import hashlib
import os
import subprocess
import sys
import time
import pathlib
import modal

app = modal.App("inkling-h100-cpp-stage")

BUILD_VOL = modal.Volume.from_name("h100-cpp-build-cache", create_if_missing=True)

cpu_image = (
    modal.Image.from_registry("pytorch/pytorch:2.7.0-cuda12.6-cudnn9-devel")
    .apt_install("ninja-build", "ccache", "git")
    .run_commands("git clone --depth 1 --branch v3.6.0 https://github.com/NVIDIA/cutlass.git /opt/cutlass")
)

h100_image = modal.Image.from_registry("pytorch/pytorch:2.7.0-cuda12.6-cudnn9-devel")

SRC_DIR = pathlib.Path(__file__).resolve().parent


def _read_local_sources():
    files = {}
    for ext in ("*.cu", "*.cpp", "*.cuh", "*.h"):
        for p in SRC_DIR.glob(ext):
            files[p.name] = p.read_text(encoding="utf-8")
    return files


@app.function(
    image=cpu_image,
    gpu=None,
    cpu=8,
    memory=16384,
    volumes={"/build_cache": BUILD_VOL},
    timeout=600,
)
def build_kernels_cpu(src_files: dict):
    import torch
    import torch.utils.cpp_extension as cpp_ext

    t0 = time.time()
    work_dir = pathlib.Path("/build_cache/work")
    ccache_dir = pathlib.Path("/build_cache/ccache")
    work_dir.mkdir(parents=True, exist_ok=True)
    ccache_dir.mkdir(parents=True, exist_ok=True)

    # Incremental source sync: only overwrite changed files so ninja skips untouched .cu!
    changed = []
    for name, content in sorted(src_files.items()):
        dst = work_dir / name
        new_h = hashlib.sha256(content.encode("utf-8")).hexdigest()
        old_h = hashlib.sha256(dst.read_bytes()).hexdigest() if dst.exists() else ""
        if new_h != old_h:
            dst.write_text(content, encoding="utf-8")
            changed.append(name)

    cu_files = sorted([p.name for p in work_dir.glob("*.cu")])
    cpp_files = sorted([p.name for p in work_dir.glob("*.cpp")])
    assert cu_files and cpp_files, f"Missing sources: cu={cu_files}, cpp={cpp_files}"

    torch_incs = cpp_ext.include_paths("cuda") + ["/usr/local/cuda/include", "/opt/cutlass/include", "/opt/cutlass/tools/util/include", str(work_dir)]
    torch_libs = cpp_ext.library_paths("cuda") + ["/usr/local/cuda/lib64"]
    # Deduplicate preserving order
    torch_incs = list(dict.fromkeys([p for p in torch_incs if os.path.isdir(p)]))
    torch_libs = list(dict.fromkeys([p for p in torch_libs if os.path.isdir(p)]))

    # Probe CUDA 12.6 cublasLt scale modes (useful telemetry for Stage 1 & 3)
    cublaslt_h = pathlib.Path("/usr/local/cuda/include/cublasLt.h")
    lt_scale_caps = []
    if cublaslt_h.exists():
        htxt = cublaslt_h.read_text(errors="ignore")
        for pat in ("CUBLASLT_MATMUL_DESC_A_SCALE_MODE", "CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F", "CUBLASLT_MATMUL_DESC_D_OUT_SCALE_POINTER"):
            if pat in htxt:
                lt_scale_caps.append(pat)

    # Check Python sysconfig include + glibcxx ABI
    import sysconfig
    py_inc = sysconfig.get_path("include")
    abi_flag = int(torch._C._GLIBCXX_USE_CXX11_ABI)

    inc_flags = " ".join([f"-I{p}" for p in torch_incs + [py_inc]])
    lib_flags = " ".join([f"-L{p}" for p in torch_libs])
    rpath_flags = " ".join([f"-Wl,-rpath,{p}" for p in torch_libs])

    nvcc_flags = (
        f"-std=c++17 -O3 --use_fast_math --expt-relaxed-constexpr --ptxas-options=-v -gencode arch=compute_90a,code=sm_90a "
        f"--compiler-options -fPIC -DTORCH_EXTENSION_NAME=inkling_h100_ops "
        f"-DTORCH_API_INCLUDE_EXTENSION_H -D_GLIBCXX_USE_CXX11_ABI={abi_flag} {inc_flags}"
    )
    cxx_flags = (
        f"-std=c++17 -O3 -fPIC -DTORCH_EXTENSION_NAME=inkling_h100_ops "
        f"-DTORCH_API_INCLUDE_EXTENSION_H -D_GLIBCXX_USE_CXX11_ABI={abi_flag} {inc_flags}"
    )
    link_libs = "-ltorch -ltorch_cpu -ltorch_python -lc10 -lc10_cuda -ltorch_cuda -lcudart -lcublasLt -lcublas"

    lines = [
        "ninja_required_version = 1.5",
        "",
        "rule nvcc_compile",
        f"  command = CCACHE_DIR=/build_cache/ccache CCACHE_TEMPDIR=/tmp/ccache_tmp CCACHE_NOHARDLINK=1 ccache nvcc {nvcc_flags} -c $in -o $out",
        "  description = NVCC $in",
        "",
        "rule cxx_compile",
        f"  command = CCACHE_DIR=/build_cache/ccache CCACHE_TEMPDIR=/tmp/ccache_tmp CCACHE_NOHARDLINK=1 ccache g++ {cxx_flags} -c $in -o $out",
        "  description = CXX $in",
        "",
        "rule link_so",
        f"  command = g++ -shared $in {lib_flags} {rpath_flags} {link_libs} -o $out",
        "  description = LINK $out",
        "",
    ]

    objs = []
    for f in cu_files:
        obj = f"{pathlib.Path(f).stem}.cu.o"
        objs.append(obj)
        lines.append(f"build {obj}: nvcc_compile {f}")
    for f in cpp_files:
        obj = f"{pathlib.Path(f).stem}.cpp.o"
        objs.append(obj)
        lines.append(f"build {obj}: cxx_compile {f}")

    so_out = "/build_cache/inkling_h100_ops.so"
    lines.append(f"build {so_out}: link_so {' '.join(objs)}")
    lines.append(f"default {so_out}")

    (work_dir / "build.ninja").write_text("\n".join(lines) + "\n", encoding="utf-8")

    os.makedirs("/tmp/ccache_tmp", exist_ok=True)
    env = os.environ.copy()
    env["CCACHE_DIR"] = "/build_cache/ccache"
    env["CCACHE_TEMPDIR"] = "/tmp/ccache_tmp"
    env["CCACHE_NOHARDLINK"] = "1"
    env["CCACHE_BASEDIR"] = str(work_dir)
    subprocess.run(["ccache", "-M", "5G"], env=env, check=False)

    res = subprocess.run(["ninja", "-j", "8", "-v"], cwd=str(work_dir), env=env, capture_output=True, text=True)
    dt = time.time() - t0
    print(f"=== CPU BUILD FINISHED in {dt:.2f}s (gpu=None, cpu=8) | changed={changed} ===", flush=True)
    print(f"=== CUDA 12.6 cublasLt scale caps: {lt_scale_caps or ['per-tensor scalar only']} ===", flush=True)
    if res.stdout:
        print(res.stdout, flush=True)
    if res.stderr:
        print(res.stderr, file=sys.stderr, flush=True)
    if res.returncode != 0:
        raise RuntimeError(f"ninja build failed with exit code {res.returncode}")

    BUILD_VOL.commit()
    return {"cpu_build_s": dt, "changed": changed, "so_path": so_out, "lt_scale_caps": lt_scale_caps}


@app.function(
    image=h100_image,
    gpu="H100",
    volumes={"/build_cache": BUILD_VOL},
    timeout=300,
    retries=0,
)
def test_on_h100(target: str = "dora"):
    import importlib.util
    import subprocess as sp
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    BUILD_VOL.reload()
    t_gpu_start = time.time()
    so_path = "/build_cache/inkling_h100_ops.so"
    assert os.path.exists(so_path), f"Compiled .so not found at {so_path}! Run build_kernels_cpu first."

    spec = importlib.util.spec_from_file_location("inkling_h100_ops", so_path)
    ops = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ops)

    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(42)
    l2_flush = torch.empty(64 * 1024 * 1024, device=device, dtype=torch.int8)

    def query_gpu_telemetry():
        try:
            out = sp.check_output(
                ["nvidia-smi", "--query-gpu=clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu", "--format=csv,noheader,nounits"],
                text=True,
            ).strip().splitlines()[0]
            parts = [p.strip() for p in out.split(",")]
            return f"SM={parts[0]}MHz | Mem={parts[1]}MHz | Power={parts[2]}W/{parts[3]}W | Temp={parts[4]}C"
        except Exception as e:
            return f"nvidia-smi N/A ({e})"

    print(f"=== H100 TEST (target={target}): GPU={torch.cuda.get_device_name(0)} | torch={torch.__version__} ===", flush=True)
    print(f"=== GPU START: {query_gpu_telemetry()} ===", flush=True)

    def bench_cg(fn, warmup=2, rep=5, flush_l2=True):
        s_cap = torch.cuda.Stream()
        s_cap.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s_cap):
            for _ in range(warmup):
                with torch.no_grad():
                    fn()
        torch.cuda.current_stream().wait_stream(s_cap)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s_cap):
            with torch.no_grad():
                fn()
        torch.cuda.current_stream().wait_stream(s_cap)
        torch.cuda.synchronize()
        for _ in range(warmup):
            g.replay()
        torch.cuda.synchronize()
        times = []
        for _ in range(rep):
            if flush_l2:
                l2_flush.zero_()
            s_ev = torch.cuda.Event(enable_timing=True)
            e_ev = torch.cuda.Event(enable_timing=True)
            s_ev.record()
            g.replay()
            e_ev.record()
            e_ev.synchronize()
            times.append(s_ev.elapsed_time(e_ev))
        times.sort()
        return (times[0], times[len(times) // 2], times[-1]), g

    if target in ("dora", "all"):
        class Cpp3xDoRAFP8Fn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, w0_fp8, w0_T_fp8, sw, sx, inv_sx, sdy, inv_sdy, la_cat, b_cat, m_cat, inv_n_cat, scale):
                orig_shape = x.shape
                x2 = x.reshape(-1, orig_shape[-1]).contiguous()
                out, h_a_cat = ops.dora_3x_fp8_fwd(x2, w0_fp8, sw, sx, inv_sx, la_cat, b_cat, m_cat, inv_n_cat, scale)
                ctx.save_for_backward(x2, w0_T_fp8, sw, sdy, inv_sdy, la_cat, b_cat, m_cat, inv_n_cat, h_a_cat, out)
                ctx.scale, ctx.orig_shape = scale, orig_shape
                return out.reshape(*orig_shape[:-1], out.shape[1])

            @staticmethod
            def backward(ctx, grad_out):
                x2, w0_T_fp8, sw, sdy, inv_sdy, la_cat, b_cat, m_cat, inv_n_cat, h_a_cat, out = ctx.saved_tensors
                go2 = grad_out.reshape(-1, out.shape[1]).contiguous()
                dx, d_la_cat, d_b_cat, dm = ops.dora_3x_fp8_bwd(
                    go2, out, x2, w0_T_fp8, sw, sdy, inv_sdy, la_cat, b_cat, m_cat, inv_n_cat, h_a_cat, ctx.scale
                )
                return dx.reshape(ctx.orig_shape), None, None, None, None, None, None, None, d_la_cat, d_b_cat, dm, None, None

        class Cpp3xFastDoRAFP8(nn.Module):
            def __init__(self, w0_3x, r=32, alpha=64):
                super().__init__()
                self.scale = float(alpha) / float(r)
                N_tot, K_d = w0_3x.shape
                N_sub = N_tot // 3
                w_f = w0_3x.contiguous().float()
                sw = (w_f.abs().amax().clamp(min=1e-8) / 448.0).view(1)
                w_fp8 = (w_f / sw).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).contiguous()
                self.register_buffer("w0_fp8", w_fp8)
                self.register_buffer("w0_T_fp8", w_fp8.t().contiguous())
                self.register_buffer("sw", sw)
                sx_val = 5.0 / 448.0
                self.register_buffer("sx", torch.tensor([sx_val], device=device, dtype=torch.float32))
                self.register_buffer("inv_sx", torch.tensor([1.0 / sx_val], device=device, dtype=torch.float32))
                self.register_buffer("sdy", torch.tensor([sx_val], device=device, dtype=torch.float32))
                self.register_buffer("inv_sdy", torch.tensor([1.0 / sx_val], device=device, dtype=torch.float32))
                self.lora_A_cat = nn.Parameter(torch.randn(3 * r, K_d, device=device, dtype=dtype) * 0.01)
                self.b_cat = nn.Parameter(torch.randn(3, N_sub, r, device=device, dtype=dtype) * 0.01)
                w_n = torch.sqrt(w_f.pow(2).sum(dim=1).clamp(min=1e-8))
                self.m_cat = nn.Parameter(w_n.to(dtype))
                self.register_buffer("cached_inv_norm", (1.0 / w_n).to(dtype))

            def forward(self, x):
                return Cpp3xDoRAFP8Fn.apply(
                    x, self.w0_fp8, self.w0_T_fp8, self.sw, self.sx, self.inv_sx, self.sdy, self.inv_sdy,
                    self.lora_A_cat, self.b_cat, self.m_cat, self.cached_inv_norm, self.scale,
                )

        # Accuracy Gate vs Exact FP32 reference!
        w0_3x = torch.randn(6912, 2304, device=device, dtype=dtype) * 0.02
        dora_cpp = Cpp3xFastDoRAFP8(w0_3x, r=32, alpha=64).to(device)
        x_d = torch.randn(1, 256, 2304, device=device, dtype=dtype, requires_grad=True)
        go_d = torch.randn(1, 256, 6912, device=device, dtype=dtype)
        y_cpp = dora_cpp(x_d)
        y_cpp.backward(go_d)

        x_ref = x_d.detach().clone().float().requires_grad_(True)
        la_r = dora_cpp.lora_A_cat.detach().clone().float().requires_grad_(True)
        bc_r = dora_cpp.b_cat.detach().clone().float().requires_grad_(True)
        mr_r = dora_cpp.m_cat.detach().clone().float().requires_grad_(True)
        parts_y = []
        for k_s in range(3):
            wk = w0_3x[k_s * 2304:(k_s + 1) * 2304].float() + dora_cpp.scale * (bc_r[k_s] @ la_r[k_s * 32:(k_s + 1) * 32])
            parts_y.append(F.linear(x_ref, wk) * (mr_r[k_s * 2304:(k_s + 1) * 2304] * dora_cpp.cached_inv_norm[k_s * 2304:(k_s + 1) * 2304].float()))
        y_ref = torch.cat(parts_y, dim=-1)
        y_ref.to(dtype).backward(go_d)

        cos_fwd = F.cosine_similarity(y_ref.flatten(), y_cpp.flatten().float(), dim=0).item()
        cos_dx = F.cosine_similarity(x_ref.grad.flatten(), x_d.grad.flatten().float(), dim=0).item()
        cos_dla = F.cosine_similarity(la_r.grad.flatten(), dora_cpp.lora_A_cat.grad.flatten().float(), dim=0).item()
        cos_db = F.cosine_similarity(bc_r.grad.flatten(), dora_cpp.b_cat.grad.flatten().float(), dim=0).item()
        cos_dm = F.cosine_similarity(mr_r.grad.flatten(), dora_cpp.m_cat.grad.flatten().float(), dim=0).item()
        assert cos_fwd >= 0.997 and cos_dx >= 0.997, f"DoRA C++ gate failed: fwd={cos_fwd}, dx={cos_dx}"
        print(f"  [GATE DoRA C++ FP8 OK] fwd={cos_fwd:.6f} dx={cos_dx:.6f} dla={cos_dla:.6f} db={cos_db:.6f} dm={cos_dm:.6f}", flush=True)
        del x_d, go_d, y_cpp, x_ref, la_r, bc_r, mr_r, y_ref

        # BS=4 CUDAGraph Benchmark (16384 tokens, 64MB L2 flush, 80 fwd + 30 bwd)
        x4 = torch.randn(4 * 4096, 2304, device=device, dtype=dtype)
        go4 = torch.randn(4 * 4096, 6912, device=device, dtype=dtype)
        ctx_d = {"out": None, "ha": None}

        def _dora_fwd():
            o, h = ops.dora_3x_fp8_fwd(
                x4, dora_cpp.w0_fp8, dora_cpp.sw, dora_cpp.sx, dora_cpp.inv_sx,
                dora_cpp.lora_A_cat, dora_cpp.b_cat, dora_cpp.m_cat, dora_cpp.cached_inv_norm, dora_cpp.scale,
            )
            ctx_d["out"], ctx_d["ha"] = o, h
            return o

        # First eager call triggers heuristic algo cache BEFORE CUDAGraph capture!
        _dora_fwd()
        _ = ops.dora_3x_fp8_bwd(
            go4, ctx_d["out"], x4, dora_cpp.w0_T_fp8, dora_cpp.sw, dora_cpp.sdy, dora_cpp.inv_sdy,
            dora_cpp.lora_A_cat, dora_cpp.b_cat, dora_cpp.m_cat, dora_cpp.cached_inv_norm, ctx_d["ha"], dora_cpp.scale,
        )
        torch.cuda.synchronize()
        prof = ops.dora_3x_fp8_fwd_profile(
            x4, dora_cpp.w0_fp8, dora_cpp.sw, dora_cpp.sx, dora_cpp.inv_sx,
            dora_cpp.lora_A_cat, dora_cpp.b_cat, dora_cpp.m_cat, dora_cpp.cached_inv_norm, dora_cpp.scale,
        )
        print(f"  [FWD BREAKDOWN ms]: ha={prof[0]:.3f} qx={prof[1]:.3f} cuBLASLt_fp8={prof[2]:.3f} wmma_epilogue={prof[3]:.3f}", flush=True)

        sf_d, gf = bench_cg(_dora_fwd, warmup=2, rep=5, flush_l2=True)
        sb_d, gb = bench_cg(
            lambda: ops.dora_3x_fp8_bwd(
                go4, ctx_d["out"], x4, dora_cpp.w0_T_fp8, dora_cpp.sw, dora_cpp.sdy, dora_cpp.inv_sdy,
                dora_cpp.lora_A_cat, dora_cpp.b_cat, dora_cpp.m_cat, dora_cpp.cached_inv_norm, ctx_d["ha"], dora_cpp.scale,
            ),
            warmup=2, rep=5, flush_l2=True,
        )
        for _ in range(15):
            gf.replay()
            gb.replay()
        pwr_str = query_gpu_telemetry()
        torch.cuda.synchronize()
        del gf, gb

        fwd_st = sf_d[1] * 80.0
        bwd_st = sb_d[1] * 30.0
        tot_st = fwd_st + bwd_st
        tf_f = ((2 * 16384 * 6912 * 2304 + 4 * 16384 * 2304 * 96) / (sf_d[1] * 1e-3)) / 1.979e15 * 100.0
        tf_b = ((2 * 16384 * 6912 * 2304 + 6 * 16384 * 2304 * 96) / (sb_d[1] * 1e-3)) / 1.979e15 * 100.0
        print(
            f"=== DORA C++ FP8 BS=4 [CG, 80f+30b]: fwd={sf_d[1]:.3f}ms ({fwd_st:.2f}ms, {tf_f:.1f}% TF) | "
            f"bwd={sb_d[1]:.3f}ms ({bwd_st:.2f}ms, {tf_b:.1f}% TF) | "
            f"STEP={tot_st:.2f}ms (vs Triton V6 78.65ms, {78.65/tot_st:.2f}x) ===",
            flush=True,
        )
        print(f"=== GPU LOAD: {pwr_str} ===", flush=True)

    if target in ("fa", "all"):
        def make_packed_doc_id(B, S, seed=21):
            g_cpu = torch.Generator(device="cpu").manual_seed(seed)
            out_d = torch.empty((B, S), dtype=torch.int16)
            for bi in range(B):
                pos = 0
                cur = 0
                while pos < S:
                    ln = int(torch.randint(120, 271, (1,), generator=g_cpu).item())
                    end = min(S, pos + ln)
                    out_d[bi, pos:end] = cur
                    cur += 1
                    pos = end
            return out_d.to(device)

        def precompute_doc_bounds(doc_id, sliding_win=1024):
            B_loc, S_loc = doc_id.shape
            q_n_start = torch.empty((B_loc, S_loc), dtype=torch.int32, device=doc_id.device)
            k_m_start = torch.arange(S_loc, dtype=torch.int32, device=doc_id.device).unsqueeze(0).expand(B_loc, S_loc).contiguous()
            k_m_end = torch.empty((B_loc, S_loc), dtype=torch.int32, device=doc_id.device)
            doc_cpu = doc_id.cpu()
            for bi in range(B_loc):
                d_row = doc_cpu[bi]
                changes = torch.nonzero(d_row[1:] != d_row[:-1], as_tuple=False).flatten() + 1
                starts = [0] + changes.tolist()
                ends = changes.tolist() + [S_loc]
                qn = torch.empty(S_loc, dtype=torch.int32)
                ke = torch.empty(S_loc, dtype=torch.int32)
                idx = torch.arange(S_loc, dtype=torch.int32)
                for s_i, e_i in zip(starts, ends):
                    qn[s_i:e_i] = torch.clamp(idx[s_i:e_i] - sliding_win + 1, min=s_i)
                    ke[s_i:e_i] = torch.clamp(idx[s_i:e_i] + sliding_win, max=e_i)
                q_n_start[bi].copy_(qn.to(doc_id.device))
                k_m_end[bi].copy_(ke.to(doc_id.device))
            return q_n_start, k_m_start, k_m_end

        # Gate 1A: Exact FP32 reference check at S=256 (packed docs)
        B_t, Hq, Hkv, S_t, D_t, R_t, WIN = 2, 24, 8, 256, 128, 16, 1024
        sm_scale = D_t ** -0.5
        q_t = torch.randn(B_t, Hq, S_t, D_t, device=device, dtype=dtype)
        k_t = torch.randn(B_t, Hkv, S_t, D_t, device=device, dtype=dtype)
        v_t = torch.randn(B_t, Hkv, S_t, D_t, device=device, dtype=dtype)
        rs_t = (torch.randn(B_t, S_t, Hq, R_t, device=device, dtype=dtype) * 0.1)
        rp_t = (torch.randn(R_t, 1024, device=device, dtype=dtype) * 0.1)
        sinks_t = torch.randn(B_t, Hq, S_t, device=device, dtype=dtype) * 0.5
        doc_t = make_packed_doc_id(B_t, S_t, seed=21)
        qn_t, kms_t, kme_t = precompute_doc_bounds(doc_t, WIN)
        go_t = torch.randn(B_t, S_t, Hq, D_t, device=device, dtype=dtype)

        q_ref = q_t.detach().float().requires_grad_(True)
        k_ref = k_t.detach().float().requires_grad_(True)
        v_ref = v_t.detach().float().requires_grad_(True)
        rs_ref = rs_t.detach().float().requires_grad_(True)
        sinks_ref = sinks_t.detach().float().requires_grad_(True)
        k_rep = k_ref.repeat_interleave(Hq // Hkv, dim=1)
        v_rep = v_ref.repeat_interleave(Hq // Hkv, dim=1)
        scores = torch.matmul(q_ref, k_rep.transpose(-2, -1)) * sm_scale
        ar = torch.arange(S_t, device=device)
        diff = ar[:, None] - ar[None, :]
        same_doc = doc_t[:, :, None] == doc_t[:, None, :]
        valid = (diff[None, :, :] >= 0) & (diff[None, :, :] < WIN) & same_doc
        rel_full = torch.matmul(rs_ref.permute(0, 2, 1, 3), rp_t.float())
        rel_idx = diff.clamp(0, 1023).view(1, 1, S_t, S_t).expand(B_t, Hq, S_t, S_t)
        rel_bias = torch.gather(rel_full, dim=-1, index=rel_idx)
        scores = scores + torch.where((diff >= 0) & (diff < 1024), rel_bias, 0.0)
        scores = scores.masked_fill(~valid[:, None, :, :], float("-inf"))
        logits = torch.cat([scores, sinks_ref.unsqueeze(-1)], dim=-1)
        probs = torch.softmax(logits, dim=-1)[..., :-1]
        o_ref = torch.matmul(probs, v_rep).transpose(1, 2).contiguous()
        o_ref.backward(go_t.float())

        o_cpp, l_cpp = ops.fa_inkling_fwd(q_t, k_t, v_t, rs_t, rp_t, sinks_t, doc_t, qn_t, sm_scale, WIN)
        dq_cpp, dk_cpp, dv_cpp, drs_cpp, dsinks_cpp = ops.fa_inkling_bwd(
            q_t, k_t, v_t, rs_t, rp_t, sinks_t, doc_t, qn_t, kms_t, kme_t, o_cpp, l_cpp, go_t, sm_scale, WIN
        )
        torch.cuda.synchronize()

        def cos(a, b):
            return F.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0).item()

        c_o = cos(o_ref, o_cpp)
        c_dq = cos(q_ref.grad, dq_cpp)
        c_dk = cos(k_ref.grad, dk_cpp)
        c_dv = cos(v_ref.grad, dv_cpp)
        c_drs = cos(rs_ref.grad, drs_cpp)
        c_ds = cos(sinks_ref.grad, dsinks_cpp)
        min_cos = min(c_o, c_dq, c_dk, c_dv, c_drs)
        print(
            f"  [GATE FA C++ S=256] o={c_o:.6f} dq={c_dq:.6f} dk={c_dk:.6f} "
            f"dv={c_dv:.6f} drs={c_drs:.6f} dsinks={c_ds:.6f} (min={min_cos:.6f})",
            flush=True,
        )
        assert min_cos >= 0.9995, f"FA C++ Gate 1A failed! min_cos={min_cos}"
        del q_t, k_t, v_t, rs_t, sinks_t, doc_t, qn_t, kms_t, kme_t, go_t, q_ref, k_ref, v_ref, rs_ref, sinks_ref, o_ref, o_cpp, l_cpp

        # BS=4 S=4096 Packed-Doc CUDAGraph Benchmark (30 layers, 64MB L2 flush)
        B4, S4 = 4, 4096
        q4 = torch.randn(B4, Hq, S4, D_t, device=device, dtype=dtype)
        k4 = torch.randn(B4, Hkv, S4, D_t, device=device, dtype=dtype)
        v4 = torch.randn(B4, Hkv, S4, D_t, device=device, dtype=dtype)
        rs4 = torch.randn(B4, S4, Hq, R_t, device=device, dtype=dtype) * 0.1
        rp4 = rp_t
        sinks4 = torch.randn(B4, Hq, S4, device=device, dtype=dtype) * 0.5
        doc4 = make_packed_doc_id(B4, S4, seed=21)
        qn4, kms4, kme4 = precompute_doc_bounds(doc4, WIN)
        go4 = torch.randn(B4, S4, Hq, D_t, device=device, dtype=dtype)

        o4, l4 = ops.fa_inkling_fwd(q4, k4, v4, rs4, rp4, sinks4, doc4, qn4, sm_scale, WIN)
        ops.fa_inkling_bwd(q4, k4, v4, rs4, rp4, sinks4, doc4, qn4, kms4, kme4, o4, l4, go4, sm_scale, WIN)
        torch.cuda.synchronize()
        fa_prof = ops.fa_inkling_bwd_profile(q4, k4, v4, rs4, rp4, sinks4, doc4, qn4, kms4, kme4, o4, l4, go4, sm_scale, WIN)
        print(f"  [FA BWD BREAKDOWN ms]: prep={fa_prof[0]:.3f} dq={fa_prof[1]:.3f} dkv={fa_prof[2]:.3f} reduce={fa_prof[3]:.3f}", flush=True)

        sf_fa, gf_fa = bench_cg(lambda: ops.fa_inkling_fwd(q4, k4, v4, rs4, rp4, sinks4, doc4, qn4, sm_scale, WIN), warmup=3, rep=7, flush_l2=True)
        sb_fa, gb_fa = bench_cg(lambda: ops.fa_inkling_bwd(q4, k4, v4, rs4, rp4, sinks4, doc4, qn4, kms4, kme4, o4, l4, go4, sm_scale, WIN), warmup=3, rep=7, flush_l2=True)
        for _ in range(15):
            gf_fa.replay()
            gb_fa.replay()
        pwr_fa = query_gpu_telemetry()
        torch.cuda.synchronize()
        del gf_fa, gb_fa

        fwd_st = sf_fa[1] * 48.0
        bwd_st = sb_fa[1] * 18.0
        tot_st = fwd_st + bwd_st
        print(
            f"=== FA C++ WGMMA BS=4 [CG, 48f+18b]: fwd={sf_fa[1]:.3f}ms ({fwd_st:.2f}ms vs V5 37.77ms) | "
            f"bwd={sb_fa[1]:.3f}ms ({bwd_st:.2f}ms vs V5 39.20ms) | "
            f"STEP={tot_st:.2f}ms (vs V5 76.97ms, {76.97/tot_st:.2f}x) ===",
            flush=True,
        )
        print(f"=== GPU LOAD (FA): {pwr_fa} ===", flush=True)

    if target in ("moe", "all"):
        E_m, H_m, I_m, R_m, TOPK_m = 128, 2304, 1152, 16, 8
        lora_scale = 2.0
        w_gu_bf = torch.randn(E_m, 2 * I_m, H_m, device=device, dtype=dtype) * 0.02
        w_dn_bf = torch.randn(E_m, H_m, I_m, device=device, dtype=dtype) * 0.02
        la_gu = torch.randn(R_m, H_m, device=device, dtype=dtype) * 0.05
        lb_gu = torch.randn(E_m, 2 * I_m, R_m, device=device, dtype=dtype) * 0.05
        la_dn = torch.randn(R_m, I_m, device=device, dtype=dtype) * 0.05
        lb_dn = torch.randn(E_m, H_m, R_m, device=device, dtype=dtype) * 0.05

        def quant_w_fp8(w):
            # w: [E, N, K] -> per-channel scale [E, N], fp8_T view [E, K, N] col-major (N contiguous? No, K contiguous!)
            wf = w.float()
            s = wf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 448.0
            q = (wf / s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
            return q.transpose(1, 2), s.squeeze(-1).contiguous()

        w_gu_fp8_T, s_gu = quant_w_fp8(w_gu_bf)
        w_dn_fp8_T, s_dn = quant_w_fp8(w_dn_bf)
        w_dn_T_fp8_T, s_dn_T = quant_w_fp8(w_dn_bf.transpose(1, 2).contiguous())
        w_gu_T_fp8_T, s_gu_T = quant_w_fp8(w_gu_bf.transpose(1, 2).contiguous())
        # Pre-quantize la_dn [16, 1152] for fast FP8 torch._scaled_mm z_dn = act_fp8 @ la_dn_fp8_T (0.04ms!)
        s_la_dn = la_dn.float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 448.0
        la_dn_fp8_T = (la_dn.float() / s_la_dn).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).t()
        s_la_dn_col = s_la_dn.t().contiguous() # [1, 16]

        def run_moe_cpp_fwd(x_in, idx_in, w_in, top_p=0.85, k_min=2):
            x_fp8, sx, exp_ends, act_tok, s_tok, s_e, s_w, inv_idx = ops.moe_route_and_quant_fp8(
                x_in, idx_in, w_in, E_m, top_p, k_min
            )
            z_gu = torch.matmul(x_in, la_gu.t())
            gu = torch._scaled_grouped_mm(x_fp8, w_gu_fp8_T, sx, s_gu, offs=exp_ends, out_dtype=dtype, use_fast_accum=True)
            act_fp8, s_act, _ = ops.moe_gu_lora_swiglu_fp8_fwd(
                gu, z_gu, lb_gu, la_dn, s_tok, s_e, act_tok, lora_scale, True
            )
            y_exp = torch._scaled_grouped_mm(act_fp8, w_dn_fp8_T, s_act, s_dn, offs=exp_ends, out_dtype=dtype, use_fast_accum=True)
            z_dn = torch._scaled_mm(act_fp8, la_dn_fp8_T, scale_a=s_act.unsqueeze(-1), scale_b=s_la_dn_col, out_dtype=dtype, use_fast_accum=False)
            y_out = ops.moe_dn_lora_scatter_fwd(y_exp, z_dn, lb_dn, inv_idx, s_tok, s_e, act_tok, w_in, lora_scale, True)
            return y_out, (gu, y_exp, exp_ends, act_tok, s_tok, s_e, s_w, inv_idx)

        def run_moe_cpp_bwd(dy_in, saved):
            gu, y_exp, exp_ends, act_tok, s_tok, s_e, s_w, inv_idx = saved
            dy_fp8, s_dy, dw_out, dz_dn = ops.moe_bwd_gather_dw_quant_fp8(
                dy_in, y_exp, lb_dn, s_tok, s_e, s_w, inv_idx, act_tok, lora_scale, True
            )
            d_act = torch._scaled_grouped_mm(dy_fp8, w_dn_T_fp8_T, s_dy, s_dn_T, offs=exp_ends, out_dtype=dtype, use_fast_accum=True)
            d_act.addmm_(dz_dn, la_dn)
            dgu_fp8, s_dgu, dz_gu = ops.moe_bwd_swiglu_quant_fp8(
                gu, d_act, dz_dn, la_dn, lb_gu, s_e, act_tok, lora_scale, True
            )
            dx_exp = torch._scaled_grouped_mm(dgu_fp8, w_gu_T_fp8_T, s_dgu, s_gu_T, offs=exp_ends, out_dtype=dtype, use_fast_accum=True)
            dx_out, dz_gu_tok = ops.moe_bwd_scatter_dx(dx_exp, dz_gu, inv_idx, True)
            dx_out.addmm_(dz_gu_tok, la_gu)
            return dx_out, dw_out

        # Gate 1D: Exact FP32 reference check (N_tok=512, top_p=1.0 for exact dense top-k=8 match)
        N_t = 512
        h_g1 = torch.randn(N_t, H_m, device=device, dtype=dtype)
        idx_g1 = torch.topk(torch.randn(N_t, E_m, device=device), TOPK_m, dim=-1).indices
        w_g1 = torch.softmax(torch.randn(N_t, TOPK_m, device=device, dtype=torch.float32), dim=-1) * TOPK_m
        go_g1 = torch.randn(N_t, H_m, device=device, dtype=dtype)

        # FP32 ref
        h_ref = h_g1.detach().float().requires_grad_(True)
        w_ref = w_g1.detach().float().requires_grad_(True)
        y_ref = torch.zeros_like(h_ref)
        for e_i in range(E_m):
            mask = idx_g1 == e_i
            if not mask.any():
                continue
            toks, ks = torch.nonzero(mask, as_tuple=True)
            xh = h_ref[toks]
            gu_r = xh @ w_gu_bf[e_i].float().t() + lora_scale * ((xh @ la_gu.float().t()) @ lb_gu[e_i].float().t())
            g_r, u_r = gu_r[:, :I_m], gu_r[:, I_m:]
            act_r = (4.0 * torch.tanh(g_r * 0.25)) * torch.sigmoid(g_r) * (25.0 * torch.tanh(u_r * 0.04))
            ye_r = act_r @ w_dn_bf[e_i].float().t() + lora_scale * ((act_r @ la_dn.float().t()) @ lb_dn[e_i].float().t())
            y_ref.index_add_(0, toks, ye_r * w_ref[toks, ks].unsqueeze(-1))
        y_ref.backward(go_g1.float())

        y_cpp, saved_g1 = run_moe_cpp_fwd(h_g1, idx_g1, w_g1, top_p=1.0, k_min=2)
        dx_cpp, dw_cpp = run_moe_cpp_bwd(go_g1, saved_g1)
        torch.cuda.synchronize()
        c_fwd = F.cosine_similarity(y_ref.flatten(), y_cpp.flatten().float(), dim=0).item()
        c_dx = F.cosine_similarity(h_ref.grad.flatten(), dx_cpp.flatten().float(), dim=0).item()
        c_dw = F.cosine_similarity(w_ref.grad.flatten(), dw_cpp.flatten().float(), dim=0).item()
        print(f"  [GATE MoE C++ FP8 N=512] fwd={c_fwd:.6f} dx={c_dx:.6f} dw={c_dw:.6f}", flush=True)
        assert c_fwd >= 0.9985 and c_dx >= 0.9985, f"MoE C++ Gate 1D failed: fwd={c_fwd}, dx={c_dx}"
        del h_g1, idx_g1, w_g1, go_g1, h_ref, w_ref, y_ref, y_cpp, saved_g1, dx_cpp, dw_cpp

        # BS=4 (N_tok=16384, top_p=0.85, k_min=2) CUDAGraph Benchmark (48 fwd + 18 bwd, 64MB L2 flush)
        N4 = 16384
        h4 = torch.randn(N4, H_m, device=device, dtype=dtype)
        idx4 = torch.topk(torch.randn(N4, E_m, device=device), TOPK_m, dim=-1).indices
        w4 = torch.softmax(torch.randn(N4, TOPK_m, device=device, dtype=torch.float32), dim=-1) * TOPK_m
        go4 = torch.randn(N4, H_m, device=device, dtype=dtype)

        ctx_m = {"saved": None}
        def _moe_fwd():
            y, s = run_moe_cpp_fwd(h4, idx4, w4, top_p=0.85, k_min=2)
            ctx_m["saved"] = s
            return y

        _moe_fwd()
        run_moe_cpp_bwd(go4, ctx_m["saved"])
        torch.cuda.synchronize()

        # Detailed per-stage breakdown (eager, warm)
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(12)]
        ev[0].record()
        x_fp8, sx, exp_ends, act_tok, s_tok, s_e, s_w, inv_idx = ops.moe_route_and_quant_fp8(h4, idx4, w4, E_m, 0.85, 2)
        ev[1].record()
        z_gu = torch.matmul(h4, la_gu.t())
        gu = torch._scaled_grouped_mm(x_fp8, w_gu_fp8_T, sx, s_gu, offs=exp_ends, out_dtype=dtype, use_fast_accum=True)
        ev[2].record()
        act_fp8, s_act, z_dn = ops.moe_gu_lora_swiglu_fp8_fwd(gu, z_gu, lb_gu, la_dn, s_tok, s_e, act_tok, lora_scale, True)
        ev[3].record()
        y_exp = torch._scaled_grouped_mm(act_fp8, w_dn_fp8_T, s_act, s_dn, offs=exp_ends, out_dtype=dtype, use_fast_accum=True)
        z_dn = torch._scaled_mm(act_fp8, la_dn_fp8_T, scale_a=s_act.unsqueeze(-1), scale_b=s_la_dn_col, out_dtype=dtype, use_fast_accum=True)
        ev[4].record()
        _ = ops.moe_dn_lora_scatter_fwd(y_exp, z_dn, lb_dn, inv_idx, s_tok, s_e, act_tok, w4, lora_scale, True)
        ev[5].record()
        dy_fp8, s_dy, dw_out, dz_dn = ops.moe_bwd_gather_dw_quant_fp8(go4, y_exp, lb_dn, s_tok, s_e, s_w, inv_idx, act_tok, lora_scale, True)
        ev[6].record()
        d_act = torch._scaled_grouped_mm(dy_fp8, w_dn_T_fp8_T, s_dy, s_dn_T, offs=exp_ends, out_dtype=dtype, use_fast_accum=True)
        ev[7].record()
        dgu_fp8, s_dgu, dz_gu = ops.moe_bwd_swiglu_quant_fp8(gu, d_act, dz_dn, la_dn, lb_gu, s_e, act_tok, lora_scale, True)
        ev[8].record()
        dx_exp = torch._scaled_grouped_mm(dgu_fp8, w_gu_T_fp8_T, s_dgu, s_gu_T, offs=exp_ends, out_dtype=dtype, use_fast_accum=True)
        ev[9].record()
        dx_o, dz_t = ops.moe_bwd_scatter_dx(dx_exp, dz_gu, inv_idx, True)
        dx_o.addmm_(dz_t, la_gu)
        ev[10].record()
        torch.cuda.synchronize()
        print(
            f"  [MoE FWD BREAKDOWN ms]: route_q={ev[0].elapsed_time(ev[1]):.3f} gmm_gu={ev[1].elapsed_time(ev[2]):.3f} "
            f"gu_lora_swiglu={ev[2].elapsed_time(ev[3]):.3f} gmm_dn={ev[3].elapsed_time(ev[4]):.3f} dn_lora_scat={ev[4].elapsed_time(ev[5]):.3f}",
            flush=True,
        )
        print(
            f"  [MoE BWD BREAKDOWN ms]: gather_dw_q={ev[5].elapsed_time(ev[6]):.3f} gmm_dact={ev[6].elapsed_time(ev[7]):.3f} "
            f"swiglu_bwd_q={ev[7].elapsed_time(ev[8]):.3f} gmm_dx={ev[8].elapsed_time(ev[9]):.3f} scat_dx={ev[9].elapsed_time(ev[10]):.3f}",
            flush=True,
        )

        sf_m, gf_m = bench_cg(_moe_fwd, warmup=2, rep=5, flush_l2=True)
        sb_m, gb_m = bench_cg(lambda: run_moe_cpp_bwd(go4, ctx_m["saved"]), warmup=2, rep=5, flush_l2=True)
        for _ in range(10):
            gf_m.replay()
            gb_m.replay()
        pwr_m = query_gpu_telemetry()
        torch.cuda.synchronize()
        del gf_m, gb_m

        fwd_st = sf_m[1] * 48.0
        bwd_st = sb_m[1] * 18.0
        tot_st = fwd_st + bwd_st
        print(
            f"=== MoE C++ FP8 BS=4 [CG, 48f+18b]: fwd={sf_m[1]:.3f}ms ({fwd_st:.2f}ms) | "
            f"bwd={sb_m[1]:.3f}ms ({bwd_st:.2f}ms) | "
            f"STEP={tot_st:.2f}ms (vs V6 281.9ms, {281.9/tot_st:.2f}x) ===",
            flush=True,
        )
        print(f"=== GPU LOAD (MoE): {pwr_m} ===", flush=True)

    if target == "train20s":
        # 20-second continuous training loop simulation of Inkling-2-Mini (BS=4, S=4096 = 16384 tok/step)
        # T-BPTT 1/3 workload per step: DoRA (80f+30b), FA (48f+18b), MoE (48f+18b)
        torch.cuda.reset_peak_memory_stats()
        B4, S4, Hq, Hkv, D_t, R_t, WIN = 4, 4096, 24, 8, 128, 16, 1024
        E_m, H_m, I_m, TOPK_m = 128, 2304, 1152, 8
        N4 = B4 * S4 # 16384 tokens/step
        sm_scale = D_t ** -0.5
        lora_scale = 2.0

        # 1. DoRA weights & activations
        w0_d = torch.randn(6912, 2304, device=device, dtype=dtype) * 0.02
        sw_d = w0_d.float().abs().amax().clamp(min=1e-8) / 448.0
        w0_fp8 = (w0_d.float() / sw_d).clamp(-448, 448).to(torch.float8_e4m3fn)
        w0_T_fp8 = w0_fp8.t().contiguous()
        sx_d = torch.tensor(1.0 / 448.0, device=device, dtype=torch.float32)
        inv_sx_d = torch.tensor(448.0, device=device, dtype=torch.float32)
        la_d = torch.randn(96, 2304, device=device, dtype=dtype) * 0.02
        lb_d = torch.randn(3, 2304, 32, device=device, dtype=dtype) * 0.02
        m_d = torch.ones(6912, device=device, dtype=dtype)
        inv_n_d = torch.ones(6912, device=device, dtype=dtype)
        x_d = torch.randn(N4, 2304, device=device, dtype=dtype)
        go_d = torch.randn(N4, 6912, device=device, dtype=dtype)

        # 2. FA weights & activations (packed docs 120-270 tok)
        g_cpu = torch.Generator(device="cpu").manual_seed(21)
        doc4 = torch.empty((B4, S4), dtype=torch.int16)
        for bi in range(B4):
            pos, cur = 0, 0
            while pos < S4:
                ln = int(torch.randint(120, 271, (1,), generator=g_cpu).item())
                end = min(S4, pos + ln)
                doc4[bi, pos:end] = cur
                cur += 1
                pos = end
        doc4 = doc4.to(device)
        qn4 = torch.empty((B4, S4), dtype=torch.int32, device=device)
        kms4 = torch.arange(S4, dtype=torch.int32, device=device).unsqueeze(0).expand(B4, S4).contiguous()
        kme4 = torch.empty((B4, S4), dtype=torch.int32, device=device)
        for bi in range(B4):
            dr = doc4[bi].cpu()
            ch = torch.nonzero(dr[1:] != dr[:-1], as_tuple=False).flatten() + 1
            starts, ends = [0] + ch.tolist(), ch.tolist() + [S4]
            idx = torch.arange(S4, dtype=torch.int32)
            qn, ke = torch.empty(S4, dtype=torch.int32), torch.empty(S4, dtype=torch.int32)
            for si, ei in zip(starts, ends):
                qn[si:ei] = torch.clamp(idx[si:ei] - WIN + 1, min=si)
                ke[si:ei] = torch.clamp(idx[si:ei] + WIN, max=ei)
            qn4[bi].copy_(qn.to(device))
            kme4[bi].copy_(ke.to(device))
        q4 = torch.randn(B4, Hq, S4, D_t, device=device, dtype=dtype)
        k4 = torch.randn(B4, Hkv, S4, D_t, device=device, dtype=dtype)
        v4 = torch.randn(B4, Hkv, S4, D_t, device=device, dtype=dtype)
        rs4 = torch.randn(B4, S4, Hq, R_t, device=device, dtype=dtype) * 0.1
        rp4 = torch.randn(R_t, 1024, device=device, dtype=dtype) * 0.1
        sinks4 = torch.randn(B4, Hq, S4, device=device, dtype=dtype) * 0.5
        go_fa = torch.randn(B4, S4, Hq, D_t, device=device, dtype=dtype)

        # 3. MoE weights & activations
        def _qw(w):
            wf = w.float()
            s = wf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 448.0
            q = (wf / s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
            return q.transpose(1, 2), s.squeeze(-1).contiguous()
        w_gu_bf = torch.randn(E_m, 2 * I_m, H_m, device=device, dtype=dtype) * 0.02
        w_dn_bf = torch.randn(E_m, H_m, I_m, device=device, dtype=dtype) * 0.02
        w_gu_fp8_T, s_gu = _qw(w_gu_bf)
        w_dn_fp8_T, s_dn = _qw(w_dn_bf)
        w_dn_T_fp8_T, s_dn_T = _qw(w_dn_bf.transpose(1, 2).contiguous())
        w_gu_T_fp8_T, s_gu_T = _qw(w_gu_bf.transpose(1, 2).contiguous())
        del w_gu_bf, w_dn_bf
        la_gu = torch.randn(R_t, H_m, device=device, dtype=dtype) * 0.05
        lb_gu = torch.randn(E_m, 2 * I_m, R_t, device=device, dtype=dtype) * 0.05
        la_dn = torch.randn(R_t, I_m, device=device, dtype=dtype) * 0.05
        lb_dn = torch.randn(E_m, H_m, R_t, device=device, dtype=dtype) * 0.05
        s_la_dn = la_dn.float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 448.0
        la_dn_fp8_T = (la_dn.float() / s_la_dn).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).t()
        s_la_dn_col = s_la_dn.t().contiguous()
        h4 = torch.randn(N4, H_m, device=device, dtype=dtype)
        idx4 = torch.topk(torch.randn(N4, E_m, device=device), TOPK_m, dim=-1).indices
        w4 = torch.softmax(torch.randn(N4, TOPK_m, device=device, dtype=torch.float32), dim=-1) * TOPK_m
        go_m = torch.randn(N4, H_m, device=device, dtype=dtype)

        # Warmup & capture 1-layer CUDAGraphs for each block
        o_d, ha_d = ops.dora_3x_fp8_fwd(x_d, w0_fp8, sw_d, sx_d, inv_sx_d, la_d, lb_d, m_d, inv_n_d, lora_scale)
        ops.dora_3x_fp8_bwd(go_d, o_d, x_d, w0_T_fp8, sw_d, sx_d, inv_sx_d, la_d, lb_d, m_d, inv_n_d, ha_d, lora_scale)
        o_fa, l_fa = ops.fa_inkling_fwd(q4, k4, v4, rs4, rp4, sinks4, doc4, qn4, sm_scale, WIN)
        ops.fa_inkling_bwd(q4, k4, v4, rs4, rp4, sinks4, doc4, qn4, kms4, kme4, o_fa, l_fa, go_fa, sm_scale, WIN)

        m_ctx = {}
        def _mf():
            xf, sx, ee, at, st, se, sw, ii = ops.moe_route_and_quant_fp8(h4, idx4, w4, E_m, 0.85, 2)
            zg = torch.matmul(h4, la_gu.t())
            gu = torch._scaled_grouped_mm(xf, w_gu_fp8_T, sx, s_gu, offs=ee, out_dtype=dtype, use_fast_accum=True)
            af, sa, _ = ops.moe_gu_lora_swiglu_fp8_fwd(gu, zg, lb_gu, la_dn, st, se, at, lora_scale, True)
            ye = torch._scaled_grouped_mm(af, w_dn_fp8_T, sa, s_dn, offs=ee, out_dtype=dtype, use_fast_accum=True)
            zd = torch._scaled_mm(af, la_dn_fp8_T, scale_a=sa.unsqueeze(-1), scale_b=s_la_dn_col, out_dtype=dtype, use_fast_accum=False)
            yo = ops.moe_dn_lora_scatter_fwd(ye, zd, lb_dn, ii, st, se, at, w4, lora_scale, True)
            m_ctx["s"] = (gu, ye, ee, at, st, se, sw, ii)
            return yo
        def _mb():
            gu, ye, ee, at, st, se, sw, ii = m_ctx["s"]
            df, sd, dw, dzd = ops.moe_bwd_gather_dw_quant_fp8(go_m, ye, lb_dn, st, se, sw, ii, at, lora_scale, True)
            da = torch._scaled_grouped_mm(df, w_dn_T_fp8_T, sd, s_dn_T, offs=ee, out_dtype=dtype, use_fast_accum=True)
            da.addmm_(dzd, la_dn)
            dgf, sdg, dzg = ops.moe_bwd_swiglu_quant_fp8(gu, da, dzd, la_dn, lb_gu, se, at, lora_scale, True)
            dxe = torch._scaled_grouped_mm(dgf, w_gu_T_fp8_T, sdg, s_gu_T, offs=ee, out_dtype=dtype, use_fast_accum=True)
            dxo, dzt = ops.moe_bwd_scatter_dx(dxe, dzg, ii, True)
            dxo.addmm_(dzt, la_gu)
            return dxo, dw

        _mf(); _mb()
        torch.cuda.synchronize()

        _, g_df = bench_cg(lambda: ops.dora_3x_fp8_fwd(x_d, w0_fp8, sw_d, sx_d, inv_sx_d, la_d, lb_d, m_d, inv_n_d, lora_scale), warmup=1, rep=2, flush_l2=False)
        _, g_db = bench_cg(lambda: ops.dora_3x_fp8_bwd(go_d, o_d, x_d, w0_T_fp8, sw_d, sx_d, inv_sx_d, la_d, lb_d, m_d, inv_n_d, ha_d, lora_scale), warmup=1, rep=2, flush_l2=False)
        _, g_ff = bench_cg(lambda: ops.fa_inkling_fwd(q4, k4, v4, rs4, rp4, sinks4, doc4, qn4, sm_scale, WIN), warmup=1, rep=2, flush_l2=False)
        _, g_fb = bench_cg(lambda: ops.fa_inkling_bwd(q4, k4, v4, rs4, rp4, sinks4, doc4, qn4, kms4, kme4, o_fa, l_fa, go_fa, sm_scale, WIN), warmup=1, rep=2, flush_l2=False)
        _, g_mf = bench_cg(_mf, warmup=1, rep=2, flush_l2=False)
        _, g_mb = bench_cg(_mb, warmup=1, rep=2, flush_l2=False)
        torch.cuda.synchronize()

        # Run 20 seconds of full training steps (80f+30b DoRA, 48f+18b FA, 48f+18b MoE per step!)
        t_loop_start = time.time()
        steps = 0
        while (time.time() - t_loop_start) < 20.0:
            for _ in range(80): g_df.replay()
            for _ in range(48): g_ff.replay()
            for _ in range(48): g_mf.replay()
            for _ in range(30): g_db.replay()
            for _ in range(18): g_fb.replay()
            for _ in range(18): g_mb.replay()
            torch.cuda.synchronize()
            steps += 1
        elapsed = time.time() - t_loop_start
        pwr_20 = query_gpu_telemetry()
        avg_step_ms = (elapsed / steps) * 1000.0
        tok_s = (steps * N4) / elapsed
        vram_alloc_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        vram_res_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)
        print(
            f"=== 20s TRAIN TEST (BS=4, 16384 tok/step, 80/30 DoRA + 48/18 FA + 48/18 MoE) ===\n"
            f"  Steps: {steps} in {elapsed:.2f}s | Avg Step: {avg_step_ms:.2f} ms | Throughput: {tok_s:,.0f} tok/s\n"
            f"  Peak VRAM (1-layer resident + CG): alloc={vram_alloc_gb:.2f} GB | reserved={vram_res_gb:.2f} GB\n"
            f"  GPU Load @20s: {pwr_20}",
            flush=True,
        )

    dt_gpu = time.time() - t_gpu_start
    print(f"=== H100 RUN FINISHED in {dt_gpu:.2f}s (Est. cost: ~${dt_gpu * 0.001097:.4f}) ===", flush=True)


@app.local_entrypoint()
def main(target: str = "dora", skip_build: bool = False, build_only: bool = False):
    t0 = time.time()
    if not skip_build:
        srcs = _read_local_sources()
        bres = build_kernels_cpu.remote(srcs)
        print(f"CPU Build Summary: {bres}", flush=True)
    if not build_only:
        test_on_h100.remote(target=target)
    print(f"Wall time: {time.time() - t0:.2f}s", flush=True)
