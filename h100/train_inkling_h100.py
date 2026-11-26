import modal

image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.7.0-cuda12.6-cudnn9-devel",
        add_python="3.11",
    )
    .pip_install("transformers>=4.45.0", "datasets", "accelerate", "huggingface_hub", "peft")
    .run_commands(
        "python -c \"from huggingface_hub import snapshot_download; snapshot_download('thinkingmachines/Inkling-2-Mini', local_dir='/model')\"",
    )
)

vol = modal.Volume.from_name("inkling-cache-vol", create_if_missing=True)
build_vol = modal.Volume.from_name("h100-cpp-build-cache", create_if_missing=True)
app = modal.App("inkling-sft-h100-prod", image=image)

CONTROL_PROMPTS = [
    "### Instruction:\nIf a train travels 60 km/h for 2.5 hours, how far does it go?\n### Response:\n",
    "### Instruction:\nWrite a Python function to reverse a string.\n### Response:\n",
    "### Instruction:\nExplain what gradient descent is in one sentence.\n### Response:\n",
]


def _format_example(ex, tokenizer):
    if "messages" in ex and isinstance(ex["messages"], list):
        msgs = ex["messages"]
        prompt_parts = []
        resp_text = ""
        for i, msg in enumerate(msgs):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if i == len(msgs) - 1 and role == "assistant":
                resp_text = content + tokenizer.eos_token
            elif role == "system":
                prompt_parts.append(f"### System:\n{content}\n")
            elif role == "user":
                prompt_parts.append(f"### Instruction:\n{content}\n")
            elif role == "assistant":
                prompt_parts.append(f"### Response:\n{content}\n")
        prompt_parts.append("### Response:\n")
        return "".join(prompt_parts), resp_text
    prompt = f"### Instruction:\n{ex.get('instruction', '')}\n"
    if ex.get("input"):
        prompt += f"### Input:\n{ex['input']}\n"
    prompt += "### Response:\n"
    resp = ex.get("output", "") + tokenizer.eos_token
    return prompt, resp


def _pack_split(examples, tokenizer, seq_len=4096):
    import torch
    prompts, resps = [], []
    for ex in examples:
        p, r = _format_example(ex, tokenizer)
        if r:
            prompts.append(p)
            resps.append(r)

    # Length-Bucketed Packing: tokenize and sort by length before packing
    all_pairs = []
    chunk = 4096
    for i in range(0, len(prompts), chunk):
        p_toks = tokenizer(prompts[i : i + chunk], add_special_tokens=False)["input_ids"]
        r_toks = tokenizer(resps[i : i + chunk], add_special_tokens=False)["input_ids"]
        for p_ids, r_ids in zip(p_toks, r_toks):
            if len(p_ids) + len(r_ids) > 0:
                all_pairs.append((p_ids, r_ids))
    all_pairs.sort(key=lambda pair: len(pair[0]) + len(pair[1]))

    packed_ids, packed_lbl, packed_doc = [], [], []
    cur_ids, cur_lbl, cur_doc = [], [], []
    doc_counter = 0
    for p_ids, r_ids in all_pairs:
        ex_len = len(p_ids) + len(r_ids)
        cur_ids.extend(p_ids)
        cur_ids.extend(r_ids)
        cur_lbl.extend([-100] * len(p_ids))
        cur_lbl.extend(r_ids)
        cur_doc.extend([doc_counter] * ex_len)
        doc_counter += 1
        while len(cur_ids) >= seq_len:
            d_slice = cur_doc[:seq_len]
            d0 = d_slice[0]
            d_rebased = [d - d0 for d in d_slice]
            packed_ids.append(cur_ids[:seq_len])
            packed_lbl.append(cur_lbl[:seq_len])
            packed_doc.append(d_rebased)
            cur_ids = cur_ids[seq_len:]
            cur_lbl = cur_lbl[seq_len:]
            cur_doc = cur_doc[seq_len:]
    return (
        torch.tensor(packed_ids, dtype=torch.int32),
        torch.tensor(packed_lbl, dtype=torch.int32),
        torch.tensor(packed_doc, dtype=torch.int16),
    )


def _load_or_build_packed_dataset(dataset_arg, tokenizer, seq_len=4096):
    import json
    import os
    import re
    import time
    import torch
    from datasets import load_dataset

    os.makedirs("/cache_vol/packed_data", exist_ok=True)
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", dataset_arg.strip("/"))
    cache_path = f"/cache_vol/packed_data/{slug}_s{seq_len}.pt"

    if os.path.exists(cache_path):
        t0 = time.time()
        data = torch.load(cache_path, map_location="cpu", weights_only=True)
        size_mb = os.path.getsize(cache_path) / (1024 ** 2)
        print(
            f"=== LOADED PACKED CACHE: {cache_path} ({size_mb:.1f} MB in {time.time()-t0:.2f}s) | "
            f"train_batches={data['train_ids'].shape[0]} ({data['train_ids'].numel()/1e6:.2f}M toks) | "
            f"val_batches={data['val_ids'].shape[0]} ({data['val_ids'].numel()/1e6:.2f}M toks) ===",
            flush=True,
        )
        return data, cache_path, 0.0, size_mb

    t0 = time.time()
    if dataset_arg.endswith(".jsonl") and os.path.exists(dataset_arg):
        with open(dataset_arg, "r", encoding="utf-8") as f:
            examples = [json.loads(line) for line in f if line.strip()]
    else:
        ds = load_dataset(dataset_arg, split="train")
        examples = [ds[i] for i in range(len(ds))]

    n_total = len(examples)
    n_val = max(200, int(n_total * 0.02))
    n_val = min(n_val, max(1, n_total // 5))
    train_ex = examples[:-n_val]
    val_ex = examples[-n_val:]

    train_ids, train_lbl, train_doc = _pack_split(train_ex, tokenizer, seq_len=seq_len)
    val_ids, val_lbl, val_doc = _pack_split(val_ex, tokenizer, seq_len=seq_len)
    if val_ids.shape[0] < 4 and train_ids.shape[0] > 8:
        need = 4 - val_ids.shape[0]
        val_ids = torch.cat([val_ids, train_ids[-need:]], dim=0)
        val_lbl = torch.cat([val_lbl, train_lbl[-need:]], dim=0)
        val_doc = torch.cat([val_doc, train_doc[-need:]], dim=0)
        train_ids, train_lbl, train_doc = train_ids[:-need], train_lbl[:-need], train_doc[:-need]

    data = {
        "train_ids": train_ids,
        "train_lbl": train_lbl,
        "train_doc": train_doc,
        "val_ids": val_ids,
        "val_lbl": val_lbl,
        "val_doc": val_doc,
        "n_examples": n_total,
    }
    torch.save(data, cache_path)
    vol.commit()
    pack_s = time.time() - t0
    size_mb = os.path.getsize(cache_path) / (1024 ** 2)
    print(
        f"=== PACKED & CACHED DATASET ({dataset_arg}, {n_total} examples) -> {cache_path} ({size_mb:.1f} MB in {pack_s:.2f}s) | "
        f"train_batches={train_ids.shape[0]} ({train_ids.numel()/1e6:.2f}M toks) | "
        f"val_batches={val_ids.shape[0]} ({val_ids.numel()/1e6:.2f}M toks) ===",
        flush=True,
    )
    return data, cache_path, pack_s, size_mb


@app.function(
    gpu="H100",
    volumes={"/cache_vol": vol, "/build_cache": build_vol},
    timeout=1800,
)
def train_h100(
    dataset: str = "yahma/alpaca-cleaned",
    seq_len: int = 4096,
    tokens_per_step: int = 16384,
    time_limit_s: float = 90.0,
    cooldown_start_s: float = 75.0,
    lr_a: float = 1.0e-4,
    lr_b: float = 4.0e-4,
    lr_moe: float = 1.5e-4,
    lr_norm: float = 1.0e-4,
    seed: int = 42,
    run_context_bench: bool = True,
    run_merge_check: bool = True,
    run_final_eval: bool = False,
    profile_1step: bool = False,
    moe_lora_start: int = 6,
    full_gc_layers: int = 10,
    moe_gc_layers: int = 8,
):
    global triton, tl
    import gc
    import importlib.util
    import math
    import os
    import queue
    import re
    import subprocess
    import sys
    import threading
    import time
    from functools import partial

    sys.modules["torchvision"] = None
    sys.modules["torchaudio"] = None

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import triton
    import triton.language as tl
    from transformers import AutoModelForImageTextToText as ModelCls, AutoTokenizer
    import transformers.models.inkling2_loop.modeling_inkling2_loop as ink_mod

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch._inductor.config.triton.cudagraphs = False

    def query_gpu_telemetry():
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=3,
            ).strip().splitlines()[0]
            sm, mem, pwr, plim, temp = [x.strip() for x in out.split(",")]
            return f"SM={sm}MHz | Mem={mem}MHz | Power={pwr}W/{plim}W | Temp={temp}C"
        except Exception:
            return "telemetry=unavailable"

    t_gpu_start = time.time()
    device = "cuda"
    dtype = torch.bfloat16
    micro_bs = max(1, tokens_per_step // seq_len)
    grad_accum = 1
    freeze_until = 6

    build_vol.reload()
    so_path = "/build_cache/inkling_h100_ops.so"
    assert os.path.exists(so_path), f"Missing prebuilt C++ extension {so_path}! Build on CPU first."
    spec = importlib.util.spec_from_file_location("inkling_h100_ops", so_path)
    ops = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ops)

    print(
        f"=== H100 PROD STACK: GPU={torch.cuda.get_device_name(0)} | torch={torch.__version__} | "
        f"seq_len={seq_len} | micro_bs={micro_bs} | tokens/step={micro_bs*seq_len} | "
        f"loaded={so_path} ===",
        flush=True,
    )
    print(f"=== GPU START: {query_gpu_telemetry()} ===", flush=True)

    tokenizer = AutoTokenizer.from_pretrained("/model")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    packed_data, cache_path, pack_s, cache_mb = _load_or_build_packed_dataset(dataset, tokenizer, seq_len=seq_len)
    train_ids_all = packed_data["train_ids"]
    train_lbl_all = packed_data["train_lbl"]
    train_doc_all = packed_data["train_doc"]
    val_ids_all = packed_data["val_ids"]
    val_lbl_all = packed_data["val_lbl"]
    val_doc_all = packed_data["val_doc"]

    # Length-Bucketed ordering for BS>1: sort train sequences by doc count so BS=4 has minimal tail idle
    doc_counts = train_doc_all[:, -1].to(torch.int32)
    bucket_order = torch.argsort(doc_counts, stable=True)
    train_ids_all = train_ids_all[bucket_order]
    train_lbl_all = train_lbl_all[bucket_order]
    train_doc_all = train_doc_all[bucket_order]

    n_train_seqs = int(train_ids_all.shape[0])
    n_train_groups = max(1, n_train_seqs // micro_bs)
    n_val_batches = int(val_ids_all.shape[0])

    # =========================================================================
    # 1. CAUSAL CONV1D V6 (Shift-Register CL 256x128 + Zero-Atomic CF 8x128)
    # =========================================================================
    @triton.jit
    def causal_conv1d_fwd_cl_k4_kernel(
        X_ptr, W_ptr, B_ptr, Out_ptr,
        stride_xb, stride_xs, stride_ob, stride_os,
        stride_wc, stride_wk,
        C_dim, S_dim, HAS_BIAS: tl.constexpr, SILU: tl.constexpr,
        BLOCK_C: tl.constexpr, CHUNK_S: tl.constexpr,
    ):
        pid_b, pid_c, pid_s = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C_dim
        s0 = pid_s * CHUNK_S
        wb = W_ptr + offs_c * stride_wc
        w0 = tl.load(wb + 0 * stride_wk, mask=mask_c, other=0.0).to(tl.float32)
        w1 = tl.load(wb + 1 * stride_wk, mask=mask_c, other=0.0).to(tl.float32)
        w2 = tl.load(wb + 2 * stride_wk, mask=mask_c, other=0.0).to(tl.float32)
        w3 = tl.load(wb + 3 * stride_wk, mask=mask_c, other=0.0).to(tl.float32)
        b_val = tl.load(B_ptr + offs_c, mask=mask_c, other=0.0).to(tl.float32) if HAS_BIAS else tl.zeros([BLOCK_C], dtype=tl.float32)
        x_b = X_ptr + pid_b * stride_xb + offs_c
        o_b = Out_ptr + pid_b * stride_ob + offs_c
        x_m3 = tl.load(x_b + (s0 - 3) * stride_xs, mask=mask_c & (s0 - 3 >= 0) & (s0 - 3 < S_dim), other=0.0).to(tl.float32)
        x_m2 = tl.load(x_b + (s0 - 2) * stride_xs, mask=mask_c & (s0 - 2 >= 0) & (s0 - 2 < S_dim), other=0.0).to(tl.float32)
        x_m1 = tl.load(x_b + (s0 - 1) * stride_xs, mask=mask_c & (s0 - 1 >= 0) & (s0 - 1 < S_dim), other=0.0).to(tl.float32)
        for step in range(CHUNK_S):
            s = s0 + step
            if s < S_dim:
                x_cur = tl.load(x_b + s * stride_xs, mask=mask_c, other=0.0).to(tl.float32)
                acc = b_val + x_m3 * w0 + x_m2 * w1 + x_m1 * w2 + x_cur * w3
                if SILU:
                    acc = acc * tl.sigmoid(acc)
                tl.store(o_b + s * stride_os, acc.to(Out_ptr.dtype.element_ty), mask=mask_c)
                x_m3, x_m2, x_m1 = x_m2, x_m1, x_cur

    @triton.jit
    def causal_conv1d_bwd_cl_k4_dx_kernel(
        dOut_ptr, X_ptr, W_ptr, B_ptr, dX_ptr,
        stride_dob, stride_doc, stride_dos,
        stride_xb, stride_xs, stride_dxb, stride_dxs,
        stride_wc, stride_wk,
        C_dim, S_dim, HAS_BIAS: tl.constexpr, SILU: tl.constexpr,
        BLOCK_C: tl.constexpr, CHUNK_S: tl.constexpr,
    ):
        pid_b, pid_c, pid_s = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C_dim
        s0 = pid_s * CHUNK_S
        wb = W_ptr + offs_c * stride_wc
        w0 = tl.load(wb + 0 * stride_wk, mask=mask_c, other=0.0).to(tl.float32)
        w1 = tl.load(wb + 1 * stride_wk, mask=mask_c, other=0.0).to(tl.float32)
        w2 = tl.load(wb + 2 * stride_wk, mask=mask_c, other=0.0).to(tl.float32)
        w3 = tl.load(wb + 3 * stride_wk, mask=mask_c, other=0.0).to(tl.float32)
        b_val = tl.load(B_ptr + offs_c, mask=mask_c, other=0.0).to(tl.float32) if HAS_BIAS else tl.zeros([BLOCK_C], dtype=tl.float32)
        x_b = X_ptr + pid_b * stride_xb + offs_c
        do_b = dOut_ptr + pid_b * stride_dob + offs_c * stride_doc
        dx_b = dX_ptr + pid_b * stride_dxb + offs_c
        if SILU:
            x_m3 = tl.load(x_b + (s0 - 3) * stride_xs, mask=mask_c & (s0 - 3 >= 0) & (s0 - 3 < S_dim), other=0.0).to(tl.float32)
            x_m2 = tl.load(x_b + (s0 - 2) * stride_xs, mask=mask_c & (s0 - 2 >= 0) & (s0 - 2 < S_dim), other=0.0).to(tl.float32)
            x_m1 = tl.load(x_b + (s0 - 1) * stride_xs, mask=mask_c & (s0 - 1 >= 0) & (s0 - 1 < S_dim), other=0.0).to(tl.float32)
        else:
            x_m3 = tl.zeros([BLOCK_C], dtype=tl.float32)
            x_m2 = tl.zeros([BLOCK_C], dtype=tl.float32)
            x_m1 = tl.zeros([BLOCK_C], dtype=tl.float32)
        g_m3 = tl.zeros([BLOCK_C], dtype=tl.float32)
        g_m2 = tl.zeros([BLOCK_C], dtype=tl.float32)
        g_m1 = tl.zeros([BLOCK_C], dtype=tl.float32)
        for step in range(CHUNK_S + 3):
            t = s0 + step
            if t < S_dim:
                g_cur = tl.load(do_b + t * stride_dos, mask=mask_c, other=0.0).to(tl.float32)
                if SILU:
                    x_cur = tl.load(x_b + t * stride_xs, mask=mask_c, other=0.0).to(tl.float32)
                    pre = b_val + x_m3 * w0 + x_m2 * w1 + x_m1 * w2 + x_cur * w3
                    sig = tl.sigmoid(pre)
                    g_cur = g_cur * (sig * (1.0 + pre * (1.0 - sig)))
                    x_m3, x_m2, x_m1 = x_m2, x_m1, x_cur
            else:
                g_cur = tl.zeros([BLOCK_C], dtype=tl.float32)
            if step >= 3:
                s_w = s0 + step - 3
                if s_w < S_dim:
                    dx_val = g_m3 * w3 + g_m2 * w2 + g_m1 * w1 + g_cur * w0
                    tl.store(dx_b + s_w * stride_dxs, dx_val.to(dX_ptr.dtype.element_ty), mask=mask_c)
            g_m3, g_m2, g_m1 = g_m2, g_m1, g_cur

    class TritonCausalConv1dFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weight, bias, activation):
            # Inkling passes x with shape [B, C, S]; sconv passes transpose(1,2) so x.stride(1)==1 (Channels-Last!)
            if x.stride(1) != 1:
                x = x.transpose(1, 2).contiguous().transpose(1, 2)
            B_c, C_c, S_c = x.shape
            K_c = weight.shape[1]
            silu = activation in ("silu", "swish")
            has_bias = bias is not None
            if K_c == 4:
                out = torch.empty_like(x)
                BLOCK_C, CHUNK_S = 256, 128
                causal_conv1d_fwd_cl_k4_kernel[(B_c, triton.cdiv(C_c, BLOCK_C), triton.cdiv(S_c, CHUNK_S))](
                    x, weight, bias if has_bias else x, out,
                    x.stride(0), x.stride(2), out.stride(0), out.stride(2),
                    weight.stride(0), weight.stride(1),
                    C_c, S_c, HAS_BIAS=has_bias, SILU=silu, BLOCK_C=BLOCK_C, CHUNK_S=CHUNK_S, num_warps=4,
                )
                ctx.save_for_backward(x, weight, bias if has_bias else torch.empty(0, device=x.device))
                ctx.has_bias, ctx.silu, ctx.K_c = has_bias, silu, K_c
                return out
            w3 = weight.unsqueeze(1)
            out = F.conv1d(F.pad(x, (K_c - 1, 0)), w3, bias=bias, groups=C_c)
            if silu:
                out = F.silu(out)
            ctx.save_for_backward(x, weight, bias if has_bias else torch.empty(0, device=x.device))
            ctx.has_bias, ctx.silu, ctx.K_c = has_bias, silu, K_c
            return out

        @staticmethod
        def backward(ctx, dout):
            x, weight, bias = ctx.saved_tensors
            if ctx.K_c == 4:
                B_c, C_c, S_c = x.shape
                dx = torch.empty_like(x)
                BLOCK_C, CHUNK_S = 256, 128
                causal_conv1d_bwd_cl_k4_dx_kernel[(B_c, triton.cdiv(C_c, BLOCK_C), triton.cdiv(S_c, CHUNK_S))](
                    dout, x, weight, bias if ctx.has_bias else x, dx,
                    dout.stride(0), dout.stride(1), dout.stride(2),
                    x.stride(0), x.stride(2), dx.stride(0), dx.stride(2),
                    weight.stride(0), weight.stride(1),
                    C_c, S_c, HAS_BIAS=ctx.has_bias, SILU=ctx.silu, BLOCK_C=BLOCK_C, CHUNK_S=CHUNK_S, num_warps=4,
                )
                return dx, None, None, None
            return None, None, None, None

    def patched_causal_conv1d_fn(x, weight, bias=None, activation=None, **kwargs):
        return TritonCausalConv1dFn.apply(x, weight, bias, activation)

    ink_mod.causal_conv1d_fn = patched_causal_conv1d_fn

    @triton.jit
    def causal_conv1d_res_fwd_cl_k4_kernel(
        X_ptr, W_ptr, Out_ptr,
        stride_xb, stride_xs, stride_ob, stride_os,
        stride_wc, stride_wk,
        C_dim, S_dim, res_scale,
        BLOCK_C: tl.constexpr, CHUNK_S: tl.constexpr,
    ):
        pid_b, pid_c, pid_s = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C_dim
        s0 = pid_s * CHUNK_S
        wb = W_ptr + offs_c * stride_wc
        w0 = tl.load(wb + 0 * stride_wk, mask=mask_c, other=0.0).to(tl.float32)
        w1 = tl.load(wb + 1 * stride_wk, mask=mask_c, other=0.0).to(tl.float32)
        w2 = tl.load(wb + 2 * stride_wk, mask=mask_c, other=0.0).to(tl.float32)
        w3 = tl.load(wb + 3 * stride_wk, mask=mask_c, other=0.0).to(tl.float32)
        x_b = X_ptr + pid_b * stride_xb + offs_c
        o_b = Out_ptr + pid_b * stride_ob + offs_c
        x_m3 = tl.load(x_b + (s0 - 3) * stride_xs, mask=mask_c & (s0 - 3 >= 0) & (s0 - 3 < S_dim), other=0.0).to(tl.float32)
        x_m2 = tl.load(x_b + (s0 - 2) * stride_xs, mask=mask_c & (s0 - 2 >= 0) & (s0 - 2 < S_dim), other=0.0).to(tl.float32)
        x_m1 = tl.load(x_b + (s0 - 1) * stride_xs, mask=mask_c & (s0 - 1 >= 0) & (s0 - 1 < S_dim), other=0.0).to(tl.float32)
        for step in range(CHUNK_S):
            s = s0 + step
            if s < S_dim:
                x_cur = tl.load(x_b + s * stride_xs, mask=mask_c, other=0.0).to(tl.float32)
                acc = (x_m3 * w0 + x_m2 * w1 + x_m1 * w2 + x_cur * (w3 + 1.0)) * res_scale
                tl.store(o_b + s * stride_os, acc.to(Out_ptr.dtype.element_ty), mask=mask_c)
                x_m3, x_m2, x_m1 = x_m2, x_m1, x_cur

    @triton.jit
    def causal_conv1d_res_bwd_cl_k4_dx_kernel(
        dOut_ptr, W_ptr, dX_ptr,
        stride_dob, stride_dos, stride_dxb, stride_dxs,
        stride_wc, stride_wk,
        C_dim, S_dim, res_scale,
        BLOCK_C: tl.constexpr, CHUNK_S: tl.constexpr,
    ):
        pid_b, pid_c, pid_s = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C_dim
        s0 = pid_s * CHUNK_S
        wb = W_ptr + offs_c * stride_wc
        w0 = tl.load(wb + 0 * stride_wk, mask=mask_c, other=0.0).to(tl.float32)
        w1 = tl.load(wb + 1 * stride_wk, mask=mask_c, other=0.0).to(tl.float32)
        w2 = tl.load(wb + 2 * stride_wk, mask=mask_c, other=0.0).to(tl.float32)
        w3 = tl.load(wb + 3 * stride_wk, mask=mask_c, other=0.0).to(tl.float32)
        do_b = dOut_ptr + pid_b * stride_dob + offs_c
        dx_b = dX_ptr + pid_b * stride_dxb + offs_c
        g_m3 = tl.zeros([BLOCK_C], dtype=tl.float32)
        g_m2 = tl.zeros([BLOCK_C], dtype=tl.float32)
        g_m1 = tl.zeros([BLOCK_C], dtype=tl.float32)
        for step in range(CHUNK_S + 3):
            t = s0 + step
            if t < S_dim:
                g_cur = tl.load(do_b + t * stride_dos, mask=mask_c, other=0.0).to(tl.float32)
            else:
                g_cur = tl.zeros([BLOCK_C], dtype=tl.float32)
            if step >= 3:
                s_w = s0 + step - 3
                if s_w < S_dim:
                    dx_val = (g_m3 * (w3 + 1.0) + g_m2 * w2 + g_m1 * w1 + g_cur * w0) * res_scale
                    tl.store(dx_b + s_w * stride_dxs, dx_val.to(dX_ptr.dtype.element_ty), mask=mask_c)
            g_m3, g_m2, g_m1 = g_m2, g_m1, g_cur

    class TritonShortConvResFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x_bsc, w_c4, res_scale):
            B_c, S_c, C_c = x_bsc.shape
            out = torch.empty_like(x_bsc)
            causal_conv1d_res_fwd_cl_k4_kernel[(B_c, triton.cdiv(C_c, 256), triton.cdiv(S_c, 128))](
                x_bsc, w_c4, out,
                x_bsc.stride(0), x_bsc.stride(1), out.stride(0), out.stride(1),
                w_c4.stride(0), w_c4.stride(1), C_c, S_c, res_scale,
                BLOCK_C=256, CHUNK_S=128, num_warps=4,
            )
            ctx.save_for_backward(w_c4)
            ctx.res_scale = res_scale
            ctx.shp = (B_c, S_c, C_c)
            return out

        @staticmethod
        def backward(ctx, grad_out):
            (w_c4,) = ctx.saved_tensors
            B_c, S_c, C_c = ctx.shp
            do_c = grad_out.contiguous()
            dx = torch.empty_like(do_c)
            causal_conv1d_res_bwd_cl_k4_dx_kernel[(B_c, triton.cdiv(C_c, 256), triton.cdiv(S_c, 128))](
                do_c, w_c4, dx,
                do_c.stride(0), do_c.stride(1), dx.stride(0), dx.stride(1),
                w_c4.stride(0), w_c4.stride(1), C_c, S_c, ctx.res_scale,
                BLOCK_C=256, CHUNK_S=128, num_warps=4,
            )
            return dx, None, None

    orig_short_conv_fwd = ink_mod.Inkling2LoopShortConvolution.forward
    def patched_short_conv_forward(self, hidden_states, dynamic_input=None, cache=None, conv_mask=None, cache_offset=0, **kwargs):
        if (
            cache is None
            and conv_mask is None
            and kwargs.get("seq_idx") is None
            and self.conv_kernel_size == 4
            and self.conv1d.bias is None
            and not self.conv1d.weight.requires_grad
            and hidden_states.ndim == 3
        ):
            return TritonShortConvResFn.apply(
                hidden_states.contiguous(),
                self.conv1d.weight.squeeze(1),
                1.0 / math.sqrt(self.conv_kernel_size + 1),
            )
        return orig_short_conv_fwd(
            self, hidden_states, dynamic_input=dynamic_input, cache=cache,
            conv_mask=conv_mask, cache_offset=cache_offset, **kwargs,
        )
    ink_mod.Inkling2LoopShortConvolution.forward = patched_short_conv_forward

    # =========================================================================
    # 2. C++ HOPPER WGMMA FLASH ATTENTION (fa_inkling.cu, 0.655ms fwd)
    # =========================================================================
    def precompute_doc_bounds_cpu(doc_id_cpu, sliding_win=1024):
        B_loc, S_loc = doc_id_cpu.shape
        q_n_start = torch.empty((B_loc, S_loc), dtype=torch.int32)
        k_m_start = torch.arange(S_loc, dtype=torch.int32).unsqueeze(0).expand(B_loc, S_loc).contiguous()
        k_m_end = torch.empty((B_loc, S_loc), dtype=torch.int32)
        idx = torch.arange(S_loc, dtype=torch.int32)
        for bi in range(B_loc):
            d_row = doc_id_cpu[bi]
            changes = torch.nonzero(d_row[1:] != d_row[:-1], as_tuple=False).flatten() + 1
            starts = [0] + changes.tolist()
            ends = changes.tolist() + [S_loc]
            qn = torch.empty(S_loc, dtype=torch.int32)
            ke = torch.empty(S_loc, dtype=torch.int32)
            for s_i, e_i in zip(starts, ends):
                qn[s_i:e_i] = torch.clamp(idx[s_i:e_i] - sliding_win + 1, min=s_i)
                ke[s_i:e_i] = torch.clamp(idx[s_i:e_i] + sliding_win, max=e_i)
            q_n_start[bi] = qn
            k_m_end[bi] = ke
        return q_n_start, k_m_start, k_m_end

    attn_ctx = {"doc_id": None, "q_n_start": None, "k_m_start": None, "k_m_end": None}

    @triton.jit
    def triton_4d_t12_kernel(
        In_ptr, Out_ptr,
        stride_in0, stride_in1, stride_in2,
        stride_out0, stride_out1, stride_out2,
        S_dim, H_dim,
        D_DIM: tl.constexpr, BLOCK_S: tl.constexpr,
    ):
        pid_s = tl.program_id(0)
        pid_bh = tl.program_id(1)
        b = pid_bh // H_dim
        h = pid_bh % H_dim
        offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
        offs_d = tl.arange(0, D_DIM)
        m_s = offs_s < S_dim
        in_ptrs = In_ptr + b * stride_in0 + offs_s[:, None] * stride_in1 + h * stride_in2 + offs_d[None, :]
        out_ptrs = Out_ptr + b * stride_out0 + h * stride_out1 + offs_s[:, None] * stride_out2 + offs_d[None, :]
        tile = tl.load(in_ptrs, mask=m_s[:, None], other=0.0)
        tl.store(out_ptrs, tile, mask=m_s[:, None])

    @triton.jit
    def triton_3d_t12_kernel(
        In_ptr, Out_ptr,
        S_dim, H_dim: tl.constexpr, BLOCK_S: tl.constexpr, BLOCK_H: tl.constexpr,
    ):
        pid_s = tl.program_id(0)
        pid_b = tl.program_id(1)
        offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
        offs_h = tl.arange(0, BLOCK_H)
        mask = (offs_s[:, None] < S_dim) & (offs_h[None, :] < H_dim)
        tile = tl.load(In_ptr + pid_b * S_dim * H_dim + offs_s[:, None] * H_dim + offs_h[None, :], mask=mask, other=0.0)
        tl.store(Out_ptr + pid_b * H_dim * S_dim + offs_h[:, None] * S_dim + offs_s[None, :], tl.trans(tile), mask=tl.trans(mask))

    def _fast_t12_contiguous_4d(t):
        if t.is_contiguous():
            return t
        if t.ndim == 4 and t.shape[3] == 128 and t.stride(3) == 1 and t.stride(1) == 128:
            B_t, H_t, S_t, D_t = t.shape
            out = torch.empty((B_t, H_t, S_t, D_t), device=t.device, dtype=t.dtype)
            triton_4d_t12_kernel[(triton.cdiv(S_t, 16), B_t * H_t)](
                t, out, t.stride(0), t.stride(2), t.stride(1),
                out.stride(0), out.stride(1), out.stride(2),
                S_t, H_t, D_DIM=128, BLOCK_S=16, num_warps=4,
            )
            return out
        return t.contiguous()

    def _fast_t12_contiguous_3d(t):
        if t.is_contiguous():
            return t
        if t.ndim == 3 and t.stride(1) == 1 and t.stride(2) == t.shape[1] and t.shape[1] <= 32:
            B_s, H_s, S_s = t.shape
            if t.stride(0) == S_s * H_s:
                out = torch.empty((B_s, H_s, S_s), device=t.device, dtype=t.dtype)
                triton_3d_t12_kernel[(triton.cdiv(S_s, 32), B_s)](
                    t, out, S_s, H_dim=H_s, BLOCK_S=32, BLOCK_H=32, num_warps=4,
                )
                return out
        return t.contiguous()

    class InklingCppFlashAttnFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, k, v, rel_states, rel_proj, sinks, doc_id, q_n_start, k_m_start, k_m_end, sm_scale, sliding_win):
            q_c = _fast_t12_contiguous_4d(q)
            k_c = _fast_t12_contiguous_4d(k)
            v_c = _fast_t12_contiguous_4d(v)
            rs_c = rel_states.contiguous()
            rp_c = rel_proj.contiguous()
            sinks_2d = _fast_t12_contiguous_3d(sinks.squeeze(-1))
            out, L_buf = ops.fa_inkling_fwd(q_c, k_c, v_c, rs_c, rp_c, sinks_2d, doc_id, q_n_start, float(sm_scale), int(sliding_win))
            ctx.save_for_backward(q_c, k_c, v_c, rs_c, rp_c, sinks_2d, doc_id, q_n_start, k_m_start, k_m_end, out, L_buf)
            ctx.sm_scale = float(sm_scale)
            ctx.sliding_win = int(sliding_win)
            return out

        @staticmethod
        def backward(ctx, grad_out):
            q_c, k_c, v_c, rs_c, rp_c, sinks_2d, doc_id, q_n_start, k_m_start, k_m_end, out, L_buf = ctx.saved_tensors
            dq, dk, dv, drs, dsinks = ops.fa_inkling_bwd(
                q_c, k_c, v_c, rs_c, rp_c, sinks_2d, doc_id, q_n_start, k_m_start, k_m_end,
                out, L_buf, grad_out.contiguous(), ctx.sm_scale, ctx.sliding_win,
            )
            return dq, dk, dv, drs, None, dsinks.unsqueeze(-1), None, None, None, None, None, None

    def _run_fa_cpp(query_states, key_states, value_states, rel_states, rel_proj, sinks, sm_scale, sliding_win):
        B_a, H_q, S_a, D_a = query_states.shape
        doc_id = attn_ctx["doc_id"]
        q_n_start = attn_ctx["q_n_start"]
        k_m_start = attn_ctx["k_m_start"]
        k_m_end = attn_ctx["k_m_end"]
        if doc_id is None or doc_id.shape != (B_a, S_a):
            doc_id = torch.zeros((B_a, S_a), device=query_states.device, dtype=torch.int16)
            idx = torch.arange(S_a, device=query_states.device, dtype=torch.int32).unsqueeze(0).expand(B_a, S_a).contiguous()
            q_n_start = torch.clamp(idx - sliding_win + 1, min=0)
            k_m_start = idx
            k_m_end = torch.clamp(idx + sliding_win, max=S_a)
        return InklingCppFlashAttnFn.apply(
            query_states, key_states, value_states, rel_states, rel_proj, sinks,
            doc_id, q_n_start, k_m_start, k_m_end, sm_scale, sliding_win,
        )

    def patched_sliding_attn_forward(
        self, hidden_states, attention_mask, relative_distance, conv_mask=None,
        cache=None, key_hidden_states=None, value_hidden_states=None, cache_offset=0, **kwargs
    ):
        key_hidden_states = hidden_states if key_hidden_states is None else key_hidden_states
        value_hidden_states = hidden_states if value_hidden_states is None else value_hidden_states
        pos_ids = kwargs["position_ids"]
        input_shape = hidden_states.shape[:-1]
        query_states = self.q_proj(hidden_states, pos_ids)
        key_states = self.k_proj(key_hidden_states, cache=cache, conv_mask=conv_mask, cache_offset=cache_offset, **kwargs)
        value_states = self.v_proj(value_hidden_states, cache=cache, conv_mask=conv_mask, cache_offset=cache_offset, **kwargs)
        rel_states = self.relative_position_bias.linear(hidden_states).view(*hidden_states.shape[:-1], self.num_heads, -1)
        rel_states = self.relative_position_bias.norm(rel_states, pos_ids)
        sinks = self.sink_proj(hidden_states).transpose(1, 2).unsqueeze(-1)
        win = self.sliding_window if self.is_sliding else 1024
        attn_output = _run_fa_cpp(query_states, key_states, value_states, rel_states, self.relative_position_bias.proj, sinks, self.scaling, win)
        return self.o_proj(attn_output.reshape(*input_shape, -1), hidden_states), None

    def patched_full_attn_forward(
        self, hidden_states, key_hidden_states, value_hidden_states, attention_mask,
        relative_distance, conv_mask=None, cache=None, cache_offset=0, **kwargs
    ):
        pos_ids = kwargs["position_ids"]
        input_shape = hidden_states.shape[:-1]
        query_states = self.q_proj(hidden_states, pos_ids)
        key_states = self.k_proj(key_hidden_states, cache=cache, conv_mask=conv_mask, cache_offset=cache_offset, **kwargs)
        value_states = self.v_proj(value_hidden_states, cache=cache, conv_mask=conv_mask, cache_offset=cache_offset, **kwargs)
        rel_states = self.relative_position_bias.linear(hidden_states).view(*hidden_states.shape[:-1], self.num_heads, -1)
        rel_states = self.relative_position_bias.norm(rel_states, pos_ids)
        sinks = self.sink_proj(hidden_states).transpose(1, 2).unsqueeze(-1)
        win = self.sliding_window if self.is_sliding else 1024
        attn_output = _run_fa_cpp(query_states, key_states, value_states, rel_states, self.relative_position_bias.proj, sinks, self.scaling, win)
        return self.o_proj(attn_output.reshape(*input_shape, -1), hidden_states), None

    # =========================================================================
    # 3. C++ FP8 MOE (moe_cutlass.cu + torch._scaled_grouped_mm + ZeroAlloc Triton LoRA)
    # =========================================================================
    @triton.jit
    def moe_z_dn_fp8_kernel(
        Act_fp8_ptr, S_act_ptr, La_dn_ptr, Exp_ends_ptr, Out_ptr,
        E_m: tl.constexpr, I_dim: tl.constexpr, R_dim: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        act_tot = tl.load(Exp_ends_ptr + E_m - 1)
        m0 = pid_m * BLOCK_M
        if m0 >= act_tot:
            return
        offs_m = m0 + tl.arange(0, BLOCK_M)
        m_m = offs_m < act_tot
        offs_r = tl.arange(0, R_dim)
        sa = tl.load(S_act_ptr + offs_m, mask=m_m, other=0.0).to(tl.float32)
        acc = tl.zeros([BLOCK_M, R_dim], dtype=tl.float32)
        for k0 in range(0, I_dim, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            a_fp8 = tl.load(Act_fp8_ptr + offs_m[:, None] * I_dim + offs_k[None, :], mask=m_m[:, None], other=0.0).to(tl.float32)
            a_bf = (a_fp8 * sa[:, None]).to(tl.bfloat16)
            la_t = tl.load(La_dn_ptr + offs_r[None, :] * I_dim + offs_k[:, None]).to(tl.bfloat16)
            acc += tl.dot(a_bf, la_t)
        tl.store(Out_ptr + offs_m[:, None] * R_dim + offs_r[None, :], acc.to(Out_ptr.dtype.element_ty), mask=m_m[:, None])

    @triton.jit
    def moe_dact_add_lora_kernel(
        dAct_ptr, dZ_ptr, La_dn_ptr, Exp_ends_ptr,
        E_m: tl.constexpr, I_dim: tl.constexpr, R_dim: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_I: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_i = tl.program_id(1)
        act_tot = tl.load(Exp_ends_ptr + E_m - 1)
        m0 = pid_m * BLOCK_M
        if m0 >= act_tot:
            return
        offs_m = m0 + tl.arange(0, BLOCK_M)
        offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
        m_m = offs_m < act_tot
        m_i = offs_i < I_dim
        offs_r = tl.arange(0, R_dim)
        dz = tl.load(dZ_ptr + offs_m[:, None] * R_dim + offs_r[None, :], mask=m_m[:, None], other=0.0).to(tl.bfloat16)
        la = tl.load(La_dn_ptr + offs_r[:, None] * I_dim + offs_i[None, :], mask=m_i[None, :], other=0.0).to(tl.bfloat16)
        cur = tl.load(dAct_ptr + offs_m[:, None] * I_dim + offs_i[None, :], mask=m_m[:, None] & m_i[None, :], other=0.0).to(tl.float32)
        res = cur + tl.dot(dz, la, out_dtype=tl.float32)
        tl.store(dAct_ptr + offs_m[:, None] * I_dim + offs_i[None, :], res.to(dAct_ptr.dtype.element_ty), mask=m_m[:, None] & m_i[None, :])

    @triton.jit
    def moe_lora_b_add_triton_kernel(
        Out_ptr, Z_ptr, B_ptr, S_tok_ptr, Exp_ends_ptr,
        lora_scale,
        N_dim: tl.constexpr, R_dim: tl.constexpr,
        USE_S_TOK: tl.constexpr,
        BLOCK_M: tl.constexpr, M_SPLIT: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        pid_e = tl.program_id(0)
        pid_ms = tl.program_id(1)
        pid_n = tl.program_id(2)
        start = tl.load(Exp_ends_ptr + pid_e - 1, mask=pid_e > 0, other=0)
        end = tl.load(Exp_ends_ptr + pid_e)
        m_s0 = start + pid_ms * M_SPLIT
        if m_s0 >= end:
            return
        m_s1 = tl.minimum(m_s0 + M_SPLIT, end)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        m_n = offs_n < N_dim
        offs_r = tl.arange(0, R_dim)
        b_t = tl.load(
            B_ptr + pid_e * N_dim * R_dim + offs_n[None, :] * R_dim + offs_r[:, None],
            mask=m_n[None, :], other=0.0,
        ).to(tl.bfloat16)
        for m0 in range(m_s0, m_s1, BLOCK_M):
            offs_m = m0 + tl.arange(0, BLOCK_M)
            m_m = offs_m < m_s1
            if USE_S_TOK:
                tok = tl.load(S_tok_ptr + offs_m, mask=m_m, other=-1)
                valid = m_m & (tok >= 0)
                z_tile = tl.load(Z_ptr + tok[:, None] * R_dim + offs_r[None, :], mask=valid[:, None], other=0.0).to(tl.bfloat16)
            else:
                valid = m_m
                z_tile = tl.load(Z_ptr + offs_m[:, None] * R_dim + offs_r[None, :], mask=valid[:, None], other=0.0).to(tl.bfloat16)
            cur = tl.load(Out_ptr + offs_m[:, None] * N_dim + offs_n[None, :], mask=valid[:, None] & m_n[None, :], other=0.0).to(tl.float32)
            res = cur + tl.dot(z_tile, b_t, out_dtype=tl.float32) * lora_scale
            tl.store(Out_ptr + offs_m[:, None] * N_dim + offs_n[None, :], res.to(Out_ptr.dtype.element_ty), mask=valid[:, None] & m_n[None, :])

    @triton.jit
    def moe_lora_dB_fp8_v2_kernel(
        G_fp8_ptr, S_g_ptr, Z_ptr, S_tok_ptr, Exp_ends_ptr, dB_ptr,
        scale_lora,
        N_dim: tl.constexpr, R_dim: tl.constexpr,
        USE_S_TOK: tl.constexpr,
        BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid_e = tl.program_id(0)
        pid_n = tl.program_id(1)
        start = tl.load(Exp_ends_ptr + pid_e - 1, mask=pid_e > 0, other=0)
        end = tl.load(Exp_ends_ptr + pid_e)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_r = tl.arange(0, R_dim)
        acc = tl.zeros([BLOCK_N, R_dim], dtype=tl.float32)
        for k0 in range(start, end, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            m_k = offs_k < end
            if USE_S_TOK:
                tok = tl.load(S_tok_ptr + offs_k, mask=m_k, other=-1)
                valid = m_k & (tok >= 0)
                z_tile = tl.load(Z_ptr + tok[:, None] * R_dim + offs_r[None, :], mask=valid[:, None], other=0.0).to(tl.bfloat16)
            else:
                valid = m_k
                z_tile = tl.load(Z_ptr + offs_k[:, None] * R_dim + offs_r[None, :], mask=valid[:, None], other=0.0).to(tl.bfloat16)
            sg = tl.load(S_g_ptr + offs_k, mask=valid, other=0.0).to(tl.float32)
            g_fp8 = tl.load(G_fp8_ptr + offs_k[None, :] * N_dim + offs_n[:, None], mask=valid[None, :] & (offs_n[:, None] < N_dim), other=0.0).to(tl.float32)
            g_bf = (g_fp8 * sg[None, :]).to(tl.bfloat16)
            acc += tl.dot(g_bf, z_tile)
        acc = acc * scale_lora
        tl.store(
            dB_ptr + pid_e * N_dim * R_dim + offs_n[:, None] * R_dim + offs_r[None, :],
            acc.to(dB_ptr.dtype.element_ty),
            mask=(offs_n[:, None] < N_dim),
        )

    @triton.jit
    def moe_lora_dA_dn_fp8_kernel(
        dZ_ptr, Act_fp8_ptr, S_act_ptr, Exp_ends_ptr, dA_ptr,
        E_m: tl.constexpr, I_dim: tl.constexpr, R_dim: tl.constexpr,
        BLOCK_I: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid_i = tl.program_id(0)
        act_tot = tl.load(Exp_ends_ptr + E_m - 1)
        offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
        offs_r = tl.arange(0, R_dim)
        acc = tl.zeros([R_dim, BLOCK_I], dtype=tl.float32)
        for k0 in range(0, act_tot, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            m_k = offs_k < act_tot
            dz_t = tl.load(dZ_ptr + offs_k[None, :] * R_dim + offs_r[:, None], mask=m_k[None, :], other=0.0).to(tl.bfloat16)
            sa = tl.load(S_act_ptr + offs_k, mask=m_k, other=0.0).to(tl.float32)
            a_fp8 = tl.load(Act_fp8_ptr + offs_k[:, None] * I_dim + offs_i[None, :], mask=m_k[:, None] & (offs_i[None, :] < I_dim), other=0.0).to(tl.float32)
            a_bf = (a_fp8 * sa[:, None]).to(tl.bfloat16)
            acc += tl.dot(dz_t, a_bf)
        tl.store(
            dA_ptr + offs_r[:, None] * I_dim + offs_i[None, :],
            acc.to(dA_ptr.dtype.element_ty),
            mask=(offs_i[None, :] < I_dim),
        )

    moe_cfg = {"top_p": 0.85, "k_min": 2}
    layer23_ctx = {"idx": None}

    def quant_w_fp8_per_channel(w):
        wf = w.float()
        s = wf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 448.0
        q = (wf / s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        return q.transpose(1, 2), s.squeeze(-1).contiguous()

    class PatchedFP8Experts(nn.Module):
        def __init__(self, orig_experts, enable_lora=False, enable_grad=False, r_lora=16, alpha_lora=32.0):
            super().__init__()
            self.num_experts = orig_experts.gate_up_proj.shape[0]
            self.enable_lora = enable_lora
            self.enable_grad = enable_grad or enable_lora
            self.r_lora = r_lora
            self.lora_scale = float(alpha_lora) / float(r_lora)

            w_gu = orig_experts.gate_up_proj.data
            w_dn = orig_experts.down_proj.data
            w_gu_fp8_T, s_gu = quant_w_fp8_per_channel(w_gu)
            w_dn_fp8_T, s_dn = quant_w_fp8_per_channel(w_dn)
            self.register_buffer("w_gu_fp8_T", w_gu_fp8_T)
            self.register_buffer("s_gu", s_gu)
            self.register_buffer("w_dn_fp8_T", w_dn_fp8_T)
            self.register_buffer("s_dn", s_dn)

            if self.enable_grad:
                w_dn_T_fp8_T, s_dn_T = quant_w_fp8_per_channel(w_dn.transpose(1, 2).contiguous())
                w_gu_T_fp8_T, s_gu_T = quant_w_fp8_per_channel(w_gu.transpose(1, 2).contiguous())
                self.register_buffer("w_dn_T_fp8_T", w_dn_T_fp8_T)
                self.register_buffer("s_dn_T", s_dn_T)
                self.register_buffer("w_gu_T_fp8_T", w_gu_T_fp8_T)
                self.register_buffer("s_gu_T", s_gu_T)
            else:
                self.register_buffer("w_dn_T_fp8_T", None)
                self.register_buffer("s_dn_T", None)
                self.register_buffer("w_gu_T_fp8_T", None)
                self.register_buffer("s_gu_T", None)

            if enable_lora:
                H_m = w_gu.shape[2]
                I_m = w_dn.shape[2]
                self.lora_A_gu = nn.Parameter(torch.randn(r_lora, H_m, device=w_gu.device, dtype=dtype) * 0.01)
                self.lora_B_gu = nn.Parameter(torch.zeros(self.num_experts, 2 * I_m, r_lora, device=w_gu.device, dtype=dtype))
                self.lora_A_dn = nn.Parameter(torch.randn(r_lora, I_m, device=w_gu.device, dtype=dtype) * 0.01)
                self.lora_B_dn = nn.Parameter(torch.zeros(self.num_experts, H_m, r_lora, device=w_gu.device, dtype=dtype))
            else:
                self.lora_A_gu = self.lora_B_gu = self.lora_A_dn = self.lora_B_dn = None

        @torch.no_grad()
        def refresh_la_dn_fp8(self):
            return

        def forward(self, hidden_states, top_k_index, top_k_weights):
            E_m = self.num_experts
            top_p = float(moe_cfg["top_p"])
            k_min = int(moe_cfg["k_min"])
            has_lora = self.enable_lora
            lora_scale = self.lora_scale
            w_gu_fp8_T, s_gu = self.w_gu_fp8_T, self.s_gu
            w_dn_fp8_T, s_dn = self.w_dn_fp8_T, self.s_dn
            w_dn_T_fp8_T, s_dn_T = self.w_dn_T_fp8_T, self.s_dn_T
            w_gu_T_fp8_T, s_gu_T = self.w_gu_T_fp8_T, self.s_gu_T

            if not torch.is_grad_enabled():
                x_fp8, sx, exp_ends, act_tok, s_tok, s_e, s_w, inv_idx = ops.moe_route_and_quant_fp8(
                    hidden_states, top_k_index, top_k_weights, E_m, top_p, k_min
                )
                gu = torch._scaled_grouped_mm(x_fp8, w_gu_fp8_T, sx, s_gu, offs=exp_ends, out_dtype=dtype, use_fast_accum=True)
                del x_fp8, sx
                if has_lora:
                    z_gu = torch.matmul(hidden_states, self.lora_A_gu.t())
                    sp_e = max(1, triton.cdiv(hidden_states.shape[0], 1024))
                    moe_lora_b_add_triton_kernel[(E_m, sp_e, triton.cdiv(gu.shape[1], 128))](
                        gu, z_gu, self.lora_B_gu, s_tok, exp_ends, lora_scale,
                        N_dim=gu.shape[1], R_dim=16, USE_S_TOK=True, BLOCK_M=64, M_SPLIT=1024, BLOCK_N=128, num_warps=4,
                    )
                    del z_gu
                    act_fp8, s_act, _ = ops.moe_gu_lora_swiglu_fp8_fwd(
                        gu, self.lora_A_dn, self.lora_B_gu, self.lora_A_dn, s_tok, s_e, act_tok, lora_scale, False
                    )
                    del gu
                    y_exp = torch._scaled_grouped_mm(act_fp8, w_dn_fp8_T, s_act, s_dn, offs=exp_ends, out_dtype=dtype, use_fast_accum=True)
                    z_dn = torch.empty((act_fp8.shape[0], 16), device=act_fp8.device, dtype=dtype)
                    moe_z_dn_fp8_kernel[(triton.cdiv(act_fp8.shape[0], 64),)](
                        act_fp8, s_act, self.lora_A_dn, exp_ends, z_dn,
                        E_m=E_m, I_dim=self.lora_A_dn.shape[1], R_dim=16, BLOCK_M=64, BLOCK_K=128, num_warps=4, num_stages=2,
                    )
                    del act_fp8, s_act
                    moe_lora_b_add_triton_kernel[(E_m, sp_e, triton.cdiv(y_exp.shape[1], 128))](
                        y_exp, z_dn, self.lora_B_dn, s_tok, exp_ends, lora_scale,
                        N_dim=y_exp.shape[1], R_dim=16, USE_S_TOK=False, BLOCK_M=64, M_SPLIT=1024, BLOCK_N=128, num_warps=4,
                    )
                    return ops.moe_dn_lora_scatter_fwd(y_exp, z_dn, self.lora_B_dn, inv_idx, s_tok, s_e, act_tok, top_k_weights, lora_scale, False)
                else:
                    dummy = hidden_states.new_empty((1, 16))
                    act_fp8, s_act, _ = ops.moe_gu_lora_swiglu_fp8_fwd(
                        gu, dummy, dummy, dummy, s_tok, s_e, act_tok, 0.0, False
                    )
                    del gu
                    y_exp = torch._scaled_grouped_mm(act_fp8, w_dn_fp8_T, s_act, s_dn, offs=exp_ends, out_dtype=dtype, use_fast_accum=True)
                    del act_fp8, s_act
                    return ops.moe_dn_lora_scatter_fwd(y_exp, dummy, dummy, inv_idx, s_tok, s_e, act_tok, top_k_weights, 0.0, False)

            class _FP8ExpertsFn(torch.autograd.Function):
                @staticmethod
                def forward(ctx, h_in, t_w, la_gu, lb_gu, la_dn, lb_dn):
                    x_fp8, sx, exp_ends, act_tok, s_tok, s_e, s_w, inv_idx = ops.moe_route_and_quant_fp8(
                        h_in, top_k_index, t_w, E_m, top_p, k_min
                    )
                    gu = torch._scaled_grouped_mm(x_fp8, w_gu_fp8_T, sx, s_gu, offs=exp_ends, out_dtype=dtype, use_fast_accum=True)
                    if has_lora:
                        z_gu = torch.matmul(h_in, la_gu.t())
                        sp_e = max(1, triton.cdiv(h_in.shape[0], 1024))
                        moe_lora_b_add_triton_kernel[(E_m, sp_e, triton.cdiv(gu.shape[1], 128))](
                            gu, z_gu, lb_gu, s_tok, exp_ends, lora_scale,
                            N_dim=gu.shape[1], R_dim=16, USE_S_TOK=True, BLOCK_M=64, M_SPLIT=1024, BLOCK_N=128, num_warps=4,
                        )
                        act_fp8, s_act, _ = ops.moe_gu_lora_swiglu_fp8_fwd(
                            gu, z_gu, lb_gu, la_dn, s_tok, s_e, act_tok, lora_scale, False
                        )
                        y_exp = torch._scaled_grouped_mm(act_fp8, w_dn_fp8_T, s_act, s_dn, offs=exp_ends, out_dtype=dtype, use_fast_accum=True)
                        z_dn = torch.empty((act_fp8.shape[0], 16), device=act_fp8.device, dtype=dtype)
                        moe_z_dn_fp8_kernel[(triton.cdiv(act_fp8.shape[0], 64),)](
                            act_fp8, s_act, la_dn, exp_ends, z_dn,
                            E_m=E_m, I_dim=la_dn.shape[1], R_dim=16, BLOCK_M=64, BLOCK_K=128, num_warps=4, num_stages=2,
                        )
                        moe_lora_b_add_triton_kernel[(E_m, sp_e, triton.cdiv(y_exp.shape[1], 128))](
                            y_exp, z_dn, lb_dn, s_tok, exp_ends, lora_scale,
                            N_dim=y_exp.shape[1], R_dim=16, USE_S_TOK=False, BLOCK_M=64, M_SPLIT=1024, BLOCK_N=128, num_warps=4,
                        )
                        y_out = ops.moe_dn_lora_scatter_fwd(y_exp, z_dn, lb_dn, inv_idx, s_tok, s_e, act_tok, t_w, lora_scale, False)
                        ctx.save_for_backward(h_in, gu, y_exp, act_fp8, s_act, z_gu, z_dn, exp_ends, act_tok, s_tok, s_e, s_w, inv_idx, la_gu, lb_gu, la_dn, lb_dn)
                    else:
                        dummy = h_in.new_empty((1, 16))
                        act_fp8, s_act, _ = ops.moe_gu_lora_swiglu_fp8_fwd(
                            gu, dummy, dummy, dummy, s_tok, s_e, act_tok, 0.0, False
                        )
                        y_exp = torch._scaled_grouped_mm(act_fp8, w_dn_fp8_T, s_act, s_dn, offs=exp_ends, out_dtype=dtype, use_fast_accum=True)
                        y_out = ops.moe_dn_lora_scatter_fwd(y_exp, dummy, dummy, inv_idx, s_tok, s_e, act_tok, t_w, 0.0, False)
                        ctx.save_for_backward(h_in, gu, y_exp, act_fp8, s_act, exp_ends, act_tok, s_tok, s_e, s_w, inv_idx)
                    return y_out

                @staticmethod
                def backward(ctx, grad_out):
                    if not has_lora:
                        h_in, gu, y_exp, act_fp8, s_act, exp_ends, act_tok, s_tok, s_e, s_w, inv_idx = ctx.saved_tensors
                        dummy = y_exp.new_empty((1, 16))
                        dy_fp8, s_dy, dw_out, _ = ops.moe_bwd_gather_dw_quant_fp8(
                            grad_out.contiguous(), y_exp, dummy, s_tok, s_e, s_w, inv_idx, act_tok, 0.0, False
                        )
                        d_act = torch._scaled_grouped_mm(dy_fp8, w_dn_T_fp8_T, s_dy, s_dn_T, offs=exp_ends, out_dtype=dtype, use_fast_accum=True)
                        dgu_fp8, s_dgu, _ = ops.moe_bwd_swiglu_quant_fp8(
                            gu, d_act, dummy, dummy, dummy, s_e, act_tok, 0.0, False
                        )
                        dx_exp = torch._scaled_grouped_mm(dgu_fp8, w_gu_T_fp8_T, s_dgu, s_gu_T, offs=exp_ends, out_dtype=dtype, use_fast_accum=True)
                        dx_out, _ = ops.moe_bwd_scatter_dx(dx_exp, dummy, inv_idx, False)
                        return dx_out, dw_out.to(grad_out.dtype), None, None, None, None

                    h_in, gu, y_exp, act_fp8, s_act, z_gu, z_dn, exp_ends, act_tok, s_tok, s_e, s_w, inv_idx, la_gu, lb_gu, la_dn, lb_dn = ctx.saved_tensors
                    dy_fp8, s_dy, dw_out, dz_dn = ops.moe_bwd_gather_dw_quant_fp8(
                        grad_out.contiguous(), y_exp, lb_dn, s_tok, s_e, s_w, inv_idx, act_tok, lora_scale, True
                    )
                    # Zero-alloc d_lb_dn [E, H, R] directly from FP8 + scale (tuned 64x32x4w2s tile)
                    d_lb_dn = torch.empty_like(lb_dn)
                    moe_lora_dB_fp8_v2_kernel[(E_m, triton.cdiv(lb_dn.shape[1], 64))](
                        dy_fp8, s_dy, z_dn, s_tok, exp_ends, d_lb_dn,
                        lora_scale, N_dim=lb_dn.shape[1], R_dim=16, USE_S_TOK=False, BLOCK_N=64, BLOCK_K=32, num_warps=4, num_stages=2,
                    )
                    # Zero-alloc d_la_dn [R, I] directly from FP8 + scale (bounded by exp_ends[-1], tuned 64x128x4w2s tile)
                    d_la_dn = torch.empty_like(la_dn)
                    moe_lora_dA_dn_fp8_kernel[(triton.cdiv(la_dn.shape[1], 64),)](
                        dz_dn, act_fp8, s_act, exp_ends, d_la_dn,
                        E_m=E_m, I_dim=la_dn.shape[1], R_dim=16, BLOCK_I=64, BLOCK_K=128, num_warps=4, num_stages=2,
                    )

                    d_act = torch._scaled_grouped_mm(dy_fp8, w_dn_T_fp8_T, s_dy, s_dn_T, offs=exp_ends, out_dtype=dtype, use_fast_accum=True)
                    moe_dact_add_lora_kernel[(triton.cdiv(d_act.shape[0], 64), triton.cdiv(la_dn.shape[1], 128))](
                        d_act, dz_dn, la_dn, exp_ends, E_m=E_m, I_dim=la_dn.shape[1], R_dim=16, BLOCK_M=64, BLOCK_I=128, num_warps=4,
                    )
                    dgu_fp8, s_dgu, dz_gu = ops.moe_bwd_swiglu_quant_fp8(
                        gu, d_act, dz_dn, la_dn, lb_gu, s_e, act_tok, lora_scale, True
                    )
                    # Zero-alloc d_lb_gu [E, 2I, R] directly from FP8 + scale (tuned 64x32x4w2s tile)
                    d_lb_gu = torch.empty_like(lb_gu)
                    moe_lora_dB_fp8_v2_kernel[(E_m, triton.cdiv(lb_gu.shape[1], 64))](
                        dgu_fp8, s_dgu, z_gu, s_tok, exp_ends, d_lb_gu,
                        lora_scale, N_dim=lb_gu.shape[1], R_dim=16, USE_S_TOK=True, BLOCK_N=64, BLOCK_K=32, num_warps=4, num_stages=2,
                    )
                    dx_exp = torch._scaled_grouped_mm(dgu_fp8, w_gu_T_fp8_T, s_dgu, s_gu_T, offs=exp_ends, out_dtype=dtype, use_fast_accum=True)
                    dx_out, dz_gu_tok = ops.moe_bwd_scatter_dx(dx_exp, dz_gu, inv_idx, True)
                    dx_out.addmm_(dz_gu_tok, la_gu)
                    d_la_gu = torch.matmul(dz_gu_tok.t(), h_in)
                    return dx_out, dw_out.to(grad_out.dtype), d_la_gu, d_lb_gu, d_la_dn, d_lb_dn

            if has_lora:
                return _FP8ExpertsFn.apply(
                    hidden_states, top_k_weights, self.lora_A_gu, self.lora_B_gu, self.lora_A_dn, self.lora_B_dn
                )
            dummy_p = hidden_states.new_empty((1, 16))
            return _FP8ExpertsFn.apply(hidden_states, top_k_weights, dummy_p, dummy_p, dummy_p, dummy_p)

    # =========================================================================
    # 4. DEPTH ATTENTION V6 (1-Pass BF16 Pair Kernel: ZERO torch.cat, Fused RMSNorm!)
    # =========================================================================
    @triton.jit
    def depth_attn_pair_fwd_bf16_kernel(
        Q_ptr, B_ptr, L_ptr, P_ptr, InvR_ptr, Out_ptr,
        stride_qm, stride_qd, stride_bn, stride_bk, stride_bd,
        stride_ln, stride_ld, stride_pn, stride_pm, stride_pk,
        stride_irn, stride_irk, stride_on, stride_om, stride_od,
        D_dim, scale, eps,
        K_PREV: tl.constexpr, M_Q: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < D_dim
        inv_d = 1.0 / D_dim
        bb = B_ptr + pid_n * stride_bn
        irb = InvR_ptr + pid_n * stride_irn

        xl_raw = tl.load(L_ptr + pid_n * stride_ln + offs_d * stride_ld, mask=mask_d, other=0.0).to(tl.float32)
        ir_l = tl.rsqrt(tl.sum(xl_raw * xl_raw, axis=0) * inv_d + eps)
        xl = xl_raw * ir_l
        tl.store(irb + K_PREV * stride_irk, ir_l)

        x0 = tl.zeros([BLOCK_D], dtype=tl.float32)
        x1 = tl.zeros([BLOCK_D], dtype=tl.float32)
        x2 = tl.zeros([BLOCK_D], dtype=tl.float32)
        x3 = tl.zeros([BLOCK_D], dtype=tl.float32)
        if K_PREV >= 1:
            r0 = tl.load(bb + 0 * stride_bk + offs_d * stride_bd, mask=mask_d, other=0.0).to(tl.float32)
            ir0 = tl.rsqrt(tl.sum(r0 * r0, axis=0) * inv_d + eps)
            x0 = r0 * ir0
            tl.store(irb + 0 * stride_irk, ir0)
        if K_PREV >= 2:
            r1 = tl.load(bb + 1 * stride_bk + offs_d * stride_bd, mask=mask_d, other=0.0).to(tl.float32)
            ir1 = tl.rsqrt(tl.sum(r1 * r1, axis=0) * inv_d + eps)
            x1 = r1 * ir1
            tl.store(irb + 1 * stride_irk, ir1)
        if K_PREV >= 3:
            r2 = tl.load(bb + 2 * stride_bk + offs_d * stride_bd, mask=mask_d, other=0.0).to(tl.float32)
            ir2 = tl.rsqrt(tl.sum(r2 * r2, axis=0) * inv_d + eps)
            x2 = r2 * ir2
            tl.store(irb + 2 * stride_irk, ir2)
        if K_PREV >= 4:
            r3 = tl.load(bb + 3 * stride_bk + offs_d * stride_bd, mask=mask_d, other=0.0).to(tl.float32)
            ir3 = tl.rsqrt(tl.sum(r3 * r3, axis=0) * inv_d + eps)
            x3 = r3 * ir3
            tl.store(irb + 3 * stride_irk, ir3)

        for m in tl.static_range(M_Q):
            qm = tl.load(Q_ptr + m * stride_qm + offs_d * stride_qd, mask=mask_d, other=0.0).to(tl.float32)
            sl = tl.sum(qm * xl, axis=0) * scale
            s0 = tl.sum(qm * x0, axis=0) * scale if K_PREV >= 1 else -1e9
            s1 = tl.sum(qm * x1, axis=0) * scale if K_PREV >= 2 else -1e9
            s2 = tl.sum(qm * x2, axis=0) * scale if K_PREV >= 3 else -1e9
            s3 = tl.sum(qm * x3, axis=0) * scale if K_PREV >= 4 else -1e9
            smax = sl
            if K_PREV >= 1: smax = tl.maximum(smax, s0)
            if K_PREV >= 2: smax = tl.maximum(smax, s1)
            if K_PREV >= 3: smax = tl.maximum(smax, s2)
            if K_PREV >= 4: smax = tl.maximum(smax, s3)
            el = tl.exp(sl - smax)
            e0 = tl.exp(s0 - smax) if K_PREV >= 1 else 0.0
            e1 = tl.exp(s1 - smax) if K_PREV >= 2 else 0.0
            e2 = tl.exp(s2 - smax) if K_PREV >= 3 else 0.0
            e3 = tl.exp(s3 - smax) if K_PREV >= 4 else 0.0
            inv_s = 1.0 / (el + e0 + e1 + e2 + e3)
            pl, p0, p1, p2, p3 = el * inv_s, e0 * inv_s, e1 * inv_s, e2 * inv_s, e3 * inv_s
            pb = P_ptr + pid_n * stride_pn + m * stride_pm
            tl.store(pb + K_PREV * stride_pk, pl)
            if K_PREV >= 1: tl.store(pb + 0 * stride_pk, p0)
            if K_PREV >= 2: tl.store(pb + 1 * stride_pk, p1)
            if K_PREV >= 3: tl.store(pb + 2 * stride_pk, p2)
            if K_PREV >= 4: tl.store(pb + 3 * stride_pk, p3)
            ov = pl * xl
            if K_PREV >= 1: ov += p0 * x0
            if K_PREV >= 2: ov += p1 * x1
            if K_PREV >= 3: ov += p2 * x2
            if K_PREV >= 4: ov += p3 * x3
            tl.store(Out_ptr + pid_n * stride_on + m * stride_om + offs_d * stride_od, ov.to(Out_ptr.dtype.element_ty), mask=mask_d)

    @triton.jit
    def depth_attn_pair_bwd_ds_kernel(
        B_ptr, L_ptr, InvR_ptr, P_ptr, dOut_ptr, dS_ptr,
        stride_bn, stride_bk, stride_bd, stride_ln, stride_ld,
        stride_irn, stride_irk, stride_pn, stride_pm, stride_pk,
        stride_don, stride_dom, stride_dod,
        D_dim, scale, K_PREV: tl.constexpr, M_Q: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < D_dim
        bb = B_ptr + pid_n * stride_bn
        irb = InvR_ptr + pid_n * stride_irn
        irl = tl.load(irb + K_PREV * stride_irk)
        xl = tl.load(L_ptr + pid_n * stride_ln + offs_d * stride_ld, mask=mask_d, other=0.0).to(tl.float32) * irl
        x0 = tl.load(bb + 0 * stride_bk + offs_d * stride_bd, mask=mask_d, other=0.0).to(tl.float32) * tl.load(irb + 0 * stride_irk) if K_PREV >= 1 else tl.zeros([BLOCK_D], dtype=tl.float32)
        x1 = tl.load(bb + 1 * stride_bk + offs_d * stride_bd, mask=mask_d, other=0.0).to(tl.float32) * tl.load(irb + 1 * stride_irk) if K_PREV >= 2 else tl.zeros([BLOCK_D], dtype=tl.float32)
        x2 = tl.load(bb + 2 * stride_bk + offs_d * stride_bd, mask=mask_d, other=0.0).to(tl.float32) * tl.load(irb + 2 * stride_irk) if K_PREV >= 3 else tl.zeros([BLOCK_D], dtype=tl.float32)
        x3 = tl.load(bb + 3 * stride_bk + offs_d * stride_bd, mask=mask_d, other=0.0).to(tl.float32) * tl.load(irb + 3 * stride_irk) if K_PREV >= 4 else tl.zeros([BLOCK_D], dtype=tl.float32)
        for m in tl.static_range(M_Q):
            dom = tl.load(dOut_ptr + pid_n * stride_don + m * stride_dom + offs_d * stride_dod, mask=mask_d, other=0.0).to(tl.float32)
            dpl = tl.sum(dom * xl, axis=0)
            dp0 = tl.sum(dom * x0, axis=0) if K_PREV >= 1 else 0.0
            dp1 = tl.sum(dom * x1, axis=0) if K_PREV >= 2 else 0.0
            dp2 = tl.sum(dom * x2, axis=0) if K_PREV >= 3 else 0.0
            dp3 = tl.sum(dom * x3, axis=0) if K_PREV >= 4 else 0.0
            pb = P_ptr + pid_n * stride_pn + m * stride_pm
            pl = tl.load(pb + K_PREV * stride_pk)
            p0 = tl.load(pb + 0 * stride_pk) if K_PREV >= 1 else 0.0
            p1 = tl.load(pb + 1 * stride_pk) if K_PREV >= 2 else 0.0
            p2 = tl.load(pb + 2 * stride_pk) if K_PREV >= 3 else 0.0
            p3 = tl.load(pb + 3 * stride_pk) if K_PREV >= 4 else 0.0
            dval = pl * dpl + p0 * dp0 + p1 * dp1 + p2 * dp2 + p3 * dp3
            dsb = dS_ptr + pid_n * stride_pn + m * stride_pm
            tl.store(dsb + K_PREV * stride_pk, pl * (dpl - dval) * scale)
            if K_PREV >= 1: tl.store(dsb + 0 * stride_pk, p0 * (dp0 - dval) * scale)
            if K_PREV >= 2: tl.store(dsb + 1 * stride_pk, p1 * (dp1 - dval) * scale)
            if K_PREV >= 3: tl.store(dsb + 2 * stride_pk, p2 * (dp2 - dval) * scale)
            if K_PREV >= 4: tl.store(dsb + 3 * stride_pk, p3 * (dp3 - dval) * scale)

    @triton.jit
    def depth_attn_pair_bwd_dx_kernel(
        Q_ptr, B_ptr, L_ptr, InvR_ptr, P_ptr, dS_ptr, dOut_ptr, dB_ptr, dL_ptr,
        stride_qm, stride_qd, stride_bn, stride_bk, stride_bd, stride_ln, stride_ld,
        stride_irn, stride_irk, stride_pn, stride_pm, stride_pk,
        stride_don, stride_dom, stride_dod, stride_dbn, stride_dbk, stride_dbd, stride_dln, stride_dld,
        D_dim, K_PREV: tl.constexpr, M_Q: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < D_dim
        inv_d = 1.0 / D_dim
        dy_l = tl.zeros([BLOCK_D], dtype=tl.float32)
        dy_0 = tl.zeros([BLOCK_D], dtype=tl.float32)
        dy_1 = tl.zeros([BLOCK_D], dtype=tl.float32)
        dy_2 = tl.zeros([BLOCK_D], dtype=tl.float32)
        dy_3 = tl.zeros([BLOCK_D], dtype=tl.float32)
        for m in tl.static_range(M_Q):
            dom = tl.load(dOut_ptr + pid_n * stride_don + m * stride_dom + offs_d * stride_dod, mask=mask_d, other=0.0).to(tl.float32)
            qm = tl.load(Q_ptr + m * stride_qm + offs_d * stride_qd, mask=mask_d, other=0.0).to(tl.float32)
            pb = P_ptr + pid_n * stride_pn + m * stride_pm
            dsb = dS_ptr + pid_n * stride_pn + m * stride_pm
            dy_l += tl.load(pb + K_PREV * stride_pk) * dom + tl.load(dsb + K_PREV * stride_pk) * qm
            if K_PREV >= 1: dy_0 += tl.load(pb + 0 * stride_pk) * dom + tl.load(dsb + 0 * stride_pk) * qm
            if K_PREV >= 2: dy_1 += tl.load(pb + 1 * stride_pk) * dom + tl.load(dsb + 1 * stride_pk) * qm
            if K_PREV >= 3: dy_2 += tl.load(pb + 2 * stride_pk) * dom + tl.load(dsb + 2 * stride_pk) * qm
            if K_PREV >= 4: dy_3 += tl.load(pb + 3 * stride_pk) * dom + tl.load(dsb + 3 * stride_pk) * qm

        irb = InvR_ptr + pid_n * stride_irn
        irl = tl.load(irb + K_PREV * stride_irk)
        yl = tl.load(L_ptr + pid_n * stride_ln + offs_d * stride_ld, mask=mask_d, other=0.0).to(tl.float32) * irl
        dxl = irl * (dy_l - yl * (tl.sum(dy_l * yl, axis=0) * inv_d))
        tl.store(dL_ptr + pid_n * stride_dln + offs_d * stride_dld, dxl.to(dL_ptr.dtype.element_ty), mask=mask_d)

        bb = B_ptr + pid_n * stride_bn
        dbb = dB_ptr + pid_n * stride_dbn
        if K_PREV >= 1:
            ir0 = tl.load(irb + 0 * stride_irk)
            y0 = tl.load(bb + 0 * stride_bk + offs_d * stride_bd, mask=mask_d, other=0.0).to(tl.float32) * ir0
            dx0 = ir0 * (dy_0 - y0 * (tl.sum(dy_0 * y0, axis=0) * inv_d))
            tl.store(dbb + 0 * stride_dbk + offs_d * stride_dbd, dx0.to(dB_ptr.dtype.element_ty), mask=mask_d)
        if K_PREV >= 2:
            ir1 = tl.load(irb + 1 * stride_irk)
            y1 = tl.load(bb + 1 * stride_bk + offs_d * stride_bd, mask=mask_d, other=0.0).to(tl.float32) * ir1
            dx1 = ir1 * (dy_1 - y1 * (tl.sum(dy_1 * y1, axis=0) * inv_d))
            tl.store(dbb + 1 * stride_dbk + offs_d * stride_dbd, dx1.to(dB_ptr.dtype.element_ty), mask=mask_d)
        if K_PREV >= 3:
            ir2 = tl.load(irb + 2 * stride_irk)
            y2 = tl.load(bb + 2 * stride_bk + offs_d * stride_bd, mask=mask_d, other=0.0).to(tl.float32) * ir2
            dx2 = ir2 * (dy_2 - y2 * (tl.sum(dy_2 * y2, axis=0) * inv_d))
            tl.store(dbb + 2 * stride_dbk + offs_d * stride_dbd, dx2.to(dB_ptr.dtype.element_ty), mask=mask_d)
        if K_PREV >= 4:
            ir3 = tl.load(irb + 3 * stride_irk)
            y3 = tl.load(bb + 3 * stride_bk + offs_d * stride_bd, mask=mask_d, other=0.0).to(tl.float32) * ir3
            dx3 = ir3 * (dy_3 - y3 * (tl.sum(dy_3 * y3, axis=0) * inv_d))
            tl.store(dbb + 3 * stride_dbk + offs_d * stride_dbd, dx3.to(dB_ptr.dtype.element_ty), mask=mask_d)

    class TritonDepthAttnPairFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q_2d, b_3d, l_2d, scale, eps, out_mn_first):
            q_c = q_2d.float().contiguous()
            M_q, D_dim = q_c.shape
            N_tok, K_prev, _ = b_3d.shape
            K_tot = K_prev + 1
            P_buf = torch.empty((N_tok, M_q, K_tot), device=l_2d.device, dtype=torch.float32)
            IR_buf = torch.empty((N_tok, K_tot), device=l_2d.device, dtype=torch.float32)
            if out_mn_first:
                out = torch.empty((M_q, N_tok, D_dim), device=l_2d.device, dtype=l_2d.dtype)
                s_on, s_om, s_od = out.stride(1), out.stride(0), out.stride(2)
            else:
                out = torch.empty((N_tok, M_q, D_dim), device=l_2d.device, dtype=l_2d.dtype)
                s_on, s_om, s_od = out.stride(0), out.stride(1), out.stride(2)
            depth_attn_pair_fwd_bf16_kernel[(N_tok,)](
                q_c, b_3d, l_2d, P_buf, IR_buf, out,
                q_c.stride(0), q_c.stride(1), b_3d.stride(0), b_3d.stride(1), b_3d.stride(2),
                l_2d.stride(0), l_2d.stride(1), P_buf.stride(0), P_buf.stride(1), P_buf.stride(2),
                IR_buf.stride(0), IR_buf.stride(1), s_on, s_om, s_od,
                D_dim, scale, eps, K_PREV=K_prev, M_Q=M_q, BLOCK_D=4096, num_warps=8,
            )
            ctx.save_for_backward(q_c, b_3d, l_2d, P_buf, IR_buf)
            ctx.meta = (scale, K_prev, M_q, D_dim, out_mn_first)
            return out

        @staticmethod
        def backward(ctx, grad_out):
            q_c, b_3d, l_2d, P_buf, IR_buf = ctx.saved_tensors
            scale, K_prev, M_q, D_dim, out_mn_first = ctx.meta
            do_c = grad_out.contiguous()
            s_don, s_dom, s_dod = (do_c.stride(1), do_c.stride(0), do_c.stride(2)) if out_mn_first else (do_c.stride(0), do_c.stride(1), do_c.stride(2))
            N_tok = l_2d.shape[0]
            dS = torch.empty((N_tok, M_q, K_prev + 1), device=l_2d.device, dtype=torch.float32)
            dB = torch.empty((N_tok, K_prev, D_dim), device=l_2d.device, dtype=b_3d.dtype)
            dL = torch.empty((N_tok, D_dim), device=l_2d.device, dtype=l_2d.dtype)
            depth_attn_pair_bwd_ds_kernel[(N_tok,)](
                b_3d, l_2d, IR_buf, P_buf, do_c, dS,
                b_3d.stride(0), b_3d.stride(1), b_3d.stride(2), l_2d.stride(0), l_2d.stride(1),
                IR_buf.stride(0), IR_buf.stride(1), P_buf.stride(0), P_buf.stride(1), P_buf.stride(2),
                s_don, s_dom, s_dod,
                D_dim, scale, K_PREV=K_prev, M_Q=M_q, BLOCK_D=4096, num_warps=8,
            )
            depth_attn_pair_bwd_dx_kernel[(N_tok,)](
                q_c, b_3d, l_2d, IR_buf, P_buf, dS, do_c, dB, dL,
                q_c.stride(0), q_c.stride(1), b_3d.stride(0), b_3d.stride(1), b_3d.stride(2),
                l_2d.stride(0), l_2d.stride(1), IR_buf.stride(0), IR_buf.stride(1),
                P_buf.stride(0), P_buf.stride(1), P_buf.stride(2),
                s_don, s_dom, s_dod,
                dB.stride(0), dB.stride(1), dB.stride(2), dL.stride(0), dL.stride(1),
                D_dim, K_PREV=K_prev, M_Q=M_q, BLOCK_D=4096, num_warps=8,
            )
            return None, dB, dL, None, None, None

    def _run_depth_attn_pair(self_da, block_states, partial_block_states, q_2d, out_mn_first=False):
        orig_prefix = partial_block_states.shape[:-1]
        hdim = partial_block_states.shape[-1]
        K_prev = block_states.shape[-2]
        eps = float(getattr(self_da.norm, "eps", 1e-6))
        if K_prev == 0:
            p_f = partial_block_states.float()
            norm_p = (p_f * torch.rsqrt(p_f.square().mean(-1, keepdim=True) + eps)).to(partial_block_states.dtype)
            if out_mn_first:
                return norm_p.unsqueeze(0).expand(q_2d.shape[0], *orig_prefix, hdim)
            return norm_p.unsqueeze(-2).expand(*orig_prefix, q_2d.shape[0], hdim).contiguous()
        if K_prev <= 4 and hdim <= 4096:
            b_3d = block_states.reshape(-1, K_prev, hdim)
            l_2d = partial_block_states.reshape(-1, hdim)
            out = TritonDepthAttnPairFn.apply(q_2d, b_3d, l_2d, hdim ** -0.5, eps, out_mn_first)
            if out_mn_first:
                return out.reshape(q_2d.shape[0], *orig_prefix, hdim)
            return out.reshape(*orig_prefix, q_2d.shape[0], hdim)
        # Fallback for K_prev > 4
        stacked = self_da.norm(torch.cat([block_states, partial_block_states.unsqueeze(-2)], dim=-2))
        q_bf = q_2d.to(stacked.dtype)
        x_3d = stacked.reshape(-1, K_prev + 1, hdim)
        probs = torch.softmax(torch.matmul(x_3d, q_bf.t()).float() * (hdim ** -0.5), dim=1)
        out = torch.bmm(probs.transpose(1, 2).to(stacked.dtype), x_3d)
        res = out.reshape(*orig_prefix, q_2d.shape[0], hdim)
        if out_mn_first:
            return res.movedim(-2, 0).contiguous()
        return res

    def patched_block_attn_res_forward(self, block_states, partial_block_states):
        hdim = partial_block_states.shape[-1]
        return _run_depth_attn_pair(self, block_states, partial_block_states, self.query.view(-1, hdim), out_mn_first=True).unbind(0)

    def patched_message_attn_res_forward(self, block_states, partial_block_states, loop_layer_idx):
        return _run_depth_attn_pair(self, block_states, partial_block_states, self.query[loop_layer_idx], out_mn_first=False)

    ink_mod.Inkling2LoopBlockAttentionResidual.forward = patched_block_attn_res_forward
    ink_mod.Inkling2LoopMessageAttentionResidual.forward = patched_message_attn_res_forward

    # =========================================================================
    # 5. LOAD MODEL + PATCH STACK (FP8 MoE, C++ FA, FP8 DoRA, CausalConv1d)
    # =========================================================================
    t_load = time.time()
    m = ModelCls.from_pretrained("/model", dtype=dtype, device_map="cuda")
    n_vis_before = torch.cuda.memory_allocated()
    if hasattr(m.model, "vision_tower"):
        del m.model.vision_tower
    gc.collect()
    torch.cuda.empty_cache()
    print(f"=== VISION TOWER FREED: {(n_vis_before - torch.cuda.memory_allocated())/1e9:.2f}GB ===", flush=True)
    m.config.use_cache = False
    m.model.language_model.config.use_cache = False

    # Patch Attention
    ink_mod.Inkling2LoopSlidingAttention.forward = patched_sliding_attn_forward
    ink_mod.Inkling2LoopFullAttention.forward = patched_full_attn_forward

    # Patch MoE Experts (FP8 Dense + LoRA on layers moe_lora_start..23)
    moe_lora_start = max(freeze_until, int(moe_lora_start))
    for idx, layer in enumerate(m.model.language_model.layers):
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
            orig_exp = layer.mlp.experts
            layer.mlp.experts = PatchedFP8Experts(orig_exp, enable_lora=(idx >= moe_lora_start), enable_grad=(idx >= freeze_until), r_lora=16, alpha_lora=32.0)
            del orig_exp
    gc.collect()
    torch.cuda.empty_cache()

    # Patch Layer 23 MLP token pruning
    orig_l23_mlp = m.model.language_model.layers[-1].mlp
    orig_l23_fwd = orig_l23_mlp.forward
    def patched_l23_mlp_forward(hidden_states):
        idx = layer23_ctx["idx"]
        if idx is None or not m.training:
            return orig_l23_fwd(hidden_states)
        B_l, S_l, H_l = hidden_states.shape
        h_flat = hidden_states.reshape(-1, H_l)
        h_sel = h_flat[idx].unsqueeze(0)
        y_sel = orig_l23_fwd(h_sel).squeeze(0)
        y_full = torch.zeros_like(h_flat)
        y_full[idx] = y_sel
        return y_full.reshape(B_l, S_l, H_l)
    orig_l23_mlp.forward = patched_l23_mlp_forward

    @triton.jit
    def fp8_amax_part_kernel(X_ptr, Part_ptr, N_elems, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        n_ctas = tl.num_programs(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        stride = n_ctas * BLOCK
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for i in range(0, N_elems, stride):
            idx = offs + i
            v = tl.load(X_ptr + idx, mask=idx < N_elems, other=0.0).to(tl.float32)
            acc = tl.maximum(acc, tl.abs(v))
        tl.store(Part_ptr + pid, tl.max(acc, axis=0))

    @triton.jit
    def fp8_quant_from_part_kernel(X_ptr, Part_ptr, Scale_ptr, Out_ptr, N_elems, BLOCK: tl.constexpr, N_PART: tl.constexpr):
        pid = tl.program_id(0)
        offs_p = tl.arange(0, N_PART)
        amax = tl.max(tl.load(Part_ptr + offs_p), axis=0)
        scale = tl.maximum(amax, 1e-8) / 448.0
        if pid == 0:
            tl.store(Scale_ptr, scale)
        inv_s = 1.0 / scale
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < N_elems
        x = tl.load(X_ptr + offs, mask=m, other=0.0).to(tl.float32)
        tl.store(Out_ptr + offs, tl.minimum(tl.maximum(x * inv_s, -448.0), 448.0).to(tl.float8e4nv), mask=m)

    @triton.jit
    def dora_1x_lora_scale_inplace_kernel(
        Out_ptr, H_a_ptr, B_ptr, M_ptr, InvN_ptr,
        M_dim, N_dim, lora_scale,
        R_LORA: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_r = tl.arange(0, R_LORA)
        m_m = offs_m < M_dim
        m_n = offs_n < N_dim
        h_tile = tl.load(H_a_ptr + offs_m[:, None] * R_LORA + offs_r[None, :], mask=m_m[:, None], other=0.0).to(tl.bfloat16)
        b_tile = tl.load(B_ptr + offs_n[:, None] * R_LORA + offs_r[None, :], mask=m_n[:, None], other=0.0).to(tl.bfloat16)
        y_base = tl.load(Out_ptr + offs_m[:, None] * N_dim + offs_n[None, :], mask=m_m[:, None] & m_n[None, :], other=0.0).to(tl.float32)
        y_pre = y_base + tl.dot(h_tile, tl.trans(b_tile), out_dtype=tl.float32) * lora_scale
        s = tl.load(M_ptr + offs_n, mask=m_n, other=1.0).to(tl.float32) * tl.load(InvN_ptr + offs_n, mask=m_n, other=1.0).to(tl.float32)
        tl.store(Out_ptr + offs_m[:, None] * N_dim + offs_n[None, :], (y_pre * s[None, :]).to(Out_ptr.dtype.element_ty), mask=m_m[:, None] & m_n[None, :])

    @triton.jit
    def dora_1x_bwd_scale_dm_kernel(
        GradOut_ptr, Out_ptr, M_ptr, InvN_ptr, dY_pre_ptr, Part_dM_ptr, Part_Max_ptr,
        M_dim, N_dim, CHUNK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        pid_cm, pid_n = tl.program_id(0), tl.program_id(1)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        m_n = offs_n < N_dim
        m_val = tl.load(M_ptr + offs_n, mask=m_n, other=1.0).to(tl.float32)
        inv_n = tl.load(InvN_ptr + offs_n, mask=m_n, other=1.0).to(tl.float32)
        s = m_val * inv_n
        inv_m = 1.0 / tl.maximum(m_val, 1e-8)
        dm_acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        amax_acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        m0 = pid_cm * CHUNK_M
        for step in range(CHUNK_M):
            m_idx = m0 + step
            if m_idx < M_dim:
                go = tl.load(GradOut_ptr + m_idx * N_dim + offs_n, mask=m_n, other=0.0).to(tl.float32)
                ov = tl.load(Out_ptr + m_idx * N_dim + offs_n, mask=m_n, other=0.0).to(tl.float32)
                dm_acc += go * ov
                dyp = go * s
                amax_acc = tl.maximum(amax_acc, tl.abs(dyp))
                tl.store(dY_pre_ptr + m_idx * N_dim + offs_n, dyp.to(dY_pre_ptr.dtype.element_ty), mask=m_n)
        tl.store(Part_dM_ptr + pid_cm * N_dim + offs_n, dm_acc * inv_m, mask=m_n)
        tl.atomic_max(Part_Max_ptr + pid_n, tl.max(amax_acc, axis=0))

    @triton.jit
    def fp8_quant_from_dynmax_kernel(X_ptr, PartMax_ptr, Scale_ptr, Out_ptr, N_elems, BLOCK: tl.constexpr, N_PMAX: tl.constexpr):
        pid = tl.program_id(0)
        offs_p = tl.arange(0, N_PMAX)
        amax = tl.max(tl.load(PartMax_ptr + offs_p), axis=0)
        scale = tl.maximum(amax, 1e-8) / 448.0
        if pid == 0:
            tl.store(Scale_ptr, scale)
        inv_s = 1.0 / scale
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < N_elems
        x = tl.load(X_ptr + offs, mask=m, other=0.0).to(tl.float32)
        tl.store(Out_ptr + offs, tl.minimum(tl.maximum(x * inv_s, -448.0), 448.0).to(tl.float8e4nv), mask=m)

    @triton.jit
    def fp8_quant_delayed_and_amax_kernel(X_ptr, Scale_ptr, NextMax_ptr, Out_ptr, N_elems, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        scale = tl.load(Scale_ptr)
        inv_s = 1.0 / tl.maximum(scale, 1e-12)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < N_elems
        x = tl.load(X_ptr + offs, mask=m, other=0.0).to(tl.float32)
        tl.store(Out_ptr + offs, tl.minimum(tl.maximum(x * inv_s, -448.0), 448.0).to(tl.float8e4nv), mask=m)
        tl.atomic_max(NextMax_ptr, tl.max(tl.abs(x), axis=0))

    @triton.jit
    def dora_1x_bwd_scale_dm_quant_delayed_kernel(
        GradOut_ptr, Out_ptr, M_ptr, InvN_ptr,
        ScaleDy_ptr, NextMaxDy_ptr,
        dY_pre_ptr, dY_fp8_ptr, Part_dM_ptr,
        M_dim, N_dim, CHUNK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        pid_cm, pid_n = tl.program_id(0), tl.program_id(1)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        m_n = offs_n < N_dim
        m_val = tl.load(M_ptr + offs_n, mask=m_n, other=1.0).to(tl.float32)
        inv_n = tl.load(InvN_ptr + offs_n, mask=m_n, other=1.0).to(tl.float32)
        s = m_val * inv_n
        inv_m = 1.0 / tl.maximum(m_val, 1e-8)
        sdy = tl.load(ScaleDy_ptr)
        inv_sdy = 1.0 / tl.maximum(sdy, 1e-12)
        dm_acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        amax_acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        m0 = pid_cm * CHUNK_M
        for step in range(CHUNK_M):
            m_idx = m0 + step
            if m_idx < M_dim:
                go = tl.load(GradOut_ptr + m_idx * N_dim + offs_n, mask=m_n, other=0.0).to(tl.float32)
                ov = tl.load(Out_ptr + m_idx * N_dim + offs_n, mask=m_n, other=0.0).to(tl.float32)
                dm_acc += go * ov
                dyp = go * s
                amax_acc = tl.maximum(amax_acc, tl.abs(dyp))
                tl.store(dY_pre_ptr + m_idx * N_dim + offs_n, dyp.to(dY_pre_ptr.dtype.element_ty), mask=m_n)
                tl.store(dY_fp8_ptr + m_idx * N_dim + offs_n, tl.minimum(tl.maximum(dyp * inv_sdy, -448.0), 448.0).to(tl.float8e4nv), mask=m_n)
        tl.store(Part_dM_ptr + pid_cm * N_dim + offs_n, dm_acc * inv_m, mask=m_n)
        tl.atomic_max(NextMaxDy_ptr, tl.max(amax_acc, axis=0))

    dora_part_256 = torch.empty((256,), device=device, dtype=torch.float32)
    MAX_DORA_SLOTS = 128
    dora_sx_buf = torch.ones((MAX_DORA_SLOTS,), device=device, dtype=torch.float32)
    dora_sdy_buf = torch.ones((MAX_DORA_SLOTS,), device=device, dtype=torch.float32)
    dora_next_max_x = torch.zeros((MAX_DORA_SLOTS,), device=device, dtype=torch.float32)
    dora_next_max_dy = torch.zeros((MAX_DORA_SLOTS,), device=device, dtype=torch.float32)
    dora_hist_x = torch.zeros((MAX_DORA_SLOTS, 4), device=device, dtype=torch.float32)
    dora_hist_dy = torch.zeros((MAX_DORA_SLOTS, 4), device=device, dtype=torch.float32)
    dora_fp8_state = {"count": 0, "hist_ptr": 0}

    def _quant_x_exact_into(x2, xf_out, sx_out):
        ne = x2.numel()
        fp8_amax_part_kernel[(256,)](x2, dora_part_256, ne, BLOCK=1024, num_warps=4)
        fp8_quant_from_part_kernel[(triton.cdiv(ne, 1024),)](x2, dora_part_256, sx_out, xf_out, ne, BLOCK=1024, N_PART=256, num_warps=4)

    class _Fused1xFP8DoRAFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x2, w0_fp8_T, w0_fp8_N, sw, la, lb, m_vec, inv_n, scale_lora, sx_del, nmax_x, sdy_del, nmax_dy, flags_box):
            M_d, K_d = x2.shape
            N_d = lb.shape[0]
            R_d = lb.shape[1]
            ha = F.linear(x2, la)
            xf = torch.empty((M_d, K_d), device=x2.device, dtype=torch.float8_e4m3fn)
            if flags_box[0]:
                fp8_quant_delayed_and_amax_kernel[(triton.cdiv(M_d * K_d, 1024),)](
                    x2, sx_del, nmax_x, xf, M_d * K_d, BLOCK=1024, num_warps=4
                )
                sx_use = sx_del
            else:
                sx_exact = torch.empty((1,), device=x2.device, dtype=torch.float32)
                _quant_x_exact_into(x2, xf, sx_exact)
                sx_del.copy_(sx_exact * 1.05)
                flags_box[0] = True
                sx_use = sx_exact
            out = torch._scaled_mm(xf, w0_fp8_T, scale_a=sx_use, scale_b=sw, out_dtype=dtype, use_fast_accum=True)
            dora_1x_lora_scale_inplace_kernel[(triton.cdiv(M_d, 64), triton.cdiv(N_d, 128))](
                out, ha, lb, m_vec, inv_n, M_d, N_d, scale_lora, R_LORA=R_d, BLOCK_M=64, BLOCK_N=128, num_warps=4
            )
            ctx.save_for_backward(x2, w0_fp8_N, sw, la, lb, m_vec, inv_n, ha, out)
            ctx.meta = (scale_lora, M_d, N_d, sdy_del, nmax_dy, flags_box)
            return out

        @staticmethod
        def backward(ctx, grad_out):
            x2, w0_fp8_N, sw, la, lb, m_vec, inv_n, ha, out = ctx.saved_tensors
            scale_lora, M_d, N_d, _sdy_del, _nmax_dy, _flags_box = ctx.meta
            go2 = grad_out.contiguous()
            CHUNK_M = 256
            n_chunks = triton.cdiv(M_d, CHUNK_M)
            n_nb = triton.cdiv(N_d, 128)
            dy_pre = torch.empty_like(go2)
            dy_fp8 = torch.empty((M_d, N_d), device=go2.device, dtype=torch.float8_e4m3fn)
            part_dm = torch.empty((n_chunks, N_d), device=go2.device, dtype=torch.float32)
            pmax = torch.zeros((64,), device=go2.device, dtype=torch.float32)
            sdy_exact = torch.empty((1,), device=go2.device, dtype=torch.float32)
            dora_1x_bwd_scale_dm_kernel[(n_chunks, n_nb)](
                go2, out, m_vec, inv_n, dy_pre, part_dm, pmax, M_d, N_d, CHUNK_M=CHUNK_M, BLOCK_N=128, num_warps=4
            )
            fp8_quant_from_dynmax_kernel[(triton.cdiv(M_d * N_d, 1024),)](dy_pre, pmax, sdy_exact, dy_fp8, M_d * N_d, BLOCK=1024, N_PMAX=64, num_warps=4)
            dm = torch.sum(part_dm, dim=0, dtype=m_vec.dtype)
            R_d = ha.shape[1]
            d_lb = torch.addmm(ha.new_empty((N_d, R_d)), dy_pre.t(), ha, beta=0.0, alpha=scale_lora)
            dha = torch.addmm(ha.new_empty((M_d, R_d)), dy_pre, lb, beta=0.0, alpha=scale_lora)
            d_la = torch.matmul(dha.t(), x2)
            dx = torch._scaled_mm(dy_fp8, w0_fp8_N, scale_a=sdy_exact, scale_b=sw, out_dtype=dtype, use_fast_accum=True)
            dx.addmm_(dha, la)
            return dx, None, None, None, d_la, d_lb, dm, None, None, None, None, None, None, None

    # Delayed-FP8 Fused Triton FastFP8DoRALinear (0 y_base/y_lora saved, NO w0_bf16 on GPU, 1-pass FP8!)
    class FastFP8DoRALinear(nn.Module):
        def __init__(self, base_layer, r=32, alpha=64):
            super().__init__()
            in_f, out_f = base_layer.in_features, base_layer.out_features
            self.in_features, self.out_features, self.r = in_f, out_f, r
            w0_f = base_layer.weight.data.float()
            w0_nsq = w0_f.pow(2).sum(dim=1)
            w0_n = torch.sqrt(w0_nsq.clamp(min=1e-8))
            self.m = nn.Parameter(w0_n.to(dtype))
            self.lora_A = nn.Parameter(torch.randn(r, in_f, device=device, dtype=dtype) * 0.01)
            self.lora_B = nn.Parameter(torch.zeros(out_f, r, device=device, dtype=dtype))
            self.scale = float(alpha) / float(r)
            self.register_buffer("w0_norm_sq", w0_nsq)
            self.register_buffer("cached_inv_norm", (1.0 / w0_n).to(dtype))
            sw = (w0_f.abs().amax().clamp(min=1e-8) / 448.0).reshape(1)
            w0_fp8 = (w0_f / sw).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
            self.register_buffer("w0_fp8_T", w0_fp8.t())
            self.register_buffer("w0_fp8_N", w0_fp8.t().contiguous().t())
            self.register_buffer("sw", sw)
            slot = dora_fp8_state["count"]
            assert slot < MAX_DORA_SLOTS, f"Too many DoRA modules ({slot})"
            dora_fp8_state["count"] = slot + 1
            self.sx_del = dora_sx_buf[slot : slot + 1]
            self.sdy_del = dora_sdy_buf[slot : slot + 1]
            self.nmax_x = dora_next_max_x[slot : slot + 1]
            self.nmax_dy = dora_next_max_dy[slot : slot + 1]
            self.flags_box = [False, False]  # [x_primed, dy_primed]

        @torch.no_grad()
        def refresh_norm(self):
            A_f = self.lora_A.float()
            B_f = self.lora_B.float()
            AAt = A_f @ A_f.T
            W0At = F.linear(A_f, self.w0_fp8_T.t().float() * self.sw).T
            norm_sq = (
                self.w0_norm_sq
                + 2.0 * self.scale * (B_f * W0At).sum(dim=1)
                + (self.scale ** 2) * (B_f * (B_f @ AAt)).sum(dim=1)
            )
            row_norm = torch.sqrt(norm_sq.clamp(min=1e-8))
            self.cached_inv_norm.copy_((1.0 / row_norm).to(self.m.dtype))

        @torch.compiler.disable
        def forward(self, x):
            orig_shape = x.shape[:-1]
            x2 = x.reshape(-1, self.in_features).contiguous()
            M_d = x2.shape[0]
            if (M_d & 15) != 0:
                M_pad = max(16, (M_d + 15) & ~15)
                x_pad = F.pad(x2, (0, 0, 0, M_pad - M_d))
                ha = F.linear(x_pad, self.lora_A)
                xf = torch.empty((M_pad, self.in_features), device=x2.device, dtype=torch.float8_e4m3fn)
                sx_buf = torch.empty((1,), device=x2.device, dtype=torch.float32)
                _quant_x_exact_into(x_pad, xf, sx_buf)
                out = torch._scaled_mm(xf, self.w0_fp8_T, scale_a=sx_buf, scale_b=self.sw, out_dtype=dtype, use_fast_accum=True)
                dora_1x_lora_scale_inplace_kernel[(triton.cdiv(M_pad, 64), triton.cdiv(self.out_features, 128))](
                    out, ha, self.lora_B, self.m, self.cached_inv_norm, M_pad, self.out_features, self.scale, R_LORA=self.r, BLOCK_M=64, BLOCK_N=128, num_warps=4
                )
                return out[:M_d].reshape(*orig_shape, self.out_features)
            if not torch.is_grad_enabled():
                ha = F.linear(x2, self.lora_A)
                xf = torch.empty((M_d, self.in_features), device=x2.device, dtype=torch.float8_e4m3fn)
                if self.training and self.flags_box[0]:
                    fp8_quant_delayed_and_amax_kernel[(triton.cdiv(M_d * self.in_features, 1024),)](
                        x2, self.sx_del, self.nmax_x, xf, M_d * self.in_features, BLOCK=1024, num_warps=4
                    )
                    sx_use = self.sx_del
                else:
                    sx_exact = torch.empty((1,), device=x2.device, dtype=torch.float32)
                    _quant_x_exact_into(x2, xf, sx_exact)
                    if self.training:
                        self.sx_del.copy_(sx_exact * 1.05)
                        self.flags_box[0] = True
                    sx_use = sx_exact
                out = torch._scaled_mm(xf, self.w0_fp8_T, scale_a=sx_use, scale_b=self.sw, out_dtype=dtype, use_fast_accum=True)
                dora_1x_lora_scale_inplace_kernel[(triton.cdiv(M_d, 64), triton.cdiv(self.out_features, 128))](
                    out, ha, self.lora_B, self.m, self.cached_inv_norm, M_d, self.out_features, self.scale, R_LORA=self.r, BLOCK_M=64, BLOCK_N=128, num_warps=4
                )
                return out.reshape(*orig_shape, self.out_features)
            y = _Fused1xFP8DoRAFn.apply(
                x2, self.w0_fp8_T, self.w0_fp8_N, self.sw, self.lora_A, self.lora_B, self.m, self.cached_inv_norm, self.scale,
                self.sx_del, self.nmax_x, self.sdy_del, self.nmax_dy, self.flags_box,
            )
            return y.reshape(*orig_shape, self.out_features)

    # Freeze all, attach FastFP8DoRALinear + unfreeze MoE-LoRA + RMSNorm (layers 6..23)
    for p in m.parameters():
        p.requires_grad_(False)

    dora_modules = []
    for name, mod in list(m.named_modules()):
        if isinstance(mod, nn.Linear) and "language_model" in name and "lm_head" not in name and mod.in_features >= 512 and mod.out_features >= 512:
            m_layer = re.search(r"layers\.(\d+)\.", name)
            if m_layer and int(m_layer.group(1)) < freeze_until:
                continue
            parent_name, attr = name.rsplit(".", 1)
            dmod = FastFP8DoRALinear(mod, r=32, alpha=64)
            setattr(m.get_submodule(parent_name), attr, dmod)
            dora_modules.append(dmod)

    for name, mod in m.named_modules():
        if isinstance(mod, PatchedFP8Experts) and mod.enable_lora:
            mod.lora_A_gu.requires_grad_(True)
            mod.lora_B_gu.requires_grad_(True)
            mod.lora_A_dn.requires_grad_(True)
            mod.lora_B_dn.requires_grad_(True)
        elif isinstance(mod, ink_mod.Inkling2LoopRMSNorm) and hasattr(mod, "weight") and mod.weight is not None:
            m_layer = re.search(r"layers\.(\d+)\.", name)
            if (m_layer and int(m_layer.group(1)) >= freeze_until) or name.endswith("language_model.norm"):
                mod.weight.requires_grad_(True)

    # Safety Guard: ALL routers strictly frozen!
    for name, p in m.named_parameters():
        if ".mlp.gate." in name:
            assert not p.requires_grad, f"Router {name} must stay frozen!"

    # Fused Triton Elementwise Kernels (RMSNorm, MulSigmoid, Scaled/LogL2Clip, MessageGateMix) -> 0 recompiles!
    @triton.jit
    def triton_rmsnorm_fwd_kernel(
        X_ptr, W_ptr, Y_ptr, InvR_ptr, stride_x, stride_y, N_dim, eps,
        HAS_W: tl.constexpr, SAVE_INVR: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        offs_n = tl.arange(0, BLOCK_N)
        m_n = offs_n < N_dim
        x = tl.load(X_ptr + pid_m * stride_x + offs_n, mask=m_n, other=0.0).to(tl.float32)
        inv_r = tl.rsqrt(tl.sum(x * x, axis=0) / N_dim + eps)
        if SAVE_INVR:
            tl.store(InvR_ptr + pid_m, inv_r)
        y = x * inv_r
        if HAS_W:
            w = tl.load(W_ptr + offs_n, mask=m_n, other=0.0).to(tl.float32)
            y = y * w
        tl.store(Y_ptr + pid_m * stride_y + offs_n, y.to(Y_ptr.dtype.element_ty), mask=m_n)

    @triton.jit
    def triton_rmsnorm_bwd_kernel(
        dY_ptr, X_ptr, W_ptr, InvR_ptr, dX_ptr, Part_dW_ptr,
        stride_dy, stride_x, stride_dx, M_dim, N_dim,
        HAS_W: tl.constexpr, NEED_DW: tl.constexpr,
        CHUNK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        pid_c = tl.program_id(0)
        offs_n = tl.arange(0, BLOCK_N)
        m_n = offs_n < N_dim
        inv_d = 1.0 / N_dim
        w = tl.load(W_ptr + offs_n, mask=m_n, other=0.0).to(tl.float32) if HAS_W else 1.0
        dw_acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        m0 = pid_c * CHUNK_M
        for s in range(CHUNK_M):
            m = m0 + s
            if m < M_dim:
                x = tl.load(X_ptr + m * stride_x + offs_n, mask=m_n, other=0.0).to(tl.float32)
                dy = tl.load(dY_ptr + m * stride_dy + offs_n, mask=m_n, other=0.0).to(tl.float32)
                inv_r = tl.load(InvR_ptr + m)
                x_hat = x * inv_r
                dy_w = dy * w if HAS_W else dy
                coeff = tl.sum(dy_w * x_hat, axis=0) * inv_d
                dx = inv_r * (dy_w - x_hat * coeff)
                tl.store(dX_ptr + m * stride_dx + offs_n, dx.to(dX_ptr.dtype.element_ty), mask=m_n)
                if NEED_DW:
                    dw_acc += dy * x_hat
        if NEED_DW:
            tl.store(Part_dW_ptr + pid_c * N_dim + offs_n, dw_acc, mask=m_n)

    class TritonRMSNormFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x2, weight, eps, has_w):
            M_d, N_d = x2.shape
            BLOCK_N = max(128, triton.next_power_of_2(N_d))
            y2 = torch.empty_like(x2)
            inv_r = torch.empty((M_d,), device=x2.device, dtype=torch.float32)
            nw = 8 if BLOCK_N >= 2048 else 4
            triton_rmsnorm_fwd_kernel[(M_d,)](
                x2, weight if has_w else x2, y2, inv_r, x2.stride(0), y2.stride(0), N_d, eps,
                HAS_W=has_w, SAVE_INVR=True, BLOCK_N=BLOCK_N, num_warps=nw,
            )
            ctx.save_for_backward(x2, weight if has_w else torch.empty(0, device=x2.device), inv_r)
            ctx.has_w, ctx.BLOCK_N, ctx.nw = has_w, BLOCK_N, nw
            return y2

        @staticmethod
        def backward(ctx, grad_out):
            x2, weight, inv_r = ctx.saved_tensors
            do2 = grad_out.contiguous()
            M_d, N_d = x2.shape
            dx2 = torch.empty_like(x2)
            need_dw = ctx.has_w and ctx.needs_input_grad[1]
            CHUNK_M = 128
            n_c = triton.cdiv(M_d, CHUNK_M)
            part_dw = torch.empty((n_c, N_d), device=x2.device, dtype=torch.float32) if need_dw else x2
            triton_rmsnorm_bwd_kernel[(n_c,)](
                do2, x2, weight if ctx.has_w else x2, inv_r, dx2, part_dw,
                do2.stride(0), x2.stride(0), dx2.stride(0), M_d, N_d,
                HAS_W=ctx.has_w, NEED_DW=need_dw, CHUNK_M=CHUNK_M, BLOCK_N=ctx.BLOCK_N, num_warps=ctx.nw,
            )
            dw = part_dw.sum(dim=0).to(weight.dtype) if need_dw else None
            return dx2, dw, None, None

    def _run_triton_rmsnorm(hidden_states, weight, eps, has_w):
        orig_shp = hidden_states.shape
        N_d = orig_shp[-1]
        x2 = hidden_states.reshape(-1, N_d).contiguous()
        if not torch.is_grad_enabled():
            M_d = x2.shape[0]
            BLOCK_N = max(128, triton.next_power_of_2(N_d))
            y2 = torch.empty_like(x2)
            triton_rmsnorm_fwd_kernel[(M_d,)](
                x2, weight if has_w else x2, y2, x2, x2.stride(0), y2.stride(0), N_d, eps,
                HAS_W=has_w, SAVE_INVR=False, BLOCK_N=BLOCK_N, num_warps=(8 if BLOCK_N >= 2048 else 4),
            )
            return y2.reshape(orig_shp)
        return TritonRMSNormFn.apply(x2, weight, eps, has_w).reshape(orig_shp)

    ink_mod.Inkling2LoopRMSNorm.forward = lambda self, hidden_states: _run_triton_rmsnorm(hidden_states, self.weight, float(self.variance_epsilon), True)
    ink_mod.Inkling2LoopUnweightedRMSNorm.forward = lambda self, hidden_states: _run_triton_rmsnorm(hidden_states, None, float(self.eps), False)

    @triton.jit
    def triton_mul_sigmoid_fwd_kernel(X_ptr, G_ptr, Out_ptr, N_elems, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < N_elems
        x = tl.load(X_ptr + offs, mask=m, other=0.0).to(tl.float32)
        g = tl.load(G_ptr + offs, mask=m, other=0.0).to(tl.float32)
        tl.store(Out_ptr + offs, (x * tl.sigmoid(g)).to(Out_ptr.dtype.element_ty), mask=m)

    @triton.jit
    def triton_mul_sigmoid_bwd_kernel(dOut_ptr, X_ptr, G_ptr, dX_ptr, dG_ptr, N_elems, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < N_elems
        do = tl.load(dOut_ptr + offs, mask=m, other=0.0).to(tl.float32)
        x = tl.load(X_ptr + offs, mask=m, other=0.0).to(tl.float32)
        g = tl.load(G_ptr + offs, mask=m, other=0.0).to(tl.float32)
        s = tl.sigmoid(g)
        tl.store(dX_ptr + offs, (do * s).to(dX_ptr.dtype.element_ty), mask=m)
        tl.store(dG_ptr + offs, (do * x * s * (1.0 - s)).to(dG_ptr.dtype.element_ty), mask=m)

    class TritonMulSigmoidFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, g):
            out = torch.empty_like(x)
            ne = x.numel()
            triton_mul_sigmoid_fwd_kernel[(triton.cdiv(ne, 1024),)](x, g, out, ne, BLOCK=1024, num_warps=4)
            ctx.save_for_backward(x, g)
            return out

        @staticmethod
        def backward(ctx, grad_out):
            x, g = ctx.saved_tensors
            do = grad_out.contiguous()
            dx, dg = torch.empty_like(x), torch.empty_like(g)
            ne = x.numel()
            triton_mul_sigmoid_bwd_kernel[(triton.cdiv(ne, 1024),)](do, x, g, dx, dg, ne, BLOCK=1024, num_warps=4)
            return dx, dg

    def patched_gated_output_forward(self, attn_output, hidden_states):
        g = self.gate_proj(hidden_states).contiguous()
        ao = attn_output.contiguous()
        if not torch.is_grad_enabled():
            out = torch.empty_like(ao)
            ne = ao.numel()
            triton_mul_sigmoid_fwd_kernel[(triton.cdiv(ne, 1024),)](ao, g, out, ne, BLOCK=1024, num_warps=4)
            return self.linear(out)
        return self.linear(TritonMulSigmoidFn.apply(ao, g))
    ink_mod.Inkling2LoopGatedOutputProjection.forward = patched_gated_output_forward

    @triton.jit
    def triton_scaled_l2_clip_fwd_kernel(
        X_ptr, Scale_ptr, Pos_ptr, Out_ptr, Norm_ptr,
        M_tot, S_dim, H_dim, threshold, alpha, log_n_floor,
        HAS_SCALE: tl.constexpr, HAS_LOG: tl.constexpr, BATCH_POS: tl.constexpr, SAVE_NORM: tl.constexpr,
        D_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        m_m = offs_m < M_tot
        h = offs_m % H_dim
        bs = offs_m // H_dim
        s_idx = bs if BATCH_POS else (bs % S_dim)
        offs_d = tl.arange(0, BLOCK_D)
        m_d = offs_d < D_DIM
        m_2d = m_m[:, None] & m_d[None, :]
        v = tl.load(X_ptr + offs_m[:, None] * D_DIM + offs_d[None, :], mask=m_2d, other=0.0).to(tl.float32)
        if HAS_SCALE:
            sc = tl.load(Scale_ptr + h[:, None] * D_DIM + offs_d[None, :], mask=m_2d, other=1.0).to(tl.float32)
            v = v * sc
        nrm = tl.sqrt(tl.sum(v * v, axis=1))
        if SAVE_NORM:
            tl.store(Norm_ptr + offs_m, nrm, mask=m_m)
        cf = tl.minimum(threshold / (nrm + 1e-6), 1.0)
        y = v * cf[:, None]
        if HAS_LOG:
            pos = tl.load(Pos_ptr + s_idx, mask=m_m, other=0).to(tl.float32)
            tau = 1.0 + alpha * tl.log(tl.maximum((pos + 1.0) / log_n_floor, 1.0))
            y = y * tau[:, None]
        tl.store(Out_ptr + offs_m[:, None] * D_DIM + offs_d[None, :], y.to(Out_ptr.dtype.element_ty), mask=m_2d)

    @triton.jit
    def triton_scaled_l2_clip_bwd_kernel(
        dOut_ptr, X_ptr, Scale_ptr, Pos_ptr, Norm_ptr, dX_ptr,
        M_tot, S_dim, H_dim, threshold, alpha, log_n_floor,
        HAS_SCALE: tl.constexpr, HAS_LOG: tl.constexpr, BATCH_POS: tl.constexpr,
        D_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        m_m = offs_m < M_tot
        h = offs_m % H_dim
        bs = offs_m // H_dim
        s_idx = bs if BATCH_POS else (bs % S_dim)
        offs_d = tl.arange(0, BLOCK_D)
        m_d = offs_d < D_DIM
        m_2d = m_m[:, None] & m_d[None, :]
        do = tl.load(dOut_ptr + offs_m[:, None] * D_DIM + offs_d[None, :], mask=m_2d, other=0.0).to(tl.float32)
        if HAS_LOG:
            pos = tl.load(Pos_ptr + s_idx, mask=m_m, other=0).to(tl.float32)
            tau = 1.0 + alpha * tl.log(tl.maximum((pos + 1.0) / log_n_floor, 1.0))
            do = do * tau[:, None]
        v = tl.load(X_ptr + offs_m[:, None] * D_DIM + offs_d[None, :], mask=m_2d, other=0.0).to(tl.float32)
        sc = tl.load(Scale_ptr + h[:, None] * D_DIM + offs_d[None, :], mask=m_2d, other=1.0).to(tl.float32) if HAS_SCALE else 1.0
        if HAS_SCALE:
            v = v * sc
        nrm = tl.load(Norm_ptr + offs_m, mask=m_m, other=1.0)
        raw = threshold / (nrm + 1e-6)
        dot_dv = tl.sum(do * v, axis=1)
        coeff = threshold / (nrm * (nrm + 1e-6) * (nrm + 1e-6) + 1e-12)
        dv = tl.where(raw[:, None] < 1.0, raw[:, None] * do - coeff[:, None] * dot_dv[:, None] * v, do)
        dx = dv * sc if HAS_SCALE else dv
        tl.store(dX_ptr + offs_m[:, None] * D_DIM + offs_d[None, :], dx.to(dX_ptr.dtype.element_ty), mask=m_2d)

    class TritonScaledL2ClipFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, sc, pos, thr, alp, nfl, has_scale, has_log):
            B_l, S_l, H_l, D_l = x.shape
            M_l = B_l * S_l * H_l
            BLOCK_D = max(16, triton.next_power_of_2(D_l))
            BLOCK_M = 32 if BLOCK_D <= 32 else 8
            out = torch.empty_like(x)
            nrm_buf = torch.empty((M_l,), device=x.device, dtype=torch.float32)
            batch_pos = has_log and (pos.shape[0] == B_l and B_l > 1)
            pos_f = pos.reshape(-1) if has_log else x
            triton_scaled_l2_clip_fwd_kernel[(triton.cdiv(M_l, BLOCK_M),)](
                x, sc if has_scale else x, pos_f, out, nrm_buf,
                M_l, S_l, H_l, thr, alp, nfl, HAS_SCALE=has_scale, HAS_LOG=has_log, BATCH_POS=batch_pos, SAVE_NORM=True,
                D_DIM=D_l, BLOCK_M=BLOCK_M, BLOCK_D=BLOCK_D, num_warps=4,
            )
            ctx.save_for_backward(x, sc if has_scale else torch.empty(0, device=x.device), pos_f, nrm_buf)
            ctx.meta = (S_l, H_l, D_l, BLOCK_M, BLOCK_D, thr, alp, nfl, has_scale, has_log, batch_pos)
            return out

        @staticmethod
        def backward(ctx, grad_out):
            x, sc, pos_f, nrm_buf = ctx.saved_tensors
            S_l, H_l, D_l, BLOCK_M, BLOCK_D, thr, alp, nfl, has_scale, has_log, batch_pos = ctx.meta
            do = grad_out.contiguous()
            dx = torch.empty_like(x)
            M_l = do.numel() // D_l
            triton_scaled_l2_clip_bwd_kernel[(triton.cdiv(M_l, BLOCK_M),)](
                do, x, sc if has_scale else x, pos_f, nrm_buf, dx,
                M_l, S_l, H_l, thr, alp, nfl, HAS_SCALE=has_scale, HAS_LOG=has_log, BATCH_POS=batch_pos,
                D_DIM=D_l, BLOCK_M=BLOCK_M, BLOCK_D=BLOCK_D, num_warps=4,
            )
            return dx, None, None, None, None, None, None, None

    def _run_l2_clip(x, sc, pos, thr, alp, nfl, has_scale, has_log):
        xc = x.contiguous()
        if not torch.is_grad_enabled():
            B_l, S_l, H_l, D_l = xc.shape
            M_l = B_l * S_l * H_l
            BLOCK_D = max(16, triton.next_power_of_2(D_l))
            BLOCK_M = 32 if BLOCK_D <= 32 else 8
            out = torch.empty_like(xc)
            batch_pos = has_log and (pos.shape[0] == B_l and B_l > 1)
            pos_f = pos.reshape(-1) if has_log else xc
            triton_scaled_l2_clip_fwd_kernel[(triton.cdiv(M_l, BLOCK_M),)](
                xc, sc if has_scale else xc, pos_f, out, xc,
                M_l, S_l, H_l, thr, alp, nfl, HAS_SCALE=has_scale, HAS_LOG=has_log, BATCH_POS=batch_pos, SAVE_NORM=False,
                D_DIM=D_l, BLOCK_M=BLOCK_M, BLOCK_D=BLOCK_D, num_warps=4,
            )
            return out
        return TritonScaledL2ClipFn.apply(xc, sc, pos, thr, alp, nfl, has_scale, has_log)

    ink_mod.Inkling2LoopScaledL2Clip.forward = lambda self, hidden_states, position_ids=None: _run_l2_clip(
        hidden_states, self.scale, position_ids, float(self.threshold), 0.0, 1.0, True, False
    )
    ink_mod.Inkling2LoopLogScaledL2Clip.forward = lambda self, hidden_states, position_ids: _run_l2_clip(
        hidden_states, self.scale, position_ids, float(self.l2_clip.threshold),
        float(self.log_scale.log_scaling_alpha), float(self.log_scale.log_scaling_n_floor), True, True
    )

    @triton.jit
    def triton_message_gate_mix_fwd_kernel(H_ptr, M_ptr, FG_ptr, MG_ptr, Out_ptr, N_elems, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mk = offs < N_elems
        h = tl.load(H_ptr + offs, mask=mk, other=0.0).to(tl.float32)
        m = tl.load(M_ptr + offs, mask=mk, other=0.0).to(tl.float32)
        fg = tl.load(FG_ptr + offs, mask=mk, other=0.0).to(tl.float32)
        mg = tl.load(MG_ptr + offs, mask=mk, other=0.0).to(tl.float32)
        th_f = 2.0 * tl.sigmoid(2.0 * fg) - 1.0
        th_m = 2.0 * tl.sigmoid(2.0 * mg) - 1.0
        tl.store(Out_ptr + offs, ((1.0 - th_f) * h + th_m * m).to(Out_ptr.dtype.element_ty), mask=mk)

    @triton.jit
    def triton_message_gate_mix_bwd_kernel(dY_ptr, H_ptr, M_ptr, FG_ptr, MG_ptr, dH_ptr, dM_ptr, dFG_ptr, dMG_ptr, N_elems, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mk = offs < N_elems
        dy = tl.load(dY_ptr + offs, mask=mk, other=0.0).to(tl.float32)
        h = tl.load(H_ptr + offs, mask=mk, other=0.0).to(tl.float32)
        m = tl.load(M_ptr + offs, mask=mk, other=0.0).to(tl.float32)
        fg = tl.load(FG_ptr + offs, mask=mk, other=0.0).to(tl.float32)
        mg = tl.load(MG_ptr + offs, mask=mk, other=0.0).to(tl.float32)
        th_f = 2.0 * tl.sigmoid(2.0 * fg) - 1.0
        th_m = 2.0 * tl.sigmoid(2.0 * mg) - 1.0
        tl.store(dH_ptr + offs, (dy * (1.0 - th_f)).to(dH_ptr.dtype.element_ty), mask=mk)
        tl.store(dM_ptr + offs, (dy * th_m).to(dM_ptr.dtype.element_ty), mask=mk)
        tl.store(dFG_ptr + offs, (dy * (-h) * (1.0 - th_f * th_f)).to(dFG_ptr.dtype.element_ty), mask=mk)
        tl.store(dMG_ptr + offs, (dy * m * (1.0 - th_m * th_m)).to(dMG_ptr.dtype.element_ty), mask=mk)

    class TritonMessageGateMixFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, h, m, fg, mg):
            out = torch.empty_like(h)
            ne = h.numel()
            triton_message_gate_mix_fwd_kernel[(triton.cdiv(ne, 1024),)](h, m, fg, mg, out, ne, BLOCK=1024, num_warps=4)
            ctx.save_for_backward(h, m, fg, mg)
            return out

        @staticmethod
        def backward(ctx, grad_out):
            h, m, fg, mg = ctx.saved_tensors
            dy = grad_out.contiguous()
            dh, dm = torch.empty_like(h), torch.empty_like(m)
            dfg, dmg = torch.empty_like(fg), torch.empty_like(mg)
            ne = h.numel()
            triton_message_gate_mix_bwd_kernel[(triton.cdiv(ne, 1024),)](dy, h, m, fg, mg, dh, dm, dfg, dmg, ne, BLOCK=1024, num_warps=4)
            return dh, dm, dfg, dmg

    def patched_message_gate_forward(self, hidden_states, message):
        w = self.down_proj.weight
        w_dt = w.dtype
        h_bf = (hidden_states if hidden_states.dtype == w_dt else hidden_states.to(w_dt)).contiguous()
        m_bf = (message if message.dtype == w_dt else message.to(w_dt)).contiguous()
        hn = _run_triton_rmsnorm(h_bf, None, float(self.norm.eps), False)
        mn = _run_triton_rmsnorm(m_bf, None, float(self.norm.eps), False)
        w_h = getattr(self, "_w_h", None)
        if w_h is None:
            H_g = h_bf.shape[-1]
            w_h = w[:, :H_g].contiguous()
            w_m = w[:, H_g:].contiguous()
            self._w_h = w_h
            self._w_m = w_m
        else:
            w_m = self._w_m
        gate_h = (F.linear(hn, w_h) + F.linear(mn, w_m)).float()
        fg = self.forget_gate_proj(gate_h).contiguous()
        mg = self.message_gate_proj(gate_h).contiguous()
        if not torch.is_grad_enabled():
            out = torch.empty_like(h_bf)
            ne = h_bf.numel()
            triton_message_gate_mix_fwd_kernel[(triton.cdiv(ne, 1024),)](h_bf, m_bf, fg, mg, out, ne, BLOCK=1024, num_warps=4)
            return out
        return TritonMessageGateMixFn.apply(h_bf, m_bf, fg, mg)
    ink_mod.Inkling2LoopMessageGate.forward = patched_message_gate_forward

    # Compile ONLY TopKRouter cleanly with contiguous input (0 stride recompiles, kills 36.8ms elementwise + Command Buffer Full!)
    compile_targets = [ink_mod.Inkling2LoopTopKRouter]
    orig_topk_fwd = ink_mod.Inkling2LoopTopKRouter.forward
    orig_uncompiled_forwards = {ink_mod.Inkling2LoopTopKRouter: orig_topk_fwd}
    _compiled_topk_core = torch.compile(orig_topk_fwd, dynamic=True)
    def _topk_dispatch(self, hidden_states):
        return _compiled_topk_core(self, hidden_states.contiguous())
    compiled_forwards = {ink_mod.Inkling2LoopTopKRouter: _topk_dispatch}
    ink_mod.Inkling2LoopTopKRouter.forward = _topk_dispatch

    def _set_compiled_mode(enable: bool):
        for cls_obj in compile_targets:
            cls_obj.forward = compiled_forwards[cls_obj] if (enable and cls_obj in compiled_forwards) else orig_uncompiled_forwards[cls_obj]

    n_depth_attn = sum(
        1 for _, mod in m.named_modules()
        if isinstance(mod, (ink_mod.Inkling2LoopBlockAttentionResidual, ink_mod.Inkling2LoopMessageAttentionResidual))
    )
    n_moe_patched = sum(1 for _, mod in m.named_modules() if isinstance(mod, PatchedFP8Experts))
    n_fa_patched = sum(
        1 for _, mod in m.named_modules()
        if isinstance(mod, (ink_mod.Inkling2LoopSlidingAttention, ink_mod.Inkling2LoopFullAttention))
    )
    n_conv1d_patched = sum(1 for _, mod in m.named_modules() if hasattr(mod, "sconv"))
    assert n_depth_attn > 0, "DepthAttn patching failed (0 modules)!"

    # =========================================================================
    # 6. T-BPTT + STOCHASTIC LOOP DEPTH + ASYMMETRIC TOP-P + CUDA GRAPH PREFIX
    # =========================================================================
    tbptt_cfg = {"grad_passes": 1, "num_passes": 3, "prefix_top_p": 0.65, "tail_top_p": 0.85}
    cg_cfg = {"enabled": True, "cache": {}}

    def run_nograd_prefix(self_tm, in_embeds, doc_id_t, qn_t, kms_t, kme_t, position_ids, n_prefix_loops):
        attn_ctx["doc_id"] = doc_id_t
        attn_ctx["q_n_start"] = qn_t
        attn_ctx["k_m_start"] = kms_t
        attn_ctx["k_m_end"] = kme_t
        old_tp = moe_cfg["top_p"]
        moe_cfg["top_p"] = tbptt_cfg["prefix_top_p"]
        loop_start, loop_end = self_tm.config.loop_layer_index_start, self_tm.config.loop_layer_index_end
        with torch.no_grad():
            hidden_states = in_embeds
            block_states = in_embeds.new_zeros(*in_embeds.shape[:-1], 0, in_embeds.shape[-1])
            for decoder_layer in self_tm.layers[:loop_start]:
                new_h = decoder_layer(hidden_states, block_states, position_ids=position_ids)
                if decoder_layer.is_block_boundary:
                    block_states = torch.cat([block_states, hidden_states.unsqueeze(-2)], dim=-2)
                hidden_states = new_h
            pre_h, pre_b = hidden_states, block_states
            num_pre_blocks = pre_b.shape[-2]

            # Pass 0 (no_grad)
            for decoder_layer in self_tm.layers[loop_start:loop_end]:
                new_h = decoder_layer(hidden_states, block_states, position_ids=position_ids)
                if decoder_layer.is_block_boundary:
                    block_states = torch.cat([block_states, hidden_states.unsqueeze(-2)], dim=-2)
                hidden_states = new_h

            # Pass 1 (no_grad) if 3-pass loop (n_prefix_loops == 2)
            if n_prefix_loops >= 2:
                prev_b = block_states[..., num_pre_blocks + 1 :, :]
                prev_h = hidden_states
                hidden_states, block_states = pre_h, pre_b
                cache_offset = self_tm.config.num_hidden_layers - loop_start
                for loop_layer_idx, decoder_layer in enumerate(self_tm.layers[loop_start:loop_end]):
                    messages = self_tm.message_input(prev_b, prev_h, loop_layer_idx)
                    new_h = decoder_layer(hidden_states, block_states, messages, cache_offset=cache_offset, position_ids=position_ids)
                    if decoder_layer.is_block_boundary:
                        block_states = torch.cat([block_states, hidden_states.unsqueeze(-2)], dim=-2)
                    hidden_states = new_h

            out_prev_b = block_states[..., num_pre_blocks + 1 :, :].contiguous()
            out_prev_h = hidden_states.contiguous()
        moe_cfg["top_p"] = old_tp
        return pre_h.contiguous(), pre_b.contiguous(), out_prev_b, out_prev_h

    def patched_text_model_forward(
        self, input_ids=None, attention_mask=None, position_ids=None, cache=None,
        inputs_embeds=None, use_cache=None, **kwargs
    ):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        B_cur, S_cur = inputs_embeds.shape[:2]
        dev_cur = inputs_embeds.device
        if position_ids is None:
            position_ids = torch.arange(S_cur, device=dev_cur).unsqueeze(0)

        loop_start, loop_end = self.config.loop_layer_index_start, self.config.loop_layer_index_end
        num_loop_layers = loop_end - loop_start
        num_passes = tbptt_cfg["num_passes"] if self.training else self.config.loop_num_passes
        grad_start_pass = (num_passes - tbptt_cfg["grad_passes"]) if self.training else num_passes

        if self.training and grad_start_pass in (1, 2) and cg_cfg["enabled"]:
            n_pref = grad_start_pass
            cg_key = (B_cur, S_cur, n_pref, tbptt_cfg["prefix_top_p"])
            saved_gc = [l.gradient_checkpointing for l in self.layers[loop_start:loop_end]]
            if cg_key not in cg_cfg["cache"]:
                for l_obj in self.layers[loop_start:loop_end]:
                    l_obj.gradient_checkpointing = False
                static_embeds = inputs_embeds.detach().clone()
                static_doc = attn_ctx["doc_id"].clone()
                static_qn = attn_ctx["q_n_start"].clone()
                static_kms = attn_ctx["k_m_start"].clone()
                static_kme = attn_ctx["k_m_end"].clone()
                static_pos = position_ids.clone()
                s_cap = cg_cfg.setdefault("stream", torch.cuda.Stream())
                s_cap.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s_cap):
                    _ = run_nograd_prefix(self, static_embeds, static_doc, static_qn, static_kms, static_kme, static_pos, n_pref)
                    _ = run_nograd_prefix(self, static_embeds, static_doc, static_qn, static_kms, static_kme, static_pos, n_pref)
                torch.cuda.current_stream().wait_stream(s_cap)
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                g_cap = torch.cuda.CUDAGraph()
                cg_pool = cg_cfg.setdefault("pool", torch.cuda.graph_pool_handle())
                with torch.cuda.graph(g_cap, pool=cg_pool, stream=s_cap):
                    static_outs = run_nograd_prefix(self, static_embeds, static_doc, static_qn, static_kms, static_kme, static_pos, n_pref)
                torch.cuda.current_stream().wait_stream(s_cap)
                g_cap.replay()
                torch.cuda.synchronize()
                cg_cfg["cache"][cg_key] = (g_cap, static_embeds, static_doc, static_qn, static_kms, static_kme, static_outs)
                for idx_l, l_obj in enumerate(self.layers[loop_start:loop_end]):
                    l_obj.gradient_checkpointing = saved_gc[idx_l]

            g_cap, static_embeds, static_doc, static_qn, static_kms, static_kme, static_outs = cg_cfg["cache"][cg_key]
            cur_doc, cur_qn, cur_kms, cur_kme = attn_ctx["doc_id"], attn_ctx["q_n_start"], attn_ctx["k_m_start"], attn_ctx["k_m_end"]
            static_embeds.copy_(inputs_embeds.detach())
            static_doc.copy_(cur_doc)
            static_qn.copy_(cur_qn)
            static_kms.copy_(cur_kms)
            static_kme.copy_(cur_kme)
            g_cap.replay()
            attn_ctx["doc_id"], attn_ctx["q_n_start"], attn_ctx["k_m_start"], attn_ctx["k_m_end"] = cur_doc, cur_qn, cur_kms, cur_kme
            pre_h, pre_b, prev_b, prev_h = static_outs
            hidden_states = pre_h.detach()
            block_states = pre_b.detach()
            previous_block_states = prev_b.detach()
            previous_hidden_states = prev_h.detach()

            # Trainable tail pass (18 layers: 12 loop + 6 post)
            moe_cfg["top_p"] = tbptt_cfg["tail_top_p"]
            cache_offset = self.config.num_hidden_layers + (n_pref - 1) * num_loop_layers - loop_start
            for loop_layer_idx, decoder_layer in enumerate(self.layers[loop_start:loop_end]):
                messages = self.message_input(previous_block_states, previous_hidden_states, loop_layer_idx)
                new_h = decoder_layer(hidden_states, block_states, messages, cache_offset=cache_offset, position_ids=position_ids)
                if decoder_layer.is_block_boundary:
                    block_states = torch.cat([block_states, hidden_states.unsqueeze(-2)], dim=-2)
                hidden_states = new_h

            for decoder_layer in self.layers[loop_end:]:
                new_h = decoder_layer(hidden_states, block_states, position_ids=position_ids)
                if decoder_layer.is_block_boundary:
                    block_states = torch.cat([block_states, hidden_states.unsqueeze(-2)], dim=-2)
                hidden_states = new_h

            hidden_states = self.final_input(block_states, hidden_states)[0]
            hidden_states = self.norm(hidden_states).to(inputs_embeds.dtype)
            return ink_mod.MoeModelOutputWithPast(last_hidden_state=hidden_states, cache=cache)

        # Standard path (Cooldown or Eval)
        hidden_states = inputs_embeds
        block_states = inputs_embeds.new_zeros(*inputs_embeds.shape[:-1], 0, inputs_embeds.shape[-1])
        with torch.no_grad():
            for decoder_layer in self.layers[:loop_start]:
                new_h = decoder_layer(hidden_states, block_states, position_ids=position_ids)
                if decoder_layer.is_block_boundary:
                    block_states = torch.cat([block_states, hidden_states.unsqueeze(-2)], dim=-2)
                hidden_states = new_h

        pre_loop_hidden_states = hidden_states.detach()
        pre_loop_block_states = block_states.detach()
        num_pre_loop_blocks = pre_loop_block_states.shape[-2]

        saved_gc = [l.gradient_checkpointing for l in self.layers[loop_start:loop_end]]
        try:
            p0_grad = self.training and (0 >= grad_start_pass)
            for idx_l, l_obj in enumerate(self.layers[loop_start:loop_end]):
                l_obj.gradient_checkpointing = saved_gc[idx_l] if p0_grad else False
            with torch.set_grad_enabled(p0_grad):
                for decoder_layer in self.layers[loop_start:loop_end]:
                    new_h = decoder_layer(hidden_states, block_states, position_ids=position_ids)
                    if decoder_layer.is_block_boundary:
                        block_states = torch.cat([block_states, hidden_states.unsqueeze(-2)], dim=-2)
                    hidden_states = new_h

            for loop_idx in range(1, num_passes):
                p_grad = self.training and (loop_idx >= grad_start_pass)
                previous_block_states = block_states[..., num_pre_loop_blocks + 1 :, :]
                previous_hidden_states = hidden_states
                if loop_idx == grad_start_pass and grad_start_pass > 0:
                    previous_block_states = previous_block_states.detach()
                    previous_hidden_states = previous_hidden_states.detach()
                hidden_states = pre_loop_hidden_states
                block_states = pre_loop_block_states
                cache_offset = self.config.num_hidden_layers + (loop_idx - 1) * num_loop_layers - loop_start
                for idx_l, l_obj in enumerate(self.layers[loop_start:loop_end]):
                    l_obj.gradient_checkpointing = saved_gc[idx_l] if p_grad else False
                with torch.set_grad_enabled(p_grad):
                    for loop_layer_idx, decoder_layer in enumerate(self.layers[loop_start:loop_end]):
                        messages = self.message_input(previous_block_states, previous_hidden_states, loop_layer_idx)
                        new_h = decoder_layer(hidden_states, block_states, messages, cache_offset=cache_offset, position_ids=position_ids)
                        if decoder_layer.is_block_boundary:
                            block_states = torch.cat([block_states, hidden_states.unsqueeze(-2)], dim=-2)
                        hidden_states = new_h
        finally:
            for idx_l, l_obj in enumerate(self.layers[loop_start:loop_end]):
                l_obj.gradient_checkpointing = saved_gc[idx_l]

        for decoder_layer in self.layers[loop_end:]:
            new_h = decoder_layer(hidden_states, block_states, position_ids=position_ids)
            if decoder_layer.is_block_boundary:
                block_states = torch.cat([block_states, hidden_states.unsqueeze(-2)], dim=-2)
            hidden_states = new_h

        hidden_states = self.final_input(block_states, hidden_states)[0]
        hidden_states = self.norm(hidden_states).to(inputs_embeds.dtype)
        return ink_mod.MoeModelOutputWithPast(last_hidden_state=hidden_states, cache=cache)

    ink_mod.Inkling2LoopTextModel.forward = patched_text_model_forward

    # Hybrid Selective GC: first `full_gc_layers` trainable layers use Full GC, next `moe_gc_layers` use MoE-only GC (Attention/DoRA/DepthAttn kept!)
    gc_fn = partial(torch.utils.checkpoint.checkpoint, use_reentrant=False)
    full_gc_end = freeze_until + max(0, int(full_gc_layers))
    moe_gc_end = full_gc_end + max(0, int(moe_gc_layers))
    n_full_gc = 0
    n_moe_gc = 0

    def _make_moe_gc_fwd(orig_mlp_fwd):
        def _wrapped(h_in):
            if torch.is_grad_enabled():
                return gc_fn(orig_mlp_fwd, h_in)
            return orig_mlp_fwd(h_in)
        return _wrapped

    for idx, layer in enumerate(m.model.language_model.layers):
        layer._gradient_checkpointing_func = gc_fn
        if freeze_until <= idx < full_gc_end:
            layer.gradient_checkpointing = True
            n_full_gc += 1
        elif full_gc_end <= idx < moe_gc_end and hasattr(layer, "mlp"):
            layer.gradient_checkpointing = False
            layer.mlp.forward = _make_moe_gc_fwd(layer.mlp.forward)
            n_moe_gc += 1
        else:
            layer.gradient_checkpointing = False

    # =========================================================================
    # 7. OPTIMIZERS: Batched Cautious Muon (Dense DoRA) + Fused AdamW (MoE-LoRA + Norm)
    # =========================================================================
    named_params = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
    dense_dora_ab = [(n, p) for n, p in named_params if n.endswith(".lora_A") or n.endswith(".lora_B")]
    dense_dora_m = [(n, p) for n, p in named_params if n.endswith(".m")]
    moe_lora_a = [(n, p) for n, p in named_params if n.endswith(".lora_A_gu") or n.endswith(".lora_A_dn")]
    moe_lora_b = [(n, p) for n, p in named_params if n.endswith(".lora_B_gu") or n.endswith(".lora_B_dn")]
    rmsnorm_named = [(n, p) for n, p in named_params if n not in {k for k, _ in dense_dora_ab + dense_dora_m + moe_lora_a + moe_lora_b}]

    print(
        f"=== MODEL LOADED & PATCHED in {time.time()-t_load:.1f}s | "
        f"Trainable={sum(p.numel() for _, p in named_params)/1e6:.2f}M "
        f"(DoRA={len(dora_modules)}, MoE={n_moe_patched}, FA={n_fa_patched}, DepthAttn={n_depth_attn}, Conv1d={n_conv1d_patched}, Norm={len(rmsnorm_named)}, Compiled={len(compiled_forwards)}, GC=Full:{n_full_gc}+MoE:{n_moe_gc}/18, MoELoRA>={moe_lora_start}) | "
        f"VRAM={torch.cuda.memory_allocated()/1e9:.2f}GB ===",
        flush=True,
    )

    def batched_ns5(G, steps=5):
        a, b, c = (3.4445, -4.7750, 2.0315)
        X = G.bfloat16()
        trans = X.size(1) > X.size(2)
        if trans:
            X = X.transpose(1, 2)
        X = X / (X.norm(dim=(1, 2), keepdim=True) + 1e-7)
        for _ in range(steps):
            A = torch.bmm(X, X.transpose(1, 2))
            B = b * A + c * torch.bmm(A, A)
            X = a * X + torch.bmm(B, X)
        if trans:
            X = X.transpose(1, 2)
        return X

    muon_buckets = {}
    for n, p in dense_dora_ab:
        key = (p.shape, n.endswith(".lora_B"))
        muon_buckets.setdefault(key, []).append(p)
    muon_groups = []
    for (shp, is_b), plist in muon_buckets.items():
        M_buf = torch.zeros((len(plist), *shp), device=device, dtype=dtype)
        muon_groups.append((plist, M_buf, is_b))

    @torch.no_grad()
    def cautious_muon_step(lr_a_cur, lr_b_cur, wd=0.01):
        for plist, M_buf, is_b in muon_groups:
            lr_cur = lr_b_cur if is_b else lr_a_cur
            grads = torch.stack([p.grad if p.grad is not None else torch.zeros_like(p) for p in plist], dim=0)
            M_buf.mul_(0.95).add_(grads)
            U = batched_ns5(M_buf, steps=5)
            rms_scale = 0.2 * math.sqrt(max(U.size(1), U.size(2)))
            mask = (U * grads) > 0
            U.mul_(mask.to(U.dtype))
            eff_lr = lr_cur * rms_scale
            if wd > 0.0:
                torch._foreach_mul_(plist, 1.0 - lr_cur * wd)
            torch._foreach_add_(plist, U.unbind(0), alpha=-eff_lr)
            for p in plist:
                p.grad = None

    adam_opt = torch.optim.AdamW(
        [
            {"params": [p for _, p in dense_dora_m], "lr": lr_a, "weight_decay": 0.0},
            {"params": [p for _, p in moe_lora_a + moe_lora_b], "lr": lr_moe, "weight_decay": 0.01},
            {"params": [p for _, p in rmsnorm_named], "lr": lr_norm, "weight_decay": 0.0},
        ],
        betas=(0.9, 0.95),
        fused=True,
    )

    moe_lora_modules = [mod for _, mod in m.named_modules() if isinstance(mod, PatchedFP8Experts) and mod.enable_lora]

    def refresh_all_dora_norms():
        for dm in dora_modules:
            dm.refresh_norm()

    def refresh_all_moe_fp8():
        for mm in moe_lora_modules:
            mm.refresh_la_dn_fp8()

    @torch.no_grad()
    def roll_dora_fp8_scales():
        n_d = dora_fp8_state["count"]
        if n_d <= 0:
            return
        hp = dora_fp8_state["hist_ptr"] & 3
        dora_fp8_state["hist_ptr"] += 1
        dora_hist_x[:n_d, hp].copy_(dora_next_max_x[:n_d])
        dora_hist_dy[:n_d, hp].copy_(dora_next_max_dy[:n_d])
        amax_x = dora_hist_x[:n_d].amax(dim=1)
        amax_dy = dora_hist_dy[:n_d].amax(dim=1)
        dora_sx_buf[:n_d].copy_(torch.where(amax_x > 0.0, amax_x.clamp(min=1e-8) * (1.05 / 448.0), dora_sx_buf[:n_d]))
        dora_sdy_buf[:n_d].copy_(torch.where(amax_dy > 0.0, amax_dy.clamp(min=1e-8) * (1.05 / 448.0), dora_sdy_buf[:n_d]))
        dora_next_max_x[:n_d].zero_()
        dora_next_max_dy[:n_d].zero_()

    # =========================================================================
    # 8. PREFETCHER (BS=micro_bs, 16k Pruned Vocab) + EVAL + GENERATE
    # =========================================================================
    V_FULL = int(m.lm_head.weight.shape[0])
    HEAD_K = 12288
    SUB_V = 16384
    HEAD_TOK_CPU = torch.arange(HEAD_K, dtype=torch.long)
    EOS_ID = int(tokenizer.eos_token_id)
    EOS_TOK_CPU = torch.tensor([EOS_ID], dtype=torch.long)
    hidden_size = m.config.get_text_config().hidden_size

    def build_cpu_batch(group_idx, gen):
        s0 = group_idx * micro_bs
        b_ids = train_ids_all[s0 : s0 + micro_bs].to(torch.long)
        b_lbl = train_lbl_all[s0 : s0 + micro_bs].to(torch.long)
        b_doc = train_doc_all[s0 : s0 + micro_bs]
        qn, kms, kme = precompute_doc_bounds_cpu(b_doc, 1024)

        lbl_shift = b_lbl[:, 1:].reshape(-1)
        valid_idx = torch.nonzero(lbl_shift != -100, as_tuple=False).squeeze(-1)
        valid_lbl = lbl_shift[valid_idx]

        l23_mask = torch.zeros((micro_bs, seq_len), dtype=torch.bool)
        l23_mask[:, :-1] = (b_lbl[:, 1:] != -100)
        for lag in (1, 2, 3):
            l23_mask[:, :-lag] |= l23_mask[:, lag:].clone()
        l23_idx = torch.nonzero(l23_mask.reshape(-1), as_tuple=False).squeeze(-1)

        uniq_toks = torch.unique(valid_lbl)
        head_union = torch.unique(torch.cat([uniq_toks, HEAD_TOK_CPU, EOS_TOK_CPU]))
        n_tail = max(0, SUB_V - head_union.numel())
        if n_tail > 0:
            cand = torch.randint(HEAD_K, V_FULL, (n_tail * 2,), generator=gen, dtype=torch.long)
            cand = torch.unique(cand)
            cand = cand[~torch.isin(cand, head_union)][:n_tail]
            sub_vocab = torch.sort(torch.cat([head_union, cand])).values
            corr_val = math.log((V_FULL - HEAD_K) / max(1, cand.numel()))
            is_tail = ~torch.isin(sub_vocab, head_union)
            log_q_corr = torch.where(is_tail, torch.full_like(sub_vocab, corr_val, dtype=torch.float32), torch.zeros_like(sub_vocab, dtype=torch.float32))
        else:
            sub_vocab = head_union[:SUB_V]
            log_q_corr = torch.zeros((SUB_V,), dtype=torch.float32)

        remap_targets = torch.searchsorted(sub_vocab, valid_lbl)
        return (
            b_ids.pin_memory(), b_doc.pin_memory(), qn.pin_memory(), kms.pin_memory(), kme.pin_memory(),
            valid_idx.pin_memory(), l23_idx.pin_memory(), sub_vocab.pin_memory(), remap_targets.pin_memory(), log_q_corr.pin_memory(),
        )

    def start_prefetch_stream(total_steps):
        q = queue.Queue(maxsize=4)
        stop_ev = threading.Event()
        def _worker():
            try:
                cur_ep = -1
                perm = None
                for step_i in range(total_steps):
                    if stop_ev.is_set():
                        break
                    ep = step_i // n_train_groups
                    pos = step_i % n_train_groups
                    if ep != cur_ep:
                        cur_ep = ep
                        ep_gen = torch.Generator().manual_seed(seed + ep * 1009)
                        perm = torch.randperm(n_train_groups, generator=ep_gen).tolist()
                    neg_gen = torch.Generator().manual_seed(seed + ep * 100003 + step_i)
                    q.put((ep, build_cpu_batch(perm[pos], neg_gen)))
                q.put(None)
            except Exception as exc:
                q.put(exc)
        th = threading.Thread(target=_worker, daemon=True)
        th.start()
        return q, stop_ev, th

    @torch.compile(dynamic=True)
    def _ce_chunk(h_c, w_m, lq, t_c):
        sub_logits = m.logit_softcap(F.linear(h_c, w_m).float()) + lq[None, :]
        return F.cross_entropy(sub_logits, t_c, reduction="sum")

    def compute_pruned_loss_16k(cpu_batch):
        b_ids, b_doc, qn, kms, kme, valid_idx, l23_idx, sub_vocab, remap_targets, log_q_corr = [
            t.to(device, non_blocking=True) for t in cpu_batch
        ]
        attn_ctx["doc_id"], attn_ctx["q_n_start"], attn_ctx["k_m_start"], attn_ctx["k_m_end"] = b_doc, qn, kms, kme
        layer23_ctx["idx"] = l23_idx
        h_out = m.model(input_ids=b_ids).last_hidden_state[:, :-1, :].reshape(-1, hidden_size)
        h_valid = h_out[valid_idx]
        w_sub = m.lm_head.weight[sub_vocab]
        n_val = h_valid.shape[0]
        tot = h_valid.new_zeros((), dtype=torch.float32)
        for i in range(0, n_val, 2048):
            tot = tot + torch.utils.checkpoint.checkpoint(
                _ce_chunk, h_valid[i : i + 2048], w_sub, log_q_corr, remap_targets[i : i + 2048], use_reentrant=False
            )
        return tot / max(1, n_val)

    @torch.no_grad()
    def eval_full_vocab_loss_all(model_obj, max_batches=None, is_merged=False):
        was_training = model_obj.training
        old_tp, old_gp, old_np, old_cg = moe_cfg["top_p"], tbptt_cfg["grad_passes"], tbptt_cfg["num_passes"], cg_cfg["enabled"]
        if not is_merged:
            refresh_all_dora_norms()
        _set_compiled_mode(False)
        model_obj.eval()
        moe_cfg["top_p"] = 1.0
        tbptt_cfg["grad_passes"] = 3
        tbptt_cfg["num_passes"] = 3
        cg_cfg["enabled"] = False
        layer23_ctx["idx"] = None
        total_loss, total_valid = 0.0, 0
        n_eval = n_val_batches if max_batches is None else min(n_val_batches, max_batches)
        for vi in range(n_eval):
            ids_b = val_ids_all[vi].unsqueeze(0).to(device=device, dtype=torch.long)
            lbl_b = val_lbl_all[vi].unsqueeze(0).to(device=device, dtype=torch.long)
            doc_b = val_doc_all[vi].unsqueeze(0)
            qn_b, kms_b, kme_b = precompute_doc_bounds_cpu(doc_b, 1024)
            attn_ctx["doc_id"] = doc_b.to(device)
            attn_ctx["q_n_start"] = qn_b.to(device)
            attn_ctx["k_m_start"] = kms_b.to(device)
            attn_ctx["k_m_end"] = kme_b.to(device)
            h_out = model_obj.model(input_ids=ids_b).last_hidden_state[:, :-1, :].reshape(-1, hidden_size)
            lbl_shift = lbl_b[:, 1:].reshape(-1)
            for i in range(0, h_out.shape[0], 512):
                lc = lbl_shift[i : i + 512]
                v_cnt = (lc != -100).sum().item()
                if v_cnt == 0:
                    continue
                logits_c = model_obj.logit_softcap(model_obj.lm_head(h_out[i : i + 512]).float())
                total_loss += F.cross_entropy(logits_c, lc, ignore_index=-100, reduction="sum").item()
                total_valid += v_cnt
        moe_cfg["top_p"], tbptt_cfg["grad_passes"], tbptt_cfg["num_passes"], cg_cfg["enabled"] = old_tp, old_gp, old_np, old_cg
        _set_compiled_mode(True)
        if was_training:
            model_obj.train()
        return total_loss / max(1, total_valid)

    @torch.no_grad()
    def generate_3_answers(model_obj, tag, max_new_tokens=96, is_merged=False):
        was_training = model_obj.training
        old_tp, old_gp, old_np, old_cg = moe_cfg["top_p"], tbptt_cfg["grad_passes"], tbptt_cfg["num_passes"], cg_cfg["enabled"]
        if not is_merged:
            refresh_all_dora_norms()
        _set_compiled_mode(False)
        model_obj.eval()
        moe_cfg["top_p"] = 1.0
        tbptt_cfg["grad_passes"] = 3
        tbptt_cfg["num_passes"] = 3
        cg_cfg["enabled"] = False
        layer23_ctx["idx"] = None
        print(f"\n=== GENERATED ANSWERS [{tag}] (max_new_tokens={max_new_tokens}) ===", flush=True)
        for q_idx, prompt in enumerate(CONTROL_PROMPTS, 1):
            inp_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
            prompt_len = inp_ids.shape[1]
            cur_ids = inp_ids
            stopped_eos = False
            for _ in range(max_new_tokens):
                S_c = cur_ids.shape[1]
                attn_ctx["doc_id"] = torch.zeros((1, S_c), device=device, dtype=torch.int16)
                idx_c = torch.arange(S_c, device=device, dtype=torch.int32).unsqueeze(0)
                attn_ctx["q_n_start"] = torch.clamp(idx_c - 1023, min=0)
                attn_ctx["k_m_start"] = idx_c
                attn_ctx["k_m_end"] = torch.clamp(idx_c + 1024, max=S_c)
                h_last = model_obj.model(input_ids=cur_ids).last_hidden_state[:, -1:, :]
                logits_last = model_obj.logit_softcap(model_obj.lm_head(h_last).float())
                next_tok = torch.argmax(logits_last[:, -1, :], dim=-1, keepdim=True)
                cur_ids = torch.cat([cur_ids, next_tok], dim=1)
                if next_tok.item() == tokenizer.eos_token_id:
                    stopped_eos = True
                    break
            gen_toks = cur_ids.shape[1] - prompt_len
            ans_text = tokenizer.decode(cur_ids[0, prompt_len:], skip_special_tokens=True)
            print(f"--- [{tag}] Q{q_idx} (stopped_by_EOS={stopped_eos}, gen_tokens={gen_toks}) ---\n{ans_text}\n", flush=True)
        moe_cfg["top_p"], tbptt_cfg["grad_passes"], tbptt_cfg["num_passes"], cg_cfg["enabled"] = old_tp, old_gp, old_np, old_cg
        _set_compiled_mode(True)
        if was_training:
            model_obj.train()

    # =========================================================================
    # 9. START CHECK OR --profile-1step
    # =========================================================================
    if profile_1step:
        import logging
        torch._logging.set_logs(recompiles=True)
        m.train()
        tbptt_cfg["grad_passes"] = 1
        tbptt_cfg["prefix_top_p"] = 0.65
        tbptt_cfg["tail_top_p"] = 0.85
        cg_cfg["enabled"] = True

        neg_gen = torch.Generator().manual_seed(seed + 1)
        batches = [build_cpu_batch(i, neg_gen) for i in range(6)]

        def _run_one_step(b_item, n_p, lr_sc):
            tbptt_cfg["num_passes"] = n_p
            adam_opt.zero_grad(set_to_none=True)
            for _, p in dense_dora_ab:
                p.grad = None
            loss_s = compute_pruned_loss_16k(b_item)
            loss_s.backward()
            torch.nn.utils.clip_grad_norm_([p for _, p in named_params], 1.0)
            cautious_muon_step(lr_a * lr_sc, lr_b * lr_sc, wd=0.01)
            adam_opt.step()
            refresh_all_moe_fp8()
            roll_dora_fp8_scales()
            return loss_s

        # 2 warmup steps (step 1: 3p, step 2: 2p -> warms up both CGs + torch.compile)
        for wi, np_w in enumerate((3, 2)):
            t_w0 = time.time()
            l_w = _run_one_step(batches[wi], np_w, 0.25)
            torch.cuda.synchronize()
            print(f"  [WARMUP step={wi+1} ({np_w}p)] loss={l_w.item():.4f} | {(time.time()-t_w0)*1000:.1f}ms", flush=True)

        # 3 timed steady-state steps (steps 3, 4, 5)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.set_sync_debug_mode("warn")
        for si, np_s in enumerate((3, 2, 3), start=3):
            torch.cuda.synchronize()
            t_s0 = time.time()
            l_s = _run_one_step(batches[si - 1], np_s, 0.5)
            torch.cuda.synchronize()
            dt_ms = (time.time() - t_s0) * 1000.0
            torch.cuda.set_sync_debug_mode("default")
            lv = l_s.item()
            torch.cuda.set_sync_debug_mode("warn")
            print(f"  [TIMED  step={si} ({np_s}p)] loss={lv:.4f} | {dt_ms:7.1f}ms ({(micro_bs*seq_len)/(dt_ms/1000.0):8,.0f} tok/s)", flush=True)
        torch.cuda.set_sync_debug_mode("default")
        peak_vram_p = torch.cuda.max_memory_allocated() / 1e9
        print(f"=== STEADY-STATE PEAK VRAM: {peak_vram_p:.2f}GB | {query_gpu_telemetry()} ===", flush=True)

        # Step 6: profiled step (3p FAST+CG) under CPU + CUDA profiler with input shapes
        torch.cuda.synchronize()
        t_p0 = time.time()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
        ) as prof:
            _ = _run_one_step(batches[5], 3, 0.5)
            torch.cuda.synchronize()
        wall_ms = (time.time() - t_p0) * 1000.0

        ka = prof.key_averages()
        launch_calls = sum(e.count for e in ka if e.key == "cudaLaunchKernel")
        print(f"\n=== PROFILE STEP 6 (BS={micro_bs}, S={seq_len}, 3p FAST+CG): Wall={wall_ms:.1f}ms | cudaLaunchKernel={launch_calls} ===", flush=True)
        print("\n--- TOP-10 SELF CUDA TIME ---", flush=True)
        print(ka.table(sort_by="self_cuda_time_total", row_limit=10), flush=True)
        print("\n--- TOP-15 BY INPUT SHAPE (Self CUDA) ---", flush=True)
        print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cuda_time_total", row_limit=15), flush=True)
        print("\n--- TOP-5 SELF CPU TIME ---", flush=True)
        print(ka.table(sort_by="self_cpu_time_total", row_limit=5), flush=True)
        gpu_total_s = time.time() - t_gpu_start
        print(f"=== PROFILE RUN FINISHED in {gpu_total_s:.1f}s (Est. cost: ~${(gpu_total_s/3600)*3.95:.4f}) ===", flush=True)
        return {"wall_ms": wall_ms, "peak_vram_gb": peak_vram_p, "launch_calls": launch_calls}

    base_eval_4 = eval_full_vocab_loss_all(m, max_batches=4)
    print(f"=== START CHECK (4 val batches, 201k vocab): EvalLoss = {base_eval_4:.4f} (expected ~1.096) ===", flush=True)

    ctx_bench_results = {}
    if run_context_bench:
        m.train()
        tbptt_cfg["num_passes"] = 3
        tbptt_cfg["grad_passes"] = 1
        tbptt_cfg["prefix_top_p"] = 0.65
        tbptt_cfg["tail_top_p"] = 0.85
        cg_cfg["enabled"] = False
        for test_s, test_bs in [(4096, 4), (16384, 1)]:
            torch.cuda.reset_peak_memory_stats()
            b_ids_t = torch.randint(0, 10000, (test_bs, test_s), device=device, dtype=torch.long)
            b_doc_t = torch.zeros((test_bs, test_s), device=device, dtype=torch.int16)
            idx_t = torch.arange(test_s, device=device, dtype=torch.int32).unsqueeze(0).expand(test_bs, test_s).contiguous()
            attn_ctx["doc_id"] = b_doc_t
            attn_ctx["q_n_start"] = torch.clamp(idx_t - 1023, min=0)
            attn_ctx["k_m_start"] = idx_t
            attn_ctx["k_m_end"] = torch.clamp(idx_t + 1024, max=test_s)
            layer23_ctx["idx"] = None
            # 1 warmup + 2 timed
            for _ in range(1):
                adam_opt.zero_grad(set_to_none=True)
                for p in [p for _, p in dense_dora_ab]:
                    p.grad = None
                h_t = m.model(input_ids=b_ids_t).last_hidden_state
                h_t.sum().backward()
            torch.cuda.synchronize()
            t0_c = time.time()
            for _ in range(2):
                adam_opt.zero_grad(set_to_none=True)
                for p in [p for _, p in dense_dora_ab]:
                    p.grad = None
                h_t = m.model(input_ids=b_ids_t).last_hidden_state
                h_t.sum().backward()
            torch.cuda.synchronize()
            dt_c = (time.time() - t0_c) / 2.0
            tps_c = (test_bs * test_s) / dt_c
            vram_c = torch.cuda.max_memory_allocated() / 1e9
            ctx_bench_results[f"s{test_s}_bs{test_bs}"] = {"ms": dt_c * 1000.0, "tok_s": tps_c, "vram_gb": vram_c}
            print(f"=== CONTEXT SCALING [S={test_s}, BS={test_bs}, 16384 tok/step]: {dt_c*1000:.1f}ms | {tps_c:,.0f} tok/s | PeakVRAM={vram_c:.2f}GB ===", flush=True)
            del b_ids_t, b_doc_t, idx_t, h_t
            adam_opt.zero_grad(set_to_none=True)
            for p in [p for _, p in dense_dora_ab]:
                p.grad = None
            torch.cuda.empty_cache()

    # =========================================================================
    # 10. 90-SECOND STRESS TEST (75s Fast + 15s Cooldown)
    # =========================================================================
    m.train()
    torch.cuda.set_sync_debug_mode("warn")
    q_pref, stop_ev, th_pref = start_prefetch_stream(500)

    t_train_start = time.time()
    opt_step = 0
    fast_steps = 0
    cd_steps = 0
    fast_time = 0.0
    cd_time = 0.0
    peak_vram_fast = 0.0
    peak_vram_cd = 0.0
    in_cooldown = False
    nan_count = 0
    max_grad_norm_seen = 0.0
    first_loss = None
    last_loss = None

    torch.cuda.reset_peak_memory_stats()
    while True:
        elapsed = time.time() - t_train_start
        if elapsed >= time_limit_s:
            break
        if (not in_cooldown) and (elapsed >= cooldown_start_s):
            in_cooldown = True
            peak_vram_fast = torch.cuda.max_memory_allocated() / 1e9
            torch.cuda.reset_peak_memory_stats()
            cg_cfg["enabled"] = False
            tbptt_cfg["num_passes"] = 3
            tbptt_cfg["grad_passes"] = 1
            tbptt_cfg["prefix_top_p"] = 1.0
            tbptt_cfg["tail_top_p"] = 1.0
            moe_cfg["top_p"] = 1.0
            print(f"\n=== ENTERING COOLDOWN at {elapsed:.1f}s (3/3 passes, top_p=1.0, CG=OFF) | Fast PeakVRAM={peak_vram_fast:.2f}GB ===", flush=True)

        if not in_cooldown:
            cg_cfg["enabled"] = True
            tbptt_cfg["prefix_top_p"] = 0.72
            tbptt_cfg["tail_top_p"] = 0.85
            # Stochastic Loop Depth: 50% steps 2 passes (36), 50% steps 3 passes (48)
            tbptt_cfg["num_passes"] = 3 if (opt_step % 2 == 0) else 2
            tbptt_cfg["grad_passes"] = 1

        item = q_pref.get()
        if item is None or isinstance(item, Exception):
            if isinstance(item, Exception):
                raise item
            break
        ep, cpu_batch = item

        t_s0 = time.time()
        adam_opt.zero_grad(set_to_none=True)
        for _, p in dense_dora_ab:
            p.grad = None

        loss = compute_pruned_loss_16k(cpu_batch)
        loss.backward()

        # Single-pass gradient diagnostics + global L2 norm clipping (reuses _foreach_norm from _grp_stats!)
        def _grp_stats(plist):
            g_list = [p.grad for _, p in plist if p.grad is not None]
            zg = len(plist) - len(g_list)
            if not g_list:
                return torch.empty((0,), device=device, dtype=torch.float32), torch.zeros((), device=device), zg, g_list
            norms = torch.stack(torch._foreach_norm(g_list))
            return norms, norms.max(), zg, g_list

        ns_dora, gn_dora_t, zg_dora, gl_dora = _grp_stats(dense_dora_ab + dense_dora_m)
        ns_ma, gn_ma_t, zg_ma, gl_ma = _grp_stats(moe_lora_a)
        ns_mb, gn_mb_t, zg_mb, gl_mb = _grp_stats(moe_lora_b)
        ns_nm, gn_nm_t, zg_nm, gl_nm = _grp_stats(rmsnorm_named)
        diag_t = torch.stack([gn_dora_t, gn_ma_t, gn_mb_t, gn_nm_t, loss.detach().float()])

        all_norms = torch.cat([ns_dora, ns_ma, ns_mb, ns_nm]).float()
        if all_norms.numel() > 0:
            total_norm = torch.linalg.vector_norm(all_norms)
            clip_coef = torch.clamp(1.0 / (total_norm + 1e-6), max=1.0)
            all_grads = gl_dora + gl_ma + gl_mb + gl_nm
            for g_dt in {g.dtype for g in all_grads}:
                torch._foreach_mul_([g for g in all_grads if g.dtype == g_dt], clip_coef)
        prog = min(1.0, elapsed / max(1.0, time_limit_s))
        lr_scale = min(1.0, (opt_step + 1) / 4.0) * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * prog)))
        cautious_muon_step(lr_a * lr_scale, lr_b * lr_scale, wd=0.01)
        for pg, base_lr in zip(adam_opt.param_groups, [lr_a, lr_moe, lr_norm]):
            pg["lr"] = base_lr * lr_scale
        adam_opt.step()
        refresh_all_moe_fp8()
        roll_dora_fp8_scales()
        if opt_step % 4 == 0:
            refresh_all_dora_norms()

        if opt_step == 0:
            bad_p = [n for n, p in named_params if not torch.isfinite(p).all()]
            bad_inv = [i for i, dm in enumerate(dora_modules) if not torch.isfinite(dm.cached_inv_norm).all()]
            if bad_p or bad_inv:
                raise RuntimeError(f"NaN after step 1 update! bad_params={bad_p[:5]} bad_inv_norm={bad_inv[:5]}")

        # Single host sync per step (after all optimizer kernels are already queued)
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("default")
        gn_dora, gn_ma, gn_mb, gn_nm, lv = diag_t.tolist()
        torch.cuda.set_sync_debug_mode("warn")
        step_max_gn = max(gn_dora, gn_ma, gn_mb, gn_nm)
        if math.isnan(step_max_gn) or math.isinf(step_max_gn) or math.isnan(lv):
            nan_count += 1
            if nan_count >= 2:
                raise RuntimeError(f"Stopping early on NaN at step {opt_step+1}: loss={lv} gn=({gn_dora},{gn_ma},{gn_mb},{gn_nm})")
        else:
            max_grad_norm_seen = max(max_grad_norm_seen, step_max_gn)

        dt_s = time.time() - t_s0
        opt_step += 1
        if opt_step == 1:
            # Reset timer after step 1 (torch.compile + CG capture warmup) so time_limit_s measures steady-state
            t_train_start = time.time()
            fast_steps = 0
            fast_time = 0.0
        elif in_cooldown:
            cd_steps += 1
            cd_time += dt_s
        else:
            fast_steps += 1
            fast_time += dt_s

        if first_loss is None:
            first_loss = lv
        last_loss = lv
        phase = "CD" if in_cooldown else f"FAST({tbptt_cfg['num_passes']}p)"
        if opt_step <= 5 or opt_step % 5 == 0 or in_cooldown:
            print(
                f"  step={opt_step:3d} [{phase:8s}] t={time.time()-t_train_start:5.1f}s | "
                f"loss={lv:.4f} | {dt_s*1000:6.1f}ms ({(micro_bs*seq_len)/dt_s:7,.0f} tok/s) | "
                f"MaxGN={step_max_gn:.3f} (zero_g: D={zg_dora} A={zg_ma} B={zg_mb} N={zg_nm})",
                flush=True,
            )

    stop_ev.set()
    torch.cuda.set_sync_debug_mode("default")
    peak_vram_cd = torch.cuda.max_memory_allocated() / 1e9 if in_cooldown else peak_vram_fast
    total_train_s = time.time() - t_train_start
    total_toks = opt_step * micro_bs * seq_len
    fast_tps = (fast_steps * micro_bs * seq_len) / max(1e-6, fast_time)
    cd_tps = (cd_steps * micro_bs * seq_len) / max(1e-6, cd_time) if cd_steps > 0 else 0.0
    avg_tps = total_toks / max(1e-6, total_train_s)

    print(
        f"\n=== 90s STRESS TEST DONE: steps={opt_step} (fast={fast_steps}, cd={cd_steps}) | "
        f"tokens={total_toks/1e6:.2f}M | Fast={fast_tps:,.0f} tok/s | CD={cd_tps:,.0f} tok/s | Avg={avg_tps:,.0f} tok/s | "
        f"loss: {first_loss:.4f} -> {last_loss:.4f} | NaN={nan_count} | MaxGN={max_grad_norm_seen:.3f} | "
        f"PeakVRAM Fast={peak_vram_fast:.2f}GB CD={peak_vram_cd:.2f}GB ===",
        flush=True,
    )
    print(f"=== GPU AFTER TRAIN: {query_gpu_telemetry()} ===", flush=True)

    # =========================================================================
    # 11. HONEST MERGE CHECK (BEFORE MERGE -> REAL MERGE -> AFTER MERGE)
    # =========================================================================
    eval_before = None
    eval_after = None
    if run_merge_check or run_final_eval:
        eval_before = eval_full_vocab_loss_all(m, max_batches=None, is_merged=False)
        lbl_ev = "EVAL BEFORE MERGE" if run_merge_check else "FINAL EVAL"
        print(f"\n=== {lbl_ev} (201k vocab, {n_val_batches} val batches): EvalLoss = {eval_before:.4f} ===", flush=True)
    if run_merge_check:
        generate_3_answers(m, "BEFORE_MERGE", max_new_tokens=96, is_merged=False)
        print("\n=== PERFORMING REAL MERGE_LORA (DoRA + MoE-LoRA + RMSNorm -> Pure BF16 Base Model) ===", flush=True)
        # Extract all adapter + norm state to CPU, then free training model and merge into clean BF16 /model
        adapter_state = {name: p.detach().cpu() for name, p in named_params}
        moe_lora_modules.clear()
        dora_modules.clear()
        compiled_forwards.clear()
        cg_cfg["cache"].clear()
        attn_ctx.update({"doc_id": None, "q_n_start": None, "k_m_start": None, "k_m_end": None})
        layer23_ctx["idx"] = None
        del m, adam_opt, muon_groups, dora_modules, moe_lora_modules, named_params
        del dense_dora_ab, dense_dora_m, moe_lora_a, moe_lora_b, rmsnorm_named
        del orig_l23_mlp, orig_l23_fwd
        del refresh_all_moe_fp8, refresh_all_dora_norms, cautious_muon_step, compute_pruned_loss_16k, _ce_chunk
        gl_dora = gl_ma = gl_mb = gl_nm = all_grads = None
        ns_dora = ns_ma = ns_mb = ns_nm = all_norms = total_norm = clip_coef = loss = diag_t = None
        torch._dynamo.reset()
        gc.collect()
        torch.cuda.empty_cache()
        print(f"=== VRAM AFTER FREEING TRAIN MODEL: {torch.cuda.memory_allocated()/1e9:.2f}GB ===", flush=True)

        # Restore original HF Attention & TextModel forward? Keep C++ FA for fast identical eval, or pure HF!
        # Using the same C++ FA + clean BF16 base weights (ZERO adapters, ZERO FP8 quantization) tests true merge!
        m_merged = ModelCls.from_pretrained("/model", dtype=dtype, device_map="cuda")
        m_merged.config.use_cache = False
        m_merged.model.language_model.config.use_cache = False
        m_merged.eval()

        n_merged_dora = 0
        n_merged_moe = 0
        n_merged_norm = 0
        with torch.no_grad():
            for name, mod in list(m_merged.named_modules()):
                # 1. Merge FastDoRA: W_eff = (W0 + scale * B @ A) * (m / ||W0 + scale * B @ A||_row)
                if isinstance(mod, nn.Linear) and f"{name}.lora_A" in adapter_state:
                    la = adapter_state[f"{name}.lora_A"].to(device=device, dtype=torch.float32)
                    lb = adapter_state[f"{name}.lora_B"].to(device=device, dtype=torch.float32)
                    mv = adapter_state[f"{name}.m"].to(device=device, dtype=torch.float32)
                    w_eff = mod.weight.data.float() + 2.0 * (lb @ la)
                    row_n = torch.sqrt(w_eff.pow(2).sum(dim=1).clamp(min=1e-8))
                    mod.weight.data.copy_((w_eff * (mv / row_n)[:, None]).to(dtype))
                    n_merged_dora += 1
                # 2. Merge MoE-LoRA: W_gu += scale * B_gu @ A_gu, W_dn += scale * B_dn @ A_dn
                elif hasattr(mod, "gate_up_proj") and f"{name}.lora_A_gu" in adapter_state:
                    la_gu = adapter_state[f"{name}.lora_A_gu"].to(device=device, dtype=torch.float32)
                    lb_gu = adapter_state[f"{name}.lora_B_gu"].to(device=device, dtype=torch.float32)
                    la_dn = adapter_state[f"{name}.lora_A_dn"].to(device=device, dtype=torch.float32)
                    lb_dn = adapter_state[f"{name}.lora_B_dn"].to(device=device, dtype=torch.float32)
                    # [E, 2I, R] @ [R, H] -> [E, 2I, H]
                    mod.gate_up_proj.data.copy_((mod.gate_up_proj.data.float() + 2.0 * torch.einsum("enr,rk->enk", lb_gu, la_gu)).to(dtype))
                    mod.down_proj.data.copy_((mod.down_proj.data.float() + 2.0 * torch.einsum("enr,rk->enk", lb_dn, la_dn)).to(dtype))
                    n_merged_moe += 1
                # 3. Copy unfrozen RMSNorm weights
                elif isinstance(mod, ink_mod.Inkling2LoopRMSNorm) and f"{name}.weight" in adapter_state:
                    mod.weight.data.copy_(adapter_state[f"{name}.weight"].to(device=device, dtype=dtype))
                    n_merged_norm += 1

        print(f"=== MERGED IN VRAM: DoRA={n_merged_dora} | MoE={n_merged_moe} | RMSNorm={n_merged_norm} (ZERO adapters remaining) ===", flush=True)
        eval_after = eval_full_vocab_loss_all(m_merged, max_batches=None, is_merged=True)
        delta_eval = abs(eval_before - eval_after)
        print(f"=== EVAL AFTER MERGE (201k vocab, {n_val_batches} val batches): EvalLoss = {eval_after:.4f} | delta = {delta_eval:.5f} (limit < 0.005) ===", flush=True)
        generate_3_answers(m_merged, "AFTER_MERGE", max_new_tokens=96, is_merged=True)

    gpu_total_s = time.time() - t_gpu_start
    print(f"=== H100 RUN FINISHED in {gpu_total_s:.1f}s (Est. cost: ~${(gpu_total_s/3600)*3.95:.4f}) ===", flush=True)
    return {
        "base_eval_4": base_eval_4,
        "eval_before_merge": eval_before,
        "eval_after_merge": eval_after,
        "steps": opt_step,
        "fast_tps": fast_tps,
        "cd_tps": cd_tps,
        "avg_tps": avg_tps,
        "peak_vram_fast": peak_vram_fast,
        "peak_vram_cd": peak_vram_cd,
        "ctx_bench": ctx_bench_results,
        "gpu_total_s": gpu_total_s,
    }


@app.function(
    gpu="H100",
    volumes={"/build_cache": build_vol},
    timeout=300,
)
def bench_micro():
    global triton, tl
    import time
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import triton
    import triton.language as tl

    t0 = time.time()
    device = "cuda"
    dtype = torch.bfloat16
    import inspect
    import transformers.models.inkling2_loop.modeling_inkling2_loop as ink_mod
    print(f"=== MICRO-BENCH #9 START GPU={torch.cuda.get_device_name(0)} ===", flush=True)

    def bench_fn(fn, warmup=3, rep=15):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(rep):
            fn()
        e1.record()
        e1.synchronize()
        return e0.elapsed_time(e1) / rep

    def cos(a, b):
        return F.cosine_similarity(a.detach().flatten().float(), b.detach().flatten().float(), dim=0).item()

    @triton.jit
    def causal_conv1d_res_fwd_cl_k4_kernel(
        X_ptr, W_ptr, B_ptr, Out_ptr,
        stride_xb, stride_xs, stride_ob, stride_os,
        stride_wc, stride_wk,
        C_dim, S_dim, res_scale, HAS_BIAS: tl.constexpr,
        BLOCK_C: tl.constexpr, CHUNK_S: tl.constexpr,
    ):
        pid_b, pid_c, pid_s = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C_dim
        s0 = pid_s * CHUNK_S
        wb = W_ptr + offs_c * stride_wc
        w0 = tl.load(wb + 0 * stride_wk, mask=mask_c, other=0.0).to(tl.float32)
        w1 = tl.load(wb + 1 * stride_wk, mask=mask_c, other=0.0).to(tl.float32)
        w2 = tl.load(wb + 2 * stride_wk, mask=mask_c, other=0.0).to(tl.float32)
        w3 = tl.load(wb + 3 * stride_wk, mask=mask_c, other=0.0).to(tl.float32)
        b_val = tl.load(B_ptr + offs_c, mask=mask_c, other=0.0).to(tl.float32) if HAS_BIAS else tl.zeros([BLOCK_C], dtype=tl.float32)
        x_b = X_ptr + pid_b * stride_xb + offs_c
        o_b = Out_ptr + pid_b * stride_ob + offs_c
        x_m3 = tl.load(x_b + (s0 - 3) * stride_xs, mask=mask_c & (s0 - 3 >= 0) & (s0 - 3 < S_dim), other=0.0).to(tl.float32)
        x_m2 = tl.load(x_b + (s0 - 2) * stride_xs, mask=mask_c & (s0 - 2 >= 0) & (s0 - 2 < S_dim), other=0.0).to(tl.float32)
        x_m1 = tl.load(x_b + (s0 - 1) * stride_xs, mask=mask_c & (s0 - 1 >= 0) & (s0 - 1 < S_dim), other=0.0).to(tl.float32)
        for step in range(CHUNK_S):
            s = s0 + step
            if s < S_dim:
                x_cur = tl.load(x_b + s * stride_xs, mask=mask_c, other=0.0).to(tl.float32)
                acc = (b_val + x_m3 * w0 + x_m2 * w1 + x_m1 * w2 + x_cur * (w3 + 1.0)) * res_scale
                tl.store(o_b + s * stride_os, acc.to(Out_ptr.dtype.element_ty), mask=mask_c)
                x_m3, x_m2, x_m1 = x_m2, x_m1, x_cur

    @triton.jit
    def causal_conv1d_res_bwd_cl_k4_dx_kernel(
        dOut_ptr, W_ptr, dX_ptr,
        stride_dob, stride_dos, stride_dxb, stride_dxs,
        stride_wc, stride_wk,
        C_dim, S_dim, res_scale,
        BLOCK_C: tl.constexpr, CHUNK_S: tl.constexpr,
    ):
        pid_b, pid_c, pid_s = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C_dim
        s0 = pid_s * CHUNK_S
        wb = W_ptr + offs_c * stride_wc
        w0 = tl.load(wb + 0 * stride_wk, mask=mask_c, other=0.0).to(tl.float32)
        w1 = tl.load(wb + 1 * stride_wk, mask=mask_c, other=0.0).to(tl.float32)
        w2 = tl.load(wb + 2 * stride_wk, mask=mask_c, other=0.0).to(tl.float32)
        w3 = tl.load(wb + 3 * stride_wk, mask=mask_c, other=0.0).to(tl.float32)
        do_b = dOut_ptr + pid_b * stride_dob + offs_c
        dx_b = dX_ptr + pid_b * stride_dxb + offs_c
        g_m3 = tl.zeros([BLOCK_C], dtype=tl.float32)
        g_m2 = tl.zeros([BLOCK_C], dtype=tl.float32)
        g_m1 = tl.zeros([BLOCK_C], dtype=tl.float32)
        for step in range(CHUNK_S + 3):
            t = s0 + step
            if t < S_dim:
                g_cur = tl.load(do_b + t * stride_dos, mask=mask_c, other=0.0).to(tl.float32)
            else:
                g_cur = tl.zeros([BLOCK_C], dtype=tl.float32)
            if step >= 3:
                s_w = s0 + step - 3
                if s_w < S_dim:
                    dx_val = (g_m3 * (w3 + 1.0) + g_m2 * w2 + g_m1 * w1 + g_cur * w0) * res_scale
                    tl.store(dx_b + s_w * stride_dxs, dx_val.to(dX_ptr.dtype.element_ty), mask=mask_c)
            g_m3, g_m2, g_m1 = g_m2, g_m1, g_cur

    @triton.jit
    def triton_scaled_l2_clip_fwd_kernel(
        X_ptr, Scale_ptr, Pos_ptr, Out_ptr, Norm_ptr,
        S_dim, H_dim, threshold, alpha, log_n_floor,
        HAS_LOG: tl.constexpr, D_DIM: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        h = pid_m % H_dim
        bs = pid_m // H_dim
        s_idx = bs % S_dim
        offs_d = tl.arange(0, BLOCK_D)
        m_d = offs_d < D_DIM
        x = tl.load(X_ptr + pid_m * D_DIM + offs_d, mask=m_d, other=0.0).to(tl.float32)
        sc = tl.load(Scale_ptr + h * D_DIM + offs_d, mask=m_d, other=1.0).to(tl.float32)
        v = x * sc
        nrm = tl.sqrt(tl.sum(v * v, axis=0))
        tl.store(Norm_ptr + pid_m, nrm)
        cf = tl.minimum(threshold / (nrm + 1e-6), 1.0)
        y = v * cf
        if HAS_LOG:
            pos = tl.load(Pos_ptr + s_idx).to(tl.float32)
            tau = 1.0 + alpha * tl.log(tl.maximum((pos + 1.0) / log_n_floor, 1.0))
            y = y * tau
        tl.store(Out_ptr + pid_m * D_DIM + offs_d, y.to(Out_ptr.dtype.element_ty), mask=m_d)

    @triton.jit
    def triton_scaled_l2_clip_bwd_kernel(
        dOut_ptr, X_ptr, Scale_ptr, Pos_ptr, Norm_ptr, dX_ptr,
        S_dim, H_dim, threshold, alpha, log_n_floor,
        HAS_LOG: tl.constexpr, D_DIM: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        h = pid_m % H_dim
        bs = pid_m // H_dim
        s_idx = bs % S_dim
        offs_d = tl.arange(0, BLOCK_D)
        m_d = offs_d < D_DIM
        do = tl.load(dOut_ptr + pid_m * D_DIM + offs_d, mask=m_d, other=0.0).to(tl.float32)
        if HAS_LOG:
            pos = tl.load(Pos_ptr + s_idx).to(tl.float32)
            tau = 1.0 + alpha * tl.log(tl.maximum((pos + 1.0) / log_n_floor, 1.0))
            do = do * tau
        x = tl.load(X_ptr + pid_m * D_DIM + offs_d, mask=m_d, other=0.0).to(tl.float32)
        sc = tl.load(Scale_ptr + h * D_DIM + offs_d, mask=m_d, other=1.0).to(tl.float32)
        nrm = tl.load(Norm_ptr + pid_m)
        v = x * sc
        raw = threshold / (nrm + 1e-6)
        if raw < 1.0:
            dot_dv = tl.sum(do * v, axis=0)
            coeff = threshold / (nrm * (nrm + 1e-6) * (nrm + 1e-6) + 1e-12)
            dv = raw * do - coeff * dot_dv * v
        else:
            dv = do
        tl.store(dX_ptr + pid_m * D_DIM + offs_d, (dv * sc).to(dX_ptr.dtype.element_ty), mask=m_d)

    # 1. Benchmark ShortConvRes
    B_c, S_c, C_c = 4, 4096, 2304
    x_sc = torch.randn(B_c, S_c, C_c, device=device, dtype=dtype, requires_grad=True)
    w_sc = torch.randn(C_c, 4, device=device, dtype=torch.float32) * 0.2
    do_sc = torch.randn(B_c, S_c, C_c, device=device, dtype=dtype)
    inv_sqrt5 = 1.0 / (5.0 ** 0.5)
    xf = x_sc.float()
    y_sc_ref = ((F.conv1d(F.pad(xf.transpose(1, 2), (3, 0)), w_sc.unsqueeze(1), groups=C_c).transpose(1, 2) + xf) * inv_sqrt5).to(dtype)
    dx_sc_ref = torch.autograd.grad(y_sc_ref, x_sc, do_sc)[0]
    y_sc_tri = torch.empty_like(x_sc)
    dx_sc_tri = torch.empty_like(x_sc)
    causal_conv1d_res_fwd_cl_k4_kernel[(B_c, triton.cdiv(C_c, 256), triton.cdiv(S_c, 128))](
        x_sc, w_sc, x_sc, y_sc_tri, x_sc.stride(0), x_sc.stride(1), y_sc_tri.stride(0), y_sc_tri.stride(1),
        w_sc.stride(0), w_sc.stride(1), C_c, S_c, inv_sqrt5, HAS_BIAS=False, BLOCK_C=256, CHUNK_S=128, num_warps=4,
    )
    causal_conv1d_res_bwd_cl_k4_dx_kernel[(B_c, triton.cdiv(C_c, 256), triton.cdiv(S_c, 128))](
        do_sc, w_sc, dx_sc_tri, do_sc.stride(0), do_sc.stride(1), dx_sc_tri.stride(0), dx_sc_tri.stride(1),
        w_sc.stride(0), w_sc.stride(1), C_c, S_c, inv_sqrt5, BLOCK_C=256, CHUNK_S=128, num_warps=4,
    )
    print(f"[BENCH ShortConvRes] y_cos={cos(y_sc_ref, y_sc_tri):.6f} | dx_cos={cos(dx_sc_ref, dx_sc_tri):.6f}", flush=True)
    ms_sc_pt = bench_fn(lambda: ((F.conv1d(F.pad(x_sc.detach().float().transpose(1, 2), (3, 0)), w_sc.unsqueeze(1), groups=C_c).transpose(1, 2) + x_sc.detach().float()) * inv_sqrt5).to(dtype))
    ms_sc_tr = bench_fn(lambda: causal_conv1d_res_fwd_cl_k4_kernel[(B_c, triton.cdiv(C_c, 256), triton.cdiv(S_c, 128))](
        x_sc, w_sc, x_sc, y_sc_tri, x_sc.stride(0), x_sc.stride(1), y_sc_tri.stride(0), y_sc_tri.stride(1),
        w_sc.stride(0), w_sc.stride(1), C_c, S_c, inv_sqrt5, HAS_BIAS=False, BLOCK_C=256, CHUNK_S=128, num_warps=4,
    ))
    ms_scb_tr = bench_fn(lambda: causal_conv1d_res_bwd_cl_k4_dx_kernel[(B_c, triton.cdiv(C_c, 256), triton.cdiv(S_c, 128))](
        do_sc, w_sc, dx_sc_tri, do_sc.stride(0), do_sc.stride(1), dx_sc_tri.stride(0), dx_sc_tri.stride(1),
        w_sc.stride(0), w_sc.stride(1), C_c, S_c, inv_sqrt5, BLOCK_C=256, CHUNK_S=128, num_warps=4,
    ))
    print(f"  ShortConvRes fwd: PT={ms_sc_pt:.3f}ms vs Triton={ms_sc_tr:.3f}ms | bwd={ms_scb_tr:.3f}ms", flush=True)

    # 2. Benchmark LogScaledL2Clip (frozen scale!)
    H_q, D_q = 24, 128
    x_l2 = torch.randn(B_c, S_c, H_q, D_q, device=device, dtype=dtype, requires_grad=True)
    sc_l2 = torch.rand(H_q, D_q, device=device, dtype=torch.float32) + 0.5
    pos_l2 = torch.arange(S_c, device=device, dtype=torch.long)
    do_l2 = torch.randn(B_c, S_c, H_q, D_q, device=device, dtype=dtype)
    thr, alp, nfl = 5.0, 0.2, 512.0
    v_ref = x_l2.float() * sc_l2
    nrm_r = torch.linalg.vector_norm(v_ref, dim=-1, keepdim=True)
    y_l2_r = v_ref * (thr / (nrm_r + 1e-6)).clamp(max=1.0)
    tau_r = 1.0 + alp * torch.log(((pos_l2.unsqueeze(0) + 1).float() / nfl).clamp(min=1.0))
    y_l2_r = (y_l2_r * tau_r[..., None, None]).to(dtype)
    dx_l2_r = torch.autograd.grad(y_l2_r, x_l2, do_l2)[0]
    y_l2_t = torch.empty_like(x_l2)
    dx_l2_t = torch.empty_like(x_l2)
    M_l2 = B_c * S_c * H_q
    nrm_buf = torch.empty((M_l2,), device=device, dtype=torch.float32)
    triton_scaled_l2_clip_fwd_kernel[(M_l2,)](x_l2, sc_l2, pos_l2, y_l2_t, nrm_buf, S_c, H_q, thr, alp, nfl, HAS_LOG=True, D_DIM=D_q, BLOCK_D=128, num_warps=4)
    triton_scaled_l2_clip_bwd_kernel[(M_l2,)](do_l2, x_l2, sc_l2, pos_l2, nrm_buf, dx_l2_t, S_c, H_q, thr, alp, nfl, HAS_LOG=True, D_DIM=D_q, BLOCK_D=128, num_warps=4)
    print(f"[BENCH LogScaledL2Clip] y_cos={cos(y_l2_r, y_l2_t):.6f} | dx_cos={cos(dx_l2_r, dx_l2_t):.6f}", flush=True)
    ms_l2_pt = bench_fn(lambda: ((x_l2.detach().float() * sc_l2) * (thr / (torch.linalg.vector_norm(x_l2.detach().float() * sc_l2, dim=-1, keepdim=True) + 1e-6)).clamp(max=1.0) * tau_r[..., None, None]).to(dtype))
    ms_l2_tr = bench_fn(lambda: triton_scaled_l2_clip_fwd_kernel[(M_l2,)](x_l2, sc_l2, pos_l2, y_l2_t, nrm_buf, S_c, H_q, thr, alp, nfl, HAS_LOG=True, D_DIM=D_q, BLOCK_D=128, num_warps=4))
    ms_l2b_tr = bench_fn(lambda: triton_scaled_l2_clip_bwd_kernel[(M_l2,)](do_l2, x_l2, sc_l2, pos_l2, nrm_buf, dx_l2_t, S_c, H_q, thr, alp, nfl, HAS_LOG=True, D_DIM=D_q, BLOCK_D=128, num_warps=4))
    print(f"  LogScaledL2Clip fwd: PT={ms_l2_pt:.3f}ms vs Triton={ms_l2_tr:.3f}ms | bwd={ms_l2b_tr:.3f}ms", flush=True)
    print(f"=== MICRO-BENCH #9 FINISHED in {time.time()-t0:.2f}s ===", flush=True)


@app.local_entrypoint()
def main(
    dataset: str = "yahma/alpaca-cleaned",
    seq_len: int = 4096,
    tokens_per_step: int = 16384,
    time_limit_s: float = 90.0,
    cooldown_start_s: float = 75.0,
    run_context_bench: bool = True,
    run_merge_check: bool = True,
    run_final_eval: bool = False,
    profile_1step: bool = False,
    bench_micro_mode: bool = False,
    moe_lora_start: int = 6,
    full_gc_layers: int = 10,
    moe_gc_layers: int = 8,
):
    if bench_micro_mode:
        bench_micro.remote()
        return
    res = train_h100.remote(
        dataset=dataset,
        seq_len=seq_len,
        tokens_per_step=tokens_per_step,
        time_limit_s=time_limit_s,
        cooldown_start_s=cooldown_start_s,
        run_context_bench=run_context_bench,
        run_merge_check=run_merge_check,
        run_final_eval=run_final_eval,
        profile_1step=profile_1step,
        moe_lora_start=moe_lora_start,
        full_gc_layers=full_gc_layers,
        moe_gc_layers=moe_gc_layers,
    )
    print(f"\nFINAL SUMMARY: {res}", flush=True)
