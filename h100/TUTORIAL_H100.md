# H100 Tutorial (`sm_90a`): Build, Run & Kernel Integration Guide

This guide covers setting up `h100/train_inkling_h100.py` on Modal or a local H100 80GB machine, plus a practical checklist for integrating the custom CUDA/C++ and Triton kernels into your own PyTorch/HuggingFace training loop cleanly.

---

## 1. Environment Requirements

- **GPU:** `NVIDIA H100 80GB HBM3` (`sm_90a`, compute capability `9.0`). `fa_inkling.cu` relies on Hopper WGMMA instructions and will not compile on A100/Ada — use [`../a100/`](../a100/) for A100.
- **CUDA Toolkit:** `12.6+` (`torch._scaled_grouped_mm`, FP8 E4M3, Hopper WGMMA).
- **PyTorch:** `2.7.0+cu126` (or newer with `_scaled_mm` / `_scaled_grouped_mm`).
- **Python packages:** `triton>=3.0`, `transformers>=4.45.0`, `datasets`, `accelerate`, `huggingface_hub`, `peft`.
- **Local C++ build tools (if not using Modal):** `ninja`, `g++`, `nvcc` 12.6+, **CUTLASS v3.6.0**.

---

## 2. Building `inkling_h100_ops.so`

### Option A — Modal CPU Builder (Easiest, zero GPU spend)

```bash
cd larpInk
modal run h100/kernels/build_and_bench.py --build-only
```
This spins up a CPU container (`CUDA 12.6`, `ninja-build`, `ccache`, CUTLASS `v3.6.0`), hashes `*.cu`/`*.cpp` in `h100/kernels/`, compiles with `-gencode arch=compute_90a,code=sm_90a`, and saves `/build_cache/inkling_h100_ops.so` in the Modal Volume `h100-cpp-build-cache`.

Run standalone C++ kernel micro-benchmarks on H100:
```bash
modal run h100/kernels/build_and_bench.py --skip-build --target full
```

### Option B — Local Linux H100 Build (Without Modal)

1. Clone CUTLASS `v3.6.0`:
   ```bash
   git clone --depth 1 --branch v3.6.0 https://github.com/NVIDIA/cutlass.git /opt/cutlass
   ```
2. Compile `inkling_h100_ops.so` (exact flags from `build_and_bench.py`):
   ```bash
   python3 - <<'EOF'
   import os, sysconfig, torch
   import torch.utils.cpp_extension as cpp_ext

   incs = [p for p in cpp_ext.include_paths("cuda") + [
       "/usr/local/cuda/include", "/opt/cutlass/include", "/opt/cutlass/tools/util/include",
       "h100/kernels", sysconfig.get_path("include")
   ] if os.path.isdir(p)]
   libs = [p for p in cpp_ext.library_paths("cuda") + ["/usr/local/cuda/lib64"] if os.path.isdir(p)]
   abi = int(torch._C._GLIBCXX_USE_CXX11_ABI)
   inc_f = " ".join(f"-I{p}" for p in dict.fromkeys(incs))
   lib_f = " ".join(f"-L{p}" for p in dict.fromkeys(libs))
   rpath_f = " ".join(f"-Wl,-rpath,{p}" for p in dict.fromkeys(libs))

   nvcc = (f"nvcc -std=c++17 -O3 --use_fast_math --expt-relaxed-constexpr "
           f"-gencode arch=compute_90a,code=sm_90a --compiler-options -fPIC "
           f"-DTORCH_EXTENSION_NAME=inkling_h100_ops -DTORCH_API_INCLUDE_EXTENSION_H "
           f"-D_GLIBCXX_USE_CXX11_ABI={abi} {inc_f}")
   cxx = (f"g++ -std=c++17 -O3 -fPIC -DTORCH_EXTENSION_NAME=inkling_h100_ops "
          f"-DTORCH_API_INCLUDE_EXTENSION_H -D_GLIBCXX_USE_CXX11_ABI={abi} {inc_f}")

   os.makedirs("build", exist_ok=True)
   for cu in ["dora_cublaslt.cu", "fa_inkling.cu", "moe_cutlass.cu"]:
       assert os.system(f"{nvcc} -c h100/kernels/{cu} -o build/{cu}.o") == 0
   assert os.system(f"{cxx} -c h100/kernels/bindings.cpp -o build/bindings.cpp.o") == 0
   assert os.system(
       f"g++ -shared build/*.o {lib_f} {rpath_f} "
       f"-ltorch -ltorch_cpu -ltorch_python -lc10 -lc10_cuda -ltorch_cuda -lcudart -lcublasLt -lcublas "
       f"-o inkling_h100_ops.so"
   ) == 0
   print("Built inkling_h100_ops.so OK")
   EOF
   ```
3. Point `so_path` in `train_inkling_h100.py` to `./inkling_h100_ops.so` (or copy it to `/build_cache/inkling_h100_ops.so`).

---

## 3. Dataset Format & Running SFT

`h100/train_inkling_h100.py` packs samples on the fly to `seq_len=4096` with `doc_id` boundaries (preventing cross-sample attention) and `-100` prompt masking.

Supported `.jsonl` / HuggingFace dataset schemas:
- **Alpaca format:** `{"instruction": "...", "input": "...", "output": "..."}`
- **ChatML / ShareGPT format:** `{"messages": [{"role": "system"|"user"|"assistant", "content": "..."}]}`

### Recommended Commands

1. **6-step speed & profiler sanity check (`--profile-1step`):**
   ```bash
   modal run h100/train_inkling_h100.py \
     --tokens-per-step 16384 \
     --moe-lora-start 6 \
     --full-gc-layers 10 \
     --moe-gc-layers 8 \
     --profile-1step
   ```
2. **25s control SFT + full 201k-vocab `EvalLoss`:**
   ```bash
   modal run h100/train_inkling_h100.py \
     --tokens-per-step 16384 \
     --moe-lora-start 6 \
     --full-gc-layers 10 \
     --moe-gc-layers 8 \
     --time-limit-s 25 \
     --cooldown-start-s 18 \
     --no-run-context-bench \
     --no-run-merge-check \
     --run-final-eval
   ```
3. **Longer SFT run + real BF16 merge verification:**
   ```bash
   modal run h100/train_inkling_h100.py \
     --dataset my_data.jsonl \
     --tokens-per-step 16384 \
     --time-limit-s 600 \
     --cooldown-start-s 520 \
     --full-gc-layers 10 \
     --moe-gc-layers 8 \
     --no-run-context-bench \
     --run-merge-check
   ```

> **Note on saving checkpoints:** `train_inkling_h100.py` was built as a speed/eval stress runner — it merges and verifies `m_merged` in VRAM without writing `.safetensors` to disk. To persist the merged HF checkpoint, add after `eval_after = eval_full_vocab_loss_all(m_merged, ...)` inside `train_h100`:
> ```python
> save_dir = "/cache_vol/merged_h100_model"
> m_merged.save_pretrained(save_dir)
> tokenizer.save_pretrained(save_dir)
> vol.commit()
> ```

---

## 4. How to Plug These Kernels Into Your Own PyTorch Code

Follow this exact order so everything works smoothly without shape bugs or Dynamo errors:

1. **Load the compiled extension:**
   ```python
   import importlib.util
   spec = importlib.util.spec_from_file_location("inkling_h100_ops", "/path/to/inkling_h100_ops.so")
   ops = importlib.util.module_from_spec(spec)
   spec.loader.exec_module(ops)
   ```

2. **Drop in `FastFP8DoRALinear` (works on any LLM `nn.Linear`):**
   - Copy `fp8_amax_part_kernel`, `fp8_quant_from_part_kernel`, `dora_1x_lora_scale_inplace_kernel`, `dora_1x_bwd_scale_dm_kernel`, `fp8_quant_from_dynmax_kernel`, `fp8_quant_delayed_and_amax_kernel`, `_Fused1xFP8DoRAFn`, and `FastFP8DoRALinear`.
   - Call `roll_dora_fp8_scales()` after every `optimizer.step()`, and call `refresh_all_dora_norms()` every ~4 steps and before `eval()` / merging.
   - Keep `@torch.compiler.disable` on `FastFP8DoRALinear.forward` so Dynamo does not trace around custom FP8 ops.

3. **Drop in `PatchedFP8Experts` (FP8 MoE + LoRA):**
   - Copy `moe_z_dn_fp8_kernel`, `moe_dact_add_lora_kernel`, `moe_lora_dB_fp8_v2_kernel`, `moe_lora_dA_dn_fp8_kernel`, `moe_lora_b_add_triton_kernel`, and `PatchedFP8Experts`.
   - **Check hardcoded dimensions in `h100/kernels/moe_cutlass.cu`:** `MOE_H = 2304`, `MOE_I = 1152`, `MOE_R = 16`. If your target model has different dimensions, update those constants at the top of `moe_cutlass.cu` and rebuild `.so`.
   - Separate `enable_lora=(idx >= moe_lora_start)` from `enable_grad=(idx >= freeze_until)` so layers that require `dx` still keep `w_dn_T_fp8_T`/`w_gu_T_fp8_T`.

4. **Drop in `fa_inkling.cu` (Hopper WGMMA Attention):**
   - Pass precomputed `doc_id` (`int16 [B, S]`) and `q_n_start`, `k_m_start`, `k_m_end` (`int32 [B, S]`).
   - Route `q, k, v` through `_fast_t12_contiguous_4d` (`[B, S, H, D] ↔ [B, H, S, D]`, `D=128`).

5. **Drop in the Triton Elementwise Stack (`RMSNorm`, `ShortConvRes`, `Scaled/LogL2Clip`, `MulSigmoid`, `MessageGateMix`):**
   - Do **not** wrap modules patched with these Triton kernels in `torch.compile` — keep `torch.compile` strictly on pure PyTorch ops like `Inkling2LoopTopKRouter` (with `.contiguous()` inputs).

6. **Hybrid Selective GC + CUDA Graph Prefix:**
   - For `BS=4, S=4096` on 80GB H100, `--full-gc-layers 10 --moe-gc-layers 8` keeps `PeakVRAM` around `~74.9 GB`.
   - When capturing multiple CUDA Graphs (e.g. `2p` and `3p` prefixes), **reuse both one shared `pool` and one shared `stream`** (`cg_cfg.setdefault("stream", torch.cuda.Stream())`) + `torch.cuda.empty_cache()` before capture — otherwise PyTorch allocates separate private pools per stream (+3–5 GB VRAM).

---

## 5. Troubleshooting

| Issue | Cause | Fix |
|---|---|---|
| `Missing prebuilt C++ extension /build_cache/inkling_h100_ops.so` | `.so` hasn't been built yet | Run `modal run h100/kernels/build_and_bench.py --build-only` |
| `CUDA out of memory` when lowering `--full-gc-layers` | Non-GC attention + DoRA activations take ~2.5–2.9 GB/layer | Increase `--full-gc-layers 10` (or `11`) and decrease `--moe-gc-layers 8` (or `7`), sum = `18` |
| `EvalLoss` regresses (`0.96+` vs `<= 0.950`) | Too few 3-pass steps (`75/25`) or `prefix_top_p` too low | Use `50/50 2p/3p` (`opt_step % 2 == 0`), `prefix_top_p >= 0.72`, and a `top_p=1.0` Cooldown |
| `nvcc fatal: Unsupported gpu architecture 'compute_90a'` | CUDA `< 12.0` or non-Hopper GPU | Upgrade CUDA to `12.6+` (or use `../a100/` on Ampere) |
| Slow step 1 (~20–35s) | One-time Triton JIT + `TopKRouter` compile + CUDA Graph capture | Warmup runs before `t_train_start`; steady-state starts on step 2–3 |
