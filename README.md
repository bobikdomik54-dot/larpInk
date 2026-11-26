# larpInk — Single-GPU H100 & A100 SFT Engine for Looped 48-Layer MoE

> **Backstory:** Built purely out of boredom together with **Antigravity (Gemini Neon 4)** — an experiment to see how fast we could push real SFT on a single **NVIDIA H100 80GB HBM3** (and **A100 80GB**) for the 48-layer looped MoE model [`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini) using custom CUDA/CUTLASS (`sm_90a`) and Triton kernels without degrading `EvalLoss`.
>
> **Status:** This project probably won't be actively developed further, **but it is 100% open for forks, kernel extraction, and further optimization** — take whatever you need and build on top of it.

---

## Real SFT Training Benchmarks (`Inkling-2-Mini`, `seq_len=4096`)

| GPU | Without Optimization | With `larpInk` | Speedup | Peak VRAM |
|---|---:|---:|---:|---:|
| **H100 80GB** (`184.28M` trainable params) | `5,748 tok/s` | **`16,979 tok/s` avg** (`19,396 tok/s` peak) | **2.95×** | `74.90 GB` |
| **A100 80GB** (`13.12M` LoRA) | `1,310 tok/s` | **`6,565 tok/s` avg** (`6,628 tok/s` peak) | **5.06×** | `54.45 GB` |

- **Accuracy:** H100 25s SFT drops full-vocab `EvalLoss` (`201k`) from `1.0187 → 0.9447` (`NaN = 0`, `zero_grad = 0`); A100 100-step SFT drops `EvalLoss` from `1.0962 → 0.9528` (and A100 `184.28M` Near-Full-FT runs at `6,274 tok/s` in `57.36 GB`).

---

## What We Built

- **Hopper WGMMA FlashAttention-3 ([`h100/kernels/fa_inkling.cu`](h100/kernels/fa_inkling.cu)):** Custom C++/CUDA WGMMA attention supporting per-head relative position bias (`rel_states @ rel_proj`), attention sinks, sliding windows, and packed `doc_id` boundaries with zero cross-document leakage.
- **FP8 Grouped MoE + Zero-Alloc MoE-LoRA ([`h100/kernels/moe_cutlass.cu`](h100/kernels/moe_cutlass.cu) + Triton):** Dynamic `top_p` expert routing & counting sort, `torch._scaled_grouped_mm` base matmuls, fused soft-capped SwiGLU + FP8 quantization, and Triton LoRA `A`/`B` kernels operating directly on FP8 activations.
- **FastFP8DoRALinear:** 1-pass Delayed FP8 quantization (`margin=1.05`, 4-step history) in forward, exact dynamic `pmax` in backward, and FP8-recomputed row norms (`refresh_norm()`) with `w0_bf16` removed from GPU VRAM.
- **Zero-Recompile Elementwise Triton Stack:** `TritonShortConvResFn`, `TritonRMSNormFn`, `TritonMulSigmoidFn`, `TritonScaledL2ClipFn`, `TritonMessageGateMixFn`, plus contiguous-slice `TritonDepthAttnPairFn` — eliminates PyTorch eager elementwise bloat and Dynamo recompiles (keeping `torch.compile` strictly on `Inkling2LoopTopKRouter`).
- **Hybrid Selective GC + T-BPTT + Shared-Stream CUDA Graphs:** Full GC on the first 10 trainable layers (`Full:10`) + **MoE-only GC** (`layer.mlp`) on the last 8 (`MoE:8`), shared `cg_stream` + `cg_pool` across `2p`/`3p` prefix graphs, `Batched Cautious Muon` (`ns5`) + `Fused AdamW`, and `Pruned 16k Vocab CE Loss`.
- **Standalone A100-80GB Branch ([`a100/`](a100/)):** Pure Triton + PyTorch stack (no `.cu` compilation) with INT8 Weight-Only Grouped MoE, Triton FlashAttention-LR-Varlen, DoRA, checkpoint saving/resume, and `merge_lora()`.

---

## Repository Layout

```text
larpInk/
├── README.md
├── .gitignore
├── h100/
│   ├── TUTORIAL_H100.md             # Step-by-step H100 build, run & integration guide
│   ├── train_inkling_h100.py        # Best H100 SFT script (19.4k peak / 17.0k avg tok/s)
│   └── kernels/
│       ├── build_and_bench.py       # CPU Modal builder for inkling_h100_ops.so + H100 micro-benches
│       ├── bindings.cpp             # PyBind11 extension bindings
│       ├── fa_inkling.cu            # Hopper WGMMA FlashAttention-3 + rel_bias + sinks + doc_id
│       ├── moe_cutlass.cu           # FP8 Top-P MoE: routing, SwiGLU, quant, scatter/gather
│       └── dora_cublaslt.cu         # cuBLASLt/WMMA DoRA kernels (optional experimental)
└── a100/
    ├── TUTORIAL_A100.md             # A100-80GB guide (INT8 MoE, checkpoints, resume, merge)
    └── train_inkling_a100.py        # Best A100 SFT script (pure Triton + PyTorch)
```

---

## Quickstart

### H100 (`sm_90a`, Modal)
```bash
# 1. Build inkling_h100_ops.so on CPU (no GPU cost)
modal run h100/kernels/build_and_bench.py --build-only

# 2. Run 6-step benchmark + profiler
modal run h100/train_inkling_h100.py --tokens-per-step 16384 --moe-lora-start 6 --full-gc-layers 10 --moe-gc-layers 8 --profile-1step

# 3. Run real SFT + full-vocab eval & merge check
modal run h100/train_inkling_h100.py --dataset yahma/alpaca-cleaned --tokens-per-step 16384 --time-limit-s 90 --cooldown-start-s 75 --no-run-context-bench --run-merge-check
```
See [**`h100/TUTORIAL_H100.md`**](h100/TUTORIAL_H100.md) for local Linux builds and how to drop these kernels into your own training code.

### A100 (`sm_80`, Modal, no C++ build required)
```bash
modal run a100/train_inkling_a100.py --dataset yahma/alpaca-cleaned --grad-accum 4 --time-limit-s 120 --cooldown-start-s 100 --merge-after
```
See [**`a100/TUTORIAL_A100.md`**](a100/TUTORIAL_A100.md) for checkpointing, `--resume`, `--merge-only`, and all CLI flags.
