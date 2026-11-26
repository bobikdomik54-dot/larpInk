# larpInk — Single-GPU H100 & A100 SFT Engine for Looped 48-Layer MoE

> **Backstory:** Built purely out of boredom together with **Antigravity (Gemini Neon 4)** — an experiment to see how fast we could push real SFT on a single **NVIDIA H100 80GB HBM3** (and **A100 80GB**) for the 48-layer looped MoE model [`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini) using custom CUDA/CUTLASS (`sm_90a`) and Triton kernels without degrading `EvalLoss`.
>
> **Status:** This project probably won't be actively developed further, **but it is 100% open for forks, kernel extraction, and further optimization** — take whatever you need and build on top of it.

---

## Real Training Benchmarks (`H100 80GB` / `H200`, 48-Layer SFT)

All runs below were measured end-to-end on `thinkingmachines/Inkling-2-Mini` (`seq_len=4096`, `tokens/step=16384`, `184.28M` trainable params: `DoRA=90`, `MoE=24`, `FA=24`, `DepthAttn=49`, `Conv1d=48`, `Norm=43`) with honest full-vocab **`EvalLoss` across all 201k tokens (`44` validation batches, `0.18M` tokens)**:

### 1. End-to-End Speed & Accuracy Progression

| Run / Config | GPU / Mem Clock | Fast 2p / 3p (`tok/s`) | Fast / CD / Avg (`tok/s`) | Peak VRAM | Power / 700W | EvalLoss (201k) |
|---|---|---:|---:|---:|---:|---:|
| **Baseline** (`50/50 2p/3p`, `top_p=0.72`, Full GC) | H100 `2619 MHz` | — | `14,866 / 13,434 / 15,090` | `74.22 / 74.44 GB` | — | `0.9471` |
| **Run A** (`BS=4, 75/25`, `top_p=0.65`, `Full:11+MoE:7`) | H200 `3201 MHz` | `19,213` (852.8ms) / — | `16,856 / 15,059 / 17,036` | `74.69 / 74.92 GB` | `644.95 W` | `0.9608` |
| **Run B** (`BS=6, 24k`, `Full:16+MoE:2`, `--profile-1step`) | H100 `2619 MHz` | `16,534` (1486ms) / `14,498` (1695ms) | — | `69.18 GB` | `550.12 W` | — |
| **Run C** (`BS=4`, `moe>=12`, `Full:11+MoE:7`) | H100 `2619 MHz` | `17,529` (934.7ms) / — | `14,932 / 13,902 / 15,279` | `74.24 / 74.46 GB` | `594.77 W` | `0.9679` |
| **Post Round-3 Profile** (`Full:10+MoE:8`, `BS=4`) | H100 `2619 MHz` | `18,984` (863.0ms) / `16,430` (997.2ms) | — | `74.37 GB` | `596.45 W` | — |
| **Final 25s SFT** (`BS=4, 50/50`, `top_p=0.72`, `Full:10+MoE:8`) | H100 `2619 MHz` | **`19,396` (844.7ms) / `16,871` (971.1ms)** | **`17,003 / 14,816 / 16,979`** | **`74.90 / 74.95 GB`** | **`572.99 W`** | **`0.9447`** |

**Final Run Highlights (`NVIDIA H100 80GB HBM3`, `SM=1980MHz`, `Mem=2619MHz`):**
- Peak 2-pass step throughput: **`19,396 tok/s`** (`844.7 ms/step`), Fast-phase avg **`17,003 tok/s`**, overall avg **`16,979 tok/s`** (`27` steps in 25s, `train_loss: 0.9604 → 0.8177`).
- **`EvalLoss = 0.9447`** (down from `base_eval_4 = 1.0187`, beating both the `0.950` target and the `0.9471` baseline).
- **Clean gradients:** `NaN = 0`, `zero_g = 0` (`D=0 A=0 B=0 N=0` across all DoRA, MoE-LoRA, and RMSNorm weights), `MaxGN = 0.207`.
- Step 6 profiler (`BS=4`, `3p`): `Self CPU = 985.6 ms`, `Self CUDA = 987.9 ms`, `Command Buffer Full` flushes cut **3.8×** (`1080 → 287`).

### 2. Custom Kernel Micro-Benchmarks on H100 (vs PyTorch / cuBLAS / WMMA)

| Kernel / Op | Baseline | larpInk | Speedup | Accuracy (`cos`) |
|---|---:|---:|---:|---:|
| **`ShortConvRes` fwd** (`4×4096×2304` BF16) | `1.542 ms` (PyTorch) | **`0.091 ms`** (Triton) | **16.9×** | `y=1.000000`, `dx=1.000000` |
| **`RMSNorm` fwd** (`16384×2304` BF16) | `0.832 ms` (PyTorch) | **`0.056 ms`** (Triton) | **14.8×** | `y=1.000000`, `dx=0.999996`, `dw=1.000000` |
| **`MulSigmoid` fwd** (`16384×3072` BF16) | `0.820 ms` (PyTorch) | **`0.103 ms`** (Triton) | **8.0×** | `y=1.000000`, `dx=1.000000`, `dg=1.000000` |
| **`LogScaledL2Clip` fwd** (`4×4096×24×128`) | `1.271 ms` (PyTorch) | **`0.304 ms`** (Triton 1D; `<7ms` total in 2D) | **4.2×+** | `y=1.000000`, `dx=1.000000` |
| **`moe_lora_b_add` (`dn`)** (`128` experts) | `~0.882 ms` (C++ WMMA) | **`0.357 ms`** (Triton `64×128`) | **2.47×** | `out=1.000000`, `y_exp=0.999968` |
| **`moe_z_dn_fp8_kernel`** (`prefix top_p=0.65`) | `0.061 ms` (`_scaled_mm`) | **`0.032 ms`** (Triton) | **1.91×** | `0.999642` |
| **`moe_dact_add_lora`** (`92k/131k`) | `0.247 ms` (cuBLAS `addmm_`) | **`0.118 ms`** (Triton) | **2.09×** | `1.000000` |
| **`Delayed FP8 DoRA Gate`** (`16k×2304→3072`) | `0.991 ms` (2-pass) | **`0.902 ms`** (1-pass fwd) | **-9%** | `fwd=0.999025`, `dx=0.999404`, `dA/dB=1.000000` |

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
