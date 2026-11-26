# larpInk — High-Throughput SFT Engine for Looped 48-Layer MoE (`H100` & `A100`)

> **О проекте:** Этот проект был создан тупо от скуки вместе с **Antigravity (Gemini Neon 4)** — ради эксперимента выжать максимум из **одной** GPU (`NVIDIA H100 80GB HBM3` и `NVIDIA A100 80GB`) на реальном SFT-обучении сложной 48-слойной циклической MoE-модели [`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini), без потери математической точности и качества (`EvalLoss`).

Внутри — полный боевой стек кастомных CUDA/CUTLASS (`sm_90a`) и Triton-ядер, гибридный selective gradient checkpointing, T-BPTT с CUDA Graphs и Batched Cautious Muon + Fused AdamW, упакованный в две самостоятельные ветки:
- [`h100/`](h100/) — флагманский стек для **Hopper (H100 80GB HBM3)**: FP8 DoRA, CUTLASS/ATen Grouped FP8 MoE, Hopper WGMMA FlashAttention-3 и набор fused Triton-ядер (`ShortConvRes`, `RMSNorm`, `DepthAttn V6`, `Scaled/LogL2Clip`, `MulSigmoid`, `MessageGateMix`).
- [`a100/`](a100/) — чистая Triton/PyTorch-версия для **Ampere (A100 80GB)** без компиляции C++ `.cu`: INT8 Weight-Only Grouped MoE, Triton FlashAttention-LR-Varlen, DoRA, чекпоинты с `resume` и встроенный `merge_lora()` в чистый BF16.

---

## Статистика и бенчмарки (`H100` / `H200`, реальный SFT 48 слоёв)

Все замеры ниже сняты end-to-end на `thinkingmachines/Inkling-2-Mini` (`seq_len=4096`, `tokens/step=16384`, `184.28M` обучаемых параметров: `DoRA=90`, `MoE=24`, `FA=24`, `DepthAttn=49`, `Conv1d=48`, `Norm=43`) с честной оценкой **`EvalLoss` по полному 201k-словарю (`44` вал-батча, `0.18M` токенов)**:

### 1. Эволюция производительности (Baseline → Финал)

| Конфигурация | GPU / Mem Clock | Fast 2p / 3p (`tok/s`) | Fast / CD / Avg (`tok/s`) | Peak VRAM | Power / 700W | EvalLoss (201k) |
|---|---|---:|---:|---:|---:|---:|
| **Baseline** (`50/50 2p/3p`, `top_p=0.72`, `Full GC`) | H100 `2619 MHz` | — | `14,866 / 13,434 / 15,090` | `74.22 / 74.44 GB` | — | `0.9471` |
| **Прогон A** (`BS=4, 75/25`, `top_p=0.65`, `Full:11+MoE:7`) | H200 `3201 MHz` | `19,213` (852.8ms) / — | `16,856 / 15,059 / 17,036` | `74.69 / 74.92 GB` | `644.95 W` | `0.9608` |
| **Прогон B** (`BS=6, 24k`, `Full:16+MoE:2`, `--profile-1step`) | H100 `2619 MHz` | `16,534` (1486ms) / `14,498` (1695ms) | — | `69.18 GB` | `550.12 W` | — |
| **Прогон C** (`BS=4`, `moe>=12`, `Full:11+MoE:7`) | H100 `2619 MHz` | `17,529` (934.7ms) / — | `14,932 / 13,902 / 15,279` | `74.24 / 74.46 GB` | `594.77 W` | `0.9679` |
| **Профиль после Round-3** (`Full:10+MoE:8`, `BS=4`) | H100 `2619 MHz` | `18,984` (863.0ms) / `16,430` (997.2ms) | — | `74.37 GB` | `596.45 W` | — |
| **Финал 25s SFT** (`BS=4, 50/50`, `top_p=0.72`, `Full:10+MoE:8`) | H100 `2619 MHz` | **`19,396` (844.7ms) / `16,871` (971.1ms)** | **`17,003 / 14,816 / 16,979`** | **`74.90 / 74.95 GB`** | **`572.99 W`** | **`0.9447`** |

**Ключевые факты по финальному прогону (`H100 80GB HBM3, SM=1980MHz, Mem=2619MHz`):**
- Пик на 2-витковом шаге: **`19,396 tok/s`** (`844.7 ms/шаг`), средняя по Fast-фазе **`17,003 tok/s`**, общая **`16,979 tok/s`** (`27` шагов за 25с, `train_loss: 0.9604 → 0.8177`).
- **`EvalLoss = 0.9447`** (со старта `base_eval_4 = 1.0187`, лучше лимита `0.950` и лучше бейзлайна `0.9471`).
- **`NaN = 0`**, **`zero_g = 0` (`D=0 A=0 B=0 N=0`)**, `MaxGN = 0.207`, стоимость 25с прогона + eval на Modal `~$0.0955`.
- В профиле 6-го шага (`BS=4`, `3p`): `Self CPU = 985.6 ms`, `Self CUDA = 987.9 ms`, число `Command Buffer Full` флашей снижено в **3.8×** (`1080 → 287`).

### 2. Микробенчи кастомных ядер на H100 (vs PyTorch/cuBLAS/WMMA)

| Ядро / Операция | Было (Baseline) | Стало (larpInk) | Ускорение | Точность (`cos`) |
|---|---:|---:|---:|---:|
| **`ShortConvRes` fwd** (`4×4096×2304` BF16) | `1.542 ms` (PyTorch) | **`0.091 ms`** (Triton) | **16.9×** | `y=1.000000`, `dx=1.000000` |
| **`RMSNorm` fwd** (`16384×2304` BF16) | `0.832 ms` (PyTorch) | **`0.056 ms`** (Triton) | **14.8×** | `y=1.000000`, `dx=0.999996`, `dw=1.000000` |
| **`MulSigmoid` fwd** (`16384×3072` BF16) | `0.820 ms` (PyTorch) | **`0.103 ms`** (Triton) | **8.0×** | `y=1.000000`, `dx=1.000000`, `dg=1.000000` |
| **`LogScaledL2Clip` fwd** (`4×4096×24×128`) | `1.271 ms` (PyTorch) | **`0.304 ms`** (Triton 1D; `<7ms` всего в 2D) | **4.2×+** | `y=1.000000`, `dx=1.000000` |
| **`moe_lora_b_add` (`dn`)** (`128` exp) | `~0.882 ms` (C++ WMMA) | **`0.357 ms`** (Triton `64×128`) | **2.47×** | `out=1.000000`, `y_exp=0.999968` |
| **`moe_z_dn_fp8_kernel`** (`prefix top_p=0.65`) | `0.061 ms` (`_scaled_mm`) | **`0.032 ms`** (Triton) | **1.91×** | `0.999642` |
| **`moe_dact_add_lora`** (`92k/131k`) | `0.247 ms` (cuBLAS `addmm_`) | **`0.118 ms`** (Triton) | **2.09×** | `1.000000` |
| **`Delayed FP8 DoRA Gate`** (`16k×2304→3072`) | `0.991 ms` (2-pass) | **`0.902 ms`** (1-pass fwd) | **-9%** | `fwd=0.999025`, `dx=0.999404`, `dA/dB=1.000000` |

---

## Структура репозитория

```text
larpInk/
├── README.md
├── .gitignore
├── h100/
│   ├── TUTORIAL_H100.md             # Подробный гайд: сборка .so, запуск и перенос ядер в свой код
│   ├── train_inkling_h100.py        # Лучшая SFT-версия под H100 (19.4k peak / 17.0k avg tok/s)
│   └── kernels/
│       ├── build_and_bench.py       # Сборка inkling_h100_ops.so на CPU (Modal) + микробенчи H100
│       ├── bindings.cpp             # PyBind11-биндинги C++/CUDA модулей
│       ├── fa_inkling.cu            # Hopper WGMMA FlashAttention-3 + rel_bias + sinks + doc_id
│       ├── moe_cutlass.cu           # FP8 Top-P MoE: routing, SwiGLU, quant, scatter/gather
│       └── dora_cublaslt.cu         # cuBLASLt/WMMA DoRA-ядра (опционально для экспериментов)
└── a100/
    ├── TUTORIAL_A100.md             # Подробный гайд для A100-80GB + чекпоинты и merge_lora
    └── train_inkling_a100.py        # Лучшая SFT-версия под A100 (INT8 MoE + Triton FA + Merge)
```

---

## Быстрый старт

### Вариант A: NVIDIA H100 80GB (`sm_90a`, максимальная скорость)

**1. Соберите нативное расширение `inkling_h100_ops.so`** (через Modal CPU-билдер — GPU при сборке не тратится):
```bash
modal run h100/kernels/build_and_bench.py --build-only
```
Это скомпилирует `fa_inkling.cu`, `moe_cutlass.cu`, `dora_cublaslt.cu`, `bindings.cpp` под `sm_90a` и сохранит `/build_cache/inkling_h100_ops.so` в Modal Volume `h100-cpp-build-cache`.

**2. Быстрая проверка на 6 шагов с профайлером (`--profile-1step`):**
```bash
modal run h100/train_inkling_h100.py --tokens-per-step 16384 --moe-lora-start 6 --full-gc-layers 10 --moe-gc-layers 8 --profile-1step
```

**3. Боевой SFT-прогон (например, 90с или полный цикл на своём `.jsonl` датасете):**
```bash
modal run h100/train_inkling_h100.py \
  --dataset yahma/alpaca-cleaned \
  --tokens-per-step 16384 \
  --time-limit-s 90 \
  --cooldown-start-s 75 \
  --full-gc-layers 10 \
  --moe-gc-layers 8 \
  --no-run-context-bench \
  --run-merge-check
```
Подробно про локальную сборку без Modal и пошаговое встраивание отдельных ядер в любой PyTorch/HF-пайплайн — в [**`h100/TUTORIAL_H100.md`**](h100/TUTORIAL_H100.md).

---

### Вариант B: NVIDIA A100 80GB (`sm_80`, без сборки C++)

Версия под A100 полностью написана на Triton + PyTorch, уже включает периодическое сохранение чекпоинтов (`best_lora.pt`, `latest_lora.pt`), `--resume` и автономный `merge_lora()` в чистые BF16 HuggingFace-веса:

```bash
modal run a100/train_inkling_a100.py \
  --dataset yahma/alpaca-cleaned \
  --grad-accum 4 \
  --time-limit-s 120 \
  --cooldown-start-s 100 \
  --merge-after
```
Подробно про параметры, датасеты, `--resume` и `--merge-only` — в [**`a100/TUTORIAL_A100.md`**](a100/TUTORIAL_A100.md).

---

## Что под капотом (стек оптимизаций)

- **Attention (`fa_inkling.cu` / `InklingFlashAttnLRVarlenFn`):** кастомный Hopper WGMMA FlashAttention (на H100) и block-wise Triton FA (на A100) с поддержкой per-head relative position bias (`rel_states @ rel_proj`), attention sinks, скользящего окна (`sliding_window`) и packed-баундари `doc_id` без cross-document утечек.
- **FP8 / INT8 MoE + Zero-Alloc MoE-LoRA:** сортировка токенов по экспертам с динамическим `top_p`, `torch._scaled_grouped_mm` на H100 (или grouped INT8 Triton GEMM на A100), кастомные ядра для LoRA `A`/`B` напрямую над FP8-буферами без раздувания памяти в BF16.
- **FastFP8DoRALinear:** weight decomposition DoRA с 1-pass Delayed FP8 в forward, динамическим точным `pmax` в backward и пересчётом норм без хранения `w0_bf16` в VRAM.
- **Zero-Recompile Elementwise Triton Stack (H100):** `TritonShortConvResFn`, `TritonRMSNormFn`, `TritonMulSigmoidFn`, `TritonScaledL2ClipFn`, `TritonMessageGateMixFn`, контигуальный выход `TritonDepthAttnPairFn` — 0 рекомпиляций `torch.compile` и минимальное число kernel launch.
- **Hybrid Selective GC + T-BPTT + CUDA Graphs:** пересчёт полного слоя только для первых 10 обучаемых слоёв (`Full:10`) и **только MoE-блока** (`layer.mlp`) для следующих 8 (`MoE:8`) + захват no-grad префикса цикла в общий `cg_pool`/`cg_stream`.
- **Оптимизаторы:** `Batched Cautious Muon` (`5` итераций Newton-Schulz) для Dense DoRA `A`/`B` + `Fused AdamW` для MoE-LoRA, DoRA `m` и RMSNorm + `Pruned 16k Vocab CE Loss`.
