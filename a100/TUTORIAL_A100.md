# A100-80GB Tutorial (`train_inkling_a100.py`)

This branch targets **NVIDIA A100 80GB (`sm_80`)** using pure **Triton + PyTorch** — no external `.cu` compilation required.

---

## 1. Stack Overview

- **Container / Environment:** `pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel`, `triton>=3.0.0`, `transformers>=4.45.0`, `flash-linear-attention`.
- **Attention:** Custom Triton `InklingFlashAttnLRVarlenFn` — tiled FlashAttention with per-head linear relative position bias, attention sinks, sliding window (`sliding_window`), and packed `doc_id` boundaries.
- **MoE:** `PatchedINT8Experts` — INT8 weight-only grouped GEMM with blockwise quantization, counting-sort `top_p` routing, and fused INT8 MoE-LoRA `A`/`B` (verified on startup via `GATE 1A`, `cos >= 0.999`).
- **DoRA + CausalConv1d:** Triton `causal_conv1d` kernels and DoRA with `refresh_all_dora_norms()` once per optimizer step.
- **Optimizers & Memory:** Batched Cautious Muon + AdamW, T-BPTT (`grad_passes=1` Fast → `2` Cooldown), CUDA Graph prefix, and Selective GC during Cooldown.
- **Full Training Lifecycle Included:**
  - Automatic adapter checkpoints (`best_lora.pt`, `latest_lora.pt` containing DoRA, MoE-LoRA, RMSNorm weights, and Muon momentum buffers).
  - Seamless checkpoint resume (`--resume`).
  - Built-in `merge_lora()` for Dense `nn.Linear` LoRA adapters with logits sanity verification (`logits_cos >= 0.999`) and pure BF16 HuggingFace `save_pretrained`. *(Note: `best_lora.pt` saves all trainable params including MoE-LoRA and RMSNorm; if you want `merge_lora()` to also fold MoE-LoRA and RMSNorm weights into the base checkpoint, copy the 6-line `n_merged_moe`/`n_merged_norm` block from [`../h100/train_inkling_h100.py`](../h100/train_inkling_h100.py).)*

---

## 2. Quickstart on Modal

### Timed SFT + Auto-Merge to BF16
```bash
cd larpInk
modal run a100/train_inkling_a100.py \
  --dataset yahma/alpaca-cleaned \
  --grad-accum 4 \
  --time-limit-s 120 \
  --cooldown-start-s 100 \
  --merge-after
```

### Step-Based SFT (Custom `.jsonl` Dataset)
```bash
modal run a100/train_inkling_a100.py \
  --dataset my_dataset.jsonl \
  --grad-accum 4 \
  --max-steps 200 \
  --eval-every 50 \
  --time-limit-s 0 \
  --output-dir /cache_vol/my_run \
  --merged-dir /cache_vol/my_merged_model \
  --merge-after
```

### Resume or Merge-Only
```bash
# Resume from latest_lora.pt
modal run a100/train_inkling_a100.py --resume --output-dir /cache_vol/my_run --max-steps 300 --time-limit-s 0

# Merge existing best_lora.pt and run generation check
modal run a100/train_inkling_a100.py --merge-only --output-dir /cache_vol/my_run --merged-dir /cache_vol/my_merged_model
```

---

## 3. CLI Flags (`a100/train_inkling_a100.py`)

| Flag | Default | Description |
|---|---|---|
| `--dataset` | `yahma/alpaca-cleaned` | HF dataset name or local `.jsonl` path (Alpaca or `messages`) |
| `--grad-accum` | `4` | Micro-batches per optimizer step |
| `--max-steps` | `100` | Optimizer step limit (used when `--time-limit-s 0`) |
| `--time-limit-s` | `120.0` | Wall-clock training budget in seconds (`0` disables timer) |
| `--cooldown-start-s` | `100.0` | Switch point to Cooldown (`CUDAGraph=OFF`, `T-BPTT 2/3`, `top_p=1.0`, `Selective(1/3)` GC) |
| `--eval-every` | `50` | Full 201k-vocab evaluation cadence (in optimizer steps) |
| `--resume` | `False` | Resume from `latest_lora.pt` |
| `--merge-after` | `False` | Run `merge_lora()` right after training finishes |
| `--merge-only` | `False` | Skip training and merge `best_lora.pt` directly |
| `--eval-prompts-after` | `True` | Generate 3 control prompt completions after training/merge |
