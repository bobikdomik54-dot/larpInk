# Туториал: A100-80GB версия (`train_inkling_a100.py`)

Эта ветка — автономное production-ready решение для **NVIDIA A100 80GB (`sm_80`)**, полностью написанное на **Triton + PyTorch** без необходимости компилировать внешние C++/CUDA `.cu` файлы.

---

## 1. Ключевые особенности A100-стека

- **Окружение:** `pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel`, `triton>=3.0.0`, `transformers>=4.45.0`, `flash-linear-attention`.
- **Attention:** кастомный Triton `InklingFlashAttnLRVarlenFn` — блочный FlashAttention с per-head linear relative position bias, attention sinks, скользящим окном (`sliding_window`) и packed-баундари `doc_id` без cross-document утечек.
- **MoE:** `PatchedINT8Experts` — INT8 Weight-Only Grouped GEMM с блочным квантованием, counting-sort `top_p`-роутингом и fused INT8 MoE-LoRA `A`/`B` (`GATE 1A` проверяет точность `cos >= 0.999` на каждом запуске).
- **DoRA + CausalConv1d:** быстрые Triton-ядра для `causal_conv1d` и DoRA с пересчётом норм `refresh_all_dora_norms()` раз в шаг оптимизатора.
- **Оптимизатор и память:** Batched Cautious Muon + AdamW, T-BPTT (`grad_passes=1` в Fast-фазе → `2` в Cooldown), CUDA Graph для префикса и Selective GC в Cooldown.
- **Полный жизненный цикл обучения из коробки:**
  - Автосохранение чекпоинтов адаптера (`best_lora.pt`, `latest_lora.pt`, включает веса DoRA, MoE-LoRA, RMSNorm и состояние Muon).
  - Продолжение обучения с чекпоинта (`--resume`).
  - Отдельная функция `merge_lora()` для слияния Dense LoRA-весов (`nn.Linear`) с проверкой косинусного сходства логитов (`logits_cos >= 0.999`) и сохранением BF16 HuggingFace-весов (`save_pretrained`). *(Примечание: `best_lora.pt` сохраняет все обучаемые параметры, включая MoE-LoRA и RMSNorm, а для полного мерджа ещё и MoE-экспертов + RMSNorm можно перенести блок `n_merged_moe` / `n_merged_norm` из [`../h100/train_inkling_h100.py`](../h100/train_inkling_h100.py).)*

---

## 2. Быстрый запуск на Modal

### Боевой SFT + автоматический Merge в чистый BF16

```bash
cd larpInk
modal run a100/train_inkling_a100.py \
  --dataset yahma/alpaca-cleaned \
  --grad-accum 4 \
  --time-limit-s 120 \
  --cooldown-start-s 100 \
  --merge-after
```

### Обучение фиксированное число шагов (без лимита по времени)

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

### Продолжение обучения (`--resume`) или только Merge (`--merge-only`)

```bash
# Дообучить с latest_lora.pt
modal run a100/train_inkling_a100.py --resume --output-dir /cache_vol/my_run --max-steps 300 --time-limit-s 0

# Смерджить уже сохранённый best_lora.pt в HF BF16 модель и проверить генерацию
modal run a100/train_inkling_a100.py --merge-only --output-dir /cache_vol/my_run --merged-dir /cache_vol/my_merged_model
```

---

## 3. Основные CLI-флаги `a100/train_inkling_a100.py`

| Флаг | По умолчанию | Описание |
|---|---|---|
| `--dataset` | `yahma/alpaca-cleaned` | HF-датасет или путь к локальному `.jsonl` (Alpaca или ChatML `messages`) |
| `--grad-accum` | `4` | Число микро-батчей на 1 шаг оптимизатора |
| `--max-steps` | `100` | Лимит шагов (при `--time-limit-s 0`) |
| `--time-limit-s` | `120.0` | Тайм-лимит обучения в секундах (`0` = отключить таймер, идти по `max_steps`) |
| `--cooldown-start-s` | `100.0` | Момент перехода в Cooldown (`CUDAGraph=OFF`, `T-BPTT 2/3`, `top_p=1.0`, `Selective(1/3)` GC) |
| `--eval-every` | `50` | Частота полной оценки по 201k-словарю |
| `--resume` | `False` | Загрузить `latest_lora.pt` и продолжить с прерванного шага |
| `--merge-after` | `False` | После обучения запустить `merge_lora()` и сохранить чистый BF16-чекпоинт |
| `--merge-only` | `False` | Только смерджить существующий `best_lora.pt` |
| `--eval-prompts-after` | `True` | Прогнать генерацию 3 контрольных промптов после тренировки/мерджа |
