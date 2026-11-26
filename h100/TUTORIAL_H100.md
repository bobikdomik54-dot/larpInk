# Подробный туториал: H100 (`sm_90a`) сборка, запуск и интеграция ядер в своё обучение

Этот гайд объясняет, как за 5 минут завести `h100/train_inkling_h100.py` на Modal или локальной H100 80GB, а также как без проблем вытащить и подключить кастомные C++/CUDA и Triton-ядра в свой собственный PyTorch-пайплайн.

---

## 1. Требования к окружению

- **GPU:** `NVIDIA H100 80GB HBM3` (`sm_90a`, compute capability `9.0`). На A100/RTX4090 код `fa_inkling.cu` (использует Hopper WGMMA) не скомпилируется — для A100 используйте ветку [`../a100/`](../a100/).
- **CUDA Toolkit:** `12.6+` (для `torch._scaled_grouped_mm` и Hopper FP8/`wgmma`).
- **PyTorch:** `2.7.0+cu126` (или новее с поддержкой `torch._scaled_mm`, `torch._scaled_grouped_mm` и `torch.float8_e4m3fn`).
- **Библиотеки:** `triton>=3.0`, `transformers>=4.45.0`, `datasets`, `accelerate`, `huggingface_hub`, `peft`.
- **Для сборки C++ модулей локально:** `ninja`, `g++`, CUDA `nvcc` 12.6+, **CUTLASS v3.6.0**.

---

## 2. Сборка нативного модуля `inkling_h100_ops.so`

### Вариант А — Через Modal (проще всего, сборка на CPU почти бесплатно)

```bash
cd larpInk
modal run h100/kernels/build_and_bench.py --build-only
```
Скрипт:
1. Поднимет CPU-контейнер с `CUDA 12.6`, `ninja-build`, `ccache` и CUTLASS `v3.6.0`.
2. Проверит SHA-256 всех `*.cu` и `*.cpp` в `h100/kernels/`.
3. Скомпилирует `fa_inkling.cu`, `moe_cutlass.cu`, `dora_cublaslt.cu` и `bindings.cpp` с `-gencode arch=compute_90a,code=sm_90a` и положит готовый `inkling_h100_ops.so` в Modal Volume `h100-cpp-build-cache` (`/build_cache/inkling_h100_ops.so`).

Проверить сами ядра отдельно (микробенч на H100):
```bash
modal run h100/kernels/build_and_bench.py --skip-build --target full
```

### Вариант Б — Локальная сборка на Linux H100 (без Modal)

1. Склонируйте CUTLASS `v3.6.0`:
   ```bash
   git clone --depth 1 --branch v3.6.0 https://github.com/NVIDIA/cutlass.git /opt/cutlass
   ```
2. Соберите `inkling_h100_ops.so` командой из `build_and_bench.py`:
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
3. В скрипте укажите путь `so_path = "./inkling_h100_ops.so"` (или скопируйте `.so` в `/build_cache/inkling_h100_ops.so`).

---

## 3. Подготовка датасета и запуск обучения

Скрипт `h100/train_inkling_h100.py` на лету пакует датасет в последовательности фиксированной длины `4096` с корректной картой документов `doc_id` (чтобы внимание не протекало между разными примерами) и маской промпта `-100`.

Поддерживаются оба популярных формата `.jsonl` (и HuggingFace Datasets):
- **Alpaca-формат:** `{"instruction": "...", "input": "...", "output": "..."}`
- **ChatML / ShareGPT-формат:** `{"messages": [{"role": "system"|"user"|"assistant", "content": "..."}]}`

### Рекомендуемые команды запуска (Modal)

1. **Профилирование и проверка скорости/VRAM за ~50–65 секунд (`--profile-1step`):**
   ```bash
   modal run h100/train_inkling_h100.py \
     --tokens-per-step 16384 \
     --moe-lora-start 6 \
     --full-gc-layers 10 \
     --moe-gc-layers 8 \
     --profile-1step
   ```
2. **Контрольный SFT-прогон на 25с + замер честного `EvalLoss` на 201k-словаре:**
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
3. **Полноценный SFT с финальным merge-тестом в BF16 (укажите свой `--time-limit-s`, например `600` сек, и `--cooldown-start-s 520`):**
   ```bash
   modal run h100/train_inkling_h100.py \
     --dataset yahma/alpaca-cleaned \
     --tokens-per-step 16384 \
     --time-limit-s 600 \
     --cooldown-start-s 520 \
     --full-gc-layers 10 \
     --moe-gc-layers 8 \
     --no-run-context-bench \
     --run-merge-check
   ```

> **Важно про сохранение весов на H100:** `train_inkling_h100.py` изначально написан как высокоскоростной stress/eval-раннер и проверяет мердж в памяти (`m_merged`), не записывая `.safetensors` на диск. Для сохранения итоговой модели добавьте после строки `eval_after = eval_full_vocab_loss_all(m_merged, ...)` в конце функции `train_h100`:
> ```python
> save_dir = "/cache_vol/merged_h100_model"
> m_merged.save_pretrained(save_dir)
> tokenizer.save_pretrained(save_dir)
> vol.commit()
> ```

---

## 4. Как встроить эти ядра в свой собственный PyTorch / HuggingFace код

Весь стек модульный — вам не обязательно использовать наш цикл целиком. Ниже точный чеклист, что и в каком порядке подключать, чтобы **всё пошло с первого раза без падений и расхождений градиентов**.

### Правило №1: Строгий порядок патчинга модели
Сначала патчите классы, **потом** загружайте или сразу после загрузки заменяйте слои, и только в самом конце настраивайте Gradient Checkpointing и Optimizer!

1. **Загрузка `.so`:**
   ```python
   import importlib.util
   spec = importlib.util.spec_from_file_location("inkling_h100_ops", "/path/to/inkling_h100_ops.so")
   ops = importlib.util.module_from_spec(spec)
   spec.loader.exec_module(ops)
   ```

2. **Подключение `FastFP8DoRALinear` к любым `nn.Linear` (работает для любой LLM, не только Inkling!):**
   - Скопируйте из `train_inkling_h100.py` ядра `fp8_amax_part_kernel`, `fp8_quant_from_part_kernel`, `dora_1x_lora_scale_inplace_kernel`, `dora_1x_bwd_scale_dm_kernel`, `fp8_quant_from_dynmax_kernel`, `fp8_quant_delayed_and_amax_kernel`, класс `_Fused1xFP8DoRAFn` и класс `FastFP8DoRALinear`.
   - **Обязательно** после каждого `optimizer.step()` вызывайте:
     - `roll_dora_fp8_scales()` — сдвигает историю `amax` для следующего шага.
     - `refresh_all_dora_norms()` (раз в 4 шага и перед любым `eval()`/мерджем) — обновляет `cached_inv_norm` без хранения `w0_bf16` в VRAM.
   - **Требования к тензорам:** `in_features`, `out_features` желательно кратны `16` (для некратного числа токенов `M_d & 15 != 0` встроен безопасный GPU-паддинг).
   - **Не ставьте `torch.compile` поверх `FastFP8DoRALinear`!** На классе уже стоит `@torch.compiler.disable`, чтобы Dynamo не плодил граф-брейки.

3. **Подключение `PatchedFP8Experts` (для MoE-слоёв с GeGLU/SwiGLU):**
   - Скопируйте `moe_z_dn_fp8_kernel`, `moe_dact_add_lora_kernel`, `moe_lora_dB_fp8_v2_kernel`, `moe_lora_dA_dn_fp8_kernel`, `moe_lora_b_add_triton_kernel` и класс `PatchedFP8Experts`.
   - **Ограничения C++ ядра `moe_cutlass.cu`:** размеры скрытого слоя в `moe_cutlass.cu` захардкожены константами `MOE_H = 2304`, `MOE_I = 1152`, `MOE_R = 16`! Если переносите на другую модель (с другим `hidden_size` или `intermediate_size`), обновите `MOE_H` / `MOE_I` / `MOE_R` в верхней части `h100/kernels/moe_cutlass.cu` и пересоберите `.so`!
   - **Раздельное отключение LoRA:** используйте `enable_lora=(idx >= moe_lora_start)` и `enable_grad=(idx >= freeze_until)` отдельно — иначе ранние слои с градиентами, но без MoE-LoRA потеряют транспонированные FP8 веса `w_dn_T_fp8_T`/`w_gu_T_fp8_T` для backward.
   - **Роутер всегда `requires_grad=False`** (но градиент `d_w` течёт обратно к выходу роутера `top_k_weights`).

4. **Подключение C++ FlashAttention (`fa_inkling.cu`) + packed `doc_id`:**
   - Потребуются буферы `doc_id` (`int16 [B, S]`) и предрасчитанные `q_n_start`, `k_m_start`, `k_m_end` (`int32 [B, S]`, см. `precompute_doc_bounds_cpu`).
   - Входные `q, k, v` подаются через `_fast_t12_contiguous_4d` (коалесцированный Triton-транспоз `[B, S, H, D] ↔ [B, H, S, D]`), `head_dim = 128`, `sliding_win = 1024`. Если меняете `head_dim`, проверьте шаблонные параметры в `fa_inkling.cu`.

5. **Подключение Triton Elementwise стека (`RMSNorm`, `ShortConvRes`, `Scaled/LogL2Clip`, `MulSigmoid`, `MessageGateMix`):**
   - Все эти классы универсальны по `batch`/`seq_len` и могут использоваться отдельно:
     - `TritonRMSNormFn`: работает для любого `N_dim <= 8192` в BF16/FP32.
     - `TritonMulSigmoidFn`: `x * sigmoid(gate)` в BF16 без FP32-тензоров.
     - `TritonShortConvResFn`: depthwise `conv1d(k=4, bias=None, weight frozen) + x * res_scale`.
     - `TritonScaledL2ClipFn`: head-wise L2-clip с опциональным `scale` и `log_scale`.
     - `TritonDepthAttnPairFn`: для `BlockAttentionResidual` (`K_prev <= 4`, `hdim <= 4096`) сразу возвращает **контигуальные** срезы `out_mn_first=True`, убирая сотни скрытых `aten::copy_`.
   - **Важно:** если заменяете этими ядрами модули модели, **не** оборачивайте их в `torch.compile` — иначе Dynamo начнёт ловить несовпадения `grad_mode`/`dtype`/`requires_grad`. В `compile_targets` оставляйте только чистый `Inkling2LoopTopKRouter` с `.contiguous()`-входом.

6. **Hybrid Selective GC + CUDA Graph Prefix (`run_nograd_prefix`):**
   - Для 80GB H100 при `BS=4, S=4096` оптимальный баланс: `--full-gc-layers 10 --moe-gc-layers 8` (`PeakVRAM ≈ 74.9 GB`).
   - Если используете CUDA Graphs для `no_grad` префикса с двумя глубинами цикла (`2p` и `3p`), **обязательно** передавайте и один общий `pool`, и **один общий `stream`** (`cg_cfg.setdefault("stream", torch.cuda.Stream())`) + `torch.cuda.empty_cache()` перед захватом, иначе CachingAllocator выделит отдельный приватный пул под каждый стрим (+3–5 GB VRAM и OOM).

---

## 5. Частые проблемы и решения (Troubleshooting)

| Симптом | Причина | Решение |
|---|---|---|
| `Missing prebuilt C++ extension /build_cache/inkling_h100_ops.so` | `.so` ещё не скомпилирован | Запустите `modal run h100/kernels/build_and_bench.py --build-only` |
| `CUDA out of memory` при уменьшении `--full-gc-layers` | Не-checkpointed слои внимания + DoRA держат ~2.5–2.9 GB/слой | Увеличьте `--full-gc-layers 10` (или `11`) и уменьшите `--moe-gc-layers 8` (или `7`), сумма = `18` |
| `EvalLoss` вырос (напр. `0.96+` вместо `<= 0.950`) | Слишком редкие 3-витковые проходы (`75/25`) или слишком низкий `prefix_top_p` | Держите в Fast-фазе `50/50 2p/3p` (`opt_step % 2 == 0`), `prefix_top_p >= 0.70–0.72` и Cooldown (`top_p=1.0`, `3p`) |
| `nvcc fatal : Unsupported gpu architecture 'compute_90a'` | Локальная CUDA < 12.0 или не H100 | Обновите CUDA Toolkit до `12.6+` (для A100 берите `../a100/`) |
| Медленный 1-й шаг (~20–35с) | Разовая JIT-компиляция Triton и `torch.compile` на `TopKRouter` + захват 2 CUDA-графов | Это одноразовый warmup перед стартом таймера `t_train_start`, со 2–3 шага выходит на полную скорость |
