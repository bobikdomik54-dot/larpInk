import modal

image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel",
        add_python="3.11",
    )
    .pip_install("triton>=3.0.0", "transformers>=4.45.0", "datasets", "accelerate", "huggingface_hub", "peft", "flash-linear-attention")
    .run_commands(
        "python -c \"from huggingface_hub import snapshot_download; snapshot_download('thinkingmachines/Inkling-2-Mini', local_dir='/model')\"",
        "python -c \"from datasets import load_dataset; ds = load_dataset('yahma/alpaca-cleaned', split='train[:500]'); ds.save_to_disk('/data')\"",
    )
)

vol = modal.Volume.from_name("inkling-cache-vol", create_if_missing=True)
app = modal.App("inkling-sft-prod", image=image)

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
    packed_ids, packed_lbl, packed_doc = [], [], []
    cur_ids, cur_lbl, cur_doc = [], [], []
    doc_counter = 0
    chunk = 4096
    for i in range(0, len(prompts), chunk):
        p_toks = tokenizer(prompts[i : i + chunk], add_special_tokens=False)["input_ids"]
        r_toks = tokenizer(resps[i : i + chunk], add_special_tokens=False)["input_ids"]
        for p_ids, r_ids in zip(p_toks, r_toks):
            ex_len = len(p_ids) + len(r_ids)
            if ex_len == 0:
                continue
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


@app.function(gpu="A100-80GB", volumes={"/cache_vol": vol}, timeout=1800)
def train_sft(
    dataset: str = "yahma/alpaca-cleaned",
    grad_accum: int = 4,
    max_steps: int = 100,
    epochs: int = 1,
    eval_every: int = 50,
    lr_a: float = 1.0e-4,
    lr_b: float = 4.0e-4,
    lr_moe: float = 1.5e-4,
    lr_norm: float = 1.0e-4,
    seed: int = 42,
    resume: bool = False,
    output_dir: str = "/cache_vol/inkling_sft_run",
    time_limit_s: float = 0.0,
    cooldown_start_s: float = 0.0,
    eval_prompts_after: bool = False,
):
    global triton, tl
    import sys
    sys.modules["torchvision"] = None
    sys.modules["torchaudio"] = None
    import gc
    import math
    import os
    import queue
    import re
    import threading
    import time
    import warnings
    from functools import partial
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

    t_gpu_start = time.time()
    device = "cuda"
    dtype = torch.bfloat16
    SEQ_LEN = 4096
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained("/model")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    packed_data, cache_path, pack_s, cache_mb = _load_or_build_packed_dataset(dataset, tokenizer, seq_len=SEQ_LEN)
    train_ids_all = packed_data["train_ids"]
    train_lbl_all = packed_data["train_lbl"]
    train_doc_all = packed_data["train_doc"]
    val_ids_all = packed_data["val_ids"]
    val_lbl_all = packed_data["val_lbl"]
    val_doc_all = packed_data["val_doc"]
    n_train_batches = int(train_ids_all.shape[0])
    n_val_batches = int(val_ids_all.shape[0])

    # =========================================================================
    # 1. FAST LAYOUT-AWARE TRITON CAUSAL CONV1D (0.16 ms CL/CF)
    # =========================================================================
    @triton.jit
    def old_causal_conv1d_fwd_kernel(
        X_ptr, W_ptr, B_ptr, Out_ptr,
        stride_xb, stride_xc, stride_xs,
        stride_wc, stride_wk,
        S_dim, K_size: tl.constexpr, HAS_BIAS: tl.constexpr, SILU: tl.constexpr,
        BLOCK_S: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_c = tl.program_id(1)
        pid_s = tl.program_id(2)
        offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
        mask_s = offs_s < S_dim
        base_x = X_ptr + pid_b * stride_xb + pid_c * stride_xc
        acc = tl.zeros([BLOCK_S], dtype=tl.float32)
        if HAS_BIAS:
            acc += tl.load(B_ptr + pid_c).to(tl.float32)
        for k in range(K_size):
            idx_s = offs_s - (K_size - 1 - k)
            m = mask_s & (idx_s >= 0)
            x_val = tl.load(base_x + idx_s * stride_xs, mask=m, other=0.0).to(tl.float32)
            w_val = tl.load(W_ptr + pid_c * stride_wc + k * stride_wk).to(tl.float32)
            acc += x_val * w_val
        if SILU:
            acc = acc * tl.sigmoid(acc)
        tl.store(Out_ptr + pid_b * stride_xb + pid_c * stride_xc + offs_s * stride_xs, acc.to(Out_ptr.dtype.element_ty), mask=mask_s)

    @triton.jit
    def old_causal_conv1d_bwd_dx_kernel(
        dOut_ptr, X_ptr, W_ptr, B_ptr, dX_ptr,
        stride_xb, stride_xc, stride_xs,
        stride_wc, stride_wk,
        S_dim, K_size: tl.constexpr, HAS_BIAS: tl.constexpr, SILU: tl.constexpr,
        BLOCK_S: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_c = tl.program_id(1)
        pid_s = tl.program_id(2)
        offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
        mask_s = offs_s < S_dim
        base_x = X_ptr + pid_b * stride_xb + pid_c * stride_xc
        base_do = dOut_ptr + pid_b * stride_xb + pid_c * stride_xc
        dx_acc = tl.zeros([BLOCK_S], dtype=tl.float32)
        for k in range(K_size):
            out_s = offs_s + (K_size - 1 - k)
            m = mask_s & (out_s < S_dim)
            dout_val = tl.load(base_do + out_s * stride_xs, mask=m, other=0.0).to(tl.float32)
            w_val = tl.load(W_ptr + pid_c * stride_wc + k * stride_wk).to(tl.float32)
            if SILU:
                pre = tl.zeros([BLOCK_S], dtype=tl.float32)
                if HAS_BIAS:
                    pre += tl.load(B_ptr + pid_c).to(tl.float32)
                for kk in range(K_size):
                    xs_idx = out_s - (K_size - 1 - kk)
                    mm = m & (xs_idx >= 0)
                    xv = tl.load(base_x + xs_idx * stride_xs, mask=mm, other=0.0).to(tl.float32)
                    wv = tl.load(W_ptr + pid_c * stride_wc + kk * stride_wk).to(tl.float32)
                    pre += xv * wv
                sig = tl.sigmoid(pre)
                dout_val = dout_val * (sig * (1.0 + pre * (1.0 - sig)))
            dx_acc += dout_val * w_val
        tl.store(dX_ptr + pid_b * stride_xb + pid_c * stride_xc + offs_s * stride_xs, dx_acc.to(dX_ptr.dtype.element_ty), mask=mask_s)


    # --- Fast Channels-Last (stride_xc == 1) K=4 kernels with 3 shift registers ---
    @triton.jit
    def causal_conv1d_fwd_cl_k4_kernel(
        X_ptr, W_ptr, B_ptr, Out_ptr,
        stride_xb, stride_xs, stride_ob, stride_os,
        stride_wc, stride_wk,
        C_dim, S_dim, HAS_BIAS: tl.constexpr, SILU: tl.constexpr,
        BLOCK_C: tl.constexpr, CHUNK_S: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_c = tl.program_id(1)
        pid_s = tl.program_id(2)
        offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C_dim
        s0 = pid_s * CHUNK_S

        w_base = W_ptr + offs_c * stride_wc
        w0 = tl.load(w_base + 0 * stride_wk, mask=mask_c, other=0.0).to(tl.float32)
        w1 = tl.load(w_base + 1 * stride_wk, mask=mask_c, other=0.0).to(tl.float32)
        w2 = tl.load(w_base + 2 * stride_wk, mask=mask_c, other=0.0).to(tl.float32)
        w3 = tl.load(w_base + 3 * stride_wk, mask=mask_c, other=0.0).to(tl.float32)
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
                x_m3 = x_m2
                x_m2 = x_m1
                x_m1 = x_cur

    @triton.jit
    def causal_conv1d_bwd_cl_k4_kernel(
        dOut_ptr, X_ptr, W_ptr, B_ptr, dX_ptr,
        stride_dob, stride_doc, stride_dos,
        stride_xb, stride_xs,
        stride_dxb, stride_dxs,
        stride_wc, stride_wk,
        C_dim, S_dim, HAS_BIAS: tl.constexpr, SILU: tl.constexpr,
        BLOCK_C: tl.constexpr, CHUNK_S: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_c = tl.program_id(1)
        pid_s = tl.program_id(2)
        offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C_dim
        s0 = pid_s * CHUNK_S

        w_base = W_ptr + offs_c * stride_wc
        w0 = tl.load(w_base + 0 * stride_wk, mask=mask_c, other=0.0).to(tl.float32)
        w1 = tl.load(w_base + 1 * stride_wk, mask=mask_c, other=0.0).to(tl.float32)
        w2 = tl.load(w_base + 2 * stride_wk, mask=mask_c, other=0.0).to(tl.float32)
        w3 = tl.load(w_base + 3 * stride_wk, mask=mask_c, other=0.0).to(tl.float32)
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
                    x_m3 = x_m2
                    x_m2 = x_m1
                    x_m1 = x_cur
            else:
                g_cur = tl.zeros([BLOCK_C], dtype=tl.float32)

            if step >= 3:
                s_w = s0 + step - 3
                if s_w < S_dim:
                    dx_val = g_m3 * w3 + g_m2 * w2 + g_m1 * w1 + g_cur * w0
                    tl.store(dx_b + s_w * stride_dxs, dx_val.to(dX_ptr.dtype.element_ty), mask=mask_c)

            g_m3 = g_m2
            g_m2 = g_m1
            g_m1 = g_cur

    # --- Fast Channels-First (stride_xs == 1) K=4 kernels (2D tile BLOCK_C x BLOCK_S) ---
    @triton.jit
    def causal_conv1d_fwd_cf_k4_kernel(
        X_ptr, W_ptr, B_ptr, Out_ptr,
        stride_xb, stride_xc, stride_ob, stride_oc,
        stride_wc, stride_wk,
        C_dim, S_dim, HAS_BIAS: tl.constexpr, SILU: tl.constexpr,
        BLOCK_C: tl.constexpr, BLOCK_S: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_c = tl.program_id(1)
        pid_s = tl.program_id(2)
        offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
        mask_c = offs_c < C_dim
        mask_s = offs_s < S_dim

        w_base = W_ptr + offs_c * stride_wc
        w0 = tl.load(w_base + 0 * stride_wk, mask=mask_c, other=0.0).to(tl.float32)[:, None]
        w1 = tl.load(w_base + 1 * stride_wk, mask=mask_c, other=0.0).to(tl.float32)[:, None]
        w2 = tl.load(w_base + 2 * stride_wk, mask=mask_c, other=0.0).to(tl.float32)[:, None]
        w3 = tl.load(w_base + 3 * stride_wk, mask=mask_c, other=0.0).to(tl.float32)[:, None]
        acc = tl.load(B_ptr + offs_c, mask=mask_c, other=0.0).to(tl.float32)[:, None] if HAS_BIAS else tl.zeros([BLOCK_C, BLOCK_S], dtype=tl.float32)

        x_base = X_ptr + pid_b * stride_xb + offs_c[:, None] * stride_xc
        x0 = tl.load(x_base + (offs_s[None, :] - 3), mask=mask_c[:, None] & (offs_s[None, :] >= 3) & mask_s[None, :], other=0.0).to(tl.float32)
        x1 = tl.load(x_base + (offs_s[None, :] - 2), mask=mask_c[:, None] & (offs_s[None, :] >= 2) & mask_s[None, :], other=0.0).to(tl.float32)
        x2 = tl.load(x_base + (offs_s[None, :] - 1), mask=mask_c[:, None] & (offs_s[None, :] >= 1) & mask_s[None, :], other=0.0).to(tl.float32)
        x3 = tl.load(x_base + offs_s[None, :], mask=mask_c[:, None] & mask_s[None, :], other=0.0).to(tl.float32)
        acc = acc + x0 * w0 + x1 * w1 + x2 * w2 + x3 * w3
        if SILU:
            acc = acc * tl.sigmoid(acc)
        tl.store(Out_ptr + pid_b * stride_ob + offs_c[:, None] * stride_oc + offs_s[None, :], acc.to(Out_ptr.dtype.element_ty), mask=mask_c[:, None] & mask_s[None, :])

    @triton.jit
    def causal_conv1d_bwd_cf_k4_kernel(
        dOut_ptr, X_ptr, W_ptr, B_ptr, dX_ptr,
        stride_dob, stride_doc, stride_dos,
        stride_xb, stride_xc,
        stride_dxb, stride_dxc,
        stride_wc, stride_wk,
        C_dim, S_dim, HAS_BIAS: tl.constexpr, SILU: tl.constexpr,
        BLOCK_C: tl.constexpr, BLOCK_S: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_c = tl.program_id(1)
        pid_s = tl.program_id(2)
        offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
        mask_c = offs_c < C_dim
        mask_s = offs_s < S_dim

        w_base = W_ptr + offs_c * stride_wc
        w0 = tl.load(w_base + 0 * stride_wk, mask=mask_c, other=0.0).to(tl.float32)[:, None]
        w1 = tl.load(w_base + 1 * stride_wk, mask=mask_c, other=0.0).to(tl.float32)[:, None]
        w2 = tl.load(w_base + 2 * stride_wk, mask=mask_c, other=0.0).to(tl.float32)[:, None]
        w3 = tl.load(w_base + 3 * stride_wk, mask=mask_c, other=0.0).to(tl.float32)[:, None]
        b_val = tl.load(B_ptr + offs_c, mask=mask_c, other=0.0).to(tl.float32)[:, None] if HAS_BIAS else tl.zeros([BLOCK_C, 1], dtype=tl.float32)

        x_base = X_ptr + pid_b * stride_xb + offs_c[:, None] * stride_xc
        do_base = dOut_ptr + pid_b * stride_dob + offs_c[:, None] * stride_doc

        # Load X window [s-3 .. s+3] once into registers (7 vectors)
        if SILU:
            xm3 = tl.load(x_base + (offs_s[None, :] - 3), mask=mask_c[:, None] & (offs_s[None, :] >= 3) & mask_s[None, :], other=0.0).to(tl.float32)
            xm2 = tl.load(x_base + (offs_s[None, :] - 2), mask=mask_c[:, None] & (offs_s[None, :] >= 2) & mask_s[None, :], other=0.0).to(tl.float32)
            xm1 = tl.load(x_base + (offs_s[None, :] - 1), mask=mask_c[:, None] & (offs_s[None, :] >= 1) & mask_s[None, :], other=0.0).to(tl.float32)
            x0v = tl.load(x_base + (offs_s[None, :] + 0), mask=mask_c[:, None] & mask_s[None, :], other=0.0).to(tl.float32)
            xp1 = tl.load(x_base + (offs_s[None, :] + 1), mask=mask_c[:, None] & ((offs_s[None, :] + 1) < S_dim), other=0.0).to(tl.float32)
            xp2 = tl.load(x_base + (offs_s[None, :] + 2), mask=mask_c[:, None] & ((offs_s[None, :] + 2) < S_dim), other=0.0).to(tl.float32)
            xp3 = tl.load(x_base + (offs_s[None, :] + 3), mask=mask_c[:, None] & ((offs_s[None, :] + 3) < S_dim), other=0.0).to(tl.float32)

        g0 = tl.load(do_base + (offs_s[None, :] + 0) * stride_dos, mask=mask_c[:, None] & mask_s[None, :], other=0.0).to(tl.float32)
        g1 = tl.load(do_base + (offs_s[None, :] + 1) * stride_dos, mask=mask_c[:, None] & ((offs_s[None, :] + 1) < S_dim), other=0.0).to(tl.float32)
        g2 = tl.load(do_base + (offs_s[None, :] + 2) * stride_dos, mask=mask_c[:, None] & ((offs_s[None, :] + 2) < S_dim), other=0.0).to(tl.float32)
        g3 = tl.load(do_base + (offs_s[None, :] + 3) * stride_dos, mask=mask_c[:, None] & ((offs_s[None, :] + 3) < S_dim), other=0.0).to(tl.float32)

        if SILU:
            p0 = b_val + xm3 * w0 + xm2 * w1 + xm1 * w2 + x0v * w3
            s0 = tl.sigmoid(p0)
            g0 = g0 * (s0 * (1.0 + p0 * (1.0 - s0)))
            p1 = b_val + xm2 * w0 + xm1 * w1 + x0v * w2 + xp1 * w3
            s1 = tl.sigmoid(p1)
            g1 = g1 * (s1 * (1.0 + p1 * (1.0 - s1)))
            p2 = b_val + xm1 * w0 + x0v * w1 + xp1 * w2 + xp2 * w3
            s2 = tl.sigmoid(p2)
            g2 = g2 * (s2 * (1.0 + p2 * (1.0 - s2)))
            p3 = b_val + x0v * w0 + xp1 * w1 + xp2 * w2 + xp3 * w3
            s3 = tl.sigmoid(p3)
            g3 = g3 * (s3 * (1.0 + p3 * (1.0 - s3)))

        dx_acc = g0 * w3 + g1 * w2 + g2 * w1 + g3 * w0
        tl.store(dX_ptr + pid_b * stride_dxb + offs_c[:, None] * stride_dxc + offs_s[None, :], dx_acc.to(dX_ptr.dtype.element_ty), mask=mask_c[:, None] & mask_s[None, :])

    conv_layout_seen = set()

    class TritonCausalConv1dFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weight, bias, activation):
            B_c, C_c, S_c = x.shape
            K_c = weight.shape[-1]
            silu = activation in ("silu", "swish")
            sig_key = (tuple(x.shape), tuple(x.stride()), K_c, activation)
            if sig_key not in conv_layout_seen and S_c == SEQ_LEN:
                conv_layout_seen.add(sig_key)
                print(f"=== CONV1D REAL MODEL INPUT: shape={tuple(x.shape)} stride={tuple(x.stride())} K={K_c} act={activation} ===", flush=True)
            out = torch.empty_like(x)
            if K_c == 4 and x.stride(1) == 1:
                BLOCK_C, CHUNK_S = 128, 64
                causal_conv1d_fwd_cl_k4_kernel[(B_c, triton.cdiv(C_c, BLOCK_C), triton.cdiv(S_c, CHUNK_S))](
                    x, weight, bias if bias is not None else x, out,
                    x.stride(0), x.stride(2), out.stride(0), out.stride(2),
                    weight.stride(0), weight.stride(1),
                    C_c, S_c, HAS_BIAS=(bias is not None), SILU=silu,
                    BLOCK_C=BLOCK_C, CHUNK_S=CHUNK_S, num_warps=4,
                )
            elif K_c == 4 and x.stride(2) == 1:
                BLOCK_C, BLOCK_S = 8, 128
                causal_conv1d_fwd_cf_k4_kernel[(B_c, triton.cdiv(C_c, BLOCK_C), triton.cdiv(S_c, BLOCK_S))](
                    x, weight, bias if bias is not None else x, out,
                    x.stride(0), x.stride(1), out.stride(0), out.stride(1),
                    weight.stride(0), weight.stride(1),
                    C_c, S_c, HAS_BIAS=(bias is not None), SILU=silu,
                    BLOCK_C=BLOCK_C, BLOCK_S=BLOCK_S, num_warps=4,
                )
            else:
                old_causal_conv1d_fwd_kernel[(B_c, C_c, triton.cdiv(S_c, 256))](
                    x, weight, bias if bias is not None else x, out,
                    x.stride(0), x.stride(1), x.stride(2), weight.stride(0), weight.stride(1),
                    S_c, K_size=K_c, HAS_BIAS=(bias is not None), SILU=silu, BLOCK_S=256, num_warps=4,
                )
            ctx.save_for_backward(x, weight, bias)
            ctx.silu = silu
            return out

        @staticmethod
        def backward(ctx, dout):
            x, weight, bias = ctx.saved_tensors
            B_c, C_c, S_c = x.shape
            K_c = weight.shape[-1]
            dx = torch.empty_like(x)
            if K_c == 4 and x.stride(1) == 1:
                BLOCK_C, CHUNK_S = 128, 64
                causal_conv1d_bwd_cl_k4_kernel[(B_c, triton.cdiv(C_c, BLOCK_C), triton.cdiv(S_c, CHUNK_S))](
                    dout, x, weight, bias if bias is not None else x, dx,
                    dout.stride(0), dout.stride(1), dout.stride(2),
                    x.stride(0), x.stride(2), dx.stride(0), dx.stride(2),
                    weight.stride(0), weight.stride(1),
                    C_c, S_c, HAS_BIAS=(bias is not None), SILU=ctx.silu,
                    BLOCK_C=BLOCK_C, CHUNK_S=CHUNK_S, num_warps=4,
                )
            elif K_c == 4 and x.stride(2) == 1:
                BLOCK_C, BLOCK_S = 8, 128
                causal_conv1d_bwd_cf_k4_kernel[(B_c, triton.cdiv(C_c, BLOCK_C), triton.cdiv(S_c, BLOCK_S))](
                    dout, x, weight, bias if bias is not None else x, dx,
                    dout.stride(0), dout.stride(1), dout.stride(2),
                    x.stride(0), x.stride(1), dx.stride(0), dx.stride(1),
                    weight.stride(0), weight.stride(1),
                    C_c, S_c, HAS_BIAS=(bias is not None), SILU=ctx.silu,
                    BLOCK_C=BLOCK_C, BLOCK_S=BLOCK_S, num_warps=4,
                )
            else:
                old_causal_conv1d_bwd_dx_kernel[(B_c, C_c, triton.cdiv(S_c, 256))](
                    dout, x, weight, bias if bias is not None else x, dx,
                    x.stride(0), x.stride(1), x.stride(2), weight.stride(0), weight.stride(1),
                    S_c, K_size=K_c, HAS_BIAS=(bias is not None), SILU=ctx.silu, BLOCK_S=256, num_warps=4,
                )
            return dx, None, None, None

    ink_mod.causal_conv1d_fn = lambda x, weight, bias=None, activation=None, **kw: TritonCausalConv1dFn.apply(x, weight, bias, activation)

    # =========================================================================
    # 2. VARLEN / DOCUMENT-PACKED LOW-RANK TOEPLITZ FLASHATTENTION
    # =========================================================================
    NUM_LR_CTAS = 216
    lr_scratch_f32 = torch.empty((NUM_LR_CTAS, 64, 128), device=device, dtype=torch.float32)

    @triton.jit
    def inkling_flash_attn_bwd_prep_kernel(
        Out, dOut, D_vec,
        stride_ob, stride_oh, stride_om, stride_od,
        stride_dob, stride_doh, stride_dom, stride_dod,
        S, H_q, BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        b = pid_bh // H_q
        h = pid_bh % H_q
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, HEAD_DIM)
        mask_m = offs_m < S
        o = tl.load(Out + b * stride_ob + h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od, mask=mask_m[:, None], other=0.0).to(tl.float32)
        do = tl.load(dOut + b * stride_dob + h * stride_doh + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod, mask=mask_m[:, None], other=0.0).to(tl.float32)
        tl.store(D_vec + pid_bh * S + offs_m, tl.sum(o * do, axis=1), mask=mask_m)

    @triton.jit
    def inkling_flash_attn_lr_varlen_fwd_kernel(
        Q, K, V, RelStates, RelProj, Sinks, DocId, Out, L_ptr, Scratch,
        stride_qb, stride_qh, stride_qm, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_rsb, stride_rsm, stride_rsh, stride_rsr,
        stride_pr, stride_pd,
        stride_sb, stride_sh, stride_sm,
        stride_docb,
        stride_ob, stride_oh, stride_om, stride_od,
        S, sm_scale, B_dim, H_q, H_kv,
        HAS_DOC: tl.constexpr, R_DIM: tl.constexpr, REL_EXTENT: tl.constexpr, IS_SLIDING: tl.constexpr, SLIDING_WIN: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, W_CHUNK: tl.constexpr, HEAD_DIM: tl.constexpr,
    ):
        cta_id = tl.program_id(0)
        num_ctas = tl.num_programs(0)
        num_m_blks = tl.cdiv(S, BLOCK_M)
        total_tiles = num_m_blks * (B_dim * H_q)

        local_m = tl.arange(0, BLOCK_M)
        local_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, HEAD_DIM)
        offs_r = tl.arange(0, R_DIM)
        offs_w = tl.arange(0, W_CHUNK)
        w_idx = BLOCK_N + local_m[:, None] - local_n[None, :]
        scratch_cta = Scratch + cta_id * (BLOCK_M * W_CHUNK)

        for tile_id in range(cta_id, total_tiles, num_ctas):
            pid_bh = tile_id // num_m_blks
            pid_m = tile_id % num_m_blks
            b = pid_bh // H_q
            h = pid_bh % H_q
            kv_h = h // (H_q // H_kv)

            offs_m = pid_m * BLOCK_M + local_m
            mask_m = offs_m < S
            m_start = pid_m * BLOCK_M
            m_end = tl.minimum((pid_m + 1) * BLOCK_M, S)

            q = tl.load(Q + b * stride_qb + h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd, mask=mask_m[:, None], other=0.0)
            rs = tl.load(RelStates + b * stride_rsb + offs_m[:, None] * stride_rsm + h * stride_rsh + offs_r[None, :] * stride_rsr, mask=mask_m[:, None], other=0.0)
            sink_val = tl.load(Sinks + b * stride_sb + h * stride_sh + offs_m * stride_sm, mask=mask_m, other=0.0).to(tl.float32)

            if HAS_DOC:
                doc_m = tl.load(DocId + b * stride_docb + offs_m, mask=mask_m, other=-1).to(tl.int32)
                doc_m_min = tl.load(DocId + b * stride_docb + m_start).to(tl.int32)
                doc_m_max = tl.load(DocId + b * stride_docb + m_end - 1).to(tl.int32)

            m_i = sink_val
            l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
            acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

            n_start = tl.maximum(0, m_start - SLIDING_WIN + 1) if IS_SLIDING else 0
            n_start_blk = n_start // BLOCK_N
            n_end_blk = tl.cdiv(m_end, BLOCK_N)

            for n_blk in range(n_start_blk, n_end_blk):
                n_s = n_blk * BLOCK_N
                n_e = tl.minimum((n_blk + 1) * BLOCK_N, S)
                active_tile = True
                if HAS_DOC:
                    doc_n_min = tl.load(DocId + b * stride_docb + n_s).to(tl.int32)
                    doc_n_max = tl.load(DocId + b * stride_docb + n_e - 1).to(tl.int32)
                    active_tile = (doc_m_max >= doc_n_min) and (doc_m_min <= doc_n_max)

                if active_tile:
                    offs_n = n_s + local_n
                    mask_n = offs_n < S
                    k = tl.load(K + b * stride_kb + kv_h * stride_kh + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd, mask=mask_n[None, :], other=0.0)
                    qk = tl.dot(q, k).to(tl.float32) * sm_scale
                    dist = offs_m[:, None] - offs_n[None, :]

                    d_0 = (pid_m - n_blk) * BLOCK_M - BLOCK_N
                    if (d_0 < REL_EXTENT) and (d_0 + W_CHUNK > 0):
                        d_col = d_0 + offs_w
                        mask_w = (d_col >= 0) & (d_col < REL_EXTENT)
                        p_chunk = tl.load(RelProj + offs_r[:, None] * stride_pr + d_col[None, :] * stride_pd, mask=mask_w[None, :], other=0.0)
                        b_dist = tl.dot(rs, p_chunk).to(tl.float32)
                        tl.debug_barrier()
                        tl.store(scratch_cta + local_m[:, None] * W_CHUNK + offs_w[None, :], b_dist)
                        tl.debug_barrier()
                        rel_mask = (dist >= 0) & (dist < REL_EXTENT) & mask_m[:, None] & mask_n[None, :]
                        rel_b = tl.load(scratch_cta + local_m[:, None] * W_CHUNK + w_idx, mask=rel_mask, other=0.0)
                        qk += rel_b

                    causal_mask = (dist >= 0) & mask_m[:, None] & mask_n[None, :]
                    if IS_SLIDING:
                        causal_mask = causal_mask & (dist < SLIDING_WIN)
                    if HAS_DOC:
                        doc_n = tl.load(DocId + b * stride_docb + offs_n, mask=mask_n, other=-2).to(tl.int32)
                        causal_mask = causal_mask & (doc_m[:, None] == doc_n[None, :])
                    qk = tl.where(causal_mask, qk, float("-inf"))
                    m_ij = tl.maximum(m_i, tl.max(qk, 1))
                    p = tl.exp(qk - m_ij[:, None])
                    alpha = tl.exp(m_i - m_ij)
                    l_i = l_i * alpha + tl.sum(p, 1)
                    acc = acc * alpha[:, None]
                    v = tl.load(V + b * stride_vb + kv_h * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd, mask=mask_n[:, None], other=0.0)
                    acc += tl.dot(p.to(v.dtype), v)
                    m_i = m_ij

            acc = acc / l_i[:, None]
            tl.store(Out + b * stride_ob + h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od, acc.to(Out.dtype.element_ty), mask=mask_m[:, None])
            tl.store(L_ptr + pid_bh * S + offs_m, m_i + tl.log(l_i), mask=mask_m)

    @triton.jit
    def inkling_flash_attn_lr_varlen_bwd_dq_kernel(
        Q, K, V, RelStates, RelProj, dOut, L_ptr, D_vec, dQ, dRelStates, Sinks, dSinks, DocId, Scratch,
        stride_qb, stride_qh, stride_qm, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_rsb, stride_rsm, stride_rsh, stride_rsr,
        stride_pr, stride_pd,
        stride_sb, stride_sh, stride_sm,
        stride_docb,
        stride_dob, stride_doh, stride_dom, stride_dod,
        S, sm_scale, B_dim, H_q, H_kv,
        HAS_DOC: tl.constexpr, R_DIM: tl.constexpr, REL_EXTENT: tl.constexpr, IS_SLIDING: tl.constexpr, SLIDING_WIN: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, W_CHUNK: tl.constexpr, HEAD_DIM: tl.constexpr,
    ):
        cta_id = tl.program_id(0)
        num_ctas = tl.num_programs(0)
        num_m_blks = tl.cdiv(S, BLOCK_M)
        total_tiles = num_m_blks * (B_dim * H_q)

        local_m = tl.arange(0, BLOCK_M)
        local_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, HEAD_DIM)
        offs_r = tl.arange(0, R_DIM)
        offs_w = tl.arange(0, W_CHUNK)
        w_idx = BLOCK_N + local_m[:, None] - local_n[None, :]
        scratch_cta = Scratch + cta_id * (BLOCK_M * W_CHUNK)

        for tile_id in range(cta_id, total_tiles, num_ctas):
            pid_bh = tile_id // num_m_blks
            pid_m = tile_id % num_m_blks
            b = pid_bh // H_q
            h = pid_bh % H_q
            kv_h = h // (H_q // H_kv)

            offs_m = pid_m * BLOCK_M + local_m
            mask_m = offs_m < S
            m_start = pid_m * BLOCK_M
            m_end = tl.minimum((pid_m + 1) * BLOCK_M, S)

            q = tl.load(Q + b * stride_qb + h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd, mask=mask_m[:, None], other=0.0)
            rs = tl.load(RelStates + b * stride_rsb + offs_m[:, None] * stride_rsm + h * stride_rsh + offs_r[None, :] * stride_rsr, mask=mask_m[:, None], other=0.0)
            do = tl.load(dOut + b * stride_dob + h * stride_doh + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod, mask=mask_m[:, None], other=0.0)
            l_i = tl.load(L_ptr + pid_bh * S + offs_m, mask=mask_m, other=0.0)
            d_i = tl.load(D_vec + pid_bh * S + offs_m, mask=mask_m, other=0.0)
            sink_val = tl.load(Sinks + b * stride_sb + h * stride_sh + offs_m * stride_sm, mask=mask_m, other=0.0).to(tl.float32)
            p_sink = tl.exp(sink_val - l_i)
            tl.store(dSinks + b * stride_sb + h * stride_sh + offs_m * stride_sm, (-p_sink * d_i).to(dSinks.dtype.element_ty), mask=mask_m)

            if HAS_DOC:
                doc_m = tl.load(DocId + b * stride_docb + offs_m, mask=mask_m, other=-1).to(tl.int32)
                doc_m_min = tl.load(DocId + b * stride_docb + m_start).to(tl.int32)
                doc_m_max = tl.load(DocId + b * stride_docb + m_end - 1).to(tl.int32)

            dq = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
            d_rs = tl.zeros([BLOCK_M, R_DIM], dtype=tl.float32)

            n_start = tl.maximum(0, m_start - SLIDING_WIN + 1) if IS_SLIDING else 0
            n_start_blk = n_start // BLOCK_N
            n_end_blk = tl.cdiv(m_end, BLOCK_N)

            for n_blk in range(n_start_blk, n_end_blk):
                n_s = n_blk * BLOCK_N
                n_e = tl.minimum((n_blk + 1) * BLOCK_N, S)
                active_tile = True
                if HAS_DOC:
                    doc_n_min = tl.load(DocId + b * stride_docb + n_s).to(tl.int32)
                    doc_n_max = tl.load(DocId + b * stride_docb + n_e - 1).to(tl.int32)
                    active_tile = (doc_m_max >= doc_n_min) and (doc_m_min <= doc_n_max)

                if active_tile:
                    offs_n = n_s + local_n
                    mask_n = offs_n < S
                    k = tl.load(K + b * stride_kb + kv_h * stride_kh + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd, mask=mask_n[None, :], other=0.0)
                    v = tl.load(V + b * stride_vb + kv_h * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd, mask=mask_n[:, None], other=0.0)
                    qk = tl.dot(q, k).to(tl.float32) * sm_scale
                    dist = offs_m[:, None] - offs_n[None, :]
                    rel_mask = (dist >= 0) & (dist < REL_EXTENT) & mask_m[:, None] & mask_n[None, :]
                    d_0 = (pid_m - n_blk) * BLOCK_M - BLOCK_N
                    has_rel = (d_0 < REL_EXTENT) and (d_0 + W_CHUNK > 0)

                    if has_rel:
                        d_col = d_0 + offs_w
                        mask_w = (d_col >= 0) & (d_col < REL_EXTENT)
                        p_chunk = tl.load(RelProj + offs_r[:, None] * stride_pr + d_col[None, :] * stride_pd, mask=mask_w[None, :], other=0.0)
                        b_dist = tl.dot(rs, p_chunk).to(tl.float32)
                        tl.debug_barrier()
                        tl.store(scratch_cta + local_m[:, None] * W_CHUNK + offs_w[None, :], b_dist)
                        tl.debug_barrier()
                        rel_b = tl.load(scratch_cta + local_m[:, None] * W_CHUNK + w_idx, mask=rel_mask, other=0.0)
                        qk += rel_b

                    causal_mask = (dist >= 0) & mask_m[:, None] & mask_n[None, :]
                    if IS_SLIDING:
                        causal_mask = causal_mask & (dist < SLIDING_WIN)
                    if HAS_DOC:
                        doc_n = tl.load(DocId + b * stride_docb + offs_n, mask=mask_n, other=-2).to(tl.int32)
                        causal_mask = causal_mask & (doc_m[:, None] == doc_n[None, :])
                    qk = tl.where(causal_mask, qk, float("-inf"))
                    p = tl.exp(qk - l_i[:, None])
                    dp = tl.dot(do, tl.trans(v)).to(tl.float32)
                    ds_raw = p * (dp - d_i[:, None])
                    ds = ds_raw * sm_scale
                    dq += tl.dot(ds.to(k.dtype), tl.trans(k))

                    if has_rel:
                        d_col = d_0 + offs_w
                        mask_w = (d_col >= 0) & (d_col < REL_EXTENT)
                        p_chunk = tl.load(RelProj + offs_r[:, None] * stride_pr + d_col[None, :] * stride_pd, mask=mask_w[None, :], other=0.0)
                        tl.debug_barrier()
                        tl.store(scratch_cta + local_m[:, None] * W_CHUNK + offs_w[None, :], 0.0)
                        tl.debug_barrier()
                        tl.store(scratch_cta + local_m[:, None] * W_CHUNK + w_idx, ds_raw, mask=rel_mask)
                        tl.debug_barrier()
                        ds_dist = tl.load(scratch_cta + local_m[:, None] * W_CHUNK + offs_w[None, :]).to(p_chunk.dtype)
                        d_rs += tl.dot(ds_dist, tl.trans(p_chunk)).to(tl.float32)

            tl.store(dQ + b * stride_qb + h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd, dq.to(dQ.dtype.element_ty), mask=mask_m[:, None])
            tl.store(dRelStates + b * stride_rsb + offs_m[:, None] * stride_rsm + h * stride_rsh + offs_r[None, :] * stride_rsr, d_rs.to(dRelStates.dtype.element_ty), mask=mask_m[:, None])

    @triton.jit
    def inkling_flash_attn_lr_varlen_bwd_dkv_kernel(
        Q, K, V, RelStates, RelProj, dOut, L_ptr, D_vec, dK, dV, DocId, Scratch,
        stride_qb, stride_qh, stride_qm, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_rsb, stride_rsm, stride_rsh, stride_rsr,
        stride_pr, stride_pd,
        stride_docb,
        stride_dob, stride_doh, stride_dom, stride_dod,
        S, sm_scale, B_dim, H_q, H_kv,
        HAS_DOC: tl.constexpr, R_DIM: tl.constexpr, REL_EXTENT: tl.constexpr, IS_SLIDING: tl.constexpr, SLIDING_WIN: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, W_CHUNK: tl.constexpr, HEAD_DIM: tl.constexpr,
    ):
        cta_id = tl.program_id(0)
        num_ctas = tl.num_programs(0)
        num_n_blks = tl.cdiv(S, BLOCK_N)
        total_tiles = num_n_blks * (B_dim * H_kv)
        G_rep: tl.constexpr = 3

        local_m = tl.arange(0, BLOCK_M)
        local_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, HEAD_DIM)
        offs_r = tl.arange(0, R_DIM)
        offs_w = tl.arange(0, W_CHUNK)
        w_idx = BLOCK_N + local_m[:, None] - local_n[None, :]
        scratch_cta = Scratch + cta_id * (BLOCK_M * W_CHUNK)

        for tile_id in range(cta_id, total_tiles, num_ctas):
            pid_bkv = tile_id // num_n_blks
            pid_n = tile_id % num_n_blks
            b = pid_bkv // H_kv
            kv_h = pid_bkv % H_kv

            n_start = pid_n * BLOCK_N
            n_end = tl.minimum((pid_n + 1) * BLOCK_N, S)
            offs_n = n_start + local_n
            mask_n = offs_n < S
            k = tl.load(K + b * stride_kb + kv_h * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd, mask=mask_n[:, None], other=0.0)
            v = tl.load(V + b * stride_vb + kv_h * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd, mask=mask_n[:, None], other=0.0)
            if HAS_DOC:
                doc_n = tl.load(DocId + b * stride_docb + offs_n, mask=mask_n, other=-2).to(tl.int32)
                doc_n_min = tl.load(DocId + b * stride_docb + n_start).to(tl.int32)
                doc_n_max = tl.load(DocId + b * stride_docb + n_end - 1).to(tl.int32)

            dk = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
            dv = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)

            m_start_blk = n_start // BLOCK_M
            m_end_blk = tl.cdiv(tl.minimum(n_end + SLIDING_WIN - 1, S), BLOCK_M) if IS_SLIDING else tl.cdiv(S, BLOCK_M)

            for g in range(G_rep):
                h = kv_h * G_rep + g
                pid_bh = b * H_q + h
                for m_blk in range(m_start_blk, m_end_blk):
                    m_s = m_blk * BLOCK_M
                    m_e = tl.minimum((m_blk + 1) * BLOCK_M, S)
                    active_tile = True
                    if HAS_DOC:
                        doc_m_min = tl.load(DocId + b * stride_docb + m_s).to(tl.int32)
                        doc_m_max = tl.load(DocId + b * stride_docb + m_e - 1).to(tl.int32)
                        active_tile = (doc_m_max >= doc_n_min) and (doc_m_min <= doc_n_max)

                    if active_tile:
                        offs_m = m_s + local_m
                        mask_m = offs_m < S
                        q = tl.load(Q + b * stride_qb + h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd, mask=mask_m[:, None], other=0.0)
                        do = tl.load(dOut + b * stride_dob + h * stride_doh + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod, mask=mask_m[:, None], other=0.0)
                        l_i = tl.load(L_ptr + pid_bh * S + offs_m, mask=mask_m, other=0.0)
                        d_i = tl.load(D_vec + pid_bh * S + offs_m, mask=mask_m, other=0.0)
                        qk = tl.dot(q, tl.trans(k)).to(tl.float32) * sm_scale
                        dist = offs_m[:, None] - offs_n[None, :]

                        d_0 = (m_blk - pid_n) * BLOCK_M - BLOCK_N
                        if (d_0 < REL_EXTENT) and (d_0 + W_CHUNK > 0):
                            rs = tl.load(RelStates + b * stride_rsb + offs_m[:, None] * stride_rsm + h * stride_rsh + offs_r[None, :] * stride_rsr, mask=mask_m[:, None], other=0.0)
                            d_col = d_0 + offs_w
                            mask_w = (d_col >= 0) & (d_col < REL_EXTENT)
                            p_chunk = tl.load(RelProj + offs_r[:, None] * stride_pr + d_col[None, :] * stride_pd, mask=mask_w[None, :], other=0.0)
                            b_dist = tl.dot(rs, p_chunk).to(tl.float32)
                            tl.debug_barrier()
                            tl.store(scratch_cta + local_m[:, None] * W_CHUNK + offs_w[None, :], b_dist)
                            tl.debug_barrier()
                            rel_mask = (dist >= 0) & (dist < REL_EXTENT) & mask_m[:, None] & mask_n[None, :]
                            rel_b = tl.load(scratch_cta + local_m[:, None] * W_CHUNK + w_idx, mask=rel_mask, other=0.0)
                            qk += rel_b

                        causal_mask = (dist >= 0) & mask_m[:, None] & mask_n[None, :]
                        if IS_SLIDING:
                            causal_mask = causal_mask & (dist < SLIDING_WIN)
                        if HAS_DOC:
                            doc_m = tl.load(DocId + b * stride_docb + offs_m, mask=mask_m, other=-1).to(tl.int32)
                            causal_mask = causal_mask & (doc_m[:, None] == doc_n[None, :])
                        qk = tl.where(causal_mask, qk, float("-inf"))
                        p = tl.exp(qk - l_i[:, None])
                        dv += tl.dot(tl.trans(p.to(do.dtype)), do)
                        dp = tl.dot(do, tl.trans(v)).to(tl.float32)
                        ds = p * (dp - d_i[:, None]) * sm_scale
                        dk += tl.dot(tl.trans(ds.to(q.dtype)), q)

            tl.store(dK + b * stride_kb + kv_h * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd, dk.to(dK.dtype.element_ty), mask=mask_n[:, None])
            tl.store(dV + b * stride_vb + kv_h * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd, dv.to(dV.dtype.element_ty), mask=mask_n[:, None])

    class InklingFlashAttnLRVarlenFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, k, v, rel_states, rel_proj, sinks, doc_id, sm_scale, is_sliding, sliding_win):
            B_a, H_q, S_a, D_a = q.shape
            H_kv = k.shape[1]
            sinks_2d = sinks.squeeze(-1).contiguous()
            has_doc = doc_id is not None
            doc_ptr = doc_id if has_doc else sinks_2d
            doc_sb = doc_id.stride(0) if has_doc else 0
            out = torch.empty((B_a, S_a, H_q, D_a), device=q.device, dtype=q.dtype)
            L_buf = torch.empty((B_a * H_q, S_a), device=q.device, dtype=torch.float32)
            inkling_flash_attn_lr_varlen_fwd_kernel[(NUM_LR_CTAS,)](
                q, k, v, rel_states, rel_proj, sinks_2d, doc_ptr, out, L_buf, lr_scratch_f32,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                rel_states.stride(0), rel_states.stride(1), rel_states.stride(2), rel_states.stride(3),
                rel_proj.stride(0), rel_proj.stride(1),
                sinks_2d.stride(0), sinks_2d.stride(1), sinks_2d.stride(2),
                doc_sb,
                out.stride(0), out.stride(2), out.stride(1), out.stride(3),
                S_a, sm_scale, B_a, H_q, H_kv,
                HAS_DOC=has_doc, R_DIM=16, REL_EXTENT=rel_proj.shape[-1], IS_SLIDING=is_sliding, SLIDING_WIN=sliding_win,
                BLOCK_M=64, BLOCK_N=64, W_CHUNK=128, HEAD_DIM=D_a, num_warps=4, num_stages=2,
            )
            ctx.save_for_backward(q, k, v, rel_states, rel_proj, sinks_2d, doc_id if has_doc else torch.empty(0, device=q.device), out, L_buf)
            ctx.has_doc, ctx.sm_scale, ctx.is_sliding, ctx.sliding_win = has_doc, sm_scale, is_sliding, sliding_win
            return out

        @staticmethod
        def backward(ctx, grad_out):
            q, k, v, rel_states, rel_proj, sinks_2d, doc_id, out, L_buf = ctx.saved_tensors
            B_a, H_q, S_a, D_a = q.shape
            H_kv = k.shape[1]
            has_doc = ctx.has_doc
            doc_ptr = doc_id if has_doc else sinks_2d
            doc_sb = doc_id.stride(0) if has_doc else 0
            D_vec = torch.empty((B_a * H_q, S_a), device=q.device, dtype=torch.float32)
            inkling_flash_attn_bwd_prep_kernel[(triton.cdiv(S_a, 64), B_a * H_q)](
                out, grad_out, D_vec,
                out.stride(0), out.stride(2), out.stride(1), out.stride(3),
                grad_out.stride(0), grad_out.stride(2), grad_out.stride(1), grad_out.stride(3),
                S_a, H_q, BLOCK_M=64, HEAD_DIM=D_a, num_warps=4,
            )
            dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
            d_rel_states = torch.empty_like(rel_states)
            dsinks = torch.empty_like(sinks_2d)
            inkling_flash_attn_lr_varlen_bwd_dq_kernel[(NUM_LR_CTAS,)](
                q, k, v, rel_states, rel_proj, grad_out, L_buf, D_vec, dq, d_rel_states, sinks_2d, dsinks, doc_ptr, lr_scratch_f32,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                rel_states.stride(0), rel_states.stride(1), rel_states.stride(2), rel_states.stride(3),
                rel_proj.stride(0), rel_proj.stride(1),
                sinks_2d.stride(0), sinks_2d.stride(1), sinks_2d.stride(2),
                doc_sb,
                grad_out.stride(0), grad_out.stride(2), grad_out.stride(1), grad_out.stride(3),
                S_a, ctx.sm_scale, B_a, H_q, H_kv,
                HAS_DOC=has_doc, R_DIM=16, REL_EXTENT=rel_proj.shape[-1], IS_SLIDING=ctx.is_sliding, SLIDING_WIN=ctx.sliding_win,
                BLOCK_M=64, BLOCK_N=64, W_CHUNK=128, HEAD_DIM=D_a, num_warps=4, num_stages=1,
            )
            inkling_flash_attn_lr_varlen_bwd_dkv_kernel[(NUM_LR_CTAS,)](
                q, k, v, rel_states, rel_proj, grad_out, L_buf, D_vec, dk, dv, doc_ptr, lr_scratch_f32,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                rel_states.stride(0), rel_states.stride(1), rel_states.stride(2), rel_states.stride(3),
                rel_proj.stride(0), rel_proj.stride(1),
                doc_sb,
                grad_out.stride(0), grad_out.stride(2), grad_out.stride(1), grad_out.stride(3),
                S_a, ctx.sm_scale, B_a, H_q, H_kv,
                HAS_DOC=has_doc, R_DIM=16, REL_EXTENT=rel_proj.shape[-1], IS_SLIDING=ctx.is_sliding, SLIDING_WIN=ctx.sliding_win,
                BLOCK_M=64, BLOCK_N=64, W_CHUNK=128, HEAD_DIM=D_a, num_warps=4, num_stages=1,
            )
            return dq, dk, dv, d_rel_states, None, dsinks.unsqueeze(-1), None, None, None, None

    attn_ctx = {"doc_id": None}

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
        attn_output = InklingFlashAttnLRVarlenFn.apply(
            query_states, key_states, value_states, rel_states, self.relative_position_bias.proj, sinks, attn_ctx["doc_id"],
            self.scaling, bool(self.is_sliding), self.sliding_window if self.is_sliding else 1024,
        )
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
        attn_output = InklingFlashAttnLRVarlenFn.apply(
            query_states, key_states, value_states, rel_states, self.relative_position_bias.proj, sinks, attn_ctx["doc_id"],
            self.scaling, bool(self.is_sliding), self.sliding_window if self.is_sliding else 1024,
        )
        return self.o_proj(attn_output.reshape(*input_shape, -1), hidden_states), None


    # =========================================================================
    # 3. ZERO-COPY MOE PIPELINE (Counting Sort + Fused Gather-Quant + BLOCK_M Tune)
    # =========================================================================
    @triton.jit
    def bincount_topp_kernel(FlatE_ptr, Counts_ptr, N_items, NUM_EXPERTS: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N_items
        e = tl.load(FlatE_ptr + offs, mask=mask, other=NUM_EXPERTS)
        valid = mask & (e < NUM_EXPERTS)
        tl.atomic_add(Counts_ptr + tl.where(valid, e, 0), 1, mask=valid)

    @triton.jit
    def counting_sort_scatter_kernel(
        FlatE_ptr, WritePtr_ptr, SortedTok_ptr, SortedE_ptr, SortedSlot_ptr, InvIdx_ptr,
        N_items, TOP_K: tl.constexpr, NUM_EXPERTS: tl.constexpr, BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N_items
        e = tl.load(FlatE_ptr + offs, mask=mask, other=NUM_EXPERTS)
        valid = mask & (e < NUM_EXPERTS)
        pos = tl.atomic_add(WritePtr_ptr + tl.where(valid, e, 0), 1, mask=valid)
        tl.store(SortedTok_ptr + pos, (offs // TOP_K).to(tl.int32), mask=valid)
        tl.store(SortedE_ptr + pos, e.to(tl.int32), mask=valid)
        tl.store(SortedSlot_ptr + pos, offs.to(tl.int32), mask=valid)
        inv_val = tl.where(valid, pos.to(tl.int32), -1)
        tl.store(InvIdx_ptr + offs, inv_val, mask=mask)

    @triton.jit
    def build_schedule_topp_kernel(
        Counts_ptr, ExpOffsets_ptr, BlockOffsets_ptr, TotalBlocks_ptr, ActiveTokens_ptr,
        BExp_ptr, BStart_ptr, BEnd_ptr,
        num_experts: tl.constexpr, BLOCK_M: tl.constexpr,
    ):
        pid_e = tl.program_id(0)
        c = tl.load(Counts_ptr + pid_e)
        e_off = tl.load(ExpOffsets_ptr + pid_e)
        b_off = tl.load(BlockOffsets_ptr + pid_e)
        n_blks = (c + BLOCK_M - 1) // BLOCK_M
        if pid_e == num_experts - 1:
            tl.store(TotalBlocks_ptr, b_off + n_blks)
            tl.store(ActiveTokens_ptr, e_off + c)
        for b in range(n_blks):
            tl.store(BExp_ptr + b_off + b, pid_e)
            s = e_off + b * BLOCK_M
            tl.store(BStart_ptr + b_off + b, s)
            tl.store(BEnd_ptr + b_off + b, tl.minimum(s + BLOCK_M, e_off + c))

    @triton.jit
    def fused_moe_gather_quant_fwd_kernel(
        H_ptr, SortedTok_ptr, Out_i8_ptr, Scale_ptr, ActiveTokens_ptr,
        stride_hm, stride_om, K_dim, BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        if pid_m >= tl.load(ActiveTokens_ptr):
            return
        tok_id = tl.load(SortedTok_ptr + pid_m)
        offs_k = tl.arange(0, BLOCK_K)
        mask_k = offs_k < K_dim
        x = tl.load(H_ptr + tok_id * stride_hm + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        scale = tl.maximum(tl.max(tl.abs(x), axis=0), 1e-8) / 127.0
        q = tl.minimum(tl.maximum(tl.extra.cuda.libdevice.rint(x / scale), -127.0), 127.0).to(tl.int8)
        tl.store(Out_i8_ptr + pid_m * stride_om + offs_k, q, mask=mask_k)
        tl.store(Scale_ptr + pid_m, scale)

    @triton.jit
    def fused_moe_gather_scale_quant_bwd_kernel(
        GradOut_ptr, W_flat_ptr, ScaleW_ptr, SortedTok_ptr, SortedSlot_ptr, SortedE_ptr,
        Out_i8_ptr, Scale_ptr, ActiveTokens_ptr,
        stride_gm, stride_se, stride_om, K_dim, BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        if pid_m >= tl.load(ActiveTokens_ptr):
            return
        tok_id = tl.load(SortedTok_ptr + pid_m)
        slot_id = tl.load(SortedSlot_ptr + pid_m)
        e = tl.load(SortedE_ptr + pid_m)
        w = tl.load(W_flat_ptr + slot_id).to(tl.float32)
        offs_k = tl.arange(0, BLOCK_K)
        mask_k = offs_k < K_dim
        g = tl.load(GradOut_ptr + tok_id * stride_gm + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        sw = tl.load(ScaleW_ptr + e * stride_se + offs_k, mask=mask_k, other=1.0).to(tl.float32)
        x = (g * w) * sw
        scale = tl.maximum(tl.max(tl.abs(x), axis=0), 1e-8) / 127.0
        q = tl.minimum(tl.maximum(tl.extra.cuda.libdevice.rint(x / scale), -127.0), 127.0).to(tl.int8)
        tl.store(Out_i8_ptr + pid_m * stride_om + offs_k, q, mask=mask_k)
        tl.store(Scale_ptr + pid_m, scale)

    @triton.jit
    def grouped_int8_gemm_blkq_kernel(
        X_ptr, ScaleX_ptr, W_ptr, ScaleW_ptr, Y_i8_ptr, Y_scale_ptr,
        block_expert_ptr, block_m_start_ptr, block_m_end_ptr, total_blocks_ptr,
        Z_ptr, B_lora_ptr, SortedTok_ptr, lora_scale,
        stride_xm, stride_xk, stride_we, stride_wn, stride_wk, stride_se, stride_sn,
        stride_ym, stride_yn, stride_ysm, stride_ysn,
        stride_zm, stride_zr, stride_be, stride_bn, stride_br,
        N_dim, K_dim,
        HAS_LORA: tl.constexpr, USE_SORTED_TOK: tl.constexpr, R_LORA: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid_m_blk = tl.program_id(0)
        if pid_m_blk >= tl.load(total_blocks_ptr):
            return
        pid_n = tl.program_id(1)
        e = tl.load(block_expert_ptr + pid_m_blk)
        m_start = tl.load(block_m_start_ptr + pid_m_blk)
        m_end = tl.load(block_m_end_ptr + pid_m_blk)
        offs_m = m_start + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = offs_m < m_end
        mask_n = offs_n < N_dim
        scale_x = tl.load(ScaleX_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)
        scale_w = tl.load(ScaleW_ptr + e * stride_se + offs_n * stride_sn, mask=mask_n, other=0.0).to(tl.float32)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
        for k0 in range(0, K_dim, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K_dim
            x_i8 = tl.load(X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk, mask=mask_m[:, None] & mask_k[None, :], other=0)
            w_i8 = tl.load(W_ptr + e * stride_we + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk, mask=mask_n[:, None] & mask_k[None, :], other=0)
            acc += tl.dot(x_i8, tl.trans(w_i8), out_dtype=tl.int32)
        out = acc.to(tl.float32) * (scale_x[:, None] * scale_w[None, :])
        if HAS_LORA:
            offs_r = tl.arange(0, R_LORA)
            if USE_SORTED_TOK:
                z_rows = tl.load(SortedTok_ptr + offs_m, mask=mask_m, other=0)
            else:
                z_rows = offs_m
            z_tile = tl.load(Z_ptr + z_rows[:, None] * stride_zm + offs_r[None, :] * stride_zr, mask=mask_m[:, None], other=0.0).to(tl.bfloat16)
            b_tile = tl.load(B_lora_ptr + e * stride_be + offs_n[:, None] * stride_bn + offs_r[None, :] * stride_br, mask=mask_n[:, None], other=0.0).to(tl.bfloat16)
            out += tl.dot(z_tile, tl.trans(b_tile), out_dtype=tl.float32) * lora_scale
        out = tl.where(mask_n[None, :], out, 0.0)
        blk_scale = tl.maximum(tl.max(tl.abs(out), axis=1), 1e-8) / 127.0
        out_i8 = tl.minimum(tl.maximum(tl.extra.cuda.libdevice.rint(out / blk_scale[:, None]), -127.0), 127.0).to(tl.int8)
        tl.store(Y_i8_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn, out_i8, mask=mask_m[:, None] & mask_n[None, :])
        tl.store(Y_scale_ptr + offs_m * stride_ysm + pid_n * stride_ysn, blk_scale, mask=mask_m)

    @triton.jit
    def dequant_matmul_z_dn_kernel(
        A_i8_ptr, ScaleA_ptr, LA_ptr, Z_dn_ptr, ActiveTokens_ptr,
        stride_am, stride_lar, stride_lak, stride_zm,
        I_dim, R_LORA: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        act_toks = tl.load(ActiveTokens_ptr)
        m0 = pid_m * BLOCK_M
        if m0 >= act_toks:
            return
        offs_m = m0 + tl.arange(0, BLOCK_M)
        mask_m = offs_m < act_toks
        offs_r = tl.arange(0, R_LORA)
        sa = tl.load(ScaleA_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)
        acc = tl.zeros((BLOCK_M, R_LORA), dtype=tl.float32)
        for k0 in range(0, I_dim, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < I_dim
            a_i8 = tl.load(A_i8_ptr + offs_m[:, None] * stride_am + offs_k[None, :], mask=mask_m[:, None] & mask_k[None, :], other=0)
            a_bf16 = (a_i8.to(tl.float32) * sa[:, None]).to(tl.bfloat16)
            w_a = tl.load(LA_ptr + offs_r[:, None] * stride_lar + offs_k[None, :] * stride_lak, mask=mask_k[None, :], other=0.0).to(tl.bfloat16)
            acc += tl.dot(a_bf16, tl.trans(w_a), out_dtype=tl.float32)
        tl.store(Z_dn_ptr + offs_m[:, None] * stride_zm + offs_r[None, :], acc.to(tl.bfloat16), mask=mask_m[:, None])

    @triton.jit
    def moe_lora_dB_dn_kernel(
        GradOut_ptr, W_flat_ptr, SortedTok_ptr, SortedSlot_ptr, Z_dn_ptr,
        ExpOffsets_ptr, Counts_ptr, dB_ptr, lora_scale,
        stride_gm, stride_zm, stride_be, stride_bn, stride_br,
        N_dn, R_LORA: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        e = tl.program_id(0)
        pid_n = tl.program_id(1)
        start = tl.load(ExpOffsets_ptr + e)
        cnt = tl.load(Counts_ptr + e)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N_dn
        offs_r = tl.arange(0, R_LORA)
        acc = tl.zeros((BLOCK_N, R_LORA), dtype=tl.float32)
        for m_off in range(0, cnt, BLOCK_M):
            loc_m = m_off + tl.arange(0, BLOCK_M)
            mask_m = loc_m < cnt
            offs_m = start + loc_m
            tok = tl.load(SortedTok_ptr + offs_m, mask=mask_m, other=0)
            slot = tl.load(SortedSlot_ptr + offs_m, mask=mask_m, other=0)
            w = tl.load(W_flat_ptr + slot, mask=mask_m, other=0.0).to(tl.float32)
            g = tl.load(GradOut_ptr + tok[:, None] * stride_gm + offs_n[None, :], mask=mask_m[:, None] & mask_n[None, :], other=0.0).to(tl.float32)
            dy = (g * w[:, None]).to(tl.bfloat16)
            z = tl.load(Z_dn_ptr + offs_m[:, None] * stride_zm + offs_r[None, :], mask=mask_m[:, None], other=0.0).to(tl.bfloat16)
            acc += tl.dot(tl.trans(dy), z, out_dtype=tl.float32)
        tl.store(dB_ptr + e * stride_be + offs_n[:, None] * stride_bn + offs_r[None, :] * stride_br, (acc * lora_scale).to(tl.bfloat16), mask=mask_n[:, None])

    @triton.jit
    def moe_lora_dz_dn_kernel(
        GradOut_ptr, W_flat_ptr, SortedTok_ptr, SortedSlot_ptr, B_ptr,
        BExp_ptr, BStart_ptr, BEnd_ptr, TotalBlocks_ptr, dZ_ptr, lora_scale,
        stride_gm, stride_be, stride_bn, stride_br, stride_dzm,
        N_dn, R_LORA: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        pid_blk = tl.program_id(0)
        if pid_blk >= tl.load(TotalBlocks_ptr):
            return
        e = tl.load(BExp_ptr + pid_blk)
        m_start = tl.load(BStart_ptr + pid_blk)
        m_end = tl.load(BEnd_ptr + pid_blk)
        offs_m = m_start + tl.arange(0, BLOCK_M)
        mask_m = offs_m < m_end
        tok = tl.load(SortedTok_ptr + offs_m, mask=mask_m, other=0)
        slot = tl.load(SortedSlot_ptr + offs_m, mask=mask_m, other=0)
        w = tl.load(W_flat_ptr + slot, mask=mask_m, other=0.0).to(tl.float32)
        offs_r = tl.arange(0, R_LORA)
        acc = tl.zeros((BLOCK_M, R_LORA), dtype=tl.float32)
        for n0 in range(0, N_dn, BLOCK_N):
            offs_n = n0 + tl.arange(0, BLOCK_N)
            mask_n = offs_n < N_dn
            g = tl.load(GradOut_ptr + tok[:, None] * stride_gm + offs_n[None, :], mask=mask_m[:, None] & mask_n[None, :], other=0.0).to(tl.float32)
            dy = (g * w[:, None]).to(tl.bfloat16)
            b = tl.load(B_ptr + e * stride_be + offs_n[:, None] * stride_bn + offs_r[None, :] * stride_br, mask=mask_n[:, None], other=0.0).to(tl.bfloat16)
            acc += tl.dot(dy, b, out_dtype=tl.float32)
        tl.store(dZ_ptr + offs_m[:, None] * stride_dzm + offs_r[None, :], (acc * lora_scale).to(tl.bfloat16), mask=mask_m[:, None])

    @triton.jit
    def moe_lora_dA_dn_split_kernel(
        dZ_ptr, A_i8_ptr, ScaleA_ptr, ActiveTokens_ptr, Partial_ptr,
        stride_dzm, stride_am, stride_ps, stride_pr,
        I_dim, SPLIT_M: tl.constexpr, R_LORA: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid_s = tl.program_id(0)
        pid_k = tl.program_id(1)
        act_toks = tl.load(ActiveTokens_ptr)
        chunk = (act_toks + SPLIT_M - 1) // SPLIT_M
        m_start = pid_s * chunk
        m_end = tl.minimum(m_start + chunk, act_toks)
        offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < I_dim
        offs_r = tl.arange(0, R_LORA)
        acc = tl.zeros((R_LORA, BLOCK_K), dtype=tl.float32)
        for m0 in range(m_start, m_end, BLOCK_M):
            offs_m = m0 + tl.arange(0, BLOCK_M)
            mask_m = offs_m < m_end
            dz = tl.load(dZ_ptr + offs_m[:, None] * stride_dzm + offs_r[None, :], mask=mask_m[:, None], other=0.0).to(tl.bfloat16)
            sa = tl.load(ScaleA_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)
            a_i8 = tl.load(A_i8_ptr + offs_m[:, None] * stride_am + offs_k[None, :], mask=mask_m[:, None] & mask_k[None, :], other=0)
            a_bf16 = (a_i8.to(tl.float32) * sa[:, None]).to(tl.bfloat16)
            acc += tl.dot(tl.trans(dz), a_bf16, out_dtype=tl.float32)
        tl.store(Partial_ptr + pid_s * stride_ps + offs_r[:, None] * stride_pr + offs_k[None, :], acc, mask=mask_k[None, :])

    @triton.jit
    def moe_lora_dB_gu_kernel(
        dGU_i8_ptr, SGU_ptr, GU_scale_ptr, SortedTok_ptr, Z_gu_tok_ptr,
        ExpOffsets_ptr, Counts_ptr, dB_ptr, lora_scale,
        stride_dm, stride_se, stride_zm, stride_be, stride_bn, stride_br,
        N_gu, R_LORA: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        e = tl.program_id(0)
        pid_n = tl.program_id(1)
        start = tl.load(ExpOffsets_ptr + e)
        cnt = tl.load(Counts_ptr + e)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N_gu
        offs_r = tl.arange(0, R_LORA)
        inv_sw = 1.0 / tl.maximum(tl.load(GU_scale_ptr + e * stride_se + offs_n, mask=mask_n, other=1.0).to(tl.float32), 1e-8)
        acc = tl.zeros((BLOCK_N, R_LORA), dtype=tl.float32)
        for m_off in range(0, cnt, BLOCK_M):
            loc_m = m_off + tl.arange(0, BLOCK_M)
            mask_m = loc_m < cnt
            offs_m = start + loc_m
            sgu = tl.load(SGU_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)
            dg_i8 = tl.load(dGU_i8_ptr + offs_m[:, None] * stride_dm + offs_n[None, :], mask=mask_m[:, None] & mask_n[None, :], other=0)
            dy = (dg_i8.to(tl.float32) * (sgu[:, None] * inv_sw[None, :])).to(tl.bfloat16)
            tok = tl.load(SortedTok_ptr + offs_m, mask=mask_m, other=0)
            z = tl.load(Z_gu_tok_ptr + tok[:, None] * stride_zm + offs_r[None, :], mask=mask_m[:, None], other=0.0).to(tl.bfloat16)
            acc += tl.dot(tl.trans(dy), z, out_dtype=tl.float32)
        tl.store(dB_ptr + e * stride_be + offs_n[:, None] * stride_bn + offs_r[None, :] * stride_br, (acc * lora_scale).to(tl.bfloat16), mask=mask_n[:, None])

    @triton.jit
    def moe_lora_dz_gu_kernel(
        dGU_i8_ptr, SGU_ptr, GU_scale_ptr, B_ptr,
        BExp_ptr, BStart_ptr, BEnd_ptr, TotalBlocks_ptr, dZ_ptr, lora_scale,
        stride_dm, stride_se, stride_be, stride_bn, stride_br, stride_dzm,
        N_gu, R_LORA: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        pid_blk = tl.program_id(0)
        if pid_blk >= tl.load(TotalBlocks_ptr):
            return
        e = tl.load(BExp_ptr + pid_blk)
        m_start = tl.load(BStart_ptr + pid_blk)
        m_end = tl.load(BEnd_ptr + pid_blk)
        offs_m = m_start + tl.arange(0, BLOCK_M)
        mask_m = offs_m < m_end
        sgu = tl.load(SGU_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)
        offs_r = tl.arange(0, R_LORA)
        acc = tl.zeros((BLOCK_M, R_LORA), dtype=tl.float32)
        for n0 in range(0, N_gu, BLOCK_N):
            offs_n = n0 + tl.arange(0, BLOCK_N)
            mask_n = offs_n < N_gu
            inv_sw = 1.0 / tl.maximum(tl.load(GU_scale_ptr + e * stride_se + offs_n, mask=mask_n, other=1.0).to(tl.float32), 1e-8)
            dg_i8 = tl.load(dGU_i8_ptr + offs_m[:, None] * stride_dm + offs_n[None, :], mask=mask_m[:, None] & mask_n[None, :], other=0)
            dy = (dg_i8.to(tl.float32) * (sgu[:, None] * inv_sw[None, :])).to(tl.bfloat16)
            b = tl.load(B_ptr + e * stride_be + offs_n[:, None] * stride_bn + offs_r[None, :] * stride_br, mask=mask_n[:, None], other=0.0).to(tl.bfloat16)
            acc += tl.dot(dy, b, out_dtype=tl.float32)
        tl.store(dZ_ptr + offs_m[:, None] * stride_dzm + offs_r[None, :], (acc * lora_scale).to(tl.bfloat16), mask=mask_m[:, None])

    @triton.jit
    def moe_reduce_dz_gu_kernel(
        dZ_sorted_ptr, InvIdx_ptr, dZ_tok_ptr,
        stride_dzm, stride_ztm,
        TOP_K: tl.constexpr, R_LORA: tl.constexpr,
    ):
        pid_t = tl.program_id(0)
        offs_r = tl.arange(0, R_LORA)
        acc = tl.zeros((R_LORA,), dtype=tl.float32)
        for k in range(TOP_K):
            src_idx = tl.load(InvIdx_ptr + pid_t * TOP_K + k)
            if src_idx >= 0:
                acc += tl.load(dZ_sorted_ptr + src_idx * stride_dzm + offs_r).to(tl.float32)
        tl.store(dZ_tok_ptr + pid_t * stride_ztm + offs_r, acc.to(tl.bfloat16))

    @triton.jit
    def grouped_int8_gemm_dx_direct_kernel(
        dY_ptr, ScaleY_ptr, W_ptr, dX_ptr,
        block_expert_ptr, block_m_start_ptr, block_m_end_ptr, total_blocks_ptr,
        stride_ym, stride_yn, stride_we, stride_wn, stride_wk, stride_xm, stride_xk,
        N_dim, K_dim, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        pid_m_blk = tl.program_id(0)
        if pid_m_blk >= tl.load(total_blocks_ptr):
            return
        pid_k = tl.program_id(1)
        e = tl.load(block_expert_ptr + pid_m_blk)
        m_start = tl.load(block_m_start_ptr + pid_m_blk)
        m_end = tl.load(block_m_end_ptr + pid_m_blk)
        offs_m = m_start + tl.arange(0, BLOCK_M)
        offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_m = offs_m < m_end
        mask_k = offs_k < K_dim
        scale_y = tl.load(ScaleY_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)
        acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.int32)
        for n0 in range(0, N_dim, BLOCK_N):
            offs_n = n0 + tl.arange(0, BLOCK_N)
            mask_n = offs_n < N_dim
            dy_i8 = tl.load(dY_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn, mask=mask_m[:, None] & mask_n[None, :], other=0)
            w_i8 = tl.load(W_ptr + e * stride_we + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk, mask=mask_n[:, None] & mask_k[None, :], other=0)
            acc += tl.dot(dy_i8, w_i8, out_dtype=tl.int32)
        out = acc.to(tl.float32) * scale_y[:, None]
        tl.store(dX_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk, out.to(tl.bfloat16), mask=mask_m[:, None] & mask_k[None, :])

    @triton.jit
    def softcapped_swiglu_quant_fwd_blkq_topp_kernel(
        GU_i8_ptr, GU_scale_ptr, Out_i8_ptr, Scale_ptr, ActiveTokens_ptr,
        stride_gum, stride_gsm, stride_om,
        I_dim, BLOCK_I: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        if pid_m >= tl.load(ActiveTokens_ptr):
            return
        offs_i = tl.arange(0, BLOCK_I)
        mask_i = offs_i < I_dim
        gu_base = GU_i8_ptr + pid_m * stride_gum
        gs_base = GU_scale_ptr + pid_m * stride_gsm
        blk_g = offs_i // 128
        blk_u = (I_dim + offs_i) // 128
        gate = tl.load(gu_base + offs_i, mask=mask_i, other=0).to(tl.float32) * tl.load(gs_base + blk_g, mask=mask_i, other=0.0)
        up = tl.load(gu_base + I_dim + offs_i, mask=mask_i, other=0).to(tl.float32) * tl.load(gs_base + blk_u, mask=mask_i, other=0.0)
        tg = tl.extra.cuda.libdevice.tanh(gate * 0.25)
        sig = tl.sigmoid(gate)
        tu = tl.extra.cuda.libdevice.tanh(up * 0.04)
        act = tl.where(mask_i, (4.0 * tg) * sig * (25.0 * tu), 0.0)
        scale = tl.maximum(tl.max(tl.abs(act), axis=0), 1e-8) / 127.0
        q = tl.minimum(tl.maximum(tl.extra.cuda.libdevice.rint(act / scale), -127.0), 127.0).to(tl.int8)
        tl.store(Out_i8_ptr + pid_m * stride_om + offs_i, q, mask=mask_i)
        tl.store(Scale_ptr + pid_m, scale)

    @triton.jit
    def softcapped_swiglu_scaled_quant_bwd_blkq_topp_kernel(
        GU_i8_ptr, GU_scale_ptr, dAct_ptr, ScaleW_ptr, SortedE_ptr, Out_i8_ptr, Scale_ptr, ActiveTokens_ptr,
        stride_gum, stride_gsm, stride_dam, stride_se, stride_om,
        I_dim, BLOCK_I: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        if pid_m >= tl.load(ActiveTokens_ptr):
            return
        e = tl.load(SortedE_ptr + pid_m)
        offs_i = tl.arange(0, BLOCK_I)
        mask_i = offs_i < I_dim
        gu_base = GU_i8_ptr + pid_m * stride_gum
        gs_base = GU_scale_ptr + pid_m * stride_gsm
        sw_base = ScaleW_ptr + e * stride_se
        blk_g = offs_i // 128
        blk_u = (I_dim + offs_i) // 128
        gate = tl.load(gu_base + offs_i, mask=mask_i, other=0).to(tl.float32) * tl.load(gs_base + blk_g, mask=mask_i, other=0.0)
        up = tl.load(gu_base + I_dim + offs_i, mask=mask_i, other=0).to(tl.float32) * tl.load(gs_base + blk_u, mask=mask_i, other=0.0)
        d_act = tl.load(dAct_ptr + pid_m * stride_dam + offs_i, mask=mask_i, other=0.0).to(tl.float32)
        sw_g = tl.load(sw_base + offs_i, mask=mask_i, other=1.0).to(tl.float32)
        sw_u = tl.load(sw_base + I_dim + offs_i, mask=mask_i, other=1.0).to(tl.float32)
        tg = tl.extra.cuda.libdevice.tanh(gate * 0.25)
        gate_cap = 4.0 * tg
        sig = tl.sigmoid(gate)
        tu = tl.extra.cuda.libdevice.tanh(up * 0.04)
        up_cap = 25.0 * tu
        d_up = d_act * (gate_cap * sig) * (1.0 - tu * tu)
        d_gate = (d_act * up_cap) * (sig * (1.0 - tg * tg) + gate_cap * sig * (1.0 - sig))
        dg_s = tl.where(mask_i, d_gate * sw_g, 0.0)
        du_s = tl.where(mask_i, d_up * sw_u, 0.0)
        max_val = tl.maximum(tl.max(tl.abs(dg_s), axis=0), tl.max(tl.abs(du_s), axis=0))
        scale = tl.maximum(max_val, 1e-8) / 127.0
        qg = tl.minimum(tl.maximum(tl.extra.cuda.libdevice.rint(dg_s / scale), -127.0), 127.0).to(tl.int8)
        qu = tl.minimum(tl.maximum(tl.extra.cuda.libdevice.rint(du_s / scale), -127.0), 127.0).to(tl.int8)
        out_base = Out_i8_ptr + pid_m * stride_om
        tl.store(out_base + offs_i, qg, mask=mask_i)
        tl.store(out_base + I_dim + offs_i, qu, mask=mask_i)
        tl.store(Scale_ptr + pid_m, scale)

    @triton.jit
    def moe_gather_reduce_fwd_blkq_topp_kernel(
        DN_i8_ptr, DN_scale_ptr, InvIdx_ptr, W_ptr, Out_ptr,
        H_dim, NUM_BLKS, TOP_K: tl.constexpr, BLOCK_H: tl.constexpr,
    ):
        pid_t = tl.program_id(0)
        pid_h = tl.program_id(1)
        offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
        mask_h = offs_h < H_dim
        blk_h = offs_h // 128
        acc = tl.zeros([BLOCK_H], dtype=tl.float32)
        for k in range(TOP_K):
            src_idx = tl.load(InvIdx_ptr + pid_t * TOP_K + k)
            if src_idx >= 0:
                w = tl.load(W_ptr + pid_t * TOP_K + k).to(tl.float32)
                val_i8 = tl.load(DN_i8_ptr + src_idx * H_dim + offs_h, mask=mask_h, other=0).to(tl.float32)
                val_sc = tl.load(DN_scale_ptr + src_idx * NUM_BLKS + blk_h, mask=mask_h, other=0.0)
                acc += (val_i8 * val_sc) * w
        tl.store(Out_ptr + pid_t * H_dim + offs_h, acc.to(Out_ptr.dtype.element_ty), mask=mask_h)

    @triton.jit
    def moe_gather_reduce_bwd_h_topp_kernel(
        DX_sorted_ptr, InvIdx_ptr, Out_dH_ptr,
        H_dim, TOP_K: tl.constexpr, BLOCK_H: tl.constexpr,
    ):
        pid_t = tl.program_id(0)
        pid_h = tl.program_id(1)
        offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
        mask_h = offs_h < H_dim
        acc = tl.zeros([BLOCK_H], dtype=tl.float32)
        for k in range(TOP_K):
            src_idx = tl.load(InvIdx_ptr + pid_t * TOP_K + k)
            if src_idx >= 0:
                val = tl.load(DX_sorted_ptr + src_idx * H_dim + offs_h, mask=mask_h, other=0.0).to(tl.float32)
                acc += val
        tl.store(Out_dH_ptr + pid_t * H_dim + offs_h, acc.to(Out_dH_ptr.dtype.element_ty), mask=mask_h)

    @triton.jit
    def moe_dw_blkq_topp_kernel(
        GradOut_ptr, DN_i8_ptr, DN_scale_ptr, InvIdx_ptr, dW_ptr,
        H_dim, NUM_BLKS, TOP_K: tl.constexpr, BLOCK_H: tl.constexpr,
    ):
        pid_t = tl.program_id(0)
        pid_k = tl.program_id(1)
        src_idx = tl.load(InvIdx_ptr + pid_t * TOP_K + pid_k)
        acc = 0.0
        if src_idx >= 0:
            for h0 in range(0, H_dim, BLOCK_H):
                offs_h = h0 + tl.arange(0, BLOCK_H)
                mask_h = offs_h < H_dim
                blk_h = offs_h // 128
                g = tl.load(GradOut_ptr + pid_t * H_dim + offs_h, mask=mask_h, other=0.0).to(tl.float32)
                dn = tl.load(DN_i8_ptr + src_idx * H_dim + offs_h, mask=mask_h, other=0).to(tl.float32) * tl.load(DN_scale_ptr + src_idx * NUM_BLKS + blk_h, mask=mask_h, other=0.0)
                acc += tl.sum(g * dn, axis=0)
        tl.store(dW_ptr + pid_t * TOP_K + pid_k, acc.to(dW_ptr.dtype.element_ty))


    # =========================================================================
    # 4. LOAD MODEL & PATCH FULL FAST STACK (INT8 ZeroCopy MoE + DepthAttn + T-BPTT 1/3 + CG + torch.compile)
    # =========================================================================
    torch.manual_seed(seed)
    m = ModelCls.from_pretrained("/model", dtype=dtype, device_map="cuda")
    if hasattr(m.model, "vision_tower"):
        del m.model.vision_tower
    gc.collect()
    torch.cuda.empty_cache()

    moe_cfg = {"top_p": 1.0, "k_min": 4, "block_m": 64, "record_stats": False, "active_sum": torch.zeros((), device=device, dtype=torch.int64), "tok_sum": 0}

    class PatchedINT8Experts(nn.Module):
        def __init__(self, orig_experts, delete_orig=True, enable_lora=False, r_lora=16, alpha_lora=32.0):
            super().__init__()
            self.num_experts = orig_experts.gate_up_proj.shape[0]
            N_gu, H_dim = orig_experts.gate_up_proj.shape[1], orig_experts.gate_up_proj.shape[2]
            N_dn, I_dim = orig_experts.down_proj.shape[1], orig_experts.down_proj.shape[2]
            gu = orig_experts.gate_up_proj.data.float()
            gu_s = gu.abs().amax(dim=2, keepdim=True).clamp(min=1e-8) / 127.0
            self.register_buffer("gu_int8", torch.round(gu / gu_s).clamp(-127, 127).to(torch.int8))
            self.register_buffer("gu_scale", gu_s.squeeze(2).to(torch.float32))
            dn = orig_experts.down_proj.data.float()
            dn_s = dn.abs().amax(dim=2, keepdim=True).clamp(min=1e-8) / 127.0
            self.register_buffer("dn_int8", torch.round(dn / dn_s).clamp(-127, 127).to(torch.int8))
            self.register_buffer("dn_scale", dn_s.squeeze(2).to(torch.float32))
            self.enable_lora = enable_lora
            self.r_lora = r_lora
            self.lora_scale = float(alpha_lora) / float(r_lora)
            if enable_lora:
                self.lora_A_gu = nn.Parameter(torch.randn(r_lora, H_dim, device=device, dtype=dtype) * 0.01)
                self.lora_B_gu = nn.Parameter(torch.zeros(self.num_experts, N_gu, r_lora, device=device, dtype=dtype))
                self.lora_A_dn = nn.Parameter(torch.randn(r_lora, I_dim, device=device, dtype=dtype) * 0.01)
                self.lora_B_dn = nn.Parameter(torch.zeros(self.num_experts, N_dn, r_lora, device=device, dtype=dtype))
            else:
                self.lora_A_gu = self.lora_B_gu = self.lora_A_dn = self.lora_B_dn = None
            if delete_orig:
                del orig_experts.gate_up_proj, orig_experts.down_proj

        def forward(self, hidden_states, top_k_index, top_k_weights):
            N_tok, H_dim = hidden_states.shape
            top_k = top_k_index.shape[1]
            top_p = moe_cfg["top_p"]
            BLOCK_M = moe_cfg["block_m"]

            if self.training and top_p < 1.0:
                k_min = moe_cfg["k_min"]
                w_sorted, order = torch.sort(top_k_weights, dim=-1, descending=True)
                idx_sorted = torch.gather(top_k_index, 1, order)
                orig_sum = w_sorted.sum(dim=-1, keepdim=True)
                w_prob = w_sorted / orig_sum.clamp(min=1e-8)
                excl_cum = torch.cumsum(w_prob, dim=-1) - w_prob
                active_mask = (excl_cum < top_p) | (torch.arange(top_k, device=hidden_states.device)[None, :] < k_min)
                w_masked = w_sorted * active_mask
                w_eff = w_masked * (orig_sum / w_masked.sum(dim=-1, keepdim=True).clamp(min=1e-8))
                flat_e = torch.where(active_mask.reshape(-1), idx_sorted.reshape(-1), self.num_experts).to(torch.int32)
            else:
                w_eff = top_k_weights
                flat_e = top_k_index.reshape(-1).to(torch.int32)

            M_flat = flat_e.numel()
            counts = torch.zeros((self.num_experts,), device=hidden_states.device, dtype=torch.int32)
            bincount_topp_kernel[(triton.cdiv(M_flat, 256),)](
                flat_e, counts, M_flat, NUM_EXPERTS=self.num_experts, BLOCK=256, num_warps=4,
            )
            num_blocks_per_exp = (counts + BLOCK_M - 1) // BLOCK_M
            exp_offsets = (torch.cumsum(counts, dim=0) - counts).to(torch.int32)
            block_offsets = (torch.cumsum(num_blocks_per_exp, dim=0) - num_blocks_per_exp).to(torch.int32)
            MAX_BLOCKS = triton.cdiv(M_flat, BLOCK_M) + self.num_experts
            b_exp = torch.empty((MAX_BLOCKS,), device=hidden_states.device, dtype=torch.int32)
            b_start = torch.empty((MAX_BLOCKS,), device=hidden_states.device, dtype=torch.int32)
            b_end = torch.empty((MAX_BLOCKS,), device=hidden_states.device, dtype=torch.int32)
            total_blocks = torch.empty((1,), device=hidden_states.device, dtype=torch.int32)
            active_tokens = torch.empty((1,), device=hidden_states.device, dtype=torch.int32)
            build_schedule_topp_kernel[(self.num_experts,)](
                counts, exp_offsets, block_offsets, total_blocks, active_tokens, b_exp, b_start, b_end,
                num_experts=self.num_experts, BLOCK_M=BLOCK_M, num_warps=1,
            )

            write_ptr = exp_offsets.clone()
            sorted_tok = torch.empty((M_flat,), device=hidden_states.device, dtype=torch.int32)
            sorted_e = torch.empty((M_flat,), device=hidden_states.device, dtype=torch.int32)
            sorted_slot = torch.empty((M_flat,), device=hidden_states.device, dtype=torch.int32)
            inv_idx = torch.empty((M_flat,), device=hidden_states.device, dtype=torch.int32)
            counting_sort_scatter_kernel[(triton.cdiv(M_flat, 256),)](
                flat_e, write_ptr, sorted_tok, sorted_e, sorted_slot, inv_idx,
                M_flat, TOP_K=top_k, NUM_EXPERTS=self.num_experts, BLOCK=256, num_warps=4,
            )

            if moe_cfg["record_stats"] and self.training and top_p < 1.0:
                moe_cfg["active_sum"] += active_tokens[0].to(torch.int64)
                moe_cfg["tok_sum"] += N_tok

            gu_int8, gu_scale = self.gu_int8, self.gu_scale
            dn_int8, dn_scale = self.dn_int8, self.dn_scale
            has_lora = self.enable_lora
            r_lora = self.r_lora
            lora_scale = self.lora_scale
            n_warps_gemm = 4 if BLOCK_M == 64 else 8
            num_experts_cur = self.num_experts

            class _ExpertsFn(torch.autograd.Function):
                @staticmethod
                def forward(ctx, h_in, t_w, la_gu, lb_gu, la_dn, lb_dn):
                    N_gu = gu_int8.shape[1]
                    num_blks_gu = triton.cdiv(N_gu, 128)
                    x_i8 = torch.empty((M_flat, H_dim), device=h_in.device, dtype=torch.int8)
                    sx = torch.empty((M_flat,), device=h_in.device, dtype=torch.float32)
                    fused_moe_gather_quant_fwd_kernel[(M_flat,)](
                        h_in, sorted_tok, x_i8, sx, active_tokens, h_in.stride(0), x_i8.stride(0), H_dim, BLOCK_K=triton.next_power_of_2(H_dim), num_warps=4,
                    )
                    if has_lora:
                        z_gu_tok = F.linear(h_in, la_gu)
                    else:
                        z_gu_tok = h_in
                    gu_i8 = torch.empty((M_flat, N_gu), device=h_in.device, dtype=torch.int8)
                    gu_blk_scale = torch.empty((M_flat, num_blks_gu), device=h_in.device, dtype=torch.float32)
                    grouped_int8_gemm_blkq_kernel[(MAX_BLOCKS, num_blks_gu)](
                        x_i8, sx, gu_int8, gu_scale, gu_i8, gu_blk_scale, b_exp, b_start, b_end, total_blocks,
                        z_gu_tok, lb_gu if has_lora else h_in, sorted_tok, lora_scale,
                        x_i8.stride(0), x_i8.stride(1), gu_int8.stride(0), gu_int8.stride(1), gu_int8.stride(2),
                        gu_scale.stride(0), gu_scale.stride(1), gu_i8.stride(0), gu_i8.stride(1), gu_blk_scale.stride(0), gu_blk_scale.stride(1),
                        z_gu_tok.stride(0), z_gu_tok.stride(1),
                        lb_gu.stride(0) if has_lora else 1, lb_gu.stride(1) if has_lora else 1, lb_gu.stride(2) if has_lora else 1,
                        N_gu, H_dim, HAS_LORA=has_lora, USE_SORTED_TOK=True, R_LORA=r_lora,
                        BLOCK_M=BLOCK_M, BLOCK_N=128, BLOCK_K=128, num_warps=n_warps_gemm, num_stages=3,
                    )
                    I_dim = N_gu // 2
                    N_dn = dn_int8.shape[1]
                    num_blks_dn = triton.cdiv(N_dn, 128)
                    a_i8 = torch.empty((M_flat, I_dim), device=h_in.device, dtype=torch.int8)
                    sa = torch.empty((M_flat,), device=h_in.device, dtype=torch.float32)
                    softcapped_swiglu_quant_fwd_blkq_topp_kernel[(M_flat,)](
                        gu_i8, gu_blk_scale, a_i8, sa, active_tokens, gu_i8.stride(0), gu_blk_scale.stride(0), a_i8.stride(0),
                        I_dim, BLOCK_I=triton.next_power_of_2(I_dim), num_warps=4,
                    )
                    if has_lora:
                        z_dn = torch.empty((M_flat, r_lora), device=h_in.device, dtype=h_in.dtype)
                        dequant_matmul_z_dn_kernel[(triton.cdiv(M_flat, 64),)](
                            a_i8, sa, la_dn, z_dn, active_tokens,
                            a_i8.stride(0), la_dn.stride(0), la_dn.stride(1), z_dn.stride(0),
                            I_dim, R_LORA=r_lora, BLOCK_M=64, BLOCK_K=128, num_warps=4,
                        )
                    else:
                        z_dn = h_in
                    dn_i8 = torch.empty((M_flat, N_dn), device=h_in.device, dtype=torch.int8)
                    dn_blk_scale = torch.empty((M_flat, num_blks_dn), device=h_in.device, dtype=torch.float32)
                    grouped_int8_gemm_blkq_kernel[(MAX_BLOCKS, num_blks_dn)](
                        a_i8, sa, dn_int8, dn_scale, dn_i8, dn_blk_scale, b_exp, b_start, b_end, total_blocks,
                        z_dn, lb_dn if has_lora else h_in, sorted_tok, lora_scale,
                        a_i8.stride(0), a_i8.stride(1), dn_int8.stride(0), dn_int8.stride(1), dn_int8.stride(2),
                        dn_scale.stride(0), dn_scale.stride(1), dn_i8.stride(0), dn_i8.stride(1), dn_blk_scale.stride(0), dn_blk_scale.stride(1),
                        z_dn.stride(0), z_dn.stride(1),
                        lb_dn.stride(0) if has_lora else 1, lb_dn.stride(1) if has_lora else 1, lb_dn.stride(2) if has_lora else 1,
                        N_dn, I_dim, HAS_LORA=has_lora, USE_SORTED_TOK=False, R_LORA=r_lora,
                        BLOCK_M=BLOCK_M, BLOCK_N=128, BLOCK_K=128, num_warps=n_warps_gemm, num_stages=3,
                    )
                    out = torch.empty_like(h_in)
                    moe_gather_reduce_fwd_blkq_topp_kernel[(N_tok, triton.cdiv(H_dim, 256))](
                        dn_i8, dn_blk_scale, inv_idx, t_w.reshape(-1), out, H_dim, num_blks_dn, TOP_K=top_k, BLOCK_H=256, num_warps=4,
                    )
                    if has_lora:
                        ctx.save_for_backward(t_w, gu_i8, gu_blk_scale, dn_i8, dn_blk_scale, h_in, a_i8, sa, z_gu_tok, z_dn, la_gu, lb_gu, la_dn, lb_dn)
                    else:
                        ctx.save_for_backward(t_w, gu_i8, gu_blk_scale, dn_i8, dn_blk_scale)
                    return out

                @staticmethod
                def backward(ctx, grad_out):
                    if has_lora:
                        t_w, gu_i8, gu_blk_scale, dn_i8, dn_blk_scale, h_in, a_i8, sa, z_gu_tok, z_dn, la_gu, lb_gu, la_dn, lb_dn = ctx.saved_tensors
                    else:
                        t_w, gu_i8, gu_blk_scale, dn_i8, dn_blk_scale = ctx.saved_tensors
                    num_blks_dn = dn_blk_scale.shape[1]
                    d_w = torch.empty_like(t_w)
                    moe_dw_blkq_topp_kernel[(N_tok, top_k)](
                        grad_out, dn_i8, dn_blk_scale, inv_idx, d_w, H_dim, num_blks_dn, TOP_K=top_k, BLOCK_H=512, num_warps=4,
                    )
                    E_d, N_d, K_d = dn_int8.shape
                    dy_i8 = torch.empty((M_flat, N_d), device=grad_out.device, dtype=torch.int8)
                    sy = torch.empty((M_flat,), device=grad_out.device, dtype=torch.float32)
                    fused_moe_gather_scale_quant_bwd_kernel[(M_flat,)](
                        grad_out, t_w.reshape(-1), dn_scale, sorted_tok, sorted_slot, sorted_e,
                        dy_i8, sy, active_tokens, grad_out.stride(0), dn_scale.stride(0), dy_i8.stride(0),
                        N_d, BLOCK_K=triton.next_power_of_2(N_d), num_warps=4,
                    )
                    d_act = torch.empty((M_flat, K_d), device=grad_out.device, dtype=grad_out.dtype)
                    grouped_int8_gemm_dx_direct_kernel[(MAX_BLOCKS, triton.cdiv(K_d, 128))](
                        dy_i8, sy, dn_int8, d_act, b_exp, b_start, b_end, total_blocks,
                        dy_i8.stride(0), dy_i8.stride(1), dn_int8.stride(0), dn_int8.stride(1), dn_int8.stride(2),
                        d_act.stride(0), d_act.stride(1), N_d, K_d, BLOCK_M=BLOCK_M, BLOCK_K=128, BLOCK_N=128, num_warps=n_warps_gemm, num_stages=3,
                    )
                    d_la_dn = d_lb_dn = d_la_gu = d_lb_gu = None
                    if has_lora:
                        d_lb_dn = torch.empty_like(lb_dn)
                        moe_lora_dB_dn_kernel[(num_experts_cur, triton.cdiv(N_d, 128))](
                            grad_out, t_w.reshape(-1), sorted_tok, sorted_slot, z_dn,
                            exp_offsets, counts, d_lb_dn, lora_scale,
                            grad_out.stride(0), z_dn.stride(0), d_lb_dn.stride(0), d_lb_dn.stride(1), d_lb_dn.stride(2),
                            N_d, R_LORA=r_lora, BLOCK_M=64, BLOCK_N=128, num_warps=4,
                        )
                        dz_dn = torch.zeros((M_flat, r_lora), device=grad_out.device, dtype=grad_out.dtype)
                        moe_lora_dz_dn_kernel[(MAX_BLOCKS,)](
                            grad_out, t_w.reshape(-1), sorted_tok, sorted_slot, lb_dn,
                            b_exp, b_start, b_end, total_blocks, dz_dn, lora_scale,
                            grad_out.stride(0), lb_dn.stride(0), lb_dn.stride(1), lb_dn.stride(2), dz_dn.stride(0),
                            N_d, R_LORA=r_lora, BLOCK_M=BLOCK_M, BLOCK_N=128, num_warps=4,
                        )
                        SPLIT_M = 32
                        part_a = torch.empty((SPLIT_M, r_lora, K_d), device=grad_out.device, dtype=torch.float32)
                        moe_lora_dA_dn_split_kernel[(SPLIT_M, triton.cdiv(K_d, 128))](
                            dz_dn, a_i8, sa, active_tokens, part_a,
                            dz_dn.stride(0), a_i8.stride(0), part_a.stride(0), part_a.stride(1),
                            K_d, SPLIT_M=SPLIT_M, R_LORA=r_lora, BLOCK_M=64, BLOCK_K=128, num_warps=4,
                        )
                        d_la_dn = part_a.sum(dim=0).to(la_dn.dtype)
                        d_act.addmm_(dz_dn, la_dn)

                    E_g, N_g, K_g = gu_int8.shape
                    I_dim = N_g // 2
                    dgu_i8 = torch.empty((M_flat, N_g), device=grad_out.device, dtype=torch.int8)
                    sgu = torch.empty((M_flat,), device=grad_out.device, dtype=torch.float32)
                    softcapped_swiglu_scaled_quant_bwd_blkq_topp_kernel[(M_flat,)](
                        gu_i8, gu_blk_scale, d_act, gu_scale, sorted_e, dgu_i8, sgu, active_tokens,
                        gu_i8.stride(0), gu_blk_scale.stride(0), d_act.stride(0), gu_scale.stride(0), dgu_i8.stride(0),
                        I_dim, BLOCK_I=triton.next_power_of_2(I_dim), num_warps=4,
                    )
                    d_x_sorted = torch.empty((M_flat, K_g), device=grad_out.device, dtype=grad_out.dtype)
                    grouped_int8_gemm_dx_direct_kernel[(MAX_BLOCKS, triton.cdiv(K_g, 128))](
                        dgu_i8, sgu, gu_int8, d_x_sorted, b_exp, b_start, b_end, total_blocks,
                        dgu_i8.stride(0), dgu_i8.stride(1), gu_int8.stride(0), gu_int8.stride(1), gu_int8.stride(2),
                        d_x_sorted.stride(0), d_x_sorted.stride(1), N_g, K_g, BLOCK_M=BLOCK_M, BLOCK_K=128, BLOCK_N=128, num_warps=n_warps_gemm, num_stages=3,
                    )
                    d_h = torch.empty((N_tok, H_dim), device=grad_out.device, dtype=grad_out.dtype)
                    moe_gather_reduce_bwd_h_topp_kernel[(N_tok, triton.cdiv(H_dim, 256))](
                        d_x_sorted, inv_idx, d_h, H_dim, TOP_K=top_k, BLOCK_H=256, num_warps=4,
                    )
                    if has_lora:
                        d_lb_gu = torch.empty_like(lb_gu)
                        moe_lora_dB_gu_kernel[(num_experts_cur, triton.cdiv(N_g, 128))](
                            dgu_i8, sgu, gu_scale, sorted_tok, z_gu_tok,
                            exp_offsets, counts, d_lb_gu, lora_scale,
                            dgu_i8.stride(0), gu_scale.stride(0), z_gu_tok.stride(0),
                            d_lb_gu.stride(0), d_lb_gu.stride(1), d_lb_gu.stride(2),
                            N_g, R_LORA=r_lora, BLOCK_M=64, BLOCK_N=128, num_warps=4,
                        )
                        dz_gu = torch.empty((M_flat, r_lora), device=grad_out.device, dtype=grad_out.dtype)
                        moe_lora_dz_gu_kernel[(MAX_BLOCKS,)](
                            dgu_i8, sgu, gu_scale, lb_gu,
                            b_exp, b_start, b_end, total_blocks, dz_gu, lora_scale,
                            dgu_i8.stride(0), gu_scale.stride(0),
                            lb_gu.stride(0), lb_gu.stride(1), lb_gu.stride(2), dz_gu.stride(0),
                            N_g, R_LORA=r_lora, BLOCK_M=BLOCK_M, BLOCK_N=128, num_warps=4,
                        )
                        dz_gu_tok = torch.empty((N_tok, r_lora), device=grad_out.device, dtype=grad_out.dtype)
                        moe_reduce_dz_gu_kernel[(N_tok,)](
                            dz_gu, inv_idx, dz_gu_tok, dz_gu.stride(0), dz_gu_tok.stride(0),
                            TOP_K=top_k, R_LORA=r_lora, num_warps=1,
                        )
                        d_la_gu = torch.matmul(dz_gu_tok.T, h_in)
                        d_h.addmm_(dz_gu_tok, la_gu)
                    return d_h, d_w, d_la_gu, d_lb_gu, d_la_dn, d_lb_dn

            return _ExpertsFn.apply(hidden_states, w_eff, self.lora_A_gu, self.lora_B_gu, self.lora_A_dn, self.lora_B_dn)

    # --- GATE 1A (Fused MoE-LoRA Accuracy vs exact PyTorch ref, non-zero A & B_e) + GATE 1C (1-expert merge) ---
    orig_exp_test = m.model.language_model.layers[6].mlp.experts
    test_moe = PatchedINT8Experts(orig_exp_test, delete_orig=False, enable_lora=True, r_lora=16, alpha_lora=32.0).to(device).train()
    with torch.no_grad():
        test_moe.lora_A_gu.normal_(0.0, 0.05)
        test_moe.lora_B_gu.normal_(0.0, 0.05)
        test_moe.lora_A_dn.normal_(0.0, 0.05)
        test_moe.lora_B_dn.normal_(0.0, 0.05)
    h_g1 = torch.randn(512, 2304, device=device, dtype=dtype, requires_grad=True)
    idx_g1 = torch.topk(torch.randn(512, 128, device=device), 8, dim=-1).indices
    w_g1 = (torch.softmax(torch.randn(512, 8, device=device, dtype=torch.float32), dim=-1) * 8.0).detach().requires_grad_(True)
    go_g1 = torch.randn(512, 2304, device=device, dtype=dtype)
    moe_cfg["top_p"] = 1.0
    moe_cfg["block_m"] = 64
    y_tri_g1 = test_moe(h_g1, idx_g1, w_g1)
    y_tri_g1.backward(go_g1)

    # Exact PyTorch reference with identical INT8-dequantized or BF16 base + non-zero LoRA
    h_ref = h_g1.detach().clone().requires_grad_(True)
    la_gu_r = test_moe.lora_A_gu.detach().clone().requires_grad_(True)
    lb_gu_r = test_moe.lora_B_gu.detach().clone().requires_grad_(True)
    la_dn_r = test_moe.lora_A_dn.detach().clone().requires_grad_(True)
    lb_dn_r = test_moe.lora_B_dn.detach().clone().requires_grad_(True)
    w_gu_base = orig_exp_test.gate_up_proj.data.float()
    w_dn_base = orig_exp_test.down_proj.data.float()
    y_ref = torch.zeros_like(h_ref, dtype=torch.float32)
    sc_r = test_moe.lora_scale
    for e_id in range(128):
        pos_t, pos_k = torch.where(idx_g1 == e_id)
        if pos_t.numel() == 0:
            continue
        x_e = h_ref[pos_t].float()
        gu_e = F.linear(x_e, w_gu_base[e_id]) + sc_r * F.linear(F.linear(x_e, la_gu_r.float()), lb_gu_r[e_id].float())
        g_e, u_e = gu_e.chunk(2, dim=-1)
        act_e = (4.0 * torch.tanh(g_e * 0.25)) * torch.sigmoid(g_e) * (25.0 * torch.tanh(u_e * 0.04))
        dn_e = F.linear(act_e, w_dn_base[e_id]) + sc_r * F.linear(F.linear(act_e, la_dn_r.float()), lb_dn_r[e_id].float())
        y_ref.index_add_(0, pos_t, dn_e * w_g1[pos_t, pos_k].unsqueeze(-1))
    y_ref.to(dtype).backward(go_g1)

    m_fwd = F.cosine_similarity(y_ref.flatten(), y_tri_g1.flatten().float(), dim=0).item()
    m_dx = F.cosine_similarity(h_ref.grad.flatten().float(), h_g1.grad.flatten().float(), dim=0).item()
    m_dA_gu = F.cosine_similarity(la_gu_r.grad.flatten().float(), test_moe.lora_A_gu.grad.flatten().float(), dim=0).item()
    m_dB_gu = F.cosine_similarity(lb_gu_r.grad.flatten().float(), test_moe.lora_B_gu.grad.flatten().float(), dim=0).item()
    m_dA_dn = F.cosine_similarity(la_dn_r.grad.flatten().float(), test_moe.lora_A_dn.grad.flatten().float(), dim=0).item()
    m_dB_dn = F.cosine_similarity(lb_dn_r.grad.flatten().float(), test_moe.lora_B_dn.grad.flatten().float(), dim=0).item()
    print(
        f"=== GATE 1A (FUSED MOE-LORA): fwd={m_fwd:.6f} | dx={m_dx:.6f} | "
        f"gu(dA={m_dA_gu:.6f}, dB={m_dB_gu:.6f}) | dn(dA={m_dA_dn:.6f}, dB={m_dB_dn:.6f}) ===",
        flush=True,
    )
    assert min(m_fwd, m_dx, m_dA_gu, m_dB_gu, m_dA_dn, m_dB_dn) >= 0.999, "Gate 1A Fused MoE-LoRA failed!"

    # Gate 1C (part 1: 1-expert merge math check)
    with torch.no_grad():
        x_m1 = torch.randn(32, 2304, device=device, dtype=dtype)
        w_gu_m0 = (w_gu_base[0] + sc_r * (test_moe.lora_B_gu[0].float() @ test_moe.lora_A_gu.float())).to(dtype)
        y_gu_ad = F.linear(x_m1, orig_exp_test.gate_up_proj.data[0]) + sc_r * F.linear(F.linear(x_m1, test_moe.lora_A_gu), test_moe.lora_B_gu[0])
        y_gu_mg = F.linear(x_m1, w_gu_m0)
        cos_exp_merge = F.cosine_similarity(y_gu_ad.flatten().float(), y_gu_mg.flatten().float(), dim=0).item()
    assert cos_exp_merge >= 0.9999, f"Gate 1C expert merge failed: {cos_exp_merge}"
    del test_moe, h_g1, idx_g1, w_g1, go_g1, y_tri_g1, h_ref, la_gu_r, lb_gu_r, la_dn_r, lb_dn_r, w_gu_base, w_dn_base, y_ref, x_m1, w_gu_m0, y_gu_ad, y_gu_mg
    torch.cuda.empty_cache()

    ink_mod.Inkling2LoopSlidingAttention.forward = patched_sliding_attn_forward
    ink_mod.Inkling2LoopFullAttention.forward = patched_full_attn_forward

    n_layers = len(m.model.language_model.layers)
    freeze_until = n_layers // 4  # 6
    for name, mod in list(m.named_modules()):
        if name.endswith(".mlp.experts") and hasattr(mod, "gate_up_proj"):
            m_layer = re.search(r"layers\.(\d+)\.", name)
            l_idx = int(m_layer.group(1)) if m_layer else 0
            parent_name, attr = name.rsplit(".", 1)
            setattr(m.get_submodule(parent_name), attr, PatchedINT8Experts(mod, delete_orig=True, enable_lora=(l_idx >= freeze_until), r_lora=16, alpha_lora=32.0))
    gc.collect()
    torch.cuda.empty_cache()

    layer23_ctx = {"idx": None}
    orig_l23_mlp_fwd = m.model.language_model.layers[23].mlp.forward

    def patched_l23_mlp_forward(hidden_states: torch.Tensor) -> torch.Tensor:
        idx = layer23_ctx["idx"]
        if idx is None or not m.training:
            return orig_l23_mlp_fwd(hidden_states)
        bsz, seqlen, hdim = hidden_states.shape
        h_flat = hidden_states.reshape(-1, hdim)
        h_sub = h_flat[idx].unsqueeze(0)
        y_sub = orig_l23_mlp_fwd(h_sub).squeeze(0)
        out = torch.zeros_like(h_flat)
        out.index_copy_(0, idx, y_sub)
        return out.reshape(bsz, seqlen, hdim)

    m.model.language_model.layers[23].mlp.forward = patched_l23_mlp_forward

    # =========================================================================
    # 5. FUSED TRITON DEPTH-ATTENTION (BlockAttnResidual & MessageAttnResidual)
    # =========================================================================
    @triton.jit
    def depth_attn_logits_kernel(
        Q_ptr, X_ptr, Logits_ptr,
        stride_qm, stride_qd, stride_xn, stride_xk, stride_xd, stride_ln, stride_lm, stride_lk,
        D_dim, scale, BLOCK_D: tl.constexpr,
    ):
        pid_n, pid_m, pid_k = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        q_base = Q_ptr + pid_m * stride_qm
        x_base = X_ptr + pid_n * stride_xn + pid_k * stride_xk
        acc = 0.0
        for d0 in range(0, D_dim, BLOCK_D):
            offs_d = d0 + tl.arange(0, BLOCK_D)
            mask_d = offs_d < D_dim
            acc += tl.sum(tl.load(q_base + offs_d * stride_qd, mask=mask_d, other=0.0).to(tl.float32) * tl.load(x_base + offs_d * stride_xd, mask=mask_d, other=0.0).to(tl.float32), axis=0)
        tl.store(Logits_ptr + pid_n * stride_ln + pid_m * stride_lm + pid_k * stride_lk, acc * scale)

    @triton.jit
    def depth_attn_softmax_v_kernel(
        Logits_ptr, X_ptr, P_ptr, Out_ptr,
        stride_ln, stride_lm, stride_lk, stride_xn, stride_xk, stride_xd, stride_on, stride_om, stride_od,
        K_dim, D_dim, MAX_K: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        pid_n, pid_m, pid_d = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        offs_k = tl.arange(0, MAX_K)
        mask_k = offs_k < K_dim
        logits = tl.load(Logits_ptr + pid_n * stride_ln + pid_m * stride_lm + offs_k * stride_lk, mask=mask_k, other=float("-inf")).to(tl.float32)
        exps = tl.where(mask_k, tl.exp(logits - tl.max(logits, axis=0)), 0.0)
        p_vec = exps / tl.sum(exps, axis=0)
        if pid_d == 0:
            tl.store(P_ptr + pid_n * stride_ln + pid_m * stride_lm + offs_k * stride_lk, p_vec, mask=mask_k)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D_dim
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        x_n_base = X_ptr + pid_n * stride_xn
        for k in range(K_dim):
            p_k = tl.sum(tl.where(offs_k == k, p_vec, 0.0), axis=0)
            acc += p_k * tl.load(x_n_base + k * stride_xk + offs_d * stride_xd, mask=mask_d, other=0.0).to(tl.float32)
        tl.store(Out_ptr + pid_n * stride_on + pid_m * stride_om + offs_d * stride_od, acc, mask=mask_d)

    @triton.jit
    def depth_attn_bwd_dval_kernel(
        Out_ptr, dOut_ptr, Dval_ptr,
        stride_on, stride_om, stride_od, stride_don, stride_dom, stride_dod,
        M_q, D_dim, BLOCK_D: tl.constexpr,
    ):
        pid_n, pid_m = tl.program_id(0), tl.program_id(1)
        o_base, do_base = Out_ptr + pid_n * stride_on + pid_m * stride_om, dOut_ptr + pid_n * stride_don + pid_m * stride_dom
        acc = 0.0
        for d0 in range(0, D_dim, BLOCK_D):
            offs_d = d0 + tl.arange(0, BLOCK_D)
            mask_d = offs_d < D_dim
            acc += tl.sum(tl.load(o_base + offs_d * stride_od, mask=mask_d, other=0.0).to(tl.float32) * tl.load(do_base + offs_d * stride_dod, mask=mask_d, other=0.0).to(tl.float32), axis=0)
        tl.store(Dval_ptr + pid_n * M_q + pid_m, acc)

    @triton.jit
    def depth_attn_bwd_ds_kernel(
        dOut_ptr, X_ptr, P_ptr, Dval_ptr, dS_ptr,
        stride_don, stride_dom, stride_dod, stride_xn, stride_xk, stride_xd, stride_ln, stride_lm, stride_lk,
        M_q, D_dim, scale, BLOCK_D: tl.constexpr,
    ):
        pid_n, pid_m, pid_k = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        do_base, x_base = dOut_ptr + pid_n * stride_don + pid_m * stride_dom, X_ptr + pid_n * stride_xn + pid_k * stride_xk
        dp = 0.0
        for d0 in range(0, D_dim, BLOCK_D):
            offs_d = d0 + tl.arange(0, BLOCK_D)
            mask_d = offs_d < D_dim
            dp += tl.sum(tl.load(do_base + offs_d * stride_dod, mask=mask_d, other=0.0).to(tl.float32) * tl.load(x_base + offs_d * stride_xd, mask=mask_d, other=0.0).to(tl.float32), axis=0)
        p_mk = tl.load(P_ptr + pid_n * stride_ln + pid_m * stride_lm + pid_k * stride_lk).to(tl.float32)
        d_val = tl.load(Dval_ptr + pid_n * M_q + pid_m).to(tl.float32)
        tl.store(dS_ptr + pid_n * stride_ln + pid_m * stride_lm + pid_k * stride_lk, p_mk * (dp - d_val) * scale)

    @triton.jit
    def depth_attn_bwd_dx_kernel(
        dOut_ptr, Q_ptr, P_ptr, dS_ptr, dX_ptr,
        stride_don, stride_dom, stride_dod, stride_qm, stride_qd, stride_ln, stride_lm, stride_lk, stride_xn, stride_xk, stride_xd,
        M_q, D_dim, BLOCK_D: tl.constexpr,
    ):
        pid_n, pid_k, pid_d = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D_dim
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        for m_i in range(M_q):
            p_mk = tl.load(P_ptr + pid_n * stride_ln + m_i * stride_lm + pid_k * stride_lk).to(tl.float32)
            ds_mk = tl.load(dS_ptr + pid_n * stride_ln + m_i * stride_lm + pid_k * stride_lk).to(tl.float32)
            acc += p_mk * tl.load(dOut_ptr + pid_n * stride_don + m_i * stride_dom + offs_d * stride_dod, mask=mask_d, other=0.0).to(tl.float32) + ds_mk * tl.load(Q_ptr + m_i * stride_qm + offs_d * stride_qd, mask=mask_d, other=0.0).to(tl.float32)
        tl.store(dX_ptr + pid_n * stride_xn + pid_k * stride_xk + offs_d * stride_xd, acc, mask=mask_d)

    class TritonDepthAttnFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q_2d, x_3d, scale):
            q_c, x_c = q_2d.contiguous(), x_3d.contiguous()
            M_q, D_dim = q_c.shape
            N_tok, K_dim, _ = x_c.shape
            logits = torch.empty((N_tok, M_q, K_dim), device=x_c.device, dtype=torch.float32)
            P_buf = torch.empty((N_tok, M_q, K_dim), device=x_c.device, dtype=torch.float32)
            out = torch.empty((N_tok, M_q, D_dim), device=x_c.device, dtype=torch.float32)
            depth_attn_logits_kernel[(N_tok, M_q, K_dim)](
                q_c, x_c, logits, q_c.stride(0), q_c.stride(1), x_c.stride(0), x_c.stride(1), x_c.stride(2),
                logits.stride(0), logits.stride(1), logits.stride(2), D_dim, scale, BLOCK_D=512, num_warps=4,
            )
            depth_attn_softmax_v_kernel[(N_tok, M_q, triton.cdiv(D_dim, 256))](
                logits, x_c, P_buf, out, logits.stride(0), logits.stride(1), logits.stride(2),
                x_c.stride(0), x_c.stride(1), x_c.stride(2), out.stride(0), out.stride(1), out.stride(2),
                K_dim, D_dim, MAX_K=max(8, triton.next_power_of_2(K_dim)), BLOCK_D=256, num_warps=4,
            )
            ctx.save_for_backward(q_c, x_c, P_buf, out)
            ctx.scale = scale
            return out

        @staticmethod
        def backward(ctx, grad_out):
            q_c, x_c, P_buf, out = ctx.saved_tensors
            do_c = grad_out.contiguous()
            M_q, D_dim = q_c.shape
            N_tok, K_dim, _ = x_c.shape
            Dval = torch.empty((N_tok, M_q), device=x_c.device, dtype=torch.float32)
            dS = torch.empty((N_tok, M_q, K_dim), device=x_c.device, dtype=torch.float32)
            dX = torch.empty_like(x_c)
            depth_attn_bwd_dval_kernel[(N_tok, M_q)](out, do_c, Dval, out.stride(0), out.stride(1), out.stride(2), do_c.stride(0), do_c.stride(1), do_c.stride(2), M_q, D_dim, BLOCK_D=512, num_warps=4)
            depth_attn_bwd_ds_kernel[(N_tok, M_q, K_dim)](do_c, x_c, P_buf, Dval, dS, do_c.stride(0), do_c.stride(1), do_c.stride(2), x_c.stride(0), x_c.stride(1), x_c.stride(2), P_buf.stride(0), P_buf.stride(1), P_buf.stride(2), M_q, D_dim, ctx.scale, BLOCK_D=512, num_warps=4)
            depth_attn_bwd_dx_kernel[(N_tok, K_dim, triton.cdiv(D_dim, 256))](do_c, q_c, P_buf, dS, dX, do_c.stride(0), do_c.stride(1), do_c.stride(2), q_c.stride(0), q_c.stride(1), P_buf.stride(0), P_buf.stride(1), P_buf.stride(2), dX.stride(0), dX.stride(1), dX.stride(2), M_q, D_dim, BLOCK_D=256, num_warps=4)
            return None, dX, None

    def patched_block_attn_res_forward(self, block_states, partial_block_states):
        stacked = self.norm(torch.cat([block_states, partial_block_states.unsqueeze(-2)], dim=-2).float())
        orig_prefix, K_dim, hdim = stacked.shape[:-2], stacked.shape[-2], stacked.shape[-1]
        q_2d = self.query.float().view(-1, hdim)
        out = TritonDepthAttnFn.apply(q_2d, stacked.reshape(-1, K_dim, hdim), hdim**-0.5)
        return out.reshape(*orig_prefix, q_2d.shape[0], hdim).unbind(-2)

    def patched_message_attn_res_forward(self, block_states, partial_block_states, loop_layer_idx):
        stacked = self.norm(torch.cat([block_states, partial_block_states.unsqueeze(-2)], dim=-2).float())
        orig_prefix, K_dim, hdim = stacked.shape[:-2], stacked.shape[-2], stacked.shape[-1]
        q_2d = self.query[loop_layer_idx].float()
        out = TritonDepthAttnFn.apply(q_2d, stacked.reshape(-1, K_dim, hdim), hdim**-0.5)
        return out.reshape(*orig_prefix, q_2d.shape[0], hdim)

    ink_mod.Inkling2LoopBlockAttentionResidual.forward = patched_block_attn_res_forward
    ink_mod.Inkling2LoopMessageAttentionResidual.forward = patched_message_attn_res_forward

    # =========================================================================
    # 6. TRUNCATED BPTT + 30-PASS CUDA GRAPH PREFIX
    # =========================================================================
    tbptt_cfg = {"grad_passes": 3}
    cg_cfg = {"enabled": False, "cache": {}, "verified": False}

    def run_30pass_nograd_prefix(self, in_embeds, doc_id_t, position_ids):
        attn_ctx["doc_id"] = doc_id_t
        loop_start, loop_end = self.config.loop_layer_index_start, self.config.loop_layer_index_end
        with torch.no_grad():
            hidden_states = in_embeds
            block_states = in_embeds.new_zeros(*in_embeds.shape[:-1], 0, in_embeds.shape[-1])
            for decoder_layer in self.layers[:loop_start]:
                new_h = decoder_layer(hidden_states, block_states, position_ids=position_ids)
                if decoder_layer.is_block_boundary:
                    block_states = torch.cat([block_states, hidden_states.unsqueeze(-2)], dim=-2)
                hidden_states = new_h
            pre_h, pre_b = hidden_states, block_states
            num_pre_blocks = pre_b.shape[-2]

            # Pass 0 (no_grad)
            for decoder_layer in self.layers[loop_start:loop_end]:
                new_h = decoder_layer(hidden_states, block_states, position_ids=position_ids)
                if decoder_layer.is_block_boundary:
                    block_states = torch.cat([block_states, hidden_states.unsqueeze(-2)], dim=-2)
                hidden_states = new_h

            # Pass 1 (no_grad)
            prev_b = block_states[..., num_pre_blocks + 1 :, :]
            prev_h = hidden_states
            hidden_states, block_states = pre_h, pre_b
            cache_offset = self.config.num_hidden_layers - loop_start
            for loop_layer_idx, decoder_layer in enumerate(self.layers[loop_start:loop_end]):
                messages = self.message_input(prev_b, prev_h, loop_layer_idx)
                new_h = decoder_layer(hidden_states, block_states, messages, cache_offset=cache_offset, position_ids=position_ids)
                if decoder_layer.is_block_boundary:
                    block_states = torch.cat([block_states, hidden_states.unsqueeze(-2)], dim=-2)
                hidden_states = new_h

            out_prev_b = block_states[..., num_pre_blocks + 1 :, :].contiguous()
            out_prev_h = hidden_states.contiguous()
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
        num_passes = self.config.loop_num_passes
        grad_start_pass = (num_passes - tbptt_cfg["grad_passes"]) if self.training else num_passes

        # Fast path: CUDA Graph for 30-pass no_grad prefix in T-BPTT 1/3
        if self.training and grad_start_pass == 2 and cg_cfg["enabled"]:
            cg_key = (B_cur, S_cur, moe_cfg["top_p"], moe_cfg["k_min"], moe_cfg["block_m"])
            saved_gc = [l.gradient_checkpointing for l in self.layers[loop_start:loop_end]]
            if cg_key not in cg_cfg["cache"]:
                for l_obj in self.layers[loop_start:loop_end]:
                    l_obj.gradient_checkpointing = False
                old_rec = moe_cfg["record_stats"]
                moe_cfg["record_stats"] = False
                static_embeds = inputs_embeds.detach().clone()
                static_doc = attn_ctx["doc_id"].clone() if attn_ctx["doc_id"] is not None else torch.zeros((B_cur, S_cur), device=dev_cur, dtype=torch.int16)
                static_pos = position_ids.clone()
                # 2 warmup runs in side stream
                s_cap = torch.cuda.Stream()
                s_cap.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s_cap):
                    ref_outs = run_30pass_nograd_prefix(self, static_embeds, static_doc, static_pos)
                    _ = run_30pass_nograd_prefix(self, static_embeds, static_doc, static_pos)
                torch.cuda.current_stream().wait_stream(s_cap)
                g_cap = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g_cap, stream=s_cap):
                    static_outs = run_30pass_nograd_prefix(self, static_embeds, static_doc, static_pos)
                torch.cuda.current_stream().wait_stream(s_cap)
                g_cap.replay()
                torch.cuda.synchronize()
                if not cg_cfg["verified"]:
                    cg_cos = F.cosine_similarity(ref_outs[3].flatten().float(), static_outs[3].flatten().float(), dim=0).item()
                    print(f"=== GATE 1C (30-PASS CUDAGRAPH PREFIX): cos={cg_cos:.6f} (key={cg_key}) ===", flush=True)
                    assert cg_cos >= 0.99999, f"CUDA Graph prefix mismatch: {cg_cos}"
                    cg_cfg["verified"] = True
                cg_cfg["cache"][cg_key] = (g_cap, static_embeds, static_doc, static_outs)
                moe_cfg["record_stats"] = old_rec
                for idx_l, l_obj in enumerate(self.layers[loop_start:loop_end]):
                    l_obj.gradient_checkpointing = saved_gc[idx_l]

            g_cap, static_embeds, static_doc, static_outs = cg_cfg["cache"][cg_key]
            cur_doc = attn_ctx["doc_id"]
            static_embeds.copy_(inputs_embeds.detach())
            if cur_doc is not None:
                static_doc.copy_(cur_doc)
            g_cap.replay()
            attn_ctx["doc_id"] = cur_doc
            pre_h, pre_b, prev_b, prev_h = static_outs
            hidden_states = pre_h.detach()
            block_states = pre_b.detach()
            previous_block_states = prev_b.detach()
            previous_hidden_states = prev_h.detach()

            # Pass 2 (the single gradient pass!)
            cache_offset = self.config.num_hidden_layers + num_loop_layers - loop_start
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

        # Standard path (Exact 3/3, Fast-Safe 2/3, or eval)
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

    # =========================================================================
    # 7. FastDoRALinear (layers 6..23) + Unfrozen RMSNorm + Gate 1B/1C + Optimizers
    # =========================================================================
    class FastDoRALinear(nn.Module):
        def __init__(self, base_layer, r=32, alpha=64):
            super().__init__()
            in_f, out_f = base_layer.in_features, base_layer.out_features
            self.base = base_layer
            self.lora_A = nn.Parameter(torch.randn(r, in_f, device=device, dtype=dtype) * 0.01)
            self.lora_B = nn.Parameter(torch.zeros(out_f, r, device=device, dtype=dtype))
            self.scale = float(alpha) / float(r)
            w0_f = base_layer.weight.data.float()
            w0_nsq = w0_f.pow(2).sum(dim=1)
            w0_n = torch.sqrt(w0_nsq.clamp(min=1e-8))
            self.m = nn.Parameter(w0_n.to(dtype))
            self.register_buffer("w0_norm_sq", w0_nsq)
            self.register_buffer("cached_inv_norm", (1.0 / w0_n).to(dtype))

        @torch.no_grad()
        def refresh_norm(self):
            A_f = self.lora_A.float()
            B_f = self.lora_B.float()
            AAt = A_f @ A_f.T
            W0At = F.linear(A_f, self.base.weight.data.float()).T
            norm_sq = (
                self.w0_norm_sq
                + 2.0 * self.scale * (B_f * W0At).sum(dim=1)
                + (self.scale ** 2) * (B_f * (B_f @ AAt)).sum(dim=1)
            )
            row_norm = torch.sqrt(norm_sq.clamp(min=1e-8))
            self.cached_inv_norm.copy_((1.0 / row_norm).to(self.m.dtype))

        def forward(self, x):
            y = self.base(x) + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scale
            return y * (self.m * self.cached_inv_norm)

    # --- GATE 1B (FastDoRALinear vs naive materialized DoRA) + GATE 1C (1-layer DoRA merge) ---
    base_lin_t = nn.Linear(2304, 2304, bias=False, device=device, dtype=dtype)
    dora_t = FastDoRALinear(base_lin_t, r=32, alpha=64).to(device)
    with torch.no_grad():
        dora_t.lora_A.normal_(0.0, 0.05)
        dora_t.lora_B.normal_(0.0, 0.05)
        dora_t.m.mul_(1.0 + 0.05 * torch.randn_like(dora_t.m))
    dora_t.refresh_norm()
    x_d1 = torch.randn(1, 256, 2304, device=device, dtype=dtype, requires_grad=True)
    go_d1 = torch.randn_like(x_d1)
    y_fast = dora_t(x_d1)
    y_fast.backward(go_d1)
    x_d2 = x_d1.detach().clone().requires_grad_(True)
    W_eff_n = base_lin_t.weight.data.float() + dora_t.scale * (dora_t.lora_B.float() @ dora_t.lora_A.float())
    row_n_n = torch.sqrt(W_eff_n.pow(2).sum(dim=1).clamp(min=1e-8)).detach()
    y_naive = F.linear(x_d2.float(), W_eff_n) * (dora_t.m.float() / row_n_n)
    y_naive.to(dtype).backward(go_d1)
    dora_fwd_cos = F.cosine_similarity(y_naive.flatten(), y_fast.flatten().float(), dim=0).item()
    dora_dx_cos = F.cosine_similarity(x_d2.grad.flatten().float(), x_d1.grad.flatten().float(), dim=0).item()
    with torch.no_grad():
        W_merged_dora = (W_eff_n * (dora_t.m.float() / row_n_n)[:, None]).to(dtype)
        y_merged_dora = F.linear(x_d1.detach(), W_merged_dora)
        cos_dora_merge = F.cosine_similarity(y_fast.detach().flatten().float(), y_merged_dora.flatten().float(), dim=0).item()
    print(f"=== GATE 1B (FASTDORA): fwd_cos={dora_fwd_cos:.6f} | bwd_dx_cos={dora_dx_cos:.6f} ===", flush=True)
    print(f"=== GATE 1C (1-LAYER MERGE CHECK): dora_merge_cos={cos_dora_merge:.6f} | moe_expert_merge_cos={cos_exp_merge:.6f} ===", flush=True)
    assert dora_fwd_cos >= 0.9999 and dora_dx_cos >= 0.9999, "Gate 1B FastDoRA failed!"
    assert cos_dora_merge >= 0.9999 and cos_exp_merge >= 0.9999, "Gate 1C merge check failed!"
    del base_lin_t, dora_t, x_d1, x_d2, go_d1, y_fast, W_eff_n, row_n_n, y_naive, W_merged_dora, y_merged_dora
    torch.cuda.empty_cache()

    # Freeze all, then attach 90 FastDoRALinear + unfreeze MoE-LoRA (layers 6..23) + unfreeze RMSNorm (layers 6..23 + final norm)
    for p in m.parameters():
        p.requires_grad_(False)

    dora_modules = []
    for name, mod in list(m.named_modules()):
        if isinstance(mod, nn.Linear) and "lm_head" not in name and mod.in_features >= 512 and mod.out_features >= 512:
            m_layer = re.search(r"layers\.(\d+)\.", name)
            if m_layer and int(m_layer.group(1)) < freeze_until:
                continue
            parent_name, attr = name.rsplit(".", 1)
            dmod = FastDoRALinear(mod, r=32, alpha=64)
            setattr(m.get_submodule(parent_name), attr, dmod)
            dora_modules.append(dmod)

    # Unfreeze MoE-LoRA params (layers 6..23) & RMSNorm weights (layers 6..23 + final norm)
    for name, mod in m.named_modules():
        if isinstance(mod, PatchedINT8Experts) and mod.enable_lora:
            mod.lora_A_gu.requires_grad_(True)
            mod.lora_B_gu.requires_grad_(True)
            mod.lora_A_dn.requires_grad_(True)
            mod.lora_B_dn.requires_grad_(True)
        elif isinstance(mod, ink_mod.Inkling2LoopRMSNorm) and hasattr(mod, "weight") and mod.weight is not None:
            m_layer = re.search(r"layers\.(\d+)\.", name)
            if (m_layer and int(m_layer.group(1)) >= freeze_until) or name.endswith("language_model.norm"):
                mod.weight.requires_grad_(True)

    # SAFETY GUARD #2: Assert ALL routers (mlp.gate) are strictly frozen!
    for name, p in m.named_parameters():
        if ".mlp.gate." in name:
            assert not p.requires_grad, f"Router {name} must stay frozen!"

    compile_targets = [
        ink_mod.Inkling2LoopRMSNorm,
        ink_mod.Inkling2LoopUnweightedRMSNorm,
        ink_mod.Inkling2LoopMessageGate,
        ink_mod.Inkling2LoopL2Clip,
        ink_mod.Inkling2LoopScaledL2Clip,
        ink_mod.Inkling2LoopLogScaledL2Clip,
        ink_mod.Inkling2LoopTopKRouter,
        ink_mod.Inkling2LoopQueryProjection,
        ink_mod.Inkling2LoopGatedOutputProjection,
        ink_mod.Inkling2LoopRelativePositionBias,
        FastDoRALinear,
    ]
    orig_uncompiled_forwards = {cls_obj: cls_obj.forward for cls_obj in compile_targets}
    compiled_forwards = {}
    torch._dynamo.config.cache_size_limit = 64
    for cls_obj in compile_targets:
        try:
            cfn = torch.compile(cls_obj.forward)
            compiled_forwards[cls_obj] = cfn
            cls_obj.forward = cfn
        except Exception:
            pass

    def _set_compiled_mode(enable: bool):
        for cls_obj in compile_targets:
            cls_obj.forward = compiled_forwards[cls_obj] if (enable and cls_obj in compiled_forwards) else orig_uncompiled_forwards[cls_obj]

    m.config.use_cache = False
    m.model.language_model.config.use_cache = False
    gc_fn = partial(torch.utils.checkpoint.checkpoint, use_reentrant=False)
    for layer in m.model.language_model.layers:
        layer._gradient_checkpointing_func = gc_fn
        layer.gradient_checkpointing = False

    def set_gc_mode(mode):
        for idx, layer in enumerate(m.model.language_model.layers):
            if mode == "OFF":
                layer.gradient_checkpointing = False
            elif mode == "Selective(1/3)":
                layer.gradient_checkpointing = (idx >= freeze_until) and ((idx - freeze_until) % 3 == 0)
            else:
                layer.gradient_checkpointing = (idx >= freeze_until)

    named_params = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
    params = [p for _, p in named_params]

    dense_dora_ab = [(n, p) for n, p in named_params if n.endswith(".lora_A") or n.endswith(".lora_B")]
    dense_dora_m = [(n, p) for n, p in named_params if n.endswith(".m")]
    moe_lora_a = [(n, p) for n, p in named_params if n.endswith(".lora_A_gu") or n.endswith(".lora_A_dn")]
    moe_lora_b = [(n, p) for n, p in named_params if n.endswith(".lora_B_gu") or n.endswith(".lora_B_dn")]
    rmsnorm_named = [(n, p) for n, p in named_params if n not in {k for k, _ in dense_dora_ab + dense_dora_m + moe_lora_a + moe_lora_b}]

    n_dense_p = sum(p.numel() for _, p in dense_dora_ab + dense_dora_m)
    n_moe_p = sum(p.numel() for _, p in moe_lora_a + moe_lora_b)
    n_norm_p = sum(p.numel() for _, p in rmsnorm_named)
    print(
        f"=== NEAR-FULL-FT PARAMS: total={sum(p.numel() for p in params)/1e6:.2f}M "
        f"(Dense DoRA={n_dense_p/1e6:.2f}M [{len(dense_dora_ab)+len(dense_dora_m)}], "
        f"MoE-LoRA={n_moe_p/1e6:.2f}M [{len(moe_lora_a)+len(moe_lora_b)}], "
        f"RMSNorm={n_norm_p/1e6:.2f}M [{len(rmsnorm_named)}]) | Routers=FROZEN ===",
        flush=True,
    )

    def batched_ns5(G, steps=5):
        a, b, c = (3.4445, -4.7750, 2.0315)
        X = G.bfloat16()
        trans = X.size(1) > X.size(2)
        if trans:
            X = X.transpose(1, 2)
        norms = X.flatten(1).norm(dim=1, keepdim=True).unsqueeze(-1) + 1e-7
        X = X / norms
        for _ in range(steps):
            A = torch.bmm(X, X.transpose(1, 2))
            B_mat = b * A + c * torch.bmm(A, A)
            X = a * X + torch.bmm(B_mat, X)
        if trans:
            X = X.transpose(1, 2)
        return X

    # SAFETY GUARD #1: ONLY Dense FastDoRA (A, B) in Muon; MoE-LoRA + DoRA.m + RMSNorm in fused AdamW!
    param_groups = {}
    for pname, p in dense_dora_ab:
        is_b = pname.endswith(".lora_B")
        param_groups.setdefault((tuple(p.shape), is_b), []).append(p)
    muon_groups = [
        (plist, torch.zeros((len(plist), *shape), device=device, dtype=dtype), is_b)
        for (shape, is_b), plist in param_groups.items() if len(shape) == 2
    ]

    adamw_opt = torch.optim.AdamW(
        [
            {"params": [p for _, p in moe_lora_a + moe_lora_b], "lr": lr_moe, "weight_decay": 0.01},
            {"params": [p for _, p in dense_dora_m + rmsnorm_named], "lr": lr_norm, "weight_decay": 0.0},
        ],
        fused=True,
    )

    @torch.no_grad()
    def cautious_muon_step(lr_a_cur, lr_b_cur, wd=0.01):
        for plist, M_buf, is_b in muon_groups:
            lr_cur = lr_b_cur if is_b else lr_a_cur
            G_stack = torch.stack([p.grad if p.grad is not None else torch.zeros_like(p) for p in plist])
            M_buf.mul_(0.95).add_(G_stack)
            U = batched_ns5(M_buf, steps=5)
            rms_scale = 0.2 * math.sqrt(max(U.size(1), U.size(2)))
            mask = (U * G_stack) > 0
            U.mul_(mask.to(U.dtype))
            eff_lr = lr_cur * rms_scale
            for i, p in enumerate(plist):
                if wd > 0.0:
                    p.mul_(1.0 - lr_cur * wd)
                p.add_(U[i], alpha=-eff_lr)
                p.grad = None

    @torch.no_grad()
    def refresh_all_dora_norms():
        for dmod in dora_modules:
            dmod.refresh_norm()

    # =========================================================================
    # 8. STREAMING CPU PREFETCHER (pin_memory + non_blocking + fresh Log-Q 16k)
    # =========================================================================
    V_FULL = m.lm_head.weight.shape[0]
    HEAD_K = 12288
    SUB_V = 16384
    HEAD_TOK_CPU = torch.arange(HEAD_K, dtype=torch.long)
    EOS_ID = int(tokenizer.eos_token_id)
    EOS_TOK_CPU = torch.tensor([EOS_ID], dtype=torch.long)
    hidden_size = m.config.get_text_config().hidden_size

    def build_cpu_microbatch(batch_idx, gen, verify_assert=False):
        b_ids = train_ids_all[batch_idx].unsqueeze(0).to(torch.long)
        b_lbl = train_lbl_all[batch_idx].unsqueeze(0).to(torch.long)
        b_doc = train_doc_all[batch_idx].unsqueeze(0)
        lbl_shift = b_lbl[:, 1:].reshape(-1)
        valid_idx = torch.nonzero(lbl_shift != -100, as_tuple=False).squeeze(-1)
        valid_lbl = lbl_shift[valid_idx]

        l23_mask = torch.zeros((1, SEQ_LEN), dtype=torch.bool)
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
            log_q_corr = torch.where(
                is_tail,
                torch.full_like(sub_vocab, corr_val, dtype=torch.float32),
                torch.zeros_like(sub_vocab, dtype=torch.float32),
            )
        else:
            sub_vocab = head_union[:SUB_V]
            log_q_corr = torch.zeros((SUB_V,), dtype=torch.float32)

        if verify_assert:
            eos_pos = int(torch.searchsorted(sub_vocab, EOS_ID).item())
            assert sub_vocab[eos_pos].item() == EOS_ID and log_q_corr[eos_pos].item() == 0.0, "EOS log_q_corr != 0!"
            assert (log_q_corr[torch.searchsorted(sub_vocab, uniq_toks)] == 0.0).all().item(), "uniq_toks log_q_corr != 0!"
            assert (log_q_corr[torch.searchsorted(sub_vocab, HEAD_TOK_CPU)] == 0.0).all().item(), "HEAD_TOK log_q_corr != 0!"

        remap_targets = torch.searchsorted(sub_vocab, valid_lbl)
        return (
            b_ids.pin_memory(), b_doc.pin_memory(), valid_idx.pin_memory(), l23_idx.pin_memory(),
            sub_vocab.pin_memory(), remap_targets.pin_memory(), log_q_corr.pin_memory(),
        )

    def start_prefetch_stream(start_micro, total_micros):
        q = queue.Queue(maxsize=4)
        stop_ev = threading.Event()

        def _worker():
            try:
                cur_ep = -1
                perm = None
                for m_idx in range(start_micro, total_micros):
                    if stop_ev.is_set():
                        break
                    ep = m_idx // n_train_batches
                    pos = m_idx % n_train_batches
                    if ep != cur_ep:
                        cur_ep = ep
                        ep_gen = torch.Generator().manual_seed(seed + ep * 1009)
                        perm = torch.randperm(n_train_batches, generator=ep_gen).tolist()
                    batch_id = perm[pos]
                    neg_gen = torch.Generator().manual_seed(seed + ep * 100003 + m_idx)
                    item = build_cpu_microbatch(batch_id, neg_gen, verify_assert=(m_idx < start_micro + 2))
                    q.put((ep, item))
                q.put(None)
            except Exception as exc:
                q.put(exc)

        th = threading.Thread(target=_worker, daemon=True)
        th.start()
        return q, stop_ev, th

    def compute_pruned_loss_16k(cpu_batch):
        b_ids, b_doc, valid_idx, l23_idx, sub_vocab, remap_targets, log_q_corr = [
            t.to(device, non_blocking=True) for t in cpu_batch
        ]
        attn_ctx["doc_id"] = b_doc
        layer23_ctx["idx"] = l23_idx
        h_out = m.model(input_ids=b_ids).last_hidden_state[:, :-1, :].reshape(-1, hidden_size)
        h_valid = h_out[valid_idx]
        sub_logits = F.linear(h_valid, m.lm_head.weight[sub_vocab]).float()
        sub_logits = m.logit_softcap(sub_logits) + log_q_corr[None, :]
        return F.cross_entropy(sub_logits, remap_targets)

    @torch.no_grad()
    def eval_full_vocab_loss_all(max_batches=None):
        was_training = m.training
        old_top_p, old_gp, old_cg = moe_cfg["top_p"], tbptt_cfg["grad_passes"], cg_cfg["enabled"]
        refresh_all_dora_norms()
        m.eval()
        moe_cfg["top_p"] = 1.0
        tbptt_cfg["grad_passes"] = 3
        cg_cfg["enabled"] = False
        layer23_ctx["idx"] = None
        total_loss, total_valid = 0.0, 0
        n_eval = n_val_batches if max_batches is None else min(n_val_batches, max_batches)
        for vi in range(n_eval):
            ids_b = val_ids_all[vi].unsqueeze(0).to(device=device, dtype=torch.long)
            lbl_b = val_lbl_all[vi].unsqueeze(0).to(device=device, dtype=torch.long)
            doc_b = val_doc_all[vi].unsqueeze(0).to(device=device)
            attn_ctx["doc_id"] = doc_b
            h_out = m.model(input_ids=ids_b).last_hidden_state[:, :-1, :].reshape(-1, hidden_size)
            lbl_shift = lbl_b[:, 1:].reshape(-1)
            for i in range(0, h_out.shape[0], 512):
                lc = lbl_shift[i : i + 512]
                v_cnt = (lc != -100).sum().item()
                if v_cnt == 0:
                    continue
                logits_c = m.logit_softcap(m.lm_head(h_out[i : i + 512]).float())
                total_loss += F.cross_entropy(logits_c, lc, ignore_index=-100, reduction="sum").item()
                total_valid += v_cnt
        moe_cfg["top_p"], tbptt_cfg["grad_passes"], cg_cfg["enabled"] = old_top_p, old_gp, old_cg
        if was_training:
            m.train()
        return total_loss / max(1, total_valid)

    @torch.no_grad()
    def generate_3_answers_in_vram(tag, max_new_tokens=96):
        was_training = m.training
        old_top_p, old_gp, old_cg = moe_cfg["top_p"], tbptt_cfg["grad_passes"], cg_cfg["enabled"]
        refresh_all_dora_norms()
        _set_compiled_mode(False)
        m.eval()
        moe_cfg["top_p"] = 1.0
        tbptt_cfg["grad_passes"] = 3
        cg_cfg["enabled"] = False
        layer23_ctx["idx"] = None
        print(f"\n=== GENERATED ANSWERS [{tag}] (max_new_tokens={max_new_tokens}) ===", flush=True)
        for q_idx, prompt in enumerate(CONTROL_PROMPTS, 1):
            inp_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
            prompt_len = inp_ids.shape[1]
            cur_ids = inp_ids
            stopped_eos = False
            for _ in range(max_new_tokens):
                attn_ctx["doc_id"] = torch.zeros((1, cur_ids.shape[1]), device=device, dtype=torch.int16)
                h_last = m.model(input_ids=cur_ids).last_hidden_state[:, -1:, :]
                logits_last = m.logit_softcap(m.lm_head(h_last).float())
                next_tok = torch.argmax(logits_last[:, -1, :], dim=-1, keepdim=True)
                cur_ids = torch.cat([cur_ids, next_tok], dim=1)
                if next_tok.item() == tokenizer.eos_token_id:
                    stopped_eos = True
                    break
            gen_toks = cur_ids.shape[1] - prompt_len
            ans_text = tokenizer.decode(cur_ids[0, prompt_len:], skip_special_tokens=True)
            print(f"--- [{tag}] Q{q_idx} (stopped_by_EOS={stopped_eos}, gen_tokens={gen_toks}) ---\n{ans_text}\n", flush=True)
        moe_cfg["top_p"], tbptt_cfg["grad_passes"], cg_cfg["enabled"] = old_top_p, old_gp, old_cg
        _set_compiled_mode(True)
        if was_training:
            m.train()

    def save_checkpoint(path, opt_step, epoch, best_eval):
        ckpt = {
            "lora_state": {name: p.detach().cpu() for name, p in named_params},
            "muon_state": [M_buf.detach().cpu() for _, M_buf, _ in muon_groups],
            "opt_step": opt_step,
            "epoch": epoch,
            "best_eval_loss": best_eval,
            "lora_r": 32,
            "lora_alpha": 64,
            "freeze_until": freeze_until,
        }
        torch.save(ckpt, path)
        vol.commit()

    # =========================================================================
    # 9. PRODUCTION / 2-MINUTE NEAR-FULL-FT LOOP (Fast + Cooldown + In-VRAM Eval)
    # =========================================================================
    if time_limit_s > 0.0:
        total_opt_steps = 100000
        est_total_opt_steps = max(20, int(time_limit_s / 2.6))
        warmup_steps = 4
    else:
        total_opt_steps = max_steps if max_steps > 0 else max(1, (n_train_batches * epochs) // grad_accum)
        est_total_opt_steps = total_opt_steps
        warmup_steps = max(1, int(0.05 * total_opt_steps))
    total_micro_steps = total_opt_steps * grad_accum

    def get_lr_scale(opt_step, elapsed_s=0.0):
        if opt_step < warmup_steps:
            return float(opt_step + 1) / float(warmup_steps)
        if time_limit_s > 0.0:
            t_warm = min(15.0, time_limit_s * 0.1)
            prog = max(0.0, min(1.0, (elapsed_s - t_warm) / max(1.0, time_limit_s - t_warm)))
        else:
            prog = float(opt_step - warmup_steps) / float(max(1, est_total_opt_steps - warmup_steps))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    latest_path = os.path.join(output_dir, "latest_lora.pt")
    best_path = os.path.join(output_dir, "best_lora.pt")
    start_opt_step = 0
    best_eval_loss = float("inf")

    if resume and os.path.exists(latest_path):
        ckpt = torch.load(latest_path, map_location="cpu", weights_only=True)
        with torch.no_grad():
            for name, p in named_params:
                if name in ckpt["lora_state"]:
                    p.copy_(ckpt["lora_state"][name].to(device))
            for (_, M_buf, _), m_saved in zip(muon_groups, ckpt["muon_state"]):
                M_buf.copy_(m_saved.to(device))
        start_opt_step = int(ckpt["opt_step"])
        best_eval_loss = float(ckpt["best_eval_loss"])
        print(f"=== RESUMED FROM {latest_path}: opt_step={start_opt_step} | best_eval={best_eval_loss:.4f} ===", flush=True)

    # Configure Phase 1 (Fast): T-BPTT 1/3, Top-P 0.85, k_min=2, 30-pass CUDAGraph=ON, GC=OFF
    set_gc_mode("OFF")
    tbptt_cfg["grad_passes"] = 1
    moe_cfg["top_p"] = 0.85
    moe_cfg["k_min"] = 2
    moe_cfg["block_m"] = 64
    cg_cfg["enabled"] = True
    m.train()
    gc.collect()
    torch.cuda.empty_cache()

    # Warmup 2 micro-steps (compile + 30-pass CUDAGraph capture), then restore pre-warmup weights
    saved_pre_warm = [p.detach().cpu().clone() for p in params]
    saved_pre_muon = [M_buf.detach().cpu().clone() for _, M_buf, _ in muon_groups]
    w_gen = torch.Generator().manual_seed(seed)
    w_batch = build_cpu_microbatch(0, w_gen, verify_assert=True)
    refresh_all_dora_norms()
    for _ in range(2):
        lw = compute_pruned_loss_16k(w_batch)
        lw.backward()
        cautious_muon_step(lr_a, lr_b)
        adamw_opt.step()
        adamw_opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    del lw, w_batch
    with torch.no_grad():
        for p, p0 in zip(params, saved_pre_warm):
            p.copy_(p0.to(device))
            p.grad = None
        for (_, M_buf, _), m0 in zip(muon_groups, saved_pre_muon):
            M_buf.copy_(m0.to(device))
    adamw_opt.state.clear()
    refresh_all_dora_norms()
    del saved_pre_warm, saved_pre_muon
    gc.collect()
    torch.cuda.empty_cache()

    # Quick start check (4 val batches in time-limit mode, full val otherwise)
    if time_limit_s > 0.0:
        eval_0_4b = eval_full_vocab_loss_all(max_batches=4)
        print(f"=== START SANITY CHECK (first 4 val batches, 201k vocab): {eval_0_4b:.4f} (full 44-batch base = 1.0962) ===", flush=True)
        eval_history = [(0, eval_0_4b)]
    else:
        eval_0 = eval_full_vocab_loss_all()
        if eval_0 < best_eval_loss:
            best_eval_loss = eval_0
            save_checkpoint(best_path, start_opt_step, 0.0, best_eval_loss)
        print(f"=== STEP {start_opt_step} EVAL LOSS (201k vocab, {n_val_batches} val batches): {eval_0:.4f} ===", flush=True)
        eval_history = [(start_opt_step, eval_0)]

    q_stream, stop_ev, th_stream = start_prefetch_stream(start_opt_step * grad_accum, min(total_micro_steps, 20000))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    train_wall_s = 0.0
    fast_wall_s, fast_toks, fast_steps, fast_vram = 0.0, 0, 0, 0.0
    cd_wall_s, cd_toks, cd_steps, cd_vram = 0.0, 0, 0, 0.0
    in_cooldown = False
    nan_detected = False
    max_grad_norm = 0.0
    tokens_done = 0
    last_opt_step = start_opt_step
    t_seg = time.time()

    for opt_step in range(start_opt_step + 1, total_opt_steps + 1):
        # Check Phase 2 (Cooldown) transition BEFORE starting the next opt_step
        if time_limit_s > 0.0 and cooldown_start_s > 0.0 and (not in_cooldown) and train_wall_s >= cooldown_start_s:
            torch.cuda.synchronize()
            fast_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
            print(
                f"=== ENTERING COOLDOWN @ step {opt_step-1} (fast_time={fast_wall_s:.2f}s, "
                f"fast_tok/s={fast_toks/max(1e-6, fast_wall_s):.1f}, fast_VRAM={fast_vram:.2f} GB) -> "
                f"CUDAGraph=OFF, T-BPTT 2/3, top_p=1.0 (top_k=8), GC=Selective(1/3) ===",
                flush=True,
            )
            in_cooldown = True
            cg_cfg["enabled"] = False
            tbptt_cfg["grad_passes"] = 2
            moe_cfg["top_p"] = 1.0
            moe_cfg["k_min"] = 8
            set_gc_mode("Selective(1/3)")
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            t_seg = time.time()

        scale_lr = get_lr_scale(opt_step - 1, train_wall_s)
        lr_a_cur = lr_a * scale_lr
        lr_b_cur = lr_b * scale_lr
        adamw_opt.param_groups[0]["lr"] = lr_moe * scale_lr
        adamw_opt.param_groups[1]["lr"] = lr_norm * scale_lr

        # Refresh exact DoRA norms ONCE per optimizer step (0 weight materialization)
        refresh_all_dora_norms()
        accum_loss = 0.0
        cur_ep_float = 0.0
        for _ga in range(grad_accum):
            pkt = q_stream.get()
            if isinstance(pkt, Exception):
                raise pkt
            ep_idx, cpu_batch = pkt
            cur_ep_float = ((opt_step - 1) * grad_accum + _ga + 1) / float(n_train_batches)
            # Check SYNC DEBUG WARNINGS == 0 on 1 micro-step at opt_step == 2, _ga == 0
            if opt_step == 2 and _ga == 0:
                torch.cuda.synchronize()
                train_wall_s += time.time() - t_seg
                if not in_cooldown:
                    fast_wall_s += time.time() - t_seg
                torch.cuda.set_sync_debug_mode("warn")
                with warnings.catch_warnings(record=True) as w_sync:
                    warnings.simplefilter("always")
                    loss_m = compute_pruned_loss_16k(cpu_batch) / float(grad_accum)
                    loss_m.backward()
                torch.cuda.set_sync_debug_mode("default")
                sync_warns = sum(1 for w in w_sync if "sync" in str(w.message).lower() or "memcpy" in str(w.message).lower())
                print(f"=== STEP 2 SYNC DEBUG CHECK: sync_warns={sync_warns} ===", flush=True)
                assert sync_warns == 0, f"Detected {sync_warns} CUDA sync warnings!"
                t_seg = time.time()
            else:
                loss_m = compute_pruned_loss_16k(cpu_batch) / float(grad_accum)
                loss_m.backward()
            if not torch.isfinite(loss_m).item():
                nan_detected = True
            accum_loss += loss_m.item()
            tokens_done += SEQ_LEN
            if in_cooldown:
                cd_toks += SEQ_LEN
            else:
                fast_toks += SEQ_LEN

        if opt_step == 2:
            z_dora = sum(1 for _, p in dense_dora_ab + dense_dora_m if p.grad is None or p.grad.abs().sum().item() == 0)
            z_moe_a = sum(1 for _, p in moe_lora_a if p.grad is None or p.grad.abs().sum().item() == 0)
            z_moe_b = sum(1 for _, p in moe_lora_b if p.grad is None or p.grad.abs().sum().item() == 0)
            z_norm = sum(1 for _, p in rmsnorm_named if p.grad is None or p.grad.abs().sum().item() == 0)
            print(
                f"=== STEP 2 ZERO-GRAD CHECK: DenseDoRA={z_dora}/{len(dense_dora_ab)+len(dense_dora_m)} | "
                f"MoE_A={z_moe_a}/{len(moe_lora_a)} | MoE_B={z_moe_b}/{len(moe_lora_b)} | RMSNorm={z_norm}/{len(rmsnorm_named)} ===",
                flush=True,
            )
            assert (z_dora + z_moe_a + z_moe_b + z_norm) == 0, "Zero-grad params detected on step 2!"

        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0).item())
        if not math.isfinite(grad_norm):
            nan_detected = True
        max_grad_norm = max(max_grad_norm, grad_norm)
        cautious_muon_step(lr_a_cur, lr_b_cur, wd=0.01)
        adamw_opt.step()
        adamw_opt.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        step_dt = time.time() - t_seg
        train_wall_s += step_dt
        if in_cooldown:
            cd_wall_s += step_dt
            cd_steps += 1
        else:
            fast_wall_s += step_dt
            fast_steps += 1
        last_opt_step = opt_step

        if opt_step % 10 == 0 or (time_limit_s > 0.0 and train_wall_s >= time_limit_s) or opt_step == total_opt_steps:
            tok_s = tokens_done / max(1e-6, train_wall_s)
            vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            phase_tag = "CD" if in_cooldown else "FAST"
            print(
                f"[{phase_tag} step {opt_step:3d}] t={train_wall_s:5.1f}s | tok/s={tok_s:.1f} | "
                f"train_loss={accum_loss:.4f} | lr_A={lr_a_cur:.2e} lr_B={lr_b_cur:.2e} lr_moe={adamw_opt.param_groups[0]['lr']:.2e} | "
                f"grad_norm={grad_norm:.4f} | Peak VRAM={vram_gb:.2f} GB",
                flush=True,
            )

        if time_limit_s > 0.0 and train_wall_s >= time_limit_s:
            break

        if time_limit_s <= 0.0 and ((eval_every > 0 and opt_step % eval_every == 0) or opt_step == total_opt_steps):
            ev_loss = eval_full_vocab_loss_all()
            eval_history.append((opt_step, ev_loss))
            save_checkpoint(latest_path, opt_step, cur_ep_float, best_eval_loss)
            if ev_loss < best_eval_loss:
                best_eval_loss = ev_loss
                save_checkpoint(best_path, opt_step, cur_ep_float, best_eval_loss)
            print(f"=== EVAL @ STEP {opt_step} (train_time={train_wall_s:.1f}s): EvalLoss(201k)={ev_loss:.4f} (best={best_eval_loss:.4f}) ===", flush=True)

        t_seg = time.time()

    stop_ev.set()
    th_stream.join(timeout=5.0)
    if in_cooldown:
        cd_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
    else:
        fast_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
    vram_peak = max(fast_vram, cd_vram)
    avg_tok_s = tokens_done / max(1e-6, train_wall_s)
    fast_tok_s = fast_toks / max(1e-6, fast_wall_s)
    cd_tok_s = cd_toks / max(1e-6, cd_wall_s) if cd_wall_s > 0 else 0.0

    # Instant In-VRAM Full 44-batch 201k EvalLoss
    final_eval = eval_full_vocab_loss_all()
    eval_history.append((last_opt_step, final_eval))
    print(
        f"\n=== FINAL NEAR-FULL-FT SUMMARY | opt_steps={last_opt_step} (Fast={fast_steps}, Cooldown={cd_steps}) | "
        f"tokens={tokens_done/1e6:.2f}M | train_wall={train_wall_s:.2f}s | "
        f"tok/s: Fast={fast_tok_s:.1f}, Cooldown={cd_tok_s:.1f}, Avg={avg_tok_s:.1f} | "
        f"Peak VRAM: Fast={fast_vram:.2f} GB, Cooldown={cd_vram:.2f} GB | "
        f"EvalLoss(201k, 44 val batches)={final_eval:.4f} (vs base 1.0962, dLoss={final_eval-1.0962:+.4f}) | "
        f"NaN/Inf={nan_detected} | MaxGradNorm={max_grad_norm:.4f} ===",
        flush=True,
    )

    if eval_prompts_after:
        generate_3_answers_in_vram(f"NEAR-FULL-FT ({last_opt_step} steps, {train_wall_s:.0f}s)", max_new_tokens=96)

    total_gpu_s = time.time() - t_gpu_start
    print(f"\n=== TOTAL A100 TIME: {total_gpu_s:.1f}s (Est. cost: ~${(total_gpu_s/3600)*2.50:.4f}) ===", flush=True)
    return {
        "best_path": best_path,
        "latest_path": latest_path,
        "avg_tok_s": avg_tok_s,
        "vram_peak": vram_peak,
        "eval_history": eval_history,
        "nan_detected": nan_detected,
        "train_gpu_s": total_gpu_s,
    }


@app.function(gpu="A100-80GB", volumes={"/cache_vol": vol}, timeout=1200)
def merge_lora(
    lora_ckpt_path: str = "/cache_vol/inkling_sft_run/best_lora.pt",
    merged_dir: str = "/cache_vol/merged_inkling_sft",
    run_eval_prompts: bool = True,
):
    import sys
    sys.modules["torchvision"] = None
    sys.modules["torchaudio"] = None
    import gc
    import os
    import re
    import time
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoModelForImageTextToText as ModelCls, AutoTokenizer

    t0 = time.time()
    device = "cuda"
    dtype = torch.bfloat16
    vol.reload()

    print(f"=== MERGE_LORA: loading base /model + {lora_ckpt_path} ===", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("/model")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    ckpt = torch.load(lora_ckpt_path, map_location="cpu", weights_only=True)
    lora_state = ckpt["lora_state"]
    r = int(ckpt.get("lora_r", 32))
    alpha = int(ckpt.get("lora_alpha", 64))
    freeze_until = int(ckpt.get("freeze_until", 6))
    scale = float(alpha) / float(r)

    m = ModelCls.from_pretrained("/model", dtype=dtype, device_map="cuda")
    m.eval()

    class _LoRAWrap(nn.Module):
        def __init__(self, base_layer, la, lb, sc):
            super().__init__()
            self.base = base_layer
            self.lora_A = nn.Parameter(la.to(device=device, dtype=dtype), requires_grad=False)
            self.lora_B = nn.Parameter(lb.to(device=device, dtype=dtype), requires_grad=False)
            self.scale = sc

        def forward(self, x):
            return self.base(x) + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scale

    wrapped = []
    for name, mod in list(m.named_modules()):
        if isinstance(mod, nn.Linear) and "lm_head" not in name and mod.in_features >= 512 and mod.out_features >= 512:
            m_layer = re.search(r"layers\.(\d+)\.", name)
            if m_layer and int(m_layer.group(1)) < freeze_until:
                continue
            la_key = f"{name}.lora_A"
            lb_key = f"{name}.lora_B"
            if la_key in lora_state and lb_key in lora_state:
                parent_name, attr = name.rsplit(".", 1)
                wmod = _LoRAWrap(mod, lora_state[la_key], lora_state[lb_key], scale)
                setattr(m.get_submodule(parent_name), attr, wmod)
                wrapped.append((parent_name, attr, wmod))

    print(f"=== ATTACHED {len(wrapped)} LoRA MODULES FOR SANITY CHECK ===", flush=True)
    test_ids = tokenizer(CONTROL_PROMPTS[0], add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    with torch.no_grad():
        lora_logits = m(input_ids=test_ids, use_cache=False).logits.float()

    with torch.no_grad():
        for parent_name, attr, wmod in wrapped:
            w_base = wmod.base.weight.data
            w_merged = (w_base.float() + scale * (wmod.lora_B.float() @ wmod.lora_A.float())).to(torch.bfloat16)
            wmod.base.weight.data.copy_(w_merged)
            setattr(m.get_submodule(parent_name), attr, wmod.base)
    del wrapped
    gc.collect()
    torch.cuda.empty_cache()

    with torch.no_grad():
        merged_logits = m(input_ids=test_ids, use_cache=False).logits.float()

    logits_cos = F.cosine_similarity(lora_logits.flatten(), merged_logits.flatten(), dim=0).item()
    max_abs_diff = (lora_logits - merged_logits).abs().max().item()
    top1_match = (lora_logits.argmax(dim=-1) == merged_logits.argmax(dim=-1)).float().mean().item()
    print(f"=== MERGE SANITY CHECK: logits_cos={logits_cos:.6f} | max_abs_diff={max_abs_diff:.6f} | top1_match={top1_match*100:.2f}% ===", flush=True)
    assert logits_cos >= 0.999, f"Merge sanity check failed: cos={logits_cos}"
    del lora_logits, merged_logits

    os.makedirs(merged_dir, exist_ok=True)
    t_save = time.time()
    m.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()
    m.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    vol.commit()
    print(f"=== SAVED PURE HF CHECKPOINT TO {merged_dir} in {time.time()-t_save:.1f}s ===", flush=True)

    del m
    gc.collect()
    torch.cuda.empty_cache()

    if run_eval_prompts:
        print(f"\n=== LOADING CLEAN MERGED HF CHECKPOINT (ZERO CUSTOM CODE): {merged_dir} ===", flush=True)
        m_clean = ModelCls.from_pretrained(merged_dir, dtype=dtype, device_map="cuda")
        m_clean.eval()
        with torch.no_grad():
            for q_idx, prompt in enumerate(CONTROL_PROMPTS, 1):
                inp_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
                prompt_len = inp_ids.shape[1]
                cur_ids = inp_ids
                stopped_eos = False
                for _ in range(64):
                    out_logits = m_clean(input_ids=cur_ids, use_cache=False).logits[:, -1, :]
                    next_tok = torch.argmax(out_logits, dim=-1, keepdim=True)
                    cur_ids = torch.cat([cur_ids, next_tok], dim=1)
                    if next_tok.item() == tokenizer.eos_token_id:
                        stopped_eos = True
                        break
                gen_toks = cur_ids.shape[1] - prompt_len
                ans_text = tokenizer.decode(cur_ids[0, prompt_len:], skip_special_tokens=True)
                print(f"--- [MERGED BF16] Q{q_idx} (stopped_by_EOS={stopped_eos}, gen_tokens={gen_toks}) ---\n{ans_text}\n", flush=True)

    merge_s = time.time() - t0
    print(f"=== MERGE_LORA TOTAL TIME: {merge_s:.1f}s (Est. cost: ~${(merge_s/3600)*2.50:.4f}) ===", flush=True)
    return {"logits_cos": logits_cos, "max_abs_diff": max_abs_diff, "merged_dir": merged_dir, "merge_gpu_s": merge_s}


@app.local_entrypoint()
def main(
    dataset: str = "yahma/alpaca-cleaned",
    grad_accum: int = 4,
    max_steps: int = 100,
    epochs: int = 1,
    eval_every: int = 50,
    lr_a: float = 1.0e-4,
    lr_b: float = 4.0e-4,
    lr_moe: float = 1.5e-4,
    lr_norm: float = 1.0e-4,
    seed: int = 42,
    resume: bool = False,
    output_dir: str = "/cache_vol/inkling_sft_run",
    merged_dir: str = "/cache_vol/merged_inkling_sft",
    merge_only: bool = False,
    merge_after: bool = False,
    time_limit_s: float = 120.0,
    cooldown_start_s: float = 100.0,
    eval_prompts_after: bool = True,
):
    if merge_only:
        merge_lora.remote(lora_ckpt_path=f"{output_dir}/best_lora.pt", merged_dir=merged_dir, run_eval_prompts=True)
        return

    res_train = train_sft.remote(
        dataset=dataset,
        grad_accum=grad_accum,
        max_steps=max_steps,
        epochs=epochs,
        eval_every=eval_every,
        lr_a=lr_a,
        lr_b=lr_b,
        lr_moe=lr_moe,
        lr_norm=lr_norm,
        seed=seed,
        resume=resume,
        output_dir=output_dir,
        time_limit_s=time_limit_s,
        cooldown_start_s=cooldown_start_s,
        eval_prompts_after=eval_prompts_after,
    )
    if merge_after:
        res_merge = merge_lora.remote(
            lora_ckpt_path=res_train["best_path"],
            merged_dir=merged_dir,
            run_eval_prompts=True,
        )
        tot_s = res_train["train_gpu_s"] + res_merge["merge_gpu_s"]
        print(f"\n=== GRAND TOTAL A100 TIME: {tot_s:.1f}s (Est. cost: ~${(tot_s/3600)*2.50:.4f}) ===", flush=True)
