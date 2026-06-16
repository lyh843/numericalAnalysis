"""
Block-sparse attention with int8 Q/K quantization (Triton).

Adapted from thu-ml/SpargeAttn (Apache-2.0):
  - Triton_SpargeAttn/triton_kernel_example.py
  - spas_sage_attn/quant_per_block.py

Compared to attention/sparse.py, this variant:
  - Quantizes Q and K in 64x64 tiles with one scale per tile.
  - Folds log2(e)/sqrt(d) into Q's dequant scale so the inner softmax can use
    exp2 instead of exp.
  - Uses int8 tensor-core matmul (tl.dot of int8 inputs -> int32, dequantized
    by `* q_scale * k_scale`).
  - Optional smooth_k (subtract per-channel K mean) — mathematically a no-op
    for softmax output but improves K's int8 dynamic range.
"""

import math

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from .sparse import _pad_seq_len, _select_k_blocks


@triton.jit
def _quantize_per_block_kernel(
    x_ptr, q_ptr, s_ptr,
    stride_x_batch, stride_x_head, stride_x_num, stride_x_dim,
    stride_q_batch, stride_q_head, stride_q_num, stride_q_dim,
    stride_s_batch, stride_s_head, stride_s_num, stride_s_dim,
    num_heads,
    n_ctx,
    head_dim,
    scale_multiplier,
    BLOCK_SIZE: tl.constexpr,
    QUANT_BLOCK_DIM: tl.constexpr,
):
    pid_seq_blk = tl.program_id(0)
    pid_dim_blk = tl.program_id(1)
    pid_batch_head = tl.program_id(2)
    b = pid_batch_head // num_heads
    h = pid_batch_head % num_heads

    offs_n = pid_seq_blk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_d = pid_dim_blk * QUANT_BLOCK_DIM + tl.arange(0, QUANT_BLOCK_DIM)
    mask = (offs_n[:, None] < n_ctx) & (offs_d[None, :] < head_dim)

    x_ptrs = (
        x_ptr
        + b * stride_x_batch
        + h * stride_x_head
        + offs_n[:, None] * stride_x_num
        + offs_d[None, :] * stride_x_dim
    )
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    max_abs = tl.max(tl.max(tl.abs(x), axis=1), axis=0)
    scale = tl.maximum(max_abs / 127.0, 1e-8)
    dequant_scale = scale * scale_multiplier

    q = tl.clamp(libdevice.round(x / scale), -127.0, 127.0).to(tl.int8)

    q_ptrs = (
        q_ptr
        + b * stride_q_batch
        + h * stride_q_head
        + offs_n[:, None] * stride_q_num
        + offs_d[None, :] * stride_q_dim
    )
    tl.store(q_ptrs, q, mask=mask)

    s_ptr = (
        s_ptr
        + b * stride_s_batch
        + h * stride_s_head
        + pid_seq_blk * stride_s_num
        + pid_dim_blk * stride_s_dim
    )
    tl.store(s_ptr, dequant_scale)


def _quantize_per_block(x, block_size, scale_multiplier=1.0):
    B, H, N, D = x.shape
    quant_block_dim = 64
    num_seq_blocks = N // block_size
    num_dim_blocks = triton.cdiv(D, quant_block_dim)
    q = torch.empty((B, H, N, D), device=x.device, dtype=torch.int8)
    scales = torch.empty(
        (B, H, num_seq_blocks, num_dim_blocks),
        device=x.device,
        dtype=torch.float32,
    )

    grid = (num_seq_blocks, num_dim_blocks, B * H)

    _quantize_per_block_kernel[grid](
        x, q, scales,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        scales.stride(0), scales.stride(1), scales.stride(2), scales.stride(3),
        H,
        N,
        D,
        scale_multiplier,
        BLOCK_SIZE=block_size,
        QUANT_BLOCK_DIM=quant_block_dim,
        num_warps=4,
        num_stages=2,
    )

    return q, scales


@triton.jit
def _sparse_int8_attention_fwd(
    q_ptr, q_scale_ptr, k_ptr, k_scale_ptr, v_ptr, idx_ptr, out_ptr,
    stride_q_batch, stride_q_head, stride_q_num, stride_q_dim,
    stride_qs_batch, stride_qs_head, stride_qs_seq, stride_qs_dim,
    stride_k_batch, stride_k_head, stride_k_num, stride_k_dim,
    stride_ks_batch, stride_ks_head, stride_ks_seq, stride_ks_dim,
    stride_v_batch, stride_v_head, stride_v_num, stride_v_dim,
    stride_i_batch, stride_i_head, stride_i_qblk, stride_i_topk,
    stride_o_batch, stride_o_head, stride_o_num, stride_o_dim,
    num_heads,
    n_ctx,
    head_dim,
    topk_blocks,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    QUANT_BLOCK_DIM: tl.constexpr,
    NUM_DIM_BLOCKS: tl.constexpr,
):
    pid_q_block = tl.program_id(0)
    pid_batch_head = tl.program_id(1)
    b = pid_batch_head // num_heads
    h = pid_batch_head % num_heads

    offs_q = pid_q_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_k = tl.arange(0, BLOCK_SIZE)
    offs_dim = tl.arange(0, BLOCK_DIM)
    row_mask = offs_q < n_ctx

    if NUM_DIM_BLOCKS == 1:
        offs_qk_dim = tl.arange(0, QUANT_BLOCK_DIM)
        q_ptrs = (
            q_ptr
            + b * stride_q_batch
            + h * stride_q_head
            + offs_q[:, None] * stride_q_num
            + offs_qk_dim[None, :] * stride_q_dim
        )
        q_mask = row_mask[:, None] & (offs_qk_dim[None, :] < head_dim)
        q_i8_single = tl.load(q_ptrs, mask=q_mask, other=0)
        q_scale_ptrs = (
            q_scale_ptr
            + b * stride_qs_batch
            + h * stride_qs_head
            + pid_q_block * stride_qs_seq
        )
        q_scale_single = tl.load(q_scale_ptrs)

    m_i = tl.full((BLOCK_SIZE,), float("-inf"), dtype=tl.float32)
    d_i = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    o_i = tl.zeros((BLOCK_SIZE, BLOCK_DIM), dtype=tl.float32)

    idx_base = idx_ptr + b * stride_i_batch + h * stride_i_head + pid_q_block * stride_i_qblk
    for topk_idx in tl.range(0, topk_blocks):
        block_id = tl.load(idx_base + topk_idx * stride_i_topk)
        start_n = block_id * BLOCK_SIZE
        cur_n = start_n + offs_k
        col_mask = cur_n < n_ctx

        if NUM_DIM_BLOCKS == 1:
            k_ptrs = (
                k_ptr
                + b * stride_k_batch
                + h * stride_k_head
                + cur_n[:, None] * stride_k_num
                + offs_qk_dim[None, :] * stride_k_dim
            )
            k_mask = col_mask[:, None] & (offs_qk_dim[None, :] < head_dim)
            k_i8 = tl.load(k_ptrs, mask=k_mask, other=0)
            k_scale_ptrs = (
                k_scale_ptr
                + b * stride_ks_batch
                + h * stride_ks_head
                + block_id * stride_ks_seq
            )
            k_scale = tl.load(k_scale_ptrs)

            qk_i32 = tl.dot(q_i8_single, tl.trans(k_i8))
            qk = qk_i32.to(tl.float32) * q_scale_single * k_scale
        else:
            qk = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
            for dim_blk in tl.static_range(0, NUM_DIM_BLOCKS):
                offs_qk_dim = dim_blk * QUANT_BLOCK_DIM + tl.arange(0, QUANT_BLOCK_DIM)

                q_ptrs = (
                    q_ptr
                    + b * stride_q_batch
                    + h * stride_q_head
                    + offs_q[:, None] * stride_q_num
                    + offs_qk_dim[None, :] * stride_q_dim
                )
                q_mask = row_mask[:, None] & (offs_qk_dim[None, :] < head_dim)
                q_i8 = tl.load(q_ptrs, mask=q_mask, other=0)
                q_scale_ptrs = (
                    q_scale_ptr
                    + b * stride_qs_batch
                    + h * stride_qs_head
                    + pid_q_block * stride_qs_seq
                    + dim_blk * stride_qs_dim
                )
                q_scale = tl.load(q_scale_ptrs)

                k_ptrs = (
                    k_ptr
                    + b * stride_k_batch
                    + h * stride_k_head
                    + cur_n[:, None] * stride_k_num
                    + offs_qk_dim[None, :] * stride_k_dim
                )
                k_mask = col_mask[:, None] & (offs_qk_dim[None, :] < head_dim)
                k_i8 = tl.load(k_ptrs, mask=k_mask, other=0)
                k_scale_ptrs = (
                    k_scale_ptr
                    + b * stride_ks_batch
                    + h * stride_ks_head
                    + block_id * stride_ks_seq
                    + dim_blk * stride_ks_dim
                )
                k_scale = tl.load(k_scale_ptrs)

                qk_i32 = tl.dot(q_i8, tl.trans(k_i8))
                qk += qk_i32.to(tl.float32) * q_scale * k_scale

        qk = tl.where(row_mask[:, None] & col_mask[None, :], qk, float("-inf"))
        qk = tl.where(row_mask[:, None], qk, 0.0)

        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        m_new = tl.where(row_mask, m_new, 0.0)

        alpha = tl.where(row_mask, tl.math.exp2(m_i - m_new), 0.0)
        p = tl.where(row_mask[:, None], tl.math.exp2(qk - m_new[:, None]), 0.0)

        d_i = d_i * alpha + tl.sum(p, axis=1)

        v_ptrs = (
            v_ptr
            + b * stride_v_batch
            + h * stride_v_head
            + cur_n[:, None] * stride_v_num
            + offs_dim[None, :] * stride_v_dim
        )
        v_mask = col_mask[:, None] & (offs_dim[None, :] < head_dim)
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)

        o_i = o_i * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    o_i = tl.where(row_mask[:, None], o_i / d_i[:, None], 0.0)

    out_ptrs = (
        out_ptr
        + b * stride_o_batch
        + h * stride_o_head
        + offs_q[:, None] * stride_o_num
        + offs_dim[None, :] * stride_o_dim
    )
    out_mask = row_mask[:, None] & (offs_dim[None, :] < head_dim)
    tl.store(out_ptrs, o_i.to(out_ptr.dtype.element_ty), mask=out_mask)


def sparse_int8_attention(q, k, v, attn_mask=None, topk_ratio=0.5,
                          block_size=64, smooth_k=False):
    """Block-sparse attention with int8 Q/K quantization.

    Args:
        q: Query tensor, shape [B, num_heads, N, head_dim]
        k: Key tensor, shape [B, num_heads, N, head_dim]
        v: Value tensor, shape [B, num_heads, N, head_dim]
        attn_mask: Attention mask (optional, ignored for API compatibility)
        topk_ratio: Ratio of K blocks to select per Q block (default 0.5)
        block_size: Block size for block selection and Triton tiles (default 64)
        smooth_k: Whether to subtract per-channel K mean before quantization (default False)

    Returns:
        Output tensor, shape [B, num_heads, N, head_dim]

    Notes:
        - This keeps the same sparse block selection logic as `sparse_attention`.
        - Q/K use one symmetric int8 scale per 64x64 tile.
        - Q's stored dequant scale already includes log2(e)/sqrt(d), so the
          kernel can use exp2 for the online softmax update.
    """
    q_pad, seq_len = _pad_seq_len(q, block_size)
    k_pad, _ = _pad_seq_len(k, block_size)
    v_pad, _ = _pad_seq_len(v, block_size)

    block_indices = _select_k_blocks(q_pad, k_pad, seq_len, topk_ratio, block_size)

    if smooth_k:
        valid = (torch.arange(k_pad.shape[2], device=k_pad.device) < seq_len).to(k_pad.dtype)
        valid = valid.view(1, 1, -1, 1)
        denom = valid.sum(dim=2, keepdim=True).clamp_min(1.0)
        k_mean = (k_pad * valid).sum(dim=2, keepdim=True) / denom
        k_quant_src = k_pad - k_mean
    else:
        k_quant_src = k_pad

    q_scale_mul = 1.4426950408889634 / math.sqrt(q_pad.shape[-1])
    q_i8, q_scales = _quantize_per_block(q_pad, block_size, scale_multiplier=q_scale_mul)
    k_i8, k_scales = _quantize_per_block(k_quant_src, block_size, scale_multiplier=1.0)

    B, H, padded_len, D = q_pad.shape
    out = torch.empty((B, H, padded_len, D), device=v_pad.device, dtype=v_pad.dtype)
    grid = (padded_len // block_size, B * H)
    block_dim = triton.next_power_of_2(D)
    quant_block_dim = 64
    num_dim_blocks = triton.cdiv(D, quant_block_dim)
    topk_blocks = block_indices.shape[-1]
    num_warps = 4 if D <= 64 else 8

    _sparse_int8_attention_fwd[grid](
        q_i8, q_scales, k_i8, k_scales, v_pad, block_indices, out,
        q_i8.stride(0), q_i8.stride(1), q_i8.stride(2), q_i8.stride(3),
        q_scales.stride(0), q_scales.stride(1), q_scales.stride(2), q_scales.stride(3),
        k_i8.stride(0), k_i8.stride(1), k_i8.stride(2), k_i8.stride(3),
        k_scales.stride(0), k_scales.stride(1), k_scales.stride(2), k_scales.stride(3),
        v_pad.stride(0), v_pad.stride(1), v_pad.stride(2), v_pad.stride(3),
        block_indices.stride(0), block_indices.stride(1), block_indices.stride(2), block_indices.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        H,
        seq_len,
        D,
        topk_blocks,
        BLOCK_SIZE=block_size,
        BLOCK_DIM=block_dim,
        QUANT_BLOCK_DIM=quant_block_dim,
        NUM_DIM_BLOCKS=num_dim_blocks,
        num_warps=num_warps,
        num_stages=2,
    )

    return out[:, :, :seq_len, :]
