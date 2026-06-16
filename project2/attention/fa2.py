"""
Triton-based Flash Attention 2 implementation (forward only).
Optimized for inference without backward pass.
"""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _flash_attention_2_fwd(
    q_ptr, k_ptr, v_ptr, out_ptr,
    stride_q_batch, stride_q_head, stride_q_num, stride_q_dim,
    stride_k_batch, stride_k_head, stride_k_num, stride_k_dim,
    stride_v_batch, stride_v_head, stride_v_num, stride_v_dim,
    stride_o_batch, stride_o_head, stride_o_num, stride_o_dim,
    num_heads,
    n_ctx,
    head_dim,
    scale,
    BLOCK_Q_LEN: tl.constexpr,
    BLOCK_K_LEN: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    pid_q_block = tl.program_id(0)
    pid_batch_head = tl.program_id(1)
    b = pid_batch_head // num_heads
    h = pid_batch_head % num_heads

    offs_q_block = pid_q_block * BLOCK_Q_LEN + tl.arange(0, BLOCK_Q_LEN)
    offs_k = tl.arange(0, BLOCK_K_LEN)
    offs_dim = tl.arange(0, BLOCK_DIM)
    row_mask = offs_q_block < n_ctx

    q_ptr = q_ptr + b * stride_q_batch + h * stride_q_head + offs_q_block[:, None] * stride_q_num + offs_dim[None, :] * stride_q_dim
    q_mask = (offs_q_block[:, None] < n_ctx) & (offs_dim[None, :] < head_dim)
    q = tl.load(q_ptr, mask=q_mask, other=0.0).to(tl.float32)

    m_i = tl.full((BLOCK_Q_LEN,), float("-inf"), dtype=tl.float32)
    d_i = tl.zeros((BLOCK_Q_LEN,), dtype=tl.float32)
    o_i = tl.zeros((BLOCK_Q_LEN, BLOCK_DIM), dtype=tl.float32)

    for start_n in tl.range(0, n_ctx, BLOCK_K_LEN):
        cur_n = start_n + offs_k

        k_ptrs = k_ptr + b * stride_k_batch + h * stride_k_head + cur_n[:, None] * stride_k_num + offs_dim[None, :] * stride_k_dim
        k_mask = (cur_n[:, None] < n_ctx) & (offs_dim[None, :] < head_dim)
        k = tl.load(k_ptrs, mask=k_mask, other=0.0).to(tl.float32)

        qk = tl.dot(q, tl.trans(k)) * scale
        qk = tl.where((offs_q_block[:, None] < n_ctx) & (cur_n[None, :] < n_ctx), qk, float("-inf"))
        qk = tl.where(row_mask[:, None], qk, 0.0)

        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        m_new = tl.where(row_mask, m_new, 0.0)

        alpha = tl.where(row_mask, tl.exp(m_i - m_new), 0.0)
        p = tl.where(row_mask[:, None], tl.exp(qk - m_new[:, None]), 0.0)

        d_i = d_i * alpha + tl.sum(p, axis=1)

        v_ptrs = v_ptr + b * stride_v_batch + h * stride_v_head + cur_n[:, None] * stride_v_num + offs_dim[None, :] * stride_v_dim
        v_mask = (cur_n[:, None] < n_ctx) & (offs_dim[None, :] < head_dim)
        v = tl.load(v_ptrs, mask=v_mask, other=0.0).to(tl.float32)

        o_i = o_i * alpha[:, None] + tl.dot(p, v)
        m_i = m_new

    o_i = tl.where(row_mask[:, None], o_i / d_i[:, None], 0.0)

    out_ptrs = out_ptr + b * stride_o_batch + h * stride_o_head + offs_q_block[:, None] * stride_o_num + offs_dim[None, :] * stride_o_dim
    out_mask = (offs_q_block[:, None] < n_ctx) & (offs_dim[None, :] < head_dim)
    tl.store(out_ptrs, o_i.to(out_ptr.dtype.element_ty), mask=out_mask)


def flash_attention_2(q, k, v, attn_mask=None, dropout_p=0.0, training=False):
    """
    Flash Attention 2 forward pass using Triton.

    Args:
        q: Query tensor, shape [B, num_heads, N, head_dim]
        k: Key tensor, shape [B, num_heads, N, head_dim]
        v: Value tensor, shape [B, num_heads, N, head_dim]
        attn_mask: Attention mask (optional, ignored for API compatibility)
        dropout_p: Dropout probability (default 0.0, ignored for API compatibility)
        training: Training mode flag (default False, ignored for API compatibility)

    Returns:
        Output tensor, shape [B, num_heads, N, head_dim]
    """

    B, H, N, D = q.shape
    out = torch.empty_like(q)

    BLOCK_Q_LEN = 128
    BLOCK_K_LEN = 64
    BLOCK_DIM = triton.next_power_of_2(D)

    grid = (triton.cdiv(N, BLOCK_Q_LEN), B * H)
    num_warps = 4 if D <= 64 else 8

    _flash_attention_2_fwd[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        H,
        N,
        D,
        1.0 / math.sqrt(D),
        BLOCK_Q_LEN=BLOCK_Q_LEN,
        BLOCK_K_LEN=BLOCK_K_LEN,
        BLOCK_DIM=BLOCK_DIM,
        num_warps=num_warps,
        num_stages=2,
    )

    return out
