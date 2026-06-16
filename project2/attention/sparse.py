"""
Block-sparse attention with PyTorch block selection and a Triton kernel.

This file follows task1.md for subtask 1.3:
  - Block selection is implemented in PyTorch.
  - Sparse attention is implemented with a Triton online-softmax kernel.
  - Inputs/outputs use shape [B, H, N, D].
"""

import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


def _pad_seq_len(x, block_size):
    seq_len = x.shape[2]
    padded_len = triton.cdiv(seq_len, block_size) * block_size
    if padded_len == seq_len:
        return x, seq_len
    return F.pad(x, (0, 0, 0, padded_len - seq_len)), seq_len


def _pooled_blocks(x, seq_len, block_size):
    padded_len = x.shape[2]
    num_blocks = padded_len // block_size
    x_blocks = x.reshape(x.shape[0], x.shape[1], num_blocks, block_size, x.shape[3])

    if seq_len == padded_len:
        return x_blocks.mean(dim=3)

    valid = (torch.arange(padded_len, device=x.device) < seq_len).to(x.dtype)
    valid = valid.view(num_blocks, block_size)
    counts = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
    valid = valid.view(1, 1, num_blocks, block_size, 1)

    return (x_blocks * valid).sum(dim=3) / counts.view(1, 1, num_blocks, 1)


def _select_k_blocks(q, k, seq_len, topk_ratio, block_size):
    num_q_blocks = q.shape[2] // block_size
    num_k_blocks = k.shape[2] // block_size
    topk = int(num_k_blocks * topk_ratio)

    if topk_ratio == 1.0:
        return torch.arange(
            num_k_blocks, device=q.device, dtype=torch.int32
        ).view(1, 1, 1, num_k_blocks).expand(q.shape[0], q.shape[1], num_q_blocks, num_k_blocks)

    q_pooled = _pooled_blocks(q, seq_len, block_size)
    k_pooled = _pooled_blocks(k, seq_len, block_size)
    block_scores = torch.matmul(q_pooled, k_pooled.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    block_indices = torch.topk(block_scores, k=topk, dim=-1, sorted=False).indices
    block_indices = torch.sort(block_indices, dim=-1).values
    return block_indices.to(torch.int32)


@triton.jit
def _sparse_attention_fwd(
    q_ptr, k_ptr, v_ptr, idx_ptr, out_ptr,
    stride_q_batch, stride_q_head, stride_q_num, stride_q_dim,
    stride_k_batch, stride_k_head, stride_k_num, stride_k_dim,
    stride_v_batch, stride_v_head, stride_v_num, stride_v_dim,
    stride_i_batch, stride_i_head, stride_i_qblk, stride_i_topk,
    stride_o_batch, stride_o_head, stride_o_num, stride_o_dim,
    num_heads,
    n_ctx,
    head_dim,
    topk_blocks,
    scale,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    pid_q_block = tl.program_id(0)
    pid_batch_head = tl.program_id(1)
    b = pid_batch_head // num_heads
    h = pid_batch_head % num_heads

    offs_q = pid_q_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_k = tl.arange(0, BLOCK_SIZE)
    offs_dim = tl.arange(0, BLOCK_DIM)
    row_mask = offs_q < n_ctx

    q_ptrs = (
        q_ptr
        + b * stride_q_batch
        + h * stride_q_head
        + offs_q[:, None] * stride_q_num
        + offs_dim[None, :] * stride_q_dim
    )
    q_mask = row_mask[:, None] & (offs_dim[None, :] < head_dim)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    m_i = tl.full((BLOCK_SIZE,), float("-inf"), dtype=tl.float32)
    d_i = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    o_i = tl.zeros((BLOCK_SIZE, BLOCK_DIM), dtype=tl.float32)

    idx_base = idx_ptr + b * stride_i_batch + h * stride_i_head + pid_q_block * stride_i_qblk
    for topk_idx in tl.range(0, topk_blocks):
        block_id = tl.load(idx_base + topk_idx * stride_i_topk)
        cur_n = block_id * BLOCK_SIZE + offs_k
        col_mask = cur_n < n_ctx

        k_ptrs = (
            k_ptr
            + b * stride_k_batch
            + h * stride_k_head
            + cur_n[:, None] * stride_k_num
            + offs_dim[None, :] * stride_k_dim
        )
        k_mask = col_mask[:, None] & (offs_dim[None, :] < head_dim)
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        qk = tl.dot(q, tl.trans(k)) * scale
        qk = tl.where(row_mask[:, None] & col_mask[None, :], qk, float("-inf"))
        qk = tl.where(row_mask[:, None], qk, 0.0)

        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        m_new = tl.where(row_mask, m_new, 0.0)

        alpha = tl.where(row_mask, tl.exp(m_i - m_new), 0.0)
        p = tl.where(row_mask[:, None], tl.exp(qk - m_new[:, None]), 0.0)
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


def sparse_attention(q, k, v, attn_mask=None, dropout_p=0.0, training=False,
                     topk_ratio=0.5, block_size=64):
    """
    Block-sparse attention.

    Args:
        q, k, v: Tensors with shape [B, H, N, D].
        attn_mask: Unsupported in this project; kept for API compatibility.
        dropout_p: Ignored for API compatibility.
        training: Ignored for API compatibility.
        topk_ratio: Ratio used in block selection.
        block_size: Sequence block size used by selection and the Triton kernel.
    """
    del dropout_p, training

    q_pad, seq_len = _pad_seq_len(q, block_size)
    k_pad, _ = _pad_seq_len(k, block_size)
    v_pad, _ = _pad_seq_len(v, block_size)

    block_indices = _select_k_blocks(q_pad, k_pad, seq_len, topk_ratio, block_size)

    bsz, num_heads, padded_len, head_dim = q_pad.shape
    out = torch.empty(
        (bsz, num_heads, padded_len, head_dim),
        device=q_pad.device,
        dtype=q_pad.dtype,
    )
    block_dim = triton.next_power_of_2(head_dim)
    topk_blocks = block_indices.shape[-1]
    num_warps = 4 if head_dim <= 64 else 8
    scale = 1.0 / math.sqrt(head_dim)

    _sparse_attention_fwd[(padded_len // block_size, bsz * num_heads)](
        q_pad, k_pad, v_pad, block_indices, out,
        q_pad.stride(0), q_pad.stride(1), q_pad.stride(2), q_pad.stride(3),
        k_pad.stride(0), k_pad.stride(1), k_pad.stride(2), k_pad.stride(3),
        v_pad.stride(0), v_pad.stride(1), v_pad.stride(2), v_pad.stride(3),
        block_indices.stride(0), block_indices.stride(1), block_indices.stride(2), block_indices.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        num_heads,
        seq_len,
        head_dim,
        topk_blocks,
        scale,
        BLOCK_SIZE=block_size,
        BLOCK_DIM=block_dim,
        num_warps=num_warps,
        num_stages=2,
    )

    return out[:, :, :seq_len, :]
