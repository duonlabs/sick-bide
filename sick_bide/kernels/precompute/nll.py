import math
from typing import Tuple
import torch
import triton
import triton.language as tl

from ..utils import DTYPECAT, CAT2ELS2DTY
from .integral import _sick_integral_kernel, precompute_blocks_fwd
from ..utils import bide_logits


@triton.jit
def sick_nll_kernel_bwd(
    p1_ptr, p2_ptr, r_ptr, m_ptr, p1_grad_ptr, p2_grad_ptr, r_grad_ptr,
    n_bits: tl.constexpr, hidden_size: tl.constexpr, p_size: tl.constexpr,
    P1_BLOCK_SIZE: tl.constexpr,
):
    # Handle ids
    n_pid, p1_pid = tl.program_id(0), tl.program_id(1) # Batch pid, combinatorial pid
    p1_id = p1_pid * P1_BLOCK_SIZE + tl.arange(0, P1_BLOCK_SIZE) # [P1_BLOCK_SIZE] - Input value
    h_id = tl.arange(0, hidden_size) # [hidden_size] - Hidden id
    # Load tensors
    r_offset = n_pid * hidden_size + h_id # [hidden_size]
    r = tl.load(r_ptr + r_offset) # [hidden_size]
    m = tl.load(m_ptr + n_pid) # []
    p_strided_n = n_pid * p_size * hidden_size
    p1_offset = p_strided_n + p1_id[:, None] * hidden_size + h_id[None, :] # [P1_BLOCK_SIZE, hidden_size]
    p1 = tl.load(p1_ptr + p1_offset) # [P1_BLOCK_SIZE, hidden_size]
    # Compute the integral over the block
    r_grad = tl.zeros((hidden_size,), dtype=tl.float32) # [N_BLOCK_SIZE]
    p1_grad = tl.zeros((P1_BLOCK_SIZE, hidden_size), dtype=tl.float32) # [P1_BLOCK_SIZE, hidden_size]
    p2_offset = p_strided_n + tl.arange(0, hidden_size) # [hidden_size]
    for _ in range(p_size):
        z = p1 + tl.load(p2_ptr + p2_offset) # [P1_BLOCK_SIZE, hidden_size]
        relu_mask = z > 0 # [P1_BLOCK_SIZE, hidden_size]
        h = tl.where(relu_mask, z, 0) # [P1_BLOCK_SIZE, hidden_size]
        logits = tl.sum(h*r, axis=-1) # [P1_BLOCK_SIZE]
        stable_exp_logit = tl.exp(logits - m) # [P1_BLOCK_SIZE]
        r_grad += tl.sum(h*stable_exp_logit[:, None], axis=-2) # [hidden_size]
        z_grad = tl.where(relu_mask, r, 0) * stable_exp_logit[:, None]# [P1_BLOCK_SIZE, hidden_size]
        tl.atomic_add(p2_grad_ptr + p2_offset, tl.sum(z_grad, axis=0)) # [hidden_size]
        p1_grad += z_grad # [P1_BLOCK_SIZE, hidden_size]
        p2_offset += hidden_size
    # Store the results
    tl.atomic_add(r_grad_ptr + r_offset, r_grad) # [hidden_size]
    tl.atomic_add(p1_grad_ptr + p1_offset, p1_grad) # [P1_BLOCK_SIZE, hidden_size]
        

@torch.compile
def precompute_blocks_bwd(p1_grad: torch.Tensor, p2_grad: torch.Tensor) -> torch.Tensor:
    n_bits = int(math.log2(p1_grad.shape[1]))
    x = torch.arange(2**n_bits, device=p1_grad.device) # [2**n_bits] - Generate all possible inputs
    x = ((x.unsqueeze(-1) >> torch.arange(n_bits, device=p1_grad.device))&1).to(p1_grad.dtype) * 2 - 1 # [2**n_bits, n_bits] - Convert to binary representation
    dL_dW1, dL_dW2 = p1_grad.permute(0,2,1) @ x, p2_grad.permute(0,2,1) @ x # [B, H, n_bits]
    return torch.cat([dL_dW1, dL_dW2], dim=-1) # [B, H, 2*n_bits]

@torch.compile
def compute_dW_dr_from_acc(W_acc: torch.Tensor, r_acc: torch.Tensor, W: torch.Tensor, r: torch.Tensor, y: torch.Tensor, lse: torch.Tensor, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    x = ((y.unsqueeze(-1).view(CAT2ELS2DTY[DTYPECAT.UINT][y.element_size()]).long() >> torch.arange(W.shape[-1], device=W.device))&1).to(W.dtype) * 2 - 1 # [B, N]
    h = torch.nn.functional.relu(x[:, None, :] @ W.permute(0, 2, 1)).squeeze(1) # [B, H]
    W_grad = (W_acc/lse.exp()[:, None, None] - (torch.where((h>0)[:, :, None], x[:, None, :], 0.0) * r[:, :, None]))*grad_output[:, None, None] # [B, H, N]
    r_grad = (r_acc/lse.exp()[:, None]-h)*grad_output[:, None] # [B, H]
    return W_grad.to(W.dtype), r_grad.to(r.dtype)


class _sick_nll_kernel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y: torch.Tensor, W: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """
            y: [B, ...]
            W: [B, H, N]
            r: [B, H]

            -> [B, ...]
        """
        lse, m = _sick_integral_kernel.forward(ctx, W, r)
        logits = bide_logits(W, r, y)
        ctx.save_for_backward(W, r, y, lse, m)
        return -logits + m + lse

    @staticmethod
    def backward(ctx, grad_output, block_size: Tuple[int] = (256,), num_warps: int = 4):
        """
        grad_output: [B, ...]
        """
        W, r, y, lse, m = ctx.saved_tensors
        batch_size, hidden_size, n_bits = W.shape
        p_size = 2**(n_bits//2)
        n_p1_blocks = p_size // block_size[0] # Number of blocks for the combinatorial dimension
        # Grad tensors
        r_grad = torch.zeros_like(r) # [H]
        p1, p2 = precompute_blocks_fwd(W) # [B, p_size, H], [1, p_size, H] - Precompute the blocks
        p1_grad = torch.zeros_like(p1) # [B, p_size, H]
        p2_grad = torch.zeros_like(p2) # [1, p_size, H]
        sick_nll_kernel_bwd[(batch_size, n_p1_blocks)](
            p1, p2, r, m, p1_grad, p2_grad, r_grad,
            n_bits, hidden_size, p_size,
            *block_size, num_warps=num_warps
        )
        W_grad = precompute_blocks_bwd(p1_grad, p2_grad)
        W_grad, r_grad = compute_dW_dr_from_acc(W_grad, r_grad, W, r, y, lse, grad_output)
        return None, W_grad, r_grad
    
sick_nll_kernel = _sick_nll_kernel.apply