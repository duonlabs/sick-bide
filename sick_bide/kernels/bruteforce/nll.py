from typing import Tuple
import torch
import triton
import triton.language as tl

from ..utils import DTYPECAT, CAT2ELS2DTY
from .integral import _sick_integral_kernel
from ..utils import bide_logits


@triton.jit
def sick_nll_kernel_bwd(
    W_ptr, r_ptr, m_ptr, W_grad_ptr, r_grad_ptr,
    n_bits: tl.constexpr, hidden_size: tl.constexpr,
    N_BLOCK_SIZE: tl.constexpr, K_BLOCK_SIZE: tl.constexpr, N_UNROLL: tl.constexpr
):
    # Handle ids
    n_pid, k_pid = tl.program_id(0), tl.program_id(1) # Batch pid, combinatorial pid
    n_id = n_pid * N_BLOCK_SIZE + tl.arange(0, N_BLOCK_SIZE) # [N_BLOCK_SIZE] - Batch id
    b_id = tl.arange(0, n_bits) # [n_bits] - Bit id
    h_id = tl.arange(0, hidden_size) # [hidden_size] - Hidden id
    # Load weights
    W = tl.load(
        W_ptr
        + n_id[:, None, None] * hidden_size * n_bits # [N_BLOCK_SIZE, 1, 1]
        + h_id[None, None, :] * n_bits # [1, 1, hidden_size]
        + b_id[None, :, None] # [1, n_bits, 1]
    ) # [N_BLOCK_SIZE, n_bits, hidden_size]
    r = tl.load(r_ptr + n_id[:, None] * hidden_size + h_id) # [N_BLOCK_SIZE, hidden_size]
    m = tl.load(m_ptr + n_id) # [N_BLOCK_SIZE]
    # Compute the integral over the block
    r_grad = tl.zeros((N_BLOCK_SIZE,hidden_size), dtype=tl.float32) # [N_BLOCK_SIZE]
    W_grad = tl.zeros((N_BLOCK_SIZE, hidden_size, n_bits), dtype=tl.float32) # [N_BLOCK_SIZE, hidden_size, n_bits]
    k_id = k_pid * K_BLOCK_SIZE * N_UNROLL + tl.arange(0, K_BLOCK_SIZE) # [K_BLOCK_SIZE] - Input value
    for _ in range(N_UNROLL):
        b = ((k_id[:, None] >> b_id[None, :]) & 1).to(W_ptr.dtype.element_ty) * 2 - 1 # [K_BLOCK_SIZE, n_bits]
        z = tl.dot(b[None].broadcast_to(W.shape[0], b.shape[0], b.shape[1]), W, out_dtype=W_ptr.dtype.element_ty) # [N_BLOCK_SIZE, K_BLOCK_SIZE, hidden_size]
        relu_mask = z > 0 # [N_BLOCK_SIZE, K_BLOCK_SIZE, hidden_size]
        h = tl.where(relu_mask, z, 0) # [N_BLOCK_SIZE, K_BLOCK_SIZE, hidden_size]
        logits = tl.sum(h*r[:, None, :], axis=-1) # [N_BLOCK_SIZE, K_BLOCK_SIZE]
        stable_exp_logit = tl.exp(logits - m[:, None]) # [N_BLOCK_SIZE, K_BLOCK_SIZE]
        r_grad += tl.sum(h*stable_exp_logit[:, :, None], axis=1) # [N_BLOCK_SIZE, hidden_size]
        W_grad += tl.dot(
            (relu_mask.to(r_ptr.dtype.element_ty) * r[:, None, :]).trans(0, 2, 1), # [N_BLOCK_SIZE, hidden_size, K_BLOCK_SIZE]
            b[None, :, :] * stable_exp_logit[:, :, None].to(W_ptr.dtype.element_ty), # [N_BLOCK_SIZE, K_BLOCK_SIZE, n_bits]
            out_dtype=W_ptr.dtype.element_ty
        ) # [N_BLOCK_SIZE, hidden_size, n_bits]
        k_id += K_BLOCK_SIZE
    # Store the results
    tl.atomic_add(r_grad_ptr + n_id[:, None] * hidden_size + h_id, r_grad) # [N_BLOCK_SIZE, hidden_size]
    tl.atomic_add(
        W_grad_ptr
        + n_id[:, None, None] * hidden_size * n_bits # [N_BLOCK_SIZE, 1, 1]
        + h_id[None, :, None] * n_bits # [1, hidden_size, 1]
        + b_id[None, None, :], # [1, 1, n_bits]
        W_grad # [N_BLOCK_SIZE, hidden_size, n_bits]
    )

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
    def backward(ctx, grad_output, block_size: Tuple[int] = (4, 64), n_unroll: int = 64, num_warps: int = 4):
        """
        grad_output: [B, ...]
        """
        W, r, y, lse, m = ctx.saved_tensors
        batch_size, hidden_size, n_bits = W.shape
        n_comb = 2**n_bits
        n_block_size, k_block_size = block_size
        n_n_blocks = batch_size // n_block_size # Number of blocks for the batch dimension
        n_k_blocks = n_comb // (k_block_size * n_unroll) # Number of blocks for the combinatorial dimension
        W_grad = torch.zeros_like(W) # [H, N]
        r_grad = torch.zeros_like(r) # [H]
        sick_nll_kernel_bwd[(n_n_blocks,n_k_blocks)](
            W, r, m, W_grad, r_grad, n_bits, hidden_size, *block_size, n_unroll, num_warps=num_warps
        )
        W_grad, r_grad = compute_dW_dr_from_acc(W_grad, r_grad, W, r, y, lse, grad_output)
        return None, W_grad, r_grad
    
sick_nll_kernel = _sick_nll_kernel.apply