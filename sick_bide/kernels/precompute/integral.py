import torch
import triton
import triton.language as tl

from typing import Tuple
from ..utils import combine_softmax_integral_blocks

@triton.jit
def sick_integral_kernel_fwd(
    p1_ptr, p2_ptr, r_ptr, lses_ptr, ms_ptr,
    hidden_size: tl.constexpr, p_size: tl.constexpr,
    P1_BLOCK_SIZE: tl.constexpr,
):
    # Handle ids
    n_pid, p1_pid = tl.program_id(0), tl.program_id(1) # Batch pid, combinatorial pid
    p1_id = p1_pid * P1_BLOCK_SIZE + tl.arange(0, P1_BLOCK_SIZE) # [P1_BLOCK_SIZE] - Input value
    h_id = tl.arange(0, hidden_size) # [hidden_size] - Hidden id
    # Load tensors
    r = tl.load(r_ptr + n_pid * hidden_size + h_id) # [hidden_size]
    p_strided_n = n_pid * p_size * hidden_size
    p1 = tl.load(
        p1_ptr + p_strided_n
        + p1_id[:, None] * hidden_size # [P1_BLOCK_SIZE, 1]
        + h_id[None, :] # [1, hidden_size]
    ) # [P1_BLOCK_SIZE, hidden_size]
    # Compute the integral over the block
    m = -float("inf")
    se = 0.0
    p2_ptr = p2_ptr + p_strided_n + tl.arange(0, hidden_size) # [hidden_size]
    for _ in range(p_size):
        h = tl.maximum(p1 + tl.load(p2_ptr), 0.0) # [P1_BLOCK_SIZE, hidden_size]
        logits = tl.sum(h*r, axis=-1) # [P1_BLOCK_SIZE]
        new_m = tl.maximum(m, tl.max(logits, axis=-1)) # []
        se = se * tl.exp(m - new_m) + tl.sum(tl.exp(logits-new_m), axis=-1) # []
        m = new_m
        p2_ptr += hidden_size
    # Store the results
    n_p1_blocks = p_size // P1_BLOCK_SIZE
    tl.store(lses_ptr + n_pid * n_p1_blocks + p1_pid, tl.log(se))
    tl.store(ms_ptr + n_pid * n_p1_blocks + p1_pid, m)

@torch.compile
def precompute_blocks_fwd(W: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
        W: [B, H, N]
    
        -> [B, 2**(N/2), H], [1, 2**(N/2), H]
    """
    n_bits = W.shape[-1] // 2
    x = torch.arange(2**n_bits, device=W.device) # [2**N] - Generate all possible inputs
    x = ((x.unsqueeze(-1) >> torch.arange(n_bits, device=W.device))&1).to(W.dtype) * 2 - 1 # [2**N, N] - Convert to binary representation
    return x @ W[..., :n_bits].permute(0, 2, 1), x @ W[..., n_bits:].permute(0, 2, 1)

class _sick_integral_kernel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, W: torch.Tensor, r: torch.Tensor, block_size: Tuple[int] = (256,), num_warps: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            W: [B, H, N]
            r: [B, H]

            -> [B]
        """
        # Validate arguments
        assert W.is_cuda and r.is_cuda, "W and r must be on the same device"
        if not W.is_contiguous():
            W = W.contiguous()
        if not r.is_contiguous():
            r = r.contiguous()
        # Compute kernel parameters
        batch_size, hidden_size, n_bits = W.shape
        p_size = 2**(n_bits//2)
        n_p1_blocks = p_size // block_size[0] # Number of blocks for the combinatorial dimension
        # Result tensors
        lses = torch.zeros(batch_size, n_p1_blocks, dtype=torch.float32, device=W.device)
        ms = torch.zeros(batch_size, n_p1_blocks, dtype=torch.float32, device=W.device)
        # Do the work !
        p1, p2 = precompute_blocks_fwd(W) # [B, p_size, H], [1, p_size, H] - Precompute the blocks
        sick_integral_kernel_fwd[(batch_size, n_p1_blocks)]( # Run the kernel
            p1, p2, r, lses, ms,
            hidden_size, p_size,
            *block_size, num_warps=num_warps
        )
        return combine_softmax_integral_blocks(lses, ms)

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError

sick_integral_kernel  = _sick_integral_kernel.apply