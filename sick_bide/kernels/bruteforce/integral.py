import torch
import triton
import triton.language as tl

from typing import Tuple
from ..utils import combine_softmax_integral_blocks

@triton.jit
def sick_integral_kernel_fwd(
    W_ptr, r_ptr, lses_ptr, ms_ptr,
    n_bits: tl.constexpr, hidden_size: tl.constexpr,
    N_BLOCK_SIZE: tl.constexpr, K_BLOCK_SIZE: tl.constexpr, N_UNROLL: tl.constexpr
):
    # Handle ids
    n_pid, k_pid = tl.program_id(0), tl.program_id(1) # Batch pid, combinatorial pid
    n_id = n_pid * N_BLOCK_SIZE + tl.arange(0, N_BLOCK_SIZE) # [N_BLOCK_SIZE] - Batch id
    b_id = tl.arange(0, n_bits) # [n_bits] - Bit id
    h_id = tl.arange(0, hidden_size) # [hidden_size] - Hidden id
    k_id = k_pid * K_BLOCK_SIZE * N_UNROLL + tl.arange(0, K_BLOCK_SIZE) # [K_BLOCK_SIZE] - Input value
    # Load weights
    W = tl.load(
        W_ptr
        + n_id[:, None, None] * hidden_size * n_bits # [N_BLOCK_SIZE, 1, 1]
        + h_id[None, None, :] * n_bits # [1, 1, hidden_size]
        + b_id[None, :, None] # [1, n_bits, 1]
    ) # [N_BLOCK_SIZE, n_bits, hidden_size]
    r = tl.load(r_ptr + n_id[:, None] * hidden_size + h_id) # [N_BLOCK_SIZE, hidden_size]
    # Compute the integral over the block
    m = tl.full((N_BLOCK_SIZE,), -float("inf"), dtype=tl.float32) # [N_BLOCK_SIZE]
    se = tl.zeros((N_BLOCK_SIZE,), dtype=tl.float32) # [N_BLOCK_SIZE]
    for _ in range(N_UNROLL):
        b = ((k_id[:, None] >> b_id[None, :]) & 1).to(W_ptr.dtype.element_ty) * 2 - 1 # [K_BLOCK_SIZE, n_bits]
        z = tl.dot(b[None].broadcast_to(W.shape[0], b.shape[0], b.shape[1]), W, out_dtype=W_ptr.dtype.element_ty) # [N_BLOCK_SIZE, K_BLOCK_SIZE, hidden_size]
        h = tl.maximum(z, 0) # [N_BLOCK_SIZE, K_BLOCK_SIZE, hidden_size]
        logits = tl.sum(h*r[:, None, :], axis=-1) # [N_BLOCK_SIZE, K_BLOCK_SIZE]
        new_m = tl.maximum(m, tl.max(logits, axis=-1)) # [N_BLOCK_SIZE]
        se = se * tl.exp(m - new_m) + tl.sum(tl.exp(logits-new_m[:, None]), axis=-1)
        m = new_m
        k_id += K_BLOCK_SIZE
    # Store the results
    tl.store(lses_ptr + n_id * tl.num_programs(1) + k_pid, tl.log(se))
    tl.store(ms_ptr + n_id * tl.num_programs(1) + k_pid, m)

class _sick_integral_kernel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, W: torch.Tensor, r: torch.Tensor, block_size: Tuple[int] = (8, 16), n_unroll: int = 128, num_warps: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            W: [B, H, N]
            r: [B, H]

            -> [B]
        """
        n_block_size, k_block_size = block_size
        batch_size, hidden_size, n_bits = W.shape
        n_comb = 2**n_bits # Number of possible combinations
        n_n_blocks = batch_size // n_block_size # Number of blocks for the batch dimension
        n_k_blocks = n_comb // (k_block_size * n_unroll) # Number of blocks for the combinatorial dimension
        lses = torch.zeros(batch_size, n_k_blocks, dtype=torch.float32, device=W.device)
        ms = torch.zeros(batch_size, n_k_blocks, dtype=torch.float32, device=W.device)
        assert W.is_cuda and r.is_cuda, "W and r must be on the same device"
        if not W.is_contiguous():
            W = W.contiguous()
        if not r.is_contiguous():
            r = r.contiguous()
        # Call the kernel
        # sick_integral_kernel_fwd[(1,1)](
        sick_integral_kernel_fwd[(n_n_blocks,n_k_blocks)](
            W, r, lses, ms, n_bits, hidden_size, *block_size, n_unroll, num_warps=num_warps
        )
        return combine_softmax_integral_blocks(lses, ms)

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError

sick_integral_kernel  = _sick_integral_kernel.apply