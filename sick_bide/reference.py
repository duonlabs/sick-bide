import torch
from typing import Optional, Tuple

from .utils import DTYPECAT, CAT2ELS2DTY

@torch.compile
def sick_integral_kernel(W: torch.Tensor, r: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
        W: [B, H, N]
        r: [B, H]

        -> [B, ...]
    """
    assert W.shape[0] == r.shape[0]
    v = torch.arange(2**W.shape[-1], device=W.device) # [2**N] - Generate all possible inputs
    x = ((v.unsqueeze(-1) >> torch.arange(W.shape[-1], device=W.device))&1).to(W.dtype) * 2 - 1 # [2**N, N] - Convert to binary representation
    logits = (torch.nn.functional.relu(x[None] @ W.permute(0, 2, 1)) @ r[:, :, None]).squeeze(-1) # [B, 2**N] - MLP forward pass
    m = logits.max(-1).values # [B] - Compute the max logit for numerical stability
    logits = logits - m[:, None] # [B, 2**N] - Subtract the max logit for numerical stability
    return torch.logsumexp(logits, dim=-1), m

@torch.compile
def sick_log_cdf_kernel(W: torch.Tensor, r: torch.Tensor, u: torch.Tensor, dtype_cat: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:  # noqa: F821
    """
        W: [B, H, N]
        r: [B, H]
        u: [B, ...]
        dtype_cat: [B]

        -> [B, ...]
    """
    assert W.shape[0] == r.shape[0]
    v = torch.arange(2**W.shape[-1], device=W.device) # [2**N] - Generate all possible inputs
    x = ((v.unsqueeze(-1) >> torch.arange(W.shape[-1], device=W.device))&1).to(W.dtype) * 2 - 1 # [2**N, N] - Convert to binary representation
    logits = (torch.nn.functional.relu(x[None] @ W.permute(0, 2, 1)) @ r[:, :, None]).squeeze(-1) # [B, 2**N] - MLP forward pass
    m = logits.max(-1, keepdim=True).values # [B, 1] - Compute the max logit for numerical stability
    logits = logits - m # [B, 2**N] - Subtract the max logit for numerical stability
    lse = torch.logsumexp(logits, dim=-1, keepdim=True) # [B] - Compute the log-sum-exp
    # Mask Values that are superior to u
    mask = logits.new_zeros(u.shape + v.shape, dtype=torch.bool) # [B, 2**N] - Mask for the CDF
    u, u_es = u.unsqueeze(-1), u.element_size()
    v, dtype_cat = v.to(CAT2ELS2DTY[DTYPECAT.UINT][u_es]), dtype_cat[..., *((None,)*(len(u.shape)-len(dtype_cat.shape)))] # [B, 2**N], [B]
    # mask |= (dtype_cat == DTYPECAT.UINT.value) & (v.view(dt:=CAT2ELS2DTY[DTYPECAT.UINT][u_es]) <= u.view(dt)) # [B, 2**N] - UINTs /!\ RuntimeError: "compare_cuda" not implemented for 'UInt16'
    mask |= (dtype_cat == DTYPECAT.INT.value) & (v.view(dt:=CAT2ELS2DTY[DTYPECAT.INT][u_es]) <= u.view(dt)) # [B, 2**N] - INTs
    mask |= (dtype_cat == DTYPECAT.FLOAT.value) & (v.view(dt:=CAT2ELS2DTY[DTYPECAT.FLOAT][u_es]) <= u.view(dt)) & torch.isfinite(u.view(dt)) # [B, 2**N] - FLOATs
    logits -= torch.finfo(logits.dtype).tiny # We want all logits to be < 0, not <=0
    log_cdfs = torch.logsumexp( # Compute masked integral normalized by the log-sum-exp
        logits.view(logits.shape[0], *(1,)*(len(u.shape)-2),logits.shape[-1]) # [B, ..., 2**N]
        * torch.where(mask, 1.0, float("inf")) # [B, ..., 2**N]
    , dim=-1) - lse.view(lse.shape[0], *(1,)*(len(u.shape)-2)) # [B]
    return log_cdfs

@torch.compile
def sick_log_prob_kernel(x: torch.Tensor, W: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """
        x: [B]
        W: [H, N]
        r: [H]

        -> [B]
    """
    assert W.shape[0] == r.shape[0]
    x = ((x.unsqueeze(-1).view(CAT2ELS2DTY[DTYPECAT.UINT][x.element_size()]).long() >> torch.arange(W.shape[1], device=W.device))&1).to(W.dtype) * 2 - 1 # [B, N]
    logits = torch.nn.functional.relu(x @ W.T) @ r # [B]
    log_sum_exp, max_logit = sick_integral_kernel(W, r) # [], []
    return logits - max_logit - log_sum_exp # [B]

@torch.compile
def sick_nll_kernel(y: torch.Tensor, W: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """
        y: [B, ...]
        W: [B, H, N]
        r: [B, H]

        -> [B, ...]
    """
    x = torch.arange(2**W.shape[-1], device=W.device) # [2**N] - Generate all possible inputs
    x = ((x.unsqueeze(-1) >> torch.arange(W.shape[-1], device=W.device))&1).to(W.dtype) * 2 - 1 # [2**N, N] - Convert to binary representation
    logits = (torch.nn.functional.relu(x[None] @ W.permute(0, 2, 1)) @ r[:, :, None]).squeeze(-1) # [B, 2**N] - MLP forward pass
    loss = torch.nn.functional.cross_entropy(
        logits.expand(*logits.shape[:2], *y.shape[1:]), # [B, 2**N, ...]
        y.view(CAT2ELS2DTY[DTYPECAT.UINT][y.element_size()]).long(), # [B, ...] - Interpret y as indices
        reduction="none" # Do not reduce for consistency with the kernel
    )
    return loss
