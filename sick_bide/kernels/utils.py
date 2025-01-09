import torch

from ..utils import DTYPECAT, CAT2ELS2DTY

@torch.compile
def combine_softmax_integral_blocks(lses: torch.Tensor, ms: torch.Tensor) -> torch.Tensor:
    m = ms.max(-1).values
    lse = torch.logsumexp((ms - m[:, None]) + lses, -1)
    return lse, m

@torch.compile
def value_to_binary_representation(value: torch.Tensor, n_bits: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
        value: [B]
        n_bits: int
    """
    return ((value.unsqueeze(-1).view(CAT2ELS2DTY[DTYPECAT.UINT][value.element_size()]).long() >> torch.arange(n_bits, device=value.device))&1).to(dtype) * 2 - 1 # [B, N]

@torch.compile
def bide_logits(W: torch.Tensor, r: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
        W: [B, H, N]
        r: [B, H]
        v: [B]
    """
    x = value_to_binary_representation(v, W.shape[-1], W.dtype) # [B, N]
    logits = (torch.nn.functional.relu(x[:, None, :] @ W.permute(0, 2, 1)) @ r[..., None]).squeeze(-1).squeeze(-1) # [B]
    return logits