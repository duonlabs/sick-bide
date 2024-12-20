import torch

from sick_bide.kernels import sick_nll_kernel
from sick_bide.reference import sick_log_prob_kernel, sick_log_cdf_kernel
# from sick_bide.reference import sick_nll_kernel, sick_log_prob_kernel, sick_log_cdf_kernel

from sick_bide.utils import compute_log_bucket_width


class BIDE(torch.nn.Module):
    """ Binary Implicit Distribution Encoding. The BIDE layer maps arbitrary values to logits leveraging their binary representation. """
    def __init__(self, W: torch.Tensor, r: torch.Tensor):
        """
            W: [B, H, N]
            r: [B, H]
        """
        super().__init__()
        self.batch_size, self.hidden_size, self.n_bits = W.shape
        # The SICK layer is a simple 1 hidden layer MLP with a single output
        self.W = W
        self.r = r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            x: [B, ...]
        """
        assert x.element_size() == self.n_bits // 8, f"Input tensor must have {self.n_bits} bits elements"
        log_prob = sick_log_prob_kernel(x, self.W, self.r)
        return log_prob
    
    def log_density(self, x: torch.Tensor) -> torch.Tensor:
        """
            x: [B, ...]
        """
        log_prob = self.forward(x)
        log_bucket_width = compute_log_bucket_width(x)
        log_density = log_prob - log_bucket_width
        return log_density
    
    def log_cdf(self, x: torch.Tensor) -> torch.Tensor:
        """
            x: [B, ...]
        """
        log_cdf = sick_log_cdf_kernel(self.W.to(torch.float16), self.r.to(torch.float16), x).to(torch.float32)
        return log_cdf
    
    def nll(self, y: torch.Tensor) -> torch.Tensor:
        """
            x: [B, ...]
        """
        nll = sick_nll_kernel(y, self.W.to(torch.float16), self.r.to(torch.float16)).to(torch.float32)
        return nll.mean()