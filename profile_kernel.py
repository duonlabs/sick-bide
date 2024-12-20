import torch
from sick_bide.kernels.precompute.integral import _sick_integral_kernel
from sick_bide.kernels.precompute.nll import _sick_nll_kernel

device = "cuda" if torch.cuda.is_available() else "cpu"


class CustomCTX:
    def __init__(self):
        self.saved_tensors = []
    def save_for_backward(self, *args):
        self.saved_tensors = args

batch_size = 64
n_bits = 16
hidden_factor = 2
hidden_size = n_bits * hidden_factor

W = torch.randn(batch_size, hidden_size, n_bits, device=device, dtype=torch.float16)
r = torch.randn(batch_size, hidden_size, device=device, dtype=torch.float16)
y = torch.randn(batch_size, device=device, dtype=torch.float16)
dy = torch.randn(batch_size, device=device)

ctx = CustomCTX()

_sick_nll_kernel.forward(ctx, y, W, r)
_sick_nll_kernel.backward(ctx, dy)

# print("32bits kernel")
# print("Kernel times:", run_benchmark(sick_integral_kernel, n_bits=32))