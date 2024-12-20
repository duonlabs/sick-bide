import torch
import triton
from sick_bide.reference import sick_integral_kernel as sick_integral_reference, sick_nll_kernel as sick_nll_reference
from sick_bide.kernels.bruteforce.integral import _sick_integral_kernel as _sick_integral_kernel_bruteforce
from sick_bide.kernels.bruteforce.nll import _sick_nll_kernel as _sick_nll_kernel_bruteforce
from sick_bide.kernels.precompute.integral import _sick_integral_kernel as _sick_integral_kernel_precompute
from sick_bide.kernels.precompute.nll import _sick_nll_kernel as _sick_nll_kernel_precompute

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision("medium")

class CustomCTX:
    def __init__(self):
        self.saved_tensors = []
    def save_for_backward(self, *args):
        self.saved_tensors = args

def run_benchmark(f, *args, n_bits: int = 16, hidden_factor: int = 2, quantiles=(0.1, 0.5, 0.9)):
    return triton.testing.do_bench(lambda: f(*args),quantiles=quantiles)

n_bits = 16
hidden_factor = 2
batch_size = 64
hidden_size = n_bits * hidden_factor
W = torch.randn(batch_size, hidden_size, n_bits, device=device)
r = torch.randn(batch_size, hidden_size, device=device)
y = torch.randn(batch_size, device=device, dtype=torch.float16)
dy = torch.randn(batch_size, device=device)
ctx = CustomCTX()
print("sick_integral_reference times:", run_benchmark(sick_integral_reference, W, r))
print("_sick_integral_kernel_bruteforce times:", run_benchmark(_sick_integral_kernel_bruteforce.forward, ctx, W, r))
print("_sick_integral_kernel_precompute times:", run_benchmark(_sick_integral_kernel_precompute.forward, ctx, W, r))
print("sick_integral_reference f16 times:", run_benchmark(sick_integral_reference, W.to(torch.float16), r.to(torch.float16)))
print("sick_integral_kernel_bruteforce f16 times:", run_benchmark(_sick_integral_kernel_bruteforce.forward, ctx, W.to(torch.float16), r.to(torch.float16)))
print("sick_integral_kernel_precompute f16 times:", run_benchmark(_sick_integral_kernel_precompute.forward, ctx, W.to(torch.float16), r.to(torch.float16)))
print("sick_nll_reference times:", run_benchmark(sick_nll_reference, y, W, r))
print("sick_nll_kernel_bruteforce times:", run_benchmark(_sick_nll_kernel_bruteforce.forward, ctx, y, W, r))
print("sick_nll_kernel_precompute times:", run_benchmark(_sick_nll_kernel_precompute.forward, ctx, y, W, r))
print("sick_nll_reference f16 times:", run_benchmark(sick_nll_reference, y.to(torch.float16), W.to(torch.float16), r.to(torch.float16)))
print("sick_nll_kernel_bruteforce f16 times:", run_benchmark(_sick_nll_kernel_bruteforce.forward, ctx, y.to(torch.float16), W.to(torch.float16), r.to(torch.float16)))
print("sick_nll_kernel_precompute f16 times:", run_benchmark(_sick_nll_kernel_precompute.forward, ctx, y.to(torch.float16), W.to(torch.float16), r.to(torch.float16)))
print("sick_nll_kernel_bruteforce backwards times:", run_benchmark(_sick_nll_kernel_bruteforce.backward, ctx, dy))
print("sick_nll_kernel_compute_precompute backwards times:", run_benchmark(_sick_nll_kernel_precompute.backward, ctx, dy))
print("sick_nll_kernel_bruteforce backwards f16 times:", run_benchmark(_sick_nll_kernel_bruteforce.backward, ctx, dy.to(torch.float16)))
print("sick_nll_kernel_compute_precompute backwards f16 times:", run_benchmark(_sick_nll_kernel_precompute.backward, ctx, dy.to(torch.float16)))
print("LLM batching")
batch_size = 32*512
W = torch.randn(batch_size, hidden_size, n_bits, device=device)
r = torch.randn(batch_size, hidden_size, device=device)
y = torch.randn(batch_size, device=device, dtype=torch.float16)
dy = torch.randn(batch_size, device=device)
print("sick_integral_kernel_bruteforce times:", run_benchmark(_sick_integral_kernel_bruteforce.forward, ctx, W, r))
print("sick_integral_kernel_precompute times:", run_benchmark(_sick_integral_kernel_precompute.forward, ctx, W, r))
print("sick_integral_kernel_bruteforce f16 times:", run_benchmark(_sick_integral_kernel_bruteforce.forward, ctx, W.to(torch.float16), r.to(torch.float16)))
print("sick_integral_kernel_precompute f16 times:", run_benchmark(_sick_integral_kernel_precompute.forward, ctx, W.to(torch.float16), r.to(torch.float16)))
print("sick_nll_kernel_bruteforce times:", run_benchmark(_sick_nll_kernel_bruteforce.forward, ctx, y, W, r))
print("sick_nll_kernel_precompute times:", run_benchmark(_sick_nll_kernel_precompute.forward, ctx, y, W, r))
print("sick_nll_kernel_bruteforce f16 times:", run_benchmark(_sick_nll_kernel_bruteforce.forward, ctx, y.to(torch.float16), W.to(torch.float16), r.to(torch.float16)))
print("sick_nll_kernel_precompute f16 times:", run_benchmark(_sick_nll_kernel_precompute.forward, ctx, y.to(torch.float16), W.to(torch.float16), r.to(torch.float16)))
print("sick_nll_kernel_bruteforce backwards times:", run_benchmark(_sick_nll_kernel_bruteforce.backward, ctx, dy))
print("sick_nll_kernel_compute_precompute backwards times:", run_benchmark(_sick_nll_kernel_precompute.backward, ctx, dy))
print("sick_nll_kernel_bruteforce backwards f16 times:", run_benchmark(_sick_nll_kernel_bruteforce.backward, ctx, dy.to(torch.float16)))
print("sick_nll_kernel_compute_precompute backwards f16 times:", run_benchmark(_sick_nll_kernel_precompute.backward, ctx, dy.to(torch.float16)))