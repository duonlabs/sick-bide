import torch
import triton
import argparse

from functools import partial
from sick_bide.tune import tune

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision("medium")

class CustomCTX:
    def __init__(self):
        self.saved_tensors = []
    def save_for_backward(self, *args):
        self.saved_tensors = args

n_bits = 16
hidden_factor = 2
batch_size = 1024
hidden_size = n_bits * hidden_factor
reference_batch_size = 128
W = torch.randn(batch_size, hidden_size, n_bits, device=device, dtype=torch.float16)
r = torch.randn(batch_size, hidden_size, device=device, dtype=torch.float16)
y = torch.randn(batch_size, device=device, dtype=torch.float16)
dy = torch.randn(batch_size, device=device)
ctx = CustomCTX()

parser = argparse.ArgumentParser(description="Kernel selection for tune.py")
parser.add_argument("--kernel", type=str, required=True, choices=["bruteforce_forward", "precompute_forward", "bruteforce_backward", "precompute_backward"], help="Select the kernel to run")
args = parser.parse_args()

def get_expected_values_forward():
    from sick_bide.reference import sick_integral_kernel as sick_integral_reference
    expected_lse = torch.zeros(batch_size, device=device, dtype=torch.float32)
    expected_m = torch.zeros(batch_size, device=device, dtype=torch.float32)
    for i in range(batch_size//reference_batch_size):
        i_start, i_end = i*reference_batch_size, (i+1)*reference_batch_size
        expected_lse[i_start:i_end], expected_m[i_start:i_end] = sick_integral_reference(W[i_start:i_end], r[i_start:i_end])
    return expected_lse, expected_m

def get_expected_values_backward():
    from sick_bide.reference import sick_nll_kernel as sick_nll_reference
    W_grad_expected = torch.zeros_like(W)
    r_grad_expected = torch.zeros_like(r)
    for i in range(batch_size//reference_batch_size):
        i_start, i_end = i*reference_batch_size, (i+1)*reference_batch_size
        W.grad, r.grad = None, None
        sick_nll_reference(y[i_start:i_end], W[i_start:i_end], r[i_start:i_end]).backward(dy[i_start:i_end])
        W_grad_expected[i_start:i_end] = W.grad[i_start:i_end]
        r_grad_expected[i_start:i_end] = r.grad[i_start:i_end]
    return W_grad_expected, r_grad_expected

def bruteforce_forward():
    from sick_bide.kernels.bruteforce.integral import _sick_integral_kernel as _sick_integral_kernel_bruteforce
    expected_lse, expected_m = get_expected_values_forward()
    return tune(
        lambda **args: partial(_sick_integral_kernel_bruteforce.forward, block_size=(args["n_block_size"], args["k_block_size"]), n_unroll=args["n_unroll"], num_warps=args["num_warps"]),
        {
            "n_block_size": [4, 8, 16],
            "k_block_size": [16, 64, 128],
            "n_unroll": [64, 128, 256],
            "num_warps": [2, 4, 8]
        },
        (expected_lse, expected_m),
        ctx, W, r
    )

def precompute_forward():
    from sick_bide.kernels.precompute.integral import _sick_integral_kernel as _sick_integral_kernel_precompute
    expected_lse, expected_m = get_expected_values_forward()
    return tune(
        lambda **args: partial(_sick_integral_kernel_precompute.forward, block_size=(args["p1_block_size"],args["p2_block_size"]), num_warps=args["num_warps"]),
        {
            "p1_block_size": [64, 128, 256],
            "p2_block_size": [8, 16, 32, 64],
            "num_warps": [2, 4]
        },
        (expected_lse, expected_m),
        ctx, W, r
    )

def bruteforce_backward():
    from sick_bide.kernels.bruteforce.nll import _sick_nll_kernel as _sick_nll_kernel_bruteforce
    W.requires_grad_(), r.requires_grad_()
    W_grad_expected, r_grad_expected = get_expected_values_backward()
    W.grad, r.grad = None, None
    _sick_nll_kernel_bruteforce.forward(ctx, y, W, r)
    return tune(
        lambda **args: partial(_sick_nll_kernel_bruteforce.backward, block_size=(args["n_block_size"], args["k_block_size"]), n_unroll=args["n_unroll"], num_warps=args["num_warps"]),
        {
            "n_block_size": [2, 4, 8],
            "k_block_size": [64, 128, 256],
            "n_unroll": [32, 64, 128],
            "num_warps": [2, 4, 8]
        },
        (None, W_grad_expected, r_grad_expected),
        ctx, dy
    )

def precompute_backward():
    from sick_bide.kernels.precompute.nll import _sick_nll_kernel as _sick_nll_kernel_precompute
    W.requires_grad_(), r.requires_grad_()
    W_grad_expected, r_grad_expected = get_expected_values_backward()
    W.grad, r.grad = None, None
    _sick_nll_kernel_precompute.forward(ctx, y, W, r)
    return tune(
        lambda **args: partial(_sick_nll_kernel_precompute.backward, block_size=(args["p1_block_size"], args["p2_block_size"]), num_warps=args["num_warps"]),
        {
            "p1_block_size": [8, 16, 32, 64, 128, 256],
            "p2_block_size": [1, 2, 4, 8],
            "num_warps": [2, 4]
        },
        (None, W_grad_expected, r_grad_expected),
        ctx, dy
    )

kernel_functions = {
    "bruteforce_forward": bruteforce_forward,
    "precompute_forward": precompute_forward,
    "bruteforce_backward": bruteforce_backward,
    "precompute_backward": precompute_backward
}

res = kernel_functions[args.kernel]()

for k, v in res[:10]:
    print(k, v)
