import torch

from sick_bide.reference import sick_integral_kernel as sick_integral_reference, sick_nll_kernel as sick_nll_reference
from sick_bide.kernels import sick_integral_kernel, sick_nll_kernel

def test_sick_integral_kernel_forward(W: torch.Tensor, r: torch.Tensor):
    logsumexp_reference, m_reference = sick_integral_reference(W, r)
    logsumexp, m = sick_integral_kernel(W, r)
    torch.testing.assert_close(logsumexp, logsumexp_reference, atol=1e-1, rtol=1e-1)
    torch.testing.assert_close(m, m_reference, atol=1e-1, rtol=1e-1)
    # float16
    logsumexp, m = sick_integral_kernel(W.to(torch.float16), r.to(torch.float16))
    torch.testing.assert_close(logsumexp, logsumexp_reference, atol=1e-1, rtol=1e-1)
    torch.testing.assert_close(m, m_reference, atol=1e-1, rtol=1e-1)


def test_sick_nll_kernel_forward(y: torch.Tensor, W: torch.Tensor, r: torch.Tensor):
    nll_reference = sick_nll_reference(y, W, r)
    nll = sick_nll_kernel(y, W, r)
    torch.testing.assert_close(nll, nll_reference, atol=1e-1, rtol=1e-1)
    # float16
    nll = sick_nll_kernel(y, W.to(torch.float16), r.to(torch.float16))
    torch.testing.assert_close(nll, nll_reference, atol=1e-1, rtol=1e-1)


def test_sick_nll_kernel_backward(y: torch.Tensor, W: torch.Tensor, r: torch.Tensor):
    W.requires_grad_(True), r.requires_grad_(True)
    W.grad, r.grad = None, None
    g = torch.randn_like(y)
    sick_nll_reference(y, W, r).backward(g)
    grad_W_reference = W.grad.clone()
    grad_r_reference = r.grad.clone()
    W.grad, r.grad = None, None
    sick_nll_kernel(y, W, r).backward(g)
    grad_W = W.grad.clone()
    grad_r = r.grad.clone()
    torch.testing.assert_close(grad_W, grad_W_reference, atol=1e-1, rtol=1e-1)
    torch.testing.assert_close(grad_r, grad_r_reference, atol=1e-1, rtol=1e-1)
    # float16
    W.grad, r.grad = None, None
    sick_nll_kernel(y, W.to(torch.float16), r.to(torch.float16)).backward(g)
    grad_W = W.grad.clone()
    grad_r = r.grad.clone()
    torch.testing.assert_close(grad_W, grad_W_reference, atol=5e-1, rtol=5e-1)
    torch.testing.assert_close(grad_r, grad_r_reference, atol=5e-1, rtol=5e-1)