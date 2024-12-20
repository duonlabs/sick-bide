import pytest
import torch
import lovely_tensors as lt

from sick_bide.utils import NBITS2FLOAT_DTYPE
torch.set_printoptions(sci_mode=False)

lt.monkey_patch()

@pytest.fixture
def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

@pytest.fixture
def n_bits() -> int:
    return 16

@pytest.fixture
def hidden_factor() -> int:
    return 2

@pytest.fixture
def hidden_size(n_bits: int, hidden_factor: int) -> int:
    return n_bits * hidden_factor

@pytest.fixture
def batch_size() -> int:
    return 64

@pytest.fixture
def y(batch_size: int, n_bits: int, device: str) -> torch.Tensor:
    return torch.randn(batch_size, device=device).to(NBITS2FLOAT_DTYPE[n_bits])

@pytest.fixture
def W(batch_size: int, hidden_size: int, n_bits: int, device: str) -> torch.Tensor:
    return torch.randn(batch_size, hidden_size, n_bits, device=device)

@pytest.fixture
def r(batch_size: int, hidden_size: int, device: str) -> torch.Tensor:
    return torch.randn(batch_size, hidden_size, device=device)