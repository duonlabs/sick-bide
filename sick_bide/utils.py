import torch
import math
import dataclasses

@dataclasses.dataclass
class FloatProperties:
    mantissa_bits: int
    exponent_bits: int
    exponent_bias: int

FLOAT_PROPERTIES = {
    torch.float16: FloatProperties(mantissa_bits=10, exponent_bits=5, exponent_bias=15),
    torch.bfloat16: FloatProperties(mantissa_bits=7, exponent_bits=8, exponent_bias=127),
    torch.float32: FloatProperties(mantissa_bits=23, exponent_bits=8, exponent_bias=127),
    torch.float64: FloatProperties(mantissa_bits=52, exponent_bits=11, exponent_bias=1023),
}

EL_SIZE2INT_DTYPE = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}
EL_SIZE2UINT_DTYPE = {1: torch.uint8, 2: torch.uint16, 4: torch.uint32, 8: torch.uint64}
NBITS2FLOAT_DTYPE = {16: torch.float16, 32: torch.float32, 64: torch.float64}

def compute_log_bucket_width(f):
    # Assuming f is a float16 number represented as a 16-bit integer
    f = f.view(EL_SIZE2INT_DTYPE[f.element_size()])
    # Extract exponent (bits 10-14)
    E = ((f >> 10) & 0x1F).to(torch.float32)
    # Compute the delta_v for each value
    delta_v = torch.zeros(f.shape, device=f.device)
    delta_v[E == 0] = 2 ** (-24)
    delta_v[E == 31] = torch.nan # Undefined for infinity or NaN
    delta_v[(E != 0) & (E != 31)] = 2 ** (E - 25)        

    return delta_v.log()