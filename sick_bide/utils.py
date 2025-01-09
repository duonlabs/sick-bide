import torch
import dataclasses

from enum import Enum, auto

class DTYPECAT(Enum):
    UINT = auto()    # uint8, uint16, uint32, uint64
    INT = auto()     # int8, int16, int32, int64
    FLOAT = auto()   # float16, float32, float64
    BOOL = auto()    # bool
    COMPLEX = auto() # complex64

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

CAT2DTY = {
    DTYPECAT.FLOAT: {torch.float16, torch.bfloat16, torch.float32, torch.float64},
    DTYPECAT.UINT: {torch.uint8, torch.uint16, torch.uint32, torch.uint64},
    DTYPECAT.INT: {torch.int8, torch.int16, torch.int32, torch.int64},
    DTYPECAT.BOOL: {torch.bool},
    DTYPECAT.COMPLEX: {torch.complex64},
}

DTY2CAT = {dtype: cat for cat, dtypes in CAT2DTY.items() for dtype in dtypes}

DTY2ELS = {
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.float32: 4,
    torch.float64: 8,
    torch.uint8: 1,
    torch.uint16: 2,
    torch.uint32: 4,
    torch.uint64: 8,
    torch.int8: 1,
    torch.int16: 2,
    torch.int32: 4,
    torch.int64: 8,
    torch.bool: 1,
    torch.complex64: 8,
}

CAT2ELS2DTY = {cat:dict(map(lambda d: (DTY2ELS[d], d), dtypes)) for cat, dtypes in CAT2DTY.items()}
GENERIC_CAT = DTYPECAT.INT
ELS2GEN = CAT2ELS2DTY[GENERIC_CAT]
DTY2GEN = {dtype: ELS2GEN[DTY2ELS[dtype]] for dtype in DTY2ELS.keys()}

def compute_log_bucket_width(f):
    # Assuming f is a float16 number represented as a 16-bit integer
    f = f.view(DTY2GEN[f.element_size()])
    # Extract exponent (bits 10-14)
    E = ((f >> 10) & 0x1F).to(torch.float32)
    # Compute the delta_v for each value
    delta_v = torch.zeros(f.shape, device=f.device)
    delta_v[E == 0] = 2 ** (-24)
    delta_v[E == 31] = torch.nan # Undefined for infinity or NaN
    delta_v[(E != 0) & (E != 31)] = 2 ** (E - 25)        

    return delta_v.log()