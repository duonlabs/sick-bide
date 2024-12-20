import threading
import torch
import triton

from typing import Any, Callable, Dict, Generator, List, Tuple, Union
from concurrent.futures import Future, TimeoutError

def run_benchmark(f, *args, n_bits: int = 16, hidden_factor: int = 2, quantiles=(0.1, 0.5, 0.9)):
    return triton.testing.do_bench(lambda: f(*args),quantiles=quantiles)

def run_with_timeout(func, timeout, *args, **kwargs):
    future = Future()

    def target():
        try:
            result = func(*args, **kwargs)
            if not future.cancelled():  # Check if the Future is still active
                future.set_result(result)
        except Exception as e:
            if not future.cancelled():
                future.set_exception(e)

    thread = threading.Thread(target=target, daemon=True)
    thread.start()

    try:
        return future.result(timeout=timeout)  # Wait for the result with a timeout
    except TimeoutError:
        if thread.is_alive():
            future.cancel()  # Optional: Mark the future as canceled
        raise TimeoutError("Function timed out")

def enumerate_space(space: Dict[str, List[Any]]) -> Generator[Dict[str, Any], None, None]:
    if not space:
        yield {}
        return
    k = next(iter(space.keys()))
    v = space.pop(k)
    for c in enumerate_space(space):
        yield from ({k: x, **c} for x in v)

def compile_and_check_valid(fn, expected, *args, **kwargs):
    actual = fn(*args, **kwargs)
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, atol=5e-1, rtol=5e-1)
    else:
        for e, a in zip(expected, actual):
            torch.testing.assert_close(a, e, atol=5, rtol=1e-1)
    
def tune(
    func: Callable[..., Any],
    space: Dict[str, List[Any]],
    expected: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    *args, **kwargs
):
    configs = {}
    for config in enumerate_space(space):
        print(config)
        fn = func(**config)
        res = [float("inf"), float("inf"), float("inf")]
        try:
            run_with_timeout(compile_and_check_valid, 5, fn, expected, *args, **kwargs)
            try:
                res = run_with_timeout(run_benchmark, 5, fn, *args, **kwargs)
            except TimeoutError:
                print("Time out during execution")
            except Exception as e:
                print("Error during execution:", e)
        except TimeoutError:
            print("Time out during compilation")
        except AssertionError:
            print("Output was incorrect")
        except Exception as e:
            print("Error during compilation:", e)
        print("_sick_integral_kernel_smart times:", res)
        configs[str(config)] = res
    return sorted(configs.items(), key=lambda x: x[1][1])
