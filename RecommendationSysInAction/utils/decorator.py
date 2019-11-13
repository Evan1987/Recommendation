
import time
from typing import Callable


def profiler(name: str, repeats: int):
    def wrapper(func: Callable):
        def inner_wrapper(*args, **kwargs):
            res = None
            tic = time.time()
            for _ in range(repeats):
                res = func(*args, **kwargs)
            toc = time.time()
            print(f"{name} cost: {(toc - tic) / repeats} sec")
            return res
        return inner_wrapper
    return wrapper



