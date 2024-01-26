import time
import functools

def time_function(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"Function {func.__name__!r} in {func.__code__.co_filename.split('/')[-1]} executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper
