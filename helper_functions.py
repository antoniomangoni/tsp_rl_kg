import time
import functools
import inspect

ENABLE_TIMING = False

def time_function(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"Function {func.__name__!r} in {func.__code__.co_filename.split('/')[-1]} executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper if ENABLE_TIMING else func

def debug_function(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f'Called from {inspect.stack()[1].filename.split("/")[-1]}')
        print(f'Line {inspect.stack()[1].lineno}')
        print(f'Function {inspect.stack()[1].function}')
        print(f'Code: {inspect.stack()[1].code_context}')
        print(f'Index: {inspect.stack()[1].index}')
        print(f'Locals: {inspect.stack()[1].locals}')
        return func(*args, **kwargs)
    return wrapper