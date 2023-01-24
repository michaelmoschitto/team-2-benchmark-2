from time import time
import numpy as np  
  
def timer_func(func,):
    # This function shows the execution time of 
    # the function object passed
    def wrap_func(*args, **kwargs):

        times = []
        for i in range(10):
            t1 = time()
            result = func(*args, **kwargs)
            t2 = time()
            times.append(t2-t1)
        print(f'Function {func.__name__!r} executed in an avg of {(np.mean(times)):.4f}s')
        return result
    return wrap_func