import time
import random

a = list(range(1000000))
random.shuffle(a)

def loop(x):
    start = time.time()
    b = sorted(x)
    for i in range(1000):
        for i in range(10000):
            pass
    end =  time.time()
    return f'{(end - start):.4f}'
loop(a)