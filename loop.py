import random
import time
start = time.time()
a = list(range(100000))
random.shuffle(a)
a = sorted(a)
for i in range(100000):
    pass
end =  time.time()
print(f'{(end - start):.4f}')