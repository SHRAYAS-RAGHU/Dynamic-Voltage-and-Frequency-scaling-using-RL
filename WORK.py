import time
s = time.time()

for i in range(10000):
    for _ in  range(1000):
        a = 1

print(time.time() - s, 's')