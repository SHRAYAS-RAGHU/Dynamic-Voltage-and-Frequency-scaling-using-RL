import random
a = list(range(10000))

for i in range(10000):
    for j in range(i+1, 10000):
        if a[i] > a[j]:
            a[j], a[i] = a[i], a[j]