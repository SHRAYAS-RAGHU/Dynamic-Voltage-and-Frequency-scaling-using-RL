import random
import time
from cpufreq import cpuFreq
cpu = cpuFreq()

def loop():
    start = time.time()
    for i in range(1000):
        for i in range(10000):
            pass
    end =  time.time()
    return f'{(end - start):.4f}'
govnr = cpu.available_governors

for g in govnr:
    cpu.set_governors(g)
    print(f'GOVERNOR : {g.upper()}    EXECUTION_TIME : {loop()} S     FREQ : {cpu.get_frequencies()[0]} Mhz')
cpu.set_governors('ondemand')
