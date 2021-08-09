import numpy as np
from collections import defaultdict
from cpufreq import cpuFreq

cpu = cpuFreq()
d = defaultdict(list)

from CPU_LOAD import measure_load
from MEMORY import mem_free

for i in range(100):
    d['util'].append(measure_load())
    d['frq'].append(cpu.get_frequencies()[0])
    d['memfree'].append(mem_free())

import pandas as pd
d = pd.DataFrame(d)

import matplotlib.pyplot as plt

plt.subplot(3,1,1)
plt.plot(d.util)

plt.subplot(3,1,2)
plt.plot(d.frq)

plt.subplot(3,1,3)
plt.plot(d.memfree)
plt.show()