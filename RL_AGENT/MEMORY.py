import subprocess as SP
import re
import matplotlib.pyplot as plt
def mem_free():
    a = SP.Popen(['cat', '/proc/meminfo'],\
                    stderr=SP.PIPE, stdout=SP.PIPE, shell=False)
    a = str(a.communicate()[0]).strip('\'b').split('\\n')[:3]
    tot = re.search(r'[\d]+', a[0]).group()
    val = re.search(r'[\d]+', a[2]).group()
    return int(val)/(int(tot)-600000)
"""
m = []
for i in range(1000):
    m.append(mem_free())
plt.plot(list(range(i+1)), m)
plt.show()
"""
