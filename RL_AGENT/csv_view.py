import pandas as pd

a = pd.read_csv(r'/home/pi/Desktop/PROJECT/RL_AGENT/New_training/FROM_SAMPLE_DATA_LOG/JULY_04_15_49_.csv')
"""
all_files = [r'/home/pi/Desktop/PROJECT/RL_AGENT/STAT_JUN_7_00_30_.csv',r'/home/pi/Desktop/PROJECT/RL_AGENT/STAT_JUN_7_00_44_.csv']

li = []

for filename in all_files:

    df = pd.read_csv(filename, index_col=None, header=0)

    li.append(df)

a = pd.concat(li, axis=0, ignore_index=True)
"""
a['index'] = [i for i in range(0, len(a.values))]
print(a.columns)
x = a.index
temp = a.TEMP
AVG_UTIL = a.AVG_UTIL
FREQ = a.FREQ / 1000
UTIL = a.UTIL
rew = a.REW
import matplotlib.pyplot as plt

#plt.plot(x, FREQ)
#"""
plt.subplot(411)
plt.plot(x, temp)
plt.ylabel('temp')
plt.subplot(412)
plt.plot(x, FREQ)
plt.ylabel('FREQ')
plt.subplot(413)
plt.plot(x, AVG_UTIL)
plt.ylabel('AVG_UTIL')
plt.subplot(414)
plt.plot(x, rew)
plt.ylabel('rew')

plt.show()