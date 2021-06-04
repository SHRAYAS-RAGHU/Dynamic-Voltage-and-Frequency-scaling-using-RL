import pandas as pd

a = pd.read_csv(r'/home/pi/Desktop/PROJECT/RL_AGENT/STAT.csv')

x = a.iloc[:, 0]
temp = a.TEMP
AVG_UTIL = a.AVG_UTIL
FREQ = a.FREQ / 1000
UTIL = a.UTIL

import matplotlib.pyplot as plt

plt.subplot(311)
plt.plot(x, temp)
plt.plot(x, AVG_UTIL)
plt.subplot(312)
plt.plot(x, FREQ)
plt.subplot(313)
plt.plot(x, UTIL)
plt.show()