from numpy.lib import utils
import pandas as pd

a = pd.read_csv(r'/home/pi/Desktop/PROJECT/RL_AGENT/STAT_new.csv')
#print(a.head())
x = a.iloc[:, 0]
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
#plt.plot(x, AVG_UTIL)
plt.subplot(412)
plt.plot(x, FREQ)
plt.ylabel('FREQ')
plt.subplot(413)
plt.plot(x, UTIL)
plt.ylabel('UTIL')
plt.subplot(414)
plt.plot(x, rew)
plt.ylabel('rew')
#"""
plt.show()