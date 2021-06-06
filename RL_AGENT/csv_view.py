from numpy.lib import utils
import pandas as pd

one = pd.read_csv(r'/home/pi/Desktop/PROJECT/RL_AGENT/STAT__JUN_6_14_00_.csv')
two = pd.read_csv(r'/home/pi/Desktop/PROJECT/RL_AGENT/STAT__JUN_6_15_26_.csv')
three = pd.read_csv(r'/home/pi/Desktop/PROJECT/RL_AGENT/STAT__JUN_6_15_41_.csv')
l = [one, two, three]
for a in l:
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
    plt.subplot(412)
    plt.plot(x, FREQ)
    plt.ylabel('FREQ')
    plt.subplot(413)
    plt.plot(x, AVG_UTIL)
    plt.ylabel('AVG_UTIL')
    plt.subplot(414)
    plt.plot(x, rew)
    plt.ylabel('rew')
    #"""
    plt.show()