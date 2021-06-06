import matplotlib.pyplot as plt
import numpy as np
ratio = np.array(list(range(0,100)))
ratio_rew = (ratio <= 20) * (-1.11 * ratio ** 2 + 22.22 * ratio - 101.1) \
                    + (ratio >= 21) * (-100-ratio)

plt.plot(ratio, ratio_rew)
plt.grid(True)
plt.show()