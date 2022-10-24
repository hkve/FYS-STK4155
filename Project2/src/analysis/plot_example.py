import matplotlib.pyplot as plt
import numpy as np

import plot_utils

x = np.linspace(-2*np.pi,2*np.pi, 1000)
fig, ax = plt.subplots()

ax.set_title(r"$V(\gamma)$")
ax.set_ylabel(r"$y$ blabla")
ax.set_xlabel(r"$x$ blabla")

for i in [1,2,3]:
    ax.plot(x, np.sin(x)/i, label=f"blabla {i}")


ax.legend()
plt.show()