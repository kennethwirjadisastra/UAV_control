from fixedWing12ODES import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


XX, YY = np.meshgrid(np.linspace(0, 3, 100), np.linspace(0, 3, 100))
ZZ = np.sin(XX*XX + YY*YY)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(XX, YY, ZZ, cmap='viridis')
plt.show()
