from fixedWing12ODES import *              # keep if you need it later
import numpy as np
import pyvista as pv                       # pip install pyvista

# ----- data -------------------------------------------------------------
x = np.linspace(0, 3, 100)
y = np.linspace(0, 3, 100)
xx, yy = np.meshgrid(x, y)
zz = np.sin(xx**2 + yy**2)

# PyVista wants a StructuredGrid for a regular surface
grid = pv.StructuredGrid(xx, yy, zz)

# ----- plot -------------------------------------------------------------
plotter = pv.Plotter()
plotter.add_mesh(grid, cmap="viridis", smooth_shading=True)
plotter.add_axes()                         # interactive triad
plotter.set_background("white")            # optional
plotter.show()                             # Z‑axis stays upright
