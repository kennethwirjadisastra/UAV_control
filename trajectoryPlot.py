from planeODES import *              # keep if you need it later
import numpy as np
import pyvista as pv                       # pip install pyvista

# simulate a flight path
params = {
    "mass": 1200.0,
    "I": (800.0, 1200.0, 1000.0),
    "throttle": 0.6,
    "delta_e": -0.05,  # small pitch-up
    "delta_a":  0.00,
    "delta_r":  0.00,
    "Tmax": 2000.0
}

# Initial state (rest, level, 1000 m altitude)
state0 = np.zeros(12)
state0[11] = -1000.0  # z_d (down is positive)

t, y = simulate(state0, t_final=10.0, dt=0.01, params=params)

trajX = y[:, 9]
trajY = y[:, 10]
trajZ = -y[:, 11]

points = np.column_stack((trajX, trajY, trajZ))

# Define line connectivity (each point connects to the next)
lines = np.hstack(([len(points)], np.arange(len(points))))

# Create PolyData for the trajectory
trajectory = pv.PolyData()
trajectory.points = points
trajectory.lines = lines

# Plot
plotter = pv.Plotter()
plotter.add_mesh(trajectory, color="navy", line_width=2, label="Plane Trajectory")
plotter.add_axes()
plotter.add_legend()
plotter.show()