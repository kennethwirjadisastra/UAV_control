import bpy
import numpy as np

# Load trajectory
traj = np.loadtxt('trajectory.csv', delimiter=',')  # shape (N, 7): [x, y, z, qx, qy, qz, qw]

car = bpy.data.objects["Car"]  # your imported car object

for i, (x, y, z, qx, qy, qz, qw) in enumerate(traj):
    car.location = (x, y, z)
    car.rotation_mode = 'QUATERNION'
    car.rotation_quaternion = (qw, qx, qy, qz)
    car.keyframe_insert(data_path="location", frame=i)
    car.keyframe_insert(data_path="rotation_quaternion", frame=i)
