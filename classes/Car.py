import numpy as np
from matplotlib import pyplot as plt
from classes.Vehicle import Vehicle
from scipy.spatial.transform import Rotation as R


def pacejka_lateral_force(alpha):
    """
    Computes lateral tire force using Pacejka's Magic Formula.
    
    Parameters:
    - alpha: slip angle in radians (can be scalar or np.array)
    - B, C, D, E: Pacejka parameters (stiffness, shape, peak, curvature)
    
    Returns:
    - lateral force F_y
    """

    B = 10.0    # stiffness factor
    C = 1.9     # shape factor
    D = 1.0     # peak factor (max force normalized to 1)
    E = 0.97    # curvature factor

    term = B * alpha
    return D * np.sin(C * np.arctan(term - E * (term - np.arctan(term))))


def suspension_force(wheel_displacment, wheel_speed, resting_compression=9.81*1610/4):
    k = 25000  # N/m        restoritive force
    c = 2000   # Ns/m       damping force
    return -k * (wheel_displacment) - c * wheel_speed - resting_compression


def project_and_normalize(vectors, normal):
    proj = vectors - np.outer(np.dot(vectors, normal), normal)
    norms = np.linalg.norm(proj, axis=-1, keepdims=True)
    return proj / (norms + 1e-6)    # to avoid 0/0 error when velocity is zero


def signed_angle(a, b, axis):
    cross = np.cross(a, b)
    dot = np.sum(a * b, axis=-1)
    sign = np.sum(cross * axis, axis=-1)
    return np.arctan2(sign, dot)


# approximate stats from a tesla model 3
class Car(Vehicle):
    def __init__(self, position: np.ndarray = None, velocity: np.ndarray = None, 
            quaternion: np.ndarray = None, angular_velocity: np.ndarray = None,
            mass: np.ndarray = None, inertia: np.ndarray = None):
        kwargs = {}
        kwargs['position']          = position
        kwargs['velocity']          = velocity
        kwargs['quaternion']        = quaternion
        kwargs['angular_velocity']  = angular_velocity
        kwargs['mass']              = mass if mass else 1610
        kwargs['inertia']           = inertia if inertia else np.diag([550, 3200, 5600])
        
        super().__init__(**kwargs)

        self.wheel_torque               = 750                           # per rear tire
        self.wheel_coef_friction        = 0.9                           # dry coef of friction

        # wheel order [back_left, back_right, front_left, front_right]
        self.wheel_resting_positions    = np.array([                    # displacments from center of mass in meters
                                            [-1.44, -0.81, -0.55], 
                                            [-1.44, 0.81, -0.55], 
                                            [1.44, -0.81, -0.55], 
                                            [1.44, 0.81, -0.55]
                                        ])
        
        self.suspension_attach_pos      = np.array([                     # location where wheels are attached by the suspension to the body
                                            [-1.44, -0.81, 0], 
                                            [-1.44, 0.81, 0], 
                                            [1.44, -0.81, 0], 
                                            [1.44, 0.81, 0]
                                        ])

        self.max_steering_speed         = 0.5                           # radians per second at the front wheels
        self.max_steering_angle         = 0.61                          # radians at the front tire

    def compute_forces_and_moments(self, state, action) -> tuple[np.ndarray, np.ndarray]:
        pos, vel, quat, ang_vel     = state
        print(quat)
        rot_mat                     = R.from_quat(quat, scalar_first=True).as_matrix()
        throttle, steer             = np.clip(action, [-1.0, -1.0], [1.0, 1.0])


        ang_vel_mat                 = np.array([
                                        [0, -ang_vel[2], ang_vel[1]],
                                        [ang_vel[2], 0, -ang_vel[0]],
                                        [-ang_vel[1], ang_vel[0], 0]
                                    ])
        

        # suspension forces
        suspension_axis             = rot_mat @ np.array([0, 0, -1])
        wheel_world_positions       = pos + (rot_mat @ self.wheel_resting_positions.T).T
        wheel_suspension_heights    = wheel_world_positions[:, 2] / (suspension_axis[2] + 1e-6)
        wheel_body_positions        = self.wheel_resting_positions + wheel_suspension_heights[:,None] * np.array([0, 0, -1])[None,:]
        tire_velocities             = vel + (rot_mat @ (ang_vel_mat @ wheel_body_positions.T)).T
        wheel_suspension_speeds     = np.dot(tire_velocities, suspension_axis)

        suspension_mags             = suspension_force(wheel_suspension_heights, wheel_suspension_speeds)
        suspension_forces           = suspension_mags[:,None] * suspension_axis[None,:]

        # tire forces
        steer_angle                 = steer * self.max_steering_angle
        coss, sins                  = np.cos(steer_angle), np.sin(steer_angle)
        tire_directions             = (rot_mat @ np.array([[1, 0, 0], [1, 0, 0], [coss, sins, 0], [coss, sins, 0]]).T).T


        up_dir                      = rot_mat @ np.array([0, 0, 1])
        tire_proj_vels              = project_and_normalize(tire_velocities, up_dir)
        tire_proj_dirs              = project_and_normalize(tire_directions, up_dir)
        slip_angles                 = signed_angle(tire_proj_vels, tire_proj_dirs, up_dir)
        tire_normal_forces          = suspension_mags * suspension_axis[2]
        tire_perp_dirs              = (rot_mat @ np.array([[0, 1, 0], [0, 1, 0], [-sins, coss, 0], [-sins, coss, 0]]).T).T
        tire_lateral_forces         = (tire_normal_forces * pacejka_lateral_force(slip_angles))[:,None] * tire_perp_dirs

        # throttle forces
        # Assumes rear wheel drive (RWD), force applied only to back wheels
        throttle_tire_force_mag     = throttle * self.wheel_torque * np.array([1.0, 1.0, 0.0, 0.0])
        tire_throttle_forces        = throttle_tire_force_mag[:,None] * tire_directions

        # group the forces and compute the moments
        force_of_gravity            = np.array([0, 0, -9.81 * self.mass])[None,:]

        forces                      = np.concatenate([suspension_forces, tire_lateral_forces + tire_throttle_forces, force_of_gravity], axis=0)
        force_locations             = np.concatenate([(rot_mat @ self.suspension_attach_pos.T).T, (rot_mat @ wheel_body_positions.T).T, [[0, 0, 0]]], axis=0)
        moments                     = np.cross(force_locations, forces)

        return np.sum(forces, axis=0), np.sum(moments, axis=0)


###################################
## ---------- testing ---------- ##
###################################

from visualization.FourViewPlot import FourViewPlot
from classes.TargetPath import TargetPath

if __name__ == '__main__':
    # initial state
    position            = np.array([0.0, 0.0, 0.9])
    velocity            = np.array([0, 0, 0])
    quaternion          = np.array([1.0, 0.0, 0.0, 0.0])
    angular_velocity    = np.array([0.3, 0, 0.0])

    car = Car(position, velocity, quaternion, angular_velocity)

    # action plan and delta time
    action_plan = np.ones((1200, 2)) * np.array([0.5, 0.8])[None,:]
    dts = 0.01 * np.ones(1200)

    p, v, q, w, t = car.simulate_trajectory(car.get_state(), action_plan, dts)
    
    # target path
    ts = np.cumsum(dts)
    wx = ts*5
    wy = 5*(1-np.cos(ts))
    wz = ts*0

    waypoints = np.stack([wx, wy, wz])
    targetPath = TargetPath(waypoints)

    fourPlot = FourViewPlot()
    fourPlot.addTrajectory(p, 'Vehicle', color='b')
    fourPlot.addTrajectory(waypoints.T, 'TargetPath', color='g')
    fourPlot.show()


    traj = np.concatenate([p,q], axis=1) # shape (N, 7): [x, y, z, qw, qx, qy, qz]
    header = 'x,y,z,qw,qz,qy,qz'
    np.savetxt('trajectory.csv', traj, delimiter=',')