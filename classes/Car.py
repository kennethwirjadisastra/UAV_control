import numpy as np
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


def suspension_force(wheel_displacment, wheel_speed):
    k = 25000  # N/m        restoritive force
    c = 2000   # Ns/m       damping force
    return -k * (wheel_displacment) - c * wheel_speed


def project_and_normalize(vectors, normal):
    proj = vectors - np.outer(np.dot(vectors, normal), normal)
    norms = np.linalg.norm(proj, axis=-1, keepdims=True)
    return proj / norms


def signed_angle(a, b, axis):
    cross = np.cross(a, b)
    dot = np.sum(a * b, axis=-1)
    sign = np.sum(cross * axis, axis=-1)
    return np.arctan2(sign, dot)


# approximate stats from a tesla model 3
class Car(Vehicle):
    def __init__(self, **kwargs):
        kwargs['mass'] = 1610
        super().__init__(**kwargs)

        self.mass                       = 1610                          # mass in kg
        self.wheel_torque               = 750                           # per rear tire
        self.wheel_coef_friction        = 0.9                           # dry coef of friction

        # wheel order [back_left, back_right, front_left, front_right]
        self.wheel_resting_positions    = np.array([                    # displacments from center of mass in meters
                                            [-1.44, -0.81, -0.55], 
                                            [-1.44, 0.81, -0.55], 
                                            [1.44, -0.81, -0.55], 
                                            [1.44, 0.81, -0.55]
                                        ])
        
        self.wheel_attach_positions    = np.array([                     # location where wheels are attached by the suspension to the body
                                            [-1.44, -0.81, 0], 
                                            [-1.44, 0.81, 0], 
                                            [1.44, -0.81, 0], 
                                            [1.44, 0.81, 0]
                                        ])

        self.max_steering_speed         = 0.5                           # radians per second at the front wheels
        self.max_steering_angle         = 0.61                          # radians at the front tire

    def compute_forces_and_moments(self, state, action) -> tuple[np.ndarray, np.ndarray]:
        pos, vel, quat, ang_vel     = state
        rot_mat                     = R.from_quat(quat).as_matrix()
        throttle, steer             = np.clip(action, [0.0, 1.0], [-1.0, 1.0])


        ang_vel_mat                 = np.array([
                                        [0, -ang_vel[2], ang_vel[1]],
                                        [ang_vel[2], 0, -ang_vel[0]],
                                        [-ang_vel[1], ang_vel[0], 0]
                                    ])
        

        # suspension forces
        suspension_axis             = rot_mat @ np.array([0, 0, -1])
        wheel_world_positions       = pos + (rot_mat @ self.wheel_resting_positions.T).T
        wheel_suspension_heights    = wheel_world_positions[:, 2] / (suspension_axis[2] + 1e-6)

        wheel_suspension_speeds     = (vel[2] + (rot_mat @ (ang_vel_mat @ self.wheel_resting_positions.T))[2,:]) / (suspension_axis[2] + 1e-6)

        suspension_forces           = suspension_force(wheel_suspension_heights, wheel_suspension_speeds) * suspension_axis
        force_of_gravity            = np.array([0, 0, -9.81 * self.mass])

        # tire forces
        steer_angle                 = steer * self.max_steering_angle
        coss, sins                  = np.cos(steer_angle), np.sin(steer_angle)
        tire_directions             = rot_mat @ np.array([[1, 0, 0], [1, 0, 0], [coss, sins, 0], [coss, sins, 0]])
        wheel_body_positions        = self.wheel_resting_positions + wheel_suspension_heights * np.array([0, 0, -1])
        tire_velocities             = vel + rot_mat @ (ang_vel_mat @ wheel_body_positions)

        up_dir                      = rot_mat @ np.array([0, 0, 1])
        tire_proj_vels              = project_and_normalize(tire_velocities, up_dir)
        tire_proj_dirs              = project_and_normalize(tire_directions, up_dir)
        slip_angles                 = signed_angle(tire_proj_vels, tire_proj_dirs, up_dir)
        tire_normal_forces          = suspension_forces * suspension_axis[2]
        tire_perp_dirs              = rot_mat @ np.array([[0, 1, 0], [0, 1, 0], [-sins, coss, 0], [-sins, coss, 0]])
        tire_lateral_forces         = tire_normal_forces * pacejka_lateral_force(slip_angles) * tire_perp_dirs

        # throttle forces

        # Assumes rear wheel drive (RWD), force applied only to back wheels
        throttle_tire_force_mag     = throttle*self.wheel_torque*np.array([1.0, 1.0, 0.0, 0.0])  # (4,3)
        
        tire_throttle_forces        = throttle_tire_force_mag * tire_directions     # (4,3)

        forces                      = np.concatenate([suspension_forces, force_of_gravity])
        force_locations             = np.concatenate([rot_mat @ self.wheel_attach_positions, [0, 0, 0]])
        moments                     = np.cross([force_locations, forces])

        return np.sum(forces, axis=0), np.sum(moments, axis=0)
