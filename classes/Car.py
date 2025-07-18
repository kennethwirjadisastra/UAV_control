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

    # Clamp slip angle to avoid numerical issues (optional)
    alpha = np.clip(alpha, -np.pi/2, np.pi/2)
    
    term = B * alpha
    return D * np.sin(C * np.arctan(term - E * (term - np.arctan(term))))


def suspension_force(wheel_displacment, wheel_speed):
    k = 25000  # N/m        restoritive force
    c = 2000   # Ns/m       damping force
    return -k * (wheel_displacment) - c * wheel_speed


# approximate stats from a tesla model 3
class Car(Vehicle):
    def __init__(self, **kwargs):
        kwargs['mass'] = 1610
        super().__init__(**kwargs)

        self.mass                       = 1610                          # mass in kg
        self.wheel_torque               = 750                           # per rear tire
        self.wheel_coef_friction        = 0.9                           # dry coef of friction

        self.wheel_resting_positions    = np.array([                    # displacments from center of mass in meters
                                            [-1.44, -0.81, -0.55], 
                                            [-1.44, 0.81, -0.55], 
                                            [1.44, -0.81, -0.55], 
                                            [1.44, 0.81, -0.55]
                                        ])

        self.max_steering_speed     = 0.5                               # radians per second at the front wheels

    def compute_forces_and_moments(self, state, action) -> tuple[np.ndarray, np.ndarray]:
        pos, vel, quat, ang_vel = state
        rot_mat = R.from_quat(quat).as_matrix()

        suspension_axis         = rot_mat @ np.array([0, 0, 1])
        wheel_world_positions   = pos + self.wheel_positions





        throttle, steer = np.clip(action, [0.0, 1.0], [-1.0, 1.0])


        return force, moment
