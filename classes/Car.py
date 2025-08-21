import torch as pt
import numpy as np
from matplotlib import pyplot as plt
from classes.Vehicle import Vehicle, StateTensor


def pacejka_lateral_force(alpha):
    """
    Computes lateral tire force using Pacejka's Magic Formula.
    
    Parameters:
    - alpha: slip angle in radians (can be scalar or tensor)
    - B, C, D, E: Pacejka parameters (stiffness, shape, peak, curvature)
    
    Returns:
    - lateral force F_y
    """

    B = 10.0    # stiffness factor
    C = 1.9     # shape factor
    D = 1.0     # peak factor (max force normalized to 1)
    E = 0.97    # curvature factor

    term = B * alpha
    return D * pt.sin(C * pt.atan(term - E * (term - pt.atan(term))))


def suspension_force(wheel_displacement, wheel_speed, resting_compression=9.81*1610/4):
    k = 25000  # N/m        restorative force
    c = 2000   # Ns/m       damping force
    return -k * (wheel_displacement) - c * wheel_speed - resting_compression        # (B, 1, 4) - (B, 1, 4) - (1)


def project_and_normalize(vectors, normal):     # (*B, 3, N), (*B, 3, 1)
    # Project vectors onto plane orthogonal to normal
    dot     = pt.sum(vectors * normal, dim=-2, keepdim=True)  # sum over the 3-component axis
    proj    = vectors - dot * normal
    norms   = pt.norm(proj, dim=-2, keepdim=True)
    return proj / (norms + 1e-6)


def signed_angle(a, b, axis):                   # (*B, 3, N), (*B, 3, N), (*B, 3, 1) -> (*B, 3, N)
    cross   = pt.cross(a, b, dim=-2)
    dot     = pt.sum(a * b, dim=-2, keepdim=True) + 1e-6
    sign    = pt.sum(cross * axis, dim=-2, keepdim=True)
    return pt.atan2(sign, dot)


# def ackermann_steering_angles(steering_angle, wheelbase, track_width):
#     turning_radius = wheelbase / pt.tan(steering_angle)
#     inner_angle = pt.arctan(wheelbase / (turning_radius - track_width / 2))
#     outer_angle = pt.arctan(wheelbase / (turning_radius + track_width / 2))
#     return inner_angle, outer_angle

# temporary change to fix grad issue
def ackermann_steering_angles(steering_angle, wheelbase, track_width):
    return steering_angle, steering_angle


# approximate stats from a tesla model 3
class Car(Vehicle):
    def __init__(self, state: StateTensor = None, mass: float = None, inertia: pt.Tensor = None):
        kwargs = {}
        kwargs['state']             = state if state is not None else StateTensor()
        kwargs['mass']              = mass if mass is not None else 1610
        kwargs['inertia']           = inertia if inertia is not None else pt.diag(pt.tensor([550, 3200, 5600], dtype=pt.float32))
        
        super().__init__(**kwargs)

        self.wheel_torque               = 750 * 9.81                    # per rear tire in newtons
        self.wheel_coef_friction        = 0.9                           # dry coef of friction
        self.wheel_base                 = 2.88                          # distance between front and rear axles
        self.track_width                = 1.62                          # distance between left and right wheels
        self.com_height                 = 0.55                          # height of center of mass above ground (resting suspension)

        # wheel order [back_left, back_right, front_left, front_right]
        self.wheel_resting_positions    = pt.tensor([                    # displacements from center of mass in meters
                                            [-self.wheel_base/2,    -self.track_width/2,    -self.com_height], 
                                            [-self.wheel_base/2,    self.track_width/2,     -self.com_height], 
                                            [self.wheel_base/2,     -self.track_width/2,    -self.com_height], 
                                            [self.wheel_base/2,     self.track_width/2,     -self.com_height]
                                        ], dtype=self.dtype, device=self.device).T
        
        self.suspension_attach_pos      = pt.tensor([                     # location where wheels are attached by the suspension to the body
                                            [-self.wheel_base/2,    -self.track_width/2,    0], 
                                            [-self.wheel_base/2,    self.track_width/2,     0], 
                                            [self.wheel_base/2,     -self.track_width/2,    0], 
                                            [self.wheel_base/2,     self.track_width/2,     0]
                                        ], dtype=self.dtype, device=self.device).T

        self.max_steering_speed         = 0.5                           # radians per second at the front wheels
        self.max_steering_angle         = 0.61                          # radians at the front tire

    # takes a state (B, 13) and corresponding action (B, 2)
    # returns forces (B, 3, N) and moments (B, 3, N)
    def compute_forces(self, state: StateTensor, action: pt.Tensor) -> tuple[pt.Tensor, pt.Tensor]:
        pos, vel                        = state.pos, state.vel
        device, dtype, batch            = state.device, state.dtype, state.batch_size
        rot_mat, angvel_mat             = state.rot_mat, state.angvel_mat
        throttle, steer                 = action[..., 0], action[..., 1]                                                    # (B, 1), (B, 1)

        # suspension forces
        z_down                          = pt.tensor([0, 0, -1], device=device, dtype=dtype).unsqueeze(-1)                   # (3, 1)
        suspension_axis                 = rot_mat @ z_down                                                                  # (B, 3, 3) @ (3, 1) -> (B, 3, 1)
        wheel_world_positions           = pos.unsqueeze(-1) + (rot_mat @ self.wheel_resting_positions)                      # (B, 3, 3) @ (3, 4) -> (3, 4)
        wheel_suspension_heights        = wheel_world_positions[..., 2:3, :] / (suspension_axis[..., 2:3, :] + 1e-6)        # (B, 3, 4) -> (B, 1, 4)
        wheel_body_positions            = self.wheel_resting_positions + wheel_suspension_heights * z_down                  # (3, 4) + ((B, 1, 4) * (3, 1)) -> (B, 3, 4)
        tire_velocities                 = vel.unsqueeze(-1) + (rot_mat @ (angvel_mat @ wheel_body_positions))               # (B, 3, 1) + ((B, 3, 3) @ ((B, 3, 3) @ (B, 3, 4)) -> (B, 3, 4)
        wheel_suspension_speeds         = pt.sum(tire_velocities * suspension_axis, dim=-2, keepdim=True)                   # (B, 3, 4) * (3, 1) -> (B, 3, 4) -> (B, 4)
        suspension_mags                 = suspension_force(wheel_suspension_heights, wheel_suspension_speeds)               # (B, 1, 4), (B, 1, 4) -> (B, 1, 4)
        suspension_forces               = suspension_mags * suspension_axis                                                 # (B, 1, 4) * (3, 1) -> (B, 3, 4)

        # tire forces
        virtual_steer_angle             = steer.unsqueeze(-1) * self.max_steering_angle                                     # (B, 1) * (1) -> (B, 1)
        steering_mask                   = pt.tensor([0, 0, 1, 1], dtype=dtype, device=device)                               # (4)
        steering_angles                 = virtual_steer_angle * steering_mask                                               # (B, 4) * (4) -> (B, 4)
        scos, ssin, szero               = pt.cos(steering_angles), pt.sin(steering_angles), pt.zeros_like(steering_angles)  # (B, 4), (B, 4), (B, 4)
        tire_dirs                       = rot_mat @ pt.stack([scos, ssin, szero], dim=-2)                                   # (B, 3, 3) @ (B, 3, 4) -> (B, 3, 4)
        tire_perp_dirs                  = rot_mat @ pt.stack([-ssin, scos, szero], dim=-2)                                  # (B, 3, 3) @ (B, 3, 4) -> (B, 3, 4)

        # tire directions
        up_dir                          = -suspension_axis                                                                  # (B, 3, 1)
        tire_proj_vels                  = project_and_normalize(tire_velocities, up_dir)                                    # (B, 3, 4), (B, 3, 1) -> (B, 3, 4)
        tire_proj_dirs                  = project_and_normalize(tire_dirs, up_dir)                                          # (B, 3, 4), (B, 3, 1) -> (B, 3, 4)
        slip_angles                     = signed_angle(tire_proj_vels, tire_proj_dirs, up_dir)                              # (B, 3, 4), (B, 3, 4) -> (B, 1, 4)
        tire_normal_forces              = suspension_mags * suspension_axis[..., 2:3, :]                                    # (B, 1, 4) * (B, 1, 1) -> (B, 1, 4)
        tire_lateral_forces             = tire_normal_forces * pacejka_lateral_force(slip_angles) * tire_perp_dirs          # ((B, 1, 4) * (B, 1, 4)) * (B, 3, 4) -> (B, 3, 4)


        # throttle forces
        # Assumes rear wheel drive (RWD), force applied only to back wheels
        acceleration_mask               = pt.tensor([1, 1, 0, 0], device=device, dtype=dtype)                               # (4)
        throttle_tire_force_mag         = (throttle.unsqueeze(-1) * self.wheel_torque * acceleration_mask).unsqueeze(-2)    # (B, 1) * (1) * (4) -> (B, 1, 4)
        tire_throttle_forces            = throttle_tire_force_mag * tire_dirs                                               # (B, 1, 4) * (B, 3, 4) -> (B, 3, 4)

        # group the forces and compute the moments
        batch_zero_vec                  = pt.zeros((*batch, 3, 1), dtype=dtype, device=device)                              # (B, 3, 1)
        force_of_gravity                = (9.81 * self.mass * z_down) + batch_zero_vec                                      # (3, 1) + (*B, 3, 1) -> (*B, 3, 1)
        forces                          = pt.cat([                                                                          # (B, 3, 4), (B, 3, 4), (B, 3, 4), (3, 1) -> (B, 3, 13)
                                            suspension_forces, tire_lateral_forces, tire_throttle_forces, force_of_gravity
                                        ], dim=-1)

        suspension_force_location       = rot_mat @ self.suspension_attach_pos                                              # (B, 3, 3) @ (3, 4) -> (B, 3, 4)
        tire_force_location             = rot_mat @ wheel_body_positions                                                    # (B, 3, 3) @ (3, 4) -> (B, 3, 4)
        gravity_force_location          = batch_zero_vec
        force_locations                 = pt.cat([suspension_force_location, tire_force_location, tire_force_location, gravity_force_location], dim=-1)

        return forces, force_locations, rot_mat

    # takes a state (B, 13) and corresponding action (B, 2)
    # returns net force (B, 3) and net moment (B, 3)
    def compute_forces_and_moments(self, state, action) -> tuple[pt.Tensor, pt.Tensor]:
        # state     = (B, 13)
        # forces    = (B, 3, N)
        # moments   = (B, 3, N)
        
        forces, force_locations, rot_mat    = self.compute_forces(state, action)

        # forces                          = pt.cat([suspension_forces, tire_lateral_forces + tire_throttle_forces, force_of_gravity], dim=0)
        # force_locations                 = pt.cat([(rot_mat @ self.suspension_attach_pos.T).T, (rot_mat @ wheel_body_positions.T).T, pt.zeros((1, 3), device=rot_mat.device)], dim=0)
        moments                             = pt.cross(force_locations, forces, dim=-2)          # (B, 3, N), (B, 3, N) -> (B, 3, N)

        return pt.sum(forces, dim=-1), (rot_mat.T @ pt.sum(moments, dim=-1, keepdim=True)).squeeze(-1)  # (B, 3, N) -> (B, 3); (B, 3, 3) @ (B, 3, 1) -> (B, 3, 1) -> (B, 3)
    

###################################
## ---------- testing ---------- ##
###################################


from classes.TargetPath import TargetPath
from tqdm import trange  # tqdm range iterator
from util.optimize import optimize_along_path



if __name__ == '__main__':
    init_state = StateTensor(
        pos     = [0, 0, 0.55],
        vel     = [0, 0, 0],
        quat    = [1, 0, 0, 0],
        angvel  = [0, 0, 0],
    )

    # action plan and delta time
    tf = 5
    dt = 0.05
    nt = int(tf / dt)
    action_plan = pt.ones((nt, 2)) * pt.tensor([0.5, 0.0])[None,:]
    action_plan.requires_grad_(True)
    dts = dt * pt.ones(nt)

    # target path
    ts = pt.linspace(0, tf, 100)
    wx = 10*ts
    wy = 10*(1-pt.cos(ts * 0.8))
    wz = ts*0

    waypoints = pt.stack([wx, wy, wz]).T

    optimize_along_path(
        vehicle=Car(init_state), action_plan=action_plan, delta_time=dts, target=TargetPath(waypoints), 
        steps=100, lr=2e-3, discount_rate=0.25, acc_reg=1e-3, plot_freq=10
    )