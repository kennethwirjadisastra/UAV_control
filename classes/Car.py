import torch as pt
import numpy as np
from matplotlib import pyplot as plt
from classes.Vehicle import Vehicle
from util.quaternion import quaternion_to_matrix


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
    return -k * (wheel_displacement) - c * wheel_speed - resting_compression


def project_and_normalize(vectors, normal):
    # Project vectors onto plane orthogonal to normal
    # vectors shape (N,3), normal shape (3,)
    dot = pt.sum(vectors * normal[None, :], dim=-1, keepdim=True)  # (N,1)
    proj = vectors - dot * normal[None, :]
    norms = pt.linalg.norm(proj, dim=-1, keepdim=True)
    return proj / (norms + 1e-6)  # avoid div by zero


def signed_angle(a, b, axis):
    cross = pt.linalg.cross(a, b)
    dot = pt.sum(a * b, dim=-1)
    sign = pt.sum(cross * axis, dim=-1)
    return pt.atan2(sign, dot)


# approximate stats from a tesla model 3
class Car(Vehicle):
    def __init__(self, position: pt.Tensor = None, velocity: pt.Tensor = None, 
            quaternion: pt.Tensor = None, angular_velocity: pt.Tensor = None,
            mass: pt.Tensor = None, inertia: pt.Tensor = None):
        kwargs = {}
        kwargs['position']          = position
        kwargs['velocity']          = velocity
        kwargs['quaternion']        = quaternion
        kwargs['angular_velocity']  = angular_velocity
        kwargs['mass']              = mass if mass is not None else 1610
        kwargs['inertia']           = inertia if inertia is not None else pt.diag(pt.tensor([550, 3200, 5600], dtype=pt.float32))
        
        super().__init__(**kwargs)

        self.wheel_torque               = 750                           # per rear tire
        self.wheel_coef_friction        = 0.9                           # dry coef of friction

        # wheel order [back_left, back_right, front_left, front_right]
        self.wheel_resting_positions    = pt.tensor([                    # displacements from center of mass in meters
                                            [-1.44, -0.81, -0.55], 
                                            [1.44, -0.81, -0.55], 
                                            [-1.44, 0.81, -0.55], 
                                            [1.44, 0.81, -0.55]
                                        ], dtype=pt.float32)
        
        self.suspension_attach_pos      = pt.tensor([                     # location where wheels are attached by the suspension to the body
                                            [-1.44, -0.81, 0], 
                                            [1.44, -0.81, 0], 
                                            [-1.44, 0.81, 0], 
                                            [1.44, 0.81, 0]
                                        ], dtype=pt.float32)

        self.max_steering_speed         = 0.5                           # radians per second at the front wheels
        self.max_steering_angle         = 0.61                          # radians at the front tire

    def compute_forces_and_moments(self, state, action) -> tuple[pt.Tensor, pt.Tensor]:
        pos, vel, quat, ang_vel         = state
        # print(quat)
        rot_mat                         = quaternion_to_matrix(quat)
        throttle, steer                 = pt.clip(action, -1.0, 1.0)
        ang_vel_mat                     = pt.tensor([
                                            [0, -ang_vel[2], ang_vel[1]],
                                            [ang_vel[2], 0, -ang_vel[0]],
                                            [-ang_vel[1], ang_vel[0], 0]
                                        ], dtype=pt.float32, device=ang_vel.device)

        # suspension forces
        suspension_axis                 = rot_mat @ pt.tensor([0, 0, -1], device=rot_mat.device, dtype=pt.float32)
        wheel_world_positions           = pos + (rot_mat @ self.wheel_resting_positions.T).T
        wheel_suspension_heights        = wheel_world_positions[:, 2] / (suspension_axis[2] + 1e-6)
        wheel_body_positions            = self.wheel_resting_positions + wheel_suspension_heights[:, None] * pt.tensor([0, 0, -1], device=rot_mat.device, dtype=pt.float32)
        tire_velocities                 = vel + (rot_mat @ (ang_vel_mat @ wheel_body_positions.T)).T
        wheel_suspension_speeds         = pt.sum(tire_velocities * suspension_axis[None, :], dim=-1)

        suspension_mags                 = suspension_force(wheel_suspension_heights, wheel_suspension_speeds)
        suspension_forces               = suspension_mags[:, None] * suspension_axis[None, :]

        # tire forces
        steer_angle                     = steer * self.max_steering_angle
        coss, sins                      = pt.cos(steer_angle), pt.sin(steer_angle)
        tire_directions                 = (rot_mat @ pt.tensor([[1, 0, 0], [1, 0, 0], [coss, sins, 0], [coss, sins, 0]], device=rot_mat.device, dtype=pt.float32).T).T
        # tire_directions                 = (rot_mat @ pt.stack([pt.tensor([1.0, 0.0, 0.0], device=steer.device)] * 2 + [pt.stack([coss, sins, pt.tensor(0.0, device=steer.device)])] * 2).T).T

        up_dir                          = rot_mat @ pt.tensor([0, 0, 1], device=rot_mat.device, dtype=pt.float32)
        tire_proj_vels                  = project_and_normalize(tire_velocities, up_dir)
        tire_proj_dirs                  = project_and_normalize(tire_directions, up_dir)
        slip_angles                     = signed_angle(tire_proj_vels, tire_proj_dirs, up_dir)
        tire_normal_forces              = suspension_mags * suspension_axis[2]
        tire_perp_dirs                  = (rot_mat @ pt.tensor([[0, 1, 0], [0, 1, 0], [-sins, coss, 0], [-sins, coss, 0]], device=rot_mat.device, dtype=pt.float32).T).T
        tire_lateral_forces             = (tire_normal_forces * pacejka_lateral_force(slip_angles))[:, None] * tire_perp_dirs

        # throttle forces
        # Assumes rear wheel drive (RWD), force applied only to back wheels
        throttle_tire_force_mag         = throttle * self.wheel_torque * pt.tensor([1.0, 1.0, 0.0, 0.0], device=rot_mat.device, dtype=pt.float32)
        tire_throttle_forces            = throttle_tire_force_mag[:, None] * tire_directions

        # group the forces and compute the moments
        force_of_gravity                = pt.tensor([0, 0, -9.81 * self.mass], device=rot_mat.device, dtype=pt.float32)[None, :]

        forces                          = pt.cat([suspension_forces, tire_lateral_forces + tire_throttle_forces, force_of_gravity], dim=0)
        force_locations                 = pt.cat([(rot_mat @ self.suspension_attach_pos.T).T, (rot_mat @ wheel_body_positions.T).T, pt.zeros((1, 3), device=rot_mat.device)], dim=0)
        moments                         = pt.linalg.cross(force_locations, forces)

        return pt.sum(forces, dim=0), rot_mat.T @ pt.sum(moments, dim=0)


###################################
## ---------- testing ---------- ##
###################################


from visualization.FourViewPlot import FourViewPlot
from classes.TargetPath import TargetPath

if __name__ == '__main__':
    # initial state
    position            = pt.tensor([0.0, 0.0, 0.55])
    velocity            = pt.tensor([0, 0, 0])
    quaternion          = pt.tensor([1.0, 0.0, 0.0, 0.0])
    angular_velocity    = pt.tensor([0.0, 0, 0.0])

    car = Car(position, velocity, quaternion, angular_velocity)

    # action plan and delta time
    action_plan = pt.ones((2000, 2)) * pt.tensor([0.5, 0.25])[None,:]
    action_plan.requires_grad_(False)
    dts = 0.04 * pt.ones(2000)

    for i in range(1):
        p, v, q, w, t = car.simulate_trajectory(car.get_state(), action_plan, dts)

        # loss = pt.norm(p[-1,0])
        # loss.backward()
        # print(action_plan.grad)
    
    # target path
    ts = pt.cumsum(dts, dim=0)
    wx = ts
    wy = 5*(1-pt.cos(ts/5))
    wz = ts*0

    waypoints = pt.stack([wx, wy, wz])
    targetPath = TargetPath(waypoints)

    fourPlot = FourViewPlot()
    fourPlot.addTrajectory(p.detach().cpu().numpy(), 'Vehicle', color='b')
    fourPlot.addTrajectory(waypoints.T, 'TargetPath', color='g')
    fourPlot.show()


    traj = pt.concatenate([p,q], axis=1) # shape (N, 7): [x, y, z, qw, qx, qy, qz]
    header = 'x,y,z,qw,qz,qy,qz'
    np.savetxt('trajectory.csv', traj.detach().cpu().numpy(), delimiter=',')