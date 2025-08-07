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
                                        ], dtype=pt.float32)
        
        self.suspension_attach_pos      = pt.tensor([                     # location where wheels are attached by the suspension to the body
                                            [-self.wheel_base/2,    -self.track_width/2,    0], 
                                            [-self.wheel_base/2,    self.track_width/2,     0], 
                                            [self.wheel_base/2,     -self.track_width/2,    0], 
                                            [self.wheel_base/2,     self.track_width/2,     0]
                                        ], dtype=pt.float32)

        self.max_steering_speed         = 0.5                           # radians per second at the front wheels
        self.max_steering_angle         = 0.61                          # radians at the front tire

    def compute_forces(self, state, action) -> tuple[pt.Tensor, pt.Tensor]:
        pos, vel, quat, ang_vel         = state
        device, dtype                   = pos.device, pt.float32
        rot_mat                         = quaternion_to_matrix(quat)
        throttle, steer                 = pt.clip(action, -1.0, 1.0)
        ang_vel_mat                     = pt.tensor([
                                            [0, -ang_vel[2], ang_vel[1]],
                                            [ang_vel[2], 0, -ang_vel[0]],
                                            [-ang_vel[1], ang_vel[0], 0]
                                        ], dtype=dtype, device=ang_vel.device)

        # suspension forces
        suspension_axis                 = rot_mat @ pt.tensor([0, 0, -1], device=device, dtype=dtype)
        wheel_world_positions           = pos + (rot_mat @ self.wheel_resting_positions.T).T
        wheel_suspension_heights        = wheel_world_positions[:, 2] / (suspension_axis[2] + 1e-6)
        wheel_body_positions            = self.wheel_resting_positions + wheel_suspension_heights[:, None] * pt.tensor([0, 0, -1], device=device, dtype=dtype)
        tire_velocities                 = vel + (rot_mat @ (ang_vel_mat @ wheel_body_positions.T)).T
        wheel_suspension_speeds         = pt.sum(tire_velocities * suspension_axis[None, :], dim=-1)

        suspension_mags                 = suspension_force(wheel_suspension_heights, wheel_suspension_speeds)
        suspension_forces               = suspension_mags[:, None] * suspension_axis[None, :]

        # tire forces
        virtual_steer_angle             = steer * self.max_steering_angle
        FL_steer, FR_steer              = ackermann_steering_angles(virtual_steer_angle, self.wheel_base, self.track_width)
        FL_cos, FL_sin, FR_cos, FR_sin  = pt.cos(FL_steer), pt.sin(FL_steer), pt.cos(FR_steer), pt.sin(FR_steer)
        # tire_directions                 = (rot_mat @ pt.tensor([[1, 0, 0], [1, 0, 0], [coss, sins, 0], [coss, sins, 0]], device=device, dtype=dtype).T).T
        tire_directions                 = (rot_mat @ pt.stack([
                                            pt.tensor([1.0, 0.0, 0.0], device=steer.device), 
                                            pt.tensor([1.0, 0.0, 0.0], device=steer.device), 
                                            pt.stack([FL_cos, FL_sin, pt.tensor(0.0, device=steer.device)]), 
                                            pt.stack([FR_cos, FR_sin, pt.tensor(0.0, device=steer.device)])
                                        ]).T).T

        up_dir                          = rot_mat @ pt.tensor([0, 0, 1], device=device, dtype=dtype)
        tire_proj_vels                  = project_and_normalize(tire_velocities, up_dir)
        tire_proj_dirs                  = project_and_normalize(tire_directions, up_dir)
        slip_angles                     = signed_angle(tire_proj_vels, tire_proj_dirs, up_dir)
        tire_normal_forces              = suspension_mags * suspension_axis[2]
        tire_perp_dirs                  = (rot_mat @ pt.tensor([[0, 1, 0], [0, 1, 0], [-FL_sin, FL_cos, 0], [-FR_sin, FR_cos, 0]], device=device, dtype=dtype).T).T
        tire_lateral_forces             = (tire_normal_forces * pacejka_lateral_force(slip_angles))[:, None] * tire_perp_dirs

        # throttle forces
        # Assumes rear wheel drive (RWD), force applied only to back wheels
        throttle_tire_force_mag         = throttle * self.wheel_torque * pt.tensor([1.0, 1.0, 0.0, 0.0], device=device, dtype=dtype)
        tire_throttle_forces            = throttle_tire_force_mag[:, None] * tire_directions

        # group the forces and compute the moments
        force_of_gravity                = pt.tensor([0, 0, -9.81 * self.mass], device=device, dtype=dtype)[None, :]
        

        forces                          = pt.cat([suspension_forces, tire_lateral_forces, tire_throttle_forces, force_of_gravity], dim=0)

        suspension_force_location       = (rot_mat @ self.suspension_attach_pos.T).T
        tire_force_location             = (rot_mat @ wheel_body_positions.T).T
        gravity_force_location          = pt.zeros((1, 3), device=device, dtype=dtype)
        force_locations                 = pt.cat([suspension_force_location, tire_force_location, tire_force_location, gravity_force_location], dim=0)

        return forces, force_locations, rot_mat

    def compute_forces_and_moments(self, state, action) -> tuple[pt.Tensor, pt.Tensor]:
        forces, force_locations, rot_mat    = self.compute_forces(state, action)

        # forces                          = pt.cat([suspension_forces, tire_lateral_forces + tire_throttle_forces, force_of_gravity], dim=0)
        # force_locations                 = pt.cat([(rot_mat @ self.suspension_attach_pos.T).T, (rot_mat @ wheel_body_positions.T).T, pt.zeros((1, 3), device=rot_mat.device)], dim=0)
        moments                             = pt.linalg.cross(force_locations, forces)

        return pt.sum(forces, dim=0), rot_mat.T @ pt.sum(moments, dim=0)
    

###################################
## ---------- testing ---------- ##
###################################


from visualization.FourViewPlot import FourViewPlot
from classes.TargetPath import TargetPath
from torch.autograd.functional import jacobian
from tqdm import trange  # tqdm range iterator



if __name__ == '__main__':
    # initial state
    position            = pt.tensor([0.0, 0.0, 0.55])
    velocity            = pt.tensor([0, 0, 0])
    quaternion          = pt.tensor([1.0, 0.0, 0.0, 0.0])
    angular_velocity    = pt.tensor([0.0, 0, 0.0])

    car = Car(position, velocity, quaternion, angular_velocity)

    # action plan and delta time
    tf = 5
    dt = 0.05
    nt = int(tf / dt)
    action_plan = pt.ones((nt, 2)) * pt.tensor([0.5, 0.5])[None,:]
    action_plan.requires_grad_(True)
    dts = dt * pt.ones(nt)

    # target path
    ts = pt.linspace(0, tf, 100)
    wx = 5*ts
    wy = 5*(1-pt.cos(ts))
    wz = ts*0

    waypoints = pt.stack([wx, wy, wz]).T
    targetPath = TargetPath(waypoints)


    # action_plan = action_plan.clone().detach().requires_grad_(True)
    optimizer = pt.optim.Adam([action_plan], lr=1e-2)

    num_iters = 100

    for epoch in trange(num_iters, desc='Optimizing action plan', unit='iter'):
        optimizer.zero_grad()
        p, v, q, w, t = car.simulate_trajectory(car.get_state(), action_plan, dts)
        with pt.no_grad():
            ds = pt.cumsum(pt.norm((p[1:] - p[:-1]).detach(), dim=1), dim=0)
            Y_p = targetPath.distance_interpolate(ds)
        losses = ((p[1:,0:2] - Y_p[:,0:2]) ** 2).sum(dim=1)                                                  # per point L_2^2 loss
        loss = losses.sum(dim=0)
        loss.backward()
        action_plan.grad = pt.nan_to_num(action_plan.grad, nan=0.0)
        optimizer.step()
    

    print(p[-1])

    def save_traj_to_csv():
        traj_force_vecs = []
        traj_force_locs = []
        for pos, vel, quat, omega, action in zip(p[:-1], v[:-1], q[:-1], w[:-1], action_plan):
            state = [pos, vel, quat, omega]
            force_vecs, force_locs, _ = car.compute_forces(state, action)
            traj_force_vecs.append(force_vecs)
            traj_force_locs.append(force_locs)
        traj_force_vecs = pt.stack(traj_force_vecs)
        traj_force_locs = pt.stack(traj_force_locs) + p[:-1,None,:]

        # Reshape and save to CSV
        folder = 'blender/trajectories/'

        np.savetxt(folder + 'traj_force_vecs.csv', traj_force_vecs.detach().numpy().reshape(traj_force_vecs.shape[0], -1), delimiter=',')
        np.savetxt(folder + 'traj_force_locs.csv', traj_force_locs.detach().numpy().reshape(traj_force_locs.shape[0], -1), delimiter=',')

        traj = pt.concatenate([p,q], axis=1) # shape (N, 7): [x, y, z, qw, qx, qy, qz]
        np.savetxt(folder + 'traj.csv', traj.detach().cpu().numpy(), delimiter=',')
    save_traj_to_csv()

    


    fourPlot = FourViewPlot()
    fourPlot.addTrajectory(p.detach().cpu().numpy(), 'Vehicle', color='blue')
    fourPlot.addTrajectory(waypoints, 'TargetPath', color='red')
    fourPlot.addScatter(p.detach().cpu().numpy(), 'X_p', color='cyan')
    fourPlot.addScatter(Y_p.detach().cpu().numpy(), 'Y_p', color='orange')
    fourPlot.show()


