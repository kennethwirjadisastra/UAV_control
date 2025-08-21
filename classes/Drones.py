import torch as pt
import numpy as np
import matplotlib.pyplot as plt
from classes.Vehicle import StateTensor, Vehicle
from util.quaternion import quaternion_to_matrix
from visualization.FourViewPlot import FourViewPlot
from classes.TargetPath import TargetPath
from tqdm import trange

class Quadcopter(Vehicle):
    def __init__(self, position: pt.Tensor = None, velocity: pt.Tensor = None, 
            quaternion: pt.Tensor = None, angular_velocity: pt.Tensor = None,
            mass: pt.Tensor = None, inertia: pt.Tensor = None):
        kwargs = {}
        kwargs['position']          = position
        kwargs['velocity']          = velocity
        kwargs['quaternion']        = quaternion
        kwargs['angular_velocity']  = angular_velocity
        kwargs['mass']              = mass if mass is not None else 0.5
        kwargs['inertia']           = inertia if inertia is not None else pt.diag(pt.tensor([0.003, 0.003, 0.005], dtype=pt.float32))
        
        super().__init__(**kwargs)

        self.peak_motor_thrust      = 5.0   # thrust per motor in newtons
        self.drag_coef              = 1.1   # dry coef of friction

                                                                    # motor order [back_left, back_right, front_left, front_right]
        self.B_thrust_locs          = pt.tensor([                   # displacements from center of mass in meters
                                            [-0.08, -0.08, -0.01], 
                                            [0.08, -0.08, -0.01], 
                                            [-0.08, 0.08, -0.01], 
                                            [0.08, 0.08, -0.01]
                                        ], dtype=pt.float32)
        
        self.B_thrust_dir           = pt.tensor([0.0, 0.0, 1.0])
        

class Quadcopter(Vehicle):
    def __init__(self, state: StateTensor=None, mass: float=None, inertia: pt.Tensor=None):
        kwargs = {}
        kwargs['state']             = state if state is not None else StateTensor()
        kwargs['mass']              = mass if mass is not None else 0.5
        kwargs['inertia']           = inertia if inertia is not None else pt.diag(pt.tensor([0.3, 0.3, 0.5], dtype=pt.float32))

        super().__init__(**kwargs)

        self.peak_motor_thrust      = 5.0   # thrust per motor in newtons
        self.drag_coef              = 1.1   # dry coef of friction

                                                                    # motor order [back_left, back_right, front_left, front_right]
        self.B_thrust_locs          = pt.tensor([                   # displacements from center of mass in meters
                                            [-0.08, -0.08, -0.01], 
                                            [0.08, -0.08, -0.01], 
                                            [-0.08, 0.08, -0.01], 
                                            [0.08, 0.08, -0.01]
                                        ], dtype=pt.float32)
        
        self.B_thrust_dir           = pt.tensor([
                                        [0.0, 0.0, 1.0]
                                    ])

    '''# expects net world force and net body moment
    def compute_forces_and_moments(self, state, action) -> tuple[pt.Tensor, pt.Tensor]:
        pos, vel, quat, ang_vel     = state.pos, state.vel, state.quat, state.angvel            # (B, 3), (B, 3), (B, 4), (B, 3)
        device, dtype, batch        = state.device, state.dtype, state.batch_size
        rot_mat, angvel_mat         = state.rot_mat, state.angvel_mat                           # (B, 3, 3), (B, 3, 3)
        throttle                    = pt.clip(action, -1.0, 1.0)
        #ang_vel_mat                 = pt.tensor([
        #                                [0, -ang_vel[2], ang_vel[1]],
        #                                [ang_vel[2], 0, -ang_vel[0]],
        #                                [-ang_vel[1], ang_vel[0], 0]
        #                            ], dtype=pt.float32, device=ang_vel.device)

        W_drag                  = -self.drag_coef * vel
        B_thrust                = self.peak_motor_thrust * throttle[:,None] * self.B_thrust_dir
        W_thrust                = (rot_mat @ B_thrust.T).T
        W_gravity               = pt.tensor([0, 0, -9.81 * self.mass])
        B_moments               = pt.linalg.cross(self.B_thrust_locs, B_thrust)

        print(W_thrust.shape, W_gravity[None, :].shape, W_drag[None, :].shape)

        return pt.sum(W_thrust, dim=0) + W_drag + W_gravity, pt.sum(B_moments, dim=0)'''

    
    def compute_forces(self, state: StateTensor, action: pt.Tensor) -> tuple [pt.Tensor, pt.Tensor]:
        pos, vel, quat, ang_vel = state.pos, state.vel, state.quat, state.angvel                        # (B, 3), (B, 3), (B, 4), (B, 3)
        device, dtype, batch    = state.device, state.dtype, state.batch_size
        rot_mat, angvel_mat     = state.rot_mat, state.angvel_mat                                       # (B, 3, 3), (B, 3, 3)
        throttle                = pt.clip(action, -1.0, 1.0)                                            # (B, 4)


        W_drag                  = -self.drag_coef * vel                                                 # (1,) * (B, 3) -> (B, 3)
        B_thrust                = self.peak_motor_thrust * throttle[:,None] * self.B_thrust_dir         # (1,) * (B, 1, 4) *  (3, 1) -> (B, 4, 3)
        W_thrust                = (rot_mat @ B_thrust.T)                                                # (B, 3, 3) @ (B, 3, 4) -> (B, 3, 4)
        W_gravity               = pt.tensor([0, 0, -9.81 * self.mass])                                  # (3, )


        forces = pt.cat([W_drag[:, None], W_thrust, W_gravity[:, None]], dim=-1)

        drag_location           = state.pos
        thrust_locations        = self.B_thrust_locs + state.pos
        gravity_locations       = state.pos

        force_locations         = pt.cat([drag_location[:,None], thrust_locations.T, gravity_locations[:,None]], dim=-1)
        print(forces.shape, force_locations.shape)

        return forces, force_locations, rot_mat


    def compute_forces_and_moments(self, state: StateTensor, action: pt.Tensor) -> tuple [pt.Tensor, pt.Tensor]:
        '''pos, vel, quat, ang_vel     = state.pos, state.vel, state.quat, state.angvel            # (B, 3), (B, 3), (B, 4), (B, 3)
        device, dtype, batch        = state.device, state.dtype, state.batch_size
        rot_mat, angvel_mat         = state.rot_mat, state.angvel_mat                           # (B, 3, 3), (B, 3, 3)
        throttle                    = pt.clip(action, -1.0, 1.0)

        
        forces, force_locations     = self.compute_forces(state, action)
        
        B_thrust                    = self.peak_motor_thrust * throttle[:,None] * self.B_thrust_dir
        B_moments                   = pt.cross(self.B_thrust_locs, B_thrust, dim=-1)

        return forces, B_moments'''

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


if __name__ == '__main__':
    # initial state
    position            = pt.tensor([0.0, 0.0, 5.0])
    velocity            = pt.tensor([0, 0, 0])
    quaternion          = pt.tensor([1.0, 0.0, 0.0, 0.0])
    angular_velocity    = pt.tensor([0.0, 0, 0.0])

    init_state = StateTensor(
        pos     = [0.0, 0.0, 5.0],
        vel     = [0.0, 0.0, 0.0],
        quat    = [1.0, 0.0, 0.0, 0.0],
        angvel  = [0.0, 0.0, 0.0]
    )

    drone = Quadcopter(init_state)

    # action plan and delta time
    tf = 1
    dt = 0.05
    nstep = int(tf / dt)
    action_plan = pt.ones((nstep, 4)) * pt.tensor([0, 0, 0, 0])[None,:]
    action_plan.requires_grad_(False)
    dts = dt * pt.ones(nstep)
    

    # target path
    ts = pt.linspace(0, tf, 100)
    wx = 10*ts
    wy = 10*(1-pt.cos(ts))
    wz = 5.0*pt.ones_like(ts)

    waypoints = pt.stack([wx, wy, wz]).T
    np.savetxt('blender/trajectories/drone_target.csv', waypoints, delimiter=',')
    targetPath = TargetPath(waypoints)

    optimizer = pt.optim.Adam([action_plan])

    fourPlot = FourViewPlot()
    fourPlot.addTrajectory(waypoints, 'TargetPath', color='red')

    num_iters = 10
    for epoch in trange(num_iters, desc='Optimizing action plan', unit='iter'):
        optimizer.zero_grad()
        X_p, X_v, q, w, t  = drone.simulate_trajectory(init_state, action_plan, dts)

        arclength           = pt.cumsum(pt.norm((X_p[1:] - X_p[:-1]).detach(), dim=1), dim=0)
        Y_p: pt.tensor      = targetPath.distance_interpolate(arclength)


    # take 1 step for each forcing term with grad tracking
        losses = ((X_p[1:,0:2] - Y_p[:,0:2]) ** 2).sum(dim=1)                                                  # per point L_2^2 loss
        time_scale = pt.exp(-0.1*t[...,1:])

        loss = (losses * time_scale).sum(dim=0)
        loss.backward()
        action_plan.grad = pt.nan_to_num(action_plan.grad, nan=0.0)
        optimizer.step()

        if (num_iters - epoch) % 5 == 1:
            fourPlot.addTrajectory(X_p.detach().cpu().numpy(), 'Vehicle', color='blue')
            fourPlot.addScatter(X_p.detach().cpu().numpy(), 'X_p', color='cyan')
            fourPlot.addScatter(Y_p.detach().cpu().numpy(), 'Y_p', color='orange')
            fourPlot.show()
    plt.show()

    

    print(X_p[-1])

    def save_traj_to_csv():
        with pt.no_grad():
            state = StateTensor(state_vec=pt.cat([X_p[:-1], X_v[:-1], q[:-1], w[:-1]], dim=-1))
            force_vecs, force_locs, _ = drone.compute_forces(state, action_plan)
            force_locs += X_p[:-1,:,None]

            # Reshape and save to CSV
            folder = 'blender/trajectories/'

            np.savetxt(folder + 'traj_force_vecs.csv', force_vecs.cpu().numpy().reshape(force_vecs.shape[0], -1), delimiter=',')
            np.savetxt(folder + 'traj_force_locs.csv', force_locs.cpu().numpy().reshape(force_locs.shape[0], -1), delimiter=',')

            traj = pt.concatenate([X_p, q], axis=1) # shape (N, 7): [x, y, z, qw, qx, qy, qz]
            np.savetxt(folder + 'traj.csv', traj.detach().cpu().numpy(), delimiter=',')
    save_traj_to_csv()