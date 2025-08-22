import torch as pt
import numpy as np
import matplotlib.pyplot as plt
from classes.Vehicle import StateTensor, Vehicle
from util.quaternion import quaternion_to_matrix
from visualization.FourViewPlot import FourViewPlot
from classes.TargetPath import TargetPath
from tqdm import trange

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
                                        ], dtype=pt.float32).T                      # (3, 4)
        
        self.B_thrust_dir           = pt.tensor([0.0, 0.0, 1.0]).unsqueeze(-1)      # (3, 1)
        self.z_down                 = pt.tensor([0, 0, -1], dtype=pt.float32, device=None).unsqueeze(-1)


    def compute_forces(self, state: StateTensor, action: pt.Tensor) -> tuple [pt.Tensor, pt.Tensor]:
        vel                     = state.vel                     # (B, 3)
        BW_rot_mat              = state.rot_mat                 # (B, 3, 3)
        device, dtype, batch    = state.device, state.dtype, state.batch_size

        throttle                = pt.clip(action, -1.0, 1.0).unsqueeze(-2)                              # (B, 1, 4)

        W_drag                  = (-self.drag_coef * vel).unsqueeze(-1)                                 # (1) * (B, 3) -> (B, 3, 1)
        B_thrust                = self.peak_motor_thrust * throttle * self.B_thrust_dir                 # (1) * (B, 1, 4) * (3, 1) -> (B, 3, 4)
        W_thrust                = BW_rot_mat @ B_thrust                                                 # (B, 3, 3) @ (B, 3, 4) -> (B, 3, 4)

        batch_zero_vec          = pt.zeros((*batch, 3, 1), dtype=dtype, device=device)                  # (B, 3, 1)
        W_gravity               = 9.81 * self.mass * self.z_down + batch_zero_vec                       # (3, 1) + (*B, 3, 1) -> (*B, 3, 1)
        W_forces                = pt.cat([W_drag, W_thrust, W_gravity], dim=-1)     # (B, 3, 6)

        thrust_locations        = BW_rot_mat @ self.B_thrust_locs
        W_force_locations       = pt.cat([batch_zero_vec, thrust_locations, batch_zero_vec], dim=-1)   # (B, 3, 6)

        return W_forces, W_force_locations, BW_rot_mat


    def compute_forces_and_moments(self, state: StateTensor, action: pt.Tensor) -> tuple [pt.Tensor, pt.Tensor]:
        forces, force_locations, rot_mat    = self.compute_forces(state, action)
        moments                             = pt.cross(force_locations, forces, dim=-2)                     # (B, 3, N), (B, 3, N) -> (B, 3, N)
        return pt.sum(forces, dim=-1), (rot_mat.T @ pt.sum(moments, dim=-1, keepdim=True)).squeeze(-1)      # (B, 3, N) -> (B, 3); (B, 3, 3) @ (B, 3, 1) -> (B, 3, 1) -> (B, 3)
    

###################################
## ---------- testing ---------- ##
###################################

from util.optimize import optimize_along_path

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

    # action plan and delta time
    tf = 3
    dt = 0.05
    nstep = int(tf / dt)
    action_plan = pt.ones((nstep, 4)) * pt.tensor([0.2, 0.2, 0.2, 0.2])[None,:]
    action_plan.requires_grad_(True)
    dts = dt * pt.ones(nstep)
    

    # target path
    ts = pt.linspace(0, tf, 100)
    wx = 10*ts/2
    wy = 10*(1-pt.cos(ts/2))
    wz = 5.0*pt.ones_like(ts)

    waypoints = pt.stack([wx, wy, wz]).T
    np.savetxt('blender/trajectories/drone_target.csv', waypoints, delimiter=',')

    optimize_along_path(
        vehicle=Quadcopter(init_state), action_plan=action_plan, delta_time=dts, target=TargetPath(waypoints), 
        steps=1000, lr=2e-3, discount_rate=0.25, acc_reg=1e-3, plot_freq=10
    )