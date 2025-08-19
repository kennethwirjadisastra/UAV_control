import torch as pt
from torchdiffeq import odeint
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from util.quaternion import quaternion_derivative, quaternion_to_matrix

class StateTensor(pt.Tensor):
    def __new__(cls, state_vec=None, pos=None, vel=None, quat=None, angvel=None):
        if state_vec is None:
            state_vec = pt.tensor([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=pt.float32)
        else:
            state_vec = pt.as_tensor(state_vec, dtype=pt.float32)

        obj = pt.Tensor._make_subclass(cls, state_vec)
        if pos is not None:
            obj[..., 0:3] = pt.as_tensor(pos, dtype=pt.float32)
        if vel is not None:
            obj[..., 3:6] = pt.as_tensor(vel, dtype=pt.float32)
        if quat is not None:
            obj[..., 6:10] = pt.as_tensor(quat, dtype=pt.float32)
        if angvel is not None:
            obj[..., 10:13] = pt.as_tensor(angvel, dtype=pt.float32)
        return obj

    @property
    def pos(self) -> pt.Tensor:
        return self[..., 0:3]

    @property
    def vel(self) -> pt.Tensor:
        return self[..., 3:6]

    @property
    def quat(self) -> pt.Tensor:
        return self[..., 6:10]

    @property
    def angvel(self) -> pt.Tensor:
        return self[..., 10:13]
    
    @property
    def angvel_mat(self) -> pt.Tensor:
        wx, wy, wz      = self[..., 10], self[..., 11], self[..., 12]
        mat             = pt.zeros((*self.shape[:-1], 3, 3), device=self.device, dtype=self.dtype)
        mat[..., 0, 1]  = -wz
        mat[..., 0, 2]  =  wy
        mat[..., 1, 0]  =  wz
        mat[..., 1, 2]  = -wx
        mat[..., 2, 0]  = -wy
        mat[..., 2, 1]  =  wx
        return mat
    
    @property
    def rot_mat(self) -> pt.Tensor:
        return quaternion_to_matrix(self[..., 6:10])
    
    @property
    def batch_size(self) -> tuple:
        return self.shape[:-1]


# vehicle class
class Vehicle(ABC):
    def __init__(self, state: StateTensor = None, mass: float = None, inertia: pt.Tensor = None):
        # state of the system
        self.state              = StateTensor() if state is None else state

        # system characteristics
        self.mass               = pt.as_tensor(mass) if mass is not None else pt.tensor(1)
        self.inertia            = pt.eye(3, dtype=pt.float32) if inertia is None else inertia.clone().detach().to(dtype=pt.float32)
        self.inv_inertia        = pt.linalg.inv(self.inertia)

    def _state_derivative(self, state, force, moment):
        q, w = state[..., 6:10], state[..., 10:13]
        # wdot = self.inv_inertia @ (moment - pt.linalg.cross(w, self.inertia @ w))

        return pt.cat([
            state[..., 3:6], 
            force / self.mass, 
            quaternion_derivative(q, w),
            (moment - pt.cross(w, w @ self.inertia)) @ self.inv_inertia
        ], dim=-1)

    def rk4_step(self, state, force, moment, dt):
        # Computes the future state of the system after 1 step of Runge-Kutta 4th order integration

        # state     = (B, 13) pos(3)+vel(3)+quat(4)+angvel(3)
        # force     = (B, 3)
        # moment    = (B, 3)
        # dt        = (B, 1)

        k1 = self._state_derivative(state, force, moment)
        k2 = self._state_derivative(state + k1*dt/2, force, moment)
        k3 = self._state_derivative(state + k2*dt/2, force, moment)
        k4 = self._state_derivative(state + k3*dt, force, moment)

        next_state = state + (k1 + 2*k2 + 2*k3 + k4) * dt/6
        next_state[...,6:10] = next_state[...,6:10] / (pt.norm(next_state[...,6:10], dim=-1, keepdim=True) + 1e-12)
        return next_state

    @abstractmethod
    def compute_forces_and_moments(self, state, action):
        # Takes the vehicle in a given state and taking an action
        # Returns the resulting net forces and net moments
        pass

    def simulate_trajectory(self, initial_state, action_plan, dts):
        # initial state     = (B, 13)
        # action plan       = (B, N, D)
        # dts               = (B, N)

        # Simulates the system forward in time given a sequence of actions and time steps
        N                   = action_plan.shape[-2]                                     # N is the number of timesteps
        time                = pt.cat([dts[..., 0:1]*0, pt.cumsum(dts, dim=-1)])
        states              = [initial_state]

        for i in range(N):
            action, dt      = action_plan[..., i, :], dts[..., i]
            force, moment   = self.compute_forces_and_moments(states[-1], action)
            states.append(
                self.rk4_step(states[-1], force, moment, dt)
            )

        states = StateTensor(pt.stack(states))

        return states[..., 0:3], states[..., 3:6], states[..., 6:10], states[..., 10:13], time
    


    # def simulate_trajectory(self, initial_state, action_plan, dts):
    #     device, dtype = action_plan.device, action_plan.dtype
    #     state = tuple(s.clone() for s in initial_state)  # avoid modifying original
    #     p_list, v_list, q_list, w_list = [state[0]], [state[1]], [state[2]], [state[3]]


    #     for action, dt in zip(action_plan, dts):
    #         force, moment = self.compute_forces_and_moments(state, action)
    #         state = self.rk4_step(state, force, moment, dt)

    #         p_list.append(state[0])
    #         v_list.append(state[1])
    #         q_list.append(state[2])
    #         w_list.append(state[3])

    #     # Stack into tensors
    #     p = pt.stack(p_list)
    #     v = pt.stack(v_list)
    #     q = pt.stack(q_list)
    #     w = pt.stack(w_list)
    #     t = pt.cat([pt.tensor([0.0], dtype=dtype, device=device), pt.cumsum(dts, dim=0)])

    #     for z in [p, v, q, t[:, None]]:
    #         print(z.shape)

    #     return p, v, q, w, t


    def __repr__(self):
        return (f"Vehicle(position={self.position}, velocity={self.velocity}, "
                f"quaternion={self.quaternion}, angular_velocity={self.angular_velocity})")
    


    """simulating the trajectory using an ode solver for accuracy
        def simulate_trajectory(self, initial_state, action_plan, dts):
        N = len(action_plan)
        t = pt.cat([pt.tensor([0.0]), pt.cumsum(dts, dim=0)])  # time points

        # Flatten initial state to 1D tensor
        def flatten_state(state):
            return pt.cat([s.reshape(-1) for s in state])

        def unflatten_state(x):
            pos = x[0:3]
            vel = x[3:6]
            quat = x[6:10]
            ang_vel = x[10:13]
            return [pos, vel, quat, ang_vel]

        # Action interpolation function: given time t, find corresponding action by piecewise constant
        def get_action_at_time(current_t):
            idx = pt.searchsorted(t, current_t, right=True) - 1
            idx = pt.clamp(idx, 0, N-1)
            return action_plan[idx]

        def dynamics(t_now, x):
            state = unflatten_state(x)
            action = get_action_at_time(t_now)
            force, moment = self.compute_forces_and_moments(state, action)
            derivatives = self._compute_state_derivative(state, force, moment)
            # Flatten derivatives
            return pt.cat([d.reshape(-1) for d in derivatives])

        x0 = flatten_state(initial_state)
        traj = odeint(dynamics, x0, t, method='dopri5', rtol=1e-4, atol=1e-6)  # shape (N+1, state_dim)

        # Unpack trajectory back to tensors (N+1, 3) etc.
        p = traj[:, 0:3]
        v = traj[:, 3:6]
        q = traj[:, 6:10]
        w = traj[:, 10:13]

        return p, v, q, w, t
    """