import torch as pt
from torchdiffeq import odeint
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from util.quaternion import quaternion_derivative, quaternion_to_matrix

class VehicleState():
    def __init__(self, state=None, pos=None, vel=None, quat=None, angvel=None):
        self.state = pt.tensor([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=pt.float32) if state is None else state  
        if pos is not None:
            self.state[0:3] = pt.as_tensor(pos, dtype=pt.float32)
        if vel is not None:
            self.state[3:6] = pt.as_tensor(vel, dtype=pt.float32)
        if quat is not None:
            self.state[6:10] = pt.as_tensor(quat, dtype=pt.float32)
        if angvel is not None:
            self.state[10:13] = pt.as_tensor(angvel, dtype=pt.float32)

    @property
    def pos(self):
        return self.state[0:3]
    
    @property
    def vel(self):
        return self.state[3:6]
    
    @property
    def quat(self):
        return self.state[6:10]
    
    @property
    def angvel(self):
        return self.state[10:13]



# vehicle class
class Vehicle(ABC):
    def __init__(self, position=None, velocity=None, 
                 quaternion=None, angular_velocity=None,
                 mass=None, inertia=None):

        # state of the system
        self.position           = pt.tensor([0.0, 0.0, 0.0], dtype=pt.float32) if position is None else position.clone().detach().to(dtype=pt.float32)
        self.velocity           = pt.tensor([0.0, 0.0, 0.0], dtype=pt.float32) if velocity is None else velocity.clone().detach().to(dtype=pt.float32)
        self.quaternion         = pt.tensor([1.0, 0.0, 0.0, 0.0], dtype=pt.float32) if quaternion is None else quaternion.clone().detach().to(dtype=pt.float32)
        self.angular_velocity   = pt.tensor([0.0, 0.0, 0.0], dtype=pt.float32) if angular_velocity is None else angular_velocity.clone().detach().to(dtype=pt.float32)

        # system characteristics
        self.mass               = 1.0 if mass is None else float(mass)
        self.inertia            = pt.eye(3, dtype=pt.float32) if inertia is None else inertia.clone().detach().to(dtype=pt.float32)
        self.inv_inertia        = pt.linalg.inv(self.inertia)

    @property
    def rotation_matrix(self):
        # Returns rotation matrix corresponding to current quaternion (scalar-first convention)
        return quaternion_to_matrix(self.quaternion)

    def get_state(self):
        # Returns current state as list of tensors
        return [self.position, self.velocity, self.quaternion, self.angular_velocity]

    def set_state(self, state):
        # Sets state from list of tensors
        self.position, self.velocity, self.quaternion, self.angular_velocity = state

    def _compute_state_derivative(self, state, force, moment):
        # Computes derivative of state given forces and moments
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

        k1 = self._compute_state_derivative(state, force, moment)
        k2 = self._compute_state_derivative(state + k1*dt/2, force, moment)
        k3 = self._compute_state_derivative(state + k2*dt/2, force, moment)
        k4 = self._compute_state_derivative(state + k3*dt, force, moment)

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
        N   = action_plan.shape(-2)         # N is the number of timesteps
        B   = initial_state.shape[:-2]      # B is the tensor of trajectories


        # 3d position, 3d velocity, 4d quaternion, 3d angular velocity, time
        states  = pt.zeros((*B, N+1, 13))
        time    = pt.cat([dts[..., 0]*0, pt.cumsum(dts, dim=-1)])

        state = initial_state
        states[..., 0, :] = state

        for i in range(N):
            action              = action_plan[..., i, :]
            dt                  = dts[..., i]
            force, moment       = self.compute_forces_and_moments(state, action)
            state               = self.rk4_step(state, force, moment, dt)
            state[..., i+1, :]  = state

        return state[..., 0:3], state[..., 3:6], state[..., 6:10], state[..., 10:13], time
    


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