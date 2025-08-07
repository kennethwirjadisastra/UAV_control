import torch as pt
from torchdiffeq import odeint
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from util.quaternion import quaternion_derivative, quaternion_to_matrix

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
        p, v, q, w = state  # position, velocity, quaternion, angular velocity
        dpdt = v
        dvdt = force / self.mass
        dqdt = quaternion_derivative(q, w)
        wdot = self.inv_inertia @ (moment - pt.linalg.cross(w, self.inertia @ w))
        return [dpdt, dvdt, dqdt, wdot]

    def rk4_step(self, state, force, moment, dt):
        # Computes the future state of the system after 1 step of Runge-Kutta 4th order integration
        def euler_step(state, dynamics, dt):
            return [x + dxdt * dt for x, dxdt in zip(state, dynamics)]

        k1 = self._compute_state_derivative(state, force, moment)
        k2 = self._compute_state_derivative(euler_step(state, k1, dt/2), force, moment)
        k3 = self._compute_state_derivative(euler_step(state, k2, dt/2), force, moment)
        k4 = self._compute_state_derivative(euler_step(state, k3, dt), force, moment)

        next_state = [s + dt/6 * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]) for i, s in enumerate(state)]
        next_state[2] = next_state[2] / (pt.norm(next_state[2]) + 1e-12)
        return next_state

    @abstractmethod
    def compute_forces_and_moments(self, state, action):
        # Takes the vehicle in a given state and taking an action
        # Returns the resulting net forces and net moments
        pass

    def simulate_trajectory(self, initial_state, action_plan, dts):
        # Simulates the system forward in time given a sequence of actions and time steps
        N = len(action_plan)

        # 3d position, 3d velocity, 4d quaternion, 3d angular velocity, time
        p = pt.zeros((N+1, 3))
        v = pt.zeros((N+1, 3))
        q = pt.zeros((N+1, 4))
        w = pt.zeros((N+1, 3))
        t = pt.cat([pt.tensor([0.0]), pt.cumsum(pt.tensor(dts), dim=0)])

        state = initial_state
        p[0], v[0], q[0], w[0] = state

        for i, (action, dt) in enumerate(zip(action_plan, dts)):
            force, moment = self.compute_forces_and_moments(state, action)
            state = self.rk4_step(state, force, moment, dt)
            p[i+1], v[i+1], q[i+1], w[i+1] = state
        
        # for z in [p,v,q,t[:,None]]:
        #     print(z.shape)

        return p, v, q, w, t
    


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