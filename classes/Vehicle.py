import torch as pt
# from torchdiffeq import odeint
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from util.quaternion import quaternion_derivative, quaternion_to_matrix
from util.functions import add_default_arg
from classes.StateTensor import StateTensor
# from classes.ActionPlan import ActionPlan


# vehicle class
class Vehicle(ABC):
    actions = []        # List of actions the vehicle can do ex: Car(Vehicle) has actions = ['Accelerate', 'Steer']

    def __init__(self, state: StateTensor = None, mass: float = None, inertia: pt.Tensor = None, **kwargs):
        self.dtype              = add_default_arg(kwargs,  'dtype',    pt.float32)
        self.device             = add_default_arg(kwargs,  'device',   None)
        self.actions            = add_default_arg(kwargs,  'actions',  {})

        # state of the system
        self.state              = StateTensor(**kwargs) if state is None else state

        # system characteristics
        self.mass               = pt.tensor(1, dtype=self.dtype, device=self.device) if mass is None else pt.as_tensor(mass, dtype=self.dtype, device=self.device) 
        self.inertia            = pt.eye(3, dtype=self.dtype, device=self.device) if inertia is None else inertia.clone().detach().to(dtype=self.dtype, device=self.device)
        self.inv_inertia        = pt.linalg.inv(self.inertia)


    def _state_derivative(self, state, force, moment):
        q, w = state[..., 6:10], state[..., 10:13]

        return pt.cat([
            state[..., 3:6], 
            force / self.mass, 
            quaternion_derivative(q, w),
            #(moment - pt.cross(w, w @ self.inertia, dim=-1)) @ self.inv_inertia
            self.inv_inertia @ (moment - pt.cross(w, self.inv_inertia @ w, dim=-1))
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

        raw_next_state = state + (k1 + 2*k2 + 2*k3 + k4) * dt/6
        next_state = pt.cat([
            raw_next_state[..., 0:6], 
            raw_next_state[..., 6:10] / (pt.norm(raw_next_state[...,6:10], dim=-1, keepdim=True) + 1e-12),
            raw_next_state[..., 10:13]
        ], dim=-1)
        return next_state

    @abstractmethod
    def compute_forces_and_moments(self, state, action):
        # Takes the vehicle in a given state and taking an action
        # Returns the resulting net forces and net moments
        pass

    # def simulate_trajectory(self, initial_state: StateTensor, action_plan: ActionPlan, dt: float):
    #     # initial state     = (B, 13)
    #     # AP.action         = (B, M, D)
    #     # AP.delta_time     = (B, M)
    #     # AP.time           = (B, M)

    #     # N                 = ceil((AP.tf - AP.t0) / dt)

    #     # action tensor     = (B, N, D)
    #     # dt tensor         = (B, N)

    #     action_tensor       = action_plan.rasterize()


    #     # Simulates the system forward in time given a sequence of actions and time steps
    #     N                   = action_plan.shape[-2]                                     # N is the number of timesteps
    #     time                = pt.cat([dts[..., 0:1]*0, pt.cumsum(dts, dim=-1)])
    #     states_list         = [initial_state]

    #     for i in range(N):
    #         action, dt      = action_plan[..., i, :], dts[..., i]
    #         force, moment   = self.compute_forces_and_moments(states_list[-1], action)
    #         states_list.append(
    #             self.rk4_step(states_list[-1], force, moment, dt)
    #         )

    #     states_tensor = pt.stack(states_list)
    #     # states = StateTensor(states, requires_grad=states.requires_grad)

    #     # return states.pos, states.vel, states.quat, states.angvel, time
    #     return states_tensor[...,0:3], states_tensor[...,3:6], states_tensor[...,6:10], states_tensor[...,10:13], time
    

    def simulate_trajectory(self, initial_state: StateTensor, action_tensor: pt.Tensor, dts: pt.Tensor) -> tuple[StateTensor, pt.Tensor]:
        # initial state     = (B, 13)
        # AP.action         = (B, M, D)
        # AP.delta_time     = (B, M)

        # Simulates the system forward in time given a sequence of actions and time steps
        N                   = action_tensor.shape[-2]                                     # N is the number of timesteps
        time                = pt.cat([dts[..., 0:1]*0, pt.cumsum(dts, dim=-1)])
        states_list         = [initial_state]

        for i in range(N):
            action, dt      = action_tensor[..., i, :], dts[..., i]
            force, moment   = self.compute_forces_and_moments(states_list[-1], action)
            states_list.append(
                self.rk4_step(states_list[-1], force, moment, dt)
            )

        states_tensor = StateTensor.from_tensor(pt.stack(states_list))
        # states = StateTensor(states, requires_grad=states.requires_grad)

        # return states.pos, states.vel, states.quat, states.angvel, time
        # return states_tensor[...,0:3], states_tensor[...,3:6], states_tensor[...,6:10], states_tensor[...,10:13], time
        return states_tensor, time
    
    


    def __repr__(self):
        return (f"Vehicle(position={self.position}, velocity={self.velocity}, "
                f"quaternion={self.quaternion}, angular_velocity={self.angular_velocity})")
    