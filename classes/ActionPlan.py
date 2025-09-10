from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from classes.Vehicle import Vehicle

import torch as pt
from util.functions import add_default_arg
from util.quaternion import quaternion_to_matrix

import matplotlib.pyplot as plt
import numpy as np

# shape of the plan is (*B, N, A+1)
# expects a time tensor of shape (*B, N, 1)
class ActionPlan:
    def __init__(self, vehicle: Vehicle, resolution: int = 20, min_dt: float = 0.05, **kwargs):
        # Tensor kwargs
        add_default_arg(kwargs, 'dtype', pt.float32)
        add_default_arg(kwargs, 'device', None)
        # requires_grad = kwargs.get('requires_grad', False)
        kwargs.pop('requires_grad', None)

        self.vehicle        = vehicle

        # Plan variables
        self.R              = resolution                                # Number of timesteps
        self.D              = len(vehicle.__class__.actions)            # Action dimensions         cumsum([1, 2, 3]) = [1, 3, 6]
        self.delta_time     = pt.ones((self.R,), **kwargs) * min_dt     # (R)
        self.action         = pt.zeros((self.R + 1, self.D), **kwargs)  # (R + 1, D)                1 more action than the dts for linear interpolate
        self.kwargs         = kwargs
        self.min_dt         = min_dt

        # require gradient
        self.action.requires_grad_()
        self.delta_time.requires_grad_()

        self.update()
        self.print()

    def print(self):
        print(self.action, self.delta_time, self.time)

    def set(self, action: pt.Tensor = None, delta_time: pt.tensor = None):
        if action is not None:
            with pt.no_grad():
                self.action = pt.as_tensor(action)
        if delta_time is not None:
            with pt.no_grad():
                self.delta_time = pt.as_tensor(delta_time)
            self.update()

    def update(self):
        with pt.no_grad():
            self.delta_time     = pt.clamp(self.delta_time, min=self.min_dt)
        self.time           = pt.cat([pt.tensor([0]), pt.cumsum(self.delta_time, dim=-1)])        # (R + 1)
        
    # max_dt is the largest dt acceptable for the integrator
    # k is the number of subdivisions for each action sample
    # returns the average action at each timestep and the dts
    def rasterize(self, max_dt: float, k: int = 4) -> tuple[pt.Tensor, pt.Tensor]:
        N           = pt.ceil(self.time[-1] / max_dt).to(pt.int64)          # (1)
        dt          = self.time[-1] / N                                     # (1)
        ts          = pt.arange(N, **self.kwargs) * dt                      # (N) * (1) -> (N)
        frac        = dt * pt.linspace(1 / (2*k), 1 - 1 / (2*k), k)         # (k)                           ex: (1/8, 3/8, 5/8, 7/8) for k = 4
        ts_sub      = ts.unsqueeze(-1) + frac                               # (N, 1) + (k) -> (N, k)

        with pt.no_grad():
            idx_0   = pt.searchsorted(self.time, ts_sub) - 1                # (N, k) finds indices of action that is being taken at each sub divided timestep
            idx_1   = pt.clamp(idx_0 + 1, 0, self.R)
            dims    = pt.arange(self.D).unsqueeze(-1)                       # (D, 1)


        r_1         = ((self.time[idx_1] - ts_sub) / self.delta_time[idx_0]).unsqueeze(-2)
                                                                            # r_1 = 1 - r, proportion across the current action, Should be bounded by [0,]
        a_0         = self.action[idx_0.unsqueeze(-2), dims]                # action[(N, 1, k), (D, 1)] -> (N, k, D) - first action in the lerp
        a_1         = self.action[idx_1.unsqueeze(-2), dims]                # action[(N, 1, k), (D, 1)] -> (N, k, D) - second action in the lerp
        action_sub  = r_1 * a_0 + (1-r_1) * a_1                             # linear interpolation of the subdivided actions

        action      = action_sub.mean(dim=-1)                               # (N, D)
        return action, pt.ones((N,)) * dt
    
    def plot(self, max_dt: float):
        time = self.time.detach().cpu().numpy()
        action = self.action.detach().cpu().numpy()

        action_r, dt_r = self.rasterize(max_dt, k=1)
        time_r_1 = pt.cumsum(dt_r, dim=-1)
        time_r_0 = time_r_1 - dt_r

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
            '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', 
            '#bcbd22', '#17becf']
        

        for i in range(self.D):
            plt.plot(time, action[:, i], '--', label=self.vehicle.__class__.actions[i], color = colors[i])
            label = [f'Rasterized {self.vehicle.__class__.actions[i]}'] + [None for _ in range(time_r_0.shape[0]-1)]
            plt.plot([time_r_0, time_r_1], [action_r[:, i], action_r[:, i]], label=label, color = colors[i])

        plt.legend()
        plt.show()

###################################
## ---------- testing ---------- ##
###################################


if __name__ == '__main__':
    from classes.Car import Car

    car = Car()

    plan = ActionPlan(car, 3, 0.01)

    plan.set(
        action      = [[1, 2], [3, 1], [4, 5], [0, 2]],
        delta_time  = [0.5, 1, 0.3]
    )

    plan.print()
    plan.plot(0.05)


