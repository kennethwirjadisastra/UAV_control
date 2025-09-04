import torch as pt
from util.functions import add_default_arg
from util.quaternion import quaternion_to_matrix
from classes.Vehicle import Vehicle

# shape of the plan is (*B, N, A+1)
# expects a time tensor of shape (*B, N, 1)
class ActionPlan:
    def __init__(self, vehicle: Vehicle, delta_time_tensor: pt.tensor, action_tensor: pt.tensor= None, **kwargs):
        # Tensor kwargs
        add_default_arg(kwargs, 'dtype', pt.float32)
        add_default_arg(kwargs, 'device', None)
        requires_grad = kwargs.get('requires_grad', False)
        kwargs.pop('requires_grad', None)

        # Plan variables
        self.B              = delta_time_tensor.shape[:-1]                                                                          # Batch size
        self.N              = delta_time_tensor.shape[-1]                                                                           # Number of timesteps
        self.D              = len(vehicle.__class__.actions)                                                                        # Action dimensions
        self.delta_time     = delta_time_tensor                                                                                     # (B, N)
        self.time           = pt.cumsum(delta_time_tensor, dim=-1)                                                                  # (B, N)
        self.action         = action_tensor if action_tensor is not None else pt.zeros((*self.B, self.N, self.D), **kwargs)         # (B, N, D)
        
    # Takes action start time and delta time stepsize
    # returns the average action over the step period
    def rasterize(self, max_dt: float, k=4):
        N           = pt.ceil(pt.max(self.time[..., -1]) / max_dt)                  # (1)
        dt          = self.time[..., -1] / N                                        # (B)
        ts          = pt.arange(N, dtype=pt.float32).unsqueeze(0) * dt              # (1, N) * (B) -> (B, N)
        frac        = pt.linspace(1 / (2*k), 1 - 1 / (2*k), k)                      # (k)                           ex: (1/8, 3/8, 5/8, 7/8) for k = 4

        ts_sub      = ts.unsqueeze(-1) + dt.view(-1, 1, 1) * frac                   # (B, N, 1) + (B, 1, 1) * (k) -> (B, N, k)
        idx         = pt.searchsorted(self.time, ts_sub)                            # (*M, k) finds indices of action that is being taken at each sub divided timestep
        
        dims        = pt.arange(self.D)                                             # (D)
        action_sub  = self.action[..., idx.unsqueeze(-1), dims]                     # (*B, *M, k, D)
        action      = action_sub.mean(dim=-2)                                       # (*B, M, D)
        return action
        
if __name__ == '__main__':
    from classes.Car import Car

    ap = pt.tensor([    # (7 + 1, 2)
        [0.01, 0.02],
        [0.04, 0.05],
        [0.08, 0.09],
        [0.11, 0.15],
        [0.21, 0.27],
        [0.33, 0.39],
        [0.54, 0.87],
        [0.00, 0.00],             # added at the end to indicate no action after the plan ends
    ])

    t = pt.tensor([     # (7)
        0,
        0.5,
        1.25,
        1.6,
        2.1,
        2.4,
        3.3,
    ])
    dt = pt.cat([pt.tensor([0]), t[1:] - t[:-1]])

    print(dt)

    car = Car()

    plan = ActionPlan(car, dt, ap)

    # k = 4
    # frac = pt.linspace(1 / (2*k), 1 - 1 / (2*k), k)
    # print(frac)
    # ts = pt.tensor([[1, 1.5, 2], [3, 7, 8]])        # (2, 3)
    # dts = pt.tensor([[0.5, 0.5, 1], [4, 1, 2]])
    # ts_sub = ts.unsqueeze(-1) + dts.unsqueeze(-1) * frac
    # print(ts_sub.shape)

    # dims = pt.arange(2)

    # idx = pt.searchsorted(t, ts_sub)                        # (2, 3)
    # action_sub = ap[..., idx.unsqueeze(-1), dims]           # (6, 2)
    # print(action_sub)
    # action = action_sub.float().mean(dim=-1, keepdim=False)
    # print(action)


