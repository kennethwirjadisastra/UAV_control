import torch
import matplotlib.pyplot as plt

class WaypointPath:
    def __init__(self, t, waypoints):
        self.n_steps = len(t) - 1
        self.t = t
        self.m_waypoints = len(waypoints)

        self.waypoints = waypoints.type(torch.float32)              # (M,2)
        self.segs = self.waypoints[1:,:] - self.waypoints[:-1,:]    # (M-1,2)

        self.seg_lens = torch.norm(self.segs, dim=1)  # length of each linear segment
        self.total_len = torch.sum(self.seg_lens)     # length of all linear segments combined
        
        self.cum_lens = torch.cat([torch.tensor([0], device=waypoints.device),
                                   torch.cumsum(self.seg_lens, dim=0)],
                                   dim=0) # cumulative length of path at each waypoint
        
    def normalize(self):
        self.norm_cum_lens = self.cum_lens / self.total_len # normalized cumulative length at each waypoint
        self.norm_seg_lens = self.norm_cum_lens[1:] - self.norm_cum_lens[:-1] # length of each segment in the distance normalized path
        

    def normalize(self):
        self.norm_cum_lens = self.cum_lens / self.total_len # normalized cumulative length at each waypoint
        self.norm_seg_lens = self.norm_cum_lens[1:] - self.norm_cum_lens[:-1] # length of each segment in the distance normalized path
        

    def step_targets(self):
        # normalize path length by distance
        self.normalize()

        step_idx = torch.searchsorted(self.norm_cum_lens[1:], self.t, right=True) # segment index that each step falls after
        step_idx = torch.clamp(step_idx, 0, self.m_waypoints - 2)

        # Start end end segments of each step's cumulative normalized distance
        seg_start = self.waypoints[step_idx]
        seg_end = self.waypoints[step_idx + 1]

        norm_seg_start = self.norm_cum_lens[step_idx] # starting normalized distance of each step's segment
        norm_seg_len = self.norm_seg_lens[step_idx] # segment length of each step's segment

        # distance from normalized cumulative segment start to normalized cumulative step
        step_lens = ((self.t - norm_seg_start) / norm_seg_len).reshape(-1,1)

        # compute target position of each step
        step_targets = seg_start + step_lens*(seg_end - seg_start)
        
        return step_targets

    


x = torch.linspace(0,2*torch.pi,50)
sinx = torch.sin(x)
waypoints = torch.stack([x, sinx], dim=1)


n_steps = 10
# N steps ==> N+1 waypoints
t = torch.linspace(0, 1, n_steps+1) # step lengths along normalized path

path = WaypointPath(t, waypoints)
path.normalize()
path_steps = path.step_targets()


plt.scatter(*path.waypoints.T, label='Waypoints')
plt.plot(*path.waypoints.T, ls='--', label='Linear Path')
plt.scatter(*path_steps.T, marker='*', label='Step Targets', zorder=2, s=200)
plt.legend()
plt.show()