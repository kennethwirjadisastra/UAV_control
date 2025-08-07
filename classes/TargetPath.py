import torch as pt
import matplotlib.pyplot as plt

# Function that interpolates between the waypoints (M,d) normalized by length
# creates targets for the path to converge to
# waypoints includes the starting location of the vehicle
class TargetPath:
    # initialized with a list of waypoints (at least 1 for the vehicles current location and one for the final target)
    def __init__(self, waypoints: pt.tensor):
        assert waypoints.ndim == 2
        self.waypoints = waypoints
        self.M = waypoints.shape[0]

        # precompute the cumulative lengths and scaled segment vectors
        segments = waypoints[1:,:] - waypoints[:-1,:]
        lengths = pt.linalg.norm(segments, axis=-1)
        self.total_length = pt.sum(lengths)
        norm_lenghts = lengths / self.total_length
        print(norm_lenghts.shape)
        self.normalized_cumulative_lengths = pt.concatenate([pt.zeros(1), pt.cumsum(norm_lenghts, dim=0)])
        self.scaled_segment_vectors = (segments / lengths[:,None]) * self.total_length

    def normalized_interpolate(self, t: pt.tensor) -> pt.tensor:
        indices = pt.searchsorted(self.normalized_cumulative_lengths, t, side='right') - 1
        indices = pt.clamp(indices, 0, self.M - 2)

        residual_ts = t - self.normalized_cumulative_lengths[indices]
        return self.waypoints[indices] + residual_ts[:,None] * self.scaled_segment_vectors[indices]
    
    def distance_interpolate(self, d: pt.tensor) -> pt.tensor:
        return self.normalized_interpolate(d / self.total_length)

###################################
## ---------- testing ---------- ##
###################################

if __name__ == '__main__':
    s = pt.linspace(0, 2, 33)
    x = s
    y = 1 + pt.cos(s)
    waypoints = pt.stack([x, y], axis=1)

    N = 10
    t = pt.linspace(0, 1, N+1)

    targetPath = TargetPath(waypoints)
    trajectoryTargets = targetPath.distance_interpolate(t)

    plt.scatter(*targetPath.waypoints.T, label='Waypoints')
    plt.plot(*targetPath.waypoints.T, ls='--', label='Linear Path')
    plt.scatter(*trajectoryTargets.T, marker='*', label='Step Targets', zorder=2, s=200)
    plt.legend()
    plt.axis('equal')
    plt.show()

    print(trajectoryTargets.shape)