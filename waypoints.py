import numpy as np
import matplotlib.pyplot as plt

# Function that interpolates between the waypoints (M,d) normalized by length
# creates targets for the path to converge to
# waypoints includes the starting location of the vehicle
class TargetPath:
    # initialized with a list of waypoints (atleast 1 for the vehicles current location and one for the final target)
    def __init__(self, waypoints: np.array):
        assert waypoints.ndim == 2
        self.waypoints = waypoints
        self.M = waypoints.shape[0]

        # precompute the cumulative lengths and scaled segment vectors
        segments = waypoints[1:,:] - waypoints[:-1,:]
        lengths = np.linalg.norm(segments, axis=-1)
        total_length = np.sum(lengths)
        norm_lenghts = lengths / total_length
        self.normalized_cumulative_lengths = np.concat([[0], np.cumsum(norm_lenghts)])
        self.scaled_segment_vectors = (segments / lengths[:,None]) * total_length

    def normalized_interpolate(self, t: np.array) -> np.array:
        indices = np.searchsorted(self.normalized_cumulative_lengths, t, side='right') - 1
        indices = np.clip(indices, 0, self.M - 2)

        residual_ts = t - self.normalized_cumulative_lengths[indices]
        return self.waypoints[indices] + residual_ts[:,None] * self.scaled_segment_vectors[indices]

    

s = np.linspace(0, 2, 33)
x = np.cos(3*s)
y = np.sin(s)
waypoints = np.stack([x, y], axis=1)

N = 10
t = np.linspace(0, 1, N+1)

targetPath = TargetPath(waypoints)
trajectoryTargets = targetPath.normalized_interpolate(t)

plt.scatter(*targetPath.waypoints.T, label='Waypoints')
plt.plot(*targetPath.waypoints.T, ls='--', label='Linear Path')
plt.scatter(*trajectoryTargets.T, marker='*', label='Step Targets', zorder=2, s=200)
plt.legend()
plt.show()