import numpy as np
from scipy.spatial.transform import Rotation as R

# necessary functions
def quaternion_derivative(q, omega):
    qx, qy, qz, qw = q
    ox, oy, oz = omega
    dq = 0.5 * np.array([
        qw*ox + qy*oz - qz*oy,
        qw*oy + qz*ox - qx*oz,
        qw*oz + qx*oy - qy*ox,
        -qx*ox - qy*oy - qz*oz
    ])
    return dq

# computes the dynamics of the system given the current state, net forces, net moments, mass and intertia
def dynamics(state, force, moment, mass, inertia):
    p, v, q, w = state  # position, velocity, quaternion, angular velocity

    dpdt = v
    dvdt = force / mass
    dqdt = quaternion_derivative(q, w)
    wdot = np.linalg.inv(inertia) @ (moment - np.cross(w, inertia @ w))

    return np.array([dpdt, dvdt, dqdt, wdot], dtype=object)


# vehicle class
class Vehicle:
    def __init__(self, position: np.ndarray = None, velocity: np.ndarray = None, 
            quaternion: np.ndarray = None, angular_velocity: np.ndarray = None):
        
        self.position = np.array(position if position else [0.0, 0.0, 0.0])
        self.velocity = np.array(velocity if velocity else [0.0, 0.0, 0.0])
        self.quaternion = np.array(quaternion if quaternion else [0.0, 0.0, 0.0, 1.0])
        self.angular_velocity = np.array(angular_velocity if angular_velocity else [0.0, 0.0, 0.0])

    @property
    def rotation(self):
        return R.from_quat(self.quaternion)

    def rk4_step(self, dt):
        state = [self.position, self.velocity, self.quaternion, self.angular_velocity]
        f = self.force
        m = self.moment
        I = self.inertia
        mass = self.mass

        def add_state(state, delta):
            return [s + d for s, d in zip(state, delta)]

        k1 = dynamics(state, f, m, mass, I)
        k2 = dynamics(add_state(state, [dt/2 * k for k in k1]), f, m, mass, I)
        k3 = dynamics(add_state(state, [dt/2 * k for k in k2]), f, m, mass, I)
        k4 = dynamics(add_state(state, [dt * k for k in k3]), f, m, mass, I)

        new_state = [s + dt/6 * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]) for i, s in enumerate(state)]

        # Normalize quaternion
        new_state[2] /= np.linalg.norm(new_state[2])

        self.position, self.velocity, self.quaternion, self.angular_velocity = new_state

    def forward(self, dt: float | np.ndarray, n_steps: int = 64):
        n_steps = len(dt) if not np.isscalar(dt) else n_steps
        time = dt*np.ones(n_steps) if np.isscalar(dt) else np.cumsum(dt)

        if np.isscalar(dt):
            time = np.linspace(0.0, n_steps*dt, num=n_steps+1, endpoint=True)
        else:
            n_steps = len(dt)
            time = np.concatenate([[0.0], np.cumsum(dt)])

        # 3d position, 3d velocity, 4d quaternion, 3d angular velocity, time
        trajectory = np.zeros((n_steps+1, 14))
        
        trajectory[0] = np.concatenate([self.get_state().copy(), [time]])

        for i in range(n_steps):
            # update state
            self.set_steate(self.rk4_step(dt[i]))
            time += dt[i]

            trajectory[i+1] = np.concatenate([self.get_state.copy(), [time]])
        
        return trajectory

    def __repr__(self):
        return (f"Vehicle(position={self.position}, velocity={self.velocity}, "
                f"quaternion={self.quaternion}, angular_velocity={self.angular_velocity})")
