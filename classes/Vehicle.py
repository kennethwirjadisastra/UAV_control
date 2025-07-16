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

# vehicle class
class Vehicle:
    def __init__(self, position: np.ndarray = None, velocity: np.ndarray = None, 
            quaternion: np.ndarray = None, angular_velocity: np.ndarray = None,
            mass: np.ndarray = None, inertia: np.ndarray = None):
        
        # state of the system
        self.position           = np.array(position if position else [0.0, 0.0, 0.0])
        self.velocity           = np.array(velocity if velocity else [0.0, 0.0, 0.0])
        self.quaternion         = np.array(quaternion if quaternion else [0.0, 0.0, 0.0, 1.0])
        self.angular_velocity   = np.array(angular_velocity if angular_velocity else [0.0, 0.0, 0.0])

        # system characteristics
        self.mass               = np.array(mass if mass else [1])
        self.inertia            = np.array(inertia if inertia else np.eye(3))
        self.inv_inertia        = np.linalg.inv(inertia)


    @property
    def rotation(self):
        return R.from_quat(self.quaternion)
    
    def get_state(self):
        return np.array([self.position, self.velocity, self.quaternion, self.angular_velocity], dtype=object)
    
    def set_state(self, state):
        self.position, self.velocity, self.quaternion, self.angular_velocity = state
    
    # computes the dynamics of the system given the current state, net force and net moment
    def dynamics(self, state, force, moment):
        p, v, q, w = state  # position, velocity, quaternion, angular velocity
        dpdt = v
        dvdt = force / self.mass
        dqdt = quaternion_derivative(q, w)
        wdot = self.inv_inertia @ (moment - np.cross(w, self.inertia @ w))
        return np.array([dpdt, dvdt, dqdt, wdot], dtype=object)
    

    # computes the future state of the system after 1 step of rung-kutta 4 integration
    def rk4_step(self, state, force, moment, dt):
        def euler_step(state, dynamics, dt):
            return [x + dxdt*dt for x, dxdt in zip(state, dynamics)]

        k1 = self.dynamics(state, force, moment)
        k2 = self.dynamics(euler_step(state, k1, dt/2), force, moment)
        k3 = self.dynamics(euler_step(state, k2, dt/2), force, moment)
        k4 = self.dynamics(euler_step(state, k3, dt), force, moment)

        next_state = [s + dt/6 * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]) for i, s in enumerate(state)]
        next_state[2] /= np.linalg.norm(next_state[2])
        return next_state

    def __repr__(self):
        return (f"Vehicle(position={self.position}, velocity={self.velocity}, "
                f"quaternion={self.quaternion}, angular_velocity={self.angular_velocity})")
