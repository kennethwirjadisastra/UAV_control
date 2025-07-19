import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

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
class Vehicle(ABC):
    def __init__(self, position: np.ndarray = None, velocity: np.ndarray = None, 
            quaternion: np.ndarray = None, angular_velocity: np.ndarray = None,
            mass: np.ndarray = None, inertia: np.ndarray = None):
        
        # state of the system
        self.position           = np.array([0.0, 0.0, 0.0] if position is None else position)
        self.velocity           = np.array([0.0, 0.0, 0.0] if velocity is None else velocity)
        self.quaternion         = np.array([0.0, 0.0, 0.0, 1.0] if quaternion is None else quaternion)
        self.angular_velocity   = np.array([0.0, 0.0, 0.0] if angular_velocity is None else angular_velocity)

        # system characteristics
        self.mass               = 1.0 if mass is None else float(mass)
        self.inertia            = np.array(np.eye(3) if inertia is None else inertia)
        self.inv_inertia        = np.linalg.inv(self.inertia)

    @property
    def rotation_matrix(self):
        return R.from_quat(self.quaternion).as_matrix()
    

    def get_state(self):
        return np.array([self.position, self.velocity, self.quaternion, self.angular_velocity], dtype=object)
    

    def set_state(self, state):
        self.position, self.velocity, self.quaternion, self.angular_velocity = state
    

    def _compute_state_derivative(self, state, force, moment):
        p, v, q, w = state  # position, velocity, quaternion, angular velocity
        dpdt = v
        gravity_force = np.array([0,0,-9.81])
        dvdt = force / self.mass + gravity_force
        dqdt = quaternion_derivative(q, w)
        wdot = self.inv_inertia @ (moment - np.cross(w, self.inertia @ w))
        return np.array([dpdt, dvdt, dqdt, wdot], dtype=object)
    
    
    # computes the future state of the system after 1 step of rung-kutta 4th order integration
    def rk4_step(self, state, force, moment, dt):
        def euler_step(state, dynamics, dt):
            return [x + dxdt*dt for x, dxdt in zip(state, dynamics)]

        k1 = self._compute_state_derivative(state, force, moment)
        k2 = self._compute_state_derivative(euler_step(state, k1, dt/2), force, moment)
        k3 = self._compute_state_derivative(euler_step(state, k2, dt/2), force, moment)
        k4 = self._compute_state_derivative(euler_step(state, k3, dt), force, moment)

        next_state = [s + dt/6 * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]) for i, s in enumerate(state)]
        next_state[2] /= np.linalg.norm(next_state[2])
        return next_state
    

    # takes the vehicle in a given state and taking an action
    # returns the resulting net forces and net moments
    @abstractmethod
    def compute_forces_and_moments(self, state, action) -> tuple[np.array, np.array]:
        pass

    def simulate_trajectory(self, initial_state: np.ndarray, action_plan: np.ndarray, dts: np.ndarray):
        N = len(action_plan)

        # 3d position, 3d velocity, 4d quaternion, 3d angular velocity, time
        p = np.zeros((N+1, 3))
        v = np.zeros((N+1, 3))
        q = np.zeros((N+1, 4))
        w = np.zeros((N+1, 3))
        t = np.hstack([[0], np.cumsum(dts, axis=0)])

        state = initial_state
        p[0], v[0], q[0], w[0] = state

        for i, (action, dt) in enumerate(zip(action_plan, dts)):
            force, moment = self.compute_forces_and_moments(state, action)
            state = self.rk4_step(state, force, moment, dt)
            p[i+1], v[i+1], q[i+1], w[i+1] = state

        for z in [p, v, q, w, t[:,None]]:
            print(z.shape)
            
        return p, v, q, w, t

    def __repr__(self):
        return (f"Vehicle(position={self.position}, velocity={self.velocity}, "
                f"quaternion={self.quaternion}, angular_velocity={self.angular_velocity})")
    