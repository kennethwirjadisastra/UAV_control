import torch as pt
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

# necessary functions
def quaternion_derivative(q, omega):
    # Computes the time derivative of a quaternion given angular velocity omega
    qw, qx, qy, qz = q
    ox, oy, oz = omega
    dq = 0.5 * pt.stack([
        -qx*ox - qy*oy - qz*oz,
         qw*ox + qy*oz - qz*oy,
         qw*oy + qz*ox - qx*oz,
         qw*oz + qx*oy - qy*ox
    ])
    return dq

# vehicle class
class Vehicle(ABC):
    def __init__(self, position=None, velocity=None, 
                 quaternion=None, angular_velocity=None,
                 mass=None, inertia=None):

        # state of the system
        self.position           = pt.tensor([0.0, 0.0, 0.0] if position is None else position, dtype=pt.float32)
        self.velocity           = pt.tensor([0.0, 0.0, 0.0] if velocity is None else velocity, dtype=pt.float32)
        self.quaternion         = pt.tensor([1.0, 0.0, 0.0, 0.0] if quaternion is None else quaternion, dtype=pt.float32)
        self.angular_velocity   = pt.tensor([0.0, 0.0, 0.0] if angular_velocity is None else angular_velocity, dtype=pt.float32)

        # system characteristics
        self.mass               = 1.0 if mass is None else float(mass)
        self.inertia            = pt.tensor(pt.eye(3) if inertia is None else inertia, dtype=pt.float32)
        self.inv_inertia        = pt.linalg.inv(self.inertia)

    @property
    def rotation_matrix(self):
        # Returns rotation matrix corresponding to current quaternion (scalar-first convention)
        return pt.tensor(R.from_quat(self.quaternion.tolist(), scalar_first=True).as_matrix(), dtype=pt.float32)

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
        wdot = self.inv_inertia @ (moment - pt.cross(w, self.inertia @ w))
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
        
        for z in [p,v,q,t[:,None]]:
            print(z.shape)

        return p, v, q, w, t

    def __repr__(self):
        return (f"Vehicle(position={self.position}, velocity={self.velocity}, "
                f"quaternion={self.quaternion}, angular_velocity={self.angular_velocity})")
    