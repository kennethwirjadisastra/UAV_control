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

    def simulate_trajectory(self, initial_state, action_plan, dt):
        pass

    def __repr__(self):
        return (f"Vehicle(position={self.position}, velocity={self.velocity}, "
                f"quaternion={self.quaternion}, angular_velocity={self.angular_velocity})")
    

###################################
## ---------- testing ---------- ##
###################################

if __name__ == '__main__':
    position = np.array([0.0,0.0,1000.0])
    velocity = np.array([100.0, 200.0, 0.0])
    quaternion = [0.0, 0.0, 0.0, 1.0]
    angular_velocity = [0.0, 0.0, 0.0]
    mass = 1
    inertia = np.eye(3)

    dt = 0.01
    n_steps = 1000

    force = np.zeros((n_steps, 3))
    x = np.linspace(-np.pi,np.pi,n_steps)
    force[:,0] = 50*np.cos(x)
    force[:,1] = np.exp(x)
    force[:,2] = 9.81*mass*np.ones(n_steps) # oppose gravity (should remain at constant height)

    UAV = Vehicle(position=position, velocity=velocity, quaternion=quaternion, angular_velocity=angular_velocity,
                  mass=mass, inertia=inertia)

    p, v, q, w, t = UAV.forward(dt, force=force, n_steps=n_steps)
    
    fig = plt.figure(figsize=(16,6))
    ax1 = fig.add_subplot(1, 2, 1, projection=None)
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')

    ax1.plot(t, p[:,2], label='Aircraft Height (m)')
    ax1.set_title('Aircraft Height')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Height (m)')
    ax1.legend()
    ax1.grid()

    ax2.plot(*p.T, label='Aircraft Trajectory')
    ax2.scatter(*p[0,:], label='Initial Position')
    ax2.scatter(*p[-1,:], label='Final Position')
    ax2.set_title('Aircraft Trajectory')
    ax2.set_xlabel('X Position')
    ax2.set_ylabel('Y Position')
    ax2.set_zlabel('Z Position')
    ax2.grid()
    ax2.legend()
    plt.show()