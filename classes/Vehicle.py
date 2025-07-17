import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

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
        self.inv_inertia        = np.linalg.inv(self.inertia)


    @property
    def rotation(self):
        return R.from_quat(self.quaternion)
    
    def get_state(self):
        return np.array([self.position, self.velocity, self.quaternion, self.angular_velocity], dtype=object)
    
    def set_state(self, state):
        self.position, self.velocity, self.quaternion, self.angular_velocity = state

    def state_13D(self):
        state = np.zeros(13)
        state[0:3] = self.position
        state[3:6] = self.velocity
        state[6:10] = self.quaternion
        state[10:] = self.angular_velocity
        return state
    
    # computes the dynamics of the system given the current state, net force and net moment
    def dynamics(self, state, force, moment):
        p, v, q, w = state  # position, velocity, quaternion, angular velocity
        dpdt = v
        gravity_force = np.array([0,0,-9.81])
        dvdt = force / self.mass + gravity_force
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
    
    def forward(self, dt: float | np.ndarray, 
                force=None, moment=None, n_steps: int = 64):

        if np.isscalar(dt):
            time = np.linspace(0.0, n_steps*dt, num=n_steps+1, endpoint=True) # (n_steps+1,)
            dt = dt*np.ones(n_steps)
        else:
            n_steps = len(dt)
            time = np.concatenate([[0.0], np.cumsum(dt)]) # (n_steps+1,)
        
        force = np.zeros((n_steps, 3)) if force is None else np.asarray(force)
        moment = np.zeros((n_steps, 3)) if moment is None else np.asarray(moment)

        if force.ndim == 1:
            force = np.tile(force, (n_steps, 1)) # (n_steps, 3)
        if moment.ndim == 1:
            moment = np.tile(moment, (n_steps, 1)) # (n_steps, 3)

        assert force.shape == (n_steps, 3), "force must be (n_steps, 3)"
        assert moment.shape == (n_steps, 3), "moment must be (n_steps, 3)"

        # 3d position, 3d velocity, 4d quaternion, 3d angular velocity ==> 13 dimensional state
        path = np.zeros((n_steps+1, 13))
        
        path[0] = self.state_13D()

        for i in range(n_steps):
            # update state
            self.set_state(self.rk4_step(self.get_state(), force[i], moment[i], dt[i]))
            path[i+1] = self.state_13D()
        
        trajectory = np.hstack([path, time[:,None]]) # (n_steps, 14) includes time

        return trajectory

    def __repr__(self):
        return (f"Vehicle(position={self.position}, velocity={self.velocity}, "
                f"quaternion={self.quaternion}, angular_velocity={self.angular_velocity})")
if __name__ == '__main__':
    UAV = Vehicle()
    dt = 0.01
    n_steps = 1000
    force = np.array([0, 0, 9.81]) * UAV.mass # oppose gravity (should hover in space)

    trajectory = UAV.forward(dt, force=force, n_steps=n_steps)

    plt.plot(trajectory[:,-1], trajectory[:,2], label='Aircraft Height (m)')
    plt.title('Aircraft Height')
    plt.xlabel('Time (s)')
    plt.ylabel('Height (m)')
    plt.legend()
    plt.grid()
    plt.show()