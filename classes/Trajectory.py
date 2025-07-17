import torch as pt
import numpy as np
import matplotlib.pyplot as plt
import Vehicle

from Vehicle import Vehicle  # assuming Vehicle is an ABC

class DummyVehicle(Vehicle):
    def compute_forces_and_moments(self, *args, **kwargs):
        raise NotImplementedError("This dummy vehicle requires you to manually provide forces and moments.")
    

class Trajectory:
    def __init__(self, vehicle):
        self.vehicle = vehicle


    def forward(self, actions: np.ndarray, dt: np.ndarray, moments: np.ndarray=None):
        vehicle = self.vehicle
        N = len(actions)
        moments = np.zeros((N, 3)) if moments is None else moments

        assert len(actions) == len(moments) == len(dt) - 1

        # 3d position, 3d velocity, 4d quaternion, 3d angular velocity, time
        p = np.zeros((N+1, 3))
        v = np.zeros((N+1, 3))
        q = np.zeros((N+1, 4))
        w = np.zeros((N+1, 3))
        t = np.cumsum(dt, axis=0)

        p[0], v[0], q[0], w[0] = vehicle.get_state()

        for i in range(N):
            # update state
            vehicle.set_state(vehicle.rk4_step(vehicle.get_state(), actions[i], moments[i], dt[i]))
            p[i+1], v[i+1], q[i+1], w[i+1] = vehicle.get_state()

        return p, v, q, w, t

###################################
## ---------- testing ---------- ##
###################################
if __name__ == '__main__':
    position = np.array([0.0,0.0,1000.0])
    velocity = np.array([100.0, 200.0, 0.0])
    quaternion = np.array([0.0, 0.0, 0.0, 1.0])
    angular_velocity = np.array([0.0, 0.0, 0.0])
    mass = 1
    inertia = np.eye(3)

    N = 1000
    dt = 0.01*np.ones(N+1)

    actions = np.zeros((N, 3))
    x = np.linspace(-np.pi,np.pi,N)
    actions[:,0] = 50*np.cos(x)
    actions[:,1] = np.exp(x)
    actions[:,2] = 9.81*mass*np.ones(N) # oppose gravity (should remain at constant height)

    UAV = DummyVehicle(position=position, velocity=velocity, quaternion=quaternion, angular_velocity=angular_velocity,
                  mass=mass, inertia=inertia)
    UAV_traj = Trajectory(UAV)

    p, v, q, w, t = UAV_traj.forward(actions=actions, dt=dt)
    
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