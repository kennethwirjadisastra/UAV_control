import torch as pt
import numpy as np
import matplotlib.pyplot as plt
from Vehicle import Vehicle

class DummyVehicle(Vehicle):
    def compute_forces_and_moments(self, *args, **kwargs):
        raise NotImplementedError("This dummy vehicle requires you to manually provide forces and moments.")
    

class Trajectory:
    def __init__(self, vehicle):
        self.vehicle = vehicle


    def forward(self, actions: np.ndarray, dt: np.ndarray, moments: np.ndarray=None,
                update_state=True, return_all=False):
        '''
        Takes vehicle and propagates its state forward in time given
        actions, moments, and dt using rk4 integration scheme
        '''
        vehicle = self.vehicle
        N = len(actions)
        moments = np.zeros((N, 3)) if moments is None else moments

        assert len(actions) == len(moments) == len(dt), "actions, moments, and dt must have same length"

        # 3d position, 3d velocity, 4d quaternion, 3d angular velocity, time
        p = np.zeros((N+1, 3))
        v = np.zeros((N+1, 3))
        q = np.zeros((N+1, 4))
        w = np.zeros((N+1, 3))
        t = np.hstack([[0], np.cumsum(dt, axis=0)])

        # initial state
        state = vehicle.get_state()
        p[0], v[0], q[0], w[0] = state

        for i in range(N):
            # propagate state
            state = vehicle.rk4_step(state, actions[i], moments[i], dt[i])
            p[i+1], v[i+1], q[i+1], w[i+1] = state
            
            # update state
            if update_state:
                vehicle.set_state(state)

        # returns (N+1, 14) if return_all is true, otherwise p, v, q, w, t separately
        if return_all:
            return np.hstack([p,v,q,w,t[:,None]])
        return p,v,q,w,t

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
    dt = 0.01*np.ones(N)

    actions = np.zeros((N, 3))
    x = np.linspace(-np.pi,np.pi,N)
    actions[:,0] = 50*np.cos(x)
    actions[:,1] = np.exp(x)
    actions[:,2] = 9.81*mass*np.ones(N) # oppose gravity (should remain at constant height)

    UAV = DummyVehicle(position=position, velocity=velocity, quaternion=quaternion, 
                       angular_velocity=angular_velocity, mass=mass, inertia=inertia)
    UAV_traj = Trajectory(UAV)

    print(f'State Before forward(): {np.hstack(UAV.get_state())}\n')
    traj = UAV_traj.forward(actions=actions, dt=dt, update_state=False, return_all=True) # return full state as (N,14) tensor
    # position, veloctiy, quaternions, angular velocity, time
    p = traj[:,0:3]
    v = traj[:,3:6]
    q = traj[:,6:10]
    w = traj[:,10:13]
    t = traj[:,13]
    print(f'State After forward(): {np.hstack(UAV.get_state())} (update_state=False)\n') # should be same as state before forward
    print(f'Trajectory End: {np.hstack([p[-1], v[-1], q[-1], w[-1]])}\n') # should be different from state before forward
    
    # plotting
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