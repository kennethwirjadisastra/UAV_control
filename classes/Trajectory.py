import torch as pt
import matplotlib.pyplot as plt
from classes.Vehicle import Vehicle

class DummyVehicle(Vehicle):
    def compute_forces_and_moments(self, *args, **kwargs):
        raise NotImplementedError("This dummy vehicle requires you to manually provide forces and moments.")

class Trajectory:
    def __init__(self, vehicle: Vehicle):
        self.vehicle = vehicle

    def forward(self, actions: pt.Tensor, moments: pt.Tensor, dt: pt.Tensor,
                update_state=True, return_all=False):
        '''
        Propagate vehicle state forward in time using RK4 given
        actions, moments, and time steps `dt`.
        '''
        vehicle = self.vehicle
        device = actions.device
        N = len(actions)

        assert actions.shape == moments.shape == (N, 3), "Expected shape (N, 3)"
        assert dt.shape == (N,), "dt must have shape (N,)"

        # Preallocate state arrays
        p = pt.zeros((N+1, 3), device=device)
        v = pt.zeros((N+1, 3), device=device)
        q = pt.zeros((N+1, 4), device=device)
        w = pt.zeros((N+1, 3), device=device)
        t = pt.cat([pt.zeros(1, device=device), pt.cumsum(dt, dim=0)])

        state = vehicle.get_state()
        p[0], v[0], q[0], w[0] = state

        for i in range(N):
            state = vehicle.rk4_step(state, actions[i], moments[i], dt[i])
            p[i+1], v[i+1], q[i+1], w[i+1] = state
            if update_state:
                vehicle.set_state(state)

        if return_all:
            return pt.cat([p, v, q, w, t[:, None]], dim=1)  # shape: (N+1, 14)

        return [p, v, q, w, t]

###################################
## ---------- testing ---------- ##
###################################
if __name__ == '__main__':
    position = pt.tensor([0.0,0.0,1000.0])
    velocity = pt.tensor([100.0, 200.0, 0.0])
    quaternion = pt.tensor([0.0, 0.0, 0.0, 1.0])
    angular_velocity = pt.tensor([0.0, 0.0, 0.0])
    mass = 1
    inertia = pt.eye(3)

    N = 1000
    dt = 0.01*pt.ones(N)

    actions = pt.zeros((N, 3))
    x = pt.linspace(-pt.pi,pt.pi,N)
    actions[:,0] = 50*pt.cos(x)
    actions[:,1] = pt.exp(x)
    actions[:,2] = 9.81*mass*pt.ones(N) # oppose gravity (should remain at constant height)
    moments = pt.random.randn(N,3)

    UAV = DummyVehicle(position=position, velocity=velocity, quaternion=quaternion, 
                       angular_velocity=angular_velocity, mass=mass, inertia=inertia)
    UAV_traj = Trajectory(UAV)

    print(f'State Before forward(): {pt.hstack(UAV.get_state())}\n')
    traj = UAV_traj.forward(actions=actions, dt=dt, moments=moments, update_state=False, return_all=True) # return full state as (N,14) tensor
    # position, veloctiy, quaternions, angular velocity, time
    p = traj[:,0:3]
    v = traj[:,3:6]
    q = traj[:,6:10]
    w = traj[:,10:13]
    t = traj[:,13]
    print(f'State After forward(): {pt.hstack(UAV.get_state())} (update_state=False)\n') # should be same as state before forward
    print(f'Trajectory End: {pt.hstack([p[-1], v[-1], q[-1], w[-1]])}\n') # should be different from state before forward
    
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