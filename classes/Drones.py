import torch as pt
from classes.Vehicle import Vehicle
from util.quaternion import quaternion_to_matrix

class Quadcopter(Vehicle):
    def __init__(self, position: pt.Tensor = None, velocity: pt.Tensor = None, 
            quaternion: pt.Tensor = None, angular_velocity: pt.Tensor = None,
            mass: pt.Tensor = None, inertia: pt.Tensor = None):
        kwargs = {}
        kwargs['position']          = position
        kwargs['velocity']          = velocity
        kwargs['quaternion']        = quaternion
        kwargs['angular_velocity']  = angular_velocity
        kwargs['mass']              = mass if mass is not None else 0.5
        kwargs['inertia']           = inertia if inertia is not None else pt.diag(pt.tensor([0.003, 0.003, 0.005], dtype=pt.float32))
        
        super().__init__(**kwargs)

        self.peak_motor_thrust      = 5.0   # thrust per motor in newtons
        self.drag_coef              = 1.1   # dry coef of friction

                                                                    # motor order [back_left, back_right, front_left, front_right]
        self.B_thrust_locs          = pt.tensor([                   # displacements from center of mass in meters
                                            [-0.08, -0.08, -0.01], 
                                            [0.08, -0.08, -0.01], 
                                            [-0.08, 0.08, -0.01], 
                                            [0.08, 0.08, -0.01]
                                        ], dtype=pt.float32)
        
        self.B_thrust_dir           = pt.tensor([
                                        [0.0, 0.0, 1.0]
                                    ])
        

    # expects net world force and net body moment
    def compute_forces_and_moments(self, state, action) -> tuple[pt.Tensor, pt.Tensor]:
        pos, vel, quat, ang_vel = state
        rot_mat                 = quaternion_to_matrix(quat)
        throttle                = pt.clip(action, -1.0, 1.0)
        ang_vel_mat             = pt.tensor([
                                    [0, -ang_vel[2], ang_vel[1]],
                                    [ang_vel[2], 0, -ang_vel[0]],
                                    [-ang_vel[1], ang_vel[0], 0]
                                ], dtype=pt.float32, device=ang_vel.device)

        W_drag                  = -self.drag_coef * vel
        B_thrust                = self.peak_motor_thrust * throttle[:,None] * self.B_thrust_dir
        W_thrust                = (rot_mat @ B_thrust.T).T
        W_gravity               = pt.tensor([0, 0, -9.81 * self.mass])
        B_moments               = pt.linalg.cross(self.B_thrust_locs, B_thrust)

        print(W_thrust.shape, W_gravity[None, :].shape, W_drag[None, :].shape)

        return pt.sum(W_thrust, dim=0) + W_drag + W_gravity, pt.sum(B_moments, dim=0)
    

###################################
## ---------- testing ---------- ##
###################################


from visualization.FourViewPlot import FourViewPlot
from classes.TargetPath import TargetPath

if __name__ == '__main__':
    # initial state
    position            = pt.tensor([0.0, 0.0, 5.0])
    velocity            = pt.tensor([0, 0, 0])
    quaternion          = pt.tensor([1.0, 0.0, 0.0, 0.0])
    angular_velocity    = pt.tensor([0.0, 0, 0.0])

    car = Quadcopter(position, velocity, quaternion, angular_velocity)

    # action plan and delta time
    tf = 1
    dt = 0.05
    nstep = int(tf / dt)
    action_plan = pt.ones((nstep, 4)) * pt.tensor([0, 0, 0, 0])[None,:]
    action_plan.requires_grad_(False)
    dts = dt * pt.ones(nstep)

    # for i in range(1):
    p, v, q, w, t = car.simulate_trajectory(car.get_state(), action_plan, dts)

        # loss = pt.norm(p[-1,0])
        # loss.backward()
        # print(action_plan.grad)
    
    # target path
    ts = pt.cumsum(dts, dim=0)
    wx = ts
    wy = 5*(1-pt.cos(ts/5))
    wz = ts*0

    waypoints = pt.stack([wx, wy, wz])
    targetPath = TargetPath(waypoints)

    fourPlot = FourViewPlot()
    fourPlot.addTrajectory(p.detach().cpu().numpy(), 'Vehicle', color='b')
    fourPlot.addTrajectory(waypoints.T, 'TargetPath', color='g')
    fourPlot.show()


    traj = pt.concatenate([p,q], axis=1) # shape (N, 7): [x, y, z, qw, qx, qy, qz]
    header = 'x,y,z,qw,qz,qy,qz'
    np.savetxt('trajectory.csv', traj.detach().cpu().numpy(), delimiter=',')