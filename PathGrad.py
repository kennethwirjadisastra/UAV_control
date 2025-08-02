import torch as pt
import matplotlib.pyplot as plt
from classes.TargetPath import TargetPath
from classes.Vehicle import Vehicle
from classes.Car import Car
from visualization.FourViewPlot import FourViewPlot
from classes.TargetPath import TargetPath

class PathGrad:
    def __init__(self, vehicle, dt_fine=0.001, device='cpu'):
        self.vehicle = vehicle
        self.dt_fine = dt_fine
        self.device = device

    @pt.no_grad()
    def X_high_res(self, x0, action_plan):
        state = [s.clone() for s in x0]
        X_high_res = [state]

        for action in action_plan:
            force, moment = self.vehicle.compute_forces_and_moments(state, action)
            state = self.vehicle.rk4_step(state, force, moment, self.dt_fine)
            X_high_res.append([s.clone() for s in state])

        return X_high_res  # list of [position, velocity, quaternion, angular_velocity]

    
    def X_low_res(self, x0, action_plan, X_high_res, low_res_steps):
        #action_plan = action_plan.clone().detach().requires_grad_(True)
        N = len(action_plan)
        T = N * self.dt_fine
        dt_coarse = T / low_res_steps
        fine_steps_per_coarse_step = int(round(dt_coarse/self.dt_fine))
        state = [s.clone() for s in x0]
        X_low_res = [state]

        for i in range(low_res_steps):
            action = action_plan[i * fine_steps_per_coarse_step]
            force, moment = self.vehicle.compute_forces_and_moments(state, action)
            state = self.vehicle.rk4_step(state, force, moment, dt_coarse)
            p_low, v_low, q_low, w_low = state
            p_high, v_high, q_high, w_high = X_high_res[(i+1) * fine_steps_per_coarse_step]
            epsilon = [p_high-p_low, v_high-v_low, q_high-q_low, w_high-w_low]

            
            corrected_state = [s.clone()-epsilon[i] for i, s in enumerate(state)]
            X_low_res.append(corrected_state)
        
        return X_low_res
    
    def target_path(self, waypoints, low_res_steps):
        t = pt.linspace(0, 1, low_res_steps+1)
        targetPath = TargetPath(waypoints)
        trajectoryTargets = targetPath.normalized_interpolate(t)

        return trajectoryTargets
    
    def loss(self, X_low_res, target_path):
        p = pt.stack([state[0] for state in X_low_res])
        loss = pt.mean((p - target_path) ** 2)

        return loss
    

if __name__ == '__main__':
    device = 'cuda' if pt.cuda.is_available() else 'cpu'

    # initial state
    position            = pt.tensor([0.0, 0.0, 0.55])
    velocity            = pt.tensor([0, 0, 0])
    quaternion          = pt.tensor([1.0, 0.0, 0.0, 0.0])
    angular_velocity    = pt.tensor([0.0, 0, 0.0])
    
    car = Car(position, velocity, quaternion, angular_velocity)

    action_plan = pt.ones((2000, 2)) * pt.tensor([0.5, 0.5])[None,:]
    action_plan.requires_grad_(True)
    action_plan.to(device)
    dts = 0.04 * pt.ones(2000)

    
    path = PathGrad(car, 0.004, device=action_plan.device)
    x0 = path.vehicle.get_state()
    optimizer = pt.optim.Adam([action_plan], lr=1e-2)

    for i in range(10):
        high_res_path = path.X_high_res(x0, action_plan)
        p_high = pt.stack([s[0] for s in high_res_path], dim=1)
        

        low_res_path = path.X_low_res(x0, action_plan, high_res_path, low_res_steps=200)
        p_low = pt.stack([s[0] for s in low_res_path], dim=1)

        # target path
        dts_coarse = 0.4*pt.ones(200)
        ts = pt.cumsum(dts_coarse, dim=0)
        wx = ts
        wy = 5*(1-pt.cos(ts/5))
        wz = 0.5500 * pt.ones(200)

        waypoints = pt.stack([wx, wy, wz])
        targetPath = TargetPath(waypoints)


        loss = pt.sum((waypoints - p_low[:,1:])**2)
        print(loss)
        loss.backward()
        optimizer.step()
    print(action_plan.grad[::10])

    plt.plot(waypoints[0,:], waypoints[1,:], label='Target Path')
    plt.plot(p_high[0,:], p_high[1,:], label='High Res Path')
    plt.plot(p_low.detach().numpy()[0,:], p_low.detach().numpy()[1,:], label='Low Res Path')
    plt.show()