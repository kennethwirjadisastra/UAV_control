import torch as pt
from classes.TargetPath import TargetPath
from classes.Vehicle import Vehicle
from classes.Car import Car

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

    def X_low_res(self, x0, action_plan, X_high_res):
        T = len(action_plan) * self.dt_fine
        N = T // self.dt_coarse

        action_idx = pt.linspace(0, len(X_high_res)-2, steps=N).round().long()
        state = [s.clone() for s in x0]
        X_low_res = [state]

        fine_steps_per_coarse_step = self.dt_coarse//self.dt_fine
        for i, idx in enumerate(action_idx):
            action = action_plan[idx]
            force, moment = self.vehicle.compute_forces_and_moments(state, action)
            state = self.vehicle.rk4_step(state, force, moment, self.dt_coarse)
            p_low, v_low, q_low, w_low = state
            p_high, v_high, q_high, w_high = X_high_res[(i + 1) * fine_steps_per_coarse_step]
            correction = [p_high-p_low, v_high-v_low, q_high-q_low, w_high-w_low]

            X_low_res.append([s.clone()-correction[i] for i, s in enumerate(state)])
        
        return X_low_res
    
    def X_low_res(self, x0, action_plan, X_high_res, low_res_steps):
        action_plan = action_plan.clone().detach().requires_grad_(True)
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
    
    