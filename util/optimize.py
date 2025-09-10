import torch as pt
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from classes.Vehicle import Vehicle, StateTensor
from classes.TargetPath import TargetPath
from classes.ActionPlan import ActionPlan
from visualization.FourViewPlot import FourViewPlot

def optimize_along_path(
        vehicle: Vehicle, action_plan: ActionPlan, max_dt: float, target: TargetPath, 
        steps: int = 100, lr: float = 0.03,
        discount_rate: float = 0.25, acc_reg: float = 0.001,
        plot_freq: int = 5, save_folder = 'blender/trajectories/'
    ):
    
    # Plotting and saving
    vehicle_name = type(vehicle).__name__
    np.savetxt(save_folder + vehicle_name + '/target.csv', target.waypoints, delimiter=',')
    fourPlot = FourViewPlot()
    fourPlot.addTrajectory(target.waypoints, 'TargetPath', color='red')

    # Optimizer
    optimizer = pt.optim.Adam([action_plan.action, action_plan.delta_time], lr=lr)
    for step in trange(steps, desc='Optimizing action plan', unit='iter'):
        optimizer.zero_grad()

        # compute the fowards trajectory
        action_tensor, dts  = action_plan.rasterize(max_dt)

        X_p, X_v, q, w, t   = vehicle.simulate_trajectory(vehicle.state, action_tensor, dts)

        # assign targets for each point along the path with no grad
        with pt.no_grad():
            raw_arc_dists   = pt.cumsum(pt.norm((X_p[1:] - X_p[:-1]), dim=-1), dim=-1)  # unscaled
            arc_dists       = (target.total_length / raw_arc_dists[-1]) * raw_arc_dists
            Y_p: pt.tensor  = target.distance_interpolate(arc_dists)

        # compute the loss
        dist_losses = ((X_p[1:] - Y_p[:]) ** 2).sum(dim=1)                          # per point L_2^2 loss
        acc_losses  = ((X_v[1:] - X_v[:-1]) ** 2).sum(dim=1) / dts ** 2             # per point acceleration ^ 2 loss
        time_scale  = discount_rate ** t[...,1:]                                    # negaive exponential scaling with discount rate
        loss = ((dist_losses + acc_reg * acc_losses) * time_scale).sum(dim=0)

        # compute the backwards gradient and update the action plan
        loss.backward()

        time_scale_prime = (discount_rate ** action_plan.time).unsqueeze(-1)

        action_plan.action.grad = pt.nan_to_num(action_plan.action.grad, nan=0.0) / time_scale_prime  # normalize the loss (undo the time scale)


        optimizer.step()
        action_plan.update()

        action_plan.print()


        if step == 0 or (steps - step) % plot_freq == 1:
            fourPlot.addTrajectory(X_p.detach().cpu().numpy(), 'Vehicle', color='blue')
            fourPlot.addScatter(X_p.detach().cpu().numpy(), 'X_p', color='cyan')
            fourPlot.addScatter(Y_p.detach().cpu().numpy(), 'Y_p', color='orange')
            fourPlot.show()

    # save the trajectory
    with pt.no_grad():
        state = StateTensor(state_vec=pt.cat([X_p[:-1], X_v[:-1], q[:-1], w[:-1]], dim=-1))
        action_tensor, _                = action_plan.rasterize(max_dt)
        force_vecs, force_locs, _       = vehicle.compute_forces(state, action_tensor)
        force_locs += X_p[:-1,:,None]

        # Reshape and save to CSV

        np.savetxt(save_folder + vehicle_name + '/traj_force_vecs.csv', force_vecs.mT.cpu().numpy().reshape(force_vecs.shape[0], -1), delimiter=',')
        np.savetxt(save_folder + vehicle_name + '/traj_force_locs.csv', force_locs.mT.cpu().numpy().reshape(force_locs.shape[0], -1), delimiter=',')

        action_plan.save_to_file(save_folder + vehicle_name + '_acton_plan/')

        traj = pt.concatenate([X_p, q], axis=1) # shape (N, 7): [x, y, z, qw, qx, qy, qz]
        np.savetxt(save_folder + vehicle_name + '/traj.csv', traj.detach().cpu().numpy(), delimiter=',')
    plt.show()