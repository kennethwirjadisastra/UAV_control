import torch as pt
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from classes.Vehicle import Vehicle, StateTensor
from classes.TargetPath import TargetPath
from classes.ActionPlan import ActionPlan
from visualization.MultiViewPlot import MultiViewPlot

def optimize_along_path(
        vehicle: Vehicle, action_plan: ActionPlan, max_dt: float, target: TargetPath, 
        steps: int = 100, lr: float = 0.03,
        discount_rate: float = 0.25, acc_reg: float = 0.001,
        plot_freq: int = 5, save_folder = 'blender/trajectories/'
    ):
    
    # Plotting and saving
    vehicle_name = type(vehicle).__name__
    np.savetxt(save_folder + vehicle_name + '/target.csv', target.waypoints, delimiter=',')
    multiPlot = MultiViewPlot()
    multiPlot.addTrajectory(target.waypoints.pos, 'TargetPath', color='red')

    # Optimizer
    optimizer = pt.optim.Adam([action_plan.action, action_plan.delta_time], lr=lr)
    action_tensor = None
    for step in trange(steps, desc='Optimizing action plan', unit='iter'):
        optimizer.zero_grad()

        # compute the fowards trajectory
        action_tensor, dts  = action_plan.rasterize(max_dt)

        X, t   = vehicle.simulate_trajectory(vehicle.state, action_tensor, dts)

        # assign targets for each point along the path with no grad
        with pt.no_grad():
            raw_arc_dists   = pt.cumsum(pt.norm((X.pos[1:] - X.pos[:-1]), dim=-1), dim=-1)  # unscaled
            arc_dists       = (target.total_length / raw_arc_dists[-1]) * raw_arc_dists
            Y_p: pt.tensor  = target.distance_interpolate(arc_dists).pos

        # compute the loss
        dist_losses = ((X.pos[1:] - Y_p[:]) ** 2).sum(dim=1)                        # per point L_2^2 loss
        acc_losses  = ((X.vel[1:] - X.vel[:-1]) ** 2).sum(dim=1) / dts ** 2         # per point acceleration ^ 2 loss
        time_scale  = discount_rate ** t[...,1:]                                    # negaive exponential scaling with discount rate
        loss = ((dist_losses + acc_reg * acc_losses) * time_scale).sum(dim=0)

        # compute the backwards gradient and update the action plan
        loss.backward()

        time_scale_prime = (discount_rate ** action_plan.time).unsqueeze(-1)

        action_plan.action.grad = pt.nan_to_num(action_plan.action.grad, nan=0.0) / time_scale_prime  # normalize the loss (undo the time scale)


        optimizer.step()
        action_plan.update()

        # action_plan.print()


        multiPlot.addLoss(dist_losses.mean().detach().cpu().numpy(), step)
        if step == 0 or (steps - step) % plot_freq == 1:
            multiPlot.addTrajectory(X.pos.detach().cpu().numpy(), 'Vehicle', color='blue')
            multiPlot.addScatter(X.pos.detach().cpu().numpy(), 'X_p', color='cyan')
            multiPlot.addScatter(Y_p.detach().cpu().numpy(), 'Y_p', color='orange')
            multiPlot.addAction(action_plan, action_plan.min_dt)
            multiPlot.show()

    # save the trajectory
    with pt.no_grad():
        force_vecs, force_locs, _       = vehicle.compute_forces(X[:-1], action_tensor)
        force_locs += X.pos[:-1,:,None]

        # Reshape and save to CSV
        np.savetxt(save_folder + vehicle_name + '/traj_force_vecs.csv', force_vecs.mT.cpu().numpy().reshape(force_vecs.shape[0], -1), delimiter=',')
        np.savetxt(save_folder + vehicle_name + '/traj_force_locs.csv', force_locs.mT.cpu().numpy().reshape(force_locs.shape[0], -1), delimiter=',')

        action_plan.save_to_file(save_folder + vehicle_name)

        traj = pt.concatenate([X.pos, X.quat], axis=1) # shape (N, 7): [x, y, z, qw, qx, qy, qz]
        np.savetxt(save_folder + vehicle_name + '/traj.csv', traj.detach().cpu().numpy(), delimiter=',')
    plt.show()


def optimize_start_end(
        vehicle: Vehicle, action_plan: ActionPlan, max_dt: float, target: StateTensor, 
        steps: int = 100, lr: float = 0.03,
        discount_rate: float = 0.25, acc_reg: float = 0.001,
        plot_freq: int = 5, save_folder = 'blender/trajectories/'
    ):
    
    # Plotting and saving
    vehicle_name = type(vehicle).__name__
    multiPlot = MultiViewPlot()
    multiPlot.addTrajectory(target.pos[None], 'TargetPath', color='red')

    # Optimizer
    optimizer = pt.optim.Adam([action_plan.action, action_plan.delta_time], lr=lr)
    action_tensor = None
    for step in trange(steps, desc='Optimizing action plan', unit='iter'):
        optimizer.zero_grad()

        # compute the fowards trajectory
        action_tensor, dts  = action_plan.rasterize(max_dt)

        X, t   = vehicle.simulate_trajectory(vehicle.state, action_tensor, dts)

        # assign targets for each point along the path with no grad
        '''with pt.no_grad():
            raw_arc_dists   = pt.cumsum(pt.norm((X.pos[1:] - X.pos[:-1]), dim=-1), dim=-1)  # unscaled
            arc_dists       = (target.total_length / raw_arc_dists[-1]) * raw_arc_dists
            Y_p: pt.tensor  = target.distance_interpolate(arc_dists).pos'''

        # compute the loss
        dist_losses = ((X.pos[-1] - target.pos) ** 2).sum(dim=-1)                        # per point L_2^2 loss
        acc_losses  = ((X.vel[1:] - X.vel[:-1]) ** 2).sum(dim=1) / dts ** 2         # per point acceleration ^ 2 loss
        loss = (dist_losses + acc_reg * acc_losses).sum(dim=0)

        # compute the backwards gradient and update the action plan
        loss.backward()

        action_plan.action.grad = pt.nan_to_num(action_plan.action.grad, nan=0.0)  # normalize the loss (undo the time scale)


        optimizer.step()
        action_plan.update()

        # action_plan.print()


        multiPlot.addLoss(dist_losses.mean().detach().cpu().numpy(), step)
        if step == 0 or (steps - step) % plot_freq == 1:
            multiPlot.addTrajectory(X.pos.detach().cpu().numpy(), 'Vehicle', color='blue')
            multiPlot.addScatter(X.pos.detach().cpu().numpy(), 'X_p', color='cyan')
            multiPlot.addScatter(target.pos[None].detach().cpu().numpy(), 'Target', color='orange')
            multiPlot.addAction(action_plan, action_plan.min_dt)
            multiPlot.show()

    # save the trajectory
    with pt.no_grad():
        force_vecs, force_locs, _       = vehicle.compute_forces(X[:-1], action_tensor)
        force_locs += X.pos[:-1,:,None]

        # Reshape and save to CSV
        np.savetxt(save_folder + vehicle_name + '/traj_force_vecs.csv', force_vecs.mT.cpu().numpy().reshape(force_vecs.shape[0], -1), delimiter=',')
        np.savetxt(save_folder + vehicle_name + '/traj_force_locs.csv', force_locs.mT.cpu().numpy().reshape(force_locs.shape[0], -1), delimiter=',')

        action_plan.save_to_file(save_folder + vehicle_name)

        traj = pt.concatenate([X.pos, X.quat], axis=1) # shape (N, 7): [x, y, z, qw, qx, qy, qz]
        np.savetxt(save_folder + vehicle_name + '/traj.csv', traj.detach().cpu().numpy(), delimiter=',')
    plt.show()