import torch as pt
import numpy as np

from util.optimize import optimize_along_path
from classes.StateTensor import StateTensor
from classes.Drone import Quadcopter
from classes.ActionPlan import ActionPlan
from classes.TargetPath import TargetPath

from util.blendScene import blendScene
from pathlib import Path

if __name__ == '__main__':
    # initial state
    position            = pt.tensor([0.0, 0.0, 5.0])
    velocity            = pt.tensor([0, 0, 0])
    quaternion          = pt.tensor([1.0, 0.0, 0.0, 0.0])
    angular_velocity    = pt.tensor([0.0, 0, 0.0])

    init_state = StateTensor(
        pos     = [0.0, 0.0, 5.0],
        vel     = [0.0, 0.0, 0.0],
        quat    = [1.0, 0.0, 0.0, 0.0],
        angvel  = [0.0, 0.0, 0.0]
    )

    # action plan and delta time
    tf = 3
    dt = 0.05
    nstep = int(tf / dt)
    action_plan = pt.ones((nstep, 4)) * pt.tensor([0.2, 0.2, 0.2, 0.2])[None,:]
    action_plan.requires_grad_(True)
    dts = dt * pt.ones(nstep)
    

    # target path
    ts = pt.linspace(0, tf, 100)
    wx = 10*ts/2
    wy = 10*(1-pt.cos(ts/2))
    wz = 5.0*pt.ones_like(ts)

    waypoints = StateTensor(pos=pt.stack([wx, wy, wz]).T)
    np.savetxt('blender/trajectories/drone_target.csv', waypoints, delimiter=',')

    drone = Quadcopter(init_state)
    plan = ActionPlan(Quadcopter, resolution=10)

    #optimize_along_path(
    #    vehicle=Quadcopter(init_state), action_plan=action_plan, delta_time=dts, target=TargetPath(waypoints), 
    #    steps=300, lr=5e-2, discount_rate=0.25, acc_reg=1e-3, plot_freq=10
    #)

    optimize_along_path(
        vehicle=drone, action_plan=plan, max_dt=0.05, target=TargetPath(waypoints), 
        steps=300, lr=5e-2, discount_rate=0.25, acc_reg=1e-3, plot_freq=10
    )

    cwd = Path(__file__).resolve().parent
    blendScene(scene_file='DroneScene.blend',
               traj_script='renderDrone.py',
               traj_path=cwd/'../blender/trajectories/Quadcopter',
               dt=dt)