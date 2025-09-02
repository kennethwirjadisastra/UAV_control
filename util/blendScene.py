import subprocess
from pathlib import Path

def find_blender_exe():
    # List of common roots to search
    search_roots = [
        Path('C:/Program Files'),
        Path('C:/Program Files (x86)'),
        Path.home() / 'AppData/Local/Programs',
        Path('C:/')
    ]

    for root in search_roots:
        if not root.exists():
            continue
        # Recursively search for blender.exe
        for exe_path in root.rglob('blender.exe'):
            return exe_path  # return the first match
    return None

def blendScene(scene_file: str, traj_script: str, traj_path: str, dt: float=None):
    # blender executable
    # blender_exe = Path(r'C:\Program Files\Blender Foundation\Blender 4.5\blender.exe')
    blender_exe = find_blender_exe()

    if blender_exe is None:
        print('blender.exe not installed')
        return None

    print(blender_exe)

    # project's root directory
    root_dir = Path(__file__).resolve().parents[1]

    # .blend file to run
    blend_file = root_dir / 'blender' / scene_file

    # trajectory rendering python script
    render_file = root_dir / 'blender' / traj_script

    # open blender file and render vehicle trajectory
    print('Opening Blender')
    subprocess.run([blender_exe,
                    str(blend_file),
                    '--python', 
                    str(render_file),
                    '--', str(traj_path), str(dt)])

    return None

if __name__ == '__main__':
    # cwd = 'UAV_control/util'
    # blender directory = 'UAV_control/blender'
    blender_path = Path(__file__).resolve().parent/'../blender'
    dt = 0.05

    blendScene(scene_file=blender_path/'CarScene.blend',
               traj_script=blender_path/'renderCar.py',
               traj_path=blender_path/'trajectories/Car',
               dt=dt)
    
    blendScene(scene_file=blender_path/'DroneScene.blend',
               traj_script=blender_path/'renderDrone.py',
               traj_path=blender_path/'trajectories/Quadcopter',
               dt=dt)