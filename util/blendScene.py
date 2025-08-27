import subprocess
from pathlib import Path


def blendScene(scene_file: str, traj_script: str):
    # blender executable
    blender_exe = Path(r'C:\Program Files\Blender Foundation\Blender 4.5\blender.exe')

    # project's root directory
    root_dir = Path(__file__).resolve().parents[1]

    # .blend file to run
    blend_file = root_dir / "blender" / scene_file

    # trajectory rendering python script
    render_file = root_dir / "blender" / traj_script

    # open blender file and render vehicle trajectory
    print('Opening Blender')
    subprocess.run([blender_exe, str(blend_file), "--python", str(render_file)])

    return None