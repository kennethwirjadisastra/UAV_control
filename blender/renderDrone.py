import bpy
import csv
import sys
from pathlib import Path
from mathutils import Quaternion
from mathutils import Vector

if __name__ == '__main__':
    # Add the project folder to Python path
    # UAV_control/blender
    base_dir = Path(__file__).parent
    if str(base_dir) not in sys.path:
        sys.path.append(str(base_dir))
    
    from render_functions import create_force_arrow, create_path_curve, create_legend,delete_objects, delete_collection, insert_vehicle_frame, insert_force_frame

    # set fps and trajectory csv files
    argv = sys.argv
    if '--' in argv:
        args = argv[argv.index('--') + 1:]
        base_path = Path(args[0])
        dt = float(args[1])

    csv_path        = base_path / 'traj.csv'
    csv_force_loc   = base_path / 'traj_force_locs.csv'
    csv_force_dir   = base_path / 'traj_force_vecs.csv'
    csv_target      = base_path / 'target.csv'

    fps = round(1.0/dt)
    bpy.context.scene.render.fps = fps
    bpy.context.scene.render.fps_base = 1.0

    drone = bpy.data.objects['Drone']
    drone.rotation_mode = 'QUATERNION'
    if drone.animation_data:
            drone.animation_data_clear()

    # Delete force arrows and paths
    delete_objects(prefixes = ['drag', 'throttle', 'gravity', 'target', 'true'])
    delete_collection('Legend')

    legend_entries = {
        'Drag':          (0, 1, 0, 1),          # drag forces (green arrows)
        'Throttle':      (0.2, 1.0, 1.0, 1.0),  # throttle forces (cyan arrows)
        'Gravity':       (0, 0, 0, 1),          # gravity force (black arrow)
        'True Path':     (0, 0, 1, 1),          # true path (blue curve)
        'Target Path':   (1, 0, 0, 1)           # target_path (red curve)
    }
    create_legend(legend_entries, start_location=(2, 4, 4), marker_size=0.125, text_size=0.5)
    drag = create_force_arrow('drag', color=legend_entries['Drag'], radius=0.0125)

    throttle_rear_left = create_force_arrow('throttle_rear_left', color=legend_entries['Throttle'], radius=0.0125)
    throttle_rear_right = create_force_arrow('throttle_rear_right', color=legend_entries['Throttle'], radius=0.0125)
    throttle_front_left = create_force_arrow('throttle_front_left', color=legend_entries['Throttle'], radius=0.0125)
    throttle_front_right = create_force_arrow('throttle_front_right', color=legend_entries['Throttle'], radius=0.0125)

    gravity = create_force_arrow('gravity', color=legend_entries['Gravity'], radius=0.0125)

    force_arrows = [
        drag,
        throttle_rear_left,
        throttle_rear_right,
        throttle_front_left,
        throttle_front_right,
        gravity
    ]

    true_path = []
    target_path = []

    with open(csv_path, newline='') as path_file, \
        open(csv_force_loc, newline='') as force_loc_file, \
        open(csv_force_dir, newline='') as force_dir_file:
        
        path_reader = list(csv.reader(path_file))
        force_loc_reader = csv.reader(force_loc_file)
        force_dir_reader = csv.reader(force_dir_file)

        frame = 1  # start at frame 1
        
        for path_row, force_loc_row, force_dir_row in zip(path_reader, force_loc_reader, force_dir_reader):
            insert_vehicle_frame(path_row, drone, frame)
            insert_force_frame(force_arrows, force_loc_row, force_dir_row, frame, scale=6)

            true_pos = (float(path_row[0]), float(path_row[1]), float(path_row[2]))
            true_path.append(true_pos)
            
            frame += 1
        
        path_end = path_reader[-1]
        insert_vehicle_frame(path_end, drone, frame)
        true_pos = (float(path_end[0]), float(path_end[1]), float(path_end[2]))
        true_path.append(true_pos)

        with open(csv_target, newline='') as target_file:
            target_reader = csv.reader(target_file)
            for target_row in target_reader:
                target_pos = (float(target_row[0]), float(target_row[1]), float(target_row[2]))
                target_path.append(target_pos)

        # set end frame
        bpy.context.scene.frame_end = frame
        
        # plot true and target paths
        true_path_curve = create_path_curve('true_path', true_path, color=legend_entries['True Path'], radius=0.0125) 
        target_path_curve = create_path_curve('target_path', target_path, color=legend_entries['Target Path'], radius=0.0125)

        # Deselect all objects
        bpy.ops.object.select_all(action='DESELECT')