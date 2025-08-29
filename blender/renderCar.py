import bpy
import sys
from pathlib import Path
import csv
from mathutils import Quaternion
from mathutils import Vector

import os

print(os.getcwd())

def create_force_arrow(name, color):
    # Create cylinder (shaft)
    bpy.ops.mesh.primitive_cylinder_add(radius=0.05, depth=1, location=(0, 0, 0))
    shaft = bpy.context.active_object
    shaft.name = f"{name}_shaft"
    
    # Move origin to the base of the shaft (so scaling/rotation acts from the bottom)
    bpy.context.scene.cursor.location = (0, 0, -0.5)  # base of unit-height cylinder
    bpy.ops.object.origin_set(type='ORIGIN_CURSOR')

    # Create cone (head)
    bpy.ops.mesh.primitive_cone_add(radius1=0.1, depth=0.2, location=(0, 0, 0.6))  # tip offset above shaft
    cone = bpy.context.active_object
    cone.name = f"{name}_head"

    # Parent cone to shaft
    cone.parent = shaft
    cone.matrix_parent_inverse = shaft.matrix_world.inverted()

    # Join into single object
    bpy.context.view_layer.objects.active = shaft
    cone.select_set(True)
    shaft.select_set(True)
    bpy.ops.object.join()
    arrow = bpy.context.active_object  # shaft (cylinder) + head (cone)

    # Set color
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
    mat.diffuse_color = color  # (R, G, B, Alpha)
    arrow.data.materials.append(mat)

    return arrow

def create_path_curve(name, steps, color):
    curve = bpy.data.curves.new(name=name, type='CURVE')
    curve.dimensions = '3D'
    curve.resolution_u = 2

    polyline = curve.splines.new('POLY')
    polyline.points.add(len(steps)-1)

    for i, step in enumerate(steps):
        polyline.points[i].co = (step[0], step[1], step[2], 1)

    path = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(path)

    mat = bpy.data.materials.new(name+'_mat')
    mat.diffuse_color = color  # RGBA
    mat.use_nodes = False
    path.data.materials.append(mat)

    path.data.bevel_depth = 0.1
    path.data.bevel_resolution = 3

    return path

def delete_objects(prefixes=None, suffixes=None):
    for obj in bpy.data.objects:
        if prefixes is not None and any(obj.name.startswith(prefix) for prefix in prefixes):
            bpy.data.objects.remove(obj, do_unlink=True)
        if suffixes is not None and any(obj.name.endswith(suffix) for suffix in suffixes):
            bpy.data.objects.remove(obj, do_unlink=True)

if __name__ == '__main__':
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

    car = bpy.data.objects['Car']
    car.rotation_mode = 'QUATERNION'
    if car.animation_data:
            car.animation_data_clear()

    # Delete force arrows and paths
    delete_objects(prefixes = ['lat', 'susp', 'throttle', 'gravity', 'target', 'true'])

    # suspension forces (bright purple arrows)
    susp_rear_left = create_force_arrow('susp_rear_left', color=(0.8, 0.2, 1.0, 1.0))
    susp_rear_right = create_force_arrow('susp_rear_right', color=(0.8, 0.2, 1.0, 1.0))
    susp_front_left = create_force_arrow('susp_front_left', color=(0.8, 0.2, 1.0, 1.0))
    susp_front_right = create_force_arrow('susp_front_right', color=(0.8, 0.2, 1.0, 1.0))

    # lateral forces (green arrows)
    lat_rear_left = create_force_arrow('lat_rear_left', color=(0, 1, 0, 1))
    lat_rear_right = create_force_arrow('lat_rear_right', color=(0, 1, 0, 1))
    lat_front_left = create_force_arrow('lat_front_left', color=(0, 1, 0, 1))
    lat_front_right = create_force_arrow('lat_front_right', color=(0, 1, 0, 1))

    # throttle forces (cyan arrows)
    throttle_rear_left = create_force_arrow('throttle_rear_left', color=(0.2, 1.0, 1.0, 1.0))
    throttle_rear_right = create_force_arrow('throttle_rear_right', color=(0.2, 1.0, 1.0, 1.0))
    throttle_front_left = create_force_arrow('throttle_front_left', color=(0.2, 1.0, 1.0, 1.0))
    throttle_front_right = create_force_arrow('throttle_front_right', color=(0.2, 1.0, 1.0, 1.0))

    # gravity force (black arrow)
    gravity = create_force_arrow('gravity', color=(0, 0, 0, 1))

    force_arrows = [
        susp_rear_left,
        susp_rear_right,
        susp_front_left,
        susp_front_right,
        lat_rear_left,
        lat_rear_right,
        lat_front_left,
        lat_front_right,
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
        
        path_reader = csv.reader(path_file)
        force_loc_reader = csv.reader(force_loc_file)
        force_dir_reader = csv.reader(force_dir_file)

        frame = 1  # start at frame 1
        
        for path_row, force_loc_row, force_dir_row in zip(path_reader, force_loc_reader, force_dir_reader):
            # Parse car's position/quaternion and force lcoation/directions from each row
            pos_x = float(path_row[0])
            pos_y = float(path_row[1])
            pos_z = float(path_row[2])
            rot_w = float(path_row[3])
            rot_x = float(path_row[4])
            rot_y = float(path_row[5])
            rot_z = float(path_row[6])

        

            car.location = (pos_x, pos_y, pos_z)
            quat = Quaternion((rot_w, rot_x, rot_y, rot_z))
            car.rotation_quaternion = quat

            # Insert keyframes at the current frame
            car.keyframe_insert(data_path="location", frame=frame)
            car.keyframe_insert(data_path="rotation_quaternion", frame=frame)

            for i, arrow in enumerate(force_arrows):
                force_pos = Vector((
                    float(force_loc_row[3*i]),
                    float(force_loc_row[3*i+1]),
                    float(force_loc_row[3*i+2])
                ))

                force_vec = Vector((
                    float(force_dir_row[3*i]),
                    float(force_dir_row[3*i+1]),
                    float(force_dir_row[3*i+2]),
                ))

                force_dir = force_vec.normalized()
                force_mag = force_vec.length / 1000

                default_dir = Vector((0, 0, 1)) # default parallel to z axis
                force_quat = default_dir.rotation_difference(force_vec.normalized())


                arrow.location = force_pos
                arrow.rotation_mode = 'QUATERNION'
                arrow.rotation_quaternion = force_quat
                arrow.scale = (1, 1, force_mag)

                arrow.keyframe_insert(data_path='location', frame=frame)
                arrow.keyframe_insert(data_path='rotation_quaternion', frame=frame)
                arrow.keyframe_insert(data_path='scale', frame=frame)

            true_pos = (float(path_row[0]), float(path_row[1]), float(path_row[2]))
            true_path.append(true_pos)
            
            frame += 1

        with open(csv_target, newline='') as target_file:
            target_reader = csv.reader(target_file)
            for target_row in target_reader:
                target_pos = (float(target_row[0]), float(target_row[1]), float(target_row[2]))
                target_path.append(target_pos)

        # set end frame
        bpy.context.scene.frame_end = frame - 1

        # plot true and target paths
        true_path_curve = create_path_curve('true_path', true_path, color=(0, 0, 1, 1)) # blue curve
        target_path_curve = create_path_curve('target_path', target_path, color=(1, 0, 0, 1)) # red curve
