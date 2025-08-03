import bpy
import csv
from mathutils import Quaternion
from mathutils import Vector

import os

print(os.getcwd())

def create_force_arrow(name, color):
    import bpy

    # Create a cylinder to represent the force arrow
    bpy.ops.mesh.primitive_cylinder_add(radius=0.02, depth=1, location=(0, 0, 0))
    arrow = bpy.context.active_object
    arrow.name = name

    # Move origin to the base of the arrow (so scaling/rotation acts from the bottom)
    bpy.context.scene.cursor.location = (0, 0, -0.5)  # base of unit-height cylinder
    bpy.ops.object.origin_set(type='ORIGIN_CURSOR')

    # Assign material and color
    mat = bpy.data.materials.new(name + "_mat")
    mat.diffuse_color = color  # expects (R, G, B, A)
    arrow.data.materials.append(mat)

    return arrow


# Replace with your actual CSV file path
csv_path = 'C:/Users/kenne/Documents/projects/UAV_control/git/blender/trajectories/traj.csv'
csv_force_loc = 'C:/Users/kenne/Documents/projects/UAV_control/git/blender/trajectories/traj_force_locs.csv'
csv_force_dir = 'C:/Users/kenne/Documents/projects/UAV_control/git/blender/trajectories/traj_force_vecs.csv'

car = bpy.data.objects['Car']
car.rotation_mode = 'QUATERNION'

# suspension forces (red arrows)
susp_rear_left = create_force_arrow('susp_rear_left', color=(1, 0, 0, 1))
susp_rear_right = create_force_arrow('susp_rear_right', color=(1, 0, 0, 1))
susp_front_left = create_force_arrow('susp_front_left', color=(1, 0, 0, 1))
susp_front_right = create_force_arrow('susp_front_right', color=(1, 0, 0, 1))

# lateral forces (green arrows)
lat_rear_left = create_force_arrow('lat_rear_left', color=(0, 1,  0, 1))
lat_rear_right = create_force_arrow('lat_rear_right', color=(0, 1,  0, 1))
lat_front_left = create_force_arrow('lat_front_left', color=(0, 1,  0, 1))
lat_front_right = create_force_arrow('lat_front_right', color=(0, 1,  0, 1))

# throttle forces (blue arrows)
throttle_rear_left = create_force_arrow('throttle_rear_left', color=(0, 0, 1, 1))
throttle_rear_right = create_force_arrow('throttle_rear_right', color=(0, 0, 1, 1))
throttle_front_left = create_force_arrow('throttle_front_left', color=(0, 0, 1, 1))
throttle_front_right = create_force_arrow('throttle_front_right', color=(0, 0, 1, 1))

# gravity force (black arrow)
gravity = create_force_arrow('gravity', color=(0, 0, 0, 1))

force_arrows = [
    susp_rear_left,
    susp_rear_right,
    susp_front_left,
    susp_front_right,
    lat_rear_left,
    lat_front_right,
    lat_front_left,
    lat_front_right,
    throttle_rear_left,
    throttle_rear_right,
    throttle_front_left,
    throttle_front_right,
    gravity
]


with open(csv_path, newline='') as path_file, \
     open(csv_force_loc, newline='') as force_loc_file, \
     open(csv_force_dir, newline='') as force_dir_file:
    
    path_reader = csv.reader(path_file)
    force_loc_reader = csv.reader(force_loc_file)
    force_dir_reader = csv.reader(force_dir_file)

    frame = 1  # start at frame 1
    
    for car_row, force_loc_row, force_dir_row in zip(path_reader, force_loc_reader, force_dir_reader):
        # Parse car's position/quaternion and force lcoation/directions from each row
        pos_x = float(car_row[0])
        pos_y = float(car_row[1])
        pos_z = float(car_row[2])
        rot_w = float(car_row[3])
        rot_x = float(car_row[4])
        rot_y = float(car_row[5])
        rot_z = float(car_row[6])

     

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

            if arrow.name[:3] == 'lat':
                print(arrow.location, arrow.rotation_quaternion, arrow.scale)


        
        frame += 1
