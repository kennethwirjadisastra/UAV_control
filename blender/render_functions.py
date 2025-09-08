import bpy
import sys
from pathlib import Path
import csv
from mathutils import Quaternion
from mathutils import Vector
from math import radians


def create_force_arrow(name, color, radius=0.05):
    # Create cylinder (shaft)
    bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=1, location=(0, 0, 0))
    shaft = bpy.context.active_object
    shaft.name = f"{name}_shaft"
    
    # Move origin to the base of the shaft (so scaling/rotation acts from the bottom)
    bpy.context.scene.cursor.location = (0, 0, -0.5)  # base of unit-height cylinder
    bpy.ops.object.origin_set(type='ORIGIN_CURSOR')

    # Create cone (head)
    bpy.ops.mesh.primitive_cone_add(radius1=radius*2, depth=0.2, location=(0, 0, 0.6))  # tip offset above shaft
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
    shaft.data.materials.append(mat)

    return arrow

def create_path_curve(name, steps, color, radius=0.1):
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

    path.data.bevel_depth = radius
    path.data.bevel_resolution = 3

    return path


def create_legend(entries_dict, start_location=(0, 0, 0), x_step=0.0, y_step=0.5, z_step=0.5,
                       marker_size=0.25, collection_name='Legend', text_rotation=(45,0,0), text_size=1.0):
    # Create collection
    legend_collection = bpy.data.collections.get(collection_name)
    if not legend_collection:
        legend_collection = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(legend_collection)

    x_start, y_start, z_start = start_location

    for i, (name, color) in enumerate(entries_dict.items()):

        # Compute position for this entry
        x = x_start + i * x_step
        y = y_start + i * y_step
        z = z_start + i * z_step

        # Marker
        bpy.ops.mesh.primitive_cube_add(location=(x - 0.2, y, z), scale=(marker_size, marker_size, marker_size))
        marker = bpy.context.object
        mat = bpy.data.materials.new(name + '_Mat')
        mat.diffuse_color = color
        marker.data.materials.append(mat)

        # Text label
        bpy.ops.object.text_add(location=(x, y, z))
        txt_obj = bpy.context.object    
        txt_obj.data.body = name
        txt_obj.data.size = text_size
        mat_text = bpy.data.materials.new(name + '_TextMat')
        mat_text.diffuse_color = color
        txt_obj.data.materials.append(mat_text)

        # Apply text rotation
        rot_x, rot_y, rot_z = text_rotation
        txt_obj.rotation_euler = (radians(rot_x), radians(rot_y), radians(rot_z))

        # Move objects into collection
        legend_collection.objects.link(txt_obj)
        bpy.context.scene.collection.objects.unlink(txt_obj)
        legend_collection.objects.link(marker)
        bpy.context.scene.collection.objects.unlink(marker)

    return legend_collection



def delete_objects(prefixes=None, suffixes=None):
    '''
    Deletes objects in scene based on prefix or suffix
    '''
    for obj in bpy.data.objects:
        if prefixes is not None and any(obj.name.startswith(prefix) for prefix in prefixes):
            bpy.data.objects.remove(obj, do_unlink=True)
        if suffixes is not None and any(obj.name.endswith(suffix) for suffix in suffixes):
            bpy.data.objects.remove(obj, do_unlink=True)
    return None

def delete_collection(collection):
    '''
    Removes collection and its objects from scene
    '''
    coll = bpy.data.collections.get(collection)
    if coll is None:
        print(f"Collection '{collection}' not found")
        return None
    
    # Delete all objects in the collection
    for obj in list(coll.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

    # Unlink from parent collections
    for parent in bpy.data.collections:
        if coll.name in parent.children.keys():
            parent.children.unlink(coll)

    # If the collection is linked to the scene directly, unlink it
    for scene in bpy.data.scenes:
        if coll in scene.collection.children.values():
            scene.collection.children.unlink(coll)

    # Finally, remove the collection itself
    bpy.data.collections.remove(coll)
    
    return None


def insert_vehicle_frame(path_row, vehicle, frame):
    pos_x = float(path_row[0])
    pos_y = float(path_row[1])
    pos_z = float(path_row[2])
    rot_w = float(path_row[3])
    rot_x = float(path_row[4])
    rot_y = float(path_row[5])
    rot_z = float(path_row[6])

    vehicle.location = (pos_x, pos_y, pos_z)
    quat = Quaternion((rot_w, rot_x, rot_y, rot_z))
    vehicle.rotation_quaternion = quat

    # Insert keyframes at the current frame
    vehicle.keyframe_insert(data_path='location', frame=frame)
    vehicle.keyframe_insert(data_path='rotation_quaternion', frame=frame)

    return None


def insert_force_frame(force_arrows, force_loc_row, force_dir_row, frame, scale=1000):
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
        force_mag = force_vec.length / scale

        default_dir = Vector((0, 0, 1)) # default parallel to z axis
        force_quat = default_dir.rotation_difference(force_vec.normalized())


        arrow.location = force_pos
        arrow.rotation_mode = 'QUATERNION'
        arrow.rotation_quaternion = force_quat
        arrow.scale = (1, 1, force_mag)

        arrow.keyframe_insert(data_path='location', frame=frame)
        arrow.keyframe_insert(data_path='rotation_quaternion', frame=frame)
        arrow.keyframe_insert(data_path='scale', frame=frame)

    return None