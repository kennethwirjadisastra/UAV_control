import bpy
import numpy as np

class BlenderVehicle:
    def __init__(self, obj=None, name=None):
        if obj:
            self.obj = obj
        else:
            bpy.ops.mesh.primitive_cube_add(size=2)
            self.obj = bpy.context.active_object
            self.obj.name = name

    def set_location(self, x, y, z):
        self.obj.location = (x, y, z)

    def set_rotation_quaternion(self, qw, qx, qy, qz):
        # Blender uses quaternion in (w, x, y, z) order
        self.obj.rotation_mode = 'QUATERNION'
        self.obj.rotation_quaternion = (qw, qx, qy, qz)

def delete_existing_vehicle(vehicle):
    objs_to_delete = [obj for obj in bpy.data.objects if obj.name.startswith(vehicle)]
    for obj in objs_to_delete:
        bpy.data.objects.remove(obj, do_unlink=True)

def animate_car(car, trajectory):
    scene = bpy.context.scene
    frame_start = scene.frame_start
    for i, row in enumerate(trajectory):
        frame = frame_start + i
        x, y, z, qw, qx, qy, qz = row
        
        car.set_location(x, y, z)
        car.set_rotation_quaternion(qw, qx, qy, qz)
        
        # Insert keyframes for location and rotation
        car.obj.keyframe_insert(data_path="location", frame=frame)
        car.obj.keyframe_insert(data_path="rotation_quaternion", frame=frame)

if __name__ == '__main__':  
    

    
    car_obj = bpy.data.objects.get("Tesla Model 3")
    if car_obj is None:
        raise Exception("Car object not found in scene!")
        
    car_obj.rotation_euler = (0, 0, -np.pi/2)  # Rotate -90 degrees around Z
    bpy.context.view_layer.objects.active = car_obj
    car_obj.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
        
    car = BlenderVehicle(car_obj)
    
    traj = np.loadtxt('trajectory.csv', delimiter=',')
    animate_car(car, traj)