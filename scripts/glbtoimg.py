import bpy

# Check Blender version
blender_version = bpy.app.version
print(f"Blender version: {blender_version}")

# Step 1: Import FBX model
fbx_file_path = "../static/show/Binary_Dept.fbx"
bpy.ops.import_scene.fbx(filepath=fbx_file_path)

# Step 2: Add a camera to the scene and position it inside the model
camera = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
bpy.context.collection.objects.link(camera)
bpy.context.scene.camera = camera

# Adjust camera location and rotation for a better starting point inside the model
coords = [
    (0, 0, -4),
    #(0.5,1,-4),
    (1, 1, -4),
    (-1, -1, -4),
]

for i, j, k in coords:
    camera.location = (i, j, k)
    camera.rotation_euler = (0, 0, 0)  # Facing along the X-axis

    # Step 3: Add lighting to avoid harsh shadows (optional)
    if not bpy.data.lights:
        light_data = bpy.data.lights.new(name="Light", type='POINT')
        light_data.energy = 1000  # Adjust light intensity
        light_object = bpy.data.objects.new(name="Light", object_data=light_data)
        bpy.context.collection.objects.link(light_object)
        light_object.location = (0, 0, 10)  # Place light above the scene

    # Disable shadows in lighting
    for light in bpy.data.lights:
        light.use_shadow = False  # Disable shadow for lights

    # Step 4: Adjust environment lighting to avoid additional shadows
    bpy.context.scene.world.use_nodes = False  # Disable HDRI environment lighting

    # Step 5: Define a camera travel path
    curve_data = bpy.data.curves.new('camera_path', type='CURVE')
    curve_data.dimensions = '3D'
    camera_path = bpy.data.objects.new('camera_path', curve_data)
    bpy.context.collection.objects.link(camera_path)

    # Add a spline to the curve
    spline = curve_data.splines.new('BEZIER')
    spline.bezier_points.add(4)  # Define the number of points (5 in total)
    spline.bezier_points[0].co = (0, 0, 5)    # Starting point
    spline.bezier_points[1].co = (5, 5, 5)    # Midpoints
    spline.bezier_points[2].co = (10, -5, 5)
    spline.bezier_points[3].co = (15, 0, 5)
    spline.bezier_points[4].co = (20, 5, 5)   # Ending point

    # Step 6: Animate camera along the path
    camera_constraint = camera.constraints.new('FOLLOW_PATH')
    camera_constraint.target = camera_path
    camera_constraint.use_fixed_location = True

    # Set the scene frame range for the camera animation
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = 1  # Number of frames for travel

    # Step 7: Modify render settings to minimize shadows
    # Set render engine to Eevee (can be changed to Cycles if preferred)
    bpy.context.scene.render.engine = 'BLENDER_EEVEE'

    # Try disabling soft shadows if available
    if hasattr(bpy.context.scene.eevee, 'use_soft_shadows'):
        bpy.context.scene.eevee.use_soft_shadows = False  # Disable soft shadows

    # Try disabling contact shadows if available (this may not exist in all versions)
    if hasattr(bpy.context.scene.eevee, 'use_contact_shadows'):
        bpy.context.scene.eevee.use_contact_shadows = False  # Disable contact shadows

    # Set the resolution and output format
    bpy.context.scene.render.resolution_x = 1920
    bpy.context.scene.render.resolution_y = 1080
    bpy.context.scene.render.image_settings.file_format = 'PNG'

    # Step 8: Animate camera moving through each frame
    for frame in range(bpy.context.scene.frame_start, bpy.context.scene.frame_end):
        bpy.context.scene.frame_set(frame)

        # Update camera position on the path at the current frame
        camera_constraint.offset_factor = frame / bpy.context.scene.frame_end

        # Render and save image at each frame
        bpy.context.scene.render.filepath = f'../static/show/extracted_images/frame_{frame:03d}_{i}_{j}.png'
        bpy.ops.render.render(write_still=True)

print("Image extraction complete!")
