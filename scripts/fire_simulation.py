import pyrosim
import trimesh
import numpy as np
from pygltflib import GLTF2

class GLBModel:
    def __init__(self, file_path):
        self.file_path = file_path
        self.meshes = []
        self.load_model()

    def load_model(self):
        # Load the GLB file using trimesh
        try:
            self.mesh = trimesh.load(self.file_path)
            print(f"Model loaded successfully: {self.file_path}")
            self.extract_meshes()
        except Exception as e:
            print(f"Error loading model: {e}")

    def extract_meshes(self):
        # Extract mesh data; trimesh handles this internally
        if isinstance(self.mesh, trimesh.Scene):
            for name, geom in self.mesh.geometry.items():
                self.meshes.append(geom)
        elif isinstance(self.mesh, trimesh.Trimesh):
            self.meshes.append(self.mesh)
        else:
            print("Unsupported mesh type.")

    def get_primitives(self):
    # Placeholder: Convert meshes to primitives that pyrosim can understand
        primitives = []
        for i, mesh in enumerate(self.meshes):
            # Attempt to fit bounding boxes
            bounding_box = mesh.bounding_box_oriented
            extents = bounding_box.extents
            center = bounding_box.centroid
            # Log the extents and center
            print(f"Mesh {i}: Extents = {extents}, Center = {center}")

            # Check for non-positive dimensions
            if extents[0] <= 0 or extents[1] <= 0 or extents[2] <= 0:
                print(f"Warning: Mesh {i} has non-positive extents: {extents}")
                continue  # Skip this mesh or handle it as needed

            # Create a box primitive
            primitive = {
            'type': 'box',
            'x': center[0],
            'y': center[1],
            'z': center[2],
            'length': extents[0],
            'width': extents[1],
            'height': extents[2],
            'mass': 1.0,
            'color': [1.0, 0.0, 0.0]  # Red color as default
            }
            primitives.append(primitive)
        return primitives

def setup_simulation(primitives, fire_origin, sim_duration=100, dt=0.05):
    # Initialize the Simulator
    sim = pyrosim.Simulator(eval_time=sim_duration, dt=dt, gravity=-9.81)

    body_ids = []
    # Create simulation bodies based on primitives
    for i, prim in enumerate(primitives):
        if prim['type'] == 'box':
            body_id = sim.send_box(
                x=prim['x'],
                y=prim['y'],
                z=prim['z'],
                length=prim['length'],
                width=prim['width'],
                height=prim['height'],
                mass=prim['mass'],
                r=prim['color'][0],
                g=prim['color'][1],
                b=prim['color'][2]
            )
            body_ids.append(body_id)
            print(f"Added box with ID {body_id} at position ({prim['x']}, {prim['y']}, {prim['z']})")

    # Add fire simulation elements
    fire_force_x = 0
    fire_force_y = 0
    fire_force_z = 50  # Upward force

    # Apply external fire force to the world
    success = sim.send_external_force(
        body_id=-1,  # Use -1 for the world
        x=fire_force_x,
        y=fire_force_y,
        z=fire_force_z,
        time=0  # Apply at the start of the simulation
    )
    
    if success:
        print(f"Applied external fire force at origin {fire_origin}")
    else:
        print("Failed to apply external fire force.")

    # Optionally, add sensors or other elements as needed
    light_id = sim.send_light_source(body_id=-1)  # Attach to the world
    print(f"Added light source with ID {light_id}")

    # Start the simulation
    sim.start()
    sim.wait_to_finish()


def main():
    # Path to your .glb file
    glb_file_path = "../static/show/model.glb"

    # Load and parse the GLB model
    model = GLBModel(glb_file_path)
    primitives = model.get_primitives()

    if not primitives:
        print("No primitives extracted from the model. Exiting.")
        return

    # Define the origin of the fire (x, y, z)
    fire_origin = [0, 0, 0]  # Adjust based on your model's coordinate system

    # Set up and run the simulation
    setup_simulation(primitives, fire_origin, sim_duration=100, dt=0.05)

if __name__ == "__main__":
    main()
