import pathfinder

# Step 1: Create a new Pathfinder project
project = pathfinder.Project()

# Step 2: Import the 3D model
model_path = "path/to/your/model.fbx"  # Replace with your file path
project.import_geometry(model_path)

# Step 3: Access the floors in the model
floors = project.get_floors()
print("Available floors:")
for i, floor in enumerate(floors):
    print(f"{i}: {floor.name}")

# Step 4: Choose a floor (update as needed)
# Example: Selecting the first floor
selected_floor = floors[0]
print(f"Selected floor: {selected_floor.name}")

# Step 5: Add humans to the selected floor
humans = [
    {"name": "Agent1", "x": 5.0, "y": 10.0},  # X, Y on the selected floor
    {"name": "Agent2", "x": 15.0, "y": 12.0},
]

for human in humans:
    agent = project.add_human()  # Add a new human agent
    agent.name = human["name"]
    agent.position = (human["x"], human["y"], selected_floor.elevation)  # Add Z based on floor elevation

# Step 6: Save the Pathfinder project
project.save("path/to/your/pathfinder_project.pth")

# Step 7: Run the simulation
simulation = project.simulation()
simulation.run()
print("Simulation completed.")
