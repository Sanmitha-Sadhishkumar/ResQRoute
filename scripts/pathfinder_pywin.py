import win32com.client  # Import pywin32 COM client
import time

def access_thunderhead_pathfinder(project_path):
    """
    Access Thunderhead Pathfinder using COM automation, open a project, run a simulation,
    and save the results.

    Args:
        project_path (str): The full path to the Pathfinder project file (.fpa).
    """
    try:
        # Connect to Thunderhead Pathfinder COM server
        app = win32com.client.Dispatch("Thunderhead.Pathfinder.Application")
        print("Connected to Thunderhead Pathfinder.")

        # Optional: Open a new or existing project
        app.OpenProject(project_path)  # Open project file
        print(f"Project {project_path} opened successfully.")

        # Example: Access some application parameters (e.g., simulation settings)
        simulation_settings = app.SimulationSettings
        print("Simulation Settings:")
        print(f"Simulated Time: {simulation_settings.SimulationTime}")
        print(f"Time Step: {simulation_settings.TimeStep}")
        
        # Example: Running a simulation
        app.StartSimulation()  # Start simulation
        print("Simulation started...")

        # Wait for simulation to complete (you might want to adjust this based on the simulation time)
        while app.SimulationRunning:
            print("Simulation is running...")
            time.sleep(1)  # Wait for a second before checking again
        
        print("Simulation completed.")

        # Save the results (optional)
        output_path = r"C:\path\to\your\output\results.txt"  # Replace with desired output file path
        app.SaveResults(output_path)
        print(f"Results saved to {output_path}.")

    except Exception as e:
        print(f"An error occurred: {e}")
