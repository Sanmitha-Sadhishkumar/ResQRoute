import subprocess
import os

# Path to the SketchUp executable
sketchup_exe = '"D:/Program Files/SketchUp/sketchup.exe"'

# Path to the Ruby script (provide the full path or relative path if in the Plugins folder)
ruby_script = "extract_materials.rb"  # Replace with the full path if needed

# Ensure the script exists before attempting to run it
if not os.path.exists(ruby_script):
    print(f"Error: Ruby script '{ruby_script}' not found!")
else:
    # Command to run SketchUp with the Ruby script
    command = f'{sketchup_exe} -RubyStartup "{ruby_script}"'

    # Run the command
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

    # Capture output and errors
    stdout, stderr = process.communicate()

    # Print the output
    if stdout:
        print(stdout.decode())
    if stderr:
        print(stderr.decode())
