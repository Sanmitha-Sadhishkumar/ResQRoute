import cv2
import numpy as np

# Function to load and preprocess the PNG floor plan
def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Thresholding the image to get a binary image (black and white)
    _, binary_image = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Optionally, dilate the image to connect any gaps in walls
    kernel = np.ones((3, 3), np.uint8)
    dilated_image = cv2.dilate(binary_image, kernel, iterations=2)
    
    return dilated_image

# Function to detect lines in the floor plan using Hough Transform
def detect_lines(binary_image):
    # Use Canny edge detection to find edges
    edges = cv2.Canny(binary_image, 50, 150, apertureSize=3)
    
    # Use Hough Line Transform to detect lines in the edge-detected image
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)
    
    return lines

# Function to convert detected lines into 3D walls
def convert_to_3d_walls(lines, wall_height=3.0, wall_thickness=0.2):
    walls_3d = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Calculate direction of the wall
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        if length == 0:
            continue
        
        # Normalize the direction vector
        ux = dx / length
        uy = dy / length
        
        # Perpendicular vector for thickness
        px = -uy
        py = ux
        
        # Offset for thickness
        offset_x = (px * wall_thickness) / 2
        offset_y = (py * wall_thickness) / 2
        
        # Define the four corners of the wall (extruded vertically)
        bottom_z = 0.0
        top_z = wall_height
        vertices = [
            (x1 + offset_x, y1 + offset_y, bottom_z),
            (x2 + offset_x, y2 + offset_y, bottom_z),
            (x2 + offset_x, y2 + offset_y, top_z),
            (x1 + offset_x, y1 + offset_y, top_z)
        ]
        walls_3d.append(vertices)
    
    return walls_3d

# Function to generate FDS file for 3D walls
def generate_fds_file(walls_3d, fds_file):
    with open(fds_file, 'w') as f:
        f.write("&HEAD CHID='FloorPlan3D', TITLE='3D Model from PNG Floor Plan'\n/\n\n")
        f.write("&MATERIAL ID='WALLS', CONDUCTIVITY=1.0, SPECIFIC_HEAT=1000.0, DENSITY=1000.0 /\n\n")
        
        for i, wall in enumerate(walls_3d, start=1):
            f.write("&FACES ID='WALL_{0}', MATERIAL='WALLS', BC='INTERNAL', SURF_ID='WALL_SURF_{0}'\n".format(i))
            for vertex in wall:
                f.write("  {:.3f}, {:.3f}, {:.3f},\n".format(*vertex))
            f.write("  {:.3f}, {:.3f}, {:.3f}\n/\n\n".format(*wall[0]))
        
        f.write("&END\n")

# Main function to execute the conversion process
def main():
    image_path = "../floor_plan/ground floor.png"  # Input PNG floor plan
    fds_file = "floorplan3D.fds"       # Output FDS file

    # Preprocess the image
    print("Preprocessing image...")
    binary_image = preprocess_image(image_path)

    # Detect lines corresponding to walls
    print("Detecting lines in the floor plan...")
    lines = detect_lines(binary_image)
    
    if lines is None:
        print("No lines detected.")
        return

    print(f"Detected {len(lines)} walls.")

    # Convert 2D lines to 3D walls
    print("Converting 2D lines to 3D walls...")
    walls_3d = convert_to_3d_walls(lines)

    # Generate the FDS file
    print("Generating FDS file...")
    generate_fds_file(walls_3d, fds_file)
    print(f"FDS file '{fds_file}' generated successfully.")

if __name__ == "__main__":
    main()
