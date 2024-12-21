import re

# Function to parse the .obj file
def parse_obj(obj_file):
    vertices = {}
    faces = []

    with open(obj_file, 'r') as file:
        current_material = None
        
        for line in file:
            # Parse vertices (v x y z)
            if line.startswith('v '):
                parts = line.strip().split()
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                vertices[len(vertices) + 1] = (x, y, z)
            
            # Parse faces (f vertex1/uv1/normal1 vertex2/uv2/normal2 ...)
            elif line.startswith('f '):
                parts = line.strip().split()[1:]  # Skip the "f"
                face_vertices = []
                for part in parts:
                    vertex_index = int(part.split('/')[0])
                    face_vertices.append(vertex_index)
                faces.append((current_material, face_vertices))
            
            # Parse material (usemtl)
            elif line.startswith('usemtl '):
                current_material = line.strip().split()[1]

    return vertices, faces


# Function to parse the .mtl file and get the material names
def parse_mtl(mtl_file):
    materials = {}
    current_material = None

    with open(mtl_file, 'r') as file:
        for line in file:
            if line.startswith('newmtl '):
                current_material = line.strip().split()[1]
                materials[current_material] = []
    
    return materials


# Function to extract coordinates for a given material
def extract_coordinates(obj_file, mtl_file, target_material):
    vertices, faces = parse_obj(obj_file)
    materials = parse_mtl(mtl_file)

    # Extract faces for the target material
    coordinates = []
    for material, face_vertices in faces:
        if material == target_material:
            for vertex_index in face_vertices:
                if vertex_index in vertices:
                    coordinates.append(vertices[vertex_index])

    return coordinates


# Example usage
obj_file = '../Dept GF-1.obj'  # Replace with your .obj file path
mtl_file = '../Dept GF-1.mtl'  # Replace with your .mtl file path
target_material = 'LPG'  # Specify the material name you're looking for

coordinates = extract_coordinates(obj_file, mtl_file, target_material)

# Output the coordinates for the 'LPG' material
for coord in coordinates:
    print(f"X: {coord[0]}, Y: {coord[1]}, Z: {coord[2]}")
