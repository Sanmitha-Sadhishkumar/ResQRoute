import cv2
import os

import os
import cv2

def room_split_func(input_image_path, output_folder, meter_ratio):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Load the image
    image = cv2.imread(input_image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the image to binary
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Border padding
    padding = 5  # Adjust this to control how much border to include

    # Exclude the largest rectangle by area
    areas = [cv2.contourArea(contour) for contour in contours]
    max_area = max(areas)
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) < max_area]

    # Dictionary to store room sizes
    room_sizes = {}

    # String to accumulate all room details
    room_details_string = ""

    # Annotate and save each cropped rectangle
    for i, contour in enumerate(filtered_contours):
        # Get bounding rectangle for each contour
        x, y, w, h = cv2.boundingRect(contour)

        # Avoid noise (small rectangles)
        if w > 10 and h > 10:
            # Expand the bounding box to include the border
            x_start = max(0, x - padding)
            y_start = max(0, y - padding)
            x_end = min(image.shape[1], x + w + padding)
            y_end = min(image.shape[0], y + h + padding)

            # Crop the rectangle
            cropped = image[y_start:y_end, x_start:x_end]
            output_path = os.path.join(output_folder, f'room_{i}.png')
            cv2.imwrite(output_path, cropped)

            # Convert dimensions to meters
            length_m = w * int(meter_ratio)
            breadth_m = h * int(meter_ratio)
            area_m2 = length_m * breadth_m

            # Store size and coordinates in the dictionary
            room_sizes[f'room_{i}'] = {
                'length_m': length_m,
                'breadth_m': breadth_m,
                'area_m2': area_m2,
                'coordinates': {'x': x, 'y': y}
            }

            # Annotate the original image at the center of the rectangle
            center_x = x + w // 2
            center_y = y + h // 2
            cv2.putText(
                image,
                f'Room {i}',
                (center_x - 30, center_y),  # Adjust offset for better alignment
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),  # Red color for text
                1,
                cv2.LINE_AA
            )
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)  # Green rectangle

            # Accumulate room details in the string
            room_details_string += f"Room: {i}, length: {length_m}, breadth: {breadth_m}, area: {area_m2}\n"

    # Save the annotated image
    annotated_image_path = os.path.join(output_folder, 'annotated_image.png')
    cv2.imwrite(annotated_image_path, image)

    # Save the room sizes dictionary to a text file
    room_sizes_path = os.path.join(output_folder, 'room_sizes.txt')
    with open(room_sizes_path, 'w') as f:
        for room, details in room_sizes.items():
            f.write(f"{room}: {details}\n")

    # Return the accumulated room details as a single string
    return room_sizes, room_details_string

def fire_extinguisher_calculator(materials, area, room):
    print("Fire Extinguisher Recommendation Tool")
    print("=====================================")
    
    # Initialize recommendations
    extinguishers = []

    # Determine fire classifications based on materials
    if any(item in materials for item in [
        "wood", "furniture", "pallets", "firewood",
        "paper", "cardboard", "documents", "books", "packaging materials",
        "cloth", "clothing", "curtains", "upholstery",
        "plastics", "polystyrene", "polyethylene"
    ]):
        extinguishers.append(("Class A", "Water or Foam Extinguisher"))

    if any(item in materials for item in [
        "oil", "motor oil", "lubricants",
        "paint", "oil-based paint", "solvent-based paint",
        "gasoline", "diesel fuel",
        "alcohol", "ethanol", "methanol",
        "solvent", "acetone", "turpentine", "toluene",
        "propane", "butane", "kerosene"
    ]):
        extinguishers.append(("Class B", "Foam, CO₂, or Dry Powder Extinguisher"))

    if any(item in materials for item in [
        "electronics", "computers", "servers", "appliances",
        "wiring", "cables", "circuit boards", "power strips",
        "electric motors", "transformers"
    ]):
        extinguishers.append(("Class C", "CO₂ or Dry Powder Extinguisher (non-conductive)"))

    if any(item in materials for item in [
        "metals", "magnesium", "sodium", "potassium",
        "aluminum", "aluminum shavings", "aluminum powder",
        "titanium", "lithium", "lithium batteries", "tools"
    ]):
        extinguishers.append(("Class D", "Specialized Dry Powder Extinguisher"))

    if any(item in materials for item in [
        "kitchen grease", "animal fat", "lard",
        "cooking oil", "vegetable oil", "canola oil", "olive oil",
        "deep fryer oil", "butter", "margarine"
    ]):
        extinguishers.append(("Class K", "Wet Chemical Extinguisher"))

    # Calculate quantity based on area and class
    quantity = {}
    for extinguisher in extinguishers:
        fire_class, extinguisher_type = extinguisher
        if fire_class == "Class A":
            # 1 extinguisher per 200 m²
            quantity[extinguisher] = max(1, min(2, int(area / 200)))
        elif fire_class in ["Class B", "Class C", "Class K"]:
            # 1 extinguisher per 30 m²
            quantity[extinguisher] = max(1, min(2, int(area / 30)))
        elif fire_class == "Class D":
            # Typically 1 extinguisher for specific hazards
            quantity[extinguisher] = 1

    # Adjust for very small areas
    if area <= 50:  # Small rooms: suggest 1 extinguisher max per fire class
        for extinguisher in quantity:
            quantity[extinguisher] = 1

    # Display results
    result = [f"\n\nRecommended Fire Extinguishers for {room}: ({area} m²)"]
    for extinguisher, count in quantity.items():
        fire_class, extinguisher_type = extinguisher
        result.append(f"- {count} units of {extinguisher_type} ({fire_class})")

    # Summary
    result.append("\nEnsure extinguishers are placed according to NFPA 10 guidelines.")
    result.append("Class A: Max travel distance 75 ft | Class B: Max 30 ft | Class C: Near electrical equipment.")
    result.append("Class D: Near flammable metals | Class K: Within 30 ft of cooking appliances.")

    return "\n".join(result)
