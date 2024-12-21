import cv2
from ultralytics import YOLO
import math
import cvzone
from sort import Sort
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
import firebase_admin
from firebase_admin import db, credentials
from firebase_initialize import *

def preprocess_image(image):
    image = cv2.resize(image, (640, 640))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = img_to_array(image)
    image = preprocess_input(image)
    return image

# Load the Keras model
second_cnn_model = load_model('../models/cnn_model_epoch20.h5')

def sort_algo(video_source):
    result=''''''
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    model = YOLO("../models/yolov8_30.pt")
    classes = {0: 'person'}

    result_array = [classes[i] for i in range(len(classes))]
    line_coords = [593, 500, 958, 500]

    tracker = Sort()
    totalCount = []
    
    # Dictionary to store distances and speeds
    track_id_distance = {}
    track_id_time = {}
    frame_rate = 30  # Assuming a frame rate of 30 FPS
    time_per_frame = 1 / frame_rate  # Time in seconds

    # Initialize a dictionary to map track IDs to CNN results
    track_id_to_cnn_result = {}
    track_id_to_class = {}

    # Define CNN class labels
    cnn_classes = {0: 'adult', 1: 'children', 2: 'elderly'}

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream.")
            break
        
        results = model(frame, stream=True)
        detections = np.empty((0, 5))
        cropped_images = []
        track_ids_for_crops = []  # To keep track of which track ID corresponds to each cropped image

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(frame, (x1, y1, w, h), l=5, rt=2, colorC=(255, 215, 0), colorR=(255, 99, 71))

                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])

                detections = np.vstack((detections, np.array([x1, y1, x2, y2, conf])))
                cropped = frame[y1:y2, x1:x2]
                if cropped.size > 0:
                    input_image = preprocess_image(cropped)
                    cropped_images.append(input_image)
                    # Temporarily store class as track_id will be assigned after tracking
                    track_ids_for_crops.append(cls)

        tracks = tracker.update(detections)

        cnn_predictions = []
        input_batch_np = np.array(cropped_images)
        if len(cropped_images) > 0:
            predictions = second_cnn_model.predict(input_batch_np)
            predicted_classes = np.argmax(predictions, axis=1).tolist()
        else:
            predicted_classes = []

        for i, track in enumerate(tracks):
            track_id = track[4]
            if track_id not in track_id_to_cnn_result and i < len(predicted_classes):
                # Assign CNN result to track ID
                track_id_to_cnn_result[track_id] = predicted_classes[i]
            elif track_id not in track_id_to_cnn_result:
                track_id_to_cnn_result[track_id] = None  # No prediction available

        cv2.line(frame, (line_coords[0], line_coords[1]), (line_coords[2], line_coords[3]), color=(255, 0, 0), thickness=3)

        for i, track in enumerate(tracks):
            x1, y1, x2, y2, track_id = track
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            cls = track_id_to_class.get(track_id, 0)
            cvzone.putTextRect(frame, f'{result_array[cls]} ID:{int(track_id)}', 
                              (max(0, x1), max(35, y1 - 20)), 
                              scale=1, thickness=1, offset=3, colorR=(255, 99, 71))

            cnn_pred = track_id_to_cnn_result.get(track_id)
            if cnn_pred is not None:
                cnn_label = cnn_classes.get(cnn_pred, "Unknown")
                cvzone.putTextRect(frame, f'CNN: {cnn_label}', 
                                  (max(0, x1), y2 + 20), 
                                  scale=1, thickness=1, offset=3, colorR=(0, 0, 0))

            # Calculate center and track distance
            cx, cy = x1 + w // 2, y1 + h // 2
            if track_id not in track_id_distance:
                track_id_distance[track_id] = [(cx, cy)]  # Initialize with the first position
                track_id_time[track_id] = 0  # Initialize time

            else:
                # Calculate distance traveled
                last_position = track_id_distance[track_id][-1]
                distance = math.sqrt((cx - last_position[0]) ** 2 + (cy - last_position[1]) ** 2)
                track_id_distance[track_id].append((cx, cy))

                # Update the total time for this track_id
                track_id_time[track_id] += time_per_frame

            # Check if the object crossed the line
            if line_coords[0] < cx < line_coords[2] and (line_coords[1] - 10) < cy < (line_coords[3] + 10):
                if track_id not in totalCount:
                    totalCount.append(track_id)

        # Calculate and display speeds
        speeds = {}
        for track_id, positions in track_id_distance.items():
            if len(positions) > 1:  # Ensure there is at least one distance traveled
                total_distance = 0
                for j in range(1, len(positions)):
                    last_pos = positions[j-1]
                    curr_pos = positions[j]
                    total_distance += math.sqrt((curr_pos[0] - last_pos[0]) ** 2 + (curr_pos[1] - last_pos[1]) ** 2)
                total_time = track_id_time[track_id]
                if total_time > 0:
                    speed = total_distance / total_time  # Speed in pixels per second
                    speeds[track_id] = speed  # Store the speed
        
        original_height, original_width = frame.shape[:2]
        window_width = 1280
        window_height = 640
        width_changed, height_changed = original_width, original_height
        if original_height>window_height:
            scaling_factor = (original_height/600)
            width_changed = original_width/scaling_factor
            height_changed = original_height/scaling_factor
        elif original_width>window_width:
            scaling_factor = (original_width/1200)
            width_changed = original_width/scaling_factor
            height_changed = original_height/scaling_factor

        print(width_changed, height_changed)
        frame_resized = cv2.resize(frame, (int(width_changed), int(height_changed)))
    
    # Display the resized frame
        cv2.imshow("Image", frame_resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    print("Distances:", track_id_distance)
    #result+=f"Distances: {track_id_distance}"
    print("Speeds:", speeds)
    result+=f"\nSpeeds: "
    for i, j in speeds.items():
        result+=f"\n{i} : {j}"
    
    # Calculate average speed per class
    class_speeds = {'adult': [], 'children': [], 'elderly': []}
    ref = db.reference('/')

    data = ref.get()
    
    for track_id, speed in speeds.items():
        cnn_pred = track_id_to_cnn_result.get(track_id)
        if cnn_pred is not None:
            class_label = cnn_classes.get(cnn_pred, None)
            if class_label:
                class_speeds[class_label].append(speed)
    
    speed = {
    'children': data.get('children'),
    'adult': data.get('adult'),
    'elderly': data.get('elderly')
    }
    # Compute and print average speeds

    print("\nAverage Speeds per Class:")
    result+="\n\nAverage Speeds per Class:"
    for class_label, speed_list in class_speeds.items():
        if speed_list:
            average_speed = sum(speed_list) / len(speed_list)
            average_speed = (average_speed+speed[class_label])/2
            print(f"{class_label.capitalize()}: {average_speed:.2f} pixels/second")
            result+=f"\n{class_label.capitalize()}: {average_speed:.2f} pixels/second"
            if average_speed==0:
                average_speed=0.1
            ref = db.reference('/').update({class_label:average_speed})
        else:
            print(f"{class_label.capitalize()}: No data available")
            result+=f"\n{class_label.capitalize()}: No data available"
    return result

def pre_sort_algo(video_source):
    result=''''''
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    model = YOLO("yolov8n.pt")
    classes = {0: 'person'}

    result_array = [classes[i] for i in range(len(classes))]
    line_coords = [593, 500, 958, 500]

    tracker = Sort()
    totalCount = []
    
    # Dictionary to store distances and speeds
    track_id_distance = {}
    track_id_time = {}
    frame_rate = 30  # Assuming a frame rate of 30 FPS
    time_per_frame = 1 / frame_rate  # Time in seconds

    # Initialize a dictionary to map track IDs to CNN results
    track_id_to_cnn_result = {}
    track_id_to_class = {}

    # Define CNN class labels
    cnn_classes = {0: 'adult', 1: 'children', 2: 'elderly'}

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream.")
            break
        
        results = model(frame, stream=True)
        detections = np.empty((0, 5))
        cropped_images = []
        track_ids_for_crops = []  # To keep track of which track ID corresponds to each cropped image

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(frame, (x1, y1, w, h), l=5, rt=2, colorC=(255, 215, 0), colorR=(255, 99, 71))

                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])

                detections = np.vstack((detections, np.array([x1, y1, x2, y2, conf])))
                cropped = frame[y1:y2, x1:x2]
                if cropped.size > 0:
                    input_image = preprocess_image(cropped)
                    cropped_images.append(input_image)
                    # Temporarily store class as track_id will be assigned after tracking
                    track_ids_for_crops.append(cls)

        tracks = tracker.update(detections)

        cnn_predictions = []
        input_batch_np = np.array(cropped_images)
        if len(cropped_images) > 0:
            predictions = second_cnn_model.predict(input_batch_np)
            predicted_classes = np.argmax(predictions, axis=1).tolist()
        else:
            predicted_classes = []

        for i, track in enumerate(tracks):
            track_id = track[4]
            if track_id not in track_id_to_cnn_result and i < len(predicted_classes):
                # Assign CNN result to track ID
                track_id_to_cnn_result[track_id] = predicted_classes[i]
            elif track_id not in track_id_to_cnn_result:
                track_id_to_cnn_result[track_id] = None  # No prediction available

        cv2.line(frame, (line_coords[0], line_coords[1]), (line_coords[2], line_coords[3]), color=(255, 0, 0), thickness=3)

        for i, track in enumerate(tracks):
            x1, y1, x2, y2, track_id = track
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            cls = track_id_to_class.get(track_id, 0)
            cvzone.putTextRect(frame, f'{result_array[cls]} ID:{int(track_id)}', 
                              (max(0, x1), max(35, y1 - 20)), 
                              scale=1, thickness=1, offset=3, colorR=(255, 99, 71))

            cnn_pred = track_id_to_cnn_result.get(track_id)
            if cnn_pred is not None:
                cnn_label = cnn_classes.get(cnn_pred, "Unknown")
                cvzone.putTextRect(frame, f'CNN: {cnn_label}', 
                                  (max(0, x1), y2 + 20), 
                                  scale=1, thickness=1, offset=3, colorR=(0, 0, 0))

            # Calculate center and track distance
            cx, cy = x1 + w // 2, y1 + h // 2
            if track_id not in track_id_distance:
                track_id_distance[track_id] = [(cx, cy)]  # Initialize with the first position
                track_id_time[track_id] = 0  # Initialize time

            else:
                # Calculate distance traveled
                last_position = track_id_distance[track_id][-1]
                distance = math.sqrt((cx - last_position[0]) ** 2 + (cy - last_position[1]) ** 2)
                track_id_distance[track_id].append((cx, cy))

                # Update the total time for this track_id
                track_id_time[track_id] += time_per_frame

            # Check if the object crossed the line
            if line_coords[0] < cx < line_coords[2] and (line_coords[1] - 10) < cy < (line_coords[3] + 10):
                if track_id not in totalCount:
                    totalCount.append(track_id)

        # Calculate and display speeds
        speeds = {}
        for track_id, positions in track_id_distance.items():
            if len(positions) > 1:  # Ensure there is at least one distance traveled
                total_distance = 0
                for j in range(1, len(positions)):
                    last_pos = positions[j-1]
                    curr_pos = positions[j]
                    total_distance += math.sqrt((curr_pos[0] - last_pos[0]) ** 2 + (curr_pos[1] - last_pos[1]) ** 2)
                total_time = track_id_time[track_id]
                if total_time > 0:
                    speed = total_distance / total_time  # Speed in pixels per second
                    speeds[track_id] = speed  # Store the speed
        
        original_height, original_width = frame.shape[:2]
        window_width = 1280
        window_height = 640
        width_changed, height_changed = original_width, original_height
        if original_height>window_height:
            scaling_factor = (original_height/600)
            width_changed = original_width/scaling_factor
            height_changed = original_height/scaling_factor
        elif original_width>window_width:
            scaling_factor = (original_width/1200)
            width_changed = original_width/scaling_factor
            height_changed = original_height/scaling_factor

        print(width_changed, height_changed)
        frame_resized = cv2.resize(frame, (int(width_changed), int(height_changed)))
    
    # Display the resized frame
        cv2.imshow("Image", frame_resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    print("Distances:", track_id_distance)
    #result+=f"Distances: {track_id_distance}"
    print("Speeds:", speeds)
    result+=f"\nSpeeds: "
    for i, j in speeds.items():
        result+=f"\n{i} : {j}"
    
    # Calculate average speed per class
    class_speeds = {'adult': [], 'children': [], 'elderly': []}
    ref = db.reference('/')

    data = ref.get()
    
    for track_id, speed in speeds.items():
        cnn_pred = track_id_to_cnn_result.get(track_id)
        if cnn_pred is not None:
            class_label = cnn_classes.get(cnn_pred, None)
            if class_label:
                class_speeds[class_label].append(speed)
    
    speed = {
    'children': data.get('children'),
    'adult': data.get('adult'),
    'elderly': data.get('elderly')
    }
    # Compute and print average speeds

    print("\nAverage Speeds per Class:")
    result+="\n\nAverage Speeds per Class:"
    for class_label, speed_list in class_speeds.items():
        if speed_list:
            average_speed = sum(speed_list) / len(speed_list)
            average_speed = (average_speed+speed[class_label])/2
            print(f"{class_label.capitalize()}: {average_speed:.2f} pixels/second")
            result+=f"\n{class_label.capitalize()}: {average_speed:.2f} pixels/second"
            if average_speed==0:
                average_speed=0.1
            ref = db.reference('/').update({class_label:average_speed})
        else:
            print(f"{class_label.capitalize()}: No data available")
            result+=f"\n{class_label.capitalize()}: No data available"
    return result

# file_path = "../samples/1.mp4"
# result=pre_sort_algo(file_path)