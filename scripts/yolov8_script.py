from ultralytics import YOLO
import matplotlib
matplotlib.use('Agg')

def yolov8_train(epochs):
    model = YOLO("yolov8n.yaml")
    model = YOLO("yolov8n.pt")
    model = YOLO("yolov8n.yaml").load("yolov8n.pt")

    results = model.train(data="../human_data.yaml", epochs=int(epochs), imgsz=640)

def yolov8_detect(model, image_path):
    model = YOLO(model)
    results = model.predict(source=image_path, save=True, imgsz=640)
    return results

def pre_yolov8_detect(model, image_path):
    model = YOLO(model)
    results = model.predict(source=image_path, save=True, imgsz=640)
    return results

def yolov8_val(model):
    model = YOLO(model)
    results = model.val(data="../hazardousdata.yaml", imgsz=640)
    return results

def hsyolov8_train(epochs):
    model = YOLO("yolov8n.yaml")
    model = YOLO("yolov8n.pt")
    model = YOLO("yolov8n.yaml").load("yolov8n.pt")

    results = model.train(data="../hazardousdata.yaml", epochs=int(epochs), imgsz=640)

def hsyolov8_detect(model, image_path):
    model = YOLO(model)
    results = model.predict(source=image_path, save=True, imgsz=640)
    return results

def hsyolov8_val(model):
    model = YOLO(model)
    results = model.val(data="../hazardousdata.yaml", imgsz=640)
    return results