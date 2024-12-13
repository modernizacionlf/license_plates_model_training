from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.yaml")  # Build a new model from scratch

# Use the model
results = model.train(data="./data/data.yaml", epochs=85, imgsz=640, batch=4, patience=10)
