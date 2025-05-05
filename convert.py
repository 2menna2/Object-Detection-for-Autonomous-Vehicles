
from ultralytics import YOLO

# Load the model
model = YOLO("best.pt")

# Export to ONNX
model.export(format="onnx")

