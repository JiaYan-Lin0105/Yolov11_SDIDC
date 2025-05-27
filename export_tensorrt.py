from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("v11m_best.pt")

# Export the model to TensorRT format
model.export(format="engine")  # creates 'yolo11n.engine'

# Load the exported TensorRT model
tensorrt_model = YOLO("v11m_best.engine")

# Run inference
