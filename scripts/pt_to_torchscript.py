from ultralytics import YOLO

# Load your custom YOLOv8 model
model = YOLO('C:/company/Yolov8classification/roboflow/runs/classify/train3/weights/best.pt')

# Export the model to ONNX format
# output_path = 'C:/company/Yolov8/scripts/runs/detect/model.onnx'
model.export(format='torchscript')