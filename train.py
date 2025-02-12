import os
from ultralytics import YOLO
from roboflow import Roboflow
from dotenv import load_dotenv

load_dotenv()

rf = Roboflow(api_key=os.getenv("ROBOFLOW"))
project = rf.workspace("robosub").project("pre-qualification-gate")
version = project.version(4)
dataset = version.download("yolov8",location="datasets")

# Load a pre-trained YOLOv10n model
model = YOLO("yolov10n.pt")


model.train(data="datasets/Pre-qualification-Gate-4/data.yaml", epochs=50, imgsz=640, batch=16, device='mps')

# Perform object detection on an image
results = model("datasets/Pre-qualification-Gate-4/test/images/frame110_jpg.rf.abc86f03d152466fb80f5471826d260e.jpg")

# Display the results
results[0].show()