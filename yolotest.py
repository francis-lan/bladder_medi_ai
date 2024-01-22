from ultralytics import YOLO
import numpy as np

model = YOLO("yolov8n.yaml")
model = YOLO("yolov8n.pt")

model.train(data = "D:/User-Data/Downloads/blaff.v1i.yolov8/data.yaml",epochs = 20)
metrics = model.val()  # evaluate model performance on the validation set
results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
path = model.export(format="onnx")  # export the model to ONNX format