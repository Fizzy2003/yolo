from ultralytics import YOLO
from ultralytics import RTDETR

# 算法1:YOLOv8 + BiFPN + MCAttn
pretrained_model = './yolov8l.pt'
model_structure = 'yolov8l-p2.yaml'
dataset = 'VisDrone.yaml'
model = YOLO(model_structure).load(pretrained_model)

# # 算法2:RTDETR + BiFPN + MCAttn
# pretrained_model = './rtdetr-l.pt'
# model_structure = 'rtdetr-l_BiFPN_MCAttn.yaml'
# dataset = 'train_data.yaml'
# model = RTDETR(model_structure).load(pretrained_model)

model.train(data=dataset, epochs=250, batch=8, imgsz=640, workers=8, patience=30)
