from ultralytics import YOLO

# 使用模型进行检测
yolo = YOLO(model="")

result = yolo(source="chip.jpg", save=True, conf=0.5)



