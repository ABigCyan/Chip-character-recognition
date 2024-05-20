from ultralytics import YOLO

# pt格式模型转换成onnx格式，注意opset=12，否则后面转换为rknn会报错

model = YOLO(model="")

model.export(format='onnx', imgsz=(640,640), opset=12, simplify=True)
