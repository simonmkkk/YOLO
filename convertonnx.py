from ultralytics import YOLO

# 一次性 export（只需執行一次）
model = YOLO("yolo26n.pt")
model.export(format="onnx", half=True)  # 生成 yolo26n.engine