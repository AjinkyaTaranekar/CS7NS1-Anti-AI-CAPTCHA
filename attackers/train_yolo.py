from ultralytics import YOLO

# Carica un modello pre-addestrato (nano = più veloce)
model = YOLO('yolov8n.pt') 

# Addestra
# imgsz=640
# epochs=20
# batch=16
results = model.train(data='data.yaml', epochs=20, imgsz=420, batch=16, name='captcha_yolo')