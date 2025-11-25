from ultralytics import YOLO

# Carica un modello pre-addestrato (nano = più veloce)
model = YOLO('yolov8n.pt') 

# Addestra
# imgsz=640: dimensione immagine
# epochs=20: bastano poche epoche con YOLO
# batch=16: riduci se finisci la memoria
results = model.train(data='data.yaml', epochs=20, imgsz=420, batch=16, name='captcha_yolo')