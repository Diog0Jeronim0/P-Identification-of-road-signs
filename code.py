from ultralytics import YOLO

# Load YOLOv8 pre-trained model
model = YOLO('yolov8n.pt')  


# Train the model with your dataset
results = model.train(data='C:/Users/diogo/desafio16.5/data.yaml', epochs=65  , 
imgsz=420, batch=8)  
