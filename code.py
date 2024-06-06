from ultralytics import YOLO

# Carregar o modelo pré-treinado YOLOv8
model = YOLO('yolov8n.pt')  


# Treinar o modelo com o seu conjunto de dados
results = model.train(data='C:/Users/diogo/desafio16.5/data.yaml', epochs=65  , 
imgsz=420, batch=8)  
