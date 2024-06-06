from ultralytics import YOLO

# Load the trained model
model_path = 'C:/Users/diogo/desafio16.5/runs/detect/train/weights/best.pt'  
model = YOLO(model_path)

# Save the trained model
save_path = 'C:/Users/diogo/desafio16.5/jeronimo_2model.pt'
model.save(save_path)

print(f"Modelo salvo em: {save_path}")
