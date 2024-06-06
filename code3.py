from ultralytics import YOLO

# Carregar o modelo treinado
model_path = 'C:/Users/diogo/desafio16.5/runs/detect/train/weights/best.pt'  
model = YOLO(model_path)

# Salvar o modelo treinado
save_path = 'C:/Users/diogo/desafio16.5/jeronimo_2model.pt'
model.save(save_path)

print(f"Modelo salvo em: {save_path}")
