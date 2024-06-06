import cv2
from ultralytics import YOLO

# Carregar o modelo
modelo = YOLO("jeronimo_2model.pt")

sinais_de_transito_classes = [59, 64, 136, 239, 269, 293, 337, 242, 291, 256, 367]

# Abrir o vídeo
video_input_path = "original_video2.mp4"
video_output_path = "video_output.mp4"

cap = cv2.VideoCapture(video_input_path)

# Obter informações sobre o vídeo
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Obter a largura original do vídeo
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Obter a altura original do vídeo
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))

# Nova resolução desejada
new_width = 1280
new_height = 560

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Redimensionar o quadro para a nova resolução
    frame_resized = cv2.resize(frame, (new_width, new_height))
    
    # Fazer a detecção
    resultados = modelo.predict(source=frame_resized, task='detect', conf=0.775)
    
    # Iterar sobre cada objecto Detection e desenhar as detecções
    for resultado in resultados:
        image_with_detections = resultado.plot()
        out.write(image_with_detections)
    
    # Mostrar o frame com detecções (opcional)
    cv2.imshow('Detections', image_with_detections)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar os recursos
cap.release()
out.release()
cv2.destroyAllWindows()
