import cv2
from ultralytics import YOLO

# Load the model
model = YOLO("jeronimo_2model.pt")

# Traffic sign classes
traffic_sign_classes = [59, 64, 136, 239, 269, 293, 337, 242, 291, 256, 367]

# Open the video
video_input_path = "original_video2.mp4"
video_output_path = "video_output.mp4"

cap = cv2.VideoCapture(video_input_path)

# Get video information
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Get the original width of the video
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Get the original height of the video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))

# Desired new resolution
new_width = 1280
new_height = 560

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Resize the frame to the new resolution
    frame_resized = cv2.resize(frame, (new_width, new_height))
    
    # Perform detection
    results = model.predict(source=frame_resized, task='detect', conf=0.775)
    
    # Iterate over each Detection object and draw the detections
    for result in results:
        image_with_detections = result.plot()
        out.write(image_with_detections)
    
    # Display the frame with detections (optional)
    cv2.imshow('Detections', image_with_detections)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
