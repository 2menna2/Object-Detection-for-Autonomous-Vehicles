from ultralytics import YOLO
import cv2

model = YOLO("model/best.pt")  

video_path = "video/project_video.mp4"  
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Couldn't open video")
    exit()

screen_width = 1920  
screen_height = 1080  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    annotated_frame = results[0].plot()

    resized_frame = cv2.resize(annotated_frame, (screen_width, screen_height))

    cv2.imshow("YOLOv8 Video Prediction", resized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
