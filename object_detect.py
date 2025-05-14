from ultralytics import YOLO
import cv2

model = YOLO("model/best.pt")  # مسار الموديل بتاعك

video_path = "video/project_video.mp4"  # ← غيري ده لمسار الفيديو اللي عندك
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Couldn't open video")
    exit()

screen_width = 1920  # ممكن تغيرها حسب حجم شاشتك
screen_height = 1080  # ممكن تغيرها حسب حجم شاشتك

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
