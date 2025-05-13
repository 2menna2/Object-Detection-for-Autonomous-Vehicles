import cv2
import numpy as np
from .model_utils import preprocess_image, postprocess

def run_inference_on_video(video_path, output_path, session, input_name, class_names):
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img_input = preprocess_image(frame)
        outputs = session.run(None, {input_name: img_input})
        detections = postprocess(outputs)

        for box, conf, class_id in detections:
            x, y, w_box, h_box = box
            x1 = int((x - w_box / 2) * frame.shape[1] / 640)
            y1 = int((y - h_box / 2) * frame.shape[0] / 640)
            x2 = int((x + w_box / 2) * frame.shape[1] / 640)
            y2 = int((y + h_box / 2) * frame.shape[0] / 640)

            label = f"{class_names[class_id]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        out.write(frame)

    cap.release()
    out.release()
