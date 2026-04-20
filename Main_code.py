import cv2
from ultralytics import YOLO

print("===  Real-Time AI Object Detection ===")

detector_model = YOLO("yolov8n.pt")

video_cap = cv2.VideoCapture(0)

while True:
    ret,vedio_frame = video_cap.read()

    if not ret:
        print("Camera error")
        break

    detection_results = detector_model(vedio_frame)

    output_frame = detection_results[0].plot()
    cv2.putText(output_frame, "Ayesha AI Vision", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Ayesha AI vision", output_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        print("exiting... detection system....")
        break

video_cap.release()
cv2.destroyAllWindows()