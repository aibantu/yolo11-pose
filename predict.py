from ultralytics import YOLO
import cv2

model = YOLO("yolo11s-pose.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)
    annotated_frame = results[0].plot()
    cv2.imshow("YOLO11 Inference", annotated_frame)

    # 增大等待时间参数，这里改为 10 毫秒
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()