import cv2
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results
import time
import gc

model = YOLO("yolo11n-pose.pt") # 加载模型
cap = cv2.VideoCapture(0)  # 打开摄像头
target_id = 1              # 设定目标ID
target_exists = True       # 标志位，设定存在目标
while cap.isOpened():
    success, frame = cap.read()  # 从摄像头读取一帧
    print(type(frame))
    if success:
        results = model.track(frame, persist=True) # 运行track

        filtered_results = []                      # 过滤结果
        target_present_in_frame = False
        for r in results:
            boxes = r.boxes                        # 过滤目标边框
            keypoints = r.keypoints                # 过滤关键点
            current_boxes = [box for box in boxes if box.id == target_id]   # 检查当前帧中是否存在目标id对应的目标框
            if current_boxes:
                target_present_in_frame = True
                r.boxes = current_boxes
                new_kp_data = keypoints.data[0].unsqueeze(0) if keypoints.data.size(0) > 0 else torch.empty((1, keypoints.data.shape[1], 3))
                r.keypoints.data = new_kp_data
                filtered_results.append(r)

        # 如果当前帧中不存在目标id对应的目标，尝试寻找新的目标
        if not target_present_in_frame and target_exists:
            target_exists = False
            for r in results:
                boxes = r.boxes
                if boxes:
                    # 将新出现的第一个目标的id作为新的跟踪对象id
                    target_id = boxes[0].id
                    target_exists = True
                    break

        if filtered_results:
            annotated_frame = filtered_results[0].plot()  
            cv2.imshow("YOLO11 Tracking", annotated_frame)
            if cv2.waitKey(6) & 0xFF == ord('q'):
                break
            gc.collect()
            filtered_results = []

    else:
        break

cap.release()
cv2.destroyAllWindows()