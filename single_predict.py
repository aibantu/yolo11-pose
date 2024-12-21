import cv2
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results
import time
import gc
# Load the YOLO11 model
model = YOLO("yolo11s-pose.pt")

# Open the video file
# video_path = "path/to/video.mp4"
cap = cv2.VideoCapture(0)
target_id = 1
# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO11 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        filtered_results = []
        for r in results:
            boxes = r.boxes
            keypoints = r.keypoints
            new_boxes = [box for box in boxes if box.id == target_id]  # 使用列表推导式筛选符合条件的box
            if new_boxes:  # 如果有符合条件的box，才构建新的Results对象
                r.boxes = new_boxes  # 更新当前结果中的boxes为筛选后的boxes
                new_kp_data = keypoints.data[0].unsqueeze(0)
                r.keypoints.data = new_kp_data
                keypoints = None
                filtered_results.append(r)  # 将更新后的结果添加到filtered_results中
            else:
                pass

        if filtered_results:  # 确保有符合条件的结果才进行后续可视化操作
            # Visualize the results on the frame
            annotated_frame = filtered_results[0].plot()

            # Display the annotated frame
            cv2.imshow("YOLO11 Tracking", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(6) & 0xFF == ord('q'):
                break

            gc.collect()

            # results = []
            filtered_results = []
        else:
            gc.collect()

            # results = []
            filtered_results = []



    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()



