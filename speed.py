import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import Tracker
import time
import csv
import imageio  # For saving GIFs

# Load YOLO model
model = YOLO('yolov8s.pt')

# Mouse callback to get pixel positions
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print([x, y])

cv2.namedWindow('Traffic Monitoring')
cv2.setMouseCallback('Traffic Monitoring', RGB)

# Video capture setup
cap = cv2.VideoCapture('veh2.mp4')

fps = int(cap.get(cv2.CAP_PROP_FPS))
tracker = Tracker()

# Line positions and offset
cy1, cy2, offset = 322, 368, 6

# Trackers and counters
vh_down, vh_up = {}, {}
counter_down, counter_up = [], []

# Read class names
with open("coco.txt", "r") as f:
    class_list = f.read().split("\n")

# List to store frames for GIF
frames_for_gif = []

count = 0

# CSV setup
with open('car_details.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Car ID", "Direction", "Speed (Km/h)", "Timestamp"])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        count += 1
        if count % 3 != 0:  # Skip frames for speed
            continue

        # Resize frame for processing
        frame = cv2.resize(frame, (1020, 500))

        # YOLO detection
        results = model.predict(frame)
        boxes = results[0].boxes.data
        px = pd.DataFrame(boxes).astype("float")
        detections = []

        for _, row in px.iterrows():
            x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
            class_id = int(row[5])
            if 'car' in class_list[class_id]:
                detections.append([x1, y1, x2, y2])

        # Update tracker
        bbox_id = tracker.update(detections)

        for bbox in bbox_id:
            x3, y3, x4, y4, obj_id = bbox
            cx, cy = (x3 + x4) // 2, (y3 + y4) // 2

            # Draw bounding box
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)

            # Down direction
            if cy1 - offset < cy < cy1 + offset:
                vh_down[obj_id] = time.time()
            if obj_id in vh_down and cy2 - offset < cy < cy2 + offset and obj_id not in counter_down:
                elapsed_time = time.time() - vh_down[obj_id]
                speed_kmh = (10 / elapsed_time) * 3.6  # distance = 10 meters
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                writer.writerow([obj_id, "Down", int(speed_kmh), timestamp])
                counter_down.append(obj_id)
                cv2.putText(frame, f"{int(speed_kmh)} Km/h", (x4, y4),
                            cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

            # Up direction
            if cy2 - offset < cy < cy2 + offset:
                vh_up[obj_id] = time.time()
            if obj_id in vh_up and cy1 - offset < cy < cy1 + offset and obj_id not in counter_up:
                elapsed_time = time.time() - vh_up[obj_id]
                speed_kmh = (10 / elapsed_time) * 3.6
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                writer.writerow([obj_id, "Up", int(speed_kmh), timestamp])
                counter_up.append(obj_id)
                cv2.putText(frame, f"{int(speed_kmh)} Km/h", (x4, y4),
                            cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

        # Draw lines
        cv2.line(frame, (274, cy1), (814, cy1), (255, 255, 255), 1)
        cv2.putText(frame, 'HOSEN', (277, 320), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
        cv2.line(frame, (177, cy2), (927, cy2), (255, 255, 255), 1)
        cv2.putText(frame, 'ARAFAT', (182, 367), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

        # Display counts
        cv2.putText(frame, f'Going down: {len(counter_down)}', (60, 90),
                    cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f'Going up: {len(counter_up)}', (60, 130),
                    cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

        # Append frame for GIF (convert BGR to RGB & resize)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.uint8)
        frame_rgb = cv2.resize(frame_rgb, (480, int(480 * frame.shape[0] / frame.shape[1])))
        frames_for_gif.append(frame_rgb)

        # Display
        cv2.imshow("Traffic Monitoring", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break

# Save GIF
if frames_for_gif:
    duration = 1 / (fps / 3)  # Adjust for skipped frames
    imageio.mimsave('traffic_output.gif', frames_for_gif, duration=duration)
    print("GIF saved successfully!")
else:
    print("No frames captured, GIF not saved.")

cap.release()
cv2.destroyAllWindows()
