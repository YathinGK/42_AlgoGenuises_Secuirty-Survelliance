import cv2
import torch
import time

# Load YOLO model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Use 'yolov5n', 'yolov5m', etc., for other variants

# Path to the video
video_path = "new.mp4"

# Open the video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Function to check if bounding boxes overlap
def is_overlapping(box1, box2):
    x1, y1, x1w, y1h = box1
    x2, y2, x2w, y2h = box2

    # Check if boxes overlap
    if x1w < x2 or x2w < x1 or y1h < y2 or y2h < y1:
        return False, None

    # Calculate overlap region (intersection)
    overlap_x = (max(x1, x2) + min(x1w, x2w)) // 2
    overlap_y = (max(y1, y2) + min(y1h, y2h)) // 2
    return True, (overlap_x, overlap_y)

# Dictionary to track collision status
collisions = {}

# Process the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = model(frame)
    detections = results.xyxy[0]  # Bounding boxes

    cars = []
    for *box, conf, cls in detections.cpu().numpy():
        label = results.names[int(cls)]
        if label == "car":  # Filter only cars
            x1, y1, x2, y2 = map(int, box)
            cars.append((x1, y1, x2, y2))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Check for overlapping cars and track collisions
    for i in range(len(cars)):
        for j in range(i + 1, len(cars)):
            overlapping, overlap_point = is_overlapping(cars[i], cars[j])
            key = (i, j)  # Unique pair identifier for cars

            if overlapping:
                collisions[key] = overlap_point  # Mark collision
            elif key in collisions:
                # Keep displaying "collision" if it was previously detected
                overlap_point = collisions[key]
                cv2.putText(frame, "collision", overlap_point, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Car Collision Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()