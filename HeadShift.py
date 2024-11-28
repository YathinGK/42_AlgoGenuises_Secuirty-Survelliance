import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Define key landmarks for head pose estimation
NOSE_TIP = 1
CHIN = 152
LEFT_EYE_INNER = 133
RIGHT_EYE_INNER = 362
LEFT_EAR = 234
RIGHT_EAR = 454

# Define the 3D model points of a human face
model_points = np.array([
    (0.0, 0.0, 0.0),         # Nose tip
    (0.0, -150.0, -10.0),    # Chin
    (-65.0, 75.0, -10.0),    # Left eye inner
    (65.0, 75.0, -10.0),     # Right eye inner
    (-110.0, -50.0, -30.0),  # Left ear
    (110.0, -50.0, -30.0)    # Right ear
], dtype=np.float32)

# Camera matrix (assumes webcam resolution is 640x480)
focal_length = 640
center = (320, 240)
camera_matrix = np.array([
    [focal_length, 0, center[0]],
    [0, focal_length, center[1]],
    [0, 0, 1]
], dtype=np.float32)

# Distortion coefficients (assuming no lens distortion)
dist_coeffs = np.zeros((4, 1))

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture video.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape
            landmarks = face_landmarks.landmark

            # Get the 2D coordinates of key landmarks
            image_points = np.array([
                (landmarks[NOSE_TIP].x * w, landmarks[NOSE_TIP].y * h),       # Nose tip
                (landmarks[CHIN].x * w, landmarks[CHIN].y * h),               # Chin
                (landmarks[LEFT_EYE_INNER].x * w, landmarks[LEFT_EYE_INNER].y * h),  # Left eye inner
                (landmarks[RIGHT_EYE_INNER].x * w, landmarks[RIGHT_EYE_INNER].y * h), # Right eye inner
                (landmarks[LEFT_EAR].x * w, landmarks[LEFT_EAR].y * h),       # Left ear
                (landmarks[RIGHT_EAR].x * w, landmarks[RIGHT_EAR].y * h)      # Right ear
            ], dtype=np.float32)

            # Solve PnP to estimate the head pose
            success, rotation_vector, translation_vector = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs
            )

            # Get rotation angles
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)

            pitch, yaw, roll = angles[0] * 180 / np.pi, angles[1] * 180 / np.pi, angles[2] * 180 / np.pi

            # Define thresholds for classification
            if pitch > 15:  
                direction = "Looking forward"
            elif yaw > 15:  # Looking left
                direction = "Looking Left"
            elif yaw < -15:  # Looking right
                direction = "Looking Right"
            elif pitch < -10:  # Looking up
                direction = "Looking Up"
            else:
                direction = "Looking down"

            # Display direction and angles
            cv2.putText(frame, f"Direction: {direction}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Pitch: {pitch:.2f}, Yaw: {yaw:.2f}, Roll: {roll:.2f}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Head Pose Estimation", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
face_mesh.close()