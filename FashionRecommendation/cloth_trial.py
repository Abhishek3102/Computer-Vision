import cv2
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture(0)
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()
tshirt = cv2.imread("tshirt.png", cv2.IMREAD_UNCHANGED)

if tshirt is None:
    print("Error loading image")
    cap.release()
    cv2.destroyAllWindows()
    exit()

def overlay_image(background, overlay, x, y, w, h):
    overlay = cv2.resize(overlay, (w, h))
    if overlay.shape[2] == 4:
        for i in range(overlay.shape[0]):
            for j in range(overlay.shape[1]):
                if overlay[i, j, 3] != 0:
                    if y + i < background.shape[0] and x + j < background.shape[1]:
                        background[y + i, x + j] = overlay[i, j][:3]
    return background

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        landmarks = results.pose_landmarks.landmark

        h, w, _ = frame.shape
        key_points = [
            mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
            mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.RIGHT_ELBOW,
            mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP
        ]
        
        points = {kp: (int(landmarks[kp].x * w), int(landmarks[kp].y * h)) for kp in key_points}
        
        for _, (x, y) in points.items():
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        
        shoulder_width = abs(points[mp_pose.PoseLandmark.RIGHT_SHOULDER][0] - points[mp_pose.PoseLandmark.LEFT_SHOULDER][0])
        chest_height = abs(points[mp_pose.PoseLandmark.LEFT_HIP][1] - points[mp_pose.PoseLandmark.LEFT_SHOULDER][1])
        
        tshirt_x = points[mp_pose.PoseLandmark.LEFT_SHOULDER][0]
        tshirt_y = points[mp_pose.PoseLandmark.LEFT_SHOULDER][1]
        
        frame = overlay_image(frame, tshirt, tshirt_x, tshirt_y, shoulder_width, chest_height)

    cv2.imshow("Virtual Try-On", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
