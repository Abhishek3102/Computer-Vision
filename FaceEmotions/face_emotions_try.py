import cv2
from deepface import DeepFace
import time  # Import the time module

cap = cv2.VideoCapture(0)

while True:
    pTime = time.time()  # Start time tracking for FPS
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture image")
        break

    try:
        # Analyze the frame for emotion detection
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        
        # Access dominant emotion
        dominant_emotion = result[0]['dominant_emotion']  
        
        # Calculate FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        
        # Display the detected emotion and FPS on the frame
        cv2.putText(frame, f'Emotion: {dominant_emotion}', (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'FPS: {int(fps)}', (50, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    except Exception as e:
        print(f"Error: {e}")

    # Show the frame with the emotion text overlay
    cv2.imshow('Emotion Detection', frame)

    # Press 'q' to quit the webcam feed
    key = cv2.waitKey(7)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('s'):
        img_name = f"emotion_detection_{time.time()}.png"
        cv2.imwrite(img_name, frame) 
        print(f"Image saved as {img_name}")

cap.release()
cv2.destroyAllWindows()
