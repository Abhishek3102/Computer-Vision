import cv2
from deepface import DeepFace
import time

# Initialize webcam capture
cap = cv2.VideoCapture(0)

while True:
    pTime = time.time()  # Track time for FPS calculation
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture image")
        break

    try:
        # Analyze the frame for emotion, age, and gender
        result = DeepFace.analyze(frame, actions=['age', 'gender'], enforce_detection=False)

        # Access the predicted age and gender
        age = result[0]['age']
        gender = result[0]['gender']
        
        # Log the result to help you investigate
        print(f"Predicted Gender: {gender} - Predicted Age: {age}")
        
        # Calculate FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        
        # Display the age, gender, and FPS on the frame
        cv2.putText(frame, f'Age: {age}', (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Gender: {gender}', (50, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'FPS: {int(fps)}', (50, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    except Exception as e:
        print(f"Error: {e}")

    # Show the frame with the text overlay
    cv2.imshow('Age and Gender Detection', frame)

    # Press 'q' to quit the webcam feed
    key = cv2.waitKey(7)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('s'):
        img_name = f"age_gender_prediction_{time.time()}.png"
        cv2.imwrite(img_name, frame)  # Save the frame
        print(f"Image saved as {img_name}")

# Release the capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
