import os
import cv2
import mediapipe as mp
import time

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        # Initialize the MediaPipe hands module
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mp_draw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        # Convert the image color space from BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
        # Process the image and find hands
        results = self.hands.process(img_rgb)
    
        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        return img

def main():
    pTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    # Initialize a counter for saving multiple images
    img_counter = 0

    while True:
        success, img = cap.read()
        if not success:
            print("Error: Failed to capture image.")
            break

        img = detector.findHands(img)

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)

        # Press 'q' to exit the loop
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

        # Press 's' to save the image
        elif key & 0xFF == ord('s'):
            img_name = f"captured_image_{img_counter}.png"
            cv2.imwrite(img_name, img)
            print(f"Image saved as {img_name}")
            img_counter += 1

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
