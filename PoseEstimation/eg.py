import cv2
import time
import PoseModule as pm
cap = cv2.VideoCapture('videoplayback.mp4')
pTime = 0
detector = pm.poseDetector()
while True:
    success, img = cap.read()
    img = detector.findPose(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) !=0:
        # According to image of CV Body points, 28 is the position for right ankle.
        print(lmList[28])
        cv2.circle(img, (lmList[28][1], lmList[28][2]), 15, (0, 0, 255), cv2.FILLED)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
cap.release()
cv2.destroyAllWindows()