# import cv2
# import time
# import os
# import HandTrackingModule as htm
# wCam, hCam = 640, 480
# cap = cv2.VideoCapture(1)
# cap.set(3, wCam)
# cap.set(4, hCam)
# folderPath = "FingerImages"
# myList = os.listdir(folderPath)
# print(myList)
# overlayList = []
# for imPath in myList:
#     image = cv2.imread(f'{folderPath}/{imPath}')
#     # print(f'{folderPath}/{imPath}')
#     overlayList.append(image)
# print(len(overlayList))
# pTime = 0
# detector = htm.handDetector(detectionCon=0.75)
# tipIds = [4, 8, 12, 16, 20]
# while True:
#     success, img = cap.read()
#     img = detector.findHands(img)
#     lmList = detector.findPosition(img, draw=False)
#     # print(lmList)
#     if len(lmList) != 0:
#         fingers = []
#         # Thumb
#         if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
#             fingers.append(1)
#         else:
#             fingers.append(0)
#         # 4 Fingers
#         for id in range(1, 5):
#             if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
#                 fingers.append(1)
#             else:
#                 fingers.append(0)
#         # print(fingers)
#         totalFingers = fingers.count(1)
#         print(totalFingers)
#         h, w, c = overlayList[totalFingers - 1].shape
#         img[0:h, 0:w] = overlayList[totalFingers - 1]
#         cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
#         cv2.putText(img, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN,
#                     10, (255, 0, 0), 25)
#     cTime = time.time()
#     fps = 1 / (cTime - pTime)
#     pTime = cTime
#     cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN,
#                 3, (255, 0, 0), 3)
#     cv2.imshow("Image", img)
#     cv2.waitKey(1)

import cv2
import time
import math
import HandTrackingModule as htm

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

pTime = 0
detector = htm.handDetector(detectionCon=0.7)

while True:
    success, img = cap.read()
    if not success:
        break

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        fingers = []
        
        if lmList[4][1] < lmList[3][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(8, 21, 4):
            if lmList[id][2] < lmList[id - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        totalFingers = fingers.count(1)
        
        cv2.putText(img, f'Fingers: {totalFingers}', (40, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

        if totalFingers > 0: 
            cv2.putText(img, "Press 's' to save image", (40, 150), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

    cv2.imshow("Img", img)
    
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('s'):
        img_name = f"finger_count_{time.strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(img_name, img)
        print(f"Image saved as: {img_name}")

cap.release()
cv2.destroyAllWindows()
