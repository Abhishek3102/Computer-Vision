import cv2
import time
import numpy as np
import PoseModule as pm

net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

cap = cv2.VideoCapture(0)
detector = pm.poseDetector()

count = 0
dir = 0
pTime = 0
save_images = False

def detect_dumbbell_in_hand(img):
    height, width, _ = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  
                if classes[class_id] == 'sports ball' or classes[class_id] == 'person':  
                    return True  
    return False 

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.resize(img, (1280, 720))
    img = detector.findPose(img, False)
    lmList = detector.findPosition(img, False)

    dumbbell_in_hand = detect_dumbbell_in_hand(img)  

    if len(lmList) != 0:
        angle = detector.findAngle(img, 12, 14, 16)
        per = np.interp(angle, (210, 310), (0, 100))
        bar = np.interp(angle, (220, 310), (650, 100))

        color = (255, 0, 255)
        if per == 100 and dumbbell_in_hand:  
            color = (0, 255, 0)
            if dir == 0:
                count += 0.5
                dir = 1
        if per == 0 and dumbbell_in_hand:  
            color = (0, 255, 0)
            if dir == 1:
                count += 0.5
                dir = 0

        cv2.rectangle(img, (1100, 100), (1175, 650), color, 3)
        cv2.rectangle(img, (1100, int(bar)), (1175, 650), color, cv2.FILLED)
        cv2.putText(img, f'{int(per)} %', (1100, 75), cv2.FONT_HERSHEY_PLAIN, 4, color, 4)

        cv2.rectangle(img, (0, 450), (250, 720), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(int(count)), (45, 670), cv2.FONT_HERSHEY_PLAIN, 15, (255, 0, 0), 25)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (50, 100), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)

    cv2.imshow("Image", img)

    key = cv2.waitKey(1)

    if key & 0xFF == ord('q'):  
        break
    elif key & 0xFF == ord('s'):  
        img_name = f"curl_count_{int(count)}.png"
        cv2.imwrite(img_name, img)
        print(f"Saved image: {img_name}")

cap.release()
cv2.destroyAllWindows()
