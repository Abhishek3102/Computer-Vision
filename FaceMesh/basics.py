import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture("messi_zizou.mp4")
pTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)

while True:
    success, img = cap.read()
    if not success:
        print("Error: Unable to capture video frame.")
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            # Use FACEMESH_TESSELATION to draw the mesh
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_TESSELATION, drawSpec, drawSpec)
            for id, lm in enumerate(faceLms.landmark):
                ih, iw, ic = img.shape
                x, y = int(lm.x * iw), int(lm.y * ih)
                print(id, x, y)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow("Image", img)
    
    key = cv2.waitKey(10)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('s'):
        img_name = f"face_mesh_{time.time()}.png"
        cv2.imwrite(img_name, img)
        print(f"Image saved as {img_name}")

cap.release()
cv2.destroyAllWindows()
