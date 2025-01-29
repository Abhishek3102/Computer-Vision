import cv2
import numpy as np

cloth = cv2.imread("tshirt.png")
gray = cv2.cvtColor(cloth, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
mask = 255 - mask
cv2.imwrite("cloth_mask.jpg", mask)
cv2.imshow("Cloth Mask", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
