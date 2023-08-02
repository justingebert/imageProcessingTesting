import cv2 as cv
import numpy as np

img = cv.imread("../images/tycan.jpg")
blank = np.zeros(img.shape[:2], dtype="uint8")




cv.imshow("TyCan", img)
cv.waitKey(0)
