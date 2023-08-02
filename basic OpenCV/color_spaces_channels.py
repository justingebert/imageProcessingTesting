import cv2 as cv
import numpy as np

img = cv.imread("../images/tycan.jpg")
r, g, b = cv.split(img)
blank = np.zeros(img.shape[:2], dtype="uint8")

blue = cv.merge([b, blank, blank])
green = cv.merge([blank, g, blank])
red = cv.merge([blank, blank, r])

cv.imshow("TyCan", img)
cv.imshow("TyCan Blue", blue)
cv.imshow("TyCan Green", green)
cv.imshow("TyCan Red", red)

cv.waitKey(0)