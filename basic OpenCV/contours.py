import cv2 as cv
import numpy as np

img = cv.imread("../images/tycan.jpg")
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blank = np.zeros((img.shape[0],img.shape[1], 3), dtype="uint8")

blur = cv.GaussianBlur(img_gray, (3, 3), cv.BORDER_DEFAULT)
canny = cv.Canny(blur, 150, 175)

ret, thresh = cv.threshold(img_gray, 125, 255, cv.THRESH_BINARY)

contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
print(f"{len(contours)} contour(s) found!")

cv.drawContours(blank, contours, -1, (0, 0, 255), 1)

cv.imshow("TyCan Canny", blank)
cv.waitKey(0)