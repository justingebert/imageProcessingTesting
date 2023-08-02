import cv2 as cv

img = cv.imread("images/tycan.jpg")
grey = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
blur = cv.GaussianBlur(img, (3, 3), 0)
canny = cv.Canny(blur, 125, 175)

dialate = cv.dilate(canny, (7, 7), iterations=1)
erode = cv.erode(dialate, (7, 7), iterations=1)

cv.imshow("TyCan Canny", dialate)
cv.waitKey(0)
