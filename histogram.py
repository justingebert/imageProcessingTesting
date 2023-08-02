import cv2 as cv

img = cv.imread("images/tycan.jpg")
grey = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

hist = cv.calcHist([grey], [0], None, [256], [0, 256])


cv.imshow("TyCan Canny", dialate)
cv.waitKey(0)
