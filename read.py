import cv2 as cv

img = cv.imread("images/tycan.jpg")

def resize(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

cv.imshow("TyCan", img)
cv.waitKey(0)