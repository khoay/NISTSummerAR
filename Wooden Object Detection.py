import cv2
import numpy as np

path = 'Resources/wooden block.jpg'

myColors = [[0, 33, 41, 255, 69, 106],
           [0, 41, 69, 33, 255, 106]]


def findColor(img, myColors):
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for color in myColors:
        lower = np.array(color[0:3])
        upper = np.array(color[3:6])
        mask = cv2.inRange(imgHSV, lower, upper)
        imgContour = getContours(mask)
        cv2.imshow("Mask " + str(color[0]), Resize(mask))
        #cv2.imshow("countrou", Resize(imgContour))


def getContours(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    x, y, w, h = 0, 0, 0, 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            cv2.drawContours(imgResult, cnt, -1, (255, 0, 1), 3)
            peri = cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,0.02 * peri,True)
            x, y, w, h = cv2.boundingRect(approx)
        return x + w // 2, y

def Resize(img):
    scale_percentage = 19
    width = int(img.shape[1] * scale_percentage / 100)
    height = int(img.shape[0] * scale_percentage / 100)
    # print("width =", width)
    # print("height = ", height)
    imgResize = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    return imgResize


while True:
    img = cv2.imread(path)
    imgResult = img.copy()
    findColor(img, myColors)

    # imgHSVResize = cv2.resize(imgHSV, (width, height), interpolation=cv2.INTER_AREA)

    cv2.imshow("Result", Resize(imgResult))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break