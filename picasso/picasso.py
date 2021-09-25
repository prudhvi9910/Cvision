import cv2
import time
import numpy as np
import os
from HandTrackor import HandDetector

cWidth, cHeight = 1280, 720
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, cWidth)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cHeight)
detector = HandDetector(min_detect_conf=0.85)
drawcolor = ()
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

side_bars = os.listdir("Resources")
print(side_bars)
side_overlay = []

for side_bar in side_bars:
    image = cv2.imread(f"Resources/{side_bar}")
    side_overlay.append(image)
sider = side_overlay[0]

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = detector.findHands(frame)
    h, w, c = sider.shape
    frame[0:h, 0:w] = sider
    lm_list = detector.findPosition(frame, draw=False)
    if lm_list:
        x1, y1 = lm_list[8][1:]
        x2, y2 = lm_list[12][1:]
        fingers_status = detector.fingersUp()
        total_fing_up = sum(fingers_status)
        if total_fing_up == 2:
            xp = yp = 0
            if y1 < 121:
                if 235<x1<470:
                    sider = side_overlay[1]
                    drawcolor = (71, 99, 255)
                elif 470<x1<720:
                    sider = side_overlay[2]
                    drawcolor = (239, 255, 0)
                elif 720<x1<965:
                    sider = side_overlay[3]
                    drawcolor = (0, 255, 0)
                elif 0<x1<235:
                    sider = side_overlay[0]
                elif 965<x1<1200:
                    sider = side_overlay[4]
                    drawcolor = (0, 0, 0)
        elif total_fing_up == 1:
            cv2.circle(frame, (x1, y1), 10, drawcolor, cv2.FILLED)
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
            if drawcolor == (0, 0, 0):
                cv2.line(frame, (xp, yp), (x1, y1), drawcolor, 50)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawcolor, 50)
            cv2.line(frame, (xp,yp), (x1, y1), drawcolor, 15)
            cv2.line(imgCanvas, (xp, yp), (x1, y1), drawcolor, 15)
            xp, yp = x1, y1

    grayframe = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, invframe = cv2.threshold(grayframe, 50, 255, cv2.THRESH_BINARY_INV)
    invframe = cv2.cvtColor(invframe, cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame,invframe)
    frame = cv2.bitwise_or(frame, imgCanvas)

    frame[0:h, 0:w] = sider
    cv2.imshow("Picasso", frame)
    if cv2.waitKey(1) and 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
