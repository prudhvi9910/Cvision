from HandTrackor import HandDetector
import cv2
import numpy as np
import time
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

cWidth, cHeight = 640, 480

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
vol_range = volume.GetVolumeRange()
min_vol = vol_range[0]
max_vol = vol_range[1]

hd = HandDetector(min_detect_conf=0.7)


def main():
    c_time = 0
    p_time = 0
    cap = cv2.VideoCapture(0)
    cap.set(3, cWidth)
    cap.set(4, cHeight)

    while True:
        success, img = cap.read()
        img = hd.findHands(img, draw=False)
        lm_list = hd.findPosition(img, draw=False)

        if lm_list:
            x1, y1 = lm_list[4][1], lm_list[4][2]
            x2, y2 = lm_list[8][1], lm_list[8][2]
            # mid_x, mid_y = (x1+y1)//2, (x2+y2)//2
            cv2.circle(img, (x1, y1), 7, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 7, (255, 0, 0), cv2.FILLED)
            # cv2.circle(img, (mid_x, mid_y), 7, (255, 0, 0), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

            length = math.hypot(x2 - x1, y2 - y1)
            vol = np.interp(length, [30, 270], [-65, 0])
            vol_bar = np.interp(length, [30, 270], [400, 70])
            vol_perc = np.interp(length, [30, 270], [0, 100])
            volume.SetMasterVolumeLevel(vol, None)

            cv2.rectangle(img, (60, 70), (80, 400), (0, 255, 0), 2)
            cv2.rectangle(img, (60, int(vol_bar)), (80, 400), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, f'Volume: {int(vol_perc)} %', (20, 430), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        c_time = time.time()
        fps = 1/(c_time-p_time)
        p_time = c_time
        # img = cv2.flip(img, 1)
        cv2.putText(img, f'FPS: {str(int(fps))}', (30, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        cv2.imshow("Volume Controller", img)
        if cv2.waitKey(1) and 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
