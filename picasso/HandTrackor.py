import time

import cv2
import numpy as np
import mediapipe as mp


class HandDetector:
    """
    Detects the hands in the live camera
    """
    def __init__(self, mode=False, max_hands=2, min_detect_conf=0.5, min_track_conf=0.5):
        """
        init module
        :param mode:Whether to treat the input images as a batch of static
                    and possibly unrelated images, or a video stream
        :param max_hands: Max number of hands to be tracked
        :param min_detect_conf: Minimum Detection Confidence
        :param min_track_conf: Minimum Tracking Confidence
        """
        self.mode = mode
        self.max_hands = max_hands
        self.min_track_conf = min_track_conf
        self.min_detect_conf = min_detect_conf

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.max_hands, self.min_detect_conf, self.min_track_conf)
        self.mpDraw = mp.solutions.drawing_utils
        self.finger_tid_ids = [4, 8, 12, 16, 20]
        self.lm_list = []

    def findHands(self, img, draw=True):
        """
        Detects the hands and tracks it
        :param img: Image of the hand
        :param draw: if true, show the hand points and draws the connections to all the hand points
        :return: img with hand detected
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

    def fingersUp(self):
        """

        :return:
        """
        fingers = []

        # For thumb
        if self.lm_list[self.finger_tid_ids[0]][1] < self.lm_list[self.finger_tid_ids[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # For remaining four fingers
        for id in range(1, 5):
            if self.lm_list[self.finger_tid_ids[id]][2] < self.lm_list[self.finger_tid_ids[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

    def findPosition(self, img, hand_no=0, draw=True):
        """
        Finds the position of all the hand points
        :param img: img of detected hand
        :param hand_no: Index of the detected hands
        :param draw: if true, draws circles on the detected hand points
        :return: list of positions of all the 21 hand points with their corresponding ids
        """
        self.lm_list = []
        if self.results.multi_hand_landmarks:
            for id, lm in enumerate(self.results.multi_hand_landmarks[hand_no].landmark):
                height, width, color = img.shape
                x_coord, y_coord = int(lm.x * width), int(lm.y * height)
                self.lm_list.append([id, x_coord, y_coord])
                if draw:
                    cv2.circle(img, (x_coord, y_coord), 7, (255, 0, 0), cv2.FILLED)
        return self.lm_list


def main():
    """

    :return:
    """
    c_time = 0
    p_time = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lm_list = detector.findPosition(img, draw=False)

        c_time = time.time()
        fps = 1/(c_time-p_time)
        p_time = c_time

        cv2.putText(img, str(int(fps)), (10,40), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("Picasso", img)
        if cv2.waitKey(1) and 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()
