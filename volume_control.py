import cv2
import mediapipe as mp
import numpy as np
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# ------------------ Audio Setup (Windows) ------------------
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None
)
volume = cast(interface, POINTER(IAudioEndpointVolume))
minVol, maxVol = volume.GetVolumeRange()[:2]

# ------------------ MediaPipe Setup ------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            lm = hand.landmark

            # Thumb tip (4) & Index finger tip (8)
            x1, y1 = int(lm[4].x * frame.shape[1]), int(lm[4].y * frame.shape[0])
            x2, y2 = int(lm[8].x * frame.shape[1]), int(lm[8].y * frame.shape[0])

            # Draw
            cv2.circle(frame, (x1, y1), 8, (255, 0, 0), -1)
            cv2.circle(frame, (x2, y2), 8, (255, 0, 0), -1)
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)

            # Distance
            length = math.hypot(x2 - x1, y2 - y1)

            # Map distance to volume
            vol = np.interp(length, [30, 200], [minVol, maxVol])
            volume.SetMasterVolumeLevel(vol, None)

            # Volume bar
            volBar = np.interp(length, [30, 200], [400, 150])
            volPercent = np.interp(length, [30, 200], [0, 100])

            cv2.rectangle(frame, (50, 150), (85, 400), (0, 255, 0), 3)
            cv2.rectangle(frame, (50, int(volBar)), (85, 400), (0, 255, 0), -1)
            cv2.putText(
                frame,
                f'{int(volPercent)} %',
                (40, 450),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Gesture Volume Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
