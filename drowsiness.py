from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import pygame
import imutils
import time
import dlib
import cv2

def sound_alarm(path):
    pygame.mixer.init()
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.04 * C)
    return ear

def detect_drowsiness():
    shape_predictor_path = r"C:\Users\Akash\Documents\project\V_Project\shape_predictor_68_face_landmarks.dat"
    alarm_sound_path = r"C:\Users\Akash\Documents\project\V_Project\wake_up.mp3"
    webcam_index = 0

    EYE_AR_THRESH = 0.3
    EYE_AR_CONSEC_FRAMES = 48
    COUNTER = 0
    ALARM_ON = False
    LAST_DETECTION = time.time()

    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor_path)
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    print("[INFO] starting video stream thread...")
    vs = VideoStream(src=webcam_index).start()
    time.sleep(1.0)
    
    pygame.mixer.init()
    
    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        drowsy = False  # Initialize as False for non-drowsy state

        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            if ear < EYE_AR_THRESH:
                COUNTER += 1
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    drowsy = True
                    if not ALARM_ON:
                        ALARM_ON = True
                        if alarm_sound_path != "":
                            t = Thread(target=sound_alarm, args=(alarm_sound_path,))
                            t.daemon = True
                            t.start()
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    LAST_DETECTION = time.time()
            else:
                COUNTER = 0
                if ALARM_ON:
                    # Check if enough time has passed since the last detection
                    if time.time() - LAST_DETECTION > 3:  # Adjust this threshold as needed
                        ALARM_ON = False
                        pygame.mixer.music.stop()
                cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Draw face recognition box based on drowsiness state
            if drowsy:
                cv2.rectangle(frame, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0, 0, 255), 2)  # Red box for drowsy
            else:
                cv2.rectangle(frame, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0, 255, 0), 2)  # Green box for non-drowsy

        if not drowsy and ALARM_ON:  # If no longer drowsy but alarm is still on
            ALARM_ON = False
            pygame.mixer.music.stop()

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    pygame.mixer.music.stop()
    cv2.destroyAllWindows()
    vs.stop()

if __name__ == "__main__":
    detect_drowsiness()
