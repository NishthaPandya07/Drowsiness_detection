from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
import os
from pygame import mixer
import tempfile


def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = abs(top_mean[1] - low_mean[1])
    return distance

def yawn(haar_cascade_path, yawn_threshold, cam_id):

    detector = cv2.CascadeClassifier(os.path.join(haar_cascade_path,"haarcascade_frontalface_default.xml"))
    predictor = dlib.shape_predictor(os.path.join(haar_cascade_path,'shape_predictor_68_face_landmarks.dat'))
    cap = cv2.VideoCapture(cam_id)

    mixer.init()
    sound = mixer.Sound('alarm.wav')

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Error: Failed to read frame from camera")
            break
        frame = imutils.resize(frame, width=450)
        height,width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Ensure grayscale is uint8 and contiguous for dlib compatibility
        gray = np.ascontiguousarray(gray, dtype=np.uint8)

        rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
        minNeighbors=5, minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE)

        for (x, y, w, h) in rects:
            # Create dlib rectangle  
            rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
            
            # Pass numpy array directly to dlib predictor
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            distance = lip_distance(shape)

            lip = shape[48:60]
            cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

            if (distance > yawn_threshold):
                try:
                    sound.play()
                except:
                    pass
                cv2.putText(frame, "Yawn Alert", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            distance = str(distance)
            cv2.putText(frame,'distance:'+distance[0:5],(10,height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,0,255),1,cv2.LINE_AA)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break 

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    haar_cascade_path = './haar_cascade_files/'
    yawn_threshold = 15
    cam_id = 0  # Changed from -1 to 0
    yawn(haar_cascade_path, yawn_threshold,cam_id)