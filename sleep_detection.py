import cv2
import os
from tensorflow.python.keras.models import load_model
import numpy as np
from pygame import mixer
from tensorflow.python.keras.engine import data_adapter

def _is_distributed_dataset(ds):
    return isinstance(ds, data_adapter.input_lib.DistributedDatasetSpec)

data_adapter._is_distributed_dataset = _is_distributed_dataset
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def sleep(haar_cascade_path, blink_threshold, cam_id):
    mixer.init()
    sound = mixer.Sound('alarm.wav')

    leye = cv2.CascadeClassifier(os.path.join(haar_cascade_path,'haarcascade_lefteye_2splits.xml'))
    reye = cv2.CascadeClassifier(os.path.join(haar_cascade_path,'haarcascade_righteye_2splits.xml'))

    model = load_model('models/main_model.h5')
    cap = cv2.VideoCapture(cam_id)
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    score=0
    thicc=2
    rpred=[99]
    lpred=[99]

    while(True):
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Error: Failed to read frame from camera")
            break
        height,width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        left_eye = leye.detectMultiScale(gray)
        right_eye =  reye.detectMultiScale(gray)

        if(type(right_eye)==tuple):
            right_eye_status = 0
        else:
            for (x,y,w,h) in right_eye:
                r_eye=frame[y:y+h,x:x+w]
                r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
                r_eye = cv2.resize(r_eye,(24,24))
                r_eye= r_eye/255
                r_eye=  r_eye.reshape(24,24,-1)
                r_eye = np.expand_dims(r_eye,axis=0)
                rpred = model.predict(r_eye)
                right_eye_status = np.argmax(rpred,axis=1)
                break

        if(type(left_eye)==tuple):
            left_eye_status = 0
        else:
            for (x,y,w,h) in left_eye:
                l_eye=frame[y:y+h,x:x+w]
                l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)
                l_eye = cv2.resize(l_eye,(24,24))
                l_eye= l_eye/255
                l_eye=l_eye.reshape(24,24,-1)
                l_eye = np.expand_dims(l_eye,axis=0)
                lpred = model.predict(l_eye)
                left_eye_status = np.argmax(lpred,axis=1)
                break

        if(right_eye_status==0 and left_eye_status==0):
            score=score+1
            cv2.putText(frame,"Closed",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
        else:
            score=score-1
            cv2.putText(frame,"Open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
        
        if(score<0):
            score=0   
        cv2.putText(frame,'Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
        if(score>blink_threshold):
            try:
                sound.play()
            except:  # isplaying = False
                pass
            if(thicc<16):
                thicc= thicc+2
            else:
                thicc=thicc-2
                if(thicc<2):
                    thicc=2
            cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc) 
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break 

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    haar_cascade_path = './haar_cascade_files/'
    blink_threshold = 10
    cam_id = 0  # Changed from -1 to 0
    sleep(haar_cascade_path, blink_threshold, cam_id)