import cv2
import tensorflow as tf
import numpy as np
from tensorflow import keras

facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')#replace with better classifer
model = keras.models.load_model('ACCURATEMODEL.hdf5')
font = cv2.FONT_HERSHEY_SIMPLEX


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    # returns camera frames along with bounding boxes and predictions
    def get_frame(self):
        _, fr = self.video.read()
        faces = facec.detectMultiScale(fr)

        for (x, y, w, h) in faces:
            fc = fr[y:y + h, x:x + w]
            roi = cv2.resize(fc, (224, 224))
            roi = tf.keras.applications.mobilenet.preprocess_input(roi)
            roi = np.expand_dims(roi, 0)
            pred = model.predict(roi)
            p = np.argmax(pred, 1)
            if p == 0:
                state = 'NOT DROWSY'
            if p == 1:
                state = "DROWSY"
            cv2.putText(fr, str(state), (x, y), font, 1, ((0,128,128)), 2)
            cv2.rectangle(fr, (x, y), (x + w, y + h), ((0,255,0)), 2)
        cv2.imshow('Stream', fr)
        cv2.waitKey(1)

vc = VideoCamera()
while True:
    vc.get_frame()
