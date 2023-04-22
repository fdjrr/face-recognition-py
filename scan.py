import cv2
import os
import time
from PIL import Image

camera = 0
face_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
video_capture = cv2.VideoCapture(camera, cv2.CAP_DSHOW)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('data/training.xml')

while True:
	_, frame = video_capture.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	face = face_detect.detectMultiScale(gray, 1.3, 5)
	for (x, y, w, h) in face:
		cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 255, 0), 2)
		id, conf = recognizer.predict(gray[y:y+h, x:x+w])
		if (id == 1):
			person = 'Fadjrir'
		elif (id == 2):
			person = 'Krisdianto'
		cv2.putText(frame, person, (x+40, y-10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0))

	cv2.imshow('Capturing', frame)
	cv2.waitKey(1)

video_capture.release()
cv2.destroyAllWindows()
