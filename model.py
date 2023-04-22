import cv2

camera = 0
face_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
video_capture = cv2.VideoCapture(camera, cv2.CAP_DSHOW)
count_capture = 0
id = input('User ID : ')

while True:
	_, frame = video_capture.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	face = face_detect.detectMultiScale(gray, 1.3, 5)
	for (x, y, w, h) in face:
		cv2.imwrite(f'data/User.{id}.{count_capture}.jpg', gray[y:y+h, x:x+w])
		cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 255, 0), 2)
		cv2.putText(frame, 'Person', (x+40, y-10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0))
		count_capture += 1

	cv2.imshow('Capturing', frame)
	cv2.waitKey(1)
	if (count_capture > 100):
		break

video_capture.release()
cv2.destroyAllWindows()
