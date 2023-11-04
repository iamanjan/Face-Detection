import cv2
import numpy
# face detection model
harcascade = "model/haarcascade_frontalface_default.xml"

# smile detection model
#harcascade = "model/haarcascade_smile.xml"

cap = cv2.VideoCapture(0) #cap mean camera object

cap.set(3,640) # width
cap.set(4,480) # height

while True:
    success, img= cap.read()
    #load model
    facecascade =cv2.CascadeClassifier(harcascade)
    img_gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # giving  gray imag

    #to face detect
    faces=facecascade.detectMultiScale(img_gray,1.1,4)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.imshow('Face', img) # for original image use 'img' and for gray image use 'img_gray'

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break