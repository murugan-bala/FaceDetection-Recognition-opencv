#######################################
# Name : Murugan
# Date : 04-03-2019
#######################################

import cv2
import numpy as np
face_detect= cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

rec=cv2.face.LBPHFaceRecognizer_create()
cam=cv2.VideoCapture(0)
rec.read("D:\\KRISH\\FAce_training_recognition_opencvAlgorithm\\recognize\\trained.yml")
names={"1":"Murugan","2":"Suganniya",}
Id=0
font=cv2.FONT_HERSHEY_SIMPLEX
while 1:  
  
    # reads frames from a camera 
    ret, img = cam.read()  
  
    # convert to gray scale of each frames 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
  
    # Detects faces of different sizes in the input image 
    faces = face_detect.detectMultiScale(gray, 1.3, 5) 
  
    for (x,y,w,h) in faces:
        
        # To draw a rectangle in a face  
        cv2.rectangle(img,(x,y),(x+w,y+h+30),(0,0,255),2)
        Id,conf=rec.predict(gray[y:y+h,x:x+w])
        print("ID is : "+str(Id),"Confidence : "+str(conf))
        cv2.putText(img,names[str(Id)],(x,y+h),font,0.90,(0,0,255),2)
        #cv2.putText(img,str(Id),x,y+h,font,0.90,(0,0,255),2)  
    # Display an image in a window 
    cv2.imshow('Faces ',img) 
    
    if cv2.waitKey(1) ==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
      
        
 

  
