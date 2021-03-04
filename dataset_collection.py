#######################################
# Name : Murugan
# Date : 04-03-2019
#######################################

import cv2
face_detect= cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

Id=input("Enter your ID : ")
name=input("Enter your Name : ")
gender=input("Enter your Gender M/F : ")
print("Dataset collecting..........................")
cam=cv2.VideoCapture(0)
sampleNumber=0
while 1:  
  
    # reads frames from a camera 
    ret, img = cam.read()  
  
    # convert to gray scale of each frames 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
  
    # Detects faces of different sizes in the input image 
    faces = face_detect.detectMultiScale(gray, 1.3, 5) 
  
    for (x,y,w,h) in faces:
        sampleNumber +=1
        # To draw a rectangle in a face  
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)  
        cv2.imwrite("D:\\KRISH\\FAce_training_recognition_opencvAlgorithm\\Datasets\\user."+str(Id)+"."+str(sampleNumber)+".jpg",gray[y:y+h,x:x+w])
  
    # Display an image in a window 
    cv2.imshow('Faces ',img) 
    cv2.waitKey(100)
    if sampleNumber ==20:
        cam.release()
        cv2.destroyAllWindows()
        break
print("...................Completed..........................")        
 

  
