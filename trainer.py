#######################################
# Name : Murugan
# Date : 04-03-2019
#######################################

import cv2
import numpy as np
import os
from PIL import Image
recognizer = cv2.face.LBPHFaceRecognizer_create()

path="D:\\KRISH\\FAce_training_recognition_opencvAlgorithm\\Datasets"
    
def getImageswithId(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    faces=[]
    IDs=[]

    for imagepath in imagePaths:
        faceImg=Image.open(imagepath).convert('L')
        faceNp=np.array(faceImg,dtype=np.uint8)
        faces.append(faceNp)
        Id=int(os.path.split(imagepath)[-1].split('.')[1])
        print(Id)
        IDs.append(Id)
        cv2.imshow("Training",faceNp)
        cv2.waitKey(10)

    return np.array(IDs),faces
print("Training starts..................")
Ids,faces=getImageswithId(path)
recognizer.train(faces,Ids)
print("Training Ends....................")
print()
recognizer.save("D:\\KRISH\\FAce_training_recognition_opencvAlgorithm\\recognize\\trained.yml")
print("Model saved sucessfully ....................")
cv2.destroyAllWindows()

