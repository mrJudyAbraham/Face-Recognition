import cv2
from cv2 import IMWRITE_PNG_BILEVEL
import numpy as np
import face_recognition

imgElon = face_recognition.load_image_file('/home/madj/Documents/Github/Face-Recognition/musk.jpg')
imgElon = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)
img = face_recognition.load_image_file('/home/madj/Documents/Github/Face-Recognition/musk2.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
imgbill = face_recognition.load_image_file('/home/madj/Documents/Github/Face-Recognition/billgates.jpg')
imgbill = cv2.cvtColor(imgbill,cv2.COLOR_BGR2RGB)

faceLoc=face_recognition.face_locations(imgElon)[0]
encodeElon=face_recognition.face_encodings(imgElon)[0]
faceLoc1=face_recognition.face_locations(img)[0]
encodeImg=face_recognition.face_encodings(img)[0]
faceLoc2=face_recognition.face_locations(imgbill)[0]
encodeBill=face_recognition.face_encodings(imgbill)[0]

cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)
cv2.rectangle(img,(faceLoc1[3],faceLoc1[0]),(faceLoc1[1],faceLoc1[2]),(0,255,0),2)
cv2.rectangle(imgbill,(faceLoc2[3],faceLoc2[0]),(faceLoc2[1],faceLoc2[2]),(255,0,0),2)
  
results = face_recognition.compare_faces([encodeElon],encodeImg)

print(type(results))

name="Elon Musk"
if results[0]==True:
    cv2.imshow('Identified as '+name, img)
    cv2.waitKey(0)



#cv2.imshow('Elon Musk',imgElon)
#cv2.imshow('Elon ',img)
#cv2.imshow('Bill',imgbill)
#cv2.waitKey(0)

