from unittest import result
import cv2
from cv2 import IMWRITE_PNG_BILEVEL
import numpy as np
import face_recognition
import os

def decode(path):
    img = face_recognition.load_image_file(path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    faceLoc = face_recognition.face_locations(img)[0]
    faceEnc = face_recognition.face_encodings(img)[0]
    return faceEnc,faceLoc,img

def compare(known,unknown):
    knownEnc=known[0]
    unknownEnc=unknown[0]
 #  print(knownEnc)
 #  print(unknownEnc)
    result=face_recognition.compare_faces([knownEnc],unknownEnc)
    distance=face_recognition.face_distance([knownEnc],unknownEnc)
    return result,distance

def draw_rect(data):
    img=data[2]
    faceLoc=data[1]
    cv2.rectangle(img,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(0,255,0),2)
    cv2.imshow('Image',img)
    cv2.waitKey(0)


data_path="/home/madj/Documents/Github/Face-Recognition/data/"
identify_path="/home/madj/Documents/Github/Face-Recognition/identify/"
files_unknown=os.listdir(identify_path)
files_known=os.listdir(data_path)


known_data = []
unknown_data = []
for i in files_known:
    path=data_path+i
    known_data.append(decode(path))

for j in files_unknown:
    path=identify_path+j
    unknown_data.append(decode(path))

#print(known_data)
#print(unknown_data)

for i in known_data:
    for j in unknown_data:
        output=compare(i,j)
        print(output)

'''
cv2.rectangle(img,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(0,255,0),2)
cv2.imshow('Elon ',img)
cv2.waitKey(0)
'''

