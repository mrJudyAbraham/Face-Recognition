import cv2
import numpy
import face_recognition
import os
import shutil

data_path="/home/madj/Documents/Github/Face-Recognition/data/"
identify_path="/home/madj/Documents/Github/Face-Recognition/identify/"
files_unknown=os.listdir(identify_path)
files_known=os.listdir(data_path)

for i in files_known:
    known_path=data_path+i
    img_known=face_recognition.load_image_file(known_path)
    img_known=cv2.cvtColor(img_known,cv2.COLOR_BGR2RGB)
    known_enc=face_recognition.face_encodings(img_known)[0]
    fldr=i.split('.')[0]
    path=os.path.join(identify_path,fldr)
    os.mkdir(path)
    for j in files_unknown:
        unknown_path=identify_path+j
        img_unknown=face_recognition.load_image_file(unknown_path)
        img_unknown=cv2.cvtColor(img_unknown,cv2.COLOR_BGR2RGB)
        unknown_enc=face_recognition.face_encodings(img_unknown)[0]

        if(face_recognition.compare_faces([known_enc],unknown_enc))[0]==True:
            shutil.copy(unknown_path,path)
        