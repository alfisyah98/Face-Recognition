import glob
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

tujuan = glob.glob('./sampel foto/harry/*.jpg')

count = 0
for lok_gambar in tujuan:
    ini = cv2.imread(lok_gambar)
    gray = cv2.cvtColor(ini, cv2.COLOR_BGR2GRAY)
        
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,wf,hf) in faces:
        # create box on detected face
        roi = ini[y:y+hf, x:x+wf]
        cv2.imwrite('./dataset6full/harry2/'+str(count)+'.jpg', roi)
    count += 1            
    
print(tujuan)