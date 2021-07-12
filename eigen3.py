from time import ctime
import numpy, os
import matplotlib.pyplot as plt
import logging
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import cv2
import pickle
from collections import Counter
import ntplib
  
def most_frequent(List):
    occurence_count = Counter(List)
    return occurence_count.most_common(1)[0][0]

pickle_in = open("pca5b.dat","rb")
pca = pickle.load(pickle_in)

#X_train_pca = pca.transform(X_train)
#X_test_pca = pca.transform(X_test)

pickle_in = open("clf5b.dat","rb")
clf= pickle.load(pickle_in)

#nama = {0: 'alfi', 1: 'unknown'}
#nama = {0: 'alfi', 1: 'fatan',2:'unknown'}
#nama = {0: 'alfi', 1: 'falah',2:'fatan',3:'unknown'}
#nama = {0: 'alfi', 1: 'falah',2:'fatan',3:'harry',4:'unknown'}
nama = {0: 'alfi', 1: 'falah',2: 'fatan',3: 'harry',4: 'unknown',5:'windi'}




#proses face recogn
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap.set(3,640)
cap.set(4,480)

identitas=[]
ketemu = False
#360x440
while(True):
    # Capture frame-by-frame
    test = []
    face = []
    ret, frame = cap.read()
    xv, yv, cv = frame.shape
    if ret == True :
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        test = []
        for (x,y,wf,hf) in faces:
            # create box on detected face
            frame = cv2.rectangle(frame,(x,y),(x+wf,y+hf),(255,0,0),1)
            
            wajah = frame[y:y+hf,x:x+wf]
            dim = (320, 320)
  
            # resize image
            resized = cv2.resize(wajah, dim, interpolation = cv2.INTER_AREA)
            resized=cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            testImageFeatureVector=numpy.array(resized).flatten()
            test.append(testImageFeatureVector)
            testImagePCA = pca.transform(test)
            testImagePredict=clf.predict(testImagePCA)
            
            cv2.imshow('wframe', resized)
            print('ada wajah', nama[testImagePredict[0]], type(testImagePredict))
            cv2.putText(frame, "Name : " + nama[testImagePredict[0]], (x + x//10, y+hf+20), \
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            #print('.',end='')
            
            identitas.append(testImagePredict[0])
            if len(identitas) >= 30 :
                ketemu = True
                break
            
        cv2.imshow('frame', frame)
        
        if ketemu == True :
            break
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

#print (identitas)
id_ = most_frequent(identitas)
print('ada wajah', nama[id_])

#def log(id):
    #openfile(detik)
    #simpan
def print_time():
    ntp_client = ntplib.NTPClient()
    response = ntp_client.request('asia.pool.ntp.org')
    print(ctime(response.tx_time))
print_time()
#if (id== 0,1,2,3):
    #buka_kunci()
    #log(id_)