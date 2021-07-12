from time import time
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

t0= time()
#mengambil dataset dari direktori komputer
path="D:\\PERKULIAHAN\\TA VERSI NEW\\data training\\dataset3a\\"

#inisialisasi image dengan beberapa library
#Flat image Feature Vector
X=[]
#Int array of Label Vector
Y=[]

#memotong frame dataset untuk mengambil wajah
data_slice = [0,800,0,800] # [ ymin, ymax, xmin, xmax]

# resize ratio to reduce sample dimention
resize_ratio = 2.5

h = int((data_slice[1] - data_slice[0])/resize_ratio) #ymax - ymin slice, Height of image in float
w = int((data_slice[3] - data_slice[2])/resize_ratio) #xmax - xmin slice, Width of image in float 
print("Image dimension after resize (h,w) :", h, w)

n_sample = 0 #Initial sample count
label_count = 0 #Initial label count
n_classes = 0 #Initial class count

#PCA Component 
n_components = 50

#####konversi gambar ke grayscale dengan memperhatikan height dan weight.. dan mengklasifikasikan dataset

target_names = [] #Array to store the names of the persons

for directory in os.listdir(path):
    for file in os.listdir(path+directory):
        img=cv2.imread(path+directory+"/"+file)[data_slice[0]:data_slice[1], data_slice[2]:data_slice[3]]
        img=cv2.resize(img, (w,h))
        img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        featurevector=numpy.array(img).flatten()
        X.append(featurevector)
        Y.append(label_count)
        n_sample = n_sample + 1
    target_names.append(directory)
    label_count=label_count+1


print("Samples :", n_sample)
print("Class :", target_names)
n_classes = len(target_names)

###############################################################################
# Split into a training set and a test set using a stratified k fold
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.25, random_state=42)
# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction

pca = PCA(n_components=n_components, whiten=True).fit(X_train)
eigenfaces = pca.components_.reshape((n_components, h, w))

#pickle_in = open("pca6full.dat","rb")
#pca = pickle.load(pickle_in)

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

########################################################################
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf = clf.fit(X_train_pca, y_train)
#pickle_in = open("clf6full.dat","rb")
#clf= pickle.load(pickle_in)
y_pred = clf.predict(X_test_pca)
#confusion_matrix(y_test,y_pred)
print("done in %0.3fs" % (time() - t0))