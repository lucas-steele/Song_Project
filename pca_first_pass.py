#imports
import os 
import cv2   
from sklearn.utils import shuffle
import numpy as np  
import matplotlib.pyplot as plt
import random
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score

#########

#Load the data
def get_images(directory):
    Images = []
    Labels = []  # 0 for Building , 1 for forest, 2 for glacier, 3 for mountain, 4 for Sea , 5 for Street
    class_labels = {'buildings': 0,
                    'forest' : 1,
                    'glacier' : 2,
                    'mountain' : 3,
                    'sea' : 4,
                    'street' : 5
                    }
    for label in os.listdir(directory):
        
        for image_file in os.listdir(directory+label): #Extracting the file name of the image from Class Label folder
            image = cv2.imread(directory+label+r'/'+image_file) #Reading the image (OpenCV)
            image = cv2.resize(image,(150,150)) #Resize the image, Some images are different sizes. (Resizing is very Important)
            Images.append(image)
            Labels.append(class_labels[label])
    
    return shuffle(Images,Labels,random_state=817328462) #Shuffle the dataset you just prepared.
	
	
#############

Images, Labels = get_images('./seg_train/') #Extract the training images from the folders.

X_train = np.array(Images) #converting the list of images to numpy array.
y_train = np.array(Labels)

X_train = X_train.reshape(14034,67500) #Need to flatten for PCA
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.30, random_state=42) 

#100 retains ~73% 200 retains ~77% too many components causes memory error
pca = PCA(n_components=100)
pca.fit(X_train)
sum(pca.explained_variance_ratio_)

X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

#Basic unoptomized MLP Classifier
mlp = MLPClassifier(alpha=7e-05,max_iter=500,random_state=42)
mlp.fit(X_train, y_train)
print("Training set score: %f" % mlp.score(X_train, y_train))

cvs=cross_val_score(mlp, X_train, y_train, cv=3, scoring="accuracy")
print('NN  cross validation ',cvs)


print("Test set score: %f" % mlp.score(X_test, y_test))
y_predict = mlp.predict(X_test)
conf_mx=confusion_matrix(y_test, y_predict)
print(conf_mx)

'''
Training set score: 0.883030
NN  cross validation  [0.51907232 0.50427611 0.5091687 ]
Test set score: 0.550226
[[276  49  60  61  71 139]
 [ 34 492   9  12  16  86]
 [ 46  15 419  98  96  47]
 [ 71  37 141 378 101  37]
 [ 90  29 126 117 280  44]
 [109  71  33  24  25 472]]
 '''