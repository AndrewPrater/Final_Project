#Final Project
import numpy as np
import cv2  as cv #will need to download the openCV libary into your andaconda enviroment
import matplotlib.pyplot as plt #version 3.3.4

import scipy
import numpy as np
import scipy.io as spio
import sklearn.preprocessing

import math 
from sklearn.model_selection import KFold
import numpy.random as random

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics.cluster import rand_score
from numpy import mean
from numpy import std

print(cv.__version__) #Version I have installed is 3.4.2 -This is the version SURF works on

#Make sure the images are downloaded to the folder this file is in
#Use imread to change picture into an array
#the first parameter must be the path to the img
# The second parameter of this function determines if the img is read in color or grayscale

##############################################################################################################################
#FEATURE EXTRACTION
img = cv.imread(r'C:\Users\Andy Prater\ML\Project_03\img.jpg',0)

#print the dimensions of the image
print("Image Dimensions")
print(img.shape)

# Create SURF object. You can specify params here or later. The Car people
# Here I set Hessian Threshold to 400, The higher the threshold the less features will be included. (Key points are the features)
surf = cv.xfeatures2d.SURF_create(3000)


# Find keypoints and descriptors directly
kp, des = surf.detectAndCompute(img,None)

# Prints the length of the features
print("Length of features")
print(len(kp))
###########################################################################################################################
#KNN



#########################################################################################################################
#Visualizing the Keypoints
Example_img =  cv.drawKeypoints(img,kp,None,(255,0,0),4)
#print the dimensions of the new image
print("New Image Dimensions")
print(Example_img.shape)
plt.imshow(Example_img),plt.show()

#This line is here simply for debugging purposes
print(len(kp))
############################################################################################################################

#Dataset_1

Dataset_1 = np.empty([71500,64], dtype=float) #Creating the array to hold the value of the sums of each feature
Dataset_1[:,:]=0
Dataset_2 = np.empty([71500,64], dtype=float) #Creating the array to hold the value of the sums of each feature
Dataset_2[:,:]=0

# Dont FORGET TO RESHAPE THE LABEL ARRAY, SINCE THE LABEL ARRAY IS ALSO 325 BY 220
data=np.reshape(data, [71500, 64])
labels=np.reshape(labels,[71500,1])

#for feature in range (0,64):
Dataset_1=sklearn.preprocessing.normalize(data)
Dataset_2=sklearn.preprocessing.maxabs_scale(data)

###########################################################################################################################

# Divide the data (75% training, 25% test)
training_data, testing_data, label_train, label_test =sklearn.model_selection.train_test_split(Dataset_1, labels, test_size=0.25, train_size=0.75)

#Use 10 fold cross validation on the data set
kfold = KFold(n_splits =10, shuffle = True)
n_estimators=[5,10,25,50,100,200,500]

for n in n_estimators :  
    max_score=0
    best_n_norm=0

    for train_index, test_index in kfold.split(training_data):
            x_training, x_test = training_data[train_index], training_data[test_index]
            y_training, y_test = label_train[train_index], label_train[test_index]
            
            # evaluate model, Found accuracy across different folds while debugging
            forest = RandomForestClassifier(n_estimators=n,max_depth=2,random_state=0) #Create model
            forest.fit( x_training, y_training)
            prediction=forest.predict(x_test)
            true=np.reshape(y_test, len(y_test))
            scores = rand_score(true, prediction)
            avg_score=np.mean(scores)

            if avg_score>max_score:
                best_n_norm=n
                max_score=avg_score
                test_prediction=forest.predict(testing_data)
           
#score=rand_score(y_test, prediction)
print('For dataset')
print('Best n_estimators:', best_n_norm, ' avg_score:', avg_score)
true_test=np.reshape(label_test, len(label_test))
accuracy=rand_score(true_test,test_prediction)
print("Normalize_Accuracy on test dataset: ", accuracy)

################################################################################################################################

debug=0 #line for debugging
            



