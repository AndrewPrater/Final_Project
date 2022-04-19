from locale import normalize
from re import X
import cv2 as cv
import os
import numpy as np
from sklearn import neighbors

import matplotlib.pyplot as plt #version 3.3.4

import scipy
import scipy.io as spio
import sklearn.preprocessing
from sklearn.preprocessing import normalize
import math 
from sklearn.model_selection import KFold
import numpy.random as random
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics.cluster import rand_score
from numpy import mean
from numpy import std

def load_images_from_folder(folder):
	images = []
	labels = []
	for subFolder in os.listdir(folder):
		finalAddress = folder + "\\" + subFolder
		for filename in os.listdir(finalAddress):
			img = cv.imread(os.path.join(finalAddress,filename))
			identifier = filename[0]
			if identifier == 'A':
				identifier = 'a'
			if identifier == 'D':
				identifier = 'd'
			if identifier == 'M':
				identifier = 'm'
			if identifier == 'S':
				identifier = 's'
			if img is not None:
				if identifier != 'r' and identifier != 'R':
					images.append(img)
					labels.append(identifier)
	return images, labels


images1, labels1 = load_images_from_folder(r"C:\Users\coles\Documents\Classes\EEE3773 Machine Learning\train")
imagesNP = np.array(images1)
labelsNP = np.array(labels1)



def train(images, labels, hessian):
	surf = cv.xfeatures2d.SURF_create(hessian)
	features = []
	for img in images:
		kp, des = surf.detectAndCompute(img,None)
		kfinal = [0, 0, 0, 0]
		ksize = 0
		kx = []
		ky = []
		xmean = 0
		ymean = 0
		kangles = 0
		kpLength = 0
		for keypoint in kp:
			ksize += keypoint.size
			kangles += keypoint.angle
			kx.append(keypoint.pt[0])
			ky.append(keypoint.pt[1])
			kpLength += 1
		if(kpLength != 0):
			ksize = ksize/kpLength
			kangles = kangles/kpLength
			xmin = 300
			xmax = 0
			for x in kx:
				if x > xmax:
					xmax = x
				if x < xmin:
					xmin = x
			if xmax != xmin:
				xmax -= xmin
				for x in kx:
					x -= xmin
					x = x/xmax
			ymin = 300
			ymax = 0
			for y in ky:
				if y > ymax:
					ymax = y
				if y < ymin:
					ymin = y
			if ymin != ymax:
				ymax -= ymin
				for y in ky:
					y -= ymin
					y = y/ymax
			for x in kx:
				xmean += x
			for y in ky:
				ymean += y
			xmean = xmean/kpLength
			ymean = ymean/kpLength
		kfinal[0] = ksize
		kfinal[1] = kangles
		kfinal[2] = xmean
		kfinal[3] = ymean
		features.append(kfinal)
	featuresNP = np.array(features)
	return features

if __name__ == '__main__':
	#parameters
	n_neighbors = 102
	distance = 1 #1 is manhattan distance, 2 is euclidean
	hessian = 40


	here = os.path.dirname(os.path.abspath(__file__))
	images1, labels1 = load_images_from_folder('..\\train')

	features = train(images1, labels1,hessian)
	clf1 = neighbors.KNeighborsClassifier(n_neighbors, p=distance)

	scores1 = cross_val_score(clf1, features, labels1, cv=10)
	clf1.fit(features, labels1)

	
	modelfile = open('model_file', 'wb')
	pickle.dump(clf1, modelfile)
	modelfile.close()

	print("%0.2f first training data set accuracy" % scores1.mean())

