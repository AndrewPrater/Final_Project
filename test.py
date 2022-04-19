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
	for filename in os.listdir(folder):
		img = cv.imread(os.path.join(folder,filename))
		if img is not None:
			images.append(img)
	return images

def test(model, data):
	features = extract_features(data)
	results = model.predict(features)
	return results

def extract_features(images):
	hessian = 40
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
	return features

if __name__ == '__main__':
	trained_model = pickle.load(open('model_file', 'rb'))
	folder_name = '..//test'
	test_images = load_images_from_folder(folder_name)
	returnArray = test(trained_model, test_images)
	print(returnArray)