import cv2 as cv
import os

def load_images_from_folder(folder):
	images = []
	labels = []
	for subFolder in os.listdir(folder):
		finalAddress = folder + "\\" + subFolder
		for filename in os.listdir(finalAddress):
			img = cv.imread(os.path.join(finalAddress,filename))
			identifier = filename[0]
			if img is not None:
				if identifier != 'r':
					images.append(img)
					labels.append(identifier)
	return images, labels


images1, labels1 = load_images_from_folder(r"C:\Users\coles\Documents\Classes\EEE3773 Machine Learning\train")
print("Done")
print(images1.shape())