# coding=utf-8
from PIL import Image
import os.path
import matplotlib.pyplot as plt
import numpy as np
import random

path = "orl_faces"

def ImageToMatrix(filename):
	# read image
	im = Image.open(filename)
	# show image
	# im.show()
	width,height = im.size # size

	im_data = im.getdata() 
	data = np.matrix(im_data,np.uint8) #transform to maxtrix
	new_data = np.reshape(data,(height,width)) 
	return new_data

def matrix_reshape(matrix, rate=1.0):
	im = Image.fromarray(matrix)
	width,height = im.size
	nw, nh = int(width*rate), int(height*rate)
	im = im.resize((nw,nh),Image.ANTIALIAS)
	new_matrix = np.asarray(im)
	return new_matrix


def load_all_data():
	all_data = []
	# read faces from 40 diferent people
	for i in range(40):
		data_of_one = []
		# read 10 face images of one people
		for j in range(10):
			# read one image
			img_file = path+"/s"+str(i+1)+"/"+str(j+1)+".pgm"
			data_of_one.append(ImageToMatrix(img_file))
		all_data.append(data_of_one)
	return all_data

def show_random_one(all_data):
	i = random.randint(0,39)
	j = random.randint(0,9)
	print("show",i+1,j+1)
	data = all_data[i][j]
	im = Image.fromarray(data)
	plt.imshow(im)
	plt.show()

if __name__ == "__main__":
	all_data = load_all_data()
	# m = all_data[0][0]
	# matrix_reshape(m)
	# just for a test
	# show_random_one(all_data)