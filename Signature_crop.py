# to make it compatible for python 3 
# import necessary libraries
from __future__ import print_function
import cv2
import numpy as np
import os
import subprocess as sb

# Laplacian (general derivate of the image)
def laplacian(x,y):
	laplacian_matrix = []
	for i in range(231):
		dummy_array = []
		for j in range(115):
			dummy_array.append(x[i][j] ** 2 + y[i][j] ** 2)
		laplacian_matrix.append(dummy_array)
	return laplacian_matrix

# computes element wise multiplication of two arrays
def element_multiply(array,kernel):
	answer = 0
	for i in range(3):
		for j in  range(3):
			answer += array[i][j] * kernel[i][j]
	return answer

# def convolution function 	
def convolution(image,kernel):
	convolution_matrix = []
	for i in range (0,693,3):
		dummy_array = []
		for j in range (0,345,3):
			temp = image[i : i + 3, j : j + 3]
			dummy_array.append(element_multiply(temp,kernel))
		convolution_matrix.append(dummy_array)
	return convolution_matrix
	
# calculate the gradients in the image
# black to white is considered positive and vice versa 
def gradient_image(image):
	# def kernel and gradient properties
	x_kernel = [[-1,0,1],[-2,0,2],[-1,0,1]]
	gradient_x = np.array(convolution(image,x_kernel))
	y_kernel = [[-1,-2,-1],[0,0,0],[1,2,1]]
	gradient_y = np.array(convolution(image,y_kernel))
	return laplacian(gradient_x,gradient_y)

# Main function 
def main():
	image = cv2.imread('NFI-00101001.png',0)
	resized_image = cv2.resize(image,(348,696))
	# cv2.imshow('image',resized_image)
	resized_image = np.float32(resized_image) / 255.0
	height_image,width_image = resized_image.shape
	gradient_matrix = np.array(gradient_image(resized_image)) * 255
	cv2.imshow('g_image',gradient_matrix)
	cv2.imwrite('g_image.png',gradient_matrix)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

# Init routine
if __name__ == "__main__":
	main()

