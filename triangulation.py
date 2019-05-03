#!/usr/bin/env python2

'''
ENPM 673 Spring 2019: Robot Perception
Project 5: Visual Odometry

Authors:
Ashwin Varghese Kuruttukulam(ashwinvk94@gmail.com)
Rachith Prakash (rachithprakash@gmail.com)
Graduate Students in Robotics,
University of Maryland, College Park
'''

import numpy as np


def linearTriangulation(K, img1px, img2px, img1_pixels, img2_pixels):
	'''
	Perform linear triangulation from pixel coordinates (homogeneous) to get world coordinates
	:param K: calibration matrix
	:param img1px: pixel location from image 1
	:param img2px: pixel location from image 2
	:param img1_pixels:
	:param img2_pixels:
	:return:
	'''

	# append last col to K to convert it into 3x4 matrix from 3x3
	K = np.hstack([K, np.zeros((3, 1))])

	# re-project every matched point
	# define the matrix using intrinsic parameters and each image pixel coordinates
	A = [[img1px[1] * K[2, :] - K[1, :]], [img1px[0] * K[2, :] - K[1, :]], [img2px[1] * K[2, :] - K[1, :]], [img2px[1] * K[2, :] - K[1, :]]]

	# perform svd to obtain world coordinate of the respective point
	u, s, vh = np.linalg.svd(A)
	X = vh.T[:, -1]

	# normalize the vector X
	X = X / X[3]



	return X