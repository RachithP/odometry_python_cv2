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
import cv2

def calculateEpipoles(F):
	'''
	Function to calculate epipoles using the fundamental matrix
	:param F:
	:return:
	'''

	# left image epipole is the right null-space of the matrix
	eigen_values, eigen_vectors = np.linalg.eig(F)

	# sort eigen values decreasing order
	idx = eigen_values.argsort()[::-1]
	eigen_values = eigen_values[idx]
	eigen_vectors = eigen_vectors[:, idx]

	# get right-eigenvector for left epipole
	right_eigenvector = eigen_vectors[:, -1]
	right_eigenvector = right_eigenvector / right_eigenvector[2]
	# right image epipole is the left null-space of the matrix

	return right_eigenvector


def isFValid(F, img1_pixels, img2_pixels, image1, image2):
	'''
	Function to check if the obtained F is valid or not
	:param F:
	:param img1_pixels:
	:param img2_pixels:
	:return:
	'''

	for pixel in img1_pixels:
		# line = F.dot([img1_pixels[0], img1_pixels[1], 1])

		# norm = line[0]**2 + line[1]**2

		# line = line / norm


		# for now gives right eigen vector - left epipole
		left_epipole = calculateEpipoles(F)


		color = tuple([np.random.randint(0, 255) for _ in xrange(3)])
		cv2.line(image1, (int(left_epipole[0]), int(left_epipole[1])), (int(pixel[0]), int(pixel[1])), color)

		cv2.imshow('image', image1)
		cv2.waitKey(0)


