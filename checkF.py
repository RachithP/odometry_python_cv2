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


def calculateEpipoles(F):
	'''
	Function to calculate epipoles using the fundamental matrix
	:param F:
	:return:
	'''

	# left image epipole is the right null-space of the matrix
	

	# right image epipole is the left null-space of the matrix


def isFValid(F, img1_pixels, img2_pixels, image1, image2):
	'''
	Function to check if the obtained F is valid or not
	:param F:
	:param img1_pixels:
	:param img2_pixels:
	:return:
	'''

	for pixel in img1_pixels:
		line = F.dot([img1_pixels[0], img1_pixels[1], 1])

		norm = line[0]**2 + line[1]**2

		line = line / norm



