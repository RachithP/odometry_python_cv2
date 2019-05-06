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
import random


def computeCorrespondMat(im1Px, im2Px):
	'''
	Compute the matrix from matched feature locations. This matrix is used to solve svd
	:param im1Px: matched pixel coordinates from image 1
	:param im2Px: matched pixel coordinates from image 1
	:return: Matrix for fundamental matrix computation
	'''

	x1, y1 = im1Px[0][0], im1Px[0][1]
	x1_, y1_ = im2Px[0][0], im2Px[0][1]
	mat = np.array([x1 * x1_, x1 * y1_, x1, y1 * x1_, y1 * y1_, y1, x1_, y1_, 1])

	for k in range(1, len(im1Px)):
		x1, y1 = im1Px[k][0], im1Px[k][1]
		x1_, y1_ = im2Px[k][0], im2Px[k][1]
		row = np.array([x1 * x1_, x1 * y1_, x1, y1 * x1_, y1 * y1_, y1, x1_, y1_, 1])
		mat = np.vstack((mat, row))

	return mat


def computeFundamentalMatrix(pixelsImg1, pixelsImg2):
	'''
	Computes Fundamental matrix between two sets of matched pixel coordinates
	:param pixelsImg1: coordinates belong to image 1
	:param pixelsImg2: coordinates belong to image 2
	:return:
	'''

	corMatRow = computeCorrespondMat(pixelsImg1, pixelsImg2)
	u, s, vh = np.linalg.svd(corMatRow)

	lastCol = vh.T[:, vh.shape[1] - 1]

	# reshape to get F matrix
	return np.vstack((lastCol[0:3], lastCol[3:6], lastCol[6:9]))


def checkRank2(uncheckedF):
	'''
	This function checks the rank of Fundamental Matrix and enforces a rank of 2 if not 2.
	:param uncheckedF:
	:return: matrix with rank 2
	'''

	# perform SVD to obtain the singular values. Note: s is an array of singular values arranged in descreasing order
	u, s, vh = np.linalg.svd(uncheckedF)
	# check if the last singular value is 0. If not, enforce zero
	s[-1] = 0

	# Convert singular value array to diagonal form
	S = np.zeros((s.shape[0], u.shape[0]), dtype=u.dtype)
	np.fill_diagonal(S, s)

	# Re-compute F
	F = u.dot(S).dot(vh)

	return F


def normalizePoints(points):
	'''
	Scale and translate image points so that the centroid of the points are at the origin and average distance to the origin
	is equal to sqrt(2)
	:param points: array of inlier points
	:return: array of same input normalized
	'''

	points = np.asarray(points)
	x = points[:, 0]
	y = points[:, 1]

	# calculate mean along each axis
	centre = points.mean(axis=0)
	# centre around this mean
	cx = x - centre[0]
	cy = y - centre[1]

	dist = np.sqrt(np.power(cx, 2) + np.power(cy, 2))
	scale = np.sqrt(2) / dist.mean()

	norm = np.array(
		[[scale, 0, -scale * centre[0]],
		 [0, scale, -scale * centre[1]],
		 [0, 0, 1]])

	points = np.hstack((points, np.ones((points.shape[0], 1))))

	norm_points = norm.dot(points.T).T

	return norm_points[:, :2], norm


def RANSAC(pixels_img1, pixels_img2, epsilonThresh, inlierRatioThresh):
	'''
	Apply RANSAC for robust estimation of fundamental matrix
	:param pixelsImg1:
	:param pixelsImg2:
	:param epsilonThresh:
	:param inlierRatioThresh:
	:return:
	'''

	# pre-process image coordinates
	pixelsImg1, norm1 = normalizePoints(pixels_img1)
	pixelsImg2, norm2 = normalizePoints(pixels_img2)

	max_value = len(pixels_img1) - 1
	best_img1_pixel = (0, 0)  # (x,y) format
	best_img2_pixel = (0, 0)
	min_epsilon = 10  # used to obtain pixel giving least error
	count = 0

	while 1:

		# 8 random pixel coordinate indices
		randomPixelInds = np.random.randint(max_value, size=8)

		randImg1Pixels = []
		randImg2Pixels = []

		# get (x,y) of the corresponding index
		for k in randomPixelInds:
			randImg1Pixels.append(pixelsImg1[k])
			randImg2Pixels.append(pixelsImg2[k])

		# get matrix from random points selected
		randomF = computeFundamentalMatrix(randImg1Pixels, randImg2Pixels)

		# impose rank 2 constraint
		randomF = checkRank2(randomF)

		# iterate through all points to find the inliers for the above computed matrix
		inliersInds = []
		for ind in range(len(pixelsImg1)):
			img1Pixels = np.array([pixelsImg1[ind][0], pixelsImg1[ind][1], 1])
			img2Pixels = np.array([pixelsImg2[ind][0], pixelsImg2[ind][1], 1])

			# check the error condition
			epsilon = img2Pixels.T.dot(randomF).dot(img1Pixels)
			if abs(epsilon) < epsilonThresh:
				inliersInds.append(ind)
				# store the pixel coordinates that gave least error - Used for linear triangulation
				if abs(epsilon) < min_epsilon:
					min_epsilon = abs(epsilon)
					best_img1_pixel = img1Pixels[:2]
					best_img2_pixel = img2Pixels[:2]
		count += 1

		# calculate the inlier percentage and break if required number is satisfied
		inlierPercentage = float(len(inliersInds)) / len(pixelsImg1)
		if inlierPercentage > inlierRatioThresh:
			print('Yaay!!, Found inlier ratio to be ', inlierPercentage)
			break

	inlierImg1Pixels = []
	inlierImg2Pixels = []
	return_inlierpoints_img1 = []
	return_inlierpoints_img2 = []
	for k in inliersInds:
		inlierImg1Pixels.append(pixelsImg1[k])
		inlierImg2Pixels.append(pixelsImg2[k])
		return_inlierpoints_img1.append(pixels_img1[k])
		return_inlierpoints_img2.append(pixels_img2[k])

	# re-compute Fundamental matrix using all these proper matches
	inliersF = computeFundamentalMatrix(inlierImg1Pixels, inlierImg2Pixels)

	# impose rank 2 constraint
	F = checkRank2(inliersF)

	# reverse the effect of normalized coordinates
	F = norm2.T.dot(F).dot(norm1)

	# if F[2, 2] == -1:
	# 	F = -F

	return F / F[2, 2], return_inlierpoints_img1, return_inlierpoints_img2, best_img1_pixel, best_img2_pixel
