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

import cv2
import numpy as np


def getPixelCoordinates(kp1, kp2, matches):
	'''
	In this function we obtain pixel coordinates of matches features in each image
	:param kp1:
	:param kp2:
	:param matches:
	:return:
	'''

	# Initialize lists
	list_kp1 = []
	list_kp2 = []

	# For each match...
	for mat in matches:
		# Get the matching keypoints for each of the images
		img1_idx = mat.queryIdx
		img2_idx = mat.trainIdx

		# x - columns
		# y - rows
		# Get the coordinates
		(x1, y1) = kp1[img1_idx].pt
		(x2, y2) = kp2[img2_idx].pt

		# Append to each list
		list_kp1.append((x1, y1))
		list_kp2.append((x2, y2))

	return list_kp1, list_kp2


def extractMatchFeatures(image1, image2):
	'''
	This function is used to extract features between pair of images
	:param image1:
	:param image2:
	:return: pixel coordinates of matched features
	'''
	# Initiate ORB detector
	orb = cv2.ORB_create()
	# orb = cv2.SIFT()

	# find the keypoints and descriptors with ORB
	kp1, des1 = orb.detectAndCompute(image1, None)
	kp2, des2 = orb.detectAndCompute(image2, None)

	# create BFMatcher object
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
	# bf = cv2.BFMatcher()

	# Match descriptors.
	matches = bf.match(des1, des2)
	# matches = bf.knnMatch(des1, des2, k=2)

	# Sort them in the order of their distance.
	matches = sorted(matches, key=lambda x: x.distance)
	# Apply ratio test
	# good = []
	# for match in matches:
	# 	m = match[0]
	# 	n = match[1]
	# 	if m.distance < 0.75 * n.distance:
	# 		good.append(match)

	#Taking only the best 50
	matches = matches[:200]
	# matches = good

	pixelsImg1, pixelsImg2 = getPixelCoordinates(kp1, kp2, matches)

	# code to visualize the matched features
	# img1 = cv2.drawKeypoints(image1, kp1, image1, color=(0, 255, 0), flags=0)
	# img2 = cv2.drawKeypoints(image2, kp2, image2, color=(255, 0, 0), flags=0)
	# cv2.imshow('image1', img1)
	# cv2.imshow('image2', img2)
	# cv2.waitKey(0)

	return pixelsImg1, pixelsImg2


def siftFeatures(img1, img2):

	sift = cv2.xfeatures2d.SIFT_create()

	# find the keypoints and descriptors with SIFT
	kp1, des1 = sift.detectAndCompute(img1, None)
	kp2, des2 = sift.detectAndCompute(img2, None)

	# Find point matches
	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
	search_params = dict(checks=50)
	flann = cv2.FlannBasedMatcher(index_params, search_params)
	matches = flann.knnMatch(des1, des2, k=2)

	# Apply Lowe's SIFT matching ratio test
	good = []
	for m, n in matches:
		if m.distance < 0.8 * n.distance:
			good.append(m)

	src_pts = np.asarray([kp1[m.queryIdx].pt for m in good])
	dst_pts = np.asarray([kp2[m.trainIdx].pt for m in good])

	# Constrain matches to fit homography
	retval, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 100.0)
	mask = mask.ravel()

	# We select only inlier points
	pts1 = src_pts[mask == 1]
	pts2 = dst_pts[mask == 1]

	return pts1, pts2
