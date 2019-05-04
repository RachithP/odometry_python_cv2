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


def linearTriangulationEigen(K, Cset, Rset, img1_pixels, img2_pixels):
	'''
	This is linear-LeastSquare method
	Perform linear triangulation from pixel coordinates (homogeneous) to get world coordinates
	:param K: calibration matrix
	:param img1px: pixel location from image 1
	:param img2px: pixel location from image 2
	:param img1_pixels:
	:param img2_pixels:
	:return:
	'''

	K = np.array(K)
	# obtain projective matrix for first image with identity rotation and 0 translation
	P1 = K.dot(np.hstack((np.diagflat([1, 1, 1]), np.zeros((3, 1)))))
	X = []
	Xset = []

	for ind in range(len(Rset)):

		# obtain projective matrix for second image with the rotation and translation given by camera shift
		P2 = K.dot(np.hstack((Rset[ind], Cset[ind].reshape(3, 1))))

		# re-project every matched point
		# define the matrix using intrinsic parameters and each image pixel coordinates

		for i in range(len(img1_pixels)):
			img1px = img1_pixels[i]
			img2px = img2_pixels[i]

			A = np.squeeze(np.array([[img1px[1] * P1[2, :] - P1[1, :]], [img1px[0] * P1[2, :] - P1[0, :]],
									 [img2px[1] * P2[2, :] - P2[1, :]], [img2px[1] * P2[2, :] - P2[0, :]]]))

			# compute AT.A
			M = A.T.dot(A)

			# perform svd to obtain world coordinate - last eigen vector
			eigen_values, eigen_vectors = np.linalg.eig(M)

			idx = eigen_values.argsort()[::-1]
			# eigen_values = eigen_values[idx]
			eigen_vectors = eigen_vectors[:, idx]

			# normalize the vector P - last eigenvector
			P = eigen_vectors[:, -1]
			P = P / P[-1]

			X.append(P[:3])

		Xset.append(X)

	return np.array(Xset)


def linearTriangulationLS(K, Cset, Rset, img1_pixels, img2_pixels):
	'''
	Perform linear triangulation from pixel coordinates (homogeneous) to get world coordinates
	:param K: calibration matrix
	:param img1px: pixel location from image 1
	:param img2px: pixel location from image 2
	:param img1_pixels:
	:param img2_pixels:
	:return:
	'''

	K = np.array(K)
	# obtain projective matrix for first image with identity rotation and 0 translation
	P1 = K.dot(np.hstack((np.diagflat([1, 1, 1]), np.zeros((3, 1)))))
	X = []
	Xset = []

	for ind in range(len(Rset)):

		# obtain projective matrix for second image with the rotation and translation given by camera shift
		P2 = K.dot(np.hstack((Rset[ind], Cset[ind].reshape(3, 1))))

		# re-project every matched point
		# define the matrix using intrinsic parameters and each image pixel coordinates

		for i in range(len(img1_pixels)):
			img1px = img1_pixels[i]
			img2px = img2_pixels[i]

			A = np.squeeze(np.array([[img1px[1] * P1[2, :] - P1[1, :]], [img1px[0] * P1[2, :] - P1[0, :]],
									 [img2px[1] * P2[2, :] - P2[1, :]], [img2px[1] * P2[2, :] - P2[0, :]]]))

			# perform svd to obtain world coordinate of the respective point
			u, s, vh = np.linalg.svd(A)
			P = vh.T[:, -1]

			# normalize the vector X
			P = P / P[3]

			X.append(P[:3])

		Xset.append(X)

	return np.array(Xset)
