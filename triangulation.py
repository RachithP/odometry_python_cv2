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
from scipy.optimize import least_squares

def nonLinearTriangulation(uArr,vArr,PArr,initGuess):
	def homX(x):
		homx = x/x[2]
		return homx
	def functionNonTriag(x):
		Pt = P.T
		for j in range(2):
			Ptj = Pt[j]
			term1 = (u[j]-(np.matmul(Ptj[0,:],homX(x))/(Ptj[2,:]*x)))**2
			term2 = (v[j]-(np.matmul(Ptj[1,:],homX(x))/(Ptj[2,:]*x)))**2
			if j==0:
				output = term1+term2
			else:
				output += term1+term2
		return output
	res_1 = least_squares(functionNonTriag, initGuess)

	return res_1.x

def linearTriangulationEigen(K, C0, R0, Cset, Rset, img1_pixels, img2_pixels):
	'''
	This is linear-Eigen based method
	Perform linear triangulation from pixel coordinates (homogeneous) to get world coordinates
	:param K:
	:param C0:
	:param R0:
	:param Cset:
	:param Rset:
	:param img1_pixels:
	:param img2_pixels:
	:return:
	'''
	K = np.array(K)
	# obtain projective matrix for first image with identity rotation and 0 translation

	R0 = np.squeeze(R0)
	C0 = C0.reshape(3, 1)
	P1 = K.dot(np.hstack((R0, C0)))
	Xset = []

	for ind in range(len(Rset)):
		X = []

		T = -Rset[ind].dot(Cset[ind]).reshape(3, 1)
		R = Rset[ind].T
		# obtain projective matrix for second image with the rotation and translation given by camera shift
		P2 = K.dot(np.hstack((R, T)))

		# re-project every matched point
		# define the matrix using intrinsic parameters and each image pixel coordinates

		for i in range(len(img1_pixels)):
			img1px = img1_pixels[i]
			img2px = img2_pixels[i]

			A = np.squeeze(np.array([[img1px[1] * P1[2, :] - P1[1, :]], [img1px[0] * P1[2, :] - P1[0, :]],
									 [img2px[1] * P2[2, :] - P2[1, :]], [img2px[0] * P2[2, :] - P2[0, :]]]))

			# compute AT.A
			M = A.T.dot(A)

			# eigen_values, eigen_vectors = np.linalg.eig(M)
			#
			# idx = eigen_values.argsort()[::-1]
			# # eigen_values = eigen_values[idx]
			# eigen_vectors = eigen_vectors[:, idx]
			#
			# # normalize the vector P - last eigenvector
			# P = eigen_vectors[:, -1]
			# P = P / P[-1]

			# perform svd to obtain world coordinate - last eigen vector
			u, s, vh = np.linalg.svd(M)
			P = vh.T[:, -1]

			if P[3] !=0:
				# normalize the vector X
				P = P / P[3]

				X.append(P[:3])
		Xset.append(X)

	return np.array(Xset)


def linearTriangulationLS(K, C0, R0, Cset, Rset, img1_pixels, img2_pixels):
	'''
	This is linear least-square based optimization
	Perform linear triangulation from pixel coordinates (homogeneous) to get world coordinates
	:param K:
	:param C0:
	:param R0:
	:param Cset:
	:param Rset:
	:param img1_pixels:
	:param img2_pixels:
	:return:
	'''
	K = np.array(K)
	# obtain projective matrix for first image with identity rotation and 0 translation
	P1 = K.dot(np.hstack((R0, C0)))
	X = []
	Xset = []

	for ind in range(len(Rset)):

		T = -Rset[ind].dot(Cset[ind]).reshape(3, 1)
		# obtain projective matrix for second image with the rotation and translation given by camera shift
		P2 = K.dot(np.hstack((Rset[ind], T)))

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
