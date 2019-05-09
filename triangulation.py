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
	counts = []

	# points = [[-7, 5, 12], [-3, 2, 5], [-4, 3, 7], [-4,3,8], [-5,3,10], [-9,4,15], [-10,6,18], [-1,12,15]]

	for ind in range(len(Rset)):
		X = []
		# X1 = []
		# X2 = []
		# obtain projective matrix for second image with the rotation and translation given by camera shift
		P2 = K.dot(Rset[ind]).dot(np.hstack((np.identity(3), -Cset[ind].reshape(3, 1))))

		# print('---------------------------------------------------')
		# for point in points:
		# 	x1 = P1.dot([point[0], point[1], point[2], 1])
		# 	temp = K.dot(Rset[3]).dot(np.hstack((np.identity(3), -Cset[3].reshape(3, 1))))
		# 	x2 = temp.dot([point[0], point[1], point[2], 1])
		# 	x1 = x1 / x1[2]
		# 	x2 = x2 / x2[2]
		#
		# 	# re-project every matched point
		# 	# define the matrix using intrinsic parameters and each image pixel coordinates
		#
		# 	X1.append(x1)
		# 	X2.append(x2)

		cnt = 0
		for i in range(len(img1_pixels)):
			img1px = img1_pixels[i]
			img2px = img2_pixels[i]

			# img1px = X1[i]
			# img2px = X2[i]
			A = np.squeeze(np.array([[img1px[0] * P1[2, :] - P1[0, :]], [img1px[1] * P1[2, :] - P1[1, :]],
									 [img2px[0] * P2[2, :] - P2[0, :]], [img2px[1] * P2[2, :] - P2[1, :]]]))

			# print(A)
			# compute AT.A
			M = A.T.dot(A)

			# perform svd to obtain world coordinate - last eigen vector
			u, s, vh = np.linalg.svd(M)
			P = vh.T[:, -1]

			# normalize the vector X
			P = P / P[3]

			X.append(P[:3])

			cnt += (Rset[ind][2, :].dot(P[:3]-Cset[ind])>0)

		counts.append(cnt)
		Xset.append(X)

	print('counts...................:', counts)

	index = np.argmax(counts)
	print(index)
	if Cset[index][2] < 0:
		Cset[index] = -Cset[index]

	return Xset[index], Cset[index], Rset[index]

