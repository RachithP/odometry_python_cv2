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
from numpy.linalg import svd, det

def extractPose(F, K):
	'''
	This function calculates Essential Matrix from Fundamental Matrix
	:param F: Fundamental Matrix
	:param K: Calibration Matrix
	:return: Essential Matrix
	'''

	K = np.array(K)
	E = K.T.dot(F).dot(K)
	u, s, vh = svd(E)

	# impose rank 2 and equal eigen value condition
	s = [1, 1, 0]
	S = np.diag(s)

	# Re-compute E
	E = u.dot(S).dot(vh)

	# E = np.array(([0, -1, 0], [0.5, 0, 0.866], [0, 0, 0]))
	# F = np.linalg.inv(K.T).dot(E).dot(np.linalg.inv(K))

	# svd of Essential matrix to compute Rotation and translation matrices
	u, s, vh = svd(E)

	W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

	C = []
	R = []

	#  first possibility
	C1 = u[:, 2]
	R1 = u.dot(W).dot(vh)
	if det(R1) == -1:
		C1 = -C1
		R1 = -R1
	C.append(C1)
	R.append(R1)

	# second possiblity
	C2 = -u[:, 2]
	R2 = u.dot(W).dot(vh)
	if det(R2) == -1:
		C2 = -C2
		R2 = -R2
	C.append(C2)
	R.append(R2)

	# third possiblity
	C3 = u[:, 2]
	R3 = u.dot(W.T).dot(vh)
	if det(R3) == -1:
		C3 = -C3
		R3 = -R3
	C.append(C3)
	R.append(R3)

	# fourth possiblity
	C4 = -u[:, 2]
	R4 = u.dot(W.T).dot(vh)
	if det(R4) == -1:
		C4 = -C4
		R4 = -R4
	C.append(C4)
	R.append(R4)

	# print(F)
	# quit()

	return np.array(C), np.array(R)