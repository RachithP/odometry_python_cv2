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


def checkChirality(C, R, X):
	'''
	Check chirality condition and return true or false
	:param C:
	:param R:
	:param X:
	:return: 1/0
	'''
	return np.dot(R[2, :], X - C) > 0


def extractPose(F, K, world_coordinates):
	'''
	This function calculates Essential Matrix from Fundamental Matrix
	:param F: Fundamental Matrix
	:param K: Calibration Matrix
	:return: Essential Matrix
	'''

	K = np.array(K)
	E = K.T.dot(F).dot(K)
	u, s, vh = svd(E)
	if s[-1] != 0:
		s[-1] = 0

	S = np.zeros((s.shape[0], u.shape[0]), dtype=u.dtype)
	np.fill_diagonal(S, s)

	# Re-compute E
	E = u.dot(S).dot(vh)

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
	counts = []

	# check chirality condition
	for ind in range(4):
		cnt = 0
		for point in world_coordinates:
			cnt += checkChirality(C[ind], R[ind], point)
		counts.append(cnt)

	ind = np.argmax(counts)

	return C[ind], R[ind]
