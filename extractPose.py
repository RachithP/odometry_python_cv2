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
	:return: set of poses, Essential Matrix
	'''

	K = np.array(K)
	E = K.T.dot(F).dot(K)
	u, s, vh = svd(E)

	# impose rank 2 and equal eigen value condition
	s = [s[0]/2+s[1]/2, s[0]/2+s[1]/2, 0]
	S = np.diag(s)

	# Re-compute E
	E = u.dot(S).dot(vh)

	# svd of Essential matrix to compute Rotation and translation matrices
	u, s, vh = svd(E)

	W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

	C = []
	R = []

	#  first possibility
	R1 = u.dot(W.T).dot(vh)

	if u[2, 2] > 0:
		C1 = -u[:, 2]
	else:
		C1 = u[:, 2]

	# second possiblity
	R2 = u.dot(W).dot(vh)

	if np.linalg.det(R1)<0:
		R1 = -R1
	if np.linalg.det(R2)<0:
		R2 = -R2

	C.append(C1)
	R.append(R1)
	C.append(C1)
	R.append(R2)

	return E, np.array(C), np.array(R)
