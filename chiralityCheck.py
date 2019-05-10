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


def checkChirality(rots, trans, oldpoints, newpoints):
	'''
	This function checks chirality condition after triangulating the points
	:param rots: set of all rotation matrices
	:param trans: set of all translation matrices
	:param oldpoints: homogeneous coordinates of prev frame
	:param newpoints: homogeneous coordinates of new frame
	:return:
	'''
	if len(rots) != 2:
		print('LENGTH OF RT SETS IS NOT 2. QUITTING')
		quit()
	H_identity = np.identity(4)
	for i in range(oldpoints.shape[0]):
		check = [False, False]
		X1 = []
		X2 = []
		for j in range(2):
			rot = rots[j]
			t = trans[j]
			H = np.hstack((rot, t.reshape(3, 1)))
			H = np.vstack((H, [0, 0, 0, 1]))
			H = np.linalg.inv(H)
			oldpoint = oldpoints[i]
			newpoint = newpoints[i]
			Anew = np.squeeze(
				np.array([[oldpoint[0] * H_identity[2, :] - H_identity[0, :]], [oldpoint[1] * H_identity[2, :] - H_identity[1, :]],
						  [newpoint[0] * H[2, :] - H[0, :]], [newpoint[1] * H[2, :] - H[1, :]]]))
			_, _, vh = np.linalg.svd(Anew)
			v = vh.T
			# Check this
			X = v[:, -1]
			X = X / X[3]

			Xdash = np.matmul(np.linalg.inv(H), X)

			if X[2] > 0 and Xdash[2] > 0:
				check[j] = True
				if j == 0:
					X1.append(X)
				else:
					X2.append(X)

		if np.sum(check) == 1:
			if check[0]:
				temp = trans[0]
				Hprime = np.hstack((rots[0], temp.reshape(3, 1)))
				Hprime = np.vstack((Hprime, [0, 0, 0, 1]))
				Xout = X1
			else:
				temp = trans[1]
				Hprime = np.hstack((rots[1], temp.reshape(3, 1)))
				Hprime = np.vstack((Hprime, [0, 0, 0, 1]))
				Xout = X2

			Hprime = np.linalg.inv(Hprime)
			newR = Hprime[0:3, 0:3]
			newT = Hprime[0:3, 3]
			if (newT[2] < 0):
				newT = -newT
			return newR, newT, Xout

	newR = np.identity(3)
	newT = np.zeros(3)

	return newR, newT, 0
