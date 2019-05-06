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


def checkChirality(Cset, Rset, Xset):
	counts = []

	for ind in range(4):
		cnt = 0
		for point in Xset[ind]:
			# H = np.vstack((np.hstack((Rset[ind], Cset[ind].reshape(3, 1))), np.array([[0, 0, 0, 1]])))
			# point2 = np.linalg.inv(H).dot(np.array([point[0], point[1], point[2], 1]).reshape(4, 1))
			# print(point2)
			# print(point)
			# print(np.matmul(Rset[ind][2, :], (point - Cset[ind])))
			cnt += (np.matmul(Rset[ind][2, :], (point - Cset[ind]))>0)
		counts.append(cnt)

	ind = np.argmax(counts)

	print('Chirality count obtained is ', counts)
	quit()

	return np.array(Cset[ind]).reshape(3, 1), Rset[ind], Xset[ind]
