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
			# print np.matmul(Rset[ind][2, :], point - Cset[ind])
			# print np.matmul(Rset[ind][2, :], point - Cset[ind]) > 0
			cnt += (np.matmul(Rset[ind][2, :], (point - Cset[ind])) > 0)
		counts.append(cnt)
	# 	print '-------------------'
	# print counts
	ind = np.argmax(counts)

	print ind

	print('Chirality count obtained is ', counts)

	return np.array(Cset[ind]).reshape(3, 1), Rset[ind], Xset[ind]
