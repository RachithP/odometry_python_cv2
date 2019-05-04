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
			cnt += (np.dot(Rset[ind][2, :], point - Cset[ind]) > 0)
		counts.append(cnt)

	ind = np.argmax(counts)

	return Cset[ind], Rset[ind]
