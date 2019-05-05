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


def normalizePoints(points):
	'''
	Scale and translate image points so that the centroid of the points are at the origin and average distance to the origin
	is equal to sqrt(2)
	:param points: array of inlier points
	:return: array of same input normalized
	'''

	points = np.asarray(points)
	x = points[:, 0]
	y = points[:, 1]

	# calculate mean along each axis
	centre = points.mean(axis=0)
	# centre around this mean
	cx = x - centre[0]
	cy = y - centre[1]

	dist = np.sqrt(np.power(cx, 2) + np.power(cy, 2))
	scale = np.sqrt(2) / dist.mean()

	norm = np.array(
		[[scale, 0, -scale * centre[0]],
		 [0, scale, -scale * centre[1]],
		 [0, 0, 1]])

	points = np.hstack((points, np.ones((points.shape[0], 1))))

	norm_points = norm.dot(points.T).T

	return norm_points[:, :2], norm


def linear(img_points, world_points, K):
	'''
	Compute projection matrix from pairs of 2D-3D correspondences. Also called DLT-Direct Linear Transform solution.
	:param img_points: 2-D points (x,y)
	:param world_points: 3-D points (x,y,z)
	:return:
	'''

	img_points = np.array(img_points)
	world_points = np.squeeze(np.array(world_points))



	n = len(img_points)
	print n
	if len(img_points)<6:
		print 'ERROR!!'
		print 'NOT ENOUGH POINTS TO SOLVE PNP'
		quit()

	# make points homogeneous and normalize image pixel by multiplying it with inv(K)


	#normalize points
	# img_points,norm = normalizePoints(img_points)
	# print img_points
	hom_img_points = np.hstack((img_points, np.ones((img_points.shape[0], 1)))).T
	hom_world_points = np.hstack((world_points, np.ones((world_points.shape[0], 1)))).T
	# print hom_img_points


	M = np.zeros((3 * n, 12 + n))
	for i in range(n):
		M[3 * i, 0:4] = hom_world_points[:, i]
		M[3 * i + 1, 4:8] = hom_world_points[:, i]
		M[3 * i + 2, 8:12] = hom_world_points[:, i]
		M[3 * i:3 * i + 3, i + 12] = -hom_img_points[:, i]

	# svd to solve for last eigen vector
	u, s, vh = np.linalg.svd(M)
	print M.shape
	# get P matrix
	P = vh[-1, :12].reshape((3, 4))

	# remove K from P
	# RT = P
	RT = np.linalg.inv(K).dot(P)

	print RT
	# separate R and T from RT
	R = RT[:, :3]
	quit()
	u, s, vh = np.linalg.svd(R)
	if np.linalg.det(u.dot(vh)) > 0:
		R = u.dot(vh)
		T = RT[:, 3] / s[0]
	else:
		R = -u.dot(vh)
		T = -RT[:, 3] / s[0]

	# print("Translation vector", T)
	print("Determinant of rotation matrix ", np.linalg.det(R))

	C = -R.T.dot(T)

	return np.array([C]), np.array([R])



