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

import glob, argparse
import numpy as np
import copy
import cv2
import extractPose
import preProcessing as dataPrep
from ransac import RANSAC
import matchedFeaturesCoordinates as features
import triangulation
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D as axes3D
import checkF
from chiralityCheck import checkChirality

def vizMatches(image1, image2, pixelsImg1, pixelsImg2):
	'''
	Visualize the feature match between pair of images
	:param image1:
	:param image2:
	:param pixelsImg1:
	:param pixelsImg2:
	:return:
	'''
	# visualization of the matches
	h1, w1 = image1.shape[:2]
	h2, w2 = image2.shape[:2]
	view = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
	view[:h1, :w1, :] = image1
	view[:h2, w1:, :] = image2
	view[:, :, 1] = view[:, :, 0]
	view[:, :, 2] = view[:, :, 0]

	for ind in range(len(pixelsImg1)):
		# draw the keypoints
		color = tuple([np.random.randint(0, 255) for _ in xrange(3)])
		cv2.line(view, (int(pixelsImg1[ind][0]), int(pixelsImg1[ind][1])),
				 (int(pixelsImg2[ind][0] + w1), int(pixelsImg2[ind][1])), color)

	cv2.imshow("view", view)
	cv2.waitKey(0)


def vizCameraPose(T_own, T):
	'''
	Function to visualize camera movement
	:param R:
	:param T:
	:return:
	'''
	T_own = np.array(T_own)
	T = np.array(T)

	plt.plot(T_own[:, 0].flatten(), T_own[:, 2].flatten(), 'g.', label="Our implementation")
	# plt.plot(T[:, 0].flatten(), T[:, 2].flatten(), 'r.', label="Inbuilt function")
	plt.pause(0.01)
	plt.xlabel('x-axis')
	plt.ylabel('z-axis')
	plt.title('Camera movement')


def main():
	# Parse input arguments
	Parser = argparse.ArgumentParser()
	Parser.add_argument('--Path', default="../Oxford_dataset/stereo/centre",
						help='Path to dataset, Default:../Oxford_dataset/stereo/centre')
	Parser.add_argument('--ransacEpsilonThreshold', default=0.01,
						help='Threshold used for deciding inlier during RANSAC, Default:0.01')
	Parser.add_argument('--inlierRatioThreshold', default=0.8,
						help='Threshold to consider a fundamental matrix as valid, Default:0.85')

	Args = Parser.parse_args()
	path = Args.Path
	epsilonThresh = Args.ransacEpsilonThreshold
	inlierRatioThresh = Args.inlierRatioThreshold

	# pre-process data to get undistorted images
	dataPrep.undistortImage(path_to_model='./model', path_to_images=path)

	# extract calibration matrix from
	K = dataPrep.extractCalibrationMatrix(path_to_model='./model')

	# extract images from undistort
	new_path = './undistort'
	filesnumber = sorted(glob.glob(new_path + "/frame*.png"))

	# extract calibration matrix
	K = dataPrep.extractCalibrationMatrix(path_to_model='./model')

	T = []
	T_own = []
	R = []
	H = np.identity(4)
	H_own = np.identity(4)

	for imageIndex in range(50, len(filesnumber)-60):
		print('Image number:', imageIndex)
		# bgrImages, vizImages = extractImages(new_path, 20)
		# ------------Process pair of images -------------------------------------
		img1 = cv2.imread(new_path + "/frame" + str(imageIndex) + ".png", -1)

		# histogram equalization of the image
		equ1 = cv2.equalizeHist(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY))
		# blur the image
		img1_gray = cv2.GaussianBlur(equ1, (3, 3), 0)
		# second image
		img2 = cv2.imread(new_path + "/frame" + str(imageIndex + 1) + ".png", -1)
		# histogram equalization of the image
		equ2 = cv2.equalizeHist(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY))
		# blur the image
		img2_gray = cv2.GaussianBlur(equ2, (3, 3), 0)

		# extract images from the input array
		pixelsImg1, pixelsImg2 = features.extractSIFTFeatures(img1_gray, img2_gray)
		# vizMatches(img1, img2, pixelsImg1, pixelsImg2) # visualize the feature matches before RANSAC

		F, inlierImg1Pixels, inlierImg2Pixels, _, _ = RANSAC(pixelsImg1, pixelsImg2, epsilonThresh, inlierRatioThresh)
		# vizMatches(img1, img2, inlierImg1Pixels, inlierImg2Pixels) # visualize after RANSAC

		E, Cset, Rset = extractPose.extractPose(F, K)
		# take points in image frame before checking chirality condition
		points1new = np.hstack((np.array(inlierImg1Pixels), np.ones((len(inlierImg1Pixels), 1)))).T
		points2new = np.hstack((np.array(inlierImg2Pixels), np.ones((len(inlierImg2Pixels), 1)))).T
		points1k = np.linalg.inv(K).dot(points1new)
		points1 = points1k.T
		points2k = np.linalg.inv(K).dot(points2new)
		points2 = points2k.T

		# check chirality and obtain the true pose
		newR, newT, X = checkChirality(Rset, Cset, points1, points2)

		# perform non-liner triangulation
		# X = triangulation.nonLinearTriangulation(inlierImg1Pixels, inlierImg2Pixels, H_own, X)

		temp_H = np.hstack((newR, newT.reshape(3, 1)))
		temp_H = np.vstack((temp_H, [0, 0, 0, 1]))
		H_own = np.matmul(H_own, temp_H)

		print('-----------------------own', H_own[0:3, 3])
		T_own.append(H_own[0:3, 3])

#---------Inbuilt------------------
		# E_cv2, mask1 = cv2.findFundamentalMat(np.array(pixelsImg1), np.array(pixelsImg2), method=cv2.RANSAC,
		# 									  focal=964.828979, pp=(643.788025, 484.40799), prob=0.85, threshold=0.5)
		E_cv2, mask1 = cv2.findEssentialMat(np.array(pixelsImg1), np.array(pixelsImg2), method=cv2.RANSAC,
											focal=964.828979, pp=(643.788025, 484.40799), prob=0.85, threshold=3.0)
		points, r, t, mask = cv2.recoverPose(E_cv2, np.array(pixelsImg1), np.array(pixelsImg2), focal=964.828979,
											 pp=(643.788025, 484.40799), mask=mask1)

		newH = np.hstack((r.T, -r.T.dot(t.reshape(3, 1))))
		newH = np.vstack((newH, [0, 0, 0, 1]))
		H = np.matmul(H, newH)
		T.append(H[0:3, 3])
		print(H[0:3, 3])
		print('--------------------------')
#----------------------------------

		# visualize
		vizCameraPose(T_own, T)
	plt.legend()
	plt.show()

cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
