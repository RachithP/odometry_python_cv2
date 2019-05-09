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
import pnp


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


def vizCameraPose(R, T):
	'''
	Function to visualize camera movement
	:param R:
	:param T:
	:return:
	'''
	T = np.array(T)

	# fig = plt.figure(1)
	# axis = fig.add_subplot(1, 1, 1, projection="3d")
	# axis.scatter(T[:, 0].flatten(), T[:, 1].flatten(), T[:, 2].flatten(), marker=".")
	# axis.set_xlabel('x')
	# axis.set_ylabel('y')
	# axis.set_zlabel('z')
	# plt.title('Camera movement')
	# plt.pause(0.2)
	plt.plot(T[:, 0].flatten(), T[:, 2].flatten(), 'r.', label="Our implementation")
	plt.pause(0.1)
	plt.xlabel('x-axis')
	plt.ylabel('z-axis')
	plt.title('Camera movement')


def main():
	# Parse input arguments
	Parser = argparse.ArgumentParser()
	Parser.add_argument('--Path', default="../Oxford_dataset/stereo/centre",
						help='Path to dataset, Default:../Oxford_dataset/stereo/centre')
	Parser.add_argument('--ransacEpsilonThreshold', default=0.5,
						help='Threshold used for deciding inlier during RANSAC, Default:0.01')
	Parser.add_argument('--inlierRatioThreshold', default=0.85,
						help='Threshold to consider a fundamental matrix as valid, Default:0.85')

	Args = Parser.parse_args()
	path = Args.Path
	epsilonThresh = Args.ransacEpsilonThreshold
	inlierRatioThresh = Args.inlierRatioThreshold

	# pre-process data to get undistorted images
	# prep.undistortImage(path_to_model='./model', path_to_images=path)

	# extract calibration matrix from
	K = dataPrep.extractCalibrationMatrix(path_to_model='./model')

	# extract images from undistort
	new_path = './undistort'
	# bgrImages = extractImages(new_path, 20)
	filesnumber = sorted(glob.glob(new_path + "/frame*.png"))

	# extract calibration matrix
	K = dataPrep.extractCalibrationMatrix(path_to_model='./model')

	T = []
	R = []
	H = np.identity(4)

	for imageIndex in range(50, len(filesnumber) - 60):
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

		# F, inlierImg1Pixels, inlierImg2Pixels, _, _ = RANSAC(pixelsImg1, pixelsImg2, epsilonThresh, inlierRatioThresh)
		# vizMatches(img1, img2, inlierImg1Pixels, inlierImg2Pixels) # visualize after RANSAC

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
#----------------------------------

		# # check if obtained fundamental matrix is valid or not
		# # checkF.isFValid(F, inlierImg1Pixels, inlierImg2Pixels, img1_gray, img2_gray, imageIndex)
		#
		# # # get all poses (4) possible
		# Cset, Rset = extractPose.extractPose(F, K)
		#
		# # # this is to perform triangulation using Eigen method
		# X, c, r = triangulation.linearTriangulationEigen(K, np.zeros((3, 1)), np.diag([1, 1, 1]), Cset, Rset, inlierImg1Pixels, inlierImg2Pixels)
		#
		# newH = np.hstack((r.T, c.reshape(3, 1)))
		# newH = np.vstack((newH, [0, 0, 0, 1]))
		# H = np.matmul(H, newH)
		# T.append(H[0:3, 3])
		# print((H[0:3, 3]))


		# visualize
		vizCameraPose(R, T)

	plt.show()

cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
