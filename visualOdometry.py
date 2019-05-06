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
import cv2
import extractPose
import preProcessing as dataPrep
from ransac import RANSAC
import matchedFeaturesCoordinates as feature
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
		color = tuple([np.random.randint(0, 255) for _ in range(3)])
		cv2.line(view, (int(pixelsImg1[ind][0]), int(pixelsImg1[ind][1])),
				 (int(pixelsImg2[ind][0] + w1), int(pixelsImg2[ind][1])), color)

	cv2.imshow("view", view)
	cv2.waitKey(0)


def extractImages(path, number_of_images):
	'''
	In this function we store undistorted images in an array
	:param path:
	:return:
	'''
	# Read and store all images in the input folder
	filesnumber = sorted(glob.glob(path + "/frame*.png"))
	filenames = []

	# Removing first 24 images because it is too bright and feature detection will be bad
	for k in range(24, len(filesnumber)):
		filenames.append(path + "/frame" + str(k) + ".png")

	images = []
	bgr_images = []
	for filename in filenames:
		im_read = cv2.imread(filename, -1)
		# histogram equalization of the image
		equ = cv2.equalizeHist(cv2.cvtColor(im_read, cv2.COLOR_BGR2GRAY))
		# blur the image
		bgr_images.append(im_read)
		im_read = cv2.GaussianBlur(equ, (3, 3), 1)
		images.append(im_read)
	cv2.destroyAllWindows()
	print('Done extracting images....')

	return images, bgr_images


def vizCameraPose(T, T_inbuilt):
	'''
	Function to visualize camera movement
	:param R:
	:param T:
	:return:
	'''
	T = np.array(T)
	T_inbuilt = np.array(T_inbuilt)

	fig = plt.figure(1)
	# axis = fig.add_subplot(1, 1, 1, projection="3d")
	# axis.scatter(T[:, 0].flatten(), T[:, 1].flatten(), T[:, 2].flatten(), marker=".")
	# axis.set_xlabel('x')
	# axis.set_ylabel('y')
	# axis.set_zlabel('z')
	# plt.title('Camera movement')
	# axis.set_xlim(-400, 400)
	# axis.set_ylim(-400, 400)
	# axis.set_zlim(-600, 1000)
	# plt.plot(T[:, 0].flatten(), T[:, 2].flatten(), 'r.', label="Our implementation")
	plt.plot(T_inbuilt[:, 0].flatten(), T_inbuilt[:, 2].flatten(), 'b.', label="Inbuilt function implementation")
	plt.xlim(-20, 50)
	plt.ylim(-50, 50)
	plt.show()


def main():
	# Parse input arguments
	Parser = argparse.ArgumentParser()
	Parser.add_argument('--Path', default="../Oxford_dataset/stereo/centre",
						help='Path to dataset, Default:../Oxford_dataset/stereo/centre')
	Parser.add_argument('--ransacEpsilonThreshold', default=0.01,
						help='Threshold used for deciding inlier during RANSAC, Default:0.01')
	Parser.add_argument('--inlierRatioThreshold', default=0.75,
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
	filesnumber = sorted(glob.glob(new_path + "/frame*.png"))

	# extract calibration matrix
	K = dataPrep.extractCalibrationMatrix(path_to_model='./model')

	T = []
	T_inbuilt = []
	alpha = 0
	R = []
	H = np.identity(4)
	H_inbuilt = np.identity(4)

	for imageIndex in range(24, len(filesnumber)-3700):
		print(imageIndex)

		# bgrImages, vizImages = extractImages(new_path, 20)
		# ------------Process pair of images -------------------------------------
		img1 = cv2.imread(new_path + "/frame" + str(imageIndex) + ".png", -1)
		# histogram equalization of the image
		equ1 = cv2.equalizeHist(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY))
		# blur the image
		img1_gray = cv2.GaussianBlur(equ1, (3, 3), 1)
		img2 = cv2.imread(new_path + "/frame" + str(imageIndex + 1) + ".png", -1)
		# histogram equalization of the image
		equ2 = cv2.equalizeHist(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY))
		# blur the image
		img2_gray = cv2.GaussianBlur(equ2, (3, 3), 1)

		# extract images from the input array
		# pixelsImg1, pixelsImg2 = feature.extractMatchFeatures(bgrImages[imageIndex], bgrImages[imageIndex + 1])
		pixelsImg1, pixelsImg2 = feature.siftFeatures(img1_gray, img2_gray)
		# vizMatches(vizImages[imageIndex],vizImages[imageIndex + 1],pixelsImg1,pixelsImg2) # visualize the feature matches before RANSAC

		# -----------------OWN CODE START-----------------------------
		# # compute Fundamental matrix using RANSAC
		# F, inlierImg1Pixels, inlierImg2Pixels, _, _ = RANSAC(pixelsImg1, pixelsImg2, epsilonThresh, inlierRatioThresh)
		# # vizMatches(vizImages[imageIndex], vizImages[imageIndex + 1], inlierImg1Pixels, inlierImg2Pixels) # visualize after RANSAC
		# # check if obtained fundamental matrix is valid or not - This is for debugging purpose only
		# # checkF.isFValid(F, inlierImg1Pixels, inlierImg2Pixels, bgrImages[imageIndex], bgrImages[imageIndex + 1], imageIndex)

		# # get all poses (4) possible and E - Essential Matrix
		# E, Cset, Rset = extractPose.extractPose(F, K)

		# # convert points to image frame by multiplying  by inv(K)
		# points1new = np.hstack((np.array(inlierImg1Pixels), np.ones((len(inlierImg1Pixels), 1)))).T
		# points2new = np.hstack((np.array(inlierImg2Pixels), np.ones((len(inlierImg2Pixels), 1)))).T
		# points1k = np.linalg.inv(K).dot(points1new)
		# points1 = points1k.T
		# points2k = np.linalg.inv(K).dot(points2new)
		# points2 = points2k.T
		#
		# # check chirality and obtain the true pose
		# newR, newT, alpha = checkChirality(Rset, Cset, points1, points2, alpha)

		# newH = np.hstack((newR, newT.reshape(3, 1)))
		# newH = np.vstack((newH, [0, 0, 0, 1]))
		# H = np.matmul(H, newH)
		# print(H)
		# T.append(H[0:3, 3])
		# print('--------------------------')

		# ----------------END OWN functions---------------------------

		# -------------------Inbuilt Usage----------------------------
		# use in-built function for comparison
		pixelsImg1 = np.int32(pixelsImg1)
		pixelsImg2 = np.int32(pixelsImg2)
		F_inbuilt, mask = cv2.findFundamentalMat(pixelsImg1, pixelsImg2, method=cv2.FM_RANSAC, param1=0.01, param2=.99)
		# print(F_inbuilt)
		inlierImg1Pixels = pixelsImg1[mask.ravel()==1]
		inlierImg2Pixels = pixelsImg2[mask.ravel()==1]
		E_cv2, mask = cv2.findEssentialMat(inlierImg1Pixels, inlierImg2Pixels, np.array(K), cv2.FM_RANSAC, prob=0.99,
										   threshold=0.01)
		# inlierImg1Pixels = pixelsImg1[mask.ravel() == 1]
		# inlierImg2Pixels = pixelsImg2[mask.ravel() == 1]
		_, r, t, _ = cv2.recoverPose(E_cv2, inlierImg1Pixels, inlierImg2Pixels, np.array(K))

		newH_inbuilt = np.hstack((r, t.reshape(3, 1)))
		newH_inbuilt = np.vstack((newH_inbuilt, [0, 0, 0, 1]))
		H_inbuilt = np.matmul(H_inbuilt, newH_inbuilt)
		print(H_inbuilt)
		T_inbuilt.append(H_inbuilt[0:3, 3])
		print('--------------------------')
		# -------------------END inbuilt usage------------------------

	# visualize the camera pose
	vizCameraPose(T, T_inbuilt)


cv2.destroyAllWindows()

if __name__ == "__main__":
	main()
