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
from matchedFeaturesCoordinates import extractMatchFeatures
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

	# Uncomment this to run on all the images
	# for k in range(30,len(filesnumber)):
	# Removing first 30 images because it is too bright
	for k in range(30, number_of_images + 200):
		filenames.append(path + "/frame" + str(k) + ".png")

	images = []
	for filename in filenames:
		im_read = cv2.imread(filename, 0)
		images.append(im_read)

	print('Done extracting images....')

	return images


def vizCameraPose(R, T):
	'''
	Function to visualize camera movement
	:param R:
	:param T:
	:return:
	'''
	T = np.array(T)

	fig = plt.figure(1)
	axis = fig.add_subplot(1, 1, 1, projection="3d")
	axis.scatter(T[:, 0].flatten(), T[:, 1].flatten(), T[:, 2].flatten(), marker=".")
	axis.set_xlabel('x')
	axis.set_ylabel('y')
	axis.set_zlabel('z')
	plt.title('Camera movement')
	plt.pause(0.1)


def combineRT(r, t, prevRT):
	'''
	This function calculates the new R, T by multiplying present H (transformation matrix) with transformation matrix
	of previous frame w.r.t base frame
	:param r:
	:param t:
	:param prevRT:
	:return:
	'''
	temp = np.hstack((r, t.reshape(3, 1)))
	RT = np.vstack((temp, np.array([0, 0, 0, 1])))
	RT = np.matmul(prevRT, RT)
	prevRT = RT.copy()
	newR = np.array(RT[0:3, 0:3])
	newT = np.array(RT[0:3, 3])
	return newR, newT, prevRT


'''
This returns the indices in the curent frame which were present in the previous frame
'''
def findCommon(prevMatchedPixelLocations,currentMatchedPixelLocations):

	both = set(prevMatchedPixelLocations).intersection(currentMatchedPixelLocations)
	commonIndicesPrevFrame = [prevMatchedPixelLocations.index(x) for x in both]
	commonIndicesNewFrame = [currentMatchedPixelLocations.index(x) for x in both]

	return np.array(commonIndicesPrevFrame),np.array(commonIndicesNewFrame)


def main():
	# Parse input arguments
	Parser = argparse.ArgumentParser()
	Parser.add_argument('--Path', default="../Oxford_dataset/stereo/centre",
						help='Path to dataset, Default:../Oxford_dataset/stereo/centre')
	Parser.add_argument('--ransacEpsilonThreshold', default=1e-2,
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
	prevRT = np.diag([1, 1, 1, 1])

	for imageIndex in range(50, len(filesnumber) - 60):
		print(imageIndex)

		# bgrImages, vizImages = extractImages(new_path, 20)
		# ------------Process pair of images -------------------------------------
		img1 = cv2.imread(new_path + "/frame" + str(imageIndex) + ".png", -1)
		# histogram equalization of the image
		equ1 = cv2.equalizeHist(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY))
		# blur the image
		img1_gray = cv2.GaussianBlur(equ1, (5, 5), 0)
		img2 = cv2.imread(new_path + "/frame" + str(imageIndex + 1) + ".png", -1)
		# histogram equalization of the image
		equ2 = cv2.equalizeHist(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY))
		# blur the image
		img2_gray = cv2.GaussianBlur(equ2, (5, 5), 0)

		# extract images from the input array
		pixelsImg1, pixelsImg2 = extractMatchFeatures(bgrImages[imageIndex], bgrImages[imageIndex + 1])
		# vizMatches(bgrImages[imageIndex],bgrImages[imageIndex + 1],pixelsImg1,pixelsImg2) # visualize the feature matches before RANSAC

		F, inlierImg1Pixels, inlierImg2Pixels, _, _ = RANSAC(pixelsImg1, pixelsImg2, epsilonThresh, inlierRatioThresh)
		# vizMatches(bgrImages[imageIndex], bgrImages[imageIndex + 1], inlierImg1Pixels, inlierImg2Pixels) # visualize after RANSAC

		# check if obtained fundamental matrix is valid or not
		checkF.isFValid(F, inlierImg1Pixels, inlierImg2Pixels, bgrImages[imageIndex], bgrImages[imageIndex + 1],
						imageIndex)

		# do this only once - first time
		if imageIndex == 0:
			# get all poses (4) possible
			Cset, Rset = extractPose.extractPose(F, K)

			# this is to perform triangulation using LS method
			# Xset = triangulation.linearTriangulationLS(K, Cset, Rset, inlierImg1Pixels, inlierImg2Pixels)

			# this is to perform triangulation using Eigen method
			Xset = triangulation.linearTriangulationEigen(K, np.zeros((3, 1)), np.diag([1, 1, 1]), Cset, Rset,
														  inlierImg1Pixels, inlierImg2Pixels)

			# check chirality and obtain the true pose
			c, r, X = checkChirality(Cset, Rset, Xset)
			T.append(c)
			R.append(r)
			print('First camera position and orientation')
			print(c)
			print(r)

			# perform non-linear triangulation to obtain optimized set of world coordinates
			# I dont think we need this
			c_old = c
			r_old = r

			# Saving the matched pixel coordinates in the second image frame
			prevFrameMatchedPixels = inlierImg2Pixels
			prevFrameMatchedWorldPixels = X

		else:
			# Finding common values in previous frame matched points and the current matched points
			commonIndicesPrevFrame,commonIndicesNewFrame = findCommon(prevFrameMatchedPixels,inlierImg1Pixels)
			XCurr = prevFrameMatchedWorldPixels[commonIndicesPrevFrame]
			xCurr = np.array(inlierImg2Pixels)[commonIndicesNewFrame]

			# perform linear pnp to estimate new R,T - resection problem
			c_new, r_new = pnp.linear(xCurr, XCurr, K)
			quit()

			print('c_new')
			print(c_new)

			# project points seen in 3rd image into world coordinates to use for next iteration
			X_new = triangulation.linearTriangulationEigen(K, np.zeros((3, 1)), np.diag([1, 1, 1]), c_new, r_new, inlierImg1Pixels, inlierImg2Pixels)

			X = X_new

			# I dont think we need this
			c_old = c_new
			r_old = r_new
		# refine the above value using non-linear triangulation

		# Combining RT and multiplying with the previous RT
		# newR, newT, prevRT = combineRT(r, t, prevRT)
			T.append(c_new)
			R.append(r_new)

		# # cv2.imshow('prev image',bgrImages[imageIndex])
		# cv2.imshow('prev image',bgrImages[imageIndex+1])
		# cv2.waitKey(0)
		print('--------------------------')

		# visualize
		# vizCameraPose(R, T)
	# plt.show()

# cv2.destroyAllWindows()

# visualize the camera pose


if __name__ == "__main__":
	main()
