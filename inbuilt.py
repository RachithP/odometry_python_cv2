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
import project2dTo3d

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
	for k in range(30, number_of_images + 3700):
		filenames.append(path + "/frame" + str(k) + ".png")

	images = []
	for filename in filenames:
		im_read = cv2.imread(filename, 0)
		# blur the image
		im_read = cv2.GaussianBlur(im_read, (5, 5), 0)
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
	axis.set_xlim(-10,500)
	axis.set_ylim(-10,500)
	axis.set_zlim(-20,1000)
	plt.show()


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


def findCommon(prevMatchedPixelLocations, currentMatchedPixelLocations, count):
	'''
	This returns the indices of features in the curent frame which were present in the previous frame
	:param prevMatchedPixelLocations:
	:param currentMatchedPixelLocations:
	:return:
	'''
	both = set(prevMatchedPixelLocations).intersection(currentMatchedPixelLocations)
	commonIndicesPrevFrame = [prevMatchedPixelLocations.index(x) for x in both]
	commonIndicesNewFrame = [currentMatchedPixelLocations.index(x) for x in both]

	return np.array(commonIndicesPrevFrame), np.array(commonIndicesNewFrame)


def main():
	# Parse input arguments
	Parser = argparse.ArgumentParser()
	Parser.add_argument('--Path', default="../Oxford_dataset/stereo/centre",
						help='Path to dataset, Default:../Oxford_dataset/stereo/centre')
	Parser.add_argument('--ransacEpsilonThreshold', default=2e-2,
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
	bgrImages = extractImages(new_path, 20)

	# extract calibration matrix
	K = dataPrep.extractCalibrationMatrix(path_to_model='./model')

	T = []
	R = []
	prevRT = np.diagflat([1, 1, 1, 1])
	alpha = 0
	H = np.identity(4)
	for imageIndex in range(len(bgrImages) - 1):
		print imageIndex
		# extract images from the input array
		pixelsImg1, pixelsImg2 = extractMatchFeatures(bgrImages[imageIndex], bgrImages[imageIndex + 1])
		# print K
		E_cv2, mask1 = cv2.findEssentialMat(np.array(pixelsImg1), np.array(pixelsImg2), method=cv2.RANSAC,focal=964.828979, pp=(643.788025,484.40799),prob=0.99,threshold=5.0)
		points, r, t, mask = cv2.recoverPose(E_cv2,np.array(pixelsImg1), np.array(pixelsImg2), focal=964.828979, pp=(643.788025,484.40799),mask=mask1)
		# print t
		
		# try:


		# 	E_cv2, mask1 = cv2.findFundamentalMat(pixelsImg1, pixelsImg2,np.array(K),cv2.FM_RANSAC)
		# 	_, r, t, _ = cv2.recoverPose(E_cv2, pixelsImg1, pixelsImg2, np.array(K), mask=mask1)
		# 	print r,t
		# 	# vizMatches(bgrImages[imageIndex],bgrImages[imageIndex + 1],pixelsImg1,pixelsImg2) # visualize the feature matches before RANSAC

		# 	F, inlierImg1Pixels, inlierImg2Pixels, _, _ = RANSAC(pixelsImg1, pixelsImg2, epsilonThresh, inlierRatioThresh)
		# 	# vizMatches(bgrImages[imageIndex], bgrImages[imageIndex + 1], inlierImg1Pixels, inlierImg2Pixels) # visualize after RANSAC

		# 	# check if obtained fundamental matrix is valid or not
		# 	checkF.isFValid(F, inlierImg1Pixels, inlierImg2Pixels, bgrImages[imageIndex], bgrImages[imageIndex + 1],
		# 					imageIndex)

		# 	# do this only once - first time
		# 	# if imageIndex == 0:
		# 	# get all poses (4) possible and E - Essential Matrix
		# 	E, Cset, Rset = extractPose.extractPose(F, K)

		# 	points1new = np.hstack((np.array(inlierImg1Pixels), np.ones((len(inlierImg1Pixels), 1)))).T
		# 	points2new = np.hstack((np.array(inlierImg2Pixels), np.ones((len(inlierImg2Pixels), 1)))).T
		# 	points1k = np.linalg.inv(K).dot(points1new)
		# 	points1 = points1k.T
		# 	points2k = np.linalg.inv(K).dot(points2new)
		# 	points2 = points2k.T

		# 	# this is to perform triangulation using LS method
		# 	# Xset = triangulation.linearTriangulationLS(K, Cset, Rset, inlierImg1Pixels, inlierImg2Pixels)

		# 	# this is to perform triangulation using Eigen method
		# 	# Xset = triangulation.linearTriangulationEigen(K, np.zeros((3, 1)), np.diag([1, 1, 1]), Cset, Rset,
		# 	# 											  inlierImg1Pixels, inlierImg2Pixels)
		# 	# check chirality and obtain the true pose
		# 	newR, newT,alpha = checkChirality(Rset,Cset,points1, points2,alpha)

		# 	newH = np.hstack((newR,newT.reshape(3,1)))
		# 	newH = np.vstack((newH,[0,0,0,1]))
		# 	H = np.matmul(H,newH)
		# 	print H
		# 	T.append(H[0:3,3])
		# 	# R.append(r)
		# 	# print('First camera position and orientation')
		# 	# print(c)
		# 	# print(r)

		# 	# # perform non-linear triangulation to obtain optimized set of world coordinates
		# 	# # I dont think we need this
		# 	# c_old = c
		# 	# r_old = r
		# 	#
		# 	# # Saving the matched pixel coordinates in the second image frame
		# 	# prevFrameMatchedPixels = inlierImg2Pixels
		# 	# prevFrameMatchedWorldPixels = X
		# 	#
		# 	# else:
		# 	# 	# Finding common values in previous frame matched points and the current matched points
		# 	# 	commonIndicesPrevFrame, commonIndicesNewFrame = findCommon(prevFrameMatchedPixels, inlierImg1Pixels, imageIndex)
		# 	# 	XCurr = prevFrameMatchedWorldPixels[commonIndicesPrevFrame]
		# 	# 	xCurr = np.array(inlierImg2Pixels)[commonIndicesNewFrame]
		# 	#
		# 	# 	# perform linear pnp to estimate new R,T - resection problem
		# 	# 	c_new, r_new = pnp.linear(xCurr, XCurr, K)
		# 	#
		# 	# 	print('c_new')
		# 	# 	print(c_new)
		# 	#
		# 	# 	# project points seen in 3rd image into world coordinates to use for next iteration
		# 	# 	prevFrameMatchedWorldPixels = np.squeeze(triangulation.linearTriangulationEigen(K, np.zeros((3, 1)), np.diag([1, 1, 1]), c_new, r_new,
		# 	# 												   inlierImg1Pixels, inlierImg2Pixels))
		# 	#
		# 	# 	# prevFrameMatchedWorldPixels = project2dTo3d.getWorldCoordinates(inlierImg2Pixels, K, r_new, c_new)
		# 	# 	prevFrameMatchedPixels = inlierImg2Pixels
		# 	# 	# c_old = c_new
		# 	# 	# r_old = r_new
		# 	#
		# 	# 	# refine the above value using non-linear triangulation
		# 	#
		# 	# 	# Combining RT and multiplying with the previous RT
		# newR, newT, prevRT = combineRT(r, t, prevRT)
		newH = np.hstack((r,t.reshape(3,1)))
		newH = np.vstack((newH,[0,0,0,1]))
		H = np.matmul(H,newH)
		print H
		T.append(H[0:3,3])

		# 	print('--------------------------')

		# 	# if imageIndex%100==0:
		# 	# # visualize the camera pose
		# except:
		# 	print 'not enough matches'
	vizCameraPose(R, T)
		# 	quit()

cv2.destroyAllWindows()

if __name__ == "__main__":
	main()
