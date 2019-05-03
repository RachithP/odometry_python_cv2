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
		im_read = cv2.imread(filename, -1)
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

	fig = plt.figure()
	axis = fig.add_subplot(1, 1, 1, projection="3d")
	axis.scatter(T[:, 0].flatten(), T[:, 1].flatten(), T[:, 2].flatten(), marker=".")
	axis.set_xlabel('x')
	axis.set_ylabel('y')
	axis.set_zlabel('z')
	plt.title('Camera movement')
	plt.show()

def combineRT(r,t,prevRT):
	temp = np.hstack((r,t.reshape(3,1)))
	RT = np.vstack((temp,np.array([0,0,0,1])))
	RT = np.matmul(prevRT,RT)
	prevRT = RT.copy()
	newR = np.array(RT[0:3,0:3])
	newT = np.array(RT[0:3,3])
	return newR,newT,prevRT

def plotLine(image,a,b,c):
	plt.imshow(image)
	x = np.linspace(0,image.shape[1],image.shape[1])
	y = -((a*x)+c)/b
	plt.plot(x,y, linewidth=1.0)
	plt.show()

def main():
	# Parse input arguments
	Parser = argparse.ArgumentParser()
	Parser.add_argument('--Path', default="../Oxford_dataset/stereo/centre",
						help='Path to dataset, Default:../Oxford_dataset/stereo/centre')
	Parser.add_argument('--ransacEpsilonThreshold', default=0.15,
						help='Threshold used for deciding inlier during RANSAC, Default:0.15')
	Parser.add_argument('--inlierRatioThreshold', default=0.8,
						help='Threshold to consider a fundamental matrix as valid, Default:0.8')

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
	prevRT = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
	for imageIndex in range(len(bgrImages) - 1):
		pixelsImg1, pixelsImg2 = extractMatchFeatures(bgrImages[imageIndex], bgrImages[imageIndex + 1])
		# vizMatches(bgrImages[imageIndex],bgrImages[imageIndex + 1],pixelsImg1,pixelsImg2)

		F, inlierImg1Pixels, inlierImg2Pixels, _, _ = RANSAC(pixelsImg1, pixelsImg2, epsilonThresh, inlierRatioThresh)
		# vizMatches(bgrImages[imageIndex], bgrImages[imageIndex + 1], inlierImg1Pixels, inlierImg2Pixels)

		# this is to perform triangulation using LS method
		# world_coordinates = triangulation.linearTriangulationLS(K, inlierImg1Pixels, inlierImg2Pixels)

		# this is to perform triangulation using Eigen method
		world_coordinates = triangulation.linearTriangulationEigen(K, inlierImg1Pixels, inlierImg2Pixels)

		t, r = extractPose.extractPose(F, K, world_coordinates)

		# Combining RT and mulitplying with the previous RT
		newR,newT,prevRT = combineRT(r,t,prevRT)
		
		plotLine(image,a,b,c)
		
		T.append(newT)
		R.append(newR)
		
	# cv2.destroyAllWindows()

	# visualize the camera pose


if __name__ == "__main__":
	main()
