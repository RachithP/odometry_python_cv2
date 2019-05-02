#!/usr/bin/env python2

'''
ENPM 673 Spring 2019: Robot Perception
Project 5: Visual Odometry

Author:
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
from matchedFeaturesCoordinates import extractMatchFeatures

def extractImages(path, number_of_images):
	'''
	In this function we store undistorted images in an array
	:param path:
	:return:
	'''
	# Read and store all images in the input folder
	filenames = glob.glob(path + "/*.png")
	filenames.sort()

	# Removing first 30 images because it is too bright
	del filenames[:30]

	images = []
	counter = 0
	for filename in filenames:
		images.append(cv2.imread(filename, -1))
		counter += 1
		if number_of_images != None:
			if counter > number_of_images:
				break

	print('Done extracting images....')

	return images


def main():
	# Parse input arguments
	Parser = argparse.ArgumentParser()
	Parser.add_argument('--Path', default="../Oxford_dataset/stereo/centre",
						help='Path to dataset, Default:../Oxford_dataset/stereo/centre')
	Parser.add_argument('--ransacEpsilonThreshold', default=0.9,
						help='Threshold used for deciding inlier during RANSAC, Default:0.9')
	Parser.add_argument('--inlierRatioThreshold', default=0.9,
						help='Threshold to consider a fundamental matrix as valid, Default:0.9')

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
	bgrImages = extractImages(new_path, 100)

	# extract calibration matrix
	K = dataPrep.extractCalibrationMatrix(path_to_model='./model')

	R = []
	T = []
	for imageIndex in range(len(bgrImages) - 1):
		pixelsImg1, pixelsImg2 = extractMatchFeatures(bgrImages[imageIndex], bgrImages[imageIndex + 1])
		F = RANSAC(pixelsImg1, pixelsImg2, epsilonThresh, inlierRatioThresh)

		t1, r1, t2, r2, t3, r3, t4, r4 = extractPose.extractPose(F, K)

		# print(t1)
		# print('2')
		# print(t2)
		# print('3')
		# print(t3)
		# print('4')
		# print(t4)


if __name__ == "__main__":
	main()
