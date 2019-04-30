#!/usr/bin/env python2
import cv2
import numpy as np
import ReadCameraModel
import glob

__author__ = 'rachith'



def preProcessData(path_to_model):
	'''
	In this function, we pre-process the data
		1. read camera parameters from the file,
		2. Undistort the image using the previously obtained calibration matrix.
	:return:
	'''

	# Read camera parameters
	fx, fy, cx, cy, G_camera_image, LUT = ReadCameraModel(path_to_model)

	# iterate through each image, convert to BGR, undistort(function takes BGR as input)
	# Since there are a ton of images, it would be easier to read, write back the undistorted image rather than storing all the images in an array


def convert2BGR(path_to_images):
	'''
	In this function, we read images and convert the Bayer format image to BGR
	:param path_to_images:
	:return:
	'''

	images = glob.glob(path_to_images+"/*.png")
	images.sort()
	for cnt, image in enumerate(images):
		frame = cv2.imread(images[0],0)
		frame_BGR = cv2.cvtColor(frame, cv2.COLOR_BayerGR2BGR)
		cv2.imshow('image', frame_BGR)
		cv2.waitKey(0)
		quit()
		frame_BGR = cv2.cvtColor(frame, cv2.COLOR_BayerGR2BGR)
		cv2.imwrite(path_to_images+"/../undistort/frame/"+cnt+".png", frame_BGR)


def main():

	convert2BGR(path_to_images='./stereo/centre')

	# preProcessData(path_to_model='./model')

if __name__ == "__main__":
	main()