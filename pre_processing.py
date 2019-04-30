#!/usr/bin/env python2
import cv2
import numpy as np
from ReadCameraModel import ReadCameraModel
import glob
from UndistortImage import UndistortImage

__author__ = 'rachith'


def preProcessData(path_to_model, path_to_images):
	'''
	Here, we undistort the dataset images
	:param path_to_model:
	:param path_to_images:
	:return:
	'''

	# Read camera parameters
	fx, fy, cx, cy, G_camera_image, LUT = ReadCameraModel(path_to_model)

	# iterate through each image, convert to RGB, undistort(function takes all channels in input)
	images = glob.glob(path_to_images+"/*.png")
	images.sort()
	for cnt, image in enumerate(images):
		frame = cv2.imread(image, -1)
		frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BayerGR2BGR)
		undistorted_image = UndistortImage(frame_RGB, LUT)
		cv2.imwrite(path_to_images + "/../undistort/frame" + str(cnt) + ".png", undistorted_image)

def main():
	preProcessData(path_to_model='./model', path_to_images='./stereo/centre')


if __name__ == "__main__":
	main()