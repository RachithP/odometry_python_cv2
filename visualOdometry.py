#!/usr/bin/env python2

'''
ENPM 673 Spring 2019: Robot Perception
Project 5 Odometry

Author:
Ashwin Varghese Kuruttukulam(ashwinvk94@gmail.com)
Rachith Prakash (rachithprakash@gmail.com)
Graduate Students in Robotics,
University of Maryland, College Park
'''


import argparse
import pre_processing as prep
import match_features

def main():

	# Parse input arguments
	Parser = argparse.ArgumentParser()
	Parser.add_argument('--Path', default="../Oxford_dataset/stereo/centre", help='Path to dataset, Default:../Oxford_dataset/stereo/centre')
	Args = Parser.parse_args()
	path = Args.Path

	# Pre-process the data and obtain calibration matrix
	# prep.preProcessData(path_to_model='./model', path_to_images=path)
	K = prep.extractCalibrationMatrix(path_to_model='./model')
	match_features.main(["./undistort", K, 0.9, 0.9])



if __name__ == "__main__":
	main()