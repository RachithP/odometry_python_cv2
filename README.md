# Odometry!
Odometry using *SIFT* feature extraction, feature matching, localization across frames.

## Instructions to use `visualOdometry.py`
After you clone this repo, create a folder inside this repo named 'undistort'. Then run `visualOdometry.py` and provide the path to your dataset(images) as the first argument. Example:
```bash
$ python visualOdometry.py -h
usage: visualOdometry.py [-h] [--Path PATH]
                         [--ransacEpsilonThreshold RANSACEPSILONTHRESHOLD]
                         [--inlierRatioThreshold INLIERRATIOTHRESHOLD]

optional arguments:
  -h, --help            show this help message and exit
  --Path PATH           Path to dataset,
                        Default:../Oxford_dataset/stereo/centre
  --ransacEpsilonThreshold RANSACEPSILONTHRESHOLD
                        Threshold used for deciding inlier during RANSAC,
                        Default:0.9
  --inlierRatioThreshold INLIERRATIOTHRESHOLD
                        Threshold to consider a fundamental matrix as valid,
                        Default:0.9

```
After this file execution, you will have undistorted BGR images in the undistort folder. The code is written such that it takes values from undistort folder and processes it.
