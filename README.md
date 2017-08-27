# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

My submission of Udacity Self Driving Car Term1, Project 5. This project uses machine learning (SVM) and computer vision (HOG feature) to classifies and tracks vehicles in traffic. 

A sample video output:

![Sample Video Output](project_video_output.gif)

The Project
---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

Here are links to the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train your classifier.  These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself.   You are welcome and encouraged to take advantage of the recently released [Udacity labeled dataset](https://github.com/udacity/self-driving-car/tree/master/annotations) to augment your training data.  

Submission
---

- [IPython notebook](sdc1_p5_object_tracking.ipynb)
- [Project writeup](writeup.md)
- [Output video](project_video_output.mp4)

You might need to follow instructions on [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit/) before running the notebook.

All Udacity SDC Term 1 Projects
---

- [Project 1: Finding Lane Lines](https://github.com/knguyen0105/CarND-LaneLines-P1)
- [Project 2: Traffic Sign Classifier](https://github.com/knguyen0105/CarND-Traffic-Sign-Classifier)
- [Project 3: Behavior Cloning ](https://github.com/knguyen0105/CarND-Behavioral-Cloning-P3v)
- [Project 4: Advanced Lane Finding ](https://github.com/knguyen0105/CarND-Advanced-Lane-Lines)
- [Project 5: Vehicle Detection](https://github.com/knguyen0105/CarND-Vehicle-Detection)
