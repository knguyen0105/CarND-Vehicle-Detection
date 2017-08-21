## CarND-Term 1 Vehicle Detection 
### Khanh Nguyen

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./pipeline.png
[image2]: ./image0000.png
[image3]: ./image1000.png
[image4]: ./car_hog.png
[image5]: ./non_car_hog.png
[image6]: ./slide_window.png
[video1]: ./project_video_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README


### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code is contained in 'Feature extraction and utilities' section of the notebook. Function `get_hog_features`.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![A car image][image2]
![A non-car image][image3]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

** Car Image **
![Hog feature of a car image in YCrCb space][image4]

** Non-Car Image ** 

![Hog feature of a non-car image in YCrCb space][image5]

#### 2. Explain how you settled on your final choice of HOG parameters.

My approach is trying out different parameters until my classifier reaches a good accuracy level. In this project, my classifier has 98% accuracy.

My final parameters for Hog are:

```python
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
```

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using two features: hog and spatial. I turned off hist feature. The code is contained in section "Train Classifier" of the notebook.

The accuracy is around 98%.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I implemented both normal sliding window and hog-subsampling. In the end, I use sub-sampling to detect car and produce the video output. 

The code for normal sliding window is contained in section "Car detection using sliding windows. Not used when creating video". I used three scales for windows: 96x96, 128x128, and 256X256

![Detect car using normal sliding window][image6]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on 3 scales using YCrCb 3-channel HOG features plus spatially binned color the feature vector, which provided a nice result.  Here is an example.:

![Detection pipleline][image1]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

The code to filter false positive is contained in the process function

```python
bboxes = []
for scale in [1.5, 1.7, 1.9]:
bboxes = bboxes + find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

prev_boxes.append(bboxes)

heat = np.zeros_like(img[:,:,0]).astype(np.float)
for bb in prev_boxes[-5:]:
heat = add_heat(heat,bb)

#Apply threshold to help remove false positives
heat = apply_threshold(heat,4)
```
In this function, calculate a heatmap from 5 consecutive frame and construct my boxes at threshold of 4.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I spent most time on
- choosing the parameters and selecting features for the classifier. Initially, I used all three features, my best result was around 94%. It took me a while until I thought about turning off one or two features. 
- getting hog subsampling to work correctly. The code provided in the lecture use YCrCb. It tooks me a while to realize that my classifier was trained on RGB. 

I believe filtering false positive is one of potential weak links in my pipeline. Two parameters to tweak in my implementation are the number of look-back frames and the threshold. In the project, I construct a heatmap from 5 consecutive frame and require a threshold of 4, meaning that a pixel only belongs to a car if it belongs to at least 4 positive detections in 5 frames. If these numbers are high, I will reduce the number of false positive. However the risk is that I will also filter out true positive. If I keep these numbers low, I can detect more car, but also more false positives. 


