## **Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./image_proj/car_no_car.png
[image2]: ./image_proj/hog_car_no_car.jpg
[image3]: ./image_proj/sliding_windows.jpg
[image4]: ./image_proj/sliding_window.jpg
[image5]: ./image_proj/bboxes_and_heat.png
[image6]: ./image_proj/labels_map.png
[image7]: ./image_proj/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points

---
### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

HOG is extracted by get_hog_features, which lies in "training classifier" section, line 3 to line 23. 

First of all, cars and non-cars are read in section "Load data". Here is an example of car and a non-car. All the read of images in my code will be cv2, so the default color space is BGR.
Among all the image data, there are 8792 vehicles and 8968 non-vehicles, quite balanced in terms of the number of samples in each class.

![vehicle and non-vehicle][image1]

I tried various color spaces such as "LUV" / "HSV" / "BGR" / "YCrCb" with tweaking the orient being 8 / 9. the pix_per_cell is set to 8 and cell_per_block is set to 2.
so for the vehicle and non-vehicle image above, its HOG is as follows:

![HOG example][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I experimented different combinations of colorspace, channels(one channel vs. all) and orientation with images under test_images folder, the following parameters work well:
* orient = 9
* color_space = LUV
* hog_channel = 0
* pix_per_cell = 8
* cell_per_block = 2
with these parameters, the accuracy of SVM model is 0.9809

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a SVM model, in "training classifier" section, line 33 to line 45.
"LUV" color space was selected and other parameters mentioned above were used.
First of all, the data was shuffled and splited to 80% as training data and 20% as test data.
All data needs to be normalized as line 37 to line 40 did, then using LinearSVC to train the model.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The idea is to select a windows size and move the window step by step in the image and apply the model to predict if the image tile is vehicle or not. So slide_window is define in section "Sliding window" line 1 to line 26. It takes a region of interest, window size and overlap as parameters and generate the window lists for search_window to work on.

Here is the trick.

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

