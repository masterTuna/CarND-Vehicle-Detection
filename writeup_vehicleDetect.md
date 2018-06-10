## **Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_no_car.png
[image2]: ./output_images/hog_car_no_car.png
[image3]: ./output_images/sliding_windows.png
[image4]: ./output_images/pipeline_results.png
[image5]: ./output_images/heatmap.png
[image6]: ./output_images/labels_map.png
[video1]: ./test_video_boxes.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points

---
### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

HOG is extracted by get_hog_features, which lies in "training classifier" section, line 3 to line 23. 

First of all, cars and non-cars are read in section "Load data". Here is an example of car and a non-car. All the read of images in my code will be cv2, so the default color space is BGR.
Among all the image data, there are 8792 vehicles and 8968 non-vehicles, quite balanced in terms of the number of samples in each class.

![vehicle and non-vehicle][image1]

I tried various color spaces such as "LUV" / "HSV" / "BGR" / "YCrCb" with tweaking the orient being 8 / 9. the pix_per_cell is set to 8 and cell_per_block is set to 2. From my experiments, both "LUV" and "YCrCb" show very good detection results.
so for the vehicle and non-vehicle image above, its HOG is as follows:

![HOG example][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I experimented different combinations of colorspace, channels(one channel vs. all) and orientation with images under test_images folder, the following parameters work well:
* orient = 9
* color_space = YCrCb
* hog_channel = 0
* pix_per_cell = 8
* cell_per_block = 2
with these parameters, the accuracy of SVM model is 0.9856. Note that for version higher than 0.15, hog function starts to use "L2-Hys" rather than "L1" for lock_norm, which in my experiments shows weaker detection. So I choose not to use "L2-Hys".

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a SVM model, in "training classifier" section, line 33 to line 45.
"YCrCb" color space was selected and other parameters mentioned above were used.
First of all, the data was read in and to better serve the training, for each image, I flipped it and used to train the model(equivalent to have some cases that cars are from left side, in this case). Images were shuffled and splited to 80% as training data and 20% as test data.
All data needs to be normalized as line 37 to line 40 did, then using LinearSVC to train the model.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The idea is to select a windows size and move the window step by step in the image and apply the model to predict if the image tile is vehicle or not. So slide_window is define in section "Sliding window" line 1 to line 26. It takes a region of interest, window size and overlap as parameters and generate the window lists for search_window to work on.

Here is the trick. Since the size of vehicle are different for ones far away and close, I decided to use different window size for detection. Here is the windows setting:
* The region of interest is roughly from y=400px to y=660px vertically and from left to right horizontally. Five different sizes windows are applied:
    *  180 x 180, applied to y=(400px,660px)
    *  150 x 150, applied to y=(400px,600px)
    *  120 x 120, applied to y=(400px,550px)
    *  80 x 80, applied to y=(400px,500px)
    *  64 x 64, applied to y=(400px,500px)

* From the experiment, overlap=0.8 shows the best result.

![all sliding windows][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

The following image shows how the pipeline is working. The vehicle is marked in red box.
As described above, to get better performance of the classifier, binned color and histrogram of color feature as well as HOG features are used with channel 0. Different color spaces are experimented, and various windows size are combined. The overall results are overall good, with rare false positive.

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

The video of vehicle detection is located in ./test_video_boxes.mp4. 


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

For each frame of the video, I use my svm model to search the vehicles on the road. The code lies in "Pipeline for Video" section, line 13. All the parameters are as metioned before. To draw reasonable box on the vehicles, I created a heatmap for the positive detections and apply a threshold to the heatmap. Heatmap is built based on all the boxes that has positive results. After threshold was applied, only strong positives were left. I applied `scipy.ndimage.measures.label()` to the heatmap and identified individual blobs in the heatmap, which indicated each vehicle. The code lies in "Filter:heatmap and threshold" section, with three functions: `add_heat`, `apply_threshold` and `draw_labeled_bboxes`. 

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:
![heatmap][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![labels_map][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![pipeline results][image4]



---

### Discussion

The pipeline works well for the video, but I figure it might fail for some senarios such as low light shade area and some structures that appear to be different shapes which may confuse the pipeline. More data under various condistions and some data augmentation tricks may help improve the robustness of the pipeline. Beside, tweaking the sliding windows and coverage/overlap also helps.
