# Sdnd-P5-Vehicle-Detection
CarND-Vehicle-Detection

The goals / steps of this project are the following:
Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
Apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
Estimate a bounding box for vehicles detected.

Please see https://github.com/BCChangEBFE/Sdnd-P5-Vehicle-Detection/blob/master/p5.ipynb for detailed code with comments.

[//]: # (Image References)

[image1]: ./output_images/HOG_TreeTop_Sky.png "HOG Not Car"
[image2]: ./output_images/HOG_Car.png "HOG Car"
[image3]: ./output_images/VehicleSearch.png "Vehicle Search"

[video1]: ./project_result.mp4 "Vehicle Detection Video"

---

## 1. Explain how (and identify where in your code) you extracted HOG features from the training images. Explain how you settled on your final choice of HOG parameters.

HOG features is extracted per winsow using the function get_hog_features(). This is just the most streight method to extract HOG. 
All 3 channls from the RGB colour space is used as in input to HOG. 
From experiment, it is found that using all 3 channels yields a better result. Note that there are some colour spaces such as LUV where feeding all three channels into HOG extraction would cause a numerical error.

![alt text][image1]
![alt text][image2]

## 2. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

Linear SVM is chosen as my classifier in this project. Its relatively high accuracy and fast computation makes it a good fit for this job. There are three feature vectors that are fed to the classifier here, 
	1. HOG Features
	2. Spatial Features
	3. Colour Histogram Features

## 3. Describe how (and identify where in your code) you implemented a sliding window search. How did you decide what scales to search and how much to overlap windows?

I have decided to use 4 tiers of sliding windows. Knowing that we are only interested in the bottom half of the image, I divide that into 4 different distance categories. For each of the 4 categories I estimated the size of a vehicle to determine the windows size needed. Next, to decide on how much overlap is needed was more of a trial and error process. Having too much overlap would be slow and create a heat map with too large of an excitation number in many pixels. Larger windows needs to have more ovelap to allow for higher heat map excitation to identify larger cars. Smaller windows can get by with less overlaps since the slightly larger windows would tend to trigger a vehicle classification match as well. The four categories of windows are.  
  1. 32x32: overlap 0.5
  2. 64x64: overlap 0.75
  3. 96x96: overlap 0.75
  4. 128x128: overlap 0.80

The code to generate sliding window frames is written in slide_window() function.

For each of the search windows, classifier features is extracted and used to decide if the windowed image is categorized as a vehicle ornot. In general the search_windows() function runs the following logic pipeline 
  1. Calls slide_window() to extract the windowed image
  2. Feeds the windowed image to single_img_features() to generate features for the classifier
  3. Features generated is then fed to the classifier clf.predict() to determine if it is a match or now

## 4. Show some examples of test images to demonstrate how your pipeline is working. How did you optimize the performance of your classifier?
Classifier performance is optimized by pre-transforming all the colours if needed before feeding to the 3 major feature generation algorithm. A further optimization is possible if hog features is also pre calculated for each image instead of each windowed image, but this is not implemented in this version yet. 

![alt text][image3]

## 5. Provide a link to your final video output. Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

![alt text][video1]

## 6. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.
  First the bounding boxes are processed into a heatmap; for each time a pixel is included in a bounding box, the pixel get a +1 in the heatmap. The heatmap is then thresholded to retain only pixels with high heatmap excitation number. Next scipy.ndimage.measurements.label() is used to identify and create the resulting bouding vehicle box. The code bloxs corresponding to thie process is implemented in the following small functinos,
  - increment_heatmap()
  - make_heatmap()
  - apply_threshold()
  - draw_labeled_bboxes()
  - heatmap_to_windows()

Where the above helper functions are called in detect_cars.annotate_image()

## 7. Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail? What could you do to make it more robust?
  A lot of time was spend trial and error and tuning varialbe defined in secion *Definition of Classifier Variables* of the p5.idynb
  Deciding on the windows to be used for the pipeline also took quite a bit of effort. 
  
  To further trying to take advantage of historical data, the final pipeline implmted in annotate_image() is actually implemented in a class. Hence some more methods of averaging and smoothing of the results can also be experimented to make the pipeline more robust
  
  The result is not exactly robust as there are frames that not the complete vehicle body is identified in a bouding box. Also the bounding boxes are sometimes flickering. It could also be possible to implement the classifier with a different algorith. For example, deep learning could replace SVM here to provide a more robust identificatino of car pixles.
