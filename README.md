# Project: Traffic Sign Recognition Program

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

# Overview

In this project, I used deep neural networks and convolutional neural networks to classify traffic signs. I trained and validated a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, I tried out the model on images of German traffic signs that I found on the web.

# Dependencies

This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab environment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

# The Project

The goals / steps of this project were the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[vis_train]: ./images/visualization_train.jpg "Training Set Visualization"
[vis_valid]: ./images/visualization_valid.jpg "Validation Set Visualization"
[vis_test]: ./images/visualization_test.jpg "Test Set Visualization"
[img_rgb]: ./images/image_rgb.jpg "RGB image"
[img_gray]: ./images/image_gray.jpg "RGB image"
[img_tr_before]: ./images/image_translate_before.jpg "Translate: Before"
[img_tr_after]: ./images/image_translate_after.jpg "Translate: After"
[img_rot_before]: ./images/image_rotate_before.jpg "Rotate: Before"
[img_rot_after]: ./images/image_rotate_after.jpg "Rotate: After"
[img_sc_before]: ./images/image_scale_before.jpg "Scale: Before"
[img_sc_after]: ./images/image_scale_after.jpg "Scale: After"
[img_bl_before]: ./images/image_blur_before.jpg "Blur: Before"
[img_bl_after]: ./images/image_blur_after.jpg "Blur: After"
[img_pers_before]: ./images/image_pers_before.jpg "Pers. Transformation: Before"
[img_pers_after]: ./images/image_pers_after.jpg "Pers. Transformation: After"
[img_aug_before]: ./images/image_aug_before.jpg "Augment: Before"
[img_aug_after]: ./images/image_aug_after.jpg "Augment: After"
[web_test_set]: ./images/web_test_set.jpg "Web Test Set"

## Data Set Summary & Exploration

### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is bar charts showing the number of sample for each class.

Training set:

![text][vis_train]

Validation set:

![text][vis_valid]

Test set:

![text][vis_test]

## Design and Test a Model Architecture

### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

**Data pre-processing pipeline**

As the first step, I decided to convert the images to grayscale because after several experiments I found out that color information does not really help the NN to train. The NN using graycale images gave better validation accuracy.

Here is an example of a traffic sign image before and after grayscaling.

![text][img_rgb]![text][img_gray]

As the last step, I normalized the image data because it helps to prevent numerical issues in calculating a loss function, and helps a NN to train faster.

**Data augmentation**

The main reason why I decided to generate additional data was the fact that the number of samples of each label in the training set is significantly different. For example, there're 180 samples for the class 0 and 2010 samples for the class 2. I wanted the training set to has approximately the same number of samples of each class. Also, as a car moves, some images can be slightly blurred and/or viewed by the car's camera from different angles. I wanted the NN to be able to work well under such conditions.

To add more data to the the data set, I used the following techniques:

- Random transition on -2..2 pixels along the x- and y-axis

  ![text][img_tr_before]![text][img_tr_after]

- Random rotation in -10..10 degrees

  ![text][img_rot_before]![text][img_rot_after]

- Scaling by a random factor of 1..1.2

  ![text][img_sc_before]![text][img_sc_after]

- Blurring with a random kernel size of 1 or 3

  ![text][img_bl_before]![text][img_bl_after]

- Random perspective transformation

  ![text][img_pers_before]![text][img_pers_after]

The method `def augment(img)` applies from 3 to 5 randomly selected transformations described above to the image passed in.

Here is an example of an original image and an output of the `augment` method:

![text][img_aug_before]![text][img_aug_after]

The difference between the original data set and the augmented data set is the following:

- The augmented data set has the same number of samples of each class.
- The augmented data set consists of 430,000 images (10,000 images per class) which is 12 times more than the original data set size.

### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:-----------------:|:-------------------------------------------:|
| Input         		| 32x32x1 Grayscale                           |
| Convolution 5x5   | 1x1 stride, valid padding, outputs 28x28x6  |
| RELU					|						                          |
| Max pooling	      	| 2x2 stride,  outputs 14x14x6                |
| Convolution 5x5	| 1x1 stride, valid padding, outputs 10x10x16 |
| RELU					|						                          |
| Max pooling	      	| 2x2 stride,  outputs 5x5x16                 |
| Flatten				| outputs 1x400                               |
| Fully connected	| outputs 1x1024                              |
| RELU					|						                          |
| Dropout				| 50% dropout	                                 |
| Fully connected	| outputs 1x512                               |
| RELU					|						                          |
| Dropout				| 50% dropout	                                 |
| Fully connected	| outputs 1x43                                |
| Softmax				|                                             |


### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the Adam optimizer with the following hyper-parameters:

- The number of epochs of 50
- The batch size of 256
- The dropout rate of 50%
- Learning rate of 0.001 decreasing every 10 epochs by the factor of 2.

### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:

* training set accuracy of 1.000
* validation set accuracy of 0.979
* test set accuracy of 0.962

To be able to achieve the result above I ran plenty experiments. The most important steps and decisions are listed below:

- As the first architecture I chose LeNet 5 shown in the classroom as it was suggested in the project's description.
- The biggest problem of this architecture was overfitting: the training set accuracy was 1.00 while the validation accuracy was under 0.93.
- In order to prevent overfitting I've added dropout layers after the first and second fully connected layers. Experiments showed that a dropout rate of 25% is not enough and the rate of 50% works much better.
- After the step above I could achieve the desired validation set accuracy > 0.93, but I wanted to achieve at least 0.95 accuracy. After several experiments with the width of the fully connected layers I decided to set it as the following:
	- the width of the first fully connected of 1024
	- the width of the seconds fully connected of 512
- The batch size of 256 was chosen after a series of experiments with different sizes (64, 128, 256) as the best performing.
- I also ran several experiments for choosing the learning rate. The rate of 0.001 gave the best validation accuracy. However, using a constant learning rate value resulted in jittering of the validation accuracy in the end of training. To solve this issue I decided to decrease the learning rate dynamically, dividing it by 2 every 10 epochs.


## Test a Model on New Images

### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![text][web_test_set]

The seconds, fourth and fifth images might be difficult to classify because they are "perspective-transformed". The third image can also be challenging to classify as it contains a contrast object in the bottom right corner.

### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Keep right      		| Keep right |
| Speed limit (30km/h) | Speed limit (30km/h) |
| Priority road			| Priority road |
| Stop	      				| Stop				|
| Speed limit (50km/h) | Speed limit (50km/h) |

The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 96.2%.

### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For the first image, the model is absolutely sure that this is a "keep right" sign (probability of 1.0), and the image does contain a "keep right". The top five soft max probabilities were

| Probability  |     Prediction	    |
|:------------:|:-------------------:|
| 1.0 		 	 | Keep right       	|
| 0.0     	 | Speed limit (20km/h) |
| 0.0			 | Speed limit (30km/h) |
| 0.0	     	 | Speed limit (50km/h) |
| 0.0		 	 | Speed limit (60km/h) |

For the second image, the model is relatively sure that this is a "Speed limit (30km/h)" sign (probability of 0.973), and the image does contain a "Speed limit (30km/h)" sign. The top five soft max probabilities were

| Probability.      |     Prediction	    |
|:-----------------:|:-------------------:|
| 9.7316957e-01 		| Speed limit (30km/h) |
| 2.6274802e-02 		| Speed limit (20km/h) |
| 3.4359802e-04		| Children crossing |
| 1.4032122e-04	   	| Road work |
| 7.0213719e-05		| Right-of-way at the next intersection |

For the third image, the model is absolutely sure that this is a "Priority road" sign (probability of 1.0), and the image does contain a "Priority road" sign. The top five soft max probabilities were

| Probability.      |     Prediction	    |
|:-----------------:|:-------------------:|
| 1.000000e+00 		| Priority road       |
| 3.730391e-30 		| Roundabout mandatory |
| 8.071415e-31		| Stop |
| 0.000000e+00	   	| Speed limit (20km/h) |
| 0.000000e+00		| Speed limit (30km/h) |

For the fourth image, the model is absolutely sure that this is a "Stop" sign (probability of 1.0), and the image does contain a "Stop" sign. The top five soft max probabilities were

| Probability.      |     Prediction	    |
|:-----------------:|:-------------------:|
| 1.0000000e+00 		| Stop       |
| 1.8616805e-14 		| Turn left ahead |
| 1.3855617e-14		| Go straight or right |
| 7.1690952e-16	   	| Keep right |
| 2.1260214e-17		| Yield |

For the fifth image, the model is relatively sure that this is a "Speed limit (50km/h)" sign (probability of 0.56), and the image does contain a "Speed limit (50km/h)" sign. The top five soft max probabilities were

| Probability.      |     Prediction	    |
|:-----------------:|:-------------------:|
| 0.56035084 		| Speed limit (50km/h)       |
| 0.38740516 		| No passing for vehicles over 3.5 metric tons |
| 0.03605079		| Speed limit (60km/h) |
| 0.00837479	   	| Speed limit (80km/h) |
| 0.00780723		| No passing |
