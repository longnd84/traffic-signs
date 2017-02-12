#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

The following resources are consulted when doing this project: 
* LeNet lab for character recognition [LeNet]
* Traffic Sign recognition with Multi-Scale convolutional networks - Sermanet and Lecun [Traffic-Sign-Convolution]
* Couple of general internet resources, including Campushippo. Relevant sources are quoted in the code segments. 

You're reading it! and here is a link to my [project code](https://github.com/longnd84/Traffic-signs/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the third code cell of the IPython notebook.  

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 (training set and validation sets are pre-splitted with the updated dataset)
* The size of validation set is 4410
* The size of test set is 12630 
* The shape of a traffic sign image is 32 x 32 x 3 color channels (RGB)
* The number of unique classes/labels in the data set is 43. 

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the 4th code cell of the IPython notebook. Please refer to the html file for visualisation. 
I choose to visualize 5 random traffic signs annotated with the sign names.

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fifth code cell of the IPython notebook.

The following pre-processing are done on the images: 
* Convert image to grayscale. Traffic signs in Germany are designed to be color-blind friendly, so grayscale images should produce good results. More importantly, grayscale images reduce training time of neural networks for the limited laptop resource. 
* Equalize histogram to improve image contrast. Many images from the original dataset are taken in low right condition with poor contrast. 

Examples of images after pre-processing are listed after code cell 5. 

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The new dataset already pre-split the train set into train set and validation set. The validation set is useful to tune hyperparameters and prevent overfitting. 

According to the [Traffic-Sign-Convolution] paper, generating additional images with Affine transformation techniques like translate and rotate have added value for training. It allow the model to be better train for images taken from different angles than train images. 
However, i decided to begin with a simple approach (use standard data set). The result achieved is reasonable (however, this can of course be improved with this technique). This can be a future improvement. 

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the 6th cell of the ipython notebook. 
My model is a almost 1 to 1 copy of the [LeNet] architecture with a minor adjustments for the number of classes (43 traffic sign classes as opposed to 10 digits in Lenet). 
My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   					| 
| Convolution 1 5x5    	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					| Rectified Linear Unit activation layer 1		|
| Pooling		      	| 2x2 stride,  outputs 14x14x6 					|
| Convolution 2 5x5    	| 1x1 stride, same padding, outputs 10x10x16 	|
| RELU					| Rectified Linear Unit activation layer 2		|
| Pooling		      	| 2x2 stride,  outputs 5x5x16 					|
| Flatten		      	| 5x5x16 -> 400				 					|
| Fully connected 		| 400 -> 120  (layer 3) 						|
| RELU					| Rectified Linear Unit activation layer 3		|
| Fully connected 		| 120 -> 84  (layer 4) 							|
| RELU					| Rectified Linear Unit activation layer 4		|
| Fully connected 		| 84 -> 43  (number of classes )(layer 5) 		|
| Output				| Logistic Regression function. 				|
|						|												|
 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training and test the model is located in the 10th cell of the ipython notebook. 

To train the model, I used AdamOptimizer (a type of gradient descent optimizer) with minimal cross entropy as loss function (same as [Lenet]). 
I choose to have number of EPOCHs as 20. From the training sessions, the validation result peak at around EPOCHs 15-18, so 20 EPOCHs will provide a good result with an acceptable training time. 
I choose batch size of 256 since the running time is acceptable on my laptop. 
The learning train is chosen to be 0.001 since empirically it is proven to be small enough, while providing a reasonable running time on local resources. 

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is also located in the 10th code cell of the Ipython notebook.

My final model results were:
* training set accuracy of 99.7%
* validation set accuracy of 87.8%
* test set accuracy of 85.2%

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

I choose to reuse the well-known [Lenet] architectures for the following reasons: 
* It has proven to work very well in a similar class of problem, which is hand-written digits classification. 
* Multi-layer convolutional neural networks like the Lenet architecture tends to work very well for image classification, because it can recognize both image overview (when train on the scaled-down image layer) and granular details (when train on full image size).
* Convolutional technique  works very well on images classification because they take into account the information of neighbour pixels, which often provide valuable information. 
* Lenet architecture provides acceptable runtime on my local resources. 
* The final resources show very good results for training set (99.7%), and an acceptable results for test set (85%). This could be improved by generating additional images for train set as suggested [Traffic-Sign-Convolution].

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.
####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

I choose 10 images from the German traffic sign website. I choose 1 image from 10 different classes. The images chosen might have been a bit friendly to the model, since they all have good lighting conditions (so i can visually inspect the results.)
The model can recognize all 10 images correctly (which is higher than test set result of 85%, but consistent with the train set results of 99.7%). 

Please refer to the execution of code cell 8th for the visualization of the image and annotated results. 

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The top 5 softmax probability is output with the execution of the last code cell (25). 
As can be seen in the result, the model has very high certainly with its top prediction (top 1 probability results from 97.1% to 100%). The other classes get very low results in all images (0% -> 2% max). 
This adds to the confidence that the model can produce the correct results.  
