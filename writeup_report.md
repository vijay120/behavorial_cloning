#**Behavioral Cloning** 

##Author: Vijay Ramakrishnan


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model was developed using Nvidia's end-to-end deep learning model for self driving cars, as referenced from the paper, "End to End Learning for Self-Driving Cars".
It consists of the following network architecture (as followed by Kera's definition of a layer):

Layer 1: A normalization layer to normalize the pixel values
Layer 2: A cropping layer that crops out the beginning 70 pixels and the end 25 pixels of the height of the input image
Layer 3: A 5x5 convolution layer with a stride length of 2 in the width and height direction and 24 kernels. The layer is activated with a relu activation.
Layer 4: A 5x5 convolution layer with a stride length of 2 in the width and height direction and 36 kernels. The layer is activated with a relu activation.
Layer 5: A 5x5 convolution layer with a stride length of 2 in the width and height direction and 48 kernels. The layer is activated with a relu activation.
Layer 6: A 5x5 convolution layer with a stride length of 1 in the width and height direction and 64 kernels. The layer is activated with a relu activation.
Layer 7: A 5x5 convolution layer with a stride length of 1 in the width and height direction and 64 kernels. The layer is activated with a relu activation.
Layer 8: A layer that flattens the output of layer 7
Layer 9: A fully connected layer of 1164 nodes
Layer 10: A fully connected layer of 100 nodes
Layer 11: A fully connected layer of 50 nodes
Layer 12: A fully connected layer of 10 nodes
Layer 13: The output node

The RELU layers add nonlinearity.

####2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets (20% of the data was given to the validation set) to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model using the adam optimizer to train the loss function, with the loss function being the mean squared error.

####4. Appropriate training data

The training data was obtained in the following way:
Step 1) A base set from the udacity training set.
Step 2) A manually curated set from driving on the training simulator for 4 laps.
Step 3) I used the left, center and right cameras of the car for training. The left camera was mapped to the center camera's steering angle with a +0.2 shift and the right camera was mapped to the camera's steering angle with a -0.2 shift. 
Step 4) All the images were flipped along with the steering angles, to double the above dataset.
Step 5) We cropped the above and below portions of the image to remove redundant data.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to start off with a simple deep learning network and add complexity based on car performance. Simultanously, I had to find clever ways to augment the dataset starting with the udacity dataset and adding more along the way, depending on ideas I got or looking at the forums. I started making the network more complex by adding two convolution layers to the model. Once this was done, I focused on acquiring more data. At this point in time, the car simulator was doing better (ie not crashing immidiately) but it would eventually crash. 

After augmenting the udacity set using the flipping and side cameras tricks, I decided to get more data to decrease the model loss. I did this by driving car the tracks for 4 laps. Next, I decided to implement the more complex Nvidia architecture, which immidiately saw improvements to loss. The simulator could go across the whole track with crashing.

I also decided to remove 50% of all steering angels that where between -0.05 and 0.05 size since according to the distribution of steering values, those values had the highest amount of data points. As I gathered more data, I had to use a generator since the data could not fit in memory.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes:
Layer 1: A normalization layer to normalize the pixel values
Layer 2: A cropping layer that crops out the beginning 70 pixels and the end 25 pixels of the height of the input image
Layer 3: A 5x5 convolution layer with a stride length of 2 in the width and height direction and 24 kernels. The layer is activated with a relu activation.
Layer 4: A 5x5 convolution layer with a stride length of 2 in the width and height direction and 36 kernels. The layer is activated with a relu activation.
Layer 5: A 5x5 convolution layer with a stride length of 2 in the width and height direction and 48 kernels. The layer is activated with a relu activation.
Layer 6: A 5x5 convolution layer with a stride length of 1 in the width and height direction and 64 kernels. The layer is activated with a relu activation.
Layer 7: A 5x5 convolution layer with a stride length of 1 in the width and height direction and 64 kernels. The layer is activated with a relu activation.
Layer 8: A layer that flattens the output of layer 7
Layer 9: A fully connected layer of 1164 nodes
Layer 10: A fully connected layer of 100 nodes
Layer 11: A fully connected layer of 50 nodes
Layer 12: A fully connected layer of 10 nodes
Layer 13: The output node

####3. Creation of the Training Set & Training Process

I initially bootstrapped the training data by obtaining it from Udacity. I then augmented it by adding the left and right camera data and adjusting the steering angle as described earlier. I also flipped the images and negated the steering angels to get more data. I then added more data by driving around the lap 4 times. The data was then cropped to remove extra information. Finally, as described earlier, I removed some of the steering angels that were too small.

After the collection process, I had ~8200 number of data points. I then preprocessed this data by normalizing the pixel values.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 7 as evidenced by loss decreasing steadily and then stablizing. I used an adam optimizer so that manually training the learning rate wasn't necessary.