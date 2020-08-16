# Customized R-CNN model for Cat Detection using Keras and Tensorflow

This project is for an assessment test.

The main problem is to build a machine learning program for object recognition, which can detect cats in a given image. The application should be able to detect multiple cats in one picture.

A jpg (png) image file should be appropriate as an input to the application. The output should also be a jpg (png) file from the input image, with the same image size. The performance has to have small, green bounding boxes from the image on the detected cats.

Hence, below is the process of solving this:

![Process](/image/pic1.png)

## Cat Detection Dataset

![First_dataset](/image/pic2.png)

As shown above, the first step will be training of R-CNN object detector to detect cats in input images.

This dataset contains 200 images (some images contain more than one cat). The annotation was manually created using LabelImg.

The dataset is from <a href="https://academictorrents.com/details/c501571c29d16d7f41d159d699d0e7fb37092cbd">Cat Annotation Dataset</a>.

## Setting up Development Environment

For this project to configure your device, tensorflow is needed to be installed:

- <a href="https://www.tensorflow.org/install">Installing Tensorflow in Windows/Mac</a>.
  
## Project Structure

![Structure](/image/pic3.PNG)

The directory 'cats/images' is from the Cat Annotation Dateset while files in 'annotations/' are manually created. These dataset must not be confused with the one that will be created later by build dataset.py script — dataset/ — which is designed to fine-tune our MobileNet V2 model to generate a cat classifier (cat_detector.h5).

pyimagesearch folder contains:

- config.py: Holds the configuration settings that will be used in Python script selection
- iou.py: Intersection over Union (IoU) computations, a metric object detection evaluation
- nms.py: Carries out non-maximum suppression (NMS) to remove overlapping object boxes

In the following three Python scripts, the scripts in the pyimagesearch module will come in handy:

- build dataset.py: takes the cat dataset from the directories 'cats/images' and '/annotations', then generates a new cat / no_cat dataset, which will be used to fine-tune a pre-trained MobileNet V2 model on the ImageNet dataset
- fine-tune rcnn.py: Trains our fine-tuning cat classifier
- detect_object_rcnn.py: bring all the pieces together to perform rudimentary detection of R-CNN objects

## Implementing Cat Detection Configuration File

Next step is implementing the configuration file that will store key constants and settings, which will be used across multiple Python scripts.

The pyimages/config.py contains the following:

- paths for the original cat data set images and annotations for object detection
- paths to positive ( i.e., a cat) and negative ( i.e., no cat in the image input) images. When build dataset.py script is run, these directories will be filled in.
- total number of proposals for Specific Search regions to be used for training and inference, respectively
- total number of positive and negative regions to use when constructing our dataset
- model-specific constants
  - spatial dimensions input to our classification network (MobileNet, ImageNet pretrained)
  - paths of the output file to the cat classifier and mark encoder
  - minimum probability needed for a positive inference prediction (used to filter out false-positive detections) is set at 99%
  
## Measuring Cat Detection Accuracy with Intersection over Union (IoU)

Intersection over Union (IoU) metric will be used to calculate how "strong" a job our object detector does when predicting bounding boxes.

The IoU method computes the ratio between the expected bounding box and the ground-truth bounding box between the area of overlap and the area of union:

<img src="https://render.githubusercontent.com/render/math?math=IoU = \dfrac{Area of Overlap}{Area of Union}">

Examining this equation, it is simply a ratio of intersection over union:

- In the numerator, the area of overlap between the bounding box foreseen and the bounding box for ground-truth.
- The denominator is the field that includes both the projected bounding box and the bounding box of ground-truth.
- Dividing the area of overlap by the union area yields the union intersection.

IoU will be used to test the accuracy of object detection, including how often a given proposal for Selective Search overlaps with a bounding box for ground reality.

## Implementing build_dataset.py Script for Cat Detection

The steps below is an overview of the steps used to create build_dataset.py script:

![Second_dataset](/image/pic4.png)

## Fine-tuning of Cat Detection using Keras and Tensorflow

With the dataset generated through the previous part (Step # 1), fine-tuning can now be done on a CNN classification model.

By combining this classifier with Selective Search, R-CNN cat detector can be constructed.

*I have chosen to fine-tune the MobileNet V2 CNN since time is limited, which is pre-trained on the 1,000-class ImageNet dataset.*

Opening the script for fine-tuning, the breakdown is below:

- config: python configuration file consists of paths and constants
- ImageDataGenerator: for data augmentation purposes
- MobileNetV2: The MobileNet CNN architecture is standard, so TensorFlow / Keras is built-in. This pre-trained model will be used for fine tuning purposes, freeze the head of the network and remove it, then tune / train until the model performs well. Later the head will be returned during the cat detection part.
- tensorflow.keras.layers: selected types of CNN layers are used to build / replace MobileNet V2 headers.
- adam: an alternative optimiser to Stochastic Gradient Descent (SGD).
- LabelBinarizer and to_categorical: used in conjunction to perform a one-hot encoding of our class labels.
- train_test_split: conveniently helps segment the dataset into training and testing sets.
- classification_report: computes a statistical summary of our model evaluation results.
- matplotlib: python’s plotting package will be used to generate accuracy/loss curves from our training history data.

After fine-tuning the model, cat detection is now available.

## Putting the Pieces Together: Implementing the R-CNN Cat Detection Script

This script will have the following breakdown:

- command for accepting the image
- line for loading the model and associated label binarizer
- line for loading the image
- performing Selective Search
- extracting proposed bounding boxes and pre-process them
- classify the pictures from each proposed bounding box
- filter the prediction to positive (bounding boxes with cats) only
- visualize the prediction using OpenCV

## Results

![First_test](/image/pic6.PNG)

This image was the output of the customized R-CNN model. It detected the cat with 99.99% accuracy.

![Second_test](/image/pic5.PNG)

This was another output of the model. The problem with picture is the human toy in front of the cat. Since that part of the was blocked, the reduction of bounding boxes was affected. Another cause of this is the training dataset. As mentioned earlier, the annotations for the cats were manually created. Hence, it affected the prediction accuracy of the bounding boxes.

![Third_test](/image/pic7.PNG)

This is an image without a cat. After inputting to the model, the output is still the same. It is because the MobileNetV2 was customized such that it will only detect cats.

## How to deploy this model on your computer

The steps below are the guide to run this project on your local computer (Windows):

1. Install the requirements on requirements.txt
2. Either run the detect_rcnn_object.py or run through Flask
3. Running the detect_rcnn_object.py (Windows):
    - Open the conda command prompt
    - Go to the project directory
    - On the conda command prompt: python detect_object_rcnn.py --image [directory of your image]
4. Running through flask (Windows):
    - Open the conda command prompt
    - Go to the project directory
    - On the conda command prompt: set FLASK_APP=app.py
    - On the conda command prompt: flask run
    - Open the browser and put the local directory that can be seen on the conda command prompt (usually http://127.0.0.1:5000)
    - Now, you can input your image on the web application
    
*Note: I tried to deploy it on HerokuApp, however, there is a limit of 500Mb. Hence, I cannot show you a working website of this project*

## Insights

- OpenCV is useful for image processing. The deployment of this project is mostly done through OpenCV and Tensorflow.
- Training of the bounding boxes was one of the challenges in this project. Without good dataset, it can affect the predictions in the deployment part.
- **Pre-trained models are the best!** Using MobileNetV2, it was easy to create a model with good accuracy (~99%). This project could also be improved by using more cat pictures. For its other applications, other pictures can also be detected, as long as there is enough dataset for training the model.
- There can be a lot more of improvements in this project like more complexities in the bounding boxes, e.g. scaled measurement of cat's height. However, it needs more complex OpenCV techniques and Mathematics computation.
- For the models used, since there is MobileNetV2, CNN is really the only choice in order to fine tune the model. If there are around thousands of available data, then creating a model from scratch is possible, and multiple Machine and Deep Learnings can be tried.
- There are a lot of available algorithms in scikit-learn or tensorflow. It would be nice to see comparisons of those different models in further projects.

## References

- https://towardsdatascience.com/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9
- https://www.pyimagesearch.com/2020/07/13/r-cnn-object-detection-with-keras-tensorflow-and-deep-learning/
- https://academictorrents.com/details/c501571c29d16d7f41d159d699d0e7fb37092cbd
- https://github.com/tzutalin/labelImg
- https://www.tensorflow.org/install
