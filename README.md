# R-CNN Cat Detection using Deep Learning

This project is an assessment exam.
![Process](/image/pic1.png)
## Cat Detection Dataset
![First_dataset](/image/pic2.png)
As shown above, the first step will be training of R-CNN object detector to detect cats in input images.

This dataset contains 200 images (some images contain more than one cat). The annotation was manually created using LabelImg.

The dataset is from <a href="https://academictorrents.com/details/c501571c29d16d7f41d159d699d0e7fb37092cbd">Cat Annotation Dataset</a>.

## Setting up Development Environment

For this project to configure your device, tensorflow is needed to be installed:

- <a href="https://www.tensorflow.org/install">Installing Tensorflow in Windows/Mac.
  
## Project Structure
![Structure](/image/pic3.PNG)

cats/images is from the Cat Annotation Dateset while annotations/ is manually created. Thise dataset must not be confused with the one that will be created later by build dataset.py script — dataset/ — which is designed to fine-tune our MobileNet V2 model to generate a cat classifier (cat_detector.h5).

pyimagesearch folder contains:

- config.py: Holds the configuration settings that will be used in Python script selection
- iou.py: Intersection over Union (IoU) computations, a metric object detection evaluation
- nms.py: Carries out non-maximum suppression (NMS) to remove overlapping object boxes

In the following three Python scripts, the scripts in the pyimagesearch module will come in handy:

- build dataset.py: takes the cat dataset from Dat Trancats/images and /annotations, then generates a new cat / no_cat dataset, which will be used to fine-tune a pre-trained MobileNet V2 model on the ImageNet dataset
- fine-tune rcnn.py: Trains our fine-tuning raccoon classifier
- detect_object_rcnn.py: bring all the pieces together to perform rudimentary detection of R-CNN objects

## Implementing Object Detection Configuration File

Next step is implementing the configuration file that will store key constants and settings, which will be used across multiple Python scripts.

The pyimages/config.py contains the following:

- paths for the original raccoon data set images and annotations for object detection
- paths to positive ( i.e., a cat) and negative ( i.e., no cat in the image input) images. When build dataset.py script is run, these directories will be filled in.
- total number of proposals for Specific Search regions to be used for training and inference, respectively
- total number of positive and negative regions to use when constructing our dataset
- model-specific constants
  - spatial dimensions input to our classification network (MobileNet, ImageNet pretrained)
  - paths of the output file to the cat classifier and mark encoder
  - minimum probability needed for a positive inference prediction (used to filter out false-positive detections) is set at 99%
  
## Measuring object detection accuracy with Intersection over Union (IoU)

Intersection over Union (IoU) metric will be used to calculate how "strong" a job our object detector does when predicting bounding boxes.

The IoU method computes the ratio between the expected bounding box and the ground-truth bounding box between the area of overlap and the area of union:

<img src="https://render.githubusercontent.com/render/math?math=IoU = \frac{Area of Overlap}{Area of Union}">

Examining this equation, it is simply a ratio of intersection over union:

- In the numerator, the area of overlap between the bounding box foreseen and the bounding box for ground-truth.
- The denominator is the field that includes both the projected bounding box and the bounding box of ground-truth.
- Dividing the area of overlap by the union area yields the union intersection.

IoU will be used to test the accuracy of object detection, including how often a given proposal for Selective Search overlaps with a bounding box for ground reality.

## Implementing build_dataset.py Script for Object Detection
![Second_dataset](/image/pic4.png)
