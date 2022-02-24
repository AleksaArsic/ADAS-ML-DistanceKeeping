# PredictYOLOv3.py - Script for aquiring data from YOLOv3 neural network that will be used
#                    as part of the dataset used for training of CNN model

import tensorflow as tf
from utils import load_class_names, output_boxes, draw_outputs, resize_image
import cv2
import numpy as np
from yolov3 import YOLOv3Net
import os
import glob
import math
from PIL import Image, ImageOps

model_size = (416, 416, 3)  # Expected input format of the model 
num_classes = 80            # Number of classes on which the network was trained   
class_name = './data/coco.names'    # Path to the file that contains class names
max_output_size = 40                # Maximum number of bounding boxes we want to get for all classes 
max_output_size_per_class= 20       # Maximum number of bounding boxes we want to get for one class 
iou_threshold = 0.5                 # Threshold of Intersection Over Union of two bounding boxes
confidence_threshold = 0.5          # Threshold of trustworthiness that the object is detected  

cfgfile = './cfg/yolov3.cfg'                  # Path to YOLOv3 configuration file
weightfile = './weights/yolov3_weights.tf'    # Path to file that contains trained coeficients in TensorFlow format
imgPath = '../camera_sensors_output/center'        # Path to input images on which we will run YOLOv3 model
cLeftCamId = 0               # left camera id
objectsOfInterest = [0, 1, 2, 3, 5, 6, 7] # Objects of interest from coco.names classes


def loadAndResize(imgsDir):
    print ('loading  images...')

    images = []
    resized_images = []
    fileNames = []

    os.chdir(imgsDir)

    for imagePath in glob.glob("*.jpg"):

        img = cv2.imread(imagePath)
        img = np.array(img)

        images.append(img)

        # preprocess data, scale, greyscale, etc.
        resized_images.append(cv2.resize(img, dsize=(model_size[0], model_size[1]), interpolation=cv2.INTER_CUBIC))

        fileNames.append(os.path.basename(imagePath))

    print ('loading complete!')

    return [images, resized_images, fileNames]

def aquireData(fileNames, boxes, classNames):
    
    # [ [fileName, numberOfBoundingBoxes, [Classnames], [ ... ], ... ]
    outputData = []

    for i in range(len(fileNames)):
        tempData = []
        classNamesTemp = []

        tempData.append(fileNames[i])
        tempData.append(len(boxes[i]))
        
        #for j in range(len(classNames[i])):
        #    classNamesTemp.append(classNames[i][j])

        tempData.append(classNamesTemp)
        outputData.append(tempData)

    return outputData

def saveCSVFile(data):
    # Format: imgName, noOfBoundingBoxes, classNames

    csv_file = open('out_yolov3.csv', "w+") 
    # write first line in output .csv file
    line = 'imgName, noOfBoundingBoxes, classNames\n'

    csv_file.write(line)

    # write the rest of recorded data to output .csv file
    print(len(data))
    for i in range(len(data)):
        line = data[i][0] + ',' + str(data[i][1]) + ',' + str(data[i][2]) + '\n'
        csv_file.write(line)

    csv_file.close()

def main():

    # Create model
    model = YOLOv3Net(cfgfile, model_size, num_classes)
    # Load trained coeficients into model
    model.load_weights(weightfile)
    # Load class names
    #classNames = load_class_names(class_name)
	# debug, no class names
    # Load class names
    classNames = [' '] * 80
    # Bounding boxes
    boxes = []
    # detected classes
    detectedClasses = []

    # Load input images and preprocess then in input format of YOLOv3 model
    images = []
    resized_images = []
    fileNames = []
    
    # Load image data 
    [images, resized_images, fileNames] = loadAndResize(imgPath)

    # Inference on input image
    # Output predictions are set of vectors (10647) where each suits one bounding box of the location of the object
    for i in range(0, len(fileNames)):
        resized_image = []
        
        image = images[i]
        
        resized_image = tf.expand_dims(resized_images[i], 0)

        pred = model.predict(resized_image)

        # Determining bounding boxes around detected objects (for certain thresholds)
        tempBoxes, scores, tempDetectedClasses, nums = output_boxes( \
            pred, model_size,
            max_output_size=max_output_size,
            max_output_size_per_class=max_output_size_per_class,
            iou_threshold=iou_threshold,
            confidence_threshold=confidence_threshold)

        # Debug
        print(len(tempBoxes))

        boxes.append(tempBoxes)
        detectedClasses.append(tempDetectedClasses)

        if(cv2.waitKey(20) & 0xFF == ord('q')):
            cv2.destroyAllWindows()
            break

    # Aquire data in correct format
    data = aquireData(fileNames, boxes, detectedClasses)

    # Save results to .csv file
    saveCSVFile(data)

if __name__ == '__main__':
    
    # force GPU, solves Error code 126
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    main()