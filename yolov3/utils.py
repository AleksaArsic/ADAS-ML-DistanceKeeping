# utils.py - helper functions 

import tensorflow as tf
import numpy as np
import cv2

def resize_image(inputs, modelsize):
    #Scales input images to the input layer of the model 

    #Parameters:
    #    inputs: original input images
    #    modelsize: expected dimensions of input layer of the model (width, height) 

    #Return value:
    #    inputs: scaled images

    inputs= tf.image.resize(inputs, modelsize)
    return inputs


def load_class_names(file_name):
    #Load class names from the file 

    #Parameters:
    #    file_name: Name and path to the file with class names   

    #Return value:
    #    class_names: list with the name of classes 

    with open(file_name, 'r') as f:
        class_names = f.read().splitlines()
    return class_names


def non_max_suppression(inputs, model_size, max_output_size,
                        max_output_size_per_class, iou_threshold,
                        confidence_threshold):
    # Non-maximum suppression procedure.

    #Parameters:
    #    inputs: bounding boxes 
    #    model_size: model input size
    #    max_output_size: Maximal number of bounding boxes for all classes 
    #    max_output_size_per_class: Maximal number of bounding boxes for each class 
    #    iou_threshold:  Intersection over union 
    #    confidence_threshold: Object presence confidence threshold  

    #Return value:
    #    boxes: Bounding boxes after Non-max suppression procedure
    #    scores: Tensor with presence probabilities of objects for bounding boxes 
    #    classes: Tensor with classes of bounding boxes 
    #    valid_detections: Tensor that consists of number of valid detections for bounding boxes.
    #                      Only first entries to output tensors are valid. The rest up until the maximal number is filled with zeros.

    bbox, confs, class_probs = tf.split(inputs, [4, 1, -1], axis=-1)
    bbox=bbox/model_size[0]

    scores = confs * class_probs
    boxes, scores, classes, valid_detections = \
        tf.image.combined_non_max_suppression(
        boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
        scores=tf.reshape(scores, (tf.shape(scores)[0], -1,
                                   tf.shape(scores)[-1])),
        max_output_size_per_class=max_output_size_per_class,
        max_total_size=max_output_size,
        iou_threshold=iou_threshold,
        score_threshold=confidence_threshold
    )
    return boxes, scores, classes, valid_detections


def output_boxes(inputs, model_size, max_output_size, max_output_size_per_class,
                 iou_threshold, confidence_threshold):
    # Exctraction of less number of bounding boxes from all bounding boxes that are output of convolutional processing
    # As first criteria, bounding boxes that are rejected are those whose value of probability below certain threshold, e.g 0.7
    # Second step is to determine Intersection over union
    # Bounding box with the biggest value p is kept, while others with whom he intersects and IoU is above certain threshold are rejected.
    # This is known as non-maximum supression.

    #Parameters:
    #    inputs: Set of vectors (10647) where each fits one bounding box of the location of the object 
    #    model_size: Input size of the model
    #    max_output_size: Maximal number of bounding boxes for all classes 
    #    max_output_size_per_class: Maximal number of bounding boxes for each class 
    #    iou_threshold: Intersection over union 
    #    confidence_threshold: Object presence confidence threshold 

    #Return value:
    #    boxes_dicts: Dictionary of bounding boxes, probabilites, classes and number of valid detections

    # print(inputs.shape) # Dimensions of input set of vectors
    center_x, center_y, width, height, confidence, classes = \
        tf.split(inputs, [1, 1, 1, 1, 1, -1], axis=-1)

    top_left_x = center_x - width / 2.0
    top_left_y = center_y - height / 2.0
    bottom_right_x = center_x + width / 2.0
    bottom_right_y = center_y + height / 2.0

    inputs = tf.concat([top_left_x, top_left_y, bottom_right_x,
                        bottom_right_y, confidence, classes], axis=-1)

    boxes_dicts = non_max_suppression(inputs, model_size, max_output_size,
                                      max_output_size_per_class, iou_threshold, confidence_threshold)

    return boxes_dicts

def draw_outputs(img, boxes, objectness, classes, nums, class_names, id, distanceIndexPair):
    #Drawing of detected objects on image (rectangle, class name, probability)

    #Parameters:
    #    img: Image
    #    boxes: Bounding boxes for drawing 
    #    objectness: Probability that the object is detected 
    #    classes: Classes of detected objects 
    #    nums: Number of detected objects 
    #    class_names: List of class names 
    #    id: ID of the camera (left - 0, right - 1)
    #    dinstaceIndexPair: pairs of calculated distances and indexes of bounding boxes 

    #Return value:
    #    img: Output image

    boxes, objectness, classes, nums = boxes[id], objectness[id], classes[id], nums[id]
    boxes=np.array(boxes)

    print(boxes)

    for i in range(nums):
        x1y1 = tuple((boxes[i,0:2] * [img.shape[1],img.shape[0]]).astype(np.int32))
        x2y2 = tuple((boxes[i,2:4] * [img.shape[1],img.shape[0]]).astype(np.int32))

        img = cv2.rectangle(img, (x1y1), (x2y2), (255,0,0), 2)

        img = cv2.putText(img, '{} {:.4f}'.format(
            class_names[int(classes[i])], objectness[i]),
                          (x1y1), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
                          
        
        for tmp in distanceIndexPair:
            if(i == tmp[1]):
                distance = tmp[0]
                img = cv2.putText(img, '{:.4f}'.format(distance), x2y2, cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
                
    return img