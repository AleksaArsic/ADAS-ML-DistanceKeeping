'''
    DatasetGenerator.py - contains DatasetGenerator class used for dynamic dataset loading in run time for the purposes of training Neural Networks
                          using Keras and Tensorflow 
'''
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# cuts Images to target width and height
def cutImage(pilImg, targetW, targetH):

    imgWidth, imgHeight = pilImg.size

    widthCrop = imgWidth - targetW

    leftPoint = widthCrop / 2
    upperPoint = imgHeight - targetH
    rightPoint = imgWidth - leftPoint
    lowerPoint = imgHeight

    pilImg = pilImg.crop((leftPoint, upperPoint, rightPoint, lowerPoint))

    return pilImg

class DatasetGenerator(tf.keras.utils.Sequence):
  
  def __init__(self, image_filenames, labels, batch_size):
    self.image_filenames = image_filenames
    self.labels = labels
    self.batch_size = batch_size
    
    
  def __len__(self):
    return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)
  
  
  def __getitem__(self, idx):
    batch_x = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
    batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]

    return np.array([
            np.asarray(ImageOps.grayscale(cutImage(Image.open(file_name), 600, 370).resize((200, 200), Image.ANTIALIAS))) / 255.0
               for file_name in batch_x]), np.array(batch_y)

