'''
    This script is used for predicting data and testing CNN model used for 
    ADAS-ML-DistanceKeeping software
'''
import os
import random
from turtle import up
import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt
import datetime
from datetime import datetime
from time import time
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split 

from cnn import create_cnn_model

##################################ADJUSTABLE PARAMETERS##########################################################
imgs_dir = '../camera_sensors_output/dataset_aditional_part_1'
label_path = '../camera_sensors_output/dataset_aditional_part_1/dataset_aditional_part_1.csv'
outPath = '../camera_sensors_output/dataset_aditional_part_1' # output folder to save results of predicting
modelPath = './model_out/model_out_center_it4_b2_200_200_lr_0005' # path to the model
SAMPLE_DIFF_THRESHOLD = 0.05 # threshold when determing difference between results

loadSize = 4969          # how much images and labels to load
startIndexTestData = 0   # from which index to start loading images and labels

targetImgWidht = 600    # target image width 
targetImgHeght = 370    # target image height
#################################################################################################################

#################################################################################################################
images = [] # list to store training images
labels = [] # list to store values of training labels
filenames = [] # list to store image names 
#################################################################################################################

#################################################################################################################
# CNN parameters
model_name = 'CNN_distanceKeeping.h5'
in_width = 200      # width of the input in the CNN model
in_heigth = 200     # heigth of the input in the CNN model
in_channels = 1     # number of input channels to the CNN model 
output_no = 5       # number of outputs of the CNN model
#################################################################################################################

# read input .csv file
def read_csv(filepath):
    result = []
    dat_file = open(filepath,'r', encoding="utf-8-sig")
    lines=dat_file.readlines()
    for line in lines:
        if len(line)>0:
            p1 = line.find(',')
            filename = line[0:p1]
            categ = line[p1+1:]
            s = filename+','+categ
            result.append(s)
    return result

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

# load images and labels
def load_images_and_labels(images, imgs_dir, labels, label_path, loadSize, inputWidth = 100, inputHeight = 100, startIndex = 0):
    print ('loading images and labels... \n')

    filenames = []
    lines = read_csv(label_path)
    lines.pop(0) # remove header
    cnt = startIndex # loadSize counter, loads specific amount of dataset from specifix index

    for line in lines:

        if (len(line)>0):
            p1 = line.find(',')
            fname = line[0:p1]
            p1 = p1+1
            image_path = os.path.join(imgs_dir, fname)
            filenames.append(fname)

            cat=line[p1:]

            cat = cat.rstrip(',\n')
            cat = cat.split(',')

            cnt_cat = 0
            for item in cat:
                cat[cnt_cat] = float(item)
                cnt_cat = cnt_cat + 1
            cat = np.asarray(cat)

            labels.append(cat)

            img = Image.open(image_path)

            img = cutImage(img, targetImgWidht, targetImgHeght)

            img = img.resize((inputWidth, inputHeight), Image.ANTIALIAS)
            img = ImageOps.grayscale(img)
            img = np.asarray(img)
            
            img = img / 255

            images.append(img)

        cnt = cnt + 1

        if(cnt - startIndex >= loadSize):
            break

    print ('loading complete!\n')
    
    return [images, labels, filenames]

def saveCSVFile(outPath, data, filenames):
    # Format: imgName, Vpresent, Vpl, Vpr, safeToAcc

    outFilePath = os.path.join(outPath, "CNNPredict_" + datetime.now().strftime("%m_%d_%Y_%H_%M_%S") + ".csv")

    csvFile = open(outFilePath, "w+") 

    print("Saving .csv file to location", outFilePath)

    for i in range(len(data)):
        line = ''

        line += filenames[i]
        for j in range(len(data[i])):
            line += str(data[i][j]) + ','

        line += '\n'

        csvFile.write(line)

    csvFile.close()

    print("Done.")

if __name__ == '__main__':
    script_start = datetime.now()

    # force GPU, solves Error code 126
    #gpus = tf.config.experimental.list_physical_devices('GPU')
    #tf.config.experimental.set_memory_growth(gpus[0], True)

    # model location
    model_out_path = os.path.join(modelPath, model_name)

    # load images and labels for training
    lSize = loadSize
    [images, labels, filenames] = load_images_and_labels(images, imgs_dir, labels, label_path, lSize, inputWidth=in_width, inputHeight=in_heigth)

    # create CNN model
    model = create_cnn_model(in_width, in_heigth, in_channels, output_no)

    # change input data to cnn input format
    df_im = np.asarray(images)
    df_im = df_im.reshape(df_im.shape[0], in_width, in_heigth, in_channels)
    df_labels = np.asarray(labels)
    df_labels = df_labels.reshape(df_labels.shape[0], output_no)

    # load trained model
    model = tf.keras.models.load_model(model_out_path)
    
    # predict on unseen data
    predictions = model.predict(df_im, verbose = 1)

    # write test results in .csv file
    saveCSVFile(imgs_dir, predictions, filenames)

    script_end = datetime.now()
    print (script_end - script_start)