'''
    This script is used to construct and train CNN model used for 
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
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils import shuffle

from cnn import create_cnn_model
from DatasetGenerator import DatasetGenerator
import cv2

#################################################################################################################
images = [] # list to store training images
labels = [] # list to store values of training labels
filenames = [] # list to store image names 

imgs_dir = './dataset'
label_path = './dataset/dataset.csv'
output_path = './model_out/model_out_center_it6_b4_200_200_dset_12754/' # output folder to save results of training
SAMPLE_DIFF_THRESHOLD = 0.05 # threshold when determing difference between positive and negative results

epochNo = 250   # number of epochs per training, any number greater than dataset size will load whole dataset
batchSize = 4   # batch size in one epoch

loadSize = 10000          # how much images and labels to load
startIndexTestData = 0   # from which index to start loading images and labels

targetImgWidht = 600
targetImgHeight = 370
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

# Load filenames and labels
def load_filenames_and_labels(imgs_dir, label_path):
    print ('loading ' + str(loadSize) + ' filenames and labels... \n')

    filenames = []
    lines = read_csv(label_path)
    lines.pop(0) # remove header

    for line in lines:

        if ((len(line) > 0)):
            p1 = line.find(',')
            fname = line[0:p1]
            p1 = p1+1
            filenames.append(os.path.join(imgs_dir, fname))

            cat=line[p1:]

            cat = cat.rstrip(',\n')
            cat = cat.split(',')

            cnt_cat = 0
            for item in cat:
                cat[cnt_cat] = float(item)
                cnt_cat = cnt_cat + 1
            cat = np.asarray(cat)

            labels.append(cat)

    print ('loading complete!\n')
    
    return [labels, filenames]

# compare results 
def compare_results(test_labels, predictions):

    sample_difference_acc = []
    predictions_acc = [] 

    # transpose arrays for easier math
    test_labels = np.transpose(test_labels)
    predictions = np.transpose(predictions)

    for i in range(len(predictions)):
        # calculate difference accuracy between samples
        diff = abs(predictions[i] - test_labels[i]) # / 1.0
    
        sample_difference_acc = []

        for j in range(len(diff)):
            if(diff[j] <= SAMPLE_DIFF_THRESHOLD):
                sample_difference_acc.append(True)
            else:
                sample_difference_acc.append(False)

        valid_samples = 0
        invalid_samples = 0

        for j in range(len(sample_difference_acc)):
            if(sample_difference_acc[j] == True):
                valid_samples += 1
            else:
                invalid_samples += 1

        predictions_acc.append(valid_samples / len(predictions[i]))

    # transpose to normal
    test_labels = np.transpose(test_labels)
    predictions = np.transpose(predictions)

    return predictions_acc

# write test results to .csv file
def write_test_to_csv(test_labels, predictions, predictions_acc):
    s = ''
    i = 0 
    for item in test_labels:
        for field in item:
            s += str(field) 
            s += ','

        s += ','

        l = predictions[i]

        for field in l:
            s += str(field)
            s += ','

        s += '\n'
        i += 1

    s += '\n'

    for item in predictions_acc:
        s += str(item) + ','

    s += '\n'

    with open(os.path.join(output_path, 'test_predictions_results.csv'), 'w') as f:
        f.write(s)

# plot training results
def plot_training_results(val_acc, val_loss, train_acc, train_loss):

    epochs = range(1, len(train_acc) + 1)

    #Plottig the training and validation loss
    plt.plot(epochs, val_acc, 'bo', label='Validation Accuracy')
    plt.plot(epochs, train_acc, 'b', label='Train Accuracy')
    plt.title('Train and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(output_path, 'model.png'))

    plt.clf()

    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.plot(epochs, train_loss, 'b', label='Train Loss')
    plt.title('Loss / Mean Sqared Error')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(output_path, 'model_phase01_loss.png'))

if __name__ == '__main__':
    script_start = datetime.now()

    # force GPU, solves Error code 126
    #gpus = tf.config.experimental.list_physical_devices('GPU')
    #tf.config.experimental.set_memory_growth(gpus[0], True)

    # force CPU
    #tf.config.set_visible_devices([], 'GPU')

    # Show which graphics card is allocated
    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF")

    # Enable GPU memory growth
    #physical_devices = tf.config.experimental.list_physical_devices('GPU')
    #if len(physical_devices) > 0:
    #    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # model output destination
    model_out_path = os.path.join(output_path, model_name)

    # load images and labels for training
    lSize = loadSize
    [labels, filenames] = load_filenames_and_labels(imgs_dir, label_path)

    # shufle filenames and labels
    filenames_shuffled, labels_shuffled = shuffle(filenames, labels)

    filenames_shuffled = np.array(filenames_shuffled)
    labels_shuffled = np.array(labels_shuffled)

    # train test split
    train_filenames, val_filenames, train_labels, val_labels = train_test_split(
                                        filenames_shuffled, labels_shuffled, test_size=0.15, random_state=1)

    # Create train and validation dataset generators
    training_batch_generator = DatasetGenerator(train_filenames, train_labels, batchSize)
    val_batch_generator = DatasetGenerator(val_filenames, val_labels, batchSize)

    # Create CNN model
    model = create_cnn_model(in_width, in_heigth, in_channels, output_no)

    # create TensorBoard
    tensorboard = TensorBoard(log_dir = output_path + "logs_img" + "\{}".format(time()))

    # define callbacks
    callbacks = [
        EarlyStopping(monitor='val_categorical_accuracy', mode = 'max', patience=30, verbose=1),
        ReduceLROnPlateau(monitor='val_categorical_accuracy', mode = 'max', factor=0.025, patience=10, min_lr=0.000001, verbose=1),
        ModelCheckpoint(model_out_path, monitor='val_categorical_accuracy', mode = 'max', verbose=1, save_best_only=True, save_weights_only=False),
        tensorboard
    ]

    # CNN training
    model_history = model.fit_generator(generator=training_batch_generator,
                    steps_per_epoch = int(len(train_filenames) // batchSize),
                    epochs=epochNo,
                    validation_data=val_batch_generator,
                    validation_steps = int(len(val_filenames) // batchSize),
                    callbacks=callbacks,
                    verbose=1)

    # Visualizing accuracy and loss of training the model
    history_dict=model_history.history
    print(history_dict.keys())
    print(history_dict)
    val_acc = history_dict['val_categorical_accuracy']
    val_loss = history_dict['val_loss']
    train_acc = history_dict['categorical_accuracy']
    train_loss = history_dict['loss']

    #plot accuracy and loss
    plot_training_results(val_acc, val_loss, train_acc, train_loss)

    # predict on test dataset
    # reshape test dataset to appropriate dimensions of input layer of the trained CNN
    #df_im = np.asarray(test_images)
    #df_im = df_im.reshape(df_im.shape[0], in_width, in_heigth, in_channels)

    # load newly trained model
    #model = tf.keras.models.load_model(model_out_path)
    
    # predict on unseen data
    #predictions = model.predict(df_im, verbose = 1)

    # compare results between labeled test set and predictions
    #test_labels = np.asarray(test_labels)
    #predictions_acc = compare_results(test_labels, predictions)

    # write test results in .csv file
    #write_test_to_csv(test_labels, predictions, predictions_acc)

    script_end = datetime.now()
    print (script_end - script_start)