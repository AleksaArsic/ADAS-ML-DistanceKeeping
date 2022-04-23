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

from cnn import create_cnn_model

#################################################################################################################
images = [] # list to store training images
labels = [] # list to store values of training labels
filenames = [] # list to store image names 

imgs_dir = './dataset'
label_path = './dataset/dataset.csv'
output_path = './model_out/model_out_center_it6_b4_200_200/' # output folder to save results of training
SAMPLE_DIFF_THRESHOLD = 0.05 # threshold when determing difference between positive and negative results

epochNo = 250   # number of epochs per training, any number greater than dataset size will load whole dataset
batchSize = 2   # batch size in one epoch

loadSize = 8599          # how much images and labels to load
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
    print ('loading ' + str(loadSize) + ' images and labels... \n')

    filenames = []
    lines = read_csv(label_path)
    lines.pop(0) # remove header
    cnt = startIndex # loadSize counter, loads specific amount of dataset from specifix index

    for line in lines:

        if ((len(line) > 0)):
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

            img = cutImage(img, targetImgWidht, targetImgHeight)

            img = img.resize((inputWidth, inputHeight), Image.ANTIALIAS)
            img = ImageOps.grayscale(img)
            img = np.asarray(img)
            
            img = img / 255

            images.append(img)

        cnt = cnt + 1

        if(cnt - startIndex >= loadSize):
            break

    # shuffle dataset before training
    #images, labels = shuffleDataset(images, labels)

    print ('loading complete!\n')
    
    return [images, labels, filenames]

# Train-Test Dataset split
def train_test_dataset_split(images, labels):
    test_images = []
    test_labels = []

    dataset_len = len(images)

    test_len = int(dataset_len * 0.1)

    # debug
    indexes = []

    for i in range(test_len):
        n = random.randint(0, dataset_len - 1)

        test_images.append(images[n])
        test_labels.append(labels[n])

        images.pop(n)
        labels.pop(n)

        dataset_len -= 1

        indexes.append(n)

    return [test_images, test_labels]

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

def shuffleDataset(imgs, labels):

    indices = np.arange(len(imgs))
    np.random.shuffle(indices)

    imgs = np.asarray(imgs)[indices]
    imgs = imgs.tolist()
    labels = np.asarray(labels)[indices]
    labels = labels.tolist()

    return imgs, labels

def reduceDataset(imgs, labels):

    indexes = [random.randrange(0, len(imgs) - 1, 1) for i in range(int(len(imgs) / 7))]

    tempImgs = []
    tempLabels = []

    for i in range(len(indexes)):
        tempImgs.append(imgs[i])
        tempLabels.append(labels[i])

    imgs = tempImgs
    labels = tempLabels

    return imgs, labels

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
    [images, labels, filenames] = load_images_and_labels(images, imgs_dir, labels, label_path, lSize, inputWidth=in_width, inputHeight=in_heigth)

    # perform a train-test dataset split
    [test_images, test_labels] = train_test_dataset_split(images, labels)

    # create CNN model
    model = create_cnn_model(in_width, in_heigth, in_channels, output_no)

    # change input data to cnn input format
    df_im = np.asarray(images)
    df_im = df_im.reshape(df_im.shape[0], in_width, in_heigth, in_channels)
    df_labels = np.asarray(labels)
    df_labels = df_labels.reshape(df_labels.shape[0], output_no)
    tr_im, val_im, tr_cat, val_labels = train_test_split(df_im, df_labels, test_size=0.2)

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
    model_history = model.fit(df_im, df_labels, # df_im - input ; df_labels - output
                    batch_size=batchSize,
                    #batch_size=64,
                    epochs=epochNo,
                    validation_data=(val_im, val_labels),
                    callbacks=callbacks,
                    shuffle=True,
                    verbose=1)

    # Visualizing accuracy and loss of training the model
    history_dict=model_history.history
    print(history_dict.keys())
    val_acc = history_dict['val_categorical_accuracy']
    val_loss = history_dict['val_loss']
    train_acc = history_dict['categorical_accuracy']
    train_loss = history_dict['loss']

    #plot accuracy and loss
    plot_training_results(val_acc, val_loss, train_acc, train_loss)

    # predict on test dataset
    # reshape test dataset to appropriate dimensions of input layer of the trained CNN
    df_im = np.asarray(test_images)
    df_im = df_im.reshape(df_im.shape[0], in_width, in_heigth, in_channels)

    # load newly trained model
    model = tf.keras.models.load_model(model_out_path)
    
    # predict on unseen data
    predictions = model.predict(df_im, verbose = 1)

    # compare results between labeled test set and predictions
    test_labels = np.asarray(test_labels)
    predictions_acc = compare_results(test_labels, predictions)

    # write test results in .csv file
    write_test_to_csv(test_labels, predictions, predictions_acc)

    script_end = datetime.now()
    print (script_end - script_start)