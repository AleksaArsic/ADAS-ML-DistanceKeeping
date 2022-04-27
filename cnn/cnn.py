import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation

multiLabelEndId = 4 # end index of multilabel labels
multilabelThreshold = 0.07  # threshold for which multilabel labels are considered as valid
safeAccThreshold = 0.1      # threshold for which safeAcc label is considered as valid

# custom binary accuracy metric definition
def adapted_binary_accuracy(y_true, y_pred):
    
    multiLabelDiff = abs(y_true[:, 0:multiLabelEndId] - y_pred[:, 0:multiLabelEndId])
    safeAccDiff = abs(y_true[:, -1] - y_pred[:, -1])
    
    multiLabelTrue = tf.cast(tf.math.count_nonzero(multiLabelDiff <= multilabelThreshold), tf.int32)
    safeAccTrue = tf.cast(tf.math.count_nonzero(safeAccDiff <= safeAccThreshold), tf.int32) # can't use if statements in custom metrics, tf.math.count_nonzero as replacement

    return (multiLabelTrue + safeAccTrue) / (y_pred.numpy().shape[0] * y_pred.numpy().shape[1]) # types of addition in keras must be the same hence the tf.cast in calculations
    

# creates and returns convolutional neural network model
def create_cnn_model(in_width, in_height, channels, output_no):
    
    tf.keras.backend.set_floatx('float64')

    model = Sequential()

    model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(in_width, in_height, channels)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(output_no)) 
    model.add(Activation('sigmoid'))

    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9), \
                  loss='binary_crossentropy',                               \
                  metrics=[adapted_binary_accuracy])

    model.summary()

    return model