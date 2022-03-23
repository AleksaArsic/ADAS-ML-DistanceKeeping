import os
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

# creates and returns convolutional neural network model
def create_cnn_model(in_width, in_height, channels, output_no, model_out_path =''):
    
    tf.keras.backend.set_floatx('float64')

    #model = Sequential()
    input = tf.keras.Input(shape=(in_width, in_height, channels))

    conv_1 = Conv2D(16, (3, 3), activation='relu')(input)
    max_pool_1 = MaxPooling2D(pool_size=(2, 2))(conv_1)
    dropout_1 = Dropout(0.25)(max_pool_1)

    conv_2 = Conv2D(32, (3, 3), activation='relu')(dropout_1)
    max_pool_2 = MaxPooling2D(pool_size=(2, 2))(conv_2)
    droupout_2 = Dropout(0.25)(max_pool_2)

    conv_3 = Conv2D(64, (3, 3), activation='relu')(droupout_2)
    max_pool_3 = MaxPooling2D(pool_size=(2, 2))(conv_3)
    dropout_3 = Dropout(0.25)(max_pool_3)

    flatten = Flatten()(dropout_3)
    
    dense_1 = Dense(128, activation='relu')(flatten)
    dropout_4 = Dropout(0.25)(dense_1)
    
    dense_2 = Dense(64, activation='relu')(dropout_4)
    dropout_5 = Dropout(0.25)(dense_2)
    
    dense_3 = Dense(32, activation='relu')(dropout_5)

    out_bin = Dense(output_no - 1, activation='sigmoid', name='veh_pred')(dense_3)
    out_mse = Dense(1, activation='sigmoid', name='safe_acc')(dense_3)

    model = tf.keras.models.Model(inputs=input, outputs=[out_bin, out_mse])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
                  loss={'veh_pred' : 'binary_crossentropy', 
                        'safe_acc' : 'mean_squared_error'},                               
                  metrics={'veh_pred' : tf.keras.metrics.BinaryAccuracy(name='cat_acc', dtype=None),
                           'safe_acc' : tf.keras.metrics.MeanSquaredError(name="mse", dtype=None)})

    tf.keras.utils.plot_model(model, os.path.join(model_out_path, "distance_keeping_model.png"), show_shapes=True)
    
    model.summary()

    return model