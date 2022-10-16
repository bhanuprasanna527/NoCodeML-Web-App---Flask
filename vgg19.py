import re
import numpy as np
import os
from camera import VideoCamera
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import pathlib
from keras.layers import Input, Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten
from keras.models import Model

def train(train_location='/Users/bhanuprasanna/Downloads/archive/train', valid_location='/Users/bhanuprasanna/Downloads/archive/valid', project_name='Image bird', noc=450, epochs=10, img_wid=224, img_hei=224, batch_size=32, activation_function='relu'):
    train_data_dir = pathlib.Path(train_location)
    image_count = len(list(train_data_dir.glob('train/*/*.jpg')))
    valid_data_dir = pathlib.Path(valid_location)
    STEPS_PER_EPOCH = np.ceil(image_count/batch_size)

    train = ImageDataGenerator(rescale=1/255)
    valid = ImageDataGenerator(rescale=1/255)
    train_dataset = train.flow_from_directory(train_location)
    valid_dataset = valid.flow_from_directory(valid_location)

    VALID_STEPS_PER_EPOCH = np.ceil(valid_dataset.samples/batch_size)

    model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32,(3,3),strides=(1, 1),activation='relu',padding='same', input_shape=(224, 224, 3)), 
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    
    tf.keras.layers.Conv2D(64,(3,3),strides=(1, 1) ,padding='same',activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

    tf.keras.layers.Conv2D(128,(3,3),strides=(1, 1),padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

    tf.keras.layers.Conv2D(256,(3,3),strides=(1, 1),padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),


    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax'),
])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'], run_eagerly=True)
    
    history = model.fit_generator(
                    train_dataset,
                    steps_per_epoch=STEPS_PER_EPOCH,
                    validation_data = valid_dataset,
                    validation_steps = VALID_STEPS_PER_EPOCH,
                    epochs=epochs
                    )

    model.save('models/{}.h5'.format(project_name))

train()