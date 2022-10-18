import re
import numpy as np
from flask import Flask, jsonify, render_template, Response, request
import os
from camera import VideoCamera
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import pathlib
import pandas as pd


app = Flask(__name__, static_folder='static')


@app.route('/')
def home():
    img = '/static/bhanu.svg'
    return render_template("index.html", img=img)


@app.route('/train')
def train():
    img = '/static/bhanu.svg'
    return render_template("train.html", img=img)


@app.route('/image-classification', methods=['GET', 'POST'])
def img_class():
    if request.method == 'POST':
        result = request.form
        name_project = result['project']
        train_location = result['train_location']
        test_location = result['test_location']
        img_wid = int(result['width'])
        img_hei = int(result['height'])
        no_class = int(result['class'])
        epochs = int(result['epochs'])
        batch_size = int(result['batch_size'])
        print(type(name_project), type(train_location), type(test_location), type(img_wid), type(img_hei), type(no_class), type(epochs), type(batch_size))
        train(train_location, test_location, name_project, no_class, epochs, img_wid, img_hei, batch_size, activation_function='relu')

    img = '/static/bhanu.svg'
    return render_template("img_class.html", img=img)


@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def train(train_location, valid_location, project_name, noc, epochs, img_wid, img_hei, batch_size, activation_function='relu'):
    train_data_dir = pathlib.Path(train_location)
    image_count = len(list(train_data_dir.glob('train/*/*.jpg')))
    valid_data_dir = pathlib.Path(valid_location)
    STEPS_PER_EPOCH = np.ceil(image_count/batch_size)

    train = ImageDataGenerator(rescale=1/255)
    valid = ImageDataGenerator(rescale=1/255)
    train_dataset = train.flow_from_directory(train_location)
    valid_dataset = valid.flow_from_directory(valid_location)

    VALID_STEPS_PER_EPOCH = np.ceil(valid_dataset.samples/batch_size)

    # Model
    VGG19 = tf.keras.applications.VGG19(input_shape = (img_wid,img_hei,3),
                                                                include_top=False, 
                                                                weights='imagenet')
    VGG19.trainable = False
    if noc == 2:
        prediction_layer = tf.keras.layers.Dense(noc, activation_function)
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        model = tf.keras.Sequential([
        VGG19,
        global_average_layer,
        prediction_layer
        ])
        model.compile(optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'])
    else:
        prediction_layer = tf.keras.layers.Dense(noc, activation = 'softmax')
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        model = tf.keras.Sequential([
        VGG19,
        global_average_layer,
        prediction_layer
        ])
        model.compile(optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    
    history = model.fit_generator(
                    train_dataset,
                    steps_per_epoch=STEPS_PER_EPOCH,
                    validation_data = valid_dataset,
                    validation_steps = VALID_STEPS_PER_EPOCH,
                    epochs=epochs
                    )

    model.save('models/{}.h5'.format(project_name))


if __name__ == "__main__":
    app.run(debug=True)
