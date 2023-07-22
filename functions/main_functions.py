import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt

from PIL import Image

from sklearn.model_selection import train_test_split

import tensorflow as tf
from keras.utils import plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from settings import *

def create_model(input_shape):
    global model, conv2d_layers_count, dense_layers_count, max_pooling_layers_count

    model = tf.keras.Sequential()
    input_shape = ast.literal_eval(input_shape)
    model.add(tf.keras.Input(shape=input_shape))

    conv2d_layers_count, dense_layers_count, max_pooling_layers_count = 0,0,0

    return gr.update(value="Model Created")

def add_conv2d(filters, kernel_size, activation, padding):
    global model, conv2d_layers_count
    try:
        kernel_size = tuple(map(int, kernel_size.split(",")))
        model.add(tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, activation=activation, padding=padding))
        conv2d_layers_count += 1
        return gr.update(value="Added Successfully"), gr.update(value=conv2d_layers_count)
    except:
        return gr.update(value="Adding Failed")

def add_max_pooling(kernel_size):
    global model, max_pooling_layers_count
    try:
        kernel_size = tuple(map(int, kernel_size.split(",")))
        model.add(tf.keras.layers.MaxPooling2D(kernel_size))
        max_pooling_layers_count += 1
        return gr.update(value="Added Successfully"), gr.update(value=max_pooling_layers_count)
    except:
        return gr.update(value="Adding Failed")

def add_flatten():
    global model
    try:
        model.add(tf.keras.layers.Flatten())
        return gr.update(value="Added Successfully")
    except:
        return gr.update(value="Adding Failed")

def add_dense(size, activation):
    global model, dense_layers_count
    try:
        model.add(tf.keras.layers.Dense(size, activation=activation))
        dense_layers_count += 1
        return gr.update(value="Added Successfully"), gr.update(value=dense_layers_count)
    except:
        return gr.update(value="Adding Failed")

def plot_arch():
    global model, img_path
    try:
        plot_model(model, to_file=img_path)
        img_pil = Image.open(img_path)
        return gr.update(value="Saved"), img_pil
    except:
        return gr.update(value="Saving Failed")

def train_model(dataset, optimizer, loss, metrics, epochs):

    dataset = pd.read_csv(dataset.name, encoding='utf-8')
    x = np.array([tf.keras.preprocessing.image.img_to_array(tf.keras.preprocessing.image.load_img(path, target_size=(32, 32))) for path in dataset["path"]])
    y = dataset["target"]

    x = x / 255.0

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

    train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
    )

    val_datagen = ImageDataGenerator()

    batch_size = 32

    train_generator = train_datagen.flow(x_train, y_train, batch_size=batch_size)
    val_generator = val_datagen.flow(x_val, y_val, batch_size=batch_size)

    if "," in metrics:metrics = metrics.split(",")
    else:pass

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    history = model.fit(train_generator, epochs=epochs, validation_data=val_generator)

    val_loss, val_accuracy = model.evaluate(val_generator)

    df = pd.DataFrame(history.history)
    fig, ax = plt.subplots()
    for column in df.columns:
        ax.plot(df.index, df[column], label=column)
    ax.legend()

    return gr.update(value="Done"), str(f"val_loss={round(val_loss,2)} | val_accuracy={round(val_accuracy,2)}"), fig
