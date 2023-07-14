import tensorflow as tf
from keras.utils import plot_model
import ast
import gradio as gr
import shutil
from settings import *

def create_model(input_shape):
    input_shape = ast.literal_eval(input_shape)

    model = tf.keras.Sequential()
    model.add(tf.keras.Input(input_shape))
    model.save("models/sequential")

    return gr.update(value=f"Model Created with (input_shape={input_shape})")

def get_model_path():
    return "models/sequential"

def add_conv2d(filters, kernel_size, activation, padding):
    try:
        kernel_size = map(int, kernel_size.split(","))
        model = tf.keras.models.load_model(get_model_path())
        model.add(tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, activation=activation, padding=padding))
        update_layer_count("conv2d")
        return gr.update(value="Added Successfully")
    except:
        return gr.update(value="Adding Failed")

def add_max_pooling(kernel_size):
    try:
        kernel_size = map(int, kernel_size.split(","))
        model = tf.keras.models.load_model(get_model_path())
        model.add(tf.keras.layers.MaxPooling2D(kernel_size))
        update_layer_count("max_pooling")
        return gr.update(value="Added Successfully")
    except:
        return gr.update(value="Adding Failed")

def add_flatten():
    try:
        model = tf.keras.models.load_model(get_model_path())
        model.add(tf.keras.layers.Flatten())
        return gr.update(value="Added Successfully")
    except:
        return gr.update(value="Adding Failed")

def add_dense(size, activation):
    try:
        model = tf.keras.models.load_model(get_model_path())
        model.add(tf.keras.layers.Dense(size, activation=activation))
        update_layer_count("dense")
        return gr.update(value="Added Successfully")
    except:
        return gr.update(value="Adding Failed")

def plot_architecture():
    try:
        img_path = "graphs/model_architecture.png"
        model = tf.keras.models.load_model(get_model_path())
        plot_model(model, to_file=img_path)
        return gr.update(value="Saved Successfully")
    except:
        return gr.update(value="Saving Failed")

def delete_model():
    shutil.rmtree(get_model_path())

def update_layer_count(layer_type):
    
    layers_dict = {
        
       'dense':dense_layers_count,
       'conv2d': conv2d_layers_count,
       'max_pooling': max_pooling_layers_count
    }

    layers_dict[layer_type] += 1
