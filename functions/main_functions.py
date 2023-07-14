import tensorflow as tf
from keras.utils import plot_model
import ast
import gradio as gr

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
        return gr.update(value="Added Successfully")
    except:
        return gr.update(value="Adding Failed")

def add_max_pooling(kernel_size):
    try:
        kernel_size = map(int, kernel_size.split(","))
        model = tf.keras.models.load_model(get_model_path())
        model.add(tf.keras.layers.MaxPooling2D(kernel_size))
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