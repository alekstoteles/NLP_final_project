# USAGE
# Start the server:
# 	python run_keras_server.py
# Submit a request via cURL:
# 	curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
# Submita a request via Python:
#	python simple_request.py

# import the necessary packages
from __future__ import print_function

import os
from os import listdir
import flask
import io
from flask import Flask, render_template, request
from wtforms import Form, TextField, TextAreaField, validators, SubmitField, DecimalField, IntegerField
import pickle
import wget
from pathlib import Path

import numpy as np
from numpy import array
from numpy import asarray
from numpy import zeros
import pandas as pd
import tensorflow as tf
import h5py
from keras import initializers
from keras import backend as K
from keras.models import Sequential
from keras.models import Model
from keras.utils import CustomObjectScope
# from keras.models import load_model
# from tensorflow.keras.models import load_model
from tensorflow.python.keras.models import load_model
from keras.layers import Flatten
from keras.layers import GlobalMaxPooling1D
from keras.layers import Input, InputLayer
from keras.layers import Embedding, Activation, Dropout, Dense, Conv1D, MaxPooling1D, GlobalMaxPooling1D, BatchNormalization
from keras.layers.merge import Concatenate
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import one_hot
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None
tokenizer = None
embedding_matrix = None
CRC_list = []

def load_tokenizer():
    path = 'data/'
    embeddings_dictionary = dict()
    global tokenizer
    
    # load tokenizer
    with open(path + 'tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

def load_embedding_matrix():
    path = 'data/'
    embeddings_dictionary = dict()
    global embedding_matrix
    
    # initialize GloVe word embeddings matrix based on the words in the tokenizer

    glove_file = open(path + "glove/glove.6B.100d.txt", encoding="utf8")

    for line in glove_file:
        records = line.split()
        word = records[0]
        vector_dimensions = asarray(records[1:], dtype='float32')
        embeddings_dictionary[word] = vector_dimensions
    glove_file.close()

    vocab_size = len(tokenizer.word_index) + 1
    embedding_matrix = zeros((vocab_size, 100))
    for word, index in tokenizer.word_index.items():
        embedding_vector = embeddings_dictionary.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector
    print("Embedding matrix loaded.")

def convert_to_0_1_1D(a):
    """Return row - for each val in row: 1 if val > 0.5, else 0"""
    return np.round(a).astype(int)

def load_crc_list():
    # load list of CRC labels that aligns to one-hot encoding
    global CRC_list
    path = 'data/'
    CRC_list = pd.read_csv(path + 'crc_labels.csv', header=None)
    print("CRC list loaded.")

def load_crc_model():
    global model
    path = './data/'
    remote_path = 'https://kstonedev.s3-us-west-2.amazonaws.com/W266/'

    # Define custom functions used in model training
    def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall
    def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    def f1(y_true, y_pred):
        precision = precision_m(y_true, y_pred)
        recall = recall_m(y_true, y_pred)
        return 2*((precision*recall)/(precision+recall+K.epsilon()))
    def weighted_bce(y_true, y_pred):
        # weights become 2 if y_true is 1, and 1 if y_true is 0
        weights = (y_true * 2.) + (1. - y_true)
        bce = K.binary_crossentropy(y_true, y_pred)
        weighted_bce = K.mean(bce * weights)
        return weighted_bce


    # if first time, download model
    try:
        filepath = Path(path + 'model.h5').resolve(strict=True)
    except FileNotFoundError:
        filename = wget.download(remote_path + 'model.h5', out=path)
        print("\nModel dowloaded.")

    # load model
    model = load_model(path + 'model.h5', custom_objects={'weighted_bce': weighted_bce, 'f1': f1})
    print("Model loaded.")

def generate_inference(model, abstract):
    """Generate output with CRC labels for abstract"""

    abstractx = tokenizer.texts_to_sequences([abstract])
    abstractx = pad_sequences(abstractx, padding='post', maxlen=200)
    y_pred = model.predict(abstractx)
    y_pred_labels = list(np.array(CRC_list).reshape(-1,)[convert_to_0_1_1D(y_pred).astype(bool).reshape(-1,)])

    # Formatting in html
    html = ''
    html = addContent(html, header(
        'Output', color='black'))
    html = addContent(html, box(abstract))
    html = addContent(html, box(y_pred_labels))
    return f'<div>{html}</div>'


def header(text, color='black', gen_text=None):
    """Create an HTML header"""

    if gen_text:
        raw_html = f'<h1 style="margin-top:16px;color: {color};font-size:54px"><center>' + str(
            text) + '<span style="color: red">' + str(gen_text) + '</center></h1>'
    else:
        raw_html = f'<h1 style="margin-top:12px;color: {color};font-size:54px"><center>' + str(
            text) + '</center></h1>'
    return raw_html


def box(text, gen_text=None):
    """Create an HTML box of text"""

    if gen_text:
        raw_html = '<div style="padding:8px;font-size:28px;margin-top:28px;margin-bottom:14px;">' + str(
            text) + '<span style="color: red">' + str(gen_text) + '</div>'

    else:
        raw_html = '<div style="border-bottom:1px inset black;border-top:1px inset black;padding:8px;font-size: 1.1em;">' + str(
            text) + '</div>'
    return raw_html


def addContent(old_html, raw_html):
    """Add html content together"""

    old_html += raw_html
    return old_html

class ReusableForm(Form):
    abstract = TextAreaField("Enter abstract text:", validators=[validators.InputRequired()])

    # Submit button
    submit = SubmitField("Enter")

# Home page
@app.route("/", methods=['GET', 'POST'])
def home():
    """Home page of app with form"""
    # Create form
    form = ReusableForm(request.form)

    # On form entry and all conditions met
    if request.method == 'POST' and form.validate():
        # Extract information
        abstract = request.form['abstract']
        return render_template('inference_output.html', input=generate_inference(model=model, abstract=abstract))
    else:
        # Send template information to index.html
        return render_template('index.html', form=form)
    
@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the view
    data = {"success": False}

    if flask.request.method == "POST":
        sample = [flask.request.data.decode("utf-8")]

        sample = tokenizer.texts_to_sequences(sample)
        sample = pad_sequences(sample, padding='post', maxlen=200)
        y_pred = model.predict(sample)
        y_pred_labels = list(np.array(CRC_list).reshape(-1,)[convert_to_0_1_1D(y_pred).astype(bool).reshape(-1,)])
        data["success"] = True
        data["labels"] = y_pred_labels

        print(f"{y_pred_labels}")

        return flask.jsonify(data)

# if this is the main thread of execution first load the model and then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    load_crc_model()
    load_tokenizer()
    # load_embedding_matrix()
    load_crc_list()
    app.run(debug=False, threaded=False)
