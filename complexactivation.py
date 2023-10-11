# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 13:07:42 2022

@author: sb3682
"""

import numpy as np
import tensorflow as tf


def complex_flatten(inputs):
    if len(inputs.shape)==3:
        a = tf.keras.backend.int_shape(inputs[1,1,:])
        b = tf.math.divide(a,2)
        b = int(b[0])
        real = inputs[:,:,:b]
        imag = inputs[:,:,b:]
        real = tf.keras.layers.Flatten()(real)
        imag = tf.keras.layers.Flatten()(imag)
        outputs = tf.concat([real,imag],1)
    elif len(inputs.shape)==4:
        a = tf.keras.backend.int_shape(inputs[1,1,1,:])
        b = tf.math.divide(a,2)
        b = int(b[0])
        real = inputs[:,:,:,:b]
        imag = inputs[:,:,:,b:]
        real = tf.keras.layers.Flatten()(real)
        imag = tf.keras.layers.Flatten()(imag)
        outputs = tf.concat([real,imag],1)
    return outputs


def CReLU(inputs):
    if len(inputs.shape)==2:
        a = tf.keras.backend.int_shape(inputs[1,:])
        b = tf.math.divide(a,2)
        b = int(b[0])
        real = inputs[:,:b]
        imag = inputs[:,b:]
        real = tf.keras.layers.ReLU()(real)
        imag = tf.keras.layers.ReLU()(imag)
        outputs = tf.concat([real,imag],1)
    elif len(inputs.shape)==3:
        a = tf.keras.backend.int_shape(inputs[1,1,:])
        b = tf.math.divide(a,2)
        b = int(b[0])
        real = inputs[:,:,:b]
        imag = inputs[:,:,b:]
        real = tf.keras.layers.ReLU()(real)
        imag = tf.keras.layers.ReLU()(imag)
        outputs = tf.concat([real,imag],2)
    elif len(inputs.shape)==4:
        a = tf.keras.backend.int_shape(inputs[1,1,1,:])
        b = tf.math.divide(a,2)
        b = int(b[0])
        real = inputs[:,:,:,:b]
        imag = inputs[:,:,:,b:]
        real = tf.keras.layers.ReLU()(real)
        imag = tf.keras.layers.ReLU()(imag)
        outputs = tf.concat([real,imag],3)
    return outputs

def complex_softmax(inputs):
    a = tf.keras.backend.int_shape(inputs[1,:])
    b = tf.math.divide(a,2)
    b = int(b[0])
    real = inputs[:,:b]
    imag = inputs[:,b:]
    magnitude = tf.abs(tf.complex(real, imag))
    magnitude = tf.keras.layers.Softmax()(magnitude)
    return magnitude

def complex_bn(inputs):
    if len(inputs.shape)==2:
        a = tf.keras.backend.int_shape(inputs[1,:])
        b = tf.math.divide(a,2)
        b = int(b[0])
        real = inputs[:,:b]
        imag = inputs[:,b:]
        real = tf.keras.layers.BatchNormalization()(real)
        imag = tf.keras.layers.BatchNormalization()(imag)
        outputs = tf.concat([real,imag],1)
    elif len(inputs.shape)==3:
        a = tf.keras.backend.int_shape(inputs[1,1,:])
        b = tf.math.divide(a,2)
        b = int(b[0])
        real = inputs[:,:,:b]
        imag = inputs[:,:,b:]
        real = tf.keras.layers.BatchNormalization()(real)
        imag = tf.keras.layers.BatchNormalization()(imag)
        outputs = tf.concat([real,imag],2)
    elif len(inputs.shape)==4:
        a = tf.keras.backend.int_shape(inputs[1,1,1,:])
        b = tf.math.divide(a,2)
        b = int(b[0])
        real = inputs[:,:,:,:b]
        imag = inputs[:,:,:,b:]
        real = tf.keras.layers.BatchNormalization()(real)
        imag = tf.keras.layers.BatchNormalization()(imag)
        outputs = tf.concat([real,imag],3)
    return outputs

def complex_ln(inputs):
    if len(inputs.shape)==2:
        a = tf.keras.backend.int_shape(inputs[1,:])
        b = tf.math.divide(a,2)
        b = int(b[0])
        real = inputs[:,:b]
        imag = inputs[:,b:]
        real = tf.keras.layers.LayerNormalization(epsilon=1e6)(real)
        imag = tf.keras.layers.LayerNormalization(epsilon=1e6)(imag)
        outputs = tf.concat([real,imag],1)
    elif len(inputs.shape)==3:
        a = tf.keras.backend.int_shape(inputs[1,1,:])
        b = tf.math.divide(a,2)
        b = int(b[0])
        real = inputs[:,:,:b]
        imag = inputs[:,:,b:]
        real = tf.keras.layers.LayerNormalization(epsilon=1e6)(real)
        imag = tf.keras.layers.LayerNormalization(epsilon=1e6)(imag)
        outputs = tf.concat([real,imag],2)
    elif len(inputs.shape)==4:
        a = tf.keras.backend.int_shape(inputs[1,1,1,:])
        b = tf.math.divide(a,2)
        b = int(b[0])
        real = inputs[:,:,:,:b]
        imag = inputs[:,:,:,b:]
        real = tf.keras.layers.LayerNormalization(epsilon=1e6)(real)
        imag = tf.keras.layers.LayerNormalization(epsilon=1e6)(imag)
        outputs = tf.concat([real,imag],3)
    return outputs