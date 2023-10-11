# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 23:03:54 2022

@author: sb3682
"""

import tensorflow as tf
import math
from numpy import log10, roll, linspace
import numpy as np

class SincNetLayer1D(tf.keras.layers.Layer):
    def __init__(self, filter_num, filter_size, sampling_freq, **kwargs):
        self.fnum = filter_num
        self.fsize = filter_size
        self.fs = sampling_freq
        super(SincNetLayer1D, self).__init__(**kwargs)

    def build(self, input_shape):
        # Initialization of the filterbanks in mel scale.
        self.f1_real = self.add_weight(name='filt_b1_real',
                        shape=(self.fnum,),
                        initializer='glorot_uniform',
                        trainable=True)

        self.fbandwidth = self.add_weight(name='filt_band_real',
                        shape=(self.fnum,),
                        initializer='glorot_uniform',
                        trainable=True)
        
        # self.f1_im = self.f1_real

        # self.fbandwidth_im = self.fbandwidth_real
        
        mel_low = 10
        mel_high = (2595 * log10(1 + (self.fs / 2) / 700))              # Converting Hz to Mel
        mel_points = linspace(mel_low, mel_high, self.fnum//2)               # Array of equally spaced frequencies in Mel scale
        freq_points_positive = (700 * (10 ** (mel_points / 2595) - 1))
        freq_points_negative = -(700 * (10 ** (mel_points / 2595) - 1))           # Converting Mel back to Hz
        freq_points_negative = np.flip(freq_points_negative,axis=[0])
        freq_points = np.concatenate((freq_points_negative,freq_points_positive),axis=0)
        b1 = roll(freq_points, 1)
        b2 = roll(freq_points, -1)
        b1[0] = b1[1]-20
        b2[-1] = b2[-2]+20
        self.freq_scale = self.fs * 1.0
        self.set_weights([b1/self.freq_scale, (b2-b1)/(2*self.freq_scale)])
        #self.set_weights([b1_real/self.freq_scale, (b2_real-b1_real)/self.freq_scale])
        
        super(SincNetLayer1D, self).build(input_shape)  # Be sure to call this at the end
        
    def call(self, input_tensor, **kwargs):
        #####################################

        self.fc = (self.f1_real + tf.abs(self.fbandwidth)/2)
        
        ######################################
        
        # Filter window (hamming).
        n = linspace(0, self.fsize, self.fsize)
        window1 = 0.54 - 0.46 * tf.math.cos(2 * math.pi * n / self.fsize)
        self.window = tf.cast(window1, "float32")
        #window = tf.Variable(initial_value=window2, trainable=True, dtype=tf.float32, name='window')
        
        # Defining the points for sinc function in time domain.
        t_right_linspace = linspace(1, (self.fsize - 1) / 2, int((self.fsize - 1) / 2))
        self.t_right = tf.constant(t_right_linspace / self.fs, dtype=tf.float32, name='t_right')
        # Compute the filters.
        real = input_tensor[:,:,0,tf.newaxis]
        imag = input_tensor[:,:,1,tf.newaxis]
        output_list_real = []
        output_list_im = []
        for i in range(self.fnum):
            bandpass = 2 * self.fbandwidth[i]*self.fs*sinc(self.fbandwidth[i]*self.fs,self.t_right)
            bandpass = bandpass*self.window
            bandpass = bandpass/tf.math.reduce_max(bandpass)
            band_pass_real = bandpass*tf.math.cos(2*math.pi*self.fc[i]*n)
            band_pass_imag = bandpass*tf.math.sin(2*math.pi*self.fc[i]*n)
            output_list_real.append(band_pass_real)
            output_list_im.append(band_pass_imag)
            
            
        filters_real = tf.keras.backend.stack(output_list_real)                               # (80,251)
        filters_real = tf.keras.backend.transpose(filters_real)                               # (251,80)
        filters_real = tf.keras.backend.reshape(filters_real, (self.fsize, 1, self.fnum))     # (251,1,80) 

        filters_im = tf.keras.backend.stack(output_list_im)                               # (80,251)
        filters_im = tf.keras.backend.transpose(filters_im)                               # (251,80)
        filters_im = tf.keras.backend.reshape(filters_im, (self.fsize, 1, self.fnum))     # (251,1,80) 
        # Do the convolution.
        out_rr = tf.keras.backend.conv1d(real, kernel=filters_real)
        out_ii = tf.keras.backend.conv1d(imag, kernel=filters_im)
        out_ri = tf.keras.backend.conv1d(real, kernel=filters_im)
        out_ir = tf.keras.backend.conv1d(imag, kernel=filters_real)
        
        out_real = (out_rr-out_ii)
        out_imag = (out_ri+out_ir)
        # out_real = out_rr
        # out_imag = out_ir
        
        out = tf.concat([out_real, out_imag], 2)
        return out

    def compute_output_shape(self, input_shape):
        new_size = tf.python.keras.utils.conv_utils.conv_output_length(input_shape[1], self.fsize, 
                                                                       padding="valid", stride=1, dilation=1)
        return (input_shape[0],) + (new_size,) + (self.fnum,)
"""
    # Overriding get_config method as __init__ function has positional arguements
    def get_config(self):
        return {"Number of filters": self.fnum,
                "Filter size": self.fsize,
                "Sampling Frequency":self.fs}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
"""
def sinc(band, t_right):
    y_right = tf.math.sin(2 * math.pi * band * t_right) / (2 * math.pi * band * t_right)
    y_left = tf.reverse(y_right, axis=[0])
    return tf.concat([y_left, tf.constant([1], dtype=tf.float32), y_right], axis=0)

class ComplexConcatenate(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ComplexConcatenate, self).__init__(**kwargs)
        
    def build(self, input_shape):
        super(ComplexConcatenate, self).build(input_shape)
        
    def call(self, complex_inputs):
        input1 = complex_inputs[0]
        input2 = complex_inputs[1]
        input3 = complex_inputs[2]
        input1_shape = input1.shape
        real_input1 = input1[:,:,:int(input1_shape[2]/2)]
        imag_input1 = input1[:,:,int(input1_shape[2]/2):]
        input2_shape = input2.shape
        real_input2 = input2[:,:,:int(input2_shape[2]/2)]
        imag_input2 = input2[:,:,int(input2_shape[2]/2):]
        input3_shape = input3.shape
        real_input3 = input3[:,:,:int(input3_shape[2]/2)]
        imag_input3 = input3[:,:,int(input3_shape[2]/2):]
        
        output = tf.concat([real_input1, real_input2, real_input3, imag_input1, imag_input2, imag_input3], axis=2)
        return output