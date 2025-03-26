# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 23:03:54 2022

@author: sb3682
"""

import tensorflow as tf
import math
from numpy import log10, roll, linspace


class SincNetLayer1DComplex(tf.keras.layers.Layer):
    def __init__(self, filter_num, filter_size, sampling_freq, **kwargs):
        self.fnum = filter_num
        self.fsize = filter_size
        self.fs = tf.constant(sampling_freq, dtype=tf.float32)
        super(SincNetLayer1DComplex, self).__init__(**kwargs)

    def build(self, input_shape):
        # Initialization of the filterbanks in mel scale.
        mel_low = 80
        mel_high = (2595 * log10(1 + (self.fs / 2) / 700))              # Converting Hz to Mel
        mel_points = linspace(mel_low, mel_high, self.fnum)             # Array of equally spaced frequencies in Mel scale
        freq_points = (700 * (10 ** (mel_points / 2595) - 1))           # Converting Mel back to Hz
        
        b1 = roll(freq_points, 1)
        b2 = roll(freq_points, -1)
        b1[0] = 20
        b2[-1] = (self.fs / 2) - 100
        
        #########################   real
        self.f1_real = tf.Variable(initial_value=b1 / self.fs, trainable=True, name='filter_f1')
        self.fbandwith_real = tf.Variable(initial_value=(b2 - b1) / self.fs, trainable=True, name='filter_bandwidth')
        
        # Get absolutes of the first and second cut-off frequencies of the filters.
        self.f1_abs_real = tf.math.abs(self.f1_real)
        self.f2_abs_real = self.f1_abs_real + (tf.math.abs(self.fbandwith_real))
        ######################################
        
        ###########################  im
        self.f1_im = tf.Variable(initial_value=b1 / self.fs, trainable=True, name='filter_f1')
        self.fbandwith_im = tf.Variable(initial_value=(b2 - b1) / self.fs, trainable=True, name='filter_bandwidth')
        
        # Get absolutes of the first and second cut-off frequencies of the filters.
        self.f1_abs_im = tf.math.abs(self.f1_im)
        self.f2_abs_im = self.f1_abs_im + (tf.math.abs(self.fbandwith_im))
        #####################################
        
        # Filter window (hamming).
        n = linspace(0, self.fsize, self.fsize)
        window = 0.54 - 0.46 * tf.math.cos(2 * math.pi * n / self.fsize)
        self.window = tf.constant(tf.cast(window, dtype=tf.float32), dtype=tf.float32, name='window')
        
        # Defining the points for sinc function in time domain.
        t_right_linspace = linspace(1, (self.fsize - 1) / 2, int((self.fsize - 1) / 2))
        self.t_right = tf.constant(t_right_linspace / self.fs, dtype=tf.float32, name='t_right')
        
        super(SincNetLayer1DComplex, self).build(input_shape)  # Be sure to call this at the end
        
    def call(self, input_tensor, **kwargs):
        # Compute the filters.
        real = tf.math.real(input_tensor)
        imag = tf.math.imag(input_tensor)
        output_list_real = []
        output_list_im = []
        for i in range(self.fnum):
            low_pass1_real = 2 * self.f1_abs_real[i] * sinc(self.f1_abs_real[i] * self.fs, self.t_right)
            low_pass2_real = 2 * self.f2_abs_real[i] * sinc(self.f2_abs_real[i] * self.fs, self.t_right)
            band_pass_real = (low_pass2_real - low_pass1_real)
            band_pass_real = band_pass_real / tf.math.reduce_max(band_pass_real)
            output_list_real.append(band_pass_real * self.window)
        filters_real = tf.keras.backend.stack(output_list_real)                               # (80,251)
        filters_real = tf.keras.backend.transpose(filters_real)                               # (251,80)
        filters_real = tf.keras.backend.reshape(filters_real, (self.fsize, 1, self.fnum))     # (251,1,80) 
        
        for i in range(self.fnum):
            low_pass1_im = 2 * self.f1_abs_im[i] * sinc(self.f1_abs_im[i] * self.fs, self.t_right)
            low_pass2_im = 2 * self.f2_abs_im[i] * sinc(self.f2_abs_im[i] * self.fs, self.t_right)
            band_pass_im = (low_pass2_im - low_pass1_im)
            band_pass_im = band_pass_im / tf.math.reduce_max(band_pass_im)
            output_list_im.append(band_pass_im * self.window)
        filters_im = tf.keras.backend.stack(output_list_im)                               # (80,251)
        filters_im = tf.keras.backend.transpose(filters_im)                               # (251,80)
        filters_im = tf.keras.backend.reshape(filters_im, (self.fsize, 1, self.fnum))
        # Do the convolution.
        out_rr = tf.keras.backend.conv1d(real, kernel=filters_real)
        out_ii = tf.keras.backend.conv1d(imag, kernel=filters_im)
        out_ri = tf.keras.backend.conv1d(real, kernel=filters_im)
        out_ir = tf.keras.backend.conv1d(imag, kernel=filters_real)
        
        out_real = (out_rr-out_ii) 
        out_imag = (out_ri+out_ir)
        out = tf.complex(out_real, out_imag)
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

"""
Max pooling operation for 1D spatial data.
"""
class ComplexMaxPooling1D(tf.keras.layers.Layer):
    def __init__(self, pool_size = 2, strides = None, padding ='VALID', dtype = None, name = None, **kwargs):
        if strides is None:
            strides = pool_size
        self.pool_size = (pool_size, 1)
        self.strides = (strides, 1)
        self.padding = padding
        super(ComplexMaxPooling1D, self).__init__(**kwargs)
        
    def build(self, input_shape):
        super(ComplexMaxPooling1D, self).build(input_shape)
        
    def call(self, input_tensor, **kwargs):
        pool_shape = (1,) + self.pool_size + (1,)
        strides = (1,) + self.strides + (1,)
        input_tensor_nhwc = input_tensor[:,:,tf.newaxis,:]
        input_tensor_abs = tf.math.abs(input_tensor_nhwc)
        output_abs, argmax = tf.nn.max_pool_with_argmax(input=input_tensor_abs, ksize=pool_shape, strides=strides,
                                                    padding=self.padding, data_format="NHWC", include_batch_in_index=True)
        output_shape = tf.shape(tf.squeeze(output_abs, axis=2))
        return tf.reshape(tf.gather(tf.reshape(input_tensor, [-1]), argmax), output_shape)
        
