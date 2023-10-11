# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 10:55:01 2022

@author: sb3682
"""
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Pooling layers."""

import tensorflow as tf
import math
from numpy import log10, roll, linspace

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
        a = input_tensor.shape[2]
        b = tf.math.divide(a,2)
        b = tf.cast(b, tf.int32)
        input_tensor_real = input_tensor[:,:,tf.newaxis,:b]
        input_tensor_imag = input_tensor[:,:,tf.newaxis,b:]
        input_tensor = tf.dtypes.complex(input_tensor_real, input_tensor_imag, name=None)
        input_tensor_abs = tf.math.abs(input_tensor_real)
        output_abs, argmax = tf.nn.max_pool_with_argmax(input=input_tensor_abs, ksize=pool_shape, strides=strides,
                                                    padding=self.padding, data_format="NHWC", include_batch_in_index=True)
        output_shape = tf.shape(tf.squeeze(output_abs, axis=2))
        output_real = tf.reshape(tf.gather(tf.reshape(input_tensor_real, [-1]), argmax), output_shape)
        output_imag = tf.reshape(tf.gather(tf.reshape(input_tensor_imag, [-1]), argmax), output_shape)
        output = tf.concat([output_real,output_imag], axis=2)
        return output