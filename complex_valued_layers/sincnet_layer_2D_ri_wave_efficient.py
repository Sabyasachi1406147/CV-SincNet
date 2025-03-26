# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 20:15:41 2025

@author: sb3682
"""

import tensorflow as tf
import math
from numpy import log10, roll, linspace
import numpy as np
import pdb

##############################################################################
#                         Constraints
##############################################################################
class FcConstraint(tf.keras.constraints.Constraint):
    def __init__(self, fs):
        self.fs = fs

    def __call__(self, w):
        return tf.clip_by_value(w, -self.fs / 2, self.fs / 2)

class StartTimeConstraint(tf.keras.constraints.Constraint):
    def __init__(self, input_signal_length, time_length):
        self.input_signal_length = input_signal_length
        self.time_length = time_length

    def __call__(self, w):
        return tf.clip_by_value(w, 0, self.input_signal_length - self.time_length)

class TimeLengthConstraint(tf.keras.constraints.Constraint):
    def __init__(self, input_signal_length):
        self.input_signal_length = input_signal_length

    def __call__(self, w):
        max_time_length = self.input_signal_length
        min_time_length = self.input_signal_length // 10
        return tf.clip_by_value(w, min_time_length, max_time_length)


##############################################################################
#                         Sinc Function (unchanged)
##############################################################################
def sinc(band, t_right):
    """
    Do not change this function.
    """
    y_right = tf.math.sin(2 * math.pi * band * t_right) / (2 * math.pi * band * t_right)
    y_left = tf.reverse(y_right, axis=[0])
    return tf.concat([y_left, tf.constant([1], dtype=tf.float32), y_right], axis=0)


##############################################################################
#                         SincNetLayer1D
##############################################################################
class SincNetLayer1D(tf.keras.layers.Layer):
    def __init__(self, filter_num, filter_size, sampling_freq, **kwargs):
        self.fnum = filter_num
        self.fsize = filter_size
        self.fs = sampling_freq
        self.input_signal_length = 2048
        super(SincNetLayer1D, self).__init__(**kwargs)

    def build(self, input_shape):
        # ------------------ Trainable parameters ------------------ #
        seed = 42
        self.fc = self.add_weight(name='filt_fc',
                                  shape=(self.fnum,),
                                  initializer=tf.keras.initializers.GlorotUniform(seed=seed),
                                  trainable=True,
                                  constraint=FcConstraint(self.fs))

        self.fbandwidth = self.add_weight(name='filt_band',
                                          shape=(self.fnum,),
                                          initializer=tf.keras.initializers.RandomUniform(minval=1e-3, maxval=self.fs/8),
                                          trainable=True,
                                          regularizer=tf.keras.regularizers.l2(1e-4),
                                          constraint=FcConstraint(self.fs))

        self.time_length = self.add_weight(name='time_length',
                                           shape=(self.fnum,),
                                           initializer=tf.keras.initializers.RandomUniform(
                                               minval=self.input_signal_length // 10,
                                               maxval=self.input_signal_length),
                                           trainable=True,
                                           constraint=TimeLengthConstraint(self.input_signal_length))

        self.start_time = self.add_weight(name='start_time',
                                          shape=(self.fnum,),
                                          initializer=tf.keras.initializers.RandomUniform(
                                              minval=0,
                                              maxval=self.input_signal_length),
                                          trainable=True,
                                          constraint=StartTimeConstraint(
                                              self.input_signal_length,
                                              self.time_length))

        # --------------- Mel scale initialization (unchanged) --------------- #
        mel_low = 10
        mel_high = (2595 * log10(1 + (self.fs / 2) / 700))
        mel_points = linspace(mel_low, mel_high, self.fnum // 2)
        freq_points_positive = (700 * (10 ** (mel_points / 2595) - 1))
        freq_points_negative = -(700 * (10 ** (mel_points / 2595) - 1))
        freq_points_negative = np.flip(freq_points_negative, axis=[0])
        freq_points = np.concatenate((freq_points_negative, freq_points_positive), axis=0)

        b1 = roll(freq_points, 1)
        b2 = roll(freq_points, -1)
        b1[0] = b1[1] - 20
        b2[-1] = b2[-2] + 20
        self.freq_scale = self.fs * 1.0

        time_length_values = np.random.uniform(low=self.input_signal_length / 10,
                                               high=self.input_signal_length,
                                               size=self.fnum)
        max_time_length = np.max(time_length_values)
        start_time_values = np.random.uniform(low=0,
                                              high=self.input_signal_length - max_time_length,
                                              size=self.fnum)

        # Set initial values
        self.set_weights([
            b1 / self.fs,
            (b2 - b1) / (2 * self.fs),
            time_length_values,
            start_time_values
        ])

        # ------------------ Precompute/cache constants ------------------ #
        # 1) For Hamming window, we'll generate a fixed array 0..fsize
        n_linspace = np.linspace(0, self.fsize, self.fsize, dtype=np.float32)
        # 2) Hamming window
        window_np = 0.54 - 0.46 * np.cos(2 * math.pi * n_linspace / self.fsize)
        self.n_const = tf.constant(n_linspace, dtype=tf.float32)         # (fsize,)
        self.window_const = tf.constant(window_np, dtype=tf.float32)     # (fsize,)

        # 3) For Sinc's "right side"
        t_right_linspace = np.linspace(1, (self.fsize - 1) / 2, int((self.fsize - 1) / 2), dtype=np.float32)
        self.t_right_const = tf.constant(t_right_linspace / self.fs, dtype=tf.float32)  # shape ~ (fsize//2,)

        super(SincNetLayer1D, self).build(input_shape)

    # Optionally wrap in @tf.function to compile the forward pass
    @tf.function
    def call(self, input_tensor, **kwargs):
        """
        input_tensor shape: (batch_size, length=2048, 2) => [real, imag]
        output shape: (batch_size, length, 2*fnum)
        """
        # -------------------------------------------------------------------
        # 1) Gaussian time window
        # -------------------------------------------------------------------
        # We'll do broadcasting for the Gaussian envelope (rather than tile).
        real_in = input_tensor[..., 0]   # (batch, length)
        imag_in = input_tensor[..., 1]   # (batch, length)

        length = tf.shape(input_tensor)[1]  # 2048
        t = tf.range(length, dtype=tf.float32)  # shape => (length,)

        center_times = (self.start_time + self.time_length) / 2.0         # (fnum,)
        time_len_std = self.time_length / 1.665                           # (fnum,)

        # shape => (fnum, length) via broadcasting => (fnum, length)
        gauss_2d = tf.exp(-0.5 * tf.square((t[None, :] - center_times[:, None]) /
                                           (time_len_std[:, None] + 1e-9)))
        # normalize each filter's row by max
        max_per_row = tf.reduce_max(gauss_2d, axis=1, keepdims=True) + 1e-9
        gauss_2d = gauss_2d / max_per_row  # still shape (fnum, length)

        # transpose => (length, fnum), then expand => (1, length, fnum)
        gauss_2d = tf.expand_dims(tf.transpose(gauss_2d, [1, 0]), axis=0)

        # apply time mask to real/imag
        real_masked = real_in[..., tf.newaxis] * gauss_2d  # => (batch, length, fnum)
        imag_masked = imag_in[..., tf.newaxis] * gauss_2d

        # for depthwise conv => (batch, length, 1, fnum)
        real_masked_4d = tf.expand_dims(real_masked, axis=2)
        imag_masked_4d = tf.expand_dims(imag_masked, axis=2)

        # -------------------------------------------------------------------
        # 2) Vectorized Sinc filter creation
        # -------------------------------------------------------------------
        band_values = self.fbandwidth * self.fs  # (fnum,)

        # Instead of tf.map_fn, we use tf.vectorized_map if possible
        # (requires TF >= 2.0). It's often more efficient:
        @tf.function
        def _sinc_fn(b):
            return 2.0 * b * sinc(b, self.t_right_const)

        # shape => (fnum, fsize)
        bandpasses = tf.vectorized_map(_sinc_fn, band_values)

        # multiply by Hamming window
        bandpasses *= self.window_const  # broadcast => (fnum, fsize)

        # normalize row by max
        max_per_filter = tf.reduce_max(tf.abs(bandpasses), axis=1, keepdims=True) + 1e-9
        bandpasses /= max_per_filter

        # modulate by cos/sin(2*pi*fc*n)
        fc_expanded = tf.reshape(self.fc, [-1, 1])      # (fnum,1)
        n_expanded  = tf.reshape(self.n_const, [1, -1]) # (1,fsize)

        cos_val = tf.cos(2.0 * math.pi * fc_expanded * n_expanded)
        sin_val = tf.sin(2.0 * math.pi * fc_expanded * n_expanded)

        real_all = bandpasses * cos_val  # (fnum, fsize)
        imag_all = bandpasses * sin_val  # (fnum, fsize)

        # reshape => (fsize,1,fnum)
        filters_real = tf.transpose(real_all, [1, 0])  # => (fsize, fnum)
        filters_real = tf.reshape(filters_real, [self.fsize, 1, self.fnum])

        filters_im = tf.transpose(imag_all, [1, 0])    # => (fsize, fnum)
        filters_im = tf.reshape(filters_im, [self.fsize, 1, self.fnum])

        # expand => (fsize,1,fnum,1) for depthwise conv
        filters_real_4d = tf.expand_dims(filters_real, axis=-1)
        filters_im_4d   = tf.expand_dims(filters_im,   axis=-1)

        # -------------------------------------------------------------------
        # 3) Depthwise Convolution for complex multiply
        #    out_real = rr - ii
        #    out_imag = ri + ir
        # -------------------------------------------------------------------
        out_rr = tf.nn.depthwise_conv2d(real_masked_4d, filters_real_4d,
                                        strides=[1, 1, 1, 1],
                                        padding='SAME')
        out_ii = tf.nn.depthwise_conv2d(imag_masked_4d, filters_im_4d,
                                        strides=[1, 1, 1, 1],
                                        padding='SAME')
        out_ri = tf.nn.depthwise_conv2d(real_masked_4d, filters_im_4d,
                                        strides=[1, 1, 1, 1],
                                        padding='SAME')
        out_ir = tf.nn.depthwise_conv2d(imag_masked_4d, filters_real_4d,
                                        strides=[1, 1, 1, 1],
                                        padding='SAME')

        # squeeze => (batch, length, fnum)
        out_rr = tf.squeeze(out_rr, axis=2)
        out_ii = tf.squeeze(out_ii, axis=2)
        out_ri = tf.squeeze(out_ri, axis=2)
        out_ir = tf.squeeze(out_ir, axis=2)

        out_real = out_rr - out_ii
        out_imag = out_ri + out_ir

        # final => (batch, length, 2*fnum)
        return tf.concat([out_real, out_imag], axis=2)

    # Optional get_config
    """
    def get_config(self):
        config = super().get_config()
        config.update({
            "filter_num": self.fnum,
            "filter_size": self.fsize,
            "sampling_freq": self.fs,
            "input_signal_length": self.input_signal_length,
        })
        return config
    """


class ComplexConcatenate(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ComplexConcatenate, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ComplexConcatenate, self).build(input_shape)

    @tf.function
    def call(self, complex_inputs):
        real_parts = []
        imag_parts = []
        for tensor in complex_inputs:
            # Split each tensor along the last dimension into real and imaginary parts.
            real, imag = tf.split(tensor, num_or_size_splits=2, axis=-1)
            real_parts.append(real)
            imag_parts.append(imag)
        # Concatenate all real parts first, then all imaginary parts.
        output = tf.concat(real_parts + imag_parts, axis=-1)
        return output