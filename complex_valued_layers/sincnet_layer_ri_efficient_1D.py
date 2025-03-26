# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 09:03:08 2025

@author: sb3682
"""
import tensorflow as tf
import math
from numpy import log10, roll, linspace
import numpy as np

# Constraint so that f1_real + bandwidth does not exceed Nyquist
class FcConstraint(tf.keras.constraints.Constraint):
    def __init__(self, fs):
        self.fs = fs

    def __call__(self, w):
        return tf.clip_by_value(w, -self.fs / 2, self.fs / 2)

# A helper function to build the symmetric 'sinc' in a vectorized manner
def sinc_banks(band, t_right):
    """
    band:     shape (fnum, 1) -> each row is a band[i]
    t_right:  shape (1, half_size)

    Returns a tensor of shape (fnum, full_size),
    where full_size = 2*half_size + 1.
    """
    # y_right: shape = (fnum, half_size)
    y_right = (
        tf.math.sin(2.0 * math.pi * band * t_right)
        / (2.0 * math.pi * band * t_right)
    )
    y_left = tf.reverse(y_right, axis=[1])  # mirror around the center
    center = tf.ones([tf.shape(band)[0], 1], dtype=tf.float32)
    return tf.concat([y_left, center, y_right], axis=1)  # (fnum, full_size)

class SincNetLayer1D(tf.keras.layers.Layer):
    def __init__(self, filter_num, filter_size, sampling_freq, **kwargs):
        self.fnum = filter_num
        self.fsize = filter_size
        self.fs = sampling_freq
        super(SincNetLayer1D, self).__init__(**kwargs)

    def build(self, input_shape):
        # Trainable parameters for center frequencies and bandwidths
        self.f1_real = self.add_weight(
            name='filt_b1_real',
            shape=(self.fnum,),
            initializer='glorot_uniform',
            trainable=True
        )
        self.fbandwidth = self.add_weight(
            name='filt_band_real',
            shape=(self.fnum,),
            initializer='glorot_uniform',
            trainable=True,
            constraint=FcConstraint(self.fs)
        )

        # Initialize using a mel-spaced arrangement around +/- fs/2
        mel_low = 10.0
        mel_high = 2595.0 * log10(1 + (self.fs / 2) / 700.0)  # Hz->Mel
        # half the filters for negative freq, half for positive freq
        mel_points_positive = linspace(mel_low, mel_high, self.fnum // 2)
        freq_points_positive = 700.0 * (10.0 ** (mel_points_positive / 2595.0) - 1.0)

        mel_points_negative = -(mel_points_positive[::-1])
        freq_points_negative = 700.0 * (10.0 ** (mel_points_negative / 2595.0) - 1.0)

        freq_points = np.concatenate((freq_points_negative, freq_points_positive), axis=0)
        b1 = roll(freq_points, 1)
        b2 = roll(freq_points, -1)
        b1[0] = b1[1] - 20.0
        b2[-1] = b2[-2] + 20.0

        self.freq_scale = float(self.fs)
        # We store center freq in f1_real and bandwidth in fbandwidth
        # dividing by freq_scale because the forward pass multiplies by fs
        self.set_weights([
            b1 / self.freq_scale,            # self.f1_real init
            (b2 - b1) / (2 * self.freq_scale)  # self.fbandwidth init
        ])

        super(SincNetLayer1D, self).build(input_shape)  # must call at end

    @tf.function
    def call(self, input_tensor, **kwargs):
        """
        Vectorized implementation of the Sinc layer.
        input_tensor: shape [batch, time, 2] or [batch, time, 2, 1]
                      where the last dimension is real/imag stacked as (x, :, 0) = real, (x, :, 1) = imag
        Returns:
          Complex conv: shape [batch, new_time, 2*fnum]
        """
        # 1) Compute the actual center frequencies = f1 + (|bandwidth|)/2
        #    This is analogous to the original code's self.fc
        fc = self.f1_real + tf.abs(self.fbandwidth) / 2.0

        # 2) Build a Hamming window (size = self.fsize)
        n = tf.range(0, self.fsize, dtype=tf.float32)
        window = 0.54 - 0.46 * tf.math.cos(2.0 * math.pi * n / (tf.cast(self.fsize, tf.float32) - 1.0))

        # 3) Build the (fnum, fsize) bandpass filters via vectorized sinc
        half_size = (self.fsize - 1) // 2
        # time points for "right" side of the sinc
        t_right = tf.range(1, half_size + 1, dtype=tf.float32) / self.fs
        t_right = tf.reshape(t_right, (1, -1))  # shape [1, half_size]

        # shape: (fnum,) -> reshape to (fnum,1) for broadcasting
        band = self.fbandwidth * self.fs
        band = tf.reshape(band, (self.fnum, 1))

        # Sinc shape: (fnum, self.fsize)
        bandpass = sinc_banks(band, t_right)  # includes left+center+right
        bandpass = 2.0 * band * bandpass      # multiply by 2*band

        # Apply the window (broadcast window of shape (1, fsize))
        window_2d = tf.reshape(window, (1, self.fsize))
        bandpass = bandpass * window_2d

        # Normalize each filter by its maximum
        # (use absolute value if you want guaranteed positivity)
        max_per_filter = tf.reduce_max(tf.abs(bandpass), axis=1, keepdims=True)
        bandpass = bandpass / max_per_filter

        # 4) Modulate each filter by cos/sin(2 pi fc * n)
        #    fc shape: (fnum,) -> reshape to (fnum,1)
        fc_2d = tf.reshape(fc, (self.fnum, 1))

        # 'n' is 0..fsize-1
        n_2d = tf.reshape(n, (1, self.fsize))

        filt_cos = tf.math.cos(2.0 * math.pi * fc_2d * n_2d)  # shape (fnum, fsize)
        filt_sin = tf.math.sin(2.0 * math.pi * fc_2d * n_2d)

        filters_real = bandpass * filt_cos
        filters_imag = bandpass * filt_sin

        # Reshape to [kernel_size, in_channels, out_channels] = [fsize, 1, fnum]
        filters_real = tf.transpose(filters_real, perm=[1, 0])  # (fsize, fnum)
        filters_imag = tf.transpose(filters_imag, perm=[1, 0])  # (fsize, fnum)

        filters_real = tf.reshape(filters_real, (self.fsize, 1, self.fnum))
        filters_imag = tf.reshape(filters_imag, (self.fsize, 1, self.fnum))

        # 5) Split real/imag from input
        #    Suppose input_tensor is shape (batch, time, 2) with [real, imag]
        real = input_tensor[:, :, 0:1]  # shape (batch, time, 1)
        imag = input_tensor[:, :, 1:2]  # shape (batch, time, 1)

        # 6) Perform the complex conv via conv1d
        out_rr = tf.keras.backend.conv1d(real, kernel=filters_real, padding='valid')
        out_ii = tf.keras.backend.conv1d(imag, kernel=filters_imag, padding='valid')
        out_ri = tf.keras.backend.conv1d(real, kernel=filters_imag, padding='valid')
        out_ir = tf.keras.backend.conv1d(imag, kernel=filters_real, padding='valid')

        out_real = out_rr - out_ii
        out_imag = out_ri + out_ir

        # 7) Concatenate real & imaginary along channel axis
        #    final shape: (batch, new_time, 2*fnum)
        out = tf.concat([out_real, out_imag], axis=2)
        return out

    def compute_output_shape(self, input_shape):
        """
        input_shape: (batch_size, time, channels)
                     here channels = 2 for real/imag
        The output time length shrinks by (self.fsize - 1) when padding='valid'.
        """
        new_size = input_shape[1] - (self.fsize - 1)
        # final channels = 2*self.fnum
        return (input_shape[0], new_size, 2 * self.fnum)

