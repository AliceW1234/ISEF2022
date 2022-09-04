# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 05:12:45 2022

@author: m255127
"""


# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 05:12:45 2022

@author: m255127
"""


from re import T
import tensorflow as tf
import numpy as np
import tensorflow.keras
from tensorflow.keras import utils as np_utils
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from imageio import imread, imwrite
from tensorflow.keras.models import load_model
from tensorflow.keras import utils as np_utils
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.layers import Input, BatchNormalization, Activation, MaxPooling2D, Dropout, Add
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate,Flatten,Dense,Reshape
from tensorflow.keras.callbacks import CSVLogger,LearningRateScheduler,EarlyStopping
from tensorflow.keras import backend as K
from tensorflow.keras.initializers import he_normal
import os
import tensorflow as tf
import sys
from tensorflow.keras.regularizers import L2
from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import get_custom_objects
from tensorflow.nn import depth_to_space
class SubpixelConv2D(Layer):
    """ Subpixel Conv2D Layer
    upsampling a layer from (h, w, c) to (h*r, w*r, c/(r*r)),
    where r is the scaling factor, default to 4
    # Arguments
    upsampling_factor: the scaling factor
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        the second and the third dimension increased by a factor of
        `upsampling_factor`; the last layer decreased by a factor of
        `upsampling_factor^2`.
    # References
        Real-Time Single Image and Video Super-Resolution Using an Efficient
        Sub-Pixel Convolutional Neural Network Shi et Al. https://arxiv.org/abs/1609.05158
    """

    def __init__(self, upsampling_factor=4, **kwargs):
        super(SubpixelConv2D, self).__init__(**kwargs)
        self.upsampling_factor = upsampling_factor

    def build(self, input_shape):
        last_dim = input_shape[-1]
        factor = self.upsampling_factor * self.upsampling_factor
        if last_dim % (factor) != 0:
            raise ValueError('Channel ' + str(last_dim) + ' should be of '
                             'integer times of upsampling_factor^2: ' +
                             str(factor) + '.')

    def call(self, inputs, **kwargs):
        return depth_to_space( inputs, self.upsampling_factor )

    def get_config(self):
        config = { 'upsampling_factor': self.upsampling_factor, }
        base_config = super(SubpixelConv2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        factor = self.upsampling_factor * self.upsampling_factor
        input_shape_1 = None
        if input_shape[1] is not None:
            input_shape_1 = input_shape[1] * self.upsampling_factor
        input_shape_2 = None
        if input_shape[2] is not None:
            input_shape_2 = input_shape[2] * self.upsampling_factor
        dims = [ input_shape[0],
                 input_shape_1,
                 input_shape_2,
                 int(input_shape[3]/factor)
               ]
        return tuple( dims )

get_custom_objects().update({'SubpixelConv2D': SubpixelConv2D})


def ResidualDenseBlock(input,num_G):
    x1 = Conv2D(filters=num_G, kernel_initializer=he_normal(), kernel_size=(3, 3), activation='relu',
                         padding='same')(input)
    y1 = concatenate([input, x1])
    x2 = Conv2D(filters=num_G, kernel_initializer=he_normal(), kernel_size=(3, 3), activation='relu',
                         padding='same')(y1)
    y2 = concatenate([input, x1,x2])
    x3 = Conv2D(filters=num_G, kernel_initializer=he_normal(), kernel_size=(3, 3), activation='relu',
                         padding='same')(y2)
    y3 = concatenate([input, x1,x2,x3])
    x4 = Conv2D(filters=num_G, kernel_initializer=he_normal(), kernel_size=(3, 3), activation='relu',
                         padding='same')(y3)
    y4 = concatenate([input, x1,x2,x3,x4])
    x5 = Conv2D(filters=num_G, kernel_initializer=he_normal(), kernel_size=(3, 3), activation='relu',
                         padding='same')(y4)
    y5 = concatenate([input, x1,x2,x3,x4,x5])
    x6 = Conv2D(filters=num_G, kernel_initializer=he_normal(), kernel_size=(3, 3), activation='relu',
                         padding='same')(y5)
    #x6 = sharpen(x6)
    y6 = concatenate([input, x1,x2,x3,x4,x5,x6])
    y = Conv2D(filters=64, kernel_initializer=he_normal(), kernel_size=(1, 1), padding='same')(y6)
    y = Add()([y,input])
    return y

def RDN(input,num_G,channels,scale=1,sr=False):
    sfe1 = Conv2D(filters=64, kernel_initializer=he_normal(), kernel_size=(3, 3), padding='same')(input)
    sfe2 = Conv2D(filters=64, kernel_initializer=he_normal(), kernel_size=(3, 3), padding='same')(sfe1)
    x = ResidualDenseBlock(sfe2,num_G) #1
    x = ResidualDenseBlock(x,num_G) #2
    x = ResidualDenseBlock(x,num_G) #3
    x = ResidualDenseBlock(x,num_G) #4
    x = ResidualDenseBlock(x,num_G) #5
    x = ResidualDenseBlock(x,num_G) #6
    x = ResidualDenseBlock(x,num_G) #7
    x = ResidualDenseBlock(x,num_G) #8
    x = ResidualDenseBlock(x,num_G) #9
    x = ResidualDenseBlock(x,num_G) #10
    # x = ResidualDenseBlock(x,num_G) #11
    # x = ResidualDenseBlock(x,num_G) #12
    # x = ResidualDenseBlock(x,num_G) #13
    # x = ResidualDenseBlock(x,num_G) #14
    # x = ResidualDenseBlock(x,num_G) #15
    # x = ResidualDenseBlock(x,num_G) #16
    # x = ResidualDenseBlock(x,num_G) #17
    # x = ResidualDenseBlock(x,num_G) #18
    # x = ResidualDenseBlock(x,num_G) #19
    # x = ResidualDenseBlock(x,num_G) #20
    if scale>1 and sr==True:
        x = SubpixelConv2D(upsampling_factor=scale)(x)
    y = Conv2D(filters=channels, kernel_initializer=he_normal(), kernel_size=(3, 3), padding='same')(x)
    model = Model(inputs=[input], outputs=[y])
    return model
def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def get_unet(input_img, n_filters=32, dropout=0.5, batchnorm=True):
    # contracting path
    c1 = conv2d_block(input_img, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    c1_sharpen = sharpen(c1)
    p1 = MaxPooling2D((2, 2)) (c1)
    p1 = Dropout(dropout*0.5)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)
    c2_sharpen = sharpen(c2)
    p2 = MaxPooling2D((2, 2)) (c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)
    c3_sharpen = sharpen(c3)
    p3 = MaxPooling2D((2, 2)) (c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)
    c4_sharpen = sharpen(c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = conv2d_block(p4, n_filters=n_filters*16, kernel_size=3, batchnorm=batchnorm)
    
    # expansive path
    u6 = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4_sharpen])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3_sharpen])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2_sharpen])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1_sharpen], axis=3)
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    
    outputs = Conv2D(1, (1, 1), activation='relu') (c9)
    outputs = Add()([outputs,input_img])
    model = Model(inputs=[input_img], outputs=[outputs])
    return model
