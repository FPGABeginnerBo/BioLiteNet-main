# -*- coding: utf-8 -*-

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import math

from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Reshape, Lambda, Dropout
from keras import layers
from keras.optimizers import RMSprop
from keras.layers import BatchNormalization
from keras.callbacks import TensorBoard
from keras.initializers import Initializer
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
from data import image_size_dict
from secondpooling import SecondOrderPooling

from keras.utils import np_utils
import tensorflow as tf
import keras
#from tensorflow.keras.layers import InputSpec
from keras import activations
from keras.layers import Permute
from pandas.core.frame import DataFrame
from keras.engine.base_layer import Layer
from keras.layers import MaxPooling1D
from keras.optimizers import Adam
import pandas as pd
import numpy as np
from keras import backend as K
import matplotlib.pyplot as plt
from keras import regularizers, Model, Input
import seaborn as sns
import time
from functions import *

from keras.layers import Conv2D, MaxPooling2D, LSTM, Flatten, Conv1D, MaxPooling1D, Dense, Activation, \
    Dropout, GlobalMaxPooling1D, AveragePooling2D, ConvLSTM2D, GlobalMaxPooling2D, GlobalAveragePooling2D, Recurrent, Reshape, Bidirectional, \
    BatchNormalization, concatenate, activations, merge, add, Multiply, multiply, UpSampling2D

def cal_flops(model):
    run_meta = tf.RunMetadata()
    opts = tf.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.profiler.profile(graph=K.get_session().graph, run_meta=run_meta, cmd='op', options=opts)
    params = tf.profiler.profile(graph=K.get_session().graph, run_meta=run_meta, cmd='op', options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('flops'+ str(flops.total_float_ops))
    print('params'+ str(params.total_parameters))

def demo1(img_rows, img_cols, num_PC, nb_classes):
    CNNInput = Input(shape=(img_rows, img_cols, num_PC), name='i0')

    #模型架构区
    F = Dense(nb_classes, activation='softmax', name='classifier', kernel_initializer=Symmetry(n=num_PC, c=nb_classes))(
        CNNInput)
    model = Model(inputs=[CNNInput], outputs=F)
    return model

def demo(img_rows, img_cols, num_PC, nb_classes):
    return None

def get_model(img_rows, img_cols, num_PC, nb_classes, dataID=1, type='aspn', lr=0.01):
    if num_PC == 0:
        num_PC = image_size_dict[str(dataID)][2]
    if type == 'demo':
        model = demo(img_rows, img_cols, num_PC, nb_classes)
    elif type == 'demo_no_selector':
        model = demo_no_selector(img_rows, img_cols, num_PC, nb_classes)
    elif type == 'demo_no_affscale':
        model = demo_no_affscale(img_rows, img_cols, num_PC, nb_classes)
    elif type == 'demo_no_selector_no_affscale':
        model = demo_no_selector_no_affscale(img_rows, img_cols, num_PC, nb_classes)
    else:
        print('invalid model type, default use demo1 model')
        model = demo1(img_rows, img_cols, num_PC, nb_classes)

    rmsp = RMSprop(lr=lr, rho=0.9, epsilon=1e-05)
    model.compile(optimizer=rmsp, loss='categorical_crossentropy',
                          metrics=['accuracy'])
    return model

class Symmetry(Initializer):
    """N*N*C Symmetry initial
    """
    def __init__(self, n=200, c=16, seed=0):
        self.n = n
        self.c = c
        self.seed = seed

    def __call__(self, shape, dtype=None):
        rv = K.truncated_normal([self.n, self.n, self.c], 0., 1e-5, dtype=dtype, seed=self.seed)
        rv = (rv + K.permute_dimensions(rv, pattern=(1, 0, 2))) / 2.0
        return K.reshape(rv, [self.n * self.n, self.c])

def get_callbacks(decay=0.0001):
    def step_decay(epoch, lr):
        return lr * math.exp(-1 * epoch * decay)

    callbacks = []
    callbacks.append(LearningRateScheduler(step_decay))

    return callbacks
