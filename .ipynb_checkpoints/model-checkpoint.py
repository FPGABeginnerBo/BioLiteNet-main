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

# 从LiteModel.py导入的BeeSenseSelector
class BeeSenseSelector(Layer):
    def __init__(self, in_channels, k_ratio=0.5, **kwargs):
        super(BeeSenseSelector, self).__init__(**kwargs)
        self.k = int(in_channels * k_ratio)
        self.in_channels = in_channels
        self.global_pool = layers.GlobalAveragePooling2D()
        self.conv = Conv2D(in_channels, 1, activation='sigmoid')

    def build(self, input_shape):
        super(BeeSenseSelector, self).build(input_shape)

    def call(self, x):
        # 获取输入张量的形状
        input_shape = tf.shape(x)
        batch_size = input_shape[0]
        
        # 计算通道注意力分数
        x_pool = self.global_pool(x)  # [batch_size, in_channels]
        x_pool = tf.expand_dims(x_pool, axis=1)  # [batch_size, 1, in_channels]
        x_pool = tf.expand_dims(x_pool, axis=1)  # [batch_size, 1, 1, in_channels]
        scores = self.conv(x_pool)  # [batch_size, 1, 1, in_channels]
        scores = tf.squeeze(scores, axis=[1, 2])  # [batch_size, in_channels]
        
        # 确保k不超过通道数
        k = tf.minimum(self.k, self.in_channels)
        
        # 获取top-k通道
        top_k_values, top_k_indices = tf.nn.top_k(scores, k=k)
        
        # 创建mask
        mask = tf.zeros([batch_size, self.in_channels])
        batch_indices = tf.range(batch_size)
        batch_indices = tf.expand_dims(batch_indices, 1)
        batch_indices = tf.tile(batch_indices, [1, k])
        
        indices = tf.stack([batch_indices, top_k_indices], axis=2)
        updates = tf.ones_like(top_k_values)
        mask = tf.scatter_nd(indices, updates, [batch_size, self.in_channels])
        
        # 扩展mask维度以匹配输入
        mask = tf.expand_dims(tf.expand_dims(mask, axis=1), axis=1)
        return x * mask

    def compute_output_shape(self, input_shape):
        return input_shape

# 从LiteModel.py导入的AffScaleConv
class AffScaleConv(Layer):
    def __init__(self, in_channels, out_channels, base_kernel_size=3, scale_set=[0.5, 1.0, 2.0], **kwargs):
        super(AffScaleConv, self).__init__(**kwargs)
        self.scale_set = scale_set
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_kernel_size = base_kernel_size
        self.branches = []
        for s in scale_set:
            kernel_size = int(base_kernel_size * s)
            padding = int((base_kernel_size * s) // 2)
            self.branches.append(
                layers.DepthwiseConv2D(
                    kernel_size=kernel_size,
                    padding='same',
                    use_bias=False
                )
            )
        self.fuse = Conv2D(out_channels, 1)

    def build(self, input_shape):
        super(AffScaleConv, self).build(input_shape)

    def call(self, x):
        outputs = []
        for conv in self.branches:
            outputs.append(conv(x))
        return self.fuse(tf.concat(outputs, axis=-1))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], self.out_channels)

# 从LiteModel.py导入的SwarmFusionUnit
class SwarmFusionUnit(Layer):
    def __init__(self, in_channels, out_channels, k_ratio=0.5, scale_set=[0.5, 1.0, 2.0], **kwargs):
        super(SwarmFusionUnit, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.selector = BeeSenseSelector(in_channels, k_ratio)
        self.scale_conv = AffScaleConv(in_channels, out_channels, scale_set=scale_set)
        self.norm = BatchNormalization()
        self.act = Activation('relu')

    def build(self, input_shape):
        super(SwarmFusionUnit, self).build(input_shape)

    def call(self, x):
        x = self.selector(x)
        x = self.scale_conv(x)
        x = self.norm(x)
        return self.act(x)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], self.out_channels)

def attention_spatial(inputs2):
    a = Dense((inputs2.shape[3]).value, activation='softmax')(inputs2)
    return a

def attention_vertical(inputs):
    input_dim1 = int(inputs.shape[1])
    input_dim2 = int(inputs.shape[2])
    input_dim3 = int(inputs.shape[3])

    a = Permute((3, 1,2))(inputs)
    a = Reshape((input_dim3, input_dim2,input_dim1))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(input_dim2, activation='softmax')(a)
    
    a_probs = Permute((3,2,1))(a)
    return a_probs

def attention_horizontal(inputs2):
    input_dim1 = int(inputs2.shape[1])
    input_dim2 = int(inputs2.shape[2])
    input_dim3 = int(inputs2.shape[3])

    a = Permute((3, 2,1))(inputs2)
    a = Reshape((input_dim3, input_dim2,input_dim1 ))(a) # this line is not useful. It's just to know which dimension is what.

    a = Dense(input_dim2, activation='softmax')(a)

    b_probs = Permute((3,2,1))(a)
    return b_probs

def demo1(img_rows, img_cols, num_PC, nb_classes):
    CNNInput = Input(shape=(img_rows, img_cols, num_PC), name='i0')

    #模型架构区
    F = Dense(nb_classes, activation='softmax', name='classifier', kernel_initializer=Symmetry(n=num_PC, c=nb_classes))(
        CNNInput)
    model = Model(inputs=[CNNInput], outputs=F)
    return model

def demo(img_rows, img_cols, num_PC, nb_classes):
    CNNInput = Input(shape=(img_rows, img_cols, num_PC), name='i0')
    
    # 1. 光谱降维 - 使用1x1卷积降低高光谱数据维度
    x = Conv2D(32, (1, 1), padding='same', 
              kernel_regularizer=regularizers.l2(0.0005))(CNNInput)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # 2. 空间特征提取 - 双路径设计
    # 路径1: 空间细节特征
    x1 = Conv2D(16, (3, 3), padding='same',
               kernel_regularizer=regularizers.l2(0.0005))(x)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    
    # 路径2: 光谱特征
    x2 = Conv2D(16, (1, 1), padding='same',
               kernel_regularizer=regularizers.l2(0.0005))(x)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    
    # 特征融合
    x = concatenate([x1, x2])
    
    # 3. 残差连接
    shortcut = Conv2D(32, (1, 1), padding='same')(x)
    
    # 4. 特征增强 - 使用AffScaleConv
    x = AffScaleConv(in_channels=32, out_channels=32, 
                    base_kernel_size=3, scale_set=[0.75, 1.0])(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.25)(x)
    
    # 5. 添加残差连接
    x = add([x, shortcut])
    x = Activation('relu')(x)
    
    # 6. 添加BeeSenseSelector - 轻量级通道选择
    # 注意：这里使用较大的k_ratio以保留更多信息
    x = BeeSenseSelector(in_channels=32, k_ratio=0.9)(x)
    
    # 7. 特征聚合 - 使用全局池化
    x_avg = GlobalAveragePooling2D()(x)
    x_max = GlobalMaxPooling2D()(x)
    x = concatenate([x_avg, x_max])
    
    # 8. 分类器 - 使用两层全连接网络
    x = Dense(64, kernel_initializer='he_normal',
             kernel_regularizer=regularizers.l2(0.0005))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)
    
    # 9. 输出层 - 移除类别平衡策略
    x = Dense(nb_classes)(x)
    
    # 直接应用Softmax输出，不进行类别平衡调整
    output = Activation('softmax', name='classifier')(x)
    
    # 创建模型
    model = Model(inputs=CNNInput, outputs=output)
    print(model.summary())
    cal_flops(model)
    return model

def demo_no_selector(img_rows, img_cols, num_PC, nb_classes):
    """
    消融实验：移除BeeSenseSelector组件的模型变体
    """
    CNNInput = Input(shape=(img_rows, img_cols, num_PC), name='i0')
    
    # 1. 光谱降维 - 使用1x1卷积降低高光谱数据维度
    x = Conv2D(32, (1, 1), padding='same', 
              kernel_regularizer=regularizers.l2(0.0005))(CNNInput)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # 2. 空间特征提取 - 双路径设计
    # 路径1: 空间细节特征
    x1 = Conv2D(16, (3, 3), padding='same',
               kernel_regularizer=regularizers.l2(0.0005))(x)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    
    # 路径2: 光谱特征
    x2 = Conv2D(16, (1, 1), padding='same',
               kernel_regularizer=regularizers.l2(0.0005))(x)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    
    # 特征融合
    x = concatenate([x1, x2])
    
    # 3. 残差连接
    shortcut = Conv2D(32, (1, 1), padding='same')(x)
    
    # 4. 特征增强 - 使用AffScaleConv
    x = AffScaleConv(in_channels=32, out_channels=32, 
                    base_kernel_size=3, scale_set=[0.75, 1.0])(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.25)(x)
    
    # 5. 添加残差连接
    x = add([x, shortcut])
    x = Activation('relu')(x)
    
    # 6. BeeSenseSelector已移除
    # 直接进行特征聚合，不进行通道选择
    
    # 7. 特征聚合 - 使用全局池化
    x_avg = GlobalAveragePooling2D()(x)
    x_max = GlobalMaxPooling2D()(x)
    x = concatenate([x_avg, x_max])
    
    # 8. 分类器 - 使用两层全连接网络
    x = Dense(64, kernel_initializer='he_normal',
             kernel_regularizer=regularizers.l2(0.0005))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)
    
    # 9. 输出层
    x = Dense(nb_classes)(x)
    output = Activation('softmax', name='classifier')(x)
    
    # 创建模型
    model = Model(inputs=CNNInput, outputs=output)
    print(model.summary())
    cal_flops(model)
    return model

def demo_no_affscale(img_rows, img_cols, num_PC, nb_classes):
    """
    消融实验：将AffScaleConv替换为标准卷积的模型变体
    """
    CNNInput = Input(shape=(img_rows, img_cols, num_PC), name='i0')
    
    # 1. 光谱降维 - 使用1x1卷积降低高光谱数据维度
    x = Conv2D(32, (1, 1), padding='same', 
              kernel_regularizer=regularizers.l2(0.0005))(CNNInput)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # 2. 空间特征提取 - 双路径设计
    # 路径1: 空间细节特征
    x1 = Conv2D(16, (3, 3), padding='same',
               kernel_regularizer=regularizers.l2(0.0005))(x)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    
    # 路径2: 光谱特征
    x2 = Conv2D(16, (1, 1), padding='same',
               kernel_regularizer=regularizers.l2(0.0005))(x)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    
    # 特征融合
    x = concatenate([x1, x2])
    
    # 3. 残差连接
    shortcut = Conv2D(32, (1, 1), padding='same')(x)
    
    # 4. 特征增强 - 使用标准卷积替代AffScaleConv
    x = Conv2D(32, (3, 3), padding='same',
              kernel_regularizer=regularizers.l2(0.0005))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.25)(x)
    
    # 5. 添加残差连接
    x = add([x, shortcut])
    x = Activation('relu')(x)
    
    # 6. 添加BeeSenseSelector - 轻量级通道选择
    # 注意：这里使用较大的k_ratio以保留更多信息
    x = BeeSenseSelector(in_channels=32, k_ratio=0.9)(x)
    
    # 7. 特征聚合 - 使用全局池化
    x_avg = GlobalAveragePooling2D()(x)
    x_max = GlobalMaxPooling2D()(x)
    x = concatenate([x_avg, x_max])
    
    # 8. 分类器 - 使用两层全连接网络
    x = Dense(64, kernel_initializer='he_normal',
             kernel_regularizer=regularizers.l2(0.0005))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)
    
    # 9. 输出层
    x = Dense(nb_classes)(x)
    output = Activation('softmax', name='classifier')(x)
    
    # 创建模型
    model = Model(inputs=CNNInput, outputs=output)
    print(model.summary())
    cal_flops(model)
    return model

def demo_no_selector_no_affscale(img_rows, img_cols, num_PC, nb_classes):
    """
    消融实验：同时移除BeeSenseSelector并将AffScaleConv替换为标准卷积的模型变体
    """
    CNNInput = Input(shape=(img_rows, img_cols, num_PC), name='i0')
    
    # 1. 光谱降维 - 使用1x1卷积降低高光谱数据维度
    x = Conv2D(32, (1, 1), padding='same', 
              kernel_regularizer=regularizers.l2(0.0005))(CNNInput)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # 2. 空间特征提取 - 双路径设计
    # 路径1: 空间细节特征
    x1 = Conv2D(16, (3, 3), padding='same',
               kernel_regularizer=regularizers.l2(0.0005))(x)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    
    # 路径2: 光谱特征
    x2 = Conv2D(16, (1, 1), padding='same',
               kernel_regularizer=regularizers.l2(0.0005))(x)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    
    # 特征融合
    x = concatenate([x1, x2])
    
    # 3. 残差连接
    shortcut = Conv2D(32, (1, 1), padding='same')(x)
    
    # 4. 特征增强 - 使用标准卷积替代AffScaleConv
    x = Conv2D(32, (3, 3), padding='same',
              kernel_regularizer=regularizers.l2(0.0005))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.25)(x)
    
    # 5. 添加残差连接
    x = add([x, shortcut])
    x = Activation('relu')(x)
    
    # 6. BeeSenseSelector已移除
    # 直接进行特征聚合，不进行通道选择
    
    # 7. 特征聚合 - 使用全局池化
    x_avg = GlobalAveragePooling2D()(x)
    x_max = GlobalMaxPooling2D()(x)
    x = concatenate([x_avg, x_max])
    
    # 8. 分类器 - 使用两层全连接网络
    x = Dense(64, kernel_initializer='he_normal',
             kernel_regularizer=regularizers.l2(0.0005))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)
    
    # 9. 输出层
    x = Dense(nb_classes)(x)
    output = Activation('softmax', name='classifier')(x)
    
    # 创建模型
    model = Model(inputs=CNNInput, outputs=output)
    print(model.summary())
    cal_flops(model)
    return model

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
