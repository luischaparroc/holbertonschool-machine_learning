#!/usr/bin/env python3
"""DenseNet121"""
import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """Builds the DenseNet121"""

    X = K.Input(shape=(224, 224, 3))
    init = K.initializers.he_normal(seed=None)

    bn1 = K.layers.BatchNormalization()(X)
    activation1 = K.layers.Activation('relu')(bn1)

    conv1 = K.layers.Conv2D(
        filters=64,
        kernel_size=(7, 7),
        padding='same',
        strides=2,
        kernel_initializer=init
    )(activation1)

    maxpool1 = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding='same',
    )(conv1)

    dense_block_1, f1 = dense_block(maxpool1, 2 * growth_rate, growth_rate, 6)
    trans_layer_1, f2 = transition_layer(dense_block_1, f1, compression)
    dense_block_2, f3 = dense_block(trans_layer_1, f2, growth_rate, 12)
    trans_layer_2, f4 = transition_layer(dense_block_2, f3, compression)
    dense_block_3, f5 = dense_block(trans_layer_2, f4, growth_rate, 24)
    trans_layer_3, f6 = transition_layer(dense_block_3, f5, compression)
    dense_block_4, f7 = dense_block(trans_layer_3, f6, growth_rate, 16)

    avgpool = K.layers.AveragePooling2D(
        pool_size=(7, 7),
        padding='same',
    )(dense_block_4)

    softmax = K.layers.Dense(
        units=1000,
        kernel_initializer=init,
        activation='softmax',
    )(avgpool)

    model = K.Model(inputs=X, outputs=softmax)

    return model
