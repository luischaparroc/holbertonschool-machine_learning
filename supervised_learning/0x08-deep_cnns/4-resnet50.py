#!/usr/bin/env python3
"""ResNet 50"""
import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """Builds a projection block"""

    X = K.Input(shape=(224, 224, 3))
    init = K.initializers.he_normal(seed=None)

    conv1 = K.layers.Conv2D(
        filters=64,
        kernel_size=(7, 7),
        padding='same',
        strides=2,
        kernel_initializer=init
    )(X)

    bn1 = K.layers.BatchNormalization(
        axis=3
    )(conv1)

    activation1 = K.layers.Activation('relu')(bn1)

    maxpool1 = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding='same',
    )(activation1)

    proj1 = projection_block(maxpool1, [64, 64, 256], s=1)
    id_block1 = identity_block(proj1, [64, 64, 256])
    id_block2 = identity_block(id_block1, [64, 64, 256])

    proj2 = projection_block(id_block2, [128, 128, 512])
    id_block3 = identity_block(proj2, [128, 128, 512])
    id_block4 = identity_block(id_block3, [128, 128, 512])
    id_block5 = identity_block(id_block4, [128, 128, 512])

    proj3 = projection_block(id_block5, [256, 256, 1024])
    id_block6 = identity_block(proj3, [256, 256, 1024])
    id_block7 = identity_block(id_block6, [256, 256, 1024])
    id_block8 = identity_block(id_block7, [256, 256, 1024])
    id_block9 = identity_block(id_block8, [256, 256, 1024])
    id_block10 = identity_block(id_block9, [256, 256, 1024])

    proj4 = projection_block(id_block10, [512, 512, 2048])
    id_block11 = identity_block(proj4, [512, 512, 2048])
    id_block12 = identity_block(id_block11, [512, 512, 2048])

    avgpool = K.layers.AveragePooling2D(
        pool_size=(1, 1),
        strides=(7, 7),
        padding='same',
    )(id_block12)

    softmax = K.layers.Dense(
        units=1000,
        kernel_initializer=init,
        activation='softmax',
    )(avgpool)

    model = K.Model(inputs=X, outputs=softmax)

    return model
