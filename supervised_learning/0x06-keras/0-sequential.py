#!/usr/bin/env python3
"""Build model"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Builds a neural network with the Keras library"""
    model = K.Sequential()
    regularizer = K.regularizers.l2(lambtha)
    n_layers = len(layers)
    for i, (layer, activation) in enumerate(zip(layers, activations)):
        model.add(K.layers.Dense(
            units=layer,
            kernel_regularizer=regularizer,
            input_shape=(nx,),
            activation=activation,
        ))
        if n_layers != i + 1:
            model.add(K.layers.Dropout(1 - keep_prob))
    return model
