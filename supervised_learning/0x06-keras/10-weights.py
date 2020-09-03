#!/usr/bin/env python3
"""Save/Load weights"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """Saves a model's weights"""
    network.save_weights(filename, save_format=save_format)


def load_weights(network, filename):
    """Loads a model's weights"""
    network.load_weights(filename)
