#!/usr/bin/env python3
""" Calculates the shape of a matrix """
def get_length(rows):
    if type(rows) is list or type(rows) is tuple:
        return [len(rows), *get_length(rows[0])]
    return []

def matrix_shape(matrix):
    return [*get_length(matrix)]
