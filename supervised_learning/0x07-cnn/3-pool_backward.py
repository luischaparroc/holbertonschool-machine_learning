#!/usr/bin/env python3
"""Pool forward"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Performs back propagation over a pooling layer of a neural network"""
    kh, kw = kernel_shape
    m, h_dA, w_dA, c_dA = dA.shape
    sh, sw = stride

    dA_prev = np.zeros(A_prev.shape)

    for i in range(m):
        for i_oh in range(h_dA):
            for i_ow in range(w_dA):
                for i_nc in range(c_dA):
                    ysh = i_oh * sh
                    yshk = ysh + kh
                    xsw = i_ow * sw
                    xswk = xsw + kw

                    if mode == 'max':
                        a = A_prev[i]
                        slice = a[ysh:yshk, xsw:xswk, i_nc]
                        msk = (slice == np.max(slice))
                        mul = np.multiply(msk, dA[i, i_oh, i_ow, i_nc])
                        dA_prev[i, ysh: yshk, xsw: xswk, i_nc] += mul
                    elif mode == 'avg':
                        dA_var = dA[i, i_oh, i_ow, i_nc]
                        avg = dA_var / (kh * kw)
                        Z = np.ones(kernel_shape) * avg
                        dA_prev[i, ysh: yshk, xsw: xswk, i_nc] += Z

    return dA_prev
