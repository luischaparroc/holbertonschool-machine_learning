#!/usr/bin/env python3
"""Pool forward"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """Performs forward propagation over a pooling layer of a neural network"""
    m, h, w, c = A_prev.shape
    kh, kw, c, nc = W.shape
    sh, sw = stride

    if padding == 'valid':
        ph = 0
        pw = 0
    elif padding == 'same':
        ph = int(np.ceil(((h - 1) * sh + kh - h) / 2))
        pw = int(np.ceil(((w - 1) * sw + kw - w) / 2))

    oh = int(((h + 2 * ph - kh) / sh) + 1)
    ow = int(((w + 2 * pw - kw) / sw) + 1)

    input_pd = np.pad(
        A_prev,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        'constant'
    )

    dA = np.zeros(input_pd.shape)
    dW = np.zeros(W.shape)
    db = np.sum(
        dZ,
        axis=(0, 1, 2),
        keepdims=True
    )

    for i in range(m):
        for i_oh in range(oh):
            for i_ow in range(ow):
                for i_nc in range(nc):
                    ysh = i_oh * sh
                    yshk = ysh + kh
                    xsw = i_ow * sw
                    xswk = xsw + kw
                    dZ_cut = dZ[i, i_oh, i_ow, i_nc]
                    mat_dZ_W = dZ_cut * W[:, :, :, i_nc]
                    dA[i, ysh: yshk, xsw:xswk] += mat_dZ_W
                    cut = input_pd[i, ysh: yshk, xsw:xswk, :] * dZ_cut
                    dW[:, :, :, i_nc] += cut

    if padding == 'same':
        dA = dA[:, ph:-ph, pw:-pw, :]

    return dA, dW, db
