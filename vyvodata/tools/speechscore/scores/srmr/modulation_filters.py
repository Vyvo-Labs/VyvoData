# Copyright 2014 João Felipe Santos, jfsantos@emt.inrs.ca
#
# This file is part of the SRMRpy library, and is licensed under the
# MIT license: https://github.com/jfsantos/SRMRpy/blob/master/LICENSE

import numpy as np
import scipy.signal as sig


def make_modulation_filter(w0, q):
    w0_tan = np.tan(w0 / 2)
    b0 = w0_tan / q
    b = np.array([b0, 0, -b0], dtype=float)
    a = np.array(
        [(1 + b0 + w0_tan**2), (2 * w0_tan**2 - 2), (1 - b0 + w0_tan**2)], dtype=float
    )
    return b, a


def modulation_filterbank(mf, fs, q):
    return [make_modulation_filter(w0, q) for w0 in 2 * np.pi * mf / fs]


def compute_modulation_cfs(min_cf, max_cf, n):
    spacing_factor = (max_cf / min_cf) ** (1.0 / (n - 1))
    cfs = np.zeros(n)
    cfs[0] = min_cf
    for k in range(1, n):
        cfs[k] = cfs[k - 1] * spacing_factor
    return cfs


def modfilt(filter_bank, x):
    y = np.zeros((len(filter_bank), len(x)), dtype=float)
    for k, f in enumerate(filter_bank):
        y[k] = sig.lfilter(f[0], f[1], x)
    return y
