"""
NaN-tolerant reductions.

Several arrays here contain columns that are all NaN by design: the untrained
CNN has no unique contribution because it is not in the full model, and images
excluded from an experiment have no data. numpy warns on every such reduction,
which would bury real warnings in noise. These wrappers suppress that one
warning and nothing else.
"""

import warnings

import numpy as np


def _quiet(fn, a, axis=0, **kw):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"(Mean|Degrees) of empty slice|"
                                          r"Degrees of freedom <= 0",
                                RuntimeWarning)
        warnings.filterwarnings("ignore", "All-NaN slice encountered",
                                RuntimeWarning)
        return fn(a, axis=axis, **kw)


def mean(a, axis=0):
    return _quiet(np.nanmean, a, axis)


def std(a, axis=0, ddof=1):
    return _quiet(np.nanstd, a, axis, ddof=ddof)
