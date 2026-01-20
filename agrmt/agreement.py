"""
Python port of the R argmt package's agreement function.

Calculates agreement A for frequency vectors based on the algorithm
described in van der Eijk (2001).

References
----------
van der Eijk, C. (2001). Measuring Agreement in Ordered Rating Scales.
Quality and Quantity, 35(3), 325-341.
"""

import numpy as np


def _pattern_agreement(p: np.ndarray) -> float:
    """
    Calculate agreement A from a pattern vector.

    Parameters
    ----------
    p : np.ndarray
        Pattern vector (binary: only 0 and 1 are allowed).

    Returns
    -------
    float
        Agreement value for the pattern.

    Raises
    ------
    ValueError
        If input contains values other than 0 and 1.
    """
    if np.max(p) > 1:
        raise ValueError("Input is not a pattern vector (only 0 and 1 are allowed).")

    k = len(p)  # number of categories

    # Count triplets
    tdu = 0  # bimodal triplets (1-0-1 pattern)
    tu = 0   # unimodal triplets (1-1-0 or 0-1-1 pattern)

    for i in range(k - 2):
        for j in range(i + 1, k - 1):
            for m in range(j + 1, k):
                if p[i] == 1 and p[j] == 0 and p[m] == 1:
                    tdu += 1  # 101 pattern, bimodal
                if p[i] == 1 and p[j] == 1 and p[m] == 0:
                    tu += 1   # 110 pattern, unimodal
                if p[i] == 0 and p[j] == 1 and p[m] == 1:
                    tu += 1   # 011 pattern, unimodal

    # Calculate U as in equation (2)
    if tu + tdu == 0:
        u = 0.0
    else:
        u = ((k - 2) * tu - (k - 1) * tdu) / ((k - 2) * (tu + tdu))

    s = np.sum(p)  # number of non-empty categories
    a = u * (1 - (s - 1) / (k - 1))  # agreement A

    # Handle edge cases
    if np.isnan(a):
        a = 0.0  # lack of agreement, defined as 0
    if np.sum(p) == 1:
        a = 1.0  # only one value, defined as 1

    return a


def agreement(v: np.ndarray) -> float:
    """
    Calculate agreement A for a frequency vector.

    Implements the van der Eijk agreement measure for ordinal data using
    a layered algorithm that decomposes the frequency distribution.

    Parameters
    ----------
    v : array_like of int
        Frequency vector (integer counts per category). Must have at least
        3 elements and contain no negative values.

    Returns
    -------
    float
        Overall agreement value in range [-1, 1], where:
        - 1.0 indicates perfect agreement (all responses in one category)
        - 0.0 indicates uniform distribution or lack of pattern
        - -1.0 indicates perfect bimodal disagreement

    Raises
    ------
    ValueError
        If vector length < 3 or contains negative values.

    Examples
    --------
    >>> import numpy as np
    >>> agreement(np.array([30, 40, 210, 130, 530, 50, 10]))
    0.6113333333333334

    >>> agreement([0, 0, 100, 0, 0])  # Perfect agreement
    1.0

    >>> agreement([50, 0, 0, 0, 50])  # Bimodal disagreement
    -1.0
    """
    v = np.asarray(v)

    if len(v) < 3:
        raise ValueError("Length of vector < 3, agreement A is not defined.")

    if not np.issubdtype(v.dtype, np.integer):
        raise ValueError("Frequency vector must contain integers.")

    if np.min(v) < 0:
        raise ValueError("Negative values found in frequency vector.")

    v = v.astype(float)  # convert for arithmetic

    aa = 0.0       # overall agreement A
    k = len(v)     # number of categories
    n = np.sum(v)  # number of cases
    r = v.copy()   # remainder

    for _ in range(k):  # repeat for each layer
        p = (r != 0).astype(int)  # pattern vector for this layer

        if np.max(p) == 0:
            break  # remainder is empty, all layers are analyzed

        a = _pattern_agreement(p)  # agreement A for this layer

        # Get non-zero minimum of remainder
        nonzero_mask = r > 0
        m = np.min(r[nonzero_mask]) if np.any(nonzero_mask) else 0

        layer = p * m          # layer with the values
        w = np.sum(layer) / n  # weight of this layer
        aa += w * a            # add agreement of this layer to overall
        r = r - layer          # new remainder

    return aa
