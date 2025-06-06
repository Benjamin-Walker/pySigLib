import timeit

import numpy as np
import iisignature

import pysiglib

if __name__ == '__main__':

    batch_size = 100
    length = 1000
    dim = 5
    degree = 5

    X = np.random.uniform(size=(batch_size, length, dim)).astype("double")

    start = timeit.default_timer()
    sig = iisignature.sig(X, degree)
    end = timeit.default_timer()

    print(end - start)
    print(sig[0][:5])

    start = timeit.default_timer()
    sig = pysiglib.signature(X, degree)
    end = timeit.default_timer()
    print(end - start)
    print(sig[0][1:6])
