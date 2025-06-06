import timeit

import numpy as np
import torch
import sigkernel

import pysiglib

if __name__ == '__main__':

    batch_size = 10
    length = 100
    dim = 5
    dyadic_order = 0

    X = np.random.uniform(size = (batch_size, length, dim)).astype("double")
    Y = np.random.uniform(size=(batch_size, length, dim)).astype("double")

    X = torch.tensor(X, device = "cuda")
    Y = torch.tensor(Y, device = "cuda")

    static_kernel = sigkernel.LinearKernel()
    signature_kernel = sigkernel.SigKernel(static_kernel, dyadic_order)

    start = timeit.default_timer()
    kernel = signature_kernel.compute_kernel(X, Y)
    end = timeit.default_timer()

    print(end - start)
    print(kernel)

    start = timeit.default_timer()
    kernel = pysiglib.sig_kernel(X, Y, dyadic_order)
    end = timeit.default_timer()
    print(end - start)
    print(kernel)
