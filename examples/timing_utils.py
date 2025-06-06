import timeit

# import signatory
import iisignature
# import esig

import numpy as np
import sigkernel
# import jax
# from sigkerax.sigkernel import SigKernel as SigKernelJax
import torch
import matplotlib.pyplot as plt

import pysiglib

def plot_times(
        x,
        ys,
        legend,
        title,
        xlabel,
        ylabel,
        scale,
        filename
):
    plt.figure(figsize=(4, 3))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for y in ys:
        plt.plot(x, y)
    plt.legend(legend)
    plt.yscale(scale)
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    plt.savefig("plots/" + filename + ".png", dpi=300)
    plt.savefig("plots/" + filename + ".pdf", dpi=300)
    plt.show()

def timeiisig(batch_size, length, dimension, degree, device, N):
    if device != "cpu":
        raise ValueError("iisignature only supports cpu")

    X = np.random.uniform(size=(batch_size, length, dimension)).astype("double")

    best_time = float('inf')
    for _ in range(N):
        start = timeit.default_timer()
        iisignature.sig(X, degree)
        end = timeit.default_timer()
        time_ = end - start
        best_time = min(best_time, time_)
    return best_time

def timepysiglib(batch_size, length, dimension, degree, horner, parallel, device, N):
    X = np.random.uniform(size=(batch_size, length, dimension)).astype("double")
    X = torch.tensor(X, device=device)

    best_time = float('inf')
    for _ in range(N):
        torch.cuda.empty_cache()
        start = timeit.default_timer()
        pysiglib.signature(X, degree, horner = horner, parallel = parallel)
        end = timeit.default_timer()
        time_ = end - start
        best_time = min(best_time, time_)
    return best_time


def timesignatory(batch_size, length, dimension, degree, device, N):
    X = np.random.uniform(size=(batch_size, length, dimension))#.astype("double")
    X = torch.tensor(X, device=device)

    best_time = float('inf')
    for _ in range(N):
        start = timeit.default_timer()
        signatory.signature(X, degree)
        end = timeit.default_timer()
        time_ = end - start
        best_time = min(best_time, time_)
    return best_time


# def timeesig(batchSize, length, dimension, degree, device, N):
#     if device != "cpu":
#         raise ValueError("esig only supports cpu")
#     X = np.random.uniform(size=(batchSize, length, dimension)).astype("double")
#
#     best_time = float('inf')
#     for _ in range(N):
#         start = timeit.default_timer()
#         esig.stream2sig(X, degree)
#         end = timeit.default_timer()
#         time_ = end - start
#         best_time = min(best_time, time_)
#     return best_time

def timesigkernel(batch_size, length, dimension, dyadic_order, device, N):
    static_kernel = sigkernel.LinearKernel()
    signature_kernel = sigkernel.SigKernel(static_kernel, dyadic_order)

    X = np.random.uniform(size=(batch_size, length, dimension)).astype("double")
    X = torch.tensor(X, device=device)
    Y = np.random.uniform(size=(batch_size, length, dimension)).astype("double")
    Y = torch.tensor(Y, device=device)

    best_time = float('inf')
    for _ in range(N):
        try:
            if device == "cuda":
                torch.cuda.empty_cache()
            start = timeit.default_timer()
            signature_kernel.compute_kernel(X, Y)
            end = timeit.default_timer()
            time_ = end - start
            best_time = min(best_time, time_)
        except:
            continue
    return best_time


# def timesigkerax(batch_size, length, dimension, dyadic_order, device, N):
#     signature_kernel = SigKernelJax(static_kernel_kind="linear", refinement_factor=1 << dyadic_order)
#
#     X = np.random.uniform(size=(batch_size, length, dimension)).astype("double")
#     X = jax.numpy.array(X)
#     Y = np.random.uniform(size=(batch_size, length, dimension)).astype("double")
#     Y = jax.numpy.array(Y)
#
#     best_time = float('inf')
#     for _ in range(N):
#         if device == "cuda":
#             torch.cuda.empty_cache()
#         start = timeit.default_timer()
#         signature_kernel.kernel_matrix(X, Y)
#         end = timeit.default_timer()
#         time_ = end - start
#         best_time = min(best_time, time_)
#     return best_time


def timepysiglib_kernel(batch_size, length, dimension, dyadic_order, device, N):
    X = np.random.uniform(size=(batch_size, length, dimension)).astype("double")
    X = torch.tensor(X, device=device)
    Y = np.random.uniform(size=(batch_size, length, dimension)).astype("double")
    Y = torch.tensor(Y, device=device)

    best_time = float('inf')
    for _ in range(N):
        if device == "cuda":
            torch.cuda.empty_cache()
        start = timeit.default_timer()
        pysiglib.sig_kernel(X, Y, dyadic_order)
        end = timeit.default_timer()
        time_ = end - start
        best_time = min(best_time, time_)
    return best_time
