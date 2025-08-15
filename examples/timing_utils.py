# Copyright 2025 Daniil Shmelev
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

import os.path
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
    if not os.path.exists("plots"):
        os.makedirs("plots")

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

def timepysiglib(batch_size, length, dimension, degree, horner, n_jobs, device, N):
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

def timepysiglib_sig_backprop(batch_size, length, dimension, degree, n_jobs, device, N):
    X = np.random.uniform(size=(batch_size, length, dimension)).astype("double")
    X = torch.tensor(X, device=device)
    sig = np.random.uniform(size=(batch_size,pysiglib.sig_length(dimension, degree)))
    sig_derivs = np.random.uniform(size=(batch_size,pysiglib.sig_length(dimension, degree)))

    best_time = float('inf')
    for _ in range(N):
        torch.cuda.empty_cache()
        start = timeit.default_timer()
        _sig = iisignature.sig(X, degree) #just for timing comparison with iisig which recomputes the signature
        pysiglib.sig_backprop(X, sig, sig_derivs, degree, False, False, n_jobs)
        end = timeit.default_timer()
        time_ = end - start
        best_time = min(best_time, time_)
    return best_time

def timeiisig_sig_backprop(batch_size, length, dimension, degree, n_jobs, device, N):
    X = np.random.uniform(size=(batch_size, length, dimension)).astype("double")
    X = torch.tensor(X, device=device)
    sig_derivs = np.random.uniform(size=(batch_size,pysiglib.sig_length(dimension, degree) - 1))

    best_time = float('inf')
    for _ in range(N):
        torch.cuda.empty_cache()
        start = timeit.default_timer()
        iisignature.sigbackprop(sig_derivs, X, degree)
        end = timeit.default_timer()
        time_ = end - start
        best_time = min(best_time, time_)
    return best_time

def timesignatory_sig_backprop(batch_size, length, dimension, degree, n_jobs, device, N):
    X = np.random.uniform(size=(batch_size, length, dimension)).astype("double")
    X = torch.tensor(X, device=device, requires_grad=True)
    sig_derivs = np.random.uniform(size=(batch_size,pysiglib.sig_length(dimension, degree) - 1))
    sig_derivs = torch.tensor(sig_derivs)

    best_time = float('inf')
    for _ in range(N):
        torch.cuda.empty_cache()
        sig = signatory.signature(X, degree)
        start = timeit.default_timer()
        sig.backward(sig_derivs)
        end = timeit.default_timer()
        time_ = end - start
        best_time = min(best_time, time_)
    return best_time
