import pysiglib
import numpy as np
import timeit
import sigkernel
from tqdm import tqdm
import torch
import pickle

import matplotlib.pyplot as plt
import plotting_params
plotting_params.set_plotting_params(8, 10, 12)

if __name__ == '__main__':
    N = 3

    dyadicOrder = 0
    batchSize = 100
    length = 200

    def timesigkernel(dimension, device):
        static_kernel = sigkernel.LinearKernel()
        signature_kernel = sigkernel.SigKernel(static_kernel, dyadicOrder)

        X = np.random.uniform(size=(batchSize, length, dimension)).astype("double")
        X = torch.tensor(X, device = device)
        Y = np.random.uniform(size=(batchSize, length, dimension)).astype("double")
        Y = torch.tensor(Y, device = device)

        best_time = float('inf')
        for i in range(N):
            start = timeit.default_timer()
            signature_kernel.compute_kernel(X, Y)
            end = timeit.default_timer()
            time_ = end - start
            if time_ < best_time:
                best_time = time_
        return best_time

    def timepysiglib(dimension, device):
        X = np.random.uniform(size=(batchSize, length, dimension)).astype("double")
        X = torch.tensor(X, device=device)
        Y = np.random.uniform(size=(batchSize, length, dimension)).astype("double")
        Y = torch.tensor(Y, device=device)

        best_time = float('inf')
        for i in range(N):
            start = timeit.default_timer()
            pysiglib.sigKernel(X, Y, dyadicOrder)
            end = timeit.default_timer()
            time_ = end - start
            if time_ < best_time:
                best_time = time_
        return best_time

    dimArr = [i for i in range(10, 11000, 1000)]
    sigkerneltime = []
    pysiglibtime = []

    device = "cpu"

    for dimension in tqdm(dimArr):
        sigkerneltime.append(timesigkernel(dimension, device))
        pysiglibtime.append(timepysiglib(dimension, device))

    print(sigkerneltime)
    print(pysiglibtime)

    plt.figure(figsize=(4, 3))
    plt.title("Times " + device)
    plt.xlabel("Path Dimension")
    plt.ylabel("Elapsed Time (s)")
    plt.plot(dimArr, sigkerneltime)
    plt.plot(dimArr, pysiglibtime)
    plt.legend(["sigkernel", "pysiglib"])
    #plt.yscale("log")
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    #plt.savefig("sigkernel_times_dim_linear_" + device + ".png", dpi=300)
    #plt.savefig("sigkernel_times_dim_linear_" + device + ".pdf", dpi=300)
    plt.show()

    plt.figure(figsize=(4, 3))
    plt.title("Times " + device)
    plt.xlabel("Path Dimension")
    plt.ylabel("Elapsed Time (s)")
    plt.plot(dimArr, sigkerneltime)
    plt.plot(dimArr, pysiglibtime)
    plt.legend(["sigkernel", "pysiglib"])
    plt.yscale("log")
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    #plt.savefig("sigkernel_times_dim_log_" + device + ".png", dpi=300)
    #plt.savefig("sigkernel_times_dim_log_" + device + ".pdf", dpi=300)
    plt.show()

    # with open('times_avx_' + str(AVX) + '.pickle', 'wb') as handle:
    #     pickle.dump((iisigtime, mysigtime, mysigtimehorner), handle)