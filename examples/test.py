import pysiglib as sig
import numpy as np
import iisignature
import timeit
import signatory
import esig
from tqdm import tqdm
import torch
import pickle

import matplotlib.pyplot as plt
import plotting_params
plotting_params.set_plotting_params(8, 10, 12)

AVX = 2

if __name__ == '__main__':
    sig.cpsig_hello_world(5)
    sig.cusig_hello_world(5)
    print("\n")
    X = np.random.uniform(size = (32, 128, 6)).astype("double")
    torchX = torch.tensor(X)

    N = 2

    def timeiisig(degree):
        best_time = float('inf')
        for i in range(N):
            start = timeit.default_timer()
            iisignature.sig(X, degree)
            end = timeit.default_timer()
            time_ = end - start
            if time_ < best_time:
                best_time = time_
        return best_time

    def timemysig(degree, horner, parallel):
        best_time = float('inf')
        for i in range(N):
            start = timeit.default_timer()
            sig.signature(X, degree, horner = horner, parallel = parallel)
            end = timeit.default_timer()
            time_ = end - start
            if time_ < best_time:
                best_time = time_
        return best_time


    def timesignatory(degree):
        best_time = float('inf')
        for i in range(N):
            start = timeit.default_timer()
            signatory.signature(torchX, degree)
            end = timeit.default_timer()
            time_ = end - start
            if time_ < best_time:
                best_time = time_
        return best_time


    def timeesig(degree):
        best_time = float('inf')
        for i in range(N):
            start = timeit.default_timer()
            esig.stream2sig(X, degree)
            end = timeit.default_timer()
            time_ = end - start
            if time_ < best_time:
                best_time = time_
        return best_time


    isig = iisignature.sig(X, 2)
    mysig = sig.signature(X, 2)

    print(isig)
    print("\n")
    print(mysig[1:])

    degreeArr = [i for i in range(1, 8)]
    iisigtime = []
    #signatorytime = []
    #esigtime = []
    mysigtime = []
    mysigtimehorner = []
    mysigtimehornerparallel = []

    for degree in tqdm(degreeArr):
        iisigtime.append(timeiisig(degree))
        #signatorytime.append(timesignatory(degree))
        #esigtime.append(timeesig(degree))
        mysigtime.append(timemysig(degree, False, False))
        mysigtimehorner.append(timemysig(degree, True, False))
        mysigtimehornerparallel.append(timemysig(degree, True, True))

    print(iisigtime)
    #print(signatorytime)
    #print(esigtime)
    print(mysigtime)
    print(mysigtimehorner)
    print(mysigtimehornerparallel)

    plt.figure(figsize=(4, 3))
    plt.title("AVX " + str(AVX))
    plt.xlabel("Truncation Level")
    plt.ylabel("Elapsed Time (s)")
    # plt.plot(degreeArr, signatorytime)
    plt.plot(degreeArr, iisigtime)
    #plt.plot(degreeArr, esigtime)
    plt.plot(degreeArr, mysigtime, linestyle="--")
    plt.plot(degreeArr, mysigtimehorner, linestyle="--")
    plt.plot(degreeArr, mysigtimehornerparallel, linestyle="--")
    plt.legend(["iisignature", "pySigLib", "pySigLib (Horner)"])
    #plt.yscale("log")
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    # plt.savefig("signature_times_linear_avx_" + str(AVX) + ".png", dpi=300)
    # plt.savefig("signature_times_linear_avx_" + str(AVX) + ".pdf", dpi=300)
    plt.show()

    plt.figure(figsize=(4, 3))
    plt.title("AVX " + str(AVX))
    plt.xlabel("Truncation Level")
    plt.ylabel("Elapsed Time (s)")
    #plt.plot(degreeArr, signatorytime)
    plt.plot(degreeArr, iisigtime)
    #plt.plot(degreeArr, esigtime)
    plt.plot(degreeArr, mysigtime, linestyle="--")
    plt.plot(degreeArr, mysigtimehorner, linestyle="--")
    plt.plot(degreeArr, mysigtimehornerparallel, linestyle="--")
    plt.legend(["iisignature", "pySigLib", "pySigLib (Horner)"])
    plt.yscale("log")
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    #plt.savefig("signature_times_log_avx_" + str(AVX) + ".png", dpi=300)
    #plt.savefig("signature_times_log_avx_" + str(AVX) + ".pdf", dpi=300)
    plt.show()

    # with open('times_avx_' + str(AVX) + '.pickle', 'wb') as handle:
    #     pickle.dump((iisigtime, mysigtime, mysigtimehorner), handle)
