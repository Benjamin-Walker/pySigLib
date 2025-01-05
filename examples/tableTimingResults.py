import pysiglib
import numpy as np
import torch
import iisignature
import esig
import signatory
import timeit
from tqdm import tqdm

def esigbatch(X_, deg):
    for i in range(X_.shape[0]):
        esig.stream2sig(X_[i], deg)

# length = 1024
# dimension = 16
# degree = 4
# batch_size = 32

length = 128
dimension = 4
degree = 7
batch_size = 32

numRuns = 50

X = np.random.uniform(size=(batch_size, length, dimension)).astype("double")

def timeFunction(f, *args, **kwargs):
    best_time = float('inf')
    for i in tqdm(range(numRuns)):
        start = timeit.default_timer()
        f(*args, **kwargs)
        end = timeit.default_timer()
        time_ = end - start
        if time_ < best_time:
            best_time = time_
    return best_time

if __name__ == '__main__':
    #print("\nesig (serial): ", timeFunction(esigbatch, X, degree))
    # print("\niisignature (serial): ", timeFunction(iisignature.sig, X, degree))
    # print("\npysiglib (serial): ", timeFunction(pysiglib.signature, X, degree, parallel = False))
    #
    #
    # print("\nsignatory (parallel): ", timeFunction(signatory.signature, torch.tensor(X), degree))
    print("\npysiglib (parallel): ", timeFunction(pysiglib.signature, X, degree, parallel=True))