import pysiglib
import numpy as np
import matplotlib.pyplot as plt
import torch
import iisignature
import esig
import signatory
import timeit
from tqdm import tqdm

length = 100
dimension = 10
degree = 7
batch_size = 100

# length = 128
# dimension = 4
# degree = 7
# batch_size = 32

numRuns = 50

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
    X = np.random.uniform(size=(batch_size, length, dimension)).astype("double")

    dimensions = [i for i in range(1, 11)]

    times = []
    vectimes = []

    for dim in dimensions:
        X = np.random.uniform(size=(length, dim)).astype("double")
        times.append(timeFunction(pysiglib.signature, X, degree, vector = False))
        vectimes.append(timeFunction(pysiglib.signature, X, degree, vector=True))

    plt.plot(dimensions, times)
    plt.plot(dimensions, vectimes)
    plt.legend(["no vector", "vector"])
    plt.yscale("log")
    plt.show()

    print(times)
    print(vectimes)