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

import timeit

import numpy as np
import iisignature

import pysiglib

if __name__ == '__main__':

    batch_size = 100
    length = 1000
    dim = 10
    degree = 4

    X = np.random.uniform(size=(batch_size, length, dim)).astype("double")
    sig = iisignature.sig(X, degree)

    start = timeit.default_timer()
    sig_combine = iisignature.sigcombine(sig, sig, dim, degree)
    end = timeit.default_timer()

    print(end - start)
    print(sig_combine[0][:5])

    sig = pysiglib.signature(X, degree)

    start = timeit.default_timer()
    sig_combine = pysiglib.poly_mult(sig, sig, dim, degree)
    end = timeit.default_timer()
    print(end - start)
    print(sig_combine[0][1:6])
