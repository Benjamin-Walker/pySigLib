import pysiglib as sig
import numpy as np
import iisignature

if __name__ == '__main__':
    X = np.random.randint(low=0, high=10, size=(1000, 2)).astype("double")

    iisigres = iisignature.sig(X, 3)
    myres = sig.signature(X, 3)

    print(iisigres)
    print(myres[1:])

    for i in range(5):
        print("#" * 30)

    X = np.random.randint(low = 0, high = 10, size = (5, 10, 2)).astype("double")

    iisigres = iisignature.sig(X, 3)
    myres = sig.signature(X, 3)

    for i in range(5):
        print(iisigres[i])
        print(myres[i, 1:])
        print("#" * 30)