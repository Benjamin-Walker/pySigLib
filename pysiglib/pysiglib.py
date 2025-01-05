import numpy as np
import ctypes
from ctypes import c_double, c_int64, c_bool, POINTER
import os
import sys
import time
import dataHandling

dir_ = os.path.dirname(sys.modules['pysiglib'].__file__)
print(dir_)

cpsig_path = os.path.join(dir_, 'cpsig.dll')
cusig_path = os.path.join(dir_, 'cusig.dll')

#https://github.com/NVIDIA/warp/issues/24
cpsig = ctypes.CDLL(cpsig_path, winmode = 0)
cusig = ctypes.CDLL(cusig_path, winmode = 0)

def cpsig_hello_world(x):
    cpsig.cpsig_hello_world(x)

def cusig_hello_world(x):
    cusig.cusig_hello_world(x)

def getPathElement(numpyArray, lengthIndex, dimIndex):
    dataLength = numpyArray.shape[0]
    dataDimension = numpyArray.shape[1]
    dataPtr = numpyArray.ctypes.data_as(POINTER(c_double))

    cpsig.getPathElement.argtypes = (POINTER(c_double), c_int64, c_int64, c_int64, c_int64)
    cpsig.getPathElement.restype = c_double

    return cpsig.getPathElement(dataPtr, dataLength, dataDimension, lengthIndex, dimIndex)

def polyLength(degree, dimension):
    cpsig.polyLength.argtypes = (c_int64, c_int64)
    cpsig.polyLength.restype = c_int64
    return cpsig.polyLength(degree, dimension)

def signature_(path, degree, timeAug = False, leadLag = False, horner = True):
    length = path.shape[0]
    dimension = path.shape[1]
    out = np.empty(shape=polyLength(dimension, degree), dtype=np.float64)

    if np.issubdtype(path.dtype, np.integer):
        dataPtr = path.ctypes.data_as(POINTER(c_int64))
        outPtr = out.ctypes.data_as(POINTER(c_int64))

        cpsig.signatureInt.argtypes = (POINTER(c_int64), POINTER(c_int64), c_int64, c_int64, c_int64, c_bool, c_bool, c_bool)
        cpsig.signatureInt(dataPtr, outPtr, dimension, length, degree, timeAug, leadLag, horner)

    elif np.issubdtype(path.dtype, np.floating):
        dataPtr = path.ctypes.data_as(POINTER(c_double))
        outPtr = out.ctypes.data_as(POINTER(c_double))

        cpsig.signature.argtypes = (POINTER(c_double), POINTER(c_double), c_int64, c_int64, c_int64, c_bool, c_bool, c_bool)
        cpsig.signature(dataPtr, outPtr, dimension, length, degree, timeAug, leadLag, horner)

    else:
        raise ValueError("path.dtype must be integer or float, got " + str(path.dtype) + " instead.")

    return out

def batchSignature_(path, degree, timeAug = False, leadLag = False, horner = True, parallel = True):
    batchSize = path.shape[0]
    length = path.shape[1]
    dimension = path.shape[2]
    out = np.empty(shape=(batchSize, polyLength(dimension, degree)), dtype=np.float64)

    if np.issubdtype(path.dtype, np.integer):
        dataPtr = path.ctypes.data_as(POINTER(c_int64))
        outPtr = out.ctypes.data_as(POINTER(c_int64))

        cpsig.batchSignatureInt.argtypes = (POINTER(c_int64), POINTER(c_int64), c_int64, c_int64, c_int64, c_int64, c_bool, c_bool, c_bool, c_bool)
        cpsig.batchSignatureInt(dataPtr, outPtr, batchSize, dimension, length, degree, timeAug, leadLag, horner, parallel)

    elif np.issubdtype(path.dtype, np.floating):
        dataPtr = path.ctypes.data_as(POINTER(c_double))
        outPtr = out.ctypes.data_as(POINTER(c_double))

        cpsig.batchSignature.argtypes = (POINTER(c_double), POINTER(c_double), c_int64, c_int64, c_int64, c_int64, c_bool, c_bool, c_bool, c_bool)
        cpsig.batchSignature(dataPtr, outPtr, batchSize, dimension, length, degree, timeAug, leadLag, horner, parallel)

    else:
        raise ValueError("path.dtype must be integer or float, got " + str(path.dtype) + " instead.")

    return out

def signature(path, degree, timeAug = False, leadLag = False, horner = True, parallel = True):
    if len(path.shape) == 2:
        return signature_(path, degree, timeAug, leadLag, horner)
    elif len(path.shape) == 3:
        return batchSignature_(path, degree, timeAug, leadLag, horner, parallel)
    else:
        raise ValueError("path.shape must have length 2 or 3, got length " + str(path.shape) + " instead.")


#https://stackoverflow.com/questions/64478880/how-to-pass-this-numpy-array-to-c-with-ctypes

#gpu ptr might have to be c_void_p