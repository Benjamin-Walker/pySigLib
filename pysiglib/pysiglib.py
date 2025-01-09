import numpy as np
import torch
import ctypes
from ctypes import c_float, c_double, c_int32, c_int64, c_bool, POINTER, cast
import os
import sys

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

def polyLength(degree, dimension):
    cpsig.polyLength.argtypes = (c_int64, c_int64)
    cpsig.polyLength.restype = c_int64
    return cpsig.polyLength(degree, dimension)

class dataHandler:
    def __init__(self, path, degree):
        self.degree = degree

        self.getDims(path)

        if isinstance(path, np.ndarray):
            self.initNumpy(path)

        if isinstance(path, torch.Tensor):
            self.initTorch(path)

    def initNumpy(self, path):
        if path.dtype == np.int32:
            self.dtype = "int32"
            self.dataPtr = path.ctypes.data_as(POINTER(c_int32))
        elif path.dtype == np.int64:
            self.dtype = "int64"
            self.dataPtr = path.ctypes.data_as(POINTER(c_int64))
        elif path.dtype == np.float32:
            self.dtype = "float32"
            self.dataPtr = path.ctypes.data_as(POINTER(c_float))
        elif path.dtype == np.float64:
            self.dtype = "float64"
            self.dataPtr = path.ctypes.data_as(POINTER(c_double))
        else:
            raise ValueError("path.dtype must be int32, int64, float32 or float64. Got " + str(path.dtype) + " instead.")

        if self.isBatch:
            self.out = np.empty(shape=(self.batchSize, polyLength(self.dimension, self.degree)), dtype=np.float64)
        else:
            self.out = np.empty(shape=polyLength(self.dimension, self.degree), dtype=np.float64)
        self.outPtr = self.out.ctypes.data_as(POINTER(c_double))

    def initTorch(self, path):
        if path.dtype == torch.int32:
            self.dtype = "int32"
            self.dataPtr = cast(path.data_ptr(), POINTER(c_int32))
        if path.dtype == torch.int64:
            self.dtype = "int64"
            self.dataPtr = cast(path.data_ptr(), POINTER(c_int64))
        elif path.dtype == torch.float32:
            self.dtype = "float32"
            self.dataPtr = cast(path.data_ptr(), POINTER(c_float))
        elif path.dtype == torch.float64:
            self.dtype = "float64"
            self.dataPtr = cast(path.data_ptr(), POINTER(c_double))
        else:
            raise ValueError("path.dtype must be int32, int64, float32 or float64. Got " + str(path.dtype) + " instead.")

        if self.isBatch:
            self.out = torch.empty(size=(self.batchSize, polyLength(self.dimension, self.degree)), dtype=torch.float64)
        else:
            self.out = torch.empty(size=polyLength(self.dimension, self.degree), dtype=torch.float64)
        self.outPtr = cast(self.out.data_ptr(), POINTER(c_double))

    def getDims(self, path):
        if len(path.shape) == 2:
            self.isBatch = False
            self.length = path.shape[0]
            self.dimension = path.shape[1]

        elif len(path.shape) == 3:
            self.isBatch = True
            self.batchSize = path.shape[0]
            self.length = path.shape[1]
            self.dimension = path.shape[2]

        else:
            raise ValueError("path.shape must have length 2 or 3, got length " + str(path.shape) + " instead.")


def signature_(data, timeAug = False, leadLag = False, horner = True):
    if data.dtype == "int32":
        cpsig.signatureInt32.argtypes = (POINTER(c_int32), POINTER(c_double), c_int64, c_int64, c_int64, c_bool, c_bool, c_bool)
        cpsig.signatureInt32(data.dataPtr, data.outPtr, data.dimension, data.length, data.degree, timeAug, leadLag, horner)
    elif data.dtype == "int64":
        cpsig.signatureInt64.argtypes = (POINTER(c_int64), POINTER(c_double), c_int64, c_int64, c_int64, c_bool, c_bool, c_bool)
        cpsig.signatureInt64(data.dataPtr, data.outPtr, data.dimension, data.length, data.degree, timeAug, leadLag, horner)
    elif data.dtype == "float32":
        cpsig.signatureFloat.argtypes = (POINTER(c_float), POINTER(c_double), c_int64, c_int64, c_int64, c_bool, c_bool, c_bool)
        cpsig.signatureFloat(data.dataPtr, data.outPtr, data.dimension, data.length, data.degree, timeAug, leadLag, horner)
    elif data.dtype == "float64":
        cpsig.signatureDouble.argtypes = (POINTER(c_double), POINTER(c_double), c_int64, c_int64, c_int64, c_bool, c_bool, c_bool)
        cpsig.signatureDouble(data.dataPtr, data.outPtr, data.dimension, data.length, data.degree, timeAug, leadLag, horner)
    return data.out

def batchSignature_(data, timeAug = False, leadLag = False, horner = True, parallel = True):
    if data.dtype == "int32":
        cpsig.batchSignatureInt32.argtypes = (POINTER(c_int32), POINTER(c_double), c_int64, c_int64, c_int64, c_int64, c_bool, c_bool, c_bool, c_bool)
        cpsig.batchSignatureInt32(data.dataPtr, data.outPtr, data.batchSize, data.dimension, data.length, data.degree, timeAug, leadLag, horner, parallel)
    if data.dtype == "int64":
        cpsig.batchSignatureInt64.argtypes = (POINTER(c_int64), POINTER(c_double), c_int64, c_int64, c_int64, c_int64, c_bool, c_bool, c_bool, c_bool)
        cpsig.batchSignatureInt64(data.dataPtr, data.outPtr, data.batchSize, data.dimension, data.length, data.degree, timeAug, leadLag, horner, parallel)
    elif data.dtype == "float32":
        cpsig.batchSignatureFloat.argtypes = (POINTER(c_float), POINTER(c_double), c_int64, c_int64, c_int64, c_int64, c_bool, c_bool, c_bool, c_bool)
        cpsig.batchSignatureFloat(data.dataPtr, data.outPtr, data.batchSize, data.dimension, data.length, data.degree, timeAug, leadLag, horner, parallel)
    elif data.dtype == "float64":
        cpsig.batchSignatureDouble.argtypes = (POINTER(c_double), POINTER(c_double), c_int64, c_int64, c_int64, c_int64, c_bool, c_bool, c_bool, c_bool)
        cpsig.batchSignatureDouble(data.dataPtr, data.outPtr, data.batchSize, data.dimension, data.length, data.degree, timeAug, leadLag, horner, parallel)
    return data.out

def signature(path, degree, timeAug = False, leadLag = False, horner = True, parallel = True):
    data = dataHandler(path, degree)
    if data.isBatch:
        return batchSignature_(data, timeAug, leadLag, horner, parallel)
    else:
        return signature_(data, timeAug, leadLag, horner)


#https://stackoverflow.com/questions/64478880/how-to-pass-this-numpy-array-to-c-with-ctypes

#gpu ptr might have to be c_void_p