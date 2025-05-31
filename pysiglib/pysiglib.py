import numpy as np
import torch
import ctypes
from ctypes import c_float, c_double, c_int32, c_int64, c_bool, POINTER, cast
import os
import sys
import platform
from .errorCodes import errMsg
import warnings

# from line_profiler_pycharm import profile

SYSTEM = platform.system()
USE_CUDA = True

dir_ = os.path.dirname(sys.modules['pysiglib'].__file__)
print(dir_)

if SYSTEM == 'Windows':
    cpsig_path = os.path.join(dir_, 'cpsig.dll')
    #https://github.com/NVIDIA/warp/issues/24
    cpsig = ctypes.CDLL(cpsig_path, winmode = 0)

    if USE_CUDA:
        cusig_path = os.path.join(dir_, 'cusig.dll')
        cusig = ctypes.CDLL(cusig_path, winmode=0)
elif SYSTEM == 'Darwin':
    cpsig_path = os.path.join(dir_, 'libcpsig.dylib')
    cpsig = ctypes.CDLL(cpsig_path)
else:
    raise Exception("Unsupported OS during pysiglib.py")

def polyLength(dimension, degree):
    cpsig.polyLength.argtypes = (c_int64, c_int64)
    cpsig.polyLength.restype = c_int64
    out = cpsig.polyLength(dimension, degree)
    if out == 0:
        raise Exception("Integer overflow encountered in polyLength")
    return out

class sigDataHandler:
    def __init__(self, path, degree, timeAug, leadLag):
        self.degree = degree
        self.timeAug = timeAug
        self.leadLag = leadLag

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

        length_, dimension_ = self.transformedDims()
        if self.isBatch:
            self.out = np.empty(shape=(self.batchSize, polyLength(dimension_, self.degree)), dtype=np.float64)
        else:
            self.out = np.empty(shape=polyLength(dimension_, self.degree), dtype=np.float64)
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

        length_, dimension_ = self.transformedDims()
        if self.isBatch:
            self.out = torch.empty(size=(self.batchSize, polyLength(dimension_, self.degree)), dtype=torch.float64)
        else:
            self.out = torch.empty(size=(polyLength(dimension_, self.degree),), dtype=torch.float64)
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

    def transformedDims(self):
        length_ = self.length
        dimension_ = self.dimension
        if self.leadLag:
            length_ *= 2
            length_ -= 3
            dimension_ *= 2
        if self.timeAug:
            dimension_ += 1
        return (length_, dimension_)


def signature_(data, timeAug = False, leadLag = False, horner = True):
    errCode = 0
    if data.dtype == "int32":
        cpsig.signatureInt32.argtypes = (POINTER(c_int32), POINTER(c_double), c_int64, c_int64, c_int64, c_bool, c_bool, c_bool)
        cpsig.signatureInt32.restype = c_int64
        errCode = cpsig.signatureInt32(data.dataPtr, data.outPtr, data.dimension, data.length, data.degree, timeAug, leadLag, horner)
    elif data.dtype == "int64":
        cpsig.signatureInt64.argtypes = (POINTER(c_int64), POINTER(c_double), c_int64, c_int64, c_int64, c_bool, c_bool, c_bool)
        cpsig.signatureInt64.restype = c_int64
        errCode = cpsig.signatureInt64(data.dataPtr, data.outPtr, data.dimension, data.length, data.degree, timeAug, leadLag, horner)
    elif data.dtype == "float32":
        cpsig.signatureFloat.argtypes = (POINTER(c_float), POINTER(c_double), c_int64, c_int64, c_int64, c_bool, c_bool, c_bool)
        cpsig.signatureFloat.restype = c_int64
        errCode = cpsig.signatureFloat(data.dataPtr, data.outPtr, data.dimension, data.length, data.degree, timeAug, leadLag, horner)
    elif data.dtype == "float64":
        cpsig.signatureDouble.argtypes = (POINTER(c_double), POINTER(c_double), c_int64, c_int64, c_int64, c_bool, c_bool, c_bool)
        cpsig.signatureDouble.restype = c_int64
        errCode = cpsig.signatureDouble(data.dataPtr, data.outPtr, data.dimension, data.length, data.degree, timeAug, leadLag, horner)

    if errCode:
        raise Exception(errMsg[errCode] + " in signature")
    return data.out

def batchSignature_(data, timeAug = False, leadLag = False, horner = True, parallel = True):
    errCode = 0
    if data.dtype == "int32":
        cpsig.batchSignatureInt32.argtypes = (POINTER(c_int32), POINTER(c_double), c_int64, c_int64, c_int64, c_int64, c_bool, c_bool, c_bool, c_bool)
        cpsig.batchSignatureInt32.restype = c_int64
        errCode = cpsig.batchSignatureInt32(data.dataPtr, data.outPtr, data.batchSize, data.dimension, data.length, data.degree, timeAug, leadLag, horner, parallel)
    elif data.dtype == "int64":
        cpsig.batchSignatureInt64.argtypes = (POINTER(c_int64), POINTER(c_double), c_int64, c_int64, c_int64, c_int64, c_bool, c_bool, c_bool, c_bool)
        cpsig.batchSignatureInt64.restype = c_int64
        errCode = cpsig.batchSignatureInt64(data.dataPtr, data.outPtr, data.batchSize, data.dimension, data.length, data.degree, timeAug, leadLag, horner, parallel)
    elif data.dtype == "float32":
        cpsig.batchSignatureFloat.argtypes = (POINTER(c_float), POINTER(c_double), c_int64, c_int64, c_int64, c_int64, c_bool, c_bool, c_bool, c_bool)
        cpsig.batchSignatureFloat.restype = c_int64
        errCode = cpsig.batchSignatureFloat(data.dataPtr, data.outPtr, data.batchSize, data.dimension, data.length, data.degree, timeAug, leadLag, horner, parallel)
    elif data.dtype == "float64":
        cpsig.batchSignatureDouble.argtypes = (POINTER(c_double), POINTER(c_double), c_int64, c_int64, c_int64, c_int64, c_bool, c_bool, c_bool, c_bool)
        cpsig.batchSignatureDouble.restype = c_int64
        errCode = cpsig.batchSignatureDouble(data.dataPtr, data.outPtr, data.batchSize, data.dimension, data.length, data.degree, timeAug, leadLag, horner, parallel)

    if errCode:
        raise Exception(errMsg[errCode] + " in signature")
    return data.out

def signature(path, degree, timeAug = False, leadLag = False, horner = True, parallel = True):
    data = sigDataHandler(path, degree, timeAug, leadLag)
    if data.isBatch:
        return batchSignature_(data, timeAug, leadLag, horner, parallel)
    else:
        return signature_(data, timeAug, leadLag, horner)


class sigKernelDataHandler:
    def __init__(self, path1, path2, dyadicOrder):
        if isinstance(dyadicOrder, tuple) and len(dyadicOrder) == 2:
            self.dyadicOrder1 = dyadicOrder[0]
            self.dyadicOrder2 = dyadicOrder[1]
        elif isinstance(dyadicOrder, int):
            self.dyadicOrder1 = dyadicOrder
            self.dyadicOrder2 = dyadicOrder
        else:
            raise ValueError("dyadicOrder must be an integer or a tuple of length 2")

        if len(path1.shape) == 2:
            self.isBatch = False
            self.batchSize = 1
            self.length1 = path1.shape[0]
            self.dimension = path1.shape[1]
        elif len(path1.shape) == 3:
            self.isBatch = True
            self.batchSize = path1.shape[0]
            self.length1 = path1.shape[1]
            self.dimension = path1.shape[2]
        else:
            raise ValueError("path1.shape must have length 2 or 3, got length " + str(path1.shape) + " instead.")

        if len(path2.shape) == 2:
            if self.batchSize != 1:
                raise ValueError("path1, path2 have different batch sizes")
            self.length2 = path1.shape[0]
            if self.dimension != path1.shape[1]:
                raise ValueError("path1, path2 have different dimensions")
        elif len(path2.shape) == 3:
            if self.batchSize != path1.shape[0]:
                raise ValueError("path1, path2 have different batch sizes")
            self.length2 = path1.shape[1]
            if self.dimension != path1.shape[2]:
                raise ValueError("path1, path2 have different dimensions")
        else:
            raise ValueError("path2.shape must have length 2 or 3, got length " + str(path2.shape) + " instead.")

        if isinstance(path1, np.ndarray) and isinstance(path2, np.ndarray):
            self.device = "cpu"
            if path1.dtype != path2.dtype:
                warnings.warn("Path dtypes differ. This will trigger a conversion to the larger type.")
                common_dtype = np.result_type(path1, path2)
                path1_ = path1.astype(common_dtype)
                path2_ = path2.astype(common_dtype)

                self.dtype, self.dataPtr1 = self.initNumpy(path1_)
                self.dtype, self.dataPtr2 = self.initNumpy(path2_)
            else:
                self.dtype, self.dataPtr1 = self.initNumpy(path1)
                self.dtype, self.dataPtr2 = self.initNumpy(path2)

            self.out = np.empty(shape=self.batchSize, dtype=np.float64)
            self.outPtr = self.out.ctypes.data_as(POINTER(c_double))

        elif isinstance(path1, torch.Tensor) and isinstance(path2, torch.Tensor) and path1.device == path2.device:
            self.device = path1.device
            if path1.dtype != path2.dtype:
                warnings.warn("Path dtypes differ. This will trigger a conversion to the larger type.")
                common_dtype = torch.promote_types(path1.dtype, path2.dtype)
                path1_ = path1.to(common_dtype)
                path2_ = path2.to(common_dtype)
                self.dtype, self.dataPtr1 = self.initTorch(path1_)
                self.dtype, self.dataPtr2 = self.initTorch(path2_)
            else:
                self.dtype, self.dataPtr1 = self.initTorch(path1)
                self.dtype, self.dataPtr2 = self.initTorch(path2)

            self.out = torch.empty(self.batchSize, dtype=torch.float64, device = self.device)
            self.outPtr = cast(self.out.data_ptr(), POINTER(c_double))
        else:
            raise ValueError("path1, path2 must both be numpy arrays or both torch arrays on the same device")

    def initNumpy(self, path):
        if path.dtype == np.int32:
            dtype = "int32"
            dataPtr = path.ctypes.data_as(POINTER(c_int32))
        elif path.dtype == np.int64:
            dtype = "int64"
            dataPtr = path.ctypes.data_as(POINTER(c_int64))
        elif path.dtype == np.float32:
            dtype = "float32"
            dataPtr = path.ctypes.data_as(POINTER(c_float))
        elif path.dtype == np.float64:
            dtype = "float64"
            dataPtr = path.ctypes.data_as(POINTER(c_double))
        else:
            raise ValueError("path.dtype must be int32, int64, float32 or float64. Got " + str(path.dtype) + " instead.")
        return dtype, dataPtr

    def initTorch(self, path):
        if path.dtype == torch.int32:
            dtype = "int32"
            dataPtr = cast(path.data_ptr(), POINTER(c_int32))
        if path.dtype == torch.int64:
            dtype = "int64"
            dataPtr = cast(path.data_ptr(), POINTER(c_int64))
        elif path.dtype == torch.float32:
            dtype = "float32"
            dataPtr = cast(path.data_ptr(), POINTER(c_float))
        elif path.dtype == torch.float64:
            dtype = "float64"
            dataPtr = cast(path.data_ptr(), POINTER(c_double))
        else:
            raise ValueError("path.dtype must be int32, int64, float32 or float64. Got " + str(path.dtype) + " instead.")
        return dtype, dataPtr

def sigKernel_(data, gram):
    errCode = 0
    # if data.dtype == "int32":
    #     cpsig.batchSigKernelInt32.argtypes = (
    #     POINTER(c_int32), POINTER(c_int32), POINTER(c_double), c_int64, c_int64, c_int64, c_int64, c_int64, c_int64)
    #     cpsig.batchSigKernelInt32.restype = c_int64
    #     errCode = cpsig.batchSigKernelInt32(data.dataPtr1, data.dataPtr2, data.outPtr, data.batchSize, data.dimension,
    #                                    data.length1, data.length2, data.dyadicOrder1, data.dyadicOrder2)
    # elif data.dtype == "int64":
    #     cpsig.batchSigKernelInt64.argtypes = (
    #     POINTER(c_int64), POINTER(c_int64), POINTER(c_double), c_int64, c_int64, c_int64, c_int64, c_int64, c_int64)
    #     cpsig.batchSigKernelInt64.restype = c_int64
    #     errCode = cpsig.batchSigKernelInt64(data.dataPtr1, data.dataPtr2, data.outPtr, data.batchSize, data.dimension,
    #                                    data.length1, data.length2, data.dyadicOrder1, data.dyadicOrder2)
    # elif data.dtype == "float32":
    #     cpsig.batchSigKernelFloat.argtypes = (
    #     POINTER(c_float), POINTER(c_float), POINTER(c_double), c_int64, c_int64, c_int64, c_int64, c_int64, c_int64)
    #     cpsig.batchSigKernelFloat.restype = c_int64
    #     errCode = cpsig.batchSigKernelFloat(data.dataPtr1, data.dataPtr2, data.outPtr, data.batchSize, data.dimension,
    #                                    data.length1, data.length2, data.dyadicOrder1, data.dyadicOrder2)
    # elif data.dtype == "float64":
    #     cpsig.batchSigKernelDouble.argtypes = (
    #     POINTER(c_double), POINTER(c_double), POINTER(c_double), c_int64, c_int64, c_int64, c_int64, c_int64, c_int64)
    #     cpsig.batchSigKernelDouble.restype = c_int64
    #     errCode = cpsig.batchSigKernelDouble(data.dataPtr1, data.dataPtr2, data.outPtr, data.batchSize, data.dimension,
    #                                    data.length1, data.length2, data.dyadicOrder1, data.dyadicOrder2)
    #
    # if errCode:
    #     raise Exception(errMsg[errCode] + " in sigKernel")
    # return data.out

    cpsig.batchSigKernel.argtypes = (
    POINTER(c_double), POINTER(c_double), c_int64, c_int64, c_int64, c_int64, c_int64, c_int64)
    cpsig.batchSigKernel.restype = c_int64

    errCode = cpsig.batchSigKernel(cast(gram.data_ptr(), POINTER(c_double)), data.outPtr, data.batchSize, data.dimension,
                                   data.length1, data.length2, data.dyadicOrder1, data.dyadicOrder2)

    if errCode:
        raise Exception(errMsg[errCode] + " in sigKernel")
    return data.out

def sigKernelCUDA_(data):
    errCode = 0
    if data.dtype == "int32":
        cusig.batchSigKernelInt32CUDA.argtypes = (
        POINTER(c_int32), POINTER(c_int32), POINTER(c_double), c_int64, c_int64, c_int64, c_int64, c_int64, c_int64)
        cusig.batchSigKernelInt32CUDA.restype = c_int64
        errCode = cusig.batchSigKernelInt32CUDA(data.dataPtr1, data.dataPtr2, data.outPtr, data.batchSize, data.dimension,
                                       data.length1, data.length2, data.dyadicOrder1, data.dyadicOrder2)
    elif data.dtype == "int64":
        cusig.batchSigKernelInt64CUDA.argtypes = (
        POINTER(c_int64), POINTER(c_int64), POINTER(c_double), c_int64, c_int64, c_int64, c_int64, c_int64, c_int64)
        cusig.batchSigKernelInt64CUDA.restype = c_int64
        errCode = cusig.batchSigKernelInt64CUDA(data.dataPtr1, data.dataPtr2, data.outPtr, data.batchSize, data.dimension,
                                       data.length1, data.length2, data.dyadicOrder1, data.dyadicOrder2)
    elif data.dtype == "float32":
        cusig.batchSigKernelFloatCUDA.argtypes = (
        POINTER(c_float), POINTER(c_float), POINTER(c_double), c_int64, c_int64, c_int64, c_int64, c_int64, c_int64)
        cusig.batchSigKernelFloatCUDA.restype = c_int64
        errCode = cusig.batchSigKernelFloatCUDA(data.dataPtr1, data.dataPtr2, data.outPtr, data.batchSize, data.dimension,
                                       data.length1, data.length2, data.dyadicOrder1, data.dyadicOrder2)
    elif data.dtype == "float64":
        cusig.batchSigKernelDoubleCUDA.argtypes = (
        POINTER(c_double), POINTER(c_double), POINTER(c_double), c_int64, c_int64, c_int64, c_int64, c_int64, c_int64)
        cusig.batchSigKernelDoubleCUDA.restype = c_int64
        errCode = cusig.batchSigKernelDoubleCUDA(data.dataPtr1, data.dataPtr2, data.outPtr, data.batchSize, data.dimension,
                                       data.length1, data.length2, data.dyadicOrder1, data.dyadicOrder2)

    if errCode:
        raise Exception(errMsg[errCode] + " in sigKernel")
    return data.out

# @profile
def sigKernel(path1, path2, dyadicOrder):
    data = sigKernelDataHandler(path1, path2, dyadicOrder)
    if data.device.type == "cpu":
        x1 = path1[:, 1:, :] - path1[:, :-1, :]
        y1 = path2[:, 1:, :] - path2[:, :-1, :]
        gram = torch.bmm(x1, y1.permute(0,2,1))
        return sigKernel_(data, gram)
    else:
        return sigKernelCUDA_(data)

#https://stackoverflow.com/questions/64478880/how-to-pass-this-numpy-array-to-c-with-ctypes

#gpu ptr might have to be c_void_p