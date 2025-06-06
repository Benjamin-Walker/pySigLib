import os
import sys
import platform
from typing import Union
import ctypes
from ctypes import c_float, c_double, c_int32, c_int64, c_bool, POINTER, cast

import numpy as np
import torch

from .error_codes import err_msg

try:
    from ._config import SYSTEM, BUILT_WITH_CUDA, BUILT_WITH_AVX
except ImportError as exc:
    SYSTEM = None
    BUILT_WITH_CUDA = None
    BUILT_WITH_AVX = None
    raise RuntimeError("Could not import configuration properties from _config.py - package may not have been built correctly.") from exc

if SYSTEM != platform.system():
    raise RuntimeError("System on which pySigLib was built does not match the current system - package may not have been built correctly.")

dir_ = os.path.dirname(sys.modules['pysiglib'].__file__)
print(dir_)

if SYSTEM == 'Windows':
    cpsig_path = os.path.join(dir_, 'cpsig.dll')
    #https://github.com/NVIDIA/warp/issues/24
    cpsig = ctypes.CDLL(cpsig_path, winmode = 0)

    if BUILT_WITH_CUDA:
        cusig_path = os.path.join(dir_, 'cusig.dll')
        cusig = ctypes.CDLL(cusig_path, winmode=0)
elif SYSTEM == 'Darwin':
    cpsig_path = os.path.join(dir_, 'libcpsig.dylib')
    cpsig = ctypes.CDLL(cpsig_path)
else:
    raise Exception("Unsupported OS during pysiglib.py")

def poly_length(dimension : int, degree : int) -> int:
    """
    Returns the length of a truncated signature,

    .. math::
        \\sum_{i=0}^N d^i = \\frac{d^{N+1} - 1}{d - 1},

    where :math:`d` is the dimension of the underlying path and :math:`N`
    is the truncation level of the signature.

    :param dimension: Dimension of the undelying path, :math:`d`
    :type dimension: int
    :param degree: Truncation level of the signature, :math:`N`
    :type degree: int
    :return: Length of a truncated signature
    :rtype: int
    """
    cpsig.polyLength.argtypes = (c_int64, c_int64)
    cpsig.polyLength.restype = c_int64
    out = cpsig.polyLength(dimension, degree)
    if out == 0:
        raise Exception("Integer overflow encountered in polyLength")
    return out

class SigDataHandler:
    def __init__(self, path, degree, time_aug, lead_lag):
        self.degree = degree
        self.time_aug = time_aug
        self.lead_lag = lead_lag

        self.get_dims(path)

        if isinstance(path, np.ndarray):
            self.init_numpy(path)

        if isinstance(path, torch.Tensor):
            self.init_torch(path)

    def init_numpy(self, path):
        if path.dtype == np.int32:
            self.dtype = "int32"
            self.data_ptr = path.ctypes.data_as(POINTER(c_int32))
        elif path.dtype == np.int64:
            self.dtype = "int64"
            self.data_ptr = path.ctypes.data_as(POINTER(c_int64))
        elif path.dtype == np.float32:
            self.dtype = "float32"
            self.data_ptr = path.ctypes.data_as(POINTER(c_float))
        elif path.dtype == np.float64:
            self.dtype = "float64"
            self.data_ptr = path.ctypes.data_as(POINTER(c_double))
        else:
            raise ValueError("path.dtype must be int32, int64, float32 or float64. Got " + str(path.dtype) + " instead.")

        _, dimension_ = self.transformed_dims()
        if self.is_batch:
            self.out = np.empty(
                shape=(self.batch_size, poly_length(dimension_, self.degree)),
                dtype=np.float64
            )
        else:
            self.out = np.empty(shape=poly_length(dimension_, self.degree), dtype=np.float64)
        self.out_ptr = self.out.ctypes.data_as(POINTER(c_double))

    def init_torch(self, path):
        if path.dtype == torch.int32:
            self.dtype = "int32"
            self.data_ptr = cast(path.data_ptr(), POINTER(c_int32))
        if path.dtype == torch.int64:
            self.dtype = "int64"
            self.data_ptr = cast(path.data_ptr(), POINTER(c_int64))
        elif path.dtype == torch.float32:
            self.dtype = "float32"
            self.data_ptr = cast(path.data_ptr(), POINTER(c_float))
        elif path.dtype == torch.float64:
            self.dtype = "float64"
            self.data_ptr = cast(path.data_ptr(), POINTER(c_double))
        else:
            raise ValueError("path.dtype must be int32, int64, float32 or float64. Got " + str(path.dtype) + " instead.")

        _, dimension_ = self.transformed_dims()
        if self.is_batch:
            self.out = torch.empty(
                size=(self.batch_size, poly_length(dimension_, self.degree)),
                dtype=torch.float64
            )
        else:
            self.out = torch.empty(
                size=(poly_length(dimension_, self.degree),),
                dtype=torch.float64
            )
        self.out_ptr = cast(self.out.data_ptr(), POINTER(c_double))

    def get_dims(self, path):
        if len(path.shape) == 2:
            self.is_batch = False
            self.length = path.shape[0]
            self.dimension = path.shape[1]


        elif len(path.shape) == 3:
            self.is_batch = True
            self.batch_size = path.shape[0]
            self.length = path.shape[1]
            self.dimension = path.shape[2]

        else:
            raise ValueError("path.shape must have length 2 or 3, got length " + str(path.shape) + " instead.")

    def transformed_dims(self):
        length_ = self.length
        dimension_ = self.dimension
        if self.lead_lag:
            length_ *= 2
            length_ -= 3
            dimension_ *= 2
        if self.time_aug:
            dimension_ += 1
        return (length_, dimension_)


def signature_(data, time_aug = False, lead_lag = False, horner = True):
    err_code = 0
    if data.dtype == "int32":
        cpsig.signatureInt32.argtypes = (POINTER(c_int32), POINTER(c_double), c_int64, c_int64, c_int64, c_bool, c_bool, c_bool)
        cpsig.signatureInt32.restype = c_int64
        err_code = cpsig.signatureInt32(data.data_ptr, data.out_ptr, data.dimension, data.length, data.degree, time_aug, lead_lag, horner)
    elif data.dtype == "int64":
        cpsig.signatureInt64.argtypes = (POINTER(c_int64), POINTER(c_double), c_int64, c_int64, c_int64, c_bool, c_bool, c_bool)
        cpsig.signatureInt64.restype = c_int64
        err_code = cpsig.signatureInt64(data.data_ptr, data.out_ptr, data.dimension, data.length, data.degree, time_aug, lead_lag, horner)
    elif data.dtype == "float32":
        cpsig.signatureFloat.argtypes = (POINTER(c_float), POINTER(c_double), c_int64, c_int64, c_int64, c_bool, c_bool, c_bool)
        cpsig.signatureFloat.restype = c_int64
        err_code = cpsig.signatureFloat(data.data_ptr, data.out_ptr, data.dimension, data.length, data.degree, time_aug, lead_lag, horner)
    elif data.dtype == "float64":
        cpsig.signatureDouble.argtypes = (POINTER(c_double), POINTER(c_double), c_int64, c_int64, c_int64, c_bool, c_bool, c_bool)
        cpsig.signatureDouble.restype = c_int64
        err_code = cpsig.signatureDouble(data.data_ptr, data.out_ptr, data.dimension, data.length, data.degree, time_aug, lead_lag, horner)

    if err_code:
        raise Exception(err_msg[err_code] + " in signature")
    return data.out

def batch_signature_(data, time_aug = False, lead_lag = False, horner = True, parallel = True):
    err_code = 0
    if data.dtype == "int32":
        cpsig.batchSignatureInt32.argtypes = (POINTER(c_int32), POINTER(c_double), c_int64, c_int64, c_int64, c_int64, c_bool, c_bool, c_bool, c_bool)
        cpsig.batchSignatureInt32.restype = c_int64
        err_code = cpsig.batchSignatureInt32(data.data_ptr, data.out_ptr, data.batch_size, data.dimension, data.length, data.degree, time_aug, lead_lag, horner, parallel)
    elif data.dtype == "int64":
        cpsig.batchSignatureInt64.argtypes = (POINTER(c_int64), POINTER(c_double), c_int64, c_int64, c_int64, c_int64, c_bool, c_bool, c_bool, c_bool)
        cpsig.batchSignatureInt64.restype = c_int64
        err_code = cpsig.batchSignatureInt64(data.data_ptr, data.out_ptr, data.batch_size, data.dimension, data.length, data.degree, time_aug, lead_lag, horner, parallel)
    elif data.dtype == "float32":
        cpsig.batchSignatureFloat.argtypes = (POINTER(c_float), POINTER(c_double), c_int64, c_int64, c_int64, c_int64, c_bool, c_bool, c_bool, c_bool)
        cpsig.batchSignatureFloat.restype = c_int64
        err_code = cpsig.batchSignatureFloat(data.data_ptr, data.out_ptr, data.batch_size, data.dimension, data.length, data.degree, time_aug, lead_lag, horner, parallel)
    elif data.dtype == "float64":
        cpsig.batchSignatureDouble.argtypes = (POINTER(c_double), POINTER(c_double), c_int64, c_int64, c_int64, c_int64, c_bool, c_bool, c_bool, c_bool)
        cpsig.batchSignatureDouble.restype = c_int64
        err_code = cpsig.batchSignatureDouble(data.data_ptr, data.out_ptr, data.batch_size, data.dimension, data.length, data.degree, time_aug, lead_lag, horner, parallel)

    if err_code:
        raise Exception(err_msg[err_code] + " in signature")
    return data.out

def signature(
        path : Union[np.ndarray, torch.tensor],
        degree : int,
        time_aug : bool = False,
        lead_lag : bool = False,
        horner : bool = True,
        parallel : bool = True #TODO: change to n_jobs
) -> Union[np.ndarray, torch.tensor]:
    """
    Computes the truncated signature of single path or a batch of paths.
     For a single path :math:`x`, the signature is given by

    .. math::
        S(x)_{[s,t]} := \\left( 1, S(x)^{(1)}_{[s,t]}, \\ldots, S(x)^{(N)}_{[s,t]}\\right) \\in T((\\mathbb{R}^d)),
    .. math::
        S(x)^{(k)}_{[s,t]} := \\int_{s < t_1 < \\cdots < t_k < t} dx_{t_1} \\otimes dx_{t_2} \\otimes \\cdots \\otimes dx_{t_k} \\in \\left(\\mathbb{R}^d\\right)^{\\otimes k}.

    :param path: The underlying path or batch of paths, given as a `numpy.ndarray` or `torch.tensor`.
        For a single path, this must be of shape (length, dimension). For a batch of paths, this must
        be of shape (batch size, length, dimension).
    :type path: numpy.ndarray | torch.tensor
    :param degree: The truncation level of the signature, :math:`N`.
    :type degree: int
    :param time_aug: If set to True, will compute the signature of the time-augmented path, :math:`\\hat{x}_t := (t, x_t)`,
        defined as the original path with an extra channel set to time, :math:`t`.
    :type time_aug: bool
    :param lead_lag: If set to True, will compute the signatue of the path after applying the lead-lag transformation.
    :type lead_lag: bool
    :param horner: If True, will use Horner's algorithm for polynomial multiplication.
    :type horner: bool
    :param parallel: If True, will parallelise the computation.
    :type parallel: bool
    :return: Truncated signature, or a batch of truncated signatures.
    :rtype: numpy.ndarray | torch.tensor
    """
    data = SigDataHandler(path, degree, time_aug, lead_lag)
    if data.is_batch:
        return batch_signature_(data, time_aug, lead_lag, horner, parallel)
    return signature_(data, time_aug, lead_lag, horner)


class SigKernelDataHandler:
    def __init__(self, path1, path2, dyadic_order):
        if isinstance(dyadic_order, tuple) and len(dyadic_order) == 2:
            self.dyadic_order_1 = dyadic_order[0]
            self.dyadic_order_2 = dyadic_order[1]
        elif isinstance(dyadic_order, int):
            self.dyadic_order_1 = dyadic_order
            self.dyadic_order_2 = dyadic_order
        else:
            raise ValueError("dyadicOrder must be an integer or a tuple of length 2")

        if len(path1.shape) == 2:
            self.is_batch = False
            self.batch_size = 1
            self.length_1 = path1.shape[0]
            self.dimension = path1.shape[1]
        elif len(path1.shape) == 3:
            self.is_batch = True
            self.batch_size = path1.shape[0]
            self.length_1 = path1.shape[1]
            self.dimension = path1.shape[2]
        else:
            raise ValueError("path1.shape must have length 2 or 3, got length " + str(path1.shape) + " instead.")

        if len(path2.shape) == 2:
            if self.batch_size != 1:
                raise ValueError("path1, path2 have different batch sizes")
            self.length_2 = path1.shape[0]
            if self.dimension != path1.shape[1]:
                raise ValueError("path1, path2 have different dimensions")
        elif len(path2.shape) == 3:
            if self.batch_size != path1.shape[0]:
                raise ValueError("path1, path2 have different batch sizes")
            self.length_2 = path1.shape[1]
            if self.dimension != path1.shape[2]:
                raise ValueError("path1, path2 have different dimensions")
        else:
            raise ValueError("path2.shape must have length 2 or 3, got length " + str(path2.shape) + " instead.")

        if isinstance(path1, np.ndarray) and isinstance(path2, np.ndarray):
            self.device = "cpu"
            self.out = np.empty(shape=self.batch_size, dtype=np.float64)
            self.out_ptr = self.out.ctypes.data_as(POINTER(c_double))

        elif isinstance(path1, torch.Tensor) and isinstance(path2, torch.Tensor) and path1.device == path2.device:
            self.device = path1.device
            self.out = torch.empty(self.batch_size, dtype=torch.float64, device = self.device)
            self.out_ptr = cast(self.out.data_ptr(), POINTER(c_double))
        else:
            raise ValueError("path1, path2 must both be numpy arrays or both torch arrays on the same device")

def sig_kernel_(data, gram):
    cpsig.batchSigKernel.argtypes = (
    POINTER(c_double), POINTER(c_double), c_int64, c_int64, c_int64, c_int64, c_int64, c_int64)
    cpsig.batchSigKernel.restype = c_int64

    err_code = cpsig.batchSigKernel(
        cast(gram.data_ptr(), POINTER(c_double)),
        data.out_ptr,
        data.batch_size,
        data.dimension,
        data.length_1,
        data.length_2,
        data.dyadic_order_1,
        data.dyadic_order_2
    )

    if err_code:
        raise Exception(err_msg[err_code] + " in sigKernel")
    return data.out

def sig_kernel_cuda_(data, gram):
    cusig.batchSigKernelCUDA.argtypes = (
    POINTER(c_double), POINTER(c_double), c_int64, c_int64, c_int64, c_int64, c_int64, c_int64)
    cusig.batchSigKernelCUDA.restype = c_int64
    err_code = cusig.batchSigKernelCUDA(
        cast(gram.data_ptr(), POINTER(c_double)),
        data.out_ptr, data.batch_size,
        data.dimension,
        data.length_1,
        data.length_2,
        data.dyadic_order_1,
        data.dyadic_order_2
    )

    if err_code:
        raise Exception(err_msg[err_code] + " in sigKernel")
    return data.out

# @profile
def sig_kernel(
        path1 : Union[np.ndarray, torch.tensor],
        path2 : Union[np.ndarray, torch.tensor],
        dyadic_order : Union[int, tuple] #TODO: add n_jobs
) -> Union[np.ndarray, torch.tensor]: #TODO: add time-aug and lead-lag
    """
    Computes a single signature kernel or a batch of signature kernels.
    The signature kernel of two :math:`d`-dimensional paths :math:`x,y`
    is defined as

    .. math::
        k_{x,y}(s,t) := \\left< S(x)_{[0,s]}, S(y)_{[0, t]} \\right>_{T((\\mathbb{R}^d))}

    where the inner product is defined as

    .. math::
        \\left< A, B \\right> := \\sum_{k=0}^{\\infty} \\left< A_k, B_k \\right>_{\\left(\\mathbb{R}^d\\right)^{\\otimes k}}
    .. math::
        \\left< u, v \\right>_{\\left(\\mathbb{R}^d\\right)^{\\otimes k}} := \\prod_{i=1}^k \\left< u_i, v_i \\right>_{\\mathbb{R}^d}

    :param path1: The first underlying path or batch of paths, given as a `numpy.ndarray` or
        `torch.tensor`. For a single path, this must be of shape (length, dimension). For a
         batch of paths, this must be of shape (batch size, length, dimension).
    :type path1: numpy.ndarray | torch.tensor
    :param path2: The second underlying path or batch of paths, given as a `numpy.ndarray`
        or `torch.tensor`. For a single path, this must be of shape (length, dimension).
        For a batch of paths, this must be of shape (batch size, length, dimension).
    :type path2: numpy.ndarray | torch.tensor
    :param dyadic_order: If set to a positive integer :math:`\\lambda`, will refine the
        PDE grid by a factor of :math:`2^\\lambda`.
    :type dyadic_order: int | tuple
    :return: Single signature kernel or batch of signature kernels
    :rtype: numpy.ndarray | torch.tensor
    """
    data = SigKernelDataHandler(path1, path2, dyadic_order)
    x1 = path1[:, 1:, :] - path1[:, :-1, :]
    y1 = path2[:, 1:, :] - path2[:, :-1, :]
    gram = torch.bmm(x1, y1.permute(0, 2, 1))

    if data.device.type == "cpu":
        return sig_kernel_(data, gram)

    if not BUILT_WITH_CUDA:
        raise RuntimeError("pySigLib was build without CUDA - data must be moved to CPU.")
    return sig_kernel_cuda_(data, gram)
