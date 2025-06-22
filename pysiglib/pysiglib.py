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
"""
Signature Computations on CPU and GPU
"""

import os
import sys
import platform
from typing import Union
import ctypes
from ctypes import c_float, c_double, c_int32, c_int64, c_bool, POINTER, cast

import numpy as np
import torch

from .param_checks import check_type, check_type_multiple, check_non_neg, check_dtype, check_dtype_double, ensure_own_contiguous_storage
from .error_codes import err_msg

######################################################
# Figure out how pysiglib was built, in particular
# whether CUDA is being used
######################################################

try:
    from ._config import SYSTEM, BUILT_WITH_CUDA, BUILT_WITH_AVX
except ImportError as exc:
    raise RuntimeError("Could not import configuration properties from _config.py - package may not have been built correctly.") from exc

if SYSTEM != platform.system():
    raise RuntimeError("System on which pySigLib was built does not match the current system - package may not have been built correctly.")

######################################################
# Load the cpsig and cusig libraries
######################################################

# winmode = 0 is necessary here
# https://github.com/NVIDIA/warp/issues/24

dir_ = os.path.dirname(sys.modules['pysiglib'].__file__)

if SYSTEM == 'Windows':
    cpsig_path = os.path.join(dir_, 'cpsig.dll')
    cpsig = ctypes.CDLL(cpsig_path, winmode = 0)

    if BUILT_WITH_CUDA:
        cusig_path = os.path.join(dir_, 'cusig.dll')
        cusig = ctypes.CDLL(cusig_path, winmode=0)
elif SYSTEM == "Linux":
    cpsig_path = os.path.join(dir_, 'libcpsig.so')
    cpsig = ctypes.CDLL(cpsig_path, winmode=0)

    if BUILT_WITH_CUDA:
        cusig_path = os.path.join(dir_, 'libcusig.so')
        cusig = ctypes.CDLL(cusig_path, winmode=0)
elif SYSTEM == 'Darwin':
    cpsig_path = os.path.join(dir_, 'libcpsig.dylib')
    cpsig = ctypes.CDLL(cpsig_path)
else:
    raise Exception("Unsupported OS during pysiglib.py")

######################################################
# Set argtypes and restypes for all imported functions
######################################################

cpsig.sig_length.argtypes = (c_int64, c_int64)
cpsig.sig_length.restype = c_int64

cpsig.sig_combine.argtypes = (POINTER(c_double), POINTER(c_double), POINTER(c_double), c_int64, c_int64)
cpsig.sig_combine.restype = c_int64

cpsig.batch_sig_combine.argtypes = (POINTER(c_double), POINTER(c_double), POINTER(c_double), c_int64, c_int64, c_int64, c_bool)
cpsig.batch_sig_combine.restype = c_int64

cpsig.signature_int32.argtypes = (POINTER(c_int32), POINTER(c_double), c_int64, c_int64, c_int64, c_bool, c_bool, c_bool)
cpsig.signature_int32.restype = c_int64

cpsig.signature_int64.argtypes = (POINTER(c_int64), POINTER(c_double), c_int64, c_int64, c_int64, c_bool, c_bool, c_bool)
cpsig.signature_int64.restype = c_int64

cpsig.signature_float.argtypes = (POINTER(c_float), POINTER(c_double), c_int64, c_int64, c_int64, c_bool, c_bool, c_bool)
cpsig.signature_float.restype = c_int64

cpsig.signature_double.argtypes = (POINTER(c_double), POINTER(c_double), c_int64, c_int64, c_int64, c_bool, c_bool, c_bool)
cpsig.signature_double.restype = c_int64

cpsig.batch_signature_int32.argtypes = (POINTER(c_int32), POINTER(c_double), c_int64, c_int64, c_int64, c_int64, c_bool, c_bool, c_bool, c_bool)
cpsig.batch_signature_int32.restype = c_int64

cpsig.batch_signature_int64.argtypes = (POINTER(c_int64), POINTER(c_double), c_int64, c_int64, c_int64, c_int64, c_bool, c_bool, c_bool, c_bool)
cpsig.batch_signature_int64.restype = c_int64

cpsig.batch_signature_float.argtypes = (POINTER(c_float), POINTER(c_double), c_int64, c_int64, c_int64, c_int64, c_bool, c_bool, c_bool, c_bool)
cpsig.batch_signature_float.restype = c_int64

cpsig.batch_signature_double.argtypes = (POINTER(c_double), POINTER(c_double), c_int64, c_int64, c_int64, c_int64, c_bool, c_bool, c_bool, c_bool)
cpsig.batch_signature_double.restype = c_int64

cpsig.batch_sig_kernel.argtypes = (POINTER(c_double), POINTER(c_double), c_int64, c_int64, c_int64, c_int64, c_int64, c_int64)
cpsig.batch_sig_kernel.restype = c_int64

if BUILT_WITH_CUDA:
    cusig.batch_sig_kernel_cuda.argtypes = (POINTER(c_double), POINTER(c_double), c_int64, c_int64, c_int64, c_int64, c_int64, c_int64)
    cusig.batch_sig_kernel_cuda.restype = c_int64

######################################################
# Some dicts to simplify dtype cases
######################################################

DTYPES = {
    "int32": c_int32,
    "int64": c_int64,
    "float32": c_float,
    "float64": c_double
}

CPSIG_SIGNATURE = {
    "int32": cpsig.signature_int32,
    "int64": cpsig.signature_int64,
    "float32": cpsig.signature_float,
    "float64": cpsig.signature_double
}

CPSIG_BATCH_SIGNATURE = {
    "int32": cpsig.batch_signature_int32,
    "int64": cpsig.batch_signature_int64,
    "float32": cpsig.batch_signature_float,
    "float64": cpsig.batch_signature_double
}

######################################################
# Python wrappers
######################################################

def sig_length(dimension : int, degree : int) -> int:
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
    check_type(dimension, "dimension", int)
    check_type(degree, "degree", int)
    check_non_neg(dimension, "dimension")
    check_non_neg(degree, "degree")

    out = cpsig.sig_length(dimension, degree)
    if out == 0:
        raise ValueError("Integer overflow encountered in sig_length")
    return out

class PolyDataHandler:
    def __init__(self, poly1_, poly2_, dimension, degree):

        check_type_multiple(poly1_, "poly1", (np.ndarray, torch.Tensor))
        check_type_multiple(poly2_, "poly2", (np.ndarray, torch.Tensor))
        self.poly1 = ensure_own_contiguous_storage(poly1_, 4)
        self.poly2 = ensure_own_contiguous_storage(poly2_, 4)
        check_dtype_double(self.poly1, "poly1")
        check_dtype_double(self.poly2, "poly2")

        self.poly_len_ = sig_length(dimension, degree)
        if self.poly1.shape[-1] != self.poly_len_:
            raise ValueError("poly1 is of incorrect length. Expected " + str(self.poly_len_) + ", got " + str(self.poly1.shape[-1]))
        if self.poly2.shape[-1] != self.poly_len_:
            raise ValueError("poly2 is of incorrect length. Expected " + str(self.poly_len_) + ", got " + str(self.poly2.shape[-1]))

        if len(self.poly1.shape) == 1:
            self.is_batch = False
            self.batch_size = 1
            self.length_1 = self.poly1.shape[0]
        elif len(self.poly1.shape) == 2:
            self.is_batch = True
            self.batch_size = self.poly1.shape[0]
            self.length_1 = self.poly1.shape[1]
        else:
            raise ValueError("poly1.shape must have length 1 or 2, got length " + str(len(self.poly1.shape)) + " instead.")

        if len(self.poly2.shape) == 1:
            if self.batch_size != 1:
                raise ValueError("poly1, poly2 have different batch sizes")
            self.length_2 = self.poly2.shape[0]
        elif len(self.poly2.shape) == 2:
            if self.batch_size != self.poly2.shape[0]:
                raise ValueError("path1, path2 have different batch sizes")
            self.length_2 = self.poly2.shape[1]
        else:
            raise ValueError("poly2.shape must have length 1 or 2, got length " + str(len(self.poly2.shape)) + " instead.")

        self.result_length = self.batch_size * self.poly_len_

        if isinstance(self.poly1, np.ndarray) and isinstance(self.poly2, np.ndarray):
            self.poly1_ptr = self.poly1.ctypes.data_as(POINTER(c_double))
            self.poly2_ptr = self.poly2.ctypes.data_as(POINTER(c_double))
            if self.is_batch:
                self.out = np.empty(
                    shape=(self.batch_size, self.poly_len_),
                    dtype=np.float64
                )
            else:
                self.out = np.empty(
                    shape=self.poly_len_,
                    dtype=np.float64
                )
            self.out_ptr = self.out.ctypes.data_as(POINTER(c_double))

        elif isinstance(self.poly1, torch.Tensor) and isinstance(self.poly2, torch.Tensor):
            if not (self.poly1.device.type == "cpu" and self.poly2.device.type == "cpu"):
                raise ValueError("Data must be located on the cpu")
            self.poly1_ptr = cast(self.poly1.data_ptr(), POINTER(c_double))
            self.poly2_ptr = cast(self.poly2.data_ptr(), POINTER(c_double))
            if self.is_batch:
                self.out = torch.empty(
                    size=(self.batch_size, self.poly_len_),
                    dtype=torch.float64
                )
            else:
                self.out = torch.empty(
                    size=(self.poly_len_,),
                    dtype=torch.float64
                )
            self.out_ptr = cast(self.out.data_ptr(), POINTER(c_double))
        else:
            raise ValueError("path1, path2 must both be numpy arrays or both torch arrays")

def sig_combine(
        poly1 : Union[np.ndarray, torch.tensor],
        poly2 : Union[np.ndarray, torch.tensor],
        dimension : int,
        degree : int,
        parallel : bool = True
) -> Union[np.ndarray, torch.tensor]:
    """
    Combines two truncated signatures of the same degree and dimension into one signature. In particular, let :math:`x_1, x_2`
    be two paths such that the first point of :math:`x_2` is the last point of :math:`x_1`. Let :math:`S(x_1), S(x_2)`
    be the truncated signatures of :math:`x_1, x_2` respectively. Then calling this function on :math:`S(x_1), S(x_2)` returns
    the truncated signature of the concatenated path,

    .. math::

        S(x_1 * x_2) = S(x_1) \\otimes S(x_2),

    where :math:`x_1 * x_2` is the concatenation of the two paths :math:`x_1, x_2`.

    :param poly1: The first truncated tensor polynomial
    :type poly1: numpy.ndarray | torch.tensor
    :param poly2: The second truncated tensor polynomial. Must have the same degree and dimension as the first.
    :type poly2: numpy.ndarray | torch.tensor
    :param dimension: Dimension of the underlying space, :math:`d`.
    :type dimension: int
    :param degree: Truncation level of the tensor polynomial, :math:`N`
    :type degree: int
    :return: Tensor product of the truncated tensor polynomials
    :rtype: numpy.ndarray | torch.tensor

    .. note::

        The flag ``parallel`` defaults to ``True``. However, if the workload is too small, it
        may be beneficial to set this to ``False`` and run the computation serially,
        due to parallelisation overhead.

    .. note::

        Ideally, any array passed to ``pysiglib.sig_combine`` should be both contiguous and own its data.
        If this is not the case, ``pysiglib.sig_combine`` will internally create a contiguous copy, which may be
        inefficient.

    Example usage::

        import pysiglib

        batch_size = 32
        length = 100
        dimension = 5
        degree = 3

        X1 = np.random.uniform(size=(batch_size, length, dimension))
        X2 = np.random.uniform(size=(batch_size, length, dimension))
        X_concat = np.concatenate((X1, X2), axis=1)

        X2 = np.concatenate((X1[:, [-1], :], X2), axis=1) # Make sure first pt of X2 is last pt of X1
        sig1 = pysiglib.signature(X1, degree)
        sig2 = pysiglib.signature(X2, degree)

        # The tensor product...
        sig_mult = pysiglib.sig_combine(sig1, sig2, dimension, degree)

        # ... is the same as the signature of the concatenated path:
        sig = pysiglib.signature(X_concat, degree)
    """

    check_type(dimension, "dimension", int)
    check_type(degree, "degree", int)
    check_non_neg(dimension, "dimension")
    check_non_neg(degree, "degree")
    check_type(parallel, "parallel", bool)

    data = PolyDataHandler(poly1, poly2, dimension, degree)

    if data.is_batch:
        err_code = cpsig.batch_sig_combine(
            data.poly1_ptr,
            data.poly2_ptr,
            data.out_ptr,
            data.batch_size,
            dimension,
            degree,
            parallel
        )
    else:
        err_code = cpsig.sig_combine(
            data.poly1_ptr,
            data.poly2_ptr,
            data.out_ptr,
            dimension,
            degree
        )

    if err_code:
        raise Exception(err_msg[err_code] + " in signature")
    return data.out



class SigDataHandler:
    def __init__(self, path_, degree, time_aug, lead_lag):
        check_type_multiple(path_, "path",(np.ndarray, torch.Tensor))
        self.path = ensure_own_contiguous_storage(path_, 4)
        check_dtype(self.path, "path")
        check_type(degree, "degree", int)
        check_non_neg(degree, "degree")
        check_type(time_aug, "time_aug", bool)
        check_type(lead_lag, "lead_lag", bool)

        self.degree = degree
        self.time_aug = time_aug
        self.lead_lag = lead_lag

        self.get_dims(self.path)

        if isinstance(self.path, np.ndarray):
            self.init_numpy(self.path)
        elif isinstance(self.path, torch.Tensor):
            if not self.path.device.type == "cpu":
                raise ValueError("Data must be located on the cpu")
            self.init_torch(self.path)

    def init_numpy(self, path):
        self.dtype = str(path.dtype)
        self.data_ptr = path.ctypes.data_as(POINTER(DTYPES[self.dtype]))

        _, dimension_ = self.transformed_dims()
        if self.is_batch:
            self.out = np.empty(
                shape=(self.batch_size, sig_length(dimension_, self.degree)),
                dtype=np.float64
            )
        else:
            self.out = np.empty(
                shape=sig_length(dimension_, self.degree),
                dtype=np.float64
            )
        self.out_ptr = self.out.ctypes.data_as(POINTER(c_double))

    def init_torch(self, path):
        self.dtype = str(path.dtype)[6:]
        self.data_ptr = cast(path.data_ptr(), POINTER(DTYPES[self.dtype]))

        _, dimension_ = self.transformed_dims()
        if self.is_batch:
            self.out = torch.empty(
                size=(self.batch_size, sig_length(dimension_, self.degree)),
                dtype=torch.float64
            )
        else:
            self.out = torch.empty(
                size=(sig_length(dimension_, self.degree),),
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
            raise ValueError("path.shape must have length 2 or 3, got length " + str(len(path.shape)) + " instead.")

    def transformed_dims(self):
        length_ = self.length
        dimension_ = self.dimension
        if self.lead_lag:
            length_ *= 2
            length_ -= 3
            dimension_ *= 2
        if self.time_aug:
            dimension_ += 1
        return length_, dimension_


def signature_(data, time_aug = False, lead_lag = False, horner = True):
    err_code = CPSIG_SIGNATURE[data.dtype](
        data.data_ptr,
        data.out_ptr,
        data.dimension,
        data.length,
        data.degree,
        time_aug,
        lead_lag,
        horner
    )

    if err_code:
        raise Exception(err_msg[err_code] + " in signature")
    return data.out

def batch_signature_(data, time_aug = False, lead_lag = False, horner = True, parallel = True):
    err_code = CPSIG_BATCH_SIGNATURE[data.dtype](
        data.data_ptr,
        data.out_ptr,
        data.batch_size,
        data.dimension,
        data.length,
        data.degree,
        time_aug,
        lead_lag,
        horner,
        parallel
    )

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
    Computes the truncated signature of single path or a batch of paths. For
    a single path :math:`x`, the signature is given by

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

    .. note::

        Ideally, any array passed to ``pysiglib.signature`` should be both contiguous and own its data.
        If this is not the case, ``pysiglib.signature`` will internally create a contiguous copy, which may be
        inefficient.

    """
    check_type(horner, "horner", bool)

    data = SigDataHandler(path, degree, time_aug, lead_lag)
    if data.is_batch:
        check_type(parallel, "parallel", bool)
        return batch_signature_(data, time_aug, lead_lag, horner, parallel)
    return signature_(data, time_aug, lead_lag, horner)


class SigKernelDataHandler:
    def __init__(self, path1_, path2_, dyadic_order):
        check_type_multiple(path1_, "path1", (np.ndarray, torch.Tensor))
        check_type_multiple(path2_, "path2", (np.ndarray, torch.Tensor))
        self.path1 = ensure_own_contiguous_storage(path1_, 4)
        self.path2 = ensure_own_contiguous_storage(path2_, 4)
        check_dtype(self.path1, "path1")
        check_dtype(self.path2, "path2")

        if isinstance(dyadic_order, tuple) and len(dyadic_order) == 2:
            self.dyadic_order_1 = dyadic_order[0]
            self.dyadic_order_2 = dyadic_order[1]
        elif isinstance(dyadic_order, int):
            self.dyadic_order_1 = dyadic_order
            self.dyadic_order_2 = dyadic_order
        else:
            raise TypeError("dyadic_order must be an integer or a tuple of length 2")

        if self.dyadic_order_1 < 0 or self.dyadic_order_2 < 0:
            raise ValueError("dyadic_order must be a non-negative integer or tuple of non-negative integers")

        if len(self.path1.shape) == 2:
            self.is_batch = False
            self.batch_size = 1
            self.length_1 = self.path1.shape[0]
            self.dimension = self.path1.shape[1]
        elif len(self.path1.shape) == 3:
            self.is_batch = True
            self.batch_size = self.path1.shape[0]
            self.length_1 = self.path1.shape[1]
            self.dimension = self.path1.shape[2]
        else:
            raise ValueError("path1.shape must have length 2 or 3, got length " + str(len(self.path1.shape)) + " instead.")

        if len(self.path2.shape) == 2:
            if self.batch_size != 1:
                raise ValueError("path1, path2 have different batch sizes")
            self.length_2 = self.path2.shape[0]
            if self.dimension != self.path2.shape[1]:
                raise ValueError("path1, path2 have different dimensions")
        elif len(self.path2.shape) == 3:
            if self.batch_size != self.path2.shape[0]:
                raise ValueError("path1, path2 have different batch sizes")
            self.length_2 = self.path2.shape[1]
            if self.dimension != self.path2.shape[2]:
                raise ValueError("path1, path2 have different dimensions")
        else:
            raise ValueError("path2.shape must have length 2 or 3, got length " + str(len(self.path2.shape)) + " instead.")

        if isinstance(self.path1, np.ndarray) and isinstance(self.path2, np.ndarray):
            self.device = "cpu"
            self.out = np.empty(shape=self.batch_size, dtype=np.float64)
            self.out_ptr = self.out.ctypes.data_as(POINTER(c_double))

        elif isinstance(self.path1, torch.Tensor) and isinstance(self.path2, torch.Tensor) and self.path1.device == self.path2.device:
            self.device = self.path1.device.type
            self.out = torch.empty(self.batch_size, dtype=torch.float64, device = self.device)
            self.out_ptr = cast(self.out.data_ptr(), POINTER(c_double))
        else:
            raise ValueError("path1, path2 must both be numpy arrays or both torch arrays on the same device")

        self.torch_path1 = torch.as_tensor(self.path1)  # Avoids data copy
        self.torch_path2 = torch.as_tensor(self.path2)

        if self.is_batch:
            x1 = self.torch_path1[:, 1:, :] - self.torch_path1[:, :-1, :]
            y1 = self.torch_path2[:, 1:, :] - self.torch_path2[:, :-1, :]
        else:
            x1 = (self.torch_path1[1:, :] - self.torch_path1[:-1, :])[None, :, :]
            y1 = (self.torch_path2[1:, :] - self.torch_path2[:-1, :])[None, :, :]

        self.gram = torch.bmm(x1, y1.permute(0, 2, 1))

def sig_kernel_(data):

    err_code = cpsig.batch_sig_kernel(
        cast(data.gram.data_ptr(), POINTER(c_double)),
        data.out_ptr,
        data.batch_size,
        data.dimension,
        data.length_1,
        data.length_2,
        data.dyadic_order_1,
        data.dyadic_order_2
    )

    if err_code:
        raise Exception(err_msg[err_code] + " in sig_kernel")
    return data.out

def sig_kernel_cuda_(data):
    err_code = cusig.batch_sig_kernel_cuda(
        cast(data.gram.data_ptr(), POINTER(c_double)),
        data.out_ptr, data.batch_size,
        data.dimension,
        data.length_1,
        data.length_2,
        data.dyadic_order_1,
        data.dyadic_order_2
    )

    if err_code:
        raise Exception(err_msg[err_code] + " in sig_kernel")
    return data.out

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

    .. note::

        Ideally, any array passed to ``pysiglib.sig_kernel`` should be both contiguous and own its data.
        If this is not the case, ``pysiglib.sig_kernel`` will internally create a contiguous copy, which may be
        inefficient.
    """
    data = SigKernelDataHandler(path1, path2, dyadic_order)

    if data.device == "cpu":
        return sig_kernel_(data)

    if not BUILT_WITH_CUDA:
        raise RuntimeError("pySigLib was build without CUDA - data must be moved to CPU.")
    return sig_kernel_cuda_(data)
