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

from ctypes import c_double, POINTER, cast

import numpy as np
import torch

from .param_checks import check_type, check_type_multiple, check_dtype, check_dtype_double, ensure_own_contiguous_storage
from .dtypes import DTYPES

class SigInputHandler:
    """
    Handle input which is (shaped like) a signature or a batch of signatures
    """
    def __init__(self, sig_, sig_len, param_name, stacklevel = 4):

        check_type(sig_len, "sig_len", int)
        self.sig_len = sig_len
        self.param_name = param_name

        check_type_multiple(sig_, param_name, (np.ndarray, torch.Tensor))
        self.sig = ensure_own_contiguous_storage(sig_, stacklevel)
        check_dtype_double(self.sig, param_name)

        if self.sig.shape[-1] != self.sig_len:
            raise ValueError(self.param_name + " is of incorrect length. Expected " + str(self.sig_len) + ", got " + str(self.sig.shape[-1]))

        if len(self.sig.shape) == 1:
            self.is_batch = False
            self.batch_size = 1
            self.length = self.sig.shape[0]
        elif len(self.sig.shape) == 2:
            self.is_batch = True
            self.batch_size = self.sig.shape[0]
            self.length = self.sig.shape[1]
        else:
            raise ValueError(self.param_name + ".shape must have length 1 or 2, got length " + str(len(self.sig.shape)) + " instead.")

        if isinstance(self.sig, np.ndarray):
            self.type_ = "numpy"
            self.data_ptr = self.sig.ctypes.data_as(POINTER(c_double))
        elif isinstance(self.sig, torch.Tensor):
            self.type_ = "torch"
            if not self.sig.device.type == "cpu":
                raise ValueError(self.param_name + " must be located on the cpu")
            self.data_ptr = cast(self.sig.data_ptr(), POINTER(c_double))
        else:
            raise ValueError(self.param_name + " must be a numpy array or a torch array")

class DoubleSigInputHandler:
    """
    Handle a pair of inputs which are (shaped like) signatures or batches of signatures
    """
    def __init__(self, sig1, sig2, sig_len, sig1_name, sig2_name):

        check_type(sig_len, "sig_len", int)
        self.sig_len = sig_len

        self.data1 = SigInputHandler(sig1, sig_len, sig1_name, 5)
        self.data2 = SigInputHandler(sig2, sig_len, sig2_name, 5)

        if self.data1.batch_size != self.data2.batch_size:
            raise ValueError(sig1_name + ", " + sig2_name + " have different batch sizes")

        if self.data1.type_ != self.data2.type_:
            raise ValueError(sig1_name + ", " + sig2_name + " must both be numpy arrays or both torch arrays")

        self.batch_size = self.data1.batch_size
        self.is_batch = self.data1.is_batch
        self.type_ = self.data1.type_
        self.sig1_ptr = self.data1.data_ptr
        self.sig2_ptr = self.data2.data_ptr

        if self.type_ == "torch" and not (sig1.device.type == "cpu" and sig2.device.type == "cpu"):
            raise ValueError(sig1_name + ", " + sig2_name + " must be located on the cpu")

class TripleSigInputHandler:
    """
    Handle a triple of inputs which are (shaped like) signatures or batches of signatures
    """
    def __init__(self, sig1, sig2, sig3, sig_len, sig1_name, sig2_name, sig3_name):

        check_type(sig_len, "sig_len", int)
        self.sig_len = sig_len

        self.data1 = SigInputHandler(sig1, sig_len, sig1_name, 5)
        self.data2 = SigInputHandler(sig2, sig_len, sig2_name, 5)
        self.data3 = SigInputHandler(sig3, sig_len, sig3_name, 5)

        if self.data1.batch_size != self.data2.batch_size or self.data2.batch_size != self.data3.batch_size:
            raise ValueError(sig1_name + ", " + sig2_name + ", " + sig3_name + " have different batch sizes")

        if self.data1.type_ != self.data2.type_ or self.data2.type_ != self.data3.type_:
            raise ValueError(sig1_name + ", " + sig2_name + ", " + sig3_name + " must both be numpy arrays or both torch arrays")

        self.batch_size = self.data1.batch_size
        self.is_batch = self.data1.is_batch
        self.type_ = self.data1.type_
        self.sig1_ptr = self.data1.data_ptr
        self.sig2_ptr = self.data2.data_ptr
        self.sig3_ptr = self.data3.data_ptr

        if self.type_ == "torch" and (sig1.device.type != "cpu" or sig2.device.type != "cpu" or sig3.device.type != "cpu"):
            raise ValueError(sig1_name + ", " + sig2_name + ", " + sig3_name + " must be located on the cpu")

class SigOutputHandler:
    """
    Handle output which is (shaped like) a signature or a batch of signatures
    """
    def __init__(self, data, sig_len):

        self.sig_len = sig_len
        self.batch_size = data.batch_size
        self.is_batch = data.is_batch
        self.type_ = data.type_

        self.result_length = self.batch_size * self.sig_len

        if self.type_ == "numpy":
            if self.is_batch:
                self.data = np.empty(
                    shape=(self.batch_size, self.sig_len),
                    dtype=np.float64
                )
            else:
                self.data = np.empty(
                    shape=self.sig_len,
                    dtype=np.float64
                )
            self.data_ptr = self.data.ctypes.data_as(POINTER(c_double))

        else:
            if self.is_batch:
                self.data = torch.empty(
                    size=(self.batch_size, self.sig_len),
                    dtype=torch.float64
                )
            else:
                self.data = torch.empty(
                    size=(self.sig_len,),
                    dtype=torch.float64
                )
            self.data_ptr = cast(self.data.data_ptr(), POINTER(c_double))

class PathInputHandler:
    """
    Handle input which is (shaped like) a path or a batch of paths
    """
    def __init__(self, path_, time_aug, lead_lag, param_name):
        self.param_name = param_name
        check_type_multiple(path_, param_name,(np.ndarray, torch.Tensor))
        self.path = ensure_own_contiguous_storage(path_, 4)
        check_dtype(self.path, param_name)
        check_type(time_aug, "time_aug", bool)
        check_type(lead_lag, "lead_lag", bool)

        self.time_aug = time_aug
        self.lead_lag = lead_lag

        self.get_dims(self.path)

        if isinstance(self.path, np.ndarray):
            self.type_ = "numpy"
            self.dtype = str(self.path.dtype)
            self.data_ptr = self.path.ctypes.data_as(POINTER(DTYPES[self.dtype]))
        elif isinstance(self.path, torch.Tensor):
            self.type_ = "torch"
            self.dtype = str(self.path.dtype)[6:]
            self.data_ptr = cast(self.path.data_ptr(), POINTER(DTYPES[self.dtype]))

        self.length, self.dimension = self.transformed_dims()
        self.device = self.path.device.type if self.type_ == "torch" else "cpu"

    def get_dims(self, path):
        if len(path.shape) == 2:
            self.is_batch = False
            self.batch_size = 1
            self.data_length = path.shape[0]
            self.data_dimension = path.shape[1]


        elif len(path.shape) == 3:
            self.is_batch = True
            self.batch_size = path.shape[0]
            self.data_length = path.shape[1]
            self.data_dimension = path.shape[2]

        else:
            raise ValueError(self.param_name + ".shape must have length 2 or 3, got length " + str(len(path.shape)) + " instead.")

    def transformed_dims(self):
        length_ = self.data_length
        dimension_ = self.data_dimension
        if self.lead_lag:
            length_ *= 2
            length_ -= 1
            dimension_ *= 2
        if self.time_aug:
            dimension_ += 1
        return length_, dimension_

class DoublePathInputHandler:
    """
    Handle a pair of inputs which are (shaped like) paths or a batch of paths
    """
    def __init__(self, path1_, path2_, time_aug, lead_lag, path1_name = "path1", path2_name = "path2"):

        self.data1 = PathInputHandler(path1_, time_aug, lead_lag, path1_name)
        self.data2 = PathInputHandler(path2_, time_aug, lead_lag, path2_name)
        self.path1 = self.data1.path
        self.path2 = self.data2.path

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
            raise ValueError(path1_name + ".shape must have length 2 or 3, got length " + str(len(self.path1.shape)) + " instead.")

        if len(self.path2.shape) == 2:
            if self.batch_size != 1:
                raise ValueError(path1_name + ", " + path2_name + " have different batch sizes")
            self.length_2 = self.path2.shape[0]
            if self.dimension != self.path2.shape[1]:
                raise ValueError(path1_name + ", " + path2_name + " have different dimensions")
        elif len(self.path2.shape) == 3:
            if self.batch_size != self.path2.shape[0]:
                raise ValueError(path1_name + ", " + path2_name + " have different batch sizes")
            self.length_2 = self.path2.shape[1]
            if self.dimension != self.path2.shape[2]:
                raise ValueError(path1_name + ", " + path2_name + " have different dimensions")
        else:
            raise ValueError(path2_name + ".shape must have length 2 or 3, got length " + str(len(self.path2.shape)) + " instead.")

        if self.data1.type_ != self.data2.type_:
            raise ValueError(path1_name + ", " + path2_name + " must both be numpy arrays or both torch arrays")

        if self.data1.type_ == "torch" and self.path1.device != self.path2.device:
            raise ValueError(path1_name + ", " + path2_name + " must both be on the same device")

        self.type_ = self.data1.type_
        self.device = self.path1.device.type if self.type_ == "torch" else "cpu"

class ScalarOutputHandler:
    """
    Handle output which is (shaped like) a scalar or a batch of scalars
    """
    def __init__(self, data):
        if data.type_ == "numpy":
            self.device = "cpu"
            self.data = np.empty(shape=data.batch_size, dtype=np.float64)
            self.data_ptr = self.data.ctypes.data_as(POINTER(c_double))

        else:
            self.device = data.path1.device.type
            self.data = torch.empty(data.batch_size, dtype=torch.float64, device = self.device)
            self.data_ptr = cast(self.data.data_ptr(), POINTER(c_double))


class PathOutputHandler:
    """
    Handle output which is (shaped like) a path or a batch of paths
    """
    def __init__(self, length, dimension, batch_size, is_batch, type_, device):

        self.length = length
        self.dimension = dimension
        self.batch_size = batch_size
        self.is_batch = is_batch
        self.type_ = type_

        if self.type_ == "numpy":
            self.device = "cpu"
            if self.is_batch:
                self.data = np.empty(
                    shape=(self.batch_size, self.length, self.dimension),
                    dtype=np.float64
                )
            else:
                self.data = np.empty(
                    shape=(self.length, self.dimension),
                    dtype=np.float64
                )
            self.data_ptr = self.data.ctypes.data_as(POINTER(c_double))

        else:
            self.device = device
            if self.is_batch:
                self.data = torch.empty(
                    size=(self.batch_size, self.length, self.dimension),
                    dtype=torch.float64,
                    device = self.device
                )
            else:
                self.data = torch.empty(
                    size=(self.length, self.dimension),
                    dtype=torch.float64,
                    device = self.device
                )
            self.data_ptr = cast(self.data.data_ptr(), POINTER(c_double))
