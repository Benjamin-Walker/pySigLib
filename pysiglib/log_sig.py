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

from typing import Union

import numpy as np
import torch

from .param_checks import check_type, check_non_neg, log_sig_method_parser
from .error_codes import err_msg
from .dtypes import CPSIG_LOG_SIGNATURE, CPSIG_BATCH_LOG_SIGNATURE
from .sig_length import sig_length, log_sig_length
from .data_handlers import PathInputHandler, DoubleSigInputHandler, SigOutputHandler, DeviceToHost
from .load_siglib import CPSIG


######################################################
# Python wrappers
######################################################

def prepare_log_sig(
        dimension : int,
        degree : int,
        method : int,
        time_aug : bool = False,
        lead_lag : bool = False

) -> Union[np.ndarray, torch.tensor]:
    """#TODO
    """
    check_type(dimension, "dimension", int)
    check_type(degree, "degree", int)
    check_type(method, "method", int)
    check_type(time_aug, "time_aug", bool)
    check_type(lead_lag, "lead_lag", bool)

    if method == 0:
        return

    aug_dimension = (2 * dimension if lead_lag else dimension) + (1 if time_aug else 0)

    err_code = CPSIG.prepare_log_sig(
        aug_dimension,
        degree,
        method
    )

    if err_code:
        raise Exception("Error in pysiglib.prepare_log_sig: " + err_msg(err_code))

def reset_log_sig() -> Union[np.ndarray, torch.tensor]:
    """#TODO
    """
    err_code = CPSIG.reset_log_sig()

def log_signature_(data, result, degree, method):
    err_code = CPSIG_LOG_SIGNATURE[data.dtype](
        data.data_ptr,
        result.data_ptr,
        data.data_dimension,
        data.data_length,
        degree,
        data.time_aug,
        data.lead_lag,
        data.end_time,
        method
    )

    if err_code:
        raise Exception("Error in pysiglib.log_sig: " + err_msg(err_code))
    return result.data

def batch_log_signature_(data, result, degree, method, n_jobs = 1):
    err_code = CPSIG_BATCH_LOG_SIGNATURE[data.dtype](
        data.data_ptr,
        result.data_ptr,
        data.batch_size,
        data.data_dimension,
        data.data_length,
        degree,
        data.time_aug,
        data.lead_lag,
        data.end_time,
        method,
        n_jobs
    )

    if err_code:
        raise Exception("Error in pysiglib.log_sig: " + err_msg(err_code))
    return result.data

def log_sig(
        path : Union[np.ndarray, torch.tensor],
        degree : int,
        time_aug : bool = False,
        lead_lag : bool = False,
        end_time : float = 1.,
        method : str = 0,
        n_jobs : int = 1
) -> Union[np.ndarray, torch.tensor]:
    """#TODO
    """
    check_type(degree, "degree", int)
    check_type(method, "method", int)
    #method = log_sig_method_parser(method)

    # If path is on GPU, move to CPU
    device_handler = DeviceToHost([path], ["path"])
    path = device_handler.data[0]

    data = PathInputHandler(path, time_aug, lead_lag, end_time, "path")
    sig_len = log_sig_length(data.dimension, degree) if method else sig_length(data.dimension, degree)
    result = SigOutputHandler(data, sig_len)
    if data.is_batch:
        check_type(n_jobs, "n_jobs", int)
        if n_jobs == 0:
            raise ValueError("n_jobs cannot be 0")
        res = batch_log_signature_(data, result, degree, method, n_jobs)
    else:
        res = log_signature_(data, result, degree, method)

    if device_handler.device is not None:
        res = res.to(device_handler.device)
    return res

