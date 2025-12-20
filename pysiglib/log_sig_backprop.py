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
from .dtypes import CPSIG_SIG_TO_LOG_SIG_BACKPROP, CPSIG_BATCH_SIG_TO_LOG_SIG_BACKPROP
from .sig_length import sig_length, log_sig_length
from .sig import signature
from .data_handlers import SigOutputHandler, DeviceToHost, SigInputHandler
from .load_siglib import CPSIG


######################################################
# Python wrappers
######################################################

def sig_to_log_sig_backprop_(data, derivs_data, result, data_dimension, degree, time_aug, lead_lag, method):
    err_code = CPSIG_SIG_TO_LOG_SIG_BACKPROP[data.dtype](
        data.data_ptr,
        result.data_ptr,
        derivs_data.data_ptr,
        data_dimension,
        degree,
        time_aug,
        lead_lag,
        method
    )

    if err_code:
        raise Exception("Error in pysiglib.sig_to_log_sig_backprop: " + err_msg(err_code))
    return result.data

def batch_sig_to_log_sig_backprop_(data, derivs_data, result, data_dimension, degree, time_aug, lead_lag, method, n_jobs = 1):
    err_code = CPSIG_BATCH_SIG_TO_LOG_SIG_BACKPROP[data.dtype](
        data.data_ptr,
        result.data_ptr,
        derivs_data.data_ptr,
        data.batch_size,
        data_dimension,
        degree,
        time_aug,
        lead_lag,
        method,
        n_jobs
    )

    if err_code:
        raise Exception("Error in pysiglib.sig_to_log_sig_backprop: " + err_msg(err_code))
    return result.data

def sig_to_log_sig_backprop(
        sig : Union[np.ndarray, torch.tensor],
        log_sig_derivs : Union[np.ndarray, torch.tensor],
        dimension : int,
        degree : int,
        time_aug : bool = False,
        lead_lag : bool = False,
        method : str = 0,
        n_jobs : int = 1
) -> Union[np.ndarray, torch.tensor]:
    """#TODO
    """
    check_type(degree, "degree", int)
    check_type(method, "method", int)

    # If path is on GPU, move to CPU
    device_handler = DeviceToHost([sig, log_sig_derivs], ["sig", "log_sig_derivs"])
    sig, log_sig_derivs = device_handler.data

    sig_len = sig_length(dimension, degree)
    log_sig_len = log_sig_length(dimension, degree) if method else sig_length(dimension, degree)
    data = SigInputHandler(sig, sig_len, "sig")
    derivs_data = SigInputHandler(log_sig_derivs, log_sig_len, "log_sig_derivs")

    if data.dtype != derivs_data.dtype:
        raise ValueError("sig and log_sig_derivs must have the same dtype")

    result = SigOutputHandler(data, sig_len)
    if data.is_batch:
        check_type(n_jobs, "n_jobs", int)
        if n_jobs == 0:
            raise ValueError("n_jobs cannot be 0")
        res = batch_sig_to_log_sig_backprop_(data, derivs_data, result, dimension, degree, time_aug, lead_lag, method, n_jobs)
    else:
        res = sig_to_log_sig_backprop_(data, derivs_data, result, dimension, degree, time_aug, lead_lag, method)

    if device_handler.device is not None:
        res = res.to(device_handler.device)
    return res

# def log_sig_backprop_(
#         path : Union[np.ndarray, torch.tensor],
#         degree : int,
#         time_aug : bool = False,
#         lead_lag : bool = False,
#         end_time : float = 1.,
#         method : str = 0,
#         n_jobs : int = 1
# ) -> Union[np.ndarray, torch.tensor]:
#     """#TODO
#     """
#     sig_ = signature(path, degree, time_aug, lead_lag, end_time, True, n_jobs)
#     dimension = path.shape[-1]
#     log_sig_ = sig_to_log_sig(sig_, dimension, degree, time_aug, lead_lag, method, n_jobs)
#     return log_sig_