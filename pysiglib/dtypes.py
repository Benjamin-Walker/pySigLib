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

from ctypes import c_float, c_double, c_int32, c_int64
import numpy as np
import torch
from .load_siglib import CPSIG

######################################################
# Some dicts to simplify dtype cases
######################################################

DTYPES = {
    "int32": c_int32,
    "int64": c_int64,
    "float32": c_float,
    "float64": c_double
}

SUPPORTED_DTYPES = [
    np.int32,
    np.int64,
    np.float32,
    np.float64,
    torch.int32,
    torch.int64,
    torch.float32,
    torch.float64
]

SUPPORTED_DTYPES_STR = "int32, int64, float or double"

CPSIG_TRANSFORM_PATH = {
    "int32": CPSIG.transform_path_int32,
    "int64": CPSIG.transform_path_int64,
    "float32": CPSIG.transform_path_float,
    "float64": CPSIG.transform_path_double
}

CPSIG_BATCH_TRANSFORM_PATH = {
    "int32": CPSIG.batch_transform_path_int32,
    "int64": CPSIG.batch_transform_path_int64,
    "float32": CPSIG.batch_transform_path_float,
    "float64": CPSIG.batch_transform_path_double
}

CPSIG_SIGNATURE = {
    "int32": CPSIG.signature_int32,
    "int64": CPSIG.signature_int64,
    "float32": CPSIG.signature_float,
    "float64": CPSIG.signature_double
}

CPSIG_BATCH_SIGNATURE = {
    "int32": CPSIG.batch_signature_int32,
    "int64": CPSIG.batch_signature_int64,
    "float32": CPSIG.batch_signature_float,
    "float64": CPSIG.batch_signature_double
}

CPSIG_SIG_BACKPROP = {
    "int32": CPSIG.sig_backprop_int32,
    "int64": CPSIG.sig_backprop_int64,
    "float32": CPSIG.sig_backprop_float,
    "float64": CPSIG.sig_backprop_double
}

CPSIG_BATCH_SIG_BACKPROP = {
    "int32": CPSIG.batch_sig_backprop_int32,
    "int64": CPSIG.batch_sig_backprop_int64,
    "float32": CPSIG.batch_sig_backprop_float,
    "float64": CPSIG.batch_sig_backprop_double
}
