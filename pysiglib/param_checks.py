import numpy as np
import torch

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

def get_type_str(type):
    try:
        if hasattr(type, '__name__'):
            return type.__name__
        else:
            return str(type)
    except:
        return "UNKNOWN"

def check_type(param, param_name, type_):
    if not isinstance(param, type_):
        raise TypeError(param_name + " must be of type " + get_type_str(type_) + ", got " + get_type_str(type(param)) + " instead")

def check_type_multiple(param, param_name, type_tuple):
    if not isinstance(param, type_tuple):
        type_tuple_str = ' or '.join(get_type_str(type_) for type_ in type_tuple)
        raise TypeError(param_name + " must be of type " + type_tuple_str + ", got " + get_type_str(type(param)) + " instead")

def check_non_neg(param, param_name):
    if param < 0:
        raise ValueError(param_name + " must be a non-negative integer, got " + param_name + " = " + str(param))

def check_dtype(arr, arr_name):
    if arr.dtype not in SUPPORTED_DTYPES:
        raise TypeError(arr_name + ".dtype must be " + SUPPORTED_DTYPES_STR + ", got " + str(arr.dtype) + " instead")