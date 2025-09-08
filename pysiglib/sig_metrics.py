from typing import Union
from ctypes import c_double, POINTER, cast

import numpy as np
import torch

from .transform_path import transform_path
from .load_siglib import CPSIG, CUSIG, BUILT_WITH_CUDA
from .param_checks import check_type, check_type_multiple
from .error_codes import err_msg
from .data_handlers import DoublePathInputHandler, ScalarOutputHandler, GridOutputHandler, PathInputHandler

from .sig_kernel import sig_kernel_gram

def sig_score(
        sample : Union[np.ndarray, torch.tensor],
        y : Union[np.ndarray, torch.tensor],
        dyadic_order : Union[int, tuple],
        lam : float = 1.,
        time_aug : bool = False,
        lead_lag : bool = False,
        end_time : float = 1.,
        n_jobs : int = 1,
        max_batch : int = -1
) -> Union[np.ndarray, torch.tensor]:
    """
    Computes the (generalised) signature kernel score

    .. math::

        \\phi_{\\text{sig}}(\\mu, y) := \\lambda \\mathbb{E}_{x,x' \\sim \\mu}[k(x,x')] - 2\\mathbb{E}_{x\\sim \\mu}[k(x,y)].

    Given a batch of sample paths :math:`\\{x_i\\}_{i=1}^m \\sim \\mu` and a path :math:`y`,
    the score is computed using the consistent and unbiased estimator

    .. math::

        \\widehat{\\phi}_{\\text{sig}}(\\mu, y) := \\frac{\\lambda }{m(m-1)} \\sum_{j \\neq i} k(x_i, x_j) - \\frac{2}{m} \\sum_i k(x_i, y).


    :param sample: The batch of sample paths drawn from :math:`\\mu`, given as a `numpy.ndarray` or
        `torch.tensor`. This must be of shape (batch size, length, dimension).
    :type sample: numpy.ndarray | torch.tensor
    :param y: The path(s) y, given as a `numpy.ndarray` or `torch.tensor`. For a single path,
        this must be of shape (length, dimension). For a batch of paths, this must be of shape
        (batch size, length, dimension).
    :type y: numpy.ndarray | torch.tensor
    :param dyadic_order: If set to a positive integer :math:`\\lambda`, will refine the
        paths by a factor of :math:`2^\\lambda`. If set to a tuple of positive integers
        :math:`(\\lambda_1, \\lambda_2)`, will refine the first path by :math:`2^{\\lambda_1}`
        and the second path by :math:`2^{\\lambda_2}`.
    :type dyadic_order: int | tuple
    :param lam: The parameter :math:`\\lambda` of the generalised signature kernel score (default = 1.0).
    :type lam: float
    :param time_aug: If set to True, will compute the signature of the time-augmented path, :math:`\\hat{x}_t := (t, x_t)`,
        defined as the original path with an extra channel set to time, :math:`t`. This channel spans :math:`[0, t_L]`,
        where :math`t_L` is given by the parameter ``end_time``.
    :type time_aug: bool
    :param lead_lag: If set to True, will compute the signature of the path after applying the lead-lag transformation.
    :type lead_lag: bool
    :param end_time: End time for time-augmentation, :math:`t_L`.
    :type end_time: float
    :param n_jobs: (Only applicable to CPU computation) Number of threads to run in parallel.
        If n_jobs = 1, the computation is run serially. If set to -1, all available threads
        are used. For n_jobs below -1, (max_threads + 1 + n_jobs) threads are used. For example
        if n_jobs = -2, all threads but one are used.
    :type n_jobs: int
    :param max_batch: Maximum batch size to run in parallel. If the computation is failing
        due to insufficient memory, this parameter should be decreased.
        If set to -1, the entire batch is computed in parallel.
    :type max_batch: int
    :return: Signature kernel score
    :rtype: numpy.ndarray | torch.tensor

    .. note::

        Ideally, any array passed to ``pysiglib.sig_score`` should be both contiguous and own its data.
        If this is not the case, ``pysiglib.sig_score`` will internally create a contiguous copy, which may be
        inefficient.
    """

    check_type_multiple(y, "y", (np.ndarray, torch.Tensor))
    if len(y.shape) not in (2,3):
        raise ValueError("Must have len(y.shape) in (2,3)")
    if len(y.shape) == 2:
        y = y[None, :, :]

    data = DoublePathInputHandler(sample, y, time_aug, lead_lag, end_time, "sample_paths", "y", True, False)

    # Use torch for simplicity
    sample = torch.as_tensor(data.path1)
    y = torch.as_tensor(data.path2)

    B = sample.shape[0]

    xx = sig_kernel_gram(sample, sample, dyadic_order, time_aug, lead_lag, end_time, n_jobs, max_batch, False)
    xy = sig_kernel_gram(sample, y, dyadic_order, time_aug, lead_lag, end_time, n_jobs, max_batch, False)

    xx_sum = float((torch.sum(xx) - torch.sum(torch.diag(xx))) / (B * (B - 1)))
    xy_sum = torch.sum(xy, dim = 0) * (2. / B)

    res = lam * torch.full(xy_sum.shape, xx_sum, device = xy_sum.device) - xy_sum

    return res

def expected_sig_score(
        sample1 : Union[np.ndarray, torch.tensor],
        sample2 : Union[np.ndarray, torch.tensor],
        dyadic_order : Union[int, tuple],
        lam : float = 1.,
        time_aug : bool = False,
        lead_lag : bool = False,
        end_time : float = 1.,
        n_jobs : int = 1,
        max_batch : int = -1
) -> Union[np.ndarray, torch.tensor]:
    """
    Computes the expected (generalised) signature kernel score

    .. math::

        \\mathbb{E}_{y \\sim \\nu}[\\phi_{\\text{sig}}(\\mu, y)] := \\lambda \\mathbb{E}_{x,x' \\sim \\mu}[k(x,x')] - 2\\mathbb{E}_{x,y\\sim \\mu \\times \\nu}[k(x,y)].

    Given a batch of sample paths :math:`\\{x_i\\}_{i=1}^m \\sim \\mu` and :math:`\\{y_j\\}_{j=1}^n \\sim \\nu`,
    the score is computed using the unbiased estimator

    .. math::

        \\frac{\\lambda }{m(m-1)} \\sum_{j \\neq i} k(x_i, x_j) - \\frac{2}{mn} \\sum_{i,j} k(x_i, y_j).


    :param sample1: The batch of sample paths drawn from :math:`\\mu`, given as a `numpy.ndarray` or
        `torch.tensor`. This must be of shape (batch size, length, dimension).
    :type sample1: numpy.ndarray | torch.tensor
    :param sample2: The batch of sample paths drawn from :math:`\\nu`, given as a `numpy.ndarray` or
        `torch.tensor`. This must be of shape (batch size, length, dimension).
    :type sample2: numpy.ndarray | torch.tensor
    :param dyadic_order: If set to a positive integer :math:`\\lambda`, will refine the
        paths by a factor of :math:`2^\\lambda`. If set to a tuple of positive integers
        :math:`(\\lambda_1, \\lambda_2)`, will refine the first path by :math:`2^{\\lambda_1}`
        and the second path by :math:`2^{\\lambda_2}`.
    :type dyadic_order: int | tuple
    :param lam: The parameter :math:`\\lambda` of the generalised signature kernel score (default = 1.0).
    :type lam: float
    :param time_aug: If set to True, will compute the signature of the time-augmented path, :math:`\\hat{x}_t := (t, x_t)`,
        defined as the original path with an extra channel set to time, :math:`t`. This channel spans :math:`[0, t_L]`,
        where :math`t_L` is given by the parameter ``end_time``.
    :type time_aug: bool
    :param lead_lag: If set to True, will compute the signature of the path after applying the lead-lag transformation.
    :type lead_lag: bool
    :param end_time: End time for time-augmentation, :math:`t_L`.
    :type end_time: float
    :param n_jobs: (Only applicable to CPU computation) Number of threads to run in parallel.
        If n_jobs = 1, the computation is run serially. If set to -1, all available threads
        are used. For n_jobs below -1, (max_threads + 1 + n_jobs) threads are used. For example
        if n_jobs = -2, all threads but one are used.
    :type n_jobs: int
    :param max_batch: Maximum batch size to run in parallel. If the computation is failing
        due to insufficient memory, this parameter should be decreased.
        If set to -1, the entire batch is computed in parallel.
    :type max_batch: int
    :return: Expected signature kernel score
    :rtype: float

    .. note::

        Ideally, any array passed to ``pysiglib.expected_sig_score`` should be both contiguous and own its data.
        If this is not the case, ``pysiglib.expected_sig_score`` will internally create a contiguous copy, which may be
        inefficient.
    """

    return float(sig_score(sample1, sample2, dyadic_order, lam, time_aug, lead_lag, end_time, n_jobs, max_batch).mean())

def sig_mmd(
        sample1 : Union[np.ndarray, torch.tensor],
        sample2 : Union[np.ndarray, torch.tensor],
        dyadic_order : Union[int, tuple],
        time_aug : bool = False,
        lead_lag : bool = False,
        end_time : float = 1.,
        n_jobs : int = 1,
        max_batch : int = -1
) -> Union[np.ndarray, torch.tensor]:
    """
    Computes the squared maximum mean discrepancy (MMD)

    .. math::

        d(\\mu, \\nu)^2 := \\sup_f(\\mathbb{E}_{x \\sim \\mu}[f(x)] - \\mathbb{E}_{y \\sim \\nu}[f(y)]).

    .. math::

        = \\mathbb{E}_{xx' \\sim \\mu}[k(x,x')] - 2\\mathbb{E}_{x,y \\sim \\mu \\times \\nu}[k(x,y)] + \\mathbb{E}_{y,y' \\sim \\nu}[k(y,y')].

    Given a batch of sample paths :math:`\\{x_i\\}_{i=1}^m \\sim \\mu` and :math:`\\{y_j\\}_{j=1}^n \\sim \\nu`,
    the MMD is computed using the unbiased estimator

    .. math::

        \\widehat{d}(\\mu, \\nu)^2 = \\frac{1}{m(m-1)}\\sum_{j \\neq i}k(x_i, x_j) - \\frac{2}{mn}\\sum_{i,j}k(x_i, x_j) + \\frac{1}{n(n-1)}\\sum_{j \\neq i} k(y_i, y_j).


    :param sample1: The batch of sample paths drawn from :math:`\\mu`, given as a `numpy.ndarray` or
        `torch.tensor`. This must be of shape (batch size, length, dimension).
    :type sample1: numpy.ndarray | torch.tensor
    :param sample2: The batch of sample paths drawn from :math:`\\nu`, given as a `numpy.ndarray` or
        `torch.tensor`. This must be of shape (batch size, length, dimension).
    :type sample2: numpy.ndarray | torch.tensor
    :param dyadic_order: If set to a positive integer :math:`\\lambda`, will refine the
        paths by a factor of :math:`2^\\lambda`. If set to a tuple of positive integers
        :math:`(\\lambda_1, \\lambda_2)`, will refine the first path by :math:`2^{\\lambda_1}`
        and the second path by :math:`2^{\\lambda_2}`.
    :type dyadic_order: int | tuple
    :param time_aug: If set to True, will compute the signature of the time-augmented path, :math:`\\hat{x}_t := (t, x_t)`,
        defined as the original path with an extra channel set to time, :math:`t`. This channel spans :math:`[0, t_L]`,
        where :math`t_L` is given by the parameter ``end_time``.
    :type time_aug: bool
    :param lead_lag: If set to True, will compute the signature of the path after applying the lead-lag transformation.
    :type lead_lag: bool
    :param end_time: End time for time-augmentation, :math:`t_L`.
    :type end_time: float
    :param n_jobs: (Only applicable to CPU computation) Number of threads to run in parallel.
        If n_jobs = 1, the computation is run serially. If set to -1, all available threads
        are used. For n_jobs below -1, (max_threads + 1 + n_jobs) threads are used. For example
        if n_jobs = -2, all threads but one are used.
    :type n_jobs: int
    :param max_batch: Maximum batch size to run in parallel. If the computation is failing
        due to insufficient memory, this parameter should be decreased.
        If set to -1, the entire batch is computed in parallel.
    :type max_batch: int
    :return: Signature MMD
    :rtype: float

    .. note::

        Ideally, any array passed to ``pysiglib.sig_mmd`` should be both contiguous and own its data.
        If this is not the case, ``pysiglib.sig_mmd`` will internally create a contiguous copy, which may be
        inefficient.
    """
    data = DoublePathInputHandler(sample1, sample2, time_aug, lead_lag, end_time, "sample1", "sample2", True, False)

    # Use torch for simplicity
    sample1 = torch.as_tensor(data.path1)
    sample2 = torch.as_tensor(data.path2)

    m = sample1.shape[0]
    n = sample2.shape[0]

    xx = sig_kernel_gram(sample1, sample1, dyadic_order, time_aug, lead_lag, end_time, n_jobs, max_batch, False)
    xy = sig_kernel_gram(sample1, sample2, dyadic_order, time_aug, lead_lag, end_time, n_jobs, max_batch, False)
    yy = sig_kernel_gram(sample2, sample2, dyadic_order, time_aug, lead_lag, end_time, n_jobs, max_batch, False)

    xx_sum = (torch.sum(xx) - torch.sum(torch.diag(xx))) / (m * (m - 1))
    xy_sum = 2. * torch.mean(xy)
    yy_sum = (torch.sum(yy) - torch.sum(torch.diag(yy))) / (n * (n - 1))

    return float(xx_sum - xy_sum + yy_sum)
