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

from .load_siglib import CPSIG
from .param_checks import check_type, check_non_neg

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

    out = CPSIG.sig_length(dimension, degree)
    if out == 0:
        raise ValueError("Integer overflow encountered in sig_length")
    return out