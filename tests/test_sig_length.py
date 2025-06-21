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

import numpy as np
import torch

import pysiglib

np.random.seed(42)
torch.manual_seed(42)
EPSILON = 1e-10

def test_poly_length():
    assert pysiglib.poly_length(0, 0) == 1
    assert pysiglib.poly_length(0, 1) == 1
    assert pysiglib.poly_length(1, 0) == 1
    assert pysiglib.poly_length(9, 9) == 435848050
    assert pysiglib.poly_length(10, 10) == 11111111111
    assert pysiglib.poly_length(11, 11) == 313842837672
    assert pysiglib.poly_length(400, 5) == 10265664160401