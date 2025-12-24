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

from timing_utils import time_sigkernel_kernel_backprop, time_pysiglib_kernel_backprop

if __name__ == '__main__':
    cfg = {
        'batch_size': 128,
        'length': 100,
        'dimension': 32,
        'dyadic_order': 0,
        'dtype': "double",
        'device': 'cpu',
        'num_runs': 5
    }

    print("\nsigkernel (CPU): ", time_sigkernel_kernel_backprop(cfg, True))
    print("\npysiglib (CPU): ", time_pysiglib_kernel_backprop(cfg, -1, True))

    cfg['device'] = 'cuda'
    print("\nsigkernel (CUDA): ", time_sigkernel_kernel_backprop(cfg, True))
    print("\npysiglib (CUDA): ", time_pysiglib_kernel_backprop(cfg, -1, True))
