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

from tqdm import tqdm

from timing_utils import plot_times, timepysiglib_sig_backprop, timeiisig_sig_backprop, timesignatory_sig_backprop

import plotting_params
plotting_params.set_plotting_params(8, 10, 12)

# pip install signatory==1.2.6.1.9.0 --no-cache-dir --force-reinstall

import pysiglib
assert pysiglib.BUILT_WITH_AVX

if __name__ == '__main__':

    batch_size = 32
    length = 1024
    dimension = 5
    N = 5
    device = "cpu"

    degree_arr = list(range(1, 8))
    iisigtime = []
    pysiglibtime = []
    signatorytime = []

    for degree in tqdm(degree_arr):
        iisigtime.append(timeiisig_sig_backprop(batch_size, length, dimension, degree, 1,device, N))
        pysiglibtime.append(timepysiglib_sig_backprop(batch_size, length, dimension, degree, -1, device, N))
        signatorytime.append(timesignatory_sig_backprop(batch_size, length, dimension, degree, 1, device, N))

    print(iisigtime)
    print(pysiglibtime)
    print(signatorytime)

    for scale in ["linear", "log"]:
        plot_times(
            x= degree_arr,
            ys= [iisigtime, signatorytime, pysiglibtime],
            legend = ["iisignature", "signatory", "pySigLib"],
            title = "Sig Backprop",
            xlabel = "Truncation Level",
            ylabel = "Elapsed Time (s)",
            scale = scale,
            filename = "sig_backprop_times_" + scale + "_serial"
        )
