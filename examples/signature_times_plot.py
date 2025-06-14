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

from timing_utils import timeiisig, timepysiglib, timesignatory, plot_times

import plotting_params
plotting_params.set_plotting_params(8, 10, 12)

# pip install signatory==1.2.6.1.9.0 --no-cache-dir --force-reinstall

if __name__ == '__main__':

    batch_size = 32
    length = 1024
    dimension = 5
    N = 5
    device = "cpu"

    degree_arr = list(range(1, 8))
    iisigtime = []
    signatorytime = []
    pysiglibtime = []
    pysiglibtimeparallel = []
    pysiglibtimehorner = []
    pysiglibtimehornerparallel = []

    for degree in tqdm(degree_arr):
        iisigtime.append(timeiisig(batch_size, length, dimension, degree, device, N))
        signatorytime.append(timesignatory(batch_size, length, dimension, degree, device, N))
        pysiglibtime.append(timepysiglib(batch_size, length, dimension, degree, False, False, device, N))
        pysiglibtimeparallel.append(timepysiglib(batch_size, length, dimension, degree, False, True, device, N))
        pysiglibtimehorner.append(timepysiglib(batch_size, length, dimension, degree, True, False, device, N))
        pysiglibtimehornerparallel.append(timepysiglib(batch_size, length, dimension, degree, True, True, device, N))

    print(iisigtime)
    print(signatorytime)
    print(pysiglibtime)
    print(pysiglibtimeparallel)
    print(pysiglibtimehorner)
    print(pysiglibtimehornerparallel)

    for scale in ["linear", "log"]:
        plot_times(
            x= degree_arr,
            ys= [iisigtime, pysiglibtime, pysiglibtimehorner],
            legend = ["iisignature (Direct)", "pySigLib (Direct)", "pySigLib (Horner)"],
            title = "Truncated Signatures (Serial)",
            xlabel = "Truncation Level",
            ylabel = "Elapsed Time (s)",
            scale = scale,
            filename = "signature_times_" + scale + "_serial"
        )

        plot_times(
            x=degree_arr,
            ys=[signatorytime, pysiglibtimeparallel, pysiglibtimehornerparallel],
            legend=["signatory (Horner)", "pySigLib (Direct)", "pySigLib (Horner)"],
            title= "Truncated Signatures (Parallel)",
            xlabel="Truncation Level",
            ylabel="Elapsed Time (s)",
            scale=scale,
            filename="signature_times_" + scale + "_parallel"
        )
