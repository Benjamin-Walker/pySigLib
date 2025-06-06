from tqdm import tqdm
from timing_utils import timesigkernel, timepysiglib_kernel, plot_times

import plotting_params
plotting_params.set_plotting_params(8, 10, 12)

if __name__ == '__main__':

    dyadic_order = 0
    batch_size = 120
    dimension = 5
    N = 10
    device = "cpu"

    length_arr = list(range(10, 2100, 100))
    sigkerneltime = []
    pysiglibtime = []


    for length in tqdm(length_arr):
        sigkerneltime.append(timesigkernel(batch_size, length, dimension, dyadic_order, device, N))
        pysiglibtime.append(timepysiglib_kernel(batch_size, length, dimension, dyadic_order, device, N))

    print(sigkerneltime)
    print(pysiglibtime)

    for scale in ["linear", "log"]:
        plot_times(
                x=length_arr[:9],
                ys= [sigkerneltime[:9], pysiglibtime[:9]],
                legend = ["sigkernel", "pysiglib"],
                title = "Signature Kernels " + device,
                xlabel = "Path Length",
                ylabel = "Elapsed Time (s)",
                scale = scale,
                filename = "sigkernel_times_len_" + scale + "_" + device + "_1"
        )
        plot_times(
                x= length_arr,
                ys= [sigkerneltime, pysiglibtime],
                legend = ["sigkernel", "pysiglib"],
                title = "Signature Kernels " + device,
                xlabel = "Path Length",
                ylabel = "Elapsed Time (s)",
                scale = scale,
                filename = "sigkernel_times_len_" + scale + "_" + device + "_2"
        )
