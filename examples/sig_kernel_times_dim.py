from tqdm import tqdm
from timing_utils import timepysiglib_kernel, timesigkernel, plot_times

import plotting_params
plotting_params.set_plotting_params(8, 10, 12)

if __name__ == '__main__':

    dyadic_order = 0
    batch_size = 32
    length = 1000
    N = 10
    device = "cuda"

    dim_arr = list(range(10, 1100, 100))
    sigkerneltime = []
    pysiglibtime = []

    for dimension in tqdm(dim_arr):
        sigkerneltime.append(timesigkernel(batch_size, length, dimension, dyadic_order, device, N))
        pysiglibtime.append(timepysiglib_kernel(batch_size, length, dimension, dyadic_order, device, N))

    print(sigkerneltime)
    print(pysiglibtime)

    for scale in ["linear", "log"]:
        plot_times(
                x= dim_arr,
                ys= [sigkerneltime, pysiglibtime],
                legend = ["sigkernel", "pysiglib"],
                title = "Signature Kernels " + device,
                xlabel = "Path Dimension",
                ylabel = "Elapsed Time (s)",
                scale = scale,
                filename = "sigkernel_times_dim_" + scale + "_" + device
        )
