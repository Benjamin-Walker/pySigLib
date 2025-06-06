<p align="center">
<img src="https://raw.githubusercontent.com/daniil-shmelev/pySigLib/master/docs/_static/logo.png" width="350"/>
</p>

[//]: # (<h1 align='center'>sigLib</h1>)
<h2 align='center'>Signature Computations on CPU and GPU</h2>

## Installation

```
pip install pysiglib
```

pySigLib will automatically detect CUDA, provided the `CUDA_PATH` environment variable is set correctly.
To manually disable CUDA and build pySigLib for CPU only, create an environment variable `CUSIG` and set
it to `0`:

```
set CUSIG=0
pip install pysiglib
```

## Documentation

TBC

## Examples

### Signatures

pySigLib implements truncated signature transforms through the function `pysiglib.signature`,
which takes as input a path or batch of paths `X`, and a truncation `degree`.</p>

`X` can be a numpy array or a torch tensor. For a single path, `X` must be of shape
`(path length, path dimension)`. For a batch of paths, it must be of shape
`(batch size, path length, path dimension)`.</p>

The computation will run on whichever device `X` is on. For example, passing
`X = np.random.uniform(size=(32, 1000, 10))` will trigger the computation to run
on the CPU, whilst `X = torch.rand(32, 1000, 10, device = "cuda")` will run
on the CUDA device.

```python
import pysiglib
import numpy as np

X = np.random.uniform(size=(32, 1000, 10))
sig = pysiglib.signature(X, degree = 5)
```

### Signature Kernels

pySigLib implements signature kernels through the function `pysiglib.sig_kernel`,
which takes as input a pair of paths or a pair of batches of paths `X, Y`. The
`dyadic_order` parameter can be used to refine the PDE grid, giving more
accurate results. If specified as an integer, the same refinement factor
is applied to both `X` and `Y`. To apply different factors to the two paths,
`dyadic_order` can be specified as a tuple.</p>

As with signatures, `X,Y` can be numpy arrays or torch tensors, and must have the
same shapes as described above. Again, the computation will run on whichever
device `X,Y` are located on.

```python
import pysiglib
import numpy as np

X = np.random.uniform(size=(32, 1000, 10))
Y = np.random.uniform(size=(32, 1000, 10))
sig = pysiglib.sig_kernel(X, Y, dyadic_order=1)

# In cases where the paths differ in length, it may
# be advantageous to refine the PDE grid by different
# amounts for X and Y:

X = np.random.uniform(size=(32, 100, 10))
Y = np.random.uniform(size=(32, 5000, 10))
sig = pysiglib.sig_kernel(X, Y, dyadic_order=(3, 0))
```

## Citation

TBC