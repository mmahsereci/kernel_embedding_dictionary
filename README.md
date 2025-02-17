# Kernel embedding dictionary

A dictionary of kernel embeddings, accessible in Python.

## Installation from source

Download the source code. 
Activate your preferred virtual env. 
In the root directory, run the following command.

```commandline
pip3 install .
```

## Usage

Once the library is installed, import it like so.

```python
import numpy as np

from kernel_embedding_dictionary import get_embedding

ke = get_embedding("expquad", "lebesgue")

x = np.random.rand(3, 1)  # evaluate at 3 points
kernel_means = ke.mean(x)
```

Provide parameters to the measure and/or kernel

```python
import numpy as np

from kernel_embedding_dictionary import get_embedding

config_measure = {
    "ndim": 2,
    "bounds": [(0, 1), (0.5, 2.5)],
    "normalize": True
}

config_kernel = {
    "ndim": 2,
    "lengthscales": [1.0, 0.5],
}

ke = get_embedding("expquad", "lebesgue", config_kernel, config_measure)
x = np.random.rand(3, 2)  # evaluate at 3 points
kernel_means = ke.mean(x)
```

Inspect the embedding with the print command

```commandline
print(ke)
```

If you would like to get your hands on some raw kernel embedding code for your own project, please feel
free to inspect e.g. 
[this](https://github.com/mmahsereci/kernel_embedding_dictionary/blob/main/kernel_embedding_dictionary/embeddings/mean_funcs_1d.py) 
module where all univariate mean embeddings are listed. 

If you use our code in your project, please do not forget 
to cite our paper and to add the appropriate license. 

## Available Kernel embeddings

All multidimensional embeddings are based on product kernels and product measures.

| kernel / emdedding | `lebesgue` | `gaussian` |
|--------------------|:---------:|:----------:|
| `expquad`          |     x     |     x      |
| `matern12`         |           |            |
| `matern32`         |           |            |
| `matern52`         |           |            |

## Kernel configs

All kernels are product kernels of the form $\prod_{i=1}^d k(x_i, z_i)$ where $d$ is the 
dimensionality and $k$ is a univariate kernel.

If an argument is not given, a default is used or inferred. 
The available kernels configs are as follows.

`expaquad` with value $k(x_i, z_i') = e^{-\frac{(x_i - z_i)^2}{2\ell_i^2}}$.

```python
config_kernel = {
    "ndim": 2,
    "lengthscales": [1.0, 2.0],
}
```

where `ndim` = $d$ and `lengthscales` = $[\ell_1, ...\ell_d]$. 

## Measure configs

All measures are product measures of the form $\prod_{i=1}^d p(x_i, z_i)$ where $d$ is the 
dimensionality and $p$ is a univariate density.

If an argument is not given, a default is used or inferred. 
The available measure configs are as follows.

`lebesgue` with density $p(x_i) = (ub_i - lb_i)^{-1}$ (normalized) or 
$p(x_i) = 1$ (not normalized) when $lb_i\leq x_i\leq ub_i$

```python
config_measure = {
    "ndim": 2,
    "bounds": [(0, 1), (1, 2)],
    "normalize": True
}
```

where `ndim` = $d$ and `bounds` = $[(lb_1, ub_1), ... (lb_d, ub_d)]$


`gaussian` with density $p(x_i) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x_i - \mu_i)^2}{2\sigma_i^2}}$.

```python
config_measure = {
    "ndim": 2,
    "means": [-0.5, 2.8],
    "variances": [0.3, 1.2]
}
```

where `ndim` = $d$, `variances` = $[\sigma_1^2, ...\sigma_d^2]$ and `means` = $[\mu_1, ...\mu_d]$. 

## Contributing

If you would like to contribute an additional kernel embedding or other enhancements, 
please feel free to open an issue or a pull request.
