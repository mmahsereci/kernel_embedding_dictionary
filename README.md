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
|--------------------|:----------:|:----------:|
| `expquad`          |     x      |     x      |
| `matern`           |     x      |            |
| `matern12`         |     x      |     x      |
| `matern32`         |     x      |     x      |
| `matern52`         |     x      |            |
| `matern72`         |     x      |            |

## Kernel configs

All kernels are product kernels of the form $\prod_{i=1}^d k(x_i, z_i)$ where $d$ is the 
dimensionality and $k$ is a univariate kernel.

If an argument is not given, a default is used or inferred. 
The available kernels configs are as follows.

`expaquad` kernel $k(x_i, z_i) = e^{-\frac{(x_i - z_i)^2}{2\ell_i^2}}$.

In the config below, `ndim` = $d$ and `lengthscales` = $[\ell_1, ...\ell_d]$.

```python
config_kernel = {
    "ndim": 2,
    "lengthscales": [1.0, 2.0],
}
```

`matern` kernel of order $\nu = n + 1/2$ for 
$n \in N_0$ with value 
$k(x_i, z_i) = \exp( -\sqrt{2n+1} r_i ) \frac{n!}{(2n)!} \sum_{k=0}^n \frac{(n+k)!}{k!(n-k)!} ( 2\sqrt{2n+1} \, r_i )^{n-k}$ 
where $r_i = \frac{|x_i - z_i|}{\ell_i}$.

In the config below, `nu` = $\nu$, where `ndim` = $d$ and `lengthscales` = $[\ell_1, ...\ell_d]$.

```python
config_kernel = {
    "ndim": 2,
    "nu": 3.5,
    "lengthscales": [1.0, 2.0],
}
```

`matern12` kernel $k(x_i, z_i) = e^{-r_i}$ where $r_i = \frac{|x_i - z_i|}{\ell_i}$.

In the config below, `ndim` = $d$ and `lengthscales` = $[\ell_1, ...\ell_d]$.

```python
config_kernel = {
    "ndim": 2,
    "lengthscales": [1.0, 2.0],
}
```

`matern32` kernel $k(x_i, z_i) = (1 + \sqrt{3} r_i)e^{-\sqrt{3} r_i}$ where $r_i = \frac{|x_i - z_i|}{\ell_i}$.

In the config below, `ndim` = $d$ and `lengthscales` = $[\ell_1, ...\ell_d]$.

```python
config_kernel = {
    "ndim": 2,
    "lengthscales": [1.0, 2.0],
}
```

`matern52` kernel $k(x_i, z_i) = (1 + \sqrt{5} r_i +\frac{5}{3} r_i^2)e^{-\sqrt{5} r_i}$ where $r_i = \frac{|x_i - z_i|}{\ell_i}$.

In the config below, `ndim` = $d$ and `lengthscales` = $[\ell_1, ...\ell_d]$.

```python
config_kernel = {
    "ndim": 2,
    "lengthscales": [1.0, 2.0],
}
```

`matern72` kernel $k(x_i, z_i) = (1 + \sqrt{7} r_i +\frac{14}{5} r_i^2 + \frac{7\sqrt{7}}{15})e^{-\sqrt{7} r_i}$ where $r_i = \frac{|x_i - z_i|}{\ell_i}$.

In the config below, `ndim` = $d$ and `lengthscales` = $[\ell_1, ...\ell_d]$.

```python
config_kernel = {
    "ndim": 2,
    "lengthscales": [1.0, 2.0],
}
```

## Measure configs

All measures are product measures of the form $\prod_{i=1}^d p(x_i, z_i)$ where $d$ is the 
dimensionality and $p$ is a univariate density.

If an argument is not given, a default is used or inferred. 
The available measure configs are as follows.

`lebesgue` measure with density $p(x_i) = (ub_i - lb_i)^{-1}$ (normalized) or 
$p(x_i) = 1$ (not normalized) when $lb_i\leq x_i\leq ub_i$.

In the config below, `ndim` = $d$ and `bounds` = $[(lb_1, ub_1), ... (lb_d, ub_d)]$

```python
config_measure = {
    "ndim": 2,
    "bounds": [(0, 1), (1, 2)],
    "normalize": True
}
```



`gaussian` measure with density $p(x_i) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x_i - \mu_i)^2}{2\sigma_i^2}}$.

In the config below, where `ndim` = $d$, `variances` = $[\sigma_1^2, ...\sigma_d^2]$ and `means` = $[\mu_1, ...\mu_d]$. 

```python
config_measure = {
    "ndim": 2,
    "means": [-0.5, 2.8],
    "variances": [0.3, 1.2]
}
```


## Contributing

If you would like to contribute an additional kernel embedding or other enhancements, 
please feel free to open an issue or a pull request.

### Adding a new product kernel

- Add the univariate kernel function to the file [`kernel_funcs_1d.py`](https://github.com/mmahsereci/kernel_embedding_dictionary/blob/main/kernel_embedding_dictionary/kernels/kernel_funcs_1d.py).
- Create a new module under `kernel_embedding_dictionary/kernels/` and implement the classes `UnivariateKernel` and `ProductKernel`. Use the existing kernels as example.
- Add the kernel to `kernel_embedding_dictionary/kernels/__init__.py`.

Add the kernels to the following tests

- [`tests/kernel_embedding_dictionary/kernels/test_kernels.py`](https://github.com/mmahsereci/kernel_embedding_dictionary/blob/main/tests/kernel_embedding_dictionary/kernels/test_kernels.py) as fixture and to the kernel list.
- [`tests/kernel_embedding_dictionary/kernels/test_kernels_uni.py`](https://github.com/mmahsereci/kernel_embedding_dictionary/blob/main/tests/kernel_embedding_dictionary/kernels/test_kernels_uni.py) as fixture and to the kernel list.
- create a new test module under `tests/kernel_embedding_dictionary/kernels/test_<new-kernel-name>_kernel.py` using the existing ones as example.

### Adding a new product measure

- Create a new module under `kernel_embedding_dictionary/measures/` and implement the classes `UnivariateMeasure` and `ProductMeasure`. Use the existing kernels as example.
- Add the measure to `kernel_embedding_dictionary/measures/__init__.py`.

Add the kernels to the following tests

- [`tests/kernel_embedding_dictionary/measures/test_measures.py`](https://github.com/mmahsereci/kernel_embedding_dictionary/blob/main/tests/kernel_embedding_dictionary/measures/test_measures.py) as fixture and to the measure list.
- create a new test module under `tests/kernel_embedding_dictionary/measures/test_<new-measure-name>_measure.py` using the existing ones as example.

### Adding a new kernel mean embedding

### Tests

Please make sure to run `isort` and then `black` on both the `kernel_embedding_dictionary` and `tests` directory. 
Pytest can be run locally (after install) with `pytest tests/`.
