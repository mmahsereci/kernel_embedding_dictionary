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

If you like you get your hands on some raw kernel embedding code for your own project, please feel
free to inspect e.g. 
[this](https://github.com/mmahsereci/kernel_embedding_dictionary/blob/main/kernel_embedding_dictionary/embeddings/mean_funcs.py) 
module where all univariate mean embeddings are listed. 

If you use our code in your project, please do not forget 
to cite our paper and to add the appropriate license. 

## Available Kernel embeddings

All multidimensional embeddings are based on product kernels and product measures.

| kernel / emdedding | `lebesgue` | `gaussian` |
|--------------------|:---------:|:---------:|
| `expquad`          |     x     |           |
| `matern12`         |           |           |
| `matern32`         |           |           |
| `matern52`         |           |           |

## Kernel configs

If an argument is not given, a default is used or inferred.


```python
# expquad
config_kernel = {
    "ndim": 2,
    "lengthscales": [1.0, 2.0],
}
```

## Measure configs

If an argument is not given, a default is used or inferred.


```python

# lebesgue
config_measure = {
    "ndim": 2,
    "bounds": [(0, 1), (1, 2)],
    "normalize": True
}

```


```python

# gaussian
config_measure = {
    "ndim": 2,
    "means": [-0.5, 2.8],
    "variances": [0.3, 1.2]
}

```
## Contributing

If you would like to contribute an additional kernel embedding or other enhancements, 
please feel free to open an issue or a pull request.
