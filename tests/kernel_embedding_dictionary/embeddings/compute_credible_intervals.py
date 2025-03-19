# Copyright 2025 The KED Authors. All Rights Reserved.
# SPDX-License-Identifier: MIT


import numpy as np
from test_mean_values_expquad import (
    get_config_expquad_gaussian_1d_standard,
    get_config_expquad_gaussian_1d_values,
    get_config_expquad_gaussian_2d_standard,
    get_config_expquad_gaussian_2d_values,
    get_config_expquad_lebesgue_1d_standard,
    get_config_expquad_lebesgue_1d_values,
    get_config_expquad_lebesgue_2d_standard,
    get_config_expquad_lebesgue_2d_values,
)
from test_mean_values_matern12 import (
    get_config_matern12_lebesgue_1d_standard,
    get_config_matern12_lebesgue_1d_values,
    get_config_matern12_lebesgue_2d_standard,
    get_config_matern12_lebesgue_2d_values,
)
from test_mean_values_matern32 import (
    get_config_matern32_lebesgue_1d_standard,
    get_config_matern32_lebesgue_1d_values,
    get_config_matern32_lebesgue_2d_standard,
    get_config_matern32_lebesgue_2d_values,
)
from test_mean_values_matern52 import (
    get_config_matern52_lebesgue_1d_standard,
    get_config_matern52_lebesgue_1d_values,
    get_config_matern52_lebesgue_2d_standard,
    get_config_matern52_lebesgue_2d_values,
)

from kernel_embedding_dictionary._get_embedding import get_embedding

embedding_to_config_list_dict = {
    "expquad-lebesgue": [
        get_config_expquad_lebesgue_1d_standard,
        get_config_expquad_lebesgue_1d_values,
        get_config_expquad_lebesgue_2d_standard,
        get_config_expquad_lebesgue_2d_values,
    ],
    "expquad-gaussian": [
        get_config_expquad_gaussian_1d_standard,
        get_config_expquad_gaussian_1d_values,
        get_config_expquad_gaussian_2d_standard,
        get_config_expquad_gaussian_2d_values,
    ],
    "matern12-lebesgue": [
        get_config_matern12_lebesgue_1d_standard,
        get_config_matern12_lebesgue_1d_values,
        get_config_matern12_lebesgue_2d_standard,
        get_config_matern12_lebesgue_2d_values,
    ],
    "matern32-lebesgue": [
        get_config_matern32_lebesgue_1d_standard,
        get_config_matern32_lebesgue_1d_values,
        get_config_matern32_lebesgue_2d_standard,
        get_config_matern32_lebesgue_2d_values,
    ],
    "matern52-lebesgue": [
        get_config_matern52_lebesgue_1d_standard,
        get_config_matern52_lebesgue_1d_values,
        get_config_matern52_lebesgue_2d_standard,
        get_config_matern52_lebesgue_2d_values,
    ],
}

if __name__ == "__main__":
    np.random.seed(0)

    # size of credible interval in units of std
    ff = 3

    # Choose embedding
    # embedding_name = "expquad-lebesgue"
    # embedding_name = "expquad-gaussian"
    # embedding_name = "matern12-lebesgue"
    # embedding_name = "matern32-lebesgue"
    embedding_name = "matern52-lebesgue"

    get_config_func_list = embedding_to_config_list_dict[embedding_name]

    for get_config_func in get_config_func_list:

        print(f"\n=== {embedding_name} =======================================")
        print(get_config_func)
        kernel_name, measure_name, ck, cm, x_eval = get_config_func()
        ke = get_embedding(kernel_name, measure_name, ck, cm)
        print(ke)
        print()
        print(x_eval)
        print()

        if ke.ndim == 1:
            n = int(1e5)
        else:
            n = int(1e5)

        x_sample = ke._measure.sample(n)
        print(x_sample.shape)
        print(x_eval.shape)
        y_sample = ke._kernel.evaluate(x_eval, x_sample)
        print(y_sample.shape)
        y_mean = y_sample.mean(axis=1)
        y_std = y_sample.std(axis=1) / np.sqrt(n)
        print(y_mean.shape)
        print(y_std.shape)

        for y_mean_i, y_std_i in zip(y_mean, y_std):
            # print(y_mean_i)
            # print(y_std_i)
            print([y_mean_i - ff * y_std_i, y_mean_i + ff * y_std_i])
