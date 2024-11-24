"""
(C) Copyright 2021 IBM Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Created on June 30, 2021

"""

# from collections.abc import Hashable
# from typing import Dict, List, Optional, Union

import numpy as np
from anndata import AnnData
from fuse.data import OpBase
from fuse.utils.ndict import NDict


class OpReadAnnData(OpBase):
    """
    Op reading data from anndata.
    Each row will be added as a value to sample dict.
    """

    def __init__(
        self,
        data: AnnData | None = None,
        key_name: str = "data.sample_id",
        label_column: str = "label",
    ):
        """
        :param data:  input AnnData object
        :param key_name: name of value in sample_dict which will be used as the key/index
        """
        super().__init__()

        self._key_name = key_name
        self._data = data
        self.label_column = label_column
        self.gene_names = np.array(self._data.var_names)

    def __call__(
        self, sample_dict: NDict, prefix: str | None = None
    ) -> None | dict | list[dict]:
        """
        See base class

        :param prefix: specify a prefix for the sample dict keys.
                       For example, with prefix 'data.features' and a df with the columns ['height', 'weight', 'sex'],
                       the matching keys will be: 'data.features.height', 'data.features.weight', 'data.features.sex'.
        """

        key = sample_dict[self._key_name]

        # locate the required item
        sample_dict[f"{prefix}.scrna"] = self._data[key, :].X
        sample_dict[f"data.{prefix}.label"] = self._data.obs.iloc[key][
            self.label_column
        ]
        sample_dict[f"{prefix}.gene_names"] = self.gene_names

        return sample_dict
