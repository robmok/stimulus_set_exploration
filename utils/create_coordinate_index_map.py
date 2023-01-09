# -*- coding: utf-8 -*-
# Copyright 2023 Brett D. Roads. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Utility module."""
import numpy as np


def create_coordinate_index_map(fp_source_data=None):
    """Create coordinate to idx map."""
    # stimulus_set_name = fp_source_data.parts[-2]
    gen_space_vals = np.linspace(0.0, 1.0, 5, endpoint=True)

    d = {}
    counter = 1  # Start at 1 to accomodate zero masking.
    for x in gen_space_vals:
        for y in gen_space_vals:
            d['{0:.2f},{1:.2f}'.format(x, y)] = counter
            counter += 1

    return d
