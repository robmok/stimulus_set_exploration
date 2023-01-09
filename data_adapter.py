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
"""Data adapter script."""

from pathlib import Path

import numpy as np
import pandas as pd
import psiz.data

from utils.create_coordinate_index_map import create_coordinate_index_map


def main(fp_project):
    """Run script."""
    # Settings.
    fp_source = fp_project / Path('assets', 'data', 'source')
    fp_ds = fp_project / Path('assets', 'data', 'datasets')
    source_data = {
        0: fp_source / Path(
            'stimulus_set_13b', 'sub-00_norming_stimset_13b_sess0-5.pkl'
        ),
        1: fp_source / Path(
            'stimulus_set_13b', 'sub-00_norming_stimset_13b_sess1-5.pkl'
        ),
        2: fp_source / Path(
            'stimulus_set_20c', 'sub-00_norming_stimset_20c_sess0-5.pkl'
        ),
        3: fp_source / Path(
            'stimulus_set_20c', 'sub-00_norming_stimset_20c_sess1-5.pkl'
        ),
        4: fp_source / Path(
            'stimulus_set_09', 'sub-00_norming_stimset_9_sess0-5.pkl'
        ),
        5: fp_source / Path(
            'stimulus_set_09', 'sub-00_norming_stimset_9_sess1-5.pkl'
        ),
    }

    for input_id, fp_source_data in source_data.items():
        ds_name = 'ds_{}'.format(input_id)
        ds = data_adapter(fp_source_data)
        # print('Total trials retained: {0}'.format(obs.n_trial))
        ds.save(str(fp_ds / Path(ds_name)))


def data_adapter(fp_source_data):
    """Parse data for phase 0."""
    # Settings.
    max_n_reference = 2
    n_select_hardcoded = 1

    # Read in pickle data.
    df_source = pd.read_pickle(fp_source_data)
    # Grab columns we need.
    df_source = df_source[
        [
            'Trial',
            'Block',
            'Condition',
            'Response',
            'RT',
        ]
    ]

    # Create name to idx map.
    coordinate_2_idx = create_coordinate_index_map(fp_source_data)

    def as_idx(arr):
        x = arr[0]
        y = arr[1]
        return coordinate_2_idx['{0:.2f},{1:.2f}'.format(x, y)]

    response_2_idx = {
        'left': 0,
        'right': 1,
    }

    # Pre-allocate.
    stimulus_set = []
    agent_id = []
    outcome_idx = []
    rt_ms = []
    for index, row_trial in df_source.iterrows():
        # Content.
        q = as_idx(row_trial['Condition'][0])
        r_left = as_idx(row_trial['Condition'][1])
        r_right = as_idx(row_trial['Condition'][2])
        stimulus_set.append([q, r_left, r_right])

        # Behavior.
        outcome_idx.append(response_2_idx[row_trial['Response']])
        rt_ms.append(row_trial['RT'] * 1000)

    stimulus_set = np.array(stimulus_set)
    outcome_idx = np.array(outcome_idx)
    outcome_idx = np.expand_dims(outcome_idx, axis=1)

    # Create dataset.
    content = psiz.data.Rank(stimulus_set, n_select=n_select_hardcoded)
    # agent_id = psiz.data.Group(agent_id, name='agent_id')
    outcome = psiz.data.SparseCategorical(outcome_idx, depth=2)
    ds = psiz.data.Dataset([content, outcome])

    return ds.export(export_format='tfds')


if __name__ == "__main__":
    fp_project = Path.home() / Path('projects', 'rob_mok', 'exploration')
    main(fp_project)
