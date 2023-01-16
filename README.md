Author: Brett D. Roads

---
**WARNING:** This package is pre-release and the API is not stable. All APIs are subject to change and all releases are alpha.

---

# Purpose 

To explore the properties of candidate stimulus sets.

# Usage

The repository is not a full-fledged package. You will need to ensure your python environment has all the packages listed in 
`requirements.txt`.

# Experimental design and data.

The stimuli sets consist of abstract shapes.

Data source: TODO

# Project structure

Use standard embedding structure.
* `data_adapter.py` Adapt the raw data files into a model-consumable `tf.data.Dataset` object.
* Analysis scripts
    * `phase0_fit.py` Infer (and save) embedding models.
        * e.g., `python phase0_fit.py --arch_id 0 --input_id 0`
    * `phase0_plot.py` Visualize embeddings.
        * e.g., `python phase0_plot.py --arch_id 2 --input_id 0 --run_id 1`

## Model identifiers

* Model names follow `emb_{arch_id}-{input_id}-{n_dim}-{run_id}`
* Architecture IDs `arch_id`
    * See `/assets/arch_id.csv`
* Input IDs `input_id`
    * See `/assets/input_id.csv`
* Dimensionality `n_dim`
    * Only using 1D space (i.e., n_dim=1)
* Run identifier `run_id`
    * Indicates restart run. Used three restarts for all architectures.

## Results

### arch_id=0, input_id=0
TODO
Using n_restart=3, the best validation loss is 0.29 (run_id=2).
