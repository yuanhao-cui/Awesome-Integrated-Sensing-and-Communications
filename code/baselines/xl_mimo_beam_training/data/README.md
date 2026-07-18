# Data Preparation

## Optional local files

Place the following `.mat` files in this directory (or specify path via `data_path` in config):

- `pcsi.mat` — Perfect Channel State Information (CSI)
  - Variable name: `pcsi`
  - Shape: `(num_samples, N_t)` complex-valued
- `ecsi.mat` — Estimated CSI (with channel estimation errors)
  - Variable name: `ecsi`
  - Shape: `(num_samples, N_t)` complex-valued

## Data Format

Each row represents one channel realization between the XL-MIMO base station
and a user. A caller supplying these files is responsible for documenting the
channel model, normalization, split, provenance, and usage rights; the file
names alone do not establish that they are paper data.

## Synthetic Data

The default runnable path generates seeded synthetic near-field channels using
`generate_synthetic_data()` in `src/utils.py`. Those samples are repository
test data, not measurements or paper-result data.

## Public-source boundary

The first author's public repository is
[fly-winder/near-field-beamforming-using-deeplearning](https://github.com/fly-winder/near-field-beamforming-using-deeplearning).
At the audited revision it did not contain `pcsi.mat`, `ecsi.mat`, a training
dataset directory, or MATLAB data-generation scripts, and it did not include a
standard open-source license. Its training-script hyperparameters also differ
from the paper's Table III. It is therefore linked for provenance only; this
repository neither copies its artifacts nor labels locally generated samples
as author data.

The paper does not disclose the exact channel realizations used for its figures.
Supplying private `.mat` files creates a separate experiment and requires a new
provenance and comparison record before any paper-parity claim.
