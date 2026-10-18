# exmarg

Code and notebooks of the paper *Unifying stochastic and deterministic signals: a practical search for the intermittent
binary black-hole background* (T. Boybeyi, V. Mandic, J. D. Romano).

The likelihood ratio of a segment of data in two detectors, marginalized over the sky position, the orientation, the phase,
the arrival time and the distance of a merger, is written as a series of data factors $T^{(n)}_{ab}(\tau)$ (delayed cross
moments of the matched-filter outputs of the two detectors) times geometric factors $W^{(n)}_{ab}(\tau)$ (the generalized
overlap reduction functions, GORFs). The geometric factors do not depend on the data, so they are computed once and stored.
Analyzing a segment is then a table lookup for each template, at any truncation order.

## Install

```
conda create -n exmarg python=3.10 numpy scipy healpy matplotlib numba lalsuite astropy jupyter
conda activate exmarg
pip install -e .              # the package in src/exmarg
pip install cupy-cuda12x      # the GPU analysis of notebooks 2 and 5
pip install bilby dynesty     # only to rerun the nested-sampling comparison (scripts/run_bilby.py)
```

## Notebooks

Every notebook is saved with its outputs, so the figures can be seen without running anything.

| notebook | what it does | paper figures |
|---|---|---|
| `notebooks/01_tables.ipynb` | builds the W tables, saves them, draws the GORFs in delay and in frequency | 3, 4, 5, 14, 15 |
| `notebooks/02_single_segments.ipynb` | analyzes single segments on the GPU in three parameters (chirp mass, mass ratio, spin) and in two (source-frame mass, redshift), and draws our posteriors next to `bilby`/`dynesty`; convergence with the truncation order | 16, 17, 18, 20, 21 |
| `notebooks/03_pp.ipynb` | probability-probability test for any prior box of the intrinsic parameters | 19 |
| `notebooks/04_population.ipynb` | the population results of one simulated year | 6 to 13, 22 to 28 |
| `notebooks/05_speed.ipynb` | the GPU analysis of 843 segments up to SNR 15 in batches, time per segment against SNR, compared with nested sampling | |

The figures are drawn by `notebooks/figures_*.py` in the style of the paper.

## Data

In the repository:

* `data/segments/`: the simulated segments of notebook 2 (`three_parameters.npz`, `two_parameters.npz`).
* `data/banks/`: the template banks of notebook 2.
* `data/results/`: the results behind Figs. 16 to 21 (nested-sampling posteriors and run times, convergence, PP tests)
  and the GPU benchmark of notebook 5.
* `data/psd/`: noise curves.

From Zenodo (link to follow):

* `tables/sumstore_n120_u*`: the summed tables of notebooks 2 and 3, one file pair per filter power, about 17 GB.
* `tables/intrinsic/`: the three-parameter tables (`ugrid.npz`, `marg_u*.npz`), 49 GB in two records; put the files of
  both records in this one folder.
* `tables/redshift/`: the two-parameter tables (`bd_*.npz`), 40 GB.
* `data/segments/benchmark.npz`: the 843 segments of notebook 5.
* `data/year/`: the results of the year analysis used in notebook 4 (posterior samples, Bayes factors of single segments,
  Bayes factors against the observing time, information of single segments).
* the likelihood ratios of every segment of the simulated year on the grid of detector-frame total mass and distance, for
  population studies of your own (five parts).

## Scripts

```
python scripts/build_store.py 120 32 64        # W tables: n_max, nside, workers
python scripts/build_kstore.py ...             # one kernel per order
python scripts/derive_sumstores.py 120 ...     # the summed tables
python scripts/run_bilby.py three segment.npz  # the nested-sampling runs of Figs. 20 and 21 (settings only, not run here)
python scripts/benchmark_gpu.py ...            # the benchmark of notebook 5
```

`notebooks/01_tables.ipynb` builds low-order tables in seconds with the same functions.

## Layout

```
src/exmarg/
  detectors.py   detector positions, antenna patterns, noise curves
  profiles.py    the waveform (IMRPhenomXAS) of a template
  data.py        simulated segments and the matched-filter outputs
  numerics.py    numerical helpers
  priors.py      the distance prior of the tables
  tables.py      the W tables
  kernel.py      lookup tables compiled from W
  evaluate.py    the likelihood ratio of a segment for any template (CPU)
  geometry.py    the GORFs in frequency and the truncation deficit
  pp.py          probability-probability test
  gpu/           the analysis of many segments on the GPU (bank, tables, kernel, parameter estimation)
notebooks/       the notebooks and their figure functions
scripts/         construction of the tables, nested sampling, benchmark
tests/           tests (pytest)
```
