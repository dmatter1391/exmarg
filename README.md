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
conda create -n exmarg python=3.10 numpy scipy healpy matplotlib numba lalsuite jupyter
conda activate exmarg
pip install -e .              # the package in src/exmarg
pip install bilby dynesty     # only for the nested-sampling comparison of notebook 2
```

## Notebooks

Every notebook is saved with its outputs, so the figures can be seen without running anything.

| notebook | what it does | paper figures |
|---|---|---|
| `notebooks/01_tables.ipynb` | builds the W tables, saves them, draws the GORFs in delay and in frequency | 3, 4, 5, 14, 15 |
| `notebooks/02_one_segment.ipynb` | simulates a segment with a merger, computes the posterior of the chirp mass, mass ratio and spin, runs `bilby`/`dynesty` on the same segment and compares, convergence with the truncation order | 16, 17, 18, 20, 21 |
| `notebooks/03_pp.ipynb` | probability-probability test for any prior box of the intrinsic parameters | 19 |
| `notebooks/04_population.ipynb` | the population results of one simulated year | 6 to 13, 22 to 28 |

The figures are drawn by `notebooks/figures_*.py` in the style of the paper.

## Data

* `data/results/`: the results behind Figs. 16 to 21 (nested-sampling runs, convergence, PP tests).
* `data/year/`: the results of the year analysis used in notebook 4 (posterior samples, Bayes factors of single segments,
  Bayes factors against the observing time, information of single segments). To be downloaded from Zenodo (link to follow).
* `tables/`: the summed tables of notebooks 2 and 3, one file pair per filter power (`sumstore_n<n_max>_u<u>.lnS.npy` and
  `.meta.npz`, about 17 GB for $n_{\max} = 120$). To be downloaded from Zenodo (link to follow).
* `data/psd/`: noise curves.

## Building the tables

`notebooks/01_tables.ipynb` builds low-order tables in seconds. The tables of the analysis (orders up to 120) are built with
the same function by the scripts

```
python scripts/build_store.py 120 32 64        # W tables: n_max, nside, workers
python scripts/build_kstore.py ...             # one kernel per order
python scripts/derive_sumstores.py 120 ...     # the summed tables
```

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
  evaluate.py    the likelihood ratio of a segment for any template
  geometry.py    the GORFs in frequency and the truncation deficit
  pp.py          probability-probability test
  compare.py     the same segment with bilby/dynesty
notebooks/       the notebooks and their figure functions
scripts/         construction of the production tables
tests/           tests (pytest)
```
