# exmarg

Code and notebooks of the paper *Unifying stochastic and deterministic signals: a practical search for the intermittent
binary black-hole background* (T. Boybeyi, V. Mandic, J. D. Romano).

The likelihood ratio of a segment of data in two detectors, marginalized over the sky position, the orientation, the phase,
the arrival time and the distance of a merger, is written as a series of data factors $T^{(n)}_{ab}(\tau)$ (delayed cross
moments of the matched-filter outputs of the two detectors) times geometric factors $W^{(n)}_{ab}(\tau)$ (the generalized
overlap reduction functions, GORFs). The geometric factors do not depend on the data, so the series is summed once into
tables. Analyzing a segment is then Eq. (C10) of the paper: the matched filter of each template, and one read of the tables
at every arrival time of the segment and every delay between the detectors.

## Install

```
conda create -n exmarg python=3.10 numpy scipy healpy matplotlib numba lalsuite astropy jupyter
conda activate exmarg
pip install -e .              # the package in src/exmarg
pip install cupy-cuda12x      # only for the analysis on the GPU (device='gpu')
pip install bilby dynesty     # only to rerun the nested-sampling comparison (scripts/run_bilby.py)
pip install cogwheel-pe       # only to rerun the comparison with cogwheel (scripts/run_cogwheel.py; we used 1.6.0)
```

## Tables

One set of tables serves every analysis: for the LIGO Hanford and Livingston detectors, one table per signal power
$s = u y^2$ (53 nodes), with the distance $y = D_0/D_L$ ($D_0 = 10410$ Mpc) as a parameter and the series summed to
$n_{\max} = 120$. With the distance kept as a parameter the tables serve the analysis in source-frame mass and redshift;
for the analysis in $(M_{\rm det}, q, \chi)$ every template reads the nodes it reaches and the distance is integrated over its
prior in the code (`exmarg.search.pe_intrinsic`, prior `'volume'` or `'madau-dickinson'`).

The tables (40 GB) are on Zenodo, https://doi.org/10.5281/zenodo.23452946. Download and check them with

```
python scripts/fetch_tables.py 23578160     # into tables/redshift/, every file checked against data/tables/tables.sha256
```

On the GPU the tables are read into memory (about 0.75 GB per node, 40 GB in all). On the CPU they can be mapped from disk
instead (`search.load_redshift_store(..., mmap=True)`), so the analysis runs on a machine with less memory than the tables.
Every analysis takes `device='cpu'` or `device='gpu'`; the two agree to 1e-5 in ln B (notebook 5).

The tables are built in three stages by `scripts/build_tables/`, with the settings of the tables on Zenodo
(n_max = 120, HEALPix nside = 32, the 53 nodes of `data/tables/s_grid.npz`):

```
python scripts/build_tables/build_w.py <w_dir>                    # the GORFs at every node (GPU, 7 h for the 53 nodes)
python scripts/build_tables/build_k.py prepare <w_dir> <k_dir> <node>
python scripts/build_tables/build_k.py slab <w_dir> <k_dir> <node> <k> <K>   # every order, split over K tasks
python scripts/build_tables/build_k.py finalize <w_dir> <k_dir> <node>
python scripts/build_tables/build_bd.py <k_dir> tables/redshift   # the sum over the orders, the files on Zenodo
```

## Notebooks

Every notebook is saved with its outputs, so the figures can be seen without running anything. Every number and every
figure of a notebook comes from the code of this repository, except the `bilby` and `cogwheel` results, which come from
those packages with the settings in `scripts/`, and Fig. 18 (`data/results/loud_ladder.npz`), which was computed with our
earlier analysis code and nested sampling and has not yet been recomputed with `exmarg.search`.

| notebook | what it does | paper figures |
|---|---|---|
| `notebooks/00_quickstart.ipynb` | start here: simulates a segment with a merger and one with noise only, analyzes both on the CPU and draws the posterior of mass and redshift | |
| `notebooks/01_tables.ipynb` | builds the W tables, draws the GORFs in delay and in frequency | 3, 4, 5, 14, 15 |
| `notebooks/02_single_segments.ipynb` | analyzes single segments in three parameters (chirp mass, mass ratio, spin) and in two (source-frame mass, redshift), draws our posteriors next to `bilby`/`dynesty`, and the convergence in the order of the series (`scripts/compute_convergence.py`) | 16, 17, 18, 20, 21 |
| `notebooks/03_pp.ipynb` | probability-probability tests in both set-ups, with injections drawn from the prior of the analysis | 19 |
| `notebooks/04_population.ipynb` | the population results of one simulated year, from the result files on Zenodo | 6 to 13, 22 to 28 |
| `notebooks/05_speed.ipynb` | the same posteriors as `bilby` on 12 segments, then the time per segment against the SNR on the GPU and the CPU | |
| `notebooks/06_cogwheel.ipynb` | the comparison with `cogwheel`: its value against ours at finer and finer settings, and the cost of each | |

The figures are drawn by `notebooks/figures_*.py` in the style of the paper.

## Data

In the repository:

* `data/segments/`: the simulated segments of notebooks 2, 5 and 6 (`three_parameters.npz`, `two_parameters.npz`,
  `bilby_comparison.npz`).
* `data/banks/`: the template banks of notebooks 2 and 5.
* `data/results/`: the `bilby` posteriors, Bayes factors and run times, the `cogwheel` results of `scripts/run_cogwheel.py`,
  and the PP results of notebook 3.
* `data/psd/`: noise curves.

From Zenodo (https://doi.org/10.5281/zenodo.23452946):

* `tables/redshift/`: the tables, 40 GB (https://doi.org/10.5281/zenodo.23452946).
* `data/year/`: the results of the year analysis used in notebook 4 (posterior samples, Bayes factors of single segments,
  Bayes factors against the observing time, information of single segments).

## Scripts

```
python scripts/run_bilby.py three segment.npz          # the nested-sampling runs of Figs. 20 and 21 (settings)
python scripts/run_cogwheel.py <row> <M> <q> <chi>     # the cogwheel runs of notebook 6
python scripts/compute_convergence.py <tables at n_max = 1, 2, ...> gpu   # Figs. 16 and 17 (tables truncated at lower orders)
```

## Layout

```
src/exmarg/
  detectors.py   detector positions, antenna patterns, noise curves
  profiles.py    the waveform (IMRPhenomXAS) of a template
  data.py        simulated segments
  numerics.py    numerical helpers
  priors.py      the distance priors
  tables.py      the W tables (GORFs)
  geometry.py    the GORFs in frequency and the truncation deficit
  pp.py          probability-probability tests
  search/        the analysis of segments with the tables, on the CPU (numba) or the GPU (cupy):
                 segments.py, bank.py (matched filter), store.py (tables), gather.py (the table read),
                 pe.py (pe_redshift, pe_intrinsic)
notebooks/       the notebooks and their figure functions
scripts/         nested sampling and cogwheel
tests/           tests (pytest)
```
