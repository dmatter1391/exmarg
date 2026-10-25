# exmarg

Code and notebooks of the paper *Unifying stochastic and deterministic signals: a practical search for the intermittent
binary black-hole background* (T. Boybeyi, V. Mandic, J. D. Romano).

The likelihood ratio of a segment of data in two detectors, marginalized over the sky position, the orientation, the phase,
the arrival time and the distance of a merger, is written as a series of data factors $T^{(n)}_{ab}(\tau)$, the delayed cross
moments of the matched-filter outputs of the two detectors, times geometric factors $W^{(n)}_{ab}(\tau)$, the generalized
overlap reduction functions (GORFs). The geometric factors do not depend on the data, so the series is summed once into
tables, and analyzing a segment is then Eq. (C7) of the paper, the matched filter of each template followed by reads of the
tables over the arrival times of the segment and the delays between the detectors.

## Install

```
conda create -n exmarg python=3.10 numpy scipy healpy matplotlib numba lalsuite astropy jupyter
conda activate exmarg
pip install -e .              # the package in src/exmarg
pip install -e .[gpu]         # cupy, for the analysis on the GPU (device='gpu')
pip install -e .[compare]     # bilby, dynesty, cogwheel and nautilus, to rerun the comparisons in scripts/ (cogwheel 1.6.0)
```

## Tables

One set of tables serves every analysis. For the LIGO Hanford and Livingston detectors there is one table per signal power
$s = u y^2$ (53 nodes), with the distance $y = D_0/D_L$ ($D_0 = 10410$ Mpc) as a parameter and the series summed to
$n_{\max} = 120$. With the distance kept as a parameter the tables serve the analysis in source-frame mass and redshift, and
for the analysis in $(M^{\rm det}, q, \chi_{\rm eff})$ every template reads the nodes it reaches and the distance is integrated
over its prior in the code (`exmarg.search.pe_intrinsic` with the prior `'volume'` or `'madau-dickinson'`).

The tables (40 GB) and the results of the year analysis are on Zenodo, https://doi.org/10.5281/zenodo.23452946 (this DOI
always points to the latest version, record 23599554). Download them and check every file against its SHA-256 list with

```
python scripts/fetch_tables.py 23599554          # the tables, into tables/redshift/
python scripts/fetch_tables.py 23599554 year     # the results of the year analysis, into data/year/ (notebook 4)
```

On the GPU the tables are read into memory, about 0.75 GB per node. On the CPU they can be mapped from disk instead
(`search.load_redshift_store(..., mmap=True)`), so the analysis runs on a machine with less memory than the tables. The
analysis functions take `device='cpu'` or `device='gpu'` and agree to 1e-5 in ln B (notebook 5). Notebooks 0 and 6 run on
the CPU, and notebooks 2, 3, 5 and 7 are written for a GPU with at least 40 GB of memory.

The tables are built in three stages by `scripts/build_tables/`, with the settings of the tables on Zenodo, $n_{\max} = 120$,
HEALPix nside 32 and the 53 nodes of `data/tables/s_grid.npz`.

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

| Notebook | Content | Paper figures |
|---|---|---|
| `notebooks/00_quickstart.ipynb` | Start here. Simulates a segment with a merger and one with noise only, analyzes both on the CPU and draws the posterior of mass and redshift | |
| `notebooks/01_tables.ipynb` | Builds the W tables and draws the GORFs in delay and in frequency | 3, 4, 5, 14, 15 |
| `notebooks/02_single_segments.ipynb` | Analyzes single segments in three parameters (chirp mass, mass ratio, spin) and in two (source-frame mass, redshift), draws the posteriors next to `bilby`, and the convergence in the order of the series (`scripts/compute_convergence.py`) | 16, 17, 18, 20, 21 |
| `notebooks/03_pp.ipynb` | Probability-probability tests in both set-ups | 19 |
| `notebooks/04_population.ipynb` | The population results of one simulated year, from the result files on Zenodo | 6 to 13, 22 to 24, 26 to 29 |
| `notebooks/05_speed.ipynb` | The Bayes factors of `bilby` on the twelve segments of Fig. 21, then the time per segment against the SNR on the GPU and the CPU | |
| `notebooks/06_cogwheel.ipynb` | The comparison with `cogwheel` at finer and finer settings, per template and for the posterior | |
| `notebooks/07_estimated_psd.ipynb` | Noise-only segments analyzed with an estimated noise spectrum, the mean of $N$ periodograms, and the factor $N/(N-2)$ that removes the resulting shift of the likelihood ratio | |

The figures are drawn by `notebooks/figures_*.py` in the style of the paper. Figs. 1, 2 and 25 of the paper are not
reproduced here.

## Data

In the repository

* `data/segments/` holds the simulated segments of notebooks 2, 5 and 6 (`three_parameters.npz`, `two_parameters.npz`)
  and the binary of Figs. 16 and 17 at five distances (`family.npz`).
* `data/banks/` holds the template banks of notebooks 2 and 5.
* `data/results/` holds the `bilby` posteriors, Bayes factors and run times, the `cogwheel` results of
  `scripts/run_cogwheel.py` and `scripts/run_cogwheel_pe.py`, and the results of notebooks 2 and 3.
* `data/tables/` holds the nodes of the tables (`s_grid.npz`) and the SHA-256 lists of the files on Zenodo.
* `data/psd/` holds noise curves.

From Zenodo (https://doi.org/10.5281/zenodo.23452946)

* `tables/redshift/`, the tables (40 GB).
* `data/year/`, the results of the year analysis used in notebook 4 (posterior samples, Bayes factors of single segments,
  Bayes factors against the observing time, information of single segments).

## Scripts

```
python scripts/fetch_tables.py <record> [year]                         # download from Zenodo
python scripts/build_population.py                                     # the population of the paper, src/exmarg/data/demo_population.npz
python scripts/run_bilby.py three data/segments/three_parameters.npz <row>   # the nested-sampling runs of Fig. 20
python scripts/run_bilby.py two data/segments/two_parameters.npz <row>       # the nested-sampling runs of Fig. 21
python scripts/run_cogwheel.py <row> <M> <q> <chi>                     # cogwheel per template, notebook 6
python scripts/run_cogwheel_pe.py <row> <n_eff> <cap>                  # cogwheel posterior, notebook 6
python scripts/compute_convergence.py <tables at n_max = 1, 2, ...> gpu   # Figs. 16 and 17 (tables truncated at lower orders)
```

## Layout

```
src/exmarg/
  detectors.py   detector positions, antenna patterns, noise curves
  profiles.py    the waveform (IMRPhenomXAS) of a template
  data.py        simulated segments
  numerics.py    numerical helpers
  priors.py      the distance priors and the population of the paper (GWTC-5 masses, Madau-Dickinson redshifts to z = 10)
  tables.py      the W tables (GORFs)
  geometry.py    the GORFs in frequency and the truncation deficit
  pp.py          probability-probability tests
  search/        the analysis of segments with the tables, on the CPU (numba) or the GPU (cupy)
                 segments.py, bank.py (matched filter), store.py (tables), gather.py (the table read),
                 pe.py (pe_redshift, pe_intrinsic)
notebooks/       the notebooks and their figure functions
scripts/         downloads, table building, nested sampling, cogwheel and the convergence runs
tests/           tests (pytest)
```
