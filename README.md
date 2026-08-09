# exmarg

Analytic marginalization over the extrinsic parameters of a compact
binary signal in two detectors. The likelihood marginalized over sky
direction, arrival time, overall phase, polarization, inclination and
luminosity distance is written as a series of data independent geometry
tables, the generalized overlap reduction functions, contracted with
per template data integrals. The lowest order table is the overlap
reduction function of the cross-correlation search; higher orders deform
it as the signal to noise ratio grows, interpolating towards the matched
filter. With the tables precomputed, the marginalized statistic of one
template is a table lookup, so a segment posterior is a direct grid sweep
independent of the loudness.

The method targets stochastic and subthreshold signals, where no trigger
time exists and the time prior spans whole segments, and where the same
data are scanned over many population parameters. The derivation and the
results are in `paper/prd.tex`.

## Install

```
# a conda environment with numpy, scipy, healpy, matplotlib (igwn works)
pip install -e .
python -m pytest tests/ -q
```

## Example notebooks

`examples/` reproduces every figure of the paper and doubles as a tutorial
you run on your own data. Each notebook calls `exmarg` and saves its
plotted data next to the figure.

| notebook | paper figures | what it does |
|---|---|---|
| `01_motivation.ipynb` | 1, 2 | the single-parameter arrival-time experiment and the eigenspectrum of the one-event kernel (self contained, numpy only) |
| `02_tables.ipynb` | 3, 4, 5 | build the `W` tables with `w_table_dl`; sky-quadrature convergence; the generalized overlap reduction function |
| `03_pe.ipynb` | 7 | parameter estimation of one event: fetch tables, sweep the intrinsic grid, corner. Swap in your own segment |
| `04_pp.ipynb` | 10 | the P-P calibration over 596 injections, and the function to run it on your own scored grids |
| `05_sgwb.ipynb` | 11-14 | the intermittent-background rate and mass fit, truncation-depth and segment-duration studies |
| `06_speed.ipynb` | 15 | wall clock versus SNR against bilby |

The notebooks that only reproduce a figure load the small shipped arrays in
`examples/data/` and run in seconds. `03_pe` scores a live grid and needs
the tables (below).

## The tables

Scoring needs the compiled sum-store, one file pair per loudness node `u`
(`sumstore_n120_u<UUUU.UU>.{meta.npz,lnS.npy}`). The full ladder
(`u = 5..500`, 54 nodes) is ~30 GB, too large for the repository, so it
lives on Zenodo. A run fetches only the nodes its template grid spans:

```python
from fetch_tables import ensure_nodes
stems = ensure_nodes(u_lo=40, u_hi=90)   # downloads the missing nodes
```

Set the record with `export EXMARG_ZENODO=<record id>` (the record id
will be published with the archive), or point `EXMARG_STORE` at a local
build to skip the download.

### Calculating the tables

`scripts/build_store.py` builds the store from scratch on a sky and
orientation quadrature:

```
python scripts/build_store.py 120 16 32          # n_max, nside, workers
```

The tables encode the noise geometry, so they are specific to the detector
noise they were built with. The shipped curves are selected with
`exmarg.detectors.set_curve('aplus' | 'o4' | 'design' | 'ajith')`, tabulated
in `data/psd/`; the paper and the shipped ladder use the A+ design
sensitivity. For any other noise power spectral density, add your curve
and rebuild.

## Layout

```
src/exmarg/
  detectors.py   detector geometry, antenna patterns, PSD registry
  profiles.py    the intrinsic template a_nu, Psi_0
  data.py        segments and the matched-filter series z
  numerics.py    the log-arithmetic armor (no physics)
  tables.py      the W tables: sky/orientation/distance averages
  kernel.py      lookup tables compiled from W; scoring is lookups
  evaluate.py    Evaluator, SumEvaluator, grid scans, emcee
  figstyle.py    shared figure style
examples/        the notebooks above + fetch_tables + shipped plot data
scripts/         table construction (build_store, merge, kstore, sumstores)
tests/           unit and invariant tests (pytest)
paper/           the paper (prd.tex + compiled prd.pdf)
```

## Citation

The formalism and its companion (the single-event kernel of Figures 1-2)
are described in the paper; please cite it if you use this code.
