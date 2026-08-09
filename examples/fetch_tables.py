"""Fetch the precomputed sum-tables the examples score against.

Scoring needs only the compiled sum-store, one file pair per loudness
node u,

    sumstore_n120_u<UUUU.UU>.meta.npz   (small, the grid metadata)
    sumstore_n120_u<UUUU.UU>.lnS.npy    (~400 MB, the log-S table)

The raw W tables are an intermediate of the build and are not
distributed. The full ladder (u = 5..250, 35 nodes, ~14 GB) lives on
Zenodo; a run fetches only the nodes its template grid spans, so a
single-event demo pulls a few nodes and a paper-exact reproduction
pulls the whole ladder.

    from fetch_tables import ensure_nodes
    stems = ensure_nodes(u_lo=5, u_hi=30)     # local paths, downloads if absent
    # -> pass stems to exmarg.SumEvaluator

Point EXMARG_STORE at an existing store (e.g. a local build) to skip
the download entirely.
"""
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np

N_MAX = 120
# the loudness ladder the store was built on (u = 5..250, x1.12 step)
U_LADDER = np.array([
    5., 6., 7., 8., 9., 10., 11., 12., 13., 15., 17., 19., 21., 24.,
    27., 30., 34., 38., 43., 48., 54., 60., 67., 75., 84., 94., 105.,
    118., 132., 148., 166., 186., 208., 233., 250.])

# Zenodo record holding the per-node files; set after upload.
ZENODO_RECORD = os.environ.get('EXMARG_ZENODO', '')  # e.g. '15xxxxxx'
ZENODO_URL = 'https://zenodo.org/records/%s/files/%s?download=1'

# where stores live: EXMARG_STORE, else the repo's stores/ next to examples/
STORE = Path(os.environ.get(
    'EXMARG_STORE', Path(__file__).resolve().parents[1] / 'stores'))


def _name(u, ext):
    return 'sumstore_n%03d_u%07.2f.%s' % (N_MAX, u, ext)


def nodes_between(u_lo, u_hi):
    """the ladder nodes that bracket [u_lo, u_hi] (one node of margin)."""
    lo = max(int(np.searchsorted(U_LADDER, u_lo)) - 1, 0)
    hi = min(int(np.searchsorted(U_LADDER, u_hi)) + 1, U_LADDER.size)
    return U_LADDER[lo:hi]


def _fetch(name, dest):
    if not ZENODO_RECORD:
        raise FileNotFoundError(
            '%s not found in %s and EXMARG_ZENODO is unset. Either build a '
            'store with scripts/build_store.py or set the Zenodo record id.'
            % (name, STORE))
    url = ZENODO_URL % (ZENODO_RECORD, name)
    print('  downloading %s' % name, flush=True)
    tmp = dest.with_suffix(dest.suffix + '.part')
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)


def ensure_nodes(u_lo=5.0, u_hi=250.0, nodes=None):
    """Ensure the sum-store files for the requested nodes are local.

    Returns the list of stems (paths without the .meta.npz / .lnS.npy
    suffix) ready for exmarg.SumEvaluator, ordered by u.
    """
    STORE.mkdir(parents=True, exist_ok=True)
    us = np.asarray(nodes, float) if nodes is not None \
        else nodes_between(u_lo, u_hi)
    stems = []
    for u in us:
        stem = STORE / ('sumstore_n%03d_u%07.2f' % (N_MAX, u))
        for ext in ('meta.npz', 'lnS.npy'):
            f = STORE / _name(u, ext)
            if not f.exists():
                _fetch(_name(u, ext), f)
        stems.append(str(stem))
    return stems


if __name__ == '__main__':
    lo = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
    hi = float(sys.argv[2]) if len(sys.argv) > 2 else 250.0
    s = ensure_nodes(lo, hi)
    print('%d nodes ready in %s' % (len(s), STORE))
