"""Stage 3 of the tables: the sum over the orders, read by exmarg.search (Eq. C10 of the paper).

Every order is homogeneous, lnK_n(q1, q2) = 2n ln q2 + g_n(ln q1 - ln q2), so the sum over the orders

    lnS(alpha; q1, q2) = logsumexp_n [ n ln alpha + lnK_n ] = logsumexp_n [ n beta + g_n(Delta) ],
    beta = ln alpha + 2 ln q2,   Delta = ln q1 - ln q2,

depends on two variables.  g_n comes from the last row and the last column of each (q1, q2) block of stage 2.  One
table T[tau, phi, beta, Delta] (float32, phi on the first 49 of the 96 phases, since the sum is even in phi) per node,
about 750 MB.

    python scripts/build_tables/build_bd.py <k_dir> <out_dir> [node indices, default all]
    -> <out_dir>/bd_n120_s<s>.npz

BD_NMAX_LADDER=1,2,5,... also writes the tables truncated at those orders, from the same pass, into
<out_dir>_n001, ... (the tables of scripts/compute_convergence.py).  BD_INMEM=1 reads a node into memory first.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ALPHA = np.geomspace(0.16, 4.0e4, 30)              # sets the range of beta
LADDER_ENV = os.environ.get('BD_NMAX_LADDER', '').strip()


def out_dir(base, nmax, nmax_full):
    """the full order in base, the truncated tables next to it"""
    return base if nmax == nmax_full else '%s_n%03d' % (base, nmax)


def build_node(stem, base):
    meta = np.load(stem + '.meta.npz')
    s = float(meta['s_node'])
    lnq = meta['lnq'].astype(np.float64)
    tau_idx = meta['tau_idx']
    nmax_full = int(os.path.basename(stem).split('_n')[1].split('_')[0])
    ladder = [int(v) for v in LADDER_ENV.split(',')] if LADDER_ENV else [nmax_full]
    assert max(ladder) <= nmax_full
    outs = {nm: os.path.join(out_dir(base, nm, nmax_full), 'bd_n%03d_s%09.3f.npz' % (nm, s)) for nm in ladder}
    for d in {os.path.dirname(v) for v in outs.values()}:
        os.makedirs(d, exist_ok=True)
    todo = [nm for nm in ladder if not os.path.exists(outs[nm])]
    if not todo:
        print('s=%g exists for every order, skipped' % s, flush=True)
        return
    t0 = time.time()
    f = stem + '.lnK.npy'
    A = np.load(f) if int(os.environ.get('BD_INMEM', '0')) else np.load(f, mmap_mode='r')
    NORD, NTAU, NQ, _, NPHI = A.shape
    NPHIF = NPHI // 2 + 1
    last = NQ - 1
    dlq = float(lnq[1] - lnq[0])
    delta = np.concatenate([lnq - lnq[last], (lnq[last] - lnq)[::-1][1:]])       # sorted, 2 nq - 1
    dbeta = 2.0 * dlq
    b_lo = np.log(ALPHA[0]) + 2.0 * lnq[0]
    b_hi = np.log(ALPHA[-1]) + 2.0 * lnq[-1]
    NB = int(np.ceil((b_hi - b_lo) / dbeta)) + 1
    beta = b_lo + dbeta * np.arange(NB)
    n = np.arange(NORD, dtype=np.float64)
    Ts = {nm: np.empty((NTAU, NPHIF, NB, delta.size), np.float32) for nm in todo}
    E_nb = n[:, None] * beta[None, :]                                           # (NORD, NB)
    for ti in range(NTAU):
        col = np.asarray(A[:, ti, :, last, :NPHIF]).astype(np.float64)          # (NORD, q1, phi)
        row = np.asarray(A[:, ti, last, :, :NPHIF]).astype(np.float64)          # (NORD, q2, phi)
        g_neg = col - 2.0 * n[:, None, None] * lnq[last]                         # Delta <= 0
        g_pos = row - 2.0 * n[:, None, None] * lnq[None, :, None]                # Delta >= 0
        g = np.concatenate([g_neg, g_pos[:, ::-1, :][:, 1:, :]], axis=1)        # (NORD, 2nq-1, phi), sorted in Delta
        for pi in range(NPHIF):
            E = E_nb[:, :, None] + g[:, None, :, pi]                             # (NORD, NB, ND)
            # logsumexp over the orders up to each truncation, each with its own running maximum
            mx_run = np.maximum.accumulate(E, axis=0)
            for nm in todo:
                mxn = mx_run[nm]
                tot = np.exp(E[:nm + 1] - mxn[None]).sum(axis=0)
                Ts[nm][ti, pi] = (mxn + np.log(tot)).astype(np.float32)
    for nm in todo:
        if not np.isfinite(Ts[nm]).all():
            raise FloatingPointError('table has non-finite entries at n_max=%d' % nm)
        np.savez(outs[nm], T=Ts[nm], beta=beta, delta=delta, tau_idx=tau_idx, lnq=lnq, alpha=ALPHA, s_node=s, u_node=s,
                 n_max=nm, nphif=NPHIF, norm='taylor_off1')
    print('s=%g done: %d orders %s in %.0f s' % (s, len(todo), todo, time.time() - t0), flush=True)


def main():
    k_dir, base = sys.argv[1], sys.argv[2]
    s_nodes = np.load(ROOT / 'data' / 'tables' / 's_grid.npz')['s_nodes']
    pick = [int(v) for v in sys.argv[3:]] or list(range(s_nodes.size))
    for si in pick:
        build_node(os.path.join(k_dir, 'kstore_n120_s%09.3f' % s_nodes[si]), base)


if __name__ == '__main__':
    main()
