"""Stage 2 of the tables: every order n of the series at one node of s, on a grid of the two filtered-data
magnitudes and their relative phase,

    lnK_n(tau, q1, q2, phi) = ln sum_{a,b} C(n,a) C(n,b) W^(n)_ab(tau) q1^(a+b) q2^(2n-a-b) cos[(a-b) phi] - ln(s^n n!) ,

with q on 110 logarithmic points from 0.05 to 100 and phi on 96 points.  The delays are rebinned from the W grid to
the time step 1 / (8 f_max) of the filtered data.  A node is split over many tasks, each writing a disjoint set of
orders into one memory map (about 49 GB per node):

    python scripts/build_tables/build_k.py prepare  <w_dir> <k_dir> <node>
    python scripts/build_tables/build_k.py slab     <w_dir> <k_dir> <node> <k> <K>   orders k, k + K, k + 2K, ...
    python scripts/build_tables/build_k.py finalize <w_dir> <k_dir> <node>

node indexes data/tables/s_grid.npz.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap
from scipy.special import comb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from exmarg.tables import rebin
from exmarg.numerics import ln_offset

N, C, NQ, NPHI = 120, 24.0, 110, 96
QMAX, QSPAN = 100.0, 2000.0
T, FMAX = 4.0, 512.0                         # the delay grid is T / n_time = 1 / (8 FMAX), the same for every T
n_time = 8 * int(FMAX * T)
tau_all = np.fft.fftfreq(n_time) * T
t_step = T / n_time
s_nodes = np.load(ROOT / 'data' / 'tables' / 's_grid.npz')['s_nodes']


def load_node(stem):
    m = np.load(stem + '.meta.npz')
    Wa = np.load(stem + '.W.npy')
    tau_f = m['tau']
    W = {tuple(int(v) for v in k): Wa[i].astype(float) for i, k in enumerate(m['keys'])}
    keep = np.abs(tau_all) <= 1.05 * np.abs(tau_f).max()
    tau_idx = np.rint(tau_all[keep] / t_step).astype(int)
    Wd = rebin(tau_f, W, tau_all[keep])
    keys = np.array(sorted(k for k in W if k[2] <= k[1]))
    WM = np.array([Wd[tuple(k)] for k in keys])
    return keys, WM, tau_idx


def stems(w_dir, k_dir, si):
    s = float(s_nodes[si])
    return s, str(Path(w_dir) / ('sladder_n%03d_s%09.3f' % (N, s))), str(Path(k_dir) / ('kstore_n%03d_s%09.3f' % (N, s)))


def compile_order(n, s, keys, WM, lnq, by_n, phi_g, n_tau):
    """lnK_n(tau, q1, q2, phi) minus the storage offset: one GEMM over the (a, b) pairs of each phase class
    m = a - b, then one GEMM against the cosine matrix"""
    lq1 = lnq[:, None]
    lq2 = lnq[None, :]
    o_n = ln_offset(n, s, s)
    g_n = 2.0 * n * np.maximum(lq1, lq2)
    groups = {}
    for (j, a, b) in by_n[n]:
        m = a - b
        coeff = comb(n, a) * comb(n, b) * (2.0 if m else 1.0)
        groups.setdefault(m, []).append((j, a, b, coeff))
    ms = sorted(groups)
    Cm = np.empty((len(ms), n_tau * NQ * NQ))
    for i, m in enumerate(ms):
        rows = groups[m]
        Wm = np.array([cf * WM[j] for (j, a, b, cf) in rows])
        P = np.exp(np.stack([(a + b) * lq1 + (2 * n - a - b) * lq2 - g_n for (j, a, b, cf) in rows]))
        Cm[i] = (Wm.T @ P.reshape(len(rows), -1)).ravel()
    cosM = np.cos(np.outer(ms, phi_g))
    Kn = (Cm.T @ cosM).reshape(n_tau, NQ, NQ, NPHI)
    if not np.isfinite(Kn).all():
        raise FloatingPointError('kernel accumulation non-finite at order %d' % n)
    return (np.log(np.maximum(Kn, 1e-300)) + (g_n - o_n)[None, :, :, None]).astype(np.float32)


def main():
    mode, w_dir, k_dir, si = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    os.makedirs(k_dir, exist_ok=True)
    s, stem_in, stem_out = stems(w_dir, k_dir, si)
    qg = np.geomspace(QMAX / QSPAN, QMAX, NQ)
    lnq = np.log(qg)
    if mode == 'prepare':
        keys, WM, tau_idx = load_node(stem_in)
        mm = open_memmap(stem_out + '.lnK.npy', mode='w+', dtype=np.float32, shape=(N + 1, tau_idx.size, NQ, NQ, NPHI))
        del mm
        np.savez(stem_out + '.prep.npz', keys=keys, WM=WM, tau_idx=tau_idx)
        print('s=%g prepared: %d keys, %d delays' % (s, len(keys), tau_idx.size), flush=True)
    elif mode == 'slab':
        k, K = int(sys.argv[5]), int(sys.argv[6])
        p = np.load(stem_out + '.prep.npz')
        keys, WM, tau_idx = p['keys'], p['WM'], p['tau_idx']
        phi_g = np.arange(NPHI) * (2 * np.pi / NPHI)
        by_n = {n: [] for n in range(N + 1)}
        for j, (n, a, b) in enumerate(keys):
            by_n[int(n)].append((j, int(a), int(b)))
        mm = open_memmap(stem_out + '.lnK.npy', mode='r+')
        t0 = time.time()
        for n in range(k, N + 1, K):
            mm[n] = compile_order(n, s, keys, WM, lnq, by_n, phi_g, tau_idx.size)
            print('  s=%g order %d written (%.0f s)' % (s, n, time.time() - t0), flush=True)
        mm.flush()
        del mm
    elif mode == 'finalize':
        tau_idx = np.load(stem_out + '.prep.npz')['tau_idx']
        mm = open_memmap(stem_out + '.lnK.npy', mode='r')
        mid = mm[:, tau_idx.size // 2, NQ // 2, NQ // 2, :]
        bad = [n for n in range(N + 1) if not np.isfinite(mid[n]).all() or np.all(mid[n] == 0.0)]
        if bad:
            raise SystemExit('s=%g: orders not written: %s' % (s, bad))
        np.savez(stem_out + '.meta.npz', u_node=s, s_node=s, lnq=lnq, tau_idx=tau_idx, n_time=n_time, T=T, C=C, N=N)
        os.remove(stem_out + '.prep.npz')
        print('s=%g finalized: all %d orders present' % (s, N + 1), flush=True)


if __name__ == '__main__':
    main()
