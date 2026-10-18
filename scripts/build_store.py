"""build the deep Taylor-unit W stores for a u node ladder in ONE
quadrature sweep.

    python scripts/build_store.py [N] [nside] [workers] [u_lo] [u_hi] [step]

defaults: N=120, nside=8, workers=32, u ladder 5 .. 250 with x1.15
log spacing, fixed ymin = 0.4.  one npz per node, float32 rows, Taylor units (norm
flag); evaluate.load_store refuses stores without the flag.  worker
memory is bounded by shrinking the pixel chunk with N.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
import numpy as np

from exmarg.tables import w_table_par

YMIN = 0.4
MEM_BUDGET = 3e8            # bytes per worker for the power lists

if __name__ == '__main__':
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    nside = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    u_lo = float(sys.argv[4]) if len(sys.argv) > 4 else 5.0
    u_hi = float(sys.argv[5]) if len(sys.argv) > 5 else 250.0
    step = float(sys.argv[6]) if len(sys.argv) > 6 else 1.12
    # node batching: build ladder slice [batch::n_batch] (bounds the
    # master accumulator, which holds n_nodes x n_tau x n_keys)
    batch = int(sys.argv[7]) if len(sys.argv) > 7 else 0
    n_batch = int(sys.argv[8]) if len(sys.argv) > 8 else 1
    # pixel batching: build sky slice [pix_b / n_pix_b] of the pixels
    # (partial stores; sum them with scripts/merge_store.py)
    pix_b = int(sys.argv[9]) if len(sys.argv) > 9 else 0
    n_pix_b = int(sys.argv[10]) if len(sys.argv) > 10 else 1
    chunk_arg = int(sys.argv[11]) if len(sys.argv) > 11 else 0

    # whole-number ladder with ratio <= step (integers force slightly
    # larger ratios at the very bottom, e.g. 5 -> 6)
    ladder = [int(np.ceil(u_lo))]
    while ladder[-1] < u_hi:
        ladder.append(min(int(round(u_hi)),
                          max(ladder[-1] + 1,
                              int(round(ladder[-1] * step)))))
    u_nodes = np.array(ladder, float)[batch::n_batch]
    n_psi, n_chi = 2 * N + 3, N + 3
    chunk = max(1, int(MEM_BUDGET / ((4 * N + 2) * n_psi * n_chi * 8)))
    if chunk_arg > 0:
        chunk = chunk_arg
    print('nodes (%u): %s' % (u_nodes.size, np.round(u_nodes, 1)),
          flush=True)
    print('N=%u nside=%u chunk=%u workers=%u' % (N, nside, chunk,
                                                 workers), flush=True)

    import healpy as hp
    npix = hp.nside2npix(nside)
    p_lo = npix * pix_b // n_pix_b
    p_hi = npix * (pix_b + 1) // n_pix_b
    print('pixel slice %u/%u: [%u, %u)' % (pix_b, n_pix_b, p_lo,
                                            p_hi), flush=True)

    tau_f, Ws = w_table_par(u_nodes, u_nodes, YMIN, N, n_tau=2049,
                            nside=nside, chunk=chunk, workers=workers,
                            verbose=True, pix_range=(p_lo, p_hi))
    ks = sorted(k for k in Ws[0] if k[2] <= k[1])
    karr = np.array(ks, dtype=np.int16)
    for u, W in zip(u_nodes, Ws):
        stem = 'wtaylor_N%u_nside%u_u%07.2f' % (N, nside, u)
        if n_pix_b > 1:
            stem += '.part%uof%u' % (pix_b, n_pix_b)
        out = Path(__file__).parent.parent / 'stores' / (stem + '.npz')
        np.savez_compressed(out, tau=tau_f, keys=karr,
                            W=np.array([W[k] for k in ks],
                                       dtype=np.float32),
                            norm='taylor_off1', u_node=u, ymin=YMIN,
                            n_pix_b=n_pix_b)
        print('written:', out, flush=True)
