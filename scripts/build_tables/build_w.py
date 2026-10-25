"""Stage 1 of the tables: the GORFs W^(n)_ab(tau) at fixed distance, at every node of s = u y^2.

With the distance kept as a parameter the GORFs depend on the filter power and the distance only through s, so one
table per node serves every waveform at every distance.  The nodes are data/tables/s_grid.npz (53 nodes, logarithmic
from 4.3e-3 to 4000).  Settings of the tables on Zenodo: n_max = 120, HEALPix nside = 32, 257 delay points.  One
quadrature sweep over the sky serves all nodes at once (7.1 h on one GPU for the 53 nodes).

    python scripts/build_tables/build_w.py <out_dir> [node indices, default all]
    -> <out_dir>/sladder_n120_s<s>.W.npy and .meta.npz, and manifest.npz
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from w_gpu import w_table_s_gpu

ROOT = Path(__file__).resolve().parents[2]
N_MAX, NSIDE, N_TAU = 120, 32, 257
# sky pixels per GPU batch; the working set scales with nodes x chunk (24 fits 53 nodes on a 46 GB card)
CHUNK = int(os.environ.get('SLADDER_CHUNK', '24'))


def save_node(stem, s_node, tau, W):
    """one node: the (a >= b) half of the keys as a raw float32 .npy (memory-mappable) and a small .meta.npz;
    the reader mirrors (b, a)"""
    keys = sorted(k for k in W if k[1] >= k[2])
    np.save(str(stem) + '.W.npy', np.array([W[k] for k in keys], np.float32))
    np.savez(str(stem) + '.meta.npz', tau=tau.astype(np.float32), keys=np.array(keys, np.int16), norm='taylor_off1',
             s_node=float(s_node), n_max=N_MAX, nside=NSIDE, shape=np.array([len(keys), tau.size]))


def main():
    import cupy as cp
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    s_all = np.load(ROOT / 'data' / 'tables' / 's_grid.npz')['s_nodes']
    pick = [int(v) for v in sys.argv[2:]] or list(range(s_all.size))
    s_nodes = s_all[pick]
    print('%d nodes in [%.4g, %.4g], n_max %d, nside %d, %d delays' % (s_nodes.size, s_nodes[0], s_nodes[-1], N_MAX,
                                                                      NSIDE, N_TAU), flush=True)
    t0 = time.time()
    tau, W = w_table_s_gpu(s_nodes, s_nodes, N_MAX, n_tau=N_TAU, nside=NSIDE, chunk=CHUNK, verbose=True)
    cp.cuda.Stream.null.synchronize()
    seconds = time.time() - t0
    W = W if isinstance(W, list) else [W]
    for s, Wj in zip(s_nodes, W):
        save_node(out / ('sladder_n%03d_s%09.3f' % (N_MAX, s)), s, tau, Wj)
    np.savez(out / 'manifest.npz', s_nodes=s_nodes, n_max=N_MAX, nside=NSIDE, n_tau=N_TAU, s_min=float(s_nodes[0]),
             s_max=float(s_nodes[-1]), built_s=seconds)
    print('%d nodes in %.0f s, written to %s' % (s_nodes.size, seconds, out), flush=True)


if __name__ == '__main__':
    main()
