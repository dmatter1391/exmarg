"""per-order K store: the PERMANENT derived layer.  every ladder
node's full-depth (N=120) per-order tables lnK_n, streamed to a raw
mmap-able npy while compiling (GEMM path, C=24 design tops) — after
this build NOTHING is ever compiled again: sumstores of any n_max,
lnsvec scans, truncation certificates and lnLambda_max curves are
all mmap reads or cheap accumulations over these files.

    sbatch --array=0-34 ... build_kstore.py $SLURM_ARRAY_TASK_ID 35

task k builds node [k] and writes
kstore_N120_c24_u%07.2f.{lnK.npy,meta.npz}  (~49 GB raw per node).
progress: one line per 10 orders with elapsed seconds.
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
import numpy as np
from numpy.lib.format import open_memmap

from exmarg.data import make_event
from exmarg.evaluate import load_store
from exmarg.kernel import Kernel
from exmarg.tables import rebin

ROOT = Path(__file__).parent.parent
N, C = 120, 24.0
BATCH = int(sys.argv[1]) if len(sys.argv) > 1 else 0
N_BATCH = int(sys.argv[2]) if len(sys.argv) > 2 else 1

case = make_event(35.0, 30.0, 0.0, 0.0, 4.0, 20.0, 512.0,
                  sky=(1.1, 2.3), psi_s=0.7, chi_s=0.6)
n_time = 8 * int(case.nu.max())
t_step = case.T / n_time
tau_all = np.fft.fftfreq(n_time) * case.T

stores = sorted((ROOT / 'stores').glob('wtaylor_N120_nside16_u*.npz'))[BATCH::N_BATCH]
for path in stores:
    u_node = float(path.stem.split('_u')[1])
    stem = ROOT / 'stores' / ('kstore_N%d_c%02.0f_u%07.2f' % (N, C, u_node))
    if Path(str(stem) + '.meta.npz').exists():
        print('u=%g kstore exists, skipped' % u_node, flush=True)
        continue
    t0 = time.time()
    tau_f, W = load_store(path)
    print('u=%g: W loaded (%.0f s)' % (u_node, time.time() - t0),
          flush=True)
    keep = np.abs(tau_all) <= 1.05 * np.abs(tau_f).max()
    tau_idx = np.rint(tau_all[keep] / t_step).astype(int)
    Wd = rebin(tau_f, W, tau_all[keep])
    keys = np.array(sorted(k for k in W if k[2] <= k[1]))
    WM = np.array([Wd[tuple(k)] for k in keys])
    del W, Wd
    mm = open_memmap(str(stem) + '.lnK.npy', mode='w+',
                     dtype=np.float32,
                     shape=(N + 1, tau_idx.size, 110, 110, 96))
    k = Kernel(u_node, tau_idx, n_time, case.T, keys, WM,
               C * np.sqrt(u_node), verbose=True, store=mm)
    mm.flush()
    np.savez(str(stem) + '.meta.npz', u_node=u_node, lnq=k.lnq,
             tau_idx=tau_idx, n_time=n_time, T=case.T, C=C, N=N)
    print('u=%g DONE (%.0f s total)' % (u_node, time.time() - t0),
          flush=True)
