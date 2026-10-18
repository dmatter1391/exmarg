"""derive summed tables from the per-order K store — NO compiling:
stream the orders of each node's kstore mmap, accumulate with
logaddexp, snapshot at the requested truncations.

    python scripts/derive_sumstores.py 80,120 [batch] [n_batch]
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
import numpy as np

from exmarg.kernel import SumKernel, save_sumstore

ROOT = Path(__file__).parent.parent
N_SET = sorted(int(x) for x in
               (sys.argv[1] if len(sys.argv) > 1 else '80,120').split(','))
BATCH = int(sys.argv[2]) if len(sys.argv) > 2 else 0
N_BATCH = int(sys.argv[3]) if len(sys.argv) > 3 else 1

metas = sorted((ROOT / 'stores').glob('kstore_N120_c24_u*.meta.npz'))[BATCH::N_BATCH]
print('slice %d/%d: %d rungs, truncations %s'
      % (BATCH, N_BATCH, len(metas), N_SET), flush=True)

for mp in metas:
    m = np.load(mp)
    u_node = float(m['u_node'])
    have = all((ROOT / 'stores' / ('sumstore_n%03d_u%07.2f.meta.npz'
                        % (n, u_node))).exists()
               and (ROOT / 'stores' / ('sumstore_n%03d_u%07.2f.lnS.npy'
                            % (n, u_node))).exists()
               for n in N_SET)
    if have:
        print('u=%g already complete, skipped' % u_node, flush=True)
        continue
    t0 = time.time()
    mm = np.load(str(mp).replace('.meta.npz', '.lnK.npy'),
                 mmap_mode='r')
    acc = np.full(mm.shape[1:], -np.inf, dtype=np.float32)
    want = set(N_SET)
    for n in range(int(m['N']) + 1):
        np.logaddexp(acc, mm[n], out=acc)
        if n in want:
            sk = SumKernel(u_node, m['tau_idx'], int(m['n_time']),
                           float(m['T']), m['lnq'], acc.copy())
            save_sumstore(sk, n, ROOT / 'stores')
            print('  u=%g n_max=%d saved (%.0f s)'
                  % (u_node, n, time.time() - t0), flush=True)
        if n >= max(want):
            break
    print('rung u=%g done (%.0f s)' % (u_node, time.time() - t0),
          flush=True)
print('slice %d done' % BATCH, flush=True)
