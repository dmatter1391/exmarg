"""sum the pixel-sliced partial W stores into final ones.

    python scripts/merge_store.py [batch] [n_batch] [prefix]

merges the stems sliced [batch::n_batch] (so a 35-node ladder maps
onto a 35-task array), optionally only stems starting with prefix.
the deposit is linear with a global 1/npix weight, so the parts of
each u node SUM to the full table exactly.  a node is merged only
when ALL its parts are present; parts are deleted after the final
store is written and verified finite.  one node needs ~8 GB of
memory (float64 accumulator + one decompressed part).
"""
import re
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.parent
PAT = re.compile(r'(wtaylor_N\d+_nside\d+_u[\d.]+)\.part(\d+)of(\d+)\.npz$')

batch = int(sys.argv[1]) if len(sys.argv) > 1 else 0
n_batch = int(sys.argv[2]) if len(sys.argv) > 2 else 1
prefix = sys.argv[3] if len(sys.argv) > 3 else 'wtaylor_'

groups = defaultdict(dict)
for p in (ROOT / 'stores').glob(prefix + '*.part*.npz'):
    m = PAT.match(p.name)
    if m:
        groups[m.group(1)][int(m.group(2))] = (p, int(m.group(3)))

if not groups:
    sys.exit('no part files to merge')

for stem, parts in sorted(groups.items())[batch::n_batch]:
    n_expect = next(iter(parts.values()))[1]
    if len(parts) != n_expect:
        print('%s: %u/%u parts ready, skipped'
              % (stem, len(parts), n_expect), flush=True)
        continue
    tot, meta = None, None
    for i in range(n_expect):
        z = np.load(parts[i][0])
        W = z['W']
        if tot is None:
            tot = W.astype(np.float64)
        else:
            tot += W        # in-place: no float64 copy of the part
        meta = z
    if not np.isfinite(tot).all():
        print('%s: SUM NOT FINITE, not merged' % stem,
              flush=True)
        continue
    out = ROOT / 'stores' / (stem + '.npz')
    np.savez_compressed(out, tau=meta['tau'], keys=meta['keys'],
                        W=tot.astype(np.float32),
                        norm=str(meta['norm']),
                        u_node=float(meta['u_node']),
                        ymin=float(meta['ymin']))
    for i in range(n_expect):
        parts[i][0].unlink()
    print('written:', out, flush=True)
