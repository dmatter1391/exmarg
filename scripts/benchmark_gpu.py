"""Time the redshift analysis on the GPU for many segments.

The segments are sorted by SNR and analyzed in batches of `batch` segments, one call of exmarg.gpu.pe_redshift per
batch. The output file holds, for every batch, its segments, their SNR, the wall time of the call and the ln B of
each segment, plus the time to read the tables into memory and the time of the first call (which compiles the kernels).

    python scripts/benchmark_gpu.py segments.npz store_dir bank.npy out.npz [batch] [part] [n_parts]

part and n_parts split the batches between processes, one per GPU (CUDA_VISIBLE_DEVICES).
"""
import sys
import time

import numpy as np
import cupy as cp

import exmarg.gpu as gpu

segments, store_dir, bank_file, out = sys.argv[1:5]
batch = int(sys.argv[5]) if len(sys.argv) > 5 else 16
part = int(sys.argv[6]) if len(sys.argv) > 6 else 0
n_parts = int(sys.argv[7]) if len(sys.argv) > 7 else 1

start = time.time()
store = gpu.load_redshift_store(store_dir)
load_seconds = time.time() - start
print('tables read in %.0f s' % load_seconds, flush=True)

bank = gpu.mass_bank(bank_file, segments)
snr = np.load(segments)['snr']

# the first call compiles the kernels
start = time.time()
gpu.pe_redshift(segments, bank, store, rows=[0])
first_seconds = time.time() - start

order = np.argsort(snr)
batches = [order[i:i + batch] for i in range(0, order.size, batch)][part::n_parts]
rows, seconds, ln_b = [], [], []
for k, b in enumerate(batches):
    b = np.sort(b)
    cp.cuda.Device().synchronize()
    start = time.time()
    result = gpu.pe_redshift(segments, bank, store, rows=b, chunk=batch)
    seconds.append(time.time() - start)
    rows.append(b)
    ln_b.append(result['ln_b'])
    print('batch %d of %d: %d segments, SNR %.1f to %.1f, %.2f s per segment'
          % (k + 1, len(batches), b.size, snr[b].min(), snr[b].max(), seconds[-1] / b.size), flush=True)

np.savez(out, rows=np.concatenate(rows), batch_of_row=np.repeat(np.arange(len(rows)), [r.size for r in rows]),
         batch_seconds=np.array(seconds), batch_size=np.array([r.size for r in rows]), snr=snr[np.concatenate(rows)],
         ln_b=np.concatenate(ln_b), load_seconds=load_seconds, first_call_seconds=first_seconds,
         n_templates=bank.M.size, gpu=cp.cuda.runtime.getDeviceProperties(0)['name'].decode())
print('written', out)
