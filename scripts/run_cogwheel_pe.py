"""cogwheel's posterior of (chirp mass, mass ratio, spin) on one segment of data/segments/three_parameters.npz, sampled with
nautilus, the extrinsic parameters marginalized by cogwheel.  Notebook 6 draws it next to ours.

The set-up is the one of scripts/run_cogwheel.py: same data and noise variance, arrival time over the whole segment,
distance prior uniform in volume out to D0 / ymin, IMRPhenomXAS with equal aligned spins, ASD drift 1.  The marginalization
runs at one setting of the ladder of run_cogwheel.py: target number of effective samples n_eff, at most 2^cap QMC samples.

    python scripts/run_cogwheel_pe.py <row> <n_eff> <cap> [n_live=1000]
    -> data/results/cogwheel_pe_row<row>_neff<n_eff>_cap<cap>.npz
"""
import os
import sys
import tempfile
import time

import numpy as np
import pandas as pd
from cogwheel import posterior, sampling

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_cogwheel import ROOT, D_MAX, IntrinsicEqualSpinPrior, event_data


def main():
    row, neff, cap = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    n_live = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
    seg = np.load(os.path.join(ROOT, 'data', 'segments', 'three_parameters.npz'))
    T = float(seg['T'])
    Mt, qt = float(seg['true_M_det'][row]), float(seg['true_q'][row])
    ev, tr = event_data(seg, row, T / 2 - 0.05, 'row%d' % row)
    post = posterior.Posterior.from_event(event=ev, likelihood_kwargs={'t_range': tr, 'coherent_score': {'min_n_effective': neff, 'max_log2n_qmc': cap}},
                                          mchirp_guess=Mt * (qt / (1 + qt) ** 2) ** 0.6, approximant='IMRPhenomXAS',
                                          prior_class=IntrinsicEqualSpinPrior,
                                          prior_kwargs={'mchirp_range': (7, 87), 'q_min': 0.3, 'f_ref': float(seg['fmin'])})
    L = post.likelihood
    L.coherent_score.lookup_table = type(L.coherent_score.lookup_table)(d_luminosity_prior_name='euclidean',
                                                                         d_luminosity_max=D_MAX)
    L.asd_drift = np.ones_like(L.asd_drift)
    sampler = sampling.Nautilus(post, run_kwargs={'n_live': n_live, 'verbose': False})
    t0 = time.time()
    with tempfile.TemporaryDirectory() as rundir:
        sampler.run(rundir)
        samples = pd.read_feather(os.path.join(rundir, 'samples.feather'))
    seconds = time.time() - t0
    m1, m2 = samples['m1'].to_numpy(), samples['m2'].to_numpy()
    w = samples['weights'].to_numpy() if 'weights' in samples else np.ones(m1.size)
    np.savez(os.path.join(ROOT, 'data', 'results', 'cogwheel_pe_row%d_neff%d_cap%d.npz' % (row, neff, cap)),
             M=(m1 + m2).astype(np.float32), q=(m2 / m1).astype(np.float32), chi=samples['s1z'].to_numpy().astype(np.float32),
             weights=w, seconds=seconds, n_live=n_live, n_eff=neff, cap=cap)
    print('row %d, n_eff %d, cap %d: %d samples in %.0f s' % (row, neff, cap, m1.size, seconds), flush=True)


if __name__ == '__main__':
    main()
