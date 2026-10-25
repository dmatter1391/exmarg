"""cogwheel's marginalization over the extrinsic parameters on one segment and template, at finer and finer QMC settings,
with the whole cost a background search pays: one construction (the template as its own relative-binning reference, no
trigger) plus one call per segment and template.  Notebook 6 compares the result with our likelihood ratio.

Set-up matched to ours (notebook 2): the same data and noise variance (data/segments/three_parameters.npz), arrival time over
the whole segment (t_range = +-(T/2 - 0.05) s around the segment center), distance prior uniform in volume out to
D0 / ymin = 82163 Mpc, IMRPhenomXAS with equal aligned spins, ASD drift fixed to 1.  The relative-binning frequency bins
are those cogwheel chooses for this segment around the injected chirp mass.

cogwheel marginalizes the arrival time as an integral over seconds (its sky prior is a density per second of delay), so its
ln Lambda exceeds ours by ln(t_range width) = ln 3.9; the run ends with the same converged setting on half the window,
which moves ln Lambda by -ln 2 if that is the only difference.

    python scripts/run_cogwheel.py <row> <M_det> <q> <chi> [n_calls=20] [n_settings=8]     (pip install cogwheel-pe; we used 1.6.0)

n_settings < 8 stops the ladder early (at large SNR the finest settings take minutes per call); the half-window check then
uses the last setting reached instead of the seventh.
    -> data/results/cogwheel_row<row>.npz
"""
import os
import sys
import time

import numpy as np
from cogwheel import data, posterior
from cogwheel.gw_prior.combined import RegisteredPriorMixin, CombinedPrior
from cogwheel.gw_prior.mass import UniformDetectorFrameMassesPrior
from cogwheel.gw_prior.miscellaneous import ZeroTidalDeformabilityPrior, FixedReferenceFrequencyPrior
from cogwheel.likelihood import MarginalizedExtrinsicLikelihoodQAS
from cogwheel.prior import Prior, UniformPriorMixin

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D_MAX = 10410.0 / 0.1267                  # D0 / ymin of the tables (Mpc)


class EqualAlignedSpinPrior(UniformPriorMixin, Prior):
    """chi_1 = chi_2 = chi, uniform in [-0.6, 0.6], as in our bank"""
    range_dic = {'chieff': (-0.6, 0.6)}
    standard_params = ['s1z', 's2z']

    def transform(self, chieff):
        return {'s1z': chieff, 's2z': chieff}

    def inverse_transform(self, s1z, s2z):
        return {'chieff': s1z}


class IntrinsicEqualSpinPrior(RegisteredPriorMixin, CombinedPrior):
    default_likelihood_class = MarginalizedExtrinsicLikelihoodQAS
    prior_classes = [UniformDetectorFrameMassesPrior, EqualAlignedSpinPrior,
                     ZeroTidalDeformabilityPrior, FixedReferenceFrequencyPrior]


def event_data(seg, row, half_width, name):
    """cogwheel's EventData of one segment, with the reference time at the segment center"""
    T, fmax = float(seg['T']), float(seg['fmax'])
    far = np.fft.rfftfreq(int(round(T * 2 * fmax)), d=1 / (2 * fmax))
    idx = np.round(seg['f'] * T).astype(int)
    strain = np.zeros((2, far.size), complex)
    strain[0, idx], strain[1, idx] = seg['d_H'][row], seg['d_L'][row]
    psd = np.full(far.size, np.inf)
    psd[idx] = 2.0 * np.asarray(seg['sig2'], float) / T           # the noise variance of the segment file
    wht = np.where(np.isfinite(psd), 1.0 / np.sqrt(psd), 0.0)
    ev = data.EventData(eventname=name, frequencies=far, strain=strain, wht_filter=np.array([wht, wht]),
                        detector_names=('H', 'L'), tgps=1e9 + T / 2, tcoarse=T / 2)
    return ev, (-half_width, half_width)


def main():
    row, M, q, chi = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
    n_calls = int(sys.argv[5]) if len(sys.argv) > 5 else 20
    n_settings = int(sys.argv[6]) if len(sys.argv) > 6 else 8
    seg = np.load(os.path.join(ROOT, 'data', 'segments', 'three_parameters.npz'))
    T = float(seg['T'])
    Mt, qt = float(seg['true_M_det'][row]), float(seg['true_q'][row])
    kw = dict(mchirp_guess=Mt * (qt / (1 + qt) ** 2) ** 0.6, approximant='IMRPhenomXAS',
              prior_class=IntrinsicEqualSpinPrior,
              prior_kwargs={'mchirp_range': (7, 87), 'q_min': 0.3, 'f_ref': float(seg['fmin'])})
    ev, tr = event_data(seg, row, T / 2 - 0.05, 'row%d' % row)
    p = posterior.Posterior.from_event(event=ev, likelihood_kwargs={'t_range': tr}, **kw)
    wfg, base = p.likelihood.waveform_generator, dict(p.likelihood.par_dic_0)
    fbin = p.likelihood.fbin
    tmpl = {**base, 'm1': M / (1 + q), 'm2': M * q / (1 + q), 's1z': chi, 's2z': chi, 't_geocenter': 0.0}

    def build(neff, cap, t_range):
        L = MarginalizedExtrinsicLikelihoodQAS(ev, wfg, tmpl, fbin=fbin, t_range=t_range,
                                               coherent_score={'min_n_effective': neff, 'max_log2n_qmc': cap})
        L.coherent_score.lookup_table = type(L.coherent_score.lookup_table)(d_luminosity_prior_name='euclidean',
                                                                             d_luminosity_max=D_MAX)
        L.asd_drift = np.ones_like(L.asd_drift)
        return L

    tc = []
    for _ in range(3):
        t0 = time.perf_counter(); build(50, 15, tr); tc.append(time.perf_counter() - t0)
    print('row %d, template M %.2f q %.3f chi %.3f: construction %.2f s' % (row, M, q, chi, np.median(tc)), flush=True)
    ladder = [(50, 15), (100, 16), (200, 17), (500, 19), (1000, 19), (2000, 19), (10 ** 6, 19), (10 ** 6, 20)][:n_settings]
    rows = []
    for neff, cap in ladder:
        L = build(neff, cap, tr)
        t0 = time.perf_counter(); v = np.array([L.lnlike(tmpl) for _ in range(n_calls)])
        dt = (time.perf_counter() - t0) / n_calls
        rows.append((neff, cap, v.mean(), v.std(), dt))
        print('  target n_eff %7d, at most 2^%d samples: ln Lambda %.4f, std %.4f, %.3f s per call'
              % (neff, cap, v.mean(), v.std(), dt), flush=True)
    half_index = min(6, len(ladder) - 1)
    L = build(*ladder[half_index], (tr[0] / 2, tr[1] / 2))
    half = np.array([L.lnlike(tmpl) for _ in range(5)])
    print('  half window: ln Lambda %.4f, std %.4f (whole window %.4f)' % (half.mean(), half.std(), rows[half_index][2]), flush=True)
    np.savez(os.path.join(ROOT, 'data', 'results', 'cogwheel_row%d.npz' % row), template=np.array([M, q, chi]),
             construction_s=np.median(tc), ladder=np.array(rows, float), half_window=half,
             t_range_width=tr[1] - tr[0], n_calls=n_calls, half_index=half_index)


if __name__ == '__main__':
    main()
