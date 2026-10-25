"""The convergence of the series with the truncation order for one binary (Figs. 16 and 17 of the paper).

data/segments/family.npz holds one nonspinning binary (detector-frame total mass 65 Msun, mass ratio q = 6/7) at five
distances (SNR 8 to 25) in 8 s segments.  For every truncation order n_max of the tables we compute ln Lambda at the true
template of each segment (Fig. 16), and for n_max = 40, 80 and 120 the posterior of the SNR 11 segment on the bank of
data/banks/three_parameters.npy, with the distance integrated over the Madau-Dickinson prior (Fig. 17).

The tables of every order are built like the n_max = 120 tables of Zenodo; only the latter are distributed (40 GB per
order).  Pass one table folder per order:

    python scripts/compute_convergence.py tables/redshift_n001 ... tables/redshift [gpu]
    -> data/results/convergence.npz
"""
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
import exmarg.search as search
from exmarg.pp import chirp_mass

SEG = os.path.join(ROOT, 'data', 'segments', 'family.npz')
POSTERIOR_ORDERS, ROW = (40, 80, 120), 1


def main():
    device = 'gpu' if sys.argv[-1] == 'gpu' else 'cpu'
    folders = [a for a in sys.argv[1:] if a != 'gpu']
    seg = np.load(SEG)
    truth = np.stack([seg['true_M_det'], seg['true_q'], seg['true_chi']], axis=1)
    bank = np.load(os.path.join(ROOT, 'data', 'banks', 'three_parameters.npy'))
    orders, lnlam, post = [], [], {}
    for folder in folders:
        store = search.load_redshift_store(folder, pinned=(device == 'gpu'), mmap=(device == 'cpu'))
        n = int(store.n_max)
        values = []
        for k in range(truth.shape[0]):                       # each segment at its own true template
            b = search.mass_bank(truth[k][None, :], SEG, 'cpu')
            values.append(search.pe_intrinsic(SEG, b, store, rows=[k], prior='madau-dickinson')['ln_lambda'][0, 0])
        orders.append(n); lnlam.append(values)
        print('n_max %3d: ln Lambda at the true template %s' % (n, np.round(values, 2)), flush=True)
        if n in POSTERIOR_ORDERS:
            L = np.concatenate([search.pe_intrinsic(SEG, search.mass_bank(bank[p], SEG, device), store, rows=[ROW],
                                                    prior='madau-dickinson', resident=(device == 'gpu'))['ln_lambda'][0]
                                for p in np.array_split(np.arange(len(bank)), 10)])
            post[n] = L
        del store
        if device == 'gpu':
            search.store.free()
    # the posterior as samples spread uniformly inside the cells of the bank (prior flat in M_det, q, chi), as a
    # smoothed histogram in (M_c, q, chi)
    M, q, chi = bank.T
    width = {}
    cell = np.ones(len(bank))
    for name, axis in (('M', M), ('q', q), ('chi', chi)):
        values = np.unique(axis); width[name] = np.gradient(values)[np.searchsorted(values, axis)]; cell *= width[name]
    rng = np.random.default_rng(6)
    samples = {}
    for n, L in post.items():
        p = cell * np.exp(L - L.max()); p /= p.sum()
        k = rng.choice(len(bank), size=300000, p=p)
        Ms, qs = M[k] + (rng.random(k.size) - 0.5) * width['M'][k], q[k] + (rng.random(k.size) - 0.5) * width['q'][k]
        samples[n] = np.stack([chirp_mass(Ms, qs), qs, chi[k] + (rng.random(k.size) - 0.5) * width['chi'][k]], 1)
    allS = np.concatenate(list(samples.values()))
    lo, hi = np.percentile(allS, 0.2, axis=0), np.percentile(allS, 99.8, axis=0)
    rngs = [(a - 0.08 * (b - a), b + 0.08 * (b - a)) for a, b in zip(lo, hi)]
    cubes = []
    for n in POSTERIOR_ORDERS:
        H, edges = np.histogramdd(samples[n], bins=[48, 36, 36], range=rngs)
        cubes.append(gaussian_filter(H, 1.0))
    ctr = [0.5 * (e[1:] + e[:-1]) for e in edges]
    order = np.argsort(orders)
    np.savez_compressed(os.path.join(ROOT, 'data', 'results', 'convergence.npz'),
                        nmax=np.array(orders, float)[order], lnlam=np.array(lnlam).T[:, order], rho_opt=seg['snr'],
                        post_nmax=np.array(POSTERIOR_ORDERS), post_cube=np.array(cubes),
                        post_Mc=ctr[0], post_q=ctr[1], post_chi=ctr[2],
                        true_Mc=chirp_mass(truth[ROW, 0], truth[ROW, 1]), true_q=truth[ROW, 1], true_chi=truth[ROW, 2],
                        post_rho_opt=float(seg['snr'][ROW]))
    print('written data/results/convergence.npz')


if __name__ == '__main__':
    main()
