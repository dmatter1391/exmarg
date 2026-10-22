"""
probability-probability (PP) tests of the analysis: injections drawn from the prior of the analysis, each in its own
Gaussian noise, the posterior of each from exmarg.search, and the credible level at which the true value lies.

    store = exmarg.search.load_redshift_store(store_dir, pinned=True)
    levels = exmarg.pp.run_intrinsic(store, 300, device='gpu')     # (M_c, q, chi), distance integrated
    levels = exmarg.pp.run_redshift(store, 592, bank, device='gpu')  # (m1_source, z) of equal-mass binaries without spin

The prior of the injections and of the analysis is the same: the intrinsic parameters uniform in their box, isotropic sky
and orientation, uniform phase and arrival time over the segment, and the distance prior of exmarg.priors
('madau-dickinson' by default, the distance distribution of the simulated year; 'volume' is uniform in volume).
"""

import numpy as np

from .detectors import S, Fplus_Fcross, c, x, basis
from .profiles import profile
from .priors import distance_density
from .search import mass_bank, pe_intrinsic, pe_redshift
from .search.segments import D0


def chirp_mass(M, q):
    """chirp mass of a binary with total mass M and mass ratio q = m2/m1 <= 1"""
    return M * (q / (1 + q) ** 2) ** 0.6


def draw_y(rng, prior, ymin):
    """one draw of y = D0 / D_L from the distance prior"""
    density = distance_density(prior)
    if density is None:
        return ymin * rng.uniform() ** (-1.0 / 3.0)               # uniform in volume
    lny, p = density
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(lny))])
    return float(np.exp(np.interp(rng.uniform() * cdf[-1], cdf, lny)))


def simulate(n_events, seed, draw, draw_distance, T, fmin, fmax):
    """n_events segments, each with one injection, in Gaussian noise with the H noise curve in both detectors.
    draw(rng) gives the intrinsic parameters (M_det, q, chi) of an injection and draw_distance(rng) its
    y = D0 / D_L.  returns the segments (the dict read_segments takes), the injected (M_det, q, chi, y) and the
    optimal network SNR of each injection."""
    nu = np.arange(int(np.ceil(fmin * T)), int(np.floor(fmax * T)) + 1)
    f = nu / T
    sig2 = T * S('H', f) / 2.0
    d = {'H': np.empty((n_events, nu.size), complex), 'L': np.empty((n_events, nu.size), complex)}
    truth = np.empty((n_events, 4))
    snr = np.empty(n_events)
    for k in range(n_events):
        rng = np.random.default_rng(seed + k)
        M, q, chi = draw(rng)
        theta, phi = np.arccos(rng.uniform(-1, 1)), rng.uniform(0, 2 * np.pi)
        psi, cos_iota = rng.uniform(0, np.pi), rng.uniform(-1, 1)
        phase, t_arrival = rng.uniform(0, 2 * np.pi), rng.uniform(0, T)
        y = draw_distance(rng)
        nu_t, _, a1, psi0 = profile(M / (1 + q), M * q / (1 + q), chi, chi, T, fmin, fmax, 1.0)
        h = np.zeros(nu.size, complex)
        h[np.searchsorted(nu, nu_t)] = a1 / D0 * np.exp(1j * psi0)
        n_hat = basis(np.array([theta]), np.array([phi]))[0][0]
        rho2 = 0.0
        for I in 'HL':
            Fp, Fx = Fplus_Fcross(I, np.array([theta]), np.array([phi]), np.array([psi]))
            R = complex((0.5 * (1 + cos_iota ** 2) * Fp + 1j * cos_iota * Fx)[0])
            tau = n_hat @ (x[I] - x['L']) / c
            signal = y * R * np.exp(-1j * phase) * h * np.exp(-2j * np.pi * f * (t_arrival + tau))
            g = rng.standard_normal((2, nu.size))
            d[I][k] = np.sqrt(sig2 / 2.0) * (g[0] + 1j * g[1]) + signal
            rho2 += 2.0 * np.sum(np.abs(signal) ** 2 / sig2)
        truth[k] = M, q, chi, y
        snr[k] = np.sqrt(rho2)
    segments = dict(d_H=d['H'], d_L=d['L'], f=f, nu=nu, sig2=sig2, T=T, fmin=fmin, fmax=fmax)
    return segments, truth, snr


def credible_level(grid, marginal, truth):
    """fraction of the posterior below the true value, from a marginal on a grid"""
    cdf = (np.cumsum(marginal) - 0.5 * marginal) / marginal.sum()
    return float(np.interp(truth, grid, cdf))


def run_intrinsic(store, n_events, M_range=(20.0, 200.0), q_range=(0.3, 1.0), chi_range=(-0.6, 0.6),
                  prior='madau-dickinson', ymin=0.1267, n_grid=(40, 24, 20), n_samples=20000, seed=1, device='cpu',
                  T=4.0, fmin=20.0, fmax=512.0, resident=False):
    """credible levels of (M_c, q, chi) for n_events injections, uniform in the box of (M_det, q, chi);
    returns an array (n_events, 4): the three levels and the optimal network SNR of the injection.
    The analysis grid is uniform in (M_det, q, chi); the level of M_c comes from posterior samples spread uniformly
    inside the cells of the grid."""
    draw = lambda rng: (rng.uniform(*M_range), rng.uniform(*q_range), rng.uniform(*chi_range))
    segments, truth, snr = simulate(n_events, seed * 100000, draw, lambda rng: draw_y(rng, prior, ymin), T, fmin, fmax)
    axes = [np.linspace(*r, n) for r, n in zip((M_range, q_range, chi_range), n_grid)]
    width = [np.gradient(a) for a in axes]
    M, q, chi = [g.ravel() for g in np.meshgrid(*axes, indexing='ij')]
    iM, iq, ic = [g.ravel() for g in np.meshgrid(*[np.arange(n) for n in n_grid], indexing='ij')]
    bank = mass_bank(np.stack([M, q, chi], axis=1), segments, device)
    ln_lambda = pe_intrinsic(segments, bank, store, prior=prior, ymin=ymin, resident=resident)['ln_lambda']
    cell = width[0][iM] * width[1][iq] * width[2][ic]
    rng = np.random.default_rng(seed)
    levels = np.empty((n_events, 4))
    for k in range(n_events):
        post = cell * np.exp(ln_lambda[k] - ln_lambda[k].max())
        pick = rng.choice(post.size, size=n_samples, p=post / post.sum())
        Ms = M[pick] + (rng.random(n_samples) - 0.5) * width[0][iM[pick]]
        qs = q[pick] + (rng.random(n_samples) - 0.5) * width[1][iq[pick]]
        cs = chi[pick] + (rng.random(n_samples) - 0.5) * width[2][ic[pick]]
        Mt, qt, ct = truth[k, :3]
        levels[k] = (np.mean(chirp_mass(Ms, qs) < chirp_mass(Mt, qt)), np.mean(qs < qt), np.mean(cs < ct), snr[k])
    return levels


def run_redshift(store, n_events, bank, Mdet_range=(20.0, 200.0), prior='madau-dickinson', ymin=0.1267, seed=2,
                 device='cpu', T=8.0, fmin=20.0, fmax=512.0, zmax=10.0, m1_range=(0.5, 150.0), batch=32, resident=False):
    """credible levels of (m1_source, z) for n_events equal-mass binaries without spin, uniform in the detector-frame
    total mass on Mdet_range and with the distance from the distance prior; bank is the one-column bank of
    detector-frame total masses of pe_redshift.  returns an array (n_events, 3): the m1 and z levels and the
    optimal network SNR.  The (m1, z) grid of pe_redshift covers the whole prior (zmax, m1_range)."""
    from astropy.cosmology import Planck18
    draw = lambda rng: (rng.uniform(*Mdet_range), 1.0, 0.0)
    segments, truth, snr = simulate(n_events, seed * 100000, draw, lambda rng: draw_y(rng, prior, ymin), T, fmin, fmax)
    templates = mass_bank(bank, segments, device)
    m1 = np.geomspace(m1_range[0], m1_range[1], 1600)
    z = np.linspace(0.02, zmax, 1600)
    # the prior in (m1, z): the density of ln y = ln(D0 / D_L(z)) carried to z, times the Jacobian 2 (1 + z) of
    # M_det = 2 m1 (1 + z), on the cells of the grid
    lny = np.log(D0 / Planck18.luminosity_distance(z).value)
    density = distance_density(prior)
    if density is None:
        p_lny = np.where(lny >= np.log(ymin), 3.0 * np.exp(-3.0 * (lny - np.log(ymin))), 0.0)
    else:
        p_lny = np.interp(lny, density[0], density[1], left=0.0, right=0.0)
    p_z = p_lny * np.abs(np.gradient(lny, z))
    Md = 2.0 * m1[:, None] * (1.0 + z[None, :])
    inside = (Md >= Mdet_range[0]) & (Md <= Mdet_range[1])
    prior_mz = np.where(inside, p_z[None, :] * 2.0 * (1.0 + z[None, :]), 0.0) * np.gradient(m1)[:, None] * np.gradient(z)[None, :]
    zz = np.linspace(0.0, 12.0, 24001)
    z_true = np.interp(D0 / truth[:, 3], Planck18.luminosity_distance(zz).value, zz)
    levels = np.empty((n_events, 3))
    for k in range(n_events):
        if k % batch == 0:                                         # ln Lambda(m1, z) of the next batch of segments
            rows = np.arange(k, min(k + batch, n_events))
            res = pe_redshift(segments, templates, store, rows=rows, ymin=ymin, zmax=zmax, m1_range=m1_range,
                              resident=resident)
            assert np.allclose(res['m1'], m1) and np.allclose(res['z'], z)
        L = res['ln_lambda'][k % batch]
        ok = np.isfinite(L) & (prior_mz > 0)
        post = np.zeros_like(L)
        post[ok] = prior_mz[ok] * np.exp(L[ok] - L[ok].max())
        m1_true = truth[k, 0] / (2.0 * (1.0 + z_true[k]))
        levels[k] = (credible_level(m1, post.sum(axis=1), m1_true), credible_level(z, post.sum(axis=0), z_true[k]), snr[k])
    return levels
