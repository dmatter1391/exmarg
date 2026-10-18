"""
probability-probability (PP) test of the likelihood: injections drawn from the prior, a grid posterior for each, and the
credible level at which the true value lies.

    levels = exmarg.pp.run(tables, n_events=30, Mc_range=(26.5, 29.5), q_range=(0.6, 1.0), chi_range=(-0.2, 0.3))

The prior of the injections and of the analysis is the same: chirp mass, mass ratio and spin uniform in the box, isotropic
sky and orientation, uniform phase and arrival time, and the distance uniform in volume out to d_max_mpc.
"""
from multiprocessing import Pool

import numpy as np

from .data import make_data, make_event
from .evaluate import SumEvaluator, mc_q_to_m1m2


def credible_level(grid, marginal, truth):
    """fraction of the posterior below the true value, from a marginal on a grid"""
    cdf = (np.cumsum(marginal) - 0.5 * marginal) / marginal.sum()
    return float(np.interp(truth, grid, cdf))


def one_event(tables, seed, Mc_grid, q_grid, chi_grid, d_max_mpc, T=4.0, fmin=25.0, fmax=512.0):
    """draw one injection from the prior, compute its grid posterior, return the credible levels of (Mc, q, chi)"""
    rng = np.random.default_rng(seed)
    true_Mc = rng.uniform(Mc_grid[0], Mc_grid[-1])
    true_q = rng.uniform(q_grid[0], q_grid[-1])
    true_chi = rng.uniform(chi_grid[0], chi_grid[-1])
    sky = (np.arccos(rng.uniform(-1, 1)), rng.uniform(0, 2 * np.pi))
    m1, m2 = mc_q_to_m1m2(true_Mc, true_q)
    event = make_event(m1, m2, true_chi, true_chi, T, fmin, fmax, sky=sky, psi_s=rng.uniform(0, np.pi),
                       chi_s=rng.uniform(-1, 1), d_max_mpc=d_max_mpc)
    y = event.ymin * rng.uniform() ** (-1.0 / 3.0)               # uniform in volume out to d_max_mpc
    data = make_data(event, seed=int(rng.integers(2**31)), inject=True, phi_s=rng.uniform(0, 2 * np.pi),
                     t_s=rng.uniform(0, T), y_s=y)
    likelihood = SumEvaluator(event, data, tables, tc=T / 2, dt=T / 2)
    ln_lambda = np.zeros((Mc_grid.size, q_grid.size, chi_grid.size))
    for i, Mc in enumerate(Mc_grid):
        for j, q in enumerate(q_grid):
            a, b = mc_q_to_m1m2(Mc, q)
            for k, chi in enumerate(chi_grid):
                ln_lambda[i, j, k] = likelihood.lnlam(a, b, chi)
    weights = np.gradient(Mc_grid)[:, None, None] * np.gradient(q_grid)[None, :, None] * np.gradient(chi_grid)[None, None, :]
    posterior = weights * np.exp(ln_lambda - ln_lambda.max())
    return (credible_level(Mc_grid, posterior.sum(axis=(1, 2)), true_Mc),
            credible_level(q_grid, posterior.sum(axis=(0, 2)), true_q),
            credible_level(chi_grid, posterior.sum(axis=(0, 1)), true_chi),
            float(y * event.rho_mf))


def run(tables, n_events, Mc_range, q_range, chi_range, d_max_mpc=4000.0, n_grid=(13, 9, 11), seed=1, processes=32):
    """credible levels of n_events injections, one injection per process, distances uniform in volume out to d_max_mpc;
    returns an array (n_events, 4): the Mc, q, chi levels and the network SNR"""
    Mc_grid = np.linspace(Mc_range[0], Mc_range[1], n_grid[0])
    q_grid = np.linspace(q_range[0], q_range[1], n_grid[1])
    chi_grid = np.linspace(chi_range[0], chi_range[1], n_grid[2])
    jobs = [(tables, seed * 100000 + k, Mc_grid, q_grid, chi_grid, d_max_mpc) for k in range(n_events)]
    with Pool(processes) as pool:
        return np.array(pool.starmap(one_event, jobs))
