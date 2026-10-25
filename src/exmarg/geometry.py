"""
helpers that turn W tables into the curves of the paper's figures.

  display_rows(tau, W, keys, n_show)       the tables on a coarse delay grid, as densities per unit delay
  frequency_slices(nside, n_psi, n_chi)    the weak-signal GORFs in frequency, Gamma^(2) and Gamma^(4) (Fig. 5)
  truncation_deficit(u_grid, n_max_list)   the noise-mean deficit eps_{n_max}(u1, u2) of the truncated series (Fig. 14)
"""
import healpy as hp
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import gammainc

from .detectors import Fplus_Fcross, basis, c, x
from .tables import _quad, _responses, rebin
from . import priors


def display_rows(tau, W, keys, n_show):
    """W[key] rebinned onto n_show delays and divided by the number of fine bins in each coarse bin"""
    tau_show = np.linspace(tau.min(), tau.max(), n_show)
    half = 0.5 * (tau_show[1] - tau_show[0])
    edges = np.concatenate([[tau_show[0] - half], 0.5 * (tau_show[1:] + tau_show[:-1]), [tau_show[-1] + half]])
    occupancy = np.histogram(tau, edges)[0].astype(float)
    coarse = rebin(tau, W, tau_show)
    return tau_show, np.array([coarse[tuple(k)] / occupancy for k in keys])


def frequency_slices(nside=32, n_psi=24, n_chi=24):
    """Gamma^(2)_HL(f,f), Gamma^(4)_HLHL(f,f,f,f), Gamma^(4)_HHHL(f,f,f,f) on f1, and the surfaces
    Gamma^(4)_HLHL(fa,fa,fc,fc) and Gamma^(4)_HHLL(fa,fc,fc,fa) on f2 x f2, in the weak-signal limit"""
    theta, phi = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    psi = np.arange(n_psi) * (np.pi / n_psi)
    cos_iota, weights = leggauss(n_chi)
    weights = weights / 2.0 / n_psi
    R = {}
    for I in 'HL':
        fp, fx = Fplus_Fcross(I, np.repeat(theta, n_psi), np.repeat(phi, n_psi), np.tile(psi, theta.size))
        fp, fx = fp.reshape(theta.size, n_psi), fx.reshape(theta.size, n_psi)
        R[I] = 0.5 * (1 + cos_iota**2)[None, None, :] * fp[:, :, None] + 1j * cos_iota[None, None, :] * fx[:, :, None]
    pair = R['H'] * np.conj(R['L'])
    w = weights[None, None, :]
    a2 = (pair * w).sum(axis=(1, 2))
    a4 = (pair**2 * w).sum(axis=(1, 2))
    a31 = (np.abs(R['H'])**2 * pair * w).sum(axis=(1, 2))
    a22 = (np.abs(R['H'])**2 * np.abs(R['L'])**2 * w).sum(axis=(1, 2))
    delay = basis(theta, phi)[0] @ (x['H'] - x['L']) / c

    f1 = np.arange(0.0, 256.5, 0.5)
    g2 = (a2[:, None] * np.exp(2j * np.pi * f1[None, :] * delay[:, None])).mean(axis=0)
    g4_hlhl = (a4[:, None] * np.exp(4j * np.pi * f1[None, :] * delay[:, None])).mean(axis=0)
    g4_hhhl = (a31[:, None] * np.exp(2j * np.pi * f1[None, :] * delay[:, None])).mean(axis=0)

    f2 = np.logspace(1.0, 3.0, 320)
    fsum = np.arange(0.0, 2 * f2.max() + 1.0, 0.5)
    g4_sum = (a4[:, None] * np.exp(2j * np.pi * fsum[None, :] * delay[:, None])).mean(axis=0)
    surf = np.interp(f2[None, :] + f2[:, None], fsum, g4_sum.real)
    fdif = np.arange(0.0, f2.max() + 1.0, 0.5)
    g4_dif = (a22[:, None] * np.exp(2j * np.pi * fdif[None, :] * delay[:, None])).mean(axis=0)
    surfx = np.interp(np.abs(f2[None, :] - f2[:, None]), fdif, g4_dif.real)
    return f1, g2, g4_hlhl, g4_hhhl, f2, surf, surfx


def truncation_deficit(u_grid, n_max_list, nside=32, n_max_quadrature=120):
    """eps_{n_max}(u1, u2) = < P[Poisson(A y^2) > n_max] > over the orientation and the Madau-Dickinson distance prior,
    on u_grid x u_grid, indexed [i_u1, i_u2]; returns {n_max: array}"""
    lny, p_lny = priors._LNY, priors._P_LNY
    p_lny = p_lny / np.trapezoid(p_lny, lny)
    y2 = np.exp(2 * lny)
    # histogram of (|R_1|^2, |R_2|^2) over the orientation quadrature of the tables
    npix = hp.nside2npix(nside)
    psi, cg, wq = _quad(n_max_quadrature, None, None)
    edges = np.linspace(0.0, 1.0 + 1e-9, 161)
    H = np.zeros((160, 160))
    for i0 in range(0, npix, 64):
        th, ph = hp.pix2ang(nside, np.arange(i0, min(i0 + 64, npix)))
        R = _responses(th, ph, psi, cg)
        r1, r2 = (np.abs(R[1])**2).ravel(), (np.abs(R[2])**2).ravel()
        w = np.broadcast_to(wq[None, None, :], np.abs(R[1]).shape).ravel() / npix
        H += np.histogram2d(r1, r2, bins=[edges, edges], weights=w)[0]
    centers = 0.5 * (edges[1:] + edges[:-1])
    filled = H > 0
    c1 = np.broadcast_to(centers[:, None], H.shape)[filled]
    c2 = np.broadcast_to(centers[None, :], H.shape)[filled]
    hw = H[filled]
    A_grid = np.geomspace(1e-3, 400.0, 900)
    out = {}
    for n_max in n_max_list:
        G = np.array([np.trapezoid(gammainc(n_max + 1, a * y2) * p_lny, lny) for a in A_grid])
        lnG = np.log(np.maximum(G, 1e-300))
        eps = np.empty((u_grid.size, u_grid.size))
        for i, u1 in enumerate(u_grid):
            A = u1 * c1[None, :] + u_grid[:, None] * c2[None, :]
            eps[i] = (hw[None, :] * np.exp(np.interp(np.log(A), np.log(A_grid), lnG))).sum(1) / hw.sum()
        out[n_max] = eps
    return out
