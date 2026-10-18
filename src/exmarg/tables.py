"""
the W tables: everything precomputed before data exists.

  W^{(n)}_{ab}(tau) = < M_n(A) R_1^a conj(R_1)^b R_2^(n-a) conj(R_2)^(n-b)
                        delta(tau - n_hat.(x_1-x_2)/c) >_{sky, psi, chi}
  A                 = u1 |R_1|^2 + u2 |R_2|^2

in words: for every inter-detector delay tau, the average over sky
direction, polarization, inclination and distance of (detector
response monomial) x (distance damping).  the Taylor expansion of
the Bessel function, I0(2v) = sum v^{2n}/(n!)^2 exactly, is what
splits the likelihood into these data-independent tables and the
data; the truncation order N is the only approximation knob
(taylor_order picks it from data, noise_mean prices it).

storage is in OFFSET TAYLOR UNITS — rows carry 1/(n!)^2 times
exp(o_n) (numerics.ln_offset) — so every stored number is O(1) at
every order.  the arithmetic armor lives in numerics.py.

quadrature: HEALPix sky (nside 16 or finer), uniform psi (n_psi = 2N+3, exact), Gauss-
Legendre chi (n_chi = N+3).  chi -> -chi makes W real and (a,b)
symmetric; only b <= a stored.  u1, u2 arrays build a whole u-node
ladder in ONE sweep (responses and powers shared between nodes).
"""

import numpy as np
import healpy as hp
from numpy.polynomial.legendre import leggauss
from scipy.special import gammaln

from .detectors import c, x, basis, Fplus_Fcross
from .numerics import (A_FLOOR_X, bounded_exp, ln_moment_n,
                       ln_offset, moments)


def taylor(N):
    '''absolute Taylor coefficients 1/(n!)^2 (reference only; stores
    already carry them, runtime weights are ones)'''
    return np.exp(-2 * gammaln(np.arange(N + 1) + 1.0))


def taylor_order(qmax, u, margin=5.0):
    '''smallest converged truncation: terms ~ r^n/n!, r <= 2 qmax^2/u'''
    r = 2.0 * qmax**2 / u
    return int(np.ceil(r + margin * np.sqrt(max(r, 1.0))))


def patterns(n_max):
    '''the key list (n, a, b), b <= a'''
    return [(n, a, b) for n in range(n_max + 1)
            for a in range(n + 1) for b in range(a + 1)]


def _quad(n_max, n_psi, n_chi):
    '''psi grid, chi Gauss-Legendre nodes, combined weights'''
    if n_psi is None:
        n_psi = 2 * n_max + 3
    if n_chi is None:
        n_chi = n_max + 3
    psi = np.arange(n_psi) * (np.pi / n_psi)
    cg, cw = leggauss(n_chi)
    return psi, cg, cw / 2.0 / n_psi


def _responses(th, ph, psi, cg):
    '''detector responses R_I(n_hat, psi, chi) on the quadrature'''
    c2p, s2p = np.cos(2 * psi), np.sin(2 * psi)
    Ap = 0.5 * (1 + cg**2)
    R = {}
    for j, I in enumerate('HL'):
        Fp0, Fx0 = Fplus_Fcross(I, th, ph, np.zeros(th.size))
        Fp = Fp0[:, None] * c2p[None, :] - Fx0[:, None] * s2p[None, :]
        Fx = Fp0[:, None] * s2p[None, :] + Fx0[:, None] * c2p[None, :]
        R[j + 1] = (Ap[None, None, :] * Fp[:, :, None]
                    + 1j * cg[None, None, :] * Fx[:, :, None])
    return R


def _chunk_vc(th, ph, u1, u2, ymin, n_max, psi, cg, wq, keys, kidx):
    '''
    the integrand for one block of sky pixels, all keys, all u nodes.

    plan: responses and their |R| powers and phase cosines are u
    independent, built once; per (node, order) only the distance
    weight changes; per key one multiply and one quadrature sum.
    memory ~ (4 n_max + 2) * n_pix * n_psi * n_chi * 8 bytes — deep
    builds pass a small pixel chunk.

    Outputs (n_nodes, n_pix, n_keys)
    '''
    u1v = np.atleast_1d(np.asarray(u1, float))
    u2v = np.atleast_1d(np.asarray(u2, float))
    npx = th.size
    R = _responses(th, ph, psi, cg)
    S1, S2 = np.abs(R[1])**2, np.abs(R[2])**2
    m1 = np.sqrt(S1).reshape(npx, -1)
    m2 = np.sqrt(S2).reshape(npx, -1)
    dph = (np.angle(R[1]) - np.angle(R[2])).reshape(npx, -1)
    wqf = np.broadcast_to(wq[None, None, :],
                          R[1].shape).reshape(npx, -1)

    p1 = [np.ones_like(m1)]              # |R_1|^k, k = 0 .. 2N
    p2 = [np.ones_like(m2)]
    for k in range(2 * n_max):
        p1.append(p1[-1] * m1)
        p2.append(p2[-1] * m2)
    cosm = [np.cos(m * dph) for m in range(n_max + 1)]

    xs, lnAf = [], []                    # per node damping inputs
    for j in range(u1v.size):
        A = (u1v[j] * S1 + u2v[j] * S2).reshape(npx, -1)
        xj = np.clip(ymin**2 * A, A_FLOOR_X, None)
        xs.append(xj)
        lnAf.append(np.log(xj) - 2.0 * np.log(ymin))

    Vc = np.zeros((u1v.size, npx, len(keys)))
    for n in range(n_max + 1):
        # distance weight of this order, in storage units, per node
        Bw = np.stack([bounded_exp(
            ln_moment_n(n, xs[j], lnAf[j], ymin)
            - 2 * gammaln(n + 1.0)
            + ln_offset(n, u1v[j], u2v[j]))
            for j in range(u1v.size)])
        Bw *= wqf[None]
        for a in range(n + 1):
            for b in range(a + 1):
                Q = p1[a + b] * p2[2 * n - a - b] * cosm[a - b]
                Vc[:, :, kidx[(n, a, b)]] = np.einsum('jpq,pq->jp',
                                                      Bw, Q)
    if not np.isfinite(Vc).all():
        raise FloatingPointError('W integrand non-finite')
    return Vc


def _deposit(tau, order, ts, dtau, t_del, npix):
    '''linear deposit matrix of one chunk onto the sorted tau grid'''
    from scipy.sparse import csr_matrix
    n_tau = tau.size
    g = (t_del - ts[0]) / dtau
    j0 = np.clip(np.floor(g).astype(int), 0, n_tau - 2)
    fr = g - j0
    rows = order[np.concatenate([j0, j0 + 1])]
    cols = np.concatenate([np.arange(t_del.size), np.arange(t_del.size)])
    dat = np.concatenate([1.0 - fr, fr]) / npix
    return csr_matrix((dat, (rows, cols)), shape=(n_tau, t_del.size))


def _tau_grid(tau, n_tau):
    '''fine delay grid: range is the detector baseline light time'''
    dx = x['H'] - x['L']
    if tau is None:
        tau_max = np.linalg.norm(dx) / c
        tau = np.linspace(-tau_max, tau_max, n_tau)
    else:
        tau = np.asarray(tau, float)
    order = np.argsort(tau)
    ts = tau[order]
    return dx, tau, order, ts, ts[1] - ts[0]


def _collect(Wgrid, keys, scalar_u):
    '''grids -> per node key dicts with mirrored (b, a) entries'''
    out = []
    for g in Wgrid:
        W = {}
        for j, (n, a, b) in enumerate(keys):
            W[(n, a, b)] = g[:, j]
            if a != b:
                W[(n, b, a)] = g[:, j]
        out.append(W)
    return out[0] if scalar_u else out


def w_table_dl(u1, u2, ymin, n_max, tau=None, n_tau=4097, nside=32,
               n_psi=None, n_chi=None, chunk=128, verbose=False):
    '''
    serial build.  Outputs tau grid and the W dict (or a list of
    dicts when u1, u2 are node arrays), offset Taylor units.
    '''
    scalar_u = np.ndim(u1) == 0
    psi, cg, wq = _quad(n_max, n_psi, n_chi)
    dx, tau, order, ts, dtau = _tau_grid(tau, n_tau)
    npix = hp.nside2npix(nside)
    keys = patterns(n_max)
    kidx = {k: j for j, k in enumerate(keys)}
    n_nodes = 1 if scalar_u else len(u1)
    Wgrid = np.zeros((n_nodes, tau.size, len(keys)))

    for i0_pix in range(0, npix, chunk):
        pix = np.arange(i0_pix, min(i0_pix + chunk, npix))
        th, ph = hp.pix2ang(nside, pix)
        t_del = (basis(th, ph)[0] @ dx) / c
        Vc = _chunk_vc(th, ph, u1, u2, ymin, n_max, psi, cg, wq,
                       keys, kidx)
        P = _deposit(tau, order, ts, dtau, t_del, npix)
        for j in range(n_nodes):
            Wgrid[j] += P @ Vc[j]
        if verbose:
            print('w_table_dl: %u / %u pixels' % (pix[-1] + 1, npix),
                  flush=True)

    if not np.isfinite(Wgrid).all():
        raise FloatingPointError('W table has non-finite entries')
    return tau, _collect(Wgrid, keys, scalar_u)


def _par_chunk(args):
    '''one pixel chunk of the parallel build (same integrand)'''
    (lo, hi, u1, u2, ymin, n_max, nside, n_psi, n_chi) = args
    pix = np.arange(lo, hi)
    th, ph = hp.pix2ang(nside, pix)
    dx = x['H'] - x['L']
    t_del = (basis(th, ph)[0] @ dx) / c
    psi, cg, wq = _quad(n_max, n_psi, n_chi)
    keys = patterns(n_max)
    kidx = {k: j for j, k in enumerate(keys)}
    Vc = _chunk_vc(th, ph, u1, u2, ymin, n_max, psi, cg, wq,
                   keys, kidx)
    return lo, t_del, Vc


def w_table_par(u1, u2, ymin, n_max, tau=None, n_tau=4097, nside=32,
                n_psi=None, n_chi=None, chunk=64, workers=32,
                verbose=False, pix_range=None):
    '''parallel build: pixel chunks over a process pool, the master
    deposits.  identical output to the serial builder.  pix_range =
    (lo, hi) builds only that pixel slice — the deposit is linear
    and keeps the global 1/npix weight, so slices from separate runs
    SUM to the full table exactly (see scripts/merge_store.py).'''
    import os
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    from multiprocessing import Pool

    scalar_u = np.ndim(u1) == 0
    if n_psi is None:
        n_psi = 2 * n_max + 3
    if n_chi is None:
        n_chi = n_max + 3

    dx, tau, order, ts, dtau = _tau_grid(tau, n_tau)
    npix = hp.nside2npix(nside)
    keys = patterns(n_max)
    n_nodes = 1 if scalar_u else len(u1)
    Wgrid = np.zeros((n_nodes, tau.size, len(keys)))

    p_lo, p_hi = (0, npix) if pix_range is None else pix_range
    jobs = [(lo, min(lo + chunk, p_hi), u1, u2, ymin, n_max, nside,
             n_psi, n_chi) for lo in range(p_lo, p_hi, chunk)]
    done = 0
    with Pool(workers) as pool:
        for lo, t_del, Vc in pool.imap_unordered(_par_chunk, jobs):
            P = _deposit(tau, order, ts, dtau, t_del, npix)
            for j in range(n_nodes):
                Wgrid[j] += P @ Vc[j]
            done += 1
            if verbose:
                print('w_table_par: %u / %u chunks'
                      % (done, len(jobs)), flush=True)

    if not np.isfinite(Wgrid).all():
        raise FloatingPointError('W table has non-finite entries')
    return tau, _collect(Wgrid, keys, scalar_u)


def rebin(tau_fine, W_fine, tau_new):
    '''fine grid -> data grid; bin sums are conserved exactly'''
    order = np.argsort(tau_new)
    ts = tau_new[order]
    edges = np.concatenate([[ts[0] - 0.5 * (ts[1] - ts[0])],
                            0.5 * (ts[1:] + ts[:-1]),
                            [ts[-1] + 0.5 * (ts[-1] - ts[-2])]])
    idx = np.clip(np.searchsorted(edges, tau_fine) - 1, 0, ts.size - 1)
    out = {}
    for k, w in W_fine.items():
        acc = np.zeros(ts.size)
        np.add.at(acc, idx, w)
        new = np.empty_like(acc)
        new[order] = acc
        out[k] = new
    return out


def noise_mean(W, n_max, u1, u2):
    '''
    exact analytic noise expectation of the truncated score for a
    store built at (u1, u2): equals 1 for the untruncated series,
    the deficit prices the truncation.
    '''
    from math import lgamma
    lu1, lu2 = np.log(u1), np.log(u2)
    lum = min(lu1, lu2)
    E = 0.0
    for n in range(n_max + 1):
        for a in range(n + 1):
            w = W[(n, a, a)].sum()
            if w == 0.0:
                continue
            lg = (lgamma(n + 1) - lgamma(a + 1) - lgamma(n - a + 1)
                  + a * (lu1 - lum) + (n - a) * (lu2 - lum))
            E += np.sign(w) * np.exp(lg + np.log(abs(w)))
    return float(E)
