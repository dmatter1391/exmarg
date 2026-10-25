"""Stage 1 of the tables, the GPU (cupy) quadrature of the GORFs W at the nodes of s = u y^2 (used by build_w.py).

Per sky-pixel chunk the builder computes the response monomials and the exponential damping and sums them over the
(psi, chi) quadrature, in fp64 on the GPU (fp32 underflows at high order, see _chunk_vc_s_gpu).  The integrand is kept
O(1): the damping e^{-s(|R1|^2+|R2|^2)} <= 1 is bounded and the power ladder is normalised by |R|_max = sqrt(2), so every
monomial (|R1|/rmax)^p (|R2|/rmax)^q <= 1.  The removed factor rmax^{2n} = 2^n and the storage offset s^n/n! are folded
into a single per-(node, order) fp64 scale applied on the host before the tau deposit.

Speed comes from three things, in order of importance:

  1. the exp damping is order-INDEPENDENT (Bw_base, built once) and
     cos(m dph) depends only on the azimuthal index m (Bwm reused
     across every n >= m);
  2. the |R|^p ladder is stacked ONCE (P1s[k] = (|R1|/rmax)^k) and the
     monomials are gathered by fancy indexing, no Python power loop;
  3. for a fixed m ALL orders n >= m are reduced together — the
     (order, sum) columns are concatenated and pushed through one
     batched GEMM per COLCHUNK block, so a full chunk is ~150 GEMMs
     instead of ~7000 per-(n,m) matmuls.

    tau, W = w_table_s_gpu(s_array, s_array, n_max, nside=..., ...)
"""
import numpy as np
import cupy as cp
import healpy as hp
from scipy.special import gammaln

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from exmarg.detectors import c, x, basis, Fplus_Fcross
from exmarg.tables import _quad, _tau_grid, _deposit, _collect, patterns

RMAX2 = 2.0            # max |R|^2 = Ap^2 Fp^2 + cg^2 Fx^2 <= 1 + 1
LNRMAX2 = np.log(RMAX2)
LN_CAP = 700.0         # matches exmarg.numerics; never fires while the
                       # ladder stays below s ~ 8000 (ln_scale < 700 at
                       # n_max=120), a purely defensive overflow guard
COLCHUNK = 256         # monomial columns per GEMM (fp64: device-memory bound)


def _responses_gpu(th, ph, psi, cg):
    """R_I(nhat, psi, chi) on GPU; antenna patterns from the host
    geometry (cheap), the (psi, chi) dressing on device.  complex64."""
    c2p = cp.asarray(np.cos(2 * psi), cp.float32)
    s2p = cp.asarray(np.sin(2 * psi), cp.float32)
    Ap = cp.asarray(0.5 * (1 + cg**2), cp.float32)
    cgg = cp.asarray(cg, cp.float32)
    R = {}
    for j, I in enumerate('HL'):
        Fp0, Fx0 = Fplus_Fcross(I, th, ph, np.zeros(th.size))
        Fp0 = cp.asarray(Fp0, cp.float32); Fx0 = cp.asarray(Fx0, cp.float32)
        Fp = Fp0[:, None] * c2p[None, :] - Fx0[:, None] * s2p[None, :]
        Fx = Fp0[:, None] * s2p[None, :] + Fx0[:, None] * c2p[None, :]
        R[j + 1] = (Ap[None, None, :] * Fp[:, :, None]
                    + 1j * cgg[None, None, :] * Fx[:, :, None]
                    ).astype(cp.complex64)
    return R


def _blocks(n_max, keys, kidx):
    """chunk-invariant column plan: for each azimuthal m, the |R1| and
    |R2| power indices and destination key columns for every order
    n >= m, split into COLCHUNK blocks.  key (n,a,b) has p=a+b,
    q=2n-a-b, m=a-b, so a=(p+m)/2, b=(p-m)/2."""
    plan = []                                       # (m, p_blk, q_blk, col_blk)
    for m in range(n_max + 1):
        ps, qs, cols = [], [], []
        for n in range(m, n_max + 1):
            for p in range(m, 2 * n - m + 1, 2):
                ps.append(p); qs.append(2 * n - p)
                cols.append(kidx[(n, (p + m) // 2, (p - m) // 2)])
        ps = np.array(ps); qs = np.array(qs); cols = np.array(cols)
        for lo in range(0, ps.size, COLCHUNK):
            hi = min(lo + COLCHUNK, ps.size)
            plan.append((m, cp.asarray(ps[lo:hi]), cp.asarray(qs[lo:hi]),
                         cp.asarray(cols[lo:hi])))
    return plan


def _chunk_vc_s_gpu(th, ph, s1v, s2v, n_max, psi, cg, wq, plan, n_key_max):
    """raw quadrature reduction for one sky-pixel block, fp64.

    fp64 is REQUIRED, not an optimisation to drop: the normalised
    monomial (|R|/sqrt2)^{2n} is genuinely tiny at high order (down to
    ~1e-256 at the damping saddle for s~4000, n~120), far below the
    fp32 floor 1e-38 -- in fp32 every such key underflowed to 0 and the
    stored table lost the bulk of the series at loud nodes.  fp64
    (floor 1e-308) represents them; the s^n/n! 2^n scale, applied by
    the caller, then lifts them back to O(1).  Returns (npix, n_nodes,
    n_keys) float64."""
    npx = th.size
    R = _responses_gpu(th, ph, psi, cg)
    S1 = (cp.abs(R[1])**2).reshape(npx, -1).astype(cp.float64)  # |R1|^2<=2
    S2 = (cp.abs(R[2])**2).reshape(npx, -1).astype(cp.float64)
    m1 = cp.sqrt(S1 / RMAX2)                          # normalised <= 1
    m2 = cp.sqrt(S2 / RMAX2)
    dph = (cp.angle(R[1]) - cp.angle(R[2])).reshape(npx, -1).astype(cp.float64)
    wqf = cp.asarray(np.broadcast_to(wq[None, None, :], R[1].shape)
                     .reshape(npx, -1).copy(), cp.float64)

    # power ladder with the power axis LAST so P1s[:, :, idx] gives a
    # (P, Q, Ncol) block ready for matmul without a transpose copy
    Q = m1.shape[1]
    P1s = cp.empty((npx, Q, 2 * n_max + 1), cp.float64)
    P2s = cp.empty_like(P1s)
    P1s[:, :, 0] = 1.0; P2s[:, :, 0] = 1.0
    for k in range(1, 2 * n_max + 1):
        P1s[:, :, k] = P1s[:, :, k - 1] * m1
        P2s[:, :, k] = P2s[:, :, k - 1] * m2

    J = np.atleast_1d(s1v).size; K = n_key_max
    Vc = cp.zeros((npx, J, K), cp.float64)
    s1v = np.atleast_1d(np.asarray(s1v, np.float64))
    s2v = np.atleast_1d(np.asarray(s2v, np.float64))
    Bw_base = cp.stack([                              # (P, J, Q), <= 1
        cp.exp(-(float(s1v[j]) * S1 + float(s2v[j]) * S2))
        for j in range(J)], axis=1) * wqf[:, None, :]

    m_cur = -1; Bwm = None
    for (m, p_blk, q_blk, col_blk) in plan:
        if m != m_cur:
            Bwm = Bw_base * cp.cos(m * dph)[:, None, :]
            m_cur = m
        Rp = P1s[:, :, p_blk]                          # (P, Q, Ncol) copy
        Rp *= P2s[:, :, q_blk]                         # in place (fresh copy)
        Vc[:, :, col_blk] = cp.matmul(Bwm, Rp)        # (P, J, Ncol)
        del Rp
    return Vc                                          # stays on device


def w_table_s_gpu(s1, s2, n_max, tau=None, n_tau=4097, nside=32,
                  n_psi=None, n_chi=None, chunk=64, verbose=False):
    """table build at the nodes s: quadrature and deposit on the GPU in fp64, the dynamic-range scaling on the host
    (fp64).  s1 == s2 on the s-ladder."""
    scalar_s = np.ndim(s1) == 0
    psi, cg, wq = _quad(n_max, n_psi, n_chi)
    dx, tau, order, ts, dtau = _tau_grid(tau, n_tau)
    npix = hp.nside2npix(nside)
    keys = patterns(n_max)
    kidx = {k: j for j, k in enumerate(keys)}
    plan = _blocks(n_max, keys, kidx)
    s1v = np.atleast_1d(np.asarray(s1, float))
    n_nodes = 1 if scalar_s else len(s1)
    # fp64 per-(node, key) scale = s^n / n! * 2^n (folds rmax^{2n} and
    # the storage offset ln_offset(n, s, s) = ln n! + n ln s together)
    n_key = np.array([k[0] for k in keys], float)
    ln_scale = (n_key[None, :] * np.log(s1v)[:, None]
                - gammaln(n_key + 1.0)[None, :]
                + n_key[None, :] * LNRMAX2)           # (n_nodes, n_key)
    scale = np.exp(np.minimum(ln_scale, LN_CAP))

    # accumulate the UNSCALED reduction on the GPU in fp64 (the tau
    # deposit is a small sparse matmul, done on device).  fp64 is
    # mandatory: the reduction spans O(1) down to ~1e-256 across orders,
    # which fp32 cannot hold -- see _chunk_vc_s_gpu.  The huge s^n/n!
    # scale is kept off the accumulator and applied once at the end.
    Wraw = cp.zeros((n_nodes, tau.size, len(keys)), cp.float64)
    for lo in range(0, npix, chunk):
        hi = min(lo + chunk, npix)
        th, ph = hp.pix2ang(nside, np.arange(lo, hi))
        Vc = _chunk_vc_s_gpu(th, ph, s1, s2, n_max, psi, cg, wq, plan,
                             len(keys))                # (npix, J, K) device
        t_del = (basis(th, ph)[0] @ (x['H'] - x['L'])) / c
        P = _deposit(tau, order, ts, dtau, t_del, npix)
        Pg = cp.asarray(P.toarray(), cp.float64)       # (n_tau, npix), tiny
        for j in range(n_nodes):
            Wraw[j] += Pg @ Vc[:, j, :]
        del Vc
        if verbose:
            print('w_table_s_gpu: %u / %u pix' % (hi, npix), flush=True)
    if not bool(cp.isfinite(Wraw).all()):
        raise FloatingPointError('conditional W raw table non-finite')
    # apply the fp64 s^n / n! 2^n scale once, node by node on the host,
    # so the huge storage-offset factors are only ever formed in fp64
    Wgrid = np.empty((n_nodes, tau.size, len(keys)))
    for j in range(n_nodes):
        Wgrid[j] = cp.asnumpy(Wraw[j]).astype(np.float64) \
            * scale[j][None, :]
    if not np.isfinite(Wgrid).all():
        raise FloatingPointError('conditional W table non-finite')
    return tau, _collect(Wgrid, keys, scalar_s)
