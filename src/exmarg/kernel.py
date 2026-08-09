"""
the per order resummed kernel: the extrinsic average precomputed per
data cell value, one table per expansion order so the coefficients
stay OUT of the tables.

  ln s_n             = ln mean_t sum_tau K_n(tau; |z1(t+tau)|, |z2(t)|, dphi)
  ln Lambda          = logsumexp_n(ln c_n + ln s_n)   (c_n = 1 exact)
  K_n(tau; q1, q2, dphi)
                     = < M_n(A) |R1 q1 + R2 q2 e^{i dphi}|^{2n} >_{delay tau}
                       / (n!)^2
                     = sum_{ab} C(n,a) C(n,b) W^(n)_ab(tau)
                       q1^{a+b} q2^{2n-a-b} cos((a-b) dphi)

W input is in the OFFSET Taylor units of wtables (rows carry
exp(o_n)); the build removes o_n and adds the per cell exponent
bound g_n(q1, q2) = 2n max(ln q1, ln q2) so every linear
intermediate is bounded: the matmul runs on exp(powers - g_n) <= 1,
and ln K_n = ln(sum) + g_n - o_n is stored.  the runtime is pure log
space (logsumexp over cells and orders), so arbitrarily loud data
cannot overflow — there is no snr wall anywhere.

each K_n is positive and ln K_n is near linear in (ln q1, ln q2)
(the dominant monomial), so trilinear interpolation on the log grid
is accurate.  the coefficients, and with them the truncation n_max,
enter only in the final logsumexp: production, n_max scans and
calibration share ONE table with no recompilation.

templates at a different u reuse the table through the rescaling
z -> sqrt(u_node/u) z, which shifts ln q by a constant.  CAVEAT:
this maps the fixed-ymin distance prior onto an effective cutoff
ymin sqrt(u_node/u), so a single node is exact only at u = u_node;
evaluate.Evaluator brackets u between two nodes and interpolates to
cancel the first order prior shift.

the (q1, q2) axes are logarithmic; dphi = arg(z1 conj(z2)) is
uniform.  memory: (N + 1) tables of (n_tau, nq, nq, nphi) float32.
"""

import math as _math
import time as _time

import numpy as np
from numba import njit, prange
from scipy.special import comb

from .numerics import ln_offset, logsumexp


class Kernel:

    def __init__(self, u_node, tau_idx, n_time, T, keys, W, qmax,
                 nq=110, nphi=96, q_span=300.0, verbose=False,
                 store=None):
        '''
        Inputs
        * keys (K, 3) int, W (K, n_tau): W tables of the node in
          OFFSET Taylor units (see wtables), b <= a rows only; with
          such stores the exact series weight vector is ones
        * qmax: top of the q grid (largest |z| the table must cover,
          in NODE units, i.e. after the sqrt(u_node/u) rescaling)
        * store: optional (N+1, n_tau, nq, nq, nphi) float32 memmap;
          each compiled order is STREAMED into store[n] instead of
          being kept in RAM (builder mode: peak memory one order,
          the object is then NOT usable for evaluation)
        '''
        self.u_node = float(u_node)
        self.tau_idx = tau_idx
        self.n_time = int(n_time)
        self.T = float(T)
        self.nq, self.nphi = nq, nphi

        self.N = int(keys[:, 0].max())
        qg = np.geomspace(qmax / q_span, qmax, nq)
        self.lnq = np.log(qg)
        self.qlo, self.qhi = qg[0], qg[-1]
        phi_g = np.arange(nphi) * (2 * np.pi / nphi)
        n_tau = W.shape[1]

        by_n = {n: [] for n in range(self.N + 1)}
        for j, (n, a, b) in enumerate(keys):
            by_n[int(n)].append((j, int(a), int(b)))

        lq1 = self.lnq[:, None]
        lq2 = self.lnq[None, :]
        self.lnK = []
        t0 = _time.time()
        for n in range(self.N + 1):
            if verbose and (n % 10 == 0 or n == self.N):
                print('  kernel u=%g: order %d/%d (%.0f s)'
                      % (self.u_node, n, self.N, _time.time() - t0),
                      flush=True)
            o_n = ln_offset(n, self.u_node, self.u_node)
            g_n = 2.0 * n * np.maximum(lq1, lq2)
            groups = {}
            for (j, a, b) in by_n[n]:
                m = a - b
                coeff = comb(n, a) * comb(n, b) * (2.0 if m else 1.0)
                groups.setdefault(m, []).append((j, a, b, coeff))
            # phase axis via ONE matmul: the per-m broadcast add
            # rewrote the (tau, q, q, phi) block once per phase
            # class (~18 TB of traffic over a full build); collect
            # the small C_m blocks first, then a single GEMM against
            # the cosine matrix writes the block once.
            ms = sorted(groups)
            C = np.empty((len(ms), n_tau * nq * nq))
            for i, m in enumerate(ms):
                rows = groups[m]
                Wm = np.array([cf * W[j] for (j, a, b, cf) in rows])
                # exponent bounded above by zero: exp <= 1
                P = np.exp(np.stack(
                    [(a + b) * lq1 + (2 * n - a - b) * lq2 - g_n
                     for (j, a, b, cf) in rows]))
                C[i] = (Wm.T @ P.reshape(len(rows), -1)).ravel()
            cosM = np.cos(np.outer(ms, phi_g))
            Kn = (C.T @ cosM).reshape(n_tau, nq, nq, nphi)
            if not np.isfinite(Kn).all():
                raise FloatingPointError(
                    'kernel accumulation non-finite at order %u' % n)
            lnKn = (np.log(np.maximum(Kn, 1e-300))
                    + (g_n - o_n)[None, :, :, None]).astype(np.float32)
            if store is not None:
                store[n] = lnKn
            else:
                self.lnK.append(lnKn)

    def lns(self, z, u, t_center, half_width):
        '''
        ln s_n for a template with weight u, using this node's table
        through the sqrt(u_node / u) argument rescaling; pure log
        space, no loudness limit.  Outputs (N + 1,) float
        '''
        ti, corners, ln_nt = _gather_setup(self, z, u, t_center,
                                           half_width)
        out = np.empty(self.N + 1)
        for n in range(self.N + 1):
            acc = 0.0
            for (ia, ib, ic, w) in corners:
                acc = acc + w * self.lnK[n][ti, ia, ib, ic]
            out[n] = logsumexp(acc.ravel()) - ln_nt
        return out

    def lnlam(self, z, u, t_center, half_width, cn):
        '''
        ln Lambda = logsumexp_n(ln cn[n] + ln s_n);
        len(cn) - 1 <= N sets the truncation, ones = exact series
        '''
        lns = self.lns(z, u, t_center, half_width)[:len(cn)]
        return logsumexp(np.log(np.asarray(cn, float)) + lns)

    def summed(self, n_maxes):
        '''
        collapse the order axis ONCE for a set of truncations:
        ln S_m = ln sum_{n <= m} K_n, accumulated in one pass with a
        snapshot at every requested m.  the sum is over table VALUES
        so the sqrt(u_node/u) q rescaling still flows through the
        shared lookup point; the per-order tables stay c_n-free, the
        summed table is a derived cache for the runtime.

        Outputs {m: SumKernel}
        '''
        want = set(int(m) for m in n_maxes)
        bad = [m for m in want if m > self.N]
        if bad:
            raise ValueError('n_max %s > compiled N=%d' % (bad, self.N))
        out = {}
        lnS = np.full(self.lnK[0].shape, -np.inf, dtype=np.float32)
        for n in range(self.N + 1):
            np.logaddexp(lnS, self.lnK[n], out=lnS)
            if n in want:
                out[n] = SumKernel(self.u_node, self.tau_idx,
                                   self.n_time, self.T, self.lnq,
                                   lnS.copy())
        return out


class SumKernel:
    '''single-table runtime: ln S(tau, q1, q2, dphi) with the order
    sum already inside.  one trilinear gather per template instead
    of one per order — the per-order gather is latency bound at
    ~0.14 s/order, this is the same cost ONCE.  truncation n_max is
    frozen at build; keep a small ladder of these (one file per
    (u node, n_max)) and pick at load time.'''

    def __init__(self, u_node, tau_idx, n_time, T, lnq, lnS):
        self.u_node = float(u_node)
        self.tau_idx = tau_idx
        self.n_time = int(n_time)
        self.T = float(T)
        self.lnq = lnq
        self.nq = lnq.size
        self.nphi = lnS.shape[-1]
        self.qlo, self.qhi = float(np.exp(lnq[0])), float(np.exp(lnq[-1]))
        self.lnS = lnS

    def lnlam(self, z, u, t_center, half_width):
        '''ln Lambda at this table's frozen truncation'''
        ti, corners, ln_nt = _gather_setup(self, z, u, t_center,
                                           half_width)
        acc = 0.0
        for (ia, ib, ic, w) in corners:
            acc = acc + w * self.lnS[ti, ia, ib, ic]
        return logsumexp(acc.ravel()) - ln_nt


@njit(cache=True, fastmath=True)
def _eval_row(r, A0, A1, lqH, lqL, phH, phL, tau_idx, it, sh0, sh1,
              lnq00, lnq01, dlq, nq, nphi, out0, out1):
    '''one tau row of the fused two-node evaluation.  the log
    magnitudes and phases arrive PRECOMPUTED PER SAMPLE (they only
    depend on the 16k time samples, not on the 400k (tau, t) pairs
    — the per-element atan2+log were HALF the sweep, measured
    114 -> 48 ms); the node difference is only the constant
    ln sqrt(u_node/u) shift, and all 8 trilinear corners of both
    tables are read in the same pass (8 in-flight reads overlap
    their latencies — fewer corners measured SLOWER).'''
    ntw = it.size
    n_time = lqH.size
    for c in range(ntw):
        i1 = (it[c] + tau_idx[r]) % n_time
        i2 = it[c]
        lq1 = lqH[i1]
        lq2 = lqL[i2]
        dp = (phH[i1] - phL[i2]) % (2 * _math.pi)
        x3 = dp / (2 * _math.pi / nphi)
        c0 = int(x3) % nphi
        c1 = (c0 + 1) % nphi
        fc = x3 - int(x3)
        for node in range(2):
            lq0 = lnq00 if node == 0 else lnq01
            sh = sh0 if node == 0 else sh1
            A = A0 if node == 0 else A1
            x1 = (lq1 + sh - lq0) / dlq
            x2 = (lq2 + sh - lq0) / dlq
            if x1 < 0.0:
                x1 = 0.0
            if x1 > nq - 1 - 1e-9:
                x1 = nq - 1 - 1e-9
            if x2 < 0.0:
                x2 = 0.0
            if x2 > nq - 1 - 1e-9:
                x2 = nq - 1 - 1e-9
            a0 = int(x1)
            b0 = int(x2)
            fa = x1 - a0
            fb = x2 - b0
            a1 = min(a0 + 1, nq - 1)
            b1 = min(b0 + 1, nq - 1)
            v00 = (1 - fc) * A[r, a0, b0, c0] + fc * A[r, a0, b0, c1]
            v01 = (1 - fc) * A[r, a0, b1, c0] + fc * A[r, a0, b1, c1]
            v10 = (1 - fc) * A[r, a1, b0, c0] + fc * A[r, a1, b0, c1]
            v11 = (1 - fc) * A[r, a1, b1, c0] + fc * A[r, a1, b1, c1]
            val = ((1 - fa) * ((1 - fb) * v00 + fb * v01)
                   + fa * ((1 - fb) * v10 + fb * v11))
            if node == 0:
                out0[r, c] = val
            else:
                out1[r, c] = val


@njit(cache=True, fastmath=True)
def _eval_two(A0, A1, lqH, lqL, phH, phL, tau_idx, it, sh0, sh1,
              lnq00, lnq01, dlq, nq, nphi, out0, out1):
    for r in range(tau_idx.size):
        _eval_row(r, A0, A1, lqH, lqL, phH, phL, tau_idx, it, sh0,
                  sh1, lnq00, lnq01, dlq, nq, nphi, out0, out1)


@njit(cache=True, fastmath=True, parallel=True)
def _eval_two_par(A0, A1, lqH, lqL, phH, phL, tau_idx, it, sh0,
                  sh1, lnq00, lnq01, dlq, nq, nphi, out0, out1):
    for r in prange(tau_idx.size):
        _eval_row(r, A0, A1, lqH, lqL, phH, phL, tau_idx, it, sh0,
                  sh1, lnq00, lnq01, dlq, nq, nphi, out0, out1)


def sum_lnlam_pair(sk0, sk1, z, u, t_center, half_width,
                   parallel=False):
    assert z['H'].size == sk0.n_time and \
        z['L'].size == sk0.n_time, \
        'z length %d != store n_time %d' \
        % (z['H'].size, sk0.n_time)
    '''ln Lambda from a bracketing SumKernel pair, fused runtime:
    evaluates both nodes in one sweep and interpolates in ln u.
    pass the same kernel twice for a single-node evaluation.
    parallel=True uses the prange sweep (measured near-ideal
    scaling; keep False when many events share one node).'''
    t = np.arange(sk0.n_time) * (sk0.T / sk0.n_time)
    it = np.nonzero(np.abs(t - t_center) <= half_width)[0]
    out0 = np.empty((sk0.tau_idx.size, it.size))
    out1 = np.empty_like(out0)
    zH, zL = z['H'], z['L']
    lqH = 0.5 * np.log(zH.real**2 + zH.imag**2)
    lqL = 0.5 * np.log(zL.real**2 + zL.imag**2)
    phH = np.arctan2(zH.imag, zH.real)
    phL = np.arctan2(zL.imag, zL.real)
    fn = _eval_two_par if parallel else _eval_two
    fn(np.ascontiguousarray(sk0.lnS),
       np.ascontiguousarray(sk1.lnS),
       lqH, lqL, phH, phL, sk0.tau_idx, it,
       0.5 * _math.log(sk0.u_node / u),
       0.5 * _math.log(sk1.u_node / u),
       sk0.lnq[0], sk1.lnq[0], sk0.lnq[1] - sk0.lnq[0],
       sk0.nq, sk0.nphi, out0, out1)
    ln_nt = _math.log(it.size)
    ln0 = logsumexp(out0.ravel()) - ln_nt
    if sk1.u_node == sk0.u_node:
        return ln0
    ln1 = logsumexp(out1.ravel()) - ln_nt
    x = (_math.log(u / sk0.u_node)
         / _math.log(sk1.u_node / sk0.u_node))
    return (1 - x) * ln0 + x * ln1


def load_kstore(stem):
    '''per-order Kernel from a kstore pair (build_kstore.py), the
    lnK orders memory mapped — lns/lnsvec-style scans with ZERO
    compilation.  the object is evaluation-only (no lnK list copy;
    indexing mm[n] hands the gather a lazy view).'''
    z = np.load(str(stem) + '.meta.npz')
    mm = np.load(str(stem) + '.lnK.npy', mmap_mode='r')
    k = Kernel.__new__(Kernel)
    k.u_node = float(z['u_node'])
    k.tau_idx = z['tau_idx']
    k.n_time = int(z['n_time'])
    k.T = float(z['T'])
    k.lnq = z['lnq']
    k.nq = k.lnq.size
    k.nphi = mm.shape[-1]
    k.qlo = float(np.exp(k.lnq[0]))
    k.qhi = float(np.exp(k.lnq[-1]))
    k.N = mm.shape[0] - 1
    k.lnK = mm
    return k


def save_sumstore(sk, n_max, root, tag=''):
    '''write a SumKernel as raw .npy (lnS, mmap friendly) + a tiny
    meta npz.  compressed npz was measured at 1-2 min/GB to reopen;
    raw npy + mmap opens instantly and node-local page cache is
    SHARED between PE workers reading the same table.  tag marks
    families built with a different q-grid top (e.g. a loud event
    whose |z| exceeds the base family's coverage).'''
    stem = root / ('sumstore_n%03d%s_u%07.2f' % (n_max, tag,
                                                 sk.u_node))
    np.save(str(stem) + '.lnS.npy', sk.lnS)
    np.savez(str(stem) + '.meta.npz', u_node=sk.u_node,
             lnq=sk.lnq, tau_idx=sk.tau_idx, n_time=sk.n_time,
             T=sk.T, n_max=n_max)
    return stem


def load_sumstore(stem):
    '''SumKernel from a save_sumstore pair; lnS is memory mapped'''
    z = np.load(str(stem) + '.meta.npz')
    lnS = np.load(str(stem) + '.lnS.npy', mmap_mode='r')
    return SumKernel(float(z['u_node']), z['tau_idx'],
                     int(z['n_time']), float(z['T']), z['lnq'], lnS)


def _gather_setup(k, z, u, t_center, half_width):
    '''shared trilinear-lookup precompute: (tau, t-window) index
    grids, the 8 interpolation corners on (ln q1, ln q2, dphi), and
    ln of the window size.  k is a Kernel or SumKernel.'''
    t = np.arange(k.n_time) * (k.T / k.n_time)
    it = np.nonzero(np.abs(t - t_center) <= half_width)[0]
    sc = np.sqrt(k.u_node / u)

    z1 = z['H'][(it[None, :] + k.tau_idx[:, None]) % k.n_time]
    z2 = np.broadcast_to(z['L'][it][None, :], z1.shape)
    q1 = np.clip(sc * np.abs(z1), k.qlo, k.qhi)
    q2 = np.clip(sc * np.abs(z2), k.qlo, k.qhi)
    dp = np.angle(z1 * np.conj(z2)) % (2 * np.pi)

    dlq = k.lnq[1] - k.lnq[0]
    nq, nphi = k.nq, k.nphi
    x1 = np.clip((np.log(q1) - k.lnq[0]) / dlq, 0, nq - 1 - 1e-9)
    x2 = np.clip((np.log(q2) - k.lnq[0]) / dlq, 0, nq - 1 - 1e-9)
    x3 = dp / (2 * np.pi / nphi)
    i1, i2, i3 = x1.astype(int), x2.astype(int), x3.astype(int) % nphi
    f1, f2, f3 = x1 - i1, x2 - i2, x3 - x3.astype(int)
    j3 = (i3 + 1) % nphi
    ti = np.broadcast_to(np.arange(k.tau_idx.size)[:, None], z1.shape)
    corners = [(ia, ib, ic, fa * fb * fc)
               for (ia, fa) in [(i1, 1 - f1),
                                (np.minimum(i1 + 1, nq - 1), f1)]
               for (ib, fb) in [(i2, 1 - f2),
                                (np.minimum(i2 + 1, nq - 1), f2)]
               for (ic, fc) in [(i3, 1 - f3), (j3, f3)]]
    return ti, corners, np.log(it.size)
