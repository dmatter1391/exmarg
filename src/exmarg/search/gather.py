"""
the table read: for one template and one table, Eq. (C7) of the paper,

  ln Lambda = ln( sum_tau sum_t exp lnS(tau; beta, Delta, dphi) ) - ln n_time
  beta  = ln a + 2 ln q_L(t),   Delta = ln q_H(t + tau) - ln q_L(t),
  dphi  = phi_H(t + tau) - phi_L(t),

summed over the arrival times t of the segment and every lag tau of the
table, with q_I, phi_I the modulus and phase of the matched-filter output
of detector I and a = s/u the ratio of the table's signal power to the
template power (a = 1 for the intrinsic store).  lnS is read bilinearly in
(beta, Delta) and linearly in dphi from a (tau, dphi >= 0, beta, Delta)
float32 table; dphi < 0 is read from |dphi| (the tables are even in dphi).
beta is clamped to the table, ln q_L moves with it and ln q_H is kept.

gather_cpu (numba, the (template, lag) pairs in parallel over the cores)
and gather_gpu (cupy) compute the same sums and return the same array.

By default the sum runs over every arrival time.  With times from
select_times it runs over the loudest arrival times of each template, where
almost all of a signal's contribution lies, plus a uniform random draw from
the rest weighted to stand for all of them, so the sum is estimated without
bias at a fraction of the cost.
"""

import numpy as np
import numba

NPHI = 96                 # phase bins of the full circle; the tables hold NPHI/2 + 1


@numba.njit(parallel=True, cache=True)
def _gather_cpu(lqH, phH, lqL, phL, mass_ids, tab_ids, tau_idx, tables, offbs, rdb, offd, rdd, sel, lw, use_sel, out_m, out_s):
    """running (max, sum) of exp(lnS) over the arrival times (all, or sel[template] with log weights lw), for each (slot, lag, template)"""
    n_time = lqH.shape[1]
    ntau = tau_idx.size
    nsel, ns = tab_ids.shape
    nb = tables[0].shape[2]; nd = tables[0].shape[3]
    rph = NPHI / 6.283185307179586
    n_walk = sel.shape[1] if use_sel else n_time
    for job in numba.prange(nsel * ntau):
        j = job // ntau
        r = job % ntau
        mid = mass_ids[j]
        tau0 = ((tau_idx[r] % n_time) + n_time) % n_time
        for n in range(ns):
            Tn = tables[tab_ids[j, n]]
            m = -1e300; s = 0.0
            for k in range(n_walk):
                t = sel[mid, k] if use_sel else k
                i1 = t + tau0
                if i1 >= n_time:
                    i1 -= n_time
                x3 = (phH[mid, i1] - phL[mid, t]) * rph
                if x3 < 0.0:
                    x3 += NPHI
                if x3 >= NPHI:
                    x3 -= NPHI
                c0 = int(x3)
                if c0 >= NPHI:
                    c0 = NPHI - 1
                elif c0 < 0:
                    c0 = 0
                fc = x3 - c0
                c1 = (c0 + 1) % NPHI
                aH = lqH[mid, i1]; aL = lqL[mid, t]
                xbr = aL * rdb + offbs[j, n]
                xb = xbr
                if xb < 0.0:
                    xb = 0.0
                elif xb > nb - 1 - 1e-6:
                    xb = nb - 1 - 1e-6
                aLc = aL + (xb - xbr) / rdb
                xd = (aH - aLc) * rdd + offd
                if xd < 0.0:
                    xd = 0.0
                elif xd > nd - 1 - 1e-6:
                    xd = nd - 1 - 1e-6
                d0 = int(xd); fdf = xd - d0; fd0 = 1.0 - fdf
                d1 = d0 + 1 if d0 + 1 < nd else nd - 1
                b0 = int(xb); fbf = xb - b0; fb0 = 1.0 - fbf
                b1 = b0 + 1 if b0 + 1 < nb else nb - 1
                acc = 0.0
                for cc in range(2):
                    ci = c1 if cc else c0
                    if ci > NPHI // 2:
                        ci = NPHI - ci
                    fci = fc if cc else 1.0 - fc
                    v = (fb0 * (fd0 * Tn[r, ci, b0, d0] + fdf * Tn[r, ci, b0, d1])
                         + fbf * (fd0 * Tn[r, ci, b1, d0] + fdf * Tn[r, ci, b1, d1]))
                    acc += fci * v
                if use_sel:
                    acc += lw[mid, k]
                if acc > m:
                    s = s * np.exp(m - acc) + 1.0
                    m = acc
                else:
                    s += np.exp(acc - m)
            out_m[n, r, j] = m
            out_s[n, r, j] = s


def _lse_lags(om, os_, n_time, xp):
    """log-sum-exp over the lags of the running (max, sum) pairs, minus ln n_time"""
    mx = om.max(axis=1)
    tot = (os_ * xp.exp(om - mx[:, None, :])).sum(axis=1)
    return mx + xp.log(tot) - np.log(n_time)


def _offsets(lna, beta):
    return (np.ascontiguousarray(lna, np.float64) - float(beta[0])) / float(beta[1] - beta[0])


def gather_cpu(traj, tables, mass_ids, lna, tab_ids, times=None):
    """ln Lambda of (template, table) pairs on the CPU.

    traj      dict of float32 arrays lqH, phH, lqL, phL, each (n_template,
              n_time): ln modulus and phase of the matched-filter output
    tables    dict with T (list of host tables, each (ntau, NPHI/2 + 1,
              nbeta, ndelta)), tau (lags), beta, delta (grids)
    mass_ids  (nsel,) template rows to evaluate
    lna       (nsel, NS) ln a = ln(s/u) of each (template, slot)
    tab_ids   (nsel, NS) index in tables['T'] of the table each slot reads
    times     None (every arrival time) or select_times(...): (sel, lw)

    returns (NS, nsel) float64."""
    beta, delta = tables["beta"], tables["delta"]
    ids = np.ascontiguousarray(mass_ids, np.int64)
    tab = np.ascontiguousarray(tab_ids, np.int64)
    ns = tab.shape[1]
    ntau = int(np.asarray(tables["tau"]).size)
    om = np.empty((ns, ntau, ids.size)); os_ = np.empty((ns, ntau, ids.size))
    T = numba.typed.List([np.ascontiguousarray(Ti, np.float32) for Ti in tables["T"]])
    ddelta = float(delta[1] - delta[0])
    sel, lw = times if times is not None else (np.zeros((1, 1), np.int32), np.zeros((1, 1), np.float32))
    _gather_cpu(traj["lqH"], traj["phH"], traj["lqL"], traj["phL"], ids, tab,
                np.ascontiguousarray(tables["tau"], np.int64).ravel(), T, _offsets(lna, beta),
                2.0 / float(beta[1] - beta[0]), -float(delta[0]) / ddelta, 1.0 / ddelta,
                np.ascontiguousarray(sel, np.int32), np.ascontiguousarray(lw, np.float32), times is not None, om, os_)
    return _lse_lags(om, os_, traj["lqH"].shape[1], np)


_SRC = r"""
#define BLK __BLK__
#define NB __NB__
#define ND __ND__
#define NPHI __NPHI__
#define NT __NT__
#define CH __CH__
#define NS __NS__
#define NPHIF (__NPHI__/2 + 1)
#define SEL __SEL__
#define KS __KS__
#if SEL
#define TIME_OF(i) (sel[(long long)mid * KS + (i)])
#define WEIGHT_OF(i) (lw[(long long)mid * KS + (i)])
#define NWALK KS
#else
#define TIME_OF(i) (i)
#define WEIGHT_OF(i) 0.0f
#define NWALK NT
#endif

extern "C" __global__
void gather(const float* __restrict__ lqH, const float* __restrict__ phH,
            const float* __restrict__ lqL, const float* __restrict__ phL,
            const int* __restrict__ mass_ids,     // (nsel) template rows
            const int* __restrict__ tab_ids,      // (nsel, NS) table read by each (template, slot)
            const long long* __restrict__ tau_idx,
            const float* __restrict__ T,          // (ntab, ntau, NPHIF, NB, ND)
            const float* __restrict__ offbs,      // (nsel, NS) (ln a - beta0) / dbeta
            const float rdb,                      // 2 / dbeta
            const float offd,                     // -delta0 / ddelta
            const float rdd,                      // 1 / ddelta
            const long long tstride,
            const int ntau,
            const int* __restrict__ sel,          // (n_template, KS) arrival times visited (SEL = 1)
            const float* __restrict__ lw,         // (n_template, KS) their log weights
            double* __restrict__ out_m, double* __restrict__ out_s)
{
    extern __shared__ unsigned char smem[];
    unsigned short* ordr = (unsigned short*)smem;
    int* cnt = (int*)(smem + 2 * CH);
    unsigned char* cbin = (unsigned char*)(smem + 2 * CH + 4 * NPHI);
    float* fcs = (float*)(smem + 2 * CH + 4 * NPHI + CH);
    const int j = blockIdx.x, r = blockIdx.y;
    const int mid = mass_ids[j];
    const long long base = (long long)mid * NT;
    const long long tau = tau_idx[r];
    const double TWOPI = 6.283185307179586476925286766559;
    const int tau0 = (int)(((tau % NT) + NT) % NT);
    const float rph = (float)(NPHI / TWOPI);
    float offb[NS];
    #pragma unroll
    for (int n = 0; n < NS; ++n) offb[n] = offbs[j * NS + n];
    float m[NS], ssum[NS];
    #pragma unroll
    for (int n = 0; n < NS; ++n) { m[n] = -1.0e30f; ssum[n] = 0.0f; }

    for (int cs = 0; cs < NWALK; cs += CH) {
        // counting sort of this chunk of arrival times by phase bin
        for (int i = threadIdx.x; i < NPHI; i += BLK) cnt[i] = 0;
        __syncthreads();
        for (int q = threadIdx.x; q < CH; q += BLK) {
            if (cs + q >= NWALK) break;
            int t = TIME_OF(cs + q);
            int i1 = t + tau0; if (i1 >= NT) i1 -= NT;
            float x3 = (phH[base + i1] - phL[base + t]) * rph;
            if (x3 < 0.0f) x3 += (float)NPHI;
            if (x3 >= (float)NPHI) x3 -= (float)NPHI;
            int c0 = (int)x3; if (c0 >= NPHI) c0 = NPHI - 1; else if (c0 < 0) c0 = 0;
            cbin[q] = (unsigned char)c0; fcs[q] = x3 - (float)c0;
            atomicAdd(&cnt[c0], 1);
        }
        __syncthreads();
        if (threadIdx.x == 0) { int acc = 0; for (int i = 0; i < NPHI; ++i) { int c = cnt[i]; cnt[i] = acc; acc += c; } }
        __syncthreads();
        for (int q = threadIdx.x; q < CH; q += BLK) {
            if (cs + q >= NWALK) break;
            ordr[atomicAdd(&cnt[(int)cbin[q]], 1)] = (unsigned short)q;
        }
        __syncthreads();
        for (int p = threadIdx.x; p < CH; p += BLK) {
            if (cs + p >= NWALK) break;
            int off = cs + (int)ordr[p];
            int t = TIME_OF(off);
            const float wt = WEIGHT_OF(off);
            int i1 = t + tau0; if (i1 >= NT) i1 -= NT;
            const float aH = lqH[base + i1], aL = lqL[base + t];
            const int qq = (int)ordr[p];
            int c0 = (int)cbin[qq];
            float fc = fcs[qq];
            int c1 = (c0 + 1) % NPHI;
            #pragma unroll
            for (int n = 0; n < NS; ++n) {
                // beta is clamped to the table; ln q_L moves with it and ln q_H is kept
                const float xbr = fmaf(aL, rdb, offb[n]);
                float xb = xbr;
                if (xb < 0.0f) xb = 0.0f; else if (xb > NB - 1 - 1e-6f) xb = NB - 1 - 1e-6f;
                const float aLc = aL + (xb - xbr) / rdb;
                float xd = fmaf(aH - aLc, rdd, offd);
                if (xd < 0.0f) xd = 0.0f; else if (xd > ND - 1 - 1e-6f) xd = ND - 1 - 1e-6f;
                const int d0 = (int)xd; const float fdf = xd - d0, fd0 = 1.0f - fdf;
                const int d1 = (d0 + 1 < ND) ? d0 + 1 : ND - 1;
                const int b0 = (int)xb; const float fbf = xb - b0, fb0 = 1.0f - fbf;
                const int b1 = (b0 + 1 < NB) ? b0 + 1 : NB - 1;
                const float* Tn = T + (long long)tab_ids[j * NS + n] * tstride;
                float acc = 0.0f;
                for (int cc = 0; cc < 2; ++cc) {
                    int ci = cc ? c1 : c0;
                    ci = (ci <= NPHI / 2) ? ci : (NPHI - ci);
                    float fci = cc ? fc : 1.0f - fc;
                    const long long rc = ((long long)r * NPHIF + ci) * NB;
                    const float* r0 = Tn + (rc + b0) * ND;
                    const float* r1 = Tn + (rc + b1) * ND;
                    float v = fb0 * (fd0 * r0[d0] + fdf * r0[d1]) + fbf * (fd0 * r1[d0] + fdf * r1[d1]);
                    acc += fci * v;
                }
                float accd = acc + wt;
                if (accd > m[n]) {
                    float sc = expf(m[n] - accd);
                    ssum[n] = ssum[n] * sc + 1.0f;
                    m[n] = accd;
                } else {
                    ssum[n] += expf(accd - m[n]);
                }
            }
        }
        __syncthreads();
    }
    // block reduction of the running (max, sum) pairs: warp shuffles, then one warp
    __shared__ float wm[32 * NS], ws[32 * NS];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int nwarp = BLK >> 5;
    #pragma unroll
    for (int n = 0; n < NS; ++n) {
        float mv = m[n], sv = ssum[n];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            float mo = __shfl_down_sync(0xffffffffu, mv, off);
            float so = __shfl_down_sync(0xffffffffu, sv, off);
            if (mo > mv) { float f = expf(mv - mo); sv = so + sv * f; mv = mo; }
            else         { float f = expf(mo - mv); sv = sv + so * f; }
        }
        if (lane == 0) { wm[warp * NS + n] = mv; ws[warp * NS + n] = sv; }
    }
    __syncthreads();
    if (warp == 0) {
        #pragma unroll
        for (int n = 0; n < NS; ++n) {
            float mv = (lane < nwarp) ? wm[lane * NS + n] : -1.0e30f;
            float sv = (lane < nwarp) ? ws[lane * NS + n] : 0.0f;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                float mo = __shfl_down_sync(0xffffffffu, mv, off);
                float so = __shfl_down_sync(0xffffffffu, sv, off);
                if (mo > mv) { float f = expf(mv - mo); sv = so + sv * f; mv = mo; }
                else         { float f = expf(mo - mv); sv = sv + so * f; }
            }
            if (lane == 0) {
                const long long o = ((long long)n * ntau + r) * gridDim.x + j;
                out_m[o] = (double)mv; out_s[o] = (double)sv;
            }
        }
    }
}
"""


_CACHE = {}


def _chunk(nphi):
    """arrival times sorted per pass: the largest power of two whose shared memory fits the device"""
    import cupy as cp
    limit = cp.cuda.Device().attributes["MaxSharedMemoryPerBlockOptin"]
    ch = 16384
    while 7 * ch + 4 * nphi + 256 > limit:
        ch //= 2
    return ch


def _compile(nb, nd, nphi, n_time, ns, ch, blk, ks=0):
    """compile (and cache) the GPU gather for one table geometry, slot count and number of selected times (0: all)"""
    import cupy as cp
    key = (nb, nd, nphi, n_time, ns, ch, blk, ks)
    if key not in _CACHE:
        src = _SRC
        for k, v in (("__NB__", nb), ("__ND__", nd), ("__NPHI__", nphi), ("__BLK__", blk),
                     ("__NT__", n_time), ("__CH__", ch), ("__NS__", ns), ("__SEL__", int(ks > 0)), ("__KS__", max(ks, 1))):
            src = src.replace(k, str(v))
        shm = 7 * ch + 4 * nphi + 256             # ordr, cnt, cbin, fcs
        fn = cp.RawKernel(src, "gather")
        if shm > 48 * 1024:
            fn.max_dynamic_shared_size_bytes = shm
        _CACHE[key] = (fn, shm)
    return _CACHE[key]


def gather_gpu(traj, tables, mass_ids, lna, tab_ids, block=256, times=None):
    """ln Lambda of (template, table) pairs on the GPU, one launch.

    traj      dict of device float32 arrays lqH, phH, lqL, phL
    tables    dict from store.upload: T (stacked tables), tau, tstride,
              shape = (ntau, nphi, nbeta, ndelta), beta, delta
    mass_ids, lna, tab_ids  as for gather_cpu (tab_ids index the stack)
    block     threads per block
    times     None (every arrival time) or select_times(...): (sel, lw) on the device

    returns (NS, nsel) float64 on the host."""
    import cupy as cp
    ntau, nphi, nb, nd = tables["shape"]
    beta, delta = tables["beta"], tables["delta"]
    n_time = int(traj["lqH"].shape[1])
    ns = int(np.asarray(lna).shape[1])
    ks = 0 if times is None else int(times[0].shape[1])
    fn, shm = _compile(nb, nd, nphi, n_time, ns, _chunk(nphi), block, ks)
    sel, lw = times if times is not None else (cp.zeros(1, cp.int32), cp.zeros(1, cp.float32))
    ddelta = float(delta[1] - delta[0])
    dm = cp.asarray(np.ascontiguousarray(mass_ids, np.int32))
    nmass = int(dm.size)
    om = cp.empty((ns, ntau, nmass), cp.float64)
    os_ = cp.empty((ns, ntau, nmass), cp.float64)
    args = (traj["lqH"], traj["phH"], traj["lqL"], traj["phL"], dm,
            cp.asarray(np.ascontiguousarray(tab_ids, np.int32).ravel()),
            tables["tau"], tables["T"], cp.asarray(_offsets(lna, beta).astype(np.float32).ravel()),
            np.float32(2.0 / float(beta[1] - beta[0])), np.float32(-float(delta[0]) / ddelta),
            np.float32(1.0 / ddelta), np.int64(tables["tstride"]), np.int32(ntau),
            cp.ascontiguousarray(sel, cp.int32), cp.ascontiguousarray(lw, cp.float32), om, os_)
    fn((nmass, ntau), (block,), args, shared_mem=shm)
    return cp.asnumpy(_lse_lags(om, os_, n_time, cp))


def select_times(traj, tau_idx, n_loud, n_random, seed, xp=np, block=256):
    """the arrival times visited for every template.

    The n_loud times with the largest |z_L(t)|^2 + max_tau |z_H(t + tau)|^2 are
    kept exactly; n_random more are drawn uniformly (with replacement) from the
    rest and carry the log weight ln((n_time - n_loud) / n_random), so the sum
    over the selection estimates the sum over all times without bias.  The
    ranking uses the data only.  The random draw is made with numpy on the
    host, so the CPU and the GPU visit the same times.

    traj      dict with lqH, lqL (n_template, n_time) on the device of xp
    tau_idx   the lags of the tables (contiguous integers)
    block     templates ranked at once (bounds the temporary memory)

    returns (sel, lw): (n_template, n_loud + n_random) int32 times, sorted,
    and float32 log weights, on the device of xp."""
    K, S = int(n_loud), int(n_random)
    nm, nt = traj["lqH"].shape
    lags = np.asarray(tau_idx).ravel().astype(int)
    lo, hi = int(lags.min()), int(lags.max())
    if not np.array_equal(np.sort(lags), np.arange(lo, hi + 1)):
        raise ValueError("lags are not contiguous")
    k = int(np.floor(np.log2(hi - lo + 1)))
    pick = np.random.default_rng(seed).integers(0, nt - K, size=(nm, S))
    sel = xp.empty((nm, K + S), xp.int32)
    for a in range(0, nm, block):
        b = min(a + block, nm)
        # max over the lags lo..hi of |z_H(t + tau)|^2: a sliding maximum built by doubling (ln|z| is monotone)
        W = traj["lqH"][a:b]
        for i in range(k):
            W = xp.maximum(W, xp.roll(W, -(1 << i), axis=1))
        mx = xp.maximum(xp.roll(W, -lo, axis=1), xp.roll(W, -(hi - (1 << k) + 1), axis=1))
        del W
        score = xp.exp(2.0 * traj["lqL"][a:b]) + xp.exp(2.0 * mx)
        del mx
        top = xp.argpartition(-score, K, axis=1)[:, :K]
        del score
        rest = xp.ones((b - a, nt), bool)
        rest[xp.arange(b - a)[:, None], top] = False
        rest = xp.nonzero(rest)[1].reshape(b - a, nt - K)          # the other times, in time order
        sel[a:b] = xp.concatenate([top, xp.take_along_axis(rest, xp.asarray(pick[a:b]), axis=1)], axis=1)
        del top, rest
    lw = xp.zeros((nm, K + S), xp.float32)
    lw[:, K:] = np.log((nt - K) / S)
    order = xp.argsort(sel, axis=1)
    return (xp.ascontiguousarray(xp.take_along_axis(sel, order, axis=1)),
            xp.ascontiguousarray(xp.take_along_axis(lw, order, axis=1)))
