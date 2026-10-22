"""
parameter estimation of single segments with the stored tables, on the
CPU or the GPU (the same steps, device='cpu' or 'gpu').

pe_redshift   (m1_source, z) at equal mass and zero spin, the distance
              a parameter through y = D0 / D_L(z)
pe_intrinsic  (M_det, q, chi) on a template bank, the distance
              integrated inside the tables

both return ln Lambda, the likelihood ratio against noise marginalized
over the sky position, orientation, polarization, phase and arrival
time, and ln B, its prior average.  Every arrival time of the segment
and every lag of the tables enters the sum of gather.py.
"""

import numpy as np

from . import gather as _gather, store as _store
from .bank import array_module
from .segments import D0, read_segments
from ..priors import distance_density


def _dl_of_z(z):
    from astropy.cosmology import Planck18
    return Planck18.luminosity_distance(np.atleast_1d(z)).value


def _chunks(seg, chunk):
    n = seg["rows"].size
    for c0 in range(0, n, chunk):
        yield range(c0, min(c0 + chunk, n))


def _fit_chunk(bank, store, chunk, nodes_per_launch, resident):
    """GPU: the number of segments filtered at once, at most chunk, so that their trajectories
    (16 n_template n_time bytes each) fit next to the tables on the device"""
    if bank.device == "cpu":
        return chunk
    import cupy as cp
    free = cp.cuda.runtime.memGetInfo()[0] + cp.get_default_memory_pool().free_bytes()
    sizes = [t.nbytes for t in store.tables]
    if resident and getattr(store, "device", None) is None:
        free -= sum(sizes)
    elif not resident:
        free -= nodes_per_launch * max(sizes)
    per_segment = 16 * bank.u.size * bank.n_time
    return max(1, min(chunk, int(0.6 * free / per_segment)))


def _filter(bank, seg, e):
    return bank.trajectories({I: seg["d_" + I][e] for I in "HL"},
                             {I: seg["sig2_" + I] for I in "HL"})


def _node_geometry(bank, st, ymin):
    """which (node, template) pairs the redshift store covers, and their ln a and y.

    a = s/u is valid for y = sqrt(a) >= ymin and alpha_lo' < a < alpha_hi,
    alpha_lo' = min(alpha_lo, ymin^2).  a template whose a crosses
    alpha_lo' between two nodes also gets a boundary point at
    a = alpha_lo' (1 + 1e-9), read from the upper node's table."""
    nn, nm = st.s.size, bank.u.size
    valid = np.zeros((nn, nm), bool)
    lna = np.zeros((nn, nm))
    y = np.zeros((nn + 1, nm))                 # row nn: the boundary point
    bnode = np.full(nm, -1)
    blna = np.zeros(nm)
    for k in range(nn):
        alo = min(st.alpha_lo[k], ymin * ymin)
        a = st.s[k] / bank.u
        o = (np.sqrt(a) >= ymin) & (a > alo) & (a < st.alpha_hi[k])
        valid[k] = o
        lna[k][o] = np.log(a[o])
        y[k] = np.where(o, np.sqrt(a), 0.0)
        s_prev = st.s[k - 1] if k > 0 else 0.0
        low = o & (s_prev / bank.u <= alo)
        a_b = float(alo) * (1.0 + 1e-9)
        bnode[low] = k
        blna[low] = np.log(a_b)
        y[nn][low] = np.sqrt(a_b)
    return valid, lna, y, bnode, blna


def _to_source_frame(lamN, yN, Mdet, yq, m1, z, cp):
    """node values (n_node + 1, n_template) -> ln Lambda on (m1, z) (cp: numpy or cupy):
    linear in y per template, then linear in M_det = 2 m1 (1 + z) per z, each
    between the neighbouring finite values (as np.interp on the finite ones)"""
    L = cp.asarray(lamN)
    Y = cp.asarray(yN)
    ok = cp.isfinite(L) & (Y > 0)
    order = cp.argsort(cp.where(ok, Y, cp.inf), axis=0)
    Ys = cp.take_along_axis(cp.where(ok, Y, cp.inf), order, axis=0)
    Ls = cp.take_along_axis(L, order, axis=0)
    n = ok.sum(axis=0)                                            # finite nodes per template
    yq_ = cp.asarray(yq)
    j = (Ys[:, None, :] <= yq_[None, :, None]).sum(axis=0) - 1    # (nz, nm): last node <= yq
    lo = cp.clip(j, 0, cp.maximum(n - 2, 0)[None, :])
    hi = cp.minimum(lo + 1, cp.maximum(n - 1, 0)[None, :])
    cols = cp.arange(Ys.shape[1])[None, :]
    y0, y1 = Ys[lo, cols], Ys[hi, cols]
    v0, v1 = Ls[lo, cols], Ls[hi, cols]
    t = cp.where(y1 > y0, (yq_[:, None] - y0) / cp.where(y1 > y0, y1 - y0, 1.0), 1.0)
    ylast = Ys[cp.maximum(n - 1, 0), cp.arange(Ys.shape[1])][None, :]
    inside = (n[None, :] >= 2) & (yq_[:, None] >= Ys[0][None, :]) & (yq_[:, None] <= ylast)
    surf = cp.where(inside, v0 + t * (v1 - v0), -cp.inf)        # (nz, nm)

    M = cp.asarray(Mdet)
    good = cp.isfinite(surf)                                      # (nz, nm)
    rank = cp.cumsum(good, axis=1)                                # good masses up to and including each index
    ngood = rank[:, -1]
    G = cp.zeros(surf.shape, cp.int64)                            # G[iz, r] = index of the r-th good mass
    iz, im = cp.nonzero(good)
    G[iz, rank[iz, im] - 1] = im
    Md = 2.0 * cp.asarray(m1)[:, None] * (1.0 + cp.asarray(z))[None, :]   # (nm1, nz)
    p = cp.searchsorted(M, Md.ravel()).reshape(Md.shape)          # masses with M_det < Md
    zi = cp.arange(z.size)[None, :]
    below = cp.where(p > 0, rank[zi, cp.maximum(p - 1, 0)], 0)    # good masses with M_det < Md
    lo = G[zi, cp.clip(below - 1, 0, None)]
    hi = G[zi, cp.minimum(below, cp.maximum(ngood - 1, 0)[None, :])]
    lo = cp.where(below >= 1, lo, hi)
    m0, m1_ = M[lo], M[hi]
    w0, w1 = surf[zi, lo], surf[zi, hi]
    t = cp.where(m1_ > m0, (Md - m0) / cp.where(m1_ > m0, m1_ - m0, 1.0), 0.0)
    first = M[G[:, 0]][None, :]
    last = M[G[zi[0], cp.maximum(ngood - 1, 0)]][None, :]
    inb = (ngood[None, :] >= 2) & (Md >= first) & (Md <= last)
    with np.errstate(invalid="ignore"):                 # -inf - (-inf) outside the bank; those points are masked by inb
        out = cp.where(inb, w0 + t * (w1 - w0), -cp.inf)
    return out if cp is np else cp.asnumpy(out)


def _log_mean(lam, w, cp=np):
    """ln sum_i w_i exp(lam_i) for normalized weights w, lam = -inf allowed"""
    lam, w = cp.asarray(lam), cp.asarray(w)
    fin = cp.isfinite(lam) & (w > 0)
    if not bool(fin.any()):
        return -np.inf
    mx = cp.where(fin, lam, -cp.inf).max()
    return float(mx + cp.log(cp.sum(cp.where(fin, w * cp.exp(cp.where(fin, lam, mx) - mx), 0.0))))



def _tables(store, ks, beta, delta, bank, resident):
    """the tables ks for a gather: host list (CPU) or device stack (GPU), and the offset of ks[0]"""
    if bank.device == "cpu":
        return dict(T=[store.tables[k] for k in ks], tau=store.tau_idx, beta=beta, delta=delta), ks[0]
    if resident:
        if getattr(store, "device", None) is None:
            store.device = _store.upload(store.tables, store.tau_idx, beta, delta)
        return dict(store.device, beta=beta, delta=delta), 0
    return _store.upload([store.tables[k] for k in ks], store.tau_idx, beta, delta), ks[0]


def _read(bank, traj, tab, ids, lna, tab_ids, block):
    if bank.device == "cpu":
        return _gather.gather_cpu(traj, tab, ids, lna, tab_ids)
    return _gather.gather_gpu(traj, tab, ids, lna, tab_ids, block)


def _release(bank):
    if bank.device == "gpu":
        _store.free()


def _node_values(bank, seg, ev, store, geometry, groups, block, resident):
    """ln Lambda of every template at every node of the redshift store, for the segments ev:
    (len(ev), n_node + 1, n_template), row n_node the boundary point at the smallest amplitude"""
    valid, lna, _, bnode, blna = geometry
    nn = store.s.size
    traj = [_filter(bank, seg, e) for e in ev]
    lamN = np.full((len(ev), nn + 1, bank.M.size), -np.inf)
    for ks in groups:
        k0 = ks[0]
        tab, base = _tables(store, ks, store.beta[k0], store.delta[k0], bank, resident)
        ok = valid[ks]
        ids = np.nonzero(ok.any(axis=0))[0]
        lnaM = np.where(ok[:, ids], lna[ks][:, ids], 0.0).T
        tid = np.tile(np.arange(len(ks)) + (k0 - base), (ids.size, 1))
        bids = np.nonzero((bnode >= k0) & (bnode <= ks[-1]))[0]
        for j in range(len(ev)):
            if ids.size:
                val = _read(bank, traj[j], tab, ids, lnaM, tid, block)
                for i, k in enumerate(ks):
                    keep = ok[i][ids]
                    lamN[j, k, ids[keep]] = val[i][keep]
            if bids.size:
                val = _read(bank, traj[j], tab, bids, blna[bids][:, None], (bnode[bids] - base)[:, None], block)
                lamN[j, nn, bids] = val[0]
        del tab
        if not resident:
            _release(bank)
    del traj
    _release(bank)
    return lamN


def pe_redshift(segments, bank, store, rows=None, chunk=8, zmax=8.0, nz=1600,
                m1_range=(3.0, 150.0), nm1=1600, ymin=0.1267, nodes_per_launch=6, block=256,
                mdet_range=(10.0, 500.0), resident=False):
    """ln Lambda(m1_source, z) of each segment.

    segments          anything read_segments takes (see there)
    bank              mass_bank of a one-column bank of detector-frame total
                      masses, ascending, built on the same segments
    store             load_redshift_store
    rows              segment indices (default all)
    chunk             segments filtered at once (16 n_template n_time bytes each; on the GPU
                      at most what fits next to the tables)
    zmax, nz          z grid linspace(0.02, zmax, nz)
    m1_range, nm1     m1 grid geomspace(*m1_range, nm1)
    ymin              smallest y = D0 / D_L the store represents
    nodes_per_launch  store nodes read in one gather
    block             GPU threads per block
    mdet_range        support of the ln B prior
    resident          GPU: keep every table on the device across calls

    ln B averages Lambda over the prior uniform in M_det = 2 m1 (1+z) on
    mdet_range and proportional to D_L^2 in luminosity distance, i.e.
    p(m1, z) ~ (1+z) D_L^2 dD_L/dz, on the (m1, z) cells.

    returns dict: m1 (nm1,), z (nz,), ln_lambda (n_segment, nm1, nz),
    ln_b (n_segment,), rows."""
    if np.any(np.diff(bank.M) <= 0) or np.any(bank.q != 1.0) or np.any(bank.chi != 0.0):
        raise ValueError("pe_redshift needs an ascending equal-mass zero-spin bank")
    xp = array_module(bank.device)
    seg = read_segments(segments, rows)
    z = np.linspace(0.02, zmax, int(nz))
    m1 = np.geomspace(m1_range[0], m1_range[1], int(nm1))
    yq = D0 / _dl_of_z(z)
    geometry = _node_geometry(bank, store, ymin)
    nn = store.s.size
    groups = [list(range(g, min(g + nodes_per_launch, nn))) for g in range(0, nn, nodes_per_launch)]

    out = np.full((seg["rows"].size, m1.size, z.size), -np.inf)
    chunk = _fit_chunk(bank, store, chunk, nodes_per_launch, resident)
    for ev in _chunks(seg, chunk):
        lamN = _node_values(bank, seg, ev, store, geometry, groups, block, resident)
        for j, e in enumerate(ev):
            out[e] = _to_source_frame(lamN[j], geometry[2], bank.M, yq, m1, z, xp)

    dl = _dl_of_z(z)
    lnp = np.log(np.maximum((1.0 + z) * dl ** 2 * np.gradient(dl, z), 1e-300))
    Md = 2.0 * m1[:, None] * (1.0 + z[None, :])
    inside = (Md >= mdet_range[0]) & (Md <= mdet_range[1])
    w = np.where(inside, np.exp(lnp)[None, :], 0.0) * np.gradient(m1)[:, None] * np.gradient(z)[None, :]
    w /= w.sum()
    ln_b = np.array([_log_mean(L, w) for L in out])
    return dict(m1=m1, z=z, ln_lambda=out, ln_b=ln_b, rows=seg["rows"])


def distance_weights(y, prior, ymin):
    """ln of the weights that integrate the distance prior against a likelihood ratio linear in x = ln y between
    the points y (ascending): sum_j w_j Lambda(y_j) = integral dy p(y) Lambda(y) over [y_0, y_last].

    prior  'volume': p(y) = 3 ymin^3 y^-4 (uniform in volume), integrated exactly on every interval;
           'madau-dickinson': the density in ln y of exmarg.priors, integrated by the trapezoid rule
           on 401 points inside every interval"""
    density = distance_density(prior)
    x = np.log(y)
    w = np.zeros_like(x)
    for i in range(x.size - 1):
        a, b = x[i], x[i + 1]
        h = b - a
        if density is None:
            c, k = 3.0 * ymin ** 3, 3.0
            ea, eb = np.exp(-k * a), np.exp(-k * b)
            I0 = (ea - eb) / k
            I1 = (a * ea - b * eb) / k + (ea - eb) / k ** 2
            w[i] += c * (b * I0 - I1) / h
            w[i + 1] += c * (I1 - a * I0) / h
        else:
            t = np.linspace(a, b, 401)
            f = np.interp(t, density[0], density[1], left=0.0, right=0.0)
            w[i] += np.trapezoid(f * (b - t) / h, t)
            w[i + 1] += np.trapezoid(f * (t - a) / h, t)
    with np.errstate(divide="ignore"):
        return np.log(w)


def pe_intrinsic(segments, bank, store, rows=None, prior="volume", ymin=0.1267, chunk=4,
                 nodes_per_launch=6, block=256, resident=False):
    """ln Lambda of each template of the bank, the distance integrated over its prior.

    segments          anything read_segments takes (see there)
    bank              mass_bank built on the same segments, rows (M_det, q, chi)
    store             load_redshift_store
    rows              segment indices (default all)
    prior             'volume' (uniform in volume) or 'madau-dickinson' (the distance prior of the
                      simulated year, exmarg.priors)
    ymin              smallest y = D0 / D_L of the prior
    chunk             segments filtered at once (16 n_template n_time bytes each; on the GPU
                      at most what fits next to the tables)
    nodes_per_launch  store nodes read in one gather
    block             GPU threads per block
    resident          GPU: keep every table on the device across calls

    each template reads the store at every node s it reaches, y = sqrt(s/u), and
    Lambda = sum_j w_j Lambda(y_j) with the weights of distance_weights.  ln B
    is the average of Lambda over the bank, each template weighted by its cell
    (the product of the spacings of the distinct values of every axis that varies).

    returns dict: M, q, chi, u (n_template,), ln_lambda (n_segment, n_template),
    ln_b (n_segment,), rows."""
    seg = read_segments(segments, rows)
    geometry = _node_geometry(bank, store, ymin)
    yN = geometry[2]
    nn = store.s.size
    groups = [list(range(g, min(g + nodes_per_launch, nn))) for g in range(0, nn, nodes_per_launch)]
    lnW = np.full(yN.shape, -np.inf)
    for m in range(bank.M.size):
        pts = np.nonzero(yN[:, m] > 0)[0]
        if pts.size < 2:
            raise ValueError("template %d (M %.2f, u %.3f) reaches fewer than two store nodes" % (m, bank.M[m], bank.u[m]))
        order = pts[np.argsort(yN[pts, m])]
        lnW[order, m] = distance_weights(yN[order, m], prior, ymin)

    out = np.full((seg["rows"].size, bank.u.size), -np.inf)
    chunk = _fit_chunk(bank, store, chunk, nodes_per_launch, resident)
    for ev in _chunks(seg, chunk):
        lamN = _node_values(bank, seg, ev, store, geometry, groups, block, resident)
        for j, e in enumerate(ev):
            v = lnW + lamN[j]
            mx = v.max(axis=0)
            out[e] = mx + np.log(np.exp(v - mx[None, :]).sum(axis=0))

    cell = np.ones(bank.u.size)
    for ax in (bank.M, bank.q, bank.chi):
        vals = np.unique(ax)
        if vals.size > 1:
            cell *= np.gradient(vals)[np.searchsorted(vals, ax)]
    w = cell / cell.sum()
    ln_b = np.array([_log_mean(L, w) for L in out])
    return dict(M=bank.M, q=bank.q, chi=bank.chi, u=bank.u, ln_lambda=out, ln_b=ln_b,
                rows=seg["rows"])
