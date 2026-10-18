"""
single-segment parameter estimation on the GPU.

pe_redshift   (m1_source, z) at equal mass and zero spin, the distance
              a parameter through y = D0 / D_L(z)
pe_intrinsic  (M_det, q, chi) on a template bank, the distance
              integrated inside the tables

both return ln Lambda, the likelihood ratio against noise marginalized
over the sky position, orientation, polarization, phase and arrival
time (over the whole segment), and ln B, its prior average.
"""

import numpy as np
import cupy as cp

from . import kernel, store as _store
from .bank import D0, read_segments, select_times


def _dl_of_z(z):
    from astropy.cosmology import Planck18
    return Planck18.luminosity_distance(np.atleast_1d(z)).value


def _chunks(seg, chunk):
    n = seg["rows"].size
    for c0 in range(0, n, chunk):
        yield range(c0, min(c0 + chunk, n))


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


def _to_source_frame(lamN, yN, Mdet, yq, m1, z):
    """node values (n_node + 1, n_template) -> ln Lambda on (m1, z): linear in y
    per template, then linear in M_det = 2 m1 (1 + z) per z"""
    nm = Mdet.size
    surf = np.full((nm, z.size), -np.inf)
    for m in range(nm):
        v = np.isfinite(lamN[:, m]) & (yN[:, m] > 0)
        if v.sum() < 2:
            continue
        yv, Lv = yN[v, m], lamN[v, m]
        o = np.argsort(yv)
        yv, Lv = yv[o], Lv[o]
        _, iu = np.unique(yv, return_index=True)
        yv, Lv = yv[iu], Lv[iu]
        ins = (yq >= yv[0]) & (yq <= yv[-1])
        surf[m] = np.where(ins, np.interp(yq, yv, Lv), -np.inf)
    lam = np.full((m1.size, z.size), -np.inf)
    Md = 2.0 * m1[:, None] * (1.0 + z[None, :])
    for iz in range(z.size):
        col = surf[:, iz]
        good = np.isfinite(col)
        if good.sum() < 2:
            continue
        mg = Mdet[good]
        inb = (Md[:, iz] >= mg[0]) & (Md[:, iz] <= mg[-1])
        lam[inb, iz] = np.interp(Md[inb, iz], mg, col[good])
    return lam


def _log_mean(lam, w):
    """ln sum_i w_i exp(lam_i) for normalized weights w, lam = -inf allowed"""
    fin = np.isfinite(lam) & (w > 0)
    if not fin.any():
        return -np.inf
    mx = lam[fin].max()
    return mx + np.log(np.sum(w[fin] * np.exp(lam[fin] - mx)))


def pe_redshift(segments, bank, store, rows=None, chunk=8, zmax=8.0, nz=1600,
                m1_range=(3.0, 150.0), nm1=1600, ymin=0.1267, n_loud=1024, n_random=1024,
                nodes_per_launch=6, block=256, seed=1234, mdet_range=(10.0, 500.0)):
    """ln Lambda(m1_source, z) of each segment.

    segments          anything read_segments takes (see there)
    bank              mass_bank of a one-column bank of detector-frame total
                      masses, ascending, built on the same segments
    store             load_redshift_store
    rows              segment indices (default all)
    chunk             segments filtered at once; the device holds
                      16 n_template n_time bytes of filter output per segment
    zmax, nz          z grid linspace(0.02, zmax, nz)
    m1_range, nm1     m1 grid geomspace(*m1_range, nm1)
    ymin              smallest y = D0 / D_L the store represents
    n_loud, n_random  arrival times visited per template (select_times);
                      n_loud = 0 visits all of them
    nodes_per_launch  store nodes evaluated in one kernel launch
    block             threads per block (256 when nodes_per_launch > 8 or so)
    seed              the j-th segment of each chunk draws with seed + j
    mdet_range        support of the ln B prior

    ln B averages Lambda over the prior uniform in M_det = 2 m1 (1+z) on
    mdet_range and proportional to D_L^2 in luminosity distance, i.e.
    p(m1, z) ~ (1+z) D_L^2 dD_L/dz, on the (m1, z) cells.

    returns dict: m1 (nm1,), z (nz,), ln_lambda (n_segment, nm1, nz),
    ln_b (n_segment,), rows."""
    if np.any(np.diff(bank.M) <= 0) or np.any(bank.q != 1.0) or np.any(bank.chi != 0.0):
        raise ValueError("pe_redshift needs an ascending equal-mass zero-spin bank")
    seg = read_segments(segments, rows)
    z = np.linspace(0.02, zmax, int(nz))
    m1 = np.geomspace(m1_range[0], m1_range[1], int(nm1))
    yq = D0 / _dl_of_z(z)
    valid, lna, yN, bnode, blna = _node_geometry(bank, store, ymin)
    nn = store.s.size
    groups = [list(range(g, min(g + nodes_per_launch, nn))) for g in range(0, nn, nodes_per_launch)]
    use_sel = n_loud > 0

    out = np.full((seg["rows"].size, m1.size, z.size), -np.inf)
    for ev in _chunks(seg, chunk):
        traj = [_filter(bank, seg, e) for e in ev]
        sel = [select_times(t, store.tau_idx, n_loud, n_random, seed + j) if use_sel else (None, None)
               for j, t in enumerate(traj)]
        lamN = np.full((len(ev), nn + 1, bank.M.size), -np.inf)
        for ks in groups:
            k0 = ks[0]
            dev = _store.upload([store.tables[k] for k in ks], store.tau_idx,
                                store.beta[k0], store.delta[k0])
            ok = valid[ks]
            ids = np.nonzero(ok.any(axis=0))[0]
            lnaM = np.where(ok[:, ids], lna[ks][:, ids], 0.0).T
            tab = np.tile(np.arange(len(ks), dtype=np.int32), (ids.size, 1))
            bids = np.nonzero((bnode >= k0) & (bnode <= ks[-1]))[0]
            for j in range(len(ev)):
                tsel, tlw = sel[j]
                if ids.size:
                    val = kernel.gather(traj[j], dev, ids, lnaM, tab, tsel, tlw, block)
                    for i, k in enumerate(ks):
                        keep = ok[i][ids]
                        lamN[j, k, ids[keep]] = val[i][keep]
                if bids.size:
                    val = kernel.gather(traj[j], dev, bids, blna[bids][:, None],
                                        (bnode[bids] - k0)[:, None], tsel, tlw, block)
                    lamN[j, nn, bids] = val[0]
            del dev
            _store.free()
        for j, e in enumerate(ev):
            out[e] = _to_source_frame(lamN[j], yN, bank.M, yq, m1, z)
        del traj, sel
        _store.free()

    dl = _dl_of_z(z)
    lnp = np.log(np.maximum((1.0 + z) * dl ** 2 * np.gradient(dl, z), 1e-300))
    Md = 2.0 * m1[:, None] * (1.0 + z[None, :])
    inside = (Md >= mdet_range[0]) & (Md <= mdet_range[1])
    w = np.where(inside, np.exp(lnp)[None, :], 0.0) * np.gradient(m1)[:, None] * np.gradient(z)[None, :]
    w /= w.sum()
    ln_b = np.array([_log_mean(L, w) for L in out])
    return dict(m1=m1, z=z, ln_lambda=out, ln_b=ln_b, rows=seg["rows"])


def pe_intrinsic(segments, bank, store, rows=None, chunk=4, group=10, n_loud=1024, n_random=1024,
                 block=512, seed=1234):
    """ln Lambda of each template of the bank, distance integrated.

    segments          anything read_segments takes (see there)
    bank              mass_bank built on the same segments
    store             load_intrinsic_store
    rows              segment indices (default all)
    chunk             segments filtered at once (16 n_template n_time
                      bytes of device memory each)
    group             tables per copy to the device (plus 3 for the stencil)
    n_loud, n_random  arrival times visited per template (select_times);
                      n_loud = 0 visits all of them
    block             threads per block
    seed              the j-th segment of each chunk draws with seed + j

    each template reads the four tables around its power u at ln a = 0,
    and the four values are combined by Lagrange interpolation in ln u.
    ln B is the average of Lambda over the bank, each template weighted
    by its cell (the product of the spacings of the distinct values of
    every axis that varies).

    returns dict: M, q, chi, u (n_template,), ln_lambda (n_segment,
    n_template), ln_b (n_segment,), rows."""
    seg = read_segments(segments, rows)
    U = store.u
    if bank.u.min() < U[0] or bank.u.max() > U[-1]:
        raise ValueError("template power %.3f..%.3f outside the store %.3f..%.3f"
                         % (bank.u.min(), bank.u.max(), U[0], U[-1]))
    il = np.clip(np.searchsorted(U, bank.u) - 1, 0, U.size - 2)
    j0 = np.clip(il - 1, 0, U.size - 4)
    stencil = j0[:, None] + np.arange(4)[None, :]
    xs = np.log(U)[stencil]
    x = np.log(bank.u)
    wl = np.ones((bank.u.size, 4))
    for a in range(4):
        for b in range(4):
            if a != b:
                wl[:, a] *= (x - xs[:, b]) / (xs[:, a] - xs[:, b])
    groups = []
    for g0 in range(0, U.size - 3, group):
        ids = np.nonzero((j0 >= g0) & (j0 < g0 + group))[0]
        if ids.size:
            groups.append((list(range(g0, min(g0 + group + 3, U.size))), ids))
    use_sel = n_loud > 0

    out = np.full((seg["rows"].size, bank.u.size), -np.inf)
    for ev in _chunks(seg, chunk):
        traj = [_filter(bank, seg, e) for e in ev]
        sel = [select_times(t, store.tau_idx, n_loud, n_random, seed + j, rest_order="time")
               if use_sel else (None, None) for j, t in enumerate(traj)]
        for tabs, ids in groups:
            dev = _store.upload([store.tables[i] for i in tabs], store.tau_idx, store.beta, store.delta)
            tab = (stencil[ids] - tabs[0]).astype(np.int32)
            lna = np.zeros((ids.size, 4))
            for j, e in enumerate(ev):
                v = kernel.gather(traj[j], dev, ids, lna, tab, sel[j][0], sel[j][1], block)
                out[e, ids] = (wl[ids].T * v).sum(axis=0)
            del dev
            _store.free()
        del traj, sel
        _store.free()

    cell = np.ones(bank.u.size)
    for ax in (bank.M, bank.q, bank.chi):
        vals = np.unique(ax)
        if vals.size > 1:
            cell *= np.gradient(vals)[np.searchsorted(vals, ax)]
    w = cell / cell.sum()
    ln_b = np.array([_log_mean(L, w) for L in out])
    return dict(M=bank.M, q=bank.q, chi=bank.chi, u=bank.u, ln_lambda=out, ln_b=ln_b,
                rows=seg["rows"])
