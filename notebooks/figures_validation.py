"""Figures 16 to 21 of the paper from the files in data/results/.

    import numpy as np
    from figures_validation import *
    fig = fig_lnlam_nmax(np.load('../data/results/convergence.npz'))
"""
from contextlib import contextmanager

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter
from scipy.stats import kstest

from exmarg import figstyle

__all__ = ['fig_lnlam_nmax', 'fig_posterior_nmax', 'fig_loud_ladder', 'fig_pp_3p', 'fig_pp_2p',
           'fig_corner_3p', 'fig_corner_2p']


@contextmanager
def _style(**extra):
    """Paper rcParams for one figure (draw the figure inside the block so every artist keeps them)."""
    with mpl.rc_context():
        mpl.rcdefaults()
        figstyle.apply()
        plt.rcParams.update(extra)
        yield


def _levels(P, fracs=(0.9, 0.5)):
    """Density levels enclosing the fractions fracs of P."""
    s = np.sort(P.ravel())[::-1]
    c = np.cumsum(s) / s.sum()
    out = [float(s[min(np.searchsorted(c, f), s.size - 1)]) for f in fracs]
    return [v for i, v in enumerate(out) if i == 0 or v > out[i - 1]]


def _two_ticks(lo, hi):
    """Two round ticks at a quarter and three quarters of [lo, hi]."""
    w = hi - lo
    dec = max(0, int(np.ceil(-np.log10(w / 4.0))))
    t = np.round([lo + 0.25 * w, lo + 0.75 * w], dec)
    return t if t[1] > t[0] else np.round([lo + 0.25 * w, lo + 0.75 * w], dec + 1)


def fig_lnlam_nmax(d):
    """Fig. 16 (lnlmax.pdf); d = convergence.npz (nmax, lnlam, rho_opt)."""
    FS_LAB, FS_TK, FS_IN = 12, 10, 9.0
    with _style(**{'xtick.labelsize': FS_TK, 'ytick.labelsize': FS_TK, 'xtick.major.width': 1.0,
                   'ytick.major.width': 1.0, 'axes.linewidth': 1.0}):
        NM, L, rho = np.asarray(d['nmax'], float), np.asarray(d['lnlam']), np.asarray(d['rho_opt'])
        fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.7))
        nfine = np.linspace(NM[0], 120, 400)
        XL = np.linspace(30, 100, 5)
        ax.set_xlim(0, 125); ax.set_ylim(-8, L.max() * 1.08)
        for nm in (40, 80, 120):
            ax.axvline(nm, color='0.85', lw=1.0, zorder=0)
        ax.set_xlabel(r'$n_{\max}$', fontsize=FS_LAB)
        ax.set_ylabel(r'$\ln\Lambda_{n_{\max}}$', fontsize=FS_LAB)
        fig.tight_layout(pad=0.2, w_pad=1.2)
        for k, y in enumerate(L):
            smooth = PchipInterpolator(NM, y)
            nf = nfine[nfine >= NM[0]]; yf = smooth(nf)
            xl = XL[k]
            yp = yf.copy(); yp[(nf > xl - 13.5) & (nf < xl + 13.5)] = np.nan
            ax.plot(nf, yp, color='k', lw=1.6)
            dx = 4.0
            p1 = ax.transData.transform((xl - dx, float(smooth(xl - dx))))
            p2 = ax.transData.transform((xl + dx, float(smooth(xl + dx))))
            ang = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))
            ax.text(xl, float(smooth(xl)), r'$\rho_{\rm opt} = %.0f$' % rho[k], ha='center', va='center',
                    fontsize=FS_IN - 1.0, color='k', rotation=ang, rotation_mode='anchor')
        fig.canvas.draw()
    plt.close(fig)
    return fig


def fig_posterior_nmax(d):
    """Fig. 17 (grid_nmax_dl.pdf); d = convergence.npz (post_*, true_*); save with bbox_inches='tight'."""
    names = [r'$\mathcal{M}_c\,[M_\odot]$', r'$q$', r'$\chi_{\rm eff}$']
    axes = [np.asarray(d['post_Mc']), np.asarray(d['post_q']), np.asarray(d['post_chi'])]
    truth = [float(d['true_Mc']), float(d['true_q']), float(d['true_chi'])]
    cubes = np.asarray(d['post_cube'])
    labels = ['%d' % v for v in d['post_nmax']]
    cols = [figstyle.STEEL, figstyle.RED, figstyle.NAVY]
    styles = ['-', '-', '--']
    tag = r'$\rho_{\rm opt} = %.0f$' % float(d['post_rho_opt'])
    size, FL, FT = 3.15, 12, 10
    LW1, LW2, LWT = 1.0, 1.8, 0.8
    with _style(**{'xtick.labelsize': FT, 'ytick.labelsize': FT, 'xtick.major.width': 1.0,
                   'ytick.major.width': 1.0, 'axes.linewidth': 1.0}):
        fig, ax = plt.subplots(3, 3, figsize=(size, size), gridspec_kw=dict(hspace=0.0, wspace=0.0))
        for i in range(3):
            for j in range(3):
                a = ax[i, j]
                if j > i:
                    a.axis('off')
                    if i == 0 and j == 2:
                        a.text(0.98, 0.96, tag, transform=a.transAxes, ha='right', va='top', fontsize=FL)
                    continue
                for k, (H, col) in enumerate(zip(cubes, cols)):
                    P = np.maximum(H, 1e-300) / H.max()
                    if i == j:
                        m = P.sum(axis=tuple(kk for kk in range(3) if kk != i))
                        m /= np.trapezoid(m, axes[i])
                        a.plot(axes[i], m, color=col, lw=LW2, ls=styles[k], zorder=2 + k)
                    else:
                        m2 = P.sum(axis=3 - i - j).T
                        s = np.sort(m2.ravel())[::-1]
                        c = np.cumsum(s) / s.sum()
                        lv = [s[np.searchsorted(c, f)] for f in (0.9, 0.5)]
                        a.contour(axes[j], axes[i], m2, levels=lv, colors=[col], linewidths=(LW1, LW2),
                                  linestyles=styles[k], zorder=2 + k)
                if i == j:
                    a.axvline(truth[i], color='0.35', lw=LWT)
                    a.set_yticks([])
                else:
                    a.axvline(truth[j], color='0.35', lw=LWT)
                    a.axhline(truth[i], color='0.35', lw=LWT)
                    a.set_ylim(axes[i][0], axes[i][-1])
                a.set_xlim(axes[j][0], axes[j][-1])
                a.xaxis.set_major_locator(MaxNLocator(3, prune='both'))
                if i != j:
                    a.yaxis.set_major_locator(MaxNLocator(3, prune='both'))
                a.tick_params(labelsize=FT)
                a.tick_params(axis='x', labelrotation=45, pad=1)
                if i < 2:
                    a.set_xticklabels([])
                else:
                    a.set_xlabel(names[j], fontsize=FL)
                if j > 0:
                    a.set_yticklabels([])
                elif i > 0:
                    a.set_ylabel(names[i], fontsize=FL)
        fig.align_xlabels(ax[2, :])
        fig.align_ylabels(ax[:, 0])
        n = len(cubes)
        cax = ax[0, 2].inset_axes([0.08, 0.02, 0.14, 0.72])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=BoundaryNorm(np.arange(n + 1), n), cmap=ListedColormap(cols)),
                          cax=cax, ticks=np.arange(n) + 0.5)
        cb.ax.set_yticklabels(labels)
        cb.ax.tick_params(labelsize=FT, length=0)
        cb.set_label(r'$n_{\max}$', fontsize=FL, labelpad=3)
        fig.canvas.draw()
    plt.close(fig)
    return fig


def fig_loud_ladder(d):
    """Fig. 18 (loud_ladder.pdf); d = loud_ladder.npz (nmax, rho_opt, lnlam_trunc, lnlam_ref)."""
    with _style(**{'font.size': 10, 'axes.labelsize': 12, 'xtick.labelsize': 10, 'ytick.labelsize': 10}):
        orders, rho = np.asarray(d['nmax']), np.asarray(d['rho_opt'])
        y = (np.asarray(d['lnlam_trunc']) - np.asarray(d['lnlam_ref'])[None, :]) / (rho ** 2 / 2.0)
        INK = figstyle.NAVY
        rgb = np.array([int(INK[i:i + 2], 16) / 255 for i in (1, 3, 5)])
        WASH = tuple(1.0 - 0.28 * (1.0 - rgb))
        o = np.argsort(rho); rk = rho[o]
        W = max(7, rk.size // 12)

        def band(v):
            win = [v[max(0, k - W // 2):k + W // 2 + 1] for k in range(v.size)]
            return np.array([np.percentile(w, 16) for w in win]), np.array([np.percentile(w, 84) for w in win])

        fig, axs = plt.subplots(2, 3, figsize=(7.0, 3.4), sharex=True, sharey=True)
        for a, i in zip(axs.ravel(), range(len(orders))):
            lo, hi = band(y[i][o])
            a.axhline(0.0, color='0.45', lw=0.8)
            a.fill_between(rk, lo, hi, color=WASH, lw=0, zorder=2)
            a.plot(rk, 0.5 * (lo + hi), lw=1.8, color=INK, zorder=3)
            a.set_ylim(-1.12, 0.15); a.set_yticks([-1.0, -0.5, 0.0])
            a.set_xlim(np.floor(rk.min()), np.ceil(rk.max()))
            a.text(0.04, 0.08, r'$n_{\max} = %d$' % orders[i], transform=a.transAxes, fontsize=11, color=INK)
            a.grid(color='0.82', lw=0.5, alpha=0.7)
        for a in axs[1]:
            a.set_xlabel(r'$\rho_{\rm opt}$')
        fig.supylabel(r'$\left[\ln\Lambda^\star - \ln\Lambda\right] / (\rho_{\rm opt}^2/2)$', fontsize=12, x=0.01)
        fig.tight_layout(pad=0.3, h_pad=0.5, w_pad=0.6)
        fig.canvas.draw()
    plt.close(fig)
    return fig


def _pp(qs, names, cols):
    """P-P panel of Fig. 19; qs = credible levels (events x parameters)."""
    with _style(**{'axes.labelsize': 12, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
                   'legend.fontsize': 10, 'axes.linewidth': 0.9}):
        n = len(qs)
        fig, a = plt.subplots(figsize=(2.94, 2.94))
        grid = np.linspace(0, 1, 200)
        eps = 1.36 / np.sqrt(n)
        a.fill_between(grid, np.clip(grid - eps, 0, 1), np.clip(grid + eps, 0, 1), color='0.9')
        for i in range(qs.shape[1]):
            x = np.sort(qs[:, i]); y = np.arange(1, n + 1) / n
            pks = kstest(qs[:, i], 'uniform').pvalue
            a.step(x, y, where='post', color=cols[i], lw=1.6, label='%s (KS p=%.2g)' % (names[i], pks))
        a.plot([0, 1], [0, 1], 'k--', lw=0.8)
        a.set_xlabel('Credible level'); a.set_ylabel('Empirical fraction')
        a.set_xlim(0, 1); a.set_ylim(0, 1)
        a.legend(frameon=False, loc='upper left', handlelength=1.4, borderaxespad=0.3)
        fig.tight_layout(pad=0.2)
        fig.canvas.draw()
    plt.close(fig)
    return fig


def fig_pp_3p(d):
    """Fig. 19 left (pp.pdf); d = pp_3p.npz (cred_Mc, cred_q, cred_chi)."""
    qs = np.stack([d['cred_Mc'], d['cred_q'], d['cred_chi']], axis=1)
    return _pp(qs, [r'$\mathcal{M}_c$', r'$q$', r'$\chi_{\rm eff}$'], [figstyle.NAVY, '#C9807A', '#7FA8C9'])


def fig_pp_2p(d):
    """Fig. 19 right (pp_2p.pdf); d = pp_2p.npz (cred_m1, cred_z)."""
    qs = np.stack([d['cred_m1'], d['cred_z']], axis=1)
    return _pp(qs, [r'$m_1^{\rm src}$', r'$z$'], [figstyle.NAVY, '#C9807A'])


def _corner_axes(ax, i, j, n, kx, ky, rx, ry, LAB, XL, YL, FS_AX, FS_TK):
    """Limits, ticks and labels of one corner cell of Figs. 20 and 21."""
    if i != j:
        ax.set_ylim(ry); ax.set_yticks(_two_ticks(*ry))
    ax.set_xlim(rx); ax.set_xticks(_two_ticks(*rx))
    ax.tick_params(labelsize=FS_TK, length=2.5, width=0.7, pad=1)
    ax.tick_params(axis='x', labelrotation=60)
    if i < n - 1:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel(LAB[kx], fontsize=FS_AX, labelpad=2)
        XL.append(ax)
        for t in ax.get_xticklabels():
            t.set_ha('right'); t.set_rotation_mode('anchor')
    if j > 0 or i == 0:
        ax.set_yticklabels([])
    if j == 0 and i > 0:
        ax.set_ylabel(LAB[ky], fontsize=FS_AX, labelpad=2)
        YL.append(ax)


def _corner_legend(fig, XL, OURS, SAMP, GREY, FS_CAP):
    """Shared legend of Figs. 20 and 21."""
    fig.align_xlabels(XL)
    h = [plt.Line2D([], [], color=OURS, lw=2.5), plt.Line2D([], [], color=SAMP, lw=2.5),
         plt.Line2D([], [], color=GREY, lw=1.2)]
    fig.legend(h, [r'$\Lambda^\star_{n_{\max}=120}$', r'$\Lambda$, nested sampling', 'Injection'], frameon=False,
               ncol=3, fontsize=FS_CAP, loc='upper center', bbox_to_anchor=(0.5, 1.0))


def fig_corner_3p(d):
    """Fig. 20 (corner_3p_row.pdf); d = ns_3p.npz (e0_*, e1_*)."""
    OURS, SAMP, GREY = figstyle.NAVY, figstyle.RED, '0.35'
    KEYS = ['Mc', 'q', 'chi', 'ra', 'dec', 'ci', 'psi']
    BK = {'Mc': 'Mc', 'q': 'q', 'chi': 'chi', 'ra': 'ra', 'dec': 'dec', 'ci': 'cos_iota', 'psi': 'psi'}
    LAB = {'Mc': r'$\mathcal{M}_c$', 'q': r'$q$', 'chi': r'$\chi_{\rm eff}$', 'ra': 'RA', 'dec': 'Dec',
           'ci': r'$\cos\iota$', 'psi': r'$\psi$'}
    FIXED = {'ra': (0, 2 * np.pi), 'dec': (-np.pi / 2, np.pi / 2), 'ci': (-1, 1), 'psi': (0, np.pi)}
    FS_AX, FS_TK, FS_CAP, LW_T = 12, 9.5, 11, 0.8
    n = len(KEYS)
    XL = []
    with _style():
        fig = plt.figure(figsize=(7.0, 4.1))
        outer = GridSpec(1, 2, figure=fig, wspace=0.2, left=0.085, right=0.995, top=0.88, bottom=0.15)
        for b in range(2):
            e = 'e%d_' % b
            OUR = {k: np.asarray(d[e + 'ours_' + k]) for k in ('Mc', 'q', 'chi')}
            S = {k: np.asarray(d[e + 'bilby_' + BK[k]]) for k in KEYS}
            wb = np.asarray(d[e + 'bilby_weight'])
            TRU = {k: float(d[e + 'true_' + k]) for k in ('Mc', 'q', 'chi')}

            def rng_(k):
                if k in FIXED:
                    return FIXED[k]
                x = S[k]; lo_o, hi_o = np.percentile(OUR[k], [0.3, 99.7])
                lo = min(np.percentile(x, 0.3), lo_o, TRU[k]); hi = max(np.percentile(x, 99.7), hi_o, TRU[k])
                pad = 0.08 * (hi - lo); return lo - pad, hi + pad

            YL = []
            gs = GridSpecFromSubplotSpec(n, n, subplot_spec=outer[b], wspace=0.07, hspace=0.07)
            for i in range(n):
                for j in range(i + 1):
                    ax = fig.add_subplot(gs[i, j]); kx, ky = KEYS[j], KEYS[i]; rx, ry = rng_(kx), rng_(ky)
                    if i == j:
                        ax.hist(S[kx], bins=26, range=rx, weights=wb, density=True, histtype='step', color=SAMP, lw=1.5)
                        if kx in OUR:
                            ax.hist(OUR[kx], bins=26, range=rx, density=True, histtype='step', color=OURS, lw=1.8)
                            ax.axvline(TRU[kx], color=GREY, lw=LW_T)
                        ax.set_yticks([])
                    else:
                        H, xe, ye = np.histogram2d(S[kx], S[ky], bins=22, range=[rx, ry], weights=wb)
                        if H.sum():
                            H = gaussian_filter(H, 1.0); H /= H.sum(); lv = _levels(H)
                            if lv:
                                ax.contour(0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1]), H.T, levels=lv,
                                           colors=[SAMP], linewidths=(1.0, 1.8))
                        if kx in OUR and ky in OUR:
                            H2, xe, ye = np.histogram2d(OUR[kx], OUR[ky], bins=22, range=[rx, ry])
                            H2 = gaussian_filter(H2, 1.0); H2 /= H2.sum(); lv = _levels(H2)
                            if lv:
                                ax.contour(0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1]), H2.T, levels=lv,
                                           colors=[OURS], linewidths=(1.0, 1.8))
                        if kx in TRU:
                            ax.axvline(TRU[kx], color=GREY, lw=LW_T)
                        if ky in TRU:
                            ax.axhline(TRU[ky], color=GREY, lw=LW_T)
                    _corner_axes(ax, i, j, n, kx, ky, rx, ry, LAB, XL, YL, FS_AX, FS_TK)
            fig.align_ylabels(YL[-(n - 1):])
            cap = fig.add_subplot(gs[0:3, 4:]); cap.axis('off')
            cap.text(0.0, 0.9, r'$\rho_{\rm opt}=%.1f$' % float(d[e + 'rho']), fontsize=FS_CAP + 1, va='top',
                     transform=cap.transAxes)
            cap.text(0.0, 0.55, r'$\ln \mathcal{B} = %.2f$' % float(d[e + 'lnB_ours']), fontsize=FS_CAP, color=OURS,
                     va='top', transform=cap.transAxes)
            cap.text(0.0, 0.33, r'$%.2f \pm %.2f$' % (float(d[e + 'lnB_bilby']), float(d[e + 'lnB_bilby_err'])),
                     fontsize=FS_CAP, color=SAMP, va='top', transform=cap.transAxes)
        _corner_legend(fig, XL, OURS, SAMP, GREY, FS_CAP)
        fig.canvas.draw()
    plt.close(fig)
    return fig


def fig_corner_2p(d, rho):
    """Fig. 21 (corner_2p_row.pdf); d = ns_2p.npz (e0_*, e1_*), rho = ns_2p_rho.npz (e0_rho, e1_rho)."""
    OURS, SAMP, GREY = figstyle.NAVY, figstyle.RED, '0.35'
    KEYS = ['m1', 'z', 'ra', 'dec', 'ci', 'psi']
    BK = {'m1': 'm1', 'z': 'z', 'ra': 'ra', 'dec': 'dec', 'ci': 'cos_iota', 'psi': 'psi'}
    LAB = {'m1': r'$m_1^{\rm src}$', 'z': r'$z$', 'ra': 'RA', 'dec': 'Dec', 'ci': r'$\cos\iota$', 'psi': r'$\psi$'}
    FIXED = {'ra': (0.0, 2 * np.pi), 'dec': (-np.pi / 2, np.pi / 2), 'ci': (-1.0, 1.0), 'psi': (0.0, np.pi)}
    FS_AX, FS_TK, FS_CAP, LW_T = 12, 9.5, 11, 0.8
    n = len(KEYS)
    XL = []
    with _style():
        fig = plt.figure(figsize=(7.0, 4.1))
        outer = GridSpec(1, 2, figure=fig, wspace=0.2, left=0.085, right=0.995, top=0.88, bottom=0.15)
        for b in range(2):
            e = 'e%d_' % b
            m1g, zg, post = np.asarray(d[e + 'm1_grid']), np.asarray(d[e + 'z_grid']), np.asarray(d[e + 'ours_posterior'])
            S = {k: np.asarray(d[e + 'bilby_' + BK[k]]) for k in KEYS}
            TRU = {'m1': float(d[e + 'true_m1']), 'z': float(d[e + 'true_z'])}

            def rng(key):
                if key in FIXED:
                    return FIXED[key]
                g, p = (m1g, post.sum(1)) if key == 'm1' else (zg, post.sum(0))
                c = np.cumsum(p) / p.sum()
                lo, hi = np.interp([0.004, 0.996], c, g)
                lo = min(lo, np.percentile(S[key], 0.4), TRU[key])
                hi = max(hi, np.percentile(S[key], 99.6), TRU[key])
                pad = 0.07 * (hi - lo)
                return lo - pad, hi + pad

            YL = []
            gs = GridSpecFromSubplotSpec(n, n, subplot_spec=outer[b], wspace=0.08, hspace=0.08)
            for i in range(n):
                for j in range(i + 1):
                    ax = fig.add_subplot(gs[i, j])
                    kx, ky = KEYS[j], KEYS[i]
                    rx, ry = rng(kx), rng(ky)
                    if i == j:
                        ax.hist(S[kx], bins=26, range=rx, density=True, histtype='step', color=SAMP, lw=1.5)
                        if kx in ('m1', 'z'):
                            g, p = (m1g, post.sum(1)) if kx == 'm1' else (zg, post.sum(0))
                            ax.plot(g, p / np.gradient(g) / p.sum(), color=OURS, lw=1.8)
                            ax.axvline(TRU[kx], color=GREY, lw=LW_T)
                        ax.set_yticks([])
                    else:
                        H, xe, ye = np.histogram2d(S[kx], S[ky], bins=22, range=[rx, ry])
                        if H.sum():
                            H = gaussian_filter(H, 1.1); H /= H.sum()
                            lv = _levels(H)
                            if lv:
                                ax.contour(0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1]), H.T,
                                           levels=lv, colors=[SAMP], linewidths=(1.0, 1.8))
                        if (kx, ky) == ('m1', 'z'):
                            lv = _levels(post)
                            if lv:
                                ax.contour(m1g, zg, post.T, levels=lv, colors=[OURS], linewidths=(1.0, 1.8))
                        if kx in TRU:
                            ax.axvline(TRU[kx], color=GREY, lw=LW_T)
                        if ky in TRU:
                            ax.axhline(TRU[ky], color=GREY, lw=LW_T)
                    _corner_axes(ax, i, j, n, kx, ky, rx, ry, LAB, XL, YL, FS_AX, FS_TK)
            fig.align_ylabels(YL[-(n - 1):])
            cap = fig.add_subplot(gs[0:3, 3:])
            cap.axis('off')
            cap.text(0.0, 0.88, r'$\rho_{\rm opt}=%.1f$' % float(rho[e + 'rho']), ha='left', va='top',
                     fontsize=FS_CAP + 1, transform=cap.transAxes)
            cap.text(0.0, 0.60, r'$\ln \mathcal{B} = %.2f$' % float(d[e + 'lnB_ours']), ha='left', va='top',
                     fontsize=FS_CAP, color=OURS, transform=cap.transAxes)
            cap.text(0.0, 0.42, r'$%.2f \pm %.2f$' % (float(d[e + 'lnB_bilby']), float(d[e + 'lnB_bilby_err'])),
                     ha='left', va='top', fontsize=FS_CAP, color=SAMP, transform=cap.transAxes)
        _corner_legend(fig, XL, OURS, SAMP, GREY, FS_CAP)
        fig.canvas.draw()
    plt.close(fig)
    return fig


def fig_corner_cogwheel(panels):
    """posteriors of (chirp mass, mass ratio, spin): ours (grid samples, 50 and 90 percent contours) and cogwheel at each of
    its settings (weighted samples, 90 percent contour, one color per setting), one corner per segment.
    panels = list of dicts: ours_<k>, true_<k> for k in Mc, q, chi; rho; cog = list of dicts with Mc, q, chi, weight, color, label."""
    OURS, GREY = figstyle.NAVY, '0.35'
    KEYS = ['Mc', 'q', 'chi']
    LAB = {'Mc': r'$\mathcal{M}_c$', 'q': r'$q$', 'chi': r'$\chi_{\rm eff}$'}
    n = len(KEYS)

    def wq(x, w, p):
        o = np.argsort(x); c = np.cumsum(w[o]); return np.interp(p * c[-1], c, x[o])

    with _style():
        fig = plt.figure(figsize=(4.2 * len(panels), 5.4))
        outer = GridSpec(1, len(panels), figure=fig, wspace=0.25, left=0.07, right=0.98, top=0.88, bottom=0.3)
        for b, d in enumerate(panels):
            OUR = {k: np.asarray(d['ours_' + k]) for k in KEYS}
            TRU = {k: float(d['true_' + k]) for k in KEYS}

            def rng_(k):
                lo = min([np.percentile(OUR[k], 0.3), TRU[k]] + [wq(np.asarray(c[k]), np.asarray(c['weight']), 0.003) for c in d['cog']])
                hi = max([np.percentile(OUR[k], 99.7), TRU[k]] + [wq(np.asarray(c[k]), np.asarray(c['weight']), 0.997) for c in d['cog']])
                pad = 0.08 * (hi - lo)
                return lo - pad, hi + pad

            gs = GridSpecFromSubplotSpec(n, n, subplot_spec=outer[b], wspace=0.08, hspace=0.08)
            for i in range(n):
                for j in range(i + 1):
                    ax = fig.add_subplot(gs[i, j]); kx, ky = KEYS[j], KEYS[i]; rx, ry = rng_(kx), rng_(ky)
                    if i == j:
                        for c in d['cog']:
                            ax.hist(c[kx], bins=30, range=rx, weights=c['weight'], density=True, histtype='step', color=c['color'], lw=1.1)
                        ax.hist(OUR[kx], bins=30, range=rx, density=True, histtype='step', color=OURS, lw=2.2)
                        ax.axvline(TRU[kx], color=GREY, lw=0.8)
                        ax.set_yticks([])
                    else:
                        for c in d['cog']:
                            H, xe, ye = np.histogram2d(c[kx], c[ky], bins=26, range=[rx, ry], weights=c['weight'])
                            H = gaussian_filter(H, 1.0); H /= H.sum(); lv = _levels(H, fracs=(0.9,))
                            if lv:
                                ax.contour(0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1]), H.T, levels=lv, colors=[c['color']], linewidths=1.0)
                        H, xe, ye = np.histogram2d(OUR[kx], OUR[ky], bins=26, range=[rx, ry])
                        H = gaussian_filter(H, 1.0); H /= H.sum(); lv = _levels(H)
                        if lv:
                            ax.contour(0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1]), H.T, levels=lv, colors=[OURS], linewidths=(1.2, 2.2))
                        ax.axvline(TRU[kx], color=GREY, lw=0.8); ax.axhline(TRU[ky], color=GREY, lw=0.8)
                    ax.set_xlim(rx)
                    if i != j:
                        ax.set_ylim(ry)
                    ax.tick_params(labelsize=8)
                    ax.xaxis.set_major_locator(MaxNLocator(3)); ax.yaxis.set_major_locator(MaxNLocator(3))
                    if i == n - 1:
                        ax.set_xlabel(LAB[kx], fontsize=12)
                    else:
                        ax.set_xticklabels([])
                    if j == 0 and i > 0:
                        ax.set_ylabel(LAB[ky], fontsize=12)
                    elif j > 0:
                        ax.set_yticklabels([])
            fig.text(0.07 + (b + 0.5) * 0.91 / len(panels), 0.92, r'SNR $= %.1f$' % d['rho'], fontsize=12, ha='center')
        handles = [plt.Line2D([], [], color=OURS, lw=2.2, label='This Work'), plt.Line2D([], [], color=GREY, lw=0.8, label='True Value')]
        seen = set()
        for d in panels:
            for c in d['cog']:
                if c['label'] not in seen:
                    seen.add(c['label']); handles.append(plt.Line2D([], [], color=c['color'], lw=1.1, label=c['label']))
        fig.legend(handles=handles, loc='lower center', frameon=False, fontsize=10, ncol=3)
    plt.close(fig)
    return fig
