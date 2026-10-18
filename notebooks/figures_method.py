"""Plotting functions for five method figures of the paper.

Each function takes the plotted arrays and returns a matplotlib Figure
drawn in the paper's style. The style is set inside a local rc context,
so calling one function does not change the look of another.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import (LogNorm, ListedColormap, BoundaryNorm,
                               SymLogNorm)
from scipy.signal import savgol_filter

from exmarg import figstyle

# rcParams of the figures drawn at their printed width
_THIN_AXES = {'axes.linewidth': 0.6,
              'xtick.major.width': 0.6,
              'ytick.major.width': 0.6,
              'xtick.major.size': 2.5,
              'ytick.major.size': 2.5,
              'xtick.major.pad': 2.0,
              'ytick.major.pad': 2.0}

LADDER_TITLES = (r'$u_1 = u_2 = u$',
                 r'$u_1 = u, \ u_2 = u / 10$',
                 r'$u_1 = u, \ u_2 = 0$')


def _style(fs_label, fs_tick, extra=None):
    """Reset to matplotlib defaults, then install figstyle and the printed sizes."""
    mpl.rcdefaults()
    figstyle.apply()
    rc = {'axes.labelsize': fs_label,
          'xtick.labelsize': fs_tick,
          'ytick.labelsize': fs_tick}
    rc.update(_THIN_AXES)
    if extra:
        rc.update(extra)
    plt.rcParams.update(rc)


def _darken(rgba, cap=0.55):
    """Scale a colour down in luminance so that it is at most `cap`."""
    r, g, b = rgba[:3]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    f = 1.0 if lum <= cap else cap / lum
    return (r * f, g * f, b * f, 1.0)


def fig_W_family(tau_s, us, keys, rows):
    """Fig. 3 (W_family): tau_s (n_tau,) delay grid [s], us (n_u,) template powers (one row of panels each), keys (n_key, 3) table keys (n, a, b), rows (n_u, n_key, n_tau) W^{(n)}_{ab}(tau) for each u and key."""
    FS_LABEL, FS_TICK, FS_PANEL = 12, 10, 12
    with plt.rc_context():
        _style(FS_LABEL, FS_TICK)
        tms = np.asarray(tau_s) * 1e3
        keys = [tuple(int(v) for v in k) for k in keys]
        nrw, ncl = len(us), len(keys)
        fig, axs = plt.subplots(nrw, ncl, figsize=(7.0, 2.7),
                                sharex=True, sharey=True, squeeze=False)
        fig.subplots_adjust(left=0.075, right=0.955, top=0.94,
                            bottom=0.15, wspace=0.0, hspace=0.0)
        for r_, u in enumerate(us):
            for c_, key in enumerate(keys):
                ax = axs[r_, c_]
                w = rows[r_][c_]
                # each curve scaled to max |W| = 1, smoothed for display
                y = savgol_filter(w / np.abs(w).max(), 7, 3)
                ax.plot(tms, y, lw=1.6, color=figstyle.NAVY)
                ax.axhline(0, color='0.85', lw=0.6)
                ax.set_xlim(tms.min(), tms.max())
                ax.set_ylim(-1.3, 1.3)
                if r_ == 0:
                    # key label at the top unless the curve fills the centre there
                    mid = slice(len(tms) * 35 // 100, len(tms) * 65 // 100)
                    top_free = y[mid].max() < 0.45
                    ax.text(0.5, 0.86 if top_free else 0.14,
                            r'$W^{(%d)}_{%d%d}$' % key,
                            fontsize=FS_PANEL, ha='center', va='center',
                            transform=ax.transAxes)
                ax.set_xticks([-5, 0, 5])
                ax.set_yticks([-1, 0, 1])
                if r_ == nrw - 1:
                    ax.set_xlabel(r'$\tau$ [ms]', fontsize=FS_LABEL,
                                  labelpad=1)
            fig.text(0.962, 0.94 - (r_ + 0.5) * (0.94 - 0.15) / nrw,
                     r'$u = %g$' % u, fontsize=FS_PANEL,
                     rotation=270, ha='left', va='center')
        fig.text(0.0, 0.545, r'$W/\max_\tau|W|$', va='center',
                 rotation='vertical', fontsize=FS_LABEL)
    plt.close(fig)
    return fig


def fig_W_u0(tau_s, fgrid, us, rows, gammas, titles=LADDER_TITLES):
    """Fig. 4 (W_u0): tau_s (n_tau,) delay grid [s], fgrid (n_f,) frequencies [Hz], us (n_u,) ladder values u, rows (n_ladder*n_u, n_tau) W^{(1)}_{10}(tau) ladder by ladder, gammas (n_ladder*n_u, n_f) its cosine transforms, titles one column title per ladder."""
    FS_LABEL, FS_TICK, FS_PANEL = 12, 10, 12
    us = np.asarray(us)
    with plt.rc_context():
        _style(FS_LABEL, FS_TICK)
        cmap_c = ListedColormap(
            [_darken(c) for c in
             plt.get_cmap('turbo')(np.linspace(0.03, 0.97, 256))])
        norm_c = LogNorm(vmin=us.min(), vmax=us.max())

        ncol = len(titles)
        fig, ax = plt.subplots(2, ncol, figsize=(7.0, 3.6), sharex='row',
                               sharey='row', constrained_layout=True,
                               squeeze=False)
        for a_ in ax.ravel():
            a_.set_box_aspect(4.4 / 6.4)
        n = us.size
        for j, title in enumerate(titles):
            for u, r in zip(us, rows[j * n:(j + 1) * n]):
                y = savgol_filter(r / np.abs(r).max(), 7, 3)
                ax[0, j].plot(np.asarray(tau_s) * 1e3, y, lw=1.5,
                              color=cmap_c(norm_c(u)))
            for u, g in zip(us, gammas[j * n:(j + 1) * n]):
                ax[1, j].plot(fgrid, g / np.abs(g).max(), lw=1.5,
                              color=cmap_c(norm_c(u)))
            ax[0, j].axhline(0, color='0.85', lw=0.6)
            ax[1, j].axhline(0, color='0.85', lw=0.6)
            ax[0, j].set_title(title, fontsize=FS_PANEL, pad=3)
            ax[1, j].set_xlabel(r'$f$ [Hz]', fontsize=FS_LABEL, labelpad=1)
            ax[0, j].set_xlabel(r'$\tau$ [ms]', fontsize=FS_LABEL,
                                labelpad=1)
        ax[0, 0].set_ylabel(r'$W^{(1)}_{10}(\tau)/\max_\tau|W|$',
                            fontsize=FS_LABEL)
        ax[1, 0].set_ylabel(
            r'$\widetilde{W}^{(1)}_{10}(f)/\max_f|\widetilde{W}|$',
            fontsize=FS_LABEL)

        # one colour band per ladder value, edges at the geometric midpoints
        lu = np.log(us)
        cedges = np.concatenate([[1.5 * lu[0] - 0.5 * lu[1]],
                                 0.5 * (lu[1:] + lu[:-1]),
                                 [1.5 * lu[-1] - 0.5 * lu[-2]]])
        cmap = ListedColormap([cmap_c(norm_c(u)) for u in us])
        norm = BoundaryNorm(cedges, us.size)
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                          ax=ax.ravel().tolist(), pad=0.015, aspect=34,
                          ticks=lu)
        cb.ax.set_yticklabels(['%.2g' % u for u in us])
        cb.set_label(r'$u$', fontsize=FS_LABEL, labelpad=10, rotation=0,
                     va='center')
        cb.ax.tick_params(labelsize=FS_TICK, length=0)
        cb.outline.set_linewidth(0.6)
    plt.close(fig)
    return fig


def fig_orf_side(f1, g2, g4_hlhl, g4_hhhl, f2, surf, surfx):
    """Fig. 5 (orf_side): f1 (n1,) frequencies [Hz] of the equal-frequency slices, g2, g4_hlhl, g4_hhhl (n1,) Gamma2_HL(f,f), Gamma4_HLHL(f,f,f,f), Gamma4_HHHL(f,f,f,f) (complex or real), f2 (n2,) log frequency grid [Hz], surf (n2, n2) Re Gamma4_HLHL(fa,fa,fc,fc), surfx (n2, n2) Re Gamma4_HHLL(fa,fc,fc,fa)."""
    with plt.rc_context():
        mpl.rcdefaults()
        plt.rcParams.update({"font.family": "serif",
                             "mathtext.fontset": "cm",
                             "font.size": 10, "savefig.dpi": 300})
        g2, g4_hlhl, g4_hhhl = (np.asarray(g) for g in (g2, g4_hlhl, g4_hhhl))
        fig, (ax1, ax2, ax3) = plt.subplots(
            1, 3, figsize=(7.0, 2.35), layout="constrained")
        ax1.plot(f1, g2.real, color="#16263F", lw=1.8,
                 label=r"$\mathrm{Re}\,\Gamma^{(2)}_{HL}(f,f)$")
        ax1.plot(f1, g4_hlhl.real, color="#D1605E", lw=1.6,
                 label=r"$\mathrm{Re}\,\Gamma^{(4)}_{HLHL}(f,f,f,f)$")
        ax1.plot(f1, g4_hhhl.real, color="#7FA8C9", lw=1.6,
                 label=r"$\mathrm{Re}\,\Gamma^{(4)}_{HHHL}(f,f,f,f)$")
        ax1.axhline(0, color="0.5", lw=0.6)
        ax1.set_box_aspect(1)
        ax1.set_xlabel(r"$f$ [Hz]", fontsize=11)
        ax1.set_xlim(0, 256)
        ax1.grid(True, color="0.88", lw=0.5, alpha=0.6)
        ax1.set_ylim(-0.16, 0.40)

        # inset with a symlog vertical axis
        axi = ax1.inset_axes([0.30, 0.47, 0.67, 0.50])
        for g, c, lw in ((g2, "#16263F", 1.4), (g4_hlhl, "#D1605E", 1.3),
                         (g4_hhhl, "#7FA8C9", 1.3)):
            axi.plot(f1, g.real, color=c, lw=lw)
        axi.axhline(0, color="0.5", lw=0.5)
        axi.set_xlim(50, 256)
        axi.set_yscale("symlog", linthresh=1e-4, linscale=0.6)
        axi.set_ylim(-0.05, 0.05)
        axi.set_yticks([-1e-2, 0.0, 1e-2])
        axi.tick_params(labelsize=7, length=2, pad=1)
        # frequency tick labels inside the inset, on a white box
        axi.tick_params(axis="x", direction="in", pad=-9)
        axi.set_xticks([100, 200])
        axi.set_xticklabels(["100", "200"])
        for t_ in axi.get_xticklabels():
            t_.set_bbox(dict(facecolor="white", alpha=0.85, lw=0, pad=0.6))
        axi.grid(True, color="0.88", lw=0.4, alpha=0.6)
        ind = ax1.indicate_inset_zoom(axi, edgecolor="0.4", lw=0.6)
        for c_ in ind.connectors:
            c_.set_visible(False)
        ax1.legend(frameon=False, fontsize=8, loc="lower center",
                   bbox_to_anchor=(0.5, 1.0), handlelength=1.2,
                   borderaxespad=0.1, labelspacing=0.15)

        vm = max(np.abs(surf).max(), np.abs(surfx).max())
        norm = SymLogNorm(linthresh=1e-3, vmin=-vm, vmax=vm)
        for ax, S, ttl in (
                (ax2, surf,
                 r"$\mathrm{Re}\,\Gamma^{(4)}_{HLHL}(f_a, f_a, f_c, f_c)$"),
                (ax3, surfx,
                 r"$\mathrm{Re}\,\Gamma^{(4)}_{HHLL}(f_a, f_c, f_c, f_a)$")):
            im = ax.pcolormesh(f2, f2, S, cmap=plt.get_cmap("RdBu_r", 14),
                               norm=norm, rasterized=True)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_aspect("equal")
            ax.set_xlabel(r"$f_a$ [Hz]", fontsize=11)
            ax.set_ylabel(r"$f_c$ [Hz]", fontsize=11)
            ax.text(0.04, 0.96, ttl, transform=ax.transAxes, fontsize=8,
                    va="top", ha="left",
                    bbox=dict(facecolor="white", alpha=0.85, lw=0, pad=1.5))
        cb = fig.colorbar(im, ax=ax3, shrink=0.85)
        cb.set_label(r"$\mathrm{Re}\,\Gamma^{(4)}$", fontsize=11)
        cb.ax.tick_params(labelsize=8.5)
        for a in (ax1, ax2, ax3):
            a.tick_params(labelsize=9.5)
    plt.close(fig)
    return fig


def fig_W_nside(tau_s, nsides, rows, u=20.0):
    """Fig. 15 (W_nside): tau_s (n_tau,) delay grid [s], nsides (3,) HEALPix resolutions, rows (3, n_tau) W^{(1)}_{10}(tau) built at each nside, u the template power of the tables."""
    FS_LABEL, FS_TICK, FS_LEGEND = 11.5, 10, 10
    with plt.rc_context():
        _style(FS_LABEL, FS_TICK, {'legend.fontsize': FS_LEGEND})
        tms = np.asarray(tau_s) * 1e3
        cols = ['#C9807A', figstyle.NAVY, '0.55']
        fig, ax = plt.subplots(figsize=(3.15, 2.3), constrained_layout=True)
        for ns, r, col in zip(nsides, rows, cols):
            ax.plot(tms, 1e6 * np.asarray(r), lw=1.6, color=col,
                    label=r'$N_{\rm side} = %d$' % ns)
        ax.axhline(0, color='0.85', lw=0.6)
        ax.set_xlabel(r'$\tau$ [ms]', fontsize=FS_LABEL, labelpad=1)
        ax.set_ylabel(r'$W^{(1)}_{10}(\tau;\, u = %g) \times 10^{6}$' % u,
                      fontsize=FS_LABEL, labelpad=2)
        ax.legend(fontsize=FS_LEGEND, frameon=False, ncol=3,
                  loc='lower center', bbox_to_anchor=(0.5, 1.0),
                  columnspacing=0.8, handlelength=1.2, handletextpad=0.4,
                  borderaxespad=0.2)
    plt.close(fig)
    return fig


def _label_spots(cs, lo, hi, bank, y0=-0.2, diag=0.4, edge=0.25,
                 short=0.3, room=0.19):
    """Inline label positions for the contour levels, clear of the diagonal, the bank segment and the panel edges; returns (spots, beside) with beside = [(x, y, level index, rotation)]."""
    spots, beside = [], []
    for i, segs in enumerate(cs.allsegs):
        segs = [sg for sg in segs if len(sg) > 1]
        if not segs or sum(np.hypot(*np.diff(sg, axis=0).T).sum()
                           for sg in segs) < short:
            continue
        pts = np.concatenate(segs)
        x_, y_ = pts[:, 0], pts[:, 1]
        clear = np.minimum.reduce([x_ - lo, hi - x_, y_ - lo, hi - y_])
        below = (x_ - y_ > diag) & (clear > edge)
        k = np.argmin(np.abs(x_ - y_))
        if below.any():
            spots.append(tuple(pts[np.flatnonzero(below)[
                np.argmin(np.abs(y_[below] - y0))]]))
        elif x_[k] < bank[0] - 0.1 and clear[k] >= room:
            spots.append((x_[k], y_[k]))
        elif x_[k] < bank[0] - 0.1:
            j = np.argmin(y_)
            beside.append((x_[j] + 0.05, lo + 0.13, i, 0))
        elif x_[k] > bank[1] + 0.1 and clear[k] >= room:
            spots.append((x_[k], y_[k]))
        else:
            leg = (x_ - y_ > diag)
            j = np.flatnonzero(leg)[np.argmin(np.abs(y_[leg] - (y0 + 0.7)))]
            beside.append((x_[j] - 0.05, y_[j], i, 90))
    return spots, beside


def fig_eps_u1u2(u_grid, u_bank, eps):
    """Fig. 14 (eps_u1u2_unified): u_grid (n_u,) template powers on both axes, u_bank (n_bank,) template powers of the bank (red segment on the diagonal), eps dict {n_max: (n_u, n_u) deficit eps[i_u1, i_u2]}, one panel per n_max in key order."""
    FS_LABEL, FS_TICK, FS_CLABEL, FS_PANEL = 12.5, 10.5, 10, 12.5
    with plt.rc_context():
        _style(FS_LABEL, FS_TICK)
        lu = np.log10(u_grid)
        levels = [-4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0]
        fmt = {lv: r"$10^{%g}$" % lv for lv in levels}
        n_list = list(eps)
        fig, axes = plt.subplots(1, len(n_list), figsize=(5.95, 2.3),
                                 sharex=True, sharey=True, squeeze=False)
        axes = axes[0]
        lb = np.log10([np.min(u_bank), np.max(u_bank)])
        for ax, N in zip(axes, n_list):
            cs = ax.contour(lu, lu, np.log10(eps[N]).T, levels=levels,
                            colors=figstyle.NAVY, linewidths=1.5,
                            linestyles="solid")
            ax.plot(lu, lu, color="0.55", lw=1.2, ls="--")
            ax.plot(lb, lb, color="#C8433F", lw=3.0, solid_capstyle="butt")
            spots, beside = _label_spots(cs, lu[0], lu[-1], lb)
            ax.clabel(cs, fmt=fmt, fontsize=FS_CLABEL, inline=True,
                      inline_spacing=2, manual=spots)
            for t in cs.labelTexts:
                t.set_bbox(dict(fc="white", ec="none", pad=0.0))
                t.set_zorder(5)
            for xb, yb, i, rot in beside:
                ax.text(xb, yb, fmt[cs.levels[i]], fontsize=FS_CLABEL,
                        color=figstyle.NAVY, zorder=5, rotation=rot,
                        rotation_mode="anchor",
                        ha="right" if rot == 90 else "left", va="center",
                        bbox=dict(fc="white", ec="none", pad=0.0))
            ax.set_aspect("equal")
            ax.set_xlabel(r"$\log_{10} u_1$", labelpad=1)
            ax.set_title(r"$n_{\max} = %d$" % N, fontsize=FS_PANEL, pad=3)
        axes[0].set_ylabel(r"$\log_{10} u_2$", labelpad=2)
        fig.tight_layout(pad=0.2, w_pad=0.6)
    plt.close(fig)
    return fig
