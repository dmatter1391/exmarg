"""Figures of the simulated year (Figs. 6-13 and 22-28 of the paper) from the files in data/year."""
import numpy as np
import matplotlib
import matplotlib.colors as mc
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, LogLocator
from scipy.ndimage import gaussian_filter
from scipy.special import erf

PAPER_STYLE = {"font.family": "serif", "mathtext.fontset": "cm", "font.size": 15, "axes.labelsize": 20, "xtick.labelsize": 16,
               "ytick.labelsize": 16, "legend.fontsize": 15, "lines.linewidth": 3.0}
COL = {20: "#CC79A7", 40: "#009E73", 80: "#E69F00", 120: "#0072B2"}
CG = "0.3"
LAB = [r"$R_0$", r"$\alpha_1$", r"$\alpha_2$", r"$\mu_2$", r"$\gamma$", r"$\kappa$", r"$z_p$"]
DASH = (0, (3, 1.5))
ZG = np.linspace(0.0, 5.0, 200)
MG = np.geomspace(5.0, 100.0, 240)
TITLE = {"no_threshold": "No Threshold", "uninformed_th12": "Uninformed\n" + r"$\rho_{\rm th} = 12$",
         "uninformed_th8": "Uninformed\n" + r"$\rho_{\rm th} = 8$", "informed_th12": "Catalog Informed\n" + r"$\rho_{\rm th} = 12$",
         "informed_th8": "Catalog Informed\n" + r"$\rho_{\rm th} = 8$"}

# axis ranges of the corner figures of the paper (0.1 and 99.9 percentiles of the drawn posteriors)
CORNER_RANGES = {
    "searches": [(7.2913, 93.8980), (-1.9727, 2.4014), (3.9366, 8.9166), (20.0450, 46.5170), (-1.8275, 6.0914), (-1.9909, 9.9813), (0.5050, 3.9984)],
    "informed_th8": [(8.2382, 25.5315), (0.6104, 2.4089), (3.9361, 7.2332), (26.0115, 36.6190), (0.5774, 5.8028), (-1.8095, 9.9896), (0.5084, 3.9921)],
    "uninformed_th8": [(1.0004, 91.2137), (-1.9603, 3.7820), (3.6200, 9.9800), (20.0450, 45.8130), (-1.8357, 8.7148), (-1.9909, 9.9988), (0.5087, 3.9974)],
    "informed_th12": [(2.8786, 28.8861), (0.6412, 3.0433), (3.8334, 8.1741), (21.6769, 38.8865), (0.9408, 6.4028), (-1.8566, 9.9984), (0.5447, 3.9717)],
    "uninformed_th12": [(1.0020, 22.8390), (1.0798, 3.7023), (3.3462, 8.3450), (20.3846, 34.3746), (2.1169, 8.1887), (3.0295, 9.9995), (0.6534, 2.7893)],
    "no_threshold": [(1.0122, 19.5481), (1.3442, 3.6474), (3.5466, 8.2194), (20.1510, 34.2980), (2.6654, 8.5912), (3.6012, 9.9997), (0.5058, 2.2380)],
    "years": [(2.0279, 86.9987), (-1.6604, 2.4866), (3.7043, 9.9899), (20.0189, 49.9118), (-1.3326, 9.1583), (-1.9693, 9.9867), (0.5021, 3.9940)],
    "null": [(1.0004, 11.4896), (-1.9431, 5.9945), (1.0156, 9.9922), (20.0230, 49.9449), (-1.9989, 3.6972), (-1.8736, 9.9919), (0.5013, 3.9929)],
}

GWTC5_MEDIAN = {"alpha_1": 1.4816146797419463, "alpha_2": 5.418655131964411, "break_mass": 37.45099814539542,
                "delta_m_1": 3.5301760691945265, "lam_0": 0.4003832959761328, "lam_1": 0.5456918152967174,
                "mlow_1": 4.485622822005297, "mmax": 300.0, "mpp_1": 9.910922088965902, "mpp_2": 32.327257249648866,
                "sigpp_1": 0.7841400525546414, "sigpp_2": 5.726331669112833}
M1_GRID = np.linspace(2.0, 300.0, 3000)


def _style(rc, base=True):
    """Reset matplotlib to its defaults, then apply the paper style (base) and the figure's own settings rc."""
    matplotlib.rcdefaults()
    if base:
        plt.rcParams.update(PAPER_STYLE)
    plt.rcParams.update(rc)


# GWTC-5 mass model of the source-frame primary mass (broken power law and two Gaussian peaks, low-mass taper)
def planck_taper(m, m_low, delta):
    """Smooth turn-on S(m) over (m_low, m_low + delta]."""
    m = np.asarray(m, dtype=float)
    mid = (m >= m_low) & (m < m_low + delta)
    mp = m - m_low
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        f = np.exp(delta / mp + delta / (mp - delta))
    return np.where(m < m_low, 0.0, np.where(mid, 1.0 / (1.0 + f), 1.0))


def p_broken_powerlaw(m1, alpha_1, alpha_2, m_break, m1_low, m_high):
    """Broken power law with the analytic normalization."""
    m1 = np.asarray(m1, dtype=float)
    alpha_1 = np.where(alpha_1 == 1.0, alpha_1 + 1e-9, alpha_1)
    alpha_2 = np.where(alpha_2 == 1.0, alpha_2 + 1e-9, alpha_2)
    norm = m_break * ((1.0 - (m1_low / m_break) ** (1.0 - alpha_1)) / (1.0 - alpha_1)
                      + ((m_high / m_break) ** (1.0 - alpha_2) - 1.0) / (1.0 - alpha_2))
    lo = (m1 >= m1_low) & (m1 < m_break)
    hi = (m1 >= m_break) & (m1 < m_high)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        out = np.where(lo, (m1 / m_break) ** (-alpha_1), np.where(hi, (m1 / m_break) ** (-alpha_2), 0.0))
    return out / norm


def trunc_normal(m, mu, sigma, low):
    """Normal density truncated below at low."""
    m = np.asarray(m, dtype=float)
    out = np.exp(-0.5 * ((m - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    out = np.where(m < low, 0.0, out)
    return out / (0.5 * (1.0 + erf((mu - low) / (sigma * np.sqrt(2.0)))))


def pi_m1_unnorm(m1, p):
    """Mixture times taper, unnormalized."""
    lam2 = 1.0 - p["lam_0"] - p["lam_1"]
    mix = (p["lam_0"] * p_broken_powerlaw(m1, p["alpha_1"], p["alpha_2"], p["break_mass"], p["mlow_1"], p["mmax"])
           + p["lam_1"] * trunc_normal(m1, p["mpp_1"], p["sigpp_1"], p["mlow_1"])
           + lam2 * trunc_normal(m1, p["mpp_2"], p["sigpp_2"], p["mlow_1"]))
    return mix * planck_taper(m1, p["mlow_1"], p["delta_m_1"])


def pi_m1(m1, p, grid=M1_GRID):
    """Normalized source-frame primary-mass density pi_src(m1 | p)."""
    return pi_m1_unnorm(m1, p) / np.trapezoid(pi_m1_unnorm(grid, p), grid)


def _md(z, g, k, zp):
    """Madau-Dickinson shape of the merger rate, equal to 1 at z = 0."""
    return (1 + z) ** g / (1 + ((1 + z) / (1 + zp)) ** k) * (1 + (1 / (1 + zp)) ** k)


def _rate_band(s):
    """5th and 95th percentiles of R(z) on ZG over the posterior samples s (columns as in posteriors.npz)."""
    s = np.asarray(s, float); s = s[np.random.default_rng(1).choice(s.shape[0], min(6000, s.shape[0]), replace=False)]
    R = s[:, 0][:, None] * _md(ZG[None], s[:, 4][:, None], s[:, 5][:, None], s[:, 6][:, None])
    return np.percentile(R, [5, 95], axis=0)


def _mass_band(s, p0):
    """5th and 95th percentiles of pi_src(m1) on MG over the posterior samples s (alpha_1, alpha_2, mu_2 from s)."""
    s = np.asarray(s, float); s = s[np.random.default_rng(1).choice(s.shape[0], min(3000, s.shape[0]), replace=False)]
    return np.percentile(np.array([pi_m1(MG, {**p0, "alpha_1": v[1], "alpha_2": v[2], "mpp_2": v[3]}) for v in s]), [5, 95], axis=0)


def _region(color, fill=True, alpha=0.3, lw=2.4, ls="-"):
    """Drawing style of one posterior in a corner figure."""
    return dict(color=color, fill=fill, alpha=alpha, edge=True, lw=lw, ls=ls)


def _level(H, frac):
    h = np.sort(H.ravel())[::-1]; c = np.cumsum(h)
    return h[np.searchsorted(c, frac * c[-1])]


def _corner(S, sty, draw, lims=None, legend=(), injected=None, year_xbins=None, year_legend=()):
    """Corner figure: 90 percent regions of the chains S[key] drawn in the order draw with styles sty; legend = [(key, label)]."""
    _style({"font.size": 10.5, "axes.labelsize": 13, "xtick.labelsize": 10, "ytick.labelsize": 10})
    S = {q: np.asarray(v, float) for q, v in S.items()}; k = 7
    if lims is None:
        allp = np.concatenate([v[::4] for v in S.values()]); lims = [tuple(np.percentile(allp[:, i], [0.1, 99.9])) for i in range(k)]
    fig, ax = plt.subplots(k, k, figsize=(7.0, 7.0)); hs = {}
    for key in draw:
        s, st = S[key], sty[key]; c = st["color"]
        for i in range(k):
            for j in range(i + 1):
                a = ax[i, j]
                if i == j:
                    h, e = np.histogram(s[:, i], 50, range=lims[i], density=True); x = 0.5 * (e[1:] + e[:-1]); y = gaussian_filter(h, 1.2)
                    if st["fill"]:
                        a.fill_between(x, 0, y, color=c, alpha=st["alpha"], lw=0)
                    a.plot(x, y, color=c, lw=st["lw"], ls=st["ls"])
                else:
                    H, xe, ye = np.histogram2d(s[:, j], s[:, i], 45, range=[lims[j], lims[i]]); H = gaussian_filter(H, 1.4)
                    xc, yc = 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1]); lv = _level(H, 0.9)
                    if st["fill"]:
                        a.contourf(xc, yc, H.T, levels=[lv, H.max() * 1.01], colors=[c], alpha=st["alpha"])
                    a.contour(xc, yc, H.T, levels=[lv], colors=[c], linewidths=[st["lw"]], linestyles=[st["ls"]])
        hs[key] = plt.Rectangle((0, 0), 1, 1, fc=mc.to_rgba(c, st["alpha"]), ec=c, lw=st["lw"]) if st["fill"] \
            else Line2D([], [], color=c, lw=st["lw"], ls=(0, (2.6, 0.9)))
    for i in range(k):
        for j in range(k):
            a = ax[i, j]
            if j > i:
                a.axis("off"); continue
            if injected is not None:
                a.axvline(injected[j], color="0.4", lw=0.8)
            a.set_xlim(*lims[j]); a.xaxis.set_major_locator(plt.MaxNLocator(4, prune="upper")); a.yaxis.set_major_locator(plt.MaxNLocator(3))
            if i == j:
                a.set_yticks([]); a.set_ylim(0, None)
            else:
                if injected is not None:
                    a.axhline(injected[i], color="0.4", lw=0.8)
                a.set_ylim(*lims[i])
            if year_xbins:
                a.xaxis.set_major_locator(plt.MaxNLocator(year_xbins, prune="upper" if year_xbins == 4 else None)); w = lims[j][1] - lims[j][0]
                a.set_xticks([t for t in a.get_xticks() if lims[j][0] + 0.08 * w <= t <= lims[j][1] - 0.14 * w])
            if i < k - 1:
                a.set_xticklabels([])
            else:
                a.set_xlabel(LAB[j])
            if j > 0 or i == 0:
                a.set_yticklabels([])
            else:
                a.set_ylabel(LAB[i])
    if year_legend:
        ax[0, k - 1].legend([Line2D([], [], color=cl, lw=1.6) for cl, _ in year_legend], [lb for _, lb in year_legend], frameon=False,
                            loc="upper right", fontsize=11.5, handlelength=2.0, borderaxespad=0.0)
    if legend:
        ax[0, k - 1].legend([hs[q] for q, _ in legend] + [Line2D([], [], color="0.4", lw=0.8)], [lb for _, lb in legend] + ["Injected"],
                            frameon=False, loc="upper right", fontsize=11.5, handlelength=2.6, borderaxespad=0.0)
    fig.subplots_adjust(left=0.105, right=0.995, bottom=0.075, top=0.995, wspace=0.06, hspace=0.06)
    plt.close(fig)
    return fig


def fig_corner_searches(P):
    """Fig. 6: catalog alone, uninformed and catalog-informed searches at rho_th = 8, n_max = 120; P = posteriors.npz."""
    S = {"ref": P["catalog_alone_th8"], "U": P["uninformed_th8_n120"], 120: P["informed_th8_n120"]}
    sty = {"ref": _region(CG), "U": _region("#D55E00"), 120: _region(COL[120])}
    return _corner(S, sty, ["U", "ref", 120], CORNER_RANGES["searches"], legend=[("ref", "Catalog Only"), ("U", "Uninformed"), (120, "Catalog Informed")],
                   injected=P["injected"])


def fig_corner_orders(P, search):
    """Figs. 22-24: one search at n_max = 20, 40, 80, 120; P = posteriors.npz, search = informed_th8, uninformed_th8, informed_th12, uninformed_th12 or no_threshold."""
    S = {n: np.asarray(P["%s_n%d" % (search, n)], float) for n in (20, 40, 80, 120)}
    sty = {n: _region(COL[n]) for n in S}
    same = np.all(np.abs(np.median(S[80], 0) - np.median(S[120], 0)) < 0.15 * S[120].std(0))
    if same:
        sty[80] = _region(COL[80], fill=False, ls=(0, (5, 5)))
    draw = [20, 40] + ([120, 80] if same else [80, 120]); legend = [(n, r"$n_{\max} = %d$" % n) for n in (20, 40, 80, 120)]
    if search.startswith("informed"):
        S["ref"] = P["catalog_alone_" + search.split("_")[1]]; sty["ref"] = _region(CG); draw = ["ref"] + draw; legend = [("ref", "Catalog Only")] + legend
    return _corner(S, sty, draw, CORNER_RANGES[search], legend=legend, injected=P["injected"])


def _corner_years(P, sets, fill, injected, lims, xbins):
    """Corner figure of one or more searches in ten simulated years; sets = [(key prefix, colour, legend label or None)]."""
    S, sty = {}, {}
    for g, (pre, col, _) in enumerate(sets):
        for q in range(10):
            S[(g, q)] = P["%s%d" % (pre, q)]; sty[(g, q)] = _region(col, fill=fill, alpha=0.1, lw=1.2 if fill else 1.0)
    return _corner(S, sty, list(S), lims, injected=injected, year_xbins=xbins, year_legend=[(col, lb) for _, col, lb in sets if lb])


def fig_corner_years(P):
    """Fig. 7: the uninformed search at rho_th = 8, n_max = 120 in ten simulated years; P = posteriors.npz."""
    return _corner_years(P, [("uninformed_th8_year", "#D55E00", None)], True, P["injected"], CORNER_RANGES["years"], 4)


def fig_corner_null(P):
    """Fig. 28: no-threshold and uninformed searches in ten years of noise only; P = posteriors.npz."""
    sets = [("noise_no_threshold_year", "#0072B2", "No Threshold"), ("noise_uninformed_year", "0.15", "Uninformed")]
    return _corner_years(P, sets, False, None, CORNER_RANGES["null"], 3)


def fig_population_searches(P):
    """Fig. 8: bands of pi_src(m1) and R(z) of the catalog alone and the two searches at rho_th = 8, n_max = 120; P = posteriors.npz."""
    _style({"font.size": 9.5, "axes.labelsize": 11.5, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 10})
    p0 = dict(GWTC5_MEDIAN, mmax=200.0); tr = P["injected"]
    fig, (am, a) = plt.subplots(1, 2, figsize=(7.0, 2.9))
    cat = (_rate_band(P["catalog_alone_th8"]), _mass_band(P["catalog_alone_th8"], p0))
    hs = [plt.Rectangle((0, 0), 1, 1, fc=mc.to_rgba(CG, 0.3), ec=CG, lw=1.9)]; ls = ["Catalog Only"]
    for key, col, lab in (("uninformed_th8_n120", "#D55E00", "Uninformed"), ("informed_th8_n120", "#0072B2", "Catalog Informed")):
        if key.startswith("informed"):
            for ax_, x_, (lo, hi) in ((a, ZG, cat[0]), (am, MG, cat[1])):
                ax_.fill_between(x_, lo, hi, color=mc.to_rgba(CG, 0.3), lw=0); ax_.plot(x_, lo, color=CG, lw=1.9); ax_.plot(x_, hi, color=CG, lw=1.9)
        for ax_, x_, (lo, hi) in ((a, ZG, _rate_band(P[key])), (am, MG, _mass_band(P[key], p0))):
            ax_.fill_between(x_, lo, hi, color=mc.to_rgba(col, 0.3), lw=0); ax_.plot(x_, lo, color=col, lw=1.9); ax_.plot(x_, hi, color=col, lw=1.9)
        hs.append(plt.Rectangle((0, 0), 1, 1, fc=mc.to_rgba(col, 0.3), ec=col, lw=1.9)); ls.append(lab)
    a.plot(ZG, tr[0] * _md(ZG, tr[4], tr[5], tr[6]), color="k", lw=1.3, ls=":")
    am.plot(MG, pi_m1(MG, {**p0, "alpha_1": tr[1], "alpha_2": tr[2], "mpp_2": tr[3]}), color="k", lw=1.3, ls=":")
    hs.append(Line2D([], [], color="k", lw=1.3, ls=":")); ls.append("Injected")
    a.set_yscale("log"); a.set_xlim(0, 5); a.set_ylim(1.0, 1000.0); a.grid(color="0.87", lw=0.5); a.set_xlabel(r"$z$"); a.set_ylabel(r"$R(z)\ [{\rm Gpc^{-3}\,yr^{-1}}]$")
    am.set_xscale("log"); am.set_yscale("log"); am.set_xlim(5, 100); am.set_ylim(1e-5, 1.0); am.grid(color="0.87", lw=0.5)
    am.set_xticks([5, 10, 20, 50, 100]); am.set_xticklabels(["5", "10", "20", "50", "100"]); am.minorticks_off()
    am.set_xlabel(r"$m_1\ [M_\odot]$"); am.set_ylabel(r"$\pi_{\rm src}(m_1)\ [M_\odot^{-1}]$")
    fig.legend(hs, ls, frameon=False, loc="upper center", ncol=4, handlelength=1.7, columnspacing=1.6, fontsize=10)
    fig.tight_layout(pad=0.3, rect=[0, 0, 1, 0.9], w_pad=1.2)
    plt.close(fig)
    return fig


def fig_rate_versions(P):
    """Fig. 27: R(z) bands at n_max = 40, 80, 120 of the five searches, one panel each; P = posteriors.npz."""
    _style({"font.size": 9.5, "axes.labelsize": 11.5, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 10})
    orders = (40, 80, 120); tr = P["injected"]
    fig, ax = plt.subplots(1, 5, figsize=(7.0, 3.0), sharey=True)
    for a, search in zip(ax, ("no_threshold", "uninformed_th12", "uninformed_th8", "informed_th12", "informed_th8")):
        if search.startswith("informed"):
            lo, hi = _rate_band(P["catalog_alone_" + search.split("_")[1]])
            a.fill_between(ZG, lo, hi, color=mc.to_rgba(CG, 0.3), lw=0); a.plot(ZG, lo, color=CG, lw=1.8); a.plot(ZG, hi, color=CG, lw=1.8)
        B = {n: _rate_band(P["%s_n%d" % (search, n)]) for n in orders}
        same = search in ("uninformed_th12", "uninformed_th8", "informed_th8")
        for n in (40, 120, 80):
            lo, hi = B[n]
            if n == 80 and same:
                a.plot(ZG, lo, color=COL[n], lw=1.8, ls=(0, (4, 4))); a.plot(ZG, hi, color=COL[n], lw=1.8, ls=(0, (4, 4)))
            else:
                a.fill_between(ZG, lo, hi, color=mc.to_rgba(COL[n], 0.3), lw=0); a.plot(ZG, lo, color=COL[n], lw=1.8); a.plot(ZG, hi, color=COL[n], lw=1.8)
        a.plot(ZG, tr[0] * _md(ZG, tr[4], tr[5], tr[6]), color="k", lw=1.2, ls=":")
        a.set_yscale("log"); a.set_xlim(0, 5); a.set_ylim(1.0, 1000.0); a.set_xticks([0, 1, 2, 3, 4] + ([5] if a is ax[-1] else []))
        a.grid(color="0.87", lw=0.5); a.set_title(TITLE[search], fontsize=9.5, pad=4)
    ax[0].set_ylabel(r"$R(z)\ [{\rm Gpc^{-3}\,yr^{-1}}]$"); fig.supxlabel(r"$z$", fontsize=11.5, y=0.215)
    hs = [plt.Rectangle((0, 0), 1, 1, fc=mc.to_rgba(COL[n], 0.3), ec=COL[n], lw=1.8) if n != 80 else Line2D([], [], color=COL[n], lw=1.8, ls=(0, (2.6, 0.9)))
          for n in orders]
    hs += [plt.Rectangle((0, 0), 1, 1, fc=mc.to_rgba(CG, 0.3), ec=CG, lw=1.8), Line2D([], [], color="k", lw=1.2, ls=":")]
    fig.legend(hs, [r"$n_{\max} = %d$" % n for n in orders] + ["Catalog Only", "Injected"], frameon=False, loc="lower center", ncol=3,
               handlelength=1.7, columnspacing=1.6, bbox_to_anchor=(0.54, -0.02), fontsize=9.5, labelspacing=0.35)
    fig.subplots_adjust(left=0.095, right=0.985, top=0.87, bottom=0.34, wspace=0.07)
    plt.close(fig)
    return fig


def _tick(v):
    e = int(np.floor(np.log10(abs(v)) + 1e-9)); m = int(round(abs(v) / 10.0 ** e)); sg = "-" if v < 0 else ""
    return r"$%s10^{%d}$" % (sg, e) if m == 1 else r"$%s%d\times10^{%d}$" % (sg, m, e)


def fig_segment_information(I):
    """Fig. 9: s_i^(j) s_i^(k) / sqrt(F^jj F^kk) of every unresolved merger at its (m1_src, z), with the z profile of the diagonal; I = information.npz."""
    order = [1, 2, 3, 4, 5, 6, 0]
    sym = [r"\alpha_1", r"\alpha_2", r"\mu_2", r"\gamma", r"\kappa", r"z_{\rm p}", r"R_0"]
    G = np.asarray(I["s_unresolved"], float)[:, order]; FJJ = np.asarray(I["F_diagonal"], float)[order]
    PX = np.asarray(I["m1_src_unresolved"], float); PY = np.log1p(np.asarray(I["z_unresolved"], float))
    THR, VTOP, VLEG, MINSZ, ZMAX, BGSIZE = 1e-5, 1e-2, 1e-1, 0.9, 10.0, 0.7
    x0, x1 = -8.0, 125.0
    POS = ["#DCE6F0", "#A9C4DB", "#7FA8C9", "#3D6A99", "#16263F"]
    NEG = ["#F6DCDA", "#E8A9A6", "#D1605E", "#B0443E", "#6E1F1B"]
    CPOS, CNEG = LinearSegmentedColormap.from_list("pos", POS[1:]), LinearSegmentedColormap.from_list("neg", NEG[1:])
    SZ = lambda v: np.maximum(200.0 * (np.abs(v) / VLEG) ** 0.7, MINSZ)
    shade = lambda v: np.clip(np.log10(np.abs(v) / THR) / np.log10(VTOP / THR), 0.0, 1.0)
    FT, FP, FA = 13, 14, 20
    _style({"font.family": "serif", "mathtext.fontset": "cm", "font.size": FT, "axes.labelsize": FA, "xtick.labelsize": FT, "ytick.labelsize": FT,
            "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
            "axes.linewidth": 1.3, "xtick.major.width": 1.1, "ytick.major.width": 1.1, "xtick.minor.width": 0.8, "ytick.minor.width": 0.8,
            "xtick.major.size": 6.5, "ytick.major.size": 6.5, "xtick.minor.size": 3.5, "ytick.minor.size": 3.5}, base=False)
    n = 7; W = 7.0; left, bot, gap, top = 0.50, 0.66, 0.0, 0.03; STRIP = 0.35
    pw = (W - left - (0.03 + 0.36 * 0.95) - (n - 1) * gap) / n; ph = 1.2 * pw; H = bot + n * ph + (n - 1) * gap + top
    XT, YT = [10, 50, 100], [1, 2, 4, 8]
    fig = plt.figure(figsize=(W, H))
    for r in range(n):
        for c in range(r + 1):
            a = fig.add_axes([(left + c * (pw + gap)) / W, (bot + (n - 1 - r) * (ph + gap)) / H, pw / W, ph / H])
            v = G[:, r] * G[:, c] / np.sqrt(FJJ[r] * FJJ[c])
            a.set_xlim(x0, x1); a.set_ylim(0.0, np.log1p(ZMAX))
            lab = (r"$%s$" % sym[r]) if r == c else (r"$%s,\ %s$" % (sym[r], sym[c]))
            a.text(0.95, 0.94, lab, transform=a.transAxes, ha="right", va="top", fontsize=FP, zorder=5, bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.6))
            lo = np.abs(v) < THR
            a.scatter(PX[lo], PY[lo], s=BGSIZE, c="#A6A6A6", alpha=0.08, linewidths=0, rasterized=True, zorder=1)
            sz = SZ(v); o = np.nonzero(~lo)[0]; o = o[np.argsort(-np.abs(v[o]))]
            a.scatter(PX[o], PY[o], s=sz[o], c=np.where((v[o] > 0)[:, None], CPOS(shade(v[o])), CNEG(shade(v[o]))), alpha=0.9,
                      linewidths=np.where(sz[o] >= 2.0, 0.25, 0.0), edgecolors="white", rasterized=True, zorder=2)
            a.set_xticks(XT); a.set_xticks([20, 30, 40, 60, 70, 80, 90, 110, 120], minor=True)
            a.set_yticks(np.log1p(np.array(YT, float))); a.set_yticks(np.log1p(np.array([3, 5, 6, 7, 9], float)), minor=True)
            a.tick_params(axis="y", which="both", right=False)
            a.set_xticklabels([str(t) for t in XT] if r == n - 1 else [], fontsize=FT - 2); a.set_yticklabels([str(t) for t in YT] if c == 0 else [])
            if r == c:
                yb = np.linspace(*a.get_ylim(), 241); yc = 0.5 * (yb[1:] + yb[:-1])
                hst = np.histogram(PY, yb, weights=v)[0]; kw = np.exp(-0.5 * (np.arange(-12, 13) * (yb[1] - yb[0]) / 0.04) ** 2); dn = np.convolve(hst, kw / kw.sum(), "same")
                sx = fig.add_axes([(left + (c + 1) * (pw + gap)) / W, (bot + (n - 1 - r) * (ph + gap)) / H, STRIP * pw / W, ph / H]); sx.set_ylim(a.get_ylim()); sx.set_xlim(0, 1.08)
                sx.fill_betweenx(yc, 0, dn / dn.max(), color="black", alpha=0.35, linewidth=0); sx.plot(dn / dn.max(), yc, color="black", lw=1.4)
                sx.set_xticks([]); sx.set_yticks([])
                for sp in ("top", "right", "bottom"):
                    sx.spines[sp].set_visible(False)
    fig.text((left + 0.5 * (n * pw + (n - 1) * gap)) / W, 0.02 / H, r"$m_1^{\rm src}\,[M_\odot]$", ha="center", va="bottom", fontsize=FA)
    fig.text(0.04 / W, (bot + 0.5 * (n * ph + (n - 1) * gap)) / H, r"$z$", ha="left", va="center", fontsize=FA)
    gp = 0.45; L = np.log10(VLEG / THR); wl, hl = 3.9, 0.6
    lax = fig.add_axes([(left + 2.75 * (pw + gap)) / W, (bot + 5.55 * (ph + gap)) / H, wl / W, hl / H])
    lax.set_xlim(-(gp + L) * 1.03, (gp + L) * 1.03); lax.set_ylim(-36.0 * hl, 36.0 * hl)
    xs = np.linspace(0.0, L, 400); vv = THR * 10.0 ** xs; rr = np.sqrt(SZ(vv) / np.pi)
    for sg, cm in ((1.0, CPOS), (-1.0, CNEG)):
        q = [[(sg * (gp + xs[k]), -rr[k]), (sg * (gp + xs[k]), rr[k]), (sg * (gp + xs[k + 1]), rr[k + 1]), (sg * (gp + xs[k + 1]), -rr[k + 1])] for k in range(xs.size - 1)]
        col = cm(shade(0.5 * (vv[1:] + vv[:-1]))); lax.add_collection(PolyCollection(q, facecolors=col, edgecolors=col, linewidths=0.3))
    lax.scatter([0.0], [0.0], s=BGSIZE * 6, c="#A6A6A6", linewidths=0); tv = [1e-4, 1e-3, 1e-2, 1e-1]
    for sg, cm in ((1.0, CPOS), (-1.0, CNEG)):
        lax.scatter([sg * (gp + np.log10(t / THR)) for t in tv], [0.0] * len(tv), s=SZ(np.array(tv)), c=cm(shade(np.array(tv))), linewidths=0.5,
                    edgecolors="white", zorder=3, clip_on=False)
    lax.set_xticks([-(gp + np.log10(t / THR)) for t in tv[::-1]] + [0.0] + [gp + np.log10(t / THR) for t in tv])
    lax.set_xticklabels([_tick(-t) for t in tv[::-1]] + [r"$<%s$" % _tick(THR).strip("$")] + [_tick(t) for t in tv], fontsize=10)
    lax.set_yticks([]); lax.xaxis.set_ticks_position("bottom"); lax.tick_params(axis="x", direction="out", length=3, top=False)
    for sp in ("left", "right", "top"):
        lax.spines[sp].set_visible(False)
    lax.set_xlabel(r"$s_i^{(j)}\,s_i^{(k)}\,/\sqrt{F^{jj}F^{kk}}$", fontsize=FA, labelpad=4)
    plt.close(fig)
    return fig


def fig_segment_bf_hist(B, orders=(120,), phase_randomized=False):
    """Figs. 10, 13, 25: histograms of the per-segment ln B_i; B = segment_lnB.npz, orders (1, 10, 20, 40) for Fig. 25, phase_randomized=True for Fig. 13."""
    _style({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 11, "axes.labelsize": 13, "xtick.labelsize": 11, "ytick.labelsize": 11,
            "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True, "axes.linewidth": 1.2}, base=False)
    edges = np.r_[np.arange(-0.3, 1.0, 0.05), np.geomspace(1.0, 1500.0, 70)]; one = len(orders) == 1
    fig, axs = plt.subplots(1, len(orders), figsize=(3.4, 2.7) if one else (7.0, 2.6), sharey=True, squeeze=False); axs = axs[0]
    for a, n in zip(axs, orders):
        groups = ((B["noise_only_n%d" % n], "0.55", "Noise Only"), (B["unresolved_mergers_n%d" % n], "#2C7BB6", "Unresolved Mergers"),
                  (B["detected_mergers"], "#7B3294", "Detected Mergers"))
        if phase_randomized:
            groups = groups[:2] + ((B["phase_randomized_mergers_n%d" % n], "#C0392B", "Phase-Randomized Mergers"),)
        for v, col, lab in groups:
            h, _ = np.histogram(np.asarray(v, float), edges)
            a.stairs(h, edges, fill=True, color=col, alpha=0.45, linewidth=0)
            a.stairs(h, edges, color=col, linewidth=1.3, label=lab)
        a.set_xscale("symlog", linthresh=1.0, linscale=0.5); a.set_yscale("log"); XR = 20.0 if phase_randomized else 1500.0
        a.set_xlim(-0.3, XR); a.set_ylim(0.7, 3e6); xt = [t for t in ((0, 1, 10, 100, 1000) if one else (0, 1, 10, 1000)) if t <= XR]
        a.set_xticks(xt); a.set_xticklabels([str(t) if t < 1000 or one else r"$10^3$" for t in xt]); a.set_xlabel(r"$\ln \mathcal{B}_i$")
        mx = [0.2, 0.4, 0.6, 0.8] + [k * 10 ** e for e in range(0, 3) for k in range(2, 10)]
        a.xaxis.set_minor_locator(FixedLocator([t for t in mx if t <= XR]))
        a.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=100))
        a.tick_params(which="major", length=6, width=1.0); a.tick_params(which="minor", length=3.5, width=0.8)
        if not one:
            a.set_title(r"$n_{\max} = %d$" % n, fontsize=12, pad=4)
    axs[0].set_ylabel("Number of segments")
    if one:
        lg = axs[-1].legend(loc="upper right", frameon=False, fontsize=9.5, handlelength=1.6); fig.tight_layout(pad=0.3)
    else:
        lg = fig.legend(*axs[0].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False, fontsize=10.5, handlelength=1.6, columnspacing=1.2)
        fig.subplots_adjust(left=0.085, right=0.99, bottom=0.17, top=0.80, wspace=0.06)
    for hl in lg.get_lines():
        hl.set_linewidth(3.0)
    plt.close(fig)
    return fig


def _time_axes(a, last, linscale):
    """Axes of ln B against observing time."""
    a.axhline(10.0, color="0.3", lw=0.7, ls=":")
    a.set_xscale("log"); a.set_yscale("symlog", linthresh=10.0, linscale=linscale); a.set_xlim(1.0 / 24, 365.25); a.set_ylim(-3, 1e5)
    a.set_xticks([1.0 / 24, 1, 30, 365.25]); a.set_xticklabels(["1 h", "1 d", "1 mo", "1 yr" if last else ""]); a.minorticks_off()
    a.set_yticks([0, 10, 1e2, 1e3, 1e4, 1e5]); a.set_yticklabels(["0", "10", r"$10^2$", r"$10^3$", r"$10^4$", r"$10^5$"])


def fig_time_thresholds(T):
    """Fig. 11: ln B against observing time of the two searches at rho_th = 8, 10, 12, n_max = 120; T = time_curves.npz."""
    _style({"font.size": 9.5, "axes.labelsize": 11.5, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 10})
    day = lambda t: t[:, 0] * float(T["segment_duration"]) / 86400.0; TC = {8: "#7FA8C9", 10: "#3D6A99", 12: "#16263F"}
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
    for a, title, search in ((ax[0], "Catalog Informed", "informed"), (ax[1], "Uninformed", "uninformed")):
        for t in (8, 10, 12):
            c = T["%s_th%d_n120" % (search, t)]; a.plot(day(c), c[:, 1], color=TC[t], lw=2.1, ls="-")
        _time_axes(a, a is ax[1], 0.9); a.grid(color="0.85", lw=0.5)
        a.set_title(title, fontsize=10.5, pad=4); a.set_xlabel("Observing time")
    ax[0].set_ylabel(r"$\ln \mathcal{B}$")
    fig.legend([Line2D([], [], color=TC[t], lw=2.1) for t in (8, 10, 12)], [r"$\rho_{\rm th} = %d$" % t for t in (8, 10, 12)], frameon=False,
               loc="upper center", ncol=3, handlelength=1.9, columnspacing=1.4, handletextpad=0.5, fontsize=10)
    fig.tight_layout(pad=0.3, rect=[0, 0, 1, 0.91], w_pad=0.6)
    plt.close(fig)
    return fig


def fig_time_versions(T):
    """Fig. 26: ln B against observing time at n_max = 20, 40, 80, 120 of four searches, one panel each; T = time_curves.npz."""
    _style({"font.size": 9.5, "axes.labelsize": 11.5, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 10})
    fig, ax = plt.subplots(1, 4, figsize=(7.0, 2.55), sharey=True)
    for a, search in zip(ax, ("no_threshold", "uninformed_th12", "uninformed_th8", "informed_th8")):
        c120 = T["%s_n120" % search][:, 1]
        for n in (20, 40, 80, 120):
            t = T["%s_n%d" % (search, n)]; d = t[:, 0] * float(T["segment_duration"]) / 86400.0
            same = n == 80 and np.allclose(t[-1, 1], c120[-1], rtol=2e-3)
            a.plot(d, t[:, 1], color=COL[n], lw=1.7, ls=(0, (4, 4)) if same else "-", zorder=4 if same else 3)
        _time_axes(a, a is ax[-1], 0.6); a.grid(color="0.85", lw=0.5); a.set_title(TITLE[search], fontsize=9.5, pad=4)
    ax[0].set_ylabel(r"$\ln \mathcal{B}$"); fig.supxlabel("Observing time", fontsize=11.5, y=0.14)
    hs = [Line2D([], [], color=COL[n], lw=1.7, ls=(0, (2.6, 0.9)) if n == 80 else "-") for n in (20, 40, 80, 120)]
    fig.legend(hs, [r"$n_{\max} = %d$" % n for n in (20, 40, 80, 120)], frameon=False, loc="lower center", ncol=4, handlelength=1.8, columnspacing=1.4,
               bbox_to_anchor=(0.53, -0.02))
    fig.subplots_adjust(left=0.075, right=0.975, top=0.86, bottom=0.30, wspace=0.07)
    plt.close(fig)
    return fig


def fig_dkl_control(I):
    """Fig. 12: histograms of D_KL,i of the merger segments at n_max = 20 and 120, simulated and phase-randomized data; I = information.npz."""
    _style({"font.size": 10, "axes.labelsize": 12, "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9.5, "lines.linewidth": 1.8})
    fig, a = plt.subplots(figsize=(3.4, 2.9)); bins = np.linspace(-10, -1, 46); DKLCOL = {20: "#16263F", 120: "#C8433F"}
    for n in (20, 120):
        for v, ls in ((I["dkl_simulated_n%d" % n], "-"), (I["dkl_phase_randomized_n%d" % n], DASH)):
            h, _ = np.histogram(np.log10(np.maximum(np.asarray(v, float), 1e-10)), bins); a.step(bins, np.r_[h, h[-1]], where="post", color=DKLCOL[n], lw=1.8, ls=ls)
    a.set_yscale("log"); a.set_xlabel(r"$\log_{10} D_{{\rm KL},i}$"); a.set_ylabel(r"Number of mergers"); a.grid(color="0.82", lw=0.5, alpha=0.7)
    hs = [Line2D([], [], color="0.6", ls="-"), Line2D([], [], color="0.6", ls=DASH)] + [Line2D([], [], color=DKLCOL[n]) for n in (20, 120)]
    fig.legend(hs, ["Simulated Data", "Phase Randomized", r"$n_{\max} = 20$", r"$n_{\max} = 120$"], ncol=2, loc="upper center", frameon=False,
               handlelength=1.6, columnspacing=0.8, fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 1 - (0.2 * 2 + 0.1) / fig.get_figheight()], pad=0.3)
    plt.close(fig)
    return fig
