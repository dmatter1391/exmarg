"""The population of the paper (Sec. III A), tabulated once into src/exmarg/data/demo_population.npz.

Source-frame primary mass: the default mass model of GWTC-5 at the median hyperparameters, a broken power law plus two
Gaussian peaks times a Planck taper at low mass, on 2 to 200 solar masses.  Redshift: a Madau-Dickinson rate shape
(1+z)^2.7 / [1 + ((1+z)/2.9)^5.6], counted as R(z) / (1+z) dVc/dz (Planck18) on 0 < z <= 10.  Mergers are equal-mass
binaries without spin, M_det = 2 m1 (1 + z).  This is the model that drew the mergers of the simulated year.

From these two densities the script derives
  p(ln y)    the distance prior of the tables and of the analysis, y = D0 / D_L(z) >= ymin = D0 / D_L(z = 10)
  p(ln M)    the marginal of the detector-frame total mass on [20, 200] solar masses
and keeps the mass-ratio and spin densities of the three-parameter set-up (q^1.1 on [0.3, 1], a normal of mean 0.06 and
width 0.12 on [-0.6, 0.6]).

    python scripts/build_population.py
"""
from pathlib import Path

import numpy as np
from scipy.special import erf
from scipy.stats import norm
from astropy.cosmology import Planck18
import astropy.units as u

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'src' / 'exmarg' / 'data' / 'demo_population.npz'
D0 = 10410.0                                   # reference distance of the tables (Mpc)
Z_MAX, M1_LO, M1_HI = 10.0, 2.0, 200.0
M_LO, M_HI = 20.0, 200.0
Q_LO, Q_HI, Q_BETA = 0.3, 1.0, 1.1
CHI_MU, CHI_SD, CHI_LIM = 0.06, 0.12, 0.6
# GWTC-5 default mass model, marginal medians of the hyperposterior
P = dict(alpha_1=1.4816146797419463, alpha_2=5.418655131964411, break_mass=37.45099814539542, mlow_1=4.485622822005297,
         mmax=300.0, delta_m_1=3.5301760691945265, lam_0=0.4003832959761328, lam_1=0.5456918152967174,
         mpp_1=9.910922088965902, sigpp_1=0.7841400525546414, mpp_2=32.327257249648866, sigpp_2=5.726331669112833)


def taper(m, m_low, delta):
    mp = m - m_low
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        f = np.exp(delta / mp + delta / (mp - delta))
    return np.where(m < m_low, 0.0, np.where(m < m_low + delta, 1.0 / (1.0 + f), 1.0))


def broken_power_law(m, a1, a2, mb, mlo, mhi):
    norm_ = mb * ((1 - (mlo / mb) ** (1 - a1)) / (1 - a1) + ((mhi / mb) ** (1 - a2) - 1) / (1 - a2))
    return np.where((m >= mlo) & (m < mb), (m / mb) ** -a1, np.where((m >= mb) & (m < mhi), (m / mb) ** -a2, 0.0)) / norm_


def peak(m, mu, sigma, low):
    out = np.where(m < low, 0.0, np.exp(-0.5 * ((m - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi)))
    return out / (0.5 * (1 + erf((mu - low) / (sigma * np.sqrt(2)))))


def pi_m1(m):
    mix = (P['lam_0'] * broken_power_law(m, P['alpha_1'], P['alpha_2'], P['break_mass'], P['mlow_1'], P['mmax'])
           + P['lam_1'] * peak(m, P['mpp_1'], P['sigpp_1'], P['mlow_1'])
           + (1 - P['lam_0'] - P['lam_1']) * peak(m, P['mpp_2'], P['sigpp_2'], P['mlow_1']))
    return mix * taper(m, P['mlow_1'], P['delta_m_1'])


def main():
    m1 = np.linspace(M1_LO, M1_HI, 4000)
    p_m1 = pi_m1(m1); p_m1 /= np.trapezoid(p_m1, m1)
    z = np.linspace(1e-3, Z_MAX, 4000)
    dvc = (Planck18.differential_comoving_volume(z) * 4 * np.pi * u.sr).to(u.Gpc ** 3).value
    p_z = (1 + z) ** 2.7 / (1 + ((1 + z) / 2.9) ** 5.6) / (1 + z) * dvc; p_z /= np.trapezoid(p_z, z)

    # distance prior in ln y, y = D0 / D_L(z): p(ln y) = p(z) / |d ln y / dz|
    dl = Planck18.luminosity_distance(z).value
    lny_z = np.log(D0 / dl)
    ymin = float(D0 / Planck18.luminosity_distance(Z_MAX).value)
    lny = np.linspace(np.log(ymin), np.log(D0 / 10.0), 2001)
    jac = np.abs(np.gradient(lny_z, z))
    p_lny = np.interp(lny, lny_z[::-1], (p_z / jac)[::-1], left=0.0, right=0.0); p_lny /= np.trapezoid(p_lny, lny)

    # detector-frame total mass M = 2 m1 (1 + z): p(ln M) = int dz p(z) p_m1(m1) m1, truncated to [M_LO, M_HI]
    lnM = np.linspace(np.log(M_LO), np.log(M_HI), 801)
    mm = np.exp(lnM)[:, None] / (2 * (1 + z[None, :]))
    p_lnM = np.trapezoid(p_z[None, :] * np.interp(mm, m1, p_m1, left=0.0, right=0.0) * mm, z, axis=1)
    all_M = np.linspace(np.log(1.0), np.log(5000.0), 4001)
    mm_all = np.exp(all_M)[:, None] / (2 * (1 + z[None, :]))
    p_all = np.trapezoid(p_z[None, :] * np.interp(mm_all, m1, p_m1, left=0.0, right=0.0) * mm_all, z, axis=1)
    inside = np.trapezoid(p_lnM, lnM) / np.trapezoid(p_all, all_M)
    p_lnM /= np.trapezoid(p_lnM, lnM)

    q = np.linspace(Q_LO, Q_HI, 401); p_q = q ** Q_BETA; p_q /= np.trapezoid(p_q, q)
    chi = np.linspace(-CHI_LIM, CHI_LIM, 401); p_chi = norm.pdf(chi, CHI_MU, CHI_SD); p_chi /= np.trapezoid(p_chi, chi)
    np.savez(OUT, m1=m1, p_m1=p_m1, z=z, p_z=p_z, lny=lny, p_lny=p_lny, lnM=lnM, p_lnM=p_lnM, q=q, p_q=p_q, chi=chi,
             p_chi=p_chi, M_range=[M_LO, M_HI], ymin=ymin, D0=D0, z_max=Z_MAX, mass_hyperparameters=str(P),
             fraction_in_M_range=inside)
    print('written %s: ymin = %.4f (D_L = %.0f Mpc at z = %g), %.3f of the mergers have M_det in [%g, %g]'
          % (OUT, ymin, D0 / ymin, Z_MAX, inside, M_LO, M_HI))


if __name__ == '__main__':
    main()
