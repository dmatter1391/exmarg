"""
template profiles a_nu, Psi_0(f_nu) from IMRPhenomXAS.

the (2,2) mode of an aligned spin binary carries the reference phase as
a global factor, so the model at iota = 0, phiRef = 0 gives
a(f) e^{i Psi_0(f)} directly, a = |h+|, Psi_0 = unwrap arg h+.  constant
and linear-in-f pieces of Psi_0 are absorbed by the phi_0 and t
marginalizations, so no further cleaning is needed.  the model contains
the full inspiral, merger and ringdown; no frequency cut is applied
beyond the analysis band.
"""

import warnings

import numpy as np

warnings.filterwarnings('ignore', 'Wswiglal-redir-stdio')
import lal
import lalsimulation as ls

from .detectors import freqs

MPC_SI = 3.08567758149137e22


def profile(m1, m2, s1z, s2z, T, fmin, fmax, D0_mpc):
    '''
    Outputs
    * nu (B,) bin indices, f (B,), a (B,), psi0 (B,)
    '''
    hp, hc = ls.SimInspiralChooseFDWaveform(
        m1 * lal.MSUN_SI, m2 * lal.MSUN_SI,
        0.0, 0.0, s1z, 0.0, 0.0, s2z,
        D0_mpc * MPC_SI, 0.0, 0.0,
        0.0, 0.0, 0.0,
        1.0 / T, fmin, fmax, fmin,
        None, ls.IMRPhenomXAS)
    f = freqs(T, fmin, fmax)
    nu = np.rint(f * T).astype(int)
    hpv = hp.data.data[nu]
    hcv = hc.data.data[nu]
    r = np.abs(hcv[hpv != 0]) / np.abs(hpv[hpv != 0])
    assert np.max(np.abs(r - 1)) < 1e-3, 'not face on?'
    a = np.abs(hpv)
    psi0 = np.unwrap(np.angle(hpv))
    keep = a > 0
    return nu[keep], f[keep], a[keep], psi0[keep]


