"""
detector geometry and noise, Hanford and Livingston.
"""

import numpy as np

c = 299792458.0     # speed of light [m/s]

# vertex positions and arm unit vectors (LAL values)
x = {'H': np.array([-2.16141492636e6, -3.83469517889e6,  4.60035022664e6]),
     'L': np.array([-7.42760447238e4, -5.49628371971e6,  3.22425701744e6])}
u = {'H': np.array([-0.22389266154,  0.79983062746,  0.55690487831]),
     'L': np.array([-0.95457412153, -0.14158077340, -0.26218911324])}
v = {'H': np.array([-0.91397818574,  0.02609403989, -0.40492342125]),
     'L': np.array([ 0.29774156894, -0.48791033647, -0.82054461286])}

# detector tensor d^ab = (1/2)(u^a u^b - v^a v^b)
d = {I: 0.5 * (np.outer(u[I], u[I]) - np.outer(v[I], v[I])) for I in 'HL'}


def basis(theta, phi):
    '''
    Outputs
    * n_hat, theta_hat, phi_hat (last axis is the vector component)
    '''
    theta, phi = np.asarray(theta, float), np.asarray(phi, float)
    st, ct, sp, cp = np.sin(theta), np.cos(theta), np.sin(phi), np.cos(phi)
    n = np.stack([st * cp, st * sp, ct], axis=-1)
    th = np.stack([ct * cp, ct * sp, -st], axis=-1)
    ph = np.stack([-sp, cp, np.zeros_like(st)], axis=-1)
    return n, th, ph


def Fplus_Fcross(I, theta, phi, psi):
    '''
    antenna patterns of detector I at polarization angle psi
    '''
    _, th, ph = basis(theta, phi)
    ep = (np.einsum('...a,...b->...ab', th, th) -
          np.einsum('...a,...b->...ab', ph, ph))
    ex = (np.einsum('...a,...b->...ab', th, ph) +
          np.einsum('...a,...b->...ab', ph, th))
    c2, s2 = np.cos(2 * psi), np.sin(2 * psi)
    Fp = np.einsum('ab,...ab->...', d[I],
                   ep * c2[..., None, None] - ex * s2[..., None, None])
    Fx = np.einsum('ab,...ab->...', d[I],
                   ep * s2[..., None, None] + ex * c2[..., None, None])
    return Fp, Fx


from pathlib import Path as _Path

# selectable noise curve: 'ajith' is the historical analytic fit,
# the others interpolate the tabulated LIGO amplitude spectral
# densities under data/psd (aplus: T2000012 AplusDesign, o4:
# aligo_O4high, design: T1800044 aLIGODesign).  set_curve() switches
# the module; every case built afterwards carries the chosen curve in
# its sig2, and the evaluation side follows the case, so generation
# and processing cannot disagree within one dataset.
CURVE = 'ajith'
_PSD_DIR = _Path(__file__).resolve().parents[2] / 'data' / 'psd'
_PSD_FILES = {'aplus': 'AplusDesign.txt', 'o4': 'aligo_O4high.txt',
              'design': 'aLIGODesign.txt'}
_TAB = {}


def set_curve(name):
    global CURVE
    if name != 'ajith' and name not in _PSD_FILES:
        raise ValueError('unknown noise curve %r' % name)
    CURVE = name


def _table(name):
    if name not in _TAB:
        d = np.loadtxt(_PSD_DIR / _PSD_FILES[name])
        _TAB[name] = (np.log(d[:, 0]), np.log(d[:, 1] ** 2))
    return _TAB[name]


def S(I, f, f_low=10.0):
    '''
    one sided noise power spectral density [1/Hz] of the selected
    curve (module state, see set_curve).  the same curve is used for
    both detectors
    '''
    f = np.asarray(f, dtype=float)
    if CURVE == 'ajith':
        y = np.where(f > 0, f, 1.0) / 215.0
        out = 1e-49 * (y**-4.14 - 5.0 * y**-2
                       + 111.0 * (1.0 - y**2 + 0.5 * y**4)
                       / (1.0 + 0.5 * y**2))
        return np.where(f >= f_low, out, np.inf)
    lf, ls = _table(CURVE)
    lo = max(f_low, np.exp(lf[0]))
    inside = (f >= lo) & (f <= np.exp(lf[-1]))
    out = np.full(f.shape, np.inf)
    out[inside] = np.exp(np.interp(np.log(f[inside]), lf, ls))
    return out


def freqs(T, fmin, fmax):
    '''
    f_nu = nu / T with fmin <= f_nu <= fmax
    '''
    nu = np.arange(int(np.ceil(fmin * T)), int(np.floor(fmax * T)) + 1)
    return nu / T
