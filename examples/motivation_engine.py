"""Self-contained engine for the motivating single-event experiment
(paper Figures 1 and 2).

Numpy only. The detector geometry is ported from the companion project so
the notebook needs no numba, MKL or JAX; the frozen intrinsic template and
the noise spectra live in data/motivation_inputs.npz and are identical to
the ones used for the paper. Two experiments freeze every parameter of a
35+35 solar-mass binary except one:

  build_tc_matrix : slide the arrival time t_c, read the matched-filter
                    log likelihood ratio -> Figure 1.
  spectrum        : SVD of the noise-averaged likelihood-ratio covariance
                    over a continuum of loudness -> Figure 2.
"""
import numpy as np

C_LIGHT = 299792458.0          # m/s
R_EARTH = 6.371e6              # m

# name: (latitude deg, longitude deg, x-arm deg E-ccw, y-arm deg E-ccw), LAL
_DET = {"H1": (46.4551, -119.4077, 125.9994, 215.9994),
        "L1": (30.5629,  -90.7742, 197.7165, 287.7165)}


def _detector(name):
    """Response tensor D_ab = (x_a x_b - y_a y_b)/2 and position vector (m)."""
    lat, lon, ax, ay = _DET[name]
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    E = np.array([-np.sin(lon), np.cos(lon), 0.0])
    N = np.array([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon),
                  np.cos(lat)])
    U = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon),
                  np.sin(lat)])
    a, b = np.deg2rad(ax), np.deg2rad(ay)
    u = np.cos(a) * E + np.sin(a) * N
    v = np.cos(b) * E + np.sin(b) * N
    return 0.5 * (np.outer(u, u) - np.outer(v, v)), R_EARTH * U


def _antenna(D, theta, phi, psi):
    """F_plus, F_cross for angle arrays (theta, phi, psi); D is 3x3."""
    st, ct = np.sin(theta), np.cos(theta)
    sp, cp = np.sin(phi), np.cos(phi)
    e = np.stack([ct * cp, ct * sp, -st], -1)          # theta_hat
    f = np.stack([-sp, cp, np.zeros_like(st)], -1)     # phi_hat
    fp0 = np.einsum('...a,ab,...b->...', e, D, e) \
        - np.einsum('...a,ab,...b->...', f, D, f)
    fx0 = 2.0 * np.einsum('...a,ab,...b->...', e, D, f)
    c2, s2 = np.cos(2 * psi), np.sin(2 * psi)
    return c2 * fp0 + s2 * fx0, -s2 * fp0 + c2 * fx0


def _sky(theta, phi):
    st, ct = np.sin(theta), np.cos(theta)
    sp, cp = np.sin(phi), np.cos(phi)
    return np.stack([st * cp, st * sp, ct], -1)


def projected_signal(u, f, theta, phi, ci, psi, t_c, phi_c):
    """h_{I nu} at the given extrinsic angles, stacked (..., 2 n_bin):
    all H1 bins then all L1 bins. Angle arguments broadcast."""
    theta, phi, ci, psi, t_c, phi_c = np.broadcast_arrays(
        theta, phi, ci, psi, t_c, phi_c)
    shape = theta.shape
    th, ph = theta.ravel(), phi.ravel()
    parts = []
    for name in ("H1", "L1"):
        D, position = _detector(name)
        Fp, Fx = _antenna(D, th, ph, psi.ravel())
        B = Fp * 0.5 * (1.0 + ci.ravel()**2) + 1j * Fx * ci.ravel()
        tau = _sky(th, ph) @ position / C_LIGHT
        phase = np.exp(-2j * np.pi * np.outer(tau + t_c.ravel(), f)
                       - 1j * phi_c.ravel()[:, None])
        parts.append(B[:, None] * u[None, :] * phase)
    return np.concatenate(parts, axis=-1).reshape(shape + (2 * len(f),))


class Motivation:
    """The frozen single-event setup shared by Figures 1 and 2."""

    # one fixed sky / inclination / polarization point; only t_c or phi_c move
    THETA, PHI, CI, PSI, PHI_C = 1.2, 2.0, 0.7, 0.4, 0.0

    def __init__(self, inputs):
        z = np.load(inputs)
        self.u, self.f = z['u1'], z['f']
        self.T = float(z['T'])
        self.sig = np.sqrt(np.concatenate([z['s2H'], z['s2L']]))
        self.unorm = float(np.sqrt(np.sum(np.abs(self._uhat(self.T / 2)[0])**2)))

    def _uhat(self, t_c):
        h = projected_signal(self.u, self.f, self.THETA, self.PHI, self.CI,
                             self.PSI, np.atleast_1d(t_c), self.PHI_C)
        return h / self.sig

    def _uhat_phi(self, phi_c):
        h = projected_signal(self.u, self.f, self.THETA, self.PHI, self.CI,
                             self.PSI, self.T / 2, np.atleast_1d(phi_c))
        return h / self.sig

    def build_tc_matrix(self, rho, t_true, t_filt, seed0=770):
        """lnL matrix: rows = true-t_c injections, cols = filter t_c."""
        x = rho / self.unorm
        U = self._uhat(t_filt)
        norm2 = np.sum(np.abs(x * U)**2, axis=1)
        S = self._uhat(t_true)
        n = self.f.size
        M = np.empty((len(t_true), len(t_filt)))
        for i in range(len(t_true)):
            r = np.random.default_rng(seed0 + i)
            noise = (r.normal(size=2 * n) + 1j * r.normal(size=2 * n)) \
                / np.sqrt(2)
            d = noise + x * S[i]
            M[i] = 2 * np.real(d.conj() @ (x * U).T) - norm2
        return M

    def spectrum(self, rho, kind='tc', n_cell=140, half=0.020):
        """Normalized eigenspectrum of the likelihood-ratio covariance at
        loudness rho, sweeping the arrival time ('tc') or the phase ('phic')."""
        if kind == 'tc':
            grid = np.linspace(self.T / 2 - half, self.T / 2 + half, n_cell)
            U = self._uhat(grid)
        else:
            grid = np.linspace(0.0, 2 * np.pi, n_cell, endpoint=False)
            U = self._uhat_phi(grid)
        ReO = np.real(U.conj() @ U.T)
        x = rho / self.unorm
        A = 2 * x * x * ReO
        Q = np.exp(A - A.max())
        a = Q.mean(axis=0)
        b = (Q.T @ Q) / n_cell
        C = np.exp(A) * b - np.outer(a, a)
        w = np.linalg.eigvalsh(C)[::-1]
        return w / w[0]
