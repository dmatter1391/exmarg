"""
the template bank and the matched filter, on the CPU (numpy) or the GPU (cupy).

  z_I(t) = FFT_t[ conj(d_I) / sig2_I * a e^{i psi0} ]   (zero padded to n_time = 8 nu_max)

the template factor a e^{i psi0} depends on the template only, so the
bank is built once; a segment gives (n_template, n_time) arrays of
ln|z_I| and arg z_I at every arrival time of the segment.
"""

import numpy as np

from ..profiles import profile
from .segments import D0, read_segments


def array_module(device):
    """numpy for device 'cpu', cupy for device 'gpu'"""
    if device == "cpu":
        return np
    if device == "gpu":
        import cupy
        return cupy
    raise ValueError("device must be 'cpu' or 'gpu'")


class MassBank:
    """templates for one frequency grid and noise curve.

    M, q, chi  (n_template,) detector-frame total mass, mass ratio
               m2/m1 <= 1 and common aligned spin
    u          (n_template,) template power sum a^2 / sig2_H at D0
    W          (n_template, n_freq) complex64 a e^{i psi0}, on the device"""

    def __init__(self, M, q, chi, seg, device="cpu"):
        self.M, self.q, self.chi = M, q, chi
        self.device, self.xp = device, array_module(device)
        self.T, self.fmin, self.fmax = seg["T"], seg["fmin"], seg["fmax"]
        self.nu = seg["nu"]
        if not np.all(np.diff(self.nu) == 1):
            raise ValueError("frequency bins must be contiguous")
        self.lo, self.hi = int(self.nu[0]), int(self.nu[-1]) + 1
        self.n_time = 8 * int(self.nu.max())
        W = np.empty((M.size, self.nu.size), np.complex64)
        self.u = np.empty(M.size)
        for i in range(M.size):
            m1 = M[i] / (1.0 + q[i]); m2 = M[i] * q[i] / (1.0 + q[i])
            nu, _, a1, p0 = profile(m1, m2, chi[i], chi[i], self.T, self.fmin, self.fmax, 1.0)
            j = np.searchsorted(self.nu, nu)
            a = np.zeros(self.nu.size); a[j] = a1 / D0
            psi0 = np.zeros(self.nu.size); psi0[j] = p0
            W[i] = a * np.exp(1j * psi0)
            self.u[i] = float((a ** 2 / seg["sig2_H"]).sum())
        self.W = self.xp.asarray(W)

    def trajectories(self, d, sig2, chunk=1024):
        """matched filter of one segment against every template.

        d, sig2  dicts {'H': (n_freq,), 'L': (n_freq,)} of data and noise
        returns dict of float32 (n_template, n_time) arrays on the device:
        lqH, phH, lqL, phL = ln|z_I| and arg z_I."""
        xp = self.xp
        nm = self.W.shape[0]
        out = {}
        for I in "HL":
            w = xp.asarray(np.ascontiguousarray(np.conj(d[I]) / sig2[I], np.complex64))
            lq = xp.empty((nm, self.n_time), np.float32)
            ph = xp.empty((nm, self.n_time), np.float32)
            for a in range(0, nm, chunk):
                b = min(a + chunk, nm)
                pad = xp.zeros((b - a, self.n_time), np.complex64)
                pad[:, self.lo:self.hi] = self.W[a:b] * w[None, :]
                z = xp.fft.fft(pad, axis=-1)
                lq[a:b] = 0.5 * xp.log(xp.maximum(z.real ** 2 + z.imag ** 2, 1e-38))
                ph[a:b] = xp.arctan2(z.imag, z.real)
                del pad, z
            out["lq" + I], out["ph" + I] = lq, ph
        return out


def mass_bank(bank, segments, device="cpu"):
    """the template bank.

    bank      .npy path or array: (n,) detector-frame total masses (equal
              mass, zero spin) or (n, 3) rows (M_det, q, chi)
    segments  anything read_segments takes; its frequency grid, T, band
              and sig2_H define the templates
    device    'cpu' or 'gpu'"""
    B = np.load(bank) if isinstance(bank, (str, bytes)) or hasattr(bank, "__fspath__") else np.asarray(bank)
    B = np.asarray(B, float)
    if B.ndim == 1:
        B = np.stack([B, np.ones_like(B), np.zeros_like(B)], axis=1)
    seg = read_segments(segments, rows=[0])
    return MassBank(*[np.ascontiguousarray(B[:, i]) for i in range(3)], seg, device)
