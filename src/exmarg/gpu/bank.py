"""
segments, the template bank on the device, the matched filter and the
choice of arrival times.

  z_I(t) = FFT_t[ conj(d_I) / sig2_I * a e^{i psi0} ]   (zero padded to n_time = 8 nu_max)

the template factor a e^{i psi0} depends on the template only, so the
bank is built once and stays on the device; a segment uploads its two
frequency series and gets (n_template, n_time) arrays of ln|z_I| and
arg z_I back.
"""

import numpy as np
import cupy as cp

from ..profiles import profile

D0 = 10410.0              # reference distance of the stores (Mpc): y = D0 / D_L


def read_segments(segments, rows=None):
    """the arrays the analysis reads from a set of segments.

    segments  npz path, loaded npz, or dict with
                d_H, d_L  (n_segment, n_freq) complex frequency-domain data
                f         (n_freq,) frequencies (Hz)
                nu        (n_freq,) integer bins f T, contiguous
                sig2      (n_freq,) noise variance E|n|^2 of each bin,
                          or sig2_H, sig2_L one per detector
                T         segment length (s)
                fmin, fmax  frequency band of the templates (Hz)
    rows      segment indices to keep (default all)

    returns a dict with the same keys, d_H, d_L as complex128 and
    sig2_H, sig2_L as float64."""
    z = np.load(segments) if isinstance(segments, (str, bytes)) or hasattr(segments, "__fspath__") else segments
    n = z["d_H"].shape[0]
    rows = np.arange(n) if rows is None else np.asarray(rows, int)
    out = {"rows": rows, "f": np.asarray(z["f"], float), "nu": np.asarray(z["nu"], np.int64),
           "T": float(z["T"]), "fmin": float(z["fmin"]), "fmax": float(z["fmax"])}
    for I in "HL":
        out["d_" + I] = np.asarray(z["d_" + I])[rows].astype(complex)
        key = "sig2_" + I if "sig2_" + I in z else "sig2"
        out["sig2_" + I] = np.asarray(z[key], float)
    return out


class MassBank:
    """templates on the device for one frequency grid and noise curve.

    M, q, chi  (n_template,) detector-frame total mass, mass ratio
               m2/m1 <= 1 and common aligned spin
    u          (n_template,) template power sum a^2 / sig2_H at D0
    W          device (n_template, n_freq) complex64 a e^{i psi0}"""

    def __init__(self, M, q, chi, seg):
        self.M, self.q, self.chi = M, q, chi
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
        self.W = cp.asarray(W)

    def trajectories(self, d, sig2, chunk=1024):
        """matched filter of one segment against every template.

        d, sig2  dicts {'H': (n_freq,), 'L': (n_freq,)} of data and noise
        returns dict of device float32 (n_template, n_time) arrays
        lqH, phH, lqL, phL = ln|z_I| and arg z_I."""
        nm = self.W.shape[0]
        out = {}
        decode = _decode()
        for I in "HL":
            w = cp.asarray(np.ascontiguousarray(np.conj(d[I]) / sig2[I], np.complex64))
            lq = cp.empty((nm, self.n_time), cp.float32)
            ph = cp.empty((nm, self.n_time), cp.float32)
            for a in range(0, nm, chunk):
                b = min(a + chunk, nm)
                pad = cp.zeros((b - a, self.n_time), cp.complex64)
                pad[:, self.lo:self.hi] = self.W[a:b] * w[None, :]
                decode(cp.fft.fft(pad, axis=-1), lq[a:b], ph[a:b])
                del pad
            out["lq" + I], out["ph" + I] = lq, ph
        return out


def mass_bank(bank, segments):
    """the template bank on the device.

    bank      .npy path or array: (n,) detector-frame total masses (equal
              mass, zero spin) or (n, 3) rows (M_det, q, chi)
    segments  anything read_segments takes; its frequency grid, T, band
              and sig2_H define the templates"""
    B = np.load(bank) if isinstance(bank, (str, bytes)) or hasattr(bank, "__fspath__") else np.asarray(bank)
    B = np.asarray(B, float)
    if B.ndim == 1:
        B = np.stack([B, np.ones_like(B), np.zeros_like(B)], axis=1)
    seg = read_segments(segments, rows=[0])
    return MassBank(*[np.ascontiguousarray(B[:, i]) for i in range(3)], seg)


_DECODE = None


def _decode():
    global _DECODE
    if _DECODE is None:
        _DECODE = cp.ElementwiseKernel(
            "complex64 z", "float32 lq, float32 ph",
            "float re = z.real(), im = z.imag();"
            "lq = 0.5f * logf(fmaxf(re * re + im * im, 1e-38f));"
            "ph = atan2f(im, re);", "decode_z")
    return _DECODE


def select_times(traj, tau_idx, n_loud, n_random, seed, rest_order="partition"):
    """arrival times to visit for every template.

    the n_loud times with the largest |z_L(t)|^2 + max_tau |z_H(t+tau)|^2
    are kept exactly; n_random more are drawn uniformly (with
    replacement) from the rest and carry ln((n_time - n_loud)/n_random),
    so the sum over the selection estimates the full sum without bias.
    the ranking uses the data only.  rest_order sets how the rest is
    listed before the draw: 'partition' (cupy.argpartition order) or
    'time' (ascending arrival time).

    returns device (n_template, n_loud + n_random) int32 times, sorted,
    and float32 log weights."""
    K, S = int(n_loud), int(n_random)
    nm, NT = traj["lqH"].shape
    aH = cp.exp(2.0 * traj["lqH"]); aL = cp.exp(2.0 * traj["lqL"])
    # max of |z_H|^2 over the contiguous lags [lo, hi]: a sparse-table sliding maximum
    tv = np.asarray(tau_idx).ravel().astype(int)
    lo, hi = int(tv.min()), int(tv.max())
    if not np.array_equal(np.sort(tv), np.arange(lo, hi + 1)):
        raise ValueError("lags are not contiguous")
    kk = int(np.floor(np.log2(hi - lo + 1)))
    W = aH
    for k in range(kk):
        W = cp.maximum(W, cp.roll(W, -(1 << k), axis=1))
    mx = cp.maximum(cp.roll(W, -lo, axis=1), cp.roll(W, -(hi - (1 << kk) + 1), axis=1))
    del W
    score = aL + mx
    del mx, aH, aL
    if rest_order == "partition":
        part = cp.argpartition(-score, K, axis=1).astype(cp.int32)
        del score
        top, rest_all = part[:, :K], part[:, K:]
    else:
        top = _top(score, K)
        del score
        mask = cp.ones((nm, NT), bool)
        mask[cp.arange(nm)[:, None], top] = False
        rest_all = cp.nonzero(mask)[1].reshape(nm, NT - K).astype(cp.int32)
        del mask
    rng = cp.random.default_rng(seed)
    pick = rng.integers(0, NT - K, size=(nm, S), dtype=cp.int32)
    rest = cp.take_along_axis(rest_all, pick, axis=1)
    sel = cp.concatenate([top, rest], axis=1)
    del rest_all, rest, pick
    w = cp.zeros((nm, K + S), cp.float32)
    w[:, K:] = float(np.log((NT - K) / S))
    o = cp.argsort(sel, axis=1)
    sel = cp.take_along_axis(sel, o, axis=1)
    w = cp.take_along_axis(w, o, axis=1)
    return cp.ascontiguousarray(sel), cp.ascontiguousarray(w)


def _top(score, K):
    """indices of the K largest entries of each row (in no particular order)"""
    try:
        import torch
    except ImportError:
        return cp.argpartition(-score, K, axis=1)[:, :K].astype(cp.int32)
    idx = torch.topk(torch.from_dlpack(score), K, dim=1, largest=True, sorted=False).indices
    return cp.from_dlpack(idx.to(torch.int32))
