"""
the segments an analysis reads: frequency-domain data of the two
detectors, their noise variance, the frequency grid and the band.
"""

import numpy as np

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
    path = isinstance(segments, (str, bytes)) or hasattr(segments, "__fspath__")
    z = np.load(segments) if path else segments
    data = {I: _member_rows(segments, "d_" + I) if path else np.asarray(z["d_" + I]) for I in "HL"}
    n = data["H"].shape[0]
    rows = np.arange(n) if rows is None else np.asarray(rows, int)
    out = {"rows": rows, "f": np.asarray(z["f"], float), "nu": np.asarray(z["nu"], np.int64),
           "T": float(z["T"]), "fmin": float(z["fmin"]), "fmax": float(z["fmax"])}
    for I in "HL":
        out["d_" + I] = np.asarray(data[I][rows]).astype(complex)
        key = "sig2_" + I if "sig2_" + I in z else "sig2"
        out["sig2_" + I] = np.asarray(z[key], float)
    return out


def _member_rows(path, name):
    """an array of an npz file mapped from disk (only the rows used are read), or loaded if compressed"""
    import zipfile
    info = zipfile.ZipFile(path).getinfo(name + ".npy")
    if info.compress_type != 0:
        return np.load(path)[name]
    with open(path, "rb") as fh:
        fh.seek(info.header_offset)
        nlen, elen = np.frombuffer(fh.read(30)[26:30], np.uint16)
        fh.seek(info.header_offset + 30 + int(nlen) + int(elen))
        ver = np.lib.format.read_magic(fh)
        rd = np.lib.format.read_array_header_1_0 if ver == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = rd(fh)
        offset = fh.tell()
    return np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=shape, order="F" if fortran else "C")
