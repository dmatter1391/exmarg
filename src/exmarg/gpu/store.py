"""
the (beta, Delta) tables, read once into page-locked host memory and
copied to the device in groups.

redshift store   bd_*.npz, one file per signal power node s, members
                 T (tau, phi+, beta, Delta) float32, beta, delta,
                 tau_idx, alpha (amplitude axis), s_node
intrinsic store  ugrid.npz (u: template powers) and marg_uNN.npz, one
                 file per u, members T, beta, delta, tau_idx; the
                 distance is already integrated into each table
"""

import glob
import os
import struct
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import cupy as cp

NPHI = 96                 # phase bins of the full circle; the tables hold NPHI/2 + 1


def read_npz_member(path, name):
    """one uncompressed .npy member of an npz, read sequentially"""
    info = zipfile.ZipFile(path).getinfo(name)
    if info.compress_type != 0:
        raise ValueError("member %s of %s is compressed" % (name, path))
    with open(path, "rb") as fh:
        fh.seek(info.header_offset)
        nlen, elen = struct.unpack("<HH", fh.read(30)[26:30])
        fh.seek(info.header_offset + 30 + nlen + elen)
        ver = np.lib.format.read_magic(fh)
        rd = np.lib.format.read_array_header_1_0 if ver == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = rd(fh)
        arr = np.fromfile(fh, dtype=dtype, count=int(np.prod(shape)))
    return arr.reshape(shape, order="F" if fortran else "C")


def _read_table(path, pinned):
    """the T member of one store file, in page-locked memory when pinned (pageable if that fails)"""
    T = read_npz_member(path, "T.npy")
    if not pinned:
        return T
    try:
        buf = cp.cuda.alloc_pinned_memory(T.nbytes)
    except Exception:
        return T
    P = np.frombuffer(buf, T.dtype, T.size).reshape(T.shape)
    P[...] = T
    return P


def _read_tables(files, pinned, readers):
    with ThreadPoolExecutor(max_workers=readers) as ex:
        return list(ex.map(lambda f: _read_table(f, pinned), files))


@dataclass
class RedshiftStore:
    """signal power nodes s (sorted), their amplitude axis limits, grids and host tables"""
    s: np.ndarray
    alpha_lo: np.ndarray
    alpha_hi: np.ndarray
    beta: list
    delta: list
    tau_idx: np.ndarray
    tables: list


@dataclass
class IntrinsicStore:
    """template powers u (sorted), grids and host tables"""
    u: np.ndarray
    beta: np.ndarray
    delta: np.ndarray
    tau_idx: np.ndarray
    tables: list


def load_redshift_store(store_dir, pinned=True, readers=6):
    """every node of the redshift store in host memory (about 0.75 GB per node)"""
    files = glob.glob(os.path.join(store_dir, "bd_*.npz"))
    if not files:
        raise FileNotFoundError("no bd_*.npz in %s" % store_dir)
    head = []
    for f in files:
        with np.load(f) as h:
            head.append((float(h["s_node"]), float(h["alpha"][0]), float(h["alpha"][-1]),
                         h["beta"], h["delta"], h["tau_idx"], f))
    head.sort(key=lambda r: r[0])
    s, alo, ahi, beta, delta, tau, files = zip(*head)
    return RedshiftStore(s=np.array(s), alpha_lo=np.array(alo), alpha_hi=np.array(ahi),
                         beta=list(beta), delta=list(delta), tau_idx=tau[0],
                         tables=_read_tables(files, pinned, readers))


def load_intrinsic_store(store_dir, pinned=True, readers=6):
    """every table of the intrinsic store in host memory (about 1.4 GB per table)"""
    u = np.load(os.path.join(store_dir, "ugrid.npz"))["u"]
    files = [os.path.join(store_dir, "marg_u%02d.npz" % i) for i in range(u.size)]
    with np.load(files[0]) as h:
        beta, delta, tau = h["beta"], h["delta"], h["tau_idx"]
    return IntrinsicStore(u=u, beta=beta, delta=delta, tau_idx=tau,
                          tables=_read_tables(files, pinned, readers))


def upload(tables, tau_idx, beta, delta, stream=None):
    """host tables -> one contiguous device stack, in the layout kernel.gather reads.
    with a stream the copies from page-locked tables run asynchronously on it"""
    ntau, nphif, nb, nd = tables[0].shape
    tstride = int(tables[0].size)
    T = cp.empty(tstride * len(tables), np.float32)
    for i, Ti in enumerate(tables):
        T[i * tstride:(i + 1) * tstride].set(np.ascontiguousarray(Ti, np.float32).ravel(), stream=stream)
    return dict(T=T, tau=cp.asarray(np.ascontiguousarray(tau_idx, np.int64)), tstride=tstride,
                shape=(ntau, NPHI, nb, nd), beta=beta, delta=delta)


def free():
    """return the device and page-locked staging pools to the system"""
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
