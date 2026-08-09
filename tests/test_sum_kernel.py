"""the summed-table runtime (SumKernel) against the per-order path,
on a tiny synthetic kernel: (1) the tables themselves must satisfy
exp(lnS_m) = sum_{n<=m} exp(lnK_n) cell by cell, (2) lnlam from the
two paths must agree exactly when the lookups sit on grid nodes and
closely off-grid (the only difference off-grid is interpolate-then-
sum vs sum-then-interpolate)."""
import numpy as np
import pytest

from exmarg.kernel import Kernel

N, NTIME, T = 6, 64, 4.0
NQ, NPHI = 16, 8


def tiny_kernel(seed=3):
    rng = np.random.default_rng(seed)
    keys, W = [], []
    tau_idx = np.arange(-2, 3)
    for n in range(N + 1):
        for a in range(n + 1):
            for b in range(a + 1):
                keys.append((n, a, b))
                W.append(rng.uniform(0.1, 1.0, tau_idx.size))
    return Kernel(50.0, tau_idx, NTIME, T, np.array(keys),
                  np.array(W), qmax=5.0, nq=NQ, nphi=NPHI,
                  q_span=50.0)


def test_summed_tables_are_exact_partial_sums():
    k = tiny_kernel()
    sks = k.summed([2, 4, 6])
    for m, sk in sks.items():
        expect = np.logaddexp.reduce(
            np.stack([k.lnK[n] for n in range(m + 1)]), axis=0)
        assert np.allclose(sk.lnS, expect, rtol=0, atol=1e-5)


def test_lnlam_matches_on_grid():
    """lookups placed EXACTLY on grid nodes: the two paths must
    agree to float precision"""
    k = tiny_kernel()
    rng = np.random.default_rng(7)
    q = np.exp(k.lnq[rng.integers(2, NQ - 2, NTIME)])
    ph1 = 2 * np.pi / NPHI * rng.integers(0, NPHI, NTIME)
    z = {'H': q * np.exp(1j * ph1),
         'L': np.exp(k.lnq[rng.integers(2, NQ - 2, NTIME)])
              * np.ones(NTIME)}
    for m, sk in k.summed([2, 6]).items():
        a = k.lnlam(z, 50.0, 2.0, 0.5, np.ones(m + 1))
        b = sk.lnlam(z, 50.0, 2.0, 0.5)
        assert abs(a - b) < 1e-5, (m, a, b)


def test_lnlam_close_off_grid():
    """generic lookups: only the interpolation order differs"""
    k = tiny_kernel()
    rng = np.random.default_rng(11)
    z = {I: (rng.uniform(0.3, 4.0, NTIME)
             * np.exp(2j * np.pi * rng.uniform(size=NTIME)))
         for I in 'HL'}
    for m, sk in k.summed([2, 6]).items():
        a = k.lnlam(z, 50.0, 2.0, 0.5, np.ones(m + 1))
        b = sk.lnlam(z, 50.0, 2.0, 0.5)
        assert abs(a - b) < 0.02, (m, a, b)


def test_fused_pair_matches_numpy_path():
    """the fused two-node runtime against the plain SumKernel
    lookups, including the ln u bracketing weight"""
    from exmarg.kernel import sum_lnlam_pair
    ka, kb = tiny_kernel(3), tiny_kernel(4)
    kb.u_node = 60.0
    ska = ka.summed([6])[6]
    skb = kb.summed([6])[6]
    rng = np.random.default_rng(13)
    z = {I: (rng.uniform(0.3, 4.0, NTIME)
             * np.exp(2j * np.pi * rng.uniform(size=NTIME)))
         for I in 'HL'}
    u = 54.0
    ref0 = ska.lnlam(z, u, 2.0, 0.5)
    ref1 = skb.lnlam(z, u, 2.0, 0.5)
    x = np.log(u / 50.0) / np.log(60.0 / 50.0)
    want = (1 - x) * ref0 + x * ref1
    got = sum_lnlam_pair(ska, skb, z, u, 2.0, 0.5)
    assert abs(got - want) < 1e-10, (got, want)
    # single-node form: same kernel twice
    got1 = sum_lnlam_pair(ska, ska, z, u, 2.0, 0.5)
    assert abs(got1 - ref0) < 1e-10
