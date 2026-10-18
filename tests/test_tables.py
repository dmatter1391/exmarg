"""the W tables: finite, one row per (n, a, b) for every a, b <= n, built with both distance priors; the weak-signal ORF
of the Hanford-Livingston pair has its first zero near 64 Hz."""
import numpy as np

import exmarg


def test_table_keys_and_finiteness():
    tau, W = exmarg.w_table_dl(5.0, 5.0, 0.4, 3, n_tau=513, nside=8)
    assert np.isfinite(tau).all()
    expected = {(n, a, b) for n in range(4) for a in range(n + 1) for b in range(n + 1)}
    assert set(W) == expected
    for row in W.values():
        assert row.shape == tau.shape and np.isfinite(row).all()


def test_madau_dickinson_prior_builds():
    exmarg.use_distance_prior('madau-dickinson')
    try:
        tau, W = exmarg.w_table_dl(np.array([1.0, 20.0]), np.array([1.0, 20.0]), exmarg.madau_dickinson_ymin(), 2,
                                   n_tau=513, nside=8)
    finally:
        exmarg.use_distance_prior('volume')
    assert len(W) == 2
    assert all(np.isfinite(row).all() for table in W for row in table.values())


def test_weak_signal_orf_first_zero():
    f1, g2, *_ = exmarg.geometry.frequency_slices(nside=16, n_psi=12, n_chi=12)
    first_zero = f1[np.argmax(np.diff(np.sign(g2.real)) != 0)]
    assert 60.0 < first_zero < 68.0
