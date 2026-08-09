"""one production-shaped builder chunk, as a test: the full
whole-number u ladder (35 nodes), nside 16, chunk of 2 pixels —
structurally identical to the real build, only the order depth is
reduced so the quick test runs in seconds.

    python -m pytest tests/ -q            (quick, N = 24)
    EXMARG_FULL_PROBE=1 python -m pytest tests/test_build_chunk.py -s
                                          (true N = 120 probe with
                                           timing projection; ~10+ min)
"""
import os
import time
import numpy as np
import pytest

from exmarg.tables import _par_chunk

NSIDE, YMIN, CHUNK = 16, 0.4, 2


def u_ladder(u_lo=5.0, u_hi=250.0, step=1.12):
    ladder = [int(np.ceil(u_lo))]
    while ladder[-1] < u_hi:
        ladder.append(min(int(round(u_hi)),
                          max(ladder[-1] + 1,
                              int(round(ladder[-1] * step)))))
    return np.array(ladder, float)


def run_chunk(n_max):
    us = u_ladder()
    n_psi, n_chi = 2 * n_max + 3, n_max + 3
    t0 = time.time()
    lo, t_del, Vc = _par_chunk((0, CHUNK, us, us, YMIN, n_max,
                                NSIDE, n_psi, n_chi))
    return us, Vc, time.time() - t0


def test_production_chunk_quick():
    """all nodes finite, deep orders alive, expected shape"""
    n_max = 24
    us, Vc, dt = run_chunk(n_max)
    n_keys = (n_max + 1) * (n_max + 2) * (n_max + 3) // 6
    assert Vc.shape == (us.size, CHUNK, n_keys)
    assert np.isfinite(Vc).all()
    # the top order must not have underflowed to zero anywhere on
    # the ladder (the old plain-Taylor units silently zeroed it)
    top = slice(n_keys - (n_max + 1) * (n_max + 2) // 2, n_keys)
    assert np.abs(Vc[:, :, top]).max(axis=(1, 2)).min() > 0.0


@pytest.mark.skipif(not os.environ.get('EXMARG_FULL_PROBE'),
                    reason='set EXMARG_FULL_PROBE=1 for the real '
                           'N=120 timing probe (~10+ min)')
def test_production_chunk_full_probe():
    """the true production chunk: N = 120, plus a wall-time
    projection printout for the sbatch job"""
    us, Vc, dt = run_chunk(120)
    assert np.isfinite(Vc).all()
    n_chunks = int(np.ceil(12 * NSIDE**2 / CHUNK))
    tot = n_chunks * dt * 1.10
    print('\nchunk %.1f s -> total ~%.0f core-hours'
          % (dt, tot / 3600))
    for w in (32, 64, 128):
        print('  %3d cores -> ~%.1f h wall time'
              % (w, tot / w / 3600))


def test_build_entrypoint_end_to_end(tmp_path):
    """the LAUNCHER script itself (arg parsing, node/pixel slicing,
    store writing) — the library tests do not cover it, and a
    NameError here killed two cluster arrays on 2026-07-31"""
    import subprocess, sys, glob, os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r = subprocess.run(
        [sys.executable, os.path.join(root, 'scripts',
                                      'build_store.py'),
         '4', '8', '2', '5', '12', '1.3', '0', '2', '0', '2'],
        capture_output=True, text=True, cwd=root, timeout=600)
    assert r.returncode == 0, r.stderr[-500:]
    made = glob.glob(os.path.join(root,
                                  'wtaylor_N4_*part0of2.npz'))
    assert len(made) == 3
    for f in made:
        os.unlink(f)
