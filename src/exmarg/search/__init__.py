"""
the analysis of segments with the stored tables, on the CPU or the GPU.

    store = load_redshift_store(store_dir)                  # once: the tables of every signal power node
    bank = mass_bank(bank_3col_file, segments, device)      # once per frequency grid and noise curve
    res = pe_intrinsic(segments, bank, store, prior)        # M, q, chi, u, ln_lambda, ln_b (distance integrated)

    bank = mass_bank(bank_file, segments, device)           # equal mass, zero spin
    res = pe_redshift(segments, bank, store)                # m1, z, ln_lambda, ln_b

device is 'cpu' (numpy and numba, every core) or 'gpu' (cupy).  For the
GPU, load the store with pinned=True for fast copies to the device.
"""

from .segments import D0, read_segments
from .bank import mass_bank
from .store import load_redshift_store
from .pe import pe_redshift, pe_intrinsic, distance_weights
