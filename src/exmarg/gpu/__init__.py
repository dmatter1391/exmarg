"""
GPU parameter estimation of single segments (requires cupy and astropy).

    store = load_redshift_store(store_dir)          # once, page-locked host memory
    bank = mass_bank(bank_file, segments)           # once per frequency grid and noise curve
    res = pe_redshift(segments, bank, store)        # m1, z, ln_lambda, ln_b

    store = load_intrinsic_store(store_dir)
    bank = mass_bank(bank_3col_file, segments)
    res = pe_intrinsic(segments, bank, store)       # M, q, chi, u, ln_lambda, ln_b
"""

from .bank import D0, mass_bank, read_segments, select_times
from .store import load_redshift_store, load_intrinsic_store
from .pe import pe_redshift, pe_intrinsic
