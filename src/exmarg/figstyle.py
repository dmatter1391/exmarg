"""Single source of truth for the paper figure style.

Every plot script imports this module and calls apply() once,
right after importing matplotlib.  The constants below are the
ONLY font sizes and shared palette colors the paper figures use.
"""
# ---- font sizes
FS_LABEL = 20     # axis labels AND colorbar labels
FS_TICK = 16      # all tick labels, including colorbar ticks
FS_LEGEND = 15    # legends and inline annotations
FS_PANEL = 18     # panel tags, small per-panel titles (e.g. W^(n)_ab)

# ---- shared palette
NAVY = '#16263F'
RED = '#D1605E'
# blue ladder, light = shallow order, dark = deep order
LADDER7 = ['#BFD3E6', '#A8C4DC', '#7FA8C9', '#5A89B4', '#3D6A99',
           '#28496C', '#16263F']
LADDER6 = LADDER7[:6]
LADDER3 = [LADDER7[0], LADDER7[3], LADDER7[6]]
STEEL = '#7FA8C9'


def ramp(n, lo='#AECBE0', hi='#B0443E'):
    """n-color ladder interpolated between the two palette ends:
    light blue for the low end, dark brick red for the top
    (overlaid level sets in the grid corner figures). The hue ramp
    doubles as a light-to-dark ramp so neighbours stay separated
    under color vision deficiency and in greyscale print. Figures
    using it paint their truth crosshairs neutral grey, red
    belongs to the ladder top."""
    from matplotlib.colors import (LinearSegmentedColormap,
                                   to_hex)
    import numpy as np
    cm = LinearSegmentedColormap.from_list('', [lo, hi])
    return [to_hex(c) for c in cm(np.linspace(0.0, 1.0, n))]


def apply():
    """Install the shared rcParams for all paper figures."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family': 'serif',
        'mathtext.fontset': 'cm',
        'font.size': FS_LEGEND,
        'axes.labelsize': FS_LABEL,
        'xtick.labelsize': FS_TICK,
        'ytick.labelsize': FS_TICK,
        'legend.fontsize': FS_LEGEND,
        'lines.linewidth': 3.0,
    })
