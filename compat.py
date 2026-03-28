"""
compat.py — Compatibility shims for different numpy versions.
"""

import numpy as np

# numpy < 2.0 has np.trapz; numpy >= 2.0 renamed it to np.trapezoid
trapz = getattr(np, 'trapz', None) or getattr(np, 'trapezoid')
