# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
compat.py — Compatibility shims for different numpy versions.
"""

import numpy as np

# numpy < 2.0 has np.trapz; numpy >= 2.0 renamed it to np.trapezoid
trapz = getattr(np, 'trapz', None) or getattr(np, 'trapezoid')
