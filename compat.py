# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
compat.py — Compatibility shims for different numpy versions.
"""

import numpy as np

# numpy >= 2.0 renamed np.trapz to np.trapezoid and deprecated the old name.
# Prefer the new one where it exists: the old spelling still resolves on numpy 2
# but emits a DeprecationWarning on every call, which buries real warnings from
# the diagnostics under integration noise.
trapz = getattr(np, 'trapezoid', None) or getattr(np, 'trapz')
