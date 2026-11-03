"""How long OmniVoice should make a take (Spec 002, 2.2 step 2).

Kept free of torch/OmniVoice imports so the policy can be unit tested and shared by the TTS server process.
"""

import os
from typing import Optional

# A take shorter than this share of the original's speech is slowed down towards it
FILL_RATIO = float(os.environ.get("TTS_FILL_RATIO", "0.9"))
# Never slower than this (0.88 = 12% slower than the model's natural pace) ...
MIN_SPEED = float(os.environ.get("TTS_MIN_SPEED", "0.88"))
# ... and never faster than this; longer lines go back to the LLM to be shortened
MAX_SPEEDUP = float(os.environ.get("TTS_MAX_SPEEDUP", "1.2"))


def choose_duration(natural: float, target: Optional[float], limit: Optional[float], pad: float = 0.0) -> Optional[float]:
    """Returns the duration to request from the model, or None for its natural pace.

    ``natural`` is the model's own estimate of the speech length, ``target`` the time the original speaks and
    ``limit`` the latest end (including the silence after the line). ``pad`` is the silence the model adds per side.
    """
    limit = limit or target
    if not limit:
        return None
    target = min(target or limit, limit)
    budget = max(0.3, limit - 2 * pad)
    wanted = max(0.3, target - 2 * pad)

    if natural > budget:
        return max(budget, natural / MAX_SPEEDUP)
    if natural < wanted * FILL_RATIO:
        slower = min(wanted, natural / MIN_SPEED)
        return slower if slower > natural else None
    return None
