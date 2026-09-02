"""Golden digests for the reproducibility acceptance test.

Regenerate ONLY with a deliberate change to emitter physics, the link budget or
the RNG stream layout, and say so in the commit message:

    python -c "from tests.regen import main; main()"
"""

from __future__ import annotations

EPISODE_DIGESTS: dict[str, str] = {"easy": "9a5d2a655aae6a6b7cb31c484899db7b", "medium": "1181fc6ab533c33add1f8a9026d1a92c", "hard": "edd8e67e4eaac2c161c1a3b53e31693a"}
