"""Deterministic RNG substreams from a single root seed.

The naive ``np.random.seed(s)`` approach couples every consumer to every other:
adding a 16th emitter shifts the draws of the first 15, and swapping the
scheduler changes the world it is being measured in. That silently invalidates
every ablation you will ever want to run.

Instead we spawn a *tree* of independent streams from one
:class:`numpy.random.SeedSequence`, keyed by a **fixed name registry**. A stream
name maps to a stable integer, so:

* adding an emitter does not perturb existing emitters,
* changing the agent does not change the environment,
* re-running any single component in isolation reproduces it exactly.

See ``docs/architecture.md`` §13.
"""

from __future__ import annotations

import hashlib
from typing import Final

import numpy as np

__all__ = ["SeedTree", "StreamName", "rng_for", "stable_hash"]

#: Registry of stream names. Order is irrelevant (names are hashed, not indexed),
#: but the set is fixed so a typo becomes a ``KeyError`` rather than a silent new
#: stream that happens to be reproducible-but-wrong.
_STREAMS: Final[frozenset[str]] = frozenset(
    {
        "scenario",  # emitter placement, class parameter draws
        "emitter",  # per-emitter internal randomness (keyed by emitter index)
        "receiver",  # detection draws, false alarms, SNR estimation noise
        "agent",  # policy-internal randomness (epsilon, Thompson sampling)
        "torch",  # torch global seed
        "eval_bootstrap",  # resampling in the statistics layer
        "dataset",  # dataset builder episode ordering / split assignment
    }
)

StreamName = str


def stable_hash(text: str) -> int:
    """Hash a string to a stable 64-bit integer.

    ``hash()`` is salted per process in CPython, so it cannot be used for
    reproducible seeding. blake2b is stable across processes, machines and
    Python versions.

    Args:
        text: String to hash.

    Returns:
        Non-negative integer below ``2**63``.
    """
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "big") >> 1


class SeedTree:
    """A root seed plus named, mutually independent substreams.

    Example:
        >>> tree = SeedTree(42)
        >>> a = tree.rng("scenario")
        >>> b = tree.rng("emitter", 3)
        >>> # a and b are independent; re-requesting either reproduces it exactly
        >>> bool(np.all(tree.rng("scenario").random(4) == a.random(4)))
        True

    Args:
        root: Root seed. Every artefact of a run derives from this one integer.
    """

    def __init__(self, root: int) -> None:
        self.root = int(root)
        self._base = np.random.SeedSequence(self.root)

    def spawn_key(self, name: StreamName, index: int = 0) -> tuple[int, ...]:
        """Return the deterministic spawn key for a named substream.

        Args:
            name: Stream name; must be in the registry.
            index: Sub-index within the stream (e.g. emitter number).

        Returns:
            Spawn key tuple consumed by :class:`numpy.random.SeedSequence`.

        Raises:
            KeyError: If ``name`` is not a registered stream.
        """
        if name not in _STREAMS:
            raise KeyError(f"unknown RNG stream {name!r}; registered: {sorted(_STREAMS)}")
        # Truncate to 32 bits: SeedSequence spawn keys are uint32 words.
        return (stable_hash(name) % (2**32), int(index) % (2**32))

    def seed_sequence(self, name: StreamName, index: int = 0) -> np.random.SeedSequence:
        """Return the :class:`SeedSequence` for a named substream."""
        return np.random.SeedSequence(self.root, spawn_key=self.spawn_key(name, index))

    def rng(self, name: StreamName, index: int = 0) -> np.random.Generator:
        """Return a fresh generator for a named substream.

        The generator is *fresh* each call: two calls with the same
        ``(name, index)`` produce identical sequences. Callers that need to draw
        incrementally should hold onto the returned generator.

        Args:
            name: Stream name; must be in the registry.
            index: Sub-index within the stream.

        Returns:
            A PCG64 generator.
        """
        return np.random.Generator(np.random.PCG64(self.seed_sequence(name, index)))

    def torch_seed(self) -> int:
        """Return a stable 32-bit seed for torch's global RNG."""
        return int(self.seed_sequence("torch").generate_state(1, dtype=np.uint32)[0])


def rng_for(root: int, name: StreamName, index: int = 0) -> np.random.Generator:
    """Convenience wrapper for a one-off substream generator.

    Args:
        root: Root seed.
        name: Stream name; must be in the registry.
        index: Sub-index within the stream.

    Returns:
        A PCG64 generator.
    """
    return SeedTree(root).rng(name, index)
