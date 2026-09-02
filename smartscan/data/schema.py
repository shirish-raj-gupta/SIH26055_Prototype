"""Normative on-disk schema for the SmartScan RF environment dataset.

One module holds every column name, dtype and unit, so the builder, the loader,
the dataset card and the tests cannot drift apart. If you change a field here,
bump :data:`DATASET_VERSION` -- a consumer that silently reads a renamed column
as ``NaN`` is worse than one that refuses to load.

Storage decisions, and why
--------------------------
**Occupancy is bit-packed, not sparse.** At 3-16 % density a COO representation
costs 8 bytes per non-zero against 1 bit per cell dense. ``np.packbits`` gives a
fixed 156 KB per episode at ``B=128, T=10000`` and beats sparse at every density
above ~1.5 %.

**SNR and emitter id are stored only where occupied.** Both are meaningless in
empty cells (the SNR sentinel is a constant), so they ride the occupancy mask as
flat float16/int16 vectors in raster order. That is a 5x saving over dense.

**Observations are wide, not long.** One row per dwell with a ``hit_mask``
bitfield and ``K`` SNR columns, rather than one row per (dwell, channel). Long
format would be 119 M rows for the full corpus; wide is 30 M.
"""

from __future__ import annotations

from typing import Final

import numpy as np

__all__ = [
    "DATASET_VERSION",
    "INDEX_COLUMNS",
    "MANIFEST_COLUMNS",
    "NOTEBOOK_WORKING_BYTES",
    "OBSERVATION_COLUMNS",
    "SIZE_BUDGET_BYTES",
    "SPLITS",
    "TIER_COUNTS",
    "TRUTH_ARRAYS",
    "episode_id",
    "split_for_seed",
]

#: Bump on any schema change. The loader refuses a version it does not know.
DATASET_VERSION: Final[str] = "1.1.0"

#: Ceiling the builder asserts against, defaulting to Kaggle's **persistent
#: dataset quota per user**.
#:
#: Three different Kaggle limits get confused with each other; they are not
#: interchangeable and only the first constrains this builder:
#:
#: * **100 GB** -- persistent dataset storage per user, across all datasets.
#:   This is the budget below.
#: * **20 GB** -- a *notebook's* writable ``/kaggle/working`` disk. Attached
#:   datasets are mounted **read-only** at ``/kaggle/input`` and do **not**
#:   count against it, so a dataset larger than 20 GB is fine to publish and
#:   attach -- it just must be streamed rather than copied into the working
#:   directory. The training notebooks in ``notebooks/kaggle/`` stream.
#: * **~200 GB** -- a per-dataset ceiling on some account tiers. Not relied on.
SIZE_BUDGET_BYTES: Final[int] = 100 * 1024**3

#: A notebook's writable disk. Anything a notebook *writes* must fit here.
NOTEBOOK_WORKING_BYTES: Final[int] = 20 * 1024**3

#: Episodes per tier, as specified in the problem brief.
TIER_COUNTS: Final[dict[str, int]] = {"easy": 1000, "medium": 1200, "hard": 800}

#: Split proportions. Assignment is by **scenario seed**, never by time slice:
#: splitting a single episode across train and test would let a model memorise
#: the emitters it is about to be scored on, and the leakage would be invisible.
SPLITS: Final[dict[str, float]] = {"train": 0.70, "val": 0.15, "test": 0.15}

#: Arrays inside ``truth_occupancy.npz`` and their dtypes.
TRUTH_ARRAYS: Final[dict[str, str]] = {
    "occupancy_packed": "uint8",   # np.packbits of the (B, T) boolean occupancy
    "snr_db": "float16",           # SNR at occupied cells only, raster order
    "duty": "float16",             # sub-slot occupied FRACTION at occupied cells
    "emitter_id": "int16",         # strongest emitter at occupied cells, raster order
    "n_pulses": "int16",           # pulses from that emitter, occupied cells only
    "shape": "int32",              # (B, T)
}

#: ``index.parquet`` -- one row per episode.
INDEX_COLUMNS: Final[dict[str, str]] = {
    "episode_id": "string",        # '<tier>_<seed>', globally unique and stable
    "tier": "string",              # easy | medium | hard
    "seed": "int64",               # scenario seed
    "split": "string",             # train | val | test, assigned BY SEED
    "n_emitters": "int32",
    "n_channels": "int32",         # B
    "n_slots": "int32",            # T
    "dt_s": "float64",
    "occupancy_frac": "float32",   # fraction of (B, T) cells occupied
    "n_popup": "int32",
    "n_interferer": "int32",
    "config_hash": "string",       # blake2b of the resolved config
    "path": "string",              # directory relative to the dataset root
    "bytes": "int64",              # on-disk size of this episode
}

#: ``emitter_manifest.parquet`` -- one row per emitter per episode.
MANIFEST_COLUMNS: Final[dict[str, str]] = {
    "episode_id": "string",
    "emitter_id": "int16",         # >= 1; 0 is reserved for noise
    "emitter_class": "string",
    "f_centre_hz": "float64",
    "home_channel": "int16",
    "scan_period_s": "float64",    # NaN when the emitter is not scan-periodic
    "beamwidth_deg": "float64",    # NaN when not applicable
    "pri_s": "float64",            # NaN when not applicable
    "pulse_width_s": "float64",
    "hop_set": "string",           # comma-separated channel indices, '' if none
    "hop_rate_hz": "float64",
    "threat_priority": "float32",  # [0, 1]
    "t_first_active": "int32",     # slot; > 0.6*T for pop-ups
    "is_novel": "bool",
    "is_interferer": "bool",
    "detection_mode": "string",    # energy | pulse
    "eirp_dbm": "float32",
    "range_km": "float32",
}

#: ``observations.parquet`` -- one row per dwell, per replayed scheduler.
OBSERVATION_COLUMNS: Final[dict[str, str]] = {
    "episode_id": "string",
    "agent": "string",             # which scheduler produced this trace
    "step": "int32",               # decision index
    "t": "int32",                  # slot at which the dwell was observed
    "action": "int16",             # chosen centre channel
    "window_lo": "int16",          # first channel of the observed window
    "slots_elapsed": "int8",       # 1, or 1 + t_settle after a retune
    "hit_mask": "uint8",           # bit i set if channel window_lo+i declared a hit
    "true_hit_mask": "uint8",      # EVAL ONLY: hits that were genuine, not false alarms
    "snr_est_db": "list<float16>", # length K, NaN where no hit was declared
    "reward": "float32",
}


def episode_id(tier: str, seed: int) -> str:
    """Return the stable, globally unique id for an episode.

    Derived from ``(tier, seed)`` rather than a running counter so that adding
    or removing episodes never renumbers the rest.

    Args:
        tier: Difficulty tier.
        seed: Scenario seed.

    Returns:
        Identifier of the form ``'medium_20260902'``.
    """
    return f"{tier}_{int(seed)}"


def split_for_seed(seed: int, salt: str = "smartscan-v1") -> str:
    """Assign a seed to train, val or test, deterministically.

    Hashing the seed rather than slicing a sorted list means the assignment is
    stable when the corpus grows: adding episodes never moves an existing one
    between splits, so a model evaluated last week is still evaluated on the
    same held-out data this week.

    Args:
        seed: Scenario seed.
        salt: Namespace for the hash; change it to reshuffle deliberately.

    Returns:
        ``'train'``, ``'val'`` or ``'test'``.
    """
    from smartscan.seeding import stable_hash

    u = (stable_hash(f"{salt}:{seed}") % 10_000) / 10_000.0
    cumulative = 0.0
    for name, frac in SPLITS.items():
        cumulative += frac
        if u < cumulative:
            return name
    return "test"


def pack_occupancy(occupancy: np.ndarray) -> np.ndarray:
    """Bit-pack a ``(B, T)`` boolean occupancy tensor.

    Args:
        occupancy: Boolean or uint8 array of shape ``(B, T)``.

    Returns:
        Flat uint8 array of packed bits, C-order.
    """
    return np.packbits(np.ascontiguousarray(occupancy).astype(bool))


def unpack_occupancy(packed: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Invert :func:`pack_occupancy`.

    Args:
        packed: Flat uint8 array from :func:`pack_occupancy`.
        shape: Original ``(B, T)`` shape.

    Returns:
        Boolean array of shape ``shape``.
    """
    b, t = int(shape[0]), int(shape[1])
    return np.unpackbits(packed, count=b * t).astype(bool).reshape(b, t)
