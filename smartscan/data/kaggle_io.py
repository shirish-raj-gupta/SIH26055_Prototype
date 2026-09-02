"""Load the published dataset, with a fallback that cannot fail because of wifi.

Resolution order, in full:

1. An explicit local path (``SMARTSCAN_DATA`` or the ``root`` argument).
2. The local cache from a previous download.
3. Kaggle, via ``kagglehub``.
4. **Regenerate the identical episodes from their seeds.**

Step 4 is the point. The dataset is a *derived* artefact: every episode is a
deterministic function of ``(tier, seed, config)``, so a missing download is an
inconvenience rather than a failure. A live demo that dies because the venue
wifi is captive-portalled is a demo that does not happen, and the fallback costs
about 0.8 s per episode.

**No credential is ever read from, or written to, this repository.** Kaggle
credentials come from ``KAGGLE_USERNAME``/``KAGGLE_KEY`` or ``~/.kaggle/kaggle.json``,
both outside the tree and both in ``.gitignore``.
"""

from __future__ import annotations

import hashlib
import os
import warnings
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from smartscan.data.schema import (
    DATASET_VERSION,
    TIER_COUNTS,
    episode_id,
    split_for_seed,
    unpack_occupancy,
)

__all__ = [
    "DATASET_SLUG",
    "EpisodeRecord",
    "LoadedDataset",
    "OccupancyWindowDataset",
    "credentials_available",
    "load_dataset",
    "resolve_dataset_root",
    "verify_dataset",
]

#: Kaggle dataset slug. The owner is filled from KAGGLE_USERNAME at publish time.
DATASET_SLUG: str = "ew-smart-scan-rf-environment"

#: Environment variable pointing at an already-built local dataset.
ENV_LOCAL_ROOT: str = "SMARTSCAN_DATA"


def credentials_available() -> bool:
    """Whether Kaggle credentials are present, without reading their values.

    Loads ``.env`` first (never overriding a real shell export), so a user who
    filled in the template does not also have to remember to source it.

    Returns:
        True if usable credentials exist in the environment, ``.env`` or
        ``~/.kaggle/kaggle.json``.
    """
    from smartscan.credentials import credential_status

    return credential_status().kaggle


def _default_cache() -> Path:
    """Return the default cache directory for downloaded datasets."""
    base = os.environ.get("SMARTSCAN_CACHE") or (Path.home() / ".cache" / "smartscan")
    return Path(base)


@dataclass(frozen=True)
class EpisodeRecord:
    """One episode's ground truth and replayed traces.

    Attributes:
        episode_id: Stable id, ``'<tier>_<seed>'``.
        tier: Difficulty tier.
        seed: Scenario seed.
        split: ``train``, ``val`` or ``test``.
        occupancy: ``(B, T)`` bool ground-truth occupancy.
        snr_db: ``(B, T)`` float32 SNR; ``SNR_FLOOR_DB`` where unoccupied.
        duty: ``(B, T)`` float32 sub-slot occupied fraction in ``[0, 1]``. A
            frequency-agile emitter hopping inside one slot, or a 1 us pulse in
            a 1 ms slot, is only representable here -- not in ``occupancy``.
        emitter_id: ``(B, T)`` int16 strongest-emitter label, 0 for noise.
        manifest: Emitter manifest rows (a DataFrame when pandas is available).
        observations: Replayed dwell traces, or ``None`` if not loaded.
    """

    episode_id: str
    tier: str
    seed: int
    split: str
    occupancy: np.ndarray
    snr_db: np.ndarray
    emitter_id: np.ndarray
    duty: np.ndarray | None = None
    manifest: Any = None
    observations: Any = None

    @property
    def shape(self) -> tuple[int, int]:
        """``(B, T)``."""
        return self.occupancy.shape  # type: ignore[return-value]


@dataclass
class LoadedDataset:
    """A resolved dataset: where it came from and what it contains.

    Attributes:
        root: Directory holding the episodes, or ``None`` when regenerated.
        source: ``'local'``, ``'cache'``, ``'kaggle'`` or ``'regenerated'``.
        index: One row per episode (a DataFrame when pandas is available).
        split: The split requested.
        tier: The tier requested, or ``None`` for all.
    """

    root: Path | None
    source: str
    index: Any
    split: str
    tier: str | None = None

    def __len__(self) -> int:
        return 0 if self.index is None else int(len(self.index))

    def episode_ids(self) -> list[str]:
        """Return the episode ids in this dataset."""
        return [] if self.index is None else list(self.index["episode_id"])

    def load(self, which: int | str, with_observations: bool = True) -> EpisodeRecord:
        """Materialise one episode.

        Args:
            which: Row position, or an ``episode_id``.
            with_observations: Also read ``observations.parquet``.

        Returns:
            The :class:`EpisodeRecord`.
        """
        row = (
            self.index.iloc[which]
            if isinstance(which, int)
            else self.index[self.index["episode_id"] == which].iloc[0]
        )
        if self.root is None:
            return _regenerate_episode(str(row["tier"]), int(row["seed"]), with_observations)
        return _read_episode(self.root, row, with_observations)

    def __iter__(self) -> Iterator[EpisodeRecord]:
        for i in range(len(self)):
            yield self.load(i)


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #
def resolve_dataset_root(
    root: str | Path | None = None,
    cache_dir: str | Path | None = None,
    allow_download: bool = True,
    owner: str | None = None,
) -> tuple[Path | None, str]:
    """Find the dataset, trying local paths before the network.

    Args:
        root: Explicit dataset directory; short-circuits everything else.
        cache_dir: Where downloads are cached.
        allow_download: Permit a Kaggle download.
        owner: Kaggle username owning the dataset; defaults to
            ``KAGGLE_USERNAME``.

    Returns:
        ``(path, source)``. ``path`` is ``None`` when nothing was found, in
        which case the caller should regenerate from seeds.
    """
    explicit = root is not None
    if explicit:
        p = Path(root)
        if (p / "index.parquet").is_file():
            return p, "local"
        # Deliberately do NOT fall through to the implicit candidates below.
        # A caller who names a root means that root; quietly substituting a
        # different corpus from the working directory would hand back data the
        # caller never asked for, under the episode ids of the one they did.
        warnings.warn(
            f"{p} has no index.parquet; not substituting another local dataset. "
            f"Falling back to the cache, then Kaggle, then regeneration.",
            stacklevel=2,
        )
    else:
        env_root = os.environ.get(ENV_LOCAL_ROOT)
        if env_root and (Path(env_root) / "index.parquet").is_file():
            return Path(env_root), "local"

        for candidate in (Path("build/dataset"), Path("data/dataset")):
            if (candidate / "index.parquet").is_file():
                return candidate, "local"

    cache = Path(cache_dir) if cache_dir else _default_cache()
    cached = cache / DATASET_SLUG
    if (cached / "index.parquet").is_file():
        return cached, "cache"

    if not allow_download:
        return None, "regenerated"

    try:
        import kagglehub
    except ImportError:
        warnings.warn(
            "kagglehub is not installed, so the dataset cannot be downloaded; "
            "regenerating episodes from seeds instead. "
            'Install with `pip install "smartscan[kaggle]"`.',
            stacklevel=2,
        )
        return None, "regenerated"

    handle = f"{owner or os.environ.get('KAGGLE_USERNAME', 'smartscan')}/{DATASET_SLUG}"
    try:
        path = Path(kagglehub.dataset_download(handle))
    except Exception as exc:
        warnings.warn(
            f"could not download {handle} ({type(exc).__name__}: {exc}); "
            "regenerating episodes from seeds instead. This is expected offline.",
            stacklevel=2,
        )
        return None, "regenerated"

    if not (path / "index.parquet").is_file():
        inner = next((d for d in path.rglob("index.parquet")), None)
        if inner is None:
            warnings.warn(f"downloaded {handle} has no index.parquet; regenerating", stacklevel=2)
            return None, "regenerated"
        path = inner.parent
    return path, "kaggle"


def load_dataset(
    split: str = "train",
    tier: str | None = None,
    root: str | Path | None = None,
    cache_dir: str | Path | None = None,
    allow_download: bool = True,
    n_episodes: int | None = None,
    owner: str | None = None,
) -> LoadedDataset:
    """Load a split of the dataset, downloading or regenerating as needed.

    Args:
        split: ``'train'``, ``'val'``, ``'test'`` or ``'all'``.
        tier: Restrict to one tier, or ``None`` for all.
        root: Explicit dataset directory.
        cache_dir: Download cache location.
        allow_download: Permit a Kaggle download.
        n_episodes: Cap the number of episodes (useful for smoke runs and for
            the regeneration fallback, which is not free).
        owner: Kaggle username owning the dataset.

    Returns:
        A :class:`LoadedDataset`.

    Raises:
        ValueError: If ``split`` or ``tier`` is not recognised.
    """
    if split not in {"train", "val", "test", "all"}:
        raise ValueError(f"unknown split {split!r}; use train, val, test or all")
    if tier is not None and tier not in TIER_COUNTS:
        raise ValueError(f"unknown tier {tier!r}; use one of {sorted(TIER_COUNTS)}")

    path, source = resolve_dataset_root(root, cache_dir, allow_download, owner)

    if path is None:
        index = _synthetic_index(split, tier, n_episodes)
        return LoadedDataset(None, "regenerated", index, split, tier)

    import pandas as pd

    index = pd.read_parquet(path / "index.parquet")
    if split != "all":
        index = index[index["split"] == split]
    if tier is not None:
        index = index[index["tier"] == tier]
    if n_episodes is not None:
        index = index.head(n_episodes)
    return LoadedDataset(path, source, index.reset_index(drop=True), split, tier)


def _synthetic_index(split: str, tier: str | None, n_episodes: int | None) -> Any:
    """Rebuild the index from seeds alone, matching the builder's assignment."""
    import pandas as pd

    from smartscan.data.dataset_builder import seed_block

    root_seed = 20260902
    rows = []
    for t, count in TIER_COUNTS.items():
        if tier is not None and t != tier:
            continue
        for i in range(count):
            seed = root_seed + seed_block(t) + i
            s = split_for_seed(seed)
            if split != "all" and s != split:
                continue
            rows.append({"episode_id": episode_id(t, seed), "tier": t, "seed": seed, "split": s})
            if n_episodes is not None and len(rows) >= n_episodes:
                return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def _read_episode(root: Path, row: Any, with_observations: bool) -> EpisodeRecord:
    """Read one episode's files from disk."""
    import pandas as pd

    from smartscan.env.propagation import SNR_FLOOR_DB

    ep_dir = root / str(row["path"])
    z = np.load(ep_dir / "truth_occupancy.npz")
    shape = (int(z["shape"][0]), int(z["shape"][1]))
    occ = unpack_occupancy(z["occupancy_packed"], shape)

    snr = np.full(shape, SNR_FLOOR_DB, dtype=np.float32)
    snr[occ] = z["snr_db"].astype(np.float32)
    eid = np.zeros(shape, dtype=np.int16)
    eid[occ] = z["emitter_id"]

    duty = None
    if "duty" in z.files:  # absent in datasets built before schema 1.1.0
        duty = np.zeros(shape, dtype=np.float32)
        duty[occ] = z["duty"].astype(np.float32)

    obs = None
    if with_observations and (ep_dir / "observations.parquet").is_file():
        obs = pd.read_parquet(ep_dir / "observations.parquet")

    return EpisodeRecord(
        episode_id=str(row["episode_id"]),
        tier=str(row["tier"]),
        seed=int(row["seed"]),
        split=str(row["split"]),
        occupancy=occ,
        snr_db=snr,
        emitter_id=eid,
        duty=duty,
        manifest=pd.read_parquet(ep_dir / "emitter_manifest.parquet"),
        observations=obs,
    )


def _regenerate_episode(tier: str, seed: int, with_observations: bool) -> EpisodeRecord:
    """Recreate an episode from its seed, byte-identically to the published one."""
    from smartscan.agents import build_agent
    from smartscan.config import load_config
    from smartscan.data.dataset_builder import DEFAULT_AGENTS, _observation_frame
    from smartscan.env.rf_environment import build_episode, generate_scenario
    from smartscan.runner import run_episode

    cfg = load_config(f"{tier}.yaml")
    scenario = generate_scenario(seed, config=cfg)
    ep = build_episode(scenario)
    eid = episode_id(tier, seed)

    obs = None
    if with_observations:
        try:
            import pandas as pd

            results = {
                k: run_episode(cfg, seed, build_agent(k, cfg, seed, scenario),
                               scenario=scenario, episode=ep)
                for k in DEFAULT_AGENTS
            }
            obs = _observation_frame(eid, results, cfg.receiver.ibw_channels, pd)
        except ImportError:  # pragma: no cover - pandas is an optional extra
            obs = None

    return EpisodeRecord(
        episode_id=eid,
        tier=tier,
        seed=seed,
        split=split_for_seed(seed),
        occupancy=ep.occupancy > 0,
        snr_db=ep.snr_db,
        emitter_id=ep.emitter_id,
        duty=ep.duty,
        manifest=None,
        observations=obs,
    )


def verify_dataset(root: str | Path, sample: int = 25) -> dict[str, Any]:
    """Check a dataset's integrity and internal consistency.

    Verifies that every indexed episode exists, that the recorded byte counts
    match the files on disk, that no seed appears in two splits, and that the
    tensors round-trip through the packed representation.

    Args:
        root: Dataset root.
        sample: How many episodes to open and check in full.

    Returns:
        Dict with ``ok``, ``n_episodes``, ``problems`` and ``digest``.
    """
    import pandas as pd

    path = Path(root)
    index = pd.read_parquet(path / "index.parquet")
    problems: list[str] = []

    dupes = index.groupby("seed")["split"].nunique()
    if (dupes > 1).any():
        problems.append(f"{int((dupes > 1).sum())} seeds appear in more than one split")
    if index["episode_id"].duplicated().any():
        problems.append("duplicate episode_id values")

    rng = np.random.default_rng(0)
    picks = rng.choice(len(index), size=min(sample, len(index)), replace=False)
    for i in picks:
        row = index.iloc[int(i)]
        ep_dir = path / str(row["path"])
        for name in ("truth_occupancy.npz", "emitter_manifest.parquet", "observations.parquet"):
            if not (ep_dir / name).is_file():
                problems.append(f"{row['episode_id']}: missing {name}")
        if (ep_dir / "truth_occupancy.npz").is_file():
            rec = _read_episode(path, row, with_observations=False)
            if rec.occupancy.shape != (int(row["n_channels"]), int(row["n_slots"])):
                problems.append(f"{row['episode_id']}: shape mismatch")
            if abs(float(rec.occupancy.mean()) - float(row["occupancy_frac"])) > 1e-5:
                problems.append(f"{row['episode_id']}: occupancy_frac disagrees with the tensor")
        if split_for_seed(int(row["seed"])) != str(row["split"]):
            problems.append(f"{row['episode_id']}: split does not match split_for_seed")

    h = hashlib.blake2b(digest_size=16)
    for eid in sorted(index["episode_id"]):
        h.update(eid.encode())
    return {
        "ok": not problems,
        "n_episodes": int(len(index)),
        "n_checked": int(len(picks)),
        "problems": problems,
        "digest": h.hexdigest(),
        "dataset_version": DATASET_VERSION,
    }


# --------------------------------------------------------------------------- #
# torch interface
# --------------------------------------------------------------------------- #
def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            'torch is required for the dataset classes; install `pip install "smartscan[ml]"`.'
        ) from exc
    return torch


class OccupancyWindowDataset:
    """Sliding windows over replayed traces, for next-slot occupancy prediction.

    Yields ``(x, y, mask, y_true)`` where:

    * ``x`` is ``(4, B, W)`` -- visit mask, hit mask, reported SNR, staleness;
    * ``y`` is the observation-only label at ``t+1`` (what the receiver saw);
    * ``mask`` marks the channels the receiver was actually tuned to, so the
      loss is only taken where a label exists;
    * ``y_true`` is the privileged full-band label, for the distillation teacher
      **only**.

    The SNR plane carries the receiver's *reported* estimate, never the true
    SNR: a model trained on ground-truth amplitude would not be deployable.

    Args:
        dataset: A :class:`LoadedDataset`.
        window: Window length ``W`` in slots.
        stride: Slots between consecutive windows.
        agent: Which replayed trace to use, or ``None`` for all of them.
        max_windows_per_episode: Cap per episode.
        class_balanced: Oversample windows whose next slot contains activity on
            a rarely-occupied channel. Occupancy runs at a few per cent, so an
            unbalanced sampler spends almost all its budget on empty spectrum.
    """

    def __init__(
        self,
        dataset: LoadedDataset,
        window: int = 128,
        stride: int = 16,
        agent: str | None = "sequential",
        max_windows_per_episode: int = 256,
        class_balanced: bool = False,
    ) -> None:
        self.torch = _require_torch()
        self.dataset = dataset
        self.window = int(window)
        self.stride = int(stride)
        self.agent = agent
        self.max_windows = int(max_windows_per_episode)
        self.class_balanced = class_balanced
        self._cache: dict[str, tuple[np.ndarray, ...]] = {}
        self._episode_ids = dataset.episode_ids()
        self._plan: list[tuple[str, int]] = []
        self._build_plan()

    def _build_plan(self) -> None:
        """Enumerate (episode, window-start) pairs aligned to real dwell slots.

        A window is only useful if the receiver actually observed something at
        ``t0 + 1``; otherwise the label mask is empty and the window contributes
        no gradient. With ``t_settle = 2`` the receiver observes about one slot
        in three, so a fixed stride wastes roughly two thirds of the plan. We
        therefore read just the ``t`` column of the trace and place windows so
        that ``t0 + 1`` is a dwell.
        """
        for eid in self._episode_ids:
            dwells = self._dwell_slots(eid)
            usable = dwells[dwells > self.window]
            if usable.size == 0:
                continue
            chosen = usable[:: max(self.stride // 3, 1)][: self.max_windows]
            self._plan.extend((eid, int(t) - 1) for t in chosen)

    def _dwell_slots(self, eid: str) -> np.ndarray:
        """Return the slots at which the selected agent observed, cheaply.

        Reads only the two columns needed, so planning does not materialise the
        whole trace.
        """
        row = self.dataset.index[self.dataset.index["episode_id"] == eid].iloc[0]
        if self.dataset.root is not None and "path" in row:
            path = self.dataset.root / str(row["path"]) / "observations.parquet"
            if path.is_file():
                import pandas as pd

                frame = pd.read_parquet(path, columns=["agent", "t"])
                if self.agent is not None:
                    frame = frame[frame["agent"] == self.agent]
                return np.unique(frame["t"].to_numpy())
        # Regenerated datasets have no parquet on disk; fall back to the record.
        rec = self.dataset.load(eid, with_observations=True)
        if rec.observations is None or not len(rec.observations):
            n_slots = int(row["n_slots"]) if "n_slots" in row else 10_000
            return np.arange(self.window + 1, n_slots - 1)
        frame = rec.observations
        if self.agent is not None:
            frame = frame[frame["agent"] == self.agent]
        return np.unique(frame["t"].to_numpy())

    def __len__(self) -> int:
        return len(self._plan)

    def _episode_planes(self, eid: str) -> tuple[np.ndarray, ...]:
        """Build and cache the four input planes plus labels for one episode."""
        if eid in self._cache:
            return self._cache[eid]

        rec = self.dataset.load(eid, with_observations=True)
        b, t = rec.occupancy.shape
        visit = np.zeros((b, t), dtype=np.float32)
        hit = np.zeros((b, t), dtype=np.float32)
        snr = np.zeros((b, t), dtype=np.float32)

        obs = rec.observations
        if obs is not None and len(obs):
            frame = obs if self.agent is None else obs[obs["agent"] == self.agent]
            k = len(frame["snr_est_db"].iloc[0]) if len(frame) else 0
            for lo, slot, mask_bits, snr_row in zip(
                frame["window_lo"].to_numpy(), frame["t"].to_numpy(),
                frame["hit_mask"].to_numpy(), frame["snr_est_db"].to_numpy(), strict=True,
            ):
                lo, slot = int(lo), int(slot)
                visit[lo : lo + k, slot] = 1.0
                bits = np.unpackbits(np.uint8(mask_bits), bitorder="little")[:k]
                hit[lo : lo + k, slot] = bits
                vals = np.nan_to_num(np.asarray(snr_row, dtype=np.float32), nan=0.0)
                snr[lo : lo + k, slot] = vals / 40.0

        stale = np.zeros((b, t), dtype=np.float32)
        last = np.full(b, -1.0)
        log_t = np.log(max(t, 2))
        for slot in range(t):
            stale[:, slot] = np.log1p(slot - last) / log_t
            last[visit[:, slot] > 0] = slot

        planes = (visit, hit, snr, stale, rec.occupancy.astype(np.float32))
        # Bounded cache: episodes are ~5 MB of planes each.
        if len(self._cache) > 8:
            self._cache.pop(next(iter(self._cache)))
        self._cache[eid] = planes
        return planes

    def __getitem__(self, idx: int) -> tuple[Any, ...]:
        eid, t0 = self._plan[int(idx)]
        visit, hit, snr, stale, truth = self._episode_planes(eid)
        sl = slice(t0 - self.window, t0)
        x = np.stack([visit[:, sl], hit[:, sl], snr[:, sl], stale[:, sl]])
        y = hit[:, t0 + 1]
        mask = visit[:, t0 + 1] > 0
        y_true = truth[:, t0 + 1]
        as_t = self.torch.as_tensor
        return as_t(x), as_t(y), as_t(mask), as_t(y_true)

    def sampler(self, seed: int = 0) -> Any:
        """Return a class-balanced sampler, or ``None`` when balancing is off.

        Windows whose next slot contains *any* activity are upweighted so that
        roughly half of each batch carries a positive label. At 2-5 % base
        occupancy an unbalanced sampler spends 95 % of its gradient budget on
        empty spectrum, and the model converges to "always idle".

        Args:
            seed: Seed for the sampler's generator.

        Returns:
            A ``torch.utils.data.WeightedRandomSampler``, or ``None``.
        """
        if not self.class_balanced:
            return None
        from torch.utils.data import WeightedRandomSampler

        weights = np.ones(len(self), dtype=np.float64)
        for i, (eid, t0) in enumerate(self._plan):
            _, _, _, _, truth = self._episode_planes(eid)
            weights[i] = 8.0 if truth[:, t0 + 1].any() else 1.0
        gen = self.torch.Generator().manual_seed(int(seed))
        return WeightedRandomSampler(
            weights=self.torch.as_tensor(weights), num_samples=len(self),
            replacement=True, generator=gen,
        )

    def loader(self, batch_size: int = 64, seed: int = 0, **kwargs: Any) -> Any:
        """Return a ``DataLoader`` over this dataset.

        Args:
            batch_size: Batch size.
            seed: Seed for the class-balanced sampler.
            **kwargs: Forwarded to ``DataLoader``.

        Returns:
            A ``torch.utils.data.DataLoader``.
        """
        from torch.utils.data import DataLoader

        sampler = self.sampler(seed)
        return DataLoader(
            self, batch_size=batch_size, sampler=sampler,
            shuffle=(sampler is None), **kwargs,
        )


def episode_seeds(tier: str, n: int | None = None, root_seed: int = 20260902) -> Sequence[int]:
    """Return the canonical scenario seeds for a tier.

    Args:
        tier: Difficulty tier.
        n: How many; defaults to the tier's full count.
        root_seed: Base seed used by the builder.

    Returns:
        The seed sequence.
    """
    from smartscan.data.dataset_builder import seed_block

    count = n if n is not None else TIER_COUNTS[tier]
    base = root_seed + seed_block(tier)
    return [base + i for i in range(count)]
