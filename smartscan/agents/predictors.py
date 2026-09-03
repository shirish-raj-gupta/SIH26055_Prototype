"""Supervised next-slot occupancy prediction, and the scheduler it drives.

The task: from the observation history alone, predict ``P(X[b, t+1] = 1)`` for
**all** ``B`` channels -- including the 31/32 the receiver could not see. The
scheduler then tunes to the legal ``K``-window maximising expected
threat-weighted detections.

Three architectures, identical inputs, loss and parameter budget, so the
comparison is about inductive bias and nothing else:

``gru``
    A GRU over time with weights shared across channels, plus a cross-channel
    mixing convolution. Cheapest, and strong on per-channel temporal structure.

``tcn``
    Dilated 1-D convolutions in time (dilations 1, 2, ..., 64 give a 128-slot
    receptive field), shared across channels. No recurrence, so it trains fast.

``transformer``
    Channel tokens with **separate** channel and time positional encodings. The
    only one of the three that can attend across channels, which is what a
    frequency-agile emitter's hop set requires: its channels are correlated in a
    way no per-channel model can represent.

Scope, stated honestly
----------------------
A 4 s scan period is 4000 slots. No 128-slot window can represent it, so the
predictor learns **short-horizon** structure (burst persistence, hop-set
correlation, beam-dwell continuation) while the long periods are handled
analytically in :mod:`smartscan.analysis.scan_on_scan`. Claiming a 128-slot
window learns a 4 s period would be false, and the division of labour is by
design.

Masked focal loss
-----------------
Masked because at training time only visited channels carry a label. Focal
(``gamma = 2``) because occupancy runs at a few per cent positive, and plain BCE
collapses to "always idle" -- which scores 95 % accuracy and is useless.

Privileged distillation (**training time only**)
------------------------------------------------
In simulation we hold the full ``X[b, t]``. A *teacher* trains on it; the
*student* trains on observations alone with an added KL term to the teacher over
**all** channels, so the teacher supplies soft labels exactly where the student
has none. This is learning-using-privileged-information (Vapnik & Izmailov)
implemented as Hinton distillation. The deployed student consumes nothing but
observations -- enforced by :class:`PrivilegedAccess`, which raises if entered
while evaluation mode is set.
"""

from __future__ import annotations

import threading
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from smartscan.agents.base import Scheduler
from smartscan.agents.belief import BeliefState
from smartscan.config import Config, checkpoint_dir

__all__ = [
    "PrivilegedAccess",
    "SequencePredictorScheduler",
    "build_predictor",
    "build_windows",
    "masked_focal_loss",
    "train_predictor",
]

#: Input planes: visit mask, hit mask, SNR estimate, staleness.
N_PLANES: int = 4

_state = threading.local()


class PrivilegedAccess:
    """Guard around any use of ground truth during model training.

    Entering this context is the **only** legitimate way to read
    ``EpisodeTensors`` inside :mod:`smartscan.agents`, and it refuses to open
    while evaluation mode is set. The point is that "training-time only" is
    enforced structurally rather than promised in a comment.

    Example:
        >>> with PrivilegedAccess("teacher training"):
        ...     pass

    Args:
        reason: Why privileged data is needed; surfaced in the error message.

    Raises:
        RuntimeError: If entered while :func:`set_eval_mode` is active.
    """

    def __init__(self, reason: str = "") -> None:
        self.reason = reason

    def __enter__(self) -> PrivilegedAccess:
        if getattr(_state, "eval_mode", False):
            raise RuntimeError(
                f"PrivilegedAccess({self.reason!r}) opened during evaluation. Ground truth is a "
                "TRAINING-TIME-ONLY signal; the deployed student sees observations alone."
            )
        _state.privileged = True
        return self

    def __exit__(self, *exc: object) -> None:
        _state.privileged = False


def set_eval_mode(enabled: bool = True) -> None:
    """Enable or disable evaluation mode, which blocks :class:`PrivilegedAccess`."""
    _state.eval_mode = bool(enabled)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
@dataclass
class PredictorDataset:
    """Windowed training data for the occupancy predictor.

    Attributes:
        x: Inputs, shape ``(N, 4, B, W)`` float32.
        y: Observation-only targets, shape ``(N, B)`` float32.
        mask: Where ``y`` carries a real label, shape ``(N, B)`` bool.
        y_true: Privileged ground-truth targets, shape ``(N, B)`` float32.
    """

    x: np.ndarray
    y: np.ndarray
    mask: np.ndarray
    y_true: np.ndarray

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def split(self, frac: float = 0.8) -> tuple[PredictorDataset, PredictorDataset]:
        """Split chronologically into train and validation halves."""
        k = int(len(self) * frac)
        return (
            PredictorDataset(self.x[:k], self.y[:k], self.mask[:k], self.y_true[:k]),
            PredictorDataset(self.x[k:], self.y[k:], self.mask[k:], self.y_true[k:]),
        )


def _available_memory_bytes() -> int:
    """Best-effort free physical memory, or 0 if it cannot be determined.

    Deliberately dependency-free: psutil is not a requirement of this project
    and a training guard is not worth adding one for.

    Returns:
        Free bytes, or 0 when unknown (in which case callers must not guess).
    """
    try:  # Linux / macOS
        import os

        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, ValueError, OSError):
        pass
    try:  # Windows
        import ctypes

        class _Status(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        st = _Status()
        st.dwLength = ctypes.sizeof(_Status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return int(st.ullAvailPhys)
    except Exception:  # a guard must never break training
        pass
    return 0


def build_windows(
    config: Config,
    seeds: Sequence[int],
    scheduler_key: str = "sequential",
    stride: int = 16,
    max_windows_per_episode: int = 400,
) -> PredictorDataset:
    """Roll out a scheduler and cut its trace into training windows.

    Args:
        config: Resolved configuration.
        seeds: Scenario seeds to roll out.
        scheduler_key: Which policy generates the observation traces. The sweep
            is the default because its coverage is uniform, so the training
            distribution is not biased by the policy being learned against.
        stride: Slots between consecutive windows.
        max_windows_per_episode: Cap on windows drawn from one episode.

    Returns:
        The assembled :class:`PredictorDataset`.
    """
    from smartscan.agents import build_agent
    from smartscan.env.rf_environment import build_episode, generate_scenario
    from smartscan.runner import run_episode

    w = config.predictor.window_slots
    b = config.n_channels

    # This builds ONE dense float32 array, so cost is linear in episodes and
    # there is no streaming path: at 400 windows/episode with 4 planes over a
    # 128-channel, 128-slot window that is ~105 MB per episode. 128 episodes is
    # 13.4 GB, which does not fit a 16 GB box -- the run climbs until the OS
    # kills it or something else on the machine dies first. Say so up front
    # rather than after twenty minutes of window building.
    per_episode = max_windows_per_episode * 4 * b * w * 4
    projected = per_episode * len(list(seeds))
    available = _available_memory_bytes()
    if available and projected > 0.6 * available:
        safe = max(1, int(0.6 * available / max(per_episode, 1)))
        raise MemoryError(
            f"build_windows would allocate ~{projected / 1e9:.1f} GB for "
            f"{len(list(seeds))} episodes ({per_episode / 1e6:.0f} MB each) with only "
            f"~{available / 1e9:.1f} GB available. Use at most ~{safe} episodes here, "
            "or train on the GPU notebook, which streams the published corpus "
            "instead of materialising it."
        )

    xs, ys, ms, yts = [], [], [], []

    for seed in seeds:
        scenario = generate_scenario(seed, config=config)
        episode = build_episode(scenario)
        res = run_episode(
            config, seed, build_agent(scheduler_key, config, seed, scenario),
            scenario=scenario, episode=episode,
        )
        visit = res.visit_mask.astype(np.float32)
        hit = res.hit_mask.astype(np.float32)

        # SNR plane: the receiver's REPORTED estimate, not the true SNR.
        # Reading episode.snr_db here would put privileged information into the
        # student's *input*, which no PrivilegedAccess guard would catch because
        # it never opens one -- the model would simply be undeployable.
        snr_src = res.snr_plane() / 40.0

        # Staleness plane: slots since this channel was last visited, log-scaled.
        stale = np.zeros_like(visit)
        last = np.full(b, -1.0)
        for t in range(episode.n_slots):
            stale[:, t] = np.log1p(t - last) / np.log(episode.n_slots)
            last[visit[:, t] > 0] = t

        # Align window ends to slots the receiver actually observed. A window
        # whose label slot was never visited has an all-zero mask and yields no
        # gradient; at t_settle = 2 that would be two thirds of them.
        dwells = res.dwell_slots[(res.dwell_slots > w) & (res.dwell_slots < episode.n_slots - 1)]
        starts = (dwells[:: max(stride // 3, 1)] - 1)[:max_windows_per_episode]
        if starts.size == 0:
            continue
        with PrivilegedAccess("building distillation targets"):
            truth_next = episode.occupancy[:, starts + 1].T.astype(np.float32)

        for i, t0 in enumerate(starts):
            sl = slice(t0 - w, t0)
            xs.append(np.stack([visit[:, sl], hit[:, sl], snr_src[:, sl], stale[:, sl]]))
            # Observation-only label: what the receiver actually saw at t0+1,
            # valid only on channels it was tuned to.
            ys.append(hit[:, t0 + 1])
            ms.append(visit[:, t0 + 1] > 0)
            yts.append(truth_next[i])

    return PredictorDataset(
        np.asarray(xs, dtype=np.float32),
        np.asarray(ys, dtype=np.float32),
        np.asarray(ms, dtype=bool),
        np.asarray(yts, dtype=np.float32),
    )


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "torch is required for the predictor; install `pip install smartscan[ml]`."
        ) from exc
    return torch


_MODEL_CACHE: dict[str, Any] = {}


def _define_models() -> dict[str, type]:
    """Define the three architectures lazily so importing never requires torch."""
    torch = _require_torch()
    nn = torch.nn

    class GRUPredictor(nn.Module):
        """Per-channel GRU with a cross-channel mixing head."""

        def __init__(self, n_channels: int, hidden: int = 128, n_layers: int = 2, dropout: float = 0.1):
            super().__init__()
            self.n_channels = n_channels
            self.gru = nn.GRU(N_PLANES, hidden, n_layers, batch_first=True, dropout=dropout)
            # Weights are shared across channels; this conv lets them talk, which
            # is what a frequency-agile hop set needs.
            self.mix = nn.Conv1d(hidden, hidden, kernel_size=5, padding=2)
            self.head = nn.Conv1d(hidden, 1, kernel_size=1)

        def forward(self, x: Any) -> Any:
            """Args: x of shape ``(N, 4, B, W)``. Returns logits ``(N, B)``."""
            n, p, b, w = x.shape
            seq = x.permute(0, 2, 3, 1).reshape(n * b, w, p)
            out, _ = self.gru(seq)
            feat = out[:, -1].reshape(n, b, -1).permute(0, 2, 1)
            return self.head(torch.relu(self.mix(feat))).squeeze(1)

    class TCNPredictor(nn.Module):
        """Dilated temporal convolutions, weight-shared across channels."""

        def __init__(
            self, n_channels: int, hidden: int = 64, dilations: Sequence[int] = (1, 2, 4, 8, 16, 32, 64),
            dropout: float = 0.1,
        ):
            super().__init__()
            layers: list[nn.Module] = []
            in_c = N_PLANES
            for d in dilations:
                layers += [
                    nn.Conv2d(in_c, hidden, kernel_size=(1, 3), padding=(0, d), dilation=(1, d)),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
                in_c = hidden
            self.trunk = nn.Sequential(*layers)
            self.mix = nn.Conv2d(hidden, hidden, kernel_size=(5, 1), padding=(2, 0))
            self.head = nn.Conv2d(hidden, 1, kernel_size=1)

        def forward(self, x: Any) -> Any:
            """Args: x of shape ``(N, 4, B, W)``. Returns logits ``(N, B)``."""
            h = self.trunk(x)[..., -1:]
            h = torch.relu(self.mix(h))
            return self.head(h).squeeze(1).squeeze(-1)

    class TransformerPredictor(nn.Module):
        """Channel tokens with separate channel and time positional encodings."""

        def __init__(
            self, n_channels: int, hidden: int = 96, n_layers: int = 2, n_heads: int = 4,
            dropout: float = 0.1, window: int = 128,
        ):
            super().__init__()
            self.n_channels = n_channels
            # Time is summarised by a small dilated conv before attention, so the
            # attention operates over CHANNELS -- which is the axis no other
            # architecture here can model.
            self.time_encoder = nn.Sequential(
                nn.Conv2d(N_PLANES, hidden // 2, kernel_size=(1, 5), stride=(1, 4)),
                nn.ReLU(),
                nn.Conv2d(hidden // 2, hidden, kernel_size=(1, 5), stride=(1, 4)),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((n_channels, 1)),
            )
            self.channel_pos = nn.Parameter(torch.zeros(1, n_channels, hidden))
            nn.init.normal_(self.channel_pos, std=0.02)
            layer = nn.TransformerEncoderLayer(
                hidden, n_heads, dim_feedforward=hidden * 2, dropout=dropout,
                batch_first=True, norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, n_layers)
            self.head = nn.Linear(hidden, 1)

        def forward(self, x: Any) -> Any:
            """Args: x of shape ``(N, 4, B, W)``. Returns logits ``(N, B)``."""
            tok = self.time_encoder(x).squeeze(-1).permute(0, 2, 1) + self.channel_pos
            return self.head(self.encoder(tok)).squeeze(-1)

    return {"gru": GRUPredictor, "tcn": TCNPredictor, "transformer": TransformerPredictor}


def build_predictor(config: Config, arch: str | None = None) -> Any:
    """Instantiate the configured predictor architecture.

    Args:
        config: Resolved configuration.
        arch: Override ``config.predictor.arch``.

    Returns:
        A torch module.

    Raises:
        ValueError: If the architecture name is unknown.
    """
    if "models" not in _MODEL_CACHE:
        _MODEL_CACHE["models"] = _define_models()
    models = _MODEL_CACHE["models"]
    key = arch or config.predictor.arch
    if key not in models:
        raise ValueError(f"unknown predictor arch {key!r}; available: {sorted(models)}")
    pc = config.predictor
    if key == "gru":
        return models[key](config.n_channels, pc.hidden_dim, pc.n_layers, pc.dropout)
    if key == "tcn":
        return models[key](config.n_channels, max(pc.hidden_dim // 2, 32), pc.tcn_dilations, pc.dropout)
    return models[key](
        config.n_channels, pc.hidden_dim, pc.n_layers, pc.transformer_heads, pc.dropout, pc.window_slots
    )


def masked_focal_loss(
    logits: Any, targets: Any, mask: Any, gamma: float = 2.0, alpha: float = 0.25
) -> Any:
    """Focal binary cross-entropy, evaluated only where a label exists.

    ``FL = -alpha_t * (1 - p_t)**gamma * log(p_t)`` (Lin et al., 2017), averaged
    over masked entries. The ``gamma`` term down-weights easy negatives, which
    dominate a ~5 %-positive label set.

    **``alpha`` is an operating point, not a cure for imbalance.** It weights
    positives by ``alpha`` and negatives by ``1 - alpha``, so the shipped 0.25
    weights positives *down* by 3x -- the opposite of what the imbalance would
    suggest. Measured on the privileged teacher (transformer, 4 epochs, 8
    episodes), sweeping it moves the threshold and almost nothing else:

        alpha   0.25    0.50    0.75    0.90
        AUC    0.710   0.732   0.728   0.729
        recall 0.057   0.300   0.371   0.535
        prec   0.992   0.927   0.433   0.204

    Ranking quality is flat; only the precision/recall trade moves. That makes
    ``alpha`` irrelevant to :class:`SequencePredictorScheduler`, which takes an
    argmax over predicted occupancy and so depends only on the ordering. Tune it
    if you consume hard decisions; do not expect it to change scheduling.

    Args:
        logits: Raw predictions, shape ``(N, B)``.
        targets: Binary labels, shape ``(N, B)``.
        mask: Where a label exists, shape ``(N, B)``.
        gamma: Focusing parameter.
        alpha: Positive-class weight.

    Returns:
        Scalar loss.
    """
    torch = _require_torch()
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * targets + (1 - p) * (1 - targets)
    a_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = a_t * (1 - p_t).pow(gamma) * bce
    m = mask.float()
    return (loss * m).sum() / m.sum().clamp(min=1.0)


def save_predictor_checkpoint(model: Any, arch: str, path: str | Path) -> None:
    """Save weights together with the architecture that produced them.

    A bare ``state_dict`` does not say which network it came from, so loading
    one falls back to whatever ``config.predictor.arch`` happens to be. Train
    with ``--arch transformer`` against a config defaulting to ``gru`` and the
    checkpoint is simply unloadable -- which is exactly what happened.

    Args:
        model: Trained module.
        arch: Architecture key the weights belong to.
        path: Destination file.
    """
    torch = _require_torch()
    torch.save({"arch": arch, "state_dict": model.state_dict()}, Path(path))


def load_predictor_checkpoint(config: Config, path: str | Path, torch: Any = None) -> Any:
    """Rebuild a predictor from a checkpoint, honouring its recorded arch.

    Accepts both the tagged format written by :func:`save_predictor_checkpoint`
    and older bare ``state_dict`` files, which are assumed to match the config.

    Args:
        config: Resolved configuration.
        path: Checkpoint file.
        torch: Imported torch module, if the caller already has one.

    Returns:
        The loaded module, in eval-ready state.

    Raises:
        RuntimeError: If the weights do not fit the architecture named.
    """
    torch = torch or _require_torch()
    blob = torch.load(Path(path), map_location="cpu", weights_only=True)
    if isinstance(blob, dict) and "state_dict" in blob:
        arch, state = blob.get("arch"), blob["state_dict"]
    else:
        arch, state = None, blob
    model = build_predictor(config, arch)
    try:
        model.load_state_dict(state)
    except RuntimeError as exc:
        raise RuntimeError(
            f"{Path(path).name} does not fit arch {arch or config.predictor.arch!r}. "
            "Checkpoints written before architectures were recorded must be "
            "retrained, or loaded with a config whose predictor.arch matches."
        ) from exc
    return model


def _pick_device(torch: Any) -> Any:
    """Return the device to train on, verified by an actual kernel launch.

    ``torch.cuda.is_available()`` is not sufficient. It returns True for a GPU
    whose compute capability predates the installed build -- Kaggle handed out a
    Tesla P100 (sm_60) against a cu128 wheel, and the failure only appeared when
    a kernel was launched. So launch one.

    Args:
        torch: The imported torch module.

    Returns:
        A ``torch.device``.
    """
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        (torch.zeros(8, 8, device="cuda") @ torch.zeros(8, 8, device="cuda")).cpu()
        return torch.device("cuda")
    except Exception:  # a GPU that cannot run a matmul is not a GPU we can use
        return torch.device("cpu")



def _save_predictor_progress(
    state: Any, arch: str, path: str | Path | None, meta: dict[str, Any]
) -> None:
    """Write a mid-training predictor checkpoint atomically, or do nothing.

    Mirrors the RL trainers' behaviour. The write goes to a temporary file and is
    then replaced, so being killed *during* a save cannot leave a truncated
    checkpoint where a working one used to be -- the failure mode periodic
    checkpointing otherwise introduces.

    The sidecar marks the file as partial. Without it a half-trained predictor
    is indistinguishable on disk from a finished one, and the benchmark would
    score it as final.

    Args:
        state: State dict to write.
        arch: Architecture the weights belong to.
        path: Destination checkpoint; ``None`` disables.
        meta: Progress fields for the sidecar.
    """
    if path is None:
        return
    import json

    torch = _require_torch()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({"arch": arch, "state_dict": state}, tmp)
    tmp.replace(path)

    side = path.with_name(f"{path.stem}_progress.json")
    tmp_side = side.with_suffix(".tmp")
    tmp_side.write_text(json.dumps(meta, indent=2, default=float), encoding="utf-8")
    tmp_side.replace(side)



def train_predictor(
    config: Config,
    dataset: PredictorDataset | None = None,
    seeds: Sequence[int] | None = None,
    arch: str | None = None,
    verbose: bool = True,
    max_windows_per_episode: int = 400,
    loaders: tuple[Any, Any] | None = None,
    checkpoint_path: str | Path | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Train the occupancy predictor, optionally distilling from a teacher.

    Args:
        config: Resolved configuration.
        dataset: Pre-built windows; generated from ``seeds`` if omitted.
        seeds: Scenario seeds used to build the dataset.
        arch: Override ``config.predictor.arch``.
        verbose: Print per-epoch progress.
        max_windows_per_episode: Windows drawn from each episode. Memory is
            linear in this, and windows within one episode overlap heavily
            (stride 16), so lowering it buys episode diversity at the same
            cost -- 40 episodes x 200 is more varied than 25 x 400.
        loaders: Optional ``(train_loader, val_loader)`` yielding the same
            ``(x, y, mask, y_true)`` batches. Supplying these STREAMS the corpus
            instead of materialising it, which is the only way to train on more
            episodes than fit in RAM -- ``build_windows`` holds one dense array
            and caps out around 40 episodes on a 16 GB box, while the published
            corpus has 854 for MEDIUM alone.
        checkpoint_path: Save the best-so-far weights here after every epoch that
            improves. ``None`` disables. Training on the full corpus runs for
            hours and a 12 h Kaggle run has already been cancelled mid-epoch and
            lost everything, so a long run should not depend on reaching its
            last line.

    Returns:
        ``(student model, history dict)``. When distillation is enabled the
        history also carries the teacher's validation metrics.
    """
    torch = _require_torch()
    from smartscan.seeding import SeedTree

    torch.manual_seed(SeedTree(config.run.seed).torch_seed())
    if config.run.deterministic:
        torch.set_num_threads(int(config.run.torch_threads))

    device = _pick_device(torch)
    if verbose:
        print(f"  device: {device}", flush=True)

    pc = config.predictor
    streaming = loaders is not None
    if streaming:
        train_loader, val_loader = loaders
        train = val = None
        if verbose:
            print("  streaming windows from the published corpus", flush=True)
    else:
        if dataset is None:
            seeds = list(seeds or range(config.run.seed + 2000, config.run.seed + 2000 + 12))
            if verbose:
                print(f"  building windows from {len(seeds)} episodes...", flush=True)
            dataset = build_windows(
                config, seeds, max_windows_per_episode=max_windows_per_episode
            )
        train, val = dataset.split(0.8)
        if verbose:
            print(f"  train windows={len(train)} val windows={len(val)}", flush=True)

    def batches(which: str, shuffle: bool = True):
        """Yield ``(x, y, mask, y_true)`` from either source."""
        if streaming:
            for batch in (train_loader if which == "train" else val_loader):
                yield tuple(t.to(device) for t in batch)
            return
        ds = train if which == "train" else val
        idx = np.arange(len(ds))
        if shuffle:
            np.random.default_rng(config.run.seed).shuffle(idx)
        for s in range(0, len(idx), pc.batch_size):
            b = idx[s : s + pc.batch_size]
            yield (
                torch.as_tensor(ds.x[b]).to(device), torch.as_tensor(ds.y[b]).to(device),
                torch.as_tensor(ds.mask[b]).to(device), torch.as_tensor(ds.y_true[b]).to(device),
            )

    history: dict[str, Any] = {"arch": arch or pc.arch, "student_loss": [], "val_loss": []}

    # -- 1. privileged teacher (training time only) ------------------------ #
    teacher = None
    if pc.distillation.enabled:
        with PrivilegedAccess("teacher sees the full occupancy tensor"):
            teacher = build_predictor(config, arch).to(device)
            opt_t = torch.optim.Adam(teacher.parameters(), lr=pc.lr)
            full = torch.ones(1, dtype=torch.bool, device=device)
            for ep in range(pc.distillation.teacher_epochs):
                teacher.train()
                tot = n = 0.0
                for x, _y, _m, yt in batches("train"):
                    loss = masked_focal_loss(
                        teacher(x), yt, full.expand_as(yt), pc.focal_gamma, pc.focal_alpha
                    )
                    opt_t.zero_grad()
                    loss.backward()
                    opt_t.step()
                    tot += float(loss.detach())
                    n += 1
                if verbose:
                    print(f"  teacher epoch {ep + 1}/{pc.distillation.teacher_epochs} loss={tot / max(n, 1):.4f}", flush=True)
            teacher.eval()

    # -- 2. observation-only student --------------------------------------- #
    student = build_predictor(config, arch).to(device)
    opt = torch.optim.Adam(student.parameters(), lr=pc.lr)
    temp = pc.distillation.temperature

    # Keep the best-validating weights, not the last ones. On this data the
    # student's validation loss bottoms out within the first few epochs and then
    # climbs steadily, so returning the final epoch would ship the *most*
    # overfit model of the run -- and then score it against privileged truth,
    # reporting a number no deployment would ever see.
    #
    # Select on average precision, NOT on the loss. Occupancy is ~8 % positive,
    # so the masked focal loss is minimised by predicting "idle" everywhere:
    # selecting on it picked epoch 1 of a real run and shipped a model with
    # recall 0.000 and predicted_positive_rate 0.000 -- a constant "no" wearing
    # 0.91 accuracy, which is exactly the base rate. AP collapses to the base
    # rate for that model instead of rewarding it.
    #
    # The AP used for selection is computed against the OBSERVED labels under
    # their mask, never against privileged truth: the teacher may see the full
    # tensor during training, but choosing which epoch to ship with it would
    # leak privileged information into the delivered artefact.

    from smartscan.analysis.metrics import prediction_scores

    best_ap = -1.0
    best_val = float("inf")
    best_state = {k: v.detach().cpu().clone()
                  for k, v in student.state_dict().items()}
    best_epoch = 0

    for ep in range(pc.epochs):
        student.train()
        tot = n = 0.0
        for x, y, m, _yt in batches("train"):
            logits = student(x)
            loss = masked_focal_loss(logits, y, m, pc.focal_gamma, pc.focal_alpha)
            if teacher is not None and pc.distillation.lambda_kd > 0:
                with torch.no_grad():
                    soft = torch.sigmoid(teacher(x) / temp)
                # KL over ALL channels: the teacher supplies labels exactly where
                # the student has none. Training-time only.
                kd = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits / temp, soft, reduction="mean"
                )
                loss = loss + pc.distillation.lambda_kd * (temp**2) * kd
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            n += 1

        student.eval()
        obs_p: list[np.ndarray] = []
        obs_y: list[np.ndarray] = []
        with torch.no_grad():
            vl = vn = 0.0
            for x, y, m, _yt in batches("val", shuffle=False):
                logits = student(x)
                vl += float(masked_focal_loss(logits, y, m, pc.focal_gamma, pc.focal_alpha))
                vn += 1
                sel = m.cpu().numpy().astype(bool)
                if sel.any():
                    obs_p.append(torch.sigmoid(logits).cpu().numpy()[sel])
                    obs_y.append(y.cpu().numpy()[sel])
        val_loss = vl / max(vn, 1)
        val_ap = float("nan")
        if obs_p:
            val_ap = prediction_scores(
                np.concatenate(obs_y), np.concatenate(obs_p)
            )["average_precision"]
        history["student_loss"].append(tot / max(n, 1))
        history["val_loss"].append(val_loss)
        history.setdefault("val_ap", []).append(val_ap)
        improved = np.isfinite(val_ap) and val_ap > best_ap + 1e-6
        if improved:
            best_ap, best_val, best_epoch = val_ap, val_loss, ep + 1
            best_state = {k: v.detach().cpu().clone()
                          for k, v in student.state_dict().items()}
            # Persist the BEST state, not the current one -- it is what would be
            # shipped anyway, so an interrupted run leaves the same artefact a
            # completed one would have left at this point.
            _save_predictor_progress(
                best_state, arch or pc.arch, checkpoint_path,
                {"epoch": ep + 1, "epochs_planned": pc.epochs,
                 "best_val_ap": best_ap, "best_val_loss": best_val,
                 "complete": False},
            )
        if verbose:
            print(
                f"  student epoch {ep + 1}/{pc.epochs} train={tot / max(n, 1):.4f} "
                f"val={val_loss:.4f} ap={val_ap:.4f}{' *' if improved else ''}",
                flush=True,
            )
        if pc.patience and ep + 1 - best_epoch >= pc.patience:
            if verbose:
                print(
                    f"  early stop: no AP improvement in {pc.patience} epochs "
                    f"(best ap {best_ap:.4f} @ epoch {best_epoch})",
                    flush=True,
                )
            break

    student.load_state_dict(best_state)
    student.to(device)
    history["best_epoch"] = best_epoch
    history["best_val_loss"] = best_val
    history["best_val_ap"] = best_ap

    # -- 3. honest scoring against PRIVILEGED truth ------------------------ #
    from smartscan.analysis.metrics import prediction_scores

    student.eval()
    # Accumulate rather than index: a streamed corpus has no dense val.x to
    # forward in one shot, and materialising one would reintroduce the exact
    # memory ceiling streaming exists to avoid.
    probs_l, truth_l = [], []
    with torch.no_grad():
        for x, _y, _m, yt in batches("val", shuffle=False):
            probs_l.append(torch.sigmoid(student(x)).cpu().numpy())
            truth_l.append(yt.cpu().numpy())
    probs = np.concatenate(probs_l)
    truth = np.concatenate(truth_l)
    history["scores_vs_truth"] = prediction_scores(truth, probs)
    history["distilled"] = teacher is not None

    # Score the teacher on the same validation split. Without this the student's
    # number has no scale: a low AP could mean the task is hard, the observation
    # is too partial, or the training is broken, and only the privileged
    # upper bound separates them. The gap IS the result of the distillation
    # experiment, so it is reported rather than inferred.
    if teacher is not None:
        tp = []
        with torch.no_grad():
            for x, _y, _m, _yt in batches("val", shuffle=False):
                tp.append(torch.sigmoid(teacher(x)).cpu().numpy())
        history["teacher_scores_vs_truth"] = prediction_scores(truth, np.concatenate(tp))

    # Judge the model the way the scheduler uses it. `act` takes an argmax over
    # the predicted probabilities -- it RANKS channels and never applies a
    # threshold -- so "predicts no positives at 0.5" says nothing about whether
    # the model is usable, and an earlier version of this check wrongly
    # condemned a model with AP 0.322 against a 0.088 base rate on exactly that
    # basis. What would break the scheduler is an absence of *ordering*: AUC at
    # chance, AP at the base rate, or constant output making the argmax
    # arbitrary.
    scores = history["scores_vs_truth"]
    base = float(scores.get("positive_rate", float("nan")))
    ap = float(scores.get("average_precision", float("nan")))
    auc = float(scores.get("auc", float("nan")))
    lift = ap / base if base > 0 else float("nan")
    history["ap_lift_over_base_rate"] = lift
    history["degenerate"] = bool(
        np.isfinite(auc) and np.isfinite(lift) and (auc < 0.55 or lift < 1.2)
    )
    if history["degenerate"]:
        warnings.warn(
            f"predictor does not rank: AUC {auc:.3f}, AP {ap:.3f} against a "
            f"{base:.3f} base rate ({lift:.2f}x lift). The scheduler picks by "
            "argmax over these scores, so without an ordering it is choosing "
            "arbitrarily. Train on more episodes before using it.",
            RuntimeWarning,
            stacklevel=2,
        )
    elif scores.get("predicted_positive_rate", 0.0) <= 0.0:
        # Usable, but worth saying: the hard metrics reported at 0.5 are
        # meaningless for this model, and only the ranking metrics describe it.
        warnings.warn(
            f"predictor ranks well (AUC {auc:.3f}, AP {ap:.3f} vs {base:.3f} base "
            f"rate, {lift:.2f}x) but never crosses threshold 0.5, so precision, "
            "recall and F1 above are vacuous. This does not affect the scheduler, "
            "which ranks rather than thresholds; it does mean the probabilities "
            "are uncalibrated.",
            RuntimeWarning,
            stacklevel=2,
        )

    return student, history


# --------------------------------------------------------------------------- #
# Scheduler
# --------------------------------------------------------------------------- #
class SequencePredictorScheduler(Scheduler):
    """Tune to the window maximising predicted threat-weighted occupancy.

    Maintains the rolling ``(4, B, W)`` observation window itself, so it can run
    on live hardware with no access to anything but :class:`Observation`.

    Args:
        config: Resolved configuration.
        seed: Seed for tie-breaking.
        name: Optional display name.
        checkpoint: Path to trained weights.
        model: Pre-loaded model, bypassing the checkpoint.
    """

    key = "predictor"

    def __init__(
        self,
        config: Config,
        seed: int = 0,
        name: str | None = None,
        checkpoint: str | Path | None = None,
        model: Any = None,
    ) -> None:
        super().__init__(config, seed, name)
        self.torch = _require_torch()
        self.w = config.predictor.window_slots
        self.model = model
        self.coverage_weight = config.agents.coverage_weight
        self.retune_penalty = (
            config.receiver.t_settle_slots / (1.0 + config.receiver.t_settle_slots)
        ) * config.reward.w4_retune
        self._fallback: Scheduler | None = None

        path = Path(checkpoint) if checkpoint else (
            checkpoint_dir(config) / f"predictor_{config.scenario.difficulty}.pt"
        )
        if self.model is None:
            if path.is_file():
                self.model = load_predictor_checkpoint(config, path, self.torch)
                self.model.eval()
            else:
                from smartscan.agents.bandits import UCB1

                self._fallback = UCB1(config, seed)
                self.name = f"{self.name} (untrained -> ucb1 fallback)"
        self.reset()

    def reset(self) -> None:
        """Clear the rolling observation window."""
        super().reset()
        self.buffer = np.zeros((N_PLANES, self.n_channels, self.w), dtype=np.float32)
        self._t = 0
        if self._fallback is not None:
            self._fallback.reset()

    def observe(self, obs: Any) -> None:
        """Roll the observation window forward by one dwell."""
        self.buffer = np.roll(self.buffer, -1, axis=2)
        self.buffer[:, :, -1] = 0.0
        lo, hi = obs.window
        self.buffer[0, lo:hi, -1] = 1.0
        self.buffer[1, lo:hi, -1] = obs.hits.astype(np.float32)
        self.buffer[2, lo:hi, -1] = np.nan_to_num(obs.snr_est_db, nan=0.0) / 40.0
        self._t = obs.t

    def predict(self, belief: BeliefState) -> np.ndarray:
        """Return ``P(occupied at t+1)`` for every channel.

        Args:
            belief: Shared belief state, used for the staleness plane.

        Returns:
            Float64 probabilities of shape ``(B,)``.
        """
        self.buffer[3, :, -1] = np.log1p(belief.time_since_visit) / np.log(max(belief.n_slots, 2))
        with self.torch.no_grad():
            x = self.torch.as_tensor(self.buffer[None], dtype=self.torch.float32)
            return self.torch.sigmoid(self.model(x)).numpy().ravel().astype(np.float64)

    def act(self, belief: BeliefState, t: int) -> int:
        """Tune to the legal window with the highest predicted value."""
        if self._fallback is not None:
            action = self._fallback.act(belief, t)
            self.last_action = action
            return action
        p = self.predict(belief)
        # Threat proxy: down-weight channels the belief has learned look like
        # always-on interferers, and add the objective's own coverage term.
        value = p * (1.0 - 0.9 * belief.interferer_score())
        value = value + self.coverage_weight * (belief.time_since_visit / max(belief.n_slots, 1))
        action = self.argmax_legal(self.window_value(value), self.retune_penalty)
        self.last_action = action
        return action
