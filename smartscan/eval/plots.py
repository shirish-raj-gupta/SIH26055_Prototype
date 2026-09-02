"""Publication-quality figures for the results deck.

Matplotlib only (no seaborn), 300 dpi, one function per figure so each can be
regenerated in isolation. Every figure takes already-computed results rather
than re-running episodes, so a figure can never disagree with the table beside
it.

Colour is used for meaning, never decoration: ground truth is neutral grey, the
receiver's attention is blue, and intercepts are red.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from smartscan.analysis.metrics import HARD_CLASSES, kaplan_meier, time_to_first_intercept

if TYPE_CHECKING:
    from smartscan.config import Config
    from smartscan.runner import EpisodeResult

__all__ = [
    "FIGURE_DPI",
    "plot_detection_curves",
    "plot_intercept_heatmap",
    "plot_learning_curves",
    "plot_survival",
    "plot_waterfall",
    "save_all",
]

FIGURE_DPI: int = 300

#: Ground truth grey, attention blue, intercepts red.
_TRUTH = "0.75"
_ATTENTION = "#1f77b4"
_INTERCEPT = "#d62728"


def _plt() -> Any:
    """Import matplotlib with a non-interactive backend, or explain."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "matplotlib is required for figures; install `pip install smartscan[viz]`."
        ) from exc
    return plt


def _save(fig: Any, path: str | Path) -> Path:
    """Write a figure at publication resolution and close it."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=FIGURE_DPI, bbox_inches="tight")
    _plt().close(fig)
    return p


def plot_waterfall(
    results: dict[str, EpisodeResult],
    config: Config,
    path: str | Path = "reports/f1_waterfall.png",
) -> Path:
    """**F1.** Ground-truth waterfall with each scheduler's tuning trajectory.

    The figure that explains the project in one look: what was on the air, where
    the receiver was pointed, and which of the two coincided.

    Args:
        results: Mapping of scheduler key to its episode result, all on the same
            seed so the ground truth panel is identical.
        config: Resolved configuration.
        path: Output path.

    Returns:
        The path written.

    Raises:
        ValueError: If a result was run without retaining ground truth.
    """
    plt = _plt()
    keys = list(results)
    episode = results[keys[0]].episode
    if episode is None:
        raise ValueError("plot_waterfall needs results run with keep_episode=True")

    fig, axes = plt.subplots(len(keys), 1, figsize=(14, 3.0 * len(keys)), sharex=True, squeeze=False)
    truth = np.where(episode.occupancy > 0, 1.0, np.nan)
    extent = [0, config.time.episode_s, 0, config.n_channels]

    for ax, key in zip(axes[:, 0], keys, strict=True):
        res = results[key]
        ax.imshow(truth, aspect="auto", origin="lower", cmap="Greys", vmin=0, vmax=2.2,
                  extent=extent, interpolation="nearest")
        ax.plot(res.dwell_slots * episode.dt_s, res.actions, lw=0.45,
                color=_ATTENTION, alpha=0.85, label="tuned centre")
        ch, sl = np.nonzero(res.true_hit_mask)
        ax.scatter(sl * episode.dt_s, ch, s=4, color=_INTERCEPT, zorder=3, label="intercept")
        found = len({int(e) for e in episode.emitter_id[res.true_hit_mask] if e > 0})
        ax.set_ylabel("channel")
        ax.set_title(
            f"{key} — {found}/{len(episode.truth)} emitters found, {res.n_retunes} retunes, "
            f"{res.settle_slots_lost} slots lost to settling",
            loc="left", fontsize=10,
        )
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    axes[-1, 0].set_xlabel("time (s)")
    fig.suptitle(
        f"Ground truth (grey) vs receiver attention (blue) — {config.scenario.difficulty} tier, "
        f"seed {results[keys[0]].seed}",
        y=1.002,
    )
    return _save(fig, path)


def plot_survival(
    per_agent: dict[str, list[EpisodeResult]],
    path: str | Path = "reports/f2_survival.png",
    classes: frozenset[str] | None = HARD_CLASSES,
) -> Path:
    """**F2.** Kaplan-Meier time-to-first-intercept curves.

    Emitters never intercepted are right-censored, not dropped. The censored
    count is shown in the legend because it is the honest denominator.

    Args:
        per_agent: Mapping of scheduler key to its results across seeds.
        path: Output path.
        classes: Restrict to these emitter classes; ``None`` for all.

    Returns:
        The path written.
    """
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 5))
    for key, results in per_agent.items():
        durations: list[float] = []
        observed: list[bool] = []
        for res in results:
            if res.episode is None:
                continue
            d = time_to_first_intercept(res.episode, res.first_intercept, classes=classes)
            durations.extend(d["durations_s"])
            observed.extend(d["observed"])
        if not durations:
            continue
        curve = kaplan_meier(np.asarray(durations), np.asarray(observed))
        median = "inf" if not np.isfinite(curve.median) else f"{curve.median:.2f} s"
        ax.step(
            np.concatenate([[0.0], curve.times]),
            np.concatenate([[1.0], curve.survival]),
            where="post", lw=1.6,
            label=f"{key} — median {median}, {curve.n_censored} censored",
        )
    ax.set_xlabel("time since the emitter became interceptable (s)")
    ax.set_ylabel("fraction not yet intercepted")
    ax.set_title("Time to first intercept (Kaplan-Meier, censored observations retained)")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    return _save(fig, path)


def plot_detection_curves(
    result: EpisodeResult,
    config: Config,
    path: str | Path = "reports/f3_detection.png",
) -> Path:
    """**F3.** Measured Pd against SNR with exact binomial intervals, plus theory.

    Args:
        result: One episode result with ground truth retained.
        config: Resolved configuration.
        path: Output path.

    Returns:
        The path written.

    Raises:
        ValueError: If the result was run without retaining ground truth.
    """
    from smartscan.analysis.metrics import empirical_pd
    from smartscan.env.propagation import p_detect

    if result.episode is None:
        raise ValueError("plot_detection_curves needs a result with keep_episode=True")

    plt = _plt()
    curve = empirical_pd(result.episode, result.visit_mask, result.true_hit_mask)
    ok = curve["n"] > 30
    pfa = config.receiver.detector.pfa

    fig, ax = plt.subplots(figsize=(9, 5))
    if ok.any():
        ax.errorbar(
            curve["snr_centre"][ok], curve["pd"][ok],
            yerr=[curve["pd"][ok] - curve["lo"][ok], curve["hi"][ok] - curve["pd"][ok]],
            fmt="o", capsize=3, color=_ATTENTION, label="measured (95 % Clopper-Pearson)",
        )
    snr = np.linspace(-20, 40, 400)
    ax.plot(snr, p_detect(snr, n_integrate=1, pfa=pfa, swerling=1), "--", color="0.35",
            label="analytic: 1 pulse, Swerling I")
    ax.plot(snr, p_detect(snr, n_integrate=1, pfa=pfa, swerling=0), ":", color="0.55",
            label="analytic: 1 pulse, Swerling 0")
    ax.set_xlabel("true SNR (dB)")
    ax.set_ylabel("probability of detection")
    ax.set_title(f"Detection performance, Pfa = {pfa:.0e}")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    return _save(fig, path)


def plot_learning_curves(
    logs: dict[str, Any],
    path: str | Path = "reports/f4_learning.png",
) -> Path:
    """**F4.** RL learning curves with policy entropy alongside return.

    Entropy is plotted because return alone cannot distinguish "learned a good
    policy" from "has not learned to decide anything yet".

    Args:
        logs: Mapping of run label to a ``TrainLog`` (or its ``as_dict``).
        path: Output path.

    Returns:
        The path written.
    """
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for label, log in logs.items():
        d = log if isinstance(log, dict) else log.as_dict()
        if d.get("steps"):
            axes[0].plot(d["steps"], d["returns"], marker="o", ms=3, label=label)
            if d.get("entropy"):
                axes[1].plot(d["steps"], d["entropy"], marker="o", ms=3, label=label)
    axes[0].set_xlabel("environment steps")
    axes[0].set_ylabel("episode return")
    axes[0].set_title("Return")
    axes[1].set_xlabel("environment steps")
    axes[1].set_ylabel("policy entropy (nats)")
    axes[1].set_title("Entropy — is the policy deciding anything?")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    return _save(fig, path)


def plot_intercept_heatmap(
    per_agent: dict[str, list[EpisodeResult]],
    path: str | Path = "reports/f5_class_heatmap.png",
) -> Path:
    """**F5.** Intercept rate per emitter class, per scheduler.

    Shows *where* an advantage comes from, which an aggregate number hides.

    Args:
        per_agent: Mapping of scheduler key to results across seeds.
        path: Output path.

    Returns:
        The path written.
    """
    from smartscan.analysis.metrics import average_intercept_rate

    plt = _plt()
    table: dict[str, dict[str, float]] = {}
    classes: set[str] = set()
    for key, results in per_agent.items():
        acc: dict[str, list[float]] = {}
        for res in results:
            if res.episode is None:
                continue
            for cls, rate in average_intercept_rate(res.episode, res.true_hit_mask).items():
                if cls != "overall":
                    acc.setdefault(cls, []).append(rate)
                    classes.add(cls)
        table[key] = {c: float(np.mean(v)) for c, v in acc.items()}

    ordered = sorted(classes)
    keys = list(table)
    mat = np.array([[table[k].get(c, 0.0) for c in ordered] for k in keys])

    fig, ax = plt.subplots(figsize=(1.35 * len(ordered) + 3.5, 0.55 * len(keys) + 2.2))
    im = ax.imshow(mat, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(ordered)), ordered, rotation=35, ha="right")
    ax.set_yticks(range(len(keys)), keys)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center", fontsize=7,
                    color="white" if mat[i, j] > mat.max() * 0.55 else "black")
    ax.set_title("Intercepts per second, by emitter class")
    fig.colorbar(im, ax=ax, label="intercepts / s")
    return _save(fig, path)


def save_all(
    per_agent: dict[str, list[EpisodeResult]],
    config: Config,
    out_dir: str | Path = "reports",
    logs: dict[str, Any] | None = None,
) -> list[Path]:
    """Regenerate every figure the harness owns.

    Args:
        per_agent: Mapping of scheduler key to results across seeds.
        config: Resolved configuration.
        out_dir: Directory for the figures.
        logs: Optional RL training logs for F4.

    Returns:
        The paths written.
    """
    out = Path(out_dir)
    first = {k: v[0] for k, v in per_agent.items() if v}
    written = [
        plot_waterfall(first, config, out / "f1_waterfall.png"),
        plot_survival(per_agent, out / "f2_survival.png"),
        plot_intercept_heatmap(per_agent, out / "f5_class_heatmap.png"),
    ]
    sample = next(iter(first.values()), None)
    if sample is not None:
        written.append(plot_detection_curves(sample, config, out / "f3_detection.png"))
    if logs:
        written.append(plot_learning_curves(logs, out / "f4_learning.png"))
    return written
