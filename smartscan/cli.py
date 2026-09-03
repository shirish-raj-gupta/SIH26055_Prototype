"""SmartScan command line.

    smartscan run       --config configs/medium.yaml --agent sequential
    smartscan benchmark --config configs/medium.yaml
    smartscan train     --config configs/medium.yaml --what ppo
    smartscan estimate  --config configs/scan_on_scan.yaml
    smartscan ablate    --config configs/medium.yaml
    smartscan reproduce

Every command writes a JSON artefact carrying the resolved config hash, so no
number is ever orphaned from the settings that produced it.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import typer

from smartscan.config import Config, load_config

app = typer.Typer(add_completion=False, help="Smart Scan Strategy for Electronic Warfare (SIH 26055)")

_CONFIG = typer.Option("configs/medium.yaml", "--config", "-c", help="Config file or bare name.")
_SET = typer.Option(None, "--set", "-s", help="Override, repeatable: --set run.seed=7")


def _resolve(config: str, overrides: list[str] | None) -> Config:
    """Load a config with dotted ``--set`` overrides applied."""
    parsed: dict[str, Any] = {}
    for item in overrides or []:
        if "=" not in item:
            raise typer.BadParameter(f"--set expects key=value, got {item!r}")
        key, value = item.split("=", 1)
        parsed[key.strip()] = value.strip()
    return load_config(config, parsed)


def _out_dir(cfg: Config) -> Path:
    """Create and return the run output directory."""
    d = Path(cfg.run.out_dir) / cfg.run.name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _finite_median(rows: list[dict[str, Any]]):
    """Return a lookup that medians one metric across rows, ignoring non-finite values.

    Args:
        rows: Per-seed metric dicts.

    Returns:
        A callable ``key -> median``. Defined as a factory rather than a closure
        over a loop variable so the binding is explicit.
    """

    def median(key: str) -> float:
        vals = np.asarray([r.get(key, np.nan) for r in rows], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        return float(np.median(vals)) if vals.size else float("nan")

    return median


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    """Write a JSON artefact, converting numpy scalars."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    return path


@app.command()
def run(
    config: str = _CONFIG,
    agent: str = typer.Option("sequential", "--agent", "-a", help="Scheduler key."),
    seed: int | None = typer.Option(None, help="Scenario seed; defaults to run.seed."),
    n_seeds: int = typer.Option(1, help="Number of consecutive seeds to average over."),
    out: str | None = typer.Option(None, help="Metrics JSON path."),
    set_: list[str] = _SET,
) -> None:
    """Run one scheduler and write a metrics JSON (acceptance test 2)."""
    from smartscan.agents import build_agent
    from smartscan.analysis.metrics import evaluate_episode
    from smartscan.env.rf_environment import build_episode, generate_scenario
    from smartscan.runner import run_episode

    cfg = _resolve(config, set_)
    base = int(seed if seed is not None else cfg.run.seed)
    rows = []
    t0 = time.perf_counter()
    for i in range(max(n_seeds, 1)):
        s = base + i
        scenario = generate_scenario(s, config=cfg)
        episode = build_episode(scenario)
        result = run_episode(
            cfg, s, build_agent(agent, cfg, s, scenario), scenario=scenario, episode=episode
        )
        row = evaluate_episode(result, cfg)
        row.pop("_detail", None)
        rows.append(row)
        typer.echo(
            f"  seed {s}: TTFI_hard={row['ttfi_hard_median_s']:.3f}s "
            f"TWIR={row['twir_rate']:.4f} coverage={row['coverage']:.3f} "
            f"reward={row['reward_total']:.1f}"
        )

    # Guarded: a metric can be all-NaN when, say, no emitter of a class was
    # interceptable in any seed. That is information, not an error.
    _median = _finite_median(rows)
    summary = {
        k: _median(k) for k in rows[0] if isinstance(rows[0][k], (int, float, np.floating))
    }
    path = Path(out) if out else _out_dir(cfg) / f"metrics_{agent}.json"
    _write_json(path, {
        "agent": agent, "tier": cfg.scenario.difficulty, "config_hash": cfg.hash(),
        "n_seeds": len(rows), "summary": summary, "rows": rows,
        "wall_time_s": time.perf_counter() - t0,
    })
    typer.secho(f"wrote {path}", fg=typer.colors.GREEN)


@app.command()
def benchmark(
    config: str = _CONFIG,
    agents: str | None = typer.Option(None, help="Comma-separated agent keys."),
    n_seeds: int | None = typer.Option(None, help="Override run.n_seeds."),
    n_jobs: int = typer.Option(1, help="Parallel workers (joblib)."),
    out: str = typer.Option("reports", help="Output directory."),
    set_: list[str] = _SET,
) -> None:
    """Run the paired benchmark and write metrics, leaderboard and comparisons."""
    from smartscan.eval.benchmark import leaderboard_markdown, run_benchmark

    cfg = _resolve(config, set_)
    if n_seeds is not None:
        cfg = cfg.with_overrides(run={"n_seeds": n_seeds})
    keys = [a.strip() for a in agents.split(",")] if agents else None

    result = run_benchmark(cfg, agents=keys, n_jobs=n_jobs)
    out_dir = Path(out)
    result.to_json(out_dir / f"metrics_{cfg.scenario.difficulty}.json")
    md = leaderboard_markdown(result)
    (out_dir / f"leaderboard_{cfg.scenario.difficulty}.md").write_text(md, encoding="utf-8")
    typer.echo(md)
    typer.secho(f"wrote {out_dir}/", fg=typer.colors.GREEN)


@app.command()
def grid(
    tiers: str = typer.Option("easy,medium,hard", help="Comma-separated tiers."),
    agents: str | None = typer.Option(None, help="Comma-separated agent keys."),
    n_seeds: int = typer.Option(30, help="Seeds per tier."),
    n_jobs: int = typer.Option(1, help="Parallel workers (joblib)."),
    out: str = typer.Option("reports", help="Output directory."),
    figures: bool = typer.Option(True, help="Also regenerate the figures."),
) -> None:
    """Run the full {schedulers} x {tiers} x {seeds} grid and write every artefact.

    Emits ``results.parquet`` (tidy), ``leaderboard.md``, ``leaderboard.tex``
    and figures F1-F7 into the output directory, with no manual steps.
    """
    from smartscan.eval.benchmark import run_grid

    t0 = time.perf_counter()
    keys = [a.strip() for a in agents.split(",")] if agents else None
    results = run_grid(
        tiers=[t.strip() for t in tiers.split(",")], agents=keys,
        n_seeds=n_seeds, n_jobs=n_jobs, out_dir=out, figures=figures,
    )
    written = sorted(p.name for p in Path(out).iterdir() if p.is_file())
    typer.echo("")
    for name in written:
        typer.echo(f"  {name}")
    typer.secho(
        f"grid complete: {len(results)} tiers in {time.perf_counter() - t0:.1f}s -> {out}/",
        fg=typer.colors.GREEN,
    )


@app.command()
def train(
    config: str = _CONFIG,
    what: str = typer.Option("ppo", "--what", "-w", help="ppo | dqn | predictor | hybrid"),
    steps: int | None = typer.Option(None, help="Override total steps / epochs."),
    arch: str | None = typer.Option(None, help="Predictor architecture override."),
    episodes: int = typer.Option(
        16, "--episodes", help="Training episodes (predictor); more is usually better."
    ),
    windows_per_episode: int = typer.Option(
        400, "--windows-per-episode",
        help="Windows per episode (predictor). Memory is linear in this; "
             "lowering it buys more episodes for the same RAM.",
    ),
    dataset: str | None = typer.Option(
        None, "--dataset",
        help="Train the predictor by STREAMING the published corpus at this "
             "path (e.g. build/dataset) instead of regenerating ~40 episodes "
             "from seeds. Removes the RAM ceiling entirely.",
    ),
    set_: list[str] = _SET,
) -> None:
    """Train a learned scheduler and save its checkpoint."""
    cfg = _resolve(config, set_)
    ckpt_dir = Path(cfg.run.out_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tier = cfg.scenario.difficulty
    t0 = time.perf_counter()

    # Training seeds are disjoint from evaluation seeds by construction.
    # Disjoint from evaluation seeds by construction. The count is a knob
    # because 16 is not enough for the predictor: on 16 episodes the privileged
    # teacher reaches AP 0.41 while the observation-only student collapses to
    # predicting no positives at all.
    train_seeds = list(range(cfg.run.seed + 1000, cfg.run.seed + 1000 + max(episodes, 1)))

    if what in {"ppo", "dqn", "hybrid"}:
        from smartscan.agents.rl_agents import train_dqn, train_ppo

        trainer = train_ppo if what != "dqn" else train_dqn
        path = ckpt_dir / f"{what}_{tier}.pt"
        extra = {"hybrid": True} if what == "hybrid" else {}
        # Pass the destination in so the trainer checkpoints as it goes. These
        # runs take hours and have been killed mid-flight; without this the
        # whole run is lost rather than the last slice of it.
        net, log = trainer(cfg, train_seeds, total_steps=steps, checkpoint_path=path, **extra)
        import torch

        torch.save(net.state_dict(), path)
        _write_json(ckpt_dir / f"{what}_{tier}_trainlog.json", log.as_dict())
        # The progress sidecar only exists to describe a partial run.
        (ckpt_dir / f"{what}_{tier}_progress.json").unlink(missing_ok=True)
    elif what == "predictor":
        import torch

        from smartscan.agents.predictors import save_predictor_checkpoint, train_predictor

        if steps:
            # --steps must bound the WHOLE job, not just the student. The
            # privileged teacher trains first, for its own epoch count, so
            # leaving it untouched makes `--steps 12` a ~100-minute "smoke run"
            # on CPU. Cap the teacher by the same budget.
            cfg = cfg.with_overrides(
                predictor={
                    "epochs": steps,
                    "distillation": {
                        **cfg.predictor.distillation.model_dump(),
                        "teacher_epochs": min(
                            cfg.predictor.distillation.teacher_epochs, steps
                        ),
                    },
                }
            )
        loaders = None
        if dataset:
            from smartscan.data.kaggle_io import OccupancyWindowDataset, load_dataset

            tr = load_dataset("train", tier=tier, root=dataset, allow_download=False)
            va = load_dataset("val", tier=tier, root=dataset, allow_download=False)
            if tr.source == "regenerated":
                raise typer.BadParameter(
                    f"no usable corpus at {dataset!r}: it fell back to regenerating "
                    "from seeds, which is not what --dataset asked for."
                )
            typer.echo(f"  corpus: {len(tr)} train / {len(va)} val episodes ({tr.source})")
            overlap = set(tr.episode_ids()) & set(va.episode_ids())
            if overlap:
                raise typer.BadParameter(f"LEAKAGE: {len(overlap)} episodes in both splits")
            tw = OccupancyWindowDataset(
                tr, window=cfg.predictor.window_slots, stride=16,
                agent="sequential", max_windows_per_episode=windows_per_episode,
                class_balanced=True,
            )
            vw = OccupancyWindowDataset(
                va, window=cfg.predictor.window_slots, stride=64,
                agent="sequential", max_windows_per_episode=64,
            )
            typer.echo(f"  windows: {len(tw)} train / {len(vw)} val (streamed)")
            loaders = (
                tw.loader(batch_size=cfg.predictor.batch_size, seed=cfg.run.seed),
                vw.loader(batch_size=cfg.predictor.batch_size),
            )

        path = ckpt_dir / f"predictor_{tier}.pt"
        model, history = train_predictor(
            cfg, seeds=train_seeds, arch=arch,
            max_windows_per_episode=windows_per_episode,
            loaders=loaders, checkpoint_path=path,
        )
        save_predictor_checkpoint(model, history["arch"], path)
        # The sidecar only ever describes a partial run.
        (ckpt_dir / f"predictor_{tier}_progress.json").unlink(missing_ok=True)
        _write_json(ckpt_dir / f"predictor_{tier}_history.json", history)
        def _fmt(scores: dict[str, float]) -> str:
            return (
                f"ap={scores['average_precision']:.4f} auc={scores['auc']:.4f} "
                f"brier={scores['brier']:.4f} "
                f"pred_pos={scores['predicted_positive_rate']:.4f}"
            )

        base = history["scores_vs_truth"]["positive_rate"]
        typer.echo(f"  best epoch {history.get('best_epoch')} "
                   f"(val {history.get('best_val_loss', float('nan')):.4f})")
        typer.echo(f"  positive base rate: {base:.4f}  <- AP at this value is no skill")
        if "teacher_scores_vs_truth" in history:
            typer.echo(f"  teacher (privileged): {_fmt(history['teacher_scores_vs_truth'])}")
        typer.echo(f"  student (obs-only)  : {_fmt(history['scores_vs_truth'])}")
    else:
        raise typer.BadParameter(f"unknown --what {what!r}; use ppo, dqn, predictor or hybrid")

    typer.secho(f"saved {path} in {time.perf_counter() - t0:.1f}s", fg=typer.colors.GREEN)


@app.command()
def estimate(
    config: str = typer.Option("configs/scan_on_scan.yaml", "--config", "-c"),
    n_seeds: int = typer.Option(10, help="Number of scenarios."),
    out: str = typer.Option("reports/scan_on_scan.json"),
    set_: list[str] = _SET,
) -> None:
    """Validate the scan-period estimators against ground truth (acceptance test 4)."""
    from smartscan.eval.scan_validation import validate_estimators

    cfg = _resolve(config, set_)
    report = validate_estimators(cfg, n_seeds=n_seeds)
    _write_json(Path(out), report)
    typer.echo(json.dumps(report["summary"], indent=2, default=float))
    typer.secho(f"wrote {out}", fg=typer.colors.GREEN)


@app.command()
def ablate(
    config: str = _CONFIG,
    which: str = typer.Option("all", help="all | reward | ibw | retune | density | belief"),
    n_seeds: int = typer.Option(8),
    out: str = typer.Option("reports/ablation.json"),
    set_: list[str] = _SET,
) -> None:
    """Run the ablation sweeps and write a sensitivity table."""
    from smartscan.eval.ablation import run_ablations

    cfg = _resolve(config, set_)
    report = run_ablations(cfg, which=which, n_seeds=n_seeds)
    _write_json(Path(out), report)
    typer.secho(f"wrote {out}", fg=typer.colors.GREEN)


@app.command()
def demo(
    config: str = typer.Option("configs/easy.yaml", "--config", "-c"),
    n_seeds: int = typer.Option(5),
) -> None:
    """Fast end-to-end smoke run: environment, schedulers, scan-on-scan, metrics."""
    from smartscan.agents import build_agent
    from smartscan.analysis.metrics import evaluate_episode, sensitivity_db
    from smartscan.analysis.scan_on_scan import analyse_coincidence, beam_dwell_s
    from smartscan.env.rf_environment import build_episode, generate_scenario
    from smartscan.runner import run_episode

    cfg = _resolve(config, None).with_overrides(run={"n_seeds": n_seeds})
    typer.secho(f"SmartScan demo - tier {cfg.scenario.difficulty}, config {cfg.hash()[:12]}", bold=True)

    sens = sensitivity_db(cfg)
    typer.echo(
        f"\nreceiver sensitivity (Pd>=0.9 @ Pfa=1e-3): "
        f"pulse {sens['pulse_single_db']:.1f} dB, energy {sens['energy_db']:.1f} dB "
        f"(N={sens['n_integrate_energy']:.0f})"
    )

    agents = ["sequential", "ucb1", "whittle", "phase_locked"]
    typer.echo(f"\n{'agent':16}{'TTFI_hard':>11}{'TWIR':>9}{'coverage':>10}{'reward':>9}")
    for key in agents:
        rows = []
        for i in range(n_seeds):
            s = cfg.run.seed + i
            sc = generate_scenario(s, config=cfg)
            ep = build_episode(sc)
            rows.append(evaluate_episode(
                run_episode(cfg, s, build_agent(key, cfg, s, sc), scenario=sc, episode=ep), cfg
            ))
        med = _finite_median(rows)
        typer.echo(
            f"{key:16}{med('ttfi_hard_median_s'):11.3f}{med('twir_rate'):9.4f}"
            f"{med('coverage'):10.3f}{med('reward_total'):9.1f}"
        )

    te, wr = 4.0, cfg.time.dt_s
    we = beam_dwell_s(2.0, te)
    typer.echo("\nscan-on-scan: a 4.0 s / 2 deg scanner vs three receiver sweep periods")
    for tr in (0.096, 1.0, 2.0):
        r = analyse_coincidence(tr, te, wr, we, horizon_s=120.0)
        typer.echo(
            f"  Tr={tr:6.3f}s ratio={r.ratio:7.4f}={r.rational[0]}/{r.rational[1]:<3d} "
            f"blind={r.blind_fraction * 100:5.1f}%  closed-form TTI={r.closed_form_tti_s:7.2f}s"
        )
    typer.secho("\ndemo complete", fg=typer.colors.GREEN)


@app.command()
def reproduce(
    tiers: str = typer.Option("easy,medium", help="Comma-separated tiers."),
    n_seeds: int = typer.Option(30),
    n_jobs: int = typer.Option(1),
    out: str = typer.Option("reports"),
) -> None:
    """Regenerate every headline number from scratch."""
    from smartscan.eval.benchmark import run_grid

    t0 = time.perf_counter()
    run_grid(tiers=[t.strip() for t in tiers.split(",")], n_seeds=n_seeds, n_jobs=n_jobs, out_dir=out)
    typer.secho(f"reproduce complete in {time.perf_counter() - t0:.1f}s -> {out}/", fg=typer.colors.GREEN)


@app.command()
def data(
    action: str = typer.Argument("status", help="status | build | verify | card"),
    root: str = typer.Option("build/dataset", help="Dataset root."),
    split: str = typer.Option("all", help="Split to summarise."),
    tier: str | None = typer.Option(None, help="Restrict to one tier."),
    n_episodes: int | None = typer.Option(None, help="Cap episodes (smoke runs)."),
    n_jobs: int = typer.Option(1, help="Parallel workers for `build`."),
) -> None:
    """Inspect, build or verify the RF environment dataset."""
    from smartscan.data.kaggle_io import load_dataset, resolve_dataset_root, verify_dataset

    if action == "build":
        from smartscan.data.dataset_builder import build_dataset

        counts = None if n_episodes is None else {
            "easy": n_episodes, "medium": n_episodes, "hard": n_episodes
        }
        build_dataset(root, counts=counts, n_jobs=n_jobs)
        return

    if action == "verify":
        typer.echo(json.dumps(verify_dataset(root), indent=2, default=str))
        return

    if action == "card":
        card = Path(root) / "dataset_card.md"
        typer.echo(card.read_text(encoding="utf-8") if card.is_file() else f"no card at {card}")
        return

    if action != "status":
        raise typer.BadParameter(f"unknown action {action!r}; use status, build, verify or card")

    found, source = resolve_dataset_root(root if Path(root).exists() else None, allow_download=False)
    typer.echo(f"resolved     {found or '(none on disk)'}")
    typer.echo(f"source       {source}")
    ds = load_dataset(split, tier=tier, root=found, allow_download=False, n_episodes=n_episodes)
    typer.echo(f"episodes     {len(ds)}  (split={split}, tier={tier or 'all'})")
    if len(ds) and "tier" in ds.index:
        typer.echo(f"per tier     {ds.index['tier'].value_counts().to_dict()}")
        if "split" in ds.index:
            typer.echo(f"per split    {ds.index['split'].value_counts().to_dict()}")
    if found is not None:
        total = sum(f.stat().st_size for f in Path(found).rglob("*") if f.is_file())
        typer.echo(f"size         {total / 1024**3:.3f} GB")
    else:
        typer.echo("note         no local dataset; episodes regenerate from seeds on demand")


@app.command()
def external(
    config: str = _CONFIG,
    subset: str = typer.Option("archive", help="TSRD subset: archive | scan | stare."),
    split: str = typer.Option("test", help="TSRD split."),
    n_records: int = typer.Option(4, help="Pulse trains to evaluate."),
    out: str = typer.Option("reports/external_tsrd.json"),
    set_: list[str] = _SET,
) -> None:
    """Validate schedulers against the gated Turing Synthetic Radar Dataset.

    Reported SEPARATELY from synthetic benchmarks: the PDW-to-occupancy binning
    is an assumption of the bridge, and TSRD carries no threat model.
    """
    from smartscan.data.tsrd_bridge import external_validation_report

    cfg = _resolve(config, set_)
    report = external_validation_report(cfg, split=split, subset=subset, max_records=n_records)
    _write_json(Path(out), report)

    typer.echo(f"available   {report['available']}")
    typer.echo(f"source      {report['source']}")
    typer.echo(f"licence     {report.get('licence', '-')}")
    if not report["available"]:
        typer.echo("")
        typer.echo(report["reason"])
        return

    rows = report["rows"]
    agents = sorted({r["agent"] for r in rows})
    typer.echo(f"records     {report['n_records']}")
    typer.echo("")
    typer.echo(f"{'agent':16}{'TWIR':>9}{'coverage':>10}{'intercept/s':>13}{'staleMax':>10}")
    for a in agents:
        med = _finite_median([r for r in rows if r["agent"] == a])
        typer.echo(
            f"{a:16}{med('twir_rate'):9.4f}{med('coverage'):10.3f}"
            f"{med('intercept_rate_per_s'):13.1f}{med('staleness_max_s'):10.3f}"
        )
    typer.secho(f"wrote {out}", fg=typer.colors.GREEN)


@app.command()
def credentials(dotenv: str = typer.Option(".env", help="Path to a .env file.")) -> None:
    """Report which credentials are configured. Never prints a secret value."""
    from smartscan.credentials import credential_status

    typer.echo(credential_status(dotenv).report())


@app.command()
def info(config: str = _CONFIG, set_: list[str] = _SET) -> None:
    """Print the resolved configuration summary and its hash."""
    cfg = _resolve(config, set_)
    grid = cfg.grid()
    typer.echo(f"config hash    : {cfg.hash()}")
    typer.echo(f"tier           : {cfg.scenario.difficulty}  ({cfg.scenario.n_emitters} emitters)")
    typer.echo(
        f"spectrum       : {grid.f_start_hz / 1e9:.2f}-{grid.f_stop_hz / 1e9:.2f} GHz, "
        f"B={grid.n_channels} x {grid.widths_hz[0] / 1e6:.1f} MHz ({cfg.spectrum.partition})"
    )
    typer.echo(
        f"receiver       : K={cfg.receiver.ibw_channels} channels "
        f"({cfg.receiver.ibw_channels / grid.n_channels:.3f} of band), "
        f"t_settle={cfg.receiver.t_settle_slots} slots"
    )
    typer.echo(f"time           : dt={cfg.time.dt_s * 1e3:.1f} ms, T={cfg.n_slots} slots ({cfg.time.episode_s} s)")
    typer.echo(f"detector       : Pfa={cfg.receiver.detector.pfa:.1e}, Swerling {cfg.receiver.detector.swerling}")
    typer.echo(f"mix            : {cfg.scenario.mix}")
    typer.echo(f"eval agents    : {cfg.eval.agents}")


if __name__ == "__main__":
    app()
