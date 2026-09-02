"""Scheduler registry.

Adding a scheduler is a one-line entry here plus the class. The CLI, benchmark
harness and dashboard all resolve agents through :func:`build_agent`, so nothing
else needs to know the set of policies.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from smartscan.agents.base import Scheduler
from smartscan.config import Config

__all__ = ["AGENT_KEYS", "build_agent", "register"]

_REGISTRY: dict[str, Callable[..., Scheduler]] = {}


def register(key: str, factory: Callable[..., Scheduler]) -> None:
    """Register a scheduler factory under ``key``.

    Args:
        key: Short identifier used in configs and on the CLI.
        factory: Callable ``(config, seed, scenario) -> Scheduler``.
    """
    _REGISTRY[key] = factory


def _lazy(module: str, cls: str) -> Callable[..., Scheduler]:
    """Build a factory that imports its module only when first used.

    Torch-backed agents cost about a second to import, which would otherwise be
    paid by every CLI invocation including ``--help``.
    """

    def factory(config: Config, seed: int = 0, scenario: Any = None) -> Scheduler:
        import importlib

        target = getattr(importlib.import_module(module), cls)
        if cls == "PriorityRoundRobin":
            truth = (
                [t.home_channel for t in scenario.emitters] if scenario is not None else None
            )
            return target(config, seed, truth_channels=truth)
        return target(config, seed)

    return factory


register("sequential", _lazy("smartscan.agents.baselines", "SequentialSweep"))
register("random", _lazy("smartscan.agents.baselines", "RandomScan"))
register("priority_rr", _lazy("smartscan.agents.baselines", "PriorityRoundRobin"))
register("epsilon_greedy", _lazy("smartscan.agents.bandits", "EpsilonGreedy"))
register("ucb1", _lazy("smartscan.agents.bandits", "UCB1"))
register("thompson", _lazy("smartscan.agents.bandits", "ThompsonSampling"))
register("whittle", _lazy("smartscan.agents.whittle", "WhittleIndexScheduler"))
register("coprime_sweep", _lazy("smartscan.analysis.scan_on_scan", "CoprimeSweepScheduler"))
register("phase_locked", _lazy("smartscan.analysis.scan_on_scan", "PhaseLockedScheduler"))
register("predictor", _lazy("smartscan.agents.predictors", "SequencePredictorScheduler"))
register("dqn", _lazy("smartscan.agents.rl_agents", "DQNScheduler"))
register("ppo", _lazy("smartscan.agents.rl_agents", "PPOScheduler"))
register("hybrid", _lazy("smartscan.agents.hybrid", "HybridScheduler"))

#: All registered scheduler keys, in a stable display order.
AGENT_KEYS: tuple[str, ...] = tuple(_REGISTRY)


def build_agent(key: str, config: Config, seed: int = 0, scenario: Any = None) -> Scheduler:
    """Instantiate a scheduler by key.

    Args:
        key: Registered scheduler key.
        config: Resolved configuration.
        seed: Seed for policy-internal randomness.
        scenario: Scenario, passed to policies that need setup-time context
            (``priority_rr`` builds its deliberately-wrong prior from it).

    Returns:
        The constructed scheduler.

    Raises:
        KeyError: If ``key`` is not registered.
    """
    if key not in _REGISTRY:
        raise KeyError(f"unknown agent {key!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[key](config, seed, scenario)
