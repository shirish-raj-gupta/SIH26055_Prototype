"""Gymnasium-compatible environment wrapper.

The RL agents in this package train through :class:`SmartScanEnv`, a plain
``reset``/``step`` object with **no hard dependency on Gymnasium**. That is a
deliberate choice (``docs/architecture.md`` §11.3): on Python 3.13 the
SB3/Gymnasium pin matrix is the single likeliest thing to break a live judge
reproduction, and we also need action masking and bit-level determinism.

:func:`make_gym_env` wraps it as a genuine ``gymnasium.Env`` when Gymnasium is
installed, so Stable-Baselines3, RLlib or CleanRL can be dropped straight in.
A conformance test exercises that path when the extra is present.

Episodes are drawn from a **pool of scenario seeds** rather than regenerated per
reset: building the ground-truth tensors costs ~20 ms, which would otherwise
dominate a 200 k-step training run.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np

from smartscan.agents.belief import N_CHANNEL_FEATURES, N_GLOBAL_FEATURES, BeliefState
from smartscan.config import Config
from smartscan.env.receiver import Receiver
from smartscan.env.rf_environment import Scenario, build_episode, generate_scenario
from smartscan.runner import RewardAccountant

__all__ = ["SmartScanEnv", "make_gym_env", "observation_size"]


def observation_size(config: Config) -> int:
    """Return the flat observation length ``B * F + G``."""
    return config.n_channels * N_CHANNEL_FEATURES + N_GLOBAL_FEATURES


class SmartScanEnv:
    """Single-agent environment over the SmartScan receiver scheduling problem.

    Args:
        config: Resolved configuration.
        seeds: Pool of scenario seeds to sample episodes from.
        cache_episodes: Keep built episodes in memory. Costs about 10 MB per
            episode at the default grid and removes tensor construction from the
            training hot path.
        rng_seed: Seed controlling which scenario is drawn on each reset.
    """

    def __init__(
        self,
        config: Config,
        seeds: Sequence[int] | None = None,
        cache_episodes: bool = True,
        rng_seed: int = 0,
    ) -> None:
        self.cfg = config
        self.seeds = list(seeds or range(config.run.seed, config.run.seed + 16))
        self.cache_episodes = cache_episodes
        self._cache: dict[int, tuple[Scenario, Any]] = {}
        self._rng = np.random.default_rng(rng_seed)

        self.n_actions = config.n_channels
        self.obs_size = observation_size(config)
        self.belief: BeliefState | None = None
        self.receiver: Receiver | None = None
        self._accountant: RewardAccountant | None = None
        self._interferer_channels: set[int] = set()
        self._last_action: int | None = None
        self.current_seed: int | None = None

    # -- episode plumbing -------------------------------------------------- #
    def _episode_for(self, seed: int) -> tuple[Scenario, Any]:
        """Return (and optionally cache) the scenario and tensors for a seed."""
        if seed in self._cache:
            return self._cache[seed]
        scenario = generate_scenario(seed, config=self.cfg)
        episode = build_episode(scenario)
        if self.cache_episodes:
            self._cache[seed] = (scenario, episode)
        return scenario, episode

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """Start a new episode.

        Args:
            seed: Explicit scenario seed; drawn from the pool if omitted.

        Returns:
            ``(observation, info)`` where ``info["action_mask"]`` is the boolean
            legal-action mask.
        """
        chosen = int(seed if seed is not None else self._rng.choice(self.seeds))
        _scenario, episode = self._episode_for(chosen)
        self.current_seed = chosen
        self.receiver = Receiver(episode, self.cfg, seed=chosen)
        self.belief = BeliefState(self.cfg, episode.n_slots)
        self._accountant = RewardAccountant(self.cfg, episode)
        self._interferer_channels = {t.home_channel for t in episode.truth if t.is_interferer}
        self._last_action = None
        return self.observation(), self.info()

    def observation(self) -> np.ndarray:
        """Return the flat float32 belief feature vector."""
        assert self.belief is not None
        return self.belief.flat_features()

    def action_mask(self) -> np.ndarray:
        """Return the boolean legal-action mask of shape ``(B,)``."""
        assert self.receiver is not None
        return self.receiver.legal_actions()

    def info(self) -> dict[str, Any]:
        """Return the auxiliary info dict."""
        return {"action_mask": self.action_mask(), "seed": self.current_seed}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Take one dwell.

        Args:
            action: Centre-channel index. Must be legal; illegal actions raise
                rather than being silently clipped, because clipping would let a
                policy learn to over-sample the band edges for free.

        Returns:
            ``(observation, reward, terminated, truncated, info)``.
        """
        assert self.receiver is not None and self.belief is not None and self._accountant is not None
        self.belief.note_action(int(action))
        obs_rx = self.receiver.step(int(action))

        lo, hi = obs_rx.window
        genuine = obs_rx.hits & ~obs_rx.pfa_flags
        detected_ids = obs_rx.truth_ids[genuine]

        self.belief.update(obs_rx)
        stale = float(self.belief.time_since_visit.max())
        retuned = self._last_action is None or int(action) != self._last_action
        on_interferer = bool(self._interferer_channels & set(range(lo, hi)))
        reward = self._accountant.step(detected_ids, retuned, on_interferer, stale)
        self._last_action = int(action)

        terminated = bool(self.receiver.done)
        return self.observation(), float(reward), terminated, False, self.info()

    @property
    def n_slots(self) -> int:
        """Episode horizon in slots."""
        return self.cfg.n_slots


def make_gym_env(config: Config, seeds: Sequence[int] | None = None, rng_seed: int = 0) -> Any:
    """Wrap :class:`SmartScanEnv` as a ``gymnasium.Env``.

    Args:
        config: Resolved configuration.
        seeds: Pool of scenario seeds.
        rng_seed: Seed controlling episode selection.

    Returns:
        A ``gymnasium.Env`` instance.

    Raises:
        ImportError: If Gymnasium is not installed. Install the ``ml`` extra;
            the package's own PPO and DQN do not need it.
    """
    try:
        import gymnasium as gym
        from gymnasium import spaces
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "gymnasium is required for make_gym_env; install `pip install smartscan[ml]`. "
            "The bundled PPO/DQN do not need it."
        ) from exc

    class _GymSmartScan(gym.Env):
        """Gymnasium view of the SmartScan scheduling problem."""

        metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}

        def __init__(self) -> None:
            self._env = SmartScanEnv(config, seeds, rng_seed=rng_seed)
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(self._env.obs_size,), dtype=np.float32
            )
            self.action_space = spaces.Discrete(self._env.n_actions)

        def reset(self, *, seed: int | None = None, options: dict | None = None):
            super().reset(seed=seed)
            return self._env.reset(seed)

        def step(self, action):
            return self._env.step(int(action))

        def action_masks(self) -> np.ndarray:
            """Legal-action mask, in the form ``sb3_contrib.MaskablePPO`` expects."""
            return self._env.action_mask()

    return _GymSmartScan()
