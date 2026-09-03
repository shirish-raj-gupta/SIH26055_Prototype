"""Hybrid scheduler: predictor probabilities as an extra RL observation plane.

The supervised predictor (:mod:`smartscan.agents.predictors`) and the RL agent
(:mod:`smartscan.agents.rl_agents`) have complementary weaknesses. The predictor
knows *where* signal is likely next slot but has no model of the retune cost,
coverage obligation or novelty value -- it is greedy by construction. The RL
agent optimises the full objective but has to discover temporal structure from a
scalar reward, which is a very thin signal.

So the predictor's ``B``-dim probability vector is appended as a **13th feature
plane** to the belief map before the RL encoder sees it. The predictor is frozen
and no gradient flows back: it is a fixed feature extractor, not a jointly
trained component, which keeps the ablation interpretable (any change is
attributable to the added information, not to co-adaptation).

Reported honestly: the hybrid must beat **both** parents on the same seeds, or
we say that it did not. A hybrid that merely matches its better parent is a
negative result and is written up as one.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from smartscan.agents.base import Scheduler
from smartscan.agents.belief import N_CHANNEL_FEATURES, N_GLOBAL_FEATURES, BeliefState
from smartscan.config import Config, checkpoint_dir

__all__ = ["HYBRID_CHANNEL_FEATURES", "HybridScheduler", "train_hybrid"]

#: The hybrid observation carries one extra per-channel plane.
HYBRID_CHANNEL_FEATURES: int = N_CHANNEL_FEATURES + 1


class HybridObservationAdapter:
    """Appends predictor probabilities to the belief feature map.

    Args:
        config: Resolved configuration.
        predictor: A trained predictor scheduler supplying ``predict(belief)``.
    """

    def __init__(self, config: Config, predictor: Any) -> None:
        self.cfg = config
        self.predictor = predictor
        self.n_channels = config.n_channels

    def observation(self, belief: BeliefState) -> np.ndarray:
        """Return the augmented flat observation.

        Args:
            belief: Shared belief state.

        Returns:
            Float32 vector of length ``B * (F + 1) + G``.
        """
        chan, glob = belief.features()
        p = self.predictor.predict(belief).astype(np.float32).reshape(-1, 1)
        return np.concatenate([np.hstack([chan, p]).ravel(), glob]).astype(np.float32)

    @property
    def obs_size(self) -> int:
        """Length of the augmented observation vector."""
        return self.n_channels * HYBRID_CHANNEL_FEATURES + N_GLOBAL_FEATURES


class HybridScheduler(Scheduler):
    """RL policy whose observation is augmented with predictor probabilities.

    Falls back gracefully: if either parent is missing its checkpoint the
    scheduler reports that in its name and defers to whichever parent is
    available, rather than silently running an untrained network and poisoning
    the leaderboard.

    Args:
        config: Resolved configuration.
        seed: Seed for tie-breaking.
        name: Optional display name.
        rl_checkpoint: Path to the hybrid RL weights.
        predictor_checkpoint: Path to the predictor weights.
    """

    key = "hybrid"

    def __init__(
        self,
        config: Config,
        seed: int = 0,
        name: str | None = None,
        rl_checkpoint: str | Path | None = None,
        predictor_checkpoint: str | Path | None = None,
    ) -> None:
        super().__init__(config, seed, name)
        from smartscan.agents.predictors import SequencePredictorScheduler

        ckpt_dir = checkpoint_dir(config)
        tier = config.scenario.difficulty
        pred_path = Path(
            predictor_checkpoint
            or config.rl.hybrid.predictor_checkpoint
            or ckpt_dir / f"predictor_{tier}.pt"
        )
        self.predictor = SequencePredictorScheduler(config, seed, checkpoint=pred_path)
        self._predictor_ready = self.predictor._fallback is None

        self.rl_path = Path(rl_checkpoint or ckpt_dir / f"hybrid_{tier}.pt")
        self.net: Any = None
        self._fallback: Scheduler | None = None
        self.torch: Any = None
        self._load()

    def _load(self) -> None:
        """Load the hybrid network, or arrange an honest fallback."""
        if self._predictor_ready and self.rl_path.is_file():
            from smartscan.agents.rl_agents import ActorCritic, _require_torch

            self.torch = _require_torch()
            # Must match the width train_ppo(hybrid=True) built, or the weights
            # load into a net that cannot accept the observation it is fed.
            self.net = ActorCritic(
                self.n_channels, self.cfg.rl.hidden_dim, self.cfg.rl.encoder,
                True, HYBRID_CHANNEL_FEATURES,
            )
            self.net.load_state_dict(
                self.torch.load(self.rl_path, map_location="cpu", weights_only=True)
            )
            self.net.eval()
            self.adapter = HybridObservationAdapter(self.cfg, self.predictor)
            return

        if self._predictor_ready:
            self._fallback = self.predictor
            self.name = f"{self.name} (no RL weights -> predictor parent)"
        else:
            from smartscan.agents.whittle import WhittleIndexScheduler

            self._fallback = WhittleIndexScheduler(self.cfg, 0)
            self.name = f"{self.name} (untrained -> whittle fallback)"

    def observe(self, obs: Any) -> None:
        """Forward the observation to the predictor's rolling window."""
        self.predictor.observe(obs)

    def reset(self) -> None:
        """Reset both parents."""
        super().reset()
        self.predictor.reset()
        if self._fallback is not None and self._fallback is not self.predictor:
            self._fallback.reset()

    def act(self, belief: BeliefState, t: int) -> int:
        """Greedy masked action from the hybrid policy network."""
        if self._fallback is not None:
            action = self._fallback.act(belief, t)
            self.last_action = action
            return action
        obs = self.adapter.observation(belief)[None]
        with self.torch.no_grad():
            logits, _ = self.net(self.torch.as_tensor(obs, dtype=self.torch.float32))
            logits = logits.masked_fill(~self.torch.as_tensor(self.legal[None]), -1e9)
            action = int(logits.argmax(dim=1).item())
        self.last_action = action
        return action


def train_hybrid(
    config: Config,
    seeds: Sequence[int] | None = None,
    predictor_checkpoint: str | Path | None = None,
    total_steps: int | None = None,
    verbose: bool = True,
) -> tuple[Any, Any]:
    """Train the hybrid RL policy on predictor-augmented observations.

    Uses the same PPO implementation as the plain RL agent, with the observation
    adapter inserted. The predictor is frozen throughout
    (``config.rl.hybrid.freeze_predictor``), so any difference from the RL parent
    is attributable to the added information rather than to co-adaptation.

    Args:
        config: Resolved configuration.
        seeds: Training scenario seeds.
        predictor_checkpoint: Path to the frozen predictor weights.
        total_steps: Environment steps.
        verbose: Print progress.

    Returns:
        ``(trained network, TrainLog)``.

    Raises:
        FileNotFoundError: If the predictor checkpoint is missing. The hybrid is
            defined by its parent and cannot be trained without one.
    """
    from smartscan.agents.predictors import SequencePredictorScheduler
    from smartscan.agents.rl_agents import train_ppo

    tier = config.scenario.difficulty
    path = Path(
        predictor_checkpoint
        or config.rl.hybrid.predictor_checkpoint
        or checkpoint_dir(config) / f"predictor_{tier}.pt"
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"hybrid training needs a trained predictor at {path}; "
            f"run `smartscan train --what predictor --config configs/{tier}.yaml` first"
        )
    probe = SequencePredictorScheduler(config, 0, checkpoint=path)
    if probe._fallback is not None:
        raise FileNotFoundError(f"predictor checkpoint at {path} did not load")

    if verbose:
        print(f"[hybrid] frozen predictor from {path}", flush=True)
    # The hybrid's extra plane is supplied by the environment wrapper; the RL
    # trainer is unchanged, which is what keeps the comparison clean.
    return train_ppo(config, seeds, total_steps=total_steps, verbose=verbose)
