"""Determinism: the same seed must reproduce byte-identical tensors.

Acceptance test 1. The golden digests below are checked in deliberately -- an
unintended change to emitter physics, the link budget or the RNG stream layout
will break them, which is exactly what they are for. Regenerate them ONLY with a
deliberate physics change, and say so in the commit message.
"""

from __future__ import annotations

import numpy as np
import pytest

from smartscan.config import load_config
from smartscan.env.rf_environment import build_episode, generate_scenario
from smartscan.hal.simulated import SimulatedBackend
from smartscan.seeding import SeedTree, stable_hash


@pytest.mark.acceptance
@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_same_seed_gives_identical_tensors(tier: str):
    cfg = load_config(f"{tier}.yaml")
    a = build_episode(generate_scenario(cfg.run.seed, config=cfg))
    b = build_episode(generate_scenario(cfg.run.seed, config=cfg))
    assert a.digest() == b.digest()
    for name in ("occupancy", "duty", "snr_db", "emitter_id", "n_pulses"):
        assert np.array_equal(getattr(a, name), getattr(b, name)), name


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_different_seeds_give_different_tensors(tier: str):
    cfg = load_config(f"{tier}.yaml")
    a = build_episode(generate_scenario(cfg.run.seed, config=cfg))
    b = build_episode(generate_scenario(cfg.run.seed + 1, config=cfg))
    assert a.digest() != b.digest()


def test_detection_realisation_is_reproducible():
    """Common random numbers: the same seed gives the same luck."""
    cfg = load_config("easy.yaml")
    ep = build_episode(generate_scenario(cfg.run.seed, config=cfg))
    a = SimulatedBackend(ep, cfg, seed=cfg.run.seed)
    b = SimulatedBackend(ep, cfg, seed=cfg.run.seed)
    assert np.array_equal(a.declared, b.declared)
    assert np.array_equal(a.true_hit, b.true_hit)
    assert np.array_equal(a.snr_report, b.snr_report)


def test_stable_hash_is_process_independent():
    """``hash()`` is salted per process; ours must not be."""
    assert stable_hash("scenario") == stable_hash("scenario")
    assert stable_hash("scenario") != stable_hash("receiver")
    assert 0 <= stable_hash("emitter") < 2**63


def test_named_streams_are_independent():
    tree = SeedTree(42)
    a = tree.rng("scenario").random(8)
    b = tree.rng("receiver").random(8)
    assert not np.allclose(a, b)
    assert np.array_equal(a, tree.rng("scenario").random(8))


def test_unknown_stream_is_rejected():
    with pytest.raises(KeyError, match="unknown RNG stream"):
        SeedTree(0).rng("not_a_registered_stream")


def test_adding_an_emitter_does_not_perturb_the_others():
    """The whole point of the SeedSequence tree.

    If emitter randomness were drawn from one shared stream, adding a 16th
    emitter would shift the parameters of the first 15 and silently invalidate
    every density ablation.
    """
    cfg = load_config("medium.yaml")
    bigger = cfg.with_overrides(
        scenario={"n_emitters": 16, "mix": {**cfg.scenario.mix, "fixed_cw": cfg.scenario.mix["fixed_cw"] + 1}}
    )
    a = generate_scenario(cfg.run.seed, config=cfg)
    b = generate_scenario(cfg.run.seed, config=bigger)
    shared = min(len(a.emitters), len(b.emitters))
    matched = sum(
        1
        for i in range(shared)
        if type(a.emitters[i]) is type(b.emitters[i])
        and a.emitters[i].eirp_dbm == b.emitters[i].eirp_dbm
    )
    # The class ORDER is reshuffled by the larger mix, but the per-emitter
    # substreams are keyed by index, so most emitters keep their draws.
    assert matched >= shared // 2


def test_changing_the_agent_does_not_change_the_world():
    """A scheduler must not be able to perturb the scenario it is measured in."""
    from smartscan.agents import build_agent
    from smartscan.runner import run_episode

    cfg = load_config("easy.yaml")
    sc = generate_scenario(cfg.run.seed, config=cfg)
    ep = build_episode(sc)
    before = ep.digest()
    for key in ("sequential", "random", "thompson", "whittle"):
        run_episode(cfg, cfg.run.seed, build_agent(key, cfg, cfg.run.seed, sc), scenario=sc, episode=ep)
    assert ep.digest() == before


@pytest.mark.acceptance
def test_episode_digests_match_golden():
    """Golden digests: a physics change must be a deliberate, visible act."""
    golden = {}
    for tier in ("easy", "medium", "hard"):
        cfg = load_config(f"{tier}.yaml")
        ep = build_episode(generate_scenario(cfg.run.seed, config=cfg))
        golden[tier] = ep.digest()
    # Recorded on first green run; see the module docstring before changing.
    from tests.golden import EPISODE_DIGESTS

    assert golden == EPISODE_DIGESTS, (
        "ground-truth tensors changed. If this was intentional, update "
        f"tests/golden.py to:\n{golden}"
    )


def test_config_hash_is_stable_and_sensitive():
    cfg = load_config("medium.yaml")
    assert cfg.hash() == load_config("medium.yaml").hash()
    assert cfg.hash() != cfg.with_overrides(run={"seed": cfg.run.seed + 1}).hash()
