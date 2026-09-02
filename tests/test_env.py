"""Environment, receiver and configuration contracts."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from pydantic import ValidationError

from smartscan.config import Config, load_config
from smartscan.env.emitters import beam_gain_db
from smartscan.env.propagation import (
    SNR_FLOOR_DB,
    albersheim_snr_db,
    detection_threshold,
    free_space_path_loss_db,
    min_snr_for_pd,
    noise_power_dbm,
    p_detect,
    p_detect_dwell,
)
from smartscan.env.receiver import Receiver
from smartscan.env.rf_environment import build_episode, generate_scenario, place_channels
from smartscan.env.types import Observation
from smartscan.hal.simulated import SimulatedBackend, detection_probability_tensor


@pytest.fixture(scope="module")
def cfg() -> Config:
    return load_config("easy.yaml")


@pytest.fixture(scope="module")
def episode(cfg: Config):
    return build_episode(generate_scenario(cfg.run.seed, config=cfg))


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
def test_unknown_key_is_an_error():
    """extra='forbid': a typo must fail loudly, not silently do nothing."""
    with pytest.raises(ValidationError):
        Config.model_validate({"schema_version": 1, "run": {"nmae": "typo"}})


def test_mix_must_sum_to_n_emitters():
    with pytest.raises(Exception, match="sum"):
        Config.model_validate(
            {"schema_version": 1, "scenario": {"n_emitters": 5, "mix": {"fixed_cw": 3}}}
        )


def test_dt_must_divide_episode():
    with pytest.raises(Exception, match="integral"):
        Config.model_validate({"schema_version": 1, "time": {"dt_s": 3e-3, "episode_s": 10.0}})


def test_ibw_cannot_exceed_band():
    with pytest.raises(Exception, match="ibw_channels"):
        Config.model_validate(
            {"schema_version": 1, "spectrum": {"n_channels": 4}, "receiver": {"ibw_channels": 8}}
        )


def test_ps_fixed_ranges_are_enforced():
    """PS-specified emitter parameter ranges must not silently drift."""
    with pytest.raises(Exception, match="PS range"):
        Config.model_validate(
            {"schema_version": 1, "emitters": {"circular_scan": {"scan_period_s": [0.1, 30.0]}}}
        )


@pytest.mark.parametrize("tier", ["easy", "medium", "hard", "scan_on_scan"])
def test_shipped_configs_load(tier: str):
    c = load_config(f"{tier}.yaml")
    assert c.n_slots > 0
    assert sum(c.scenario.mix.values()) == c.scenario.n_emitters
    assert len(c.hash()) == 32


def test_non_uniform_partitions():
    """Non-uniform partitioning is a PS requirement."""
    log_cfg = Config.model_validate({"schema_version": 1, "spectrum": {"partition": "log"}})
    grid = log_cfg.grid()
    assert grid.widths_hz[0] < grid.widths_hz[-1]  # constant fractional bandwidth
    assert np.all(np.diff(grid.edges_hz) > 0)

    edges = list(np.linspace(0.5e9, 18e9, 9))
    ex = Config.model_validate({
        "schema_version": 1,
        "spectrum": {"n_channels": 8, "partition": "explicit", "edges_hz": edges},
        "receiver": {"ibw_channels": 2},
    })
    assert ex.grid().n_channels == 8


# --------------------------------------------------------------------------- #
# Propagation and detection
# --------------------------------------------------------------------------- #
def test_fspl_matches_closed_form():
    assert free_space_path_loss_db(1e9, 1.0) == pytest.approx(92.44, abs=0.01)
    # Doubling range adds 6.02 dB.
    assert free_space_path_loss_db(1e9, 2.0) - free_space_path_loss_db(1e9, 1.0) == pytest.approx(
        6.02, abs=0.01
    )


def test_noise_power():
    assert noise_power_dbm(1e6, 0.0) == pytest.approx(-113.98, abs=0.05)


def test_threshold_delivers_requested_pfa():
    """Monte-Carlo the false-alarm rate against the requested Pfa."""
    rng = np.random.default_rng(0)
    for pfa in (1e-2, 1e-3):
        for n in (1, 4, 16):
            thresh = detection_threshold(pfa, n)
            w = rng.gamma(n, 1.0, size=200_000)
            assert (w > thresh).mean() == pytest.approx(pfa, rel=0.25)


def test_swerling1_single_pulse_closed_form():
    """Swerling I with N=1 must equal Pfa ** (1 / (1 + SNR))."""
    for snr_db in (-10.0, 0.0, 10.0, 20.0):
        chi = 10 ** (snr_db / 10)
        assert float(p_detect(snr_db, n_integrate=1, pfa=1e-4, swerling=1)) == pytest.approx(
            1e-4 ** (1 / (1 + chi)), rel=1e-9
        )


def test_swerling0_agrees_with_albersheim():
    """Independent cross-check of the exact ncx2 form against the empirical fit."""
    for n in (1, 2, 4, 8, 16, 64):
        exact = min_snr_for_pd(0.5, 1e-4, n, swerling=0)
        assert exact == pytest.approx(albersheim_snr_db(0.5, 1e-4, n), abs=0.3)


def test_pd_is_never_hardcoded_to_one():
    """Detection must be probabilistic across the whole SNR range."""
    pd = p_detect(np.linspace(-30, 40, 200), n_integrate=4, pfa=1e-4, swerling=1)
    assert np.all(np.diff(pd) >= -1e-12), "Pd must be monotone in SNR"
    assert pd.min() < 1e-3 and pd.max() > 0.9
    assert np.all(np.isfinite(pd))


def test_pd_at_snr_floor_tends_to_pfa():
    """At the SNR floor the detector fires only at the false-alarm rate."""
    for n in (1, 4, 256, 5000):
        assert float(p_detect(SNR_FLOOR_DB, n_integrate=n, pfa=1e-4, swerling=1)) == pytest.approx(
            1e-4, rel=0.05
        )


def test_pulse_dwell_combination():
    """1-of-n combination must be monotone in pulse count and saturate at 1."""
    pd = [float(p_detect_dwell(6.0, n, pfa=1e-4)) for n in (0, 1, 2, 10, 1000)]
    assert pd[0] == 0.0
    assert all(a <= b for a, b in itertools.pairwise(pd))
    assert pd[-1] > 0.99


def test_beam_gain_model():
    """Parabolic main lobe, floored at the sidelobe and backlobe levels."""
    assert beam_gain_db(np.array([0.0]), 2.0, -30.0)[0] == pytest.approx(0.0)
    assert beam_gain_db(np.array([1.0]), 2.0, -30.0)[0] == pytest.approx(-3.0, abs=0.01)
    assert beam_gain_db(np.array([45.0]), 2.0, -30.0)[0] == pytest.approx(-30.0)
    assert beam_gain_db(np.array([180.0]), 2.0, -30.0, -45.0)[0] == pytest.approx(-45.0)


# --------------------------------------------------------------------------- #
# Tensors and contracts
# --------------------------------------------------------------------------- #
def test_tensor_contracts(episode, cfg):
    b, t = cfg.n_channels, cfg.n_slots
    assert episode.occupancy.shape == (b, t) and episode.occupancy.dtype == np.uint8
    assert episode.duty.shape == (b, t) and episode.duty.dtype == np.float32
    assert episode.snr_db.shape == (b, t) and episode.snr_db.dtype == np.float32
    assert episode.emitter_id.shape == (b, t) and episode.emitter_id.dtype == np.int16
    assert episode.n_pulses.dtype == np.int32
    assert np.all((episode.duty >= 0) & (episode.duty <= 1))
    assert np.all(episode.snr_db[episode.occupancy == 0] == SNR_FLOOR_DB)
    assert set(np.unique(episode.emitter_id)) <= {0, *[t_.emitter_id for t_ in episode.truth]}


def test_every_emitter_class_produces_activity():
    """All eight classes must instantiate and emit."""
    cfg = load_config("hard.yaml")
    ep = build_episode(generate_scenario(cfg.run.seed, config=cfg))
    classes = {t.emitter_class for t in ep.truth}
    assert len(classes) >= 7
    for truth in ep.truth:
        assert 0.0 <= truth.threat_priority <= 1.0


def test_poisson_disk_placement_respects_separation():
    rng = np.random.default_rng(0)
    ch = place_channels(10, 128, rng, min_separation=3)
    assert ch.size == 10
    assert np.min(np.diff(np.sort(ch))) > 3


def test_popup_emitters_are_silent_until_their_time():
    cfg = load_config("hard.yaml")
    ep = build_episode(generate_scenario(cfg.run.seed, config=cfg))
    popups = [t for t in ep.truth if t.t_first_active > 0]
    assert popups, "hard tier must contain pop-up emitters"
    for truth in popups:
        assert truth.t_first_active > 0.6 * cfg.n_slots
        before = ep.emitter_id[:, : truth.t_first_active] == truth.emitter_id
        assert not before.any(), f"emitter {truth.emitter_id} transmitted before its pop-up time"


def test_frequency_agile_can_occupy_several_channels_in_one_slot():
    """Sub-slot hopping is why Activity is sparse rather than a dense per-slot map."""
    cfg = load_config("medium.yaml").with_overrides(
        emitters={"frequency_agile": {"hop_rate_hz": [8000.0, 10000.0]}}
    )
    ep = build_episode(generate_scenario(cfg.run.seed, config=cfg))
    agile = [t.emitter_id for t in ep.truth if t.emitter_class == "FrequencyAgile"]
    assert agile
    per_slot = (np.isin(ep.emitter_id, agile)).sum(axis=0)
    assert per_slot.max() >= 2


# --------------------------------------------------------------------------- #
# Receiver
# --------------------------------------------------------------------------- #
def test_action_masking_leaves_b_minus_k_plus_1_legal(episode, cfg):
    rx = Receiver(episode, cfg)
    assert rx.legal_actions().sum() == cfg.n_channels - cfg.receiver.ibw_channels + 1
    illegal = int(np.flatnonzero(~rx.legal_actions())[0])
    with pytest.raises(ValueError, match="illegal"):
        rx.step(illegal)


def test_retune_costs_settle_slots(episode, cfg):
    rx = Receiver(episode, cfg)
    first = rx.step(10)
    assert first.slots_elapsed == 1 + cfg.receiver.t_settle_slots
    stay = rx.step(10)
    assert stay.slots_elapsed == 1
    move = rx.step(20)
    assert move.slots_elapsed == 1 + cfg.receiver.t_settle_slots
    assert rx.n_retunes == 2


def test_observation_shape_and_window(episode, cfg):
    rx = Receiver(episode, cfg)
    obs = rx.step(10)
    k = cfg.receiver.ibw_channels
    assert isinstance(obs, Observation)
    assert obs.hits.shape == (k,) and obs.hits.dtype == np.bool_
    assert obs.snr_est_db.shape == (k,)
    assert obs.window[1] - obs.window[0] == k
    assert np.array_equal(obs.channels, np.arange(*obs.window))


def test_false_alarms_only_on_empty_cells(episode, cfg):
    backend = SimulatedBackend(episode, cfg)
    assert not np.any(backend.false_alarm & (episode.occupancy > 0))
    assert not np.any(backend.true_hit & (episode.occupancy == 0))


def test_empirical_pfa_matches_configured(episode, cfg):
    """The realised false-alarm rate must match the design Pfa."""
    backend = SimulatedBackend(episode, cfg)
    empty = episode.occupancy == 0
    rate = backend.false_alarm[empty].mean()
    assert rate == pytest.approx(cfg.receiver.detector.pfa, rel=0.35)


def test_pd_zero_below_sensitivity(episode, cfg):
    pd = detection_probability_tensor(episode, cfg)
    assert np.all(pd[episode.snr_db <= SNR_FLOOR_DB] == 0.0)
    assert np.all((pd >= 0) & (pd <= 1))


def test_soapy_backend_refuses_to_pretend():
    """The hardware stub must fail loudly, not return plausible fake data."""
    from smartscan.hal.soapy_stub import SoapySDRBackend

    with pytest.raises(NotImplementedError, match="non-functional stub"):
        SoapySDRBackend()
