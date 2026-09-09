"""The figures-of-merit report is judge-facing, so its rendering is tested.

``collect()`` runs episodes and is exercised by ``make figures``; what is
tested here is the part that turns numbers into the table a reader sees. The
specific risk is silent degradation: a formatting change that drops the
always-idle comparison row would leave a 98.86 % accuracy standing on its own,
which is exactly the misreading the row exists to prevent.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "figures_of_merit.py"


@pytest.fixture(scope="module")
def mod():
    """Import the script by path -- scripts/ is not a package."""
    spec = importlib.util.spec_from_file_location("figures_of_merit", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def data():
    """A minimal but structurally complete report payload."""
    def tier(acc: float, base: float) -> dict:
        return {
            "intercept_rate_per_s": 137.8, "reward_total": -351.0,
            "reward_discounted": -12.0, "ttfi_median_s": 0.083,
            "ttfi_p90_s": float("inf"), "twir_rate": 0.0217,
            "interception_ratio_raw": 0.0223, "coverage": 0.857,
            "coverage_entropy": 0.98, "n_seeds": 30,
            "pd_high_snr": 1.0, "pfa": 0.0, "sensitivity_pulse_db": 18.1,
            "sensitivity_energy_db": 2.9,
            "predictor": {
                "accuracy": acc, "accuracy_of_always_idle": base,
                "average_precision": 0.531, "ap_lift_over_base_rate": 6.15,
                "auc": 0.763, "brier": 0.0594,
            },
            "reward_decomposition": {
                "_n_interferers": 5, "_n_emitters": 30, "_w5": 2.0, "_seed": 20260902,
                "whittle": {"total": -678.1,
                            "terms": {"interferer": -1010.0, "new": 204.6,
                                      "reconfirm": 159.5, "retune": -31.3,
                                      "staleness": -0.9},
                            "counts": {"interferer": 505, "new": 20,
                                       "reconfirm": 319, "retune": 3128}},
                "sequential": {"total": -1540.6,
                               "terms": {"interferer": -1920.0},
                               "counts": {"interferer": 960}},
            },
        }
    return {
        "headline_agent": "whittle",
        "scan_estimation": {
            "median_arrival_time_error_s": 0.142,
            "median_rel_error_lomb_scargle": 0.00168,
            "median_rel_error_sdif": 0.238,
            "n_emitters": 40, "n_resolved": 26,
        },
        "tiers": {"easy": tier(0.9886, 0.9641), "hard": tier(0.8863, 0.8480)},
    }


def test_fmt_handles_the_awkward_values(mod):
    assert mod._fmt(None) == "—"
    assert mod._fmt(float("nan")) == "—"
    assert mod._fmt(float("inf")) == "∞"
    # Integer specs must not be handed a float, which raises ValueError.
    assert mod._fmt(30.0, "d") == "30"
    assert mod._fmt(0.9886, ".2%") == "98.86%"


def test_every_figure_of_merit_appears(mod, data):
    """All seven the problem statement names, by number."""
    out = mod.render(data)
    for n in range(1, 8):
        assert f"| {n} |" in out, f"figure of merit {n} missing from the table"
    for name in ("empirical_pd", "empirical_pfa", "sensitivity_db",
                 "average_intercept_rate", "average_reward", "prediction_scores",
                 "average_intercept_time_error"):
        assert name in out, f"{name} not credited"


def test_accuracy_never_stands_without_its_baseline(mod, data):
    """The row that stops 98.86 % being read as a triumph.

    An always-idle model scores 96.41 % on the same data. Reporting the
    accuracy without that comparison is the single most misleading thing this
    report could do, so it is asserted rather than trusted.
    """
    out = mod.render(data)
    assert "98.86%" in out, "accuracy not reported"
    assert "96.41%" in out, "always-idle baseline missing beside the accuracy"
    assert out.index("98.86%") < out.index("96.41%"), "baseline must follow the claim"
    assert "average precision" in out.lower()
    assert "lift over base rate" in out.lower()


def test_negative_return_is_explained_not_just_printed(mod, data):
    """A bare -351.0 invites the wrong conclusion; the split must be there."""
    out = mod.render(data)
    assert "why the return is negative" in out.lower()
    assert "interferer" in out and "-1010.0" in out
    # The comparison against the open-loop sweep is the actual result.
    assert "47%" in out or "48%" in out, "decoy-dwell saving vs sequential not stated"
    assert "staleness" in out


def test_arrival_error_is_not_faked_per_tier(mod, data):
    """One measurement, one cell -- three identical columns would imply three."""
    out = mod.render(data)
    row = next(ln for ln in out.splitlines() if ln.startswith("| 7 |"))
    assert row.count("0.142") == 1, f"arrival error repeated across tiers: {row}"
    assert "†" in row and "scan_on_scan" in out


def test_render_survives_a_missing_predictor(mod, data):
    """Checkpoints are large and may be absent; the report must still build."""
    data["tiers"]["easy"]["predictor"] = {}
    out = mod.render(data)
    assert "Figures of merit" in out
    assert "| 6 |" in out
