"""Solver-level tests: closed intervals, strict gains, scale, and brute-force cross-checks."""

from __future__ import annotations

import itertools
import time

import numpy as np
import pytest

from app import solver


def _feasible_set_bruteforce(templates, energy, error, lo, hi):
    """Enumerate pressed subsets for a single take. Assertion helper only."""
    n = templates.shape[0]
    out = []
    for size in range(n + 1):
        for combo in itertools.combinations(range(n), size):
            g = solver._feasible_gains(
                templates, list(combo), energy, error, lo, hi
            )
            if g is not None:
                out.append(frozenset(combo))
    return out


def _random_instance(rng, n, m):
    templates = rng.integers(1, 4, size=(n, m)).astype(float)
    lo = rng.uniform(0.1, 0.4, size=n)
    hi = rng.uniform(0.5, 2.0, size=n)
    return templates, lo, hi


def test_strict_gain_interval() -> None:
    # lo == hi == 1 for the single key: gain is fixed.
    templates = np.array([[1.0, 0.0], [0.0, 1.0]])
    lo = np.array([1.0, 1.0])
    hi = np.array([1.0, 1.0])
    b = np.array([1.0, 0.0])
    e = np.array([0.0, 0.0])
    verdict = solver.classify(templates, (b, b), (e, e), lo, hi)
    assert verdict.kind == "same"
    sol = verdict.possibilities["same"][0]
    assert sol.pressed == [0]
    assert sol.gains[0] == pytest.approx(1.0)


def test_zero_gain_lower_rejected() -> None:
    # gain_lower == 0 would conflate "played with zero gain" and "not played".
    templates = np.array([[1.0]])
    lo = np.array([0.0])
    hi = np.array([2.0])
    b = np.array([1.0])
    e = np.array([0.0])
    with pytest.raises(solver.SolverError, match="strictly positive"):
        solver.classify(templates, (b, b), (e, e), lo, hi)


def test_fixed_gain_midpoint() -> None:
    templates = np.array([[1.0]])
    lo = np.array([0.5])
    hi = np.array([2.0])
    b = np.array([1.0])
    e = np.array([0.0])
    verdict = solver.classify(templates, (b, b), (e, e), lo, hi)
    assert verdict.kind == "same"
    assert verdict.possibilities["same"][0].gains[0] == pytest.approx(1.0)


def test_infeasible_band_energy() -> None:
    templates = np.array([[1.0]])
    lo = np.array([0.5])
    hi = np.array([1.0])
    b = np.array([5.0])
    e = np.array([0.1])
    verdict = solver.classify(templates, (b, b), (e, e), lo, hi)
    assert verdict.kind == "impossible"
    assert verdict.possibilities == {}


def test_validation_errors() -> None:
    templates = np.array([[1.0, 0.0]])
    b = np.array([1.0, 0.0])
    e = np.array([0.0, 0.0])
    with pytest.raises(solver.SolverError):
        solver.classify(templates, (b, b), (e, e), np.array([2.0]), np.array([1.0]))  # lo > hi
    with pytest.raises(solver.SolverError):
        solver.classify(-templates, (b, b), (e, e), np.array([0.5]), np.array([1.0]))  # negative
    with pytest.raises(solver.SolverError):
        solver.classify(templates, (b, b), (np.array([-1.0, 0.0]), e), np.array([0.5]), np.array([1.0]))
    with pytest.raises(solver.SolverError):
        solver.classify(templates, (b, b), (e, e), np.array([0.5]), np.array([1.0]), {(0, 5): True})


@pytest.mark.parametrize("seed", range(6))
def test_classification_matches_bruteforce(seed: int) -> None:
    """Cross-check both MILP hypotheses against exhaustive subset enumeration."""
    rng = np.random.default_rng(seed)
    n, m = 4, 2
    templates, lo, hi = _random_instance(rng, n, m)
    # Build random target energies from a randomly chosen pressed set per take.
    def synth(pressed):
        gains = np.zeros(n)
        for k in pressed:
            gains[k] = rng.uniform(lo[k], hi[k])
        return gains @ templates, gains

    true1 = set(rng.choice(n, size=rng.integers(1, 3), replace=False).tolist())
    true2 = set(rng.choice(n, size=rng.integers(1, 3), replace=False).tolist())
    b1, _ = synth(true1)
    b2, _ = synth(true2)
    e = np.full(m, 1e-9)

    feasible1 = _feasible_set_bruteforce(templates, b1, e, lo, hi)
    feasible2 = _feasible_set_bruteforce(templates, b2, e, lo, hi)
    same_possible = bool(set(feasible1) & set(feasible2))
    diff_possible = any(s1 != s2 for s1 in feasible1 for s2 in feasible2)

    verdict = solver.classify(templates, (b1, b2), (e, e), lo, hi)
    expected = (
        "both" if same_possible and diff_possible
        else "same" if same_possible
        else "different" if diff_possible
        else "impossible"
    )
    assert verdict.kind == expected
    if same_possible:
        assert verdict.possibilities["same"][0].pressed == verdict.possibilities["same"][1].pressed
    if diff_possible:
        assert (
            verdict.possibilities["different"][0].pressed
            != verdict.possibilities["different"][1].pressed
        )
    # Every witness must independently satisfy the band constraints.
    for name, takes in verdict.possibilities.items():
        for take, energy in zip(takes, (b1, b2)):
            gains = np.zeros(n)
            for k in take.pressed:
                gains[k] = take.gains[k]
            produced = gains @ templates
            assert np.allclose(produced, energy, atol=1e-6, rtol=0)


def test_24_keys_within_time_budget() -> None:
    rng = np.random.default_rng(123)
    n, m = 24, 6
    templates, lo, hi = _random_instance(rng, n, m)
    pressed = [2, 9, 17]
    gains = np.zeros(n)
    for k in pressed:
        gains[k] = rng.uniform(lo[k], hi[k])
    b = gains @ templates
    e = np.full(m, 1e-7)
    start = time.monotonic()
    verdict = solver.classify(templates, (b, b), (e, e), lo, hi, time_limit=30.0)
    elapsed = time.monotonic() - start
    assert verdict.kind in {"same", "both"}
    for name, takes in verdict.possibilities.items():
        for take, energy in zip(takes, (b, b)):
            gains = np.zeros(n)
            for k in take.pressed:
                gains[k] = take.gains[k]
            assert np.allclose(gains @ templates, energy, atol=1e-5, rtol=0)
    if verdict.kind == "same":
        assert verdict.possibilities["same"][0].pressed == pressed
    assert elapsed < 30.0


def test_confirmation_conflict_evidence() -> None:
    # Two distinguishable keys so a differing witness exists: energy is 0, so
    # key 0 can only be pressed in a take whose error band admits zero output.
    templates = np.array([[1.0], [1.0]])
    lo = np.array([0.5, 0.5])
    hi = np.array([1.0, 1.0])
    b = np.array([1.0])
    e = np.array([0.0])
    verdict = solver.classify(
        templates, (b, b), (e, e), lo, hi,
        confirmations={(0, 0): True, (1, 0): False},
    )
    assert verdict.kind == "different"
    assert verdict.evidence["models"]["same"]["reason"] == "confirmations_imply_different_press_state"


# ---------------------------------------------------------------------------
# equivalence-group aggregation
# ---------------------------------------------------------------------------
def _one_group_instance(n: int, bands: int = 3):
    """n keys sharing one template row and one gain interval."""
    row = np.full(bands, 0.7)
    templates = np.tile(row, (n, 1))
    lo = np.full(n, 0.4)
    hi = np.full(n, 1.6)
    energy = row.copy()  # one key at gain 1 reproduces it
    error = np.full(bands, 1e-9)
    return templates, lo, hi, energy, error


def test_groups_partition_identical_template_and_bounds() -> None:
    templates = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    lo = np.array([0.5, 0.5, 0.9])
    hi = np.array([2.0, 2.0, 2.0])
    groups = solver._build_groups(templates, lo, hi, {})
    assert [g.members for g in groups] == [[0, 1], [2]]
    assert groups[0].pool == [0, 1] and groups[0].pool_size == 2
    # A differing gain interval splits otherwise identical rows.
    lo2 = np.array([0.5, 0.6, 0.9])
    groups2 = solver._build_groups(templates, lo2, hi, {})
    assert [g.members for g in groups2] == [[0], [1], [2]]


def test_24_single_group_smaller_than_per_key_model() -> None:
    n = 24
    templates, lo, hi, energy, error = _one_group_instance(n)
    verdict = solver.classify(
        templates, (energy, energy), (error, error), lo, hi, time_limit=30.0
    )
    for model in ("same", "different"):
        actual = verdict.evidence["models"][model]["model_size"]
        per_key = verdict.evidence["models"][model]["per_key_model_size"]
        assert actual["variables"] < per_key["variables"]
        assert actual["constraints"] < per_key["constraints"]
    # The report still names concrete keys and per-key gains in each witness.
    for _name, takes in verdict.possibilities.items():
        for take in takes:
            assert take.gains.keys() == set(take.pressed)
            for k in take.pressed:
                assert lo[k] - 1e-7 <= take.gains[k] <= hi[k] + 1e-7


def test_adding_unconfirmed_group_key_adds_no_variables_or_constraints() -> None:
    def sizes(n: int):
        templates, lo, hi, energy, error = _one_group_instance(n)
        verdict = solver.classify(
            templates, (energy, energy), (error, error), lo, hi, time_limit=30.0
        )
        return {
            model: verdict.evidence["models"][model]["model_size"]
            for model in ("same", "different")
        }

    assert sizes(3) == sizes(7) == sizes(13) == sizes(24)

    # A confirmed member keeps its own binary/gain variables, but adding more
    # *unconfirmed* members on top of that confirmation still adds nothing.
    def confirmed_sizes(n: int):
        templates, lo, hi, energy, error = _one_group_instance(n)
        verdict = solver.classify(
            templates, (energy, energy), (error, error), lo, hi,
            confirmations={(0, 0): True}, time_limit=30.0,
        )
        return {
            model: verdict.evidence["models"][model]["model_size"]
            for model in ("same", "different")
        }

    assert confirmed_sizes(5) == confirmed_sizes(12)
    assert confirmed_sizes(5)["same"]["variables"] > sizes(5)["same"]["variables"]


def test_equal_group_count_can_still_mean_different_key_names() -> None:
    # Two keys share a template; one unit of energy is one key at gain 1.
    templates = np.array([[1.0], [1.0]])
    lo = np.array([0.5, 0.5])
    hi = np.array([1.0, 1.0])
    b = np.array([1.0])
    e = np.array([0.0])
    verdict = solver.classify(templates, (b, b), (e, e), lo, hi)
    assert verdict.kind == "both"
    same = verdict.possibilities["same"]
    different = verdict.possibilities["different"]
    # Same-set witness: identical concrete key set in both takes.
    assert same[0].pressed == same[1].pressed
    # Different-set witness: same group count (one key) but different names.
    assert different[0].pressed != different[1].pressed
    assert len(different[0].pressed) == len(different[1].pressed) == 1
    for take in different:
        produced = sum(take.gains[k] for k in take.pressed)
        assert abs(produced - 1.0) <= 1e-7


def test_group_witnesses_always_verify_against_original_templates() -> None:
    rng = np.random.default_rng(99)
    n, m = 6, 3
    base = rng.integers(0, 3, size=(3, m)).astype(float)
    assign = rng.integers(0, 3, size=n)
    templates = base[assign]
    lo = np.full(n, 0.3)
    hi = np.full(n, 1.8)
    for _ in range(25):
        pressed = rng.choice(n, size=int(rng.integers(1, 4)), replace=False)
        gains = np.zeros(n)
        for k in pressed:
            gains[k] = rng.uniform(lo[k], hi[k])
        energy = gains @ templates
        error = np.full(m, 1e-9)
        verdict = solver.classify(
            templates, (energy, energy), (error, error), lo, hi, time_limit=20.0
        )
        for _hyp, takes in verdict.possibilities.items():
            for take in takes:
                rebuilt = np.zeros(n)
                for k in take.pressed:
                    rebuilt[k] = take.gains[k]
                assert np.all(np.abs(rebuilt @ templates - energy) <= error + 1e-7)

