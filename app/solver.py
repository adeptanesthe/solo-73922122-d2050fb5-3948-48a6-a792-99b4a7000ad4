"""Scientific core: classify and reconstruct two piano takes with SciPy MILP.

Model
-----
A candidate key ``k`` has a non-negative spectral template ``T[k, :]`` over the
measured frequency bands.  A take chooses a gain ``g[k] >= 0`` for every key.
An unplayed key has gain exactly zero; a played key has a gain in the closed
interval ``[lo[k], hi[k]]``.  The take's band energies are
``sum_k g[k] * T[k, :]`` and every band must lie in ``[b - e, b + e]``.

Equivalence groups
------------------
With coarse frequency bands, several candidate keys can share an *identical*
spectral template row and identical gain bounds.  Such keys are
indistinguishable to the band equations and are collected into one equivalence
group.  Within a group, only the number of pressed keys matters for feasibility
(any ``q`` members out of ``s`` can realize it, each with a positive gain), so
fully-unconfirmed members are modelled collectively instead of one binary per
key:

* a general-integer pressed-count variable ``q`` with ``0 <= q <= s`` over the
  group's unconfirmed pool (confirmed members keep their own binaries),
* one pooled non-negative gain variable ``G`` linked by
  ``lo*q <= G <= hi*q``.  Because every member of a group shares its gain
  interval, any feasible pooled gain can be split back into ``q`` per-key gains
  inside it.

Adding another fully-unconfirmed key to a group therefore only raises an upper
bound; it introduces **no** new variable and **no** new constraint.  A key that
an editor has already confirmed in either take keeps its own binary/gain pair,
which is the only information that can distinguish group members.

The same-set hypothesis shares one pressed state per key; the different-set
hypothesis allows two states, either because the groups' pressed counts differ
(two direction binaries ``dp``/``dm``) or because the counts coincide but two
different member subsets of an interior count are chosen (``z``, possible only
while a group still holds an unconfirmed pool of at least two keys).  No key
combination is ever enumerated: HiGHS branches on the (group-level) integer
variables, and a feasible solution is expanded back to concrete key names only
afterwards.

A strict pin ``lo == hi`` is handled by the same linearisation; it remains exact
for closed intervals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

# A confirmation maps (segment index, key index) -> pressed (True/False).
Confirmations = dict[tuple[int, int], bool]

_INF = np.inf
_OPTIMAL = 0
_INFEASIBLE = 2
# Absolute slack used only to absorb floating-point noise when re-checking a
# witness; it is far below any meaningful spectral tolerance.
_FEAS_TOL = 1e-7


class SolverError(Exception):
    """Raised for malformed numerical input (distinct from infeasibility)."""


class SolverInconclusive(Exception):
    """The solver stopped without proving feasibility or infeasibility."""


@dataclass(frozen=True)
class Solution:
    """One feasible reconstruction for a take."""

    pressed: list[int]
    gains: dict[int, float]


@dataclass(frozen=True)
class Outcome:
    """Result of one hypothesis model."""

    feasible: bool
    takes: tuple[Solution, Solution] | None
    evidence: dict[str, object]


@dataclass(frozen=True)
class Verdict:
    """Three-way classification plus witnesses for every feasible hypothesis."""

    kind: Literal["same", "different", "both", "impossible"]
    # Keys "same" and/or "different" mapped to a witness pair for that hypothesis.
    possibilities: dict[str, tuple[Solution, Solution]]
    evidence: dict[str, object]


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
def _validate_templates(templates: np.ndarray) -> None:
    if templates.ndim != 2:
        raise SolverError("templates must be a 2-D array (keys x bands)")
    if templates.size == 0:
        raise SolverError("templates must not be empty")
    if not np.all(np.isfinite(templates)):
        raise SolverError("templates contain non-finite values")
    if np.any(templates < 0):
        raise SolverError("templates must be non-negative")


def _validate_energy(name: str, values: np.ndarray, bands: int) -> None:
    if values.shape != (bands,):
        raise SolverError(f"{name} must have {bands} bands, got shape {values.shape}")
    if not np.all(np.isfinite(values)):
        raise SolverError(f"{name} contains non-finite values")


def _validate_error(name: str, values: np.ndarray, bands: int) -> None:
    if values.shape != (bands,):
        raise SolverError(f"{name} must have {bands} bands, got shape {values.shape}")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise SolverError(f"{name} must be finite and non-negative")


def _validate_gains(lo: np.ndarray, hi: np.ndarray, keys: int) -> None:
    for name, arr in (("gain_lower", lo), ("gain_upper", hi)):
        if arr.shape != (keys,):
            raise SolverError(f"{name} must have {keys} entries, got shape {arr.shape}")
        if not np.all(np.isfinite(arr)) or np.any(arr < 0):
            raise SolverError(f"{name} must be finite and non-negative")
    # A played key must scale its template by a *positive* gain.  Allowing
    # gain_lower == 0 would make "played with gain 0" indistinguishable from
    # "not played", which contradicts the problem statement.
    if np.any(lo <= 0):
        raise SolverError("gain_lower must be strictly positive for every key")
    if np.any(lo > hi):
        raise SolverError("gain_lower must not exceed gain_upper for any key")


def _check_confirmations(conf: Confirmations, keys: int) -> None:
    for (seg, key), _pressed in conf.items():
        if seg not in (0, 1):
            raise SolverError(f"confirmation segment {seg} must be 0 or 1")
        if not 0 <= key < keys:
            raise SolverError(f"confirmation key index {key} out of range")


# ---------------------------------------------------------------------------
# equivalence groups
# ---------------------------------------------------------------------------
@dataclass
class _Group:
    """Keys sharing a spectral template row and a gain interval."""

    members: list[int]
    template: np.ndarray
    lo: float
    hi: float
    # Confirmed keys: key index -> set of segments that pinned it (0/1).
    confirmed: dict[int, set[int]] = field(default_factory=dict)
    # Keys not confirmed in either segment, modelled by one pooled block.
    pool: list[int] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.members)

    @property
    def pool_size(self) -> int:
        return len(self.pool)


def _build_groups(
    templates: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    conf: Confirmations,
) -> list[_Group]:
    """Partition keys into equivalence groups by (template row, lo, hi)."""
    groups: list[_Group] = []
    lookup: dict[tuple, _Group] = {}
    for k in range(templates.shape[0]):
        signature = (tuple(float(v) for v in templates[k, :]), float(lo[k]), float(hi[k]))
        group = lookup.get(signature)
        if group is None:
            group = _Group(
                members=[k],
                template=templates[k, :].copy(),
                lo=float(lo[k]),
                hi=float(hi[k]),
            )
            lookup[signature] = group
            groups.append(group)
        else:
            group.members.append(k)

    for group in groups:
        pinned: dict[int, set[int]] = {}
        for (seg, key), pressed in conf.items():
            if key in group.members:
                pinned.setdefault(key, set()).add(seg)
        group.confirmed = pinned
        group.pool = [k for k in group.members if k not in pinned]
    return groups


def _group_evidence(groups: list[_Group], conf: Confirmations) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for gid, group in enumerate(groups):
        rows.append(
            {
                "group": gid,
                "keys": list(group.members),
                "size": group.size,
                "unconfirmed_pool_size": group.pool_size,
                "confirmed_keys": sorted(group.confirmed),
                "count_bounds": {
                    "segment_0": [
                        sum(1 for k in group.confirmed if conf.get((0, k)) is True),
                        group.size - sum(1 for k in group.confirmed if conf.get((0, k)) is False),
                    ],
                    "segment_1": [
                        sum(1 for k in group.confirmed if conf.get((1, k)) is True),
                        group.size - sum(1 for k in group.confirmed if conf.get((1, k)) is False),
                    ],
                },
            }
        )
    return rows


def _reference_model_size(n: int) -> dict[str, dict[str, int]]:
    """Variable/constraint counts the original per-key MILPs would have.

    Counts exclude the shared band rows and (for the different-set model) the
    single global "at least one difference" row, which are identical in number
    between the per-key and the aggregated formulations.
    """
    return {
        # same: n binaries + 2n gains; 2 linkage rows per key per take.
        "same_set": {"variables": 3 * n, "constraints": 4 * n},
        # different: 2n binaries + 2n gains + n difference binaries;
        # 2 linkage rows per key per take (4n) + 4 difference rows per key (4n).
        "different_set": {"variables": 5 * n, "constraints": 8 * n},
    }


# ---------------------------------------------------------------------------
# sparse MILP bookkeeping
# ---------------------------------------------------------------------------
class _MilpBuilder:
    """Accumulate variables and sparse rows for one MILP."""

    def __init__(self) -> None:
        self.lb: list[float] = []
        self.ub: list[float] = []
        self.integrality: list[int] = []
        self.rows: list[dict[int, float]] = []
        self.rlb: list[float] = []
        self.rub: list[float] = []
        self.objective: list[float] = []
        # Rows added via :meth:`band_constraint` are the per-take band rows,
        # which are identical in count to the original per-key formulation and
        # are therefore reported separately rather than as aggregation savings.
        self.band_rows = 0
        # Rows added as global cross-group coupling (the single difference
        # requirement); also counted separately for a like-for-like footprint.
        self.global_rows = 0

    def var(self, lb: float, ub: float, integer: bool, cost: float = 0.0) -> int:
        idx = len(self.lb)
        self.lb.append(lb)
        self.ub.append(ub)
        self.integrality.append(1 if integer else 0)
        self.objective.append(cost)
        return idx

    def constraint(self, coefficients: dict[int, float], lb: float, ub: float) -> None:
        self.rows.append(dict(coefficients))
        self.rlb.append(lb)
        self.rub.append(ub)

    def band_constraint(self, coefficients: dict[int, float], lb: float, ub: float) -> None:
        self.constraint(coefficients, lb, ub)
        self.band_rows += 1

    def fix(self, idx: int, value: float) -> None:
        self.lb[idx] = self.ub[idx] = value

    def solve(self, time_limit: float) -> object:
        nv = len(self.lb)
        matrix = np.zeros((len(self.rows), nv))
        for i, row in enumerate(self.rows):
            for j, value in row.items():
                matrix[i, j] = value
        return milp(
            np.asarray(self.objective, dtype=float),
            integrality=np.asarray(self.integrality),
            bounds=Bounds(np.asarray(self.lb), np.asarray(self.ub)),
            constraints=LinearConstraint(matrix, np.asarray(self.rlb), np.asarray(self.rub)),
            options={"time_limit": time_limit, "mip_rel_gap": 0.0, "disp": False},
        )

    @property
    def size(self) -> dict[str, int]:
        """Aggregated model size, excluding formulation-neutral band/global rows."""
        return {
            "variables": len(self.lb),
            "constraints": len(self.rows) - self.band_rows - self.global_rows,
        }


def _solver_diagnostics(
    model: str,
    result: object,
    conf: Confirmations,
    groups: list[_Group],
    size: dict[str, int],
    n: int,
) -> dict[str, object]:
    return {
        "model": model,
        "solver_status": int(getattr(result, "status", -1)),
        "solver_message": str(getattr(result, "message", "unknown solver status")),
        "mip_node_count": getattr(result, "mip_node_count", None),
        "group_count": len(groups),
        "model_size": size,
        "per_key_model_size": _reference_model_size(n)[model],
        "confirmations": {f"{seg}:{key}": pressed for (seg, key), pressed in sorted(conf.items())},
    }


# ---------------------------------------------------------------------------
# witness extraction, expansion, and verification
# ---------------------------------------------------------------------------
def _verify(
    templates: np.ndarray,
    gains: np.ndarray,
    pressed: list[int],
    energy: np.ndarray,
    error: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
) -> bool:
    allowed = set(pressed)
    for k in range(len(gains)):
        if k in allowed:
            if gains[k] < lo[k] - _FEAS_TOL or gains[k] > hi[k] + _FEAS_TOL:
                return False
        elif gains[k] != 0.0:
            return False
    produced = gains @ templates
    if not np.all(np.isfinite(produced)):
        return False
    return bool(
        np.all(produced <= energy + error + _FEAS_TOL)
        and np.all(produced >= energy - error - _FEAS_TOL)
    )


def _feasible_gains(
    templates: np.ndarray,
    pressed: list[int],
    energy: np.ndarray,
    error: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
) -> np.ndarray | None:
    """Find gains for a fixed pressed set with a small LP (feasibility only)."""
    gains = np.zeros(templates.shape[0])
    if not pressed:
        if np.all(np.abs(energy) <= error + _FEAS_TOL):
            return gains
        return None
    idx = np.asarray(pressed, dtype=int)
    sub = templates[idx, :]
    # sub^T g <= b + e  and  -(sub^T g) <= -(b - e)
    a_ub = np.vstack([sub.T, -sub.T])
    b_ub = np.concatenate([energy + error, -(energy - error)])
    bounds = [(float(lo[k]), float(hi[k])) for k in idx]
    res = linprog(np.zeros(len(idx)), A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        return None
    for pos, k in enumerate(idx):
        gains[k] = float(res.x[pos])
    return gains


def _split_pooled_gain(pool: list[int], pooled: float, lo: float, hi: float) -> dict[int, float]:
    """Split a pooled group gain over chosen members within the shared interval."""
    q = len(pool)
    if q == 0:
        return {}
    share = pooled / q
    if lo - _FEAS_TOL <= share <= hi + _FEAS_TOL:
        share = min(max(share, lo), hi)
        return {k: share for k in pool}
    # General fallback respecting closed intervals: fill at the lower bound and
    # put the remainder on the last member.
    gains = {k: lo for k in pool}
    remainder = pooled - q * lo
    last = pool[-1]
    gains[last] += remainder
    if gains[last] > hi + _FEAS_TOL:  # pragma: no cover - guarded by the pool linkage
        gains[last] = hi
    return gains


def _build_solution(
    templates: np.ndarray,
    pressed: list[int],
    gains_map: dict[int, float],
    energy: np.ndarray,
    error: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
) -> Solution | None:
    """Verify an expanded witness; re-solve its gains once as a fallback."""
    gains = np.zeros(templates.shape[0])
    for k in pressed:
        gains[k] = float(gains_map[k])
    if not _verify(templates, gains, pressed, energy, error, lo, hi):
        recovered = _feasible_gains(templates, pressed, energy, error, lo, hi)
        if recovered is None:
            return None
        gains = recovered
        if not _verify(templates, gains, pressed, energy, error, lo, hi):
            return None
    return Solution(pressed=list(pressed), gains={k: float(gains[k]) for k in pressed})


# ---------------------------------------------------------------------------
# same-set model: one pressed state shared by both takes
# ---------------------------------------------------------------------------
def _same_set_model(
    templates: np.ndarray,
    b1: np.ndarray,
    e1: np.ndarray,
    b2: np.ndarray,
    e2: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    conf: Confirmations,
    groups: list[_Group],
    time_limit: float,
) -> Outcome:
    n, m = templates.shape
    builder = _MilpBuilder()
    layout: dict[str, object] = {"segment": 0}

    for gid, group in enumerate(groups):
        info: dict[str, object] = {"confirmed": {}}
        # A key confirmed in either take gets one shared binary and two gains.
        for key in sorted(group.confirmed):
            p0 = conf.get((0, key))
            p1 = conf.get((1, key))
            if p0 is not None and p1 is not None and p0 != p1:
                return Outcome(
                    feasible=False,
                    takes=None,
                    evidence={
                        "model": "same_set",
                        "reason": "confirmations_imply_different_press_state",
                        "key_index": key,
                        "confirmed": {"segment": 0, "pressed": p0},
                        "conflicting": {"segment": 1, "pressed": p1},
                    },
                )
            x = builder.var(0.0, 1.0, integer=True, cost=1.0)
            g1 = builder.var(0.0, group.hi, integer=False)
            g2 = builder.var(0.0, group.hi, integer=False)
            pressed = p0 if p0 is not None else p1
            if pressed is not None:
                builder.fix(x, 1.0 if pressed else 0.0)
            builder.constraint({g1: 1.0, x: -group.hi}, -_INF, 0.0)
            builder.constraint({x: group.lo, g1: -1.0}, -_INF, 0.0)
            builder.constraint({g2: 1.0, x: -group.hi}, -_INF, 0.0)
            builder.constraint({x: group.lo, g2: -1.0}, -_INF, 0.0)
            info["confirmed"][key] = (x, g1, g2)

        u = group.pool_size
        if u > 0:
            # Pool members are unconfirmed in both takes by construction, so
            # the shared count ranges over the whole pool.
            q = builder.var(0.0, float(u), integer=True, cost=1.0)
            g1 = builder.var(0.0, u * group.hi, integer=False)
            g2 = builder.var(0.0, u * group.hi, integer=False)
            builder.constraint({g1: 1.0, q: -group.hi}, -_INF, 0.0)
            builder.constraint({q: group.lo, g1: -1.0}, -_INF, 0.0)
            builder.constraint({g2: 1.0, q: -group.hi}, -_INF, 0.0)
            builder.constraint({q: group.lo, g2: -1.0}, -_INF, 0.0)
            info["q"] = q
            info["g1"] = g1
            info["g2"] = g2
        layout[gid] = info

    # One band row per take and band, expressed on group-level gains.
    for seg, energy, error, gain_kind in ((0, b1, e1, 1), (1, b2, e2, 2)):
        for j in range(m):
            row: dict[int, float] = {}
            for gid, group in enumerate(groups):
                info = layout[gid]
                for key, var_info in info["confirmed"].items():  # type: ignore[index]
                    g_idx = var_info[gain_kind]
                    row[g_idx] = row.get(g_idx, 0.0) + float(group.template[j])
                if group.pool_size > 0:
                    g_idx = info["g1"] if gain_kind == 1 else info["g2"]  # type: ignore[index]
                    row[g_idx] = row.get(g_idx, 0.0) + float(group.template[j])
            builder.band_constraint(row, float(energy[j] - error[j]), float(energy[j] + error[j]))

    result = builder.solve(time_limit)
    status = int(getattr(result, "status", -1))
    if status == _OPTIMAL:
        values = np.asarray(result.x)
        # Re-map the shared layout into per-segment expansion layouts.
        sol1 = _expand_same(templates, groups, conf, values, b1, e1, lo, hi, layout, segment=0)
        sol2 = _expand_same(templates, groups, conf, values, b2, e2, lo, hi, layout, segment=1)
        if sol1 is None or sol2 is None or sol1.pressed != sol2.pressed:
            raise SolverInconclusive("same-set witness failed verification")
        return Outcome(
            feasible=True,
            takes=(sol1, sol2),
            evidence={
                "model": "same_set",
                "status": "optimal",
                "group_count": len(groups),
                "model_size": builder.size,
                "per_key_model_size": _reference_model_size(n)["same_set"],
            },
        )
    if status == _INFEASIBLE:
        return Outcome(
            feasible=False,
            takes=None,
            evidence=_solver_diagnostics("same_set", result, conf, groups, builder.size, n),
        )
    raise SolverInconclusive(f"same-set model inconclusive: {getattr(result, 'message', status)}")


def _expand_same(
    templates: np.ndarray,
    groups: list[_Group],
    conf: Confirmations,
    values: np.ndarray,
    energy: np.ndarray,
    error: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    layout: dict[str, object],
    segment: int,
) -> Solution | None:
    pressed: list[int] = []
    gains_map: dict[int, float] = {}
    for gid, group in enumerate(groups):
        info = layout[gid]
        for key, var_info in info["confirmed"].items():  # type: ignore[index]
            x_idx, g1_idx, g2_idx = var_info
            g_idx = g1_idx if segment == 0 else g2_idx
            if int(np.rint(values[x_idx])) == 1:
                pressed.append(key)
                gains_map[key] = float(values[g_idx])
        if group.pool_size > 0:
            q = int(np.rint(values[info["q"]]))  # type: ignore[index]
            g_idx = info["g1"] if segment == 0 else info["g2"]  # type: ignore[index]
            chosen = list(group.pool[:q])
            gains_map.update(_split_pooled_gain(chosen, float(values[g_idx]), group.lo, group.hi))
            pressed.extend(chosen)
        for key in group.confirmed:
            if (segment, key) in conf and ((key in pressed) != conf[(segment, key)]):
                return None
    pressed.sort()
    return _build_solution(templates, pressed, gains_map, energy, error, lo, hi)


# ---------------------------------------------------------------------------
# different-set model: per-segment states with at least one differing key
# ---------------------------------------------------------------------------
def _different_set_model(
    templates: np.ndarray,
    b1: np.ndarray,
    e1: np.ndarray,
    b2: np.ndarray,
    e2: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    conf: Confirmations,
    groups: list[_Group],
    time_limit: float,
) -> Outcome:
    n, m = templates.shape
    builder = _MilpBuilder()
    layout: dict[str, object] = {"segment": 0}

    for gid, group in enumerate(groups):
        info: dict[str, object] = {"confirmed": {}, "indicators": []}
        indicators: list[int] = info["indicators"]  # type: ignore[assignment]
        # Confirmed keys: one binary and one gain per segment.  Each confirmed
        # key also contributes a difference indicator d = |x0 - x1|, since a
        # confirmed member can distinguish two takes even inside a pooled group.
        for key in sorted(group.confirmed):
            per_seg: dict[int, tuple[int, int]] = {}
            for seg in (0, 1):
                x = builder.var(0.0, 1.0, integer=True, cost=1.0)
                g = builder.var(0.0, group.hi, integer=False)
                pressed = conf.get((seg, key))
                if pressed is not None:
                    builder.fix(x, 1.0 if pressed else 0.0)
                builder.constraint({g: 1.0, x: -group.hi}, -_INF, 0.0)
                builder.constraint({x: group.lo, g: -1.0}, -_INF, 0.0)
                per_seg[seg] = (x, g)
            info["confirmed"][key] = per_seg
            d = builder.var(0.0, 1.0, integer=True)
            x0, x1 = per_seg[0][0], per_seg[1][0]
            builder.constraint({d: 1.0, x0: -1.0, x1: 1.0}, 0.0, _INF)
            builder.constraint({d: 1.0, x0: 1.0, x1: -1.0}, 0.0, _INF)
            builder.constraint({d: 1.0, x0: -1.0, x1: -1.0}, -_INF, 0.0)
            builder.constraint({d: 1.0, x0: 1.0, x1: 1.0}, -_INF, 2.0)
            indicators.append(d)

        u = group.pool_size
        if u > 0:
            # Pool members are unconfirmed in both takes by construction.
            qs: dict[int, int] = {}
            gs: dict[int, int] = {}
            for seg in (0, 1):
                q = builder.var(0.0, float(u), integer=True, cost=1.0)
                g = builder.var(0.0, u * group.hi, integer=False)
                builder.constraint({g: 1.0, q: -group.hi}, -_INF, 0.0)
                builder.constraint({q: group.lo, g: -1.0}, -_INF, 0.0)
                qs[seg] = q
                gs[seg] = g
            info["q"] = qs
            info["g"] = gs

            # Exact indicator that the two segments' pressed counts in the
            # pool differ.
            q0, q1 = qs[0], qs[1]
            if u == 1:
                # A singleton pool has binary counts, so the cheap exact
                # |q0 - q1| linearisation applies, matching one per-key block.
                d = builder.var(0.0, 1.0, integer=True)
                builder.constraint({d: 1.0, q0: -1.0, q1: 1.0}, 0.0, _INF)
                builder.constraint({d: 1.0, q0: 1.0, q1: -1.0}, 0.0, _INF)
                builder.constraint({d: 1.0, q0: -1.0, q1: -1.0}, -_INF, 0.0)
                builder.constraint({d: 1.0, q0: 1.0, q1: 1.0}, -_INF, 2.0)
                indicators.append(d)
                info["dp"], info["dm"] = d, d
            else:
                # For general integer counts in [0, u] two direction binaries
                # are needed: a single binary cannot be forced back to 0 when
                # the difference s = q0 - q1 is zero by linear rows alone.
                #   dp = 1 iff s >= 1 (take 0 presses more pool keys),
                #   dm = 1 iff -s >= 1 (take 1 presses more pool keys).
                # Each uses the standard two-row big-M with M = u, e.g. dp:
                #   s - u*dp <= 0  and  s - (u+1)*dp >= -u.
                dp = builder.var(0.0, 1.0, integer=True)
                dm = builder.var(0.0, 1.0, integer=True)
                builder.constraint({q0: 1.0, q1: -1.0, dp: float(-u)}, -_INF, 0.0)
                builder.constraint(
                    {q0: 1.0, q1: -1.0, dp: float(-(u + 1))}, float(-u), _INF
                )
                builder.constraint({q0: -1.0, q1: 1.0, dm: float(-u)}, -_INF, 0.0)
                builder.constraint(
                    {q0: -1.0, q1: 1.0, dm: float(-(u + 1))}, float(-u), _INF
                )
                indicators.append(dp)
                indicators.append(dm)
                info["dp"], info["dm"] = dp, dm

            # z = 1 allows equal counts realized by different member subsets.
            # Possible only for an interior count while an unconfirmed pool of
            # at least two keys remains: z = 1 forces 1 <= q_t <= u - 1.
            if u >= 2:
                z = builder.var(0.0, 1.0, integer=True)
                for q in qs.values():
                    # q - z >= 0: when z = 1, q >= 1.
                    builder.constraint({q: 1.0, z: -1.0}, 0.0, _INF)
                    # q + z <= u: when z = 1, q <= u - 1, so an interior
                    # count leaves room for two distinct q-sized subsets.
                    builder.constraint({q: 1.0, z: 1.0}, -_INF, float(u))
                indicators.append(z)
                info["z"] = z
            else:
                info["z"] = None
        layout[gid] = info

    # At least one group must realize a difference.  This is the analogue of
    # the original per-key model's single global sum-of-differences row.
    diff_row: dict[int, float] = {}
    for gid in range(len(groups)):
        for d in layout[gid]["indicators"]:  # type: ignore[index]
            diff_row[d] = 1.0
    builder.constraint(diff_row, 1.0, _INF)
    builder.global_rows += 1

    for seg, energy, error in ((0, b1, e1), (1, b2, e2)):
        for j in range(m):
            row: dict[int, float] = {}
            for gid, group in enumerate(groups):
                info = layout[gid]
                for key, per_seg in info["confirmed"].items():  # type: ignore[index]
                    g_idx = per_seg[seg][1]
                    row[g_idx] = row.get(g_idx, 0.0) + float(group.template[j])
                if group.pool_size > 0:
                    g_idx = info["g"][seg]  # type: ignore[index]
                    row[g_idx] = row.get(g_idx, 0.0) + float(group.template[j])
            builder.band_constraint(row, float(energy[j] - error[j]), float(energy[j] + error[j]))

    result = builder.solve(time_limit)
    status = int(getattr(result, "status", -1))
    if status == _OPTIMAL:
        values = np.asarray(result.x)
        sol1 = _expand_different(templates, groups, conf, values, b1, e1, lo, hi, layout, segment=0)
        sol2 = _expand_different(templates, groups, conf, values, b2, e2, lo, hi, layout, segment=1)
        if sol1 is None or sol2 is None or sol1.pressed == sol2.pressed:
            raise SolverInconclusive("different-set witness failed verification")
        return Outcome(
            feasible=True,
            takes=(sol1, sol2),
            evidence={
                "model": "different_set",
                "status": "optimal",
                "group_count": len(groups),
                "model_size": builder.size,
                "per_key_model_size": _reference_model_size(n)["different_set"],
            },
        )
    if status == _INFEASIBLE:
        return Outcome(
            feasible=False,
            takes=None,
            evidence=_solver_diagnostics("different_set", result, conf, groups, builder.size, n),
        )
    raise SolverInconclusive(f"different-set model inconclusive: {getattr(result, 'message', status)}")


def _expand_different(
    templates: np.ndarray,
    groups: list[_Group],
    conf: Confirmations,
    values: np.ndarray,
    energy: np.ndarray,
    error: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    layout: dict[str, object],
    segment: int,
) -> Solution | None:
    pressed: list[int] = []
    gains_map: dict[int, float] = {}
    for gid, group in enumerate(groups):
        info = layout[gid]
        for key, per_seg in info["confirmed"].items():  # type: ignore[index]
            x_idx, g_idx = per_seg[segment]
            if int(np.rint(values[x_idx])) == 1:
                pressed.append(key)
                gains_map[key] = float(values[g_idx])
        if group.pool_size > 0:
            q0 = int(np.rint(values[info["q"][0]]))  # type: ignore[index]
            q1 = int(np.rint(values[info["q"][1]]))  # type: ignore[index]
            q = q0 if segment == 0 else q1
            g_idx = info["g"][segment]  # type: ignore[index]
            z_idx = info.get("z")
            swap = z_idx is not None and int(np.rint(values[z_idx])) == 1
            # Two different member subsets are needed either when the pressed
            # counts differ (dp/dm) or when equal counts are swapped (z).
            # Segment 0 takes the first q pool members, segment 1 the last q;
            # both constructions stay inside the pool and differ whenever the
            # MILP required this group to supply the distinguishing key.
            distinct_here = swap or q0 != q1
            if distinct_here:
                u = group.pool_size
                start = 0 if segment == 0 else u - q
                chosen = list(group.pool[start : start + q])
            else:
                chosen = list(group.pool[:q])
            gains_map.update(_split_pooled_gain(chosen, float(values[g_idx]), group.lo, group.hi))
            pressed.extend(chosen)
        for key in group.confirmed:
            if (segment, key) in conf and ((key in pressed) != conf[(segment, key)]):
                return None
    pressed.sort()
    return _build_solution(templates, pressed, gains_map, energy, error, lo, hi)


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------
def classify(
    templates: np.ndarray,
    band_energy: tuple[np.ndarray, np.ndarray],
    band_error: tuple[np.ndarray, np.ndarray],
    gain_lower: np.ndarray,
    gain_upper: np.ndarray,
    confirmations: Confirmations | None = None,
    time_limit: float = 10.0,
) -> Verdict:
    """Classify the two takes and return witnesses for each feasible case.

    Raises :class:`SolverError` for malformed input and
    :class:`SolverInconclusive` if a model could not be decided within the
    time limit, since an undecided model cannot support a definite report.
    """
    templates = np.asarray(templates, dtype=float)
    b1, b2 = (np.asarray(v, dtype=float) for v in band_energy)
    e1, e2 = (np.asarray(v, dtype=float) for v in band_error)
    lo = np.asarray(gain_lower, dtype=float)
    hi = np.asarray(gain_upper, dtype=float)

    _validate_templates(templates)
    n, m = templates.shape
    _validate_energy("band_energy[0]", b1, m)
    _validate_energy("band_energy[1]", b2, m)
    _validate_error("band_error[0]", e1, m)
    _validate_error("band_error[1]", e2, m)
    _validate_gains(lo, hi, n)
    conf: Confirmations = dict(confirmations or {})
    _check_confirmations(conf, n)

    groups = _build_groups(templates, lo, hi, conf)

    same = _same_set_model(templates, b1, e1, b2, e2, lo, hi, conf, groups, time_limit)
    different = _different_set_model(templates, b1, e1, b2, e2, lo, hi, conf, groups, time_limit)

    possibilities: dict[str, tuple[Solution, Solution]] = {}
    if same.feasible and same.takes is not None:
        possibilities["same"] = same.takes
    if different.feasible and different.takes is not None:
        possibilities["different"] = different.takes

    if len(possibilities) == 2:
        kind: Literal["same", "different", "both", "impossible"] = "both"
    elif same.feasible:
        kind = "same"
    elif different.feasible:
        kind = "different"
    else:
        kind = "impossible"

    evidence: dict[str, object] = {
        "same_set_feasible": same.feasible,
        "different_set_feasible": different.feasible,
        "models": {"same": same.evidence, "different": different.evidence},
        "key_groups": _group_evidence(groups, conf),
        "group_count": len(groups),
    }
    if kind == "impossible":
        evidence["reason"] = "conflicting_evidence"
    return Verdict(kind=kind, possibilities=possibilities, evidence=evidence)
