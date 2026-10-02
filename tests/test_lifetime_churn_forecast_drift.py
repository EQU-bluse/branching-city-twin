import copy
import functools
import inspect
import math
import unittest

from city_twin.branches import BranchStore

from tests.test_lifetime_churn import churn_kwargs, diamond_store
from tests.test_lifetime_churn_forecast_backtests import (
    reference_backtests,
)
from tests.test_lifetime_churn_forecast_scorecards import (
    scorecards_kwargs,
)


def drift_kwargs(store_args, **overrides) -> dict:
    split_cutoff = overrides.pop("split_cutoff", 1)
    min_rate_drop = overrides.pop("min_rate_drop", (0, 1))
    drift_limit = overrides.pop("drift_limit", 50)
    kwargs = scorecards_kwargs(store_args, **overrides)
    kwargs["split_cutoff"] = split_cutoff
    kwargs["min_rate_drop"] = min_rate_drop
    kwargs["drift_limit"] = drift_limit
    return kwargs


def drift_call(store, store_args, **overrides):
    return store.lifetime_churn_forecast_drift(
        **drift_kwargs(store_args, **overrides)
    )


def _reduced(numerator: int, denominator: int) -> tuple[int, int]:
    divisor = math.gcd(numerator, denominator)
    return (numerator // divisor, denominator // divisor)


def _group(outcomes_results):
    grouped = {}
    encounter = []
    for cutoff_result in outcomes_results:
        for outcome in cutoff_result["outcomes"]:
            key = (outcome["type"], outcome["identity"])
            entry = grouped.get(key)
            if entry is None:
                entry = {
                    "type": outcome["type"],
                    "identity": outcome["identity"],
                    "predictions": 0,
                    "hit": 0,
                    "early": 0,
                    "late": 0,
                    "missed": 0,
                    "unresolved": 0,
                    "offsets": [],
                }
                grouped[key] = entry
                encounter.append(key)
            entry["predictions"] += 1
            entry[outcome["status"]] += 1
            if outcome["actual"] is not None:
                entry["offsets"].append(
                    outcome["actual"] - outcome["predicted"]
                )
    return encounter, grouped


def _scorecard(entry):
    resolved = entry["predictions"] - entry["unresolved"]
    observed = entry["hit"] + entry["early"] + entry["late"]
    if entry["offsets"]:
        mean_offset = _reduced(sum(entry["offsets"]), len(entry["offsets"]))
        max_abs_offset = max(abs(o) for o in entry["offsets"])
    else:
        mean_offset = None
        max_abs_offset = None
    return {
        "type": entry["type"],
        "identity": entry["identity"],
        "predictions": entry["predictions"],
        "resolved": resolved,
        "unresolved": entry["unresolved"],
        "hit": entry["hit"],
        "early": entry["early"],
        "late": entry["late"],
        "missed": entry["missed"],
        "observed": observed,
        "exact_rate": _reduced(entry["hit"], resolved),
        "window_rate": _reduced(observed, resolved),
        "mean_offset": mean_offset,
        "max_abs_offset": max_abs_offset,
    }


def reference_drift(
    backtests: dict[str, object],
    split_cutoff: int,
    min_resolved: int,
    min_rate_drop: tuple[int, int],
) -> dict[str, object]:
    """Independently split a backtests result and compute the drifts."""
    cutoff_results = list(backtests["backtests"])
    encounter, _full = _group(cutoff_results)
    _, baseline = _group(
        [r for r in cutoff_results if r["cutoff"] < split_cutoff]
    )
    _, recent = _group(
        [r for r in cutoff_results if r["cutoff"] >= split_cutoff]
    )

    threshold_num, threshold_den = min_rate_drop
    drifts = []
    for key in encounter:
        baseline_entry = baseline.get(key)
        recent_entry = recent.get(key)
        if baseline_entry is None or recent_entry is None:
            continue
        baseline_resolved = (
            baseline_entry["predictions"] - baseline_entry["unresolved"]
        )
        recent_resolved = (
            recent_entry["predictions"] - recent_entry["unresolved"]
        )
        if baseline_resolved < min_resolved or recent_resolved < min_resolved:
            continue
        baseline_observed = (
            baseline_entry["hit"]
            + baseline_entry["early"]
            + baseline_entry["late"]
        )
        recent_observed = (
            recent_entry["hit"]
            + recent_entry["early"]
            + recent_entry["late"]
        )
        window_num = (
            baseline_observed * recent_resolved
            - recent_observed * baseline_resolved
        )
        window_den = baseline_resolved * recent_resolved
        if window_num * threshold_den < threshold_num * window_den:
            continue
        exact_num = (
            baseline_entry["hit"] * recent_resolved
            - recent_entry["hit"] * baseline_resolved
        )
        if baseline_entry["offsets"] and recent_entry["offsets"]:
            shift_num = (
                sum(recent_entry["offsets"]) * len(baseline_entry["offsets"])
                - sum(baseline_entry["offsets"])
                * len(recent_entry["offsets"])
            )
            shift_den = len(recent_entry["offsets"]) * len(
                baseline_entry["offsets"]
            )
            shift = _reduced(shift_num, shift_den)
        else:
            shift = None
        drifts.append(
            {
                "type": baseline_entry["type"],
                "identity": baseline_entry["identity"],
                "baseline": _scorecard(baseline_entry),
                "recent": _scorecard(recent_entry),
                "exact_drop": _reduced(exact_num, window_den),
                "window_drop": _reduced(window_num, window_den),
                "mean_offset_shift": shift,
            }
        )

    def compare(left, right):
        for field in ("window_drop", "exact_drop"):
            left_num, left_den = left[field]
            right_num, right_den = right[field]
            if left_num * right_den != right_num * left_den:
                return (
                    -1
                    if left_num * right_den > right_num * left_den
                    else 1
                )
        left_offset = left["recent"]["max_abs_offset"]
        right_offset = right["recent"]["max_abs_offset"]
        left_key = left_offset if left_offset is not None else -1
        right_key = right_offset if right_offset is not None else -1
        if left_key != right_key:
            return -1 if left_key > right_key else 1
        return 0

    drifts.sort(key=functools.cmp_to_key(compare))
    totals = {
        "identities": 0,
        "baseline_predictions": 0,
        "recent_predictions": 0,
        "baseline_resolved": 0,
        "recent_resolved": 0,
    }
    for drift in drifts:
        totals["identities"] += 1
        totals["baseline_predictions"] += drift["baseline"]["predictions"]
        totals["recent_predictions"] += drift["recent"]["predictions"]
        totals["baseline_resolved"] += drift["baseline"]["resolved"]
        totals["recent_resolved"] += drift["recent"]["resolved"]
    return {"drifts": tuple(drifts), "totals": totals}


def _outcome(
    identity,
    status,
    type_name="node",
    ordinal=1,
    predicted=3,
    actual=None,
):
    return {
        "type": type_name,
        "identity": identity,
        "ordinal": ordinal,
        "predicted": predicted,
        "earliest": 2,
        "latest": 5,
        "actual": actual,
        "status": status,
    }


def _cutoff_result(cutoff, outcomes):
    totals = {
        "predictions": 0,
        "hit": 0,
        "early": 0,
        "late": 0,
        "missed": 0,
        "unresolved": 0,
    }
    for outcome in outcomes:
        totals["predictions"] += 1
        totals[outcome["status"]] += 1
    return {
        "cutoff": cutoff,
        "outcomes": tuple(outcomes),
        "totals": totals,
    }


def synthetic_backtests():
    """Two-period fixture pinning drops, shifts, filtering and ordering.

    Cutoff 1 is the baseline period and cutoffs 2 and 3 the recent one
    at split 2. A and B degrade by a half window rate, H and I mirror
    each other exactly, C degrades by a quarter, D improves, E never
    resolves recently, F is baseline-only and J is recent-only.
    """
    return [
        _cutoff_result(
            1,
            [
                _outcome("A", "hit", predicted=3, actual=3),
                _outcome("A", "late", ordinal=2, predicted=3, actual=4),
                _outcome("A", "hit", ordinal=3, predicted=4, actual=4),
                _outcome("A", "missed", ordinal=4),
                _outcome("B", "hit", predicted=3, actual=3),
                _outcome("B", "hit", ordinal=2, predicted=3, actual=3),
                _outcome("C", "hit", predicted=3, actual=3),
                _outcome("C", "missed", ordinal=2),
                _outcome("C", "missed", ordinal=3),
                _outcome("C", "hit", ordinal=4, predicted=3, actual=3),
                _outcome("D", "missed"),
                _outcome("D", "missed", ordinal=2),
                _outcome("E", "hit", predicted=3, actual=3),
                _outcome("F", "hit", predicted=3, actual=3),
                _outcome("F", "hit", ordinal=2, predicted=3, actual=3),
                _outcome("H", "hit", predicted=3, actual=3),
                _outcome("H", "hit", ordinal=2, predicted=3, actual=3),
                _outcome("H", "missed", ordinal=3),
                _outcome("H", "missed", ordinal=4),
                _outcome("I", "hit", predicted=3, actual=3),
                _outcome("I", "hit", ordinal=2, predicted=3, actual=3),
                _outcome("I", "missed", ordinal=3),
                _outcome("I", "missed", ordinal=4),
            ],
        ),
        _cutoff_result(
            2,
            [
                _outcome("A", "missed"),
                _outcome("A", "missed", ordinal=2),
                _outcome("A", "missed", ordinal=3),
                _outcome("A", "late", ordinal=4, predicted=3, actual=5),
                _outcome("B", "missed"),
                _outcome("B", "late", ordinal=2, predicted=3, actual=4),
                _outcome("C", "hit", predicted=3, actual=3),
                _outcome("C", "missed", ordinal=2),
                _outcome("C", "missed", ordinal=3),
                _outcome("C", "missed", ordinal=4),
                _outcome("D", "missed"),
                _outcome("D", "hit", ordinal=2, predicted=3, actual=3),
                _outcome("E", "unresolved"),
                _outcome("E", "unresolved", ordinal=2),
                _outcome("H", "missed"),
                _outcome("H", "missed", ordinal=2),
                _outcome("H", "missed", ordinal=3),
                _outcome("H", "missed", ordinal=4),
                _outcome("I", "missed"),
                _outcome("I", "missed", ordinal=2),
                _outcome("I", "missed", ordinal=3),
                _outcome("I", "missed", ordinal=4),
                _outcome("J", "hit", predicted=3, actual=3),
                _outcome("J", "hit", ordinal=2, predicted=3, actual=3),
            ],
        ),
    ]


def build_drift(min_resolved=2, min_rate_drop=(0, 1), drift_limit=50):
    return BranchStore._build_lifetime_churn_forecast_drift_result(
        synthetic_backtests(), 2, min_resolved, min_rate_drop, drift_limit
    )


class LifetimeChurnForecastDriftSignatureTests(unittest.TestCase):
    def test_public_signature_appends_three_parameters(self):
        parameters = list(
            inspect.signature(
                BranchStore.lifetime_churn_forecast_drift
            ).parameters.values()
        )
        self.assertEqual(
            [p.name for p in parameters],
            [
                "self",
                "reference",
                "series",
                "base_scenario",
                "axis",
                "values",
                "min_size",
                "max_size",
                "required",
                "exclusive_pairs",
                "limit",
                "windows",
                "causes",
                "direction",
                "depth",
                "node_limit",
                "change_limit",
                "lifetime_limit",
                "diff_limit",
                "window_limit",
                "total_diff_limit",
                "churn_limit",
                "streak_limit",
                "min_identities",
                "wave_limit",
                "min_waves",
                "recurrence_limit",
                "max_jitter",
                "periodicity_limit",
                "horizon",
                "forecast_limit",
                "cutoffs",
                "backtest_limit",
                "min_resolved",
                "scorecard_limit",
                "split_cutoff",
                "min_rate_drop",
                "drift_limit",
                "token",
            ],
        )
        self.assertEqual(
            [p.default for p in parameters[:-1]],
            [inspect.Parameter.empty] * (len(parameters) - 1),
        )
        self.assertIsNone(parameters[-1].default)

    def test_result_shape_and_key_order(self):
        store, args = diamond_store()
        result = drift_call(
            store,
            args,
            windows=((0,), (1,), (1,), (2,), (2,), (0,), (0,), (1,)),
            max_jitter=2,
            horizon=5,
            cutoffs=(2, 3),
            split_cutoff=3,
        )
        self.assertIsInstance(result, dict)
        self.assertEqual(list(result), ["drifts", "totals"])
        self.assertIsInstance(result["drifts"], tuple)
        self.assertEqual(
            list(result["totals"]),
            [
                "identities",
                "baseline_predictions",
                "recent_predictions",
                "baseline_resolved",
                "recent_resolved",
            ],
        )
        for drift in result["drifts"]:
            self.assertEqual(
                list(drift),
                [
                    "type",
                    "identity",
                    "baseline",
                    "recent",
                    "exact_drop",
                    "window_drop",
                    "mean_offset_shift",
                ],
            )
            for period in ("baseline", "recent"):
                self.assertEqual(
                    list(drift[period]),
                    [
                        "type",
                        "identity",
                        "predictions",
                        "resolved",
                        "unresolved",
                        "hit",
                        "early",
                        "late",
                        "missed",
                        "observed",
                        "exact_rate",
                        "window_rate",
                        "mean_offset",
                        "max_abs_offset",
                    ],
                )
            for name in ("exact_drop", "window_drop"):
                value = drift[name]
                self.assertIsInstance(value, tuple)
                self.assertEqual(len(value), 2)
                for part in value:
                    self.assertIsInstance(part, int)
                    self.assertNotIsInstance(part, bool)
            shift = drift["mean_offset_shift"]
            if shift is not None:
                self.assertIsInstance(shift, tuple)
                self.assertEqual(len(shift), 2)


class LifetimeChurnForecastDriftResultTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        return drift_call(self.store, self.args, **overrides)

    def evolution(self, **overrides):
        kwargs = churn_kwargs(self.args, **overrides)
        for name in (
            "churn_limit",
            "streak_limit",
            "min_identities",
            "wave_limit",
            "min_waves",
            "recurrence_limit",
            "max_jitter",
            "periodicity_limit",
            "horizon",
            "forecast_limit",
        ):
            kwargs.pop(name, None)
        return self.store.lifetime_evolution(**kwargs)

    def test_matches_independent_two_period_aggregation(self):
        from tests.test_lifetime_churn_waves import reference_waves

        cases = (
            (
                (0,), (1,), (1,), (2,), (2,), (0,), (0,), (1,),
            ),
            ((0,), (1,), (1,), (2,), (2,), (0,)),
            ((0,), (1,), (2,)),
            ((0, 2), (1,), (1,), (0, 1, 2)),
        )
        for windows in cases:
            evolution = self.evolution(windows=windows)
            last_wave = len(reference_waves(evolution, 1)["waves"]) - 1
            cutoff_choices = tuple(range(max(last_wave, 0)))
            if len(cutoff_choices) < 2:
                continue
            for max_jitter in (0, 2):
                for horizon in (1, 3, 6):
                    backtests = reference_backtests(
                        evolution, 1, max_jitter, horizon, cutoff_choices
                    )
                    for split_cutoff in cutoff_choices[1:]:
                        for min_resolved in (1, 2):
                            for min_rate_drop in ((0, 1), (1, 4), (1, 2)):
                                with self.subTest(
                                    windows=windows,
                                    max_jitter=max_jitter,
                                    horizon=horizon,
                                    split_cutoff=split_cutoff,
                                    min_resolved=min_resolved,
                                    min_rate_drop=min_rate_drop,
                                ):
                                    self.assertEqual(
                                        self.call(
                                            windows=windows,
                                            max_jitter=max_jitter,
                                            horizon=horizon,
                                            cutoffs=cutoff_choices,
                                            backtest_limit=100_000,
                                            min_resolved=min_resolved,
                                            scorecard_limit=100_000,
                                            split_cutoff=split_cutoff,
                                            min_rate_drop=min_rate_drop,
                                            drift_limit=100_000,
                                        ),
                                        reference_drift(
                                            backtests,
                                            split_cutoff,
                                            min_resolved,
                                            min_rate_drop,
                                        ),
                                    )

    def test_diamond_split_three_keeps_no_identity(self):
        # Every identity resolves only at cutoff 2, so the recent
        # period at split 3 has no resolved outcome for any of them.
        result = self.call(
            windows=((0,), (1,), (1,), (2,), (2,), (0,), (0,), (1,)),
            max_jitter=2,
            horizon=5,
            cutoffs=(2, 3),
            split_cutoff=3,
        )
        self.assertEqual(result["drifts"], ())
        self.assertEqual(
            result["totals"],
            {
                "identities": 0,
                "baseline_predictions": 0,
                "recent_predictions": 0,
                "baseline_resolved": 0,
                "recent_resolved": 0,
            },
        )


class LifetimeChurnForecastDriftAggregationTests(unittest.TestCase):
    """Synthetic cutoff results pin the drift aggregation directly."""

    def test_kept_identities_and_their_order(self):
        result = build_drift()
        self.assertEqual(
            [d["identity"] for d in result["drifts"]],
            ["B", "A", "H", "I", "C"],
        )

    def test_drop_and_shift_values(self):
        drifts = {d["identity"]: d for d in build_drift()["drifts"]}
        self.assertEqual(
            drifts["A"],
            {
                "type": "node",
                "identity": "A",
                "baseline": {
                    "type": "node",
                    "identity": "A",
                    "predictions": 4,
                    "resolved": 4,
                    "unresolved": 0,
                    "hit": 2,
                    "early": 0,
                    "late": 1,
                    "missed": 1,
                    "observed": 3,
                    "exact_rate": (1, 2),
                    "window_rate": (3, 4),
                    "mean_offset": (1, 3),
                    "max_abs_offset": 1,
                },
                "recent": {
                    "type": "node",
                    "identity": "A",
                    "predictions": 4,
                    "resolved": 4,
                    "unresolved": 0,
                    "hit": 0,
                    "early": 0,
                    "late": 1,
                    "missed": 3,
                    "observed": 1,
                    "exact_rate": (0, 1),
                    "window_rate": (1, 4),
                    "mean_offset": (2, 1),
                    "max_abs_offset": 2,
                },
                "exact_drop": (1, 2),
                "window_drop": (1, 2),
                "mean_offset_shift": (5, 3),
            },
        )
        self.assertEqual(drifts["B"]["exact_drop"], (1, 1))
        self.assertEqual(drifts["B"]["window_drop"], (1, 2))
        self.assertEqual(drifts["B"]["mean_offset_shift"], (1, 1))
        self.assertEqual(drifts["C"]["exact_drop"], (1, 4))
        self.assertEqual(drifts["C"]["window_drop"], (1, 4))
        self.assertEqual(drifts["C"]["mean_offset_shift"], (0, 1))

    def test_unobservable_recent_offset_gives_none_shift(self):
        drifts = {d["identity"]: d for d in build_drift()["drifts"]}
        self.assertIsNone(drifts["H"]["mean_offset_shift"])
        self.assertIsNone(drifts["H"]["recent"]["mean_offset"])
        self.assertIsNone(drifts["H"]["recent"]["max_abs_offset"])
        self.assertEqual(drifts["H"]["window_drop"], (1, 2))
        self.assertEqual(drifts["H"]["exact_drop"], (1, 2))

    def test_improving_identity_fails_the_zero_threshold(self):
        # D's window rate rises from 0 to 1/2, a negative drop.
        self.assertNotIn(
            "D", [d["identity"] for d in build_drift()["drifts"]]
        )

    def test_single_period_and_unresolved_identities_are_excluded(self):
        identities = [d["identity"] for d in build_drift()["drifts"]]
        self.assertNotIn("E", identities)  # unresolved in recent
        self.assertNotIn("F", identities)  # baseline only
        self.assertNotIn("J", identities)  # recent only

    def test_min_resolved_applies_to_both_periods(self):
        # B resolves only twice per period and drops out first.
        result = build_drift(min_resolved=3)
        self.assertEqual(
            [d["identity"] for d in result["drifts"]],
            ["A", "H", "I", "C"],
        )
        result = build_drift(min_resolved=5)
        self.assertEqual(result["drifts"], ())

    def test_min_rate_drop_threshold_uses_cross_multiplication(self):
        self.assertEqual(
            [d["identity"] for d in build_drift(min_rate_drop=(1, 2))["drifts"]],
            ["B", "A", "H", "I"],
        )
        # C's drop of exactly 1/4 reaches the 1/4 threshold.
        self.assertEqual(
            [d["identity"] for d in build_drift(min_rate_drop=(1, 4))["drifts"]],
            ["B", "A", "H", "I", "C"],
        )
        result = build_drift(min_rate_drop=(3, 4))
        self.assertEqual(result["drifts"], ())
        self.assertEqual(
            result["totals"],
            {
                "identities": 0,
                "baseline_predictions": 0,
                "recent_predictions": 0,
                "baseline_resolved": 0,
                "recent_resolved": 0,
            },
        )

    def test_totals_sum_only_the_kept_drifts(self):
        self.assertEqual(
            build_drift()["totals"],
            {
                "identities": 5,
                "baseline_predictions": 18,
                "recent_predictions": 18,
                "baseline_resolved": 18,
                "recent_resolved": 18,
            },
        )
        self.assertEqual(
            build_drift(min_rate_drop=(1, 2))["totals"],
            {
                "identities": 4,
                "baseline_predictions": 14,
                "recent_predictions": 14,
                "baseline_resolved": 14,
                "recent_resolved": 14,
            },
        )

    def test_drift_limit_raises_instead_of_truncating(self):
        with self.assertRaises(ValueError) as caught:
            build_drift(drift_limit=4)
        self.assertIn("drift limit", str(caught.exception))
        result = build_drift(drift_limit=5)
        self.assertEqual(len(result["drifts"]), 5)

    def test_empty_backtests_return_empty_drifts_and_zero_totals(self):
        result = BranchStore._build_lifetime_churn_forecast_drift_result(
            [], 1, 1, (0, 1), 1
        )
        self.assertEqual(result["drifts"], ())
        self.assertEqual(
            result["totals"],
            {
                "identities": 0,
                "baseline_predictions": 0,
                "recent_predictions": 0,
                "baseline_resolved": 0,
                "recent_resolved": 0,
            },
        )

    def test_result_shares_nothing_with_the_cutoff_results(self):
        first = ("a", ("b", 1))
        second = ("a", ("b", 2))
        backtests = [
            _cutoff_result(
                1,
                [
                    _outcome(first, "hit", predicted=3, actual=3),
                    _outcome(second, "hit", predicted=3, actual=3),
                ],
            ),
            _cutoff_result(
                2,
                [
                    _outcome(first, "missed"),
                    _outcome(second, "missed"),
                ],
            ),
        ]
        result = BranchStore._build_lifetime_churn_forecast_drift_result(
            backtests, 2, 1, (0, 1), 50
        )
        outcomes = [
            outcome
            for cutoff_result in backtests
            for outcome in cutoff_result["outcomes"]
        ]
        for drift in result["drifts"]:
            for outcome in outcomes:
                self.assertIsNot(drift, outcome)
                self.assertIsNot(drift["baseline"], outcome)
                self.assertIsNot(drift["recent"], outcome)
                self.assertIsNot(
                    drift["identity"], outcome["identity"]
                )
                self.assertIsNot(
                    drift["baseline"]["identity"], outcome["identity"]
                )
                self.assertIsNot(
                    drift["recent"]["identity"], outcome["identity"]
                )

    def test_drift_identity_is_an_isolated_copy(self):
        identity = ("left", ("right", "leaf"))
        backtests = [
            _cutoff_result(
                1,
                [
                    _outcome(
                        identity, "hit", type_name="gap",
                        predicted=3, actual=3,
                    ),
                ],
            ),
            _cutoff_result(
                2,
                [
                    _outcome(
                        identity, "missed", type_name="gap",
                    ),
                ],
            ),
        ]
        drift = BranchStore._build_lifetime_churn_forecast_drift_result(
            backtests, 2, 1, (0, 1), 1
        )["drifts"][0]
        self.assertEqual(drift["identity"], identity)
        self.assertIsNot(drift["identity"], identity)
        self.assertIsNot(drift["baseline"]["identity"], identity)
        self.assertIsNot(drift["recent"]["identity"], identity)
        self.assertIsNot(
            drift["baseline"]["identity"], drift["recent"]["identity"]
        )


class LifetimeChurnForecastDriftValidationTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        overrides.setdefault("causes", ("a0",))
        return drift_call(self.store, self.args, **overrides)

    def extended(self, **overrides):
        overrides.setdefault(
            "windows",
            ((0,), (1,), (1,), (2,), (2,), (0,), (0,), (1,)),
        )
        overrides.setdefault("horizon", 5)
        overrides.setdefault("max_jitter", 2)
        overrides.setdefault("cutoffs", (2, 3))
        overrides.setdefault("backtest_limit", 50)
        overrides.setdefault("split_cutoff", 3)
        return self.call(**overrides)

    def test_cutoffs_must_not_be_empty(self):
        with self.assertRaises(ValueError):
            self.call(cutoffs=())

    def test_split_cutoff_must_be_a_non_bool_int(self):
        for bad in (True, False, 1.0, "1", None, (1,)):
            with self.subTest(split_cutoff=bad):
                with self.assertRaises(TypeError):
                    self.extended(split_cutoff=bad)

    def test_split_cutoff_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            self.extended(split_cutoff=1)
        with self.assertRaises(ValueError):
            self.extended(split_cutoff=4)
        with self.assertRaises(ValueError):
            self.extended(split_cutoff=99)

    def test_split_cutoff_must_leave_both_periods_nonempty(self):
        # A split at the first cutoff empties the baseline period.
        with self.assertRaises(ValueError):
            self.extended(split_cutoff=2)
        # The last cutoff is a valid split: recent is that cutoff only.
        result = self.extended(split_cutoff=3)
        self.assertEqual(result["drifts"], ())

    def test_min_rate_drop_must_be_an_int_pair(self):
        for bad in (
            [0, 1],
            "01",
            0,
            (0,),
            (0, 1, 2),
            (True, 1),
            (0, False),
            (0.5, 1),
            (1, "2"),
            (None, 1),
        ):
            with self.subTest(min_rate_drop=bad):
                with self.assertRaises(TypeError):
                    self.extended(min_rate_drop=bad)

    def test_min_rate_drop_value_bounds(self):
        for bad in ((-1, 2), (1, 0), (1, -2), (3, 2), (2, 1)):
            with self.subTest(min_rate_drop=bad):
                with self.assertRaises(ValueError):
                    self.extended(min_rate_drop=bad)
        # Boundary ratios are accepted.
        self.extended(min_rate_drop=(0, 1))
        self.extended(min_rate_drop=(1, 1))

    def test_drift_limit_must_be_a_non_bool_positive_int(self):
        for bad in (True, False, 1.0, "1", None):
            with self.subTest(drift_limit=bad):
                with self.assertRaises(TypeError):
                    self.extended(drift_limit=bad)
        with self.assertRaises(ValueError):
            self.extended(drift_limit=0)
        with self.assertRaises(ValueError):
            self.extended(drift_limit=-4)

    def test_new_parameters_validated_in_order_after_scorecard_limit(self):
        # scorecard_limit fails before split_cutoff.
        with self.assertRaises(ValueError):
            self.extended(scorecard_limit=0, split_cutoff="x")
        with self.assertRaises(TypeError):
            self.extended(scorecard_limit="x", split_cutoff="x")
        # split_cutoff fails before min_rate_drop.
        with self.assertRaises(TypeError):
            self.extended(split_cutoff="x", min_rate_drop="x")
        with self.assertRaises(ValueError):
            self.extended(split_cutoff=99, min_rate_drop="x")
        # min_rate_drop fails before drift_limit.
        with self.assertRaises(TypeError):
            self.extended(min_rate_drop="x", drift_limit="x")
        with self.assertRaises(ValueError):
            self.extended(min_rate_drop=(1, 0), drift_limit="x")
        # Once min_rate_drop passes, drift_limit's error fires.
        with self.assertRaises(TypeError):
            self.extended(min_rate_drop=(0, 1), drift_limit="x")
        with self.assertRaises(ValueError):
            self.extended(min_rate_drop=(0, 1), drift_limit=0)
        # Empty cutoffs fail before the split_cutoff checks.
        with self.assertRaises(ValueError):
            self.call(cutoffs=(), split_cutoff="x")
        # All parameter checks still precede every state lookup.
        with self.assertRaises(TypeError):
            self.call(reference="ghost", cutoffs=(0, 1), split_cutoff="x")
        with self.assertRaises(ValueError):
            self.call(reference="ghost", cutoffs=(0, 1), split_cutoff=99)
        with self.assertRaises(TypeError):
            self.call(
                token="whatever", cutoffs=(0, 1), split_cutoff=1,
                min_rate_drop="x",
            )
        with self.assertRaises(ValueError):
            self.call(
                token="whatever", cutoffs=(0, 1), split_cutoff=1,
                drift_limit=0,
            )

    def test_shared_validation_errors_match_scorecards(self):
        with self.assertRaises(TypeError):
            self.call(cutoffs="x")
        with self.assertRaises(ValueError):
            self.call(cutoffs=(-1,))
        with self.assertRaises(ValueError):
            self.call(windows=((-1,),))
        with self.assertRaises(ValueError) as caught:
            self.extended(cutoffs=(2, 99), split_cutoff=3)
        self.assertIn("out of range", str(caught.exception))
        with self.assertRaises(ValueError):
            self.extended(min_resolved=0)
        with self.assertRaises(ValueError):
            self.extended(backtest_limit=0)


class LifetimeChurnForecastDriftStateTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        overrides.setdefault("causes", ("a0",))
        return drift_call(self.store, self.args, **overrides)

    def test_unknown_branch_history_node_and_cause_raise(self):
        with self.assertRaises(KeyError):
            self.call(cutoffs=(0, 1), split_cutoff=1, reference="ghost")
        with self.assertRaises(KeyError):
            self.call(
                series={
                    "a": (("r0", "a0"), ("r1", "a0"), ("r2", "a0")),
                    "b": (("r0", "a0"), ("r1", "W"), ("r2", "nope2")),
                },
                cutoffs=(0, 1),
                split_cutoff=1,
            )
        with self.assertRaises(KeyError) as caught:
            self.call(causes=("zzz",), cutoffs=(0, 1), split_cutoff=1)
        self.assertEqual(caught.exception.args[0], "zzz")

    def test_range_error_fires_after_the_state_checks(self):
        with self.assertRaises(KeyError):
            self.call(reference="ghost", cutoffs=(0, 99), split_cutoff=1)


class LifetimeChurnForecastDriftIsolationTests(unittest.TestCase):
    EXTENDED = (
        (0,), (1,), (1,), (2,), (2,), (0,), (0,), (1,),
    )

    def call(self, store, args, **overrides):
        overrides.setdefault("windows", self.EXTENDED)
        overrides.setdefault("max_jitter", 2)
        overrides.setdefault("horizon", 5)
        overrides.setdefault("cutoffs", (2, 3))
        overrides.setdefault("split_cutoff", 3)
        return drift_call(store, args, **overrides)

    def test_result_levels_are_mutually_unshared(self):
        store, args = diamond_store()
        result = self.call(store, args)
        seen_ids = []

        def collect(value):
            if isinstance(value, dict):
                seen_ids.append(id(value))
                for item in value.values():
                    collect(item)
            elif isinstance(value, (tuple, list)):
                for item in value:
                    collect(item)

        collect(result)
        self.assertEqual(len(seen_ids), len(set(seen_ids)))

    def test_mutating_result_never_affects_later_calls(self):
        store, args = diamond_store()
        first = self.call(store, args)
        pristine = copy.deepcopy(first)
        for drift in first["drifts"]:
            drift["window_drop"] = (9, 9)
            drift["baseline"]["predictions"] = 99
            drift["recent"]["resolved"] = 99
        first["totals"]["identities"] = 99
        self.assertEqual(self.call(store, args), pristine)

    def test_success_and_failure_change_nothing(self):
        store, args = diamond_store()
        heads = dict(store._heads)
        appends = copy.deepcopy(store._appends)
        merges = copy.deepcopy(store._merges)
        audit = store.audit_log()

        self.call(store, args)
        failures = [
            dict(windows="x"),
            dict(node_limit=0),
            dict(lifetime_limit=0),
            dict(window_limit=0),
            dict(churn_limit=0),
            dict(max_jitter=-1),
            dict(horizon=0),
            dict(forecast_limit=0),
            dict(cutoffs="x"),
            dict(cutoffs=(True,)),
            dict(cutoffs=(-1,)),
            dict(cutoffs=(1, 1)),
            dict(cutoffs=()),
            dict(backtest_limit=0),
            dict(min_resolved=0),
            dict(min_resolved="x"),
            dict(scorecard_limit=0),
            dict(split_cutoff=True),
            dict(split_cutoff="x"),
            dict(split_cutoff=1),
            dict(split_cutoff=2),
            dict(split_cutoff=99),
            dict(min_rate_drop="x"),
            dict(min_rate_drop=(0,)),
            dict(min_rate_drop=(True, 1)),
            dict(min_rate_drop=(-1, 2)),
            dict(min_rate_drop=(1, 0)),
            dict(min_rate_drop=(3, 2)),
            dict(drift_limit=0),
            dict(drift_limit="x"),
            dict(drift_limit=False),
            dict(cutoffs=(2, 99), split_cutoff=3),
            dict(cutoffs=(2, 3), backtest_limit=1),
            dict(causes=("zzz",)),
            dict(limit=1),
            dict(reference="ghost"),
        ]
        for overrides in failures:
            with self.subTest(overrides=overrides):
                params = dict(
                    causes=("a0",),
                    windows=self.EXTENDED,
                    max_jitter=2,
                    horizon=5,
                    cutoffs=(2, 3),
                    split_cutoff=3,
                )
                params.update(overrides)
                with self.assertRaises(Exception):
                    drift_call(store, args, **params)

        self.assertEqual(dict(store._heads), heads)
        self.assertEqual(copy.deepcopy(store._appends), appends)
        self.assertEqual(copy.deepcopy(store._merges), merges)
        self.assertEqual(store.audit_log(), audit)

    def test_existing_public_entries_are_unaffected(self):
        store, args = diamond_store()
        self.call(store, args)
        from tests.test_lifetime_churn_forecast_backtests import (
            backtests_call,
        )
        from tests.test_lifetime_churn_forecast_scorecards import (
            scorecards_call,
        )
        from tests.test_lifetime_churn_forecasts import forecasts_call

        backtests = backtests_call(
            store,
            args,
            windows=self.EXTENDED,
            max_jitter=2,
            horizon=5,
            cutoffs=(2, 3),
        )
        scorecards = scorecards_call(
            store,
            args,
            windows=self.EXTENDED,
            max_jitter=2,
            horizon=5,
            cutoffs=(2, 3),
        )
        forecasts = forecasts_call(
            store, args, windows=self.EXTENDED, max_jitter=2, horizon=5
        )
        self.assertEqual(
            backtests_call(
                store,
                args,
                windows=self.EXTENDED,
                max_jitter=2,
                horizon=5,
                cutoffs=(2, 3),
            ),
            backtests,
        )
        self.assertEqual(
            scorecards_call(
                store,
                args,
                windows=self.EXTENDED,
                max_jitter=2,
                horizon=5,
                cutoffs=(2, 3),
            ),
            scorecards,
        )
        self.assertEqual(
            forecasts_call(
                store, args, windows=self.EXTENDED, max_jitter=2, horizon=5
            ),
            forecasts,
        )


class LifetimeChurnForecastDriftTokenTests(unittest.TestCase):
    EXTENDED = (
        (0,), (1,), (1,), (2,), (2,), (0,), (0,), (1,),
    )

    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, token=None, **overrides):
        overrides.setdefault("windows", self.EXTENDED)
        overrides.setdefault("max_jitter", 2)
        overrides.setdefault("horizon", 5)
        overrides.setdefault("cutoffs", (2, 3))
        overrides.setdefault("split_cutoff", 3)
        return drift_call(self.store, self.args, token=token, **overrides)

    def test_success_consumes_one_read_and_failure_refunds(self):
        token = self.store.create_snapshot(2)
        self.call(token=token)
        # Over the backtest cap: failure refunds.
        with self.assertRaises(ValueError):
            self.call(token=token, backtest_limit=1)
        # Out-of-range cutoff: failure refunds.
        with self.assertRaises(ValueError):
            self.call(token=token, cutoffs=(2, 99), split_cutoff=3)
        # State error: failure refunds.
        with self.assertRaises(KeyError):
            self.call(token=token, causes=("zzz",))
        # Parameter validation runs before the token is touched.
        with self.assertRaises(ValueError):
            self.call(token=token, cutoffs=())
        with self.assertRaises(TypeError):
            self.call(token=token, split_cutoff="x")
        with self.assertRaises(TypeError):
            self.call(token=token, min_rate_drop="x")
        with self.assertRaises(ValueError):
            self.call(token=token, drift_limit=0)
        # One read remains after the refunded failures.
        self.call(token=token)
        with self.assertRaises(RuntimeError):
            self.call(token=token)

    def test_results_come_from_the_frozen_view(self):
        before = self.call()
        token = self.store.create_snapshot(4)
        self.store.append("a", "a2", 7, {"x": -1})
        self.store.create("d", "root")
        self.store.append("d", "d0", 8, {"z": 3})
        self.store.merge("a", "d", "M", 9, {"z": 1})
        self.store.append("main", "r3", 10, {"x": 1})
        self.assertEqual(self.call(token=token), before)


if __name__ == "__main__":
    unittest.main()
