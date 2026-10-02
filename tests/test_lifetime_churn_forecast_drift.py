import copy
import inspect
import unittest
from functools import cmp_to_key

from city_twin.branches import BranchStore

from tests.test_lifetime_churn import churn_kwargs, diamond_store
from tests.test_lifetime_churn_forecast_backtests import (
    backtests_call,
    reference_backtests,
)
from tests.test_lifetime_churn_forecast_scorecards import (
    _reduced,
    reference_scorecards,
    scorecards_call,
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


def _subtract(left: tuple[int, int], right: tuple[int, int]) -> tuple[int, int]:
    return _reduced(
        left[0] * right[1] - right[0] * left[1],
        left[1] * right[1],
    )


def reference_drift(
    backtests: dict[str, object],
    split_cutoff: int,
    min_resolved: int,
    min_rate_drop: tuple[int, int],
) -> dict[str, object]:
    """Independently aggregate a backtests result into drift records."""
    baseline_results = [
        cutoff_result
        for cutoff_result in backtests["backtests"]
        if cutoff_result["cutoff"] < split_cutoff
    ]
    recent_results = [
        cutoff_result
        for cutoff_result in backtests["backtests"]
        if cutoff_result["cutoff"] >= split_cutoff
    ]
    baseline = reference_scorecards(
        {"backtests": baseline_results}, min_resolved
    )["scorecards"]
    recent = reference_scorecards(
        {"backtests": recent_results}, min_resolved
    )["scorecards"]
    baseline_by_key = {
        (s["type"], s["identity"]): s for s in baseline
    }
    recent_by_key = {(s["type"], s["identity"]): s for s in recent}

    encounter = []
    seen = set()
    for cutoff_result in backtests["backtests"]:
        for outcome in cutoff_result["outcomes"]:
            key = (outcome["type"], outcome["identity"])
            if key not in seen:
                seen.add(key)
                encounter.append(key)

    drifts = []
    for key in encounter:
        base = baseline_by_key.get(key)
        late = recent_by_key.get(key)
        if base is None or late is None:
            continue
        exact_drop = _subtract(base["exact_rate"], late["exact_rate"])
        window_drop = _subtract(base["window_rate"], late["window_rate"])
        if (
            window_drop[0] * min_rate_drop[1]
            < min_rate_drop[0] * window_drop[1]
        ):
            continue
        if base["mean_offset"] is None or late["mean_offset"] is None:
            shift = None
        else:
            shift = _subtract(late["mean_offset"], base["mean_offset"])
        drifts.append(
            {
                "type": key[0],
                "identity": key[1],
                "baseline": base,
                "recent": late,
                "exact_drop": exact_drop,
                "window_drop": window_drop,
                "mean_offset_shift": shift,
            }
        )

    def compare(left, right):
        for name in ("window_drop", "exact_drop"):
            left_cross = left[name][0] * right[name][1]
            right_cross = right[name][0] * left[name][1]
            if left_cross != right_cross:
                return -1 if left_cross > right_cross else 1
        left_abs = left["recent"]["max_abs_offset"]
        right_abs = right["recent"]["max_abs_offset"]
        left_key = -1 if left_abs is None else left_abs
        right_key = -1 if right_abs is None else right_abs
        if left_key != right_key:
            return -1 if left_key > right_key else 1
        return 0

    drifts.sort(key=cmp_to_key(compare))
    totals = {
        "identities": 0,
        "baseline_predictions": 0,
        "recent_predictions": 0,
        "baseline_resolved": 0,
        "recent_resolved": 0,
    }
    for record in drifts:
        totals["identities"] += 1
        totals["baseline_predictions"] += record["baseline"]["predictions"]
        totals["recent_predictions"] += record["recent"]["predictions"]
        totals["baseline_resolved"] += record["baseline"]["resolved"]
        totals["recent_resolved"] += record["recent"]["resolved"]
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
        "latest": 4,
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
    """Four cutoffs pinning periods, drops, shifts, filtering and order.

    Split at 3: cutoffs 1 and 2 are the baseline period, 3 and 4 the
    recent period. A degrades (window rate 1 -> 1/4), B is steady at
    1/2, F and G degrade identically (1/2 -> 0) with no recent offset,
    H degrades to 1/6 so its drop needs reduction, C never resolves in
    the recent period, D improves (negative drop) and E is never
    observed. First-appearance order is A, B, F, G, H, C, D, E.
    """
    return [
        _cutoff_result(
            1,
            [
                _outcome("A", "hit", predicted=3, actual=3),
                _outcome("A", "hit", ordinal=2, predicted=4, actual=4),
                _outcome("B", "hit", predicted=3, actual=3),
                _outcome("F", "hit", predicted=3, actual=5),
                _outcome("G", "hit", predicted=3, actual=2),
                _outcome("H", "hit", predicted=3, actual=3),
                _outcome("C", "hit", predicted=3, actual=3),
                _outcome("C", "hit", ordinal=2, predicted=4, actual=4),
                _outcome("D", "missed"),
                _outcome("E", "missed"),
            ],
        ),
        _cutoff_result(
            2,
            [
                _outcome("A", "early", ordinal=3, predicted=3, actual=2),
                _outcome("A", "late", ordinal=4, predicted=3, actual=4),
                _outcome("B", "missed", ordinal=2),
                _outcome("F", "missed", ordinal=2),
                _outcome("G", "missed", ordinal=2),
                _outcome("H", "missed", ordinal=2),
                _outcome("C", "hit", ordinal=3, predicted=3, actual=3),
                _outcome("D", "missed", ordinal=2),
                _outcome("E", "missed", ordinal=2),
            ],
        ),
        _cutoff_result(
            3,
            [
                _outcome("A", "missed", ordinal=5),
                _outcome("A", "missed", ordinal=6),
                _outcome("B", "hit", ordinal=3, predicted=3, actual=3),
                _outcome("F", "missed", ordinal=3),
                _outcome("G", "missed", ordinal=3),
                _outcome("H", "missed", ordinal=3),
                _outcome("H", "missed", ordinal=4),
                _outcome("H", "missed", ordinal=5),
                _outcome("C", "unresolved", ordinal=4),
                _outcome("D", "hit", ordinal=3, predicted=3, actual=3),
                _outcome("E", "missed", ordinal=3),
            ],
        ),
        _cutoff_result(
            4,
            [
                _outcome("A", "missed", ordinal=7),
                _outcome("A", "hit", ordinal=8, predicted=3, actual=3),
                _outcome("B", "missed", ordinal=4),
                _outcome("F", "missed", ordinal=4),
                _outcome("G", "missed", ordinal=4),
                _outcome("H", "missed", ordinal=6),
                _outcome("H", "missed", ordinal=7),
                _outcome("H", "hit", ordinal=8, predicted=3, actual=3),
                _outcome("C", "unresolved", ordinal=5),
                _outcome("D", "hit", ordinal=4, predicted=3, actual=3),
                _outcome("E", "missed", ordinal=4),
            ],
        ),
    ]


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
            windows=LONG_WINDOWS,
            max_jitter=2,
            horizon=1,
            cutoffs=(2, 3, 4, 5, 6),
            split_cutoff=4,
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
        self.assertTrue(result["drifts"])
        scorecard_keys = [
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
        ]
        for record in result["drifts"]:
            self.assertEqual(
                list(record),
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
            self.assertEqual(list(record["baseline"]), scorecard_keys)
            self.assertEqual(list(record["recent"]), scorecard_keys)
            for name in ("exact_drop", "window_drop", "mean_offset_shift"):
                value = record[name]
                if value is None:
                    continue
                self.assertIsInstance(value, tuple)
                self.assertEqual(len(value), 2)
                for part in value:
                    self.assertIsInstance(part, int)
                    self.assertNotIsInstance(part, bool)


LONG_WINDOWS = (
    (0,), (1,), (1,), (2,), (2,), (0,), (0,), (1,),
    (1,), (2,), (2,), (0,), (0,), (1,), (1,), (2,),
)


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

    def test_matches_independent_aggregation_of_backtests(self):
        from tests.test_lifetime_churn_waves import reference_waves

        cases = (
            ((0,), (1,), (1,), (2,), (2,), (0,), (0,), (1,)),
            LONG_WINDOWS,
            ((0, 2), (1,), (1,), (0, 1, 2), (2,), (2,), (0,)),
        )
        for windows in cases:
            evolution = self.evolution(windows=windows)
            last_wave = len(reference_waves(evolution, 1)["waves"]) - 1
            cutoff_choices = tuple(range(max(last_wave, 0)))
            if len(cutoff_choices) < 2:
                continue
            for max_jitter in (0, 2):
                for horizon in (1, 3):
                    for min_resolved in (1, 2):
                        for split_cutoff in cutoff_choices[1:]:
                            for min_rate_drop in ((0, 1), (1, 3), (1, 1)):
                                with self.subTest(
                                    windows=windows,
                                    max_jitter=max_jitter,
                                    horizon=horizon,
                                    min_resolved=min_resolved,
                                    split_cutoff=split_cutoff,
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
                                            scorecard_limit=10_000,
                                            split_cutoff=split_cutoff,
                                            min_rate_drop=min_rate_drop,
                                            drift_limit=10_000,
                                        ),
                                        reference_drift(
                                            reference_backtests(
                                                evolution,
                                                1,
                                                max_jitter,
                                                horizon,
                                                cutoff_choices,
                                            ),
                                            split_cutoff,
                                            min_resolved,
                                            min_rate_drop,
                                        ),
                                    )

    def test_long_windows_pin_each_steady_identity(self):
        result = self.call(
            windows=LONG_WINDOWS,
            max_jitter=2,
            horizon=1,
            cutoffs=(2, 3, 4, 5, 6),
            split_cutoff=4,
        )
        self.assertEqual(
            [record["identity"] for record in result["drifts"]],
            [("W", "y"), ("a0", "x"), (("a0", "x"), ("W", "y"))],
        )
        for record in result["drifts"]:
            self.assertEqual(
                record["baseline"],
                {
                    "type": record["type"],
                    "identity": record["identity"],
                    "predictions": 2,
                    "resolved": 2,
                    "unresolved": 0,
                    "hit": 2,
                    "early": 0,
                    "late": 0,
                    "missed": 0,
                    "observed": 2,
                    "exact_rate": (1, 1),
                    "window_rate": (1, 1),
                    "mean_offset": (0, 1),
                    "max_abs_offset": 0,
                },
            )
            self.assertEqual(
                record["recent"],
                {
                    "type": record["type"],
                    "identity": record["identity"],
                    "predictions": 3,
                    "resolved": 3,
                    "unresolved": 0,
                    "hit": 3,
                    "early": 0,
                    "late": 0,
                    "missed": 0,
                    "observed": 3,
                    "exact_rate": (1, 1),
                    "window_rate": (1, 1),
                    "mean_offset": (0, 1),
                    "max_abs_offset": 0,
                },
            )
            self.assertEqual(record["exact_drop"], (0, 1))
            self.assertEqual(record["window_drop"], (0, 1))
            self.assertEqual(record["mean_offset_shift"], (0, 1))
        self.assertEqual(
            result["totals"],
            {
                "identities": 3,
                "baseline_predictions": 6,
                "recent_predictions": 9,
                "baseline_resolved": 6,
                "recent_resolved": 9,
            },
        )

    def test_no_qualifying_identity_returns_zero_totals(self):
        # Every steady identity scores a zero drop, so a full-ratio
        # threshold keeps nothing.
        result = self.call(
            windows=LONG_WINDOWS,
            max_jitter=2,
            horizon=1,
            cutoffs=(2, 3, 4, 5, 6),
            split_cutoff=4,
            min_rate_drop=(1, 1),
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

    def build(self, min_resolved=1, scorecard_limit=50,
              min_rate_drop=(0, 1), drift_limit=50, split_cutoff=3):
        return BranchStore._build_lifetime_churn_forecast_drift_result(
            synthetic_backtests(),
            split_cutoff,
            min_resolved,
            scorecard_limit,
            min_rate_drop,
            drift_limit,
        )

    def test_kept_records_take_the_drift_sort_order(self):
        result = self.build()
        self.assertEqual(
            [record["identity"] for record in result["drifts"]],
            ["A", "F", "G", "H", "B", "E"],
        )

    def test_drops_shifts_and_period_scorecards(self):
        records = {
            record["identity"]: record
            for record in self.build()["drifts"]
        }
        record = records["A"]
        self.assertEqual(record["type"], "node")
        self.assertEqual(record["baseline"]["predictions"], 4)
        self.assertEqual(record["baseline"]["resolved"], 4)
        self.assertEqual(record["baseline"]["exact_rate"], (1, 2))
        self.assertEqual(record["baseline"]["window_rate"], (1, 1))
        self.assertEqual(record["baseline"]["mean_offset"], (0, 1))
        self.assertEqual(record["baseline"]["max_abs_offset"], 1)
        self.assertEqual(record["recent"]["predictions"], 4)
        self.assertEqual(record["recent"]["resolved"], 4)
        self.assertEqual(record["recent"]["exact_rate"], (1, 4))
        self.assertEqual(record["recent"]["window_rate"], (1, 4))
        self.assertEqual(record["recent"]["mean_offset"], (0, 1))
        self.assertEqual(record["recent"]["max_abs_offset"], 0)
        self.assertEqual(record["exact_drop"], (1, 4))
        self.assertEqual(record["window_drop"], (3, 4))
        self.assertEqual(record["mean_offset_shift"], (0, 1))
        # H's drop reduces: 1/2 - 1/6 = 1/3.
        self.assertEqual(records["H"]["baseline"]["window_rate"], (1, 2))
        self.assertEqual(records["H"]["recent"]["window_rate"], (1, 6))
        self.assertEqual(records["H"]["window_drop"], (1, 3))
        self.assertEqual(records["H"]["exact_drop"], (1, 3))
        self.assertEqual(records["H"]["mean_offset_shift"], (0, 1))
        # B is steady: zero drops and a zero shift.
        self.assertEqual(records["B"]["window_drop"], (0, 1))
        self.assertEqual(records["B"]["exact_drop"], (0, 1))
        self.assertEqual(records["B"]["mean_offset_shift"], (0, 1))

    def test_shift_is_none_when_either_period_has_no_offset(self):
        records = {
            record["identity"]: record
            for record in self.build()["drifts"]
        }
        # F and G have a baseline offset but no recent observation; E
        # is never observed in either period.
        self.assertIsNone(records["F"]["mean_offset_shift"])
        self.assertIsNone(records["G"]["mean_offset_shift"])
        self.assertIsNone(records["E"]["mean_offset_shift"])
        self.assertEqual(records["F"]["baseline"]["mean_offset"], (2, 1))
        self.assertEqual(records["G"]["baseline"]["mean_offset"], (-1, 1))

    def test_ties_fall_back_to_first_appearance_order(self):
        # F and G share every sort key; F appears first in the backtest.
        result = self.build()
        identities = [record["identity"] for record in result["drifts"]]
        self.assertLess(identities.index("F"), identities.index("G"))
        # B's recent max_abs_offset 0 outranks E's missing offset.
        self.assertLess(identities.index("B"), identities.index("E"))

    def test_threshold_compares_by_cross_multiplication(self):
        self.assertEqual(
            [
                record["identity"]
                for record in self.build(min_rate_drop=(1, 2))["drifts"]
            ],
            ["A", "F", "G"],
        )
        # (2, 4) is the same ratio as (1, 2): equality still keeps.
        self.assertEqual(
            [
                record["identity"]
                for record in self.build(min_rate_drop=(2, 4))["drifts"]
            ],
            ["A", "F", "G"],
        )
        self.assertEqual(
            [
                record["identity"]
                for record in self.build(min_rate_drop=(1, 3))["drifts"]
            ],
            ["A", "F", "G", "H"],
        )
        self.assertEqual(
            [
                record["identity"]
                for record in self.build(min_rate_drop=(3, 4))["drifts"]
            ],
            ["A"],
        )
        self.assertEqual(
            self.build(min_rate_drop=(1, 1))["drifts"],
            (),
        )

    def test_improving_identity_is_never_kept(self):
        # D's window_drop is negative and no allowed threshold is.
        result = self.build()
        self.assertNotIn(
            "D", [record["identity"] for record in result["drifts"]]
        )

    def test_identity_unresolved_in_one_period_is_skipped(self):
        # C resolves in the baseline period but not in the recent one.
        result = self.build()
        self.assertNotIn(
            "C", [record["identity"] for record in result["drifts"]]
        )

    def test_min_resolved_applies_to_both_periods(self):
        self.assertEqual(
            [
                record["identity"]
                for record in self.build(min_resolved=2)["drifts"]
            ],
            ["A", "F", "G", "H", "B", "E"],
        )
        self.assertEqual(
            [
                record["identity"]
                for record in self.build(min_resolved=3)["drifts"]
            ],
            ["A"],
        )
        self.assertEqual(self.build(min_resolved=5)["drifts"], ())

    def test_totals_sum_only_the_kept_records(self):
        self.assertEqual(
            self.build()["totals"],
            {
                "identities": 6,
                "baseline_predictions": 14,
                "recent_predictions": 18,
                "baseline_resolved": 14,
                "recent_resolved": 18,
            },
        )
        self.assertEqual(
            self.build(min_rate_drop=(1, 2))["totals"],
            {
                "identities": 3,
                "baseline_predictions": 8,
                "recent_predictions": 8,
                "baseline_resolved": 8,
                "recent_resolved": 8,
            },
        )
        self.assertEqual(
            self.build(min_rate_drop=(1, 1))["totals"],
            {
                "identities": 0,
                "baseline_predictions": 0,
                "recent_predictions": 0,
                "baseline_resolved": 0,
                "recent_resolved": 0,
            },
        )

    def test_drift_limit_raises_instead_of_truncating(self):
        with self.assertRaises(ValueError) as caught:
            self.build(drift_limit=5)
        self.assertIn("drift limit", str(caught.exception))
        result = self.build(drift_limit=6)
        self.assertEqual(len(result["drifts"]), 6)

    def test_scorecard_limit_bounds_each_period(self):
        # The baseline period keeps eight scorecards, the recent seven.
        with self.assertRaises(ValueError) as caught:
            self.build(scorecard_limit=7)
        self.assertIn("scorecard limit", str(caught.exception))
        result = self.build(scorecard_limit=8)
        self.assertEqual(len(result["drifts"]), 6)

    def test_empty_backtests_return_empty_drifts_and_zero_totals(self):
        result = BranchStore._build_lifetime_churn_forecast_drift_result(
            [], 1, 1, 50, (0, 1), 50
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
        identity = ("a", ("b", 1))
        backtests = [
            _cutoff_result(
                1, [_outcome(identity, "hit", predicted=3, actual=3)]
            ),
            _cutoff_result(2, [_outcome(identity, "missed")]),
        ]
        result = BranchStore._build_lifetime_churn_forecast_drift_result(
            backtests, 2, 1, 50, (0, 1), 50
        )
        outcomes = [
            outcome
            for cutoff_result in backtests
            for outcome in cutoff_result["outcomes"]
        ]
        self.assertEqual(len(result["drifts"]), 1)
        for record in result["drifts"]:
            self.assertEqual(record["identity"], identity)
            for outcome in outcomes:
                self.assertIsNot(record, outcome)
                self.assertIsNot(record["identity"], outcome["identity"])
            self.assertIsNot(record["baseline"], record["recent"])
            self.assertIsNot(
                record["identity"], record["baseline"]["identity"]
            )
            self.assertIsNot(
                record["identity"], record["recent"]["identity"]
            )
            self.assertIsNot(
                record["baseline"]["identity"], record["recent"]["identity"]
            )


class LifetimeChurnForecastDriftValidationTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        overrides.setdefault("causes", ("a0",))
        return drift_call(self.store, self.args, **overrides)

    def extended(self, **overrides):
        overrides.setdefault("windows", LONG_WINDOWS)
        overrides.setdefault("horizon", 1)
        overrides.setdefault("max_jitter", 2)
        overrides.setdefault("cutoffs", (2, 3, 4, 5, 6))
        overrides.setdefault("split_cutoff", 4)
        overrides.setdefault("backtest_limit", 50)
        return self.call(**overrides)

    def test_split_cutoff_must_be_a_non_bool_int(self):
        for bad in (True, False, 1.0, "1", None, (1,)):
            with self.subTest(split_cutoff=bad):
                with self.assertRaises(TypeError):
                    self.call(cutoffs=(0, 1), split_cutoff=bad)

    def test_split_cutoff_must_lie_within_the_cutoff_span(self):
        with self.assertRaises(ValueError):
            self.extended(split_cutoff=1)
        with self.assertRaises(ValueError):
            self.extended(split_cutoff=7)
        # The last cutoff itself is a valid split: the recent period
        # keeps exactly that cutoff.
        result = self.extended(split_cutoff=6)
        self.assertEqual(
            [record["identity"] for record in result["drifts"]],
            [("W", "y"), ("a0", "x"), (("a0", "x"), ("W", "y"))],
        )

    def test_split_cutoff_must_leave_both_periods_non_empty(self):
        # A split at the first cutoff empties the baseline period.
        with self.assertRaises(ValueError):
            self.extended(split_cutoff=2)
        # A single cutoff can never split into two periods.
        with self.assertRaises(ValueError):
            self.call(cutoffs=(2,), split_cutoff=2)

    def test_empty_cutoffs_raise(self):
        with self.assertRaises(ValueError):
            self.call(cutoffs=(), split_cutoff=0)
        with self.assertRaises(ValueError):
            self.extended(cutoffs=())

    def test_min_rate_drop_must_be_a_two_int_tuple(self):
        for bad in ("x", [0, 1], 1, None, True):
            with self.subTest(min_rate_drop=bad):
                with self.assertRaises(TypeError):
                    self.extended(min_rate_drop=bad)
        for bad in ((1,), (1, 2, 3), ()):
            with self.subTest(min_rate_drop=bad):
                with self.assertRaises(ValueError):
                    self.extended(min_rate_drop=bad)
        for bad in ((True, 1), (1, False), (1.0, 1), (1, "1")):
            with self.subTest(min_rate_drop=bad):
                with self.assertRaises(TypeError):
                    self.extended(min_rate_drop=bad)

    def test_min_rate_drop_must_be_a_valid_ratio(self):
        for bad in ((-1, 1), (1, 0), (1, -2), (0, 0), (2, 1), (3, 2)):
            with self.subTest(min_rate_drop=bad):
                with self.assertRaises(ValueError):
                    self.extended(min_rate_drop=bad)
        # The extremes are both valid thresholds.
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
            self.extended(split_cutoff="x", min_rate_drop=(2, 1))
        # min_rate_drop fails before the split's range check.
        with self.assertRaises(TypeError):
            self.extended(split_cutoff=99, min_rate_drop="x")
        with self.assertRaises(ValueError):
            self.extended(split_cutoff=99, min_rate_drop=(2, 1))
        # Once min_rate_drop passes, the split's range error fires.
        with self.assertRaises(ValueError):
            self.extended(split_cutoff=99, min_rate_drop=(0, 1))
        # min_rate_drop fails before drift_limit.
        with self.assertRaises(TypeError):
            self.extended(min_rate_drop="x", drift_limit="x")
        with self.assertRaises(ValueError):
            self.extended(min_rate_drop=(2, 1), drift_limit=0)
        # Once min_rate_drop passes, drift_limit's error fires.
        with self.assertRaises(TypeError):
            self.extended(min_rate_drop=(0, 1), drift_limit="x")
        with self.assertRaises(ValueError):
            self.extended(min_rate_drop=(0, 1), drift_limit=0)
        # All parameter checks still precede every state lookup.
        with self.assertRaises(TypeError):
            self.call(reference="ghost", cutoffs=(0, 1), split_cutoff="x")
        with self.assertRaises(ValueError):
            self.call(
                reference="ghost", cutoffs=(0, 1), min_rate_drop=(2, 1)
            )
        with self.assertRaises(ValueError):
            self.call(reference="ghost", cutoffs=(0, 1), drift_limit=0)
        with self.assertRaises(ValueError):
            self.call(token="whatever", cutoffs=(0, 1), split_cutoff=5)

    def test_drift_limit_counts_kept_records_and_never_truncates(self):
        # Three identities qualify at min_rate_drop (0, 1).
        with self.assertRaises(ValueError) as caught:
            self.extended(drift_limit=2)
        self.assertIn("drift limit", str(caught.exception))
        result = self.extended(drift_limit=3)
        self.assertEqual(result["totals"]["identities"], 3)
        # A higher threshold keeps nothing, so even a cap of one passes.
        result = self.extended(min_rate_drop=(1, 1), drift_limit=1)
        self.assertEqual(result["drifts"], ())

    def test_scorecard_limit_still_bounds_each_period(self):
        with self.assertRaises(ValueError) as caught:
            self.extended(scorecard_limit=2)
        self.assertIn("scorecard limit", str(caught.exception))
        result = self.extended(scorecard_limit=3)
        self.assertEqual(result["totals"]["identities"], 3)

    def test_backtest_limit_still_bounds_all_cutoffs(self):
        with self.assertRaises(ValueError) as caught:
            self.extended(backtest_limit=5)
        self.assertIn("backtest limit", str(caught.exception))

    def test_shared_validation_errors_match_scorecards(self):
        with self.assertRaises(TypeError):
            self.call(cutoffs="x")
        with self.assertRaises(ValueError):
            self.call(cutoffs=(-1,))
        with self.assertRaises(ValueError):
            self.call(windows=((-1,),))
        with self.assertRaises(ValueError) as caught:
            self.extended(cutoffs=(2, 99))
        self.assertIn("out of range", str(caught.exception))


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
            self.call(reference="ghost", cutoffs=(99, 100), split_cutoff=100)


class LifetimeChurnForecastDriftIsolationTests(unittest.TestCase):
    def call_args(self):
        return dict(
            windows=LONG_WINDOWS,
            max_jitter=2,
            horizon=1,
            cutoffs=(2, 3, 4, 5, 6),
            split_cutoff=4,
        )

    def test_result_levels_are_mutually_unshared(self):
        store, args = diamond_store()
        result = drift_call(store, args, **self.call_args())
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
        first = drift_call(store, args, **self.call_args())
        pristine = copy.deepcopy(first)
        for record in first["drifts"]:
            record["baseline"]["predictions"] = 99
            record["recent"]["resolved"] = 99
            record["window_drop"] = (9, 9)
            record["mean_offset_shift"] = (9, 9)
        first["totals"]["identities"] = 99
        self.assertEqual(
            drift_call(store, args, **self.call_args()), pristine
        )

    def test_success_and_failure_change_nothing(self):
        store, args = diamond_store()
        heads = dict(store._heads)
        appends = copy.deepcopy(store._appends)
        merges = copy.deepcopy(store._merges)
        audit = store.audit_log()

        drift_call(store, args, **self.call_args())
        failures = [
            dict(windows="x"),
            dict(windows=((-1,),)),
            dict(node_limit=0),
            dict(change_limit=0),
            dict(lifetime_limit=0),
            dict(diff_limit=0),
            dict(window_limit=0),
            dict(total_diff_limit=0),
            dict(churn_limit=0),
            dict(streak_limit=0),
            dict(min_identities=0),
            dict(wave_limit=0),
            dict(min_waves=0),
            dict(recurrence_limit=0),
            dict(max_jitter=-1),
            dict(periodicity_limit=0),
            dict(horizon=0),
            dict(forecast_limit=0),
            dict(cutoffs="x"),
            dict(cutoffs=(True,)),
            dict(cutoffs=(-1,)),
            dict(cutoffs=(1, 1)),
            dict(cutoffs=()),
            dict(backtest_limit=0),
            dict(backtest_limit="x"),
            dict(min_resolved=0),
            dict(min_resolved="x"),
            dict(min_resolved=True),
            dict(scorecard_limit=0),
            dict(scorecard_limit="x"),
            dict(scorecard_limit=False),
            dict(split_cutoff=True),
            dict(split_cutoff="x"),
            dict(split_cutoff=1.0),
            dict(split_cutoff=1),
            dict(split_cutoff=2),
            dict(split_cutoff=7),
            dict(min_rate_drop="x"),
            dict(min_rate_drop=[0, 1]),
            dict(min_rate_drop=(1,)),
            dict(min_rate_drop=(True, 1)),
            dict(min_rate_drop=(-1, 1)),
            dict(min_rate_drop=(1, 0)),
            dict(min_rate_drop=(2, 1)),
            dict(drift_limit=0),
            dict(drift_limit="x"),
            dict(drift_limit=False),
            dict(drift_limit=2),
            dict(wave_limit=1),
            dict(cutoffs=(2, 99)),
            dict(forecast_limit=1),
            dict(backtest_limit=1),
            dict(scorecard_limit=2),
            dict(causes=("zzz",)),
            dict(limit=1),
            dict(reference="ghost"),
        ]
        for overrides in failures:
            with self.subTest(overrides=overrides):
                merged = dict(self.call_args())
                merged.update(overrides)
                with self.assertRaises(Exception):
                    drift_call(store, args, **merged)

        self.assertEqual(dict(store._heads), heads)
        self.assertEqual(copy.deepcopy(store._appends), appends)
        self.assertEqual(copy.deepcopy(store._merges), merges)
        self.assertEqual(store.audit_log(), audit)

    def test_existing_public_entries_are_unaffected(self):
        store, args = diamond_store()
        drift_call(store, args, **self.call_args())
        from tests.test_lifetime_churn_forecasts import forecasts_call
        from tests.test_lifetime_churn_waves import waves_call

        waves = waves_call(store, args, windows=LONG_WINDOWS)
        backtests = backtests_call(
            store,
            args,
            windows=LONG_WINDOWS,
            max_jitter=2,
            horizon=1,
            cutoffs=(2, 3, 4, 5, 6),
        )
        scorecards = scorecards_call(
            store,
            args,
            windows=LONG_WINDOWS,
            max_jitter=2,
            horizon=1,
            cutoffs=(2, 3, 4, 5, 6),
        )
        forecasts = forecasts_call(
            store, args, windows=LONG_WINDOWS, max_jitter=2, horizon=1
        )
        self.assertEqual(
            waves_call(store, args, windows=LONG_WINDOWS), waves
        )
        self.assertEqual(
            backtests_call(
                store,
                args,
                windows=LONG_WINDOWS,
                max_jitter=2,
                horizon=1,
                cutoffs=(2, 3, 4, 5, 6),
            ),
            backtests,
        )
        self.assertEqual(
            scorecards_call(
                store,
                args,
                windows=LONG_WINDOWS,
                max_jitter=2,
                horizon=1,
                cutoffs=(2, 3, 4, 5, 6),
            ),
            scorecards,
        )
        self.assertEqual(
            forecasts_call(
                store, args, windows=LONG_WINDOWS, max_jitter=2, horizon=1
            ),
            forecasts,
        )


class LifetimeChurnForecastDriftTokenTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, token=None, **overrides):
        return drift_call(self.store, self.args, token=token, **overrides)

    def call_args(self):
        return dict(
            windows=LONG_WINDOWS,
            max_jitter=2,
            horizon=1,
            cutoffs=(2, 3, 4, 5, 6),
            split_cutoff=4,
        )

    def test_success_consumes_one_read_and_failure_refunds(self):
        token = self.store.create_snapshot(2)
        self.call(token=token, **self.call_args())
        # Over the drift cap: failure refunds.
        with self.assertRaises(ValueError):
            self.call(token=token, drift_limit=2, **self.call_args())
        # Over the scorecard cap: failure refunds.
        with self.assertRaises(ValueError):
            self.call(token=token, scorecard_limit=2, **self.call_args())
        # Over the backtest cap: failure refunds.
        with self.assertRaises(ValueError):
            self.call(token=token, backtest_limit=1, **self.call_args())
        # Out-of-range cutoff: failure refunds.
        with self.assertRaises(ValueError):
            self.call(
                token=token, **dict(self.call_args(), cutoffs=(2, 99))
            )
        # State error: failure refunds.
        with self.assertRaises(KeyError):
            self.call(
                token=token, **dict(self.call_args(), causes=("zzz",))
            )
        # Parameter validation runs before the token is touched.
        with self.assertRaises(ValueError):
            self.call(token=token, cutoffs=(), split_cutoff=0)
        with self.assertRaises(TypeError):
            self.call(
                token=token, **dict(self.call_args(), split_cutoff="x")
            )
        with self.assertRaises(ValueError):
            self.call(
                token=token, **dict(self.call_args(), min_rate_drop=(2, 1))
            )
        with self.assertRaises(ValueError):
            self.call(token=token, **dict(self.call_args(), drift_limit=0))
        # The second allowed read succeeds; the token then expires.
        self.call(token=token, **self.call_args())
        with self.assertRaises(RuntimeError):
            self.call(token=token, **self.call_args())

    def test_results_come_from_the_frozen_view(self):
        before = self.call(**self.call_args())
        token = self.store.create_snapshot(4)
        self.store.append("a", "a2", 7, {"x": -1})
        self.store.create("d", "root")
        self.store.append("d", "d0", 8, {"z": 3})
        self.store.merge("a", "d", "M", 9, {"z": 1})
        self.store.append("main", "r3", 10, {"x": 1})
        self.assertEqual(self.call(token=token, **self.call_args()), before)


if __name__ == "__main__":
    unittest.main()
