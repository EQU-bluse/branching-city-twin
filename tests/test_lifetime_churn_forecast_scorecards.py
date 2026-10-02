import copy
import inspect
import math
import unittest

from city_twin.branches import BranchStore

from tests.test_lifetime_churn import churn_kwargs, diamond_store
from tests.test_lifetime_churn_forecast_backtests import (
    backtests_kwargs,
    reference_backtests,
)


def scorecards_kwargs(store_args, **overrides) -> dict:
    min_resolved = overrides.pop("min_resolved", 1)
    scorecard_limit = overrides.pop("scorecard_limit", 50)
    kwargs = backtests_kwargs(store_args, **overrides)
    kwargs["min_resolved"] = min_resolved
    kwargs["scorecard_limit"] = scorecard_limit
    return kwargs


def scorecards_call(store, store_args, **overrides):
    return store.lifetime_churn_forecast_scorecards(
        **scorecards_kwargs(store_args, **overrides)
    )


def _reduced(numerator: int, denominator: int) -> tuple[int, int]:
    divisor = math.gcd(numerator, denominator)
    return (numerator // divisor, denominator // divisor)


def reference_scorecards(
    backtests: dict[str, object], min_resolved: int
) -> dict[str, object]:
    """Independently aggregate a backtests result into scorecards."""
    grouped = {}
    encounter = []
    for backtest in backtests["backtests"]:
        for outcome in backtest["outcomes"]:
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

    scorecards = []
    for key in encounter:
        entry = grouped[key]
        resolved = entry["predictions"] - entry["unresolved"]
        if resolved < min_resolved:
            continue
        observed = entry["hit"] + entry["early"] + entry["late"]
        if entry["offsets"]:
            mean_offset = _reduced(
                sum(entry["offsets"]), len(entry["offsets"])
            )
            max_abs_offset = max(abs(o) for o in entry["offsets"])
        else:
            mean_offset = None
            max_abs_offset = None
        scorecards.append(
            {
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
        )
    totals = {
        "identities": 0,
        "predictions": 0,
        "resolved": 0,
        "unresolved": 0,
        "hit": 0,
        "early": 0,
        "late": 0,
        "missed": 0,
        "observed": 0,
    }
    for scorecard in scorecards:
        totals["identities"] += 1
        for name in (
            "predictions",
            "resolved",
            "unresolved",
            "hit",
            "early",
            "late",
            "missed",
            "observed",
        ):
            totals[name] += scorecard[name]
    return {"scorecards": tuple(scorecards), "totals": totals}


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
    """Two cutoffs pinning grouping, ordering, rates and offsets.

    A resolves five times (two hits, an early, a late, a missed), B
    never resolves, C resolves four times (two hits, two missed) and D
    only ever misses. First-appearance order is B, A, C, D.
    """
    return [
        _cutoff_result(
            1,
            [
                _outcome("B", "unresolved"),
                _outcome("A", "hit", predicted=3, actual=3),
                _outcome("A", "early", ordinal=2, predicted=3, actual=2),
                _outcome("A", "missed", ordinal=3),
                _outcome("C", "hit", predicted=3, actual=3),
                _outcome("C", "missed", ordinal=2),
                _outcome("D", "missed"),
            ],
        ),
        _cutoff_result(
            2,
            [
                _outcome("B", "unresolved", ordinal=2),
                _outcome("A", "hit", ordinal=4, predicted=4, actual=4),
                _outcome("A", "late", ordinal=5, predicted=3, actual=5),
                _outcome("C", "hit", ordinal=3, predicted=3, actual=3),
                _outcome("C", "missed", ordinal=4),
                _outcome("D", "missed", ordinal=2),
            ],
        ),
    ]


class LifetimeChurnForecastScorecardsSignatureTests(unittest.TestCase):
    def test_public_signature_appends_min_resolved_and_scorecard_limit(self):
        parameters = list(
            inspect.signature(
                BranchStore.lifetime_churn_forecast_scorecards
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
        windows = (
            (0,), (1,), (1,), (2,), (2,), (0,), (0,), (1,),
        )
        result = scorecards_call(
            store,
            args,
            windows=windows,
            max_jitter=2,
            horizon=5,
            cutoffs=(2, 3),
        )
        self.assertIsInstance(result, dict)
        self.assertEqual(list(result), ["scorecards", "totals"])
        self.assertIsInstance(result["scorecards"], tuple)
        self.assertEqual(
            list(result["totals"]),
            [
                "identities",
                "predictions",
                "resolved",
                "unresolved",
                "hit",
                "early",
                "late",
                "missed",
                "observed",
            ],
        )
        for scorecard in result["scorecards"]:
            self.assertEqual(
                list(scorecard),
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
            for name in ("exact_rate", "window_rate", "mean_offset"):
                value = scorecard[name]
                if value is None:
                    continue
                self.assertIsInstance(value, tuple)
                self.assertEqual(len(value), 2)
                for part in value:
                    self.assertIsInstance(part, int)
                    self.assertNotIsInstance(part, bool)


class LifetimeChurnForecastScorecardsResultTests(unittest.TestCase):
    EXTENDED = (
        (0,), (1,), (1,), (2,), (2,), (0,), (0,), (1,),
    )

    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        return scorecards_call(self.store, self.args, **overrides)

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
        cases = (
            (
                (0,), (1,), (1,), (2,), (2,), (0,), (0,), (1,),
            ),
            ((0,), (1,), (1,), (2,), (2,), (0,)),
            ((0,), (1,), (2,)),
            ((0, 2), (1,), (1,), (0, 1, 2)),
            (),
        )
        for windows in cases:
            evolution = self.evolution(windows=windows)
            from tests.test_lifetime_churn_waves import reference_waves

            last_wave = len(reference_waves(evolution, 1)["waves"]) - 1
            cutoff_choices = tuple(range(max(last_wave, 0)))
            for max_jitter in (0, 2):
                for horizon in (1, 3, 6):
                    for min_resolved in (1, 2, 3):
                        with self.subTest(
                            windows=windows,
                            max_jitter=max_jitter,
                            horizon=horizon,
                            min_resolved=min_resolved,
                        ):
                            self.assertEqual(
                                self.call(
                                    windows=windows,
                                    max_jitter=max_jitter,
                                    horizon=horizon,
                                    cutoffs=cutoff_choices,
                                    backtest_limit=10_000,
                                    min_resolved=min_resolved,
                                    scorecard_limit=10_000,
                                ),
                                reference_scorecards(
                                    reference_backtests(
                                        evolution,
                                        1,
                                        max_jitter,
                                        horizon,
                                        cutoff_choices,
                                    ),
                                    min_resolved,
                                ),
                            )

    def test_diamond_fixture_aggregates_each_identity_once(self):
        result = self.call(
            windows=self.EXTENDED,
            max_jitter=2,
            horizon=5,
            cutoffs=(2, 3),
        )
        # Each of the three steady identities is predicted five times
        # per cutoff: one hit at cutoff 2, nine unresolved overall.
        self.assertEqual(len(result["scorecards"]), 3)
        for scorecard in result["scorecards"]:
            self.assertEqual(
                scorecard,
                {
                    "type": scorecard["type"],
                    "identity": scorecard["identity"],
                    "predictions": 10,
                    "resolved": 1,
                    "unresolved": 9,
                    "hit": 1,
                    "early": 0,
                    "late": 0,
                    "missed": 0,
                    "observed": 1,
                    "exact_rate": (1, 1),
                    "window_rate": (1, 1),
                    "mean_offset": (0, 1),
                    "max_abs_offset": 0,
                },
            )
        self.assertEqual(
            result["totals"],
            {
                "identities": 3,
                "predictions": 30,
                "resolved": 3,
                "unresolved": 27,
                "hit": 3,
                "early": 0,
                "late": 0,
                "missed": 0,
                "observed": 3,
            },
        )

    def test_min_resolved_filters_every_diamond_identity(self):
        result = self.call(
            windows=self.EXTENDED,
            max_jitter=2,
            horizon=5,
            cutoffs=(2, 3),
            min_resolved=2,
        )
        self.assertEqual(result["scorecards"], ())
        self.assertEqual(
            result["totals"],
            {
                "identities": 0,
                "predictions": 0,
                "resolved": 0,
                "unresolved": 0,
                "hit": 0,
                "early": 0,
                "late": 0,
                "missed": 0,
                "observed": 0,
            },
        )

    def test_empty_cutoffs_return_empty_scorecards_and_zero_totals(self):
        result = self.call(cutoffs=())
        self.assertEqual(result["scorecards"], ())
        self.assertEqual(
            result["totals"],
            {
                "identities": 0,
                "predictions": 0,
                "resolved": 0,
                "unresolved": 0,
                "hit": 0,
                "early": 0,
                "late": 0,
                "missed": 0,
                "observed": 0,
            },
        )

    def test_empty_cutoffs_still_run_every_state_check(self):
        with self.assertRaises(KeyError):
            self.call(cutoffs=(), reference="ghost")
        with self.assertRaises(ValueError):
            self.call(cutoffs=(), limit=1)


class LifetimeChurnForecastScorecardsAggregationTests(unittest.TestCase):
    """Synthetic cutoff results pin the aggregation independently."""

    def build(self, min_resolved=1, scorecard_limit=50):
        return BranchStore._build_lifetime_churn_forecast_scorecards_result(
            synthetic_backtests(), min_resolved, scorecard_limit
        )

    def test_groups_keep_first_appearance_order_of_kept_identities(self):
        result = self.build()
        self.assertEqual(
            [s["identity"] for s in result["scorecards"]],
            ["A", "C", "D"],
        )

    def test_counts_rates_and_offsets(self):
        scorecards = {
            s["identity"]: s for s in self.build()["scorecards"]
        }
        self.assertEqual(
            scorecards["A"],
            {
                "type": "node",
                "identity": "A",
                "predictions": 5,
                "resolved": 5,
                "unresolved": 0,
                "hit": 2,
                "early": 1,
                "late": 1,
                "missed": 1,
                "observed": 4,
                "exact_rate": (2, 5),
                "window_rate": (4, 5),
                "mean_offset": (1, 4),
                "max_abs_offset": 2,
            },
        )
        self.assertEqual(
            scorecards["C"],
            {
                "type": "node",
                "identity": "C",
                "predictions": 4,
                "resolved": 4,
                "unresolved": 0,
                "hit": 2,
                "early": 0,
                "late": 0,
                "missed": 2,
                "observed": 2,
                "exact_rate": (1, 2),
                "window_rate": (1, 2),
                "mean_offset": (0, 1),
                "max_abs_offset": 0,
            },
        )

    def test_never_observed_identity_has_no_offset(self):
        scorecards = {
            s["identity"]: s for s in self.build()["scorecards"]
        }
        self.assertEqual(
            scorecards["D"],
            {
                "type": "node",
                "identity": "D",
                "predictions": 2,
                "resolved": 2,
                "unresolved": 0,
                "hit": 0,
                "early": 0,
                "late": 0,
                "missed": 2,
                "observed": 0,
                "exact_rate": (0, 1),
                "window_rate": (0, 1),
                "mean_offset": None,
                "max_abs_offset": None,
            },
        )

    def test_unresolved_only_identity_is_dropped_by_min_resolved(self):
        result = self.build()
        self.assertNotIn(
            "B", [s["identity"] for s in result["scorecards"]]
        )

    def test_min_resolved_keeps_only_sufficiently_resolved_identities(self):
        result = self.build(min_resolved=3)
        self.assertEqual(
            [s["identity"] for s in result["scorecards"]], ["A", "C"]
        )
        result = self.build(min_resolved=5)
        self.assertEqual(
            [s["identity"] for s in result["scorecards"]], ["A"]
        )
        result = self.build(min_resolved=6)
        self.assertEqual(result["scorecards"], ())

    def test_totals_sum_only_the_kept_scorecards(self):
        self.assertEqual(
            self.build()["totals"],
            {
                "identities": 3,
                "predictions": 11,
                "resolved": 11,
                "unresolved": 0,
                "hit": 4,
                "early": 1,
                "late": 1,
                "missed": 5,
                "observed": 6,
            },
        )
        self.assertEqual(
            self.build(min_resolved=3)["totals"],
            {
                "identities": 2,
                "predictions": 9,
                "resolved": 9,
                "unresolved": 0,
                "hit": 4,
                "early": 1,
                "late": 1,
                "missed": 3,
                "observed": 6,
            },
        )

    def test_scorecard_limit_raises_instead_of_truncating(self):
        with self.assertRaises(ValueError) as caught:
            self.build(scorecard_limit=2)
        self.assertIn("scorecard limit", str(caught.exception))
        result = self.build(scorecard_limit=3)
        self.assertEqual(len(result["scorecards"]), 3)

    def test_empty_backtests_return_empty_scorecards_and_zero_totals(self):
        result = (
            BranchStore._build_lifetime_churn_forecast_scorecards_result(
                [], 1, 1
            )
        )
        self.assertEqual(result["scorecards"], ())
        self.assertEqual(
            result["totals"],
            {
                "identities": 0,
                "predictions": 0,
                "resolved": 0,
                "unresolved": 0,
                "hit": 0,
                "early": 0,
                "late": 0,
                "missed": 0,
                "observed": 0,
            },
        )

    def test_scorecard_identity_is_an_isolated_copy(self):
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
            )
        ]
        scorecard = (
            BranchStore._build_lifetime_churn_forecast_scorecards_result(
                backtests, 1, 1
            )["scorecards"][0]
        )
        self.assertEqual(scorecard["identity"], identity)
        self.assertIsNot(scorecard["identity"], identity)

    def test_result_shares_nothing_with_the_cutoff_results(self):
        backtests = [
            _cutoff_result(
                1,
                [
                    _outcome(
                        ("a", ("b", 1)), "hit", predicted=3, actual=3
                    ),
                    _outcome(
                        ("a", ("b", 2)), "missed", ordinal=2
                    ),
                ],
            )
        ]
        result = (
            BranchStore._build_lifetime_churn_forecast_scorecards_result(
                backtests, 1, 50
            )
        )
        outcomes = [
            outcome
            for cutoff_result in backtests
            for outcome in cutoff_result["outcomes"]
        ]
        for scorecard in result["scorecards"]:
            for outcome in outcomes:
                self.assertIsNot(scorecard, outcome)
                self.assertIsNot(
                    scorecard["identity"], outcome["identity"]
                )


class LifetimeChurnForecastScorecardsValidationTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        overrides.setdefault("causes", ("a0",))
        return scorecards_call(self.store, self.args, **overrides)

    def extended(self, **overrides):
        overrides.setdefault(
            "windows",
            ((0,), (1,), (1,), (2,), (2,), (0,), (0,), (1,)),
        )
        overrides.setdefault("horizon", 5)
        overrides.setdefault("max_jitter", 2)
        overrides.setdefault("cutoffs", (2,))
        overrides.setdefault("backtest_limit", 50)
        return self.call(**overrides)

    def test_min_resolved_must_be_a_non_bool_positive_int(self):
        for bad in (True, False, 1.0, "1", None):
            with self.subTest(min_resolved=bad):
                with self.assertRaises(TypeError):
                    self.call(cutoffs=(0,), min_resolved=bad)
        with self.assertRaises(ValueError):
            self.call(cutoffs=(0,), min_resolved=0)
        with self.assertRaises(ValueError):
            self.call(cutoffs=(0,), min_resolved=-4)

    def test_scorecard_limit_must_be_a_non_bool_positive_int(self):
        for bad in (True, False, 1.0, "1", None):
            with self.subTest(scorecard_limit=bad):
                with self.assertRaises(TypeError):
                    self.call(cutoffs=(0,), scorecard_limit=bad)
        with self.assertRaises(ValueError):
            self.call(cutoffs=(0,), scorecard_limit=0)
        with self.assertRaises(ValueError):
            self.call(cutoffs=(0,), scorecard_limit=-4)

    def test_new_parameters_validated_in_order_after_backtest_limit(self):
        # backtest_limit fails before min_resolved.
        with self.assertRaises(ValueError):
            self.extended(backtest_limit=0, min_resolved="x")
        with self.assertRaises(TypeError):
            self.extended(backtest_limit="x", min_resolved="x")
        # min_resolved fails before scorecard_limit.
        with self.assertRaises(ValueError):
            self.extended(min_resolved=0, scorecard_limit="x")
        with self.assertRaises(TypeError):
            self.extended(min_resolved="x", scorecard_limit=0)
        # Once min_resolved passes, scorecard_limit's error fires.
        with self.assertRaises(TypeError):
            self.extended(min_resolved=1, scorecard_limit="x")
        with self.assertRaises(ValueError):
            self.extended(min_resolved=1, scorecard_limit=0)
        # All parameter checks still precede every state lookup.
        with self.assertRaises(ValueError):
            self.call(reference="ghost", min_resolved=0)
        with self.assertRaises(TypeError):
            self.call(reference="ghost", min_resolved="x")
        with self.assertRaises(ValueError):
            self.call(token="whatever", cutoffs=(0,), scorecard_limit=0)
        with self.assertRaises(TypeError):
            self.call(token="whatever", cutoffs=(0,), min_resolved="x")

    def test_scorecard_limit_counts_kept_scorecards_over_all_cutoffs(self):
        # Three identities qualify at min_resolved 1.
        with self.assertRaises(ValueError) as caught:
            self.extended(cutoffs=(2, 3), scorecard_limit=2)
        self.assertIn("scorecard limit", str(caught.exception))
        result = self.extended(cutoffs=(2, 3), scorecard_limit=3)
        self.assertEqual(result["totals"]["identities"], 3)
        # None qualify at min_resolved 2, so even a cap of one passes.
        result = self.extended(
            cutoffs=(2, 3), min_resolved=2, scorecard_limit=1
        )
        self.assertEqual(result["scorecards"], ())

    def test_over_limit_is_checked_after_full_computation(self):
        # The failing call must not truncate: the very next call with a
        # raised cap returns the complete three-scorecard set.
        with self.assertRaises(ValueError):
            self.extended(cutoffs=(2, 3), scorecard_limit=2)
        result = self.extended(cutoffs=(2, 3), scorecard_limit=3)
        self.assertEqual(result["totals"]["predictions"], 30)

    def test_backtest_limit_still_bounds_all_cutoffs(self):
        with self.assertRaises(ValueError) as caught:
            self.extended(cutoffs=(2, 3), backtest_limit=15)
        self.assertIn("backtest limit", str(caught.exception))

    def test_empty_cutoffs_never_trip_the_scorecard_cap(self):
        result = self.call(cutoffs=(), scorecard_limit=1)
        self.assertEqual(result["scorecards"], ())
        self.assertEqual(result["totals"]["identities"], 0)

    def test_shared_validation_errors_match_backtests(self):
        with self.assertRaises(TypeError):
            self.call(cutoffs="x")
        with self.assertRaises(ValueError):
            self.call(cutoffs=(-1,))
        with self.assertRaises(ValueError):
            self.call(windows=((-1,),))
        with self.assertRaises(ValueError) as caught:
            self.extended(cutoffs=(99,))
        self.assertIn("out of range", str(caught.exception))


class LifetimeChurnForecastScorecardsStateTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        overrides.setdefault("causes", ("a0",))
        return scorecards_call(self.store, self.args, **overrides)

    def test_unknown_branch_history_node_and_cause_raise(self):
        with self.assertRaises(KeyError):
            self.call(reference="ghost", cutoffs=(0,))
        with self.assertRaises(KeyError):
            self.call(
                series={
                    "a": (("r0", "a0"), ("r1", "a0"), ("r2", "a0")),
                    "b": (("r0", "a0"), ("r1", "W"), ("r2", "nope2")),
                },
                cutoffs=(0,),
            )
        with self.assertRaises(KeyError) as caught:
            self.call(causes=("zzz",), cutoffs=(0,))
        self.assertEqual(caught.exception.args[0], "zzz")

    def test_range_error_fires_after_the_state_checks(self):
        with self.assertRaises(KeyError):
            self.call(reference="ghost", cutoffs=(99,))


class LifetimeChurnForecastScorecardsIsolationTests(unittest.TestCase):
    EXTENDED = (
        (0,), (1,), (1,), (2,), (2,), (0,), (0,), (1,),
    )

    def test_result_levels_are_mutually_unshared(self):
        store, args = diamond_store()
        result = scorecards_call(
            store,
            args,
            windows=self.EXTENDED,
            max_jitter=2,
            horizon=5,
            cutoffs=(2, 3),
        )
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
        first = scorecards_call(
            store,
            args,
            windows=self.EXTENDED,
            max_jitter=2,
            horizon=5,
            cutoffs=(2, 3),
        )
        pristine = copy.deepcopy(first)
        for scorecard in first["scorecards"]:
            scorecard["predictions"] = 99
            scorecard["resolved"] = 99
            scorecard["exact_rate"] = (9, 9)
            scorecard["mean_offset"] = (9, 9)
        first["totals"]["identities"] = 99
        self.assertEqual(
            scorecards_call(
                store,
                args,
                windows=self.EXTENDED,
                max_jitter=2,
                horizon=5,
                cutoffs=(2, 3),
            ),
            pristine,
        )

    def test_success_and_failure_change_nothing(self):
        store, args = diamond_store()
        heads = dict(store._heads)
        appends = copy.deepcopy(store._appends)
        merges = copy.deepcopy(store._merges)
        audit = store.audit_log()

        scorecards_call(
            store,
            args,
            windows=self.EXTENDED,
            max_jitter=2,
            horizon=5,
            cutoffs=(2,),
        )
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
            dict(backtest_limit=0),
            dict(backtest_limit="x"),
            dict(min_resolved=0),
            dict(min_resolved="x"),
            dict(min_resolved=True),
            dict(scorecard_limit=0),
            dict(scorecard_limit="x"),
            dict(scorecard_limit=False),
            dict(wave_limit=1, cutoffs=(2,)),
            dict(cutoffs=(99,)),
            dict(forecast_limit=1, cutoffs=(2,)),
            dict(cutoffs=(2,), backtest_limit=1),
            dict(cutoffs=(2, 3), scorecard_limit=2),
            dict(causes=("zzz",), cutoffs=(2,)),
            dict(limit=1, cutoffs=(2,)),
            dict(reference="ghost", cutoffs=(2,)),
        ]
        for overrides in failures:
            with self.subTest(overrides=overrides):
                with self.assertRaises(Exception):
                    scorecards_call(
                        store,
                        args,
                        causes=("a0",),
                        windows=self.EXTENDED,
                        max_jitter=2,
                        horizon=5,
                        **overrides,
                    )

        self.assertEqual(dict(store._heads), heads)
        self.assertEqual(copy.deepcopy(store._appends), appends)
        self.assertEqual(copy.deepcopy(store._merges), merges)
        self.assertEqual(store.audit_log(), audit)

    def test_existing_public_entries_are_unaffected(self):
        store, args = diamond_store()
        scorecards_call(
            store,
            args,
            windows=self.EXTENDED,
            max_jitter=2,
            horizon=5,
            cutoffs=(2,),
        )
        from tests.test_lifetime_churn_forecast_backtests import (
            backtests_call,
        )
        from tests.test_lifetime_churn_forecasts import forecasts_call
        from tests.test_lifetime_churn_waves import waves_call

        waves = waves_call(store, args, windows=self.EXTENDED)
        backtests = backtests_call(
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
            waves_call(store, args, windows=self.EXTENDED), waves
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
            forecasts_call(
                store, args, windows=self.EXTENDED,
                max_jitter=2, horizon=5
            ),
            forecasts,
        )


class LifetimeChurnForecastScorecardsTokenTests(unittest.TestCase):
    EXTENDED = (
        (0,), (1,), (1,), (2,), (2,), (0,), (0,), (1,),
    )

    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, token=None, **overrides):
        return scorecards_call(
            self.store, self.args, token=token, **overrides
        )

    def test_success_consumes_one_read_and_failure_refunds(self):
        token = self.store.create_snapshot(2)
        self.call(
            token=token,
            windows=self.EXTENDED,
            max_jitter=2,
            horizon=5,
            cutoffs=(2, 3),
        )
        # Over the scorecard cap: failure refunds.
        with self.assertRaises(ValueError):
            self.call(
                token=token,
                windows=self.EXTENDED,
                max_jitter=2,
                horizon=5,
                cutoffs=(2, 3),
                scorecard_limit=2,
            )
        # Over the backtest cap: failure refunds.
        with self.assertRaises(ValueError):
            self.call(
                token=token,
                windows=self.EXTENDED,
                max_jitter=2,
                horizon=5,
                cutoffs=(2, 3),
                backtest_limit=1,
            )
        # Out-of-range cutoff: failure refunds.
        with self.assertRaises(ValueError):
            self.call(
                token=token,
                windows=self.EXTENDED,
                cutoffs=(99,),
            )
        # State error: failure refunds.
        with self.assertRaises(KeyError):
            self.call(
                token=token,
                causes=("zzz",),
                cutoffs=(2,),
            )
        # Parameter validation runs before the token is touched.
        with self.assertRaises(ValueError):
            self.call(token=token, min_resolved=0)
        with self.assertRaises(TypeError):
            self.call(token=token, scorecard_limit="x")
        # An empty-cutoff success still performs one read.
        self.call(token=token, cutoffs=())
        with self.assertRaises(RuntimeError):
            self.call(token=token, cutoffs=())

    def test_results_come_from_the_frozen_view(self):
        before = scorecards_call(
            self.store,
            self.args,
            windows=self.EXTENDED,
            max_jitter=2,
            horizon=5,
            cutoffs=(2, 3),
        )
        token = self.store.create_snapshot(4)
        self.store.append("a", "a2", 7, {"x": -1})
        self.store.create("d", "root")
        self.store.append("d", "d0", 8, {"z": 3})
        self.store.merge("a", "d", "M", 9, {"z": 1})
        self.store.append("main", "r3", 10, {"x": 1})
        self.assertEqual(
            self.call(
                token=token,
                windows=self.EXTENDED,
                max_jitter=2,
                horizon=5,
                cutoffs=(2, 3),
            ),
            before,
        )


if __name__ == "__main__":
    unittest.main()
