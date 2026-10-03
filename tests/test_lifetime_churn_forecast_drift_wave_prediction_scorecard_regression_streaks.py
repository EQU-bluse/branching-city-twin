import copy
import inspect
import unittest

from city_twin.branches import BranchStore

from tests.test_lifetime_churn import diamond_store
from tests.test_lifetime_churn_forecast_drift_scan import LONG_ARGS
from tests.test_lifetime_churn_forecast_drift_wave_prediction_backtests import (
    build_backtests,
    synthetic_waves,
)
from tests.test_lifetime_churn_forecast_drift_wave_prediction_scorecards import (
    build_scorecards,
    scorecards_kwargs,
)


def streaks_kwargs(store_args, **overrides) -> dict:
    baseline_splits = overrides.pop("baseline_splits", (1,))
    min_evaluated_delta = overrides.pop("min_evaluated_delta", 0)
    min_hit_rate_drop = overrides.pop("min_hit_rate_drop", (0, 1))
    min_consecutive_splits = overrides.pop("min_consecutive_splits", 1)
    regression_streak_limit = overrides.pop("regression_streak_limit", 50)
    kwargs = scorecards_kwargs(store_args, **overrides)
    kwargs["baseline_splits"] = baseline_splits
    kwargs["min_evaluated_delta"] = min_evaluated_delta
    kwargs["min_hit_rate_drop"] = min_hit_rate_drop
    kwargs["min_consecutive_splits"] = min_consecutive_splits
    kwargs["regression_streak_limit"] = regression_streak_limit
    return kwargs


def streaks_call(store, store_args, **overrides):
    return store.lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_streaks(
        **streaks_kwargs(store_args, **overrides)
    )


def build_streaks(
    records,
    scorecards,
    baseline_splits=(1,),
    min_evaluated_delta=0,
    min_hit_rate_drop=(0, 1),
    min_consecutive_splits=1,
    regression_streak_limit=50,
):
    return BranchStore._build_lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_streaks_result(
        records,
        scorecards,
        baseline_splits,
        min_evaluated_delta,
        min_hit_rate_drop,
        min_consecutive_splits,
        regression_streak_limit,
    )


def kept_scorecards(*identities):
    return tuple({"identity": identity} for identity in identities)


def backtest_record(cutoff, identity, hit, missed, unresolved=0, unexpected=0):
    return {
        "cutoff": cutoff,
        "identity": identity,
        "period": 1,
        "jitter": 0,
        "last_wave": cutoff,
        "targets": (),
        "hit": hit,
        "missed": missed,
        "unresolved": unresolved,
        "unexpected": unexpected,
    }


def regressing_records(identity):
    return (
        backtest_record(2, identity, 2, 0),
        backtest_record(4, identity, 1, 1),
        backtest_record(6, identity, 0, 2),
        backtest_record(8, identity, 1, 1),
    )


class LifetimeChurnForecastDriftWavePredictionScorecardRegressionStreaksSignatureTests(
    unittest.TestCase
):
    def test_public_signature_appends_five_parameters(self):
        parameters = list(
            inspect.signature(
                BranchStore.lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_streaks
            ).parameters.values()
        )
        names = [p.name for p in parameters]
        scorecard_names = list(
            inspect.signature(
                BranchStore.lifetime_churn_forecast_drift_wave_prediction_scorecards
            ).parameters
        )
        self.assertEqual(
            names,
            scorecard_names[:-1]
            + [
                "baseline_splits",
                "min_evaluated_delta",
                "min_hit_rate_drop",
                "min_consecutive_splits",
                "regression_streak_limit",
                "token",
            ],
        )
        self.assertEqual(
            [p.default for p in parameters[:-1]],
            [inspect.Parameter.empty] * (len(parameters) - 1),
        )
        self.assertIsNone(parameters[-1].default)

    def test_result_shape_and_key_order(self):
        result = build_streaks(
            regressing_records("A"),
            kept_scorecards("A"),
            (1, 2, 3),
            0,
            (1, 2),
        )
        self.assertEqual(list(result), ["streaks", "totals"])
        self.assertIsInstance(result["streaks"], tuple)
        streak = result["streaks"][0]
        self.assertEqual(
            list(streak),
            [
                "identity",
                "start_baseline_cutoffs",
                "end_baseline_cutoffs",
                "split_count",
                "first_regression_cutoff",
                "worst_rate_drop",
            ],
        )
        self.assertEqual(
            list(result["totals"]),
            ["identities", "streaks", "regressing_splits"],
        )


class LifetimeChurnForecastDriftWavePredictionScorecardRegressionStreaksResultTests(
    unittest.TestCase
):
    def test_consecutive_hits_form_one_streak(self):
        # Splits 1 and 2 hit (drops 2/3 and 1/2, first regressing
        # cutoffs 4 and 8); split 3 falls short of the evaluated
        # delta, so the run stops at 2.
        result = build_streaks(
            regressing_records("A"),
            kept_scorecards("A"),
            (1, 2, 3),
            0,
            (1, 2),
        )
        self.assertEqual(
            result["streaks"],
            (
                {
                    "identity": "A",
                    "start_baseline_cutoffs": 1,
                    "end_baseline_cutoffs": 2,
                    "split_count": 2,
                    "first_regression_cutoff": 4,
                    "worst_rate_drop": (2, 3),
                },
            ),
        )
        self.assertEqual(
            result["totals"],
            {"identities": 1, "streaks": 1, "regressing_splits": 2},
        )

    def test_only_scorecard_identities_are_candidates(self):
        records = regressing_records("A") + regressing_records("Z")
        result = build_streaks(
            records, kept_scorecards("A"), (1, 2), 0, (1, 2)
        )
        self.assertEqual(
            [s["identity"] for s in result["streaks"]], ["A"]
        )
        self.assertEqual(result["totals"]["identities"], 1)

    def test_empty_segment_or_zero_evaluated_position_does_not_hit(self):
        # Split 2 leaves a one-record observation segment that
        # evaluates nothing, so only split 1 hits.
        records = (
            backtest_record(2, "A", 2, 0),
            backtest_record(4, "A", 0, 2),
            backtest_record(6, "A", 0, 0, 3),
        )
        result = build_streaks(
            records, kept_scorecards("A"), (1, 2), 0, (0, 1)
        )
        self.assertEqual(
            [
                (
                    s["start_baseline_cutoffs"],
                    s["end_baseline_cutoffs"],
                    s["split_count"],
                )
                for s in result["streaks"]
            ],
            [(1, 1, 1)],
        )
        # Split 3 leaves the observation segment empty, so no
        # position hits at all.
        result = build_streaks(
            records, kept_scorecards("A"), (2, 3), 0, (0, 1)
        )
        self.assertEqual(result["streaks"], ())

    def test_min_consecutive_splits_filters_short_runs(self):
        result = build_streaks(
            regressing_records("A"),
            kept_scorecards("A"),
            (1, 2, 3),
            0,
            (1, 2),
            2,
        )
        self.assertEqual(len(result["streaks"]), 1)
        result = build_streaks(
            regressing_records("A"),
            kept_scorecards("A"),
            (1, 2, 3),
            0,
            (1, 2),
            3,
        )
        self.assertEqual(result["streaks"], ())
        self.assertEqual(
            result["totals"],
            {"identities": 0, "streaks": 0, "regressing_splits": 0},
        )

    def test_only_the_longest_run_is_kept_per_identity(self):
        # Hit pattern over splits (1, 2, 3, 4) is hit, miss, hit,
        # hit: the runs are [1] and [3, 4], so [3, 4] wins.
        records = tuple(
            backtest_record(2 * index + 2, "A", hit, 1 - hit)
            for index, hit in enumerate((1, 0, 1, 1, 1, 1, 0, 0))
        )
        result = build_streaks(
            records, kept_scorecards("A"), (1, 2, 3, 4), 0, (0, 1)
        )
        self.assertEqual(
            result["streaks"],
            (
                {
                    "identity": "A",
                    "start_baseline_cutoffs": 3,
                    "end_baseline_cutoffs": 4,
                    "split_count": 2,
                    "first_regression_cutoff": 16,
                    "worst_rate_drop": (1, 4),
                },
            ),
        )

    def test_equal_length_runs_keep_the_smaller_start(self):
        # Hit pattern over splits (1, 2, 3, 4, 5) is hit, hit, miss,
        # hit, hit: two runs of length two, and the earlier start
        # wins.
        records = tuple(
            backtest_record(2 * index + 2, "A", hit, 1 - hit)
            for index, hit in enumerate((1, 0, 0, 1, 1, 0))
        )
        result = build_streaks(
            records, kept_scorecards("A"), (1, 2, 3, 4, 5), 0, (0, 1)
        )
        self.assertEqual(
            result["streaks"],
            (
                {
                    "identity": "A",
                    "start_baseline_cutoffs": 1,
                    "end_baseline_cutoffs": 2,
                    "split_count": 2,
                    "first_regression_cutoff": 4,
                    "worst_rate_drop": (3, 5),
                },
            ),
        )

    def test_streak_aggregates_min_cutoff_and_max_drop(self):
        # Split 1 regresses first at cutoff 4 with drop 2/3, split 2
        # at cutoff 8 with drop 1/2: the streak takes the earliest
        # cutoff and the largest drop.
        result = build_streaks(
            regressing_records("A"),
            kept_scorecards("A"),
            (1, 2),
            0,
            (1, 2),
        )
        streak = result["streaks"][0]
        self.assertEqual(streak["first_regression_cutoff"], 4)
        self.assertEqual(streak["worst_rate_drop"], (2, 3))

    def test_min_evaluated_delta_and_drop_threshold_gate_each_position(
        self,
    ):
        # Split 1's observation evaluated count exceeds the baseline
        # by 4, split 2's by 0.
        result = build_streaks(
            regressing_records("A"),
            kept_scorecards("A"),
            (1, 2),
            1,
            (1, 2),
        )
        self.assertEqual(
            [(s["start_baseline_cutoffs"], s["split_count"]) for s in result["streaks"]],
            [(1, 1)],
        )
        # A threshold of 3/5 lies between split 2's drop 1/2 and
        # split 1's drop 2/3, so only split 1 still hits.
        result = build_streaks(
            regressing_records("A"),
            kept_scorecards("A"),
            (1, 2),
            0,
            (3, 5),
        )
        self.assertEqual(
            [(s["start_baseline_cutoffs"], s["split_count"]) for s in result["streaks"]],
            [(1, 1)],
        )

    def test_sort_order_count_then_drop_then_identity(self):
        records = (
            regressing_records("A")
            + (
                backtest_record(2, "B", 1, 1),
                backtest_record(4, "B", 1, 1),
                backtest_record(6, "B", 0, 2),
                backtest_record(8, "B", 0, 2),
            )
            + (
                backtest_record(2, "C", 1, 0),
                backtest_record(4, "C", 1, 0),
                backtest_record(6, "C", 0, 1),
                backtest_record(8, "C", 0, 1),
                backtest_record(10, "C", 0, 2),
                backtest_record(12, "C", 0, 2),
            )
            + (
                backtest_record(2, "D", 1, 1),
                backtest_record(4, "D", 1, 1),
                backtest_record(6, "D", 0, 2),
                backtest_record(8, "D", 0, 2),
            )
        )
        result = build_streaks(
            records,
            kept_scorecards("A", "B", "C", "D"),
            (1, 2, 3),
            0,
            (0, 1),
        )
        # C's run covers all three splits; A's drop 2/3 beats B's and
        # D's 1/2; B and D tie and sort by identity.
        self.assertEqual(
            [s["identity"] for s in result["streaks"]],
            ["C", "A", "B", "D"],
        )
        self.assertEqual(
            [s["split_count"] for s in result["streaks"]],
            [3, 2, 2, 2],
        )
        self.assertEqual(
            [s["worst_rate_drop"] for s in result["streaks"]],
            [(1, 1), (2, 3), (1, 2), (1, 2)],
        )
        self.assertEqual(
            result["totals"],
            {"identities": 4, "streaks": 4, "regressing_splits": 9},
        )

    def test_tuple_identities_sort_by_unicode_code_point(self):
        records = ()
        for identity in (("W", "y"), ("a0", "x"), ("A", "z")):
            records += (
                backtest_record(2, identity, 1, 0),
                backtest_record(4, identity, 0, 1),
            )
        result = build_streaks(
            records,
            kept_scorecards(("W", "y"), ("a0", "x"), ("A", "z")),
            (1,),
            0,
            (0, 1),
        )
        self.assertEqual(
            [s["identity"] for s in result["streaks"]],
            [("A", "z"), ("W", "y"), ("a0", "x")],
        )

    def test_streak_limit_bounds_the_complete_set_without_truncating(
        self,
    ):
        records = regressing_records("A") + regressing_records("C")
        with self.assertRaises(ValueError) as caught:
            build_streaks(
                records, kept_scorecards("A", "C"), (1, 2), 0, (1, 2), 1, 1
            )
        self.assertIn("regression streak limit", str(caught.exception))
        result = build_streaks(
            records, kept_scorecards("A", "C"), (1, 2), 0, (1, 2), 1, 2
        )
        self.assertEqual(len(result["streaks"]), 2)
        # Identities filtered out below the limit never count toward
        # it: a 3/4 threshold exceeds both positions' drops.
        result = build_streaks(
            records, kept_scorecards("A", "C"), (1, 2), 0, (3, 4), 1, 1
        )
        self.assertEqual(result["streaks"], ())

    def test_empty_records_return_empty_tuple_and_zero_totals(self):
        result = build_streaks((), ())
        self.assertEqual(result["streaks"], ())
        self.assertEqual(
            result["totals"],
            {"identities": 0, "streaks": 0, "regressing_splits": 0},
        )

    def test_every_object_is_fresh(self):
        records = (
            backtest_record(2, ("A", "x"), 2, 0),
            backtest_record(4, ("A", "x"), 0, 2),
        )
        scorecards = kept_scorecards(("A", "x"))
        result = build_streaks(records, scorecards, (1,), 0, (0, 1))
        result["streaks"][0]["identity"] = "Z"
        result["streaks"][0]["split_count"] = 999
        result["totals"]["regressing_splits"] = 999
        again = build_streaks(records, scorecards, (1,), 0, (0, 1))
        self.assertEqual(again["streaks"][0]["identity"], ("A", "x"))
        self.assertEqual(again["streaks"][0]["split_count"], 1)
        self.assertEqual(again["totals"]["regressing_splits"], 1)
        # The input records are not mutated either.
        self.assertEqual(records[0]["identity"], ("A", "x"))

        seen_ids = []

        def collect(value):
            if isinstance(value, dict):
                seen_ids.append(id(value))
                for item in value.values():
                    collect(item)
            elif isinstance(value, (tuple, list)):
                for item in value:
                    collect(item)

        collect(again)
        self.assertEqual(len(seen_ids), len(set(seen_ids)))


class LifetimeChurnForecastDriftWavePredictionScorecardRegressionStreaksChainTests(
    unittest.TestCase
):
    """The builder refines exactly what the scorecards stage keeps."""

    def test_streaks_match_the_two_cutoff_backtest(self):
        backtests = build_backtests(
            synthetic_waves(), 1, 50, 0, 50, 2, 50, (2, 4), 0, 50
        )
        scorecards = build_scorecards(backtests["backtests"])
        result = build_streaks(
            backtests["backtests"], scorecards["scorecards"], (1,), 0, (0, 1)
        )
        # B has a single record, so its observation segment is empty;
        # A and C tie on every rate and count, then sort by identity.
        self.assertEqual(
            [s["identity"] for s in result["streaks"]], ["A", "C"]
        )
        for streak in result["streaks"]:
            self.assertEqual(
                (
                    streak["start_baseline_cutoffs"],
                    streak["end_baseline_cutoffs"],
                    streak["split_count"],
                    streak["first_regression_cutoff"],
                    streak["worst_rate_drop"],
                ),
                (1, 1, 1, 4, (1, 2)),
            )
        self.assertEqual(
            result["totals"],
            {"identities": 2, "streaks": 2, "regressing_splits": 2},
        )
        # A threshold above the 1/2 drop empties the result.
        result = build_streaks(
            backtests["backtests"],
            scorecards["scorecards"],
            (1,),
            0,
            (1, 1),
        )
        self.assertEqual(result["streaks"], ())
        # Split 2 leaves no observation record, so no position hits.
        result = build_streaks(
            backtests["backtests"],
            scorecards["scorecards"],
            (1, 2),
            0,
            (0, 1),
        )
        self.assertEqual(
            [s["split_count"] for s in result["streaks"]], [1, 1]
        )

    def test_scorecard_filters_gate_the_candidates(self):
        backtests = build_backtests(
            synthetic_waves(), 1, 50, 0, 50, 2, 50, (2, 4), 0, 50
        )
        # min_cutoffs 2 removes B from the scorecards; the streak set
        # is unchanged because B never qualifies anyway, while
        # min_evaluated 5 removes every scorecard and every candidate.
        scorecards = build_scorecards(backtests["backtests"], 0, 2)
        result = build_streaks(
            backtests["backtests"], scorecards["scorecards"], (1,), 0, (0, 1)
        )
        self.assertEqual(
            [s["identity"] for s in result["streaks"]], ["A", "C"]
        )
        scorecards = build_scorecards(backtests["backtests"], 5, 1)
        result = build_streaks(
            backtests["backtests"], scorecards["scorecards"], (1,), 0, (0, 1)
        )
        self.assertEqual(result["streaks"], ())


class LifetimeChurnForecastDriftWavePredictionScorecardRegressionStreaksValidationTests(
    unittest.TestCase
):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        return streaks_call(self.store, self.args, **overrides)

    def extended(self, **overrides):
        overrides.setdefault("split_cutoffs", (4,))
        merged = dict(LONG_ARGS)
        merged.update(overrides)
        return self.call(**merged)

    def test_baseline_splits_must_be_a_nonempty_tuple_of_positive_ints(
        self,
    ):
        for bad in (True, 1, 1.0, "1", None, [1, 2]):
            with self.subTest(value=bad):
                with self.assertRaises(TypeError):
                    self.extended(baseline_splits=bad)
        for bad in ((True, 1), (1, True), (1.0, 2), (1, "2")):
            with self.subTest(value=bad):
                with self.assertRaises(TypeError):
                    self.extended(baseline_splits=bad)
        with self.assertRaises(ValueError):
            self.extended(baseline_splits=())
        for bad in ((0,), (-1,), (1, 1), (2, 1), (1, 3, 3)):
            with self.subTest(value=bad):
                with self.assertRaises(ValueError):
                    self.extended(baseline_splits=bad)
        # A strictly increasing positive tuple is legal.
        self.extended(baseline_splits=(1, 2, 4))

    def test_min_consecutive_splits_must_be_a_non_bool_positive_int(self):
        for bad in (True, False, 1.0, "1", None, (1,)):
            with self.subTest(value=bad):
                with self.assertRaises(TypeError):
                    self.extended(min_consecutive_splits=bad)
        with self.assertRaises(ValueError):
            self.extended(min_consecutive_splits=0)
        with self.assertRaises(ValueError):
            self.extended(min_consecutive_splits=-2)

    def test_regression_streak_limit_must_be_a_non_bool_positive_int(self):
        for bad in (True, False, 1.0, "1", None, (1,)):
            with self.subTest(value=bad):
                with self.assertRaises(TypeError):
                    self.extended(regression_streak_limit=bad)
        with self.assertRaises(ValueError):
            self.extended(regression_streak_limit=0)
        with self.assertRaises(ValueError):
            self.extended(regression_streak_limit=-3)

    def test_new_parameters_validated_in_signature_order(self):
        # prediction_scorecard_limit fails before baseline_splits.
        with self.assertRaises(ValueError):
            self.extended(
                prediction_scorecard_limit=0, baseline_splits="x"
            )
        with self.assertRaises(TypeError):
            self.extended(
                prediction_scorecard_limit="x", baseline_splits="x"
            )
        # baseline_splits fails before min_evaluated_delta.
        with self.assertRaises(TypeError):
            self.extended(baseline_splits="x", min_evaluated_delta="x")
        with self.assertRaises(ValueError):
            self.extended(baseline_splits=(), min_evaluated_delta=-1)
        # min_evaluated_delta fails before min_hit_rate_drop.
        with self.assertRaises(TypeError):
            self.extended(min_evaluated_delta="x", min_hit_rate_drop="x")
        with self.assertRaises(ValueError):
            self.extended(min_evaluated_delta=-1, min_hit_rate_drop=(1, 0))
        # min_hit_rate_drop fails before min_consecutive_splits.
        with self.assertRaises(TypeError):
            self.extended(min_hit_rate_drop="x", min_consecutive_splits="x")
        with self.assertRaises(ValueError):
            self.extended(
                min_hit_rate_drop=(1, 0), min_consecutive_splits=0
            )
        # min_consecutive_splits fails before regression_streak_limit.
        with self.assertRaises(TypeError):
            self.extended(
                min_consecutive_splits="x", regression_streak_limit="x"
            )
        with self.assertRaises(ValueError):
            self.extended(
                min_consecutive_splits=0, regression_streak_limit=0
            )
        # All five pass before the split membership check fires.
        with self.assertRaises(ValueError):
            self.extended(
                split_cutoffs=(99,),
                baseline_splits=(1,),
                min_evaluated_delta=0,
                min_hit_rate_drop=(0, 1),
                min_consecutive_splits=1,
                regression_streak_limit=1,
            )

    def test_all_parameter_checks_precede_state_lookups(self):
        with self.assertRaises(TypeError):
            self.call(reference="ghost", cutoffs=(0, 1), baseline_splits="x")
        with self.assertRaises(ValueError):
            self.call(reference="ghost", cutoffs=(0, 1), baseline_splits=())
        with self.assertRaises(TypeError):
            self.call(
                reference="ghost", cutoffs=(0, 1), min_evaluated_delta="x"
            )
        with self.assertRaises(ValueError):
            self.call(
                reference="ghost", cutoffs=(0, 1), min_evaluated_delta=-1
            )
        with self.assertRaises(TypeError):
            self.call(
                reference="ghost", cutoffs=(0, 1), min_hit_rate_drop="x"
            )
        with self.assertRaises(ValueError):
            self.call(
                reference="ghost", cutoffs=(0, 1), min_hit_rate_drop=(1, 0)
            )
        with self.assertRaises(TypeError):
            self.call(
                reference="ghost",
                cutoffs=(0, 1),
                min_consecutive_splits="x",
            )
        with self.assertRaises(ValueError):
            self.call(
                reference="ghost", cutoffs=(0, 1), min_consecutive_splits=0
            )
        with self.assertRaises(TypeError):
            self.call(
                reference="ghost",
                cutoffs=(0, 1),
                regression_streak_limit="x",
            )
        with self.assertRaises(ValueError):
            self.call(
                reference="ghost", cutoffs=(0, 1), regression_streak_limit=0
            )

    def test_underlying_stage_limits_and_errors_are_unchanged(self):
        with self.assertRaises(ValueError) as caught:
            self.extended(
                wave_cutoffs=(9,),
                baseline_splits=(1,),
                min_evaluated_delta=0,
                min_hit_rate_drop=(0, 1),
                min_consecutive_splits=1,
                regression_streak_limit=50,
            )
        self.assertIn("out of range", str(caught.exception))

    def test_underlying_stage_caps_still_bound_the_shared_backtest(self):
        # The per-cutoff and cross-cutoff caps belong to the shared
        # backtest stage, so they must raise before anything
        # aggregates.
        with self.assertRaises(ValueError) as caught:
            build_backtests(
                synthetic_waves(), 1, 50, 0, 50, 2, 2, (4,), 0, 50
            )
        self.assertIn("prediction result limit", str(caught.exception))
        with self.assertRaises(ValueError) as caught:
            build_backtests(
                synthetic_waves(), 1, 50, 0, 50, 2, 50, (2, 4), 0, 4
            )
        self.assertIn("prediction backtest limit", str(caught.exception))
        # The scorecard stage cap likewise raises before the streak
        # stage aggregates.
        backtests = build_backtests(
            synthetic_waves(), 1, 50, 0, 50, 2, 50, (2, 4), 0, 50
        )
        with self.assertRaises(ValueError) as caught:
            build_scorecards(backtests["backtests"], 0, 1, 2)
        self.assertIn("prediction scorecard limit", str(caught.exception))


class LifetimeChurnForecastDriftWavePredictionScorecardRegressionStreaksEmptyRunTests(
    unittest.TestCase
):
    def setUp(self):
        self.store, self.args = diamond_store()

    def zero_totals(self):
        return {"identities": 0, "streaks": 0, "regressing_splits": 0}

    def test_empty_wave_cutoffs_returns_empty_streaks_and_zero_totals(
        self,
    ):
        result = streaks_call(
            self.store,
            self.args,
            **dict(LONG_ARGS, split_cutoffs=(3, 4, 5), wave_cutoffs=()),
        )
        self.assertEqual(result["streaks"], ())
        self.assertEqual(result["totals"], self.zero_totals())

    def test_no_qualifying_identity_returns_empty_streaks(self):
        # The diamond store yields a single drift wave, so no identity
        # reaches three training appearances at cutoff 0.
        result = streaks_call(
            self.store,
            self.args,
            **dict(LONG_ARGS, split_cutoffs=(3, 4, 5), wave_cutoffs=(0,)),
        )
        self.assertEqual(result["streaks"], ())
        self.assertEqual(result["totals"], self.zero_totals())


class LifetimeChurnForecastDriftWavePredictionScorecardRegressionStreaksIsolationTests(
    unittest.TestCase
):
    def call_args(self):
        return dict(
            LONG_ARGS,
            split_cutoffs=(3, 4, 5),
            wave_horizon=10,
            prediction_result_limit=50,
            wave_cutoffs=(0,),
            match_tolerance=1,
            prediction_backtest_limit=50,
            min_evaluated=0,
            min_cutoffs=1,
            prediction_scorecard_limit=50,
            baseline_splits=(1,),
            min_evaluated_delta=0,
            min_hit_rate_drop=(0, 1),
            min_consecutive_splits=1,
            regression_streak_limit=50,
        )

    def test_success_and_failure_change_nothing(self):
        store, args = diamond_store()
        heads = dict(store._heads)
        appends = copy.deepcopy(store._appends)
        merges = copy.deepcopy(store._merges)
        audit = store.audit_log()

        streaks_call(store, args, **self.call_args())
        failures = [
            dict(wave_cutoffs="x"),
            dict(wave_cutoffs=(True,)),
            dict(wave_cutoffs=(-1,)),
            dict(wave_cutoffs=(0, 0)),
            dict(wave_cutoffs=(1, 0)),
            dict(wave_cutoffs=(9,)),
            dict(match_tolerance="x"),
            dict(match_tolerance=True),
            dict(match_tolerance=-1),
            dict(prediction_backtest_limit="x"),
            dict(prediction_backtest_limit=False),
            dict(prediction_backtest_limit=0),
            dict(prediction_result_limit=0),
            dict(drift_limit=2),
            dict(scan_limit=1),
            dict(min_evaluated="x"),
            dict(min_evaluated=True),
            dict(min_evaluated=-1),
            dict(min_cutoffs="x"),
            dict(min_cutoffs=False),
            dict(min_cutoffs=0),
            dict(prediction_scorecard_limit="x"),
            dict(prediction_scorecard_limit=False),
            dict(prediction_scorecard_limit=0),
            dict(baseline_splits="x"),
            dict(baseline_splits=(True,)),
            dict(baseline_splits=()),
            dict(baseline_splits=(0,)),
            dict(baseline_splits=(2, 2)),
            dict(min_evaluated_delta="x"),
            dict(min_evaluated_delta=True),
            dict(min_evaluated_delta=-1),
            dict(min_hit_rate_drop="x"),
            dict(min_hit_rate_drop=(1, 0)),
            dict(min_hit_rate_drop=(2, 1)),
            dict(min_consecutive_splits="x"),
            dict(min_consecutive_splits=False),
            dict(min_consecutive_splits=0),
            dict(regression_streak_limit="x"),
            dict(regression_streak_limit=False),
            dict(regression_streak_limit=0),
            dict(reference="ghost"),
        ]
        for overrides in failures:
            with self.subTest(overrides=overrides):
                merged = self.call_args()
                merged.update(overrides)
                with self.assertRaises(Exception):
                    streaks_call(store, args, **merged)

        self.assertEqual(dict(store._heads), heads)
        self.assertEqual(copy.deepcopy(store._appends), appends)
        self.assertEqual(copy.deepcopy(store._merges), merges)
        self.assertEqual(store.audit_log(), audit)


class LifetimeChurnForecastDriftWavePredictionScorecardRegressionStreaksTokenTests(
    unittest.TestCase
):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, token=None, **overrides):
        return streaks_call(
            self.store, self.args, token=token, **overrides
        )

    def call_args(self):
        return dict(
            LONG_ARGS,
            split_cutoffs=(3, 4, 5),
            wave_horizon=10,
            prediction_result_limit=50,
            wave_cutoffs=(0,),
            match_tolerance=1,
            prediction_backtest_limit=50,
            min_evaluated=0,
            min_cutoffs=1,
            prediction_scorecard_limit=50,
            baseline_splits=(1,),
            min_evaluated_delta=0,
            min_hit_rate_drop=(0, 1),
            min_consecutive_splits=1,
            regression_streak_limit=50,
        )

    def test_success_consumes_one_read_and_failure_refunds(self):
        token = self.store.create_snapshot(2)
        self.call(token=token, **self.call_args())
        # Stage failures still refund.
        with self.assertRaises(ValueError):
            self.call(token=token, **dict(self.call_args(), drift_limit=2))
        with self.assertRaises(ValueError):
            self.call(token=token, **dict(self.call_args(), scan_limit=1))
        with self.assertRaises(ValueError):
            self.call(token=token, **dict(self.call_args(), wave_cutoffs=(9,)))
        # Parameter validation runs before the token is touched.
        with self.assertRaises(TypeError):
            self.call(
                token=token, **dict(self.call_args(), baseline_splits="x")
            )
        with self.assertRaises(ValueError):
            self.call(
                token=token,
                **dict(self.call_args(), regression_streak_limit=0),
            )
        # The second allowed read succeeds; the token then expires.
        self.call(token=token, **self.call_args())
        with self.assertRaises(RuntimeError):
            self.call(token=token, **self.call_args())

    def test_tokenless_query_leaves_snapshot_quotas_intact(self):
        token = self.store.create_snapshot(1)
        self.call(**self.call_args())
        self.call(**self.call_args())
        # The single read quota is still available after tokenless calls.
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
