import copy
import inspect
import unittest

from city_twin.branches import BranchStore

from tests.test_lifetime_churn import diamond_store
from tests.test_lifetime_churn_forecast_drift_scan import LONG_ARGS
from tests.test_lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_wave_forecast_backtests import (
    build_backtests,
    forecast_backtests_kwargs,
    synthetic_waves,
)


def forecast_scorecards_kwargs(store_args, **overrides) -> dict:
    regression_wave_min_evaluated = overrides.pop(
        "regression_wave_min_evaluated", 0
    )
    regression_wave_min_cutoffs = overrides.pop(
        "regression_wave_min_cutoffs", 1
    )
    regression_wave_forecast_scorecard_limit = overrides.pop(
        "regression_wave_forecast_scorecard_limit", 50
    )
    kwargs = forecast_backtests_kwargs(store_args, **overrides)
    kwargs["regression_wave_min_evaluated"] = regression_wave_min_evaluated
    kwargs["regression_wave_min_cutoffs"] = regression_wave_min_cutoffs
    kwargs["regression_wave_forecast_scorecard_limit"] = (
        regression_wave_forecast_scorecard_limit
    )
    return kwargs


def forecast_scorecards_call(store, store_args, **overrides):
    return store.lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_wave_forecast_scorecards(
        **forecast_scorecards_kwargs(store_args, **overrides)
    )


def build_scorecards(records, min_evaluated=0, min_cutoffs=1, limit=50):
    return BranchStore._build_lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_wave_forecast_scorecards_result(
        records, min_evaluated, min_cutoffs, limit
    )


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


def call_args(**overrides):
    merged = dict(
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
        regression_limit=50,
        min_consecutive_splits=1,
        regression_streak_limit=50,
        min_active_regressions=1,
        regression_wave_limit=50,
        min_regression_wave_occurrences=1,
        regression_wave_recurrence_limit=50,
        max_regression_wave_jitter=10,
        regression_wave_periodicity_limit=50,
        regression_wave_horizon=10,
        regression_wave_forecast_limit=50,
        regression_wave_cutoffs=(0,),
        regression_wave_match_tolerance=1,
        regression_wave_forecast_backtest_limit=50,
        regression_wave_min_evaluated=0,
        regression_wave_min_cutoffs=1,
        regression_wave_forecast_scorecard_limit=50,
    )
    merged.update(overrides)
    return merged


class LifetimeChurnForecastDriftWavePredictionScorecardRegressionWaveForecastScorecardsSignatureTests(
    unittest.TestCase
):
    def test_public_signature_appends_three_parameters(self):
        parameters = list(
            inspect.signature(
                BranchStore.lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_wave_forecast_scorecards
            ).parameters.values()
        )
        names = [p.name for p in parameters]
        backtest_names = list(
            inspect.signature(
                BranchStore.lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_wave_forecast_backtests
            ).parameters
        )
        self.assertEqual(
            names,
            backtest_names[:-1]
            + [
                "regression_wave_min_evaluated",
                "regression_wave_min_cutoffs",
                "regression_wave_forecast_scorecard_limit",
                "token",
            ],
        )
        self.assertEqual(
            [p.default for p in parameters[:-1]],
            [inspect.Parameter.empty] * (len(parameters) - 1),
        )
        self.assertIsNone(parameters[-1].default)

    def test_result_shape_and_key_order(self):
        result = build_scorecards(
            (
                backtest_record(2, "A", 2, 0),
                backtest_record(4, "A", 1, 1, 2, 1),
            )
        )
        self.assertEqual(list(result), ["scorecards", "totals"])
        self.assertIsInstance(result["scorecards"], tuple)
        scorecard = result["scorecards"][0]
        self.assertEqual(
            list(scorecard),
            [
                "identity",
                "cutoffs",
                "hit",
                "missed",
                "unresolved",
                "unexpected",
                "evaluated",
                "hit_rate",
                "first_failure",
            ],
        )
        self.assertEqual(
            list(result["totals"]),
            [
                "identities",
                "cutoffs",
                "hit",
                "missed",
                "unresolved",
                "unexpected",
                "evaluated",
            ],
        )


class LifetimeChurnForecastDriftWavePredictionScorecardRegressionWaveForecastScorecardsResultTests(
    unittest.TestCase
):
    def test_grouping_aggregates_every_counter_across_cutoffs(self):
        result = build_scorecards(
            (
                backtest_record(2, "A", 2, 0, 0, 0),
                backtest_record(4, "A", 1, 1, 2, 1),
            )
        )
        self.assertEqual(len(result["scorecards"]), 1)
        scorecard = result["scorecards"][0]
        self.assertEqual(
            scorecard,
            {
                "identity": "A",
                "cutoffs": 2,
                "hit": 3,
                "missed": 1,
                "unresolved": 2,
                "unexpected": 1,
                "evaluated": 4,
                "hit_rate": (3, 4),
                "first_failure": 4,
            },
        )
        self.assertEqual(
            result["totals"],
            {
                "identities": 1,
                "cutoffs": 2,
                "hit": 3,
                "missed": 1,
                "unresolved": 2,
                "unexpected": 1,
                "evaluated": 4,
            },
        )

    def test_hit_rate_is_reduced_to_lowest_terms(self):
        result = build_scorecards((backtest_record(0, "A", 2, 2),))
        self.assertEqual(result["scorecards"][0]["hit_rate"], (1, 2))
        result = build_scorecards((backtest_record(0, "A", 4, 0),))
        self.assertEqual(result["scorecards"][0]["hit_rate"], (1, 1))

    def test_zero_evaluated_fixes_hit_rate_to_zero_zero(self):
        # Only unresolved targets: evaluated (hit + missed) is zero.
        result = build_scorecards((backtest_record(0, "A", 0, 0, 3),))
        scorecard = result["scorecards"][0]
        self.assertEqual(scorecard["evaluated"], 0)
        self.assertEqual(scorecard["hit_rate"], (0, 0))

    def test_first_failure_skips_unresolved_only_and_never_failing_is_none(
        self,
    ):
        result = build_scorecards(
            (
                backtest_record(2, "X", 0, 0, 2, 0),
                backtest_record(4, "X", 1, 0, 0, 0),
                backtest_record(6, "X", 0, 1, 0, 0),
            )
        )
        self.assertEqual(result["scorecards"][0]["first_failure"], 6)
        # Cutoff ordering in the records does not decide the answer.
        result = build_scorecards(
            (
                backtest_record(6, "X", 0, 0, 0, 1),
                backtest_record(2, "X", 0, 0, 2, 0),
                backtest_record(4, "X", 1, 0, 0, 0),
            )
        )
        self.assertEqual(result["scorecards"][0]["first_failure"], 6)
        result = build_scorecards(
            (
                backtest_record(2, "Y", 1, 0, 1, 0),
                backtest_record(4, "Y", 2, 0, 4, 0),
            )
        )
        self.assertIsNone(result["scorecards"][0]["first_failure"])

    def test_min_evaluated_and_min_cutoffs_filter(self):
        records = (
            backtest_record(2, "A", 2, 0),
            backtest_record(4, "A", 1, 1),
            backtest_record(2, "C", 1, 1),
            backtest_record(4, "C", 1, 1),
            # B appears at one cutoff with no evaluated targets.
            backtest_record(2, "B", 0, 0, 2),
        )
        self.assertEqual(
            [
                s["identity"]
                for s in build_scorecards(records, 0, 1)["scorecards"]
            ],
            ["A", "C", "B"],
        )
        self.assertEqual(
            [
                s["identity"]
                for s in build_scorecards(records, 3, 1)["scorecards"]
            ],
            ["A", "C"],
        )
        self.assertEqual(
            [
                s["identity"]
                for s in build_scorecards(records, 0, 2)["scorecards"]
            ],
            ["A", "C"],
        )
        # evaluated == threshold and cutoffs == threshold both pass.
        self.assertEqual(
            [
                s["identity"]
                for s in build_scorecards(records, 2, 2)["scorecards"]
            ],
            ["A", "C"],
        )

    def test_sort_order_rate_then_unexpected_missed_identity(self):
        records = (
            # rate 1/2, unexpected 2
            backtest_record(0, "d", 1, 1, 0, 2),
            # rate 1/2, unexpected 1, missed 1
            backtest_record(0, "c", 1, 1, 0, 1),
            # rate 1/2, unexpected 1, missed 2
            backtest_record(0, "b", 2, 2, 0, 1),
            # same counters as b, smaller Unicode identity
            backtest_record(0, "a", 2, 2, 0, 1),
            # rate 1/1 sorts first despite the largest unexpected count
            backtest_record(0, "p", 3, 0, 0, 5),
        )
        self.assertEqual(
            [s["identity"] for s in build_scorecards(records)["scorecards"]],
            ["p", "c", "a", "b", "d"],
        )

    def test_zero_zero_rate_sorts_below_every_real_rate(self):
        records = (
            backtest_record(0, "U", 0, 0, 3, 0),
            backtest_record(0, "P", 0, 0, 0, 1),
            backtest_record(0, "R", 1, 0),
        )
        # R (rate 1) first; the two zero-zero rates then sort by
        # unexpected ascending: U (none) before P (one).
        self.assertEqual(
            [s["identity"] for s in build_scorecards(records)["scorecards"]],
            ["R", "U", "P"],
        )

    def test_tuple_identities_sort_by_unicode_code_point(self):
        records = (
            backtest_record(0, ("W", "y"), 1, 1),
            backtest_record(0, ("a0", "x"), 1, 1),
            backtest_record(0, ("A", "z"), 1, 1),
        )
        self.assertEqual(
            [s["identity"] for s in build_scorecards(records)["scorecards"]],
            [("A", "z"), ("W", "y"), ("a0", "x")],
        )

    def test_totals_cutoffs_is_the_kept_identities_union(self):
        records = (
            backtest_record(2, "A", 1, 0),
            backtest_record(4, "A", 1, 0),
            # B covers cutoff 4 only.
            backtest_record(4, "B", 1, 0),
            # C fails the min_cutoffs threshold and must not add cutoff 6.
            backtest_record(6, "C", 1, 0),
        )
        result = build_scorecards(records, 0, 2)
        self.assertEqual(
            [s["identity"] for s in result["scorecards"]], ["A"]
        )
        self.assertEqual(result["totals"]["cutoffs"], 2)
        # The same union without the threshold keeps both cutoffs plus B's.
        result = build_scorecards(records, 0, 1)
        self.assertEqual(result["totals"]["cutoffs"], 3)

    def test_scorecard_limit_bounds_the_complete_set_without_truncating(self):
        records = (
            backtest_record(0, "A", 1, 0),
            backtest_record(0, "B", 1, 0),
            backtest_record(0, "C", 1, 0),
        )
        with self.assertRaises(ValueError) as caught:
            build_scorecards(records, limit=2)
        self.assertIn("forecast scorecard limit", str(caught.exception))
        result = build_scorecards(records, limit=3)
        self.assertEqual(len(result["scorecards"]), 3)
        # Identities filtered out below the limit never count toward it.
        result = build_scorecards(records, min_cutoffs=2, limit=1)
        self.assertEqual(result["scorecards"], ())

    def test_empty_records_return_empty_tuple_and_zero_totals(self):
        result = build_scorecards(())
        self.assertEqual(result["scorecards"], ())
        self.assertEqual(
            result["totals"],
            {
                "identities": 0,
                "cutoffs": 0,
                "hit": 0,
                "missed": 0,
                "unresolved": 0,
                "unexpected": 0,
                "evaluated": 0,
            },
        )

    def test_every_object_is_fresh(self):
        records = (
            backtest_record(2, ("A", "x"), 1, 0, 0, 0),
            backtest_record(4, ("A", "x"), 0, 1, 1, 1),
        )
        result = build_scorecards(records)
        result["scorecards"][0]["identity"] = "Z"
        result["scorecards"][0]["hit"] = 999
        result["totals"]["hit"] = 999
        again = build_scorecards(records)
        self.assertEqual(again["scorecards"][0]["identity"], ("A", "x"))
        self.assertEqual(again["scorecards"][0]["hit"], 1)
        self.assertEqual(again["totals"]["hit"], 1)
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


class LifetimeChurnForecastDriftWavePredictionScorecardRegressionWaveForecastScorecardsBacktestChainTests(
    unittest.TestCase
):
    """The builder aggregates exactly what a forecast backtest returns."""

    def test_scorecards_match_the_two_cutoff_backtest(self):
        backtests = build_backtests(
            synthetic_waves(), 1, 50, 0, 50, 2, 50, (2, 4), 0, 50
        )
        result = build_scorecards(backtests["backtests"])
        # B (rate 1/1) sorts before A and C (rate 3/4, tied on every
        # other key, then Unicode); B alone covers only cutoff 4.
        self.assertEqual(
            [s["identity"] for s in result["scorecards"]],
            ["B", "A", "C"],
        )
        by_identity = {s["identity"]: s for s in result["scorecards"]}
        self.assertEqual(
            {
                name: (
                    scorecard["cutoffs"],
                    scorecard["hit"],
                    scorecard["missed"],
                    scorecard["unresolved"],
                    scorecard["unexpected"],
                    scorecard["evaluated"],
                    scorecard["hit_rate"],
                    scorecard["first_failure"],
                )
                for name, scorecard in by_identity.items()
            },
            {
                "A": (2, 3, 1, 0, 0, 4, (3, 4), 4),
                "B": (1, 1, 0, 0, 1, 1, (1, 1), 4),
                "C": (2, 3, 1, 0, 0, 4, (3, 4), 4),
            },
        )
        self.assertEqual(
            result["totals"],
            {
                "identities": 3,
                "cutoffs": 2,
                "hit": 7,
                "missed": 2,
                "unresolved": 0,
                "unexpected": 1,
                "evaluated": 9,
            },
        )
        # Raising min_cutoffs removes only the single-cutoff identity.
        result = build_scorecards(backtests["backtests"], 0, 2)
        self.assertEqual(
            [s["identity"] for s in result["scorecards"]], ["A", "C"]
        )
        self.assertEqual(
            result["totals"],
            {
                "identities": 2,
                "cutoffs": 2,
                "hit": 6,
                "missed": 2,
                "unresolved": 0,
                "unexpected": 0,
                "evaluated": 8,
            },
        )

    def test_public_query_matches_the_builder_on_the_backtest_set(self):
        store, args = diamond_store()
        backtest_parameters = inspect.signature(
            BranchStore.lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_wave_forecast_backtests
        ).parameters
        full_kwargs = forecast_scorecards_kwargs(args, **call_args())
        backtests = store.lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_wave_forecast_backtests(
            **{
                key: value
                for key, value in full_kwargs.items()
                if key in backtest_parameters
            }
        )
        result = forecast_scorecards_call(store, args, **call_args())
        self.assertEqual(
            result,
            build_scorecards(backtests["backtests"], 0, 1, 50),
        )


class LifetimeChurnForecastDriftWavePredictionScorecardRegressionWaveForecastScorecardsValidationTests(
    unittest.TestCase
):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        return forecast_scorecards_call(self.store, self.args, **overrides)

    def extended(self, **overrides):
        overrides.setdefault("split_cutoffs", (4,))
        overrides.setdefault("baseline_splits", (1,))
        merged = dict(LONG_ARGS)
        merged.update(overrides)
        return self.call(**merged)

    def test_regression_wave_min_evaluated_must_be_a_non_bool_non_negative_int(
        self,
    ):
        for bad in (True, False, 1.0, "0", None, (0,)):
            with self.subTest(value=bad):
                with self.assertRaises(TypeError):
                    self.extended(regression_wave_min_evaluated=bad)
        with self.assertRaises(ValueError):
            self.extended(regression_wave_min_evaluated=-1)
        # Zero is the lower bound and succeeds.
        self.extended(regression_wave_min_evaluated=0)

    def test_regression_wave_min_cutoffs_must_be_a_non_bool_positive_int(
        self,
    ):
        for bad in (True, False, 1.0, "1", None, (1,)):
            with self.subTest(value=bad):
                with self.assertRaises(TypeError):
                    self.extended(regression_wave_min_cutoffs=bad)
        with self.assertRaises(ValueError):
            self.extended(regression_wave_min_cutoffs=0)
        with self.assertRaises(ValueError):
            self.extended(regression_wave_min_cutoffs=-2)

    def test_regression_wave_forecast_scorecard_limit_must_be_a_non_bool_positive_int(
        self,
    ):
        for bad in (True, False, 1.0, "1", None, (1,)):
            with self.subTest(value=bad):
                with self.assertRaises(TypeError):
                    self.extended(
                        regression_wave_forecast_scorecard_limit=bad
                    )
        with self.assertRaises(ValueError):
            self.extended(regression_wave_forecast_scorecard_limit=0)
        with self.assertRaises(ValueError):
            self.extended(regression_wave_forecast_scorecard_limit=-3)

    def test_new_parameters_validated_in_signature_order(self):
        # regression_wave_forecast_backtest_limit fails before
        # regression_wave_min_evaluated.
        with self.assertRaises(ValueError):
            self.extended(
                regression_wave_forecast_backtest_limit=0,
                regression_wave_min_evaluated="x",
            )
        with self.assertRaises(TypeError):
            self.extended(
                regression_wave_forecast_backtest_limit="x",
                regression_wave_min_evaluated="x",
            )
        # regression_wave_min_evaluated fails before
        # regression_wave_min_cutoffs.
        with self.assertRaises(TypeError):
            self.extended(
                regression_wave_min_evaluated="x",
                regression_wave_min_cutoffs="x",
            )
        with self.assertRaises(ValueError):
            self.extended(
                regression_wave_min_evaluated=-1,
                regression_wave_min_cutoffs=0,
            )
        # regression_wave_min_cutoffs fails before
        # regression_wave_forecast_scorecard_limit.
        with self.assertRaises(TypeError):
            self.extended(
                regression_wave_min_cutoffs="x",
                regression_wave_forecast_scorecard_limit="x",
            )
        with self.assertRaises(ValueError):
            self.extended(
                regression_wave_min_cutoffs=0,
                regression_wave_forecast_scorecard_limit=0,
            )
        # All three pass before the split membership check fires.
        with self.assertRaises(ValueError):
            self.extended(
                split_cutoffs=(99,),
                regression_wave_min_evaluated=0,
                regression_wave_min_cutoffs=1,
                regression_wave_forecast_scorecard_limit=1,
            )

    def test_all_parameter_checks_precede_state_lookups(self):
        with self.assertRaises(TypeError):
            self.call(reference="ghost", regression_wave_min_evaluated="x")
        with self.assertRaises(ValueError):
            self.call(reference="ghost", regression_wave_min_evaluated=-1)
        with self.assertRaises(TypeError):
            self.call(reference="ghost", regression_wave_min_cutoffs="x")
        with self.assertRaises(ValueError):
            self.call(reference="ghost", regression_wave_min_cutoffs=0)
        with self.assertRaises(TypeError):
            self.call(
                reference="ghost",
                regression_wave_forecast_scorecard_limit="x",
            )
        with self.assertRaises(ValueError):
            self.call(
                reference="ghost",
                regression_wave_forecast_scorecard_limit=0,
            )

    def test_underlying_stage_limits_and_errors_are_unchanged(self):
        # The diamond store yields one drift wave, so a drift wave
        # cutoff past it keeps the shared backtest stage's range error.
        with self.assertRaises(ValueError) as caught:
            self.extended(
                wave_cutoffs=(9,),
                regression_wave_min_evaluated=0,
                regression_wave_min_cutoffs=1,
                regression_wave_forecast_scorecard_limit=50,
            )
        self.assertIn("out of range", str(caught.exception))

    def test_underlying_stage_caps_still_bound_the_shared_backtest(self):
        # The per-cutoff and cross-cutoff caps belong to the shared
        # backtest stage, so they must raise before scorecards aggregate.
        with self.assertRaises(ValueError) as caught:
            build_backtests(
                synthetic_waves(), 1, 50, 0, 50, 2, 2, (4,), 0, 50
            )
        self.assertIn("forecast limit", str(caught.exception))
        with self.assertRaises(ValueError) as caught:
            build_backtests(
                synthetic_waves(), 1, 50, 0, 50, 2, 50, (2, 4), 0, 4
            )
        self.assertIn("backtest limit", str(caught.exception))


class LifetimeChurnForecastDriftWavePredictionScorecardRegressionWaveForecastScorecardsEmptyRunTests(
    unittest.TestCase
):
    def setUp(self):
        self.store, self.args = diamond_store()

    def zero_totals(self):
        return {
            "identities": 0,
            "cutoffs": 0,
            "hit": 0,
            "missed": 0,
            "unresolved": 0,
            "unexpected": 0,
            "evaluated": 0,
        }

    def test_empty_regression_wave_cutoffs_returns_empty_scorecards(self):
        result = forecast_scorecards_call(
            self.store,
            self.args,
            **call_args(regression_wave_cutoffs=()),
        )
        self.assertEqual(result["scorecards"], ())
        self.assertEqual(result["totals"], self.zero_totals())

    def test_no_regression_waves_returns_empty_scorecards(self):
        # The diamond store yields no regression waves at all, so the
        # range check is skipped and nothing trains.
        result = forecast_scorecards_call(
            self.store,
            self.args,
            **call_args(regression_wave_cutoffs=(3,)),
        )
        self.assertEqual(result["scorecards"], ())
        self.assertEqual(result["totals"], self.zero_totals())


class LifetimeChurnForecastDriftWavePredictionScorecardRegressionWaveForecastScorecardsIsolationTests(
    unittest.TestCase
):
    def test_success_and_failure_change_nothing(self):
        store, args = diamond_store()
        heads = dict(store._heads)
        appends = copy.deepcopy(store._appends)
        merges = copy.deepcopy(store._merges)
        audit = store.audit_log()

        forecast_scorecards_call(store, args, **call_args())
        failures = [
            dict(regression_wave_cutoffs="x"),
            dict(regression_wave_cutoffs=(True,)),
            dict(regression_wave_cutoffs=(-1,)),
            dict(regression_wave_cutoffs=(0, 0)),
            dict(regression_wave_cutoffs=(1, 0)),
            dict(regression_wave_match_tolerance="x"),
            dict(regression_wave_match_tolerance=True),
            dict(regression_wave_match_tolerance=-1),
            dict(regression_wave_forecast_backtest_limit="x"),
            dict(regression_wave_forecast_backtest_limit=False),
            dict(regression_wave_forecast_backtest_limit=0),
            dict(regression_wave_min_evaluated="x"),
            dict(regression_wave_min_evaluated=True),
            dict(regression_wave_min_evaluated=-1),
            dict(regression_wave_min_cutoffs="x"),
            dict(regression_wave_min_cutoffs=False),
            dict(regression_wave_min_cutoffs=0),
            dict(regression_wave_forecast_scorecard_limit="x"),
            dict(regression_wave_forecast_scorecard_limit=False),
            dict(regression_wave_forecast_scorecard_limit=0),
            dict(regression_wave_forecast_limit=0),
            dict(regression_wave_horizon=0),
            dict(drift_limit=2),
            dict(scan_limit=1),
            dict(reference="ghost"),
        ]
        for overrides in failures:
            with self.subTest(overrides=overrides):
                merged = call_args()
                merged.update(overrides)
                with self.assertRaises(Exception):
                    forecast_scorecards_call(store, args, **merged)

        self.assertEqual(dict(store._heads), heads)
        self.assertEqual(copy.deepcopy(store._appends), appends)
        self.assertEqual(copy.deepcopy(store._merges), merges)
        self.assertEqual(store.audit_log(), audit)


class LifetimeChurnForecastDriftWavePredictionScorecardRegressionWaveForecastScorecardsTokenTests(
    unittest.TestCase
):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, token=None, **overrides):
        return forecast_scorecards_call(
            self.store, self.args, token=token, **overrides
        )

    def test_success_consumes_one_read_and_failure_refunds(self):
        token = self.store.create_snapshot(2)
        self.call(token=token, **call_args())
        # Stage failures still refund.
        with self.assertRaises(ValueError):
            self.call(token=token, **call_args(drift_limit=2))
        with self.assertRaises(ValueError):
            self.call(token=token, **call_args(scan_limit=1))
        # Parameter validation runs before the token is touched.
        with self.assertRaises(TypeError):
            self.call(
                token=token,
                **call_args(regression_wave_min_evaluated="x"),
            )
        with self.assertRaises(ValueError):
            self.call(
                token=token,
                **call_args(regression_wave_forecast_scorecard_limit=0),
            )
        # The second allowed read succeeds; the token then expires.
        self.call(token=token, **call_args())
        with self.assertRaises(RuntimeError):
            self.call(token=token, **call_args())

    def test_tokenless_query_leaves_snapshot_quotas_intact(self):
        token = self.store.create_snapshot(1)
        self.call(**call_args())
        self.call(**call_args())
        # The single read quota is still available after tokenless calls.
        self.call(token=token, **call_args())
        with self.assertRaises(RuntimeError):
            self.call(token=token, **call_args())

    def test_results_come_from_the_frozen_view(self):
        before = self.call(**call_args())
        token = self.store.create_snapshot(4)
        self.store.append("a", "a2", 7, {"x": -1})
        self.store.create("d", "root")
        self.store.append("d", "d0", 8, {"z": 3})
        self.store.merge("a", "d", "M", 9, {"z": 1})
        self.store.append("main", "r3", 10, {"x": 1})
        self.assertEqual(self.call(token=token, **call_args()), before)


if __name__ == "__main__":
    unittest.main()
