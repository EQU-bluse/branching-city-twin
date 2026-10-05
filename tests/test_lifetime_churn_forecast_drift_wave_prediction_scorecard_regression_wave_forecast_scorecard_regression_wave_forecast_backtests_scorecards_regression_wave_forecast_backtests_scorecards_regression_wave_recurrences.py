import copy
import inspect
import unittest

from city_twin.branches import BranchStore

from tests.test_lifetime_churn import diamond_store
from tests.test_lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_wave_forecast_scorecard_regression_wave_forecast_backtests_scorecards_regression_wave_forecast_backtests_scorecards_regressions_streaks import (
    call_args,
    streaks_kwargs,
)


def waves_kwargs(store_args, **overrides) -> dict:
    min_active_regressions = overrides.pop(
        "regression_wave_regression_wave_regression_wave_min_active_regressions", 1
    )
    regression_wave_limit = overrides.pop(
        "regression_wave_regression_wave_regression_wave_regression_wave_limit", 50
    )
    kwargs = streaks_kwargs(store_args, **overrides)
    kwargs["regression_wave_regression_wave_regression_wave_min_active_regressions"] = (
        min_active_regressions
    )
    kwargs["regression_wave_regression_wave_regression_wave_regression_wave_limit"] = (
        regression_wave_limit
    )
    return kwargs


def waves_call(store, store_args, **overrides):
    return store.lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_wave_forecast_scorecard_regression_wave_forecast_backtests_scorecards_regression_wave_forecast_backtests_scorecards_regressions_waves(
        **waves_kwargs(store_args, **overrides)
    )


def recurrences_kwargs(store_args, **overrides) -> dict:
    min_regression_wave_occurrences = overrides.pop(
        "regression_wave_regression_wave_regression_wave_min_regression_wave_occurrences", 1
    )
    regression_wave_recurrence_limit = overrides.pop(
        "regression_wave_regression_wave_regression_wave_regression_wave_recurrence_limit", 50
    )
    kwargs = waves_kwargs(store_args, **overrides)
    kwargs[
        "regression_wave_regression_wave_regression_wave_min_regression_wave_occurrences"
    ] = min_regression_wave_occurrences
    kwargs[
        "regression_wave_regression_wave_regression_wave_regression_wave_recurrence_limit"
    ] = regression_wave_recurrence_limit
    return kwargs


def recurrences_call(store, store_args, **overrides):
    return store.lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_wave_forecast_scorecard_regression_wave_forecast_backtests_scorecards_regression_wave_forecast_backtests_scorecards_regression_wave_recurrences(
        **recurrences_kwargs(store_args, **overrides)
    )


def build_waves(
    streaks,
    regression_wave_regression_wave_regression_wave_baseline_splits=(1,),
    regression_wave_regression_wave_regression_wave_min_active_regressions=1,
    regression_wave_regression_wave_regression_wave_regression_wave_limit=50,
):
    return BranchStore._build_lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_wave_forecast_scorecard_regression_wave_forecast_backtests_scorecards_regression_wave_forecast_backtests_scorecards_regressions_waves_result(
        streaks,
        regression_wave_regression_wave_regression_wave_baseline_splits,
        regression_wave_regression_wave_regression_wave_min_active_regressions,
        regression_wave_regression_wave_regression_wave_regression_wave_limit,
    )


def build_recurrences(
    waves,
    regression_wave_regression_wave_regression_wave_min_regression_wave_occurrences=1,
    regression_wave_regression_wave_regression_wave_regression_wave_recurrence_limit=50,
):
    return BranchStore._build_lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_wave_forecast_scorecard_regression_wave_forecast_backtests_scorecards_regression_wave_forecast_backtests_scorecards_regression_wave_recurrences_result(
        waves,
        regression_wave_regression_wave_regression_wave_min_regression_wave_occurrences,
        regression_wave_regression_wave_regression_wave_regression_wave_recurrence_limit,
    )


def make_wave(start, points):
    # Each point spec is (baseline_cutoffs, active_count, identities).
    return {
        "start_baseline_cutoffs": start,
        "end_baseline_cutoffs": points[-1][0],
        "split_count": len(points),
        "peak_regressions": max(spec[1] for spec in points),
        "points": tuple(
            {
                "baseline_cutoffs": baseline_cutoffs,
                "active_count": active_count,
                "identities": tuple(identities),
            }
            for baseline_cutoffs, active_count, identities in points
        ),
    }


def make_streak(identity, start, end, worst_rate_drop=(1, 2)):
    return {
        "identity": identity,
        "start_baseline_cutoffs": start,
        "end_baseline_cutoffs": end,
        "split_count": end - start + 1,
        "first_regression_cutoff": 0,
        "worst_rate_drop": worst_rate_drop,
    }


def make_streaks(*specs):
    # Each spec is (identity, start, end); the tuple order is the
    # streak records' final order.
    return tuple(make_streak(*spec) for spec in specs)


class LifetimeChurnForecastDriftWavePredictionScorecardRegressionWaveForecastScorecardRegressionWaveForecastBacktestsScorecardsRegressionWaveForecastBacktestsScorecardsRegressionWaveRecurrencesSignatureTests(
    unittest.TestCase
):
    def test_public_signature_appends_two_parameters(self):
        parameters = list(
            inspect.signature(
                BranchStore.lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_wave_forecast_scorecard_regression_wave_forecast_backtests_scorecards_regression_wave_forecast_backtests_scorecards_regression_wave_recurrences
            ).parameters.values()
        )
        names = [p.name for p in parameters]
        wave_names = list(
            inspect.signature(
                BranchStore.lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_wave_forecast_scorecard_regression_wave_forecast_backtests_scorecards_regression_wave_forecast_backtests_scorecards_regressions_waves
            ).parameters
        )
        self.assertEqual(
            names,
            wave_names[:-1]
            + [
                "regression_wave_regression_wave_regression_wave_min_regression_wave_occurrences",
                "regression_wave_regression_wave_regression_wave_regression_wave_recurrence_limit",
                "token",
            ],
        )
        self.assertEqual(
            [p.default for p in parameters[:-1]],
            [inspect.Parameter.empty] * (len(parameters) - 1),
        )
        self.assertIsNone(parameters[-1].default)

    def test_result_shape_and_key_order(self):
        waves = (
            make_wave(1, [(1, 1, ("A",))]),
            make_wave(3, [(3, 1, ("A",))]),
        )
        result = build_recurrences(waves, 2)
        self.assertEqual(list(result), ["recurrences", "totals"])
        self.assertIsInstance(result["recurrences"], tuple)
        record = result["recurrences"][0]
        self.assertEqual(
            list(record),
            [
                "identity",
                "waves",
                "first_wave",
                "last_wave",
                "span",
                "points",
                "peak_active",
                "appearances",
            ],
        )
        self.assertEqual(
            list(record["appearances"][0]),
            [
                "wave",
                "start_baseline_cutoffs",
                "end_baseline_cutoffs",
                "points",
                "first_baseline_cutoffs",
                "last_baseline_cutoffs",
            ],
        )
        self.assertEqual(
            list(result["totals"]),
            ["recurrences", "waves", "points", "first_wave", "last_wave"],
        )


class LifetimeChurnForecastDriftWavePredictionScorecardRegressionWaveForecastScorecardRegressionWaveForecastBacktestsScorecardsRegressionWaveForecastBacktestsScorecardsRegressionWaveRecurrencesResultTests(
    unittest.TestCase
):
    def test_identity_recurring_across_waves(self):
        waves = (
            make_wave(1, [(1, 2, ("A", "B")), (2, 1, ("A",))]),
            make_wave(4, [(4, 1, ("A",)), (5, 3, ("A", "B", "C"))]),
        )
        result = build_recurrences(waves)
        self.assertEqual(len(result["recurrences"]), 3)
        record = result["recurrences"][0]
        self.assertEqual(
            record,
            {
                "identity": "A",
                "waves": 2,
                "first_wave": 0,
                "last_wave": 1,
                "span": 2,
                "points": 4,
                "peak_active": 3,
                "appearances": (
                    {
                        "wave": 0,
                        "start_baseline_cutoffs": 1,
                        "end_baseline_cutoffs": 2,
                        "points": 2,
                        "first_baseline_cutoffs": 1,
                        "last_baseline_cutoffs": 2,
                    },
                    {
                        "wave": 1,
                        "start_baseline_cutoffs": 4,
                        "end_baseline_cutoffs": 5,
                        "points": 2,
                        "first_baseline_cutoffs": 4,
                        "last_baseline_cutoffs": 5,
                    },
                ),
            },
        )
        self.assertEqual(
            result["totals"],
            {
                "recurrences": 3,
                "waves": 5,
                "points": 7,
                "first_wave": 0,
                "last_wave": 1,
            },
        )

    def test_wave_bounds_and_actual_positions_differ(self):
        # The identity only covers the middle position of the wave, so
        # the appearance keeps the wave's own bounds as start/end and
        # the identity's actual position as first/last.
        waves = (
            make_wave(
                1,
                [
                    (1, 1, ("B",)),
                    (2, 2, ("A", "B")),
                    (3, 1, ("B",)),
                ],
            ),
            make_wave(5, [(5, 1, ("A",))]),
        )
        result = build_recurrences(waves)
        record = result["recurrences"][0]
        self.assertEqual(record["identity"], "A")
        self.assertEqual(record["points"], 2)
        self.assertEqual(record["peak_active"], 2)
        appearance = record["appearances"][0]
        self.assertEqual(appearance["start_baseline_cutoffs"], 1)
        self.assertEqual(appearance["end_baseline_cutoffs"], 3)
        self.assertEqual(appearance["points"], 1)
        self.assertEqual(appearance["first_baseline_cutoffs"], 2)
        self.assertEqual(appearance["last_baseline_cutoffs"], 2)

    def test_identity_counts_once_per_wave_however_many_points_list_it(self):
        waves = (
            make_wave(1, [(1, 1, ("A",)), (2, 1, ("A",))]),
            make_wave(4, [(4, 1, ("A",)), (5, 1, ("A",))]),
        )
        result = build_recurrences(waves, 2)
        self.assertEqual(len(result["recurrences"]), 1)
        record = result["recurrences"][0]
        self.assertEqual(record["waves"], 2)
        self.assertEqual(record["points"], 4)
        self.assertEqual(
            [appearance["points"] for appearance in record["appearances"]],
            [2, 2],
        )

    def test_min_regression_wave_occurrences_filters_records(self):
        waves = (
            make_wave(1, [(1, 1, ("A", "B"))]),
            make_wave(3, [(3, 1, ("A",))]),
        )
        result = build_recurrences(waves, 2)
        self.assertEqual(len(result["recurrences"]), 1)
        self.assertEqual(result["recurrences"][0]["identity"], "A")
        self.assertEqual(
            result["totals"],
            {
                "recurrences": 1,
                "waves": 2,
                "points": 2,
                "first_wave": 0,
                "last_wave": 1,
            },
        )
        result = build_recurrences(waves, 3)
        self.assertEqual(result["recurrences"], ())
        self.assertEqual(
            result["totals"],
            {
                "recurrences": 0,
                "waves": 0,
                "points": 0,
                "first_wave": 0,
                "last_wave": 0,
            },
        )

    def test_records_sort_by_waves_points_first_wave_then_encounter(self):
        waves = (
            make_wave(1, [(1, 1, ("c", "a", "b"))]),
            make_wave(3, [(3, 1, ("a", "b")), (4, 1, ("b",))]),
            make_wave(6, [(6, 1, ("a",))]),
        )
        result = build_recurrences(waves)
        # "a" has three waves and three points; "b" has two waves and
        # three points; "c" has one wave. Encounter order across the
        # waves is c, a, b, which only breaks full ties.
        self.assertEqual(
            [record["identity"] for record in result["recurrences"]],
            ["a", "b", "c"],
        )
        # A full tie falls back to first-encounter order.
        waves = (
            make_wave(1, [(1, 1, ("c", "a"))]),
            make_wave(3, [(3, 1, ("a", "c"))]),
        )
        result = build_recurrences(waves)
        self.assertEqual(
            [record["identity"] for record in result["recurrences"]],
            ["c", "a"],
        )

    def test_regression_wave_recurrence_limit_bounds_the_complete_set(
        self,
    ):
        waves = (
            make_wave(1, [(1, 1, ("A", "B"))]),
            make_wave(3, [(3, 1, ("A", "B"))]),
        )
        with self.assertRaises(ValueError) as caught:
            build_recurrences(waves, 1, 1)
        self.assertIn(
            "regression wave recurrence limit", str(caught.exception)
        )
        result = build_recurrences(waves, 1, 2)
        self.assertEqual(len(result["recurrences"]), 2)

    def test_empty_waves_return_empty_tuple_and_zero_totals(self):
        result = build_recurrences(())
        self.assertEqual(result["recurrences"], ())
        self.assertEqual(
            result["totals"],
            {
                "recurrences": 0,
                "waves": 0,
                "points": 0,
                "first_wave": 0,
                "last_wave": 0,
            },
        )

    def test_every_object_is_fresh(self):
        waves = (
            make_wave(1, [(1, 1, (("A", "x"),))]),
            make_wave(3, [(3, 1, (("A", "x"),))]),
        )
        result = build_recurrences(waves)
        result["recurrences"][0]["waves"] = 999
        result["recurrences"][0]["appearances"][0]["points"] = 999
        result["totals"]["recurrences"] = 999
        again = build_recurrences(waves)
        self.assertEqual(again["recurrences"][0]["waves"], 2)
        self.assertEqual(
            again["recurrences"][0]["appearances"][0]["points"], 1
        )
        self.assertEqual(again["totals"]["recurrences"], 1)
        # The wave records are not mutated either.
        self.assertEqual(waves[0]["points"][0]["identities"], (("A", "x"),))

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


class LifetimeChurnForecastDriftWavePredictionScorecardRegressionWaveForecastScorecardRegressionWaveForecastBacktestsScorecardsRegressionWaveForecastBacktestsScorecardsRegressionWaveRecurrencesChainTests(
    unittest.TestCase
):
    """The recurrence builder consumes exactly what the wave stage keeps."""

    def test_recurrences_match_the_waves_result(self):
        streaks = make_streaks(("A", 1, 2), ("B", 1, 1), ("A2", 4, 5))
        waves = build_waves(streaks, (1, 2, 3, 4, 5), 1)
        self.assertEqual(len(waves["waves"]), 2)
        result = build_recurrences(waves["waves"])
        self.assertEqual(
            [record["identity"] for record in result["recurrences"]],
            ["A", "A2", "B"],
        )
        record = result["recurrences"][0]
        self.assertEqual(record["waves"], 1)
        self.assertEqual(record["points"], 2)
        self.assertEqual(record["span"], 1)
        # Only "A" reaches two waves once it also covers the second run.
        streaks = make_streaks(("A", 1, 2), ("A", 4, 5), ("B", 1, 1))
        waves = build_waves(streaks, (1, 2, 3, 4, 5), 1)
        result = build_recurrences(waves["waves"], 2)
        self.assertEqual(len(result["recurrences"]), 1)
        record = result["recurrences"][0]
        self.assertEqual(record["identity"], "A")
        self.assertEqual(record["waves"], 2)
        self.assertEqual(record["first_wave"], 0)
        self.assertEqual(record["last_wave"], 1)
        self.assertEqual(record["span"], 2)
        self.assertEqual(record["points"], 4)

    def test_public_query_matches_the_waves_query(self):
        store, args = diamond_store()
        kwargs = dict(call_args())
        waves = waves_call(store, args, **kwargs)
        result = recurrences_call(store, args, **kwargs)
        self.assertEqual(result, build_recurrences(waves["waves"]))


class LifetimeChurnForecastDriftWavePredictionScorecardRegressionWaveForecastScorecardRegressionWaveForecastBacktestsScorecardsRegressionWaveForecastBacktestsScorecardsRegressionWaveRecurrencesValidationTests(
    unittest.TestCase
):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        return recurrences_call(self.store, self.args, **overrides)

    def extended(self, **overrides):
        merged = dict(call_args())
        merged.update(overrides)
        return self.call(**merged)

    def test_min_regression_wave_occurrences_must_be_a_non_bool_positive_int(
        self,
    ):
        for bad in (True, False, 1.0, "1", None, (1,)):
            with self.subTest(value=bad):
                with self.assertRaises(TypeError):
                    self.extended(
                        regression_wave_regression_wave_regression_wave_min_regression_wave_occurrences=bad
                    )
        with self.assertRaises(ValueError):
            self.extended(
                regression_wave_regression_wave_regression_wave_min_regression_wave_occurrences=0
            )
        with self.assertRaises(ValueError):
            self.extended(
                regression_wave_regression_wave_regression_wave_min_regression_wave_occurrences=-2
            )

    def test_regression_wave_recurrence_limit_must_be_a_non_bool_positive_int(
        self,
    ):
        for bad in (True, False, 1.0, "1", None, (1,)):
            with self.subTest(value=bad):
                with self.assertRaises(TypeError):
                    self.extended(
                        regression_wave_regression_wave_regression_wave_regression_wave_recurrence_limit=bad
                    )
        with self.assertRaises(ValueError):
            self.extended(
                regression_wave_regression_wave_regression_wave_regression_wave_recurrence_limit=0
            )
        with self.assertRaises(ValueError):
            self.extended(
                regression_wave_regression_wave_regression_wave_regression_wave_recurrence_limit=-3
            )

    def test_new_parameters_validated_in_signature_order(self):
        # The deepest wave limit fails before min regression wave
        # occurrences.
        with self.assertRaises(TypeError):
            self.extended(
                regression_wave_regression_wave_regression_wave_regression_wave_limit="x",
                regression_wave_regression_wave_regression_wave_min_regression_wave_occurrences="x",
            )
        with self.assertRaises(ValueError):
            self.extended(
                regression_wave_regression_wave_regression_wave_regression_wave_limit=0,
                regression_wave_regression_wave_regression_wave_min_regression_wave_occurrences=0,
            )
        # min regression wave occurrences fails before the recurrence
        # limit.
        with self.assertRaises(TypeError):
            self.extended(
                regression_wave_regression_wave_regression_wave_min_regression_wave_occurrences="x",
                regression_wave_regression_wave_regression_wave_regression_wave_recurrence_limit="x",
            )
        with self.assertRaises(ValueError):
            self.extended(
                regression_wave_regression_wave_regression_wave_min_regression_wave_occurrences=0,
                regression_wave_regression_wave_regression_wave_regression_wave_recurrence_limit=0,
            )
        # Both pass before the split membership check fires.
        with self.assertRaises(ValueError):
            self.extended(
                split_cutoffs=(99,),
                regression_wave_regression_wave_regression_wave_baseline_splits=(1,),
                regression_wave_regression_wave_regression_wave_min_evaluated_delta=0,
                regression_wave_regression_wave_regression_wave_min_hit_rate_drop=(0, 1),
                regression_wave_regression_wave_regression_wave_regression_limit=1,
                regression_wave_regression_wave_regression_wave_min_consecutive_splits=1,
                regression_wave_regression_wave_regression_wave_regression_streak_limit=1,
                regression_wave_regression_wave_regression_wave_min_active_regressions=1,
                regression_wave_regression_wave_regression_wave_regression_wave_limit=1,
                regression_wave_regression_wave_regression_wave_min_regression_wave_occurrences=1,
                regression_wave_regression_wave_regression_wave_regression_wave_recurrence_limit=1,
            )

    def test_all_parameter_checks_precede_state_lookups(self):
        with self.assertRaises(TypeError):
            self.call(
                reference="ghost",
                regression_wave_regression_wave_regression_wave_min_regression_wave_occurrences="x",
            )
        with self.assertRaises(ValueError):
            self.call(
                reference="ghost",
                regression_wave_regression_wave_regression_wave_min_regression_wave_occurrences=0,
            )
        with self.assertRaises(TypeError):
            self.call(
                reference="ghost",
                regression_wave_regression_wave_regression_wave_regression_wave_recurrence_limit="x",
            )
        with self.assertRaises(ValueError):
            self.call(
                reference="ghost",
                regression_wave_regression_wave_regression_wave_regression_wave_recurrence_limit=0,
            )

    def test_underlying_stage_limits_and_errors_are_unchanged(self):
        with self.assertRaises(ValueError) as caught:
            self.extended(
                wave_cutoffs=(9,),
                regression_wave_regression_wave_regression_wave_baseline_splits=(1,),
                regression_wave_regression_wave_regression_wave_min_evaluated_delta=0,
                regression_wave_regression_wave_regression_wave_min_hit_rate_drop=(0, 1),
                regression_wave_regression_wave_regression_wave_regression_limit=50,
                regression_wave_regression_wave_regression_wave_min_consecutive_splits=1,
                regression_wave_regression_wave_regression_wave_regression_streak_limit=50,
                regression_wave_regression_wave_regression_wave_min_active_regressions=1,
                regression_wave_regression_wave_regression_wave_regression_wave_limit=50,
                regression_wave_regression_wave_regression_wave_min_regression_wave_occurrences=1,
                regression_wave_regression_wave_regression_wave_regression_wave_recurrence_limit=50,
            )
        self.assertIn("out of range", str(caught.exception))


class LifetimeChurnForecastDriftWavePredictionScorecardRegressionWaveForecastScorecardRegressionWaveForecastBacktestsScorecardsRegressionWaveForecastBacktestsScorecardsRegressionWaveRecurrencesEmptyRunTests(
    unittest.TestCase
):
    def setUp(self):
        self.store, self.args = diamond_store()

    def zero_totals(self):
        return {
            "recurrences": 0,
            "waves": 0,
            "points": 0,
            "first_wave": 0,
            "last_wave": 0,
        }

    def test_empty_deep_cutoffs_returns_empty_recurrences(self):
        result = recurrences_call(
            self.store,
            self.args,
            **call_args(regression_wave_regression_wave_regression_wave_cutoffs=()),
        )
        self.assertEqual(result["recurrences"], ())
        self.assertEqual(result["totals"], self.zero_totals())

    def test_no_qualifying_identity_returns_empty_recurrences(self):
        # The diamond store yields no deepest forecast regression waves, so
        # no identity can recur across them.
        result = recurrences_call(
            self.store,
            self.args,
            **call_args(regression_wave_regression_wave_regression_wave_cutoffs=(3,)),
        )
        self.assertEqual(result["recurrences"], ())
        self.assertEqual(result["totals"], self.zero_totals())


class LifetimeChurnForecastDriftWavePredictionScorecardRegressionWaveForecastScorecardRegressionWaveForecastBacktestsScorecardsRegressionWaveForecastBacktestsScorecardsRegressionWaveRecurrencesIsolationTests(
    unittest.TestCase
):
    def test_success_and_failure_change_nothing(self):
        store, args = diamond_store()
        heads = dict(store._heads)
        appends = copy.deepcopy(store._appends)
        merges = copy.deepcopy(store._merges)
        audit = store.audit_log()

        recurrences_call(store, args, **call_args())
        failures = [
            dict(regression_wave_regression_wave_regression_wave_cutoffs="x"),
            dict(regression_wave_regression_wave_regression_wave_cutoffs=(True,)),
            dict(regression_wave_regression_wave_regression_wave_cutoffs=(-1,)),
            dict(regression_wave_regression_wave_regression_wave_cutoffs=(0, 0)),
            dict(regression_wave_regression_wave_regression_wave_cutoffs=(1, 0)),
            dict(regression_wave_regression_wave_regression_wave_match_tolerance="x"),
            dict(regression_wave_regression_wave_regression_wave_match_tolerance=True),
            dict(regression_wave_regression_wave_regression_wave_match_tolerance=-1),
            dict(regression_wave_regression_wave_regression_wave_forecast_backtest_limit="x"),
            dict(
                regression_wave_regression_wave_regression_wave_forecast_backtest_limit=False
            ),
            dict(regression_wave_regression_wave_regression_wave_forecast_backtest_limit=0),
            dict(regression_wave_regression_wave_regression_wave_min_evaluated="x"),
            dict(regression_wave_regression_wave_regression_wave_min_evaluated=True),
            dict(regression_wave_regression_wave_regression_wave_min_evaluated=-1),
            dict(regression_wave_regression_wave_regression_wave_min_cutoffs="x"),
            dict(regression_wave_regression_wave_regression_wave_min_cutoffs=False),
            dict(regression_wave_regression_wave_regression_wave_min_cutoffs=0),
            dict(
                regression_wave_regression_wave_regression_wave_forecast_scorecard_limit="x"
            ),
            dict(
                regression_wave_regression_wave_regression_wave_forecast_scorecard_limit=False
            ),
            dict(regression_wave_regression_wave_regression_wave_forecast_scorecard_limit=0),
            dict(regression_wave_regression_wave_regression_wave_forecast_limit=0),
            dict(regression_wave_regression_wave_regression_wave_horizon=0),
            dict(regression_wave_regression_wave_regression_wave_baseline_splits="x"),
            dict(regression_wave_regression_wave_regression_wave_baseline_splits=False),
            dict(regression_wave_regression_wave_regression_wave_baseline_splits=()),
            dict(regression_wave_regression_wave_regression_wave_baseline_splits=(0,)),
            dict(regression_wave_regression_wave_regression_wave_baseline_splits=(2, 1)),
            dict(regression_wave_regression_wave_regression_wave_min_evaluated_delta="x"),
            dict(regression_wave_regression_wave_regression_wave_min_evaluated_delta=True),
            dict(regression_wave_regression_wave_regression_wave_min_evaluated_delta=-1),
            dict(regression_wave_regression_wave_regression_wave_min_hit_rate_drop="x"),
            dict(regression_wave_regression_wave_regression_wave_min_hit_rate_drop=(1, 0)),
            dict(regression_wave_regression_wave_regression_wave_min_hit_rate_drop=(2, 1)),
            dict(regression_wave_regression_wave_regression_wave_regression_limit="x"),
            dict(regression_wave_regression_wave_regression_wave_regression_limit=False),
            dict(regression_wave_regression_wave_regression_wave_regression_limit=0),
            dict(regression_wave_regression_wave_regression_wave_min_consecutive_splits="x"),
            dict(
                regression_wave_regression_wave_regression_wave_min_consecutive_splits=False
            ),
            dict(regression_wave_regression_wave_regression_wave_min_consecutive_splits=0),
            dict(regression_wave_regression_wave_regression_wave_regression_streak_limit="x"),
            dict(
                regression_wave_regression_wave_regression_wave_regression_streak_limit=False
            ),
            dict(regression_wave_regression_wave_regression_wave_regression_streak_limit=0),
            dict(regression_wave_regression_wave_regression_wave_min_active_regressions="x"),
            dict(regression_wave_regression_wave_regression_wave_min_active_regressions=False),
            dict(regression_wave_regression_wave_regression_wave_min_active_regressions=0),
            dict(regression_wave_regression_wave_regression_wave_regression_wave_limit="x"),
            dict(regression_wave_regression_wave_regression_wave_regression_wave_limit=False),
            dict(regression_wave_regression_wave_regression_wave_regression_wave_limit=0),
            dict(
                regression_wave_regression_wave_regression_wave_min_regression_wave_occurrences="x"
            ),
            dict(
                regression_wave_regression_wave_regression_wave_min_regression_wave_occurrences=False
            ),
            dict(
                regression_wave_regression_wave_regression_wave_min_regression_wave_occurrences=0
            ),
            dict(
                regression_wave_regression_wave_regression_wave_regression_wave_recurrence_limit="x"
            ),
            dict(
                regression_wave_regression_wave_regression_wave_regression_wave_recurrence_limit=False
            ),
            dict(
                regression_wave_regression_wave_regression_wave_regression_wave_recurrence_limit=0
            ),
            dict(drift_limit=2),
            dict(scan_limit=1),
            dict(reference="ghost"),
        ]
        for overrides in failures:
            with self.subTest(overrides=overrides):
                merged = call_args()
                merged.update(overrides)
                with self.assertRaises(Exception):
                    recurrences_call(store, args, **merged)

        self.assertEqual(dict(store._heads), heads)
        self.assertEqual(copy.deepcopy(store._appends), appends)
        self.assertEqual(copy.deepcopy(store._merges), merges)
        self.assertEqual(store.audit_log(), audit)


class LifetimeChurnForecastDriftWavePredictionScorecardRegressionWaveForecastScorecardRegressionWaveForecastBacktestsScorecardsRegressionWaveForecastBacktestsScorecardsRegressionWaveRecurrencesTokenTests(
    unittest.TestCase
):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, token=None, **overrides):
        return recurrences_call(
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
        with self.assertRaises(ValueError):
            self.call(token=token, **call_args(wave_cutoffs=(9,)))
        # Parameter validation runs before the token is touched.
        with self.assertRaises(TypeError):
            self.call(
                token=token,
                **call_args(
                    regression_wave_regression_wave_regression_wave_min_regression_wave_occurrences="x"
                ),
            )
        with self.assertRaises(ValueError):
            self.call(
                token=token,
                **call_args(
                    regression_wave_regression_wave_regression_wave_regression_wave_recurrence_limit=0
                ),
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
