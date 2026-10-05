import copy
import inspect
import unittest

from city_twin.branches import BranchStore

from tests.test_lifetime_churn import diamond_store
from tests.test_lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_wave_forecast_scorecard_regression_wave_forecast_backtests_scorecards_regression_wave_forecast_backtests_scorecards_regressions_streaks import (
    call_args,
)
from tests.test_lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_wave_forecast_scorecard_regression_wave_forecast_backtests_scorecards_regression_wave_forecast_backtests_scorecards_regression_wave_recurrences import (
    recurrences_call,
    recurrences_kwargs,
)


def periodicities_kwargs(store_args, **overrides) -> dict:
    max_regression_wave_jitter = overrides.pop(
        "regression_wave_regression_wave_regression_wave_max_regression_wave_jitter",
        10,
    )
    regression_wave_periodicity_limit = overrides.pop(
        "regression_wave_regression_wave_regression_wave_regression_wave_periodicity_limit",
        50,
    )
    kwargs = recurrences_kwargs(store_args, **overrides)
    kwargs[
        "regression_wave_regression_wave_regression_wave_max_regression_wave_jitter"
    ] = max_regression_wave_jitter
    kwargs[
        "regression_wave_regression_wave_regression_wave_regression_wave_periodicity_limit"
    ] = regression_wave_periodicity_limit
    return kwargs


def periodicities_call(store, store_args, **overrides):
    return store.lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_wave_forecast_scorecard_regression_wave_forecast_backtests_scorecards_regression_wave_forecast_backtests_scorecards_regression_wave_periodicities(
        **periodicities_kwargs(store_args, **overrides)
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


def build_periodicities(
    recurrences,
    regression_wave_regression_wave_regression_wave_max_regression_wave_jitter=10,
    regression_wave_regression_wave_regression_wave_regression_wave_periodicity_limit=50,
):
    return BranchStore._build_lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_wave_forecast_scorecard_regression_wave_forecast_backtests_scorecards_regression_wave_forecast_backtests_scorecards_regression_wave_periodicities_result(
        recurrences,
        regression_wave_regression_wave_regression_wave_max_regression_wave_jitter,
        regression_wave_regression_wave_regression_wave_regression_wave_periodicity_limit,
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


def deep_call_args(**overrides):
    merged = dict(
        call_args(),
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
        regression_wave_regression_wave_regression_wave_max_regression_wave_jitter=10,
        regression_wave_regression_wave_regression_wave_regression_wave_periodicity_limit=50,
    )
    merged.update(overrides)
    return merged


class LifetimeChurnForecastDriftWavePredictionScorecardRegressionWaveForecastScorecardRegressionWaveForecastBacktestsScorecardsRegressionWaveForecastBacktestsScorecardsRegressionWavePeriodicitiesSignatureTests(
    unittest.TestCase
):
    def test_public_signature_appends_two_parameters(self):
        parameters = list(
            inspect.signature(
                BranchStore.lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_wave_forecast_scorecard_regression_wave_forecast_backtests_scorecards_regression_wave_forecast_backtests_scorecards_regression_wave_periodicities
            ).parameters.values()
        )
        names = [p.name for p in parameters]
        recurrence_names = list(
            inspect.signature(
                BranchStore.lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_wave_forecast_scorecard_regression_wave_forecast_backtests_scorecards_regression_wave_forecast_backtests_scorecards_regression_wave_recurrences
            ).parameters
        )
        self.assertEqual(
            names,
            recurrence_names[:-1]
            + [
                "regression_wave_regression_wave_regression_wave_max_regression_wave_jitter",
                "regression_wave_regression_wave_regression_wave_regression_wave_periodicity_limit",
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
            make_wave(2, [(2, 1, ("A",))]),
            make_wave(3, [(3, 1, ("A",))]),
        )
        recurrences = build_recurrences(waves)
        result = build_periodicities(recurrences["recurrences"])
        self.assertEqual(list(result), ["periodicities", "totals"])
        self.assertIsInstance(result["periodicities"], tuple)
        record = result["periodicities"][0]
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
                "intervals",
                "period",
                "jitter",
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
            [
                "periodicities",
                "waves",
                "points",
                "first_wave",
                "last_wave",
                "mean_period",
            ],
        )


class LifetimeChurnForecastDriftWavePredictionScorecardRegressionWaveForecastScorecardRegressionWaveForecastBacktestsScorecardsRegressionWaveForecastBacktestsScorecardsRegressionWavePeriodicitiesResultTests(
    unittest.TestCase
):
    def test_steady_cadence_record(self):
        waves = (
            make_wave(1, [(1, 2, ("A",))]),
            make_wave(2, [(2, 1, ("A",))]),
            make_wave(3, [(3, 3, ("A",))]),
        )
        recurrences = build_recurrences(waves)
        result = build_periodicities(recurrences["recurrences"])
        self.assertEqual(len(result["periodicities"]), 1)
        record = result["periodicities"][0]
        self.assertEqual(record["identity"], "A")
        self.assertEqual(record["waves"], 3)
        self.assertEqual(record["first_wave"], 0)
        self.assertEqual(record["last_wave"], 2)
        self.assertEqual(record["span"], 3)
        self.assertEqual(record["points"], 3)
        self.assertEqual(record["peak_active"], 3)
        self.assertEqual(record["intervals"], (1, 1))
        self.assertEqual(record["period"], 1)
        self.assertEqual(record["jitter"], 0)
        self.assertEqual(
            result["totals"],
            {
                "periodicities": 1,
                "waves": 3,
                "points": 3,
                "first_wave": 0,
                "last_wave": 2,
                "mean_period": 1,
            },
        )

    def test_two_appearances_never_qualify(self):
        waves = (
            make_wave(1, [(1, 1, ("A",))]),
            make_wave(2, [(2, 1, ("A",))]),
        )
        recurrences = build_recurrences(waves)
        self.assertEqual(len(recurrences["recurrences"]), 1)
        result = build_periodicities(recurrences["recurrences"])
        self.assertEqual(result["periodicities"], ())
        self.assertEqual(
            result["totals"],
            {
                "periodicities": 0,
                "waves": 0,
                "points": 0,
                "first_wave": 0,
                "last_wave": 0,
                "mean_period": 0,
            },
        )

    def test_even_interval_count_takes_the_smaller_middle_period(self):
        waves = (
            make_wave(1, [(1, 1, ("A",))]),
            make_wave(2, [(2, 1, ("B",))]),
            make_wave(3, [(3, 1, ("A",))]),
            make_wave(4, [(4, 1, ("B",))]),
            make_wave(5, [(5, 1, ("C",))]),
            make_wave(6, [(6, 1, ("C",))]),
            make_wave(7, [(7, 1, ("A",))]),
        )
        # "A" takes part in waves 0, 2 and 6; "B" and "C" never reach
        # three appearances.
        recurrences = build_recurrences(waves)
        result = build_periodicities(recurrences["recurrences"])
        self.assertEqual(len(result["periodicities"]), 1)
        record = result["periodicities"][0]
        self.assertEqual(record["intervals"], (2, 4))
        self.assertEqual(record["period"], 2)
        self.assertEqual(record["jitter"], 2)

    def test_odd_interval_count_takes_the_middle_period(self):
        waves = (
            make_wave(1, [(1, 1, ("A",))]),
            make_wave(2, [(2, 1, ("A",))]),
            make_wave(3, [(3, 1, ("A",))]),
            make_wave(4, [(4, 1, ("B",))]),
            make_wave(5, [(5, 1, ("A",))]),
        )
        # "A" takes part in waves 0, 1, 2 and 4.
        recurrences = build_recurrences(waves)
        result = build_periodicities(recurrences["recurrences"])
        record = result["periodicities"][0]
        self.assertEqual(record["intervals"], (1, 1, 2))
        self.assertEqual(record["period"], 1)
        self.assertEqual(record["jitter"], 1)

    def test_max_regression_wave_jitter_filters_records(self):
        waves = (
            make_wave(1, [(1, 1, ("A", "B"))]),
            make_wave(2, [(2, 1, ("A", "B"))]),
            make_wave(3, [(3, 1, ("A",))]),
            make_wave(4, [(4, 1, ("B",))]),
        )
        # "A" has intervals (1, 1) and jitter 0; "B" has intervals
        # (1, 2) and jitter 1.
        recurrences = build_recurrences(waves)
        result = build_periodicities(
            recurrences["recurrences"],
            regression_wave_regression_wave_regression_wave_max_regression_wave_jitter=0,
        )
        self.assertEqual(
            [record["identity"] for record in result["periodicities"]],
            ["A"],
        )
        result = build_periodicities(
            recurrences["recurrences"],
            regression_wave_regression_wave_regression_wave_max_regression_wave_jitter=1,
        )
        self.assertEqual(
            [record["identity"] for record in result["periodicities"]],
            ["A", "B"],
        )

    def test_records_sort_by_jitter_waves_points_first_wave_then_encounter(
        self,
    ):
        waves = (
            make_wave(1, [(1, 1, ("c", "b", "a", "d"))]),
            make_wave(2, [(2, 1, ("a", "c", "d")), (3, 1, ("a",))]),
            make_wave(3, [(3, 1, ("a", "b", "d"))]),
            make_wave(4, [(4, 1, ("c", "d"))]),
            make_wave(5, [(5, 1, ("b",))]),
        )
        # "a": waves 0-2, intervals (1, 1), jitter 0, four points.
        # "d": waves 0-3, intervals (1, 1, 1), jitter 0, four points.
        # "b": waves 0, 2, 4, intervals (2, 2), jitter 0, three points.
        # "c": waves 0, 1, 3, intervals (1, 2), jitter 1.
        recurrences = build_recurrences(waves)
        result = build_periodicities(recurrences["recurrences"])
        # Jitter 0 first; within it waves descending ("d" has four),
        # then points descending ("a" has four, "b" three). A full tie
        # would fall back to first-encounter order across the waves.
        self.assertEqual(
            [record["identity"] for record in result["periodicities"]],
            ["d", "a", "b", "c"],
        )
        # A full tie falls back to first-encounter order.
        waves = (
            make_wave(1, [(1, 1, ("c", "a"))]),
            make_wave(2, [(2, 1, ("c", "a"))]),
            make_wave(3, [(3, 1, ("c", "a"))]),
        )
        recurrences = build_recurrences(waves)
        result = build_periodicities(recurrences["recurrences"])
        self.assertEqual(
            [record["identity"] for record in result["periodicities"]],
            ["c", "a"],
        )

    def test_regression_wave_periodicity_limit_bounds_the_complete_set(
        self,
    ):
        waves = (
            make_wave(1, [(1, 1, ("A", "B"))]),
            make_wave(2, [(2, 1, ("A", "B"))]),
            make_wave(3, [(3, 1, ("A", "B"))]),
        )
        recurrences = build_recurrences(waves)
        with self.assertRaises(ValueError) as caught:
            build_periodicities(recurrences["recurrences"], 10, 1)
        self.assertIn(
            "regression wave periodicity limit", str(caught.exception)
        )
        result = build_periodicities(recurrences["recurrences"], 10, 2)
        self.assertEqual(len(result["periodicities"]), 2)

    def test_mean_period_is_the_floored_average(self):
        waves = (
            make_wave(1, [(1, 1, ("A", "B"))]),
            make_wave(2, [(2, 1, ("A",))]),
            make_wave(3, [(3, 1, ("A", "B"))]),
            make_wave(4, [(4, 1, ("C",))]),
            make_wave(5, [(5, 1, ("B",))]),
        )
        # "A" takes part in waves 0-2 with period 1; "B" takes part in
        # waves 0, 2 and 4 with intervals (2, 2) and period 2, so the
        # floored mean of (1, 2) is 1.
        recurrences = build_recurrences(waves)
        result = build_periodicities(recurrences["recurrences"])
        self.assertEqual(result["totals"]["mean_period"], 1)

    def test_empty_recurrences_return_empty_tuple_and_zero_totals(self):
        result = build_periodicities(())
        self.assertEqual(result["periodicities"], ())
        self.assertEqual(
            result["totals"],
            {
                "periodicities": 0,
                "waves": 0,
                "points": 0,
                "first_wave": 0,
                "last_wave": 0,
                "mean_period": 0,
            },
        )

    def test_every_object_is_fresh(self):
        waves = (
            make_wave(1, [(1, 1, (("A", "x"),))]),
            make_wave(2, [(2, 1, (("A", "x"),))]),
            make_wave(3, [(3, 1, (("A", "x"),))]),
        )
        recurrences = build_recurrences(waves)
        result = build_periodicities(recurrences["recurrences"])
        result["periodicities"][0]["waves"] = 999
        result["periodicities"][0]["appearances"][0]["points"] = 999
        result["totals"]["periodicities"] = 999
        again = build_periodicities(recurrences["recurrences"])
        self.assertEqual(again["periodicities"][0]["waves"], 3)
        self.assertEqual(
            again["periodicities"][0]["appearances"][0]["points"], 1
        )
        self.assertEqual(again["totals"]["periodicities"], 1)
        # The recurrence records are not mutated either.
        self.assertEqual(recurrences["recurrences"][0]["waves"], 3)

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


class LifetimeChurnForecastDriftWavePredictionScorecardRegressionWaveForecastScorecardRegressionWaveForecastBacktestsScorecardsRegressionWaveForecastBacktestsScorecardsRegressionWavePeriodicitiesChainTests(
    unittest.TestCase
):
    """The periodicity builder consumes exactly what recurrences keep."""

    def test_periodicities_match_the_recurrences_result(self):
        waves = (
            make_wave(1, [(1, 2, ("A",))]),
            make_wave(2, [(2, 1, ("A",))]),
            make_wave(3, [(3, 3, ("A",))]),
        )
        recurrences = build_recurrences(waves)
        self.assertEqual(len(recurrences["recurrences"]), 1)
        result = build_periodicities(recurrences["recurrences"])
        self.assertEqual(len(result["periodicities"]), 1)
        record = result["periodicities"][0]
        self.assertEqual(record["identity"], "A")
        self.assertEqual(record["waves"], 3)
        self.assertEqual(record["first_wave"], 0)
        self.assertEqual(record["last_wave"], 2)
        self.assertEqual(record["span"], 3)
        self.assertEqual(record["intervals"], (1, 1))
        self.assertEqual(record["period"], 1)
        self.assertEqual(record["jitter"], 0)

    def test_public_query_matches_the_recurrences_query(self):
        store, args = diamond_store()
        kwargs = deep_call_args()
        recurrence_kwargs = dict(kwargs)
        del recurrence_kwargs[
            "regression_wave_regression_wave_regression_wave_max_regression_wave_jitter"
        ]
        del recurrence_kwargs[
            "regression_wave_regression_wave_regression_wave_regression_wave_periodicity_limit"
        ]
        recurrences = recurrences_call(store, args, **recurrence_kwargs)
        result = periodicities_call(store, args, **kwargs)
        self.assertEqual(
            result,
            build_periodicities(
                recurrences["recurrences"],
                kwargs[
                    "regression_wave_regression_wave_regression_wave_max_regression_wave_jitter"
                ],
                kwargs[
                    "regression_wave_regression_wave_regression_wave_regression_wave_periodicity_limit"
                ],
            ),
        )


class LifetimeChurnForecastDriftWavePredictionScorecardRegressionWaveForecastScorecardRegressionWaveForecastBacktestsScorecardsRegressionWaveForecastBacktestsScorecardsRegressionWavePeriodicitiesValidationTests(
    unittest.TestCase
):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        return periodicities_call(self.store, self.args, **overrides)

    def extended(self, **overrides):
        merged = dict(call_args())
        merged.update(overrides)
        return self.call(**merged)

    def test_max_regression_wave_jitter_must_be_a_non_bool_non_negative_int(
        self,
    ):
        for bad in (True, False, 1.0, "1", None, (1,)):
            with self.subTest(value=bad):
                with self.assertRaises(TypeError):
                    self.extended(
                        regression_wave_regression_wave_regression_wave_max_regression_wave_jitter=bad
                    )
        with self.assertRaises(ValueError):
            self.extended(
                regression_wave_regression_wave_regression_wave_max_regression_wave_jitter=-1
            )
        with self.assertRaises(ValueError):
            self.extended(
                regression_wave_regression_wave_regression_wave_max_regression_wave_jitter=-3
            )
        # Zero is allowed.
        self.extended(
            regression_wave_regression_wave_regression_wave_max_regression_wave_jitter=0
        )

    def test_regression_wave_periodicity_limit_must_be_a_non_bool_positive_int(
        self,
    ):
        for bad in (True, False, 1.0, "1", None, (1,)):
            with self.subTest(value=bad):
                with self.assertRaises(TypeError):
                    self.extended(
                        regression_wave_regression_wave_regression_wave_regression_wave_periodicity_limit=bad
                    )
        with self.assertRaises(ValueError):
            self.extended(
                regression_wave_regression_wave_regression_wave_regression_wave_periodicity_limit=0
            )
        with self.assertRaises(ValueError):
            self.extended(
                regression_wave_regression_wave_regression_wave_regression_wave_periodicity_limit=-3
            )

    def test_new_parameters_validated_in_signature_order(self):
        # The deepest recurrence limit fails before the deepest max
        # jitter.
        with self.assertRaises(TypeError):
            self.extended(
                regression_wave_regression_wave_regression_wave_regression_wave_recurrence_limit="x",
                regression_wave_regression_wave_regression_wave_max_regression_wave_jitter="x",
            )
        with self.assertRaises(ValueError):
            self.extended(
                regression_wave_regression_wave_regression_wave_regression_wave_recurrence_limit=0,
                regression_wave_regression_wave_regression_wave_max_regression_wave_jitter=-1,
            )
        # The deepest max jitter fails before the deepest periodicity
        # limit.
        with self.assertRaises(TypeError):
            self.extended(
                regression_wave_regression_wave_regression_wave_max_regression_wave_jitter="x",
                regression_wave_regression_wave_regression_wave_regression_wave_periodicity_limit="x",
            )
        with self.assertRaises(ValueError):
            self.extended(
                regression_wave_regression_wave_regression_wave_max_regression_wave_jitter=-1,
                regression_wave_regression_wave_regression_wave_regression_wave_periodicity_limit=0,
            )
        # All pass before the split membership check fires.
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
                regression_wave_regression_wave_regression_wave_max_regression_wave_jitter=0,
                regression_wave_regression_wave_regression_wave_regression_wave_periodicity_limit=1,
            )

    def test_all_parameter_checks_precede_state_lookups(self):
        with self.assertRaises(TypeError):
            self.call(
                reference="ghost",
                regression_wave_regression_wave_regression_wave_max_regression_wave_jitter="x",
            )
        with self.assertRaises(ValueError):
            self.call(
                reference="ghost",
                regression_wave_regression_wave_regression_wave_max_regression_wave_jitter=-1,
            )
        with self.assertRaises(TypeError):
            self.call(
                reference="ghost",
                regression_wave_regression_wave_regression_wave_regression_wave_periodicity_limit="x",
            )
        with self.assertRaises(ValueError):
            self.call(
                reference="ghost",
                regression_wave_regression_wave_regression_wave_regression_wave_periodicity_limit=0,
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


class LifetimeChurnForecastDriftWavePredictionScorecardRegressionWaveForecastScorecardRegressionWaveForecastBacktestsScorecardsRegressionWaveForecastBacktestsScorecardsRegressionWavePeriodicitiesEmptyRunTests(
    unittest.TestCase
):
    def setUp(self):
        self.store, self.args = diamond_store()

    def zero_totals(self):
        return {
            "periodicities": 0,
            "waves": 0,
            "points": 0,
            "first_wave": 0,
            "last_wave": 0,
            "mean_period": 0,
        }

    def test_empty_deep_cutoffs_returns_empty_periodicities(self):
        result = periodicities_call(
            self.store,
            self.args,
            **call_args(
                regression_wave_regression_wave_regression_wave_cutoffs=()
            ),
        )
        self.assertEqual(result["periodicities"], ())
        self.assertEqual(result["totals"], self.zero_totals())

    def test_no_qualifying_identity_returns_empty_periodicities(self):
        # The diamond store yields no deepest forecast regression waves
        # at cutoff 3, so no identity reaches three appearances.
        result = periodicities_call(
            self.store,
            self.args,
            **call_args(
                regression_wave_regression_wave_regression_wave_cutoffs=(3,)
            ),
        )
        self.assertEqual(result["periodicities"], ())
        self.assertEqual(result["totals"], self.zero_totals())


class LifetimeChurnForecastDriftWavePredictionScorecardRegressionWaveForecastScorecardRegressionWaveForecastBacktestsScorecardsRegressionWaveForecastBacktestsScorecardsRegressionWavePeriodicitiesIsolationTests(
    unittest.TestCase
):
    def test_success_and_failure_change_nothing(self):
        store, args = diamond_store()
        heads = dict(store._heads)
        appends = copy.deepcopy(store._appends)
        merges = copy.deepcopy(store._merges)
        audit = store.audit_log()

        periodicities_call(store, args, **deep_call_args())
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
            dict(
                regression_wave_regression_wave_regression_wave_max_regression_wave_jitter="x"
            ),
            dict(
                regression_wave_regression_wave_regression_wave_max_regression_wave_jitter=False
            ),
            dict(
                regression_wave_regression_wave_regression_wave_max_regression_wave_jitter=-1
            ),
            dict(
                regression_wave_regression_wave_regression_wave_regression_wave_periodicity_limit="x"
            ),
            dict(
                regression_wave_regression_wave_regression_wave_regression_wave_periodicity_limit=False
            ),
            dict(
                regression_wave_regression_wave_regression_wave_regression_wave_periodicity_limit=0
            ),
            dict(drift_limit=2),
            dict(scan_limit=1),
            dict(reference="ghost"),
        ]
        for overrides in failures:
            with self.subTest(overrides=overrides):
                merged = deep_call_args()
                merged.update(overrides)
                with self.assertRaises(Exception):
                    periodicities_call(store, args, **merged)

        self.assertEqual(dict(store._heads), heads)
        self.assertEqual(copy.deepcopy(store._appends), appends)
        self.assertEqual(copy.deepcopy(store._merges), merges)
        self.assertEqual(store.audit_log(), audit)


class LifetimeChurnForecastDriftWavePredictionScorecardRegressionWaveForecastScorecardRegressionWaveForecastBacktestsScorecardsRegressionWaveForecastBacktestsScorecardsRegressionWavePeriodicitiesTokenTests(
    unittest.TestCase
):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, token=None, **overrides):
        return periodicities_call(
            self.store, self.args, token=token, **overrides
        )

    def test_success_consumes_one_read_and_failure_refunds(self):
        token = self.store.create_snapshot(2)
        self.call(token=token, **deep_call_args())
        # Stage failures still refund.
        with self.assertRaises(ValueError):
            self.call(token=token, **deep_call_args(drift_limit=2))
        with self.assertRaises(ValueError):
            self.call(token=token, **deep_call_args(scan_limit=1))
        with self.assertRaises(ValueError):
            self.call(token=token, **deep_call_args(wave_cutoffs=(9,)))
        # Parameter validation runs before the token is touched.
        with self.assertRaises(TypeError):
            self.call(
                token=token,
                **deep_call_args(
                    regression_wave_regression_wave_regression_wave_max_regression_wave_jitter="x"
                ),
            )
        with self.assertRaises(ValueError):
            self.call(
                token=token,
                **deep_call_args(
                    regression_wave_regression_wave_regression_wave_regression_wave_periodicity_limit=0
                ),
            )
        # The second allowed read succeeds; the token then expires.
        self.call(token=token, **deep_call_args())
        with self.assertRaises(RuntimeError):
            self.call(token=token, **deep_call_args())

    def test_tokenless_query_leaves_snapshot_quotas_intact(self):
        token = self.store.create_snapshot(1)
        self.call(**deep_call_args())
        self.call(**deep_call_args())
        # The single read quota is still available after tokenless calls.
        self.call(token=token, **deep_call_args())
        with self.assertRaises(RuntimeError):
            self.call(token=token, **deep_call_args())

    def test_results_come_from_the_frozen_view(self):
        before = self.call(**deep_call_args())
        token = self.store.create_snapshot(4)
        self.store.append("a", "a2", 7, {"x": -1})
        self.store.create("d", "root")
        self.store.append("d", "d0", 8, {"z": 3})
        self.store.merge("a", "d", "M", 9, {"z": 1})
        self.store.append("main", "r3", 10, {"x": 1})
        self.assertEqual(self.call(token=token, **deep_call_args()), before)


if __name__ == "__main__":
    unittest.main()
