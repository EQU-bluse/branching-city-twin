import copy
import inspect
import unittest

from city_twin.branches import BranchStore

from tests.test_lifetime_churn import diamond_store
from tests.test_lifetime_churn_forecast_drift_scan import LONG_ARGS
from tests.test_lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_waves import (
    build_waves,
    make_streaks,
    waves_call,
)
from tests.test_lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_wave_recurrences import (
    make_wave,
    recurrences_kwargs,
)


def periodicities_kwargs(store_args, **overrides) -> dict:
    max_regression_wave_jitter = overrides.pop(
        "max_regression_wave_jitter", 10
    )
    regression_wave_periodicity_limit = overrides.pop(
        "regression_wave_periodicity_limit", 50
    )
    kwargs = recurrences_kwargs(store_args, **overrides)
    kwargs["max_regression_wave_jitter"] = max_regression_wave_jitter
    kwargs["regression_wave_periodicity_limit"] = (
        regression_wave_periodicity_limit
    )
    return kwargs


def periodicities_call(store, store_args, **overrides):
    return store.lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_wave_periodicities(
        **periodicities_kwargs(store_args, **overrides)
    )


def build_periodicities(
    waves,
    max_regression_wave_jitter=10,
    regression_wave_periodicity_limit=50,
):
    return BranchStore._build_lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_wave_periodicities_result(
        waves,
        max_regression_wave_jitter,
        regression_wave_periodicity_limit,
    )


class LifetimeChurnForecastDriftWavePredictionScorecardRegressionWavePeriodicitiesSignatureTests(
    unittest.TestCase
):
    def test_public_signature_appends_two_parameters(self):
        parameters = list(
            inspect.signature(
                BranchStore.lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_wave_periodicities
            ).parameters.values()
        )
        names = [p.name for p in parameters]
        recurrence_names = list(
            inspect.signature(
                BranchStore.lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_wave_recurrences
            ).parameters
        )
        self.assertEqual(
            names,
            recurrence_names[:-1]
            + [
                "max_regression_wave_jitter",
                "regression_wave_periodicity_limit",
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
            make_wave(5, [(5, 1, ("A",))]),
        )
        result = build_periodicities(waves)
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


class LifetimeChurnForecastDriftWavePredictionScorecardRegressionWavePeriodicitiesResultTests(
    unittest.TestCase
):
    def test_identity_recurring_at_a_steady_cadence(self):
        waves = (
            make_wave(1, [(1, 2, ("A", "B")), (2, 1, ("A",))]),
            make_wave(4, [(4, 1, ("A",)), (5, 3, ("A", "B", "C"))]),
            make_wave(7, [(7, 1, ("A",))]),
        )
        result = build_periodicities(waves)
        self.assertEqual(len(result["periodicities"]), 1)
        record = result["periodicities"][0]
        self.assertEqual(
            record,
            {
                "identity": "A",
                "waves": 3,
                "first_wave": 0,
                "last_wave": 2,
                "span": 3,
                "points": 5,
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
                    {
                        "wave": 2,
                        "start_baseline_cutoffs": 7,
                        "end_baseline_cutoffs": 7,
                        "points": 1,
                        "first_baseline_cutoffs": 7,
                        "last_baseline_cutoffs": 7,
                    },
                ),
                "intervals": (1, 1),
                "period": 1,
                "jitter": 0,
            },
        )
        self.assertEqual(
            result["totals"],
            {
                "periodicities": 1,
                "waves": 3,
                "points": 5,
                "first_wave": 0,
                "last_wave": 2,
                "mean_period": 1,
            },
        )

    def test_identities_in_fewer_than_three_waves_are_dropped(self):
        waves = (
            make_wave(1, [(1, 1, ("A", "B"))]),
            make_wave(3, [(3, 1, ("A", "B"))]),
            make_wave(5, [(5, 1, ("A",))]),
        )
        result = build_periodicities(waves)
        self.assertEqual(
            [record["identity"] for record in result["periodicities"]],
            ["A"],
        )

    def test_jitter_must_not_exceed_max_regression_wave_jitter(self):
        # "A" takes part in waves 0, 1 and 3, so the intervals are
        # (1, 2) and the jitter is one.
        waves = (
            make_wave(1, [(1, 1, ("A",))]),
            make_wave(2, [(2, 1, ("A",))]),
            make_wave(3, [(3, 1, ("B",))]),
            make_wave(5, [(5, 1, ("A",))]),
        )
        result = build_periodicities(waves, 1)
        self.assertEqual(len(result["periodicities"]), 1)
        record = result["periodicities"][0]
        self.assertEqual(record["intervals"], (1, 2))
        self.assertEqual(record["jitter"], 1)
        # Waves 0, 1 and 4 give intervals (1, 3) and a jitter of two.
        waves = (
            make_wave(1, [(1, 1, ("A",))]),
            make_wave(2, [(2, 1, ("A",))]),
            make_wave(3, [(3, 1, ("B",))]),
            make_wave(4, [(4, 1, ("B",))]),
            make_wave(5, [(5, 1, ("A",))]),
        )
        result = build_periodicities(waves, 2)
        self.assertEqual(result["periodicities"][0]["jitter"], 2)
        result = build_periodicities(waves, 1)
        self.assertEqual(result["periodicities"], ())

    def test_period_is_the_integer_median_of_the_sorted_intervals(self):
        # "A" takes part in waves 0, 2, 3 and 6, so the intervals are
        # (2, 1, 3) and the median is two.
        waves = (
            make_wave(1, [(1, 1, ("A",))]),
            make_wave(2, [(2, 1, ("B",))]),
            make_wave(3, [(3, 1, ("A",))]),
            make_wave(4, [(4, 1, ("A",))]),
            make_wave(5, [(5, 1, ("B",))]),
            make_wave(6, [(6, 1, ("C",))]),
            make_wave(7, [(7, 1, ("A",))]),
        )
        result = build_periodicities(waves)
        self.assertEqual(len(result["periodicities"]), 1)
        record = result["periodicities"][0]
        self.assertEqual(record["identity"], "A")
        self.assertEqual(record["intervals"], (2, 1, 3))
        self.assertEqual(record["period"], 2)
        self.assertEqual(record["jitter"], 2)

    def test_even_interval_count_takes_the_smaller_middle_value(self):
        # "A" takes part in waves 0, 1, 4, 6 and 10, so the intervals
        # are (1, 3, 2, 4), sorted (1, 2, 3, 4): the smaller middle
        # value is two. Every filler identity appears only once.
        fillers = {2: "B", 3: "C", 5: "D", 7: "E", 8: "F", 9: "G"}
        waves = tuple(
            make_wave(
                index + 1,
                [(index + 1, 1, (fillers.get(index, "A"),))],
            )
            for index in range(11)
        )
        result = build_periodicities(waves, 3)
        self.assertEqual(len(result["periodicities"]), 1)
        record = result["periodicities"][0]
        self.assertEqual(record["intervals"], (1, 3, 2, 4))
        self.assertEqual(record["period"], 2)
        self.assertEqual(record["jitter"], 3)

    def test_records_sort_by_jitter_waves_points_first_wave_then_encounter(
        self,
    ):
        # "a" and "b" both have jitter zero; "a" has more waves. "c"
        # has jitter one and sorts last however many points it has.
        waves = (
            make_wave(1, [(1, 1, ("c", "a", "b"))]),
            make_wave(2, [(2, 1, ("a", "b"))]),
            make_wave(3, [(3, 1, ("a", "b", "c"))]),
            make_wave(5, [(5, 1, ("a", "c"))]),
        )
        result = build_periodicities(waves)
        self.assertEqual(
            [record["identity"] for record in result["periodicities"]],
            ["a", "b", "c"],
        )
        # A full tie falls back to first-encounter order.
        waves = (
            make_wave(1, [(1, 1, ("c", "a"))]),
            make_wave(2, [(2, 1, ("c", "a"))]),
            make_wave(3, [(3, 1, ("c", "a"))]),
        )
        result = build_periodicities(waves)
        self.assertEqual(
            [record["identity"] for record in result["periodicities"]],
            ["c", "a"],
        )
        # More points break a wave-count tie before first_wave.
        waves = (
            make_wave(1, [(1, 1, ("b",)), (2, 1, ("b", "a"))]),
            make_wave(2, [(2, 1, ("a", "b"))]),
            make_wave(3, [(3, 1, ("a", "b"))]),
        )
        result = build_periodicities(waves)
        self.assertEqual(
            [record["identity"] for record in result["periodicities"]],
            ["b", "a"],
        )

    def test_regression_wave_periodicity_limit_bounds_the_complete_set(
        self,
    ):
        waves = (
            make_wave(1, [(1, 1, ("A", "B"))]),
            make_wave(2, [(2, 1, ("A", "B"))]),
            make_wave(3, [(3, 1, ("A", "B"))]),
        )
        with self.assertRaises(ValueError) as caught:
            build_periodicities(waves, 10, 1)
        self.assertIn(
            "regression wave periodicity limit", str(caught.exception)
        )
        result = build_periodicities(waves, 10, 2)
        self.assertEqual(len(result["periodicities"]), 2)

    def test_empty_waves_return_empty_tuple_and_zero_totals(self):
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

    def test_mean_period_is_the_floored_average(self):
        # "A" has period one (waves 0, 1, 2) and "B" period two
        # (waves 0, 2, 4), so the mean period floors 1.5 to one.
        waves = (
            make_wave(1, [(1, 1, ("A", "B"))]),
            make_wave(2, [(2, 1, ("A",))]),
            make_wave(3, [(3, 1, ("A", "B"))]),
            make_wave(4, [(4, 1, ("C",))]),
            make_wave(5, [(5, 1, ("B",))]),
        )
        result = build_periodicities(waves)
        self.assertEqual(len(result["periodicities"]), 2)
        self.assertEqual(result["totals"]["mean_period"], 1)
        self.assertEqual(result["totals"]["waves"], 6)
        self.assertEqual(result["totals"]["first_wave"], 0)
        self.assertEqual(result["totals"]["last_wave"], 4)

    def test_every_object_is_fresh(self):
        waves = (
            make_wave(1, [(1, 1, (("A", "x"),))]),
            make_wave(2, [(2, 1, (("A", "x"),))]),
            make_wave(3, [(3, 1, (("A", "x"),))]),
        )
        result = build_periodicities(waves)
        result["periodicities"][0]["waves"] = 999
        result["periodicities"][0]["appearances"][0]["points"] = 999
        result["totals"]["periodicities"] = 999
        again = build_periodicities(waves)
        self.assertEqual(again["periodicities"][0]["waves"], 3)
        self.assertEqual(
            again["periodicities"][0]["appearances"][0]["points"], 1
        )
        self.assertEqual(again["totals"]["periodicities"], 1)
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


class LifetimeChurnForecastDriftWavePredictionScorecardRegressionWavePeriodicitiesChainTests(
    unittest.TestCase
):
    """The periodicity builder consumes exactly what the wave stage keeps."""

    def test_periodicities_match_the_waves_result(self):
        # "B" stays active at every requested split while "A" only
        # covers 1, 3 and 5, so with two active regressions required
        # the run breaks into three single-point waves.
        streaks = make_streaks(
            ("A", 1, 1), ("A", 3, 3), ("A", 5, 5), ("B", 1, 5)
        )
        waves = build_waves(streaks, (1, 2, 3, 4, 5), 2)
        self.assertEqual(len(waves["waves"]), 3)
        result = build_periodicities(waves["waves"])
        self.assertEqual(
            [record["identity"] for record in result["periodicities"]],
            ["A", "B"],
        )
        for record in result["periodicities"]:
            self.assertEqual(record["waves"], 3)
            self.assertEqual(record["points"], 3)
            self.assertEqual(record["peak_active"], 2)
            self.assertEqual(record["intervals"], (1, 1))
            self.assertEqual(record["period"], 1)
            self.assertEqual(record["jitter"], 0)
        self.assertEqual(
            result["totals"],
            {
                "periodicities": 2,
                "waves": 6,
                "points": 6,
                "first_wave": 0,
                "last_wave": 2,
                "mean_period": 1,
            },
        )

    def test_public_query_matches_the_waves_query(self):
        store, args = diamond_store()
        call_args = dict(
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
        )
        waves = waves_call(store, args, **call_args)
        result = periodicities_call(store, args, **call_args)
        self.assertEqual(result, build_periodicities(waves["waves"]))


class LifetimeChurnForecastDriftWavePredictionScorecardRegressionWavePeriodicitiesValidationTests(
    unittest.TestCase
):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        return periodicities_call(self.store, self.args, **overrides)

    def extended(self, **overrides):
        overrides.setdefault("split_cutoffs", (4,))
        merged = dict(LONG_ARGS)
        merged.update(overrides)
        return self.call(**merged)

    def test_max_regression_wave_jitter_must_be_a_non_bool_non_negative_int(
        self,
    ):
        for bad in (True, False, 1.0, "1", None, (1,)):
            with self.subTest(value=bad):
                with self.assertRaises(TypeError):
                    self.extended(max_regression_wave_jitter=bad)
        with self.assertRaises(ValueError):
            self.extended(max_regression_wave_jitter=-1)
        # Zero is a legal jitter bound.
        self.extended(max_regression_wave_jitter=0)

    def test_regression_wave_periodicity_limit_must_be_a_non_bool_positive_int(
        self,
    ):
        for bad in (True, False, 1.0, "1", None, (1,)):
            with self.subTest(value=bad):
                with self.assertRaises(TypeError):
                    self.extended(regression_wave_periodicity_limit=bad)
        with self.assertRaises(ValueError):
            self.extended(regression_wave_periodicity_limit=0)
        with self.assertRaises(ValueError):
            self.extended(regression_wave_periodicity_limit=-3)

    def test_new_parameters_validated_in_signature_order(self):
        # regression_wave_recurrence_limit fails before
        # max_regression_wave_jitter.
        with self.assertRaises(TypeError):
            self.extended(
                regression_wave_recurrence_limit="x",
                max_regression_wave_jitter="x",
            )
        with self.assertRaises(ValueError):
            self.extended(
                regression_wave_recurrence_limit=0,
                max_regression_wave_jitter=-1,
            )
        # max_regression_wave_jitter fails before
        # regression_wave_periodicity_limit.
        with self.assertRaises(TypeError):
            self.extended(
                max_regression_wave_jitter="x",
                regression_wave_periodicity_limit="x",
            )
        with self.assertRaises(ValueError):
            self.extended(
                max_regression_wave_jitter=-1,
                regression_wave_periodicity_limit=0,
            )
        # Both pass before the split membership check fires.
        with self.assertRaises(ValueError):
            self.extended(
                split_cutoffs=(99,),
                baseline_splits=(1,),
                min_evaluated_delta=0,
                min_hit_rate_drop=(0, 1),
                regression_limit=1,
                min_consecutive_splits=1,
                regression_streak_limit=1,
                min_active_regressions=1,
                regression_wave_limit=1,
                min_regression_wave_occurrences=1,
                regression_wave_recurrence_limit=1,
                max_regression_wave_jitter=0,
                regression_wave_periodicity_limit=1,
            )

    def test_all_parameter_checks_precede_state_lookups(self):
        with self.assertRaises(TypeError):
            self.call(
                reference="ghost",
                cutoffs=(0, 1),
                max_regression_wave_jitter="x",
            )
        with self.assertRaises(ValueError):
            self.call(
                reference="ghost",
                cutoffs=(0, 1),
                max_regression_wave_jitter=-1,
            )
        with self.assertRaises(TypeError):
            self.call(
                reference="ghost",
                cutoffs=(0, 1),
                regression_wave_periodicity_limit="x",
            )
        with self.assertRaises(ValueError):
            self.call(
                reference="ghost",
                cutoffs=(0, 1),
                regression_wave_periodicity_limit=0,
            )

    def test_underlying_stage_limits_and_errors_are_unchanged(self):
        with self.assertRaises(ValueError) as caught:
            self.extended(
                wave_cutoffs=(9,),
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
            )
        self.assertIn("out of range", str(caught.exception))


class LifetimeChurnForecastDriftWavePredictionScorecardRegressionWavePeriodicitiesEmptyRunTests(
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

    def test_empty_wave_cutoffs_returns_empty_periodicities_and_zero_totals(
        self,
    ):
        result = periodicities_call(
            self.store,
            self.args,
            **dict(LONG_ARGS, split_cutoffs=(3, 4, 5), wave_cutoffs=()),
        )
        self.assertEqual(result["periodicities"], ())
        self.assertEqual(result["totals"], self.zero_totals())

    def test_no_qualifying_identity_returns_empty_periodicities(self):
        # The diamond store yields a single drift wave, so no identity
        # reaches three regression waves at cutoff 0.
        result = periodicities_call(
            self.store,
            self.args,
            **dict(LONG_ARGS, split_cutoffs=(3, 4, 5), wave_cutoffs=(0,)),
        )
        self.assertEqual(result["periodicities"], ())
        self.assertEqual(result["totals"], self.zero_totals())


class LifetimeChurnForecastDriftWavePredictionScorecardRegressionWavePeriodicitiesIsolationTests(
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
            regression_limit=50,
            min_consecutive_splits=1,
            regression_streak_limit=50,
            min_active_regressions=1,
            regression_wave_limit=50,
            min_regression_wave_occurrences=1,
            regression_wave_recurrence_limit=50,
            max_regression_wave_jitter=10,
            regression_wave_periodicity_limit=50,
        )

    def test_success_and_failure_change_nothing(self):
        store, args = diamond_store()
        heads = dict(store._heads)
        appends = copy.deepcopy(store._appends)
        merges = copy.deepcopy(store._merges)
        audit = store.audit_log()

        periodicities_call(store, args, **self.call_args())
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
            dict(baseline_splits=(False,)),
            dict(baseline_splits=()),
            dict(baseline_splits=(0,)),
            dict(baseline_splits=(2, 2)),
            dict(baseline_splits=(3, 1)),
            dict(min_evaluated_delta="x"),
            dict(min_evaluated_delta=True),
            dict(min_evaluated_delta=-1),
            dict(min_hit_rate_drop="x"),
            dict(min_hit_rate_drop=(1, 0)),
            dict(min_hit_rate_drop=(2, 1)),
            dict(regression_limit="x"),
            dict(regression_limit=False),
            dict(regression_limit=0),
            dict(min_consecutive_splits="x"),
            dict(min_consecutive_splits=False),
            dict(min_consecutive_splits=0),
            dict(regression_streak_limit="x"),
            dict(regression_streak_limit=False),
            dict(regression_streak_limit=0),
            dict(min_active_regressions="x"),
            dict(min_active_regressions=False),
            dict(min_active_regressions=0),
            dict(regression_wave_limit="x"),
            dict(regression_wave_limit=False),
            dict(regression_wave_limit=0),
            dict(min_regression_wave_occurrences="x"),
            dict(min_regression_wave_occurrences=False),
            dict(min_regression_wave_occurrences=0),
            dict(regression_wave_recurrence_limit="x"),
            dict(regression_wave_recurrence_limit=False),
            dict(regression_wave_recurrence_limit=0),
            dict(max_regression_wave_jitter="x"),
            dict(max_regression_wave_jitter=False),
            dict(max_regression_wave_jitter=-1),
            dict(regression_wave_periodicity_limit="x"),
            dict(regression_wave_periodicity_limit=False),
            dict(regression_wave_periodicity_limit=0),
            dict(reference="ghost"),
        ]
        for overrides in failures:
            with self.subTest(overrides=overrides):
                merged = self.call_args()
                merged.update(overrides)
                with self.assertRaises(Exception):
                    periodicities_call(store, args, **merged)

        self.assertEqual(dict(store._heads), heads)
        self.assertEqual(copy.deepcopy(store._appends), appends)
        self.assertEqual(copy.deepcopy(store._merges), merges)
        self.assertEqual(store.audit_log(), audit)


class LifetimeChurnForecastDriftWavePredictionScorecardRegressionWavePeriodicitiesTokenTests(
    unittest.TestCase
):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, token=None, **overrides):
        return periodicities_call(
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
            regression_limit=50,
            min_consecutive_splits=1,
            regression_streak_limit=50,
            min_active_regressions=1,
            regression_wave_limit=50,
            min_regression_wave_occurrences=1,
            regression_wave_recurrence_limit=50,
            max_regression_wave_jitter=10,
            regression_wave_periodicity_limit=50,
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
                token=token,
                **dict(self.call_args(), max_regression_wave_jitter="x"),
            )
        with self.assertRaises(ValueError):
            self.call(
                token=token,
                **dict(self.call_args(), regression_wave_periodicity_limit=0),
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
