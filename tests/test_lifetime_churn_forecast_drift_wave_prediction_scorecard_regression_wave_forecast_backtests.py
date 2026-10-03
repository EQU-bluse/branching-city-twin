import copy
import inspect
import unittest

from city_twin.branches import BranchStore

from tests.test_lifetime_churn import diamond_store
from tests.test_lifetime_churn_forecast_drift_scan import LONG_ARGS
from tests.test_lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_waves import (
    waves_call,
)
from tests.test_lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_wave_periodicities import (
    periodicities_kwargs,
)


def forecast_backtests_kwargs(store_args, **overrides) -> dict:
    regression_wave_horizon = overrides.pop("regression_wave_horizon", 3)
    regression_wave_forecast_limit = overrides.pop(
        "regression_wave_forecast_limit", 50
    )
    regression_wave_cutoffs = overrides.pop("regression_wave_cutoffs", (0,))
    regression_wave_match_tolerance = overrides.pop(
        "regression_wave_match_tolerance", 0
    )
    regression_wave_forecast_backtest_limit = overrides.pop(
        "regression_wave_forecast_backtest_limit", 50
    )
    kwargs = periodicities_kwargs(store_args, **overrides)
    kwargs["regression_wave_horizon"] = regression_wave_horizon
    kwargs["regression_wave_forecast_limit"] = regression_wave_forecast_limit
    kwargs["regression_wave_cutoffs"] = regression_wave_cutoffs
    kwargs["regression_wave_match_tolerance"] = (
        regression_wave_match_tolerance
    )
    kwargs["regression_wave_forecast_backtest_limit"] = (
        regression_wave_forecast_backtest_limit
    )
    return kwargs


def forecast_backtests_call(store, store_args, **overrides):
    return store.lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_wave_forecast_backtests(
        **forecast_backtests_kwargs(store_args, **overrides)
    )


def build_forecast_backtests(waves, *args):
    return BranchStore._build_lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_wave_forecast_backtests_result(
        waves, *args
    )


def regression_wave(index, identities):
    return {
        "start_baseline_cutoffs": index,
        "end_baseline_cutoffs": index,
        "split_count": 1,
        "peak_regressions": len(identities),
        "points": (
            {
                "baseline_cutoffs": index,
                "active_count": len(identities),
                "identities": tuple(identities),
            },
        ),
    }


def synthetic_waves():
    """Eight one-point waves pinning training, matching and windows.

    A and C ride every training wave 0-4 (period one); B rides waves
    0, 2 and 4 (period two). After a cutoff at 4, A reappears at 5 and
    7, B at 5 and 6, and C at 5 only, so a horizon of two sees A and C
    hit wave 5 and miss wave 6 while B hits wave 6 and leaves wave 5
    unexpected.
    """
    identities_per_wave = (
        ("A", "B", "C"),
        ("A", "C"),
        ("A", "B", "C"),
        ("A", "C"),
        ("A", "B", "C"),
        ("A", "B", "C"),
        ("B",),
        ("A",),
    )
    return tuple(
        regression_wave(index, identities)
        for index, identities in enumerate(identities_per_wave)
    )


def tie_waves():
    """E rides waves 0, 2 and 4 (period two), then reappears at 5 and 7."""
    identities_per_wave = (
        ("E",),
        (),
        ("E",),
        (),
        ("E",),
        ("E",),
        (),
        ("E",),
    )
    return tuple(
        regression_wave(index, identities)
        for index, identities in enumerate(identities_per_wave)
    )


def no_reuse_waves():
    """F rides waves 0-2 (period one) and reappears only at wave 3."""
    identities_per_wave = (("F",), ("F",), ("F",), ("F",), ())
    return tuple(
        regression_wave(index, identities)
        for index, identities in enumerate(identities_per_wave)
    )


class LifetimeChurnForecastDriftWavePredictionScorecardRegressionWaveForecastBacktestsSignatureTests(
    unittest.TestCase
):
    def test_public_signature_appends_three_parameters(self):
        parameters = list(
            inspect.signature(
                BranchStore.lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_wave_forecast_backtests
            ).parameters.values()
        )
        names = [p.name for p in parameters]
        forecast_names = list(
            inspect.signature(
                BranchStore.lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_wave_forecasts
            ).parameters
        )
        self.assertEqual(
            names,
            forecast_names[:-1]
            + [
                "regression_wave_cutoffs",
                "regression_wave_match_tolerance",
                "regression_wave_forecast_backtest_limit",
                "token",
            ],
        )
        self.assertEqual(
            [p.default for p in parameters[:-1]],
            [inspect.Parameter.empty] * (len(parameters) - 1),
        )
        self.assertIsNone(parameters[-1].default)

    def test_result_shape_and_key_order(self):
        result = build_forecast_backtests(
            synthetic_waves(), 1, 50, 0, 50, 2, 50, (4,), 0, 50
        )
        self.assertEqual(list(result), ["backtests", "totals"])
        self.assertIsInstance(result["backtests"], tuple)
        for record in result["backtests"]:
            self.assertEqual(
                list(record),
                [
                    "cutoff",
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
                    "targets",
                    "hit",
                    "missed",
                    "unresolved",
                    "unexpected",
                ],
            )
            self.assertIsInstance(record["targets"], tuple)
            for target in record["targets"]:
                self.assertEqual(
                    list(target),
                    ["ordinal", "wave", "earliest", "latest", "actual", "status"],
                )
        self.assertEqual(
            list(result["totals"]),
            ["cutoffs", "records", "hit", "missed", "unresolved", "unexpected"],
        )


class LifetimeChurnForecastDriftWavePredictionScorecardRegressionWaveForecastBacktestsResultTests(
    unittest.TestCase
):
    def test_training_matching_and_totals(self):
        result = build_forecast_backtests(
            synthetic_waves(), 1, 50, 0, 50, 2, 50, (4,), 0, 50
        )
        records = result["backtests"]
        # Periodicity order is A, C, B: equal jitter, then wave and
        # point counts descending, then first-encounter order.
        self.assertEqual([r["identity"] for r in records], ["A", "C", "B"])
        for record in records:
            self.assertEqual(record["cutoff"], 4)
            self.assertEqual(record["jitter"], 0)
            self.assertEqual(record["last_wave"], 4)
        self.assertEqual(records[0]["waves"], 5)
        self.assertEqual(records[0]["first_wave"], 0)
        self.assertEqual(records[0]["span"], 5)
        self.assertEqual(records[0]["points"], 5)
        self.assertEqual(records[0]["peak_active"], 3)
        self.assertEqual(records[0]["intervals"], (1, 1, 1, 1))
        self.assertEqual(records[0]["period"], 1)
        self.assertEqual(
            records[0]["targets"],
            (
                {
                    "ordinal": 1,
                    "wave": 5,
                    "earliest": 5,
                    "latest": 5,
                    "actual": 5,
                    "status": "hit",
                },
                {
                    "ordinal": 2,
                    "wave": 6,
                    "earliest": 6,
                    "latest": 6,
                    "actual": None,
                    "status": "missed",
                },
            ),
        )
        self.assertEqual(
            (records[0]["hit"], records[0]["missed"],
             records[0]["unresolved"], records[0]["unexpected"]),
            (1, 1, 0, 0),
        )
        self.assertEqual(
            [(t["wave"], t["actual"], t["status"]) for t in records[1]["targets"]],
            [(5, 5, "hit"), (6, None, "missed")],
        )
        # B's period is two, so its only target is wave 6; its wave-5
        # appearance inside the window stays unclaimed.
        self.assertEqual(records[2]["period"], 2)
        self.assertEqual(records[2]["intervals"], (2, 2))
        self.assertEqual(
            records[2]["targets"],
            (
                {
                    "ordinal": 1,
                    "wave": 6,
                    "earliest": 6,
                    "latest": 6,
                    "actual": 6,
                    "status": "hit",
                },
            ),
        )
        self.assertEqual(
            (records[2]["hit"], records[2]["missed"],
             records[2]["unresolved"], records[2]["unexpected"]),
            (1, 0, 0, 1),
        )
        self.assertEqual(
            result["totals"],
            {
                "cutoffs": 1,
                "records": 3,
                "hit": 3,
                "missed": 2,
                "unresolved": 0,
                "unexpected": 1,
            },
        )

    def test_missed_and_unresolved_split_at_the_observation_end(self):
        result = build_forecast_backtests(
            synthetic_waves(), 1, 50, 0, 50, 4, 50, (4,), 0, 50
        )
        records = result["backtests"]
        # Wave 7 is the last observed wave: targets at 8 are
        # unresolved, targets at most 7 without a match are missed.
        self.assertEqual(
            [(t["wave"], t["actual"], t["status"]) for t in records[0]["targets"]],
            [(5, 5, "hit"), (6, None, "missed"),
             (7, 7, "hit"), (8, None, "unresolved")],
        )
        self.assertEqual(
            [(t["wave"], t["actual"], t["status"]) for t in records[1]["targets"]],
            [(5, 5, "hit"), (6, None, "missed"),
             (7, None, "missed"), (8, None, "unresolved")],
        )
        self.assertEqual(
            [(t["wave"], t["actual"], t["status"]) for t in records[2]["targets"]],
            [(6, 6, "hit"), (8, None, "unresolved")],
        )
        self.assertEqual(
            result["totals"],
            {
                "cutoffs": 1,
                "records": 3,
                "hit": 4,
                "missed": 3,
                "unresolved": 3,
                "unexpected": 1,
            },
        )

    def test_window_excludes_appearances_beyond_the_horizon(self):
        # With a horizon of two the window is (4, 6], so A's wave-7
        # appearance is neither matched nor unexpected.
        result = build_forecast_backtests(
            synthetic_waves(), 1, 50, 0, 50, 2, 50, (4,), 0, 50
        )
        record_a = result["backtests"][0]
        self.assertEqual(record_a["unexpected"], 0)
        self.assertEqual(
            [t["wave"] for t in record_a["targets"]], [5, 6]
        )

    def test_equal_distances_take_the_earlier_wave(self):
        result = build_forecast_backtests(
            tie_waves(), 1, 50, 0, 50, 3, 50, (4,), 1, 50
        )
        record = result["backtests"][0]
        # Target 6 sits one wave from both 5 and 7; the earlier wave
        # wins and wave 7 is left unexpected.
        self.assertEqual(
            record["targets"],
            (
                {
                    "ordinal": 1,
                    "wave": 6,
                    "earliest": 6,
                    "latest": 6,
                    "actual": 5,
                    "status": "hit",
                },
            ),
        )
        self.assertEqual(record["unexpected"], 1)

    def test_tolerance_zero_requires_the_exact_wave(self):
        result = build_forecast_backtests(
            tie_waves(), 1, 50, 0, 50, 3, 50, (4,), 0, 50
        )
        record = result["backtests"][0]
        self.assertEqual(
            record["targets"],
            (
                {
                    "ordinal": 1,
                    "wave": 6,
                    "earliest": 6,
                    "latest": 6,
                    "actual": None,
                    "status": "missed",
                },
            ),
        )
        self.assertEqual(record["unexpected"], 2)

    def test_observed_waves_are_never_reused(self):
        result = build_forecast_backtests(
            no_reuse_waves(), 1, 50, 0, 50, 2, 50, (2,), 1, 50
        )
        record = result["backtests"][0]
        # Wave 3 claims target 3; target 4 is within tolerance of wave
        # 3 but cannot reuse it.
        self.assertEqual(
            [(t["wave"], t["actual"], t["status"]) for t in record["targets"]],
            [(3, 3, "hit"), (4, None, "missed")],
        )

    def test_records_sort_by_cutoff_then_periodicity_order(self):
        result = build_forecast_backtests(
            synthetic_waves(), 1, 50, 0, 50, 2, 50, (2, 4), 0, 50
        )
        self.assertEqual(
            [(r["cutoff"], r["identity"]) for r in result["backtests"]],
            [(2, "A"), (2, "C"), (4, "A"), (4, "C"), (4, "B")],
        )
        self.assertEqual(
            result["totals"],
            {
                "cutoffs": 2,
                "records": 5,
                "hit": 7,
                "missed": 2,
                "unresolved": 0,
                "unexpected": 1,
            },
        )

    def test_cutoff_prefix_changes_the_training_set(self):
        result = build_forecast_backtests(
            synthetic_waves(), 1, 50, 0, 50, 2, 50, (2,), 0, 50
        )
        # B rides only two of the waves 0-2, so it never reaches three
        # training appearances at cutoff 2.
        self.assertEqual(
            [r["identity"] for r in result["backtests"]], ["A", "C"]
        )
        for record in result["backtests"]:
            self.assertEqual(record["last_wave"], 2)
            self.assertEqual(
                [(t["wave"], t["actual"], t["status"]) for t in record["targets"]],
                [(3, 3, "hit"), (4, 4, "hit")],
            )

    def test_jitter_over_max_regression_wave_jitter_is_dropped(self):
        # G rides waves 0, 1 and 3: intervals (1, 2), so its jitter of
        # one only qualifies once max_regression_wave_jitter allows it.
        waves = tuple(
            regression_wave(index, identities)
            for index, identities in enumerate(
                (("G",), ("G",), (), ("G",), ("G",))
            )
        )
        result = build_forecast_backtests(
            waves, 1, 50, 0, 50, 2, 50, (3,), 0, 50
        )
        self.assertEqual(result["backtests"], ())
        result = build_forecast_backtests(
            waves, 1, 50, 1, 50, 2, 50, (3,), 0, 50
        )
        self.assertEqual(len(result["backtests"]), 1)
        record = result["backtests"][0]
        self.assertEqual(record["jitter"], 1)
        # The median period is the smaller middle interval, one.
        self.assertEqual(record["period"], 1)
        self.assertEqual(
            [(t["wave"], t["actual"], t["status"]) for t in record["targets"]],
            [(4, 4, "hit"), (5, None, "unresolved")],
        )

    def test_empty_inputs_return_empty_records_and_zero_totals(self):
        zero_totals = {
            "cutoffs": 0,
            "records": 0,
            "hit": 0,
            "missed": 0,
            "unresolved": 0,
            "unexpected": 0,
        }
        # Empty cutoff tuple.
        result = build_forecast_backtests(
            synthetic_waves(), 1, 50, 0, 50, 2, 50, (), 0, 50
        )
        self.assertEqual(result, {"backtests": (), "totals": zero_totals})
        # No waves at all: the range check is skipped and nothing
        # trains.
        result = build_forecast_backtests((), 1, 50, 0, 50, 2, 50, (3,), 0, 50)
        self.assertEqual(result, {"backtests": (), "totals": zero_totals})
        # No identity reaches three training appearances.
        result = build_forecast_backtests(
            synthetic_waves()[:1], 1, 50, 0, 50, 2, 50, (0,), 0, 50
        )
        self.assertEqual(result, {"backtests": (), "totals": zero_totals})
        # A horizon shorter than every period projects nothing.
        result = build_forecast_backtests(
            tie_waves(), 1, 50, 0, 50, 1, 50, (4,), 0, 50
        )
        self.assertEqual(result, {"backtests": (), "totals": zero_totals})
        # A horizon of one keeps only the period-one identities.
        result = build_forecast_backtests(
            synthetic_waves(), 1, 50, 0, 50, 1, 50, (4,), 0, 50
        )
        self.assertEqual(
            [r["identity"] for r in result["backtests"]], ["A", "C"]
        )

    def test_cutoff_beyond_the_last_wave_raises(self):
        with self.assertRaises(ValueError) as caught:
            build_forecast_backtests(
                synthetic_waves(), 1, 50, 0, 50, 2, 50, (8,), 0, 50
            )
        self.assertIn("out of range", str(caught.exception))
        # The check fires before any training, even for a later cutoff.
        with self.assertRaises(ValueError):
            build_forecast_backtests(
                synthetic_waves(), 1, 50, 0, 50, 2, 50, (4, 8), 0, 50
            )

    def test_stage_limits_bound_each_cutoff_without_truncating(self):
        with self.assertRaises(ValueError) as caught:
            build_forecast_backtests(
                synthetic_waves(), 1, 2, 0, 50, 2, 50, (4,), 0, 50
            )
        self.assertIn("regression wave recurrence limit", str(caught.exception))
        with self.assertRaises(ValueError) as caught:
            build_forecast_backtests(
                synthetic_waves(), 1, 50, 0, 2, 2, 50, (4,), 0, 50
            )
        self.assertIn(
            "regression wave periodicity limit", str(caught.exception)
        )

    def test_forecast_limit_bounds_each_cutoff_prediction_set(self):
        # Cutoff 4 with a horizon of two projects five targets across
        # the three identities.
        with self.assertRaises(ValueError) as caught:
            build_forecast_backtests(
                synthetic_waves(), 1, 50, 0, 50, 2, 4, (4,), 0, 50
            )
        self.assertIn("regression wave forecast limit", str(caught.exception))
        result = build_forecast_backtests(
            synthetic_waves(), 1, 50, 0, 50, 2, 5, (4,), 0, 50
        )
        self.assertEqual(len(result["backtests"]), 3)

    def test_backtest_limit_bounds_the_complete_record_set(self):
        with self.assertRaises(ValueError) as caught:
            build_forecast_backtests(
                synthetic_waves(), 1, 50, 0, 50, 2, 50, (2, 4), 0, 4
            )
        self.assertIn("forecast backtest limit", str(caught.exception))
        result = build_forecast_backtests(
            synthetic_waves(), 1, 50, 0, 50, 2, 50, (2, 4), 0, 5
        )
        self.assertEqual(result["totals"]["records"], 5)

    def test_every_object_is_fresh(self):
        waves = synthetic_waves()
        result = build_forecast_backtests(
            waves, 1, 50, 0, 50, 2, 50, (4,), 0, 50
        )
        result["backtests"][0]["targets"][0]["wave"] = 999
        result["backtests"][0]["hit"] = 999
        result["totals"]["hit"] = 999
        again = build_forecast_backtests(
            waves, 1, 50, 0, 50, 2, 50, (4,), 0, 50
        )
        self.assertEqual(again["backtests"][0]["targets"][0]["wave"], 5)
        self.assertEqual(again["backtests"][0]["hit"], 1)
        self.assertEqual(again["totals"]["hit"], 3)

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


class LifetimeChurnForecastDriftWavePredictionScorecardRegressionWaveForecastBacktestsChainTests(
    unittest.TestCase
):
    """The backtests builder consumes exactly what regression waves keep."""

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
            regression_wave_horizon=10,
            regression_wave_forecast_limit=50,
            regression_wave_cutoffs=(0,),
            regression_wave_match_tolerance=1,
            regression_wave_forecast_backtest_limit=50,
        )

    def test_public_query_matches_the_regression_waves_result(self):
        store, args = diamond_store()
        call_args = self.call_args()
        wave_args = {
            key: value
            for key, value in call_args.items()
            if key
            not in (
                "min_regression_wave_occurrences",
                "regression_wave_recurrence_limit",
                "max_regression_wave_jitter",
                "regression_wave_periodicity_limit",
                "regression_wave_horizon",
                "regression_wave_forecast_limit",
                "regression_wave_cutoffs",
                "regression_wave_match_tolerance",
                "regression_wave_forecast_backtest_limit",
            )
        }
        waves = waves_call(store, args, **wave_args)
        result = forecast_backtests_call(store, args, **call_args)
        self.assertEqual(
            result,
            build_forecast_backtests(
                waves["waves"],
                call_args["min_regression_wave_occurrences"],
                call_args["regression_wave_recurrence_limit"],
                call_args["max_regression_wave_jitter"],
                call_args["regression_wave_periodicity_limit"],
                call_args["regression_wave_horizon"],
                call_args["regression_wave_forecast_limit"],
                call_args["regression_wave_cutoffs"],
                call_args["regression_wave_match_tolerance"],
                call_args["regression_wave_forecast_backtest_limit"],
            ),
        )


class LifetimeChurnForecastDriftWavePredictionScorecardRegressionWaveForecastBacktestsValidationTests(
    unittest.TestCase
):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        return forecast_backtests_call(self.store, self.args, **overrides)

    def extended(self, **overrides):
        overrides.setdefault("split_cutoffs", (4,))
        merged = dict(LONG_ARGS)
        merged.update(overrides)
        return self.call(**merged)

    def test_regression_wave_cutoffs_must_be_a_strictly_increasing_int_tuple(
        self,
    ):
        for bad in ([0], "0", 0, 1.0, None, {0}):
            with self.subTest(value=bad):
                with self.assertRaises(TypeError):
                    self.extended(regression_wave_cutoffs=bad)
        for bad in (True, False, 1.0, "0", None):
            with self.subTest(value=bad):
                with self.assertRaises(TypeError):
                    self.extended(regression_wave_cutoffs=(bad,))
        with self.assertRaises(ValueError):
            self.extended(regression_wave_cutoffs=(-1,))
        with self.assertRaises(ValueError):
            self.extended(regression_wave_cutoffs=(0, 0))
        with self.assertRaises(ValueError):
            self.extended(regression_wave_cutoffs=(1, 0))

    def test_regression_wave_match_tolerance_must_be_a_non_bool_non_negative_int(
        self,
    ):
        for bad in (True, False, 1.0, "1", None, (1,)):
            with self.subTest(value=bad):
                with self.assertRaises(TypeError):
                    self.extended(regression_wave_match_tolerance=bad)
        with self.assertRaises(ValueError):
            self.extended(regression_wave_match_tolerance=-1)
        self.extended(regression_wave_match_tolerance=0)

    def test_regression_wave_forecast_backtest_limit_must_be_a_non_bool_positive_int(
        self,
    ):
        for bad in (True, False, 1.0, "1", None, (1,)):
            with self.subTest(value=bad):
                with self.assertRaises(TypeError):
                    self.extended(regression_wave_forecast_backtest_limit=bad)
        with self.assertRaises(ValueError):
            self.extended(regression_wave_forecast_backtest_limit=0)
        with self.assertRaises(ValueError):
            self.extended(regression_wave_forecast_backtest_limit=-3)

    def test_new_parameters_validated_in_signature_order(self):
        # regression_wave_forecast_limit fails before
        # regression_wave_cutoffs.
        with self.assertRaises(TypeError):
            self.extended(
                regression_wave_forecast_limit="x",
                regression_wave_cutoffs="x",
            )
        with self.assertRaises(ValueError):
            self.extended(
                regression_wave_forecast_limit=0,
                regression_wave_cutoffs=(-1,),
            )
        # regression_wave_cutoffs fails before
        # regression_wave_match_tolerance.
        with self.assertRaises(TypeError):
            self.extended(
                regression_wave_cutoffs="x",
                regression_wave_match_tolerance="x",
            )
        with self.assertRaises(ValueError):
            self.extended(
                regression_wave_cutoffs=(-1,),
                regression_wave_match_tolerance=-1,
            )
        # regression_wave_match_tolerance fails before
        # regression_wave_forecast_backtest_limit.
        with self.assertRaises(TypeError):
            self.extended(
                regression_wave_match_tolerance="x",
                regression_wave_forecast_backtest_limit="x",
            )
        with self.assertRaises(ValueError):
            self.extended(
                regression_wave_match_tolerance=-1,
                regression_wave_forecast_backtest_limit=0,
            )
        # All three pass before the split membership check fires.
        with self.assertRaises(ValueError):
            self.extended(
                split_cutoffs=(99,),
                regression_wave_cutoffs=(0,),
                regression_wave_match_tolerance=1,
                regression_wave_forecast_backtest_limit=1,
            )

    def test_all_parameter_checks_precede_state_lookups(self):
        with self.assertRaises(TypeError):
            self.call(
                reference="ghost",
                cutoffs=(0, 1),
                regression_wave_cutoffs="x",
            )
        with self.assertRaises(ValueError):
            self.call(
                reference="ghost",
                cutoffs=(0, 1),
                regression_wave_cutoffs=(-1,),
            )
        with self.assertRaises(TypeError):
            self.call(
                reference="ghost",
                cutoffs=(0, 1),
                regression_wave_match_tolerance="x",
            )
        with self.assertRaises(ValueError):
            self.call(
                reference="ghost",
                cutoffs=(0, 1),
                regression_wave_match_tolerance=-1,
            )
        with self.assertRaises(TypeError):
            self.call(
                reference="ghost",
                cutoffs=(0, 1),
                regression_wave_forecast_backtest_limit="x",
            )
        with self.assertRaises(ValueError):
            self.call(
                reference="ghost",
                cutoffs=(0, 1),
                regression_wave_forecast_backtest_limit=0,
            )


class LifetimeChurnForecastDriftWavePredictionScorecardRegressionWaveForecastBacktestsEmptyRunTests(
    unittest.TestCase
):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call_args(self, **overrides):
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
        )
        merged.update(overrides)
        return merged

    def zero_totals(self):
        return {
            "cutoffs": 0,
            "records": 0,
            "hit": 0,
            "missed": 0,
            "unresolved": 0,
            "unexpected": 0,
        }

    def test_empty_regression_wave_cutoffs_returns_empty_records(self):
        result = forecast_backtests_call(
            self.store,
            self.args,
            **self.call_args(regression_wave_cutoffs=()),
        )
        self.assertEqual(result["backtests"], ())
        self.assertEqual(result["totals"], self.zero_totals())

    def test_no_qualifying_identity_returns_empty_records(self):
        # The diamond store yields no regression wave set with three
        # appearances of one identity at cutoff 0.
        result = forecast_backtests_call(
            self.store, self.args, **self.call_args()
        )
        self.assertEqual(result["backtests"], ())
        self.assertEqual(result["totals"], self.zero_totals())


class LifetimeChurnForecastDriftWavePredictionScorecardRegressionWaveForecastBacktestsIsolationTests(
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
            regression_wave_horizon=10,
            regression_wave_forecast_limit=50,
            regression_wave_cutoffs=(0,),
            regression_wave_match_tolerance=1,
            regression_wave_forecast_backtest_limit=50,
        )

    def test_success_and_failure_change_nothing(self):
        store, args = diamond_store()
        heads = dict(store._heads)
        appends = copy.deepcopy(store._appends)
        merges = copy.deepcopy(store._merges)
        audit = store.audit_log()

        forecast_backtests_call(store, args, **self.call_args())
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
            dict(regression_wave_horizon=0),
            dict(regression_wave_forecast_limit=0),
            dict(wave_cutoffs="x"),
            dict(wave_cutoffs=(9,)),
            dict(match_tolerance=-1),
            dict(prediction_backtest_limit=0),
            dict(prediction_result_limit=0),
            dict(drift_limit=2),
            dict(scan_limit=1),
            dict(min_evaluated=-1),
            dict(min_cutoffs=0),
            dict(prediction_scorecard_limit=0),
            dict(baseline_splits="x"),
            dict(baseline_splits=()),
            dict(min_evaluated_delta=-1),
            dict(min_hit_rate_drop=(1, 0)),
            dict(regression_limit=0),
            dict(min_consecutive_splits=0),
            dict(regression_streak_limit=0),
            dict(min_active_regressions=0),
            dict(regression_wave_limit=0),
            dict(min_regression_wave_occurrences=0),
            dict(regression_wave_recurrence_limit=0),
            dict(max_regression_wave_jitter=-1),
            dict(regression_wave_periodicity_limit=0),
            dict(reference="ghost"),
        ]
        for overrides in failures:
            with self.subTest(overrides=overrides):
                merged = self.call_args()
                merged.update(overrides)
                with self.assertRaises(Exception):
                    forecast_backtests_call(store, args, **merged)

        self.assertEqual(dict(store._heads), heads)
        self.assertEqual(copy.deepcopy(store._appends), appends)
        self.assertEqual(copy.deepcopy(store._merges), merges)
        self.assertEqual(store.audit_log(), audit)


class LifetimeChurnForecastDriftWavePredictionScorecardRegressionWaveForecastBacktestsTokenTests(
    unittest.TestCase
):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, token=None, **overrides):
        return forecast_backtests_call(
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
            regression_wave_horizon=10,
            regression_wave_forecast_limit=50,
            regression_wave_cutoffs=(0,),
            regression_wave_match_tolerance=1,
            regression_wave_forecast_backtest_limit=50,
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
                **dict(self.call_args(), regression_wave_cutoffs="x"),
            )
        with self.assertRaises(ValueError):
            self.call(
                token=token,
                **dict(self.call_args(), regression_wave_forecast_backtest_limit=0),
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
