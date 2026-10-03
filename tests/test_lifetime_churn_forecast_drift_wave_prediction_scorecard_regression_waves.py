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
)
from tests.test_lifetime_churn_forecast_drift_wave_prediction_scorecard_regressions import (
    backtest_record,
    kept_scorecards,
    regressing_records,
)
from tests.test_lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_streaks import (
    build_streaks,
    streaks_kwargs,
)


def waves_kwargs(store_args, **overrides) -> dict:
    min_active_regressions = overrides.pop("min_active_regressions", 1)
    regression_wave_limit = overrides.pop("regression_wave_limit", 50)
    kwargs = streaks_kwargs(store_args, **overrides)
    kwargs["min_active_regressions"] = min_active_regressions
    kwargs["regression_wave_limit"] = regression_wave_limit
    return kwargs


def waves_call(store, store_args, **overrides):
    return store.lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_waves(
        **waves_kwargs(store_args, **overrides)
    )


def build_waves(
    streaks,
    baseline_splits=(1,),
    min_active_regressions=1,
    regression_wave_limit=50,
):
    return BranchStore._build_lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_waves_result(
        streaks,
        baseline_splits,
        min_active_regressions,
        regression_wave_limit,
    )


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


class LifetimeChurnForecastDriftWavePredictionScorecardRegressionWavesSignatureTests(
    unittest.TestCase
):
    def test_public_signature_appends_two_parameters(self):
        parameters = list(
            inspect.signature(
                BranchStore.lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_waves
            ).parameters.values()
        )
        names = [p.name for p in parameters]
        streak_names = list(
            inspect.signature(
                BranchStore.lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_streaks
            ).parameters
        )
        self.assertEqual(
            names,
            streak_names[:-1]
            + [
                "min_active_regressions",
                "regression_wave_limit",
                "token",
            ],
        )
        self.assertEqual(
            [p.default for p in parameters[:-1]],
            [inspect.Parameter.empty] * (len(parameters) - 1),
        )
        self.assertIsNone(parameters[-1].default)

    def test_result_shape_and_key_order(self):
        result = build_waves(make_streaks(("A", 1, 2)), (1, 2))
        self.assertEqual(list(result), ["waves", "totals"])
        self.assertIsInstance(result["waves"], tuple)
        wave = result["waves"][0]
        self.assertEqual(
            list(wave),
            [
                "start_baseline_cutoffs",
                "end_baseline_cutoffs",
                "split_count",
                "peak_regressions",
                "points",
            ],
        )
        self.assertEqual(
            list(wave["points"][0]),
            ["baseline_cutoffs", "active_count", "identities"],
        )
        self.assertEqual(
            list(result["totals"]),
            ["splits", "identities", "waves", "points", "peak_active"],
        )


class LifetimeChurnForecastDriftWavePredictionScorecardRegressionWavesResultTests(
    unittest.TestCase
):
    def test_adjacent_active_positions_form_one_wave(self):
        result = build_waves(make_streaks(("A", 1, 3)), (1, 2, 3))
        self.assertEqual(len(result["waves"]), 1)
        self.assertEqual(
            result["waves"][0],
            {
                "start_baseline_cutoffs": 1,
                "end_baseline_cutoffs": 3,
                "split_count": 3,
                "peak_regressions": 1,
                "points": (
                    {
                        "baseline_cutoffs": 1,
                        "active_count": 1,
                        "identities": ("A",),
                    },
                    {
                        "baseline_cutoffs": 2,
                        "active_count": 1,
                        "identities": ("A",),
                    },
                    {
                        "baseline_cutoffs": 3,
                        "active_count": 1,
                        "identities": ("A",),
                    },
                ),
            },
        )
        self.assertEqual(
            result["totals"],
            {
                "splits": 3,
                "identities": 1,
                "waves": 1,
                "points": 3,
                "peak_active": 1,
            },
        )

    def test_unrequested_integer_positions_do_not_break_adjacency(self):
        # Requested positions 1, 3 and 5 are adjacent even though the
        # integers 2 and 4 were never requested.
        result = build_waves(make_streaks(("A", 1, 5)), (1, 3, 5))
        self.assertEqual(len(result["waves"]), 1)
        wave = result["waves"][0]
        self.assertEqual(wave["start_baseline_cutoffs"], 1)
        self.assertEqual(wave["end_baseline_cutoffs"], 5)
        self.assertEqual(wave["split_count"], 3)
        self.assertEqual(
            [point["baseline_cutoffs"] for point in wave["points"]],
            [1, 3, 5],
        )

    def test_position_below_threshold_ends_the_wave(self):
        streaks = make_streaks(("A", 1, 1), ("B", 1, 1), ("A2", 3, 3))
        result = build_waves(streaks, (1, 2, 3), 2)
        # Only position 1 reaches two active identities; positions 2
        # and 3 fall below the threshold, so each is a wave boundary.
        self.assertEqual(len(result["waves"]), 1)
        wave = result["waves"][0]
        self.assertEqual(wave["start_baseline_cutoffs"], 1)
        self.assertEqual(wave["end_baseline_cutoffs"], 1)
        self.assertEqual(wave["split_count"], 1)
        self.assertEqual(wave["peak_regressions"], 2)
        self.assertEqual(wave["points"][0]["identities"], ("A", "B"))
        self.assertEqual(
            result["totals"],
            {
                "splits": 1,
                "identities": 2,
                "waves": 1,
                "points": 1,
                "peak_active": 2,
            },
        )

    def test_separate_runs_become_separate_waves_in_start_order(self):
        streaks = make_streaks(("A", 1, 2), ("B", 4, 5))
        result = build_waves(streaks, (1, 2, 3, 4, 5))
        self.assertEqual(len(result["waves"]), 2)
        self.assertEqual(result["waves"][0]["start_baseline_cutoffs"], 1)
        self.assertEqual(result["waves"][0]["end_baseline_cutoffs"], 2)
        self.assertEqual(result["waves"][1]["start_baseline_cutoffs"], 4)
        self.assertEqual(result["waves"][1]["end_baseline_cutoffs"], 5)
        self.assertEqual(
            result["totals"],
            {
                "splits": 4,
                "identities": 2,
                "waves": 2,
                "points": 4,
                "peak_active": 1,
            },
        )

    def test_identities_follow_the_streak_final_order(self):
        # The streak tuple order, not the identity sort order, decides
        # the point identity order.
        streaks = make_streaks(("b", 1, 2), ("A", 1, 1), ("c", 2, 2))
        result = build_waves(streaks, (1, 2))
        wave = result["waves"][0]
        self.assertEqual(wave["points"][0]["identities"], ("b", "A"))
        self.assertEqual(wave["points"][1]["identities"], ("b", "c"))
        self.assertEqual(wave["peak_regressions"], 2)

    def test_regression_wave_limit_bounds_the_complete_set(self):
        streaks = make_streaks(("A", 1, 1), ("B", 3, 3))
        with self.assertRaises(ValueError) as caught:
            build_waves(streaks, (1, 2, 3), 1, 1)
        self.assertIn("regression wave limit", str(caught.exception))
        result = build_waves(streaks, (1, 2, 3), 1, 2)
        self.assertEqual(len(result["waves"]), 2)
        # Positions below the threshold never form waves to count.
        result = build_waves(streaks, (1, 2, 3), 2, 1)
        self.assertEqual(result["waves"], ())

    def test_empty_streaks_return_empty_tuple_and_zero_totals(self):
        result = build_waves((), (1, 2, 3))
        self.assertEqual(result["waves"], ())
        self.assertEqual(
            result["totals"],
            {
                "splits": 0,
                "identities": 0,
                "waves": 0,
                "points": 0,
                "peak_active": 0,
            },
        )

    def test_every_object_is_fresh(self):
        streaks = make_streaks((("A", "x"), 1, 2))
        result = build_waves(streaks, (1, 2))
        result["waves"][0]["split_count"] = 999
        result["waves"][0]["points"][0]["active_count"] = 999
        result["totals"]["waves"] = 999
        again = build_waves(streaks, (1, 2))
        self.assertEqual(again["waves"][0]["split_count"], 2)
        self.assertEqual(again["waves"][0]["points"][0]["active_count"], 1)
        self.assertEqual(again["totals"]["waves"], 1)
        # The streak records are not mutated either.
        self.assertEqual(streaks[0]["identity"], ("A", "x"))

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


class LifetimeChurnForecastDriftWavePredictionScorecardRegressionWavesChainTests(
    unittest.TestCase
):
    """The wave builder consumes exactly what the streak stage keeps."""

    def test_waves_match_the_streaks_result(self):
        records = regressing_records("A") + regressing_records("C")
        scorecards = kept_scorecards("A", "C")
        streaks = build_streaks(
            records, scorecards, (1, 2), 0, (1, 2)
        )
        result = build_waves(streaks["streaks"], (1, 2), 2)
        self.assertEqual(len(result["waves"]), 1)
        wave = result["waves"][0]
        self.assertEqual(wave["start_baseline_cutoffs"], 1)
        self.assertEqual(wave["end_baseline_cutoffs"], 2)
        self.assertEqual(wave["split_count"], 2)
        self.assertEqual(wave["peak_regressions"], 2)
        self.assertEqual(
            [point["identities"] for point in wave["points"]],
            [("A", "C"), ("A", "C")],
        )
        self.assertEqual(
            result["totals"],
            {
                "splits": 2,
                "identities": 2,
                "waves": 1,
                "points": 2,
                "peak_active": 2,
            },
        )
        # Requiring three active identities empties the result.
        result = build_waves(streaks["streaks"], (1, 2), 3)
        self.assertEqual(result["waves"], ())
        self.assertEqual(
            result["totals"],
            {
                "splits": 0,
                "identities": 0,
                "waves": 0,
                "points": 0,
                "peak_active": 0,
            },
        )

    def test_waves_match_the_two_cutoff_backtest(self):
        backtests = build_backtests(
            synthetic_waves(), 1, 50, 0, 50, 2, 50, (2, 4), 0, 50
        )
        scorecards = build_scorecards(backtests["backtests"])
        streaks = build_streaks(
            backtests["backtests"], scorecards["scorecards"], (1,), 0, (0, 1)
        )
        result = build_waves(streaks["streaks"], (1,), 2)
        self.assertEqual(len(result["waves"]), 1)
        wave = result["waves"][0]
        self.assertEqual(wave["split_count"], 1)
        self.assertEqual(wave["peak_regressions"], 2)
        self.assertEqual(wave["points"][0]["identities"], ("A", "C"))


class LifetimeChurnForecastDriftWavePredictionScorecardRegressionWavesValidationTests(
    unittest.TestCase
):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        return waves_call(self.store, self.args, **overrides)

    def extended(self, **overrides):
        overrides.setdefault("split_cutoffs", (4,))
        merged = dict(LONG_ARGS)
        merged.update(overrides)
        return self.call(**merged)

    def test_min_active_regressions_must_be_a_non_bool_positive_int(self):
        for bad in (True, False, 1.0, "1", None, (1,)):
            with self.subTest(value=bad):
                with self.assertRaises(TypeError):
                    self.extended(min_active_regressions=bad)
        with self.assertRaises(ValueError):
            self.extended(min_active_regressions=0)
        with self.assertRaises(ValueError):
            self.extended(min_active_regressions=-2)

    def test_regression_wave_limit_must_be_a_non_bool_positive_int(self):
        for bad in (True, False, 1.0, "1", None, (1,)):
            with self.subTest(value=bad):
                with self.assertRaises(TypeError):
                    self.extended(regression_wave_limit=bad)
        with self.assertRaises(ValueError):
            self.extended(regression_wave_limit=0)
        with self.assertRaises(ValueError):
            self.extended(regression_wave_limit=-3)

    def test_new_parameters_validated_in_signature_order(self):
        # regression_streak_limit fails before min_active_regressions.
        with self.assertRaises(TypeError):
            self.extended(
                regression_streak_limit="x", min_active_regressions="x"
            )
        with self.assertRaises(ValueError):
            self.extended(
                regression_streak_limit=0, min_active_regressions=0
            )
        # min_active_regressions fails before regression_wave_limit.
        with self.assertRaises(TypeError):
            self.extended(
                min_active_regressions="x", regression_wave_limit="x"
            )
        with self.assertRaises(ValueError):
            self.extended(
                min_active_regressions=0, regression_wave_limit=0
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
            )

    def test_all_parameter_checks_precede_state_lookups(self):
        with self.assertRaises(TypeError):
            self.call(
                reference="ghost",
                cutoffs=(0, 1),
                min_active_regressions="x",
            )
        with self.assertRaises(ValueError):
            self.call(
                reference="ghost", cutoffs=(0, 1), min_active_regressions=0
            )
        with self.assertRaises(TypeError):
            self.call(
                reference="ghost",
                cutoffs=(0, 1),
                regression_wave_limit="x",
            )
        with self.assertRaises(ValueError):
            self.call(
                reference="ghost", cutoffs=(0, 1), regression_wave_limit=0
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
            )
        self.assertIn("out of range", str(caught.exception))


class LifetimeChurnForecastDriftWavePredictionScorecardRegressionWavesEmptyRunTests(
    unittest.TestCase
):
    def setUp(self):
        self.store, self.args = diamond_store()

    def zero_totals(self):
        return {
            "splits": 0,
            "identities": 0,
            "waves": 0,
            "points": 0,
            "peak_active": 0,
        }

    def test_empty_wave_cutoffs_returns_empty_waves_and_zero_totals(self):
        result = waves_call(
            self.store,
            self.args,
            **dict(LONG_ARGS, split_cutoffs=(3, 4, 5), wave_cutoffs=()),
        )
        self.assertEqual(result["waves"], ())
        self.assertEqual(result["totals"], self.zero_totals())

    def test_no_qualifying_identity_returns_empty_waves(self):
        # The diamond store yields a single drift wave, so no identity
        # reaches three training appearances at cutoff 0.
        result = waves_call(
            self.store,
            self.args,
            **dict(LONG_ARGS, split_cutoffs=(3, 4, 5), wave_cutoffs=(0,)),
        )
        self.assertEqual(result["waves"], ())
        self.assertEqual(result["totals"], self.zero_totals())


class LifetimeChurnForecastDriftWavePredictionScorecardRegressionWavesIsolationTests(
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
        )

    def test_success_and_failure_change_nothing(self):
        store, args = diamond_store()
        heads = dict(store._heads)
        appends = copy.deepcopy(store._appends)
        merges = copy.deepcopy(store._merges)
        audit = store.audit_log()

        waves_call(store, args, **self.call_args())
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
            dict(reference="ghost"),
        ]
        for overrides in failures:
            with self.subTest(overrides=overrides):
                merged = self.call_args()
                merged.update(overrides)
                with self.assertRaises(Exception):
                    waves_call(store, args, **merged)

        self.assertEqual(dict(store._heads), heads)
        self.assertEqual(copy.deepcopy(store._appends), appends)
        self.assertEqual(copy.deepcopy(store._merges), merges)
        self.assertEqual(store.audit_log(), audit)


class LifetimeChurnForecastDriftWavePredictionScorecardRegressionWavesTokenTests(
    unittest.TestCase
):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, token=None, **overrides):
        return waves_call(self.store, self.args, token=token, **overrides)

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
                **dict(self.call_args(), min_active_regressions="x"),
            )
        with self.assertRaises(ValueError):
            self.call(
                token=token,
                **dict(self.call_args(), regression_wave_limit=0),
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
