import copy
import inspect
import unittest

from city_twin.branches import BranchStore

from tests.test_lifetime_churn import diamond_store
from tests.test_lifetime_churn_forecast_drift_scan import LONG_ARGS
from tests.test_lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_waves import (
    waves_call,
)
from tests.test_lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_wave_recurrences import (
    make_wave,
    recurrences_call,
)
from tests.test_lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_wave_periodicities import (
    build_periodicities,
    periodicities_call,
    periodicities_kwargs,
)


def forecasts_kwargs(store_args, **overrides) -> dict:
    regression_wave_horizon = overrides.pop("regression_wave_horizon", 10)
    regression_wave_forecast_limit = overrides.pop(
        "regression_wave_forecast_limit", 50
    )
    kwargs = periodicities_kwargs(store_args, **overrides)
    kwargs["regression_wave_horizon"] = regression_wave_horizon
    kwargs["regression_wave_forecast_limit"] = regression_wave_forecast_limit
    return kwargs


def forecasts_call(store, store_args, **overrides):
    return store.lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_wave_forecasts(
        **forecasts_kwargs(store_args, **overrides)
    )


def build_forecasts(
    periodicities,
    boundary,
    regression_wave_horizon=10,
    regression_wave_forecast_limit=50,
):
    return BranchStore._build_lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_wave_forecasts_result(
        periodicities,
        boundary,
        regression_wave_horizon,
        regression_wave_forecast_limit,
    )


class LifetimeChurnForecastDriftWavePredictionScorecardRegressionWaveForecastsSignatureTests(
    unittest.TestCase
):
    def test_public_signature_appends_two_parameters(self):
        parameters = list(
            inspect.signature(
                BranchStore.lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_wave_forecasts
            ).parameters.values()
        )
        names = [p.name for p in parameters]
        periodicity_names = list(
            inspect.signature(
                BranchStore.lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_wave_periodicities
            ).parameters
        )
        self.assertEqual(
            names,
            periodicity_names[:-1]
            + [
                "regression_wave_horizon",
                "regression_wave_forecast_limit",
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
        periodicities = build_periodicities(
            BranchStore._build_lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_wave_recurrences_result(
                waves, 1, 50
            )["recurrences"]
        )
        result = build_forecasts(periodicities["periodicities"], 2)
        self.assertEqual(list(result), ["forecasts", "totals"])
        self.assertIsInstance(result["forecasts"], tuple)
        record = result["forecasts"][0]
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
                "forecasts",
            ],
        )
        self.assertEqual(
            list(record["forecasts"][0]),
            ["ordinal", "wave", "earliest", "latest"],
        )
        self.assertEqual(
            list(result["totals"]),
            [
                "identities",
                "predictions",
                "waves",
                "points",
                "first_wave",
                "last_wave",
            ],
        )


class LifetimeChurnForecastDriftWavePredictionScorecardRegressionWaveForecastsResultTests(
    unittest.TestCase
):
    def make_periodicities(self, waves):
        recurrences = BranchStore._build_lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_wave_recurrences_result(
            waves, 1, 50
        )
        return build_periodicities(recurrences["recurrences"])

    def test_candidates_advance_by_period_from_last_wave(self):
        waves = (
            make_wave(1, [(1, 2, ("A",))]),
            make_wave(2, [(2, 1, ("A",))]),
            make_wave(3, [(3, 3, ("A",))]),
        )
        periodicities = self.make_periodicities(waves)
        result = build_forecasts(
            periodicities["periodicities"], 2, regression_wave_horizon=3
        )
        self.assertEqual(len(result["forecasts"]), 1)
        record = result["forecasts"][0]
        self.assertEqual(record["identity"], "A")
        self.assertEqual(record["last_wave"], 2)
        self.assertEqual(
            record["forecasts"],
            (
                {"ordinal": 1, "wave": 3, "earliest": 3, "latest": 3},
                {"ordinal": 2, "wave": 4, "earliest": 4, "latest": 4},
                {"ordinal": 3, "wave": 5, "earliest": 5, "latest": 5},
            ),
        )
        self.assertEqual(
            result["totals"],
            {
                "identities": 1,
                "predictions": 3,
                "waves": 3,
                "points": 3,
                "first_wave": 0,
                "last_wave": 2,
            },
        )

    def test_candidates_must_exceed_the_boundary(self):
        waves = (
            make_wave(1, [(1, 1, ("A",))]),
            make_wave(2, [(2, 1, ("A",))]),
            make_wave(3, [(3, 1, ("A",))]),
        )
        periodicities = self.make_periodicities(waves)
        # The record's last_wave is 2; a boundary of 4 discards the
        # candidates at waves 3 and 4, which are not strictly greater,
        # and keeps waves 5 and 6 inside boundary + horizon.
        result = build_forecasts(
            periodicities["periodicities"], 4, regression_wave_horizon=2
        )
        self.assertEqual(
            result["forecasts"][0]["forecasts"],
            (
                {"ordinal": 3, "wave": 5, "earliest": 5, "latest": 5},
                {"ordinal": 4, "wave": 6, "earliest": 6, "latest": 6},
            ),
        )

    def test_earliest_and_latest_come_from_the_interval_extremes(self):
        waves = (
            make_wave(1, [(1, 1, ("A",))]),
            make_wave(2, [(2, 1, ("B",))]),
            make_wave(3, [(3, 1, ("A",))]),
            make_wave(4, [(4, 1, ("B",))]),
            make_wave(5, [(5, 1, ("C",))]),
            make_wave(6, [(6, 1, ("C",))]),
            make_wave(7, [(7, 1, ("A",))]),
        )
        # "A" takes part in waves 0, 2 and 6: intervals (2, 4),
        # period 2.
        periodicities = self.make_periodicities(waves)
        result = build_forecasts(
            periodicities["periodicities"], 6, regression_wave_horizon=4
        )
        record = result["forecasts"][0]
        self.assertEqual(record["period"], 2)
        self.assertEqual(
            record["forecasts"],
            (
                {"ordinal": 1, "wave": 8, "earliest": 8, "latest": 10},
                {"ordinal": 2, "wave": 10, "earliest": 10, "latest": 14},
            ),
        )

    def test_identity_without_candidates_keeps_no_record(self):
        waves = (
            make_wave(1, [(1, 1, ("A", "B"))]),
            make_wave(2, [(2, 1, ("A",))]),
            make_wave(3, [(3, 1, ("A", "B"))]),
            make_wave(4, [(4, 1, ("C",))]),
            make_wave(5, [(5, 1, ("B",))]),
        )
        # "A" has period 1 and projects inside the horizon; "B" has
        # period 2 from last_wave 4, so its first candidate is 6.
        periodicities = self.make_periodicities(waves)
        result = build_forecasts(
            periodicities["periodicities"], 4, regression_wave_horizon=1
        )
        self.assertEqual(
            [record["identity"] for record in result["forecasts"]], ["A"]
        )
        self.assertEqual(
            result["totals"],
            {
                "identities": 1,
                "predictions": 1,
                "waves": 3,
                "points": 3,
                "first_wave": 0,
                "last_wave": 2,
            },
        )

    def test_records_keep_the_periodicity_sort_order(self):
        waves = (
            make_wave(1, [(1, 1, ("c", "b", "a", "d"))]),
            make_wave(2, [(2, 1, ("a", "c", "d")), (3, 1, ("a",))]),
            make_wave(3, [(3, 1, ("a", "b", "d"))]),
            make_wave(4, [(4, 1, ("c", "d"))]),
            make_wave(5, [(5, 1, ("b",))]),
        )
        periodicities = self.make_periodicities(waves)
        self.assertEqual(
            [record["identity"] for record in periodicities["periodicities"]],
            ["d", "a", "b", "c"],
        )
        result = build_forecasts(
            periodicities["periodicities"], 4, regression_wave_horizon=10
        )
        self.assertEqual(
            [record["identity"] for record in result["forecasts"]],
            ["d", "a", "b", "c"],
        )

    def test_regression_wave_forecast_limit_bounds_the_candidate_total(
        self,
    ):
        waves = (
            make_wave(1, [(1, 1, ("A", "B"))]),
            make_wave(2, [(2, 1, ("A", "B"))]),
            make_wave(3, [(3, 1, ("A", "B"))]),
        )
        periodicities = self.make_periodicities(waves)
        # Two identities with period 1 each project three candidates
        # inside a horizon of three: six predictions in total.
        result = build_forecasts(
            periodicities["periodicities"],
            2,
            regression_wave_horizon=3,
            regression_wave_forecast_limit=6,
        )
        self.assertEqual(result["totals"]["predictions"], 6)
        with self.assertRaises(ValueError) as caught:
            build_forecasts(
                periodicities["periodicities"],
                2,
                regression_wave_horizon=3,
                regression_wave_forecast_limit=5,
            )
        self.assertIn(
            "regression wave forecast limit", str(caught.exception)
        )

    def test_empty_periodicities_return_empty_tuple_and_zero_totals(self):
        result = build_forecasts((), -1)
        self.assertEqual(result["forecasts"], ())
        self.assertEqual(
            result["totals"],
            {
                "identities": 0,
                "predictions": 0,
                "waves": 0,
                "points": 0,
                "first_wave": 0,
                "last_wave": 0,
            },
        )

    def test_every_object_is_fresh(self):
        waves = (
            make_wave(1, [(1, 1, (("A", "x"),))]),
            make_wave(2, [(2, 1, (("A", "x"),))]),
            make_wave(3, [(3, 1, (("A", "x"),))]),
        )
        periodicities = self.make_periodicities(waves)
        result = build_forecasts(periodicities["periodicities"], 2)
        result["forecasts"][0]["waves"] = 999
        result["forecasts"][0]["appearances"][0]["points"] = 999
        result["forecasts"][0]["forecasts"][0]["wave"] = 999
        result["totals"]["predictions"] = 999
        again = build_forecasts(periodicities["periodicities"], 2)
        self.assertEqual(again["forecasts"][0]["waves"], 3)
        self.assertEqual(
            again["forecasts"][0]["appearances"][0]["points"], 1
        )
        self.assertEqual(again["forecasts"][0]["forecasts"][0]["wave"], 3)
        self.assertEqual(again["totals"]["predictions"], 10)
        # The periodicity records are not mutated either.
        self.assertEqual(periodicities["periodicities"][0]["waves"], 3)

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


class LifetimeChurnForecastDriftWavePredictionScorecardRegressionWaveForecastsChainTests(
    unittest.TestCase
):
    """The forecast builder consumes exactly what periodicities keep."""

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

    def test_public_query_matches_the_periodicities_query(self):
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
            )
        }
        waves = waves_call(store, args, **wave_args)
        periodicities = periodicities_call(store, args, **call_args)
        result = forecasts_call(
            store,
            args,
            **dict(
                call_args,
                regression_wave_horizon=4,
                regression_wave_forecast_limit=50,
            )
        )
        self.assertEqual(
            result,
            build_forecasts(
                periodicities["periodicities"],
                len(waves["waves"]) - 1,
                4,
                50,
            ),
        )


class LifetimeChurnForecastDriftWavePredictionScorecardRegressionWaveForecastsValidationTests(
    unittest.TestCase
):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        return forecasts_call(self.store, self.args, **overrides)

    def extended(self, **overrides):
        overrides.setdefault("split_cutoffs", (4,))
        merged = dict(LONG_ARGS)
        merged.update(overrides)
        return self.call(**merged)

    def test_regression_wave_horizon_must_be_a_non_bool_positive_int(self):
        for bad in (True, False, 1.0, "1", None, (1,)):
            with self.subTest(value=bad):
                with self.assertRaises(TypeError):
                    self.extended(regression_wave_horizon=bad)
        with self.assertRaises(ValueError):
            self.extended(regression_wave_horizon=0)
        with self.assertRaises(ValueError):
            self.extended(regression_wave_horizon=-3)

    def test_regression_wave_forecast_limit_must_be_a_non_bool_positive_int(
        self,
    ):
        for bad in (True, False, 1.0, "1", None, (1,)):
            with self.subTest(value=bad):
                with self.assertRaises(TypeError):
                    self.extended(regression_wave_forecast_limit=bad)
        with self.assertRaises(ValueError):
            self.extended(regression_wave_forecast_limit=0)
        with self.assertRaises(ValueError):
            self.extended(regression_wave_forecast_limit=-3)

    def test_new_parameters_validated_in_signature_order(self):
        # regression_wave_periodicity_limit fails before
        # regression_wave_horizon.
        with self.assertRaises(TypeError):
            self.extended(
                regression_wave_periodicity_limit="x",
                regression_wave_horizon="x",
            )
        with self.assertRaises(ValueError):
            self.extended(
                regression_wave_periodicity_limit=0,
                regression_wave_horizon=0,
            )
        # regression_wave_horizon fails before
        # regression_wave_forecast_limit.
        with self.assertRaises(TypeError):
            self.extended(
                regression_wave_horizon="x",
                regression_wave_forecast_limit="x",
            )
        with self.assertRaises(ValueError):
            self.extended(
                regression_wave_horizon=0,
                regression_wave_forecast_limit=0,
            )
        # All pass before the split membership check fires.
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
                regression_wave_horizon=1,
                regression_wave_forecast_limit=1,
            )

    def test_all_parameter_checks_precede_state_lookups(self):
        with self.assertRaises(TypeError):
            self.call(
                reference="ghost",
                cutoffs=(0, 1),
                regression_wave_horizon="x",
            )
        with self.assertRaises(ValueError):
            self.call(
                reference="ghost",
                cutoffs=(0, 1),
                regression_wave_horizon=0,
            )
        with self.assertRaises(TypeError):
            self.call(
                reference="ghost",
                cutoffs=(0, 1),
                regression_wave_forecast_limit="x",
            )
        with self.assertRaises(ValueError):
            self.call(
                reference="ghost",
                cutoffs=(0, 1),
                regression_wave_forecast_limit=0,
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
                regression_wave_horizon=10,
                regression_wave_forecast_limit=50,
            )
        self.assertIn("out of range", str(caught.exception))


class LifetimeChurnForecastDriftWavePredictionScorecardRegressionWaveForecastsEmptyRunTests(
    unittest.TestCase
):
    def setUp(self):
        self.store, self.args = diamond_store()

    def zero_totals(self):
        return {
            "identities": 0,
            "predictions": 0,
            "waves": 0,
            "points": 0,
            "first_wave": 0,
            "last_wave": 0,
        }

    def test_empty_wave_cutoffs_returns_empty_forecasts_and_zero_totals(
        self,
    ):
        result = forecasts_call(
            self.store,
            self.args,
            **dict(LONG_ARGS, split_cutoffs=(3, 4, 5), wave_cutoffs=()),
        )
        self.assertEqual(result["forecasts"], ())
        self.assertEqual(result["totals"], self.zero_totals())

    def test_no_qualifying_identity_returns_empty_forecasts(self):
        # The diamond store yields a single drift wave, so no identity
        # reaches three regression wave appearances at cutoff 0.
        result = forecasts_call(
            self.store,
            self.args,
            **dict(LONG_ARGS, split_cutoffs=(3, 4, 5), wave_cutoffs=(0,)),
        )
        self.assertEqual(result["forecasts"], ())
        self.assertEqual(result["totals"], self.zero_totals())


class LifetimeChurnForecastDriftWavePredictionScorecardRegressionWaveForecastsIsolationTests(
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
        )

    def test_success_and_failure_change_nothing(self):
        store, args = diamond_store()
        heads = dict(store._heads)
        appends = copy.deepcopy(store._appends)
        merges = copy.deepcopy(store._merges)
        audit = store.audit_log()

        forecasts_call(store, args, **self.call_args())
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
            dict(regression_wave_horizon="x"),
            dict(regression_wave_horizon=False),
            dict(regression_wave_horizon=0),
            dict(regression_wave_forecast_limit="x"),
            dict(regression_wave_forecast_limit=False),
            dict(regression_wave_forecast_limit=0),
            dict(reference="ghost"),
        ]
        for overrides in failures:
            with self.subTest(overrides=overrides):
                merged = self.call_args()
                merged.update(overrides)
                with self.assertRaises(Exception):
                    forecasts_call(store, args, **merged)

        self.assertEqual(dict(store._heads), heads)
        self.assertEqual(copy.deepcopy(store._appends), appends)
        self.assertEqual(copy.deepcopy(store._merges), merges)
        self.assertEqual(store.audit_log(), audit)


class LifetimeChurnForecastDriftWavePredictionScorecardRegressionWaveForecastsTokenTests(
    unittest.TestCase
):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, token=None, **overrides):
        return forecasts_call(self.store, self.args, token=token, **overrides)

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
                **dict(self.call_args(), regression_wave_horizon="x"),
            )
        with self.assertRaises(ValueError):
            self.call(
                token=token,
                **dict(self.call_args(), regression_wave_forecast_limit=0),
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
