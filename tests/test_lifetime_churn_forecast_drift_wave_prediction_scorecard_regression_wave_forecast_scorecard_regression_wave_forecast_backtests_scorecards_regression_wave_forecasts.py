import copy
import inspect
import unittest

from city_twin.branches import BranchStore

from tests.test_lifetime_churn import diamond_store
from tests.test_lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_wave_forecast_scorecard_regression_wave_forecast_backtests_scorecards import (
    call_args,
)
from tests.test_lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_wave_forecast_scorecard_regression_wave_forecast_backtests_scorecards_regression_wave_periodicities import (
    build_periodicities,
    build_recurrences,
    deep_call_args,
    make_wave,
    periodicities_call,
    periodicities_kwargs,
)
from tests.test_lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_wave_forecast_scorecard_regression_wave_forecast_backtests_scorecards_regression_wave_recurrences import (
    waves_call,
)


def forecasts_kwargs(store_args, **overrides) -> dict:
    horizon = overrides.pop(
        "regression_wave_regression_wave_regression_wave_horizon", 10
    )
    forecast_limit = overrides.pop(
        "regression_wave_regression_wave_regression_wave_forecast_limit", 50
    )
    kwargs = periodicities_kwargs(store_args, **overrides)
    kwargs["regression_wave_regression_wave_regression_wave_horizon"] = (
        horizon
    )
    kwargs[
        "regression_wave_regression_wave_regression_wave_forecast_limit"
    ] = forecast_limit
    return kwargs


def forecasts_call(store, store_args, **overrides):
    return store.lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_wave_forecast_scorecard_regression_wave_forecast_backtests_scorecards_regression_wave_forecasts(
        **forecasts_kwargs(store_args, **overrides)
    )


def build_forecasts(
    periodicities,
    boundary,
    regression_wave_regression_wave_regression_wave_horizon=10,
    regression_wave_regression_wave_regression_wave_forecast_limit=50,
):
    return BranchStore._build_lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_wave_forecast_scorecard_regression_wave_forecast_backtests_scorecards_regression_wave_forecasts_result(
        periodicities,
        boundary,
        regression_wave_regression_wave_regression_wave_horizon,
        regression_wave_regression_wave_regression_wave_forecast_limit,
    )


def wave_set(waves):
    recurrences = build_recurrences(waves)
    return build_periodicities(recurrences["recurrences"])["periodicities"]


class LifetimeChurnForecastDriftWavePredictionScorecardRegressionWaveForecastScorecardRegressionWaveForecastBacktestsScorecardsRegressionWaveForecastsSignatureTests(
    unittest.TestCase
):
    def test_public_signature_appends_two_parameters(self):
        parameters = list(
            inspect.signature(
                BranchStore.lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_wave_forecast_scorecard_regression_wave_forecast_backtests_scorecards_regression_wave_forecasts
            ).parameters.values()
        )
        names = [p.name for p in parameters]
        periodicity_names = list(
            inspect.signature(
                BranchStore.lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_wave_forecast_scorecard_regression_wave_forecast_backtests_scorecards_regression_wave_periodicities
            ).parameters
        )
        self.assertEqual(
            names,
            periodicity_names[:-1]
            + [
                "regression_wave_regression_wave_regression_wave_horizon",
                "regression_wave_regression_wave_regression_wave_forecast_limit",
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
        result = build_forecasts(wave_set(waves), 2, 2)
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


class LifetimeChurnForecastDriftWavePredictionScorecardRegressionWaveForecastScorecardRegressionWaveForecastBacktestsScorecardsRegressionWaveForecastsResultTests(
    unittest.TestCase
):
    def test_projection_advances_from_the_last_wave_by_period(self):
        waves = (
            make_wave(1, [(1, 2, ("A",))]),
            make_wave(2, [(2, 1, ("A",))]),
            make_wave(3, [(3, 3, ("A",))]),
        )
        result = build_forecasts(wave_set(waves), 2, 2)
        self.assertEqual(len(result["forecasts"]), 1)
        record = result["forecasts"][0]
        self.assertEqual(record["identity"], "A")
        self.assertEqual(record["period"], 1)
        self.assertEqual(
            record["forecasts"],
            (
                {"ordinal": 1, "wave": 3, "earliest": 3, "latest": 3},
                {"ordinal": 2, "wave": 4, "earliest": 4, "latest": 4},
            ),
        )
        self.assertEqual(
            result["totals"],
            {
                "identities": 1,
                "predictions": 2,
                "waves": 3,
                "points": 3,
                "first_wave": 0,
                "last_wave": 2,
            },
        )

    def test_candidates_at_or_below_the_boundary_keep_their_ordinal(self):
        # A stops after wave 2; B rides every wave 0-4, so the boundary
        # is wave 4. A's first two candidates (waves 3 and 4) are
        # observed and skipped; wave 5 keeps ordinal 3.
        waves = (
            make_wave(1, [(1, 2, ("A", "B"))]),
            make_wave(2, [(2, 2, ("A", "B"))]),
            make_wave(3, [(3, 2, ("A", "B"))]),
            make_wave(4, [(4, 1, ("B",))]),
            make_wave(5, [(5, 1, ("B",))]),
        )
        result = build_forecasts(wave_set(waves), 4, 3)
        records = {
            record["identity"]: record for record in result["forecasts"]
        }
        # Equal jitter; B has five participating waves versus A's
        # three, so B sorts first.
        self.assertEqual(
            [record["identity"] for record in result["forecasts"]],
            ["B", "A"],
        )
        self.assertEqual(
            [prediction["wave"] for prediction in records["A"]["forecasts"]],
            [5, 6, 7],
        )
        self.assertEqual(
            [
                prediction["ordinal"]
                for prediction in records["A"]["forecasts"]
            ],
            [3, 4, 5],
        )
        self.assertEqual(
            [prediction["wave"] for prediction in records["B"]["forecasts"]],
            [5, 6, 7],
        )
        self.assertEqual(
            [
                prediction["ordinal"]
                for prediction in records["B"]["forecasts"]
            ],
            [1, 2, 3],
        )

    def test_earliest_and_latest_use_interval_extremes(self):
        # A rides waves 0, 1 and 3: intervals (1, 2), period 1.
        waves = (
            make_wave(1, [(1, 1, ("A",))]),
            make_wave(2, [(2, 1, ("A",))]),
            make_wave(3, [(3, 1, ("B",))]),
            make_wave(4, [(4, 1, ("A",))]),
        )
        result = build_forecasts(wave_set(waves), 3, 2)
        self.assertEqual(len(result["forecasts"]), 1)
        record = result["forecasts"][0]
        self.assertEqual(record["intervals"], (1, 2))
        self.assertEqual(
            record["forecasts"],
            (
                {"ordinal": 1, "wave": 4, "earliest": 4, "latest": 5},
                {"ordinal": 2, "wave": 5, "earliest": 5, "latest": 7},
            ),
        )

    def test_horizon_is_inclusive_and_a_record_without_candidates_dropped(
        self,
    ):
        # A rides waves 0, 2 and 4 (period two); B rides every wave.
        # With boundary 4 and horizon one, A's next wave 6 passes the
        # upper bound 5 and A is dropped, while B keeps wave 5.
        waves = (
            make_wave(1, [(1, 2, ("A", "B"))]),
            make_wave(2, [(2, 1, ("B",))]),
            make_wave(3, [(3, 2, ("A", "B"))]),
            make_wave(4, [(4, 1, ("B",))]),
            make_wave(5, [(5, 2, ("A", "B"))]),
        )
        result = build_forecasts(wave_set(waves), 4, 1)
        self.assertEqual(
            [record["identity"] for record in result["forecasts"]],
            ["B"],
        )
        self.assertEqual(
            result["forecasts"][0]["forecasts"],
            ({"ordinal": 1, "wave": 5, "earliest": 5, "latest": 5},),
        )
        self.assertEqual(result["totals"]["identities"], 1)
        self.assertEqual(result["totals"]["predictions"], 1)

    def test_horizon_inclusive_boundary_plus_horizon_is_kept(self):
        waves = (
            make_wave(1, [(1, 1, ("A",))]),
            make_wave(2, [(2, 1, ("A",))]),
            make_wave(3, [(3, 1, ("A",))]),
        )
        result = build_forecasts(wave_set(waves), 2, 1)
        self.assertEqual(
            [
                prediction["wave"]
                for prediction in result["forecasts"][0]["forecasts"]
            ],
            [3],
        )

    def test_forecast_limit_counts_predictions_not_identities(self):
        waves = (
            make_wave(1, [(1, 3, ("A", "B", "C"))]),
            make_wave(2, [(2, 3, ("A", "B", "C"))]),
            make_wave(3, [(3, 3, ("A", "B", "C"))]),
        )
        periodicities = wave_set(waves)
        with self.assertRaises(ValueError) as caught:
            build_forecasts(periodicities, 2, 2, 5)
        self.assertIn("forecast limit", str(caught.exception))
        result = build_forecasts(periodicities, 2, 2, 6)
        self.assertEqual(len(result["forecasts"]), 3)
        self.assertEqual(result["totals"]["predictions"], 6)

    def test_totals_summarise_history_and_projections(self):
        # B rides all four waves; A rides waves 0-2. Boundary 3,
        # horizon two: B projects waves 4, 5; A projects 4, 5 starting
        # from ordinal 2 (wave 3 is its skipped ordinal 1).
        waves = (
            make_wave(1, [(1, 2, ("A", "B"))]),
            make_wave(2, [(2, 2, ("A", "B"))]),
            make_wave(3, [(3, 2, ("A", "B"))]),
            make_wave(4, [(4, 1, ("B",))]),
        )
        result = build_forecasts(wave_set(waves), 3, 2)
        self.assertEqual(
            result["totals"],
            {
                "identities": 2,
                "predictions": 4,
                "waves": 7,
                "points": 7,
                "first_wave": 0,
                "last_wave": 3,
            },
        )

    def test_empty_periodicities_return_empty_tuple_and_zero_totals(self):
        result = build_forecasts((), -1, 10)
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
        periodicities = wave_set(waves)
        result = build_forecasts(periodicities, 2, 2)
        result["forecasts"][0]["waves"] = 999
        result["forecasts"][0]["appearances"][0]["points"] = 999
        result["forecasts"][0]["forecasts"][0]["wave"] = 999
        result["totals"]["predictions"] = 999
        again = build_forecasts(periodicities, 2, 2)
        self.assertEqual(again["forecasts"][0]["waves"], 3)
        self.assertEqual(
            again["forecasts"][0]["appearances"][0]["points"], 1
        )
        self.assertEqual(again["forecasts"][0]["forecasts"][0]["wave"], 3)
        self.assertEqual(again["totals"]["predictions"], 2)
        # The periodicity records are not mutated either.
        self.assertEqual(periodicities[0]["waves"], 3)
        self.assertNotIn("forecasts", periodicities[0])

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


class LifetimeChurnForecastDriftWavePredictionScorecardRegressionWaveForecastScorecardRegressionWaveForecastBacktestsScorecardsRegressionWaveForecastsChainTests(
    unittest.TestCase
):
    """The forecasts builder consumes exactly what periodicities keep."""

    def test_public_query_matches_the_periodicities_query(self):
        store, args = diamond_store()
        kwargs = deep_call_args(
            regression_wave_regression_wave_regression_wave_horizon=10,
            regression_wave_regression_wave_regression_wave_forecast_limit=50,
        )
        periodicity_kwargs = dict(kwargs)
        del periodicity_kwargs[
            "regression_wave_regression_wave_regression_wave_horizon"
        ]
        del periodicity_kwargs[
            "regression_wave_regression_wave_regression_wave_forecast_limit"
        ]
        periodicities = periodicities_call(
            store, args, **periodicity_kwargs
        )

        wave_parameters = inspect.signature(
            BranchStore.lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_wave_forecast_scorecard_regression_wave_forecast_backtests_scorecards_regression_waves
        ).parameters
        waves = waves_call(
            store,
            args,
            **{
                key: value
                for key, value in kwargs.items()
                if key in wave_parameters
            },
        )
        boundary = len(waves["waves"]) - 1

        result = forecasts_call(store, args, **kwargs)
        self.assertEqual(
            result,
            build_forecasts(
                periodicities["periodicities"],
                boundary,
                kwargs[
                    "regression_wave_regression_wave_regression_wave_horizon"
                ],
                kwargs[
                    "regression_wave_regression_wave_regression_wave_forecast_limit"
                ],
            ),
        )


class LifetimeChurnForecastDriftWavePredictionScorecardRegressionWaveForecastScorecardRegressionWaveForecastBacktestsScorecardsRegressionWaveForecastsValidationTests(
    unittest.TestCase
):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        return forecasts_call(self.store, self.args, **overrides)

    def extended(self, **overrides):
        overrides.setdefault("split_cutoffs", (4,))
        overrides.setdefault("baseline_splits", (1,))
        merged = dict(call_args())
        merged.update(overrides)
        return self.call(**merged)

    def test_horizon_must_be_a_non_bool_positive_int(self):
        for bad in (True, False, 1.0, "1", None, (1,)):
            with self.subTest(value=bad):
                with self.assertRaises(TypeError):
                    self.extended(
                        regression_wave_regression_wave_regression_wave_horizon=bad
                    )
        with self.assertRaises(ValueError):
            self.extended(
                regression_wave_regression_wave_regression_wave_horizon=0
            )
        with self.assertRaises(ValueError):
            self.extended(
                regression_wave_regression_wave_regression_wave_horizon=-2
            )

    def test_forecast_limit_must_be_a_non_bool_positive_int(self):
        for bad in (True, False, 1.0, "1", None, (1,)):
            with self.subTest(value=bad):
                with self.assertRaises(TypeError):
                    self.extended(
                        regression_wave_regression_wave_regression_wave_forecast_limit=bad
                    )
        with self.assertRaises(ValueError):
            self.extended(
                regression_wave_regression_wave_regression_wave_forecast_limit=0
            )
        with self.assertRaises(ValueError):
            self.extended(
                regression_wave_regression_wave_regression_wave_forecast_limit=-3
            )

    def test_new_parameters_validated_in_signature_order(self):
        # The deep periodicity limit fails before the deep horizon.
        with self.assertRaises(TypeError):
            self.extended(
                regression_wave_regression_wave_regression_wave_periodicity_limit="x",
                regression_wave_regression_wave_regression_wave_horizon="x",
            )
        with self.assertRaises(ValueError):
            self.extended(
                regression_wave_regression_wave_regression_wave_periodicity_limit=0,
                regression_wave_regression_wave_regression_wave_horizon=0,
            )
        # The deep horizon fails before the deep forecast limit.
        with self.assertRaises(TypeError):
            self.extended(
                regression_wave_regression_wave_regression_wave_horizon="x",
                regression_wave_regression_wave_regression_wave_forecast_limit="x",
            )
        with self.assertRaises(ValueError):
            self.extended(
                regression_wave_regression_wave_regression_wave_horizon=0,
                regression_wave_regression_wave_regression_wave_forecast_limit=0,
            )
        # All pass before the split membership check fires.
        with self.assertRaises(ValueError):
            self.extended(
                split_cutoffs=(99,),
                regression_wave_regression_wave_baseline_splits=(1,),
                regression_wave_regression_wave_min_evaluated_delta=0,
                regression_wave_regression_wave_min_hit_rate_drop=(0, 1),
                regression_wave_regression_wave_regression_limit=1,
                regression_wave_regression_wave_min_consecutive_splits=1,
                regression_wave_regression_wave_regression_streak_limit=1,
                regression_wave_regression_wave_min_active_regressions=1,
                regression_wave_regression_wave_regression_wave_limit=1,
                regression_wave_regression_wave_min_regression_wave_occurrences=1,
                regression_wave_regression_wave_regression_wave_recurrence_limit=1,
                regression_wave_regression_wave_max_regression_wave_jitter=0,
                regression_wave_regression_wave_regression_wave_periodicity_limit=1,
                regression_wave_regression_wave_regression_wave_horizon=1,
                regression_wave_regression_wave_regression_wave_forecast_limit=1,
            )

    def test_all_parameter_checks_precede_state_lookups(self):
        with self.assertRaises(TypeError):
            self.call(
                reference="ghost",
                regression_wave_regression_wave_regression_wave_horizon="x",
            )
        with self.assertRaises(ValueError):
            self.call(
                reference="ghost",
                regression_wave_regression_wave_regression_wave_horizon=0,
            )
        with self.assertRaises(TypeError):
            self.call(
                reference="ghost",
                regression_wave_regression_wave_regression_wave_forecast_limit="x",
            )
        with self.assertRaises(ValueError):
            self.call(
                reference="ghost",
                regression_wave_regression_wave_regression_wave_forecast_limit=0,
            )

    def test_underlying_stage_limits_and_errors_are_unchanged(self):
        with self.assertRaises(ValueError) as caught:
            self.extended(
                wave_cutoffs=(9,),
                regression_wave_regression_wave_baseline_splits=(1,),
                regression_wave_regression_wave_min_evaluated_delta=0,
                regression_wave_regression_wave_min_hit_rate_drop=(0, 1),
                regression_wave_regression_wave_regression_limit=50,
                regression_wave_regression_wave_min_consecutive_splits=1,
                regression_wave_regression_wave_regression_streak_limit=50,
                regression_wave_regression_wave_min_active_regressions=1,
                regression_wave_regression_wave_regression_wave_limit=50,
            )
        self.assertIn("out of range", str(caught.exception))


class LifetimeChurnForecastDriftWavePredictionScorecardRegressionWaveForecastScorecardRegressionWaveForecastBacktestsScorecardsRegressionWaveForecastsEmptyRunTests(
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

    def test_empty_deep_cutoffs_returns_empty_forecasts(self):
        result = forecasts_call(
            self.store,
            self.args,
            **call_args(regression_wave_regression_wave_cutoffs=()),
        )
        self.assertEqual(result["forecasts"], ())
        self.assertEqual(result["totals"], self.zero_totals())

    def test_no_qualifying_identity_returns_empty_forecasts(self):
        # The diamond store yields no deep forecast regression waves at
        # cutoff 3, so no identity reaches three appearances.
        result = forecasts_call(
            self.store,
            self.args,
            **call_args(regression_wave_regression_wave_cutoffs=(3,)),
        )
        self.assertEqual(result["forecasts"], ())
        self.assertEqual(result["totals"], self.zero_totals())


class LifetimeChurnForecastDriftWavePredictionScorecardRegressionWaveForecastScorecardRegressionWaveForecastBacktestsScorecardsRegressionWaveForecastsIsolationTests(
    unittest.TestCase
):
    def test_success_and_failure_change_nothing(self):
        store, args = diamond_store()
        heads = dict(store._heads)
        appends = copy.deepcopy(store._appends)
        merges = copy.deepcopy(store._merges)
        audit = store.audit_log()

        forecasts_call(store, args, **deep_call_args())
        failures = [
            dict(regression_wave_regression_wave_cutoffs="x"),
            dict(regression_wave_regression_wave_cutoffs=(True,)),
            dict(regression_wave_regression_wave_cutoffs=(-1,)),
            dict(regression_wave_regression_wave_cutoffs=(0, 0)),
            dict(regression_wave_regression_wave_cutoffs=(1, 0)),
            dict(regression_wave_regression_wave_match_tolerance="x"),
            dict(regression_wave_regression_wave_match_tolerance=True),
            dict(regression_wave_regression_wave_match_tolerance=-1),
            dict(regression_wave_regression_wave_forecast_backtest_limit="x"),
            dict(
                regression_wave_regression_wave_forecast_backtest_limit=False
            ),
            dict(regression_wave_regression_wave_forecast_backtest_limit=0),
            dict(regression_wave_regression_wave_min_evaluated="x"),
            dict(regression_wave_regression_wave_min_evaluated=True),
            dict(regression_wave_regression_wave_min_evaluated=-1),
            dict(regression_wave_regression_wave_min_cutoffs="x"),
            dict(regression_wave_regression_wave_min_cutoffs=False),
            dict(regression_wave_regression_wave_min_cutoffs=0),
            dict(
                regression_wave_regression_wave_forecast_scorecard_limit="x"
            ),
            dict(
                regression_wave_regression_wave_forecast_scorecard_limit=False
            ),
            dict(regression_wave_regression_wave_forecast_scorecard_limit=0),
            dict(regression_wave_regression_wave_forecast_limit=0),
            dict(regression_wave_regression_wave_horizon=0),
            dict(regression_wave_regression_wave_baseline_splits="x"),
            dict(regression_wave_regression_wave_baseline_splits=False),
            dict(regression_wave_regression_wave_baseline_splits=()),
            dict(regression_wave_regression_wave_baseline_splits=(0,)),
            dict(regression_wave_regression_wave_baseline_splits=(2, 1)),
            dict(regression_wave_regression_wave_min_evaluated_delta="x"),
            dict(regression_wave_regression_wave_min_evaluated_delta=True),
            dict(regression_wave_regression_wave_min_evaluated_delta=-1),
            dict(regression_wave_regression_wave_min_hit_rate_drop="x"),
            dict(regression_wave_regression_wave_min_hit_rate_drop=(1, 0)),
            dict(regression_wave_regression_wave_min_hit_rate_drop=(2, 1)),
            dict(regression_wave_regression_wave_regression_limit="x"),
            dict(regression_wave_regression_wave_regression_limit=False),
            dict(regression_wave_regression_wave_regression_limit=0),
            dict(regression_wave_regression_wave_min_consecutive_splits="x"),
            dict(
                regression_wave_regression_wave_min_consecutive_splits=False
            ),
            dict(regression_wave_regression_wave_min_consecutive_splits=0),
            dict(regression_wave_regression_wave_regression_streak_limit="x"),
            dict(
                regression_wave_regression_wave_regression_streak_limit=False
            ),
            dict(regression_wave_regression_wave_regression_streak_limit=0),
            dict(regression_wave_regression_wave_min_active_regressions="x"),
            dict(
                regression_wave_regression_wave_min_active_regressions=False
            ),
            dict(regression_wave_regression_wave_min_active_regressions=0),
            dict(regression_wave_regression_wave_regression_wave_limit="x"),
            dict(
                regression_wave_regression_wave_regression_wave_limit=False
            ),
            dict(regression_wave_regression_wave_regression_wave_limit=0),
            dict(
                regression_wave_regression_wave_min_regression_wave_occurrences="x"
            ),
            dict(
                regression_wave_regression_wave_min_regression_wave_occurrences=False
            ),
            dict(
                regression_wave_regression_wave_min_regression_wave_occurrences=0
            ),
            dict(
                regression_wave_regression_wave_regression_wave_recurrence_limit="x"
            ),
            dict(
                regression_wave_regression_wave_regression_wave_recurrence_limit=False
            ),
            dict(
                regression_wave_regression_wave_regression_wave_recurrence_limit=0
            ),
            dict(
                regression_wave_regression_wave_max_regression_wave_jitter="x"
            ),
            dict(
                regression_wave_regression_wave_max_regression_wave_jitter=False
            ),
            dict(
                regression_wave_regression_wave_max_regression_wave_jitter=-1
            ),
            dict(
                regression_wave_regression_wave_regression_wave_periodicity_limit="x"
            ),
            dict(
                regression_wave_regression_wave_regression_wave_periodicity_limit=False
            ),
            dict(
                regression_wave_regression_wave_regression_wave_periodicity_limit=0
            ),
            dict(
                regression_wave_regression_wave_regression_wave_horizon="x"
            ),
            dict(
                regression_wave_regression_wave_regression_wave_horizon=True
            ),
            dict(
                regression_wave_regression_wave_regression_wave_horizon=0
            ),
            dict(
                regression_wave_regression_wave_regression_wave_forecast_limit="x"
            ),
            dict(
                regression_wave_regression_wave_regression_wave_forecast_limit=False
            ),
            dict(
                regression_wave_regression_wave_regression_wave_forecast_limit=0
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
                    forecasts_call(store, args, **merged)

        self.assertEqual(dict(store._heads), heads)
        self.assertEqual(copy.deepcopy(store._appends), appends)
        self.assertEqual(copy.deepcopy(store._merges), merges)
        self.assertEqual(store.audit_log(), audit)


class LifetimeChurnForecastDriftWavePredictionScorecardRegressionWaveForecastScorecardRegressionWaveForecastBacktestsScorecardsRegressionWaveForecastsTokenTests(
    unittest.TestCase
):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, token=None, **overrides):
        return forecasts_call(
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
                    regression_wave_regression_wave_regression_wave_horizon="x"
                ),
            )
        with self.assertRaises(ValueError):
            self.call(
                token=token,
                **deep_call_args(
                    regression_wave_regression_wave_regression_wave_forecast_limit=0
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
