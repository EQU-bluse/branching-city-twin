import copy
import inspect
import unittest

from city_twin.branches import BranchStore

from tests.test_lifetime_churn import diamond_store
from tests.test_lifetime_churn_forecast_drift_scan import LONG_ARGS
from tests.test_lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_wave_forecast_scorecard_regression_wave_forecast_backtests import (
    build_backtests,
    synthetic_waves,
)
from tests.test_lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_wave_forecast_scorecard_regression_wave_forecast_backtests_scorecards import (
    backtest_record,
    build_scorecards,
    call_args as scorecards_call_args,
    forecast_backtest_scorecards_kwargs,
)


def regressions_kwargs(store_args, **overrides) -> dict:
    baseline_cutoffs = overrides.pop(
        "regression_wave_regression_wave_baseline_cutoffs", 1
    )
    min_evaluated_delta = overrides.pop(
        "regression_wave_regression_wave_min_evaluated_delta", 0
    )
    min_hit_rate_drop = overrides.pop(
        "regression_wave_regression_wave_min_hit_rate_drop", (0, 1)
    )
    regression_limit = overrides.pop(
        "regression_wave_regression_wave_regression_limit", 50
    )
    kwargs = forecast_backtest_scorecards_kwargs(store_args, **overrides)
    kwargs["regression_wave_regression_wave_baseline_cutoffs"] = (
        baseline_cutoffs
    )
    kwargs["regression_wave_regression_wave_min_evaluated_delta"] = (
        min_evaluated_delta
    )
    kwargs["regression_wave_regression_wave_min_hit_rate_drop"] = (
        min_hit_rate_drop
    )
    kwargs["regression_wave_regression_wave_regression_limit"] = (
        regression_limit
    )
    return kwargs


def regressions_call(store, store_args, **overrides):
    return store.lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_wave_forecast_scorecard_regression_wave_forecast_backtests_scorecards_regressions(
        **regressions_kwargs(store_args, **overrides)
    )


def build_regressions(
    records,
    scorecards,
    baseline_cutoffs=1,
    min_evaluated_delta=0,
    min_hit_rate_drop=(0, 1),
    regression_limit=50,
):
    return BranchStore._build_lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_wave_forecast_scorecard_regression_wave_forecast_backtests_scorecards_regressions_result(
        records,
        scorecards,
        baseline_cutoffs,
        min_evaluated_delta,
        min_hit_rate_drop,
        regression_limit,
    )


def kept_scorecards(*identities):
    return tuple((identity,) for identity in identities)


def call_args(**overrides):
    merged = scorecards_call_args(
        regression_wave_regression_wave_baseline_cutoffs=1,
        regression_wave_regression_wave_min_evaluated_delta=0,
        regression_wave_regression_wave_min_hit_rate_drop=(0, 1),
        regression_wave_regression_wave_regression_limit=50,
    )
    merged.update(overrides)
    return merged


class LifetimeChurnForecastDriftWavePredictionScorecardRegressionWaveForecastScorecardRegressionWaveForecastBacktestsScorecardsRegressionsSignatureTests(
    unittest.TestCase
):
    def test_public_signature_appends_four_parameters(self):
        parameters = list(
            inspect.signature(
                BranchStore.lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_wave_forecast_scorecard_regression_wave_forecast_backtests_scorecards_regressions
            ).parameters.values()
        )
        names = [p.name for p in parameters]
        scorecard_names = list(
            inspect.signature(
                BranchStore.lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_wave_forecast_scorecard_regression_wave_forecast_backtests_scorecards
            ).parameters
        )
        self.assertEqual(
            names,
            scorecard_names[:-1]
            + [
                "regression_wave_regression_wave_baseline_cutoffs",
                "regression_wave_regression_wave_min_evaluated_delta",
                "regression_wave_regression_wave_min_hit_rate_drop",
                "regression_wave_regression_wave_regression_limit",
                "token",
            ],
        )
        self.assertEqual(
            [p.default for p in parameters[:-1]],
            [inspect.Parameter.empty] * (len(parameters) - 1),
        )
        self.assertIsNone(parameters[-1].default)

    def test_result_shape_and_field_order(self):
        result = build_regressions(
            (
                backtest_record(2, "A", 2, 0),
                backtest_record(4, "A", 1, 1),
            ),
            kept_scorecards("A"),
            1,
            0,
            (0, 1),
        )
        self.assertEqual(list(result), ["regressions", "totals"])
        self.assertIsInstance(result["regressions"], tuple)
        regression = result["regressions"][0]
        self.assertEqual(
            list(regression),
            [
                "identity",
                "baseline_cutoffs",
                "observation_cutoffs",
                "baseline_evaluated",
                "observation_evaluated",
                "baseline_hit_rate",
                "observation_hit_rate",
                "rate_drop",
                "first_regression_cutoff",
            ],
        )
        self.assertEqual(
            list(result["totals"]),
            [
                "identities",
                "baseline_evaluated",
                "baseline_hit",
                "baseline_missed",
                "observation_evaluated",
                "observation_hit",
                "observation_missed",
            ],
        )


class LifetimeChurnForecastDriftWavePredictionScorecardRegressionWaveForecastScorecardRegressionWaveForecastBacktestsScorecardsRegressionsResultTests(
    unittest.TestCase
):
    def test_segments_split_and_aggregate(self):
        result = build_regressions(
            (
                backtest_record(2, "A", 2, 0),
                backtest_record(4, "A", 1, 1),
            ),
            kept_scorecards("A"),
            1,
            0,
            (0, 1),
        )
        self.assertEqual(len(result["regressions"]), 1)
        self.assertEqual(
            result["regressions"][0],
            {
                "identity": "A",
                "baseline_cutoffs": 1,
                "observation_cutoffs": 1,
                "baseline_evaluated": 2,
                "observation_evaluated": 2,
                "baseline_hit_rate": (1, 1),
                "observation_hit_rate": (1, 2),
                "rate_drop": (1, 2),
                "first_regression_cutoff": 4,
            },
        )
        self.assertEqual(
            result["totals"],
            {
                "identities": 1,
                "baseline_evaluated": 2,
                "baseline_hit": 2,
                "baseline_missed": 0,
                "observation_evaluated": 2,
                "observation_hit": 1,
                "observation_missed": 1,
            },
        )

    def test_only_scorecard_identities_are_candidates(self):
        records = (
            backtest_record(2, "A", 1, 0),
            backtest_record(4, "A", 0, 1),
            backtest_record(2, "Z", 1, 0),
            backtest_record(4, "Z", 0, 1),
        )
        result = build_regressions(
            records, kept_scorecards("A"), 1, 0, (0, 1)
        )
        self.assertEqual(
            [r["identity"] for r in result["regressions"]], ["A"]
        )
        self.assertEqual(result["totals"]["identities"], 1)

    def test_both_segments_must_hold_records(self):
        # Two records with baseline_cutoffs 2 leave the observation
        # segment empty.
        records = (
            backtest_record(2, "A", 2, 0),
            backtest_record(4, "A", 0, 2),
        )
        result = build_regressions(
            records, kept_scorecards("A"), 2, 0, (0, 1)
        )
        self.assertEqual(result["regressions"], ())
        # A single record with baseline_cutoffs 1 is empty too.
        result = build_regressions(
            records[:1], kept_scorecards("A"), 1, 0, (0, 1)
        )
        self.assertEqual(result["regressions"], ())

    def test_zero_evaluated_in_either_segment_disqualifies(self):
        # Baseline segment evaluates nothing (unresolved only).
        records = (
            backtest_record(2, "A", 0, 0, 2),
            backtest_record(4, "A", 0, 0, 1),
            backtest_record(6, "A", 0, 3),
        )
        result = build_regressions(
            records, kept_scorecards("A"), 2, 0, (0, 1)
        )
        self.assertEqual(result["regressions"], ())
        # Observation segment evaluates nothing.
        records = (
            backtest_record(2, "A", 1, 1),
            backtest_record(4, "A", 0, 0, 2),
        )
        result = build_regressions(
            records, kept_scorecards("A"), 1, 0, (0, 1)
        )
        self.assertEqual(result["regressions"], ())

    def test_min_evaluated_delta_bounds_the_observation_growth(self):
        records = (
            backtest_record(2, "A", 0, 1),
            backtest_record(4, "A", 0, 3),
            backtest_record(6, "A", 0, 1),
        )
        # Observation evaluated (4) exceeds baseline (1) by 3.
        result = build_regressions(
            records, kept_scorecards("A"), 1, 3, (0, 1)
        )
        self.assertEqual(len(result["regressions"]), 1)
        result = build_regressions(
            records, kept_scorecards("A"), 1, 4, (0, 1)
        )
        self.assertEqual(result["regressions"], ())

    def test_hit_rate_drop_threshold_is_inclusive(self):
        records = (
            backtest_record(2, "A", 2, 0),
            backtest_record(4, "A", 1, 1),
        )
        # The drop is exactly 1/2.
        result = build_regressions(
            records, kept_scorecards("A"), 1, 0, (1, 2)
        )
        self.assertEqual(len(result["regressions"]), 1)
        # 2/3 is strictly above 1/2, so the identity drops out.
        result = build_regressions(
            records, kept_scorecards("A"), 1, 0, (2, 3)
        )
        self.assertEqual(result["regressions"], ())

    def test_rate_drop_is_reduced_to_lowest_terms(self):
        records = (
            backtest_record(2, "A", 2, 0),
            backtest_record(4, "A", 1, 1),
        )
        result = build_regressions(
            records, kept_scorecards("A"), 1, 0, (0, 1)
        )
        self.assertEqual(result["regressions"][0]["rate_drop"], (1, 2))
        # A zero drop reduces to (0, 1) and still qualifies under a
        # zero threshold.
        records = (
            backtest_record(2, "A", 0, 1),
            backtest_record(4, "A", 0, 3),
        )
        result = build_regressions(
            records, kept_scorecards("A"), 1, 0, (0, 1)
        )
        self.assertEqual(result["regressions"][0]["rate_drop"], (0, 1))
        self.assertEqual(
            result["regressions"][0]["baseline_hit_rate"], (0, 1)
        )

    def test_first_regression_cutoff_tracks_accumulated_observation(self):
        # The evaluated delta is only reached once the second
        # observation record accumulates, so cutoff 6 is first.
        records = (
            backtest_record(2, "A", 0, 1),
            backtest_record(4, "A", 0, 1),
            backtest_record(6, "A", 0, 3),
        )
        result = build_regressions(
            records, kept_scorecards("A"), 1, 2, (0, 1)
        )
        self.assertEqual(
            result["regressions"][0]["first_regression_cutoff"], 6
        )
        # With the delta already met, the drop threshold decides: the
        # accumulated rate reaches 1/2 below baseline at cutoff 4.
        records = (
            backtest_record(2, "A", 2, 0),
            backtest_record(4, "A", 0, 2),
            backtest_record(6, "A", 2, 0),
        )
        result = build_regressions(
            records, kept_scorecards("A"), 1, 0, (1, 2)
        )
        regression = result["regressions"][0]
        self.assertEqual(regression["first_regression_cutoff"], 4)
        self.assertEqual(regression["rate_drop"], (1, 2))

    def test_sort_order_drop_then_observation_evaluated_then_identity(self):
        records = (
            # drop 1/1, observation evaluated 1
            backtest_record(2, "b", 1, 0),
            backtest_record(4, "b", 0, 1),
            # drop 1/1, observation evaluated 2
            backtest_record(2, "a", 1, 0),
            backtest_record(4, "a", 0, 1),
            backtest_record(6, "a", 0, 1),
            # drop 1/2
            backtest_record(2, "c", 1, 1),
            backtest_record(4, "c", 0, 2),
        )
        result = build_regressions(
            records, kept_scorecards("a", "b", "c"), 1, 0, (0, 1)
        )
        self.assertEqual(
            [r["identity"] for r in result["regressions"]],
            ["a", "b", "c"],
        )

    def test_tuple_identities_sort_by_unicode_code_point(self):
        records = (
            backtest_record(2, ("W", "y"), 1, 0),
            backtest_record(4, ("W", "y"), 0, 1),
            backtest_record(2, ("a0", "x"), 1, 0),
            backtest_record(4, ("a0", "x"), 0, 1),
            backtest_record(2, ("A", "z"), 1, 0),
            backtest_record(4, ("A", "z"), 0, 1),
        )
        result = build_regressions(
            records,
            kept_scorecards(("W", "y"), ("a0", "x"), ("A", "z")),
            1,
            0,
            (0, 1),
        )
        self.assertEqual(
            [r["identity"] for r in result["regressions"]],
            [("A", "z"), ("W", "y"), ("a0", "x")],
        )

    def test_regression_limit_bounds_the_complete_set_without_truncating(
        self,
    ):
        records = (
            backtest_record(2, "A", 2, 0),
            backtest_record(4, "A", 1, 1),
            backtest_record(2, "C", 2, 0),
            backtest_record(4, "C", 1, 1),
        )
        with self.assertRaises(ValueError) as caught:
            build_regressions(
                records, kept_scorecards("A", "C"), 1, 0, (1, 2), 1
            )
        self.assertIn("regression limit", str(caught.exception))
        result = build_regressions(
            records, kept_scorecards("A", "C"), 1, 0, (1, 2), 2
        )
        self.assertEqual(len(result["regressions"]), 2)
        # Identities filtered out below the limit never count toward it.
        result = build_regressions(
            records, kept_scorecards("A", "C"), 1, 0, (2, 3), 1
        )
        self.assertEqual(result["regressions"], ())

    def test_empty_records_return_empty_tuple_and_zero_totals(self):
        result = build_regressions((), ())
        self.assertEqual(result["regressions"], ())
        self.assertEqual(
            result["totals"],
            {
                "identities": 0,
                "baseline_evaluated": 0,
                "baseline_hit": 0,
                "baseline_missed": 0,
                "observation_evaluated": 0,
                "observation_hit": 0,
                "observation_missed": 0,
            },
        )

    def test_every_object_is_fresh(self):
        records = (
            backtest_record(2, ("A", "x"), 2, 0),
            backtest_record(4, ("A", "x"), 0, 2),
        )
        scorecards = kept_scorecards(("A", "x"))
        result = build_regressions(records, scorecards, 1, 0, (0, 1))
        result["regressions"][0]["identity"] = "Z"
        result["regressions"][0]["baseline_evaluated"] = 999
        result["totals"]["baseline_hit"] = 999
        again = build_regressions(records, scorecards, 1, 0, (0, 1))
        self.assertEqual(again["regressions"][0]["identity"], ("A", "x"))
        self.assertEqual(again["regressions"][0]["baseline_evaluated"], 2)
        self.assertEqual(again["totals"]["baseline_hit"], 2)
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


class LifetimeChurnForecastDriftWavePredictionScorecardRegressionWaveForecastScorecardRegressionWaveForecastBacktestsScorecardsRegressionsChainTests(
    unittest.TestCase
):
    """The builder refines exactly what the scorecards stage keeps."""

    def test_regressions_match_the_two_cutoff_backtest(self):
        backtests = build_backtests(
            synthetic_waves(), 1, 50, 0, 50, 2, 50, (2, 4), 0, 50
        )
        scorecards = build_scorecards(backtests["backtests"])
        result = build_regressions(
            backtests["backtests"], scorecards["scorecards"], 1, 0, (0, 1)
        )
        # B has a single record, so its observation segment is empty;
        # A and C tie on every rate and count, then sort by identity.
        self.assertEqual(
            [r["identity"] for r in result["regressions"]], ["A", "C"]
        )
        for regression in result["regressions"]:
            self.assertEqual(
                (
                    regression["baseline_cutoffs"],
                    regression["observation_cutoffs"],
                    regression["baseline_evaluated"],
                    regression["observation_evaluated"],
                    regression["baseline_hit_rate"],
                    regression["observation_hit_rate"],
                    regression["rate_drop"],
                    regression["first_regression_cutoff"],
                ),
                (1, 1, 2, 2, (1, 1), (1, 2), (1, 2), 4),
            )
        self.assertEqual(
            result["totals"],
            {
                "identities": 2,
                "baseline_evaluated": 4,
                "baseline_hit": 4,
                "baseline_missed": 0,
                "observation_evaluated": 4,
                "observation_hit": 2,
                "observation_missed": 2,
            },
        )
        # A threshold above the 1/2 drop empties the result.
        result = build_regressions(
            backtests["backtests"],
            scorecards["scorecards"],
            1,
            0,
            (1, 1),
        )
        self.assertEqual(result["regressions"], ())
        # Two baseline cutoffs leave no observation record.
        result = build_regressions(
            backtests["backtests"], scorecards["scorecards"], 2, 0, (0, 1)
        )
        self.assertEqual(result["regressions"], ())

    def test_scorecard_filters_gate_the_candidates(self):
        backtests = build_backtests(
            synthetic_waves(), 1, 50, 0, 50, 2, 50, (2, 4), 0, 50
        )
        # min_cutoffs 2 removes B from the scorecards; the regression
        # set is unchanged because B never qualifies anyway, while
        # min_evaluated 5 removes every scorecard and every candidate.
        scorecards = build_scorecards(backtests["backtests"], 0, 2)
        result = build_regressions(
            backtests["backtests"], scorecards["scorecards"], 1, 0, (0, 1)
        )
        self.assertEqual(
            [r["identity"] for r in result["regressions"]], ["A", "C"]
        )
        scorecards = build_scorecards(backtests["backtests"], 5, 1)
        result = build_regressions(
            backtests["backtests"], scorecards["scorecards"], 1, 0, (0, 1)
        )
        self.assertEqual(result["regressions"], ())

    def test_public_query_matches_the_builder_on_the_backtest_set(self):
        store, args = diamond_store()
        backtest_parameters = inspect.signature(
            BranchStore.lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_wave_forecast_scorecard_regression_wave_forecast_backtests
        ).parameters
        scorecard_parameters = inspect.signature(
            BranchStore.lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_wave_forecast_scorecard_regression_wave_forecast_backtests_scorecards
        ).parameters
        full_kwargs = regressions_kwargs(args, **call_args())
        backtests = store.lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_wave_forecast_scorecard_regression_wave_forecast_backtests(
            **{
                key: value
                for key, value in full_kwargs.items()
                if key in backtest_parameters
            }
        )
        scorecards = store.lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_wave_forecast_scorecard_regression_wave_forecast_backtests_scorecards(
            **{
                key: value
                for key, value in full_kwargs.items()
                if key in scorecard_parameters
            }
        )
        result = regressions_call(store, args, **call_args())
        self.assertEqual(
            result,
            build_regressions(
                backtests["backtests"],
                scorecards["scorecards"],
                1,
                0,
                (0, 1),
                50,
            ),
        )


class LifetimeChurnForecastDriftWavePredictionScorecardRegressionWaveForecastScorecardRegressionWaveForecastBacktestsScorecardsRegressionsValidationTests(
    unittest.TestCase
):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        return regressions_call(self.store, self.args, **overrides)

    def extended(self, **overrides):
        overrides.setdefault("split_cutoffs", (4,))
        overrides.setdefault("baseline_splits", (1,))
        merged = dict(LONG_ARGS)
        merged.update(overrides)
        return self.call(**merged)

    def test_baseline_cutoffs_must_be_a_non_bool_positive_int(self):
        for bad in (True, False, 1.0, "1", None, (1,)):
            with self.subTest(value=bad):
                with self.assertRaises(TypeError):
                    self.extended(
                        regression_wave_regression_wave_baseline_cutoffs=bad
                    )
        with self.assertRaises(ValueError):
            self.extended(
                regression_wave_regression_wave_baseline_cutoffs=0
            )
        with self.assertRaises(ValueError):
            self.extended(
                regression_wave_regression_wave_baseline_cutoffs=-2
            )

    def test_min_evaluated_delta_must_be_a_non_bool_non_negative_int(self):
        for bad in (True, False, 1.0, "0", None, (0,)):
            with self.subTest(value=bad):
                with self.assertRaises(TypeError):
                    self.extended(
                        regression_wave_regression_wave_min_evaluated_delta=bad
                    )
        with self.assertRaises(ValueError):
            self.extended(
                regression_wave_regression_wave_min_evaluated_delta=-1
            )
        # Zero is the lower bound and succeeds.
        self.extended(
            regression_wave_regression_wave_min_evaluated_delta=0
        )

    def test_min_hit_rate_drop_must_be_a_valid_fraction_tuple(self):
        for bad in (True, 1, "1/2", None, [1, 2]):
            with self.subTest(value=bad):
                with self.assertRaises(TypeError):
                    self.extended(
                        regression_wave_regression_wave_min_hit_rate_drop=bad
                    )
        for bad in ((True, 1), (1, True), (1.0, 2), (1, "2")):
            with self.subTest(value=bad):
                with self.assertRaises(TypeError):
                    self.extended(
                        regression_wave_regression_wave_min_hit_rate_drop=bad
                    )
        for bad in ((1,), (1, 2, 3)):
            with self.subTest(value=bad):
                with self.assertRaises(ValueError):
                    self.extended(
                        regression_wave_regression_wave_min_hit_rate_drop=bad
                    )
        with self.assertRaises(ValueError):
            self.extended(
                regression_wave_regression_wave_min_hit_rate_drop=(-1, 2)
            )
        with self.assertRaises(ValueError):
            self.extended(
                regression_wave_regression_wave_min_hit_rate_drop=(1, 0)
            )
        with self.assertRaises(ValueError):
            self.extended(
                regression_wave_regression_wave_min_hit_rate_drop=(1, -2)
            )
        with self.assertRaises(ValueError):
            self.extended(
                regression_wave_regression_wave_min_hit_rate_drop=(3, 2)
            )
        # The boundaries are legal.
        self.extended(
            regression_wave_regression_wave_min_hit_rate_drop=(0, 1)
        )
        self.extended(
            regression_wave_regression_wave_min_hit_rate_drop=(1, 1)
        )

    def test_regression_limit_must_be_a_non_bool_positive_int(self):
        for bad in (True, False, 1.0, "1", None, (1,)):
            with self.subTest(value=bad):
                with self.assertRaises(TypeError):
                    self.extended(
                        regression_wave_regression_wave_regression_limit=bad
                    )
        with self.assertRaises(ValueError):
            self.extended(
                regression_wave_regression_wave_regression_limit=0
            )
        with self.assertRaises(ValueError):
            self.extended(
                regression_wave_regression_wave_regression_limit=-3
            )

    def test_new_parameters_validated_in_signature_order(self):
        # The forecast scorecard limit fails before baseline_cutoffs.
        with self.assertRaises(ValueError):
            self.extended(
                regression_wave_regression_wave_forecast_scorecard_limit=0,
                regression_wave_regression_wave_baseline_cutoffs="x",
            )
        with self.assertRaises(TypeError):
            self.extended(
                regression_wave_regression_wave_forecast_scorecard_limit="x",
                regression_wave_regression_wave_baseline_cutoffs="x",
            )
        # baseline_cutoffs fails before min_evaluated_delta.
        with self.assertRaises(TypeError):
            self.extended(
                regression_wave_regression_wave_baseline_cutoffs="x",
                regression_wave_regression_wave_min_evaluated_delta="x",
            )
        with self.assertRaises(ValueError):
            self.extended(
                regression_wave_regression_wave_baseline_cutoffs=0,
                regression_wave_regression_wave_min_evaluated_delta=-1,
            )
        # min_evaluated_delta fails before min_hit_rate_drop.
        with self.assertRaises(TypeError):
            self.extended(
                regression_wave_regression_wave_min_evaluated_delta="x",
                regression_wave_regression_wave_min_hit_rate_drop="x",
            )
        with self.assertRaises(ValueError):
            self.extended(
                regression_wave_regression_wave_min_evaluated_delta=-1,
                regression_wave_regression_wave_min_hit_rate_drop=(1, 0),
            )
        # min_hit_rate_drop fails before the regression limit.
        with self.assertRaises(TypeError):
            self.extended(
                regression_wave_regression_wave_min_hit_rate_drop="x",
                regression_wave_regression_wave_regression_limit="x",
            )
        with self.assertRaises(ValueError):
            self.extended(
                regression_wave_regression_wave_min_hit_rate_drop=(1, 0),
                regression_wave_regression_wave_regression_limit=0,
            )
        # All four pass before the split membership check fires.
        with self.assertRaises(ValueError):
            self.extended(
                split_cutoffs=(99,),
                regression_wave_regression_wave_baseline_cutoffs=1,
                regression_wave_regression_wave_min_evaluated_delta=0,
                regression_wave_regression_wave_min_hit_rate_drop=(0, 1),
                regression_wave_regression_wave_regression_limit=1,
            )

    def test_all_parameter_checks_precede_state_lookups(self):
        with self.assertRaises(TypeError):
            self.call(
                reference="ghost",
                regression_wave_regression_wave_baseline_cutoffs="x",
            )
        with self.assertRaises(ValueError):
            self.call(
                reference="ghost",
                regression_wave_regression_wave_baseline_cutoffs=0,
            )
        with self.assertRaises(TypeError):
            self.call(
                reference="ghost",
                regression_wave_regression_wave_min_evaluated_delta="x",
            )
        with self.assertRaises(ValueError):
            self.call(
                reference="ghost",
                regression_wave_regression_wave_min_evaluated_delta=-1,
            )
        with self.assertRaises(TypeError):
            self.call(
                reference="ghost",
                regression_wave_regression_wave_min_hit_rate_drop="x",
            )
        with self.assertRaises(ValueError):
            self.call(
                reference="ghost",
                regression_wave_regression_wave_min_hit_rate_drop=(1, 0),
            )
        with self.assertRaises(TypeError):
            self.call(
                reference="ghost",
                regression_wave_regression_wave_regression_limit="x",
            )
        with self.assertRaises(ValueError):
            self.call(
                reference="ghost",
                regression_wave_regression_wave_regression_limit=0,
            )

    def test_underlying_stage_limits_and_errors_are_unchanged(self):
        # The diamond store yields one drift wave, so a drift-wave
        # cutoff past it keeps the shared backtest stage's range error;
        # it yields no forecast regression waves, so the deep cutoff
        # range check is skipped there exactly as in the scorecards
        # query.
        with self.assertRaises(ValueError) as caught:
            self.extended(
                wave_cutoffs=(9,),
                regression_wave_regression_wave_baseline_cutoffs=1,
                regression_wave_regression_wave_min_evaluated_delta=0,
                regression_wave_regression_wave_min_hit_rate_drop=(0, 1),
                regression_wave_regression_wave_regression_limit=50,
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
        self.assertIn("forecast limit", str(caught.exception))
        with self.assertRaises(ValueError) as caught:
            build_backtests(
                synthetic_waves(), 1, 50, 0, 50, 2, 50, (2, 4), 0, 4
            )
        self.assertIn("backtest limit", str(caught.exception))
        # The scorecard stage cap likewise raises before the
        # regression stage aggregates.
        backtests = build_backtests(
            synthetic_waves(), 1, 50, 0, 50, 2, 50, (2, 4), 0, 50
        )
        with self.assertRaises(ValueError) as caught:
            build_scorecards(backtests["backtests"], 0, 1, 2)
        self.assertIn("scorecard limit", str(caught.exception))


class LifetimeChurnForecastDriftWavePredictionScorecardRegressionWaveForecastScorecardRegressionWaveForecastBacktestsScorecardsRegressionsEmptyRunTests(
    unittest.TestCase
):
    def setUp(self):
        self.store, self.args = diamond_store()

    def zero_totals(self):
        return {
            "identities": 0,
            "baseline_evaluated": 0,
            "baseline_hit": 0,
            "baseline_missed": 0,
            "observation_evaluated": 0,
            "observation_hit": 0,
            "observation_missed": 0,
        }

    def test_empty_deep_cutoffs_returns_empty_regressions(self):
        result = regressions_call(
            self.store,
            self.args,
            **call_args(regression_wave_regression_wave_cutoffs=()),
        )
        self.assertEqual(result["regressions"], ())
        self.assertEqual(result["totals"], self.zero_totals())

    def test_no_forecast_regression_waves_returns_empty_regressions(self):
        # The diamond store yields no forecast regression waves at
        # all, so the range check is skipped and nothing trains.
        result = regressions_call(
            self.store,
            self.args,
            **call_args(regression_wave_regression_wave_cutoffs=(3,)),
        )
        self.assertEqual(result["regressions"], ())
        self.assertEqual(result["totals"], self.zero_totals())


class LifetimeChurnForecastDriftWavePredictionScorecardRegressionWaveForecastScorecardRegressionWaveForecastBacktestsScorecardsRegressionsIsolationTests(
    unittest.TestCase
):
    def test_success_and_failure_change_nothing(self):
        store, args = diamond_store()
        heads = dict(store._heads)
        appends = copy.deepcopy(store._appends)
        merges = copy.deepcopy(store._merges)
        audit = store.audit_log()

        regressions_call(store, args, **call_args())
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
            dict(regression_wave_regression_wave_forecast_backtest_limit=False),
            dict(regression_wave_regression_wave_forecast_backtest_limit=0),
            dict(regression_wave_regression_wave_forecast_limit=0),
            dict(drift_limit=2),
            dict(scan_limit=1),
            dict(regression_wave_regression_wave_min_evaluated="x"),
            dict(regression_wave_regression_wave_min_evaluated=True),
            dict(regression_wave_regression_wave_min_evaluated=-1),
            dict(regression_wave_regression_wave_min_cutoffs="x"),
            dict(regression_wave_regression_wave_min_cutoffs=False),
            dict(regression_wave_regression_wave_min_cutoffs=0),
            dict(regression_wave_regression_wave_forecast_scorecard_limit="x"),
            dict(regression_wave_regression_wave_forecast_scorecard_limit=False),
            dict(regression_wave_regression_wave_forecast_scorecard_limit=0),
            dict(regression_wave_regression_wave_baseline_cutoffs="x"),
            dict(regression_wave_regression_wave_baseline_cutoffs=False),
            dict(regression_wave_regression_wave_baseline_cutoffs=0),
            dict(regression_wave_regression_wave_min_evaluated_delta="x"),
            dict(regression_wave_regression_wave_min_evaluated_delta=True),
            dict(regression_wave_regression_wave_min_evaluated_delta=-1),
            dict(regression_wave_regression_wave_min_hit_rate_drop="x"),
            dict(regression_wave_regression_wave_min_hit_rate_drop=(1, 0)),
            dict(regression_wave_regression_wave_min_hit_rate_drop=(2, 1)),
            dict(regression_wave_regression_wave_regression_limit="x"),
            dict(regression_wave_regression_wave_regression_limit=False),
            dict(regression_wave_regression_wave_regression_limit=0),
            dict(reference="ghost"),
        ]
        for overrides in failures:
            with self.subTest(overrides=overrides):
                merged = call_args()
                merged.update(overrides)
                with self.assertRaises(Exception):
                    regressions_call(store, args, **merged)

        self.assertEqual(dict(store._heads), heads)
        self.assertEqual(copy.deepcopy(store._appends), appends)
        self.assertEqual(copy.deepcopy(store._merges), merges)
        self.assertEqual(store.audit_log(), audit)


class LifetimeChurnForecastDriftWavePredictionScorecardRegressionWaveForecastScorecardRegressionWaveForecastBacktestsScorecardsRegressionsTokenTests(
    unittest.TestCase
):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, token=None, **overrides):
        return regressions_call(
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
                    regression_wave_regression_wave_baseline_cutoffs="x"
                ),
            )
        with self.assertRaises(ValueError):
            self.call(
                token=token,
                **call_args(
                    regression_wave_regression_wave_regression_limit=0
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
