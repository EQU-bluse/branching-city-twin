import copy
import inspect
import unittest

from city_twin.branches import BranchStore

from tests.test_lifetime_churn import diamond_store
from tests.test_lifetime_churn_forecast_drift_scan import LONG_ARGS
from tests.test_lifetime_churn_forecast_drift_wave_prediction_backtests import (
    backtests_kwargs,
)


def scorecards_kwargs(store_args, **overrides) -> dict:
    min_evaluated = overrides.pop("min_evaluated", 0)
    min_cutoffs = overrides.pop("min_cutoffs", 1)
    prediction_scorecard_limit = overrides.pop(
        "prediction_scorecard_limit", 50
    )
    kwargs = backtests_kwargs(store_args, **overrides)
    kwargs["min_evaluated"] = min_evaluated
    kwargs["min_cutoffs"] = min_cutoffs
    kwargs["prediction_scorecard_limit"] = prediction_scorecard_limit
    return kwargs


def scorecards_call(store, store_args, **overrides):
    return store.lifetime_churn_forecast_drift_wave_prediction_scorecards(
        **scorecards_kwargs(store_args, **overrides)
    )


def build_scorecards(records, *args):
    return BranchStore._build_lifetime_churn_forecast_drift_wave_prediction_scorecards_result(
        records, *args
    )


def record(cutoff, identity, hit, missed, unresolved, unexpected):
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


def synthetic_records():
    """Four identities pinning aggregation, rates and first failures.

    A rides three cutoffs with one miss at cutoff 1; B fails already at
    cutoff 0 with a miss and an unexpected appearance; C is unresolved
    everywhere, so it is never evaluated and never fails; D hits every
    target at two cutoffs.
    """
    return (
        record(0, "A", 2, 0, 1, 0),
        record(1, "A", 1, 1, 0, 0),
        record(2, "A", 0, 0, 2, 0),
        record(0, "B", 0, 2, 0, 1),
        record(2, "B", 1, 0, 0, 0),
        record(1, "C", 0, 0, 3, 0),
        record(0, "D", 2, 0, 0, 0),
        record(1, "D", 2, 0, 0, 0),
    )


def tie_records():
    """Equal hit rates break ties on unexpected, missed, then identity."""
    return (
        record(0, "b", 1, 1, 0, 1),
        record(0, "a", 2, 2, 0, 1),
        record(0, "c", 1, 1, 0, 0),
        record(0, "d", 1, 2, 0, 1),
        record(0, "e", 1, 1, 0, 1),
    )


class LifetimeChurnForecastDriftWavePredictionScorecardsSignatureTests(
    unittest.TestCase
):
    def test_public_signature_appends_three_parameters(self):
        parameters = list(
            inspect.signature(
                BranchStore.lifetime_churn_forecast_drift_wave_prediction_scorecards
            ).parameters.values()
        )
        names = [p.name for p in parameters]
        backtests_names = list(
            inspect.signature(
                BranchStore.lifetime_churn_forecast_drift_wave_prediction_backtests
            ).parameters
        )
        self.assertEqual(
            names,
            backtests_names[:-1]
            + [
                "min_evaluated",
                "min_cutoffs",
                "prediction_scorecard_limit",
                "token",
            ],
        )
        self.assertEqual(
            [p.default for p in parameters[:-1]],
            [inspect.Parameter.empty] * (len(parameters) - 1),
        )
        self.assertIsNone(parameters[-1].default)

    def test_result_shape_and_key_order(self):
        result = build_scorecards(synthetic_records(), 0, 1, 50)
        self.assertEqual(list(result), ["scorecards", "totals"])
        self.assertIsInstance(result["scorecards"], tuple)
        for scorecard in result["scorecards"]:
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


class LifetimeChurnForecastDriftWavePredictionScorecardsResultTests(
    unittest.TestCase
):
    def test_aggregation_rates_and_first_failures(self):
        result = build_scorecards(synthetic_records(), 0, 1, 50)
        scorecards = {
            scorecard["identity"]: scorecard
            for scorecard in result["scorecards"]
        }
        self.assertEqual(
            scorecards["A"],
            {
                "identity": "A",
                "cutoffs": 3,
                "hit": 3,
                "missed": 1,
                "unresolved": 3,
                "unexpected": 0,
                "evaluated": 4,
                "hit_rate": (3, 4),
                "first_failure": 1,
            },
        )
        self.assertEqual(
            scorecards["B"],
            {
                "identity": "B",
                "cutoffs": 2,
                "hit": 1,
                "missed": 2,
                "unresolved": 0,
                "unexpected": 1,
                "evaluated": 3,
                "hit_rate": (1, 3),
                "first_failure": 0,
            },
        )
        # Unresolved-only records never evaluate and never fail.
        self.assertEqual(
            scorecards["C"],
            {
                "identity": "C",
                "cutoffs": 1,
                "hit": 0,
                "missed": 0,
                "unresolved": 3,
                "unexpected": 0,
                "evaluated": 0,
                "hit_rate": (0, 0),
                "first_failure": None,
            },
        )
        self.assertEqual(
            scorecards["D"],
            {
                "identity": "D",
                "cutoffs": 2,
                "hit": 4,
                "missed": 0,
                "unresolved": 0,
                "unexpected": 0,
                "evaluated": 4,
                "hit_rate": (1, 1),
                "first_failure": None,
            },
        )

    def test_hit_rate_is_reduced(self):
        result = build_scorecards(
            (record(0, "A", 2, 2, 0, 0),), 0, 1, 50
        )
        self.assertEqual(result["scorecards"][0]["hit_rate"], (1, 2))

    def test_first_failure_counts_unexpected_but_not_unresolved(self):
        result = build_scorecards(
            (
                record(2, "A", 0, 0, 1, 0),
                record(0, "A", 0, 0, 0, 2),
                record(1, "A", 0, 1, 0, 0),
            ),
            0,
            1,
            50,
        )
        # Cutoff 0 already fails on unexpected appearances alone.
        self.assertEqual(result["scorecards"][0]["first_failure"], 0)

    def test_scorecards_sort_by_rate_unexpected_missed_identity(self):
        result = build_scorecards(tie_records(), 0, 1, 50)
        self.assertEqual(
            [s["identity"] for s in result["scorecards"]],
            ["c", "b", "e", "a", "d"],
        )

    def test_min_evaluated_and_min_cutoffs_filter(self):
        records = synthetic_records()
        # C never evaluates; the rest stay.
        result = build_scorecards(records, 1, 1, 50)
        self.assertEqual(
            [s["identity"] for s in result["scorecards"]], ["D", "A", "B"]
        )
        # Only A reaches three distinct cutoffs.
        result = build_scorecards(records, 0, 3, 50)
        self.assertEqual(
            [s["identity"] for s in result["scorecards"]], ["A"]
        )
        # A and D evaluate at least four times over two cutoffs.
        result = build_scorecards(records, 4, 2, 50)
        self.assertEqual(
            [s["identity"] for s in result["scorecards"]], ["D", "A"]
        )

    def test_totals_sum_kept_scorecards_and_union_cutoffs(self):
        result = build_scorecards(synthetic_records(), 0, 1, 50)
        self.assertEqual(
            result["totals"],
            {
                "identities": 4,
                "cutoffs": 3,
                "hit": 8,
                "missed": 3,
                "unresolved": 6,
                "unexpected": 1,
                "evaluated": 11,
            },
        )
        # The cutoff union shrinks with the kept set.
        result = build_scorecards(synthetic_records(), 0, 2, 50)
        self.assertEqual(result["totals"]["identities"], 3)
        self.assertEqual(result["totals"]["cutoffs"], 3)
        result = build_scorecards(synthetic_records(), 4, 1, 50)
        self.assertEqual(
            result["totals"],
            {
                "identities": 2,
                "cutoffs": 3,
                "hit": 7,
                "missed": 1,
                "unresolved": 3,
                "unexpected": 0,
                "evaluated": 8,
            },
        )

    def test_no_qualifying_identity_returns_empty_and_zero_totals(self):
        result = build_scorecards(synthetic_records(), 99, 1, 50)
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
        result = build_scorecards((), 0, 1, 50)
        self.assertEqual(result["scorecards"], ())
        self.assertEqual(result["totals"]["identities"], 0)

    def test_scorecard_limit_bounds_the_kept_set_without_truncating(self):
        result = build_scorecards(synthetic_records(), 0, 1, 4)
        self.assertEqual(len(result["scorecards"]), 4)
        with self.assertRaises(ValueError) as caught:
            build_scorecards(synthetic_records(), 0, 1, 3)
        self.assertIn("scorecard limit exceeded", str(caught.exception))

    def test_every_object_is_fresh(self):
        records = synthetic_records()
        result = build_scorecards(records, 0, 1, 50)
        again = build_scorecards(records, 0, 1, 50)
        self.assertEqual(result, again)
        for scorecard in result["scorecards"]:
            scorecard["hit"] = 99
            scorecard["hit_rate"] = (9, 9)
        result["totals"]["hit"] = 99
        self.assertEqual(build_scorecards(records, 0, 1, 50), again)


class LifetimeChurnForecastDriftWavePredictionScorecardsValidationTests(
    unittest.TestCase
):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        return scorecards_call(self.store, self.args, **overrides)

    def extended(self, **overrides):
        merged = dict(
            LONG_ARGS,
            split_cutoffs=(3, 4, 5),
            wave_cutoffs=(0,),
            match_tolerance=1,
            prediction_backtest_limit=50,
            min_evaluated=0,
            min_cutoffs=1,
            prediction_scorecard_limit=50,
        )
        merged.update(overrides)
        return self.call(**merged)

    def test_min_evaluated_must_be_a_non_bool_non_negative_int(self):
        for bad in (True, "1", 1.5, None, (1,)):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    self.extended(min_evaluated=bad)
        with self.assertRaises(ValueError):
            self.extended(min_evaluated=-1)
        self.extended(min_evaluated=0)

    def test_min_cutoffs_must_be_a_non_bool_positive_int(self):
        for bad in (True, "1", 1.5, None, (1,)):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    self.extended(min_cutoffs=bad)
        with self.assertRaises(ValueError):
            self.extended(min_cutoffs=0)
        with self.assertRaises(ValueError):
            self.extended(min_cutoffs=-2)

    def test_prediction_scorecard_limit_must_be_a_non_bool_positive_int(self):
        for bad in (False, "1", 1.5, None, (1,)):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    self.extended(prediction_scorecard_limit=bad)
        with self.assertRaises(ValueError):
            self.extended(prediction_scorecard_limit=0)
        with self.assertRaises(ValueError):
            self.extended(prediction_scorecard_limit=-3)

    def test_new_parameters_validated_in_signature_order(self):
        # prediction_backtest_limit fails before min_evaluated.
        with self.assertRaises(ValueError):
            self.extended(prediction_backtest_limit=0, min_evaluated="x")
        with self.assertRaises(TypeError):
            self.extended(prediction_backtest_limit="x", min_evaluated="x")
        # min_evaluated fails before min_cutoffs.
        with self.assertRaises(TypeError):
            self.extended(min_evaluated="x", min_cutoffs="x")
        with self.assertRaises(ValueError):
            self.extended(min_evaluated=-1, min_cutoffs=0)
        # min_cutoffs fails before prediction_scorecard_limit.
        with self.assertRaises(TypeError):
            self.extended(min_cutoffs="x", prediction_scorecard_limit="x")
        with self.assertRaises(ValueError):
            self.extended(min_cutoffs=0, prediction_scorecard_limit=0)
        # All three pass before the split membership check fires.
        with self.assertRaises(ValueError):
            self.extended(
                split_cutoffs=(99,),
                min_evaluated=0,
                min_cutoffs=1,
                prediction_scorecard_limit=1,
            )

    def test_all_parameter_checks_precede_state_lookups(self):
        with self.assertRaises(TypeError):
            self.call(reference="ghost", cutoffs=(0, 1), min_evaluated="x")
        with self.assertRaises(ValueError):
            self.call(reference="ghost", cutoffs=(0, 1), min_evaluated=-1)
        with self.assertRaises(TypeError):
            self.call(reference="ghost", cutoffs=(0, 1), min_cutoffs="x")
        with self.assertRaises(ValueError):
            self.call(reference="ghost", cutoffs=(0, 1), min_cutoffs=0)
        with self.assertRaises(TypeError):
            self.call(
                reference="ghost",
                cutoffs=(0, 1),
                prediction_scorecard_limit="x",
            )
        with self.assertRaises(ValueError):
            self.call(
                reference="ghost",
                cutoffs=(0, 1),
                prediction_scorecard_limit=0,
            )


class LifetimeChurnForecastDriftWavePredictionScorecardsEmptyRunTests(
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

    def test_empty_wave_cutoffs_returns_empty_scorecards_and_zero_totals(self):
        result = scorecards_call(
            self.store,
            self.args,
            **dict(LONG_ARGS, split_cutoffs=(3, 4, 5), wave_cutoffs=()),
        )
        self.assertEqual(result["scorecards"], ())
        self.assertEqual(result["totals"], self.zero_totals())

    def test_no_waves_returns_empty_scorecards_and_zero_totals(self):
        result = scorecards_call(
            self.store,
            self.args,
            **dict(
                LONG_ARGS, cutoffs=(), split_cutoffs=(), wave_cutoffs=(0,)
            ),
        )
        self.assertEqual(result["scorecards"], ())
        self.assertEqual(result["totals"], self.zero_totals())

    def test_no_qualifying_identity_returns_empty_scorecards(self):
        # The diamond store yields a single drift wave, so no identity
        # reaches three training appearances at cutoff 0.
        result = scorecards_call(
            self.store,
            self.args,
            **dict(LONG_ARGS, split_cutoffs=(3, 4, 5), wave_cutoffs=(0,)),
        )
        self.assertEqual(result["scorecards"], ())
        self.assertEqual(result["totals"], self.zero_totals())

    def test_cutoff_beyond_the_last_wave_raises(self):
        with self.assertRaises(ValueError) as caught:
            scorecards_call(
                self.store,
                self.args,
                **dict(LONG_ARGS, split_cutoffs=(3, 4, 5), wave_cutoffs=(1,)),
            )
        self.assertIn("out of range", str(caught.exception))


class LifetimeChurnForecastDriftWavePredictionScorecardsIsolationTests(
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
        )

    def test_success_and_failure_change_nothing(self):
        store, args = diamond_store()
        heads = dict(store._heads)
        appends = copy.deepcopy(store._appends)
        merges = copy.deepcopy(store._merges)
        audit = store.audit_log()

        scorecards_call(store, args, **self.call_args())
        failures = [
            dict(min_evaluated="x"),
            dict(min_evaluated=True),
            dict(min_evaluated=-1),
            dict(min_cutoffs="x"),
            dict(min_cutoffs=False),
            dict(min_cutoffs=0),
            dict(prediction_scorecard_limit="x"),
            dict(prediction_scorecard_limit=True),
            dict(prediction_scorecard_limit=0),
            dict(wave_cutoffs="x"),
            dict(wave_cutoffs=(9,)),
            dict(match_tolerance=-1),
            dict(prediction_backtest_limit=0),
            dict(drift_limit=2),
            dict(scan_limit=1),
            dict(reference="ghost"),
        ]
        for overrides in failures:
            with self.subTest(overrides=overrides):
                merged = self.call_args()
                merged.update(overrides)
                with self.assertRaises(Exception):
                    scorecards_call(store, args, **merged)

        self.assertEqual(dict(store._heads), heads)
        self.assertEqual(copy.deepcopy(store._appends), appends)
        self.assertEqual(copy.deepcopy(store._merges), merges)
        self.assertEqual(store.audit_log(), audit)


class LifetimeChurnForecastDriftWavePredictionScorecardsTokenTests(
    unittest.TestCase
):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, token=None, **overrides):
        return scorecards_call(self.store, self.args, token=token, **overrides)

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
        with self.assertRaises(ValueError):
            self.call(
                token=token,
                **dict(self.call_args(), prediction_scorecard_limit=0),
            )
        # Parameter validation runs before the token is touched.
        with self.assertRaises(TypeError):
            self.call(token=token, **dict(self.call_args(), min_evaluated="x"))
        with self.assertRaises(ValueError):
            self.call(token=token, **dict(self.call_args(), min_cutoffs=0))
        # The second allowed read succeeds; the token then expires.
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
