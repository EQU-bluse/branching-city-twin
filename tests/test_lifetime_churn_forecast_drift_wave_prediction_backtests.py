import copy
import inspect
import unittest

from city_twin.branches import BranchStore

from tests.test_lifetime_churn import diamond_store
from tests.test_lifetime_churn_forecast_drift_scan import LONG_ARGS
from tests.test_lifetime_churn_forecast_drift_wave_predictions import (
    predictions_kwargs,
)


def backtests_kwargs(store_args, **overrides) -> dict:
    wave_cutoffs = overrides.pop("wave_cutoffs", (0,))
    match_tolerance = overrides.pop("match_tolerance", 1)
    prediction_backtest_limit = overrides.pop("prediction_backtest_limit", 50)
    kwargs = predictions_kwargs(store_args, **overrides)
    kwargs["wave_cutoffs"] = wave_cutoffs
    kwargs["match_tolerance"] = match_tolerance
    kwargs["prediction_backtest_limit"] = prediction_backtest_limit
    return kwargs


def backtests_call(store, store_args, **overrides):
    return store.lifetime_churn_forecast_drift_wave_prediction_backtests(
        **backtests_kwargs(store_args, **overrides)
    )


def wave(*identity_groups):
    """One one-point wave whose point carries the given identities."""
    return {
        "start_split": 0,
        "end_split": 0,
        "length": 1,
        "peak_identities": 1,
        "points": tuple(
            {
                "split_cutoff": 0,
                "active_count": len(group),
                "identities": tuple(group),
            }
            for group in identity_groups
        ),
    }


def synthetic_waves():
    """Seven waves; A and C train on 0/2/4, B only ever appears twice.

    A reappears exactly on wave 6, C one wave early on wave 5 and B
    never reaches three training waves at any cutoff.
    """
    return (
        wave(("A",), ("C",)),
        wave(("B",),),
        wave(("A",), ("C",)),
        wave(("B",),),
        wave(("A",), ("C",)),
        wave(("B",), ("C",)),
        wave(("A",),),
    )


def build(waves, cutoffs, jitter=0, horizon=4, tolerance=0,
          result_limit=50, backtest_limit=50):
    return BranchStore._build_lifetime_churn_forecast_drift_wave_prediction_backtests_result(
        waves, cutoffs, jitter, horizon, tolerance, result_limit,
        backtest_limit,
    )


class LifetimeChurnForecastDriftWavePredictionBacktestsSignatureTests(
    unittest.TestCase
):
    def test_public_signature_appends_three_parameters(self):
        parameters = list(
            inspect.signature(
                BranchStore.lifetime_churn_forecast_drift_wave_prediction_backtests
            ).parameters.values()
        )
        names = [p.name for p in parameters]
        prediction_names = list(
            inspect.signature(
                BranchStore.lifetime_churn_forecast_drift_wave_predictions
            ).parameters
        )
        self.assertEqual(
            names,
            prediction_names[:-1]
            + [
                "wave_cutoffs",
                "match_tolerance",
                "prediction_backtest_limit",
                "token",
            ],
        )
        self.assertEqual(
            [p.default for p in parameters[:-1]],
            [inspect.Parameter.empty] * (len(parameters) - 1),
        )
        self.assertIsNone(parameters[-1].default)

    def test_result_shape_and_key_order(self):
        result = build(synthetic_waves(), (4,))
        self.assertEqual(list(result), ["backtests", "totals"])
        self.assertIsInstance(result["backtests"], tuple)
        for record in result["backtests"]:
            self.assertEqual(
                list(record),
                [
                    "cutoff",
                    "identity",
                    "period",
                    "jitter",
                    "last_wave",
                    "targets",
                    "totals",
                ],
            )
            self.assertIsInstance(record["targets"], tuple)
            for target in record["targets"]:
                self.assertEqual(list(target), ["wave", "actual", "status"])
            self.assertEqual(
                list(record["totals"]),
                ["hit", "missed", "unresolved", "unexpected"],
            )
        self.assertEqual(
            list(result["totals"]),
            ["cutoffs", "records", "hit", "missed", "unresolved",
             "unexpected"],
        )


class LifetimeChurnForecastDriftWavePredictionBacktestsResultTests(
    unittest.TestCase
):
    def test_training_uses_only_waves_up_to_the_cutoff(self):
        # At cutoff 3 no identity reaches three training waves.
        result = build(synthetic_waves(), (3,))
        self.assertEqual(result["backtests"], ())
        self.assertEqual(
            result["totals"],
            {
                "cutoffs": 0,
                "records": 0,
                "hit": 0,
                "missed": 0,
                "unresolved": 0,
                "unexpected": 0,
            },
        )

    def test_targets_match_observed_waves_within_tolerance(self):
        result = build(synthetic_waves(), (4,), tolerance=0)
        records = {record["identity"]: record for record in result["backtests"]}
        self.assertEqual(list(records), ["A", "C"])
        a = records["A"]
        self.assertEqual(
            (a["cutoff"], a["period"], a["jitter"], a["last_wave"]),
            (4, 2, 0, 4),
        )
        self.assertEqual(
            [(t["wave"], t["actual"], t["status"]) for t in a["targets"]],
            [(6, 6, "hit"), (8, None, "unresolved")],
        )
        self.assertEqual(
            a["totals"],
            {"hit": 1, "missed": 0, "unresolved": 1, "unexpected": 0},
        )
        c = records["C"]
        # C's observed wave 5 is past the zero tolerance, so target 6
        # is missed (still inside the observation end) and wave 5 is
        # left unexpected.
        self.assertEqual(
            [(t["wave"], t["actual"], t["status"]) for t in c["targets"]],
            [(6, None, "missed"), (8, None, "unresolved")],
        )
        self.assertEqual(
            c["totals"],
            {"hit": 0, "missed": 1, "unresolved": 1, "unexpected": 1},
        )
        self.assertEqual(
            result["totals"],
            {
                "cutoffs": 1,
                "records": 2,
                "hit": 1,
                "missed": 1,
                "unresolved": 2,
                "unexpected": 1,
            },
        )

    def test_tolerance_turns_near_waves_into_hits(self):
        result = build(synthetic_waves(), (4,), tolerance=1)
        c = next(
            record
            for record in result["backtests"]
            if record["identity"] == "C"
        )
        self.assertEqual(
            [(t["wave"], t["actual"], t["status"]) for t in c["targets"]],
            [(6, 5, "hit"), (8, None, "unresolved")],
        )
        self.assertEqual(c["totals"]["unexpected"], 0)

    def test_equal_distance_takes_the_earlier_wave_and_never_reuses(self):
        waves = (
            wave(("A",),),
            wave(),
            wave(("A",),),
            wave(),
            wave(("A",),),
            wave(("A",),),
            wave(),
            wave(("A",),),
        )
        # Targets 6 and 8; observed 5 and 7. Target 6 is equidistant
        # from 5 and 7, so the earlier 5 matches and 7 is left for 8.
        result = build(waves, (4,), tolerance=1)
        targets = result["backtests"][0]["targets"]
        self.assertEqual(
            [(t["wave"], t["actual"], t["status"]) for t in targets],
            [(6, 5, "hit"), (8, 7, "hit")],
        )

    def test_jitter_over_max_wave_jitter_is_excluded(self):
        waves = (
            wave(("A",),),
            wave(("A",),),
            wave(),
            wave(),
            wave(("A",),),
            wave(),
            wave(),
        )
        # Intervals 1 and 3 give jitter 2 and the even-count median 1.
        self.assertEqual(build(waves, (4,), jitter=1)["backtests"], ())
        result = build(waves, (4,), jitter=2)
        record = result["backtests"][0]
        self.assertEqual((record["period"], record["jitter"]), (1, 2))

    def test_records_follow_the_periodicity_order_per_cutoff(self):
        waves = (
            wave(("X",), ("Y",)),
            wave(("Z",),),
            wave(("X",), ("Y",), ("Z",)),
            wave(),
            wave(("X", "Y"), ("Y",), ("Z",)),
            wave(),
            wave(),
        )
        recurrences = (
            BranchStore._build_lifetime_churn_forecast_drift_wave_recurrences_result(
                waves[:5], 1, 50
            )
        )
        periodicities = (
            BranchStore._build_lifetime_churn_forecast_drift_wave_periodicities_result(
                recurrences["recurrences"], 1, 50
            )
        )
        expected = [p["identity"] for p in periodicities["periodicities"]]
        result = build(waves, (4,), jitter=1, horizon=3)
        self.assertEqual(
            [record["identity"] for record in result["backtests"]], expected
        )
        for record in result["backtests"]:
            periodicity = next(
                p
                for p in periodicities["periodicities"]
                if p["identity"] == record["identity"]
            )
            self.assertEqual(
                (record["period"], record["jitter"], record["last_wave"]),
                (
                    periodicity["period"],
                    periodicity["jitter"],
                    periodicity["last_wave"],
                ),
            )

    def test_records_sort_by_cutoff_then_training_order(self):
        # At cutoff 5, C's training waves 0/2/4/5 gain jitter 1 and
        # drop out while B's 1/3/5 reach three steady waves.
        result = build(synthetic_waves(), (4, 5), tolerance=1)
        self.assertEqual(
            [(record["cutoff"], record["identity"])
             for record in result["backtests"]],
            [(4, "A"), (4, "C"), (5, "A"), (5, "B")],
        )
        self.assertEqual(result["totals"]["cutoffs"], 2)
        self.assertEqual(result["totals"]["records"], 4)

    def test_limits_bound_without_truncating(self):
        with self.assertRaises(ValueError) as caught:
            build(synthetic_waves(), (4,), result_limit=1)
        self.assertIn("prediction result limit", str(caught.exception))
        with self.assertRaises(ValueError) as caught:
            build(synthetic_waves(), (4, 5), backtest_limit=3)
        self.assertIn("prediction backtest limit", str(caught.exception))
        result = build(synthetic_waves(), (4, 5), backtest_limit=4)
        self.assertEqual(len(result["backtests"]), 4)

    def test_empty_cutoffs_and_empty_waves_return_zero_totals(self):
        empty = {
            "cutoffs": 0,
            "records": 0,
            "hit": 0,
            "missed": 0,
            "unresolved": 0,
            "unexpected": 0,
        }
        result = build(synthetic_waves(), ())
        self.assertEqual(result, {"backtests": (), "totals": empty})
        result = build((), (1, 2))
        self.assertEqual(result, {"backtests": (), "totals": empty})

    def test_cutoff_beyond_the_last_wave_raises(self):
        with self.assertRaises(ValueError) as caught:
            build(synthetic_waves(), (4, 7))
        self.assertIn("out of range", str(caught.exception))

    def test_every_object_is_fresh(self):
        result = build(synthetic_waves(), (4,), tolerance=1)
        again = build(synthetic_waves(), (4,), tolerance=1)
        seen_ids = []

        def collect(value):
            if isinstance(value, dict):
                seen_ids.append(id(value))
                for item in value.values():
                    collect(item)
            elif isinstance(value, (tuple, list)):
                for item in value:
                    collect(item)

        collect(result)
        collect(again)
        self.assertEqual(len(seen_ids), len(set(seen_ids)))
        result["backtests"][0]["targets"][0]["wave"] = 999
        self.assertEqual(
            again["backtests"][0]["targets"][0]["wave"], 6
        )


class LifetimeChurnForecastDriftWavePredictionBacktestsValidationTests(
    unittest.TestCase
):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        return backtests_call(self.store, self.args, **overrides)

    def extended(self, **overrides):
        overrides.setdefault("split_cutoffs", (4,))
        merged = dict(LONG_ARGS)
        merged.update(overrides)
        return self.call(**merged)

    def test_wave_cutoffs_must_be_strictly_increasing_int_tuple(self):
        for bad in ([0], "0", 0, {0: 1}):
            with self.subTest(value=bad):
                with self.assertRaises(TypeError):
                    self.extended(wave_cutoffs=bad)
        for bad in ((True,), (0.5,), ("1",), (None,)):
            with self.subTest(value=bad):
                with self.assertRaises(TypeError):
                    self.extended(wave_cutoffs=bad)
        for bad in ((-1,), (1, 0), (2, 2), (0, -3)):
            with self.subTest(value=bad):
                with self.assertRaises(ValueError):
                    self.extended(wave_cutoffs=bad)
        self.assertEqual(
            self.extended(wave_cutoffs=())["backtests"], ()
        )

    def test_match_tolerance_must_be_non_bool_non_negative_int(self):
        for bad in (True, False, 1.0, "1", None, (1,)):
            with self.subTest(value=bad):
                with self.assertRaises(TypeError):
                    self.extended(match_tolerance=bad)
        for bad in (-1, -7):
            with self.subTest(value=bad):
                with self.assertRaises(ValueError):
                    self.extended(match_tolerance=bad)
        self.assertEqual(self.extended(match_tolerance=0)["backtests"], ())

    def test_prediction_backtest_limit_must_be_non_bool_positive_int(self):
        for bad in (True, False, 1.0, "1", None, (1,)):
            with self.subTest(value=bad):
                with self.assertRaises(TypeError):
                    self.extended(prediction_backtest_limit=bad)
        for bad in (0, -3):
            with self.subTest(value=bad):
                with self.assertRaises(ValueError):
                    self.extended(prediction_backtest_limit=bad)

    def test_new_parameters_validated_in_signature_order(self):
        # prediction_result_limit fails before wave_cutoffs.
        with self.assertRaises(ValueError):
            self.extended(prediction_result_limit=0, wave_cutoffs=[0])
        with self.assertRaises(TypeError):
            self.extended(prediction_result_limit="x", wave_cutoffs=[0])
        # wave_cutoffs fails before match_tolerance.
        with self.assertRaises(TypeError):
            self.extended(wave_cutoffs=[0], match_tolerance="x")
        with self.assertRaises(ValueError):
            self.extended(wave_cutoffs=(-1,), match_tolerance=-1)
        # match_tolerance fails before prediction_backtest_limit.
        with self.assertRaises(TypeError):
            self.extended(match_tolerance="x", prediction_backtest_limit=0)
        with self.assertRaises(ValueError):
            self.extended(match_tolerance=-1, prediction_backtest_limit=0)
        # All three pass before the split membership check fires.
        with self.assertRaises(ValueError):
            self.extended(split_cutoffs=(99,))

    def test_all_parameter_checks_precede_state_lookups(self):
        with self.assertRaises(TypeError):
            self.call(reference="ghost", wave_cutoffs=[0])
        with self.assertRaises(ValueError):
            self.call(reference="ghost", wave_cutoffs=(-1,))
        with self.assertRaises(TypeError):
            self.call(reference="ghost", match_tolerance="x")
        with self.assertRaises(ValueError):
            self.call(reference="ghost", match_tolerance=-1)
        with self.assertRaises(TypeError):
            self.call(reference="ghost", prediction_backtest_limit="x")
        with self.assertRaises(ValueError):
            self.call(reference="ghost", prediction_backtest_limit=0)


class LifetimeChurnForecastDriftWavePredictionBacktestsEmptyRunTests(
    unittest.TestCase
):
    def setUp(self):
        self.store, self.args = diamond_store()

    def test_empty_cutoffs_and_short_wave_set_return_zero_totals(self):
        empty = {
            "cutoffs": 0,
            "records": 0,
            "hit": 0,
            "missed": 0,
            "unresolved": 0,
            "unexpected": 0,
        }
        result = backtests_call(
            self.store,
            self.args,
            **dict(LONG_ARGS, split_cutoffs=(3, 4, 5), wave_cutoffs=()),
        )
        self.assertEqual(result, {"backtests": (), "totals": empty})
        # The diamond store yields a single wave, so no cutoff can
        # train three waves and even an in-range cutoff stays empty.
        result = backtests_call(
            self.store,
            self.args,
            **dict(LONG_ARGS, split_cutoffs=(3, 4, 5), wave_cutoffs=(0,)),
        )
        self.assertEqual(result, {"backtests": (), "totals": empty})

    def test_cutoff_beyond_the_last_wave_raises(self):
        with self.assertRaises(ValueError) as caught:
            backtests_call(
                self.store,
                self.args,
                **dict(LONG_ARGS, split_cutoffs=(3, 4, 5),
                       wave_cutoffs=(0, 1)),
            )
        self.assertIn("out of range", str(caught.exception))


class LifetimeChurnForecastDriftWavePredictionBacktestsIsolationTests(
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
        )

    def test_success_and_failure_change_nothing(self):
        store, args = diamond_store()
        heads = dict(store._heads)
        appends = copy.deepcopy(store._appends)
        merges = copy.deepcopy(store._merges)
        audit = store.audit_log()

        backtests_call(store, args, **self.call_args())
        failures = [
            dict(wave_cutoffs=[0]),
            dict(wave_cutoffs=(True,)),
            dict(wave_cutoffs=(-1,)),
            dict(wave_cutoffs=(0, 0)),
            dict(wave_cutoffs=(9,)),
            dict(match_tolerance="x"),
            dict(match_tolerance=True),
            dict(match_tolerance=-1),
            dict(prediction_backtest_limit=0),
            dict(prediction_backtest_limit="x"),
            dict(prediction_backtest_limit=False),
            dict(prediction_result_limit=0),
            dict(drift_limit=2),
            dict(scan_limit=1),
            dict(reference="ghost"),
        ]
        for overrides in failures:
            with self.subTest(overrides=overrides):
                merged = self.call_args()
                merged.update(overrides)
                with self.assertRaises(Exception):
                    backtests_call(store, args, **merged)

        self.assertEqual(dict(store._heads), heads)
        self.assertEqual(copy.deepcopy(store._appends), appends)
        self.assertEqual(copy.deepcopy(store._merges), merges)
        self.assertEqual(store.audit_log(), audit)


class LifetimeChurnForecastDriftWavePredictionBacktestsTokenTests(
    unittest.TestCase
):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, token=None, **overrides):
        return backtests_call(
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
        )

    def test_success_consumes_one_read_and_failure_refunds(self):
        token = self.store.create_snapshot(2)
        self.call(token=token, **self.call_args())
        # Stage failures still refund.
        with self.assertRaises(ValueError):
            self.call(token=token, **dict(self.call_args(), drift_limit=2))
        with self.assertRaises(ValueError):
            self.call(token=token, **dict(self.call_args(), scan_limit=1))
        # Parameter validation runs before the token is touched.
        with self.assertRaises(TypeError):
            self.call(token=token, **dict(self.call_args(), wave_cutoffs="x"))
        with self.assertRaises(ValueError):
            self.call(
                token=token,
                **dict(self.call_args(), prediction_backtest_limit=0),
            )
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
