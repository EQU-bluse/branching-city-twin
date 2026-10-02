import copy
import inspect
import unittest

from city_twin.branches import BranchStore

from tests.test_lifetime_churn import diamond_store
from tests.test_lifetime_churn_forecast_drift_scan import LONG_ARGS
from tests.test_lifetime_churn_forecast_drift_waves import waves_kwargs


def periodicities_kwargs(store_args, **overrides) -> dict:
    min_wave_occurrences = overrides.pop("min_wave_occurrences", 1)
    recurrence_result_limit = overrides.pop("recurrence_result_limit", 50)
    max_wave_jitter = overrides.pop("max_wave_jitter", 10)
    periodicity_result_limit = overrides.pop("periodicity_result_limit", 50)
    kwargs = waves_kwargs(store_args, **overrides)
    kwargs["min_wave_occurrences"] = min_wave_occurrences
    kwargs["recurrence_result_limit"] = recurrence_result_limit
    kwargs["max_wave_jitter"] = max_wave_jitter
    kwargs["periodicity_result_limit"] = periodicity_result_limit
    return kwargs


def predictions_kwargs(store_args, **overrides) -> dict:
    wave_horizon = overrides.pop("wave_horizon", 10)
    prediction_result_limit = overrides.pop("prediction_result_limit", 50)
    kwargs = periodicities_kwargs(store_args, **overrides)
    kwargs["wave_horizon"] = wave_horizon
    kwargs["prediction_result_limit"] = prediction_result_limit
    return kwargs


def predictions_call(store, store_args, **overrides):
    return store.lifetime_churn_forecast_drift_wave_predictions(
        **predictions_kwargs(store_args, **overrides)
    )


def periodicity_record(identity, period, jitter, last_wave):
    """Build one periodicity-shaped record for the prediction builder."""
    return {
        "identity": identity,
        "waves": 3,
        "first_wave": last_wave - 2 * period,
        "last_wave": last_wave,
        "span": 2 * period + 1,
        "points": 3,
        "peak_active": 1,
        "appearances": (),
        "intervals": (period, period),
        "period": period,
        "jitter": jitter,
    }


class LifetimeChurnForecastDriftWavePredictionsSignatureTests(unittest.TestCase):
    def test_public_signature_appends_horizon_and_prediction_limit(self):
        parameters = list(
            inspect.signature(
                BranchStore
                .lifetime_churn_forecast_drift_wave_predictions
            ).parameters.values()
        )
        self.assertEqual(
            [p.name for p in parameters],
            [
                "self",
                "reference",
                "series",
                "base_scenario",
                "axis",
                "values",
                "min_size",
                "max_size",
                "required",
                "exclusive_pairs",
                "limit",
                "windows",
                "causes",
                "direction",
                "depth",
                "node_limit",
                "change_limit",
                "lifetime_limit",
                "diff_limit",
                "window_limit",
                "total_diff_limit",
                "churn_limit",
                "streak_limit",
                "min_identities",
                "wave_limit",
                "min_waves",
                "recurrence_limit",
                "max_jitter",
                "periodicity_limit",
                "horizon",
                "forecast_limit",
                "cutoffs",
                "backtest_limit",
                "min_resolved",
                "scorecard_limit",
                "split_cutoffs",
                "min_rate_drop",
                "drift_limit",
                "scan_limit",
                "min_splits",
                "streak_result_limit",
                "min_active_identities",
                "wave_result_limit",
                "min_wave_occurrences",
                "recurrence_result_limit",
                "max_wave_jitter",
                "periodicity_result_limit",
                "wave_horizon",
                "prediction_result_limit",
                "token",
            ],
        )
        self.assertEqual(
            [p.default for p in parameters[:-1]],
            [inspect.Parameter.empty] * (len(parameters) - 1),
        )
        self.assertIsNone(parameters[-1].default)

    def test_result_shape_and_key_order(self):
        records = (periodicity_record("A", 2, 0, 4),)
        result = (
            BranchStore
            ._build_lifetime_churn_forecast_drift_wave_predictions_result(
                records, 4, 50
            )
        )
        self.assertIsInstance(result, dict)
        self.assertEqual(list(result), ["predictions", "totals"])
        self.assertIsInstance(result["predictions"], tuple)
        self.assertEqual(
            list(result["totals"]),
            ["predictions", "occurrences", "first_wave", "last_wave"],
        )
        record = result["predictions"][0]
        self.assertEqual(
            list(record),
            ["identity", "period", "jitter", "last_wave", "next_waves"],
        )
        self.assertIsInstance(record["next_waves"], tuple)
        for next_wave in record["next_waves"]:
            self.assertEqual(list(next_wave), ["wave", "offset"])


class LifetimeChurnForecastDriftWavePredictionsResultTests(unittest.TestCase):
    def build(self, records, wave_horizon=10, prediction_result_limit=50):
        return (
            BranchStore
            ._build_lifetime_churn_forecast_drift_wave_predictions_result(
                records, wave_horizon, prediction_result_limit
            )
        )

    def test_waves_step_by_period_from_last_wave(self):
        # last_wave 4, period 2 -> 6, 8 inside a horizon of 5, while 10
        # is six waves away and must be excluded.
        result = self.build(
            (periodicity_record("A", 2, 0, 4),), wave_horizon=5
        )
        record = result["predictions"][0]
        self.assertEqual(
            record["next_waves"],
            (
                {"wave": 6, "offset": 2},
                {"wave": 8, "offset": 4},
            ),
        )
        self.assertEqual(
            (record["identity"], record["period"], record["jitter"],
             record["last_wave"]),
            ("A", 2, 0, 4),
        )
        self.assertEqual(
            result["totals"],
            {"predictions": 1, "occurrences": 2,
             "first_wave": 6, "last_wave": 8},
        )

    def test_horizon_is_inclusive_at_the_boundary(self):
        result = self.build(
            (periodicity_record("A", 3, 1, 0),), wave_horizon=3
        )
        self.assertEqual(
            result["predictions"][0]["next_waves"],
            ({"wave": 3, "offset": 3},),
        )

    def test_horizon_shorter_than_period_omits_the_identity(self):
        result = self.build(
            (periodicity_record("A", 4, 0, 0),), wave_horizon=3
        )
        self.assertEqual(result["predictions"], ())
        self.assertEqual(
            result["totals"],
            {"predictions": 0, "occurrences": 0,
             "first_wave": 0, "last_wave": 0},
        )

    def test_totals_span_every_kept_record(self):
        # A: last 2, period 2, horizon 5 -> waves 4, 6 (offset 2, 4).
        # B: last 3, period 1, horizon 5 -> waves 4..8 (offsets 1..5).
        # C: last 0, period 9, horizon 5 -> no future wave, omitted.
        records = (
            periodicity_record("A", 2, 0, 2),
            periodicity_record("B", 1, 0, 3),
            periodicity_record("C", 9, 0, 0),
        )
        result = self.build(records, wave_horizon=5)
        self.assertEqual(
            [[w["wave"] for w in r["next_waves"]]
             for r in result["predictions"]],
            [[4, 6], [4, 5, 6, 7, 8]],
        )
        self.assertEqual(
            result["totals"],
            {"predictions": 2, "occurrences": 7,
             "first_wave": 4, "last_wave": 8},
        )

    def test_record_order_is_the_periodicity_order(self):
        # A projects one wave, B projects three: the order must stay
        # A, B instead of being reordered by the prediction count.
        records = (
            periodicity_record("A", 5, 0, 0),
            periodicity_record("B", 1, 2, 0),
        )
        result = self.build(records, wave_horizon=5)
        self.assertEqual(
            [r["identity"] for r in result["predictions"]], ["A", "B"]
        )
        self.assertEqual(
            [len(r["next_waves"]) for r in result["predictions"]], [1, 5]
        )

    def test_offsets_are_always_wave_minus_last_wave(self):
        result = self.build(
            (periodicity_record("A", 2, 0, 7),), wave_horizon=6
        )
        for next_wave in result["predictions"][0]["next_waves"]:
            self.assertEqual(
                next_wave["offset"], next_wave["wave"] - 7
            )

    def test_next_waves_are_ascending(self):
        result = self.build(
            (periodicity_record("A", 1, 0, 0),), wave_horizon=4
        )
        waves = [w["wave"] for w in result["predictions"][0]["next_waves"]]
        self.assertEqual(waves, [1, 2, 3, 4])

    def test_empty_periodicity_set_yields_zeros(self):
        result = self.build((), wave_horizon=5)
        self.assertEqual(result["predictions"], ())
        self.assertEqual(
            result["totals"],
            {"predictions": 0, "occurrences": 0,
             "first_wave": 0, "last_wave": 0},
        )

    def test_prediction_result_limit_bounds_final_records_without_truncating(
        self,
    ):
        # The third identity projects nothing inside the horizon and so
        # does not count toward the final-record cap.
        records = (
            periodicity_record("A", 1, 0, 0),
            periodicity_record("B", 1, 0, 0),
            periodicity_record("C", 9, 0, 0),
        )
        with self.assertRaises(ValueError) as caught:
            self.build(records, wave_horizon=2, prediction_result_limit=1)
        self.assertIn("prediction result limit", str(caught.exception))
        result = self.build(
            records, wave_horizon=2, prediction_result_limit=2
        )
        self.assertEqual(len(result["predictions"]), 2)

    def test_result_objects_are_fresh_and_unshared(self):
        source = periodicity_record(("W", "y"), 2, 0, 4)
        result = self.build((source,), wave_horizon=4)
        record = result["predictions"][0]
        self.assertEqual(record["identity"], ("W", "y"))
        self.assertIsNot(record["identity"], source["identity"])
        seen_ids = []

        def collect(value):
            if isinstance(value, dict):
                seen_ids.append(id(value))
                for item in value.values():
                    collect(item)
            elif isinstance(value, tuple):
                for item in value:
                    collect(item)

        collect(result)
        self.assertEqual(len(seen_ids), len(set(seen_ids)))
        # Mutating the result never reaches the periodicity record.
        record["next_waves"][0]["wave"] = 99
        self.assertEqual(source["last_wave"], 4)


class LifetimeChurnForecastDriftWavePredictionsValidationTests(
    unittest.TestCase
):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        overrides.setdefault("causes", ("a0",))
        return predictions_call(self.store, self.args, **overrides)

    def extended(self, **overrides):
        overrides.setdefault(
            "split_cutoffs", (3, 4, 5, 6)
        )
        merged = dict(LONG_ARGS)
        merged.update(overrides)
        return self.call(**merged)

    def test_wave_horizon_must_be_a_non_bool_positive_int(self):
        for bad in (True, False, 1.0, "1", None, (1,)):
            with self.subTest(wave_horizon=bad):
                with self.assertRaises(TypeError):
                    self.extended(wave_horizon=bad)
        with self.assertRaises(ValueError):
            self.extended(wave_horizon=0)
        with self.assertRaises(ValueError):
            self.extended(wave_horizon=-3)

    def test_prediction_result_limit_must_be_a_non_bool_positive_int(self):
        for bad in (True, False, 1.0, "1", None, (1,)):
            with self.subTest(prediction_result_limit=bad):
                with self.assertRaises(TypeError):
                    self.extended(prediction_result_limit=bad)
        with self.assertRaises(ValueError):
            self.extended(prediction_result_limit=0)
        with self.assertRaises(ValueError):
            self.extended(prediction_result_limit=-3)

    def test_new_parameters_validated_in_signature_order(self):
        # periodicity_result_limit fails before wave_horizon.
        with self.assertRaises(ValueError):
            self.extended(
                periodicity_result_limit=0, wave_horizon=0
            )
        with self.assertRaises(TypeError):
            self.extended(
                periodicity_result_limit="x", wave_horizon=0
            )
        # wave_horizon fails before prediction_result_limit.
        with self.assertRaises(ValueError):
            self.extended(wave_horizon=0, prediction_result_limit=0)
        with self.assertRaises(TypeError):
            self.extended(wave_horizon="x", prediction_result_limit=0)
        # Once wave_horizon passes, the prediction limit error fires.
        with self.assertRaises(TypeError):
            self.extended(wave_horizon=1, prediction_result_limit="x")
        with self.assertRaises(ValueError):
            self.extended(wave_horizon=1, prediction_result_limit=0)
        # Both still precede the split membership check and every
        # state lookup.
        with self.assertRaises(ValueError):
            self.extended(split_cutoffs=(99,), wave_horizon=0)
        with self.assertRaises(TypeError):
            self.extended(split_cutoffs=(99,), wave_horizon="x")
        with self.assertRaises(ValueError):
            self.call(reference="ghost", wave_horizon=0)
        with self.assertRaises(ValueError):
            self.call(token="whatever", prediction_result_limit=0)

    def test_earlier_stage_caps_still_fire_ahead(self):
        # The LONG batch never yields a periodicity, so its periodicity
        # cap cannot trip; the recurrence cap is validated first.
        with self.assertRaises(ValueError) as caught:
            self.extended(
                recurrence_result_limit=0, wave_horizon=1
            )
        self.assertIn("recurrence_result_limit", str(caught.exception))
        with self.assertRaises(ValueError) as caught:
            self.extended(drift_limit=2, wave_horizon=1)
        self.assertIn("drift limit", str(caught.exception))
        with self.assertRaises(ValueError) as caught:
            self.extended(scan_limit=1, wave_horizon=1)
        self.assertIn("scan limit", str(caught.exception))

    def test_unknown_branch_history_node_and_cause_raise(self):
        with self.assertRaises(KeyError):
            self.call(
                reference="ghost", cutoffs=(0, 1), split_cutoffs=(1,)
            )
        with self.assertRaises(KeyError):
            self.call(
                causes=("zzz",), cutoffs=(0, 1), split_cutoffs=(1,)
            )


class LifetimeChurnForecastDriftWavePredictionsEmptyTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        return predictions_call(self.store, self.args, **overrides)

    def test_single_wave_batch_has_nothing_to_forecast(self):
        result = self.call(**LONG_ARGS, split_cutoffs=(3, 4, 5, 6))
        self.assertEqual(result["predictions"], ())
        self.assertEqual(
            result["totals"],
            {"predictions": 0, "occurrences": 0,
             "first_wave": 0, "last_wave": 0},
        )

    def test_empty_split_cutoffs_return_zeros(self):
        result = self.call(
            **dict(LONG_ARGS, cutoffs=()), split_cutoffs=()
        )
        self.assertEqual(result["predictions"], ())
        self.assertEqual(
            result["totals"],
            {"predictions": 0, "occurrences": 0,
             "first_wave": 0, "last_wave": 0},
        )

    def test_short_horizon_returns_zeros(self):
        result = self.call(
            **LONG_ARGS, split_cutoffs=(3, 4, 5, 6), wave_horizon=1
        )
        # No periodicity exists at all in a one-wave batch, so every
        # horizon behaves identically to the empty case.
        self.assertEqual(result["predictions"], ())
        self.assertEqual(result["totals"]["occurrences"], 0)


class LifetimeChurnForecastDriftWavePredictionsIsolationTests(
    unittest.TestCase
):
    def call_args(self):
        return dict(
            LONG_ARGS, split_cutoffs=(3, 4, 5, 6), wave_horizon=3
        )

    def test_result_levels_are_mutually_unshared(self):
        store, args = diamond_store()
        result = predictions_call(store, args, **self.call_args())
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
        self.assertEqual(len(seen_ids), len(set(seen_ids)))

    def test_mutating_result_never_affects_later_calls(self):
        store, args = diamond_store()
        first = predictions_call(store, args, **self.call_args())
        pristine = copy.deepcopy(first)
        first["totals"]["occurrences"] = 99
        second = predictions_call(store, args, **self.call_args())
        self.assertEqual(second, pristine)
        # The periodicity query feeding the prediction is unaffected.
        periodic_args = {
            key: value
            for key, value in self.call_args().items()
            if key not in ("wave_horizon", "prediction_result_limit")
        }
        periodic = store.lifetime_churn_forecast_drift_wave_periodicities(
            **periodicities_kwargs(args, **periodic_args)
        )
        self.assertEqual(periodic["periodicities"], ())

    def test_success_and_failure_change_nothing(self):
        store, args = diamond_store()
        heads = dict(store._heads)
        appends = copy.deepcopy(store._appends)
        merges = copy.deepcopy(store._merges)
        audit = store.audit_log()

        predictions_call(store, args, **self.call_args())
        failures = [
            dict(windows="x"),
            dict(wave_horizon=0),
            dict(wave_horizon="x"),
            dict(wave_horizon=True),
            dict(prediction_result_limit=0),
            dict(prediction_result_limit="x"),
            dict(prediction_result_limit=False),
            dict(periodicity_result_limit=0),
            dict(max_wave_jitter="x"),
            dict(recurrence_result_limit=0),
            dict(streak_result_limit=0),
            dict(drift_limit=2),
            dict(scan_limit=1),
            dict(wave_result_limit=0),
            dict(causes=("zzz",)),
            dict(reference="ghost"),
        ]
        for overrides in failures:
            with self.subTest(overrides=overrides):
                merged = self.call_args()
                merged.update(overrides)
                with self.assertRaises(Exception):
                    predictions_call(store, args, **merged)

        self.assertEqual(dict(store._heads), heads)
        self.assertEqual(copy.deepcopy(store._appends), appends)
        self.assertEqual(copy.deepcopy(store._merges), merges)
        self.assertEqual(store.audit_log(), audit)


class LifetimeChurnForecastDriftWavePredictionsTokenTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, token=None, **overrides):
        return predictions_call(
            self.store, self.args, token=token, **overrides
        )

    def call_args(self):
        return dict(
            LONG_ARGS, split_cutoffs=(3, 4, 5), wave_horizon=3
        )

    def test_success_consumes_one_read_and_failure_refunds(self):
        token = self.store.create_snapshot(2)
        self.call(token=token, **self.call_args())
        # Parameter validation fails before the token is touched.
        with self.assertRaises(ValueError):
            self.call(
                token=token, **dict(self.call_args(), wave_horizon=0)
            )
        with self.assertRaises(TypeError):
            self.call(
                token=token,
                **dict(self.call_args(), prediction_result_limit="x"),
            )
        # A failing earlier stage inside the frozen view refunds.
        with self.assertRaises(ValueError):
            self.call(
                token=token, **dict(self.call_args(), drift_limit=2)
            )
        with self.assertRaises(ValueError):
            self.call(
                token=token, **dict(self.call_args(), scan_limit=1)
            )
        # A state error refunds as well.
        with self.assertRaises(KeyError):
            self.call(
                token=token, **dict(self.call_args(), causes=("zzz",))
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
        self.assertEqual(
            self.call(token=token, **self.call_args()), before
        )


if __name__ == "__main__":
    unittest.main()
