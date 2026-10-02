import copy
import inspect
import unittest

from city_twin.branches import BranchStore

from tests.test_lifetime_churn import diamond_store
from tests.test_lifetime_churn_forecast_drift_scan import LONG_ARGS
from tests.test_lifetime_churn_forecast_drift_waves import waves_kwargs


def predictions_kwargs(store_args, **overrides) -> dict:
    min_wave_occurrences = overrides.pop("min_wave_occurrences", 1)
    recurrence_result_limit = overrides.pop("recurrence_result_limit", 50)
    max_wave_jitter = overrides.pop("max_wave_jitter", 0)
    periodicity_result_limit = overrides.pop("periodicity_result_limit", 50)
    wave_horizon = overrides.pop("wave_horizon", 10)
    prediction_result_limit = overrides.pop("prediction_result_limit", 50)
    kwargs = waves_kwargs(store_args, **overrides)
    kwargs["min_wave_occurrences"] = min_wave_occurrences
    kwargs["recurrence_result_limit"] = recurrence_result_limit
    kwargs["max_wave_jitter"] = max_wave_jitter
    kwargs["periodicity_result_limit"] = periodicity_result_limit
    kwargs["wave_horizon"] = wave_horizon
    kwargs["prediction_result_limit"] = prediction_result_limit
    return kwargs


def predictions_call(store, store_args, **overrides):
    return store.lifetime_churn_forecast_drift_wave_predictions(
        **predictions_kwargs(store_args, **overrides)
    )


def synthetic_periodicities():
    return (
        {"identity": "X", "period": 2, "jitter": 0, "last_wave": 1},
        {"identity": "Y", "period": 3, "jitter": 1, "last_wave": 2},
        {"identity": "Z", "period": 6, "jitter": 2, "last_wave": 0},
    )


class LifetimeChurnForecastDriftWavePredictionsSignatureTests(unittest.TestCase):
    def test_public_signature_appends_two_parameters(self):
        parameters = list(
            inspect.signature(
                BranchStore.lifetime_churn_forecast_drift_wave_predictions
            ).parameters.values()
        )
        names = [p.name for p in parameters]
        periodicity_names = list(
            inspect.signature(
                BranchStore.lifetime_churn_forecast_drift_wave_periodicities
            ).parameters
        )
        self.assertEqual(
            names, periodicity_names[:-1] + [
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
        result = BranchStore._build_lifetime_churn_forecast_drift_wave_predictions_result(
            synthetic_periodicities(), 5, 50
        )
        self.assertEqual(list(result), ["predictions", "totals"])
        self.assertIsInstance(result["predictions"], tuple)
        for record in result["predictions"]:
            self.assertEqual(
                list(record),
                ["identity", "period", "jitter", "last_wave", "next_waves"],
            )
            self.assertIsInstance(record["next_waves"], tuple)
            for wave in record["next_waves"]:
                self.assertEqual(list(wave), ["wave", "offset"])
        self.assertEqual(
            list(result["totals"]),
            ["predictions", "occurrences", "first_wave", "last_wave"],
        )


class LifetimeChurnForecastDriftWavePredictionsResultTests(unittest.TestCase):
    def test_waves_project_from_last_wave_by_period_within_horizon(self):
        result = BranchStore._build_lifetime_churn_forecast_drift_wave_predictions_result(
            synthetic_periodicities(), 5, 50
        )
        records = result["predictions"]
        # Z's first projection (offset 6) lies past the horizon, so Z
        # drops; X and Y keep periodicity order regardless of counts.
        self.assertEqual([r["identity"] for r in records], ["X", "Y"])
        self.assertEqual(
            records[0]["next_waves"],
            ({"wave": 3, "offset": 2}, {"wave": 5, "offset": 4}),
        )
        self.assertEqual(
            records[1]["next_waves"],
            ({"wave": 5, "offset": 3},),
        )
        self.assertEqual(
            result["totals"],
            {"predictions": 2, "occurrences": 3, "first_wave": 3, "last_wave": 5},
        )

    def test_exact_horizon_distance_is_kept(self):
        result = BranchStore._build_lifetime_churn_forecast_drift_wave_predictions_result(
            synthetic_periodicities(), 6, 50
        )
        z = next(r for r in result["predictions"] if r["identity"] == "Z")
        self.assertEqual(z["next_waves"], ({"wave": 6, "offset": 6},))
        self.assertEqual(result["totals"]["occurrences"], 6)
        self.assertEqual(result["totals"]["last_wave"], 8)

    def test_empty_periodicities_and_short_horizon(self):
        result = BranchStore._build_lifetime_churn_forecast_drift_wave_predictions_result(
            (), 5, 50
        )
        self.assertEqual(result["predictions"], ())
        self.assertEqual(
            result["totals"],
            {"predictions": 0, "occurrences": 0, "first_wave": 0, "last_wave": 0},
        )
        result = BranchStore._build_lifetime_churn_forecast_drift_wave_predictions_result(
            synthetic_periodicities(), 1, 50
        )
        self.assertEqual(result["predictions"], ())
        self.assertEqual(result["totals"]["predictions"], 0)

    def test_records_keep_periodicity_order(self):
        # Y projects one wave and X two, but X stays first.
        result = BranchStore._build_lifetime_churn_forecast_drift_wave_predictions_result(
            synthetic_periodicities(), 4, 50
        )
        self.assertEqual(
            [r["identity"] for r in result["predictions"]], ["X", "Y"]
        )

    def test_prediction_limit_bounds_without_truncating(self):
        with self.assertRaises(ValueError) as caught:
            BranchStore._build_lifetime_churn_forecast_drift_wave_predictions_result(
                synthetic_periodicities(), 5, 1
            )
        self.assertIn("prediction result limit", str(caught.exception))
        result = BranchStore._build_lifetime_churn_forecast_drift_wave_predictions_result(
            synthetic_periodicities(), 5, 2
        )
        self.assertEqual(len(result["predictions"]), 2)

    def test_every_object_is_fresh(self):
        source = (
            {
                "identity": {"a": [1]},
                "period": 2,
                "jitter": 0,
                "last_wave": 1,
            },
        )
        result = BranchStore._build_lifetime_churn_forecast_drift_wave_predictions_result(
            source, 5, 50
        )
        identity = result["predictions"][0]["identity"]
        self.assertEqual(identity, {"a": (1,)})
        identity["b"] = 2
        result["predictions"][0]["next_waves"][0]["wave"] = 999
        again = BranchStore._build_lifetime_churn_forecast_drift_wave_predictions_result(
            source, 5, 50
        )
        self.assertEqual(
            again["predictions"][0]["next_waves"][0],
            {"wave": 3, "offset": 2},
        )
        self.assertEqual(again["predictions"][0]["identity"], {"a": (1,)})

        result2 = BranchStore._build_lifetime_churn_forecast_drift_wave_predictions_result(
            synthetic_periodicities(), 5, 50
        )
        seen_ids = []

        def collect(value):
            if isinstance(value, dict):
                seen_ids.append(id(value))
                for item in value.values():
                    collect(item)
            elif isinstance(value, (tuple, list)):
                for item in value:
                    collect(item)

        collect(result2)
        self.assertEqual(len(seen_ids), len(set(seen_ids)))


class LifetimeChurnForecastDriftWavePredictionsValidationTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        return predictions_call(self.store, self.args, **overrides)

    def extended(self, **overrides):
        overrides.setdefault("split_cutoffs", (4,))
        merged = dict(LONG_ARGS)
        merged.update(overrides)
        return self.call(**merged)

    def test_new_parameters_must_be_non_bool_positive_ints(self):
        for name in ("wave_horizon", "prediction_result_limit"):
            for bad in (True, False, 1.0, "1", None, (1,)):
                with self.subTest(parameter=name, value=bad):
                    with self.assertRaises(TypeError):
                        self.extended(**{name: bad})
            with self.assertRaises(ValueError):
                self.extended(**{name: 0})
            with self.assertRaises(ValueError):
                self.extended(**{name: -3})

    def test_new_parameters_validated_in_signature_order(self):
        # periodicity_result_limit fails before wave_horizon.
        with self.assertRaises(ValueError):
            self.extended(periodicity_result_limit=0, wave_horizon=0)
        with self.assertRaises(TypeError):
            self.extended(periodicity_result_limit="x", wave_horizon="x")
        # wave_horizon fails before prediction_result_limit.
        with self.assertRaises(ValueError):
            self.extended(wave_horizon=0, prediction_result_limit=0)
        with self.assertRaises(TypeError):
            self.extended(wave_horizon="x", prediction_result_limit="x")
        # Both pass before the split membership check fires.
        with self.assertRaises(ValueError):
            self.extended(split_cutoffs=(99,))

    def test_all_parameter_checks_precede_state_lookups(self):
        with self.assertRaises(TypeError):
            self.call(
                reference="ghost", cutoffs=(0, 1), wave_horizon="x"
            )
        with self.assertRaises(ValueError):
            self.call(reference="ghost", cutoffs=(0, 1), wave_horizon=0)
        with self.assertRaises(TypeError):
            self.call(
                reference="ghost",
                cutoffs=(0, 1),
                prediction_result_limit="x",
            )
        with self.assertRaises(ValueError):
            self.call(
                reference="ghost", cutoffs=(0, 1), prediction_result_limit=0
            )


class LifetimeChurnForecastDriftWavePredictionsEmptyRunTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def test_empty_split_tuple_returns_empty_predictions_and_zero_totals(self):
        result = predictions_call(
            self.store,
            self.args,
            **dict(LONG_ARGS, cutoffs=(), split_cutoffs=()),
        )
        self.assertEqual(result["predictions"], ())
        self.assertEqual(
            result["totals"],
            {"predictions": 0, "occurrences": 0, "first_wave": 0, "last_wave": 0},
        )


class LifetimeChurnForecastDriftWavePredictionsIsolationTests(unittest.TestCase):
    def call_args(self):
        return dict(
            LONG_ARGS,
            split_cutoffs=(3, 4, 5),
            wave_horizon=10,
            prediction_result_limit=50,
        )

    def test_success_and_failure_change_nothing(self):
        store, args = diamond_store()
        heads = dict(store._heads)
        appends = copy.deepcopy(store._appends)
        merges = copy.deepcopy(store._merges)
        audit = store.audit_log()

        predictions_call(store, args, **self.call_args())
        failures = [
            dict(wave_horizon=0),
            dict(wave_horizon="x"),
            dict(wave_horizon=True),
            dict(prediction_result_limit=0),
            dict(prediction_result_limit="x"),
            dict(prediction_result_limit=False),
            dict(periodicity_result_limit=0),
            dict(drift_limit=2),
            dict(scan_limit=1),
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
        return predictions_call(self.store, self.args, token=token, **overrides)

    def call_args(self):
        return dict(
            LONG_ARGS,
            split_cutoffs=(3, 4, 5),
            wave_horizon=10,
            prediction_result_limit=50,
        )

    def test_success_consumes_one_read_and_failure_refunds(self):
        token = self.store.create_snapshot(2)
        self.call(token=token, **self.call_args())
        # Stage failures still refund.
        with self.assertRaises(ValueError):
            self.call(token=token, **dict(self.call_args(), drift_limit=2))
        with self.assertRaises(ValueError):
            self.call(
                token=token, **dict(self.call_args(), scan_limit=1)
            )
        # Parameter validation runs before the token is touched.
        with self.assertRaises(TypeError):
            self.call(token=token, **dict(self.call_args(), wave_horizon="x"))
        with self.assertRaises(ValueError):
            self.call(
                token=token,
                **dict(self.call_args(), prediction_result_limit=0),
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
