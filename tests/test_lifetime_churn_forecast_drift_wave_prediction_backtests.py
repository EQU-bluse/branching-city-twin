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
    match_tolerance = overrides.pop("match_tolerance", 0)
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


def build_backtests(waves, *args):
    return BranchStore._build_lifetime_churn_forecast_drift_wave_prediction_backtests_result(
        waves, *args
    )


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
        {
            "start_split": index,
            "end_split": index,
            "length": 1,
            "peak_identities": len(identities),
            "points": (
                {
                    "split_cutoff": index,
                    "active_count": len(identities),
                    "identities": tuple(identities),
                },
            ),
        }
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
        {
            "start_split": index,
            "end_split": index,
            "length": 1,
            "peak_identities": len(identities),
            "points": (
                {
                    "split_cutoff": index,
                    "active_count": len(identities),
                    "identities": tuple(identities),
                },
            ),
        }
        for index, identities in enumerate(identities_per_wave)
    )


def no_reuse_waves():
    """F rides waves 0-2 (period one) and reappears only at wave 3."""
    identities_per_wave = (("F",), ("F",), ("F",), ("F",), ())
    return tuple(
        {
            "start_split": index,
            "end_split": index,
            "length": 1,
            "peak_identities": len(identities),
            "points": (
                {
                    "split_cutoff": index,
                    "active_count": len(identities),
                    "identities": tuple(identities),
                },
            ),
        }
        for index, identities in enumerate(identities_per_wave)
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
        result = build_backtests(
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
                    "period",
                    "jitter",
                    "last_wave",
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
                    list(target), ["wave", "offset", "actual", "status"]
                )
        self.assertEqual(
            list(result["totals"]),
            ["cutoffs", "records", "hit", "missed", "unresolved", "unexpected"],
        )


class LifetimeChurnForecastDriftWavePredictionBacktestsResultTests(
    unittest.TestCase
):
    def test_training_matching_and_totals(self):
        result = build_backtests(
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
        self.assertEqual(records[0]["period"], 1)
        self.assertEqual(
            records[0]["targets"],
            (
                {"wave": 5, "offset": 1, "actual": 5, "status": "hit"},
                {"wave": 6, "offset": 2, "actual": None, "status": "missed"},
            ),
        )
        self.assertEqual(
            (records[0]["hit"], records[0]["missed"],
             records[0]["unresolved"], records[0]["unexpected"]),
            (1, 1, 0, 0),
        )
        self.assertEqual(
            records[1]["targets"],
            (
                {"wave": 5, "offset": 1, "actual": 5, "status": "hit"},
                {"wave": 6, "offset": 2, "actual": None, "status": "missed"},
            ),
        )
        # B's period is two, so its only target is wave 6; its wave-5
        # appearance inside the window stays unclaimed.
        self.assertEqual(records[2]["period"], 2)
        self.assertEqual(
            records[2]["targets"],
            ({"wave": 6, "offset": 2, "actual": 6, "status": "hit"},),
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
        result = build_backtests(
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
        result = build_backtests(
            synthetic_waves(), 1, 50, 0, 50, 2, 50, (4,), 0, 50
        )
        record_a = result["backtests"][0]
        self.assertEqual(record_a["unexpected"], 0)
        self.assertEqual(
            [t["wave"] for t in record_a["targets"]], [5, 6]
        )

    def test_equal_distances_take_the_earlier_wave(self):
        result = build_backtests(
            tie_waves(), 1, 50, 0, 50, 3, 50, (4,), 1, 50
        )
        record = result["backtests"][0]
        # Target 6 sits one wave from both 5 and 7; the earlier wave
        # wins and wave 7 is left unexpected.
        self.assertEqual(
            record["targets"],
            ({"wave": 6, "offset": 2, "actual": 5, "status": "hit"},),
        )
        self.assertEqual(record["unexpected"], 1)

    def test_tolerance_zero_requires_the_exact_wave(self):
        result = build_backtests(
            tie_waves(), 1, 50, 0, 50, 3, 50, (4,), 0, 50
        )
        record = result["backtests"][0]
        self.assertEqual(
            record["targets"],
            ({"wave": 6, "offset": 2, "actual": None, "status": "missed"},),
        )
        self.assertEqual(record["unexpected"], 2)

    def test_observed_waves_are_never_reused(self):
        result = build_backtests(
            no_reuse_waves(), 1, 50, 0, 50, 2, 50, (2,), 1, 50
        )
        record = result["backtests"][0]
        # Wave 3 claims target 3; target 4 is within tolerance of wave
        # 3 but cannot reuse it.
        self.assertEqual(
            record["targets"],
            (
                {"wave": 3, "offset": 1, "actual": 3, "status": "hit"},
                {"wave": 4, "offset": 2, "actual": None, "status": "missed"},
            ),
        )

    def test_records_sort_by_cutoff_then_periodicity_order(self):
        result = build_backtests(
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
        result = build_backtests(
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
                record["targets"],
                (
                    {"wave": 3, "offset": 1, "actual": 3, "status": "hit"},
                    {"wave": 4, "offset": 2, "actual": 4, "status": "hit"},
                ),
            )

    def test_jitter_over_max_wave_jitter_is_dropped(self):
        # G rides waves 0, 1 and 3: intervals (1, 2), so its jitter of
        # one only qualifies once max_wave_jitter allows it.
        waves = tuple(
            {
                "start_split": index,
                "end_split": index,
                "length": 1,
                "peak_identities": len(identities),
                "points": (
                    {
                        "split_cutoff": index,
                        "active_count": len(identities),
                        "identities": tuple(identities),
                    },
                ),
            }
            for index, identities in enumerate(
                (("G",), ("G",), (), ("G",), ("G",))
            )
        )
        result = build_backtests(waves, 1, 50, 0, 50, 2, 50, (3,), 0, 50)
        self.assertEqual(result["backtests"], ())
        result = build_backtests(waves, 1, 50, 1, 50, 2, 50, (3,), 0, 50)
        self.assertEqual(len(result["backtests"]), 1)
        record = result["backtests"][0]
        self.assertEqual(record["jitter"], 1)
        # The median period is the smaller middle interval, one.
        self.assertEqual(record["period"], 1)
        self.assertEqual(
            record["targets"],
            (
                {"wave": 4, "offset": 1, "actual": 4, "status": "hit"},
                {"wave": 5, "offset": 2, "actual": None, "status": "unresolved"},
            ),
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
        result = build_backtests(
            synthetic_waves(), 1, 50, 0, 50, 2, 50, (), 0, 50
        )
        self.assertEqual(result, {"backtests": (), "totals": zero_totals})
        # No waves at all: the range check is skipped and nothing
        # trains.
        result = build_backtests((), 1, 50, 0, 50, 2, 50, (3,), 0, 50)
        self.assertEqual(result, {"backtests": (), "totals": zero_totals})
        # No identity reaches three training appearances.
        result = build_backtests(
            synthetic_waves()[:1], 1, 50, 0, 50, 2, 50, (0,), 0, 50
        )
        self.assertEqual(result, {"backtests": (), "totals": zero_totals})
        # A horizon shorter than every period projects nothing.
        result = build_backtests(
            tie_waves(), 1, 50, 0, 50, 1, 50, (4,), 0, 50
        )
        self.assertEqual(result, {"backtests": (), "totals": zero_totals})
        # A horizon of one keeps only the period-one identities.
        result = build_backtests(
            synthetic_waves(), 1, 50, 0, 50, 1, 50, (4,), 0, 50
        )
        self.assertEqual(
            [r["identity"] for r in result["backtests"]], ["A", "C"]
        )

    def test_cutoff_beyond_the_last_wave_raises(self):
        with self.assertRaises(ValueError) as caught:
            build_backtests(
                synthetic_waves(), 1, 50, 0, 50, 2, 50, (8,), 0, 50
            )
        self.assertIn("out of range", str(caught.exception))
        # The check fires before any training, even for a later cutoff.
        with self.assertRaises(ValueError):
            build_backtests(
                synthetic_waves(), 1, 50, 0, 50, 2, 50, (4, 8), 0, 50
            )

    def test_stage_limits_bound_each_cutoff_without_truncating(self):
        with self.assertRaises(ValueError) as caught:
            build_backtests(
                synthetic_waves(), 1, 2, 0, 50, 2, 50, (4,), 0, 50
            )
        self.assertIn("recurrence result limit", str(caught.exception))
        with self.assertRaises(ValueError) as caught:
            build_backtests(
                synthetic_waves(), 1, 50, 0, 2, 2, 50, (4,), 0, 50
            )
        self.assertIn("periodicity result limit", str(caught.exception))
        with self.assertRaises(ValueError) as caught:
            build_backtests(
                synthetic_waves(), 1, 50, 0, 50, 2, 2, (4,), 0, 50
            )
        self.assertIn("prediction result limit", str(caught.exception))
        result = build_backtests(
            synthetic_waves(), 1, 50, 0, 50, 2, 3, (4,), 0, 50
        )
        self.assertEqual(len(result["backtests"]), 3)

    def test_backtest_limit_bounds_the_complete_record_set(self):
        with self.assertRaises(ValueError) as caught:
            build_backtests(
                synthetic_waves(), 1, 50, 0, 50, 2, 50, (2, 4), 0, 4
            )
        self.assertIn("prediction backtest limit", str(caught.exception))
        result = build_backtests(
            synthetic_waves(), 1, 50, 0, 50, 2, 50, (2, 4), 0, 5
        )
        self.assertEqual(result["totals"]["records"], 5)

    def test_every_object_is_fresh(self):
        waves = synthetic_waves()
        result = build_backtests(waves, 1, 50, 0, 50, 2, 50, (4,), 0, 50)
        result["backtests"][0]["targets"][0]["wave"] = 999
        result["backtests"][0]["hit"] = 999
        result["totals"]["hit"] = 999
        again = build_backtests(waves, 1, 50, 0, 50, 2, 50, (4,), 0, 50)
        self.assertEqual(
            again["backtests"][0]["targets"][0],
            {"wave": 5, "offset": 1, "actual": 5, "status": "hit"},
        )
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

    def test_wave_cutoffs_must_be_a_strictly_increasing_int_tuple(self):
        for bad in ([0], "0", 0, 1.0, None, {0}):
            with self.subTest(value=bad):
                with self.assertRaises(TypeError):
                    self.extended(wave_cutoffs=bad)
        for bad in (True, False, 1.0, "0", None):
            with self.subTest(value=bad):
                with self.assertRaises(TypeError):
                    self.extended(wave_cutoffs=(bad,))
        with self.assertRaises(ValueError):
            self.extended(wave_cutoffs=(-1,))
        with self.assertRaises(ValueError):
            self.extended(wave_cutoffs=(0, 0))
        with self.assertRaises(ValueError):
            self.extended(wave_cutoffs=(1, 0))

    def test_match_tolerance_must_be_a_non_bool_non_negative_int(self):
        for bad in (True, False, 1.0, "1", None, (1,)):
            with self.subTest(value=bad):
                with self.assertRaises(TypeError):
                    self.extended(match_tolerance=bad)
        with self.assertRaises(ValueError):
            self.extended(match_tolerance=-1)
        self.extended(match_tolerance=0)

    def test_prediction_backtest_limit_must_be_a_non_bool_positive_int(self):
        for bad in (True, False, 1.0, "1", None, (1,)):
            with self.subTest(value=bad):
                with self.assertRaises(TypeError):
                    self.extended(prediction_backtest_limit=bad)
        with self.assertRaises(ValueError):
            self.extended(prediction_backtest_limit=0)
        with self.assertRaises(ValueError):
            self.extended(prediction_backtest_limit=-3)

    def test_new_parameters_validated_in_signature_order(self):
        # prediction_result_limit fails before wave_cutoffs.
        with self.assertRaises(ValueError):
            self.extended(prediction_result_limit=0, wave_cutoffs="x")
        with self.assertRaises(TypeError):
            self.extended(prediction_result_limit="x", wave_cutoffs="x")
        # wave_cutoffs fails before match_tolerance.
        with self.assertRaises(TypeError):
            self.extended(wave_cutoffs="x", match_tolerance="x")
        with self.assertRaises(ValueError):
            self.extended(wave_cutoffs=(-1,), match_tolerance=-1)
        # match_tolerance fails before prediction_backtest_limit.
        with self.assertRaises(TypeError):
            self.extended(
                match_tolerance="x", prediction_backtest_limit="x"
            )
        with self.assertRaises(ValueError):
            self.extended(match_tolerance=-1, prediction_backtest_limit=0)
        # All three pass before the split membership check fires.
        with self.assertRaises(ValueError):
            self.extended(
                split_cutoffs=(99,),
                wave_cutoffs=(0,),
                match_tolerance=1,
                prediction_backtest_limit=1,
            )

    def test_all_parameter_checks_precede_state_lookups(self):
        with self.assertRaises(TypeError):
            self.call(reference="ghost", cutoffs=(0, 1), wave_cutoffs="x")
        with self.assertRaises(ValueError):
            self.call(reference="ghost", cutoffs=(0, 1), wave_cutoffs=(-1,))
        with self.assertRaises(TypeError):
            self.call(reference="ghost", cutoffs=(0, 1), match_tolerance="x")
        with self.assertRaises(ValueError):
            self.call(reference="ghost", cutoffs=(0, 1), match_tolerance=-1)
        with self.assertRaises(TypeError):
            self.call(
                reference="ghost",
                cutoffs=(0, 1),
                prediction_backtest_limit="x",
            )
        with self.assertRaises(ValueError):
            self.call(
                reference="ghost", cutoffs=(0, 1), prediction_backtest_limit=0
            )


class LifetimeChurnForecastDriftWavePredictionBacktestsEmptyRunTests(
    unittest.TestCase
):
    def setUp(self):
        self.store, self.args = diamond_store()

    def zero_totals(self):
        return {
            "cutoffs": 0,
            "records": 0,
            "hit": 0,
            "missed": 0,
            "unresolved": 0,
            "unexpected": 0,
        }

    def test_empty_wave_cutoffs_returns_empty_records_and_zero_totals(self):
        result = backtests_call(
            self.store,
            self.args,
            **dict(LONG_ARGS, split_cutoffs=(3, 4, 5), wave_cutoffs=()),
        )
        self.assertEqual(result["backtests"], ())
        self.assertEqual(result["totals"], self.zero_totals())

    def test_no_waves_returns_empty_records_and_zero_totals(self):
        result = backtests_call(
            self.store,
            self.args,
            **dict(
                LONG_ARGS, cutoffs=(), split_cutoffs=(), wave_cutoffs=(0,)
            ),
        )
        self.assertEqual(result["backtests"], ())
        self.assertEqual(result["totals"], self.zero_totals())

    def test_no_qualifying_identity_returns_empty_records(self):
        # The diamond store yields a single drift wave, so no identity
        # reaches three training appearances at cutoff 0.
        result = backtests_call(
            self.store,
            self.args,
            **dict(LONG_ARGS, split_cutoffs=(3, 4, 5), wave_cutoffs=(0,)),
        )
        self.assertEqual(result["backtests"], ())
        self.assertEqual(result["totals"], self.zero_totals())

    def test_cutoff_beyond_the_last_wave_raises(self):
        with self.assertRaises(ValueError) as caught:
            backtests_call(
                self.store,
                self.args,
                **dict(LONG_ARGS, split_cutoffs=(3, 4, 5), wave_cutoffs=(1,)),
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
        return backtests_call(self.store, self.args, token=token, **overrides)

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
        with self.assertRaises(ValueError):
            self.call(token=token, **dict(self.call_args(), wave_cutoffs=(9,)))
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
