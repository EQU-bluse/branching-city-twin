import copy
import inspect
import unittest

from city_twin.branches import BranchStore

from tests.test_lifetime_churn import diamond_store
from tests.test_lifetime_churn_forecast_drift import LONG_WINDOWS
from tests.test_lifetime_churn_forecast_drift_scan import LONG_ARGS, scan_kwargs


def waves_kwargs(store_args, **overrides) -> dict:
    min_splits = overrides.pop("min_splits", 1)
    streak_result_limit = overrides.pop("streak_result_limit", 50)
    min_active_identities = overrides.pop("min_active_identities", 1)
    wave_result_limit = overrides.pop("wave_result_limit", 50)
    kwargs = scan_kwargs(store_args, **overrides)
    kwargs["min_splits"] = min_splits
    kwargs["streak_result_limit"] = streak_result_limit
    kwargs["min_active_identities"] = min_active_identities
    kwargs["wave_result_limit"] = wave_result_limit
    return kwargs


def waves_call(store, store_args, **overrides):
    return store.lifetime_churn_forecast_drift_waves(
        **waves_kwargs(store_args, **overrides)
    )


def streaks_call(store, store_args, **overrides):
    overrides.pop("min_active_identities", None)
    overrides.pop("wave_result_limit", None)
    kwargs = waves_kwargs(store_args, **overrides)
    del kwargs["min_active_identities"]
    del kwargs["wave_result_limit"]
    return store.lifetime_churn_forecast_drift_streaks(**kwargs)


def synthetic_scans():
    """Four requested positions pinning coverage, order and breaks.

    Position 3 drifts A, B; position 4 drifts A only; position 5
    drifts A, B, C; position 6 drifts B, C. Streaks keep A over 3-5,
    B over 3-4 and C over 5-6, so with ``min_active_identities`` of
    two the active positions are 3 (A, B) and 5 (A, C): position 4
    falls below the threshold and ends the first wave, and position 6
    stays below it, so two one-point waves result.
    """
    scans = tuple(
        {
            "split_cutoff": split_cutoff,
            "drifts": tuple(
                {"identity": identity} for identity in identities
            ),
            "totals": {},
        }
        for split_cutoff, identities in (
            (3, ("A", "B")),
            (4, ("A",)),
            (5, ("A", "B", "C")),
            (6, ("B", "C")),
        )
    )
    streaks = (
        {
            "identity": "A",
            "start_split": 3,
            "end_split": 5,
            "length": 3,
            "points": tuple({"split_cutoff": s} for s in (3, 4, 5)),
        },
        {
            "identity": "B",
            "start_split": 3,
            "end_split": 4,
            "length": 2,
            "points": tuple({"split_cutoff": s} for s in (3, 4)),
        },
        {
            "identity": "C",
            "start_split": 5,
            "end_split": 6,
            "length": 2,
            "points": tuple({"split_cutoff": s} for s in (5, 6)),
        },
    )
    return scans, streaks


class LifetimeChurnForecastDriftWavesSignatureTests(unittest.TestCase):
    def test_public_signature_appends_two_parameters(self):
        parameters = list(
            inspect.signature(
                BranchStore.lifetime_churn_forecast_drift_waves
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
                "token",
            ],
        )
        self.assertEqual(
            [p.default for p in parameters[:-1]],
            [inspect.Parameter.empty] * (len(parameters) - 1),
        )
        self.assertIsNone(parameters[-1].default)

    def test_result_shape_and_key_order(self):
        store, args = diamond_store()
        result = waves_call(store, args, **LONG_ARGS, split_cutoffs=(3, 4, 5))
        self.assertIsInstance(result, dict)
        self.assertEqual(list(result), ["waves", "totals"])
        self.assertIsInstance(result["waves"], tuple)
        self.assertEqual(len(result["waves"]), 1)
        wave = result["waves"][0]
        self.assertEqual(
            list(wave),
            ["start_split", "end_split", "length", "peak_identities", "points"],
        )
        self.assertIsInstance(wave["points"], tuple)
        for point in wave["points"]:
            self.assertEqual(
                list(point), ["split_cutoff", "active_count", "identities"]
            )
            self.assertIsInstance(point["identities"], tuple)
            self.assertEqual(point["active_count"], len(point["identities"]))
        self.assertEqual(
            list(result["totals"]),
            ["splits", "identities", "waves", "points", "peak_active"],
        )


class LifetimeChurnForecastDriftWavesResultTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        return waves_call(self.store, self.args, **overrides)

    def test_full_run_forms_one_wave_over_every_requested_position(self):
        result = self.call(**LONG_ARGS, split_cutoffs=(3, 4, 5, 6))
        self.assertEqual(len(result["waves"]), 1)
        wave = result["waves"][0]
        self.assertEqual(wave["start_split"], 3)
        self.assertEqual(wave["end_split"], 6)
        self.assertEqual(wave["length"], 4)
        self.assertEqual(wave["peak_identities"], 3)
        self.assertEqual(
            [point["split_cutoff"] for point in wave["points"]],
            [3, 4, 5, 6],
        )
        for point in wave["points"]:
            self.assertEqual(point["active_count"], 3)
            self.assertEqual(
                point["identities"],
                (("W", "y"), ("a0", "x"), (("a0", "x"), ("W", "y"))),
            )
        self.assertEqual(
            result["totals"],
            {
                "splits": 4,
                "identities": 3,
                "waves": 1,
                "points": 4,
                "peak_active": 3,
            },
        )

    def test_positions_below_threshold_bound_the_wave(self):
        # With min_resolved=2 only splits 4 and 5 keep drift records,
        # so the wave covers exactly those requested positions.
        result = self.call(
            **dict(LONG_ARGS, min_resolved=2), split_cutoffs=(3, 4, 5, 6)
        )
        self.assertEqual(len(result["waves"]), 1)
        wave = result["waves"][0]
        self.assertEqual(
            [point["split_cutoff"] for point in wave["points"]], [4, 5]
        )
        self.assertEqual(wave["start_split"], 4)
        self.assertEqual(wave["end_split"], 5)
        self.assertEqual(wave["length"], 2)
        self.assertEqual(result["totals"]["splits"], 2)
        self.assertEqual(result["totals"]["points"], 2)

    def test_unrequested_cutoffs_do_not_break_continuity(self):
        # Requesting only 3 and 5 skips 4, yet the two positions are
        # adjacent requested positions and form one wave.
        result = self.call(**LONG_ARGS, split_cutoffs=(3, 5))
        self.assertEqual(len(result["waves"]), 1)
        wave = result["waves"][0]
        self.assertEqual(wave["start_split"], 3)
        self.assertEqual(wave["end_split"], 5)
        self.assertEqual(wave["length"], 2)

    def test_min_active_identities_selects_positions(self):
        result = self.call(
            **LONG_ARGS, split_cutoffs=(3, 4, 5), min_active_identities=3
        )
        self.assertEqual(len(result["waves"]), 1)
        self.assertEqual(result["waves"][0]["length"], 3)
        result = self.call(
            **LONG_ARGS, split_cutoffs=(3, 4, 5), min_active_identities=4
        )
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

    def test_min_splits_filters_streaks_before_positions_are_counted(self):
        # Every streak spans 3-6, so a min_splits of five keeps none.
        result = self.call(
            **LONG_ARGS, split_cutoffs=(3, 4, 5, 6), min_splits=5
        )
        self.assertEqual(result["waves"], ())
        self.assertEqual(result["totals"]["waves"], 0)
        self.assertEqual(result["totals"]["peak_active"], 0)

    def test_empty_split_tuple_returns_empty_waves_and_zero_totals(self):
        result = self.call(
            **dict(LONG_ARGS, cutoffs=()), split_cutoffs=()
        )
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

    def test_wave_identities_match_streak_coverage_and_record_order(self):
        split_cutoffs = (3, 4, 5, 6)
        result = self.call(**LONG_ARGS, split_cutoffs=split_cutoffs)
        streaks = streaks_call(
            self.store, self.args, **LONG_ARGS, split_cutoffs=split_cutoffs
        )["streaks"]
        scans = self.store.lifetime_churn_forecast_drift_scan(
            **scan_kwargs(self.args, **LONG_ARGS, split_cutoffs=split_cutoffs)
        )["scans"]
        for wave in result["waves"]:
            for point in wave["points"]:
                covering = {
                    streak["identity"]
                    for streak in streaks
                    if point["split_cutoff"]
                    in [p["split_cutoff"] for p in streak["points"]]
                }
                scan = next(
                    scan
                    for scan in scans
                    if scan["split_cutoff"] == point["split_cutoff"]
                )
                expected = tuple(
                    record["identity"]
                    for record in scan["drifts"]
                    if record["identity"] in covering
                )
                self.assertEqual(point["identities"], expected)
                self.assertEqual(point["active_count"], len(expected))

    def test_wave_builder_splits_runs_and_orders_identities(self):
        scans, streaks = synthetic_scans()
        result = BranchStore._build_lifetime_churn_forecast_drift_waves_result(
            scans, streaks, 2, 10
        )
        self.assertEqual(len(result["waves"]), 2)
        first, second = result["waves"]
        self.assertEqual(first["start_split"], 3)
        self.assertEqual(first["end_split"], 3)
        self.assertEqual(first["length"], 1)
        self.assertEqual(first["peak_identities"], 2)
        self.assertEqual(
            first["points"],
            (
                {
                    "split_cutoff": 3,
                    "active_count": 2,
                    "identities": ("A", "B"),
                },
            ),
        )
        self.assertEqual(second["start_split"], 5)
        self.assertEqual(second["end_split"], 5)
        # Identities follow the position's drift record order, not the
        # streak order: A then C, with B absent from the coverage.
        self.assertEqual(
            second["points"][0]["identities"], ("A", "C")
        )
        self.assertEqual(
            result["totals"],
            {
                "splits": 2,
                "identities": 3,
                "waves": 2,
                "points": 2,
                "peak_active": 2,
            },
        )

    def test_wave_builder_threshold_and_empty_inputs(self):
        scans, streaks = synthetic_scans()
        result = BranchStore._build_lifetime_churn_forecast_drift_waves_result(
            scans, streaks, 3, 10
        )
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
        result = BranchStore._build_lifetime_churn_forecast_drift_waves_result(
            (), (), 1, 1
        )
        self.assertEqual(result["waves"], ())
        self.assertEqual(result["totals"]["waves"], 0)


class LifetimeChurnForecastDriftWavesValidationTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        overrides.setdefault("causes", ("a0",))
        return waves_call(self.store, self.args, **overrides)

    def extended(self, **overrides):
        overrides.setdefault("split_cutoffs", (4,))
        merged = dict(LONG_ARGS)
        merged.update(overrides)
        return self.call(**merged)

    def test_min_active_identities_must_be_a_non_bool_positive_int(self):
        for bad in (True, False, 1.0, "1", None, (1,)):
            with self.subTest(min_active_identities=bad):
                with self.assertRaises(TypeError):
                    self.extended(min_active_identities=bad)
        with self.assertRaises(ValueError):
            self.extended(min_active_identities=0)
        with self.assertRaises(ValueError):
            self.extended(min_active_identities=-2)

    def test_wave_result_limit_must_be_a_non_bool_positive_int(self):
        for bad in (True, False, 1.0, "1", None, (1,)):
            with self.subTest(wave_result_limit=bad):
                with self.assertRaises(TypeError):
                    self.extended(wave_result_limit=bad)
        with self.assertRaises(ValueError):
            self.extended(wave_result_limit=0)
        with self.assertRaises(ValueError):
            self.extended(wave_result_limit=-2)

    def test_new_parameters_validated_in_signature_order(self):
        # scan_limit fails before min_splits.
        with self.assertRaises(ValueError):
            self.extended(scan_limit=0, min_splits="x")
        # min_splits fails before streak_result_limit.
        with self.assertRaises(TypeError):
            self.extended(min_splits="x", streak_result_limit="x")
        with self.assertRaises(ValueError):
            self.extended(min_splits=0, streak_result_limit=0)
        # streak_result_limit fails before min_active_identities.
        with self.assertRaises(TypeError):
            self.extended(streak_result_limit="x", min_active_identities="x")
        with self.assertRaises(ValueError):
            self.extended(streak_result_limit=0, min_active_identities=0)
        # min_active_identities fails before wave_result_limit.
        with self.assertRaises(TypeError):
            self.extended(min_active_identities="x", wave_result_limit="x")
        with self.assertRaises(ValueError):
            self.extended(min_active_identities=0, wave_result_limit=0)
        # wave_result_limit fails before the split membership check.
        with self.assertRaises(ValueError):
            self.extended(split_cutoffs=(99,), wave_result_limit=0)
        with self.assertRaises(TypeError):
            self.extended(split_cutoffs=(99,), wave_result_limit="x")
        # Only after every parameter passes does membership fire.
        with self.assertRaises(ValueError):
            self.extended(
                split_cutoffs=(99,),
                min_active_identities=1,
                wave_result_limit=1,
            )

    def test_all_parameter_checks_precede_state_lookups(self):
        with self.assertRaises(TypeError):
            self.call(
                reference="ghost", cutoffs=(0, 1), min_active_identities="x"
            )
        with self.assertRaises(ValueError):
            self.call(
                reference="ghost", cutoffs=(0, 1), min_active_identities=0
            )
        with self.assertRaises(TypeError):
            self.call(
                reference="ghost", cutoffs=(0, 1), wave_result_limit="x"
            )
        with self.assertRaises(ValueError):
            self.call(reference="ghost", cutoffs=(0, 1), wave_result_limit=0)
        with self.assertRaises(ValueError):
            self.call(
                token="whatever", cutoffs=(0, 1), split_cutoffs=(5,)
            )

    def test_unknown_branch_history_node_and_cause_raise(self):
        with self.assertRaises(KeyError):
            self.call(
                reference="ghost", cutoffs=(0, 1), split_cutoffs=(1,)
            )
        with self.assertRaises(KeyError):
            self.call(causes=("zzz",), cutoffs=(0, 1), split_cutoffs=(1,))


class LifetimeChurnForecastDriftWavesLimitTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        return waves_call(self.store, self.args, **overrides)

    def call_args(self):
        return dict(LONG_ARGS, split_cutoffs=(3, 4, 5))

    def test_streak_result_limit_still_bounds_the_streak_stage(self):
        # Three identities streak over 3-5, so a cap of two fails
        # before any wave is built.
        with self.assertRaises(ValueError) as caught:
            self.call(**self.call_args(), streak_result_limit=2)
        self.assertIn("streak result limit", str(caught.exception))
        result = self.call(**self.call_args(), streak_result_limit=3)
        self.assertEqual(result["totals"]["waves"], 1)

    def test_drift_and_scan_limits_still_bound_the_scan_stage(self):
        with self.assertRaises(ValueError) as caught:
            self.call(**self.call_args(), drift_limit=2)
        self.assertIn("drift limit", str(caught.exception))
        with self.assertRaises(ValueError) as caught:
            self.call(**self.call_args(), scan_limit=1)
        self.assertIn("scan limit", str(caught.exception))

    def test_wave_result_limit_bounds_waves_without_truncating(self):
        scans, streaks = synthetic_scans()
        with self.assertRaises(ValueError) as caught:
            BranchStore._build_lifetime_churn_forecast_drift_waves_result(
                scans, streaks, 2, 1
            )
        self.assertIn("wave result limit", str(caught.exception))
        result = BranchStore._build_lifetime_churn_forecast_drift_waves_result(
            scans, streaks, 2, 2
        )
        self.assertEqual(result["totals"]["waves"], 2)


class LifetimeChurnForecastDriftWavesIsolationTests(unittest.TestCase):
    def call_args(self):
        return dict(LONG_ARGS, split_cutoffs=(3, 4, 5))

    def test_result_levels_are_mutually_unshared(self):
        store, args = diamond_store()
        result = waves_call(store, args, **self.call_args())
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
        first = waves_call(store, args, **self.call_args())
        pristine = copy.deepcopy(first)
        for wave in first["waves"]:
            for point in wave["points"]:
                point["active_count"] = 99
            wave["length"] = 99
        first["totals"]["waves"] = 99
        self.assertEqual(
            waves_call(store, args, **self.call_args()), pristine
        )

    def test_success_and_failure_change_nothing(self):
        store, args = diamond_store()
        heads = dict(store._heads)
        appends = copy.deepcopy(store._appends)
        merges = copy.deepcopy(store._merges)
        audit = store.audit_log()

        waves_call(store, args, **self.call_args())
        failures = [
            dict(windows="x"),
            dict(scorecard_limit=0),
            dict(split_cutoffs="x"),
            dict(min_splits=0),
            dict(streak_result_limit=0),
            dict(min_active_identities=0),
            dict(min_active_identities="x"),
            dict(min_active_identities=True),
            dict(wave_result_limit=0),
            dict(wave_result_limit="x"),
            dict(wave_result_limit=False),
            dict(streak_result_limit=2),
            dict(drift_limit=2),
            dict(scan_limit=1),
            dict(wave_limit=1),
            dict(cutoffs=(2, 99)),
            dict(backtest_limit=1),
            dict(causes=("zzz",)),
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


class LifetimeChurnForecastDriftWavesTokenTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, token=None, **overrides):
        return waves_call(self.store, self.args, token=token, **overrides)

    def call_args(self):
        return dict(LONG_ARGS, split_cutoffs=(3, 4, 5))

    def test_success_consumes_one_read_and_failure_refunds(self):
        token = self.store.create_snapshot(2)
        self.call(token=token, **self.call_args())
        # Over the per-scan drift cap: failure refunds.
        with self.assertRaises(ValueError):
            self.call(
                token=token, **dict(self.call_args(), drift_limit=2)
            )
        # Over the total scan cap: failure refunds.
        with self.assertRaises(ValueError):
            self.call(
                token=token, **dict(self.call_args(), scan_limit=1)
            )
        # Over the streak result cap: failure refunds.
        with self.assertRaises(ValueError):
            self.call(
                token=token, **dict(self.call_args(), streak_result_limit=2)
            )
        # Over the backtest cap: failure refunds.
        with self.assertRaises(ValueError):
            self.call(
                token=token, **dict(self.call_args(), backtest_limit=1)
            )
        # State error: failure refunds.
        with self.assertRaises(KeyError):
            self.call(
                token=token, **dict(self.call_args(), causes=("zzz",))
            )
        # Parameter validation runs before the token is touched.
        with self.assertRaises(TypeError):
            self.call(
                token=token,
                **dict(self.call_args(), min_active_identities="x"),
            )
        with self.assertRaises(ValueError):
            self.call(
                token=token, **dict(self.call_args(), wave_result_limit=0)
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
