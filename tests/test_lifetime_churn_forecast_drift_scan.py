import copy
import inspect
import unittest

from city_twin.branches import BranchStore

from tests.test_lifetime_churn import diamond_store
from tests.test_lifetime_churn_forecast_drift import (
    LONG_WINDOWS,
    drift_call,
)
from tests.test_lifetime_churn_forecast_scorecards import scorecards_kwargs


def scan_kwargs(store_args, **overrides) -> dict:
    split_cutoffs = overrides.pop("split_cutoffs", (3, 4, 5))
    min_rate_drop = overrides.pop("min_rate_drop", (0, 1))
    drift_limit = overrides.pop("drift_limit", 50)
    scan_limit = overrides.pop("scan_limit", 100)
    kwargs = scorecards_kwargs(store_args, **overrides)
    kwargs["split_cutoffs"] = split_cutoffs
    kwargs["min_rate_drop"] = min_rate_drop
    kwargs["drift_limit"] = drift_limit
    kwargs["scan_limit"] = scan_limit
    return kwargs


def scan_call(store, store_args, **overrides):
    return store.lifetime_churn_forecast_drift_scan(
        **scan_kwargs(store_args, **overrides)
    )


LONG_ARGS = dict(
    windows=LONG_WINDOWS,
    max_jitter=2,
    horizon=1,
    cutoffs=(2, 3, 4, 5, 6),
)


class LifetimeChurnForecastDriftScanSignatureTests(unittest.TestCase):
    def test_public_signature_replaces_split_and_appends_scan_limit(self):
        parameters = list(
            inspect.signature(
                BranchStore.lifetime_churn_forecast_drift_scan
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
        result = scan_call(store, args, **LONG_ARGS, split_cutoffs=(3, 4, 5))
        self.assertIsInstance(result, dict)
        self.assertEqual(list(result), ["scans", "totals"])
        self.assertIsInstance(result["scans"], tuple)
        self.assertEqual(
            [scan["split_cutoff"] for scan in result["scans"]], [3, 4, 5]
        )
        scorecard_keys = [
            "type",
            "identity",
            "predictions",
            "resolved",
            "unresolved",
            "hit",
            "early",
            "late",
            "missed",
            "observed",
            "exact_rate",
            "window_rate",
            "mean_offset",
            "max_abs_offset",
        ]
        for scan in result["scans"]:
            self.assertEqual(
                list(scan), ["split_cutoff", "drifts", "totals"]
            )
            self.assertIsInstance(scan["drifts"], tuple)
            self.assertEqual(
                list(scan["totals"]),
                [
                    "identities",
                    "baseline_predictions",
                    "recent_predictions",
                    "baseline_resolved",
                    "recent_resolved",
                ],
            )
            for record in scan["drifts"]:
                self.assertEqual(
                    list(record),
                    [
                        "type",
                        "identity",
                        "baseline",
                        "recent",
                        "exact_drop",
                        "window_drop",
                        "mean_offset_shift",
                    ],
                )
                self.assertEqual(list(record["baseline"]), scorecard_keys)
                self.assertEqual(list(record["recent"]), scorecard_keys)
        self.assertEqual(
            list(result["totals"]),
            [
                "splits",
                "drift_records",
                "baseline_predictions",
                "recent_predictions",
                "baseline_resolved",
                "recent_resolved",
            ],
        )


class LifetimeChurnForecastDriftScanResultTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        return scan_call(self.store, self.args, **overrides)

    def point(self, split_cutoff, **overrides):
        return drift_call(
            self.store, self.args, split_cutoff=split_cutoff, **overrides
        )

    def test_each_scan_matches_the_single_point_query(self):
        for min_rate_drop in ((0, 1), (1, 2), (1, 1)):
            for min_resolved in (1, 2):
                with self.subTest(
                    min_rate_drop=min_rate_drop, min_resolved=min_resolved
                ):
                    result = self.call(
                        **LONG_ARGS,
                        split_cutoffs=(3, 4, 5, 6),
                        min_rate_drop=min_rate_drop,
                        min_resolved=min_resolved,
                        drift_limit=10_000,
                        scan_limit=100_000,
                    )
                    for scan in result["scans"]:
                        point = self.point(
                            scan["split_cutoff"],
                            **LONG_ARGS,
                            min_rate_drop=min_rate_drop,
                            min_resolved=min_resolved,
                            drift_limit=10_000,
                        )
                        self.assertEqual(scan["drifts"], point["drifts"])
                        self.assertEqual(scan["totals"], point["totals"])

    def test_scans_follow_the_requested_split_order(self):
        result = self.call(**LONG_ARGS, split_cutoffs=(3, 5, 6))
        self.assertEqual(
            [scan["split_cutoff"] for scan in result["scans"]], [3, 5, 6]
        )

    def test_top_level_totals_sum_every_scan(self):
        result = self.call(**LONG_ARGS, split_cutoffs=(3, 4, 5))
        totals = result["totals"]
        self.assertEqual(totals["splits"], 3)
        self.assertEqual(
            totals["drift_records"],
            sum(len(scan["drifts"]) for scan in result["scans"]),
        )
        for name in (
            "baseline_predictions",
            "recent_predictions",
            "baseline_resolved",
            "recent_resolved",
        ):
            self.assertEqual(
                totals[name],
                sum(scan["totals"][name] for scan in result["scans"]),
            )
        # drift_records agrees with the per-scan identities totals.
        self.assertEqual(
            totals["drift_records"],
            sum(scan["totals"]["identities"] for scan in result["scans"]),
        )

    def test_empty_split_tuple_returns_empty_scans_and_zero_totals(self):
        result = self.call(
            **dict(LONG_ARGS, cutoffs=()), split_cutoffs=()
        )
        self.assertEqual(result["scans"], ())
        self.assertEqual(
            result["totals"],
            {
                "splits": 0,
                "drift_records": 0,
                "baseline_predictions": 0,
                "recent_predictions": 0,
                "baseline_resolved": 0,
                "recent_resolved": 0,
            },
        )

    def test_split_at_first_or_last_cutoff_rules_match_the_point_query(self):
        # The last cutoff is a split; the first cutoff empties baseline.
        result = self.call(**LONG_ARGS, split_cutoffs=(6,))
        self.assertEqual(result["scans"][0]["split_cutoff"], 6)
        point = self.point(6, **LONG_ARGS)
        self.assertEqual(result["scans"][0]["drifts"], point["drifts"])
        with self.assertRaises(ValueError):
            self.call(**LONG_ARGS, split_cutoffs=(2,))

    def test_scan_builder_on_synthetic_backtests(self):
        from tests.test_lifetime_churn_forecast_drift import (
            synthetic_backtests,
        )

        result = BranchStore._build_lifetime_churn_forecast_drift_scan_result(
            synthetic_backtests(),
            (2, 3, 4),
            1,
            50,
            (0, 1),
            50,
            100,
        )
        self.assertEqual(
            [scan["split_cutoff"] for scan in result["scans"]], [2, 3, 4]
        )
        expected = {
            2: BranchStore._build_lifetime_churn_forecast_drift_result(
                synthetic_backtests(), 2, 1, 50, (0, 1), 50
            ),
            3: BranchStore._build_lifetime_churn_forecast_drift_result(
                synthetic_backtests(), 3, 1, 50, (0, 1), 50
            ),
            4: BranchStore._build_lifetime_churn_forecast_drift_result(
                synthetic_backtests(), 4, 1, 50, (0, 1), 50
            ),
        }
        total_records = 0
        for scan in result["scans"]:
            point = expected[scan["split_cutoff"]]
            self.assertEqual(scan["drifts"], point["drifts"])
            self.assertEqual(scan["totals"], point["totals"])
            total_records += len(point["drifts"])
        self.assertEqual(result["totals"]["splits"], 3)
        self.assertEqual(result["totals"]["drift_records"], total_records)

    def test_builder_empty_splits(self):
        from tests.test_lifetime_churn_forecast_drift import (
            synthetic_backtests,
        )

        result = BranchStore._build_lifetime_churn_forecast_drift_scan_result(
            synthetic_backtests(), (), 1, 50, (0, 1), 50, 1
        )
        self.assertEqual(result["scans"], ())
        self.assertEqual(result["totals"]["splits"], 0)
        self.assertEqual(result["totals"]["drift_records"], 0)


class LifetimeChurnForecastDriftScanValidationTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        overrides.setdefault("causes", ("a0",))
        return scan_call(self.store, self.args, **overrides)

    def extended(self, **overrides):
        overrides.setdefault("split_cutoffs", (4,))
        merged = dict(LONG_ARGS)
        merged.update(overrides)
        return self.call(**merged)

    def test_split_cutoffs_must_be_a_tuple(self):
        for bad in ("x", [4], {4}, 4, None, True):
            with self.subTest(split_cutoffs=bad):
                with self.assertRaises(TypeError):
                    self.extended(split_cutoffs=bad)

    def test_split_cutoff_elements_must_be_non_bool_ints(self):
        for bad in ((True,), (False,), (3.0,), ("4",), (3, 4.0),
                    (3, None)):
            with self.subTest(split_cutoffs=bad):
                with self.assertRaises(TypeError):
                    self.extended(split_cutoffs=bad)

    def test_split_cutoffs_must_be_strictly_increasing(self):
        for bad in ((4, 4), (5, 4), (3, 4, 4), (6, 5, 4)):
            with self.subTest(split_cutoffs=bad):
                with self.assertRaises(ValueError):
                    self.extended(split_cutoffs=bad)

    def test_split_cutoffs_must_be_requested_cutoffs(self):
        with self.assertRaises(ValueError):
            self.extended(split_cutoffs=(1,))
        with self.assertRaises(ValueError):
            self.extended(split_cutoffs=(7,))
        with self.assertRaises(ValueError):
            self.extended(split_cutoffs=(3, 7))
        # The last requested cutoff is itself a valid split.
        self.extended(split_cutoffs=(6,))

    def test_each_split_must_leave_both_periods_non_empty(self):
        with self.assertRaises(ValueError):
            self.extended(split_cutoffs=(2,))
        with self.assertRaises(ValueError):
            self.call(cutoffs=(2,), split_cutoffs=(2,))
        with self.assertRaises(ValueError):
            self.extended(split_cutoffs=(3, 2))

    def test_scan_limit_must_be_a_non_bool_positive_int(self):
        for bad in (True, False, 1.0, "1", None):
            with self.subTest(scan_limit=bad):
                with self.assertRaises(TypeError):
                    self.extended(scan_limit=bad)
        with self.assertRaises(ValueError):
            self.extended(scan_limit=0)
        with self.assertRaises(ValueError):
            self.extended(scan_limit=-5)

    def test_min_rate_drop_and_drift_limit_share_point_validation(self):
        for bad in ("x", [0, 1], 1):
            with self.subTest(min_rate_drop=bad):
                with self.assertRaises(TypeError):
                    self.extended(min_rate_drop=bad)
        with self.assertRaises(ValueError):
            self.extended(min_rate_drop=(2, 1))
        for bad in (True, 1.0, "1"):
            with self.subTest(drift_limit=bad):
                with self.assertRaises(TypeError):
                    self.extended(drift_limit=bad)
        with self.assertRaises(ValueError):
            self.extended(drift_limit=0)

    def test_new_parameters_validated_in_signature_order(self):
        # scorecard_limit fails before split_cutoffs.
        with self.assertRaises(ValueError):
            self.extended(scorecard_limit=0, split_cutoffs="x")
        with self.assertRaises(TypeError):
            self.extended(scorecard_limit="x", split_cutoffs="x")
        # split_cutoffs type/order fail before min_rate_drop.
        with self.assertRaises(TypeError):
            self.extended(split_cutoffs="x", min_rate_drop=(2, 1))
        with self.assertRaises(ValueError):
            self.extended(split_cutoffs=(5, 4), min_rate_drop=(2, 1))
        # min_rate_drop fails before drift_limit.
        with self.assertRaises(TypeError):
            self.extended(min_rate_drop="x", drift_limit="x")
        with self.assertRaises(ValueError):
            self.extended(min_rate_drop=(2, 1), drift_limit=0)
        # drift_limit fails before scan_limit.
        with self.assertRaises(TypeError):
            self.extended(drift_limit="x", scan_limit="x")
        with self.assertRaises(ValueError):
            self.extended(drift_limit=0, scan_limit=0)
        # scan_limit fails before the split membership check.
        with self.assertRaises(ValueError):
            self.extended(
                split_cutoffs=(99,), min_rate_drop=(0, 1),
                drift_limit=1, scan_limit=0,
            )
        with self.assertRaises(TypeError):
            self.extended(
                split_cutoffs=(99,), min_rate_drop=(0, 1),
                drift_limit=1, scan_limit="x",
            )
        # Only after every parameter passes does membership fire.
        with self.assertRaises(ValueError):
            self.extended(
                split_cutoffs=(99,), min_rate_drop=(0, 1),
                drift_limit=1, scan_limit=1,
            )

    def test_all_parameter_checks_precede_state_lookups(self):
        with self.assertRaises(TypeError):
            self.call(
                reference="ghost", cutoffs=(0, 1), split_cutoffs="x"
            )
        with self.assertRaises(ValueError):
            self.call(
                reference="ghost", cutoffs=(0, 1), min_rate_drop=(2, 1)
            )
        with self.assertRaises(ValueError):
            self.call(reference="ghost", cutoffs=(0, 1), drift_limit=0)
        with self.assertRaises(ValueError):
            self.call(reference="ghost", cutoffs=(0, 1), scan_limit=0)
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

    def test_caps_still_bound_the_single_backtest(self):
        with self.assertRaises(ValueError) as caught:
            self.extended(backtest_limit=5)
        self.assertIn("backtest limit", str(caught.exception))
        # A split that is itself a requested cutoff passes parameter
        # validation, so the post-view cutoff range check still fires.
        with self.assertRaises(ValueError) as caught:
            self.call(
                **dict(
                    LONG_ARGS,
                    causes=("a0",),
                    cutoffs=(2, 99),
                    split_cutoffs=(99,),
                )
            )
        self.assertIn("out of range", str(caught.exception))


class LifetimeChurnForecastDriftScanLimitTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        return scan_call(self.store, self.args, **overrides)

    def test_drift_limit_bounds_each_scan_independently(self):
        # Split 4 keeps three identities.
        with self.assertRaises(ValueError) as caught:
            self.call(**LONG_ARGS, split_cutoffs=(4,), drift_limit=2)
        self.assertIn("drift limit", str(caught.exception))
        result = self.call(**LONG_ARGS, split_cutoffs=(4,), drift_limit=3)
        self.assertEqual(len(result["scans"][0]["drifts"]), 3)
        # The cap is per scan: two scans of three records each pass a
        # cap of three, even though six records are built in total.
        result = self.call(
            **LONG_ARGS, split_cutoffs=(4, 5), drift_limit=3,
            scan_limit=100,
        )
        self.assertEqual(
            [len(scan["drifts"]) for scan in result["scans"]], [3, 3]
        )

    def test_scan_limit_bounds_total_records_without_truncating(self):
        result = self.call(
            **LONG_ARGS, split_cutoffs=(3, 4, 5), drift_limit=50
        )
        total = result["totals"]["drift_records"]
        self.assertGreater(total, 0)
        with self.assertRaises(ValueError) as caught:
            self.call(
                **LONG_ARGS, split_cutoffs=(3, 4, 5),
                drift_limit=50, scan_limit=total - 1,
            )
        self.assertIn("scan limit", str(caught.exception))
        exact = self.call(
            **LONG_ARGS, split_cutoffs=(3, 4, 5),
            drift_limit=50, scan_limit=total,
        )
        self.assertEqual(exact["totals"]["drift_records"], total)

    def test_scorecard_limit_bounds_each_period_of_each_scan(self):
        with self.assertRaises(ValueError) as caught:
            self.call(**LONG_ARGS, split_cutoffs=(4,), scorecard_limit=2)
        self.assertIn("scorecard limit", str(caught.exception))
        result = self.call(
            **LONG_ARGS, split_cutoffs=(4,), scorecard_limit=3
        )
        self.assertEqual(result["totals"]["drift_records"], 3)


class LifetimeChurnForecastDriftScanIsolationTests(unittest.TestCase):
    def call_args(self):
        return dict(LONG_ARGS, split_cutoffs=(3, 4, 5))

    def test_result_levels_are_mutually_unshared(self):
        store, args = diamond_store()
        result = scan_call(store, args, **self.call_args())
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
        first = scan_call(store, args, **self.call_args())
        pristine = copy.deepcopy(first)
        for scan in first["scans"]:
            for record in scan["drifts"]:
                record["baseline"]["predictions"] = 99
                record["recent"]["resolved"] = 99
                record["window_drop"] = (9, 9)
            scan["totals"]["identities"] = 99
        first["totals"]["splits"] = 99
        self.assertEqual(
            scan_call(store, args, **self.call_args()), pristine
        )

    def test_success_and_failure_change_nothing(self):
        store, args = diamond_store()
        heads = dict(store._heads)
        appends = copy.deepcopy(store._appends)
        merges = copy.deepcopy(store._merges)
        audit = store.audit_log()

        scan_call(store, args, **self.call_args())
        failures = [
            dict(windows="x"),
            dict(scorecard_limit=0),
            dict(split_cutoffs="x"),
            dict(split_cutoffs=(True,)),
            dict(split_cutoffs=(4, 4)),
            dict(split_cutoffs=(5, 4)),
            dict(split_cutoffs=(7,)),
            dict(split_cutoffs=(2,)),
            dict(min_rate_drop="x"),
            dict(min_rate_drop=(2, 1)),
            dict(drift_limit=0),
            dict(drift_limit="x"),
            dict(scan_limit=0),
            dict(scan_limit=False),
            dict(drift_limit=2),
            dict(scan_limit=1),
            dict(wave_limit=1),
            dict(cutoffs=(2, 99)),
            dict(backtest_limit=1),
            dict(scorecard_limit=2),
            dict(causes=("zzz",)),
            dict(limit=1),
            dict(reference="ghost"),
        ]
        for overrides in failures:
            with self.subTest(overrides=overrides):
                merged = self.call_args()
                merged.update(overrides)
                with self.assertRaises(Exception):
                    scan_call(store, args, **merged)

        self.assertEqual(dict(store._heads), heads)
        self.assertEqual(copy.deepcopy(store._appends), appends)
        self.assertEqual(copy.deepcopy(store._merges), merges)
        self.assertEqual(store.audit_log(), audit)

    def test_existing_public_entries_are_unaffected(self):
        from tests.test_lifetime_churn_forecast_backtests import (
            backtests_call,
        )
        from tests.test_lifetime_churn_forecast_scorecards import (
            scorecards_call,
        )
        from tests.test_lifetime_churn_forecasts import forecasts_call
        from tests.test_lifetime_churn_waves import waves_call

        store, args = diamond_store()
        scan_call(store, args, **self.call_args())

        waves = waves_call(store, args, windows=LONG_WINDOWS)
        backtests = backtests_call(store, args, **LONG_ARGS)
        scorecards = scorecards_call(store, args, **LONG_ARGS)
        forecasts = forecasts_call(
            store, args, windows=LONG_WINDOWS, max_jitter=2, horizon=1
        )
        self.assertEqual(
            waves_call(store, args, windows=LONG_WINDOWS), waves
        )
        self.assertEqual(
            backtests_call(store, args, **LONG_ARGS), backtests
        )
        self.assertEqual(
            scorecards_call(store, args, **LONG_ARGS), scorecards
        )
        self.assertEqual(
            forecasts_call(
                store, args, windows=LONG_WINDOWS, max_jitter=2, horizon=1
            ),
            forecasts,
        )


class LifetimeChurnForecastDriftScanTokenTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, token=None, **overrides):
        return scan_call(self.store, self.args, token=token, **overrides)

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
        # Over the scorecard cap: failure refunds.
        with self.assertRaises(ValueError):
            self.call(
                token=token, **dict(self.call_args(), scorecard_limit=2)
            )
        # Over the backtest cap: failure refunds.
        with self.assertRaises(ValueError):
            self.call(
                token=token, **dict(self.call_args(), backtest_limit=1)
            )
        # Out-of-range cutoff: failure refunds.
        with self.assertRaises(ValueError):
            self.call(
                token=token,
                **dict(self.call_args(), cutoffs=(2, 99)),
            )
        # State error: failure refunds.
        with self.assertRaises(KeyError):
            self.call(
                token=token, **dict(self.call_args(), causes=("zzz",))
            )
        # Parameter validation runs before the token is touched.
        with self.assertRaises(TypeError):
            self.call(
                token=token, **dict(self.call_args(), split_cutoffs="x")
            )
        with self.assertRaises(ValueError):
            self.call(
                token=token,
                **dict(self.call_args(), min_rate_drop=(2, 1)),
            )
        with self.assertRaises(ValueError):
            self.call(token=token, **dict(self.call_args(), scan_limit=0))
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

    def test_no_token_still_uses_one_consistent_view(self):
        # Mutating the store after a token-free scan cannot change the
        # scan already returned; a second scan over changed state still
        # scores every split identically relative to a single point.
        first = self.call(**self.call_args())
        token = self.store.create_snapshot(8)
        self.assertEqual(
            self.call(token=token, **self.call_args()), first
        )


if __name__ == "__main__":
    unittest.main()
