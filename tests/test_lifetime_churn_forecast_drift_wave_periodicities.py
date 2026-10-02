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
    periodicity_result_limit = overrides.pop(
        "periodicity_result_limit", 50
    )
    kwargs = waves_kwargs(store_args, **overrides)
    kwargs["min_wave_occurrences"] = min_wave_occurrences
    kwargs["recurrence_result_limit"] = recurrence_result_limit
    kwargs["max_wave_jitter"] = max_wave_jitter
    kwargs["periodicity_result_limit"] = periodicity_result_limit
    return kwargs


def periodicities_call(store, store_args, **overrides):
    return store.lifetime_churn_forecast_drift_wave_periodicities(
        **periodicities_kwargs(store_args, **overrides)
    )


def synthetic_waves():
    """Eight waves pinning participation, cadence and point counting.

    A appears in every wave (and twice in wave 2), B in waves 0-3, C
    in waves 0, 2, 4, D in waves 3-6, G in waves 0, 1, 4, 6, 7, E in
    waves 0, 1, 7 and F only in wave 1. Wave 2 lists A and B at two
    of its points, so each takes part once but covers two positions.
    """
    membership = (
        ("A", "B", "C", "E", "G"),
        ("A", "B", "E", "F", "G"),
        ("A", "B", "C"),
        ("A", "B", "D"),
        ("A", "C", "D", "G"),
        ("A", "D"),
        ("A", "D", "G"),
        ("A", "E", "G"),
    )
    waves = []
    for serial, identities in enumerate(membership):
        points = [
            {
                "split_cutoff": 10 + serial,
                "active_count": len(identities),
                "identities": tuple(identities),
            }
        ]
        if serial == 2:
            points.append(
                {
                    "split_cutoff": 99,
                    "active_count": 2,
                    "identities": ("A", "B"),
                }
            )
        waves.append(
            {
                "start_split": points[0]["split_cutoff"],
                "end_split": points[-1]["split_cutoff"],
                "length": len(points),
                "peak_identities": max(
                    point["active_count"] for point in points
                ),
                "points": tuple(points),
            }
        )
    return tuple(waves)


def build(waves, max_wave_jitter=5, periodicity_result_limit=50):
    return (
        BranchStore
        ._build_lifetime_churn_forecast_drift_wave_periodicities_result(
            waves, max_wave_jitter, periodicity_result_limit
        )
    )


class LifetimeChurnForecastDriftWavePeriodicitiesSignatureTests(
    unittest.TestCase
):
    def test_public_signature_appends_two_parameters(self):
        parameters = list(
            inspect.signature(
                BranchStore
                .lifetime_churn_forecast_drift_wave_periodicities
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
                "token",
            ],
        )
        self.assertEqual(
            [p.default for p in parameters[:-1]],
            [inspect.Parameter.empty] * (len(parameters) - 1),
        )
        self.assertIsNone(parameters[-1].default)

    def test_result_shape_and_key_order(self):
        result = build(synthetic_waves())
        self.assertIsInstance(result, dict)
        self.assertEqual(list(result), ["periodicities", "totals"])
        self.assertIsInstance(result["periodicities"], tuple)
        for record in result["periodicities"]:
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
                ],
            )
            self.assertIsInstance(record["appearances"], tuple)
            for appearance in record["appearances"]:
                self.assertEqual(
                    list(appearance),
                    [
                        "wave",
                        "start_split",
                        "end_split",
                        "points",
                        "first_split",
                        "last_split",
                    ],
                )
            self.assertIsInstance(record["intervals"], tuple)
        self.assertEqual(
            list(result["totals"]),
            [
                "periodicities",
                "waves",
                "points",
                "first_wave",
                "last_wave",
                "mean_period",
            ],
        )


class LifetimeChurnForecastDriftWavePeriodicitiesResultTests(
    unittest.TestCase
):
    def setUp(self):
        self.waves = synthetic_waves()

    def test_three_wave_minimum_and_interval_fields(self):
        result = build(self.waves)
        by_identity = {
            record["identity"]: record
            for record in result["periodicities"]
        }
        # F appears in one wave only and keeps no record.
        self.assertNotIn("F", by_identity)
        # C appears exactly three waves two serials apart.
        c_record = by_identity["C"]
        self.assertEqual(c_record["waves"], 3)
        self.assertEqual(c_record["first_wave"], 0)
        self.assertEqual(c_record["last_wave"], 4)
        self.assertEqual(c_record["span"], 5)
        self.assertEqual(c_record["intervals"], (2, 2))
        self.assertEqual(c_record["period"], 2)
        self.assertEqual(c_record["jitter"], 0)

    def test_points_dedup_within_a_wave_but_count_positions(self):
        by_identity = {
            record["identity"]: record
            for record in build(self.waves)["periodicities"]
        }
        a_record = by_identity["A"]
        self.assertEqual(a_record["waves"], 8)
        # Eight participating waves, two positions covered in wave 2.
        self.assertEqual(a_record["points"], 9)
        self.assertEqual(a_record["peak_active"], 5)
        wave_two = a_record["appearances"][2]
        self.assertEqual(wave_two["wave"], 2)
        self.assertEqual(wave_two["points"], 2)
        self.assertEqual(wave_two["first_split"], 12)
        self.assertEqual(wave_two["last_split"], 99)

    def test_median_takes_lower_middle_for_even_interval_counts(self):
        by_identity = {
            record["identity"]: record
            for record in build(self.waves)["periodicities"]
        }
        # G intervals in appearance order are 1, 3, 2, 1; sorted they
        # are 1, 1, 2, 3 and the lower middle value is 1.
        g_record = by_identity["G"]
        self.assertEqual(g_record["intervals"], (1, 3, 2, 1))
        self.assertEqual(g_record["period"], 1)
        self.assertEqual(g_record["jitter"], 2)
        e_record = by_identity["E"]
        self.assertEqual(e_record["intervals"], (1, 6))
        self.assertEqual(e_record["period"], 1)
        self.assertEqual(e_record["jitter"], 5)

    def test_jitter_filter_keeps_only_steady_cadences(self):
        identities = lambda cap: [
            record["identity"]
            for record in build(self.waves, max_wave_jitter=cap)[
                "periodicities"
            ]
        ]
        self.assertEqual(identities(0), ["A", "B", "D", "C"])
        self.assertEqual(identities(2), ["A", "B", "D", "C", "G"])
        self.assertEqual(identities(5), ["A", "B", "D", "C", "G", "E"])

    def test_records_sort_by_jitter_then_waves_points_first_encounter(self):
        result = build(self.waves)
        ordered = [
            (
                record["identity"],
                record["jitter"],
                record["waves"],
                record["points"],
                record["first_wave"],
            )
            for record in result["periodicities"]
        ]
        self.assertEqual(
            ordered,
            [
                ("A", 0, 8, 9, 0),
                ("B", 0, 4, 5, 0),
                ("D", 0, 4, 4, 3),
                ("C", 0, 3, 3, 0),
                ("G", 2, 5, 5, 0),
                ("E", 5, 3, 3, 0),
            ],
        )

    def test_totals_sum_records_and_floor_the_mean_period(self):
        self.assertEqual(
            build(self.waves)["totals"],
            {
                "periodicities": 6,
                "waves": 27,
                "points": 29,
                "first_wave": 0,
                "last_wave": 7,
                # Periods 1, 1, 1, 2, 1, 1 sum to 7; floor(7 / 6).
                "mean_period": 1,
            },
        )

    def test_first_eight_fields_match_recurrences_semantics(self):
        periodicities = build(self.waves)["periodicities"]
        recurrences = (
            BranchStore
            ._build_lifetime_churn_forecast_drift_wave_recurrences_result(
                self.waves, 1, 50
            )["recurrences"]
        )
        recurrence_of = {
            record["identity"]: record for record in recurrences
        }
        for record in periodicities:
            expected = recurrence_of[record["identity"]]
            for field in (
                "identity",
                "waves",
                "first_wave",
                "last_wave",
                "span",
                "points",
                "peak_active",
                "appearances",
            ):
                self.assertEqual(record[field], expected[field])

    def test_limit_checks_the_complete_candidate_set(self):
        with self.assertRaises(ValueError) as caught:
            build(self.waves, max_wave_jitter=5, periodicity_result_limit=5)
        self.assertIn("periodicity result limit", str(caught.exception))
        result = build(
            self.waves, max_wave_jitter=5, periodicity_result_limit=6
        )
        self.assertEqual(len(result["periodicities"]), 6)
        # The cap counts the jitter-filtered set, not every recurrence.
        result = build(
            self.waves, max_wave_jitter=0, periodicity_result_limit=4
        )
        self.assertEqual(len(result["periodicities"]), 4)

    def test_empty_waves_return_empty_tuple_and_six_zero_totals(self):
        self.assertEqual(
            build(()),
            {
                "periodicities": (),
                "totals": {
                    "periodicities": 0,
                    "waves": 0,
                    "points": 0,
                    "first_wave": 0,
                    "last_wave": 0,
                    "mean_period": 0,
                },
            },
        )


class LifetimeChurnForecastDriftWavePeriodicitiesEntryTests(
    unittest.TestCase
):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        return periodicities_call(self.store, self.args, **overrides)

    def call_args(self):
        return dict(LONG_ARGS, split_cutoffs=(3, 4, 5, 6))

    def test_single_qualifying_wave_recurs_fewer_than_three_times(self):
        result = self.call(**self.call_args())
        self.assertEqual(result["periodicities"], ())
        self.assertEqual(
            result["totals"],
            {
                "periodicities": 0,
                "waves": 0,
                "points": 0,
                "first_wave": 0,
                "last_wave": 0,
                "mean_period": 0,
            },
        )

    def test_empty_split_tuple_returns_empty_periodicities(self):
        result = self.call(**dict(LONG_ARGS, cutoffs=()), split_cutoffs=())
        self.assertEqual(result["periodicities"], ())
        self.assertEqual(result["totals"]["periodicities"], 0)
        self.assertEqual(result["totals"]["mean_period"], 0)

    def test_recurrence_parameters_are_validated_but_not_binding(self):
        # A demanding recurrence threshold and tight cap neither filter
        # nor cap a periodicities query (the empty result here comes
        # from the single wave stage, never from three appearances).
        result = self.call(
            **self.call_args(),
            min_wave_occurrences=99,
            recurrence_result_limit=1,
        )
        self.assertEqual(result["periodicities"], ())
        # They still keep the recurrences entry's type and range rules.
        with self.assertRaises(TypeError):
            self.call(
                **self.call_args(), min_wave_occurrences="x"
            )
        with self.assertRaises(ValueError):
            self.call(**self.call_args(), recurrence_result_limit=0)

    def test_all_stages_still_apply_their_own_caps(self):
        with self.assertRaises(ValueError) as caught:
            self.call(**self.call_args(), drift_limit=2)
        self.assertIn("drift limit", str(caught.exception))
        with self.assertRaises(ValueError) as caught:
            self.call(**self.call_args(), scan_limit=1)
        self.assertIn("scan limit", str(caught.exception))
        with self.assertRaises(ValueError) as caught:
            self.call(**self.call_args(), wave_result_limit=0)
        self.assertIn("wave_result_limit", str(caught.exception))


class LifetimeChurnForecastDriftWavePeriodicitiesValidationTests(
    unittest.TestCase
):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        overrides.setdefault("causes", ("a0",))
        return periodicities_call(self.store, self.args, **overrides)

    def extended(self, **overrides):
        overrides.setdefault("split_cutoffs", (4,))
        merged = dict(LONG_ARGS)
        merged.update(overrides)
        return self.call(**merged)

    def test_max_wave_jitter_must_be_a_non_bool_non_negative_int(self):
        for bad in (True, False, 1.0, "1", None, (1,)):
            with self.subTest(max_wave_jitter=bad):
                with self.assertRaises(TypeError):
                    self.extended(max_wave_jitter=bad)
        with self.assertRaises(ValueError):
            self.extended(max_wave_jitter=-1)
        # Zero is a legal, maximally strict cadence.
        result = self.extended(max_wave_jitter=0)
        self.assertEqual(result["periodicities"], ())

    def test_periodicity_result_limit_must_be_a_non_bool_positive_int(self):
        for bad in (True, False, 1.0, "1", None, (1,)):
            with self.subTest(periodicity_result_limit=bad):
                with self.assertRaises(TypeError):
                    self.extended(periodicity_result_limit=bad)
        with self.assertRaises(ValueError):
            self.extended(periodicity_result_limit=0)
        with self.assertRaises(ValueError):
            self.extended(periodicity_result_limit=-2)

    def test_new_parameters_validated_in_signature_order(self):
        # recurrence_result_limit fails before max_wave_jitter.
        with self.assertRaises(TypeError):
            self.extended(
                recurrence_result_limit="x", max_wave_jitter="x"
            )
        with self.assertRaises(ValueError):
            self.extended(
                recurrence_result_limit=0, max_wave_jitter=-1
            )
        # max_wave_jitter fails before periodicity_result_limit.
        with self.assertRaises(TypeError):
            self.extended(
                max_wave_jitter="x", periodicity_result_limit="x"
            )
        with self.assertRaises(ValueError):
            self.extended(
                max_wave_jitter=-1, periodicity_result_limit=0
            )
        # periodicity_result_limit fails before the split membership
        # check.
        with self.assertRaises(ValueError):
            self.extended(
                split_cutoffs=(99,), periodicity_result_limit=0
            )
        with self.assertRaises(TypeError):
            self.extended(
                split_cutoffs=(99,), periodicity_result_limit="x"
            )
        # Only after every parameter passes does membership fire.
        with self.assertRaises(ValueError):
            self.extended(
                split_cutoffs=(99,),
                max_wave_jitter=0,
                periodicity_result_limit=1,
            )

    def test_all_parameter_checks_precede_state_lookups(self):
        with self.assertRaises(TypeError):
            self.call(
                reference="ghost",
                cutoffs=(0, 1),
                max_wave_jitter="x",
            )
        with self.assertRaises(ValueError):
            self.call(
                reference="ghost",
                cutoffs=(0, 1),
                max_wave_jitter=-1,
            )
        with self.assertRaises(TypeError):
            self.call(
                reference="ghost",
                cutoffs=(0, 1),
                periodicity_result_limit="x",
            )
        with self.assertRaises(ValueError):
            self.call(
                reference="ghost",
                cutoffs=(0, 1),
                periodicity_result_limit=0,
            )

    def test_unknown_branch_history_node_and_cause_raise(self):
        with self.assertRaises(KeyError):
            self.call(
                reference="ghost", cutoffs=(0, 1), split_cutoffs=(1,)
            )
        with self.assertRaises(KeyError):
            self.call(causes=("zzz",), cutoffs=(0, 1), split_cutoffs=(1,))


class LifetimeChurnForecastDriftWavePeriodicitiesIsolationTests(
    unittest.TestCase
):
    def test_result_levels_are_mutually_unshared(self):
        result = build(synthetic_waves())
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

    def test_result_shares_nothing_with_the_wave_records(self):
        waves = synthetic_waves()
        result = build(waves)
        wave_ids = set()
        record_ids = set()

        def collect(value, bucket):
            if isinstance(value, dict):
                bucket.add(id(value))
                for item in value.values():
                    collect(item, bucket)
            elif isinstance(value, (tuple, list)):
                bucket.add(id(value))
                for item in value:
                    collect(item, bucket)

        collect(waves, wave_ids)
        collect(result, record_ids)
        self.assertFalse(wave_ids & record_ids)

    def test_mutating_result_never_affects_later_calls(self):
        waves = synthetic_waves()
        first = build(waves)
        pristine = copy.deepcopy(first)
        for record in first["periodicities"]:
            record["jitter"] = 99
            record["intervals"] = (99,)
            for appearance in record["appearances"]:
                appearance["points"] = 99
        first["totals"]["mean_period"] = 99
        self.assertEqual(build(waves), pristine)

    def test_success_and_failure_change_nothing(self):
        store, args = diamond_store()
        call_args = dict(LONG_ARGS, split_cutoffs=(3, 4, 5))
        heads = dict(store._heads)
        appends = copy.deepcopy(store._appends)
        merges = copy.deepcopy(store._merges)
        audit = store.audit_log()

        periodicities_call(store, args, **call_args)
        failures = [
            dict(windows="x"),
            dict(split_cutoffs="x"),
            dict(max_wave_jitter="x"),
            dict(max_wave_jitter=True),
            dict(max_wave_jitter=-1),
            dict(periodicity_result_limit="x"),
            dict(periodicity_result_limit=False),
            dict(periodicity_result_limit=0),
            dict(recurrence_result_limit="x"),
            dict(recurrence_result_limit=0),
            dict(streak_result_limit=2),
            dict(drift_limit=2),
            dict(scan_limit=1),
            dict(wave_limit=1),
            dict(cutoffs=(2, 99)),
            dict(causes=("zzz",)),
            dict(reference="ghost"),
        ]
        for overrides in failures:
            with self.subTest(overrides=overrides):
                merged = periodicities_kwargs(args, **call_args)
                merged.update(overrides)
                with self.assertRaises(Exception):
                    store.lifetime_churn_forecast_drift_wave_periodicities(
                        **merged
                    )

        self.assertEqual(dict(store._heads), heads)
        self.assertEqual(copy.deepcopy(store._appends), appends)
        self.assertEqual(copy.deepcopy(store._merges), merges)
        self.assertEqual(store.audit_log(), audit)


class LifetimeChurnForecastDriftWavePeriodicitiesTokenTests(
    unittest.TestCase
):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, token=None, **overrides):
        return periodicities_call(
            self.store, self.args, token=token, **overrides
        )

    def call_args(self):
        return dict(LONG_ARGS, split_cutoffs=(3, 4, 5))

    def test_success_consumes_one_read_and_failure_refunds(self):
        token = self.store.create_snapshot(2)
        self.call(token=token, **self.call_args())
        with self.assertRaises(ValueError):
            self.call(
                token=token, **dict(self.call_args(), drift_limit=2)
            )
        with self.assertRaises(ValueError):
            self.call(
                token=token, **dict(self.call_args(), scan_limit=1)
            )
        with self.assertRaises(ValueError):
            self.call(
                token=token, **dict(self.call_args(), backtest_limit=1)
            )
        with self.assertRaises(KeyError):
            self.call(
                token=token, **dict(self.call_args(), causes=("zzz",))
            )
        # Parameter validation runs before the token is touched.
        with self.assertRaises(TypeError):
            self.call(
                token=token, **dict(self.call_args(), max_wave_jitter="x")
            )
        with self.assertRaises(ValueError):
            self.call(
                token=token,
                **dict(self.call_args(), periodicity_result_limit=0),
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
