import copy
import inspect
import unittest

from city_twin.branches import BranchStore

from tests.test_lifetime_churn import diamond_store
from tests.test_lifetime_churn_forecast_drift_scan import (
    LONG_ARGS,
    scan_call,
    scan_kwargs,
)


def waves_kwargs(store_args, **overrides) -> dict:
    min_splits = overrides.pop("min_splits", 2)
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
    kwargs = waves_kwargs(store_args, **overrides)
    kwargs.pop("min_active_identities")
    kwargs.pop("wave_result_limit")
    return store.lifetime_churn_forecast_drift_streaks(**kwargs)


def reference_waves(
    scans, streaks, min_active_identities, wave_result_limit
):
    """Independently overlay kept streaks into maximal active waves."""
    covering = {scan["split_cutoff"]: set() for scan in scans}
    for streak in streaks:
        for point in streak["points"]:
            covering[point["split_cutoff"]].add(streak["identity"])

    active_points = []
    for scan in scans:
        identities = []
        for record in scan["drifts"]:
            identity = record["identity"]
            if identity in covering[scan["split_cutoff"]]:
                if identity not in identities:
                    identities.append(identity)
        active_points.append(
            {
                "split_cutoff": scan["split_cutoff"],
                "active_count": len(identities),
                "identities": tuple(identities),
            }
        )

    waves = []
    current = []
    for point in active_points:
        if point["active_count"] >= min_active_identities:
            current.append(point)
        elif current:
            waves.append(current)
            current = []
    if current:
        waves.append(current)

    if len(waves) > wave_result_limit:
        raise ValueError("wave result limit exceeded")

    built = []
    split_seen = set()
    identity_seen = set()
    peak_active = 0
    point_count = 0
    for members in waves:
        wave = {
            "start_split": members[0]["split_cutoff"],
            "end_split": members[-1]["split_cutoff"],
            "length": len(members),
            "peak_identities": max(
                point["active_count"] for point in members
            ),
            "points": tuple(copy.deepcopy(point) for point in members),
        }
        built.append(wave)
        point_count += len(members)
        peak_active = max(peak_active, wave["peak_identities"])
        for point in members:
            split_seen.add(point["split_cutoff"])
            identity_seen.update(point["identities"])
    return {
        "waves": tuple(built),
        "totals": {
            "splits": len(split_seen),
            "identities": len(identity_seen),
            "waves": len(built),
            "points": point_count,
            "peak_active": peak_active,
        },
    }


def synthetic_scan(split_cutoff, identities):
    return {
        "split_cutoff": split_cutoff,
        "drifts": tuple(
            {
                "type": "node",
                "identity": identity,
                "baseline": None,
                "recent": None,
                "exact_drop": (0, 1),
                "window_drop": (0, 1),
                "mean_offset_shift": None,
            }
            for identity in identities
        ),
        "totals": {},
    }


# Six requested positions.  Drift presence:
#   1: A          2: A, B       3: A, B, C     4: C, A
#   5: C          6: B, C
# so A drifts 1-4, B drifts 2-3 and 6, C drifts 3-6.
SYNTHETIC_SCANS = (
    synthetic_scan(1, ("A",)),
    synthetic_scan(2, ("A", "B")),
    synthetic_scan(3, ("A", "B", "C")),
    synthetic_scan(4, ("C", "A")),
    synthetic_scan(5, ("C",)),
    synthetic_scan(6, ("B", "C")),
)


class LifetimeChurnForecastDriftWavesSignatureTests(unittest.TestCase):
    def test_public_signature_appends_two_wave_parameters(self):
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
        result = waves_call(
            store, args, **LONG_ARGS, split_cutoffs=(3, 4, 5, 6)
        )
        self.assertIsInstance(result, dict)
        self.assertEqual(list(result), ["waves", "totals"])
        self.assertIsInstance(result["waves"], tuple)
        self.assertEqual(
            list(result["totals"]),
            ["splits", "identities", "waves", "points", "peak_active"],
        )
        for wave in result["waves"]:
            self.assertIsInstance(wave, dict)
            self.assertEqual(
                list(wave),
                ["start_split", "end_split", "length",
                 "peak_identities", "points"],
            )
            self.assertIsInstance(wave["points"], tuple)
            for point in wave["points"]:
                self.assertEqual(
                    list(point),
                    ["split_cutoff", "active_count", "identities"],
                )
                self.assertIsInstance(point["identities"], tuple)


class LifetimeChurnForecastDriftWavesResultTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        return waves_call(self.store, self.args, **overrides)

    def test_fixture_keeps_one_wave_over_every_active_split(self):
        result = self.call(
            **LONG_ARGS, split_cutoffs=(3, 4, 5, 6),
            min_splits=2, min_active_identities=3,
        )
        self.assertEqual(len(result["waves"]), 1)
        wave = result["waves"][0]
        self.assertEqual(
            (wave["start_split"], wave["end_split"], wave["length"]),
            (3, 6, 4),
        )
        self.assertEqual(wave["peak_identities"], 3)
        identities = (
            ("W", "y"),
            ("a0", "x"),
            (("a0", "x"), ("W", "y")),
        )
        self.assertEqual(
            [point["split_cutoff"] for point in wave["points"]],
            [3, 4, 5, 6],
        )
        for point in wave["points"]:
            self.assertEqual(point["active_count"], 3)
            self.assertEqual(point["identities"], identities)
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

    def test_threshold_above_every_count_returns_empty_waves(self):
        result = self.call(
            **LONG_ARGS, split_cutoffs=(3, 4, 5, 6),
            min_active_identities=4,
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

    def test_min_splits_filters_before_active_counts(self):
        # No identity can sustain five requested positions.
        result = self.call(
            **LONG_ARGS, split_cutoffs=(3, 4, 5, 6),
            min_splits=5, min_active_identities=1,
        )
        self.assertEqual(result["waves"], ())
        self.assertEqual(result["totals"]["waves"], 0)

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

    def test_matches_independent_overlay_of_scan_and_streaks(self):
        for split_cutoffs in ((3, 4, 5), (3, 5, 6), (4, 5), (6,)):
            for min_splits in (1, 2, 3):
                for min_active in (1, 2, 3, 4):
                    with self.subTest(
                        split_cutoffs=split_cutoffs,
                        min_splits=min_splits,
                        min_active=min_active,
                    ):
                        kwargs = dict(
                            LONG_ARGS,
                            split_cutoffs=split_cutoffs,
                            drift_limit=10_000,
                            scan_limit=100_000,
                            streak_result_limit=10_000,
                            wave_result_limit=10_000,
                            min_splits=min_splits,
                            min_active_identities=min_active,
                        )
                        result = self.call(**kwargs)
                        scan = self._scan(split_cutoffs)
                        streaks = self._streaks(split_cutoffs, min_splits)
                        self.assertEqual(
                            result,
                            reference_waves(
                                scan["scans"],
                                streaks["streaks"],
                                min_active,
                                10_000,
                            ),
                        )

    def _scan(self, split_cutoffs):
        return scan_call(
            self.store,
            self.args,
            **dict(
                LONG_ARGS,
                split_cutoffs=split_cutoffs,
                drift_limit=10_000,
                scan_limit=100_000,
            ),
        )

    def _streaks(self, split_cutoffs, min_splits):
        return BranchStore._build_lifetime_churn_forecast_drift_streaks_result(
            self._scan(split_cutoffs)["scans"], min_splits, 10_000
        )

    def test_identity_order_follows_the_position_drift_records(self):
        result = self.call(
            **LONG_ARGS, split_cutoffs=(3, 4, 5, 6),
            min_active_identities=1,
        )
        scan = self._scan((3, 4, 5, 6))
        for wave in result["waves"]:
            for point in wave["points"]:
                matching = next(
                    item for item in scan["scans"]
                    if item["split_cutoff"] == point["split_cutoff"]
                )
                self.assertEqual(
                    point["identities"],
                    tuple(
                        record["identity"]
                        for record in matching["drifts"]
                    ),
                )


class LifetimeChurnForecastDriftWavesBuilderTests(unittest.TestCase):
    def build(self, scans=SYNTHETIC_SCANS, min_splits=1, min_active=2,
              wave_result_limit=50):
        streaks = (
            BranchStore._build_lifetime_churn_forecast_drift_streaks_result(
                scans, min_splits, 10_000
            )["streaks"]
        )
        return BranchStore._build_lifetime_churn_forecast_drift_waves_result(
            scans, streaks, min_active, wave_result_limit
        )

    def test_threshold_gap_splits_into_two_maximal_waves(self):
        result = self.build()
        self.assertEqual(
            [
                (wave["start_split"], wave["end_split"], wave["length"])
                for wave in result["waves"]
            ],
            [(2, 4, 3), (6, 6, 1)],
        )
        first, second = result["waves"]
        self.assertEqual(first["peak_identities"], 3)
        self.assertEqual(
            [point["active_count"] for point in first["points"]],
            [2, 3, 2],
        )
        # Position 4 lists its drift records C-before-A; the wave keeps
        # that position's record order rather than any streak order.
        self.assertEqual(
            [point["identities"] for point in first["points"]],
            [("A", "B"), ("A", "B", "C"), ("C", "A")],
        )
        self.assertEqual(
            second["points"],
            (
                {
                    "split_cutoff": 6,
                    "active_count": 2,
                    "identities": ("B", "C"),
                },
            ),
        )
        self.assertEqual(second["peak_identities"], 2)
        self.assertEqual(
            result["totals"],
            {
                "splits": 4,
                "identities": 3,
                "waves": 2,
                "points": 4,
                "peak_active": 3,
            },
        )

    def test_unrequested_cutoffs_do_not_break_continuity(self):
        scans = (
            synthetic_scan(1, ("A", "B")),
            synthetic_scan(3, ("A", "B")),
            synthetic_scan(5, ("B", "A")),
        )
        result = self.build(scans=scans)
        self.assertEqual(len(result["waves"]), 1)
        wave = result["waves"][0]
        self.assertEqual(
            (wave["start_split"], wave["end_split"], wave["length"]),
            (1, 5, 3),
        )
        self.assertEqual(
            [point["split_cutoff"] for point in wave["points"]],
            [1, 3, 5],
        )
        self.assertEqual(
            [point["identities"] for point in wave["points"]],
            [("A", "B"), ("A", "B"), ("B", "A")],
        )

    def test_streaks_filtered_by_min_splits_never_count_as_active(self):
        # B's longest run is two positions; with min_splits=3 only A
        # (1-4) and C (3-6) remain, so no position reaches three
        # active identities.
        result = self.build(min_splits=3, min_active=3)
        self.assertEqual(result["waves"], ())
        self.assertEqual(result["totals"]["waves"], 0)
        single = self.build(min_splits=3, min_active=1)
        # A covers 1-4 and C covers 3-6: one wave across all positions.
        self.assertEqual(len(single["waves"]), 1)
        self.assertEqual(
            (
                single["waves"][0]["start_split"],
                single["waves"][0]["end_split"],
            ),
            (1, 6),
        )

    def test_identities_are_deduplicated_within_a_position(self):
        scans = (
            {
                "split_cutoff": 1,
                "drifts": (
                    {"identity": "A"},
                    {"identity": "A"},
                    {"identity": "B"},
                ),
                "totals": {},
            },
        )
        streaks = (
            {
                "identity": "A",
                "points": ({"split_cutoff": 1},),
            },
            {
                "identity": "B",
                "points": ({"split_cutoff": 1},),
            },
        )
        result = (
            BranchStore._build_lifetime_churn_forecast_drift_waves_result(
                scans, tuple(streaks), 1, 50
            )
        )
        self.assertEqual(
            result["waves"][0]["points"][0]["identities"], ("A", "B")
        )
        self.assertEqual(
            result["waves"][0]["points"][0]["active_count"], 2
        )

    def test_empty_inputs(self):
        streaks = (
            BranchStore._build_lifetime_churn_forecast_drift_streaks_result(
                SYNTHETIC_SCANS, 1, 10_000
            )["streaks"]
        )
        zeroed = {
            "splits": 0,
            "identities": 0,
            "waves": 0,
            "points": 0,
            "peak_active": 0,
        }
        self.assertEqual(
            BranchStore._build_lifetime_churn_forecast_drift_waves_result(
                (), (), 1, 50
            ),
            {"waves": (), "totals": zeroed},
        )
        self.assertEqual(
            BranchStore._build_lifetime_churn_forecast_drift_waves_result(
                SYNTHETIC_SCANS, (), 1, 50
            ),
            {"waves": (), "totals": zeroed},
        )
        self.assertEqual(
            BranchStore._build_lifetime_churn_forecast_drift_waves_result(
                SYNTHETIC_SCANS, streaks, 99, 50
            ),
            {"waves": (), "totals": zeroed},
        )

    def test_wave_result_limit_bounds_without_truncating(self):
        with self.assertRaises(ValueError) as caught:
            self.build(wave_result_limit=1)
        self.assertIn("wave result limit", str(caught.exception))
        result = self.build(wave_result_limit=2)
        self.assertEqual(len(result["waves"]), 2)


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
        for bad in (True, False, 1.0, "1", None):
            with self.subTest(min_active_identities=bad):
                with self.assertRaises(TypeError):
                    self.extended(min_active_identities=bad)
        with self.assertRaises(ValueError):
            self.extended(min_active_identities=0)
        with self.assertRaises(ValueError):
            self.extended(min_active_identities=-2)

    def test_wave_result_limit_must_be_a_non_bool_positive_int(self):
        for bad in (True, False, 1.0, "1", None):
            with self.subTest(wave_result_limit=bad):
                with self.assertRaises(TypeError):
                    self.extended(wave_result_limit=bad)
        with self.assertRaises(ValueError):
            self.extended(wave_result_limit=0)
        with self.assertRaises(ValueError):
            self.extended(wave_result_limit=-7)

    def test_new_parameters_validated_in_signature_order(self):
        # streak_result_limit fails before min_active_identities.
        with self.assertRaises(ValueError):
            self.extended(streak_result_limit=0, min_active_identities="x")
        with self.assertRaises(TypeError):
            self.extended(
                streak_result_limit="x", min_active_identities="x"
            )
        # min_active_identities fails before wave_result_limit.
        with self.assertRaises(ValueError):
            self.extended(
                min_active_identities=0, wave_result_limit="x"
            )
        with self.assertRaises(TypeError):
            self.extended(
                min_active_identities="x", wave_result_limit="x"
            )
        # wave_result_limit fails before the split membership check.
        with self.assertRaises(ValueError):
            self.extended(
                split_cutoffs=(99,), wave_result_limit=0
            )
        with self.assertRaises(TypeError):
            self.extended(
                split_cutoffs=(99,), wave_result_limit="x"
            )
        # Only after every parameter passes does membership fire.
        with self.assertRaises(ValueError):
            self.extended(split_cutoffs=(99,))

    def test_all_parameter_checks_precede_state_lookups(self):
        with self.assertRaises(TypeError):
            self.call(
                reference="ghost", cutoffs=(0, 1),
                split_cutoffs=(1,), min_active_identities="x",
            )
        with self.assertRaises(ValueError):
            self.call(
                reference="ghost", cutoffs=(0, 1),
                split_cutoffs=(1,), min_active_identities=0,
            )
        with self.assertRaises(ValueError):
            self.call(
                reference="ghost", cutoffs=(0, 1),
                split_cutoffs=(1,), wave_result_limit=0,
            )
        with self.assertRaises(ValueError):
            self.call(
                token="whatever", cutoffs=(0, 1), split_cutoffs=(5,),
                min_active_identities=0,
            )

    def test_unknown_branch_history_node_and_cause_raise(self):
        with self.assertRaises(KeyError):
            self.call(
                reference="ghost", cutoffs=(0, 1), split_cutoffs=(1,)
            )
        with self.assertRaises(KeyError):
            self.call(
                causes=("zzz",), cutoffs=(0, 1), split_cutoffs=(1,)
            )

    def test_earlier_stage_caps_still_fire(self):
        with self.assertRaises(ValueError) as caught:
            self.extended(drift_limit=2)
        self.assertIn("drift limit", str(caught.exception))
        with self.assertRaises(ValueError) as caught:
            self.extended(scan_limit=1)
        self.assertIn("scan limit", str(caught.exception))
        with self.assertRaises(ValueError) as caught:
            self.extended(streak_result_limit=0)
        self.assertIn("streak_result_limit", str(caught.exception))
        with self.assertRaises(ValueError) as caught:
            self.extended(backtest_limit=5)
        self.assertIn("backtest limit", str(caught.exception))


class LifetimeChurnForecastDriftWavesIsolationTests(unittest.TestCase):
    def call_args(self):
        return dict(
            LONG_ARGS,
            split_cutoffs=(3, 4, 5, 6),
            min_splits=1,
            min_active_identities=1,
        )

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

    def test_result_shares_no_object_with_scan_records(self):
        store, args = diamond_store()
        result = waves_call(store, args, **self.call_args())
        scan = scan_call(
            store,
            args,
            **dict(
                LONG_ARGS,
                split_cutoffs=(3, 4, 5, 6),
                drift_limit=50,
                scan_limit=100,
            ),
        )
        result_ids = set()

        def collect(value):
            if isinstance(value, dict):
                result_ids.add(id(value))
                for item in value.values():
                    collect(item)
            elif isinstance(value, tuple):
                result_ids.add(id(value))
                for item in value:
                    collect(item)

        collect(result)
        scan_ids = set()

        def collect_scan(value):
            if isinstance(value, dict):
                scan_ids.add(id(value))
                for item in value.values():
                    collect_scan(item)
            elif isinstance(value, tuple):
                scan_ids.add(id(value))
                for item in value:
                    collect_scan(item)

        collect_scan(scan)
        self.assertFalse(result_ids & scan_ids)

    def test_mutating_result_never_affects_later_calls(self):
        store, args = diamond_store()
        first = waves_call(store, args, **self.call_args())
        pristine = copy.deepcopy(first)
        for wave in first["waves"]:
            wave["points"] = wave["points"] + (
                {
                    "split_cutoff": 99,
                    "active_count": 99,
                    "identities": ("evil",),
                },
            )
            for point in wave["points"]:
                point["identities"] += ("tampered",)
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
            dict(split_cutoffs=(True,)),
            dict(split_cutoffs=(4, 4)),
            dict(split_cutoffs=(7,)),
            dict(split_cutoffs=(2,)),
            dict(min_rate_drop="x"),
            dict(drift_limit=0),
            dict(drift_limit="x"),
            dict(scan_limit=0),
            dict(min_splits=0),
            dict(streak_result_limit="x"),
            dict(drift_limit=2),
            dict(scan_limit=1),
            dict(min_active_identities=0),
            dict(min_active_identities="x"),
            dict(min_active_identities=False),
            dict(wave_result_limit=0),
            dict(wave_result_limit="x"),
            dict(wave_result_limit=True),
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
                    waves_call(store, args, **merged)

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
        from tests.test_lifetime_churn_waves import waves_call as churn_waves_call

        store, args = diamond_store()
        waves_call(store, args, **self.call_args())

        churn_waves = churn_waves_call(store, args, windows=LONG_ARGS["windows"])
        backtests = backtests_call(store, args, **LONG_ARGS)
        scorecards = scorecards_call(store, args, **LONG_ARGS)
        forecasts = forecasts_call(
            store,
            args,
            windows=LONG_ARGS["windows"],
            max_jitter=2,
            horizon=1,
        )
        scan = scan_call(
            store, args, **dict(LONG_ARGS, split_cutoffs=(3, 4, 5))
        )
        streaks = streaks_call(
            store,
            args,
            **dict(LONG_ARGS, split_cutoffs=(3, 4, 5)),
        )
        self.assertEqual(
            churn_waves_call(store, args, windows=LONG_ARGS["windows"]),
            churn_waves,
        )
        self.assertEqual(
            backtests_call(store, args, **LONG_ARGS), backtests
        )
        self.assertEqual(
            scorecards_call(store, args, **LONG_ARGS), scorecards
        )
        self.assertEqual(
            forecasts_call(
                store, args, windows=LONG_ARGS["windows"],
                max_jitter=2, horizon=1,
            ),
            forecasts,
        )
        self.assertEqual(
            scan_call(store, args, **dict(LONG_ARGS, split_cutoffs=(3, 4, 5))),
            scan,
        )
        self.assertEqual(
            streaks_call(store, args, **dict(LONG_ARGS, split_cutoffs=(3, 4, 5))),
            streaks,
        )


class LifetimeChurnForecastDriftWavesTokenTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, token=None, **overrides):
        return waves_call(self.store, self.args, token=token, **overrides)

    def call_args(self):
        return dict(
            LONG_ARGS,
            split_cutoffs=(3, 4, 5),
            min_splits=1,
            min_active_identities=1,
        )

    def test_success_consumes_one_read_and_failure_refunds(self):
        token = self.store.create_snapshot(2)
        self.call(token=token, **self.call_args())
        # Failures at every stage refund the reserved read.
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
                token=token, **dict(self.call_args(), streak_result_limit=0)
            )
        with self.assertRaises(ValueError):
            self.call(
                token=token, **dict(self.call_args(), scorecard_limit=2)
            )
        with self.assertRaises(ValueError):
            self.call(
                token=token, **dict(self.call_args(), backtest_limit=1)
            )
        with self.assertRaises(ValueError):
            self.call(
                token=token, **dict(self.call_args(), min_active_identities=0)
            )
        with self.assertRaises(ValueError):
            self.call(
                token=token, **dict(self.call_args(), wave_result_limit=0)
            )
        with self.assertRaises(ValueError):
            self.call(
                token=token,
                **dict(self.call_args(), cutoffs=(2, 99)),
            )
        with self.assertRaises(KeyError):
            self.call(
                token=token, **dict(self.call_args(), causes=("zzz",))
            )
        # Parameter validation runs before the token is touched.
        with self.assertRaises(TypeError):
            self.call(
                token=token, **dict(self.call_args(), split_cutoffs="x")
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
