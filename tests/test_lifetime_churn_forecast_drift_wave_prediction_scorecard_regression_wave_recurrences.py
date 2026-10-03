import copy
import inspect
import unittest

from city_twin.branches import BranchStore

from tests.test_lifetime_churn import diamond_store
from tests.test_lifetime_churn_forecast_drift_scan import LONG_ARGS
from tests.test_lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_waves import (
    build_waves,
    make_streaks,
    waves_call,
    waves_kwargs,
)

METHOD = (
    "lifetime_churn_forecast_drift_wave_prediction_scorecard_"
    "regression_wave_recurrences"
)
BUILDER = (
    "_build_lifetime_churn_forecast_drift_wave_prediction_scorecard_"
    "regression_wave_recurrences_result"
)


def recurrences_kwargs(store_args, **overrides) -> dict:
    min_occurrences = overrides.pop(
        "min_regression_wave_occurrences", 1
    )
    recurrence_limit = overrides.pop(
        "regression_wave_recurrence_limit", 50
    )
    kwargs = waves_kwargs(store_args, **overrides)
    kwargs["min_regression_wave_occurrences"] = min_occurrences
    kwargs["regression_wave_recurrence_limit"] = recurrence_limit
    return kwargs


def recurrences_call(store, store_args, **overrides):
    return getattr(store, METHOD)(
        **recurrences_kwargs(store_args, **overrides)
    )


def build_recurrences(
    waves,
    min_regression_wave_occurrences=1,
    regression_wave_recurrence_limit=50,
):
    return getattr(BranchStore, BUILDER)(
        waves,
        min_regression_wave_occurrences,
        regression_wave_recurrence_limit,
    )


def _point(baseline_cutoffs, *identities):
    return {
        "baseline_cutoffs": baseline_cutoffs,
        "active_count": len(identities),
        "identities": tuple(identities),
    }


def _wave(start, end, *points):
    return {
        "start_baseline_cutoffs": start,
        "end_baseline_cutoffs": end,
        "split_count": len(points),
        "peak_regressions": max(
            point["active_count"] for point in points
        ),
        "points": tuple(points),
    }


def reference_recurrences(waves, min_occurrences):
    """Independently group a regression waves result into recurrences."""
    encounter: dict[object, int] = {}
    next_encounter = 0
    per_identity: dict[object, list[dict[str, object]]] = {}
    for serial, wave in enumerate(waves):
        member_points: dict[object, list[tuple[object, int]]] = {}
        for point in wave["points"]:
            for identity in point["identities"]:
                if identity not in encounter:
                    encounter[identity] = next_encounter
                    next_encounter += 1
                member_points.setdefault(identity, []).append(
                    (
                        point["baseline_cutoffs"],
                        point["active_count"],
                    )
                )
        for identity, located in member_points.items():
            per_identity.setdefault(identity, []).append(
                {
                    "wave": serial,
                    "start_baseline_cutoffs": wave[
                        "start_baseline_cutoffs"
                    ],
                    "end_baseline_cutoffs": wave[
                        "end_baseline_cutoffs"
                    ],
                    "points": tuple(
                        {
                            "baseline_cutoffs": cutoff,
                            "active_count": active,
                        }
                        for cutoff, active in located
                    ),
                    "first_baseline_cutoffs": located[0][0],
                    "last_baseline_cutoffs": located[-1][0],
                }
            )

    records = []
    for identity, appearances in per_identity.items():
        if len(appearances) < min_occurrences:
            continue
        records.append(
            {
                "identity": identity,
                "waves": len(appearances),
                "first_wave": appearances[0]["wave"],
                "last_wave": appearances[-1]["wave"],
                "span": appearances[-1]["wave"]
                - appearances[0]["wave"]
                + 1,
                "points": sum(
                    len(appearance["points"]) for appearance in appearances
                ),
                "peak_active": max(
                    member_point["active_count"]
                    for appearance in appearances
                    for member_point in appearance["points"]
                ),
                "appearances": tuple(appearances),
            }
        )

    records.sort(
        key=lambda record: (
            -record["waves"],
            -record["points"],
            record["first_wave"],
            encounter[record["identity"]],
        )
    )
    totals = {
        "recurrences": len(records),
        "waves": 0,
        "points": 0,
        "first_wave": 0,
        "last_wave": 0,
    }
    if records:
        totals["waves"] = sum(record["waves"] for record in records)
        totals["points"] = sum(record["points"] for record in records)
        totals["first_wave"] = min(
            record["first_wave"] for record in records
        )
        totals["last_wave"] = max(
            record["last_wave"] for record in records
        )
    return {"recurrences": tuple(records), "totals": totals}


class RegressionWaveRecurrencesSignatureTests(unittest.TestCase):
    def test_public_signature_appends_two_parameters(self):
        parameters = list(
            inspect.signature(getattr(BranchStore, METHOD)).parameters.values()
        )
        names = [p.name for p in parameters]
        wave_names = list(
            inspect.signature(
                BranchStore.lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_waves
            ).parameters
        )
        self.assertEqual(
            names,
            wave_names[:-1]
            + [
                "min_regression_wave_occurrences",
                "regression_wave_recurrence_limit",
                "token",
            ],
        )
        self.assertEqual(
            [p.default for p in parameters[:-1]],
            [inspect.Parameter.empty] * (len(parameters) - 1),
        )
        self.assertIsNone(parameters[-1].default)

    def test_result_shape_and_key_order(self):
        waves = (
            _wave(1, 2, _point(1, "A", "B"), _point(2, "A")),
            _wave(4, 4, _point(4, "A", "C")),
        )
        result = build_recurrences(waves)
        self.assertEqual(list(result), ["recurrences", "totals"])
        self.assertIsInstance(result["recurrences"], tuple)
        record = result["recurrences"][0]
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
            ],
        )
        self.assertIsInstance(record["appearances"], tuple)
        appearance = record["appearances"][0]
        self.assertEqual(
            list(appearance),
            [
                "wave",
                "start_baseline_cutoffs",
                "end_baseline_cutoffs",
                "points",
                "first_baseline_cutoffs",
                "last_baseline_cutoffs",
            ],
        )
        self.assertEqual(
            list(appearance["points"][0]),
            ["baseline_cutoffs", "active_count"],
        )
        self.assertEqual(
            list(result["totals"]),
            ["recurrences", "waves", "points", "first_wave", "last_wave"],
        )


class RegressionWaveRecurrencesResultTests(unittest.TestCase):
    def test_within_wave_repetition_counts_once_but_points_accumulate(self):
        waves = (
            _wave(1, 2, _point(1, "A"), _point(2, "A", "B")),
            _wave(4, 4, _point(4, "A", "C")),
        )
        result = build_recurrences(waves, 2)
        self.assertEqual(len(result["recurrences"]), 1)
        record = result["recurrences"][0]
        self.assertEqual(record["identity"], "A")
        self.assertEqual(
            (
                record["waves"],
                record["first_wave"],
                record["last_wave"],
                record["span"],
            ),
            (2, 0, 1, 2),
        )
        # A appears at three actual points even though it joins only
        # two waves.
        self.assertEqual(record["points"], 3)
        self.assertEqual(record["peak_active"], 2)
        first, second = record["appearances"]
        self.assertEqual(first["wave"], 0)
        self.assertEqual(
            (
                first["start_baseline_cutoffs"],
                first["end_baseline_cutoffs"],
            ),
            (1, 2),
        )
        self.assertEqual(len(first["points"]), 2)
        self.assertEqual(first["first_baseline_cutoffs"], 1)
        self.assertEqual(first["last_baseline_cutoffs"], 2)
        self.assertEqual(second["wave"], 1)
        self.assertEqual(len(second["points"]), 1)
        self.assertEqual(
            (
                second["start_baseline_cutoffs"],
                second["end_baseline_cutoffs"],
                second["first_baseline_cutoffs"],
                second["last_baseline_cutoffs"],
            ),
            (4, 4, 4, 4),
        )
        self.assertEqual(
            result["totals"],
            {
                "recurrences": 1,
                "waves": 2,
                "points": 3,
                "first_wave": 0,
                "last_wave": 1,
            },
        )

    def test_peak_active_uses_the_largest_point_count(self):
        waves = (
            _wave(1, 1, _point(1, "A", "B", "C")),
            _wave(3, 3, _point(3, "A", "B")),
        )
        result = build_recurrences(waves, 2)
        by_identity = {
            record["identity"]: record for record in result["recurrences"]
        }
        self.assertEqual(by_identity["A"]["peak_active"], 3)
        self.assertEqual(by_identity["B"]["peak_active"], 3)

    def test_filter_sort_and_totals(self):
        # A in three waves (5 points); B and C in waves 0 and 2 at two
        # points each; both tie on every counter, but C is encountered
        # first (point 1 lists C, point 2 introduces B). D below would
        # be a single-wave identity and is filtered.
        waves = (
            _wave(
                1, 2,
                _point(1, "A", "C"),
                _point(2, "A", "B", "C"),
            ),
            _wave(4, 4, _point(4, "A", "D")),
            _wave(
                6, 7,
                _point(6, "A", "B", "C"),
                _point(7, "A"),
            ),
        )
        result = build_recurrences(waves, 2)
        self.assertEqual(
            [record["identity"] for record in result["recurrences"]],
            ["A", "C", "B"],
        )
        record_a, record_c, record_b = result["recurrences"]
        self.assertEqual(
            (
                record_a["waves"],
                record_a["points"],
                record_a["first_wave"],
                record_a["last_wave"],
                record_a["span"],
            ),
            (3, 5, 0, 2, 3),
        )
        self.assertEqual(
            (record_b["waves"], record_b["points"]), (2, 2)
        )
        self.assertEqual(
            (record_b["first_wave"], record_b["last_wave"]), (0, 2)
        )
        self.assertEqual(
            (record_c["waves"], record_c["points"]), (2, 3)
        )
        self.assertEqual(record_c["span"], 3)
        self.assertEqual(
            result["totals"],
            {
                "recurrences": 3,
                "waves": 7,
                "points": 10,
                "first_wave": 0,
                "last_wave": 2,
            },
        )

    def test_points_tie_break_sorts_by_points_first_wave_encounter(self):
        # A joins two waves at three points; everyone else joins two at
        # two points. B starts in wave 0 like the rest, but the point
        # tuple order gives the encounter tie break.
        waves = (
            _wave(1, 1, _point(1, "B2", "A2", "B", "A")),
            _wave(
                3, 4,
                _point(3, "A"),
                _point(4, "A", "B"),
            ),
            _wave(6, 6, _point(6, "A2", "B2")),
        )
        result = build_recurrences(waves, 2)
        self.assertEqual(
            [record["identity"] for record in result["recurrences"]],
            ["A", "B2", "A2", "B"],
        )

    def test_first_wave_breaks_the_point_count_tie(self):
        # X and Y both join two waves at two points, but Y's first
        # wave is wave 1 while X's is wave 0.
        waves = (
            _wave(1, 1, _point(1, "X")),
            _wave(3, 3, _point(3, "Y")),
            _wave(5, 5, _point(5, "X", "Y")),
        )
        result = build_recurrences(waves, 2)
        self.assertEqual(
            [record["identity"] for record in result["recurrences"]],
            ["X", "Y"],
        )
        x_record, y_record = result["recurrences"]
        self.assertEqual((x_record["waves"], x_record["points"]), (2, 2))
        self.assertEqual((x_record["first_wave"], x_record["span"]), (0, 3))
        self.assertEqual((y_record["first_wave"], y_record["span"]), (1, 2))

    def test_empty_wave_set_yields_zeros(self):
        result = build_recurrences(())
        self.assertEqual(result["recurrences"], ())
        self.assertEqual(
            result["totals"],
            {
                "recurrences": 0,
                "waves": 0,
                "points": 0,
                "first_wave": 0,
                "last_wave": 0,
            },
        )

    def test_min_occurrences_above_every_participation_is_empty(self):
        waves = (
            _wave(1, 1, _point(1, "A", "B")),
            _wave(3, 3, _point(3, "A")),
        )
        result = build_recurrences(waves, 3)
        self.assertEqual(result["recurrences"], ())
        self.assertEqual(result["totals"]["recurrences"], 0)
        self.assertEqual(result["totals"]["waves"], 0)
        self.assertEqual(result["totals"]["points"], 0)

    def test_recurrence_limit_bounds_the_complete_set(self):
        waves = (
            _wave(1, 1, _point(1, "A", "B", "C")),
            _wave(3, 3, _point(3, "A", "B", "C")),
        )
        with self.assertRaises(ValueError) as caught:
            build_recurrences(waves, 1, 2)
        self.assertIn("recurrence limit", str(caught.exception))
        result = build_recurrences(waves, 1, 3)
        self.assertEqual(len(result["recurrences"]), 3)
        # An empty result never trips the cap.
        result = build_recurrences(waves, 3, 1)
        self.assertEqual(result["recurrences"], ())

    def test_matches_independent_grouping_of_wave_sets(self):
        cases = (
            (make_streaks(("A", 1, 3)), (1, 2, 3), 1),
            (make_streaks(("A", 1, 3)), (1, 2, 3), 2),
            (
                make_streaks(("A", 1, 1), ("B", 1, 1), ("A2", 3, 3)),
                (1, 2, 3),
                1,
            ),
            (
                make_streaks(("A", 1, 1), ("B", 1, 1), ("A2", 3, 3)),
                (1, 2, 3),
                2,
            ),
            (make_streaks(("A", 1, 2), ("B", 4, 5)), (1, 2, 3, 4, 5), 1),
            (make_streaks(("A", 1, 2), ("B", 4, 5)), (1, 2, 3, 4, 5), 2),
            (
                make_streaks(
                    ("b", 1, 2), ("A", 1, 1), ("c", 2, 2)
                ),
                (1, 2),
                1,
            ),
            (make_streaks(("A", 1, 5)), (1, 3, 5), 1),
            ((), (1, 2, 3), 1),
        )
        for streaks, splits, min_occurrences in cases:
            with self.subTest(splits=splits, min_occurrences=min_occurrences):
                wave_set = build_waves(streaks, splits)["waves"]
                self.assertEqual(
                    build_recurrences(wave_set, min_occurrences),
                    reference_recurrences(wave_set, min_occurrences),
                )

    def test_every_object_is_fresh_and_unshared(self):
        waves = (
            _wave(1, 2, _point(1, ("A", "x")), _point(2, ("A", "x"))),
            _wave(4, 4, _point(4, ("A", "x"), "B")),
        )
        result = build_recurrences(waves, 2)
        record = result["recurrences"][0]
        record["waves"] = 999
        record["appearances"][0]["points"][0]["active_count"] = 999
        result["totals"]["waves"] = 999
        again = build_recurrences(waves, 2)
        self.assertEqual(again["recurrences"][0]["waves"], 2)
        self.assertEqual(
            again["recurrences"][0]["appearances"][0]["points"][0][
                "active_count"
            ],
            1,
        )
        self.assertEqual(again["totals"]["waves"], 2)
        # The source waves are not mutated either.
        self.assertEqual(waves[0]["points"][0]["identities"], (("A", "x"),))

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


class RegressionWaveRecurrencesValidationTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        return recurrences_call(self.store, self.args, **overrides)

    def extended(self, **overrides):
        overrides.setdefault("split_cutoffs", (4,))
        merged = dict(LONG_ARGS)
        merged.update(overrides)
        return self.call(**merged)

    def test_min_occurrences_must_be_a_non_bool_positive_int(self):
        for bad in (True, False, 1.0, "1", None, (1,)):
            with self.subTest(value=bad):
                with self.assertRaises(TypeError):
                    self.extended(min_regression_wave_occurrences=bad)
        with self.assertRaises(ValueError):
            self.extended(min_regression_wave_occurrences=0)
        with self.assertRaises(ValueError):
            self.extended(min_regression_wave_occurrences=-2)

    def test_recurrence_limit_must_be_a_non_bool_positive_int(self):
        for bad in (True, False, 1.0, "1", None, (1,)):
            with self.subTest(value=bad):
                with self.assertRaises(TypeError):
                    self.extended(regression_wave_recurrence_limit=bad)
        with self.assertRaises(ValueError):
            self.extended(regression_wave_recurrence_limit=0)
        with self.assertRaises(ValueError):
            self.extended(regression_wave_recurrence_limit=-3)

    def test_new_parameters_validated_after_wave_limit_in_signature_order(self):
        # regression_wave_limit fails before the two new parameters.
        with self.assertRaises(TypeError):
            self.extended(
                regression_wave_limit="x",
                min_regression_wave_occurrences="x",
            )
        with self.assertRaises(ValueError):
            self.extended(
                regression_wave_limit=0,
                min_regression_wave_occurrences=0,
            )
        # min_occurrences fails before recurrence_limit.
        with self.assertRaises(TypeError):
            self.extended(
                min_regression_wave_occurrences="x",
                regression_wave_recurrence_limit="x",
            )
        with self.assertRaises(ValueError):
            self.extended(
                min_regression_wave_occurrences=0,
                regression_wave_recurrence_limit=0,
            )
        # Once the earlier parameters pass, the recurrence cap error
        # fires, and all of them still precede the split membership
        # check and every state lookup.
        with self.assertRaises(TypeError):
            self.extended(regression_wave_recurrence_limit="x")
        with self.assertRaises(ValueError):
            self.call(
                reference="ghost",
                cutoffs=(0, 1),
                min_regression_wave_occurrences=0,
            )
        with self.assertRaises(ValueError):
            self.call(
                token="whatever",
                cutoffs=(0, 1),
                regression_wave_recurrence_limit=0,
            )
        with self.assertRaises(ValueError):
            self.extended(
                split_cutoffs=(99,),
                baseline_splits=(1,),
                min_evaluated_delta=0,
                min_hit_rate_drop=(0, 1),
                regression_limit=1,
                min_consecutive_splits=1,
                regression_streak_limit=1,
                min_active_regressions=1,
                regression_wave_limit=1,
                min_regression_wave_occurrences=1,
                regression_wave_recurrence_limit=1,
            )

    def test_underlying_stage_limits_and_errors_are_unchanged(self):
        with self.assertRaises(ValueError) as caught:
            self.extended(
                wave_cutoffs=(9,),
                baseline_splits=(1,),
                min_evaluated_delta=0,
                min_hit_rate_drop=(0, 1),
                regression_limit=50,
                min_consecutive_splits=1,
                regression_streak_limit=50,
                min_active_regressions=1,
                regression_wave_limit=50,
                min_regression_wave_occurrences=1,
                regression_wave_recurrence_limit=50,
            )
        self.assertIn("out of range", str(caught.exception))


class RegressionWaveRecurrencesEmptyRunTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def zero_totals(self):
        return {
            "recurrences": 0,
            "waves": 0,
            "points": 0,
            "first_wave": 0,
            "last_wave": 0,
        }

    def test_empty_wave_cutoffs_returns_empty_tuple_and_zero_totals(self):
        result = recurrences_call(
            self.store,
            self.args,
            **dict(LONG_ARGS, split_cutoffs=(3, 4, 5), wave_cutoffs=()),
        )
        self.assertEqual(result["recurrences"], ())
        self.assertEqual(result["totals"], self.zero_totals())

    def test_no_qualifying_identity_returns_empty_recurrences(self):
        result = recurrences_call(
            self.store,
            self.args,
            **dict(
                LONG_ARGS,
                split_cutoffs=(3, 4, 5),
                wave_cutoffs=(0,),
                min_regression_wave_occurrences=2,
            ),
        )
        self.assertEqual(result["recurrences"], ())
        self.assertEqual(result["totals"], self.zero_totals())


class RegressionWaveRecurrencesIsolationTests(unittest.TestCase):
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
            baseline_splits=(1,),
            min_evaluated_delta=0,
            min_hit_rate_drop=(0, 1),
            regression_limit=50,
            min_consecutive_splits=1,
            regression_streak_limit=50,
            min_active_regressions=1,
            regression_wave_limit=50,
            min_regression_wave_occurrences=1,
            regression_wave_recurrence_limit=50,
        )

    def test_mutating_result_never_affects_later_calls(self):
        store, args = diamond_store()
        first = recurrences_call(store, args, **self.call_args())
        pristine = copy.deepcopy(first)
        for record in first["recurrences"]:
            record["waves"] = 99
            record["appearances"] = record["appearances"] + (
                {
                    "wave": 9,
                    "start_baseline_cutoffs": 9,
                    "end_baseline_cutoffs": 9,
                    "points": ({"baseline_cutoffs": 9,
                                "active_count": 9},),
                    "first_baseline_cutoffs": 9,
                    "last_baseline_cutoffs": 9,
                },
            )
            for appearance in record["appearances"]:
                for point in appearance["points"]:
                    point["active_count"] = 99
        self.assertEqual(
            recurrences_call(store, args, **self.call_args()),
            pristine,
        )

    def test_success_and_failure_change_nothing(self):
        store, args = diamond_store()
        heads = dict(store._heads)
        appends = copy.deepcopy(store._appends)
        merges = copy.deepcopy(store._merges)
        audit = store.audit_log()

        recurrences_call(store, args, **self.call_args())
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
            dict(min_evaluated="x"),
            dict(min_evaluated=True),
            dict(min_evaluated=-1),
            dict(min_cutoffs="x"),
            dict(min_cutoffs=False),
            dict(min_cutoffs=0),
            dict(prediction_scorecard_limit="x"),
            dict(prediction_scorecard_limit=False),
            dict(prediction_scorecard_limit=0),
            dict(baseline_splits="x"),
            dict(baseline_splits=(False,)),
            dict(baseline_splits=()),
            dict(baseline_splits=(0,)),
            dict(baseline_splits=(2, 2)),
            dict(baseline_splits=(3, 1)),
            dict(min_evaluated_delta="x"),
            dict(min_evaluated_delta=True),
            dict(min_evaluated_delta=-1),
            dict(min_hit_rate_drop="x"),
            dict(min_hit_rate_drop=(1, 0)),
            dict(min_hit_rate_drop=(2, 1)),
            dict(regression_limit="x"),
            dict(regression_limit=False),
            dict(regression_limit=0),
            dict(min_consecutive_splits="x"),
            dict(min_consecutive_splits=False),
            dict(min_consecutive_splits=0),
            dict(regression_streak_limit="x"),
            dict(regression_streak_limit=False),
            dict(regression_streak_limit=0),
            dict(min_active_regressions="x"),
            dict(min_active_regressions=False),
            dict(min_active_regressions=0),
            dict(regression_wave_limit="x"),
            dict(regression_wave_limit=False),
            dict(regression_wave_limit=0),
            dict(min_regression_wave_occurrences="x"),
            dict(min_regression_wave_occurrences=False),
            dict(min_regression_wave_occurrences=0),
            dict(regression_wave_recurrence_limit="x"),
            dict(regression_wave_recurrence_limit=False),
            dict(regression_wave_recurrence_limit=0),
            dict(reference="ghost"),
        ]
        for overrides in failures:
            with self.subTest(overrides=overrides):
                merged = self.call_args()
                merged.update(overrides)
                with self.assertRaises(Exception):
                    recurrences_call(store, args, **merged)

        self.assertEqual(dict(store._heads), heads)
        self.assertEqual(copy.deepcopy(store._appends), appends)
        self.assertEqual(copy.deepcopy(store._merges), merges)
        self.assertEqual(store.audit_log(), audit)

    def test_existing_regression_waves_query_is_unaffected(self):
        store, args = diamond_store()
        recurrences_call(store, args, **self.call_args())
        wave_args = {
            key: value
            for key, value in self.call_args().items()
            if key
            not in (
                "min_regression_wave_occurrences",
                "regression_wave_recurrence_limit",
            )
        }
        waves = waves_call(store, args, **wave_args)
        self.assertEqual(waves_call(store, args, **wave_args), waves)


class RegressionWaveRecurrencesTokenTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, token=None, **overrides):
        return recurrences_call(
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
            min_evaluated=0,
            min_cutoffs=1,
            prediction_scorecard_limit=50,
            baseline_splits=(1,),
            min_evaluated_delta=0,
            min_hit_rate_drop=(0, 1),
            regression_limit=50,
            min_consecutive_splits=1,
            regression_streak_limit=50,
            min_active_regressions=1,
            regression_wave_limit=50,
            min_regression_wave_occurrences=1,
            regression_wave_recurrence_limit=50,
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
            self.call(
                token=token, **dict(self.call_args(), wave_cutoffs=(9,))
            )
        # Parameter validation runs before the token is touched.
        with self.assertRaises(TypeError):
            self.call(
                token=token,
                **dict(self.call_args(),
                       min_regression_wave_occurrences="x"),
            )
        with self.assertRaises(ValueError):
            self.call(
                token=token,
                **dict(self.call_args(),
                       regression_wave_recurrence_limit=0),
            )
        # The second allowed read succeeds; the token then expires.
        self.call(token=token, **self.call_args())
        with self.assertRaises(RuntimeError):
            self.call(token=token, **self.call_args())

    def test_tokenless_query_leaves_snapshot_quotas_intact(self):
        token = self.store.create_snapshot(1)
        self.call(**self.call_args())
        self.call(**self.call_args())
        # The single read quota is still available after tokenless calls.
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
