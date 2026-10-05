import copy
import inspect
import unittest

from city_twin.branches import BranchStore

from tests.test_lifetime_churn import diamond_store
from tests.test_lifetime_churn_forecast_drift_scan import LONG_ARGS
from tests.test_lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_wave_forecast_scorecard_regression_wave_forecast_backtests import (
    build_backtests,
    synthetic_waves,
)
from tests.test_lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_wave_forecast_scorecard_regression_wave_forecast_backtests_scorecards import (
    backtest_record,
    call_args as scorecards_call_args,
)
from tests.test_lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_wave_forecast_scorecard_regression_wave_forecast_backtests_scorecards_regression_wave_forecast_backtests_scorecards_regressions import (
    DEEP_BACKTESTS,
    DEEP_SCORECARD_DEFAULTS,
    DEEP_SCORECARDS,
    build_deep_scorecards,
    deep_scorecards_kwargs,
    kept_scorecards,
    regressing_records,
)

REGRESSIONS = (
    BranchStore.lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_wave_forecast_scorecard_regression_wave_forecast_backtests_scorecards_regression_wave_forecast_backtests_scorecards_regressions
)
STREAKS = (
    BranchStore.lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_wave_forecast_scorecard_regression_wave_forecast_backtests_scorecards_regression_wave_forecast_backtests_scorecards_regressions_streaks
)

STREAK_DEFAULTS = dict(
    regression_wave_regression_wave_regression_wave_baseline_splits=(1,),
    regression_wave_regression_wave_regression_wave_min_evaluated_delta=0,
    regression_wave_regression_wave_regression_wave_min_hit_rate_drop=(0, 1),
    regression_wave_regression_wave_regression_wave_regression_limit=50,
    regression_wave_regression_wave_regression_wave_min_consecutive_splits=1,
    regression_wave_regression_wave_regression_wave_regression_streak_limit=50,
)


def streaks_kwargs(store_args, **overrides) -> dict:
    streak = {
        name: overrides.pop(name, default)
        for name, default in STREAK_DEFAULTS.items()
    }
    kwargs = deep_scorecards_kwargs(store_args, **overrides)
    kwargs.update(streak)
    return kwargs


def streaks_call(store, store_args, **overrides):
    return store.lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_wave_forecast_scorecard_regression_wave_forecast_backtests_scorecards_regression_wave_forecast_backtests_scorecards_regressions_streaks(
        **streaks_kwargs(store_args, **overrides)
    )


def build_streaks(
    records,
    scorecards,
    baseline_splits=(1,),
    min_evaluated_delta=0,
    min_hit_rate_drop=(0, 1),
    regression_limit=50,
    min_consecutive_splits=1,
    regression_streak_limit=50,
):
    return BranchStore._build_lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_wave_forecast_scorecard_regression_wave_forecast_backtests_scorecards_regression_wave_forecast_backtests_scorecards_regressions_streaks_result(
        records,
        scorecards,
        baseline_splits,
        min_evaluated_delta,
        min_hit_rate_drop,
        regression_limit,
        min_consecutive_splits,
        regression_streak_limit,
    )


def call_args(**overrides):
    merged = scorecards_call_args(**DEEP_SCORECARD_DEFAULTS)
    merged.update(overrides)
    return merged


def gap_records(identity):
    # Hits at splits 1 and 3 but not 2: the run breaks at split 2.
    return (
        backtest_record(2, identity, 2, 0),
        backtest_record(4, identity, 0, 2),
        backtest_record(6, identity, 2, 0),
        backtest_record(8, identity, 0, 2),
        backtest_record(10, identity, 0, 2),
        backtest_record(12, identity, 0, 2),
    )


def two_run_records(identity):
    # Hits at splits 1, 2, 4 and 5 but not 3: two runs of length two.
    return (
        backtest_record(2, identity, 1, 0),
        backtest_record(4, identity, 1, 0),
        backtest_record(6, identity, 0, 1),
        backtest_record(8, identity, 1, 0),
        backtest_record(10, identity, 1, 0),
        backtest_record(12, identity, 0, 1),
        backtest_record(14, identity, 0, 1),
        backtest_record(16, identity, 0, 1),
        backtest_record(18, identity, 0, 1),
        backtest_record(20, identity, 0, 1),
    )


def long_run_records(identity):
    # Hits at splits 1, 2 and 3: one run of length three.
    return (
        backtest_record(2, identity, 2, 0),
        backtest_record(4, identity, 2, 0),
        backtest_record(6, identity, 2, 0),
        backtest_record(8, identity, 0, 2),
        backtest_record(10, identity, 0, 2),
        backtest_record(12, identity, 0, 2),
    )


def steep_records(identity):
    # Hits at splits 1 and 2 with a worst drop of 1/1.
    return (
        backtest_record(2, identity, 1, 0),
        backtest_record(4, identity, 1, 0),
        backtest_record(6, identity, 0, 2),
        backtest_record(8, identity, 0, 2),
    )


class SignatureTests(unittest.TestCase):
    def test_public_signature_swaps_and_appends_parameters(self):
        parameters = list(inspect.signature(STREAKS).parameters.values())
        names = [p.name for p in parameters]
        regression_names = list(inspect.signature(REGRESSIONS).parameters)
        self.assertEqual(
            names,
            [
                "regression_wave_regression_wave_regression_wave_baseline_splits"
                if name
                == "regression_wave_regression_wave_regression_wave_baseline_cutoffs"
                else name
                for name in regression_names[:-1]
            ]
            + [
                "regression_wave_regression_wave_regression_wave_min_consecutive_splits",
                "regression_wave_regression_wave_regression_wave_regression_streak_limit",
                "token",
            ],
        )
        self.assertEqual(
            [p.default for p in parameters[:-1]],
            [inspect.Parameter.empty] * (len(parameters) - 1),
        )
        self.assertIsNone(parameters[-1].default)

    def test_result_shape_and_key_order(self):
        result = build_streaks(
            regressing_records("A"),
            kept_scorecards("A"),
            (1, 2),
            0,
            (1, 2),
        )
        self.assertEqual(list(result), ["streaks", "totals"])
        self.assertIsInstance(result["streaks"], tuple)
        streak = result["streaks"][0]
        self.assertEqual(
            list(streak),
            [
                "identity",
                "start_baseline_cutoffs",
                "end_baseline_cutoffs",
                "split_count",
                "first_regression_cutoff",
                "worst_rate_drop",
            ],
        )
        self.assertEqual(
            list(result["totals"]),
            ["identities", "streaks", "regressing_splits"],
        )


class ResultTests(unittest.TestCase):
    def test_adjacent_hits_connect_into_one_streak(self):
        result = build_streaks(
            regressing_records("A"),
            kept_scorecards("A"),
            (1, 2),
            0,
            (1, 2),
        )
        self.assertEqual(len(result["streaks"]), 1)
        self.assertEqual(
            result["streaks"][0],
            {
                "identity": "A",
                "start_baseline_cutoffs": 1,
                "end_baseline_cutoffs": 2,
                "split_count": 2,
                "first_regression_cutoff": 4,
                "worst_rate_drop": (2, 3),
            },
        )
        self.assertEqual(
            result["totals"],
            {"identities": 1, "streaks": 1, "regressing_splits": 2},
        )

    def test_min_consecutive_splits_filters_short_runs(self):
        result = build_streaks(
            regressing_records("A"),
            kept_scorecards("A"),
            (1, 2),
            0,
            (1, 2),
            50,
            3,
        )
        self.assertEqual(result["streaks"], ())
        self.assertEqual(
            result["totals"],
            {"identities": 0, "streaks": 0, "regressing_splits": 0},
        )

    def test_gap_breaks_the_run(self):
        records = gap_records("B")
        # Two runs of one position each; the smaller start wins the
        # per-identity tie.
        result = build_streaks(
            records, kept_scorecards("B"), (1, 2, 3), 0, (1, 2)
        )
        self.assertEqual(
            result["streaks"],
            (
                {
                    "identity": "B",
                    "start_baseline_cutoffs": 1,
                    "end_baseline_cutoffs": 1,
                    "split_count": 1,
                    "first_regression_cutoff": 4,
                    "worst_rate_drop": (4, 5),
                },
            ),
        )
        self.assertEqual(
            result["totals"],
            {"identities": 1, "streaks": 1, "regressing_splits": 1},
        )
        # Requiring two consecutive splits drops both runs.
        result = build_streaks(
            records, kept_scorecards("B"), (1, 2, 3), 0, (1, 2), 50, 2
        )
        self.assertEqual(result["streaks"], ())

    def test_only_the_longest_run_survives_per_identity(self):
        # Two runs of two positions each; the earlier start wins.
        result = build_streaks(
            two_run_records("D"),
            kept_scorecards("D"),
            (1, 2, 3, 4, 5),
            0,
            (1, 2),
        )
        self.assertEqual(
            result["streaks"],
            (
                {
                    "identity": "D",
                    "start_baseline_cutoffs": 1,
                    "end_baseline_cutoffs": 2,
                    "split_count": 2,
                    "first_regression_cutoff": 6,
                    "worst_rate_drop": (3, 4),
                },
            ),
        )
        self.assertEqual(
            result["totals"],
            {"identities": 1, "streaks": 1, "regressing_splits": 2},
        )

    def test_only_scorecard_identities_are_candidates(self):
        records = regressing_records("A") + regressing_records("Z")
        result = build_streaks(
            records, kept_scorecards("A"), (1, 2), 0, (1, 2)
        )
        self.assertEqual(
            [s["identity"] for s in result["streaks"]], ["A"]
        )

    def test_sort_order_count_then_drop_then_identity(self):
        records = (
            long_run_records("F")
            + steep_records("G")
            + regressing_records("A")
            + regressing_records("H")
        )
        result = build_streaks(
            records,
            kept_scorecards("F", "G", "A", "H"),
            (1, 2, 3),
            0,
            (1, 2),
        )
        # F runs three positions; G and the pair run two. G's worst
        # drop 1/1 beats the pair's 2/3; A and H tie on everything and
        # order by identity.
        self.assertEqual(
            [s["identity"] for s in result["streaks"]],
            ["F", "G", "A", "H"],
        )
        self.assertEqual(result["streaks"][0]["split_count"], 3)
        self.assertEqual(result["streaks"][0]["worst_rate_drop"], (1, 1))
        self.assertEqual(result["streaks"][1]["worst_rate_drop"], (1, 1))
        self.assertEqual(result["streaks"][2]["worst_rate_drop"], (2, 3))
        self.assertEqual(
            result["totals"],
            {"identities": 4, "streaks": 4, "regressing_splits": 3},
        )

    def test_tuple_identities_sort_by_unicode_code_point(self):
        records = (
            regressing_records(("W", "y"))
            + regressing_records(("a0", "x"))
            + regressing_records(("A", "z"))
        )
        result = build_streaks(
            records,
            kept_scorecards(("W", "y"), ("a0", "x"), ("A", "z")),
            (1, 2),
            0,
            (1, 2),
        )
        self.assertEqual(
            [s["identity"] for s in result["streaks"]],
            [("A", "z"), ("W", "y"), ("a0", "x")],
        )

    def test_regression_streak_limit_bounds_the_complete_set(self):
        records = regressing_records("A") + regressing_records("C")
        with self.assertRaises(ValueError) as caught:
            build_streaks(
                records,
                kept_scorecards("A", "C"),
                (1, 2),
                0,
                (1, 2),
                50,
                1,
                1,
            )
        self.assertIn("regression streak limit", str(caught.exception))
        result = build_streaks(
            records,
            kept_scorecards("A", "C"),
            (1, 2),
            0,
            (1, 2),
            50,
            1,
            2,
        )
        self.assertEqual(len(result["streaks"]), 2)
        # Identities filtered out below the limit never count toward it.
        result = build_streaks(
            records,
            kept_scorecards("A", "C"),
            (2,),
            0,
            (2, 3),
            50,
            1,
            1,
        )
        self.assertEqual(result["streaks"], ())

    def test_regression_limit_bounds_each_split_position(self):
        records = regressing_records("A") + regressing_records("C")
        # Each position keeps two regressions, over the limit of one.
        with self.assertRaises(ValueError) as caught:
            build_streaks(
                records,
                kept_scorecards("A", "C"),
                (1, 2),
                0,
                (1, 2),
                1,
            )
        self.assertIn("regression limit", str(caught.exception))
        result = build_streaks(
            records,
            kept_scorecards("A", "C"),
            (1, 2),
            0,
            (1, 2),
            2,
        )
        self.assertEqual(len(result["streaks"]), 2)

    def test_empty_records_return_empty_tuple_and_zero_totals(self):
        result = build_streaks((), ())
        self.assertEqual(result["streaks"], ())
        self.assertEqual(
            result["totals"],
            {"identities": 0, "streaks": 0, "regressing_splits": 0},
        )

    def test_every_object_is_fresh(self):
        records = (
            backtest_record(2, ("A", "x"), 2, 0),
            backtest_record(4, ("A", "x"), 0, 2),
        )
        scorecards = kept_scorecards(("A", "x"))
        result = build_streaks(records, scorecards, (1,), 0, (0, 1))
        result["streaks"][0]["identity"] = "Z"
        result["streaks"][0]["split_count"] = 999
        result["totals"]["streaks"] = 999
        again = build_streaks(records, scorecards, (1,), 0, (0, 1))
        self.assertEqual(again["streaks"][0]["identity"], ("A", "x"))
        self.assertEqual(again["streaks"][0]["split_count"], 1)
        self.assertEqual(again["totals"]["streaks"], 1)
        # The input records are not mutated either.
        self.assertEqual(records[0]["identity"], ("A", "x"))

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


class ChainTests(unittest.TestCase):
    """The builder refines exactly what the deepest scorecards stage keeps."""

    def test_streaks_match_the_two_cutoff_backtest(self):
        backtests = build_backtests(
            synthetic_waves(), 1, 50, 0, 50, 2, 50, (2, 4), 0, 50
        )
        scorecards = build_deep_scorecards(backtests["backtests"])
        result = build_streaks(
            backtests["backtests"], scorecards["scorecards"], (1,), 0, (0, 1)
        )
        # B has a single record, so its observation segment is empty;
        # A and C tie on every rate and count, then sort by identity.
        self.assertEqual(
            [s["identity"] for s in result["streaks"]], ["A", "C"]
        )
        for streak in result["streaks"]:
            self.assertEqual(
                (
                    streak["start_baseline_cutoffs"],
                    streak["end_baseline_cutoffs"],
                    streak["split_count"],
                    streak["first_regression_cutoff"],
                    streak["worst_rate_drop"],
                ),
                (1, 1, 1, 4, (1, 2)),
            )
        self.assertEqual(
            result["totals"],
            {"identities": 2, "streaks": 2, "regressing_splits": 1},
        )
        # A threshold above the 1/2 drop empties the result.
        result = build_streaks(
            backtests["backtests"],
            scorecards["scorecards"],
            (1,),
            0,
            (1, 1),
        )
        self.assertEqual(result["streaks"], ())
        # Two baseline splits leave no observation record at split 2,
        # so only split 1 hits and no run reaches length two.
        result = build_streaks(
            backtests["backtests"],
            scorecards["scorecards"],
            (1, 2),
            0,
            (0, 1),
            50,
            2,
        )
        self.assertEqual(result["streaks"], ())

    def test_scorecard_filters_gate_the_candidates(self):
        backtests = build_backtests(
            synthetic_waves(), 1, 50, 0, 50, 2, 50, (2, 4), 0, 50
        )
        scorecards = build_deep_scorecards(backtests["backtests"], 0, 2)
        result = build_streaks(
            backtests["backtests"], scorecards["scorecards"], (1,), 0, (0, 1)
        )
        self.assertEqual(
            [s["identity"] for s in result["streaks"]], ["A", "C"]
        )
        scorecards = build_deep_scorecards(backtests["backtests"], 5, 1)
        result = build_streaks(
            backtests["backtests"], scorecards["scorecards"], (1,), 0, (0, 1)
        )
        self.assertEqual(result["streaks"], ())

    def test_public_query_matches_the_builder_on_the_backtest_set(self):
        store, args = diamond_store()
        backtest_parameters = inspect.signature(DEEP_BACKTESTS).parameters
        scorecard_parameters = inspect.signature(DEEP_SCORECARDS).parameters
        full_kwargs = streaks_kwargs(args, **call_args())
        backtests = store.lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_wave_forecast_scorecard_regression_wave_forecast_backtests_scorecards_regression_wave_forecast_backtests(
            **{
                key: value
                for key, value in full_kwargs.items()
                if key in backtest_parameters
            }
        )
        scorecards = store.lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_wave_forecast_scorecard_regression_wave_forecast_backtests_scorecards_regression_wave_forecast_backtests_scorecards(
            **{
                key: value
                for key, value in full_kwargs.items()
                if key in scorecard_parameters
            }
        )
        result = streaks_call(store, args, **call_args())
        self.assertEqual(
            result,
            build_streaks(
                backtests["backtests"],
                scorecards["scorecards"],
                (1,),
                0,
                (0, 1),
                50,
                1,
                50,
            ),
        )


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        return streaks_call(self.store, self.args, **overrides)

    def extended(self, **overrides):
        overrides.setdefault("split_cutoffs", (4,))
        overrides.setdefault("baseline_splits", (1,))
        merged = dict(LONG_ARGS)
        merged.update(overrides)
        return self.call(**merged)

    def test_baseline_splits_must_be_a_nonempty_tuple_of_positive_ints(
        self,
    ):
        for bad in (True, 1, 1.0, "1", None, [1, 2], {1: 2}):
            with self.subTest(value=bad):
                with self.assertRaises(TypeError):
                    self.extended(
                        regression_wave_regression_wave_regression_wave_baseline_splits=bad
                    )
        for bad in ((True, 1), (1, True), (1.0, 2), (1, "2"), (None, 2)):
            with self.subTest(value=bad):
                with self.assertRaises(TypeError):
                    self.extended(
                        regression_wave_regression_wave_regression_wave_baseline_splits=bad
                    )
        with self.assertRaises(ValueError):
            self.extended(
                regression_wave_regression_wave_regression_wave_baseline_splits=()
            )
        with self.assertRaises(ValueError):
            self.extended(
                regression_wave_regression_wave_regression_wave_baseline_splits=(0,)
            )
        with self.assertRaises(ValueError):
            self.extended(
                regression_wave_regression_wave_regression_wave_baseline_splits=(-1, 2)
            )
        with self.assertRaises(ValueError):
            self.extended(
                regression_wave_regression_wave_regression_wave_baseline_splits=(1, 1)
            )
        with self.assertRaises(ValueError):
            self.extended(
                regression_wave_regression_wave_regression_wave_baseline_splits=(2, 1)
            )
        # The boundaries are legal.
        self.extended(
            regression_wave_regression_wave_regression_wave_baseline_splits=(1,)
        )
        self.extended(
            regression_wave_regression_wave_regression_wave_baseline_splits=(1, 2, 3)
        )

    def test_min_consecutive_splits_must_be_a_non_bool_positive_int(self):
        for bad in (True, False, 1.0, "1", None, (1,)):
            with self.subTest(value=bad):
                with self.assertRaises(TypeError):
                    self.extended(
                        regression_wave_regression_wave_regression_wave_min_consecutive_splits=bad
                    )
        with self.assertRaises(ValueError):
            self.extended(
                regression_wave_regression_wave_regression_wave_min_consecutive_splits=0
            )
        with self.assertRaises(ValueError):
            self.extended(
                regression_wave_regression_wave_regression_wave_min_consecutive_splits=-2
            )

    def test_regression_streak_limit_must_be_a_non_bool_positive_int(self):
        for bad in (True, False, 1.0, "1", None, (1,)):
            with self.subTest(value=bad):
                with self.assertRaises(TypeError):
                    self.extended(
                        regression_wave_regression_wave_regression_wave_regression_streak_limit=bad
                    )
        with self.assertRaises(ValueError):
            self.extended(
                regression_wave_regression_wave_regression_wave_regression_streak_limit=0
            )
        with self.assertRaises(ValueError):
            self.extended(
                regression_wave_regression_wave_regression_wave_regression_streak_limit=-3
            )

    def test_new_parameters_validated_in_signature_order(self):
        # The deepest scorecard limit fails before the new baseline
        # splits.
        with self.assertRaises(ValueError):
            self.extended(
                regression_wave_regression_wave_regression_wave_forecast_scorecard_limit=0,
                regression_wave_regression_wave_regression_wave_baseline_splits="x",
            )
        with self.assertRaises(TypeError):
            self.extended(
                regression_wave_regression_wave_regression_wave_forecast_scorecard_limit="x",
                regression_wave_regression_wave_regression_wave_baseline_splits="x",
            )
        # baseline splits fail before min evaluated delta.
        with self.assertRaises(TypeError):
            self.extended(
                regression_wave_regression_wave_regression_wave_baseline_splits="x",
                regression_wave_regression_wave_regression_wave_min_evaluated_delta="x",
            )
        with self.assertRaises(ValueError):
            self.extended(
                regression_wave_regression_wave_regression_wave_baseline_splits=(),
                regression_wave_regression_wave_regression_wave_min_evaluated_delta=-1,
            )
        # min evaluated delta fails before min hit rate drop.
        with self.assertRaises(TypeError):
            self.extended(
                regression_wave_regression_wave_regression_wave_min_evaluated_delta="x",
                regression_wave_regression_wave_regression_wave_min_hit_rate_drop="x",
            )
        with self.assertRaises(ValueError):
            self.extended(
                regression_wave_regression_wave_regression_wave_min_evaluated_delta=-1,
                regression_wave_regression_wave_regression_wave_min_hit_rate_drop=(1, 0),
            )
        # min hit rate drop fails before the regression limit.
        with self.assertRaises(TypeError):
            self.extended(
                regression_wave_regression_wave_regression_wave_min_hit_rate_drop="x",
                regression_wave_regression_wave_regression_wave_regression_limit="x",
            )
        with self.assertRaises(ValueError):
            self.extended(
                regression_wave_regression_wave_regression_wave_min_hit_rate_drop=(1, 0),
                regression_wave_regression_wave_regression_wave_regression_limit=0,
            )
        # the regression limit fails before min consecutive splits.
        with self.assertRaises(TypeError):
            self.extended(
                regression_wave_regression_wave_regression_wave_regression_limit="x",
                regression_wave_regression_wave_regression_wave_min_consecutive_splits="x",
            )
        with self.assertRaises(ValueError):
            self.extended(
                regression_wave_regression_wave_regression_wave_regression_limit=0,
                regression_wave_regression_wave_regression_wave_min_consecutive_splits=0,
            )
        # min consecutive splits fails before the regression streak
        # limit.
        with self.assertRaises(TypeError):
            self.extended(
                regression_wave_regression_wave_regression_wave_min_consecutive_splits="x",
                regression_wave_regression_wave_regression_wave_regression_streak_limit="x",
            )
        with self.assertRaises(ValueError):
            self.extended(
                regression_wave_regression_wave_regression_wave_min_consecutive_splits=0,
                regression_wave_regression_wave_regression_wave_regression_streak_limit=0,
            )
        # All six pass before the split membership check fires.
        with self.assertRaises(ValueError):
            self.extended(
                split_cutoffs=(99,),
                regression_wave_regression_wave_regression_wave_baseline_splits=(1,),
                regression_wave_regression_wave_regression_wave_min_evaluated_delta=0,
                regression_wave_regression_wave_regression_wave_min_hit_rate_drop=(0, 1),
                regression_wave_regression_wave_regression_wave_regression_limit=1,
                regression_wave_regression_wave_regression_wave_min_consecutive_splits=1,
                regression_wave_regression_wave_regression_wave_regression_streak_limit=1,
            )

    def test_all_parameter_checks_precede_state_lookups(self):
        with self.assertRaises(TypeError):
            self.call(
                reference="ghost",
                regression_wave_regression_wave_regression_wave_baseline_splits="x",
            )
        with self.assertRaises(ValueError):
            self.call(
                reference="ghost",
                regression_wave_regression_wave_regression_wave_baseline_splits=(),
            )
        with self.assertRaises(ValueError):
            self.call(
                reference="ghost",
                regression_wave_regression_wave_regression_wave_baseline_splits=(0,),
            )
        with self.assertRaises(TypeError):
            self.call(
                reference="ghost",
                regression_wave_regression_wave_regression_wave_min_consecutive_splits="x",
            )
        with self.assertRaises(ValueError):
            self.call(
                reference="ghost",
                regression_wave_regression_wave_regression_wave_min_consecutive_splits=0,
            )
        with self.assertRaises(TypeError):
            self.call(
                reference="ghost",
                regression_wave_regression_wave_regression_wave_regression_streak_limit="x",
            )
        with self.assertRaises(ValueError):
            self.call(
                reference="ghost",
                regression_wave_regression_wave_regression_wave_regression_streak_limit=0,
            )

    def test_underlying_stage_limits_and_errors_are_unchanged(self):
        # The diamond store yields one drift wave, so a drift-wave
        # cutoff past it keeps the shared backtest stage's range error;
        # it yields no deepest forecast regression waves, so the deep
        # cutoff range check is skipped there exactly as in the deepest
        # scorecards query.
        with self.assertRaises(ValueError) as caught:
            self.extended(
                wave_cutoffs=(9,),
                regression_wave_regression_wave_regression_wave_baseline_splits=(1,),
                regression_wave_regression_wave_regression_wave_min_evaluated_delta=0,
                regression_wave_regression_wave_regression_wave_min_hit_rate_drop=(0, 1),
                regression_wave_regression_wave_regression_wave_regression_limit=50,
                regression_wave_regression_wave_regression_wave_min_consecutive_splits=1,
                regression_wave_regression_wave_regression_wave_regression_streak_limit=50,
            )
        self.assertIn("out of range", str(caught.exception))

    def test_underlying_stage_caps_still_bound_the_shared_backtest(self):
        # The per-cutoff and cross-cutoff caps belong to the shared
        # backtest stage, so they must raise before anything
        # aggregates.
        with self.assertRaises(ValueError) as caught:
            build_backtests(
                synthetic_waves(), 1, 50, 0, 50, 2, 2, (4,), 0, 50
            )
        self.assertIn("forecast limit", str(caught.exception))
        with self.assertRaises(ValueError) as caught:
            build_backtests(
                synthetic_waves(), 1, 50, 0, 50, 2, 50, (2, 4), 0, 4
            )
        self.assertIn("backtest limit", str(caught.exception))
        # The deepest scorecard stage cap likewise raises before the
        # regression stage aggregates.
        backtests = build_backtests(
            synthetic_waves(), 1, 50, 0, 50, 2, 50, (2, 4), 0, 50
        )
        with self.assertRaises(ValueError) as caught:
            build_deep_scorecards(backtests["backtests"], 0, 1, 2)
        self.assertIn("scorecard limit", str(caught.exception))


class EmptyRunTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def zero_totals(self):
        return {"identities": 0, "streaks": 0, "regressing_splits": 0}

    def test_empty_deep_cutoffs_returns_empty_streaks(self):
        result = streaks_call(
            self.store,
            self.args,
            **call_args(
                regression_wave_regression_wave_regression_wave_cutoffs=()
            ),
        )
        self.assertEqual(result["streaks"], ())
        self.assertEqual(result["totals"], self.zero_totals())

    def test_no_deep_forecast_regression_waves_returns_empty_streaks(
        self,
    ):
        # The diamond store yields no deepest forecast regression
        # waves, so the range check is skipped and nothing aggregates.
        result = streaks_call(
            self.store,
            self.args,
            **call_args(
                regression_wave_regression_wave_regression_wave_cutoffs=(3,)
            ),
        )
        self.assertEqual(result["streaks"], ())
        self.assertEqual(result["totals"], self.zero_totals())


class IsolationTests(unittest.TestCase):
    def test_success_and_failure_change_nothing(self):
        store, args = diamond_store()
        heads = dict(store._heads)
        appends = copy.deepcopy(store._appends)
        merges = copy.deepcopy(store._merges)
        audit = store.audit_log()

        streaks_call(store, args, **call_args())
        failures = [
            dict(regression_wave_regression_wave_regression_wave_cutoffs="x"),
            dict(
                regression_wave_regression_wave_regression_wave_cutoffs=(True,)
            ),
            dict(regression_wave_regression_wave_regression_wave_cutoffs=(-1,)),
            dict(
                regression_wave_regression_wave_regression_wave_cutoffs=(0, 0)
            ),
            dict(
                regression_wave_regression_wave_regression_wave_cutoffs=(1, 0)
            ),
            dict(
                regression_wave_regression_wave_regression_wave_match_tolerance="x"
            ),
            dict(
                regression_wave_regression_wave_regression_wave_match_tolerance=True
            ),
            dict(
                regression_wave_regression_wave_regression_wave_match_tolerance=-1
            ),
            dict(
                regression_wave_regression_wave_regression_wave_forecast_backtest_limit="x"
            ),
            dict(
                regression_wave_regression_wave_regression_wave_forecast_backtest_limit=False
            ),
            dict(
                regression_wave_regression_wave_regression_wave_forecast_backtest_limit=0
            ),
            dict(
                regression_wave_regression_wave_regression_wave_min_evaluated="x"
            ),
            dict(
                regression_wave_regression_wave_regression_wave_min_evaluated=True
            ),
            dict(
                regression_wave_regression_wave_regression_wave_min_evaluated=-1
            ),
            dict(
                regression_wave_regression_wave_regression_wave_min_cutoffs="x"
            ),
            dict(
                regression_wave_regression_wave_regression_wave_min_cutoffs=False
            ),
            dict(
                regression_wave_regression_wave_regression_wave_min_cutoffs=0
            ),
            dict(
                regression_wave_regression_wave_regression_wave_forecast_scorecard_limit="x"
            ),
            dict(
                regression_wave_regression_wave_regression_wave_forecast_scorecard_limit=False
            ),
            dict(
                regression_wave_regression_wave_regression_wave_forecast_scorecard_limit=0
            ),
            dict(regression_wave_regression_wave_regression_wave_forecast_limit=0),
            dict(regression_wave_regression_wave_regression_wave_horizon=0),
            dict(
                regression_wave_regression_wave_regression_wave_baseline_splits="x"
            ),
            dict(
                regression_wave_regression_wave_regression_wave_baseline_splits=False
            ),
            dict(
                regression_wave_regression_wave_regression_wave_baseline_splits=()
            ),
            dict(
                regression_wave_regression_wave_regression_wave_baseline_splits=(0,)
            ),
            dict(
                regression_wave_regression_wave_regression_wave_baseline_splits=(2, 1)
            ),
            dict(
                regression_wave_regression_wave_regression_wave_min_evaluated_delta="x"
            ),
            dict(
                regression_wave_regression_wave_regression_wave_min_evaluated_delta=True
            ),
            dict(
                regression_wave_regression_wave_regression_wave_min_evaluated_delta=-1
            ),
            dict(
                regression_wave_regression_wave_regression_wave_min_hit_rate_drop="x"
            ),
            dict(
                regression_wave_regression_wave_regression_wave_min_hit_rate_drop=(1, 0)
            ),
            dict(
                regression_wave_regression_wave_regression_wave_min_hit_rate_drop=(2, 1)
            ),
            dict(
                regression_wave_regression_wave_regression_wave_regression_limit="x"
            ),
            dict(
                regression_wave_regression_wave_regression_wave_regression_limit=False
            ),
            dict(
                regression_wave_regression_wave_regression_wave_regression_limit=0
            ),
            dict(
                regression_wave_regression_wave_regression_wave_min_consecutive_splits="x"
            ),
            dict(
                regression_wave_regression_wave_regression_wave_min_consecutive_splits=False
            ),
            dict(
                regression_wave_regression_wave_regression_wave_min_consecutive_splits=0
            ),
            dict(
                regression_wave_regression_wave_regression_wave_regression_streak_limit="x"
            ),
            dict(
                regression_wave_regression_wave_regression_wave_regression_streak_limit=False
            ),
            dict(
                regression_wave_regression_wave_regression_wave_regression_streak_limit=0
            ),
            dict(drift_limit=2),
            dict(scan_limit=1),
            dict(reference="ghost"),
        ]
        for overrides in failures:
            with self.subTest(overrides=overrides):
                merged = call_args()
                merged.update(overrides)
                with self.assertRaises(Exception):
                    streaks_call(store, args, **merged)

        self.assertEqual(dict(store._heads), heads)
        self.assertEqual(copy.deepcopy(store._appends), appends)
        self.assertEqual(copy.deepcopy(store._merges), merges)
        self.assertEqual(store.audit_log(), audit)


class TokenTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, token=None, **overrides):
        return streaks_call(
            self.store, self.args, token=token, **overrides
        )

    def test_success_consumes_one_read_and_failure_refunds(self):
        token = self.store.create_snapshot(2)
        self.call(token=token, **call_args())
        # Stage failures still refund.
        with self.assertRaises(ValueError):
            self.call(token=token, **call_args(drift_limit=2))
        with self.assertRaises(ValueError):
            self.call(token=token, **call_args(scan_limit=1))
        with self.assertRaises(ValueError):
            self.call(token=token, **call_args(wave_cutoffs=(9,)))
        # Parameter validation runs before the token is touched.
        with self.assertRaises(TypeError):
            self.call(
                token=token,
                **call_args(
                    regression_wave_regression_wave_regression_wave_baseline_splits="x"
                ),
            )
        with self.assertRaises(ValueError):
            self.call(
                token=token,
                **call_args(
                    regression_wave_regression_wave_regression_wave_regression_streak_limit=0
                ),
            )
        # The second allowed read succeeds; the token then expires.
        self.call(token=token, **call_args())
        with self.assertRaises(RuntimeError):
            self.call(token=token, **call_args())

    def test_tokenless_query_leaves_snapshot_quotas_intact(self):
        token = self.store.create_snapshot(1)
        self.call(**call_args())
        self.call(**call_args())
        # The single read quota is still available after tokenless calls.
        self.call(token=token, **call_args())
        with self.assertRaises(RuntimeError):
            self.call(token=token, **call_args())

    def test_results_come_from_the_frozen_view(self):
        before = self.call(**call_args())
        token = self.store.create_snapshot(4)
        self.store.append("a", "a2", 7, {"x": -1})
        self.store.create("d", "root")
        self.store.append("d", "d0", 8, {"z": 3})
        self.store.merge("a", "d", "M", 9, {"z": 1})
        self.store.append("main", "r3", 10, {"x": 1})
        self.assertEqual(self.call(token=token, **call_args()), before)


if __name__ == "__main__":
    unittest.main()
