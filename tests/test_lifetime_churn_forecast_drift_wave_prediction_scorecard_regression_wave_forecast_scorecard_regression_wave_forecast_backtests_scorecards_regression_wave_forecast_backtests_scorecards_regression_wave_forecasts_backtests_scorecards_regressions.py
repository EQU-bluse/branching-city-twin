import copy
import inspect
import unittest

from city_twin.branches import BranchStore

from tests.test_lifetime_churn import diamond_store
from tests.test_lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_wave_forecast_scorecard_regression_wave_forecast_backtests_scorecards_regression_wave_forecast_backtests_scorecards_regression_wave_forecasts_backtests_scorecards import (
    BACKTESTS_QUERY,
    SCORECARDS_QUERY,
    backtest_record,
    backtests_kwargs,
    scorecards_kwargs,
)


REGRESSIONS_QUERY = SCORECARDS_QUERY + "_regressions"

PREFIX = "regression_wave_regression_wave_regression_wave_regression_wave"
BASELINE_CUTOFFS = PREFIX + "_baseline_cutoffs"
MIN_EVALUATED_DELTA = PREFIX + "_min_evaluated_delta"
MIN_HIT_RATE_DROP = PREFIX + "_min_hit_rate_drop"
REGRESSION_LIMIT = PREFIX + "_regression_limit"

# Field offsets inside the totals tuple, in field order.
T_IDENTITIES = 0
T_BASELINE_EVALUATED = 1
T_BASELINE_HIT = 2
T_BASELINE_MISSED = 3
T_OBSERVATION_EVALUATED = 4
T_OBSERVATION_HIT = 5
T_OBSERVATION_MISSED = 6


def regressions_kwargs(store_args, **overrides) -> dict:
    baseline_cutoffs = overrides.pop(BASELINE_CUTOFFS, 1)
    min_evaluated_delta = overrides.pop(MIN_EVALUATED_DELTA, 0)
    min_hit_rate_drop = overrides.pop(MIN_HIT_RATE_DROP, (0, 1))
    regression_limit = overrides.pop(REGRESSION_LIMIT, 50)
    kwargs = scorecards_kwargs(store_args, **overrides)
    kwargs[BASELINE_CUTOFFS] = baseline_cutoffs
    kwargs[MIN_EVALUATED_DELTA] = min_evaluated_delta
    kwargs[MIN_HIT_RATE_DROP] = min_hit_rate_drop
    kwargs[REGRESSION_LIMIT] = regression_limit
    return kwargs


def regressions_call(store, store_args, **overrides):
    return getattr(store, REGRESSIONS_QUERY)(
        **regressions_kwargs(store_args, **overrides)
    )


def build_regressions(
    records,
    scorecards,
    baseline_cutoffs=1,
    min_evaluated_delta=0,
    min_hit_rate_drop=(0, 1),
    regression_limit=50,
):
    return BranchStore._build_lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_wave_forecast_scorecard_regression_wave_forecast_backtests_scorecards_regression_wave_forecast_backtests_scorecards_regression_wave_forecasts_backtests_scorecards_regressions_result(
        records,
        scorecards,
        baseline_cutoffs,
        min_evaluated_delta,
        min_hit_rate_drop,
        regression_limit,
    )


def kept_scorecards(*identities):
    # The builder only reads the identity at tuple offset zero.
    return tuple((identity,) for identity in identities)


def regressing_records(identity):
    return (
        backtest_record(2, identity, 2, 0),
        backtest_record(4, identity, 1, 1),
        backtest_record(6, identity, 0, 2),
        backtest_record(8, identity, 1, 1),
    )


class SignatureTests(unittest.TestCase):
    def test_public_signature_appends_four_parameters(self):
        parameters = list(
            inspect.signature(
                getattr(BranchStore, REGRESSIONS_QUERY)
            ).parameters.values()
        )
        names = [p.name for p in parameters]
        scorecard_names = list(
            inspect.signature(getattr(BranchStore, SCORECARDS_QUERY)).parameters
        )
        self.assertEqual(
            names,
            scorecard_names[:-1]
            + [
                BASELINE_CUTOFFS,
                MIN_EVALUATED_DELTA,
                MIN_HIT_RATE_DROP,
                REGRESSION_LIMIT,
                "token",
            ],
        )
        self.assertEqual(
            [p.default for p in parameters[:-1]],
            [inspect.Parameter.empty] * (len(parameters) - 1),
        )
        self.assertIsNone(parameters[-1].default)

    def test_result_shape_and_key_order(self):
        result = build_regressions(
            regressing_records("A"), kept_scorecards("A"), 2, 0, (1, 2)
        )
        self.assertEqual(list(result), ["regressions", "totals"])
        self.assertIsInstance(result["regressions"], tuple)
        regression = result["regressions"][0]
        self.assertIsInstance(regression, dict)
        self.assertEqual(
            list(regression),
            [
                "identity",
                "baseline_cutoffs",
                "observation_cutoffs",
                "baseline_evaluated",
                "observation_evaluated",
                "baseline_hit_rate",
                "observation_hit_rate",
                "rate_drop",
                "first_regression_cutoff",
            ],
        )
        self.assertIsInstance(result["totals"], tuple)
        self.assertEqual(len(result["totals"]), 7)


class ResultTests(unittest.TestCase):
    def test_segments_split_and_aggregate(self):
        result = build_regressions(
            regressing_records("A"), kept_scorecards("A"), 1, 0, (1, 4)
        )
        (regression,) = result["regressions"]
        # Baseline is the cutoff-2 record (2/2 hit); the observation
        # segment holds the remaining three records (2/6 hit).
        self.assertEqual(
            regression,
            {
                "identity": "A",
                "baseline_cutoffs": 1,
                "observation_cutoffs": 3,
                "baseline_evaluated": 2,
                "observation_evaluated": 6,
                "baseline_hit_rate": (1, 1),
                "observation_hit_rate": (1, 3),
                "rate_drop": (2, 3),
                "first_regression_cutoff": 4,
            },
        )
        self.assertEqual(
            result["totals"], (1, 2, 2, 0, 6, 2, 4)
        )

    def test_only_scorecard_identities_are_candidates(self):
        records = regressing_records("A") + regressing_records("B")
        result = build_regressions(
            records, kept_scorecards("A"), 1, 0, (0, 1)
        )
        self.assertEqual(
            [r["identity"] for r in result["regressions"]], ["A"]
        )

    def test_both_segments_must_hold_records(self):
        # A single record leaves an empty observation segment.
        result = build_regressions(
            (backtest_record(2, "A", 1, 0),),
            kept_scorecards("A"),
            1,
            0,
            (0, 1),
        )
        self.assertEqual(result["regressions"], ())
        # Two baseline cutoffs swallow every record.
        result = build_regressions(
            regressing_records("A")[:2],
            kept_scorecards("A"),
            2,
            0,
            (0, 1),
        )
        self.assertEqual(result["regressions"], ())

    def test_zero_evaluated_in_either_segment_disqualifies(self):
        records = (
            backtest_record(2, "A", 0, 0, 3, 0),
            backtest_record(4, "A", 0, 1),
        )
        result = build_regressions(
            records, kept_scorecards("A"), 1, 0, (0, 1)
        )
        self.assertEqual(result["regressions"], ())
        records = (
            backtest_record(2, "A", 1, 0),
            backtest_record(4, "A", 0, 0, 2, 0),
        )
        result = build_regressions(
            records, kept_scorecards("A"), 1, 0, (0, 1)
        )
        self.assertEqual(result["regressions"], ())

    def test_min_evaluated_delta_bounds_the_observation_growth(self):
        records = (
            backtest_record(2, "A", 2, 0),
            backtest_record(4, "A", 0, 2),
            backtest_record(6, "A", 0, 2),
        )
        # Observation evaluated 4 vs baseline 2: delta 2.
        result = build_regressions(
            records, kept_scorecards("A"), 1, 2, (0, 1)
        )
        self.assertEqual(len(result["regressions"]), 1)
        result = build_regressions(
            records, kept_scorecards("A"), 1, 3, (0, 1)
        )
        self.assertEqual(result["regressions"], ())

    def test_hit_rate_drop_threshold_is_inclusive(self):
        records = (
            backtest_record(2, "A", 3, 1),
            backtest_record(4, "A", 1, 3),
        )
        # Baseline 3/4, observation 1/4: the drop is exactly 1/2.
        result = build_regressions(
            records, kept_scorecards("A"), 1, 0, (1, 2)
        )
        self.assertEqual(len(result["regressions"]), 1)
        result = build_regressions(
            records, kept_scorecards("A"), 1, 0, (2, 3)
        )
        self.assertEqual(result["regressions"], ())

    def test_rate_drop_is_reduced_to_lowest_terms(self):
        records = (
            backtest_record(2, "A", 3, 1),
            backtest_record(4, "A", 1, 3),
        )
        result = build_regressions(
            records, kept_scorecards("A"), 1, 0, (0, 1)
        )
        (regression,) = result["regressions"]
        self.assertEqual(regression["baseline_hit_rate"], (3, 4))
        self.assertEqual(regression["observation_hit_rate"], (1, 4))
        self.assertEqual(regression["rate_drop"], (1, 2))

    def test_first_regression_cutoff_tracks_accumulated_observation(self):
        records = (
            backtest_record(2, "A", 2, 0),
            backtest_record(4, "A", 2, 0),
            backtest_record(6, "A", 0, 2),
            backtest_record(8, "A", 0, 2),
        )
        # After cutoff 6 the accumulated observation is 2/4 hit with an
        # evaluated delta of two: both thresholds are first met at 6.
        result = build_regressions(
            records, kept_scorecards("A"), 1, 2, (1, 2)
        )
        (regression,) = result["regressions"]
        self.assertEqual(regression["first_regression_cutoff"], 6)
        # A 2/3 drop threshold is first met only after cutoff 8.
        result = build_regressions(
            records, kept_scorecards("A"), 1, 0, (2, 3)
        )
        (regression,) = result["regressions"]
        self.assertEqual(regression["first_regression_cutoff"], 8)

    def test_sort_order_drop_then_observation_evaluated_then_identity(
        self,
    ):
        records = (
            # "A" drops to zero over two evaluated observations.
            backtest_record(1, "A", 2, 0),
            backtest_record(2, "A", 0, 2),
            # "B" drops to zero over four evaluated observations.
            backtest_record(1, "B", 2, 0),
            backtest_record(2, "B", 0, 2),
            backtest_record(3, "B", 0, 2),
            # "C" ties "B" fully; Unicode code point order decides.
            backtest_record(1, "C", 4, 0),
            backtest_record(2, "C", 0, 4),
        )
        result = build_regressions(
            records, kept_scorecards("A", "B", "C"), 1, 0, (0, 1)
        )
        self.assertEqual(
            [r["identity"] for r in result["regressions"]],
            ["B", "C", "A"],
        )

    def test_regression_limit_bounds_the_complete_set_without_truncating(
        self,
    ):
        records = (
            regressing_records("A")
            + regressing_records("B")
            + regressing_records("C")
        )
        scorecards = kept_scorecards("A", "B", "C")
        with self.assertRaises(ValueError) as caught:
            build_regressions(records, scorecards, 1, 0, (0, 1), 2)
        self.assertIn("regression limit exceeded", str(caught.exception))
        # Exactly at the limit passes.
        result = build_regressions(records, scorecards, 1, 0, (0, 1), 3)
        self.assertEqual(len(result["regressions"]), 3)

    def test_empty_records_return_empty_tuple_and_seven_zero_totals(self):
        result = build_regressions((), kept_scorecards("A"), 1, 0, (0, 1))
        self.assertEqual(result["regressions"], ())
        self.assertEqual(result["totals"], (0, 0, 0, 0, 0, 0, 0))

    def test_every_object_is_fresh(self):
        records = regressing_records("A")
        result = build_regressions(
            records, kept_scorecards("A"), 1, 0, (0, 1)
        )
        collected = []

        def collect(value):
            if isinstance(value, (dict, list, tuple, set)):
                collected.append(value)
                for item in (
                    value.values() if isinstance(value, dict) else value
                ):
                    collect(item)

        collect(result)
        for value in collected:
            if isinstance(value, dict):
                value["polluted"] = True
            elif isinstance(value, list):
                value.append("polluted")
        again = build_regressions(
            records, kept_scorecards("A"), 1, 0, (0, 1)
        )
        self.assertEqual(
            again,
            build_regressions(
                regressing_records("A"), kept_scorecards("A"), 1, 0, (0, 1)
            ),
        )


class ChainTests(unittest.TestCase):
    """The public query refines exactly what the scorecards stage keeps."""

    def test_public_query_matches_the_builder_composition(self):
        store, args = diamond_store()
        backtests = getattr(store, BACKTESTS_QUERY)(
            **backtests_kwargs(args)
        )
        scorecards = getattr(store, SCORECARDS_QUERY)(
            **scorecards_kwargs(args)
        )
        result = regressions_call(store, args)
        self.assertEqual(
            result,
            build_regressions(
                backtests["backtests"],
                scorecards["scorecards"],
                1,
                0,
                (0, 1),
                50,
            ),
        )

    def test_public_query_matches_with_tighter_thresholds(self):
        store, args = diamond_store()
        overrides = {
            BASELINE_CUTOFFS: 2,
            MIN_EVALUATED_DELTA: 1,
            MIN_HIT_RATE_DROP: (1, 4),
            REGRESSION_LIMIT: 7,
        }
        backtests = getattr(store, BACKTESTS_QUERY)(
            **backtests_kwargs(args)
        )
        scorecards = getattr(store, SCORECARDS_QUERY)(
            **scorecards_kwargs(args)
        )
        result = regressions_call(store, args, **overrides)
        self.assertEqual(
            result,
            build_regressions(
                backtests["backtests"],
                scorecards["scorecards"],
                2,
                1,
                (1, 4),
                7,
            ),
        )


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        return regressions_call(self.store, self.args, **overrides)

    def test_baseline_cutoffs_must_be_a_non_bool_positive_int(self):
        for bad in (True, False, 1.0, "1", None, (1,)):
            with self.subTest(value=bad):
                with self.assertRaises(TypeError):
                    self.call(**{BASELINE_CUTOFFS: bad})
        with self.assertRaises(ValueError):
            self.call(**{BASELINE_CUTOFFS: 0})
        with self.assertRaises(ValueError):
            self.call(**{BASELINE_CUTOFFS: -2})

    def test_min_evaluated_delta_must_be_a_non_bool_non_negative_int(self):
        for bad in (True, False, 1.0, "1", None, (1,)):
            with self.subTest(value=bad):
                with self.assertRaises(TypeError):
                    self.call(**{MIN_EVALUATED_DELTA: bad})
        with self.assertRaises(ValueError):
            self.call(**{MIN_EVALUATED_DELTA: -1})

    def test_min_hit_rate_drop_must_be_a_valid_fraction_tuple(self):
        for bad in ([0, 1], "1/2", 0, None):
            with self.subTest(value=bad):
                with self.assertRaises(TypeError):
                    self.call(**{MIN_HIT_RATE_DROP: bad})
        for bad in ((True, 1), (1, True), (0.5, 1), ("1", 1)):
            with self.subTest(value=bad):
                with self.assertRaises(TypeError):
                    self.call(**{MIN_HIT_RATE_DROP: bad})
        for bad in ((1,), (1, 2, 3), (-1, 2), (1, 0), (1, -2), (3, 2)):
            with self.subTest(value=bad):
                with self.assertRaises(ValueError):
                    self.call(**{MIN_HIT_RATE_DROP: bad})
        # The boundary fractions pass validation.
        self.call(**{MIN_HIT_RATE_DROP: (0, 1)})
        self.call(**{MIN_HIT_RATE_DROP: (1, 1)})

    def test_regression_limit_must_be_a_non_bool_positive_int(self):
        for bad in (True, False, 1.0, "1", None, (1,)):
            with self.subTest(value=bad):
                with self.assertRaises(TypeError):
                    self.call(**{REGRESSION_LIMIT: bad})
        with self.assertRaises(ValueError):
            self.call(**{REGRESSION_LIMIT: 0})
        with self.assertRaises(ValueError):
            self.call(**{REGRESSION_LIMIT: -3})

    def test_new_parameters_validated_in_signature_order(self):
        scorecard_limit = (
            "regression_wave_regression_wave_regression_wave_"
            "regression_wave_forecast_scorecard_limit"
        )
        # The scorecard limit fails before the new baseline cutoffs.
        with self.assertRaises(TypeError):
            self.call(**{scorecard_limit: "x", BASELINE_CUTOFFS: "x"})
        with self.assertRaises(ValueError):
            self.call(**{scorecard_limit: 0, BASELINE_CUTOFFS: 0})
        # The baseline cutoffs fail before the minimum evaluated delta.
        with self.assertRaises(TypeError):
            self.call(**{BASELINE_CUTOFFS: "x", MIN_EVALUATED_DELTA: "x"})
        with self.assertRaises(ValueError):
            self.call(**{BASELINE_CUTOFFS: 0, MIN_EVALUATED_DELTA: -1})
        # The minimum evaluated delta fails before the hit-rate drop.
        with self.assertRaises(TypeError):
            self.call(**{MIN_EVALUATED_DELTA: "x", MIN_HIT_RATE_DROP: "x"})
        with self.assertRaises(ValueError):
            self.call(**{MIN_EVALUATED_DELTA: -1, MIN_HIT_RATE_DROP: (1,)})
        # The hit-rate drop fails before the regression limit.
        with self.assertRaises(TypeError):
            self.call(**{MIN_HIT_RATE_DROP: "x", REGRESSION_LIMIT: "x"})
        with self.assertRaises(ValueError):
            self.call(**{MIN_HIT_RATE_DROP: (1,), REGRESSION_LIMIT: 0})
        # All pass before the split membership check fires.
        with self.assertRaises(ValueError) as caught:
            self.call(
                split_cutoffs=(99,),
                **{
                    BASELINE_CUTOFFS: 1,
                    MIN_EVALUATED_DELTA: 0,
                    MIN_HIT_RATE_DROP: (0, 1),
                    REGRESSION_LIMIT: 1,
                },
            )
        self.assertIn("split_cutoff 99", str(caught.exception))

    def test_all_parameter_checks_precede_state_lookups(self):
        with self.assertRaises(TypeError):
            self.call(reference="ghost", **{BASELINE_CUTOFFS: "x"})
        with self.assertRaises(ValueError):
            self.call(reference="ghost", **{BASELINE_CUTOFFS: 0})
        with self.assertRaises(TypeError):
            self.call(reference="ghost", **{MIN_EVALUATED_DELTA: "x"})
        with self.assertRaises(ValueError):
            self.call(reference="ghost", **{MIN_EVALUATED_DELTA: -1})
        with self.assertRaises(TypeError):
            self.call(reference="ghost", **{MIN_HIT_RATE_DROP: "x"})
        with self.assertRaises(ValueError):
            self.call(reference="ghost", **{MIN_HIT_RATE_DROP: (1,)})
        with self.assertRaises(TypeError):
            self.call(reference="ghost", **{REGRESSION_LIMIT: "x"})
        with self.assertRaises(ValueError):
            self.call(reference="ghost", **{REGRESSION_LIMIT: 0})

    def test_underlying_stage_limits_and_errors_are_unchanged(self):
        with self.assertRaises(ValueError) as caught:
            self.call(wave_cutoffs=(9,))
        self.assertIn("out of range", str(caught.exception))
        with self.assertRaises(ValueError):
            self.call(drift_limit=2)
        with self.assertRaises(ValueError):
            self.call(scan_limit=1)


class EmptyRunTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def test_empty_deepest_cutoffs_returns_empty_regressions(self):
        result = regressions_call(
            self.store,
            self.args,
            regression_wave_regression_wave_regression_wave_cutoffs=(),
        )
        self.assertEqual(result["regressions"], ())
        self.assertEqual(result["totals"], (0, 0, 0, 0, 0, 0, 0))

    def test_no_qualifying_identity_returns_empty_regressions(self):
        result = regressions_call(
            self.store,
            self.args,
            **{MIN_EVALUATED_DELTA: 100},
        )
        self.assertEqual(result["regressions"], ())
        self.assertEqual(result["totals"], (0, 0, 0, 0, 0, 0, 0))


class IsolationTests(unittest.TestCase):
    def test_success_and_failure_change_nothing(self):
        store, args = diamond_store()
        heads = dict(store._heads)
        appends = copy.deepcopy(store._appends)
        merges = copy.deepcopy(store._merges)
        audit = store.audit_log()

        regressions_call(store, args)
        failures = [
            {BASELINE_CUTOFFS: "x"},
            {BASELINE_CUTOFFS: True},
            {BASELINE_CUTOFFS: 0},
            {MIN_EVALUATED_DELTA: "x"},
            {MIN_EVALUATED_DELTA: False},
            {MIN_EVALUATED_DELTA: -1},
            {MIN_HIT_RATE_DROP: "x"},
            {MIN_HIT_RATE_DROP: (1,)},
            {MIN_HIT_RATE_DROP: (True, 1)},
            {MIN_HIT_RATE_DROP: (-1, 2)},
            {MIN_HIT_RATE_DROP: (1, 0)},
            {MIN_HIT_RATE_DROP: (3, 2)},
            {REGRESSION_LIMIT: "x"},
            {REGRESSION_LIMIT: False},
            {REGRESSION_LIMIT: 0},
            dict(
                regression_wave_regression_wave_regression_wave_cutoffs="x"
            ),
            dict(
                regression_wave_regression_wave_regression_wave_cutoffs=(
                    True,
                )
            ),
            dict(
                regression_wave_regression_wave_regression_wave_cutoffs=(-1,)
            ),
            dict(
                regression_wave_regression_wave_regression_wave_cutoffs=(
                    0,
                    0,
                )
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
            dict(drift_limit=2),
            dict(scan_limit=1),
            dict(reference="ghost"),
        ]
        for overrides in failures:
            with self.subTest(overrides=overrides):
                with self.assertRaises(Exception):
                    regressions_call(store, args, **overrides)

        self.assertEqual(dict(store._heads), heads)
        self.assertEqual(copy.deepcopy(store._appends), appends)
        self.assertEqual(copy.deepcopy(store._merges), merges)
        self.assertEqual(store.audit_log(), audit)


class TokenTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, token=None, **overrides):
        kwargs = regressions_kwargs(self.args, **overrides)
        if token is not None:
            kwargs["token"] = token
        return getattr(self.store, REGRESSIONS_QUERY)(**kwargs)

    def test_success_consumes_one_read_and_failure_refunds(self):
        token = self.store.create_snapshot(2)
        self.call(token=token)
        # Stage failures still refund.
        with self.assertRaises(ValueError):
            self.call(token=token, drift_limit=2)
        with self.assertRaises(ValueError):
            self.call(token=token, scan_limit=1)
        with self.assertRaises(ValueError):
            self.call(token=token, wave_cutoffs=(9,))
        # Parameter validation runs before the token is touched.
        with self.assertRaises(TypeError):
            self.call(token=token, **{BASELINE_CUTOFFS: "x"})
        with self.assertRaises(ValueError):
            self.call(token=token, **{MIN_HIT_RATE_DROP: (1,)})
        with self.assertRaises(ValueError):
            self.call(token=token, **{REGRESSION_LIMIT: 0})
        # The second allowed read succeeds; the token then expires.
        self.call(token=token)
        with self.assertRaises(RuntimeError):
            self.call(token=token)

    def test_tokenless_query_leaves_snapshot_quotas_intact(self):
        token = self.store.create_snapshot(1)
        self.call()
        self.call()
        # The single read quota is still available after tokenless calls.
        self.call(token=token)
        with self.assertRaises(RuntimeError):
            self.call(token=token)

    def test_results_come_from_the_frozen_view(self):
        before = self.call()
        token = self.store.create_snapshot(4)
        self.store.append("a", "a2", 7, {"x": -1})
        self.store.create("d", "root")
        self.store.append("d", "d0", 8, {"z": 3})
        self.store.merge("a", "d", "M", 9, {"z": 1})
        self.store.append("main", "r3", 10, {"x": 1})
        self.assertEqual(self.call(token=token), before)


if __name__ == "__main__":
    unittest.main()
