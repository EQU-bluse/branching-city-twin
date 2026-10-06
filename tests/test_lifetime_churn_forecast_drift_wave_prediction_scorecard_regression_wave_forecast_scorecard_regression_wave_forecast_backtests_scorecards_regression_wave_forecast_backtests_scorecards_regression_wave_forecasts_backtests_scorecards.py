import copy
import inspect
import unittest

from city_twin.branches import BranchStore

from tests.test_lifetime_churn import diamond_store
from tests.test_lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_wave_forecast_scorecard_regression_wave_forecast_backtests_scorecards_regression_wave_periodicities import (
    deep_call_args,
    periodicities_kwargs,
)


BACKTESTS_QUERY = "lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_wave_forecast_scorecard_regression_wave_forecast_backtests_scorecards_regression_wave_forecast_backtests_scorecards_regression_wave_forecasts_backtests"
SCORECARDS_QUERY = BACKTESTS_QUERY + "_scorecards"

MIN_EVALUATED = (
    "regression_wave_regression_wave_regression_wave_regression_wave_min_evaluated"
)
MIN_CUTOFFS = (
    "regression_wave_regression_wave_regression_wave_regression_wave_min_cutoffs"
)
SCORECARD_LIMIT = "regression_wave_regression_wave_regression_wave_regression_wave_forecast_scorecard_limit"

# Field offsets inside a scorecard tuple, in field order.
IDENTITY = 0
CUTOFFS = 1
HIT = 2
MISSED = 3
UNRESOLVED = 4
UNEXPECTED = 5
EVALUATED = 6
HIT_RATE = 7
FIRST_FAILURE = 8

# Field offsets inside the totals tuple, in field order.
T_IDENTITIES = 0
T_CUTOFFS = 1
T_HIT = 2
T_MISSED = 3
T_UNRESOLVED = 4
T_UNEXPECTED = 5
T_EVALUATED = 6


def backtests_kwargs(store_args, **overrides) -> dict:
    merged = deep_call_args()
    merged.update(overrides)
    horizon = merged.pop(
        "regression_wave_regression_wave_regression_wave_horizon", 10
    )
    forecast_limit = merged.pop(
        "regression_wave_regression_wave_regression_wave_forecast_limit", 50
    )
    cutoffs = merged.pop(
        "regression_wave_regression_wave_regression_wave_cutoffs", (0,)
    )
    match_tolerance = merged.pop(
        "regression_wave_regression_wave_regression_wave_match_tolerance", 1
    )
    backtest_limit = merged.pop(
        "regression_wave_regression_wave_regression_wave_forecast_backtest_limit",
        50,
    )
    merged.setdefault("cutoffs", (2, 3, 4, 5, 6))
    merged.setdefault("split_cutoffs", (4,))
    kwargs = periodicities_kwargs(store_args, **merged)
    kwargs["regression_wave_regression_wave_regression_wave_horizon"] = horizon
    kwargs[
        "regression_wave_regression_wave_regression_wave_forecast_limit"
    ] = forecast_limit
    kwargs["regression_wave_regression_wave_regression_wave_cutoffs"] = cutoffs
    kwargs[
        "regression_wave_regression_wave_regression_wave_match_tolerance"
    ] = match_tolerance
    kwargs[
        "regression_wave_regression_wave_regression_wave_forecast_backtest_limit"
    ] = backtest_limit
    deepest = dict(
        regression_wave_regression_wave_regression_wave_min_evaluated=0,
        regression_wave_regression_wave_regression_wave_min_cutoffs=1,
        regression_wave_regression_wave_regression_wave_forecast_scorecard_limit=50,
        regression_wave_regression_wave_regression_wave_baseline_splits=(1,),
        regression_wave_regression_wave_regression_wave_min_evaluated_delta=0,
        regression_wave_regression_wave_regression_wave_min_hit_rate_drop=(0, 1),
        regression_wave_regression_wave_regression_wave_regression_limit=50,
        regression_wave_regression_wave_regression_wave_min_consecutive_splits=1,
        regression_wave_regression_wave_regression_wave_regression_streak_limit=50,
        regression_wave_regression_wave_regression_wave_min_active_regressions=1,
        regression_wave_regression_wave_regression_wave_regression_wave_limit=50,
        regression_wave_regression_wave_regression_wave_min_regression_wave_occurrences=1,
        regression_wave_regression_wave_regression_wave_regression_wave_recurrence_limit=50,
        regression_wave_regression_wave_regression_wave_max_regression_wave_jitter=10,
        regression_wave_regression_wave_regression_wave_regression_wave_periodicity_limit=50,
        regression_wave_regression_wave_regression_wave_regression_wave_horizon=10,
        regression_wave_regression_wave_regression_wave_regression_wave_forecast_limit=50,
        regression_wave_regression_wave_regression_wave_regression_wave_cutoffs=(0,),
        regression_wave_regression_wave_regression_wave_regression_wave_match_tolerance=1,
        regression_wave_regression_wave_regression_wave_regression_wave_forecast_backtest_limit=50,
    )
    deepest.update(kwargs)
    return deepest


def scorecards_kwargs(store_args, **overrides) -> dict:
    min_evaluated = overrides.pop(MIN_EVALUATED, 0)
    min_cutoffs = overrides.pop(MIN_CUTOFFS, 1)
    scorecard_limit = overrides.pop(SCORECARD_LIMIT, 50)
    kwargs = backtests_kwargs(store_args, **overrides)
    kwargs[MIN_EVALUATED] = min_evaluated
    kwargs[MIN_CUTOFFS] = min_cutoffs
    kwargs[SCORECARD_LIMIT] = scorecard_limit
    return kwargs


def backtests_call(store, store_args, **overrides):
    return getattr(store, BACKTESTS_QUERY)(
        **backtests_kwargs(store_args, **overrides)
    )


def scorecards_call(store, store_args, **overrides):
    return getattr(store, SCORECARDS_QUERY)(
        **scorecards_kwargs(store_args, **overrides)
    )


def build_scorecards(records, min_evaluated=0, min_cutoffs=1, limit=50):
    return BranchStore._build_lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_wave_forecast_scorecard_regression_wave_forecast_backtests_scorecards_regression_wave_forecast_backtests_scorecards_regression_wave_forecasts_backtests_scorecards_result(
        records, min_evaluated, min_cutoffs, limit
    )


def backtest_record(cutoff, identity, hit, missed, unresolved=0, unexpected=0):
    return {
        "cutoff": cutoff,
        "identity": identity,
        "period": 1,
        "jitter": 0,
        "last_wave": cutoff,
        "targets": (),
        "hit": hit,
        "missed": missed,
        "unresolved": unresolved,
        "unexpected": unexpected,
    }


class SignatureTests(unittest.TestCase):
    def test_public_signature_appends_three_parameters(self):
        parameters = list(
            inspect.signature(
                getattr(BranchStore, SCORECARDS_QUERY)
            ).parameters.values()
        )
        names = [p.name for p in parameters]
        backtest_names = list(
            inspect.signature(getattr(BranchStore, BACKTESTS_QUERY)).parameters
        )
        self.assertEqual(
            names,
            backtest_names[:-1]
            + [MIN_EVALUATED, MIN_CUTOFFS, SCORECARD_LIMIT, "token"],
        )
        self.assertEqual(
            [p.default for p in parameters[:-1]],
            [inspect.Parameter.empty] * (len(parameters) - 1),
        )
        self.assertIsNone(parameters[-1].default)

    def test_result_shape_and_field_order(self):
        result = build_scorecards(
            (
                backtest_record(2, "A", 2, 0),
                backtest_record(4, "A", 1, 1, 2, 1),
            )
        )
        self.assertEqual(list(result), ["scorecards", "totals"])
        self.assertIsInstance(result["scorecards"], tuple)
        scorecard = result["scorecards"][0]
        self.assertIsInstance(scorecard, tuple)
        self.assertEqual(
            scorecard,
            ("A", 2, 3, 1, 2, 1, 4, (3, 4), 4),
        )
        self.assertIsInstance(result["totals"], tuple)
        self.assertEqual(len(result["totals"]), 7)
        self.assertEqual(result["totals"], (1, 2, 3, 1, 2, 1, 4))


class ResultTests(unittest.TestCase):
    def test_identity_aggregates_across_cutoffs(self):
        result = build_scorecards(
            (
                backtest_record(1, "A", 1, 0),
                backtest_record(1, "B", 0, 1),
                backtest_record(2, "A", 0, 1, 1, 2),
                backtest_record(2, "B", 3, 0),
                backtest_record(3, "A", 2, 0),
            )
        )
        scorecards = {
            scorecard[IDENTITY]: scorecard
            for scorecard in result["scorecards"]
        }
        self.assertEqual(
            scorecards["A"],
            ("A", 3, 3, 1, 1, 2, 4, (3, 4), 2),
        )
        self.assertEqual(
            scorecards["B"],
            ("B", 2, 3, 1, 0, 0, 4, (3, 4), 1),
        )
        self.assertEqual(result["totals"], (2, 3, 6, 2, 1, 2, 8))

    def test_hit_rate_is_reduced_and_zero_evaluated_is_fixed(self):
        result = build_scorecards(
            (
                backtest_record(1, "A", 2, 2),
                backtest_record(1, "B", 0, 0, 3, 0),
            )
        )
        scorecards = {
            scorecard[IDENTITY]: scorecard
            for scorecard in result["scorecards"]
        }
        self.assertEqual(scorecards["A"][HIT_RATE], (1, 2))
        self.assertEqual(scorecards["B"][HIT_RATE], (0, 0))
        self.assertIsNone(scorecards["B"][FIRST_FAILURE])

    def test_first_failure_tracks_missed_or_unexpected_not_unresolved(
        self,
    ):
        result = build_scorecards(
            (
                backtest_record(1, "A", 0, 0, 5, 0),
                backtest_record(2, "A", 1, 0),
                backtest_record(3, "A", 0, 0, 0, 2),
                backtest_record(4, "A", 0, 1),
            )
        )
        scorecard = result["scorecards"][0]
        # The cutoff-3 unexpected count fails before the cutoff-4 miss.
        self.assertEqual(scorecard[FIRST_FAILURE], 3)

    def test_min_evaluated_and_min_cutoffs_filter(self):
        records = (
            backtest_record(1, "A", 1, 0),
            backtest_record(2, "A", 1, 0),
            backtest_record(1, "B", 0, 0, 2, 0),
            backtest_record(2, "B", 0, 0, 1, 0),
            backtest_record(1, "C", 2, 0),
        )
        result = build_scorecards(records, min_evaluated=1, min_cutoffs=2)
        self.assertEqual(
            [scorecard[IDENTITY] for scorecard in result["scorecards"]],
            ["A"],
        )
        self.assertEqual(result["totals"], (1, 2, 2, 0, 0, 0, 2))

    def test_scorecards_sort_by_rate_then_unexpected_missed_identity(self):
        records = (
            # "A" and "B" tie at 1/2; fewer unexpected wins.
            backtest_record(1, "A", 1, 1, 0, 2),
            backtest_record(1, "B", 2, 2, 0, 1),
            # "C" and "D" tie at 1/2 with no unexpected; fewer missed wins.
            backtest_record(1, "C", 1, 1),
            backtest_record(1, "D", 2, 2),
            # "E" and "F" tie fully; Unicode code point order decides.
            backtest_record(1, "F", 1, 1),
            backtest_record(1, "E", 1, 1),
            # "G" has the best rate; "H" evaluates nothing and sorts last.
            backtest_record(1, "G", 3, 1),
            backtest_record(1, "H", 0, 0, 4, 0),
        )
        result = build_scorecards(records)
        self.assertEqual(
            [scorecard[IDENTITY] for scorecard in result["scorecards"]],
            ["G", "C", "E", "F", "D", "B", "A", "H"],
        )

    def test_scorecard_limit_bounds_the_complete_set(self):
        records = (
            backtest_record(1, "A", 1, 0),
            backtest_record(1, "B", 1, 0),
            backtest_record(1, "C", 1, 0),
        )
        with self.assertRaises(ValueError) as caught:
            build_scorecards(records, limit=2)
        self.assertIn("scorecard limit exceeded", str(caught.exception))
        # Exactly at the limit passes.
        result = build_scorecards(records, limit=3)
        self.assertEqual(len(result["scorecards"]), 3)

    def test_empty_records_return_empty_tuple_and_zero_totals(self):
        result = build_scorecards(())
        self.assertEqual(result["scorecards"], ())
        self.assertEqual(result["totals"], (0, 0, 0, 0, 0, 0, 0))

    def test_every_object_is_fresh(self):
        records = (
            backtest_record(1, "A", 1, 0),
            backtest_record(2, "A", 0, 1),
        )
        result = build_scorecards(records)
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
        again = build_scorecards(records)
        self.assertEqual(
            again,
            build_scorecards(
                (
                    backtest_record(1, "A", 1, 0),
                    backtest_record(2, "A", 0, 1),
                )
            ),
        )


class ChainTests(unittest.TestCase):
    """The scorecards builder consumes exactly the backtests records."""

    def test_public_query_matches_the_backtests_query(self):
        store, args = diamond_store()
        backtests = backtests_call(store, args)
        result = scorecards_call(store, args)
        self.assertEqual(
            result,
            build_scorecards(backtests["backtests"], 0, 1, 50),
        )

    def test_filters_apply_to_the_same_records(self):
        store, args = diamond_store()
        backtests = backtests_call(store, args)
        result = scorecards_call(
            store,
            args,
            **{MIN_EVALUATED: 1, MIN_CUTOFFS: 1, SCORECARD_LIMIT: 50},
        )
        self.assertEqual(
            result,
            build_scorecards(backtests["backtests"], 1, 1, 50),
        )


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        return scorecards_call(self.store, self.args, **overrides)

    def test_min_evaluated_must_be_a_non_bool_non_negative_int(self):
        for bad in (True, False, 1.0, "1", None, (1,)):
            with self.subTest(value=bad):
                with self.assertRaises(TypeError):
                    self.call(**{MIN_EVALUATED: bad})
        with self.assertRaises(ValueError):
            self.call(**{MIN_EVALUATED: -1})

    def test_min_cutoffs_must_be_a_non_bool_positive_int(self):
        for bad in (True, False, 1.0, "1", None, (1,)):
            with self.subTest(value=bad):
                with self.assertRaises(TypeError):
                    self.call(**{MIN_CUTOFFS: bad})
        with self.assertRaises(ValueError):
            self.call(**{MIN_CUTOFFS: 0})
        with self.assertRaises(ValueError):
            self.call(**{MIN_CUTOFFS: -2})

    def test_scorecard_limit_must_be_a_non_bool_positive_int(self):
        for bad in (True, False, 1.0, "1", None, (1,)):
            with self.subTest(value=bad):
                with self.assertRaises(TypeError):
                    self.call(**{SCORECARD_LIMIT: bad})
        with self.assertRaises(ValueError):
            self.call(**{SCORECARD_LIMIT: 0})
        with self.assertRaises(ValueError):
            self.call(**{SCORECARD_LIMIT: -3})

    def test_new_parameters_validated_in_signature_order(self):
        # The backtest limit fails before the new minimum evaluated.
        with self.assertRaises(TypeError):
            self.call(
                regression_wave_regression_wave_regression_wave_forecast_backtest_limit="x",
                **{MIN_EVALUATED: "x"},
            )
        with self.assertRaises(ValueError):
            self.call(
                regression_wave_regression_wave_regression_wave_forecast_backtest_limit=0,
                **{MIN_EVALUATED: -1},
            )
        # The minimum evaluated fails before the minimum cutoffs.
        with self.assertRaises(TypeError):
            self.call(**{MIN_EVALUATED: "x", MIN_CUTOFFS: "x"})
        with self.assertRaises(ValueError):
            self.call(**{MIN_EVALUATED: -1, MIN_CUTOFFS: 0})
        # The minimum cutoffs fails before the scorecard limit.
        with self.assertRaises(TypeError):
            self.call(**{MIN_CUTOFFS: "x", SCORECARD_LIMIT: "x"})
        with self.assertRaises(ValueError):
            self.call(**{MIN_CUTOFFS: 0, SCORECARD_LIMIT: 0})
        # All pass before the split membership check fires.
        with self.assertRaises(ValueError) as caught:
            self.call(
                split_cutoffs=(99,),
                **{MIN_EVALUATED: 0, MIN_CUTOFFS: 1, SCORECARD_LIMIT: 1},
            )
        self.assertIn("split_cutoff 99", str(caught.exception))

    def test_all_parameter_checks_precede_state_lookups(self):
        with self.assertRaises(TypeError):
            self.call(reference="ghost", **{MIN_EVALUATED: "x"})
        with self.assertRaises(ValueError):
            self.call(reference="ghost", **{MIN_EVALUATED: -1})
        with self.assertRaises(TypeError):
            self.call(reference="ghost", **{MIN_CUTOFFS: "x"})
        with self.assertRaises(ValueError):
            self.call(reference="ghost", **{MIN_CUTOFFS: 0})
        with self.assertRaises(TypeError):
            self.call(reference="ghost", **{SCORECARD_LIMIT: "x"})
        with self.assertRaises(ValueError):
            self.call(reference="ghost", **{SCORECARD_LIMIT: 0})

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

    def test_empty_deepest_cutoffs_returns_empty_scorecards(self):
        result = scorecards_call(
            self.store,
            self.args,
            regression_wave_regression_wave_regression_wave_cutoffs=(),
        )
        self.assertEqual(result["scorecards"], ())
        self.assertEqual(result["totals"], (0, 0, 0, 0, 0, 0, 0))

    def test_no_qualifying_identity_returns_empty_scorecards(self):
        result = scorecards_call(
            self.store,
            self.args,
            **{MIN_EVALUATED: 100},
        )
        self.assertEqual(result["scorecards"], ())
        self.assertEqual(result["totals"], (0, 0, 0, 0, 0, 0, 0))


class IsolationTests(unittest.TestCase):
    def test_success_and_failure_change_nothing(self):
        store, args = diamond_store()
        heads = dict(store._heads)
        appends = copy.deepcopy(store._appends)
        merges = copy.deepcopy(store._merges)
        audit = store.audit_log()

        scorecards_call(store, args)
        failures = [
            {MIN_EVALUATED: "x"},
            {MIN_EVALUATED: True},
            {MIN_EVALUATED: -1},
            {MIN_CUTOFFS: "x"},
            {MIN_CUTOFFS: False},
            {MIN_CUTOFFS: 0},
            {SCORECARD_LIMIT: "x"},
            {SCORECARD_LIMIT: False},
            {SCORECARD_LIMIT: 0},
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
                    scorecards_call(store, args, **overrides)

        self.assertEqual(dict(store._heads), heads)
        self.assertEqual(copy.deepcopy(store._appends), appends)
        self.assertEqual(copy.deepcopy(store._merges), merges)
        self.assertEqual(store.audit_log(), audit)


class TokenTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, token=None, **overrides):
        return scorecards_call(self.store, self.args, token=token, **overrides)

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
            self.call(token=token, **{MIN_EVALUATED: "x"})
        with self.assertRaises(ValueError):
            self.call(token=token, **{MIN_CUTOFFS: 0})
        with self.assertRaises(ValueError):
            self.call(token=token, **{SCORECARD_LIMIT: 0})
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
