import copy
import inspect
import unittest

from city_twin.branches import BranchStore

from tests.test_lifetime_churn import diamond_store
from tests.test_lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_wave_forecast_scorecard_regression_wave_forecast_backtests_scorecards_regression_wave_forecast_backtests_scorecards_regressions_wave_recurrences import (
    DEEP_CUTOFFS,
    MIN_OCCURRENCES,
    RECURRENCE_LIMIT,
    RECURRENCES_QUERY,
    WAVE_LIMIT,
    build_recurrences,
    make_wave,
    recurrences_kwargs,
)

PERIODICITIES_QUERY = "lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_wave_forecast_scorecard_regression_wave_forecast_backtests_scorecards_regression_wave_forecast_backtests_scorecards_regressions_wave_periodicities"

P5 = "regression_wave_" * 5
MAX_JITTER = P5 + "max_regression_wave_jitter"
PERIODICITY_LIMIT = P5 + "regression_wave_periodicity_limit"

ZERO_TOTALS = {
    "periodicities": 0,
    "waves": 0,
    "points": 0,
    "first_wave": 0,
    "last_wave": 0,
    "mean_period": 0,
}


def periodicities_kwargs(store_args, **overrides) -> dict:
    max_jitter = overrides.pop(MAX_JITTER, 10)
    periodicity_limit = overrides.pop(PERIODICITY_LIMIT, 50)
    kwargs = recurrences_kwargs(store_args, **overrides)
    kwargs[MAX_JITTER] = max_jitter
    kwargs[PERIODICITY_LIMIT] = periodicity_limit
    return kwargs


def periodicities_call(store, store_args, **overrides):
    return getattr(store, PERIODICITIES_QUERY)(
        **periodicities_kwargs(store_args, **overrides)
    )


def build_periodicities(
    recurrences,
    max_regression_wave_jitter=10,
    periodicity_limit=50,
):
    return BranchStore._build_lifetime_churn_forecast_drift_wave_prediction_scorecard_regression_wave_forecast_scorecard_regression_wave_forecast_backtests_scorecards_regression_wave_forecast_backtests_scorecards_regressions_wave_periodicities_result(
        recurrences,
        max_regression_wave_jitter,
        periodicity_limit,
    )


class SignatureTests(unittest.TestCase):
    def test_public_signature_appends_two_parameters(self):
        parameters = list(
            inspect.signature(
                getattr(BranchStore, PERIODICITIES_QUERY)
            ).parameters.values()
        )
        names = [p.name for p in parameters]
        recurrence_names = list(
            inspect.signature(
                getattr(BranchStore, RECURRENCES_QUERY)
            ).parameters
        )
        self.assertEqual(
            names,
            recurrence_names[:-1]
            + [MAX_JITTER, PERIODICITY_LIMIT, "token"],
        )
        self.assertEqual(
            [p.default for p in parameters[:-1]],
            [inspect.Parameter.empty] * (len(parameters) - 1),
        )
        self.assertIsNone(parameters[-1].default)

    def test_result_shape_and_key_order(self):
        waves = (
            make_wave(1, [(1, 1, ("A",))]),
            make_wave(2, [(2, 1, ("A",))]),
            make_wave(3, [(3, 1, ("A",))]),
        )
        recurrences = build_recurrences(waves)
        result = build_periodicities(recurrences["recurrences"])
        self.assertEqual(list(result), ["periodicities", "totals"])
        self.assertIsInstance(result["periodicities"], tuple)
        record = result["periodicities"][0]
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
        self.assertEqual(
            list(record["appearances"][0]),
            [
                "wave",
                "start_reference_cutoffs",
                "end_reference_cutoffs",
                "points",
                "first_reference_cutoffs",
                "last_reference_cutoffs",
            ],
        )
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


class PeriodicitiesResultTests(unittest.TestCase):
    def test_steady_cadence_record(self):
        waves = (
            make_wave(1, [(1, 2, ("A",))]),
            make_wave(2, [(2, 1, ("A",))]),
            make_wave(3, [(3, 3, ("A",))]),
        )
        recurrences = build_recurrences(waves)
        result = build_periodicities(recurrences["recurrences"])
        self.assertEqual(len(result["periodicities"]), 1)
        record = result["periodicities"][0]
        self.assertEqual(record["identity"], "A")
        self.assertEqual(record["waves"], 3)
        self.assertEqual(record["first_wave"], 0)
        self.assertEqual(record["last_wave"], 2)
        self.assertEqual(record["span"], 3)
        self.assertEqual(record["points"], 3)
        self.assertEqual(record["peak_active"], 3)
        self.assertEqual(record["intervals"], (1, 1))
        self.assertEqual(record["period"], 1)
        self.assertEqual(record["jitter"], 0)
        self.assertEqual(
            result["totals"],
            {
                "periodicities": 1,
                "waves": 3,
                "points": 3,
                "first_wave": 0,
                "last_wave": 2,
                "mean_period": 1,
            },
        )

    def test_two_appearances_never_qualify(self):
        waves = (
            make_wave(1, [(1, 1, ("A",))]),
            make_wave(2, [(2, 1, ("A",))]),
        )
        recurrences = build_recurrences(waves)
        self.assertEqual(len(recurrences["recurrences"]), 1)
        result = build_periodicities(recurrences["recurrences"])
        self.assertEqual(result["periodicities"], ())
        self.assertEqual(result["totals"], ZERO_TOTALS)

    def test_even_interval_count_takes_the_smaller_middle_period(self):
        waves = (
            make_wave(1, [(1, 1, ("A",))]),
            make_wave(2, [(2, 1, ("B",))]),
            make_wave(3, [(3, 1, ("A",))]),
            make_wave(4, [(4, 1, ("B",))]),
            make_wave(5, [(5, 1, ("C",))]),
            make_wave(6, [(6, 1, ("C",))]),
            make_wave(7, [(7, 1, ("A",))]),
        )
        # "A" takes part in waves 0, 2 and 6; "B" and "C" never reach
        # three appearances.
        recurrences = build_recurrences(waves)
        result = build_periodicities(recurrences["recurrences"])
        self.assertEqual(len(result["periodicities"]), 1)
        record = result["periodicities"][0]
        self.assertEqual(record["intervals"], (2, 4))
        self.assertEqual(record["period"], 2)
        self.assertEqual(record["jitter"], 2)

    def test_odd_interval_count_takes_the_middle_period(self):
        waves = (
            make_wave(1, [(1, 1, ("A",))]),
            make_wave(2, [(2, 1, ("A",))]),
            make_wave(3, [(3, 1, ("A",))]),
            make_wave(4, [(4, 1, ("B",))]),
            make_wave(5, [(5, 1, ("A",))]),
        )
        # "A" takes part in waves 0, 1, 2 and 4.
        recurrences = build_recurrences(waves)
        result = build_periodicities(recurrences["recurrences"])
        record = result["periodicities"][0]
        self.assertEqual(record["intervals"], (1, 1, 2))
        self.assertEqual(record["period"], 1)
        self.assertEqual(record["jitter"], 1)

    def test_max_regression_wave_jitter_filters_records(self):
        waves = (
            make_wave(1, [(1, 1, ("A", "B"))]),
            make_wave(2, [(2, 1, ("A", "B"))]),
            make_wave(3, [(3, 1, ("A",))]),
            make_wave(4, [(4, 1, ("B",))]),
        )
        # "A" has intervals (1, 1) and jitter 0; "B" has intervals
        # (1, 2) and jitter 1.
        recurrences = build_recurrences(waves)
        result = build_periodicities(
            recurrences["recurrences"],
            max_regression_wave_jitter=0,
        )
        self.assertEqual(
            [record["identity"] for record in result["periodicities"]],
            ["A"],
        )
        result = build_periodicities(
            recurrences["recurrences"],
            max_regression_wave_jitter=1,
        )
        self.assertEqual(
            [record["identity"] for record in result["periodicities"]],
            ["A", "B"],
        )

    def test_records_sort_by_jitter_waves_points_first_wave_then_encounter(
        self,
    ):
        waves = (
            make_wave(1, [(1, 1, ("c", "b", "a", "d"))]),
            make_wave(2, [(2, 1, ("a", "c", "d")), (3, 1, ("a",))]),
            make_wave(3, [(3, 1, ("a", "b", "d"))]),
            make_wave(4, [(4, 1, ("c", "d"))]),
            make_wave(5, [(5, 1, ("b",))]),
        )
        # "a": waves 0-2, intervals (1, 1), jitter 0, four points.
        # "d": waves 0-3, intervals (1, 1, 1), jitter 0, four points.
        # "b": waves 0, 2, 4, intervals (2, 2), jitter 0, three points.
        # "c": waves 0, 1, 3, intervals (1, 2), jitter 1.
        recurrences = build_recurrences(waves)
        result = build_periodicities(recurrences["recurrences"])
        # Jitter 0 first; within it waves descending ("d" has four),
        # then points descending ("a" has four, "b" three). A full tie
        # would fall back to first-encounter order across the waves.
        self.assertEqual(
            [record["identity"] for record in result["periodicities"]],
            ["d", "a", "b", "c"],
        )
        # A full tie falls back to first-encounter order.
        waves = (
            make_wave(1, [(1, 1, ("c", "a"))]),
            make_wave(2, [(2, 1, ("c", "a"))]),
            make_wave(3, [(3, 1, ("c", "a"))]),
        )
        recurrences = build_recurrences(waves)
        result = build_periodicities(recurrences["recurrences"])
        self.assertEqual(
            [record["identity"] for record in result["periodicities"]],
            ["c", "a"],
        )

    def test_periodicity_limit_bounds_the_complete_set(self):
        waves = (
            make_wave(1, [(1, 1, ("A", "B"))]),
            make_wave(2, [(2, 1, ("A", "B"))]),
            make_wave(3, [(3, 1, ("A", "B"))]),
        )
        recurrences = build_recurrences(waves)
        with self.assertRaises(ValueError) as caught:
            build_periodicities(recurrences["recurrences"], 10, 1)
        self.assertIn("periodicity limit", str(caught.exception))
        result = build_periodicities(recurrences["recurrences"], 10, 2)
        self.assertEqual(len(result["periodicities"]), 2)

    def test_mean_period_is_the_floored_average(self):
        waves = (
            make_wave(1, [(1, 1, ("A", "B"))]),
            make_wave(2, [(2, 1, ("A",))]),
            make_wave(3, [(3, 1, ("A", "B"))]),
            make_wave(4, [(4, 1, ("C",))]),
            make_wave(5, [(5, 1, ("B",))]),
        )
        # "A" takes part in waves 0-2 with period 1; "B" takes part in
        # waves 0, 2 and 4 with intervals (2, 2) and period 2, so the
        # floored mean of (1, 2) is 1.
        recurrences = build_recurrences(waves)
        result = build_periodicities(recurrences["recurrences"])
        self.assertEqual(result["totals"]["mean_period"], 1)

    def test_empty_recurrences_return_empty_tuple_and_zero_totals(self):
        result = build_periodicities(())
        self.assertEqual(result["periodicities"], ())
        self.assertEqual(result["totals"], ZERO_TOTALS)

    def test_every_object_is_fresh(self):
        waves = (
            make_wave(1, [(1, 1, (("A", "x"),))]),
            make_wave(2, [(2, 1, (("A", "x"),))]),
            make_wave(3, [(3, 1, (("A", "x"),))]),
        )
        recurrences = build_recurrences(waves)
        result = build_periodicities(recurrences["recurrences"])
        result["periodicities"][0]["waves"] = 999
        result["periodicities"][0]["appearances"][0]["points"] = 999
        result["totals"]["periodicities"] = 999
        again = build_periodicities(recurrences["recurrences"])
        self.assertEqual(again["periodicities"][0]["waves"], 3)
        self.assertEqual(
            again["periodicities"][0]["appearances"][0]["points"], 1
        )
        self.assertEqual(again["totals"]["periodicities"], 1)
        # The recurrence records are not mutated either.
        self.assertEqual(recurrences["recurrences"][0]["waves"], 3)

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
    """The periodicity builder consumes exactly what recurrences keep."""

    def test_periodicities_match_the_recurrences_result(self):
        waves = (
            make_wave(1, [(1, 2, ("A",))]),
            make_wave(2, [(2, 1, ("A",))]),
            make_wave(3, [(3, 3, ("A",))]),
        )
        recurrences = build_recurrences(waves)
        self.assertEqual(len(recurrences["recurrences"]), 1)
        result = build_periodicities(recurrences["recurrences"])
        self.assertEqual(len(result["periodicities"]), 1)
        record = result["periodicities"][0]
        self.assertEqual(record["identity"], "A")
        self.assertEqual(record["waves"], 3)
        self.assertEqual(record["first_wave"], 0)
        self.assertEqual(record["last_wave"], 2)
        self.assertEqual(record["span"], 3)
        self.assertEqual(record["intervals"], (1, 1))
        self.assertEqual(record["period"], 1)
        self.assertEqual(record["jitter"], 0)

    def test_public_query_matches_the_recurrences_query(self):
        store, args = diamond_store()
        kwargs = periodicities_kwargs(args)
        recurrence_kwargs = dict(kwargs)
        del recurrence_kwargs[MAX_JITTER]
        del recurrence_kwargs[PERIODICITY_LIMIT]
        recurrences = getattr(store, RECURRENCES_QUERY)(**recurrence_kwargs)
        result = getattr(store, PERIODICITIES_QUERY)(**kwargs)
        self.assertEqual(
            result,
            build_periodicities(
                recurrences["recurrences"],
                kwargs[MAX_JITTER],
                kwargs[PERIODICITY_LIMIT],
            ),
        )


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        return periodicities_call(self.store, self.args, **overrides)

    def test_max_regression_wave_jitter_must_be_a_non_bool_non_negative_int(
        self,
    ):
        for bad in (True, False, 1.0, "1", None, (1,)):
            with self.subTest(value=bad):
                with self.assertRaises(TypeError):
                    self.call(**{MAX_JITTER: bad})
        with self.assertRaises(ValueError):
            self.call(**{MAX_JITTER: -1})
        with self.assertRaises(ValueError):
            self.call(**{MAX_JITTER: -3})
        # Zero is allowed.
        self.call(**{MAX_JITTER: 0})

    def test_periodicity_limit_must_be_a_non_bool_positive_int(self):
        for bad in (True, False, 1.0, "1", None, (1,)):
            with self.subTest(value=bad):
                with self.assertRaises(TypeError):
                    self.call(**{PERIODICITY_LIMIT: bad})
        with self.assertRaises(ValueError):
            self.call(**{PERIODICITY_LIMIT: 0})
        with self.assertRaises(ValueError):
            self.call(**{PERIODICITY_LIMIT: -3})

    def test_new_parameters_validated_in_signature_order(self):
        # The deepest recurrence limit fails before the deepest max
        # jitter.
        with self.assertRaises(TypeError):
            self.call(**{RECURRENCE_LIMIT: "x", MAX_JITTER: "x"})
        with self.assertRaises(ValueError):
            self.call(**{RECURRENCE_LIMIT: 0, MAX_JITTER: -1})
        # The deepest max jitter fails before the deepest periodicity
        # limit.
        with self.assertRaises(TypeError):
            self.call(**{MAX_JITTER: "x", PERIODICITY_LIMIT: "x"})
        with self.assertRaises(ValueError):
            self.call(**{MAX_JITTER: -1, PERIODICITY_LIMIT: 0})
        # Both pass before the split membership check fires.
        with self.assertRaises(ValueError):
            self.call(
                split_cutoffs=(99,),
                **{MAX_JITTER: 0, PERIODICITY_LIMIT: 1},
            )

    def test_all_parameter_checks_precede_state_lookups(self):
        with self.assertRaises(TypeError):
            self.call(reference="ghost", **{MAX_JITTER: "x"})
        with self.assertRaises(ValueError):
            self.call(reference="ghost", **{MAX_JITTER: -1})
        with self.assertRaises(TypeError):
            self.call(reference="ghost", **{PERIODICITY_LIMIT: "x"})
        with self.assertRaises(ValueError):
            self.call(reference="ghost", **{PERIODICITY_LIMIT: 0})

    def test_underlying_stage_limits_and_errors_are_unchanged(self):
        with self.assertRaises(ValueError) as caught:
            self.call(wave_cutoffs=(9,))
        self.assertIn("out of range", str(caught.exception))


class EmptyRunTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def test_empty_deep_cutoffs_returns_empty_periodicities(self):
        result = periodicities_call(self.store, self.args, **{DEEP_CUTOFFS: ()})
        self.assertEqual(result["periodicities"], ())
        self.assertEqual(result["totals"], ZERO_TOTALS)

    def test_no_qualifying_identity_returns_empty_periodicities(self):
        # The diamond store yields no deepest regression waves, so no
        # identity can recur across three of them.
        result = periodicities_call(self.store, self.args)
        self.assertEqual(result["periodicities"], ())
        self.assertEqual(result["totals"], ZERO_TOTALS)


class IsolationTests(unittest.TestCase):
    def test_success_and_failure_change_nothing(self):
        store, args = diamond_store()
        heads = dict(store._heads)
        appends = copy.deepcopy(store._appends)
        merges = copy.deepcopy(store._merges)
        audit = store.audit_log()

        periodicities_call(store, args)
        failures = [
            {DEEP_CUTOFFS: "x"},
            {DEEP_CUTOFFS: (True,)},
            {DEEP_CUTOFFS: (-1,)},
            {DEEP_CUTOFFS: (0, 0)},
            {DEEP_CUTOFFS: (1, 0)},
            {MIN_OCCURRENCES: "x"},
            {MIN_OCCURRENCES: False},
            {MIN_OCCURRENCES: 0},
            {RECURRENCE_LIMIT: "x"},
            {RECURRENCE_LIMIT: False},
            {RECURRENCE_LIMIT: 0},
            {WAVE_LIMIT: "x"},
            {WAVE_LIMIT: False},
            {WAVE_LIMIT: 0},
            {MAX_JITTER: "x"},
            {MAX_JITTER: False},
            {MAX_JITTER: -1},
            {PERIODICITY_LIMIT: "x"},
            {PERIODICITY_LIMIT: False},
            {PERIODICITY_LIMIT: 0},
            dict(drift_limit=2),
            dict(scan_limit=1),
            dict(reference="ghost"),
        ]
        for overrides in failures:
            with self.subTest(overrides=overrides):
                with self.assertRaises(Exception):
                    periodicities_call(store, args, **overrides)

        self.assertEqual(dict(store._heads), heads)
        self.assertEqual(copy.deepcopy(store._appends), appends)
        self.assertEqual(copy.deepcopy(store._merges), merges)
        self.assertEqual(store.audit_log(), audit)


class TokenTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, token=None, **overrides):
        return periodicities_call(self.store, self.args, token=token, **overrides)

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
            self.call(token=token, **{MAX_JITTER: "x"})
        with self.assertRaises(ValueError):
            self.call(token=token, **{PERIODICITY_LIMIT: 0})
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
