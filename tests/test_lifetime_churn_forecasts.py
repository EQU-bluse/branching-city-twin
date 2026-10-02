import copy
import inspect
import unittest

from city_twin.branches import BranchStore

from tests.test_lifetime_churn import churn_kwargs, diamond_store
from tests.test_lifetime_churn_waves import reference_waves
from tests.test_lifetime_churn_periodicities import (
    periodicities_kwargs,
    reference_periodicities,
)


def forecasts_kwargs(store_args, **overrides) -> dict:
    horizon = overrides.pop("horizon", 10)
    forecast_limit = overrides.pop("forecast_limit", 50)
    kwargs = periodicities_kwargs(store_args, **overrides)
    kwargs["horizon"] = horizon
    kwargs["forecast_limit"] = forecast_limit
    return kwargs


def forecasts_call(store, store_args, **overrides):
    return store.lifetime_churn_forecasts(
        **forecasts_kwargs(store_args, **overrides)
    )


def reference_forecasts(
    evolution: dict[str, object],
    min_identities: int,
    max_jitter: int,
    horizon: int,
) -> dict[str, object]:
    """Independently project periodicities of an evolution result."""
    waves = reference_waves(evolution, min_identities)["waves"]
    periodicities = reference_periodicities(
        evolution, min_identities, max_jitter
    )["periodicities"]
    boundary = len(waves) - 1
    edge = boundary + horizon

    records = []
    predictions = 0
    total_waves = 0
    total_segments = 0
    total_added = 0
    total_removed = 0
    total_changed = 0
    for periodicity in periodicities:
        last = periodicity["last"]
        period = periodicity["period"]
        intervals = periodicity["intervals"]
        entries = []
        ordinal = 1
        while True:
            wave = last + ordinal * period
            if wave > edge:
                break
            if wave > boundary:
                entries.append(
                    {
                        "ordinal": ordinal,
                        "wave": wave,
                        "earliest": last + ordinal * min(intervals),
                        "latest": last + ordinal * max(intervals),
                    }
                )
            ordinal += 1
        if not entries:
            continue
        record = {key: periodicity[key] for key in periodicity}
        record["forecasts"] = tuple(entries)
        records.append(record)
        predictions += len(entries)
        total_waves += periodicity["waves"]
        total_segments += periodicity["segments"]
        total_added += periodicity["added"]
        total_removed += periodicity["removed"]
        total_changed += periodicity["changed"]
    return {
        "forecasts": tuple(records),
        "totals": {
            "identities": len(records),
            "predictions": predictions,
            "waves": total_waves,
            "segments": total_segments,
            "added": total_added,
            "removed": total_removed,
            "changed": total_changed,
        },
    }


class LifetimeChurnForecastsSignatureTests(unittest.TestCase):
    def test_public_signature_appends_horizon_and_forecast_limit(self):
        parameters = list(
            inspect.signature(
                BranchStore.lifetime_churn_forecasts
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
        result = forecasts_call(
            store,
            args,
            windows=((0,), (1,), (1,), (2,), (2,), (0,)),
            horizon=3,
        )
        self.assertIsInstance(result, dict)
        self.assertEqual(list(result), ["forecasts", "totals"])
        self.assertIsInstance(result["forecasts"], tuple)
        self.assertEqual(
            list(result["totals"]),
            [
                "identities",
                "predictions",
                "waves",
                "segments",
                "added",
                "removed",
                "changed",
            ],
        )
        for record in result["forecasts"]:
            self.assertIsInstance(record, dict)
            self.assertEqual(
                list(record),
                [
                    "type",
                    "identity",
                    "waves",
                    "first",
                    "last",
                    "span",
                    "segments",
                    "added",
                    "removed",
                    "changed",
                    "appearances",
                    "intervals",
                    "period",
                    "jitter",
                    "forecasts",
                ],
            )
            self.assertIsInstance(record["forecasts"], tuple)
            self.assertTrue(record["forecasts"])
            for forecast in record["forecasts"]:
                self.assertEqual(
                    list(forecast),
                    ["ordinal", "wave", "earliest", "latest"],
                )


class LifetimeChurnForecastsResultTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        return forecasts_call(self.store, self.args, **overrides)

    def evolution(self, **overrides):
        kwargs = churn_kwargs(self.args, **overrides)
        for name in (
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
        ):
            kwargs.pop(name, None)
        return self.store.lifetime_evolution(**kwargs)

    def test_matches_independent_projection_of_evolution(self):
        cases = (
            (((0,), (1,), (1,), (2,), (2,), (0,)), 1, 0, 1),
            (((0,), (1,), (1,), (2,), (2,), (0,)), 1, 0, 3),
            (((0,), (1,), (1,), (2,), (2,), (0,)), 1, 2, 5),
            (
                ((0,), (1,), (1,), (2,), (2,), (0,), (0,), (1,)),
                1,
                0,
                1,
            ),
            (
                ((0,), (1,), (1,), (2,), (2,), (0,), (0,), (1,)),
                1,
                2,
                4,
            ),
            (((0, 1), (0, 1), (0, 1)), 1, 1, 2),
            (((2,), (0,)), 1, 1, 3),
            ((), 1, 1, 3),
        )
        for windows, min_identities, max_jitter, horizon in cases:
            with self.subTest(
                windows=windows,
                min_identities=min_identities,
                max_jitter=max_jitter,
                horizon=horizon,
            ):
                self.assertEqual(
                    self.call(
                        windows=windows,
                        min_identities=min_identities,
                        max_jitter=max_jitter,
                        horizon=horizon,
                    ),
                    reference_forecasts(
                        self.evolution(windows=windows),
                        min_identities,
                        max_jitter,
                        horizon,
                    ),
                )

    def test_diamond_fixture_projects_three_cadenced_identities(self):
        result = self.call(
            windows=((0,), (1,), (1,), (2,), (2,), (0,)),
            max_jitter=0,
            horizon=3,
        )
        self.assertEqual(len(result["forecasts"]), 3)
        for record in result["forecasts"]:
            self.assertEqual(record["last"], 2)
            self.assertEqual(record["period"], 1)
            self.assertEqual(
                [(f["ordinal"], f["wave"]) for f in record["forecasts"]],
                [(1, 3), (2, 4), (3, 5)],
            )
            for forecast in record["forecasts"]:
                self.assertEqual(forecast["earliest"], forecast["wave"])
                self.assertEqual(forecast["latest"], forecast["wave"])
        self.assertEqual(
            result["totals"],
            {
                "identities": 3,
                "predictions": 9,
                "waves": 9,
                "segments": 9,
                "added": 2,
                "removed": 2,
                "changed": 5,
            },
        )

    def test_horizon_is_inclusive_and_forecasts_stay_beyond_boundary(self):
        # Last observed wave serial is 2; horizon one reaches exactly
        # wave 3 and keeps it.
        windows = ((0,), (1,), (1,), (2,), (2,), (0,))
        one = self.call(windows=windows, horizon=1)
        self.assertEqual(
            [f["wave"] for f in one["forecasts"][0]["forecasts"]],
            [3],
        )
        self.assertEqual(one["totals"]["predictions"], 3)

    def test_horizon_shorter_than_period_selects_nothing(self):
        # The steady identities have period one; projecting with the
        # builder against a zero-width horizon still yields seven zeros,
        # proving no observed wave is ever re-issued as a forecast.
        evolution = self.evolution(
            windows=((0,), (1,), (1,), (2,), (2,), (0,))
        )
        periodicities = reference_periodicities(evolution, 1, 0)[
            "periodicities"
        ]
        waves = reference_waves(evolution, 1)["waves"]
        result = BranchStore._build_lifetime_churn_forecasts_result(
            waves, periodicities, 0, 50
        )
        self.assertEqual(result["forecasts"], ())
        self.assertEqual(
            result["totals"],
            {
                "identities": 0,
                "predictions": 0,
                "waves": 0,
                "segments": 0,
                "added": 0,
                "removed": 0,
                "changed": 0,
            },
        )

    def test_records_keep_periodicity_order(self):
        result = self.call(
            windows=(
                (0,), (1,), (1,), (2,), (2,), (0,), (0,), (1,),
            ),
            max_jitter=2,
            horizon=4,
        )
        jitters = [r["jitter"] for r in result["forecasts"]]
        self.assertEqual(jitters, sorted(jitters))
        for record in result["forecasts"]:
            ordinals = [f["ordinal"] for f in record["forecasts"]]
            self.assertEqual(ordinals, sorted(ordinals))

    def test_empty_windows_causes_and_single_window_return_seven_zeros(self):
        for overrides in (
            {"windows": ()},
            {"causes": ()},
            {"windows": ((0, 1, 2),)},
        ):
            with self.subTest(overrides=overrides):
                result = self.call(**overrides)
                self.assertEqual(result["forecasts"], ())
                self.assertEqual(
                    result["totals"],
                    {
                        "identities": 0,
                        "predictions": 0,
                        "waves": 0,
                        "segments": 0,
                        "added": 0,
                        "removed": 0,
                        "changed": 0,
                    },
                )


class LifetimeChurnForecastsGroupingTests(unittest.TestCase):
    @staticmethod
    def _member(kind, identity, segments=1, added=0, removed=0, changed=0):
        return {
            "type": kind,
            "identity": identity,
            "segments": segments,
            "added": added,
            "removed": removed,
            "changed": changed,
        }

    def _wave(self, serial, members):
        return {
            "from": serial * 2,
            "to": serial * 2 + 1,
            "segments": 1,
            "identities": len(members),
            "added": sum(m["added"] for m in members),
            "removed": sum(m["removed"] for m in members),
            "changed": sum(m["changed"] for m in members),
            "members": tuple(members),
            "changes": (),
        }

    def _periodicities(self, waves, max_jitter=0):
        return BranchStore._build_lifetime_churn_periodicities_result(
            waves, max_jitter, None
        )["periodicities"]

    def test_ordinal_skips_candidates_still_inside_observed_history(self):
        # A churns at waves 0, 1, 2; later observed waves 3 and 4 carry
        # one-off fillers. boundary is 4, last is 2: ordinals 1 and 2
        # land on observed waves and are skipped, ordinal 3 is the first
        # forecast even though its ordinal is still three.
        waves = tuple(
            self._wave(
                serial,
                (self._member("node", "A"),)
                if serial < 3
                else (self._member("node", f"Z{serial}"),),
            )
            for serial in range(5)
        )
        periodicities = self._periodicities(waves)
        result = BranchStore._build_lifetime_churn_forecasts_result(
            waves, periodicities, 3, 100
        )
        record = result["forecasts"][0]
        self.assertEqual(record["identity"], "A")
        self.assertEqual(record["last"], 2)
        self.assertEqual(
            [(f["ordinal"], f["wave"]) for f in record["forecasts"]],
            [(3, 5), (4, 6), (5, 7)],
        )
        self.assertEqual(result["totals"]["predictions"], 3)

    def test_earliest_latest_use_observed_interval_extremes(self):
        # A at waves 0, 2, 3 -> intervals (2, 1), period 1, jitter 1.
        # boundary 3; horizon two keeps waves 4 and 5 only.
        filler = self._member("node", "Z")
        waves = (
            self._wave(0, (self._member("node", "A"),)),
            self._wave(1, (filler,)),
            self._wave(2, (self._member("node", "A"),)),
            self._wave(3, (self._member("node", "A"),)),
        )
        periodicities = self._periodicities(waves, max_jitter=1)
        result = BranchStore._build_lifetime_churn_forecasts_result(
            waves, periodicities, 2, 100
        )
        forecasts = result["forecasts"][0]["forecasts"]
        self.assertEqual(
            [
                (f["ordinal"], f["wave"], f["earliest"], f["latest"])
                for f in forecasts
            ],
            [
                (1, 4, 4, 5),
                (2, 5, 5, 7),
            ],
        )

    def test_identity_without_an_in_range_candidate_is_dropped(self):
        # Period three with a horizon of one from wave 6 reaches only
        # wave 7: the first candidate is wave 9, so A is not selected.
        a_serials = {0, 3, 6}
        waves = tuple(
            self._wave(
                serial,
                (self._member("node", "A"),)
                if serial in a_serials
                else (self._member("node", f"Z{serial}"),),
            )
            for serial in range(7)
        )
        periodicities = self._periodicities(waves)
        self.assertEqual(len(periodicities), 1)
        result = BranchStore._build_lifetime_churn_forecasts_result(
            waves, periodicities, 1, 100
        )
        self.assertEqual(result["forecasts"], ())
        self.assertEqual(
            result["totals"],
            {
                "identities": 0,
                "predictions": 0,
                "waves": 0,
                "segments": 0,
                "added": 0,
                "removed": 0,
                "changed": 0,
            },
        )

    def test_forecast_limit_counts_predictions_across_identities(self):
        waves = tuple(
            self._wave(
                serial,
                (
                    self._member("node", "A"),
                    self._member("node", "B"),
                    self._member("edge", "C"),
                ),
            )
            for serial in range(3)
        )
        periodicities = self._periodicities(waves)
        self.assertEqual(len(periodicities), 3)
        ok = BranchStore._build_lifetime_churn_forecasts_result(
            waves, periodicities, 2, 6
        )
        self.assertEqual(ok["totals"]["predictions"], 6)
        with self.assertRaises(ValueError) as caught:
            BranchStore._build_lifetime_churn_forecasts_result(
                waves, periodicities, 2, 5
            )
        self.assertIn("forecast limit", str(caught.exception))

    def test_empty_wave_and_periodicity_sets_yield_seven_zeros(self):
        result = BranchStore._build_lifetime_churn_forecasts_result(
            (), (), 3, 100
        )
        self.assertEqual(result["forecasts"], ())
        self.assertEqual(
            result["totals"],
            {
                "identities": 0,
                "predictions": 0,
                "waves": 0,
                "segments": 0,
                "added": 0,
                "removed": 0,
                "changed": 0,
            },
        )

    def test_forecast_records_share_no_input_object(self):
        waves = (
            self._wave(0, (self._member("node", "A", changed=1),)),
            self._wave(1, (self._member("node", "A"),)),
            self._wave(2, (self._member("node", "A"),)),
        )
        periodicities = self._periodicities(waves)
        source = periodicities[0]
        result = BranchStore._build_lifetime_churn_forecasts_result(
            waves, periodicities, 2, 100
        )
        record = result["forecasts"][0]
        self.assertIsNot(record, source)
        self.assertIsNot(record["appearances"], source["appearances"])
        for built, original in zip(
            record["appearances"], source["appearances"]
        ):
            self.assertIsNot(built, original)
        self.assertIsNot(record["intervals"], source["intervals"])
        self.assertEqual(record["intervals"], source["intervals"])


class LifetimeChurnForecastsLimitValidationTests(unittest.TestCase):
    EXTENDED = ((0,), (1,), (1,), (2,), (2,), (0,))

    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        overrides.setdefault("causes", ("a0",))
        return forecasts_call(self.store, self.args, **overrides)

    def extended(self, **overrides):
        overrides.setdefault("windows", self.EXTENDED)
        return self.call(**overrides)

    def test_horizon_must_be_a_non_bool_positive_int(self):
        for bad in (True, False, 1.0, "1", None):
            with self.subTest(horizon=bad):
                with self.assertRaises(TypeError):
                    self.call(horizon=bad)
        with self.assertRaises(ValueError):
            self.call(horizon=0)
        with self.assertRaises(ValueError):
            self.call(horizon=-3)
        # One is the boundary that must be accepted.
        self.extended(horizon=1)

    def test_forecast_limit_must_be_a_non_bool_positive_int(self):
        for bad in (True, False, 1.0, "1", None):
            with self.subTest(forecast_limit=bad):
                with self.assertRaises(TypeError):
                    self.call(forecast_limit=bad)
        with self.assertRaises(ValueError):
            self.call(forecast_limit=0)
        with self.assertRaises(ValueError):
            self.call(forecast_limit=-3)

    def test_new_limits_validated_in_order_after_periodicity_limit(self):
        # periodicity_limit fails before horizon.
        with self.assertRaises(ValueError):
            self.extended(periodicity_limit=0, horizon=0)
        with self.assertRaises(TypeError):
            self.extended(periodicity_limit="x", horizon=0)
        # horizon fails before forecast_limit.
        with self.assertRaises(ValueError):
            self.extended(horizon=0, forecast_limit=0)
        with self.assertRaises(TypeError):
            self.extended(horizon="x", forecast_limit=0)
        # Once horizon passes, forecast_limit's error fires.
        with self.assertRaises(TypeError):
            self.extended(horizon=1, forecast_limit="x")
        with self.assertRaises(ValueError):
            self.extended(horizon=1, forecast_limit=0)
        # Both still precede every state lookup and the token lookup.
        with self.assertRaises(ValueError):
            self.call(reference="ghost", horizon=0)
        with self.assertRaises(ValueError):
            self.call(token="whatever", forecast_limit=0)
        with self.assertRaises(TypeError):
            self.call(reference="ghost", horizon="x")

    def test_prediction_count_beyond_limit_raises_value_error(self):
        with self.assertRaises(ValueError) as caught:
            self.extended(horizon=3, forecast_limit=8)
        self.assertIn("forecast limit", str(caught.exception))
        result = self.extended(horizon=3, forecast_limit=9)
        self.assertEqual(result["totals"]["predictions"], 9)
        # An empty result never trips the cap.
        result = self.call(forecast_limit=1)
        self.assertEqual(result["forecasts"], ())

    def test_periodicity_limit_is_validated_but_not_enforced(self):
        with self.assertRaises(ValueError):
            self.call(periodicity_limit=0)
        with self.assertRaises(TypeError):
            self.call(periodicity_limit="x")
        # A cap of one would bound the periodicities query; forecasts
        # still returns every periodic identity.
        result = self.extended(horizon=1, periodicity_limit=1)
        self.assertEqual(len(result["forecasts"]), 3)

    def test_wave_limit_is_still_enforced_ahead_of_forecasts(self):
        with self.assertRaises(ValueError) as caught:
            self.extended(wave_limit=1, forecast_limit=10)
        self.assertIn("wave limit", str(caught.exception))

    def test_earlier_limits_validated_but_not_enforced(self):
        for name in (
            "churn_limit",
            "streak_limit",
            "min_waves",
            "recurrence_limit",
        ):
            with self.assertRaises(ValueError):
                self.call(**{name: 0})
            with self.assertRaises(TypeError):
                self.call(**{name: "x"})
        result = self.extended(
            churn_limit=1,
            streak_limit=1,
            min_waves=99,
            recurrence_limit=1,
            horizon=1,
        )
        self.assertEqual(len(result["forecasts"]), 3)

    def test_batch_caps_still_fire_ahead_of_forecast_cap(self):
        with self.assertRaises(ValueError) as caught:
            self.call(total_diff_limit=5, forecast_limit=1)
        self.assertIn("total diff", str(caught.exception))
        # Empty batches never trip the forecast cap.
        self.assertEqual(
            self.call(windows=(), forecast_limit=1)["forecasts"],
            (),
        )

    def test_shared_validation_errors_match_periodicities(self):
        with self.assertRaises(TypeError):
            self.call(windows="x")
        with self.assertRaises(ValueError):
            self.call(windows=((-1,),))
        with self.assertRaises(ValueError):
            self.call(direction="sideways")
        with self.assertRaises(ValueError):
            self.call(window_limit=2, windows=((0,), (1,), (2,)))


class LifetimeChurnForecastsStateTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        overrides.setdefault("causes", ("a0",))
        return forecasts_call(self.store, self.args, **overrides)

    def test_unknown_branch_history_node_and_cause_raise(self):
        with self.assertRaises(KeyError):
            self.call(reference="ghost")
        with self.assertRaises(KeyError):
            self.call(
                series={
                    "a": (("r0", "a0"), ("r1", "a0"), ("r2", "a0")),
                    "b": (("r0", "a0"), ("r1", "W"), ("r2", "nope2")),
                }
            )
        with self.assertRaises(KeyError) as caught:
            self.call(causes=("zzz",))
        self.assertEqual(caught.exception.args[0], "zzz")

    def test_empty_windows_still_run_every_state_check(self):
        with self.assertRaises(KeyError):
            self.call(windows=(), reference="ghost")
        with self.assertRaises(ValueError):
            self.call(windows=(), limit=1)


class LifetimeChurnForecastsIsolationTests(unittest.TestCase):
    EXTENDED = ((0,), (1,), (1,), (2,), (2,), (0,))

    def test_result_levels_are_mutually_unshared(self):
        store, args = diamond_store()
        result = forecasts_call(
            store, args, windows=self.EXTENDED, horizon=3
        )
        seen_ids = []

        def collect(value):
            if isinstance(value, dict):
                seen_ids.append(id(value))
                for item in value.values():
                    collect(item)
            elif isinstance(value, tuple):
                for item in value:
                    collect(item)

        collect(result)
        self.assertEqual(len(seen_ids), len(set(seen_ids)))

    def test_mutating_result_never_affects_later_calls(self):
        store, args = diamond_store()
        first = forecasts_call(
            store, args, windows=self.EXTENDED, horizon=3
        )
        pristine = copy.deepcopy(first)
        for record in first["forecasts"]:
            record["intervals"] += (99,)
            record["period"] = 99
            record["jitter"] = 99
            record["forecasts"] += (
                {"ordinal": 9, "wave": 99, "earliest": 99, "latest": 99},
            )
            record["waves"] = 99
            for appearance in record["appearances"]:
                appearance["segments"] = 99
        self.assertEqual(
            forecasts_call(
                store, args, windows=self.EXTENDED, horizon=3
            ),
            pristine,
        )

    def test_success_and_failure_change_nothing(self):
        store, args = diamond_store()
        heads = dict(store._heads)
        appends = copy.deepcopy(store._appends)
        merges = copy.deepcopy(store._merges)
        audit = store.audit_log()

        forecasts_call(store, args, windows=self.EXTENDED, horizon=3)
        failures = [
            dict(windows="x"),
            dict(windows=((-1,),)),
            dict(windows=((9,),)),
            dict(causes=[1]),
            dict(direction="up"),
            dict(node_limit=0),
            dict(change_limit=0),
            dict(lifetime_limit=0),
            dict(diff_limit=0),
            dict(window_limit=0),
            dict(total_diff_limit=0),
            dict(churn_limit=0),
            dict(streak_limit=0),
            dict(min_identities=0),
            dict(min_identities="x"),
            dict(wave_limit=0),
            dict(wave_limit="x"),
            dict(min_waves=0),
            dict(min_waves="x"),
            dict(recurrence_limit=0),
            dict(recurrence_limit="x"),
            dict(max_jitter=-1),
            dict(max_jitter="x"),
            dict(periodicity_limit=0),
            dict(periodicity_limit="x"),
            dict(horizon=0),
            dict(horizon="x"),
            dict(forecast_limit=0),
            dict(forecast_limit="x"),
            dict(wave_limit=1, windows=self.EXTENDED),
            dict(
                forecast_limit=1,
                windows=self.EXTENDED,
                horizon=3,
            ),
            dict(causes=("zzz",)),
            dict(limit=1),
            dict(reference="ghost"),
        ]
        for overrides in failures:
            with self.subTest(overrides=overrides):
                with self.assertRaises(Exception):
                    forecasts_call(
                        store, args, causes=("a0",), **overrides
                    )

        self.assertEqual(dict(store._heads), heads)
        self.assertEqual(copy.deepcopy(store._appends), appends)
        self.assertEqual(copy.deepcopy(store._merges), merges)
        self.assertEqual(store.audit_log(), audit)

    def test_existing_public_entries_are_unaffected(self):
        store, args = diamond_store()
        forecasts_call(store, args, windows=self.EXTENDED, horizon=3)
        from tests.test_lifetime_churn_periodicities import (
            periodicities_call,
        )

        periodicities = periodicities_call(
            store, args, windows=self.EXTENDED
        )
        self.assertEqual(
            periodicities_call(store, args, windows=self.EXTENDED),
            periodicities,
        )


class LifetimeChurnForecastsTokenTests(unittest.TestCase):
    EXTENDED = ((0,), (1,), (1,), (2,), (2,), (0,))

    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, token=None, **overrides):
        return forecasts_call(
            self.store, self.args, token=token, **overrides
        )

    def test_success_consumes_one_read_and_failure_refunds(self):
        token = self.store.create_snapshot(2)
        self.call(token=token, windows=self.EXTENDED, horizon=3)
        # Over the forecast cap: failure refunds.
        with self.assertRaises(ValueError):
            self.call(
                token=token,
                windows=self.EXTENDED,
                horizon=3,
                forecast_limit=1,
            )
        # Over the wave cap inside the shared batch: failure refunds.
        with self.assertRaises(ValueError):
            self.call(
                token=token,
                windows=self.EXTENDED,
                wave_limit=1,
            )
        with self.assertRaises(KeyError):
            self.call(token=token, causes=("zzz",))
        # Parameter validation runs before the token is touched.
        with self.assertRaises(ValueError):
            self.call(token=token, forecast_limit=0)
        with self.assertRaises(TypeError):
            self.call(token=token, horizon="x")
        self.call(token=token)
        with self.assertRaises(RuntimeError):
            self.call(token=token)

    def test_results_come_from_the_frozen_view(self):
        before = copy.deepcopy(self.call())
        token = self.store.create_snapshot(4)
        self.store.append("a", "a2", 7, {"x": -1})
        self.store.create("d", "root")
        self.store.append("d", "d0", 8, {"z": 3})
        self.store.merge("a", "d", "M", 9, {"z": 1})
        self.store.append("main", "r3", 10, {"x": 1})
        self.assertEqual(self.call(token=token), before)


if __name__ == "__main__":
    unittest.main()
