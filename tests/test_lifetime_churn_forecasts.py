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
    """Independently project a reference periodicity set forward."""
    waves = reference_waves(evolution, min_identities)["waves"]
    boundary = len(waves) - 1
    upper = boundary + horizon
    periodicities = reference_periodicities(
        evolution, min_identities, max_jitter
    )["periodicities"]

    records = []
    for record in periodicities:
        last = record["last"]
        period = record["period"]
        smallest = min(record["intervals"])
        largest = max(record["intervals"])
        predictions = []
        ordinal = 1
        while True:
            wave = last + ordinal * period
            if wave > upper:
                break
            if wave > boundary:
                predictions.append(
                    {
                        "ordinal": ordinal,
                        "wave": wave,
                        "earliest": last + ordinal * smallest,
                        "latest": last + ordinal * largest,
                    }
                )
            ordinal += 1
        if not predictions:
            continue
        projected = dict(record)
        projected["forecasts"] = tuple(predictions)
        records.append(projected)

    return {
        "forecasts": tuple(records),
        "totals": {
            "identities": len(records),
            "predictions": sum(len(r["forecasts"]) for r in records),
            "waves": sum(r["waves"] for r in records),
            "segments": sum(r["segments"] for r in records),
            "added": sum(r["added"] for r in records),
            "removed": sum(r["removed"] for r in records),
            "changed": sum(r["changed"] for r in records),
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
            self.assertIsInstance(record["appearances"], tuple)
            self.assertIsInstance(record["intervals"], tuple)
            self.assertIsInstance(record["forecasts"], tuple)
            for appearance in record["appearances"]:
                self.assertEqual(
                    list(appearance),
                    ["wave", "from", "to", "segments",
                     "added", "removed", "changed"],
                )
            for prediction in record["forecasts"]:
                self.assertEqual(
                    list(prediction),
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
            (((0,), (1,), (2,)), 1, 0, 1),
            (((0,), (1,), (2,)), 1, 2, 4),
            (((0,), (1,), (1,), (2,)), 1, 0, 2),
            (((0,), (1,), (1,), (2,), (2,), (0,)), 1, 0, 1),
            (((0,), (1,), (1,), (2,), (2,), (0,)), 1, 0, 3),
            (((0,), (1,), (1,), (2,), (2,), (0,)), 1, 1, 5),
            (
                ((0,), (1,), (1,), (2,), (2,), (0,), (0,), (1,)),
                1,
                0,
                2,
            ),
            (
                ((0,), (1,), (1,), (2,), (2,), (0,), (0,), (1,)),
                1,
                2,
                4,
            ),
            (((0, 2), (1,), (1,), (0, 1, 2)), 1, 1, 3),
            (((2,), (0,)), 1, 1, 4),
            (((0, 1), (0, 1), (0, 1)), 1, 1, 2),
            (((0, 1, 2),), 1, 1, 3),
            ((), 1, 1, 1),
        )
        for windows, min_identities, max_jitter, horizon in cases:
            with self.subTest(windows=windows,
                              min_identities=min_identities,
                              max_jitter=max_jitter,
                              horizon=horizon):
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

    def test_diamond_fixture_projects_three_waves_within_horizon(self):
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
                [p["ordinal"] for p in record["forecasts"]], [1, 2, 3]
            )
            self.assertEqual(
                [p["wave"] for p in record["forecasts"]], [3, 4, 5]
            )
            for prediction in record["forecasts"]:
                self.assertEqual(
                    prediction["earliest"], prediction["wave"]
                )
                self.assertEqual(
                    prediction["latest"], prediction["wave"]
                )
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

    def test_horizon_one_keeps_only_the_first_candidate(self):
        result = self.call(
            windows=((0,), (1,), (1,), (2,), (2,), (0,)),
            horizon=1,
        )
        self.assertEqual(len(result["forecasts"]), 3)
        for record in result["forecasts"]:
            self.assertEqual(len(record["forecasts"]), 1)
            self.assertEqual(record["forecasts"][0]["wave"], 3)
        self.assertEqual(result["totals"]["predictions"], 3)

    def test_predictions_respect_the_global_boundary(self):
        from tests.test_lifetime_churn_waves import waves_call

        windows = (
            (0,), (1,), (1,), (2,), (2,), (0,), (0,), (1,),
        )
        result = self.call(windows=windows, max_jitter=2, horizon=5)
        boundary = len(waves_call(
            self.store, self.args, windows=windows
        )["waves"]) - 1
        for record in result["forecasts"]:
            self.assertLessEqual(record["last"], boundary)
            for prediction in record["forecasts"]:
                self.assertGreater(prediction["wave"], boundary)
                self.assertLessEqual(
                    prediction["wave"], boundary + 5
                )
                self.assertEqual(
                    prediction["wave"],
                    record["last"]
                    + prediction["ordinal"] * record["period"],
                )

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

    @staticmethod
    def _periodicity(waves, identity="A", type_name="node", **counts):
        result = BranchStore._build_lifetime_churn_periodicities_result(
            waves, 100, None
        )
        by_identity = {
            (r["type"], r["identity"]): r
            for r in result["periodicities"]
        }
        return by_identity[(type_name, identity)]

    def test_wave_ordinal_and_earliest_latest_formulas(self):
        # last 2, period 2 with steady intervals (2, 2): candidates
        # 4 and 6 inside a horizon of five (upper bound 7).
        waves = (
            self._wave(0, (self._member("node", "A"),)),
            self._wave(1, (self._member("node", "Z1"),)),
            self._wave(2, (self._member("node", "A"),)),
            self._wave(3, (self._member("node", "Z3"),)),
            self._wave(4, (self._member("node", "A"),)),
        )
        record = self._periodicity(waves)
        result = BranchStore._build_lifetime_churn_forecasts_result(
            (record,), 4, 3, 100
        )
        forecasts = result["forecasts"][0]["forecasts"]
        self.assertEqual(
            forecasts,
            (
                {"ordinal": 1, "wave": 6, "earliest": 6, "latest": 6},
            ),
        )
        result = BranchStore._build_lifetime_churn_forecasts_result(
            (record,), 4, 5, 100
        )
        forecasts = result["forecasts"][0]["forecasts"]
        self.assertEqual(
            [(p["ordinal"], p["wave"]) for p in forecasts],
            [(1, 6), (2, 8)],
        )

    def test_jitter_spreads_earliest_and_latest(self):
        # A at waves 0, 1, 4 -> intervals (1, 3), period is the lower
        # median of the sorted pair, one; last is 4. Candidates advance
        # by one while earliest/latest use the smallest/largest gap.
        waves = (
            self._wave(0, (self._member("node", "A"),)),
            self._wave(1, (
                self._member("node", "A"),
                self._member("node", "Z1"),
            )),
            self._wave(2, (self._member("node", "Z2"),)),
            self._wave(3, (self._member("node", "Z3"),)),
            self._wave(4, (
                self._member("node", "A"),
                self._member("node", "Z4"),
            )),
        )
        record = self._periodicity(waves)
        self.assertEqual(record["intervals"], (1, 3))
        self.assertEqual(record["period"], 1)
        result = BranchStore._build_lifetime_churn_forecasts_result(
            (record,), 4, 2, 100
        )
        forecasts = result["forecasts"][0]["forecasts"]
        self.assertEqual(
            [(p["ordinal"], p["wave"]) for p in forecasts],
            [(1, 5), (2, 6)],
        )
        # latest is free to run past the horizon upper bound.
        self.assertEqual(
            [(p["earliest"], p["latest"]) for p in forecasts],
            [(5, 7), (6, 10)],
        )

    def test_candidates_at_or_before_boundary_keep_ordinal_numbering(self):
        # A synthetic record last seen at wave 1 with a steady period
        # of two; the observation boundary is wave 4. The ordinal-one
        # candidate (wave 3) is not beyond the boundary, so it is
        # skipped, but ordinal numbering is not restarted: the first
        # kept prediction is ordinal two.
        record = {
            "type": "node",
            "identity": "A",
            "waves": 3,
            "first": 0,
            "last": 1,
            "span": 2,
            "segments": 3,
            "added": 0,
            "removed": 0,
            "changed": 3,
            "appearances": (),
            "intervals": (2, 2),
            "period": 2,
            "jitter": 0,
        }
        result = BranchStore._build_lifetime_churn_forecasts_result(
            (record,), 4, 2, 100
        )
        self.assertEqual(len(result["forecasts"]), 1)
        self.assertEqual(
            [
                (p["ordinal"], p["wave"], p["earliest"], p["latest"])
                for p in result["forecasts"][0]["forecasts"]
            ],
            [(2, 5, 5, 5)],
        )

    def test_last_before_boundary_with_large_period_is_dropped(self):
        # Last seen at wave 0, period four, boundary 4, horizon two:
        # ordinal one lands on wave 4 (not strictly beyond the
        # boundary) and ordinal two on wave 8 (past the upper bound
        # 6), so no candidate is in range and the identity is dropped.
        record = {
            "type": "node",
            "identity": "A",
            "waves": 3,
            "first": 0,
            "last": 0,
            "span": 1,
            "segments": 3,
            "added": 0,
            "removed": 0,
            "changed": 3,
            "appearances": (),
            "intervals": (4, 4),
            "period": 4,
            "jitter": 0,
        }
        result = BranchStore._build_lifetime_churn_forecasts_result(
            (record,), 4, 2, 100
        )
        self.assertEqual(result["forecasts"], ())
        self.assertEqual(result["totals"]["identities"], 0)
        self.assertEqual(result["totals"]["predictions"], 0)

    def test_identity_without_an_in_range_candidate_is_dropped(self):
        waves = tuple(
            self._wave(
                serial,
                (self._member("node", "A"),)
                if serial in (0, 2, 4)
                else (self._member("node", f"Z{serial}"),),
            )
            for serial in range(5)
        )
        record = self._periodicity(waves)
        self.assertEqual((record["last"], record["period"]), (4, 2))
        # Boundary 4, horizon one: the first candidate is wave 6, past
        # the upper bound 5 -- the identity must not be selected.
        result = BranchStore._build_lifetime_churn_forecasts_result(
            (record,), 4, 1, 100
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

    def test_empty_periodicity_set_yields_seven_zeros(self):
        result = BranchStore._build_lifetime_churn_forecasts_result(
            (), -1, 5, 100
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

    def test_records_keep_periodicity_order_and_all_fields(self):
        waves = (
            self._wave(0, (
                self._member("node", "A", changed=1),
                self._member("edge", "B", changed=1),
            )),
            self._wave(1, (
                self._member("node", "A"),
                self._member("edge", "B"),
            )),
            self._wave(2, (
                self._member("node", "A"),
                self._member("edge", "B"),
            )),
        )
        periodicity_result = (
            BranchStore._build_lifetime_churn_periodicities_result(
                waves, 0, None
            )
        )
        records = periodicity_result["periodicities"]
        result = BranchStore._build_lifetime_churn_forecasts_result(
            records, 2, 2, 100
        )
        self.assertEqual(
            [(r["type"], r["identity"]) for r in result["forecasts"]],
            [(r["type"], r["identity"]) for r in records],
        )
        for projected, source in zip(result["forecasts"], records):
            for key in (
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
            ):
                self.assertEqual(projected[key], source[key])

    def test_totals_count_predictions_and_historical_evidence(self):
        waves = (
            self._wave(0, (
                self._member("node", "A", added=1),
                self._member("edge", "B", removed=1),
            )),
            self._wave(1, (
                self._member("node", "A", changed=2),
                self._member("edge", "B"),
            )),
            self._wave(2, (
                self._member("node", "A"),
                self._member("edge", "B"),
            )),
        )
        periodicities = (
            BranchStore
            ._build_lifetime_churn_periodicities_result(waves, 0, None)
            ["periodicities"]
        )
        result = BranchStore._build_lifetime_churn_forecasts_result(
            periodicities, 2, 2, 100
        )
        self.assertEqual(result["totals"]["identities"], 2)
        self.assertEqual(result["totals"]["predictions"], 4)
        self.assertEqual(result["totals"]["waves"], 6)
        self.assertEqual(result["totals"]["segments"], 6)
        self.assertEqual(result["totals"]["added"], 1)
        self.assertEqual(result["totals"]["removed"], 1)
        self.assertEqual(result["totals"]["changed"], 2)

    def test_forecast_limit_counts_predictions_not_identities(self):
        waves = (
            self._wave(0, (
                self._member("node", "A"),
                self._member("node", "B"),
            )),
            self._wave(1, (
                self._member("node", "A"),
                self._member("node", "B"),
            )),
            self._wave(2, (
                self._member("node", "A"),
                self._member("node", "B"),
            )),
        )
        periodicities = (
            BranchStore
            ._build_lifetime_churn_periodicities_result(waves, 0, None)
            ["periodicities"]
        )
        # Two identities, one prediction each: two identities exceed a
        # cap of one only when predictions, not identities, are counted.
        with self.assertRaises(ValueError) as caught:
            BranchStore._build_lifetime_churn_forecasts_result(
                periodicities, 2, 1, 1
            )
        self.assertIn("forecast limit", str(caught.exception))
        # One prediction per identity fits a cap of two even though the
        # cap equals the identity count.
        result = BranchStore._build_lifetime_churn_forecasts_result(
            periodicities, 2, 1, 2
        )
        self.assertEqual(len(result["forecasts"]), 2)
        # A wider horizon yields four predictions and trips the same
        # cap even though the identity count is unchanged.
        with self.assertRaises(ValueError):
            BranchStore._build_lifetime_churn_forecasts_result(
                periodicities, 2, 2, 2
            )

    def test_output_shares_no_object_with_the_periodicity_input(self):
        waves = (
            self._wave(0, (self._member("node", "A"),)),
            self._wave(1, (self._member("node", "A"),)),
            self._wave(2, (self._member("node", "A"),)),
        )
        record = self._periodicity(waves)
        result = BranchStore._build_lifetime_churn_forecasts_result(
            (record,), 2, 2, 100
        )
        projected = result["forecasts"][0]
        self.assertIsNot(projected, record)
        self.assertIsNot(projected["appearances"], record["appearances"])
        self.assertIsNot(projected["intervals"], record["intervals"])
        for projected_appearance, source_appearance in zip(
            projected["appearances"], record["appearances"]
        ):
            self.assertIsNot(projected_appearance, source_appearance)
        record["intervals"] = (99,)
        record["appearances"][0]["wave"] = 99
        record["period"] = 99
        self.assertEqual(projected["intervals"], (1, 1))
        self.assertEqual(projected["period"], 1)
        self.assertEqual(projected["appearances"][0]["wave"], 0)


class LifetimeChurnForecastsLimitValidationTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        overrides.setdefault("causes", ("a0",))
        return forecasts_call(self.store, self.args, **overrides)

    def extended(self, **overrides):
        overrides.setdefault(
            "windows", ((0,), (1,), (1,), (2,), (2,), (0,))
        )
        overrides.setdefault("horizon", 3)
        return self.call(**overrides)

    def test_horizon_must_be_a_non_bool_positive_int(self):
        for bad in (True, False, 1.0, "1", None):
            with self.subTest(horizon=bad):
                with self.assertRaises(TypeError):
                    self.call(horizon=bad)
        with self.assertRaises(ValueError):
            self.call(horizon=0)
        with self.assertRaises(ValueError):
            self.call(horizon=-4)
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
            self.call(forecast_limit=-4)

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
            self.extended(forecast_limit=8)
        self.assertIn("forecast limit", str(caught.exception))
        result = self.extended(forecast_limit=9)
        self.assertEqual(result["totals"]["predictions"], 9)
        # An empty result never trips the cap.
        result = self.call(forecast_limit=1)
        self.assertEqual(result["forecasts"], ())

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
            "periodicity_limit",
        ):
            with self.assertRaises(ValueError):
                self.call(**{name: 0})
            with self.assertRaises(TypeError):
                self.call(**{name: "x"})
        # Caps of one would bound the earlier queries; forecasts
        # ignores every one of them and still returns all three
        # steady identities with their predictions.
        result = self.extended(
            churn_limit=1,
            streak_limit=1,
            min_waves=99,
            recurrence_limit=1,
            periodicity_limit=1,
        )
        self.assertEqual(len(result["forecasts"]), 3)
        self.assertEqual(result["totals"]["predictions"], 9)

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
                {"ordinal": 9, "wave": 99,
                 "earliest": 99, "latest": 99},
            )
            record["waves"] = 99
            for appearance in record["appearances"]:
                appearance["segments"] = 99
            for prediction in record["forecasts"]:
                prediction["wave"] = 99
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

        forecasts_call(
            store, args, windows=self.EXTENDED, horizon=3
        )
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
            dict(forecast_limit=1, windows=self.EXTENDED, horizon=3),
            dict(causes=("zzz",)),
            dict(limit=1),
            dict(reference="ghost"),
        ]
        for overrides in failures:
            with self.subTest(overrides=overrides):
                with self.assertRaises(Exception):
                    forecasts_call(
                        store,
                        args,
                        causes=("a0",),
                        horizon=3,
                        **overrides,
                    )

        self.assertEqual(dict(store._heads), heads)
        self.assertEqual(copy.deepcopy(store._appends), appends)
        self.assertEqual(copy.deepcopy(store._merges), merges)
        self.assertEqual(store.audit_log(), audit)

    def test_existing_public_entries_are_unaffected(self):
        store, args = diamond_store()
        forecasts_call(
            store, args, windows=self.EXTENDED, horizon=3
        )
        from tests.test_lifetime_churn_periodicities import (
            periodicities_call,
        )
        from tests.test_lifetime_churn_recurrences import recurrences_call
        from tests.test_lifetime_churn_waves import waves_call

        waves = waves_call(store, args)
        recurrences = recurrences_call(
            store, args,
            windows=((0,), (1,), (1,), (2,)), min_waves=2,
        )
        periodicities = periodicities_call(
            store, args, windows=self.EXTENDED
        )
        self.assertEqual(waves_call(store, args), waves)
        self.assertEqual(
            recurrences_call(
                store, args,
                windows=((0,), (1,), (1,), (2,)), min_waves=2,
            ),
            recurrences,
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
            self.call(token=token, horizon=0)
        with self.assertRaises(TypeError):
            self.call(token=token, forecast_limit="x")
        self.call(token=token)
        with self.assertRaises(RuntimeError):
            self.call(token=token)

    def test_results_come_from_the_frozen_view(self):
        before = copy.deepcopy(self.call(horizon=3))
        token = self.store.create_snapshot(4)
        self.store.append("a", "a2", 7, {"x": -1})
        self.store.create("d", "root")
        self.store.append("d", "d0", 8, {"z": 3})
        self.store.merge("a", "d", "M", 9, {"z": 1})
        self.store.append("main", "r3", 10, {"x": 1})
        self.assertEqual(self.call(token=token, horizon=3), before)


if __name__ == "__main__":
    unittest.main()
