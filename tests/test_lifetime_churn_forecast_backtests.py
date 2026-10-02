import copy
import inspect
import unittest

from city_twin.branches import BranchStore

from tests.test_lifetime_churn import churn_kwargs, diamond_store
from tests.test_lifetime_churn_waves import reference_waves
from tests.test_lifetime_churn_forecasts import forecasts_kwargs


def backtests_kwargs(store_args, **overrides) -> dict:
    cutoffs = overrides.pop("cutoffs", ())
    backtest_limit = overrides.pop("backtest_limit", 50)
    kwargs = forecasts_kwargs(store_args, **overrides)
    kwargs["cutoffs"] = cutoffs
    kwargs["backtest_limit"] = backtest_limit
    return kwargs


def backtests_call(store, store_args, **overrides):
    return store.lifetime_churn_forecast_backtests(
        **backtests_kwargs(store_args, **overrides)
    )


def reference_backtests(
    evolution: dict[str, object],
    min_identities: int,
    max_jitter: int,
    horizon: int,
    cutoffs: tuple[int, ...],
) -> dict[str, object]:
    """Independently backtest per-cutoff projections against later waves."""
    waves = reference_waves(evolution, min_identities)["waves"]
    observed: dict[tuple[str, object], list[int]] = {}
    for serial, wave in enumerate(waves):
        for member in wave["members"]:
            observed.setdefault(
                (member["type"], member["identity"]), []
            ).append(serial)
    final_serial = len(waves) - 1

    zero = {
        "predictions": 0,
        "hit": 0,
        "early": 0,
        "late": 0,
        "missed": 0,
        "unresolved": 0,
    }
    backtests = []
    totals = dict(zero)
    type_order = {"node": 0, "edge": 1, "gap": 2}
    for cutoff in cutoffs:
        training = waves[: cutoff + 1]
        # Rebuild each identity's training evidence from only the
        # cutoff's waves, mirroring the periodicity aggregation.
        encounter: dict[tuple[str, object], int] = {}
        next_encounter = 0
        per_identity: dict[
            tuple[str, object], dict[str, object]
        ] = {}
        for serial, wave in enumerate(training):
            for member in wave["members"]:
                key = (member["type"], member["identity"])
                if key not in encounter:
                    encounter[key] = next_encounter
                    next_encounter += 1
                    per_identity[key] = {
                        "serials": [],
                        "segments": 0,
                        "added": 0,
                        "removed": 0,
                        "changed": 0,
                    }
                evidence = per_identity[key]
                evidence["serials"].append(serial)
                evidence["segments"] += member["segments"]
                evidence["added"] += member["added"]
                evidence["removed"] += member["removed"]
                evidence["changed"] += member["changed"]

        cutoff_totals = dict(zero)
        outcomes = []
        records = {}
        ordered_keys = []
        for key, evidence in per_identity.items():
            serials = evidence["serials"]
            if len(serials) < 3:
                continue
            intervals = tuple(b - a for a, b in zip(serials, serials[1:]))
            if max(intervals) - min(intervals) > max_jitter:
                continue
            ordered = sorted(intervals)
            middle = len(ordered) // 2
            period = (
                ordered[middle - 1]
                if len(ordered) % 2 == 0
                else ordered[middle]
            )
            records[key] = (serials[-1], period, min(intervals),
                            max(intervals), intervals)
            ordered_keys.append(key)

        # Mirror the periodicity sort key exactly so outcomes follow the
        # existing periodicity record order.
        ordered_keys.sort(
            key=lambda key: (
                max(records[key][4]) - min(records[key][4]),
                -len(per_identity[key]["serials"]),
                -per_identity[key]["segments"],
                -(
                    per_identity[key]["added"]
                    + per_identity[key]["removed"]
                    + per_identity[key]["changed"]
                ),
                per_identity[key]["serials"][0],
                type_order[key[0]],
                encounter[key],
            )
        )

        upper = cutoff + horizon
        for key in ordered_keys:
            last, period, smallest, largest, _intervals = records[key]
            ordinal = 1
            while True:
                wave = last + ordinal * period
                if wave > upper:
                    break
                if wave > cutoff:
                    earliest = last + ordinal * smallest
                    latest = last + ordinal * largest
                    appearances = observed.get(key, ())
                    match = None
                    for serial in appearances:
                        if serial >= earliest:
                            if serial <= latest:
                                match = serial
                            break
                    if match is not None:
                        status = (
                            "hit" if match == wave
                            else "early" if match < wave
                            else "late"
                        )
                        actual = match
                    elif latest <= final_serial:
                        status = "missed"
                        actual = None
                    else:
                        status = "unresolved"
                        actual = None
                    outcomes.append(
                        {
                            "type": key[0],
                            "identity": key[1],
                            "ordinal": ordinal,
                            "predicted": wave,
                            "earliest": earliest,
                            "latest": latest,
                            "actual": actual,
                            "status": status,
                        }
                    )
                    cutoff_totals["predictions"] += 1
                    cutoff_totals[status] += 1
                ordinal += 1

        for name in zero:
            totals[name] += cutoff_totals[name]
        backtests.append(
            {
                "cutoff": cutoff,
                "outcomes": tuple(outcomes),
                "totals": cutoff_totals,
            }
        )

    return {"backtests": tuple(backtests), "totals": totals}


class LifetimeChurnForecastBacktestsSignatureTests(unittest.TestCase):
    def test_public_signature_appends_cutoffs_and_backtest_limit(self):
        parameters = list(
            inspect.signature(
                BranchStore.lifetime_churn_forecast_backtests
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
        windows = (
            (0,), (1,), (1,), (2,), (2,), (0,), (0,), (1,),
        )
        result = backtests_call(
            store, args, windows=windows, horizon=3, cutoffs=(2,)
        )
        self.assertIsInstance(result, dict)
        self.assertEqual(list(result), ["backtests", "totals"])
        self.assertIsInstance(result["backtests"], tuple)
        self.assertEqual(
            list(result["totals"]),
            ["predictions", "hit", "early", "late", "missed",
             "unresolved"],
        )
        for backtest in result["backtests"]:
            self.assertIsInstance(backtest, dict)
            self.assertEqual(
                list(backtest), ["cutoff", "outcomes", "totals"]
            )
            self.assertIsInstance(backtest["outcomes"], tuple)
            self.assertEqual(
                list(backtest["totals"]),
                ["predictions", "hit", "early", "late", "missed",
                 "unresolved"],
            )
            for outcome in backtest["outcomes"]:
                self.assertIsInstance(outcome, dict)
                self.assertEqual(
                    list(outcome),
                    [
                        "type",
                        "identity",
                        "ordinal",
                        "predicted",
                        "earliest",
                        "latest",
                        "actual",
                        "status",
                    ],
                )


class LifetimeChurnForecastBacktestsResultTests(unittest.TestCase):
    EXTENDED = (
        (0,), (1,), (1,), (2,), (2,), (0,), (0,), (1,),
    )

    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        return backtests_call(self.store, self.args, **overrides)

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

    def test_matches_independent_backtest_of_evolution(self):
        cases = (
            (self.EXTENDED, 1, 0, 2, (2,)),
            (self.EXTENDED, 1, 0, 3, (2,)),
            (self.EXTENDED, 1, 0, 3, (2, 3)),
            (self.EXTENDED, 1, 2, 5, (2,)),
            (
                ((0,), (1,), (1,), (2,), (2,), (0,)),
                1,
                0,
                3,
                (2,),
            ),
            (
                ((0,), (1,), (1,), (2,), (2,), (0,)),
                1,
                0,
                3,
                (0, 1, 2),
            ),
        )
        for windows, min_identities, max_jitter, horizon, cutoffs in cases:
            with self.subTest(windows=windows,
                              min_identities=min_identities,
                              max_jitter=max_jitter,
                              horizon=horizon,
                              cutoffs=cutoffs):
                self.assertEqual(
                    self.call(
                        windows=windows,
                        min_identities=min_identities,
                        max_jitter=max_jitter,
                        horizon=horizon,
                        cutoffs=cutoffs,
                    ),
                    reference_backtests(
                        self.evolution(windows=windows),
                        min_identities,
                        max_jitter,
                        horizon,
                        cutoffs,
                    ),
                )

    def test_hits_match_the_fixture_periodic_waves(self):
        result = self.call(
            windows=self.EXTENDED, max_jitter=0, horizon=3, cutoffs=(2,)
        )
        backtest = result["backtests"][0]
        self.assertEqual(backtest["cutoff"], 2)
        self.assertEqual(backtest["totals"]["predictions"], 9)
        self.assertEqual(backtest["totals"]["hit"], 3)
        self.assertEqual(backtest["totals"]["unresolved"], 6)
        hits = [o for o in backtest["outcomes"] if o["status"] == "hit"]
        self.assertEqual(len(hits), 3)
        for outcome in hits:
            self.assertEqual(outcome["ordinal"], 1)
            self.assertEqual(outcome["predicted"], 3)
            self.assertEqual(outcome["actual"], 3)
        self.assertEqual(result["totals"], backtest["totals"])

    def test_global_totals_sum_every_cutoff(self):
        result = self.call(
            windows=self.EXTENDED, max_jitter=0, horizon=3, cutoffs=(2, 3)
        )
        self.assertEqual(len(result["backtests"]), 2)
        self.assertEqual(
            [b["cutoff"] for b in result["backtests"]], [2, 3]
        )
        aggregate = {
            name: sum(b["totals"][name] for b in result["backtests"])
            for name in (
                "predictions", "hit", "early", "late",
                "missed", "unresolved",
            )
        }
        self.assertEqual(result["totals"], aggregate)
        # The cutoff at the final wave observes nothing after it.
        last = result["backtests"][1]
        self.assertEqual(last["totals"]["hit"], 0)
        self.assertEqual(last["totals"]["missed"], 0)
        self.assertGreater(last["totals"]["unresolved"], 0)

    def test_predictions_are_ordered_by_periodicity_then_ordinal(self):
        result = self.call(
            windows=self.EXTENDED, horizon=3, cutoffs=(2,)
        )
        outcomes = result["backtests"][0]["outcomes"]
        forecasts = self.store.lifetime_churn_forecasts(
            **forecasts_kwargs(
                self.args,
                windows=(
                    (0,), (1,), (1,), (2,), (2,), (0,),
                ),
                horizon=3,
            )
        )
        expected = []
        for record in forecasts["forecasts"]:
            for prediction in record["forecasts"]:
                expected.append(
                    (record["type"], record["identity"],
                     prediction["ordinal"])
                )
        self.assertEqual(
            [(o["type"], o["identity"], o["ordinal"]) for o in outcomes],
            expected,
        )

    def test_empty_cutoffs_return_empty_backtests_and_zero_totals(self):
        result = self.call(windows=self.EXTENDED, cutoffs=())
        self.assertEqual(result["backtests"], ())
        self.assertEqual(
            result["totals"],
            {
                "predictions": 0,
                "hit": 0,
                "early": 0,
                "late": 0,
                "missed": 0,
                "unresolved": 0,
            },
        )

    def test_empty_wave_set_with_empty_cutoffs_still_zeros(self):
        for overrides in (
            {"windows": ()},
            {"causes": ()},
            {"windows": ((0, 1, 2),)},
        ):
            with self.subTest(overrides=overrides):
                result = self.call(**{**overrides, "cutoffs": ()})
                self.assertEqual(result["backtests"], ())
                self.assertEqual(result["totals"]["predictions"], 0)

    def test_actual_is_none_only_for_missed_and_unresolved(self):
        result = self.call(
            windows=self.EXTENDED, max_jitter=2, horizon=6, cutoffs=(2,)
        )
        for outcome in result["backtests"][0]["outcomes"]:
            if outcome["status"] in ("hit", "early", "late"):
                self.assertIsNotNone(outcome["actual"])
                self.assertIsInstance(outcome["actual"], int)
            else:
                self.assertIn(
                    outcome["status"], ("missed", "unresolved")
                )
                self.assertIsNone(outcome["actual"])


class LifetimeChurnForecastBacktestsStatusTests(unittest.TestCase):
    """Exercise every status through synthetic wave membership."""

    @staticmethod
    def _member(type_name, identity, segments=1, added=0, removed=0,
                changed=0):
        return {
            "type": type_name,
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

    def _build(self, waves, cutoffs, *, max_jitter=0, horizon=10):
        return BranchStore._build_lifetime_churn_backtests_result(
            waves, list(cutoffs), max_jitter, horizon, 100, 100
        )

    def test_hit_early_late_classification_by_first_appearance(self):
        # A appears at waves 0, 1, 2 (steady period 1). Cutoff 2 gives
        # predicted wave 3 with range 3..3. Observed at 3 is a hit; a
        # jittered range 2..4 with first appearance 2 is early and at 4
        # is late.
        base = tuple(
            self._wave(s, (self._member("node", "A"),)) for s in range(3)
        )
        hit_waves = base + (
            self._wave(3, (self._member("node", "A"),)),
        )
        result = self._build(hit_waves, (2,), horizon=2)
        statuses = {
            (o["ordinal"], o["status"]): o
            for o in result["backtests"][0]["outcomes"]
        }
        self.assertEqual(statuses[(1, "hit")]["actual"], 3)

        # Intervals (1, 1) keep earliest == latest, so to get early/late
        # use jitter: A at 0, 2, 3 gives intervals (2, 1), last 3.
        waves = (
            self._wave(0, (self._member("node", "A"),)),
            self._wave(1, (self._member("node", "Z1"),)),
            self._wave(2, (self._member("node", "A"),)),
            self._wave(3, (self._member("node", "A"),)),
            self._wave(4, (self._member("node", "Z4"),)),
            self._wave(5, (self._member("node", "A"),)),
        )
        # last=3, intervals (2,1), period median=1: ordinal 1 predicts
        # wave 4 with earliest 4, latest 5; first appearance is 5 -> late.
        result = self._build(waves, (3,), horizon=3, max_jitter=1)
        outcomes = result["backtests"][0]["outcomes"]
        first = next(o for o in outcomes if o["ordinal"] == 1)
        self.assertEqual(
            (first["predicted"], first["earliest"], first["latest"]),
            (4, 4, 5),
        )
        self.assertEqual(first["status"], "late")
        self.assertEqual(first["actual"], 5)

    def test_jitter_range_spreads_early_hit_late_and_missed(self):
        # A trains at waves 0, 1, 3, 6: intervals (1, 2, 3), period 2,
        # last 6. Ordinal one predicts wave 8 with earliest 7 and
        # latest 9; the next observed appearance classifies each case.
        for observed_A, expected in (
            (7, ("early", 7)),
            (8, ("hit", 8)),
            (9, ("late", 9)),
            (10, ("missed", None)),
        ):
            with self.subTest(observed_A=observed_A):
                waves = tuple(
                    self._wave(
                        serial,
                        (
                            self._member("node", f"Z{serial}"),
                            self._member("node", "A"),
                        )
                        if serial in (0, 1, 3, 6, observed_A)
                        else (self._member("node", f"Z{serial}"),),
                    )
                    for serial in range(10)
                )
                result = self._build(
                    waves, (6,), horizon=4, max_jitter=2
                )
                outcome = next(
                    o
                    for o in result["backtests"][0]["outcomes"]
                    if o["ordinal"] == 1
                )
                self.assertEqual(
                    (
                        outcome["predicted"],
                        outcome["earliest"],
                        outcome["latest"],
                    ),
                    (8, 7, 9),
                )
                self.assertEqual(
                    (outcome["status"], outcome["actual"]), expected
                )

    def test_missed_when_interval_fully_observed_without_appearance(self):
        # A at waves 0, 1, 2 then absent; cutoff 2 predicts 3 with range
        # 3..3, and wave 3 is fully observed without A -> missed.
        waves = (
            self._wave(0, (self._member("node", "A"),)),
            self._wave(1, (self._member("node", "A"),)),
            self._wave(2, (self._member("node", "A"),)),
            self._wave(3, (self._member("node", "Z"),)),
        )
        result = self._build(waves, (2,), horizon=1)
        outcome = result["backtests"][0]["outcomes"][0]
        self.assertEqual(outcome["status"], "missed")
        self.assertIsNone(outcome["actual"])
        self.assertEqual(
            result["backtests"][0]["totals"],
            {"predictions": 1, "hit": 0, "early": 0, "late": 0,
             "missed": 1, "unresolved": 0},
        )

    def test_unresolved_when_latest_passes_the_last_observed_wave(self):
        waves = (
            self._wave(0, (self._member("node", "A"),)),
            self._wave(1, (self._member("node", "A"),)),
            self._wave(2, (self._member("node", "A"),)),
        )
        # Cutoff 2 is the final wave: predicted 3, range 3..3, latest 3
        # is beyond the final observed serial 2 -> unresolved.
        result = self._build(waves, (2,), horizon=1)
        outcome = result["backtests"][0]["outcomes"][0]
        self.assertEqual(outcome["status"], "unresolved")
        self.assertIsNone(outcome["actual"])

    def test_appearance_before_earliest_never_matches(self):        # A at 0, 1, 2 (period 1); jitter via wider intervals is not
        # available, so build a record with intervals (2, 2): A at
        # 0, 2, 4; cutoff 4, ordinal 1 predicts 6, range 6..6. A also
        # appears at wave 5 (before earliest 6) and not at 6; wave 6 is
        # observed, so the wave-5 appearance must not count and the
        # prediction is missed.
        waves = (
            self._wave(0, (self._member("node", "A"),)),
            self._wave(1, (self._member("node", "Z1"),)),
            self._wave(2, (self._member("node", "A"),)),
            self._wave(3, (self._member("node", "Z3"),)),
            self._wave(4, (self._member("node", "A"),)),
            self._wave(5, (self._member("node", "A"),)),
            self._wave(6, (self._member("node", "Z6"),)),
        )
        result = self._build(waves, (4,), horizon=3)
        outcome = next(
            o
            for o in result["backtests"][0]["outcomes"]
            if o["ordinal"] == 1
        )
        self.assertEqual(
            (outcome["predicted"], outcome["earliest"],
             outcome["latest"]),
            (6, 6, 6),
        )
        self.assertEqual(outcome["status"], "missed")
        self.assertIsNone(outcome["actual"])

    def test_cutoff_uses_only_its_own_prefix_for_training(self):
        # A appears in waves 0,1,2 and again at 4. At cutoff 2 the
        # steady set trains on 0,1,2 only; wave 3 (no A) is a missed
        # observation and wave 4 is never seen as training.
        waves = (
            self._wave(0, (self._member("node", "A"),)),
            self._wave(1, (self._member("node", "A"),)),
            self._wave(2, (self._member("node", "A"),)),
            self._wave(3, (self._member("node", "Z"),)),
            self._wave(4, (self._member("node", "A"),)),
        )
        result = self._build(waves, (2,), horizon=2)
        by_ordinal = {
            o["ordinal"]: o
            for o in result["backtests"][0]["outcomes"]
        }
        self.assertEqual(by_ordinal[1]["status"], "missed")
        # Ordinal two predicts wave 4 with range 4..4; the reappearance
        # at wave 4 lands on that center and settles as a fresh hit.
        self.assertEqual(by_ordinal[2]["status"], "hit")
        self.assertEqual(by_ordinal[2]["actual"], 4)


class LifetimeChurnForecastBacktestsCutoffValidationTests(
    unittest.TestCase
):
    EXTENDED = (
        (0,), (1,), (1,), (2,), (2,), (0,), (0,), (1,),
    )

    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        overrides.setdefault("causes", ("a0",))
        overrides.setdefault("windows", self.EXTENDED)
        return backtests_call(self.store, self.args, **overrides)

    def test_cutoffs_must_be_a_tuple(self):
        for bad in ([0, 1], {0: None}, (x for x in (0,)), "x", 0, None):
            with self.subTest(cutoffs=bad):
                with self.assertRaises(TypeError):
                    self.call(cutoffs=bad)

    def test_cutoff_elements_must_be_non_bool_ints(self):
        for bad in (True, False, 1.0, "1", None):
            with self.subTest(cutoff=bad):
                with self.assertRaises(TypeError):
                    self.call(cutoffs=(0, bad))
                with self.assertRaises(TypeError):
                    self.call(cutoffs=(bad,))

    def test_cutoffs_negative_duplicate_unordered_raise_value_error(self):
        with self.assertRaises(ValueError):
            self.call(cutoffs=(-1,))
        with self.assertRaises(ValueError):
            self.call(cutoffs=(0, 0))
        with self.assertRaises(ValueError):
            self.call(cutoffs=(2, 1))
        with self.assertRaises(ValueError):
            self.call(cutoffs=(1, 3, 3))

    def test_cutoffs_out_of_wave_range_raise_value_error(self):
        # The extended fixture has four waves, serials 0..3.
        with self.assertRaises(ValueError) as caught:
            self.call(cutoffs=(4,))
        self.assertIn("out of range", str(caught.exception))
        with self.assertRaises(ValueError):
            self.call(cutoffs=(3, 4))
        # With fewer waves the bound tightens.
        with self.assertRaises(ValueError):
            self.call(
                windows=((0,), (1,), (1,), (2,), (2,), (0,)),
                cutoffs=(3,),
            )
        # The last valid serial is accepted.
        self.call(cutoffs=(3,))

    def test_empty_wave_set_rejects_every_nonempty_cutoff(self):
        with self.assertRaises(ValueError):
            self.call(windows=(), cutoffs=(0,))
        with self.assertRaises(ValueError):
            self.call(causes=(), cutoffs=(0,))

    def test_backtest_limit_must_be_a_non_bool_positive_int(self):
        for bad in (True, False, 1.0, "1", None):
            with self.subTest(backtest_limit=bad):
                with self.assertRaises(TypeError):
                    self.call(backtest_limit=bad)
        with self.assertRaises(ValueError):
            self.call(backtest_limit=0)
        with self.assertRaises(ValueError):
            self.call(backtest_limit=-3)

    def test_validation_order_after_forecast_limit(self):
        # forecast_limit fails before cutoffs and backtest_limit.
        with self.assertRaises(ValueError):
            self.call(forecast_limit=0, cutoffs="x")
        with self.assertRaises(TypeError):
            self.call(forecast_limit="x", cutoffs="x")
        # cutoffs container/elements fail before backtest_limit.
        with self.assertRaises(TypeError):
            self.call(cutoffs="x", backtest_limit=0)
        with self.assertRaises(TypeError):
            self.call(cutoffs=(0, "x"), backtest_limit=0)
        # Once cutoffs types pass, backtest_limit's error fires, still
        # before any state lookup or token touch.
        with self.assertRaises(TypeError):
            self.call(cutoffs=(0,), backtest_limit="x")
        with self.assertRaises(ValueError):
            self.call(cutoffs=(0,), backtest_limit=0)
        with self.assertRaises(ValueError):
            self.call(reference="ghost", backtest_limit=0)
        with self.assertRaises(ValueError):
            self.call(token="whatever", backtest_limit=0)

    def test_backtest_limit_counts_predictions_across_all_cutoffs(self):
        # With horizon three, cutoff 2 yields nine predictions and
        # cutoff 3 another nine; the cross-cutoff total must trip the
        # cap.
        self.call(cutoffs=(2,), horizon=3, backtest_limit=9)
        with self.assertRaises(ValueError) as caught:
            self.call(cutoffs=(2,), horizon=3, backtest_limit=8)
        self.assertIn("backtest limit", str(caught.exception))
        with self.assertRaises(ValueError) as caught:
            self.call(cutoffs=(2, 3), horizon=3, backtest_limit=17)
        self.assertIn("backtest limit", str(caught.exception))
        # Eighteen is the exact cross-cutoff total and must fit.
        result = self.call(
            cutoffs=(2, 3), horizon=3, backtest_limit=18
        )
        self.assertEqual(result["totals"]["predictions"], 18)
        # Empty cutoffs never trip even a cap of one.
        result = self.call(cutoffs=(), backtest_limit=1)
        self.assertEqual(result["backtests"], ())

    def test_forecast_limit_still_bounds_each_cutoff(self):
        with self.assertRaises(ValueError) as caught:
            self.call(
                cutoffs=(2,), horizon=3, forecast_limit=8,
                backtest_limit=100,
            )
        self.assertIn("forecast limit", str(caught.exception))

    def test_wave_limit_still_enforced_ahead_of_cutoff_range(self):
        with self.assertRaises(ValueError) as caught:
            self.call(wave_limit=1, cutoffs=(9,))
        self.assertIn("wave limit", str(caught.exception))


class LifetimeChurnForecastBacktestsStateTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        overrides.setdefault("causes", ("a0",))
        return backtests_call(self.store, self.args, **overrides)

    def test_unknown_branch_history_node_and_cause_raise(self):
        with self.assertRaises(KeyError):
            self.call(reference="ghost", cutoffs=(0,))
        with self.assertRaises(KeyError):
            self.call(causes=("zzz",), cutoffs=(0,))

    def test_empty_windows_still_run_every_state_check(self):
        with self.assertRaises(KeyError):
            self.call(windows=(), cutoffs=(), reference="ghost")
        with self.assertRaises(ValueError):
            self.call(windows=(), cutoffs=(), limit=1)


class LifetimeChurnForecastBacktestsIsolationTests(unittest.TestCase):
    EXTENDED = (
        (0,), (1,), (1,), (2,), (2,), (0,), (0,), (1,),
    )

    def test_result_levels_are_mutually_unshared(self):
        store, args = diamond_store()
        result = backtests_call(
            store, args,
            windows=self.EXTENDED, horizon=3, cutoffs=(2, 3),
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
        # Local and global totals are distinct dicts even when equal.
        self.assertIsNot(
            result["totals"], result["backtests"][0]["totals"]
        )
        self.assertIsNot(
            result["backtests"][0]["totals"],
            result["backtests"][1]["totals"],
        )

    def test_mutating_result_never_affects_later_calls(self):
        store, args = diamond_store()
        first = backtests_call(
            store, args,
            windows=self.EXTENDED, horizon=3, cutoffs=(2,),
        )
        pristine = copy.deepcopy(first)
        for backtest in first["backtests"]:
            backtest["cutoff"] = 99
            backtest["totals"]["hit"] = 99
            for outcome in backtest["outcomes"]:
                outcome["status"] = "hit"
                outcome["actual"] = 99
                outcome["predicted"] = 99
                identity = outcome["identity"]
                if isinstance(identity, tuple):
                    outcome["identity"] = identity + ("z",)
        self.assertEqual(
            backtests_call(
                store, args,
                windows=self.EXTENDED, horizon=3, cutoffs=(2,),
            ),
            pristine,
        )

    def test_success_and_failure_change_nothing(self):
        store, args = diamond_store()
        heads = dict(store._heads)
        appends = copy.deepcopy(store._appends)
        merges = copy.deepcopy(store._merges)
        audit = store.audit_log()

        backtests_call(
            store, args, windows=self.EXTENDED, horizon=3, cutoffs=(2,),
        )
        failures = [
            dict(cutoffs="x"),
            dict(cutoffs=(-1,)),
            dict(cutoffs=(9,)),
            dict(cutoffs=(2, 1)),
            dict(cutoffs=(True,)),
            dict(backtest_limit=0),
            dict(backtest_limit="x"),
            dict(wave_limit=1, windows=self.EXTENDED, cutoffs=(0,)),
            dict(forecast_limit=1, cutoffs=(2,), horizon=3),
            dict(backtest_limit=1, cutoffs=(2, 3), horizon=3),
            dict(causes=("zzz",), cutoffs=(0,)),
            dict(reference="ghost", cutoffs=(0,)),
        ]
        for overrides in failures:
            with self.subTest(overrides=overrides):
                with self.assertRaises(Exception):
                    backtests_call(
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

    def test_existing_forecasts_entry_is_unaffected(self):
        store, args = diamond_store()
        backtests_call(
            store, args, windows=self.EXTENDED, horizon=3, cutoffs=(2,),
        )
        from tests.test_lifetime_churn_forecasts import forecasts_call

        forecasts = forecasts_call(
            store, args, windows=self.EXTENDED, horizon=3
        )
        self.assertEqual(
            forecasts_call(
                store, args, windows=self.EXTENDED, horizon=3
            ),
            forecasts,
        )


class LifetimeChurnForecastBacktestsTokenTests(unittest.TestCase):
    EXTENDED = (
        (0,), (1,), (1,), (2,), (2,), (0,), (0,), (1,),
    )

    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, token=None, **overrides):
        return backtests_call(
            self.store, self.args, token=token, **overrides
        )

    def test_success_consumes_one_read_and_failure_refunds(self):
        token = self.store.create_snapshot(3)
        self.call(
            token=token, windows=self.EXTENDED, horizon=3, cutoffs=(2,),
        )
        # Over the backtest cap: failure refunds.
        with self.assertRaises(ValueError):
            self.call(
                token=token,
                windows=self.EXTENDED,
                horizon=3,
                cutoffs=(2, 3),
                backtest_limit=1,
            )
        # Cutoff range error on the frozen view: failure refunds.
        with self.assertRaises(ValueError):
            self.call(token=token, cutoffs=(9,))
        # State failure: refunds.
        with self.assertRaises(KeyError):
            self.call(token=token, causes=("zzz",), cutoffs=(0,))
        # Parameter validation (including cutoff sign and order) runs
        # before the token is touched, so these never consume a read.
        with self.assertRaises(TypeError):
            self.call(token=token, cutoffs="x")
        with self.assertRaises(ValueError):
            self.call(token=token, cutoffs=(-1,))
        with self.assertRaises(ValueError):
            self.call(token=token, cutoffs=(2, 1))
        with self.assertRaises(ValueError):
            self.call(token=token, backtest_limit=0)
        # Exactly two more successful reads fit the allowance of three.
        self.call(token=token, cutoffs=())
        self.call(
            token=token, windows=self.EXTENDED, horizon=3, cutoffs=(2,),
        )
        with self.assertRaises(RuntimeError):
            self.call(token=token, cutoffs=())

    def test_results_come_from_the_frozen_view(self):
        before_kwargs = forecasts_kwargs(
            self.args,
            windows=((0,), (1,), (1,), (2,), (2,), (0,)),
            horizon=3,
        )
        token = self.store.create_snapshot(4)
        self.store.append("a", "a2", 7, {"x": -1})
        self.store.create("d", "root")
        self.store.append("d", "d0", 8, {"z": 3})
        self.store.merge("a", "d", "M", 9, {"z": 1})
        self.store.append("main", "r3", 10, {"x": 1})
        # The frozen view sees only the six-window history's three
        # waves, so cutoff 2 is valid and cutoff 3 is out of range.
        result = self.call(
            token=token,
            windows=((0,), (1,), (1,), (2,), (2,), (0,)),
            horizon=3,
            cutoffs=(2,),
        )
        self.assertEqual(result["totals"]["predictions"], 9)
        with self.assertRaises(ValueError):
            self.call(
                token=token,
                windows=((0,), (1,), (1,), (2,), (2,), (0,)),
                cutoffs=(3,),
            )


if __name__ == "__main__":
    unittest.main()
