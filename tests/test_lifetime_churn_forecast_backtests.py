import copy
import inspect
import unittest

from city_twin.branches import BranchStore

from tests.test_lifetime_churn import churn_kwargs, diamond_store
from tests.test_lifetime_churn_waves import reference_waves, waves_call
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


def _reference_periodicities(waves, max_jitter: int):
    """Independently derive steady-cadence records from given waves.

    Mirrors the store's periodicity construction, including the full
    record sort order, so the backtest reference iterates identities in
    exactly the order the outcomes must take.
    """
    type_order = {"node": 0, "edge": 1, "gap": 2}
    encounter = {}
    next_encounter = 0
    per_identity = {}
    for serial, wave in enumerate(waves):
        for member in wave["members"]:
            key = (member["type"], member["identity"])
            if key not in encounter:
                encounter[key] = next_encounter
                next_encounter += 1
            per_identity.setdefault(key, []).append(
                {
                    "wave": serial,
                    "segments": member["segments"],
                    "added": member["added"],
                    "removed": member["removed"],
                    "changed": member["changed"],
                }
            )

    records = []
    for (type_name, identity), appearances in per_identity.items():
        if len(appearances) < 3:
            continue
        serials = [appearance["wave"] for appearance in appearances]
        intervals = tuple(
            later - earlier
            for earlier, later in zip(serials, serials[1:])
        )
        jitter = max(intervals) - min(intervals)
        if jitter > max_jitter:
            continue
        ordered = sorted(intervals)
        middle = len(ordered) // 2
        period = (
            ordered[middle - 1]
            if len(ordered) % 2 == 0
            else ordered[middle]
        )
        records.append(
            {
                "type": type_name,
                "identity": identity,
                "waves": len(appearances),
                "first": serials[0],
                "last": serials[-1],
                "segments": sum(a["segments"] for a in appearances),
                "added": sum(a["added"] for a in appearances),
                "removed": sum(a["removed"] for a in appearances),
                "changed": sum(a["changed"] for a in appearances),
                "intervals": intervals,
                "period": period,
                "jitter": jitter,
            }
        )
    records.sort(
        key=lambda record: (
            record["jitter"],
            -record["waves"],
            -record["segments"],
            -(
                record["added"]
                + record["removed"]
                + record["changed"]
            ),
            record["first"],
            type_order[record["type"]],
            encounter[(record["type"], record["identity"])],
        )
    )
    return records


def reference_backtests(
    evolution: dict[str, object],
    min_identities: int,
    max_jitter: int,
    horizon: int,
    cutoffs: tuple[int, ...],
) -> dict[str, object]:
    """Independently backtest prefix forecasts against the full waves."""
    waves = reference_waves(evolution, min_identities)["waves"]
    last_wave = len(waves) - 1
    members_by_wave = [
        {(m["type"], m["identity"]) for m in wave["members"]}
        for wave in waves
    ]
    observed = {}
    for serial, members in enumerate(members_by_wave):
        for key in members:
            observed.setdefault(key, []).append(serial)

    def zero_totals():
        return {
            "predictions": 0,
            "hit": 0,
            "early": 0,
            "late": 0,
            "missed": 0,
            "unresolved": 0,
        }

    backtests = []
    grand = zero_totals()
    for cutoff in cutoffs:
        periodicities = _reference_periodicities(
            waves[: cutoff + 1], max_jitter
        )
        upper = cutoff + horizon
        outcomes = []
        totals = zero_totals()
        for record in periodicities:
            last = record["last"]
            period = record["period"]
            smallest = min(record["intervals"])
            largest = max(record["intervals"])
            ordinal = 1
            while True:
                wave = last + ordinal * period
                if wave > upper:
                    break
                if wave > cutoff:
                    earliest = last + ordinal * smallest
                    latest = last + ordinal * largest
                    actual = None
                    status = None
                    for serial in observed.get(
                        (record["type"], record["identity"]), ()
                    ):
                        if serial < cutoff + 1 or serial < max(
                            earliest, cutoff + 1
                        ):
                            continue
                        if serial > min(latest, last_wave):
                            break
                        actual = serial
                        if serial == wave:
                            status = "hit"
                        elif serial < wave:
                            status = "early"
                        else:
                            status = "late"
                        break
                    if status is None:
                        status = (
                            "missed"
                            if latest <= last_wave
                            else "unresolved"
                        )
                    outcomes.append(
                        {
                            "type": record["type"],
                            "identity": record["identity"],
                            "ordinal": ordinal,
                            "predicted": wave,
                            "earliest": earliest,
                            "latest": latest,
                            "actual": actual,
                            "status": status,
                        }
                    )
                    totals["predictions"] += 1
                    totals[status] += 1
                ordinal += 1
        backtests.append(
            {
                "cutoff": cutoff,
                "outcomes": tuple(outcomes),
                "totals": totals,
            }
        )
        for key in grand:
            grand[key] += totals[key]
    return {"backtests": tuple(backtests), "totals": grand}


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
            store,
            args,
            windows=windows,
            max_jitter=2,
            horizon=5,
            cutoffs=(2, 3),
        )
        self.assertIsInstance(result, dict)
        self.assertEqual(list(result), ["backtests", "totals"])
        self.assertIsInstance(result["backtests"], tuple)
        self.assertEqual(
            list(result["totals"]),
            ["predictions", "hit", "early", "late", "missed", "unresolved"],
        )
        self.assertEqual(
            [bt["cutoff"] for bt in result["backtests"]], [2, 3]
        )
        for backtest in result["backtests"]:
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
                self.assertIn(
                    outcome["status"],
                    ("hit", "early", "late", "missed", "unresolved"),
                )


class LifetimeChurnForecastBacktestsResultTests(unittest.TestCase):
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
            (
                (0,), (1,), (1,), (2,), (2,), (0,), (0,), (1,),
            ),
            ((0,), (1,), (1,), (2,), (2,), (0,)),
            ((0,), (1,), (2,)),
            ((0, 2), (1,), (1,), (0, 1, 2)),
            (),
        )
        for windows in cases:
            waves = reference_waves(
                self.evolution(windows=windows), 1
            )["waves"]
            last_wave = len(waves) - 1
            cutoff_choices = tuple(range(max(last_wave, 0)))
            for max_jitter in (0, 2):
                for horizon in (1, 3, 6):
                    with self.subTest(
                        windows=windows,
                        max_jitter=max_jitter,
                        horizon=horizon,
                        cutoffs=cutoff_choices,
                    ):
                        self.assertEqual(
                            self.call(
                                windows=windows,
                                max_jitter=max_jitter,
                                horizon=horizon,
                                cutoffs=cutoff_choices,
                                backtest_limit=10_000,
                            ),
                            reference_backtests(
                                self.evolution(windows=windows),
                                1,
                                max_jitter,
                                horizon,
                                cutoff_choices,
                            ),
                        )

    def test_diamond_fixture_scores_hits_at_the_observed_wave(self):
        windows = (
            (0,), (1,), (1,), (2,), (2,), (0,), (0,), (1,),
        )
        result = self.call(
            windows=windows,
            max_jitter=2,
            horizon=5,
            cutoffs=(0, 1, 2, 3),
        )
        by_cutoff = {bt["cutoff"]: bt for bt in result["backtests"]}
        # Cutoffs 0 and 1 train fewer than three appearances, so they
        # produce no periodicity and no prediction.
        for cutoff in (0, 1):
            self.assertEqual(by_cutoff[cutoff]["outcomes"], ())
            self.assertEqual(
                by_cutoff[cutoff]["totals"]["predictions"], 0
            )
        # At cutoff 2 the three steady identities all reappear at
        # predicted wave 3: three hits and twelve still-open ranges.
        self.assertEqual(
            by_cutoff[2]["totals"],
            {
                "predictions": 15,
                "hit": 3,
                "early": 0,
                "late": 0,
                "missed": 0,
                "unresolved": 12,
            },
        )
        hits = [
            outcome
            for outcome in by_cutoff[2]["outcomes"]
            if outcome["status"] == "hit"
        ]
        self.assertEqual(len(hits), 3)
        for outcome in hits:
            self.assertEqual(outcome["ordinal"], 1)
            self.assertEqual(outcome["predicted"], 3)
            self.assertEqual(outcome["actual"], 3)
        # At cutoff 3 every prediction points beyond the last observed
        # wave: all fifteen stay unresolved.
        self.assertEqual(
            by_cutoff[3]["totals"],
            {
                "predictions": 15,
                "hit": 0,
                "early": 0,
                "late": 0,
                "missed": 0,
                "unresolved": 15,
            },
        )
        self.assertEqual(
            result["totals"],
            {
                "predictions": 30,
                "hit": 3,
                "early": 0,
                "late": 0,
                "missed": 0,
                "unresolved": 27,
            },
        )

    def test_outcomes_keep_periodicity_then_ordinal_order(self):
        windows = (
            (0,), (1,), (1,), (2,), (2,), (0,), (0,), (1,),
        )
        # Cutoff 2 trains on exactly the first three observed waves;
        # derive the periodicity order from that same prefix directly.
        prefix_waves = waves_call(
            self.store, self.args, windows=windows
        )["waves"][:3]
        prefix_periodicities = (
            BranchStore._build_lifetime_churn_periodicities_result(
                prefix_waves, 2, None
            )["periodicities"]
        )
        expected_order = [
            (record["type"], record["identity"])
            for record in prefix_periodicities
        ]
        result = self.call(
            windows=windows,
            max_jitter=2,
            horizon=5,
            cutoffs=(2,),
        )
        outcomes = result["backtests"][0]["outcomes"]
        # The identities repeat once per ordinal, keeping the
        # periodicity record order within each ordinal block.
        distinct = []
        for outcome in outcomes:
            key = (outcome["type"], outcome["identity"])
            if key not in distinct:
                distinct.append(key)
        self.assertEqual(distinct, expected_order)
        for key in expected_order:
            ordinal_list = [
                outcome["ordinal"]
                for outcome in outcomes
                if (outcome["type"], outcome["identity"]) == key
            ]
            self.assertEqual(ordinal_list, sorted(ordinal_list))
            self.assertEqual(
                ordinal_list, list(range(1, len(ordinal_list) + 1))
            )

    def test_empty_cutoffs_return_empty_backtests_and_zero_totals(self):
        result = self.call(cutoffs=())
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

    def test_empty_cutoffs_still_run_every_state_check(self):
        with self.assertRaises(KeyError):
            self.call(cutoffs=(), reference="ghost")
        with self.assertRaises(ValueError):
            self.call(cutoffs=(), limit=1)


class LifetimeChurnForecastBacktestsScoringTests(unittest.TestCase):
    """Synthetic member sets pin the five statuses independently."""

    @staticmethod
    def _record(
        identity="A",
        type_name="node",
        ordinal=1,
        predicted=3,
        earliest=2,
        latest=4,
    ):
        return (
            {
                "type": type_name,
                "identity": identity,
                "forecasts": (
                    {
                        "ordinal": ordinal,
                        "wave": predicted,
                        "earliest": earliest,
                        "latest": latest,
                    },
                ),
            },
        )

    @staticmethod
    def _members(appearances, n=6, identity="A", type_name="node"):
        return [
            (
                {(type_name, identity)}
                if serial in appearances
                else {(type_name, f"Z{serial}")}
            )
            for serial in range(n)
        ]

    def test_center_appearance_is_a_hit(self):
        outcome = BranchStore._build_lifetime_churn_backtest_cutoff(
            1, self._record(), self._members({3}), 5
        )["outcomes"][0]
        self.assertEqual(
            (outcome["status"], outcome["actual"]), ("hit", 3)
        )

    def test_first_appearance_before_center_is_early(self):
        outcome = BranchStore._build_lifetime_churn_backtest_cutoff(
            1, self._record(), self._members({2}), 5
        )["outcomes"][0]
        self.assertEqual(
            (outcome["status"], outcome["actual"]), ("early", 2)
        )

    def test_first_appearance_after_center_is_late(self):
        outcome = BranchStore._build_lifetime_churn_backtest_cutoff(
            1, self._record(), self._members({4}), 5
        )["outcomes"][0]
        self.assertEqual(
            (outcome["status"], outcome["actual"]), ("late", 4)
        )

    def test_the_earliest_match_wins_even_when_center_also_appears(self):
        early = BranchStore._build_lifetime_churn_backtest_cutoff(
            1, self._record(), self._members({2, 3}), 5
        )["outcomes"][0]
        self.assertEqual(
            (early["status"], early["actual"]), ("early", 2)
        )
        hit = BranchStore._build_lifetime_churn_backtest_cutoff(
            1, self._record(), self._members({3, 4}), 5
        )["outcomes"][0]
        self.assertEqual((hit["status"], hit["actual"]), ("hit", 3))

    def test_fully_observed_interval_without_match_is_missed(self):
        outcome = BranchStore._build_lifetime_churn_backtest_cutoff(
            1, self._record(), self._members(set()), 5
        )["outcomes"][0]
        self.assertEqual(
            (outcome["status"], outcome["actual"]), ("missed", None)
        )

    def test_open_interval_without_match_is_unresolved(self):
        records = self._record(predicted=4, earliest=4, latest=7)
        unresolved = BranchStore._build_lifetime_churn_backtest_cutoff(
            1, records, self._members(set()), 5
        )["outcomes"][0]
        self.assertEqual(
            (unresolved["status"], unresolved["actual"]),
            ("unresolved", None),
        )
        # A match at the last observed wave inside the interval still
        # resolves the prediction.
        late = BranchStore._build_lifetime_churn_backtest_cutoff(
            1, records, self._members({5}), 5
        )["outcomes"][0]
        self.assertEqual((late["status"], late["actual"]), ("late", 5))

    def test_training_waves_are_never_observations(self):
        # earliest reaches back into wave 1, but wave 1 is part of the
        # training set (cutoff 1); appearing there must not score.
        records = self._record(
            predicted=3, earliest=1, latest=4
        )
        outcome = BranchStore._build_lifetime_churn_backtest_cutoff(
            1, records, self._members({1}), 5
        )["outcomes"][0]
        self.assertEqual(
            (outcome["status"], outcome["actual"]), ("missed", None)
        )

    def test_totals_and_ordering_over_several_predictions(self):
        records = (
            {
                "type": "node",
                "identity": "A",
                "forecasts": (
                    {"ordinal": 1, "wave": 2, "earliest": 2, "latest": 2},
                    {"ordinal": 2, "wave": 3, "earliest": 3, "latest": 3},
                ),
            },
            {
                "type": "node",
                "identity": "B",
                "forecasts": (
                    {"ordinal": 1, "wave": 3, "earliest": 3, "latest": 4},
                ),
            },
        )
        members = [
            {("node", "A")} if serial in (2,) else {("node", "B")}
            if serial in (4,) else set()
            for serial in range(6)
        ]
        cutoff_result = BranchStore._build_lifetime_churn_backtest_cutoff(
            1, records, members, 5
        )
        statuses = [
            (outcome["type"], outcome["identity"], outcome["ordinal"],
             outcome["status"], outcome["actual"])
            for outcome in cutoff_result["outcomes"]
        ]
        self.assertEqual(
            statuses,
            [
                ("node", "A", 1, "hit", 2),
                ("node", "A", 2, "missed", None),
                ("node", "B", 1, "late", 4),
            ],
        )
        self.assertEqual(
            cutoff_result["totals"],
            {
                "predictions": 3,
                "hit": 1,
                "early": 0,
                "late": 1,
                "missed": 1,
                "unresolved": 0,
            },
        )

    def test_outcome_identity_is_an_isolated_copy(self):
        identity = ("left", ("right", "leaf"))
        records = self._record(
            identity=identity,
            type_name="gap",
            predicted=3,
            earliest=3,
            latest=3,
        )
        members = self._members(
            {3}, identity=identity, type_name="gap"
        )
        outcome = BranchStore._build_lifetime_churn_backtest_cutoff(
            1, records, members, 5
        )["outcomes"][0]
        self.assertEqual(outcome["status"], "hit")
        self.assertEqual(outcome["identity"], identity)
        self.assertIsNot(outcome["identity"], identity)


class LifetimeChurnForecastBacktestsValidationTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        overrides.setdefault("causes", ("a0",))
        return backtests_call(self.store, self.args, **overrides)

    def extended(self, **overrides):
        overrides.setdefault(
            "windows",
            ((0,), (1,), (1,), (2,), (2,), (0,), (0,), (1,)),
        )
        overrides.setdefault("horizon", 5)
        overrides.setdefault("max_jitter", 2)
        overrides.setdefault("cutoffs", (2,))
        overrides.setdefault("backtest_limit", 50)
        return self.call(**overrides)

    def test_cutoffs_must_be_a_tuple(self):
        for bad in ([], [0], (x for x in ()), {}, "x", 0, None):
            with self.subTest(cutoffs=bad):
                with self.assertRaises(TypeError):
                    self.call(cutoffs=bad, backtest_limit=1)

    def test_cutoff_elements_must_be_non_bool_ints(self):
        for bad in (True, False, 1.0, "1", None):
            with self.subTest(cutoff=bad):
                with self.assertRaises(TypeError):
                    self.call(cutoffs=(bad,), backtest_limit=1)
        with self.assertRaises(TypeError):
            self.call(cutoffs=(0, bad), backtest_limit=1)

    def test_cutoff_values_must_be_non_negative_ordered_and_unique(self):
        with self.assertRaises(ValueError):
            self.call(cutoffs=(-1,), backtest_limit=1)
        with self.assertRaises(ValueError):
            self.call(cutoffs=(0, -1), backtest_limit=1)
        with self.assertRaises(ValueError):
            self.call(cutoffs=(1, 1), backtest_limit=1)
        with self.assertRaises(ValueError):
            self.call(cutoffs=(2, 1), backtest_limit=1)

    def test_cutoff_range_is_checked_against_the_actual_waves(self):
        # The extended eight-window pattern yields four waves, so
        # serial four is out of range while serial three is the last
        # valid cutoff.
        with self.assertRaises(ValueError) as caught:
            self.extended(cutoffs=(4,), backtest_limit=100)
        self.assertIn("out of range", str(caught.exception))
        self.extended(cutoffs=(3,), backtest_limit=100)

    def test_backtest_limit_must_be_a_non_bool_positive_int(self):
        for bad in (True, False, 1.0, "1", None):
            with self.subTest(backtest_limit=bad):
                with self.assertRaises(TypeError):
                    self.call(cutoffs=(0,), backtest_limit=bad)
        with self.assertRaises(ValueError):
            self.call(cutoffs=(0,), backtest_limit=0)
        with self.assertRaises(ValueError):
            self.call(cutoffs=(0,), backtest_limit=-4)

    def test_new_parameters_validated_in_order_after_forecast_limit(self):
        # forecast_limit fails before cutoffs.
        with self.assertRaises(ValueError):
            self.extended(forecast_limit=0, cutoffs="x")
        with self.assertRaises(TypeError):
            self.extended(forecast_limit="x", cutoffs="x")
        # cutoffs fail before backtest_limit.
        with self.assertRaises(ValueError):
            self.extended(cutoffs=(-1,), backtest_limit=0)
        with self.assertRaises(TypeError):
            self.extended(cutoffs="x", backtest_limit=0)
        with self.assertRaises(TypeError):
            self.extended(cutoffs=(0, "x"), backtest_limit=0)
        # Once cutoffs pass, backtest_limit's error fires.
        with self.assertRaises(TypeError):
            self.extended(cutoffs=(0,), backtest_limit="x")
        with self.assertRaises(ValueError):
            self.extended(cutoffs=(0,), backtest_limit=0)
        # All parameter checks still precede every state lookup.
        with self.assertRaises(ValueError):
            self.call(reference="ghost", cutoffs=(-1,))
        with self.assertRaises(TypeError):
            self.call(reference="ghost", cutoffs="x")
        with self.assertRaises(ValueError):
            self.call(token="whatever", cutoffs=(0,), backtest_limit=0)

    def test_backtest_limit_counts_predictions_over_all_cutoffs(self):
        # Cutoff 2 alone makes fifteen predictions.
        with self.assertRaises(ValueError) as caught:
            self.extended(cutoffs=(2,), backtest_limit=14)
        self.assertIn("backtest limit", str(caught.exception))
        result = self.extended(cutoffs=(2,), backtest_limit=15)
        self.assertEqual(result["totals"]["predictions"], 15)
        # Cutoffs 2 and 3 make thirty; a cap of fifteen passes each
        # cutoff's forecast_limit but trips the cross-cutoff cap.
        with self.assertRaises(ValueError) as caught:
            self.extended(cutoffs=(2, 3), backtest_limit=15)
        self.assertIn("backtest limit", str(caught.exception))

    def test_over_limit_is_checked_after_full_computation(self):
        # The failing call must not truncate: the very next call with a
        # raised cap returns the complete thirty-prediction set.
        with self.assertRaises(ValueError):
            self.extended(cutoffs=(2, 3), backtest_limit=29)
        result = self.extended(cutoffs=(2, 3), backtest_limit=30)
        self.assertEqual(result["totals"]["predictions"], 30)

    def test_forecast_limit_still_bounds_each_cutoff(self):
        with self.assertRaises(ValueError) as caught:
            self.extended(
                cutoffs=(2, 3),
                forecast_limit=14,
                backtest_limit=10_000,
            )
        self.assertIn("forecast limit", str(caught.exception))

    def test_empty_cutoffs_never_trip_the_backtest_cap(self):
        result = self.call(cutoffs=(), backtest_limit=1)
        self.assertEqual(result["backtests"], ())
        self.assertEqual(result["totals"]["predictions"], 0)

    def test_wave_limit_is_still_enforced_ahead_of_scoring(self):
        with self.assertRaises(ValueError) as caught:
            self.extended(wave_limit=1, backtest_limit=10_000)
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
        result = self.extended(
            churn_limit=1,
            streak_limit=1,
            min_waves=99,
            recurrence_limit=1,
            periodicity_limit=1,
            cutoffs=(2,),
            backtest_limit=100,
        )
        self.assertEqual(result["totals"]["predictions"], 15)

    def test_batch_caps_still_fire_ahead_of_scoring(self):
        with self.assertRaises(ValueError) as caught:
            self.call(
                total_diff_limit=5,
                cutoffs=(0,),
                backtest_limit=100,
            )
        self.assertIn("total diff", str(caught.exception))

    def test_shared_validation_errors_match_forecasts(self):
        with self.assertRaises(TypeError):
            self.call(windows="x")
        with self.assertRaises(ValueError):
            self.call(windows=((-1,),))
        with self.assertRaises(ValueError):
            self.call(direction="sideways")


class LifetimeChurnForecastBacktestsStateTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        overrides.setdefault("causes", ("a0",))
        return backtests_call(self.store, self.args, **overrides)

    def test_unknown_branch_history_node_and_cause_raise(self):
        with self.assertRaises(KeyError):
            self.call(reference="ghost", cutoffs=(0,), backtest_limit=1)
        with self.assertRaises(KeyError):
            self.call(
                series={
                    "a": (("r0", "a0"), ("r1", "a0"), ("r2", "a0")),
                    "b": (("r0", "a0"), ("r1", "W"), ("r2", "nope2")),
                },
                cutoffs=(0,),
                backtest_limit=1,
            )
        with self.assertRaises(KeyError) as caught:
            self.call(causes=("zzz",), cutoffs=(0,), backtest_limit=1)
        self.assertEqual(caught.exception.args[0], "zzz")

    def test_range_error_fires_after_the_state_checks(self):
        # The unknown branch must surface KeyError before the actual
        # wave count is known and the cutoff could be range-checked.
        with self.assertRaises(KeyError):
            self.call(
                reference="ghost", cutoffs=(99,), backtest_limit=1
            )


class LifetimeChurnForecastBacktestsIsolationTests(unittest.TestCase):
    EXTENDED = (
        (0,), (1,), (1,), (2,), (2,), (0,), (0,), (1,),
    )

    def test_result_levels_are_mutually_unshared(self):
        store, args = diamond_store()
        result = backtests_call(
            store,
            args,
            windows=self.EXTENDED,
            max_jitter=2,
            horizon=5,
            cutoffs=(2, 3),
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
            elif isinstance(value, list):
                for item in value:
                    collect(item)

        collect(result)
        self.assertEqual(len(seen_ids), len(set(seen_ids)))

    def test_mutating_result_never_affects_later_calls(self):
        store, args = diamond_store()
        first = backtests_call(
            store,
            args,
            windows=self.EXTENDED,
            max_jitter=2,
            horizon=5,
            cutoffs=(2, 3),
        )
        pristine = copy.deepcopy(first)
        for backtest in first["backtests"]:
            backtest["cutoff"] = 99
            for outcome in backtest["outcomes"]:
                outcome["ordinal"] = 99
                outcome["predicted"] = 99
                outcome["actual"] = 99
                outcome["status"] = "changed"
        self.assertEqual(
            backtests_call(
                store,
                args,
                windows=self.EXTENDED,
                max_jitter=2,
                horizon=5,
                cutoffs=(2, 3),
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
            store,
            args,
            windows=self.EXTENDED,
            max_jitter=2,
            horizon=5,
            cutoffs=(2,),
        )
        failures = [
            dict(windows="x"),
            dict(windows=((-1,),)),
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
            dict(cutoffs="x"),
            dict(cutoffs=(True,)),
            dict(cutoffs=(-1,)),
            dict(cutoffs=(1, 1)),
            dict(cutoffs=(2, 1)),
            dict(backtest_limit=0),
            dict(backtest_limit="x"),
            dict(wave_limit=1, cutoffs=(2,)),
            dict(cutoffs=(99,)),
            dict(forecast_limit=1, cutoffs=(2,)),
            dict(cutoffs=(2,), backtest_limit=1),
            dict(causes=("zzz",), cutoffs=(2,)),
            dict(limit=1, cutoffs=(2,)),
            dict(reference="ghost", cutoffs=(2,)),
        ]
        for overrides in failures:
            with self.subTest(overrides=overrides):
                with self.assertRaises(Exception):
                    backtests_call(
                        store,
                        args,
                        causes=("a0",),
                        windows=self.EXTENDED,
                        max_jitter=2,
                        horizon=5,
                        **overrides,
                    )

        self.assertEqual(dict(store._heads), heads)
        self.assertEqual(copy.deepcopy(store._appends), appends)
        self.assertEqual(copy.deepcopy(store._merges), merges)
        self.assertEqual(store.audit_log(), audit)

    def test_existing_public_entries_are_unaffected(self):
        store, args = diamond_store()
        backtests_call(
            store,
            args,
            windows=self.EXTENDED,
            max_jitter=2,
            horizon=5,
            cutoffs=(2,),
        )
        from tests.test_lifetime_churn_forecasts import forecasts_call
        from tests.test_lifetime_churn_periodicities import (
            periodicities_call,
        )

        waves = waves_call(store, args, windows=self.EXTENDED)
        forecasts = forecasts_call(
            store, args, windows=self.EXTENDED, max_jitter=2, horizon=5
        )
        periodicities = periodicities_call(
            store, args, windows=self.EXTENDED, max_jitter=2
        )
        self.assertEqual(
            waves_call(store, args, windows=self.EXTENDED), waves
        )
        self.assertEqual(
            forecasts_call(
                store, args, windows=self.EXTENDED,
                max_jitter=2, horizon=5
            ),
            forecasts,
        )
        self.assertEqual(
            periodicities_call(
                store, args, windows=self.EXTENDED, max_jitter=2
            ),
            periodicities,
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
        token = self.store.create_snapshot(2)
        self.call(
            token=token,
            windows=self.EXTENDED,
            max_jitter=2,
            horizon=5,
            cutoffs=(2, 3),
        )
        # Over the backtest cap: failure refunds.
        with self.assertRaises(ValueError):
            self.call(
                token=token,
                windows=self.EXTENDED,
                max_jitter=2,
                horizon=5,
                cutoffs=(2, 3),
                backtest_limit=1,
            )
        # Out-of-range cutoff: failure refunds.
        with self.assertRaises(ValueError):
            self.call(
                token=token,
                windows=self.EXTENDED,
                cutoffs=(99,),
                backtest_limit=100,
            )
        # State error: failure refunds.
        with self.assertRaises(KeyError):
            self.call(
                token=token,
                causes=("zzz",),
                cutoffs=(2,),
                backtest_limit=100,
            )
        # Parameter validation runs before the token is touched.
        with self.assertRaises(ValueError):
            self.call(token=token, cutoffs=(-1,))
        with self.assertRaises(TypeError):
            self.call(token=token, backtest_limit="x")
        # An empty-cutoff success still performs one read.
        self.call(token=token, cutoffs=())
        with self.assertRaises(RuntimeError):
            self.call(token=token, cutoffs=())

    def test_results_come_from_the_frozen_view(self):
        from tests.test_lifetime_churn_forecasts import forecasts_kwargs

        before = backtests_call(
            self.store,
            self.args,
            windows=self.EXTENDED,
            max_jitter=2,
            horizon=5,
            cutoffs=(2, 3),
        )
        token = self.store.create_snapshot(4)
        self.store.append("a", "a2", 7, {"x": -1})
        self.store.create("d", "root")
        self.store.append("d", "d0", 8, {"z": 3})
        self.store.merge("a", "d", "M", 9, {"z": 1})
        self.store.append("main", "r3", 10, {"x": 1})
        self.assertEqual(
            self.call(
                token=token,
                windows=self.EXTENDED,
                max_jitter=2,
                horizon=5,
                cutoffs=(2, 3),
            ),
            before,
        )


if __name__ == "__main__":
    unittest.main()
