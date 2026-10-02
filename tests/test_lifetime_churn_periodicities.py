import copy
import inspect
import unittest

from city_twin.branches import BranchStore

from tests.test_lifetime_churn import churn_kwargs, diamond_store
from tests.test_lifetime_churn_waves import reference_waves, waves_kwargs
from tests.test_lifetime_churn_recurrences import recurrences_kwargs


def periodicities_kwargs(store_args, **overrides) -> dict:
    max_jitter = overrides.pop("max_jitter", 0)
    periodicity_limit = overrides.pop("periodicity_limit", 50)
    kwargs = recurrences_kwargs(store_args, **overrides)
    kwargs["max_jitter"] = max_jitter
    kwargs["periodicity_limit"] = periodicity_limit
    return kwargs


def periodicities_call(store, store_args, **overrides):
    return store.lifetime_churn_periodicities(
        **periodicities_kwargs(store_args, **overrides)
    )


def reference_periodicities(
    evolution: dict[str, object],
    min_identities: int,
    max_jitter: int,
) -> dict[str, object]:
    """Independently group a lifetime_evolution result into periods."""
    type_order = {"node": 0, "edge": 1, "gap": 2}
    waves = reference_waves(evolution, min_identities)["waves"]

    encounter: dict[tuple, int] = {}
    next_encounter = 0
    per_identity: dict[tuple, list[dict[str, object]]] = {}
    for serial, wave in enumerate(waves):
        for member in wave["members"]:
            key = (member["type"], member["identity"])
            if key not in encounter:
                encounter[key] = next_encounter
                next_encounter += 1
            per_identity.setdefault(key, []).append(
                {
                    "wave": serial,
                    "from": wave["from"],
                    "to": wave["to"],
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
        intervals = tuple(
            later["wave"] - earlier["wave"]
            for earlier, later in zip(appearances, appearances[1:])
        )
        jitter = max(intervals) - min(intervals)
        if jitter > max_jitter:
            continue
        ordered = sorted(intervals)
        middle = len(ordered) // 2
        if len(ordered) % 2 == 0:
            period = ordered[middle - 1]
        else:
            period = ordered[middle]
        records.append(
            {
                "type": type_name,
                "identity": identity,
                "waves": len(appearances),
                "first": appearances[0]["wave"],
                "last": appearances[-1]["wave"],
                "span": appearances[-1]["wave"] - appearances[0]["wave"] + 1,
                "segments": sum(a["segments"] for a in appearances),
                "added": sum(a["added"] for a in appearances),
                "removed": sum(a["removed"] for a in appearances),
                "changed": sum(a["changed"] for a in appearances),
                "appearances": tuple(appearances),
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
            -(record["added"] + record["removed"] + record["changed"]),
            record["first"],
            type_order[record["type"]],
            encounter[(record["type"], record["identity"])],
        )
    )
    return {
        "periodicities": tuple(records),
        "totals": {
            "periodicities": len(records),
            "waves": sum(r["waves"] for r in records),
            "segments": sum(r["segments"] for r in records),
            "added": sum(r["added"] for r in records),
            "removed": sum(r["removed"] for r in records),
            "changed": sum(r["changed"] for r in records),
        },
    }


class LifetimeChurnPeriodicitiesSignatureTests(unittest.TestCase):
    def test_public_signature_appends_max_jitter_and_periodicity_limit(self):
        parameters = list(
            inspect.signature(
                BranchStore.lifetime_churn_periodicities
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
        result = periodicities_call(
            store,
            args,
            windows=((0,), (1,), (1,), (2,), (2,), (0,)),
        )
        self.assertIsInstance(result, dict)
        self.assertEqual(list(result), ["periodicities", "totals"])
        self.assertIsInstance(result["periodicities"], tuple)
        self.assertEqual(
            list(result["totals"]),
            ["periodicities", "waves", "segments",
             "added", "removed", "changed"],
        )
        for record in result["periodicities"]:
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
                ],
            )
            self.assertIsInstance(record["appearances"], tuple)
            self.assertIsInstance(record["intervals"], tuple)
            for appearance in record["appearances"]:
                self.assertEqual(
                    list(appearance),
                    ["wave", "from", "to", "segments",
                     "added", "removed", "changed"],
                )


class LifetimeChurnPeriodicitiesResultTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        return periodicities_call(self.store, self.args, **overrides)

    def evolution(self, **overrides):
        kwargs = churn_kwargs(self.args, **overrides)
        kwargs.pop("churn_limit")
        kwargs.pop("streak_limit", None)
        kwargs.pop("min_identities", None)
        kwargs.pop("wave_limit", None)
        kwargs.pop("min_waves", None)
        kwargs.pop("recurrence_limit", None)
        kwargs.pop("max_jitter", None)
        kwargs.pop("periodicity_limit", None)
        return self.store.lifetime_evolution(**kwargs)

    def test_matches_independent_grouping_of_evolution(self):
        cases = (
            (((0,), (1,), (2,)), 1, 0),
            (((0,), (1,), (2,)), 1, 2),
            (((0,), (1,), (1,), (2,)), 1, 0),
            (((0,), (1,), (1,), (2,)), 1, 3),
            (((0,), (1,), (1,), (2,), (2,), (0,)), 1, 0),
            (((0,), (1,), (1,), (2,), (2,), (0,)), 1, 1),
            (((0,), (1,), (1,), (2,), (2,), (0,)), 1, 5),
            (((0,), (1,), (1,), (2,), (2,), (0,)), 2, 0),
            (
                ((0,), (1,), (1,), (2,), (2,), (0,), (0,), (1,)),
                1,
                0,
            ),
            (
                ((0,), (1,), (1,), (2,), (2,), (0,), (0,), (1,)),
                1,
                2,
            ),
            (((0, 2), (1,), (1,), (0, 1, 2)), 1, 1),
            (((2,), (0,)), 1, 1),
            (((0, 1), (0, 1), (0, 1)), 1, 1),
            (((0, 1, 2),), 1, 1),
            ((), 1, 1),
        )
        for windows, min_identities, max_jitter in cases:
            with self.subTest(windows=windows,
                              min_identities=min_identities,
                              max_jitter=max_jitter):
                self.assertEqual(
                    self.call(windows=windows,
                              min_identities=min_identities,
                              max_jitter=max_jitter),
                    reference_periodicities(
                        self.evolution(windows=windows),
                        min_identities,
                        max_jitter,
                    ),
                )

    def test_diamond_fixture_three_adjacent_waves(self):
        result = self.call(
            windows=((0,), (1,), (1,), (2,), (2,), (0,)),
            max_jitter=0,
        )
        self.assertEqual(len(result["periodicities"]), 3)
        by_identity = {
            (r["type"], r["identity"]): r
            for r in result["periodicities"]
        }
        node_w = by_identity[("node", ("W", "y"))]
        self.assertEqual(
            (node_w["waves"], node_w["first"], node_w["last"],
             node_w["span"], node_w["segments"]),
            (3, 0, 2, 3, 3),
        )
        self.assertEqual(
            (node_w["added"], node_w["removed"], node_w["changed"]),
            (1, 1, 1),
        )
        self.assertEqual(node_w["intervals"], (1, 1))
        self.assertEqual(node_w["period"], 1)
        self.assertEqual(node_w["jitter"], 0)
        self.assertEqual(
            [a["wave"] for a in node_w["appearances"]], [0, 1, 2]
        )
        for record in result["periodicities"]:
            waves = [a["wave"] for a in record["appearances"]]
            self.assertEqual(waves, sorted(waves))
            self.assertEqual(waves[0], record["first"])
            self.assertEqual(waves[-1], record["last"])
            self.assertEqual(
                record["intervals"],
                tuple(b - a for a, b in zip(waves, waves[1:])),
            )
        self.assertEqual(
            result["totals"],
            {"periodicities": 3, "waves": 9, "segments": 9,
             "added": 2, "removed": 2, "changed": 5},
        )

    def test_fewer_than_three_waves_returns_empty(self):
        # Two separated waves give recurrences but never periodicities.
        result = self.call(
            windows=((0,), (1,), (1,), (2,)),
            min_waves=2,
            max_jitter=100,
        )
        self.assertEqual(result["periodicities"], ())
        self.assertEqual(
            result["totals"],
            {"periodicities": 0, "waves": 0, "segments": 0,
             "added": 0, "removed": 0, "changed": 0},
        )

    def test_records_sort_jitter_waves_segments_changes_first(self):
        result = self.call(
            windows=(
                (0,), (1,), (1,), (2,), (2,), (0,), (0,), (1,),
            ),
            max_jitter=2,
        )
        keys = [
            (
                r["jitter"],
                -r["waves"],
                -r["segments"],
                -(r["added"] + r["removed"] + r["changed"]),
                r["first"],
                {"node": 0, "edge": 1, "gap": 2}[r["type"]],
            )
            for r in result["periodicities"]
        ]
        self.assertEqual(keys, sorted(keys))

    def test_empty_windows_causes_and_single_window_return_zeros(self):
        for overrides in (
            {"windows": ()},
            {"causes": ()},
            {"windows": ((0, 1, 2),)},
        ):
            with self.subTest(overrides=overrides):
                result = self.call(**overrides)
                self.assertEqual(result["periodicities"], ())
                self.assertEqual(
                    result["totals"],
                    {"periodicities": 0, "waves": 0, "segments": 0,
                     "added": 0, "removed": 0, "changed": 0},
                )

    def test_zero_jitter_keeps_only_exact_cadence(self):
        # The store's extended fixture is perfectly cadenced, so the
        # strictest jitter still keeps every three-wave identity.
        strict = self.call(
            windows=((0,), (1,), (1,), (2,), (2,), (0,)),
            max_jitter=0,
        )
        relaxed = self.call(
            windows=((0,), (1,), (1,), (2,), (2,), (0,)),
            max_jitter=5,
        )
        self.assertEqual(strict, relaxed)


class LifetimeChurnPeriodicitiesGroupingTests(unittest.TestCase):
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

    def test_three_equal_intervals_kept_with_median_period(self):
        waves = (
            self._wave(0, (self._member("node", "A", changed=1),)),
            self._wave(1, (self._member("node", "A"),)),
            self._wave(2, (self._member("node", "A"),)),
        )
        result = BranchStore._build_lifetime_churn_periodicities_result(
            waves, 0, 100
        )
        self.assertEqual(len(result["periodicities"]), 1)
        record = result["periodicities"][0]
        self.assertEqual(record["identity"], "A")
        self.assertEqual(record["intervals"], (1, 1))
        self.assertEqual(record["period"], 1)
        self.assertEqual(record["jitter"], 0)
        self.assertEqual(
            (record["waves"], record["first"], record["last"],
             record["span"]),
            (3, 0, 2, 3),
        )

    def test_two_appearances_never_qualify(self):
        waves = (
            self._wave(0, (self._member("node", "A"),)),
            self._wave(2, (self._member("node", "A"),)),
        )
        for max_jitter in (0, 10):
            with self.subTest(max_jitter=max_jitter):
                result = (
                    BranchStore
                    ._build_lifetime_churn_periodicities_result(
                        waves, max_jitter, 100
                    )
                )
                self.assertEqual(result["periodicities"], ())
                self.assertEqual(result["totals"]["periodicities"], 0)

    def test_jitter_threshold_filters_and_even_median_takes_lower(self):
        # A at waves 0, 2, 3 -> intervals (2, 1), sorted (1, 2): even
        # count, period is the smaller middle value, jitter is one.
        # Wave 1 exists with an unrelated member only.
        filler = self._member("node", "Z")
        waves = (
            self._wave(0, (self._member("node", "A"), filler)),
            self._wave(1, (filler,)),
            self._wave(2, (self._member("node", "A"),)),
            self._wave(3, (self._member("node", "A"),)),
        )
        rejected = BranchStore._build_lifetime_churn_periodicities_result(
            waves, 0, 100
        )
        self.assertEqual(rejected["periodicities"], ())
        accepted = BranchStore._build_lifetime_churn_periodicities_result(
            waves, 1, 100
        )
        record = accepted["periodicities"][0]
        self.assertEqual(record["intervals"], (2, 1))
        self.assertEqual(record["period"], 1)
        self.assertEqual(record["jitter"], 1)

    def test_even_interval_count_uses_lower_middle(self):
        # Five appearances -> four intervals 2,3,1,4; sorted
        # (1, 2, 3, 4): period 2, jitter 3. The non-A waves each carry a
        # one-off filler, so no second identity can become periodic.
        waves = tuple(
            self._wave(
                serial,
                (self._member("node", "A"),)
                if serial in (0, 2, 5, 6, 10)
                else (self._member("node", f"Z{serial}"),),
            )
            for serial in range(11)
        )
        rejected = BranchStore._build_lifetime_churn_periodicities_result(
            waves, 2, 100
        )
        self.assertEqual(rejected["periodicities"], ())
        accepted = BranchStore._build_lifetime_churn_periodicities_result(
            waves, 3, 100
        )
        record = accepted["periodicities"][0]
        self.assertEqual(record["intervals"], (2, 3, 1, 4))
        self.assertEqual(record["period"], 2)
        self.assertEqual(record["jitter"], 3)
        self.assertEqual(record["waves"], 5)

    def test_repeated_segments_in_one_wave_count_once(self):
        # Churning on two segments inside wave 0 is still one
        # appearance; three waves total give intervals (1, 1).
        heavy = self._member("node", "A", segments=2, changed=2)
        waves = (
            self._wave(0, (heavy,)),
            self._wave(1, (self._member("node", "A", segments=1),)),
            self._wave(2, (self._member("node", "A", segments=1),)),
        )
        result = BranchStore._build_lifetime_churn_periodicities_result(
            waves, 0, 100
        )
        record = result["periodicities"][0]
        self.assertEqual(record["waves"], 3)
        self.assertEqual(record["intervals"], (1, 1))
        self.assertEqual(record["segments"], 4)

    def test_sort_jitter_then_waves_then_first_then_type_encounter(self):
        # P: waves 0,2,4 -> jitter 0, waves 3
        # R: waves 0,1,2,4 -> intervals (1,1,2) jitter 1, waves 4
        # Q: waves 0,1,3 -> intervals (1,2) jitter 1, waves 3, node
        # T: waves 0,2,3 -> jitter 1, waves 3, edge (ties Q on every
        # numeric key and first; the node, edge, gap grouping wins)
        # S: waves 1,3,4 -> jitter 1, waves 3, first 1
        p = self._member("node", "P")
        r = self._member("node", "R")
        q = self._member("node", "Q")
        t = self._member("edge", "T")
        s = self._member("gap", "S")
        waves = (
            self._wave(0, (p, r, q, t)),
            self._wave(1, (r, q, s)),
            self._wave(2, (p, r, t)),
            self._wave(3, (q, t, s)),
            self._wave(4, (p, r, s)),
        )
        result = BranchStore._build_lifetime_churn_periodicities_result(
            waves, 2, 100
        )
        self.assertEqual(
            [rec["identity"] for rec in result["periodicities"]],
            ["P", "R", "Q", "T", "S"],
        )
        jitters = [rec["jitter"] for rec in result["periodicities"]]
        self.assertEqual(jitters, sorted(jitters))

    def test_empty_wave_set_yields_zeros(self):
        result = BranchStore._build_lifetime_churn_periodicities_result(
            (), 0, 100
        )
        self.assertEqual(result["periodicities"], ())
        self.assertEqual(
            result["totals"],
            {"periodicities": 0, "waves": 0, "segments": 0,
             "added": 0, "removed": 0, "changed": 0},
        )

    def test_totals_sum_only_the_final_records(self):
        # Wave serials are tuple positions; position 1 carries only an
        # unrelated filler and so opens a gap in A's and B's cadence.
        #   A at 0(added 1), 2(changed 2), 3, 4 -> intervals
        #   (2, 1, 1), jitter 1, period 1;
        #   B at 0(removed 1), 3, 5 -> intervals (3, 2), jitter 1,
        #   period 2 (lower middle of the sorted pair).
        filler = self._member("gap", "Z")
        waves = (
            self._wave(0, (
                self._member("node", "A", added=1),
                self._member("edge", "B", removed=1),
            )),
            self._wave(1, (filler,)),
            self._wave(2, (self._member("node", "A", changed=2),)),
            self._wave(3, (
                self._member("node", "A"),
                self._member("edge", "B"),
            )),
            self._wave(4, (self._member("node", "A"),)),
            self._wave(5, (self._member("edge", "B"),)),
        )
        result = BranchStore._build_lifetime_churn_periodicities_result(
            waves, 1, 100
        )
        by_identity = {
            rec["identity"]: rec for rec in result["periodicities"]
        }
        self.assertEqual(set(by_identity), {"A", "B"})
        record_a = by_identity["A"]
        self.assertEqual(
            (record_a["waves"], record_a["segments"]), (4, 4)
        )
        self.assertEqual(record_a["intervals"], (2, 1, 1))
        self.assertEqual(record_a["period"], 1)
        self.assertEqual(
            (record_a["added"], record_a["changed"]), (1, 2)
        )
        record_b = by_identity["B"]
        self.assertEqual(record_b["intervals"], (3, 2))
        self.assertEqual(record_b["period"], 2)
        self.assertEqual(record_b["removed"], 1)
        self.assertEqual(
            result["totals"],
            {
                "periodicities": 2,
                "waves": 7,
                "segments": 7,
                "added": 1,
                "removed": 1,
                "changed": 2,
            },
        )

    def test_periodicity_limit_raises_after_full_set_built(self):
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
        with self.assertRaises(ValueError) as caught:
            BranchStore._build_lifetime_churn_periodicities_result(
                waves, 0, 1
            )
        self.assertIn("periodicity limit", str(caught.exception))


class LifetimeChurnPeriodicitiesLimitValidationTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        overrides.setdefault("causes", ("a0",))
        return periodicities_call(self.store, self.args, **overrides)

    def extended(self, **overrides):
        overrides.setdefault(
            "windows", ((0,), (1,), (1,), (2,), (2,), (0,))
        )
        return self.call(**overrides)

    def test_max_jitter_must_be_a_non_bool_non_negative_int(self):
        for bad in (True, False, 1.0, "0", None):
            with self.subTest(max_jitter=bad):
                with self.assertRaises(TypeError):
                    self.call(max_jitter=bad)
        with self.assertRaises(ValueError):
            self.call(max_jitter=-1)
        with self.assertRaises(ValueError):
            self.call(max_jitter=-4)
        # Zero is the boundary that must be accepted.
        self.extended(max_jitter=0)

    def test_periodicity_limit_must_be_a_non_bool_positive_int(self):
        for bad in (True, False, 1.0, "1", None):
            with self.subTest(periodicity_limit=bad):
                with self.assertRaises(TypeError):
                    self.call(periodicity_limit=bad)
        with self.assertRaises(ValueError):
            self.call(periodicity_limit=0)
        with self.assertRaises(ValueError):
            self.call(periodicity_limit=-4)

    def test_new_limits_validated_in_order_after_recurrence_limit(self):
        # recurrence_limit fails before max_jitter.
        with self.assertRaises(ValueError):
            self.extended(recurrence_limit=0, max_jitter=-1)
        with self.assertRaises(TypeError):
            self.extended(recurrence_limit="x", max_jitter=-1)
        # max_jitter fails before periodicity_limit.
        with self.assertRaises(ValueError):
            self.extended(max_jitter=-1, periodicity_limit=0)
        with self.assertRaises(TypeError):
            self.extended(max_jitter="x", periodicity_limit=0)
        # Once max_jitter passes, periodicity_limit's error fires.
        with self.assertRaises(TypeError):
            self.extended(max_jitter=0, periodicity_limit="x")
        with self.assertRaises(ValueError):
            self.extended(max_jitter=0, periodicity_limit=0)
        # Both still precede every state lookup and the token lookup.
        with self.assertRaises(ValueError):
            self.call(reference="ghost", max_jitter=-1)
        with self.assertRaises(ValueError):
            self.call(token="whatever", periodicity_limit=0)
        with self.assertRaises(TypeError):
            self.call(reference="ghost", max_jitter="x")

    def test_periodicity_count_beyond_limit_raises_value_error(self):
        with self.assertRaises(ValueError) as caught:
            self.extended(periodicity_limit=2)
        self.assertIn("periodicity limit", str(caught.exception))
        result = self.extended(periodicity_limit=3)
        self.assertEqual(len(result["periodicities"]), 3)
        # An empty result never trips the cap.
        result = self.call(periodicity_limit=1)
        self.assertEqual(result["periodicities"], ())

    def test_wave_limit_is_still_enforced_ahead_of_periodicities(self):
        with self.assertRaises(ValueError) as caught:
            self.extended(wave_limit=1, periodicity_limit=10)
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
        # Caps of one would bound the earlier queries; periodicities
        # ignores them and still returns all three steady identities.
        result = self.extended(
            churn_limit=1,
            streak_limit=1,
            min_waves=99,
            recurrence_limit=1,
        )
        self.assertEqual(len(result["periodicities"]), 3)

    def test_batch_caps_still_fire_ahead_of_periodicity_cap(self):
        with self.assertRaises(ValueError) as caught:
            self.call(total_diff_limit=5, periodicity_limit=1)
        self.assertIn("total diff", str(caught.exception))
        # Empty batches never trip the periodicity cap.
        self.assertEqual(
            self.call(windows=(), periodicity_limit=1)["periodicities"],
            (),
        )

    def test_shared_validation_errors_match_recurrences(self):
        with self.assertRaises(TypeError):
            self.call(windows="x")
        with self.assertRaises(ValueError):
            self.call(windows=((-1,),))
        with self.assertRaises(ValueError):
            self.call(direction="sideways")
        with self.assertRaises(ValueError):
            self.call(window_limit=2, windows=((0,), (1,), (2,)))


class LifetimeChurnPeriodicitiesStateTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        overrides.setdefault("causes", ("a0",))
        return periodicities_call(self.store, self.args, **overrides)

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


class LifetimeChurnPeriodicitiesIsolationTests(unittest.TestCase):
    EXTENDED = ((0,), (1,), (1,), (2,), (2,), (0,))

    def test_result_levels_are_mutually_unshared(self):
        store, args = diamond_store()
        result = periodicities_call(
            store, args, windows=self.EXTENDED
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
        first = periodicities_call(store, args, windows=self.EXTENDED)
        pristine = copy.deepcopy(first)
        for record in first["periodicities"]:
            record["intervals"] += (99,)
            record["period"] = 99
            record["jitter"] = 99
            record["appearances"] += (
                {"wave": 9, "from": 9, "to": 10, "segments": 1,
                 "added": 0, "removed": 0, "changed": 0},
            )
            record["waves"] = 99
            for appearance in record["appearances"]:
                appearance["segments"] = 99
        self.assertEqual(
            periodicities_call(store, args, windows=self.EXTENDED),
            pristine,
        )

    def test_success_and_failure_change_nothing(self):
        store, args = diamond_store()
        heads = dict(store._heads)
        appends = copy.deepcopy(store._appends)
        merges = copy.deepcopy(store._merges)
        audit = store.audit_log()

        periodicities_call(store, args, windows=self.EXTENDED)
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
            dict(wave_limit=1, windows=self.EXTENDED),
            dict(periodicity_limit=1, windows=self.EXTENDED),
            dict(causes=("zzz",)),
            dict(limit=1),
            dict(reference="ghost"),
        ]
        for overrides in failures:
            with self.subTest(overrides=overrides):
                with self.assertRaises(Exception):
                    periodicities_call(
                        store, args, causes=("a0",), **overrides
                    )

        self.assertEqual(dict(store._heads), heads)
        self.assertEqual(copy.deepcopy(store._appends), appends)
        self.assertEqual(copy.deepcopy(store._merges), merges)
        self.assertEqual(store.audit_log(), audit)

    def test_existing_public_entries_are_unaffected(self):
        store, args = diamond_store()
        periodicities_call(store, args, windows=self.EXTENDED)
        from tests.test_lifetime_churn_recurrences import recurrences_call
        from tests.test_lifetime_churn_waves import waves_call

        waves = waves_call(store, args)
        recurrences = recurrences_call(
            store, args,
            windows=((0,), (1,), (1,), (2,)), min_waves=2,
        )
        self.assertEqual(waves_call(store, args), waves)
        self.assertEqual(
            recurrences_call(
                store, args,
                windows=((0,), (1,), (1,), (2,)), min_waves=2,
            ),
            recurrences,
        )


class LifetimeChurnPeriodicitiesTokenTests(unittest.TestCase):
    EXTENDED = ((0,), (1,), (1,), (2,), (2,), (0,))

    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, token=None, **overrides):
        return periodicities_call(
            self.store, self.args, token=token, **overrides
        )

    def test_success_consumes_one_read_and_failure_refunds(self):
        token = self.store.create_snapshot(2)
        self.call(token=token, windows=self.EXTENDED)
        # Over the periodicity cap: failure refunds.
        with self.assertRaises(ValueError):
            self.call(
                token=token,
                windows=self.EXTENDED,
                periodicity_limit=1,
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
            self.call(token=token, periodicity_limit=0)
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
