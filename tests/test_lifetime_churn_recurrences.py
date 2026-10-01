import copy
import inspect
import unittest

from city_twin.branches import BranchStore

from tests.test_lifetime_churn import churn_kwargs, diamond_store
from tests.test_lifetime_churn_waves import (
    reference_waves,
    waves_call,
    waves_kwargs,
)


def recurrences_kwargs(store_args, **overrides) -> dict:
    min_waves = overrides.pop("min_waves", 1)
    recurrence_limit = overrides.pop("recurrence_limit", 50)
    kwargs = waves_kwargs(store_args, **overrides)
    kwargs["min_waves"] = min_waves
    kwargs["recurrence_limit"] = recurrence_limit
    return kwargs


def recurrences_call(store, store_args, **overrides):
    return store.lifetime_churn_recurrences(
        **recurrences_kwargs(store_args, **overrides)
    )


def reference_recurrences(
    evolution: dict[str, object],
    min_identities: int,
    min_waves: int,
) -> dict[str, object]:
    """Independently group a lifetime_evolution result into recurrences."""
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
        if len(appearances) < min_waves:
            continue
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
            }
        )

    records.sort(
        key=lambda record: (
            -record["waves"],
            -record["segments"],
            -(record["added"] + record["removed"] + record["changed"]),
            record["first"],
            type_order[record["type"]],
            encounter[(record["type"], record["identity"])],
        )
    )
    return {
        "recurrences": tuple(records),
        "totals": {
            "recurrences": len(records),
            "waves": sum(r["waves"] for r in records),
            "segments": sum(r["segments"] for r in records),
            "added": sum(r["added"] for r in records),
            "removed": sum(r["removed"] for r in records),
            "changed": sum(r["changed"] for r in records),
        },
    }


class LifetimeChurnRecurrencesSignatureTests(unittest.TestCase):
    def test_public_signature_appends_min_waves_and_recurrence_limit(self):
        parameters = list(
            inspect.signature(
                BranchStore.lifetime_churn_recurrences
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
        result = recurrences_call(
            store, args, windows=((0,), (1,), (1,), (2,)), min_waves=2
        )
        self.assertIsInstance(result, dict)
        self.assertEqual(list(result), ["recurrences", "totals"])
        self.assertIsInstance(result["recurrences"], tuple)
        self.assertEqual(
            list(result["totals"]),
            ["recurrences", "waves", "segments",
             "added", "removed", "changed"],
        )
        for record in result["recurrences"]:
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
                ],
            )
            self.assertIsInstance(record["appearances"], tuple)
            for appearance in record["appearances"]:
                self.assertEqual(
                    list(appearance),
                    ["wave", "from", "to", "segments",
                     "added", "removed", "changed"],
                )


class LifetimeChurnRecurrencesResultTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        return recurrences_call(self.store, self.args, **overrides)

    def evolution(self, **overrides):
        kwargs = churn_kwargs(self.args, **overrides)
        kwargs.pop("churn_limit")
        kwargs.pop("streak_limit", None)
        kwargs.pop("min_identities", None)
        kwargs.pop("wave_limit", None)
        kwargs.pop("min_waves", None)
        kwargs.pop("recurrence_limit", None)
        return self.store.lifetime_evolution(**kwargs)

    def test_matches_independent_grouping_of_evolution(self):
        for windows, min_identities, min_waves in (
            (((0,), (1,), (2,)), 1, 1),
            (((0,), (1,), (2,)), 1, 2),
            (((0,), (1,), (1,), (2,)), 1, 1),
            (((0,), (1,), (1,), (2,)), 1, 2),
            (((0,), (1,), (1,), (2,)), 1, 3),
            (((0,), (1,), (1,), (2,)), 2, 1),
            (((0,), (1,), (1,), (2,)), 2, 2),
            (((0, 2), (1,), (1,), (0, 1, 2)), 1, 1),
            (((0, 2), (1,), (1,), (0, 1, 2)), 2, 2),
            (((2,), (0,)), 1, 1),
            (((0, 1), (0, 1), (0, 1)), 1, 1),
            (((0, 1, 2),), 1, 1),
            ((), 1, 1),
        ):
            with self.subTest(windows=windows,
                              min_identities=min_identities,
                              min_waves=min_waves):
                self.assertEqual(
                    self.call(windows=windows,
                              min_identities=min_identities,
                              min_waves=min_waves),
                    reference_recurrences(
                        self.evolution(windows=windows),
                        min_identities,
                        min_waves,
                    ),
                )

    def test_diamond_fixture_two_separated_waves(self):
        result = self.call(
            windows=((0,), (1,), (1,), (2,)), min_waves=2
        )
        self.assertEqual(len(result["recurrences"]), 3)
        by_identity = {
            (r["type"], r["identity"]): r
            for r in result["recurrences"]
        }
        node_w = by_identity[("node", ("W", "y"))]
        self.assertEqual(
            (node_w["waves"], node_w["first"], node_w["last"],
             node_w["span"], node_w["segments"]),
            (2, 0, 1, 2, 2),
        )
        self.assertEqual(
            (node_w["added"], node_w["removed"], node_w["changed"]),
            (1, 0, 1),
        )
        self.assertEqual(
            [a["wave"] for a in node_w["appearances"]], [0, 1]
        )
        first, second = node_w["appearances"]
        self.assertEqual(
            (first["from"], first["to"], first["segments"],
             first["added"], first["removed"], first["changed"]),
            (0, 1, 1, 1, 0, 0),
        )
        self.assertEqual(
            (second["from"], second["to"], second["segments"],
             second["added"], second["removed"], second["changed"]),
            (2, 3, 1, 0, 0, 1),
        )
        # Every record's appearances stay in ascending wave order.
        for record in result["recurrences"]:
            waves = [a["wave"] for a in record["appearances"]]
            self.assertEqual(waves, sorted(waves))
            self.assertEqual(waves[0], record["first"])
            self.assertEqual(waves[-1], record["last"])
        self.assertEqual(
            result["totals"],
            {"recurrences": 3, "waves": 6, "segments": 6,
             "added": 2, "removed": 0, "changed": 4},
        )

    def test_single_wave_with_min_waves_one(self):
        result = self.call(min_waves=1)
        self.assertEqual(len(result["recurrences"]), 3)
        for record in result["recurrences"]:
            self.assertEqual(record["waves"], 1)
            self.assertEqual(
                (record["first"], record["last"], record["span"]),
                (0, 0, 1),
            )
            self.assertEqual(len(record["appearances"]), 1)
            appearance = record["appearances"][0]
            self.assertEqual(appearance["wave"], 0)
            self.assertEqual(
                (appearance["from"], appearance["to"]), (0, 2)
            )
        self.assertEqual(result["totals"]["recurrences"], 3)
        self.assertEqual(result["totals"]["waves"], 3)

    def test_single_wave_with_min_waves_two_is_empty(self):
        result = self.call(min_waves=2)
        self.assertEqual(result["recurrences"], ())
        self.assertEqual(
            result["totals"],
            {"recurrences": 0, "waves": 0, "segments": 0,
             "added": 0, "removed": 0, "changed": 0},
        )

    def test_records_sort_waves_segments_changes_first_type_encounter(self):
        result = self.call(
            windows=((0,), (1,), (1,), (2,)), min_waves=1
        )
        keys = [
            (
                -r["waves"],
                -r["segments"],
                -(r["added"] + r["removed"] + r["changed"]),
                r["first"],
                {"node": 0, "edge": 1, "gap": 2}[r["type"]],
            )
            for r in result["recurrences"]
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
                self.assertEqual(result["recurrences"], ())
                self.assertEqual(
                    result["totals"],
                    {"recurrences": 0, "waves": 0, "segments": 0,
                     "added": 0, "removed": 0, "changed": 0},
                )

    def test_appearances_match_the_wave_members_and_boundaries(self):
        waves = waves_call(
            self.store, self.args,
            windows=((0,), (1,), (1,), (2,)),
        )["waves"]
        result = self.call(
            windows=((0,), (1,), (1,), (2,)), min_waves=1
        )
        for record in result["recurrences"]:
            for appearance in record["appearances"]:
                wave = waves[appearance["wave"]]
                self.assertEqual(
                    (appearance["from"], appearance["to"]),
                    (wave["from"], wave["to"]),
                )
                member = next(
                    member
                    for member in wave["members"]
                    if member["type"] == record["type"]
                    and member["identity"] == record["identity"]
                )
                self.assertEqual(
                    (appearance["segments"], appearance["added"],
                     appearance["removed"], appearance["changed"]),
                    (member["segments"], member["added"],
                     member["removed"], member["changed"]),
                )


class LifetimeChurnRecurrencesGroupingTests(unittest.TestCase):
    @staticmethod
    def _member(kind, identity, segments, added, removed, changed):
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

    def test_filter_span_sums_and_sort_order(self):
        # A recurs across all three waves; B and C across two waves with
        # different segment counts; D appears once and is filtered.
        a = self._member("node", "A", 1, 1, 0, 0)
        a2 = self._member("node", "A", 1, 0, 0, 0)
        a3 = self._member("node", "A", 1, 0, 0, 0)
        b0 = self._member("node", "B", 2, 0, 0, 2)
        b2 = self._member("node", "B", 2, 1, 0, 1)
        c0 = self._member("node", "C", 1, 0, 1, 0)
        c2 = self._member("node", "C", 1, 0, 0, 1)
        d = self._member("edge", "D", 1, 1, 0, 0)
        e = self._member("gap", "E", 1, 0, 1, 0)
        e2 = self._member("gap", "E", 1, 0, 0, 1)
        waves = (
            self._wave(0, (a, b0, c0, d, e)),
            self._wave(1, (a2,)),
            self._wave(2, (a3, b2, c2, e2)),
        )
        result = BranchStore._build_lifetime_churn_recurrences_result(
            waves, 2, 100
        )
        self.assertEqual(
            [r["identity"] for r in result["recurrences"]],
            ["A", "B", "C", "E"],
        )

        record_a = result["recurrences"][0]
        self.assertEqual(
            (record_a["waves"], record_a["first"], record_a["last"],
             record_a["span"]),
            (3, 0, 2, 3),
        )
        self.assertEqual(record_a["segments"], 3)
        self.assertEqual(
            (record_a["added"], record_a["removed"],
             record_a["changed"]),
            (1, 0, 0),
        )
        self.assertEqual(
            [a["wave"] for a in record_a["appearances"]], [0, 1, 2]
        )

        record_b = result["recurrences"][1]
        self.assertEqual(
            (record_b["waves"], record_b["segments"],
             record_b["added"], record_b["changed"]),
            (2, 4, 1, 3),
        )
        self.assertEqual(record_b["first"], 0)
        self.assertEqual(record_b["last"], 2)
        self.assertEqual(record_b["span"], 3)

        record_c = result["recurrences"][2]
        self.assertEqual((record_c["segments"], record_c["removed"]),
                         (2, 1))

        # E (gap) ties with C on every counter and first; the node,
        # edge, gap grouping puts C first.
        record_e = result["recurrences"][3]
        self.assertEqual(record_e["type"], "gap")

        self.assertEqual(
            result["totals"],
            {"recurrences": 4, "waves": 9, "segments": 11,
             "added": 2, "removed": 2, "changed": 5},
        )

    def test_min_waves_three_keeps_only_the_triple_recurrence(self):
        a = self._member("node", "A", 1, 0, 0, 1)
        b = self._member("edge", "B", 1, 1, 0, 0)
        waves = (
            self._wave(0, (a, b)),
            self._wave(1, (a,)),
            self._wave(2, (a, b)),
        )
        result = BranchStore._build_lifetime_churn_recurrences_result(
            waves, 3, 100
        )
        self.assertEqual(
            [r["identity"] for r in result["recurrences"]], ["A"]
        )
        self.assertEqual(
            result["totals"],
            {"recurrences": 1, "waves": 3, "segments": 3,
             "added": 0, "removed": 0, "changed": 3},
        )

    def test_empty_wave_set_yields_zeros(self):
        result = BranchStore._build_lifetime_churn_recurrences_result(
            (), 1, 100
        )
        self.assertEqual(result["recurrences"], ())
        self.assertEqual(
            result["totals"],
            {"recurrences": 0, "waves": 0, "segments": 0,
             "added": 0, "removed": 0, "changed": 0},
        )

    def test_appearance_counters_sum_into_record(self):
        x0 = self._member("node", "X", 2, 1, 2, 3)
        x1 = self._member("node", "X", 1, 4, 5, 6)
        other = self._member("node", "Q", 1, 0, 0, 1)
        # X takes part in wave 0 and wave 2 but skips wave 1.
        waves = (
            self._wave(0, (x0,)),
            self._wave(1, (other,)),
            self._wave(2, (x1,)),
        )
        result = BranchStore._build_lifetime_churn_recurrences_result(
            waves, 2, 100
        )
        record = result["recurrences"][0]
        self.assertEqual(record["segments"], 3)
        self.assertEqual(
            (record["added"], record["removed"], record["changed"]),
            (5, 7, 9),
        )
        self.assertEqual(
            (record["first"], record["last"], record["span"]),
            (0, 2, 3),
        )
        self.assertEqual(
            [a["wave"] for a in record["appearances"]], [0, 2]
        )


class LifetimeChurnRecurrencesLimitValidationTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        overrides.setdefault("causes", ("a0",))
        return recurrences_call(self.store, self.args, **overrides)

    def test_min_waves_must_be_a_non_bool_positive_int(self):
        for bad in (True, False, 1.0, "1", None):
            with self.subTest(min_waves=bad):
                with self.assertRaises(TypeError):
                    self.call(min_waves=bad)
        with self.assertRaises(ValueError):
            self.call(min_waves=0)
        with self.assertRaises(ValueError):
            self.call(min_waves=-4)

    def test_recurrence_limit_must_be_a_non_bool_positive_int(self):
        for bad in (True, False, 1.0, "1", None):
            with self.subTest(recurrence_limit=bad):
                with self.assertRaises(TypeError):
                    self.call(recurrence_limit=bad)
        with self.assertRaises(ValueError):
            self.call(recurrence_limit=0)
        with self.assertRaises(ValueError):
            self.call(recurrence_limit=-4)

    def test_new_limits_validated_in_order_after_wave_limit(self):
        # wave_limit fails before min_waves.
        with self.assertRaises(ValueError):
            self.call(wave_limit=0, min_waves=0)
        with self.assertRaises(TypeError):
            self.call(wave_limit="x", min_waves=0)
        # min_waves fails before recurrence_limit.
        with self.assertRaises(ValueError):
            self.call(min_waves=0, recurrence_limit=0)
        with self.assertRaises(TypeError):
            self.call(min_waves="x", recurrence_limit=0)
        # Once the earlier parameters pass, recurrence_limit's error fires.
        with self.assertRaises(TypeError):
            self.call(min_waves=1, recurrence_limit="x")
        # Both still precede every state lookup and the token lookup.
        with self.assertRaises(ValueError):
            self.call(reference="ghost", min_waves=0)
        with self.assertRaises(ValueError):
            self.call(token="whatever", recurrence_limit=0)

    def test_recurrence_count_beyond_limit_raises_value_error(self):
        # Two separated waves share three identities; a limit of two trips.
        with self.assertRaises(ValueError) as caught:
            self.call(
                windows=((0,), (1,), (1,), (2,)),
                min_waves=2,
                recurrence_limit=2,
            )
        self.assertIn("recurrence limit", str(caught.exception))
        result = self.call(
            windows=((0,), (1,), (1,), (2,)),
            min_waves=2,
            recurrence_limit=3,
        )
        self.assertEqual(len(result["recurrences"]), 3)
        # An empty result never trips the cap.
        result = self.call(min_waves=2, recurrence_limit=1)
        self.assertEqual(result["recurrences"], ())

    def test_wave_limit_is_still_enforced_ahead_of_recurrences(self):
        # Two waves trip the wave cap even though the recurrence query
        # would happily return records.
        with self.assertRaises(ValueError) as caught:
            self.call(
                windows=((0,), (1,), (1,), (2,)),
                wave_limit=1,
                min_waves=10,
            )
        self.assertIn("wave limit", str(caught.exception))

    def test_churn_and_streak_limits_are_validated_but_not_enforced(self):
        with self.assertRaises(ValueError):
            self.call(churn_limit=0)
        with self.assertRaises(ValueError):
            self.call(streak_limit=0)
        with self.assertRaises(TypeError):
            self.call(churn_limit="x")
        with self.assertRaises(TypeError):
            self.call(streak_limit="x")
        # Limits of one would cap churn/streaks queries; recurrences
        # ignore them.
        result = self.call(
            churn_limit=1, streak_limit=1,
            windows=((0,), (1,), (1,), (2,)), min_waves=2,
        )
        self.assertEqual(len(result["recurrences"]), 3)

    def test_batch_caps_still_fire_ahead_of_recurrence_cap(self):
        with self.assertRaises(ValueError) as caught:
            self.call(total_diff_limit=5, recurrence_limit=1)
        self.assertIn("total diff", str(caught.exception))
        # Empty batches never trip the recurrence cap.
        self.assertEqual(
            self.call(windows=(), recurrence_limit=1)["recurrences"],
            (),
        )

    def test_shared_validation_errors_match_waves(self):
        with self.assertRaises(TypeError):
            self.call(windows="x")
        with self.assertRaises(ValueError):
            self.call(windows=((-1,),))
        with self.assertRaises(ValueError):
            self.call(direction="sideways")
        with self.assertRaises(ValueError):
            self.call(window_limit=2, windows=((0,), (1,), (2,)))


class LifetimeChurnRecurrencesStateTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        overrides.setdefault("causes", ("a0",))
        return recurrences_call(self.store, self.args, **overrides)

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


class LifetimeChurnRecurrencesIsolationTests(unittest.TestCase):
    def test_result_levels_are_mutually_unshared(self):
        store, args = diamond_store()
        result = recurrences_call(
            store, args, windows=((0,), (1,), (1,), (2,)), min_waves=2
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
        first = recurrences_call(
            store, args, windows=((0,), (1,), (1,), (2,)), min_waves=2
        )
        pristine = copy.deepcopy(first)
        for record in first["recurrences"]:
            record["appearances"] += (
                {"wave": 9, "from": 9, "to": 10, "segments": 1,
                 "added": 0, "removed": 0, "changed": 0},
            )
            record["waves"] = 99
            for appearance in record["appearances"]:
                appearance["segments"] = 99
        self.assertEqual(
            recurrences_call(
                store, args,
                windows=((0,), (1,), (1,), (2,)), min_waves=2,
            ),
            pristine,
        )

    def test_success_and_failure_change_nothing(self):
        store, args = diamond_store()
        heads = dict(store._heads)
        appends = copy.deepcopy(store._appends)
        merges = copy.deepcopy(store._merges)
        audit = store.audit_log()

        recurrences_call(
            store, args,
            windows=((0,), (1,), (1,), (2,)), min_waves=2,
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
            dict(wave_limit=1, windows=((0,), (1,), (1,), (2,))),
            dict(recurrence_limit=1,
                 windows=((0,), (1,), (1,), (2,)), min_waves=2),
            dict(causes=("zzz",)),
            dict(limit=1),
            dict(reference="ghost"),
        ]
        for overrides in failures:
            with self.subTest(overrides=overrides):
                with self.assertRaises(Exception):
                    recurrences_call(
                        store, args, causes=("a0",), **overrides
                    )

        self.assertEqual(dict(store._heads), heads)
        self.assertEqual(copy.deepcopy(store._appends), appends)
        self.assertEqual(copy.deepcopy(store._merges), merges)
        self.assertEqual(store.audit_log(), audit)

    def test_existing_public_entries_are_unaffected(self):
        store, args = diamond_store()
        recurrences_call(
            store, args, windows=((0,), (1,), (1,), (2,)), min_waves=2
        )
        waves = waves_call(store, args)
        self.assertEqual(waves_call(store, args), waves)


class LifetimeChurnRecurrencesTokenTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, token=None, **overrides):
        return recurrences_call(
            self.store, self.args, token=token, **overrides
        )

    def test_success_consumes_one_read_and_failure_refunds(self):
        token = self.store.create_snapshot(2)
        self.call(
            token=token,
            windows=((0,), (1,), (1,), (2,)), min_waves=2,
        )
        # Over the recurrence cap: failure refunds.
        with self.assertRaises(ValueError):
            self.call(
                token=token,
                windows=((0,), (1,), (1,), (2,)),
                min_waves=2,
                recurrence_limit=1,
            )
        # Over the wave cap inside the shared batch: failure refunds.
        with self.assertRaises(ValueError):
            self.call(
                token=token,
                windows=((0,), (1,), (1,), (2,)),
                wave_limit=1,
            )
        with self.assertRaises(KeyError):
            self.call(token=token, causes=("zzz",))
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
