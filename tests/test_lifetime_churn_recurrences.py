import copy
import inspect
import unittest

from city_twin.branches import BranchStore

from tests.test_lifetime_churn import churn_kwargs, diamond_store
from tests.test_lifetime_churn_waves import waves_call, waves_kwargs


def recurrences_kwargs(store_args, **overrides) -> dict:
    kwargs = waves_kwargs(store_args, **overrides)
    kwargs["min_waves"] = overrides.pop("min_waves", 1)
    kwargs["recurrence_limit"] = overrides.pop("recurrence_limit", 50)
    return kwargs


def recurrences_call(store, store_args, **overrides):
    return store.lifetime_churn_recurrences(
        **recurrences_kwargs(store_args, **overrides)
    )


def reference_recurrences(
    waves: dict[str, object], min_waves: int
) -> dict[str, object]:
    """Independently aggregate a lifetime_churn_waves result by identity."""
    type_order = {"node": 0, "edge": 1, "gap": 2}
    appearances = {}
    encounter = {}
    next_encounter = 0
    for wave_number, wave in enumerate(waves["waves"]):
        for member in wave["members"]:
            key = (member["type"], member["identity"])
            if key not in encounter:
                encounter[key] = next_encounter
                next_encounter += 1
                appearances[key] = []
            appearances[key].append(
                {
                    "wave": wave_number,
                    "from": wave["from"],
                    "to": wave["to"],
                    "segments": member["segments"],
                    "added": member["added"],
                    "removed": member["removed"],
                    "changed": member["changed"],
                }
            )

    qualifying = [
        key for key, entries in appearances.items()
        if len(entries) >= min_waves
    ]
    qualifying.sort(
        key=lambda key: (
            -len(appearances[key]),
            -sum(e["segments"] for e in appearances[key]),
            -sum(
                e["added"] + e["removed"] + e["changed"]
                for e in appearances[key]
            ),
            appearances[key][0]["wave"],
            type_order[key[0]],
            encounter[key],
        )
    )

    records = []
    total_waves = 0
    total_segments = 0
    total_added = 0
    total_removed = 0
    total_changed = 0
    for key in qualifying:
        entries = appearances[key]
        segments = sum(e["segments"] for e in entries)
        added = sum(e["added"] for e in entries)
        removed = sum(e["removed"] for e in entries)
        changed = sum(e["changed"] for e in entries)
        first = entries[0]["wave"]
        last = entries[-1]["wave"]
        total_waves += len(entries)
        total_segments += segments
        total_added += added
        total_removed += removed
        total_changed += changed
        records.append(
            {
                "type": key[0],
                "identity": key[1],
                "waves": len(entries),
                "first": first,
                "last": last,
                "span": last - first + 1,
                "segments": segments,
                "added": added,
                "removed": removed,
                "changed": changed,
                "appearances": tuple(dict(e) for e in entries),
            }
        )
    return {
        "recurrences": tuple(records),
        "totals": {
            "recurrences": len(records),
            "waves": total_waves,
            "segments": total_segments,
            "added": total_added,
            "removed": total_removed,
            "changed": total_changed,
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
            store, args, windows=((0,), (1,), (1,), (2,))
        )
        self.assertIsInstance(result, dict)
        self.assertEqual(list(result), ["recurrences", "totals"])
        self.assertIsInstance(result["recurrences"], tuple)
        self.assertEqual(
            list(result["totals"]),
            ["recurrences", "waves", "segments", "added", "removed",
             "changed"],
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
            self.assertEqual(
                record["span"], record["last"] - record["first"] + 1
            )
            for appearance in record["appearances"]:
                self.assertEqual(
                    list(appearance),
                    ["wave", "from", "to", "segments", "added", "removed",
                     "changed"],
                )


class LifetimeChurnRecurrencesResultTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        return recurrences_call(self.store, self.args, **overrides)

    def waves(self, **overrides):
        kwargs = waves_kwargs(self.args, **overrides)
        return self.store.lifetime_churn_waves(**kwargs)

    def test_matches_independent_grouping_of_waves(self):
        for windows, min_identities, min_waves in (
            (((0,), (1,), (1,), (2,)), 1, 1),
            (((0,), (1,), (1,), (2,)), 1, 2),
            (((0,), (1,), (1,), (2,)), 1, 3),
            (((0,), (1,), (2,)), 1, 1),
            (((0,), (1,), (2,)), 1, 2),
            (((0,), (1,), (1,), (2,)), 2, 1),
            (((0,), (1,), (1,), (2,)), 2, 2),
            (((0, 2), (1,), (0, 1, 2)), 1, 2),
            ((), 1, 1),
            (((0, 1, 2),), 1, 1),
        ):
            with self.subTest(windows=windows,
                              min_identities=min_identities,
                              min_waves=min_waves):
                waves = self.waves(
                    windows=windows, min_identities=min_identities
                )
                self.assertEqual(
                    self.call(
                        windows=windows,
                        min_identities=min_identities,
                        min_waves=min_waves,
                    ),
                    reference_recurrences(waves, min_waves),
                )

    def test_two_waves_every_member_recurs(self):
        result = self.call(windows=((0,), (1,), (1,), (2,)))
        self.assertEqual(len(result["recurrences"]), 3)
        for record in result["recurrences"]:
            self.assertEqual(record["waves"], 2)
            self.assertEqual(
                (record["first"], record["last"], record["span"]),
                (0, 1, 2),
            )
            self.assertEqual(len(record["appearances"]), 2)
            self.assertEqual(
                [a["wave"] for a in record["appearances"]], [0, 1]
            )
            self.assertEqual(
                (
                    record["appearances"][0]["from"],
                    record["appearances"][0]["to"],
                ),
                (0, 1),
            )
            self.assertEqual(
                (
                    record["appearances"][1]["from"],
                    record["appearances"][1]["to"],
                ),
                (2, 3),
            )
        self.assertEqual(
            result["totals"],
            {"recurrences": 3, "waves": 6, "segments": 6, "added": 2,
             "removed": 0, "changed": 4},
        )

    def test_min_waves_filters_and_single_wave_yields_none_at_two(self):
        # The default diamond batch is one wave: min_waves=2 keeps no one.
        result = self.call(min_waves=2)
        self.assertEqual(result["recurrences"], ())
        self.assertEqual(
            result["totals"],
            {"recurrences": 0, "waves": 0, "segments": 0, "added": 0,
             "removed": 0, "changed": 0},
        )
        # min_waves=1 keeps every member of that wave.
        result = self.call(min_waves=1)
        self.assertEqual(len(result["recurrences"]), 3)
        self.assertEqual(result["totals"]["recurrences"], 3)
        self.assertEqual(result["totals"]["waves"], 3)

    def test_empty_windows_causes_and_quiet_batch_return_zeros(self):
        for overrides in (
            {"windows": ()},
            {"causes": ()},
            {"windows": ((0, 1, 2),)},
            {"min_identities": 99},
        ):
            with self.subTest(overrides=overrides):
                result = self.call(**overrides)
                self.assertEqual(result["recurrences"], ())
                self.assertEqual(
                    result["totals"],
                    {"recurrences": 0, "waves": 0, "segments": 0,
                     "added": 0, "removed": 0, "changed": 0},
                )

    def test_records_sort_by_waves_segments_changes_first_type(self):
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

    def test_appearances_share_boundaries_with_the_waves_result(self):
        waves = self.waves(windows=((0,), (1,), (1,), (2,)))
        result = self.call(windows=((0,), (1,), (1,), (2,)))
        for record in result["recurrences"]:
            for appearance in record["appearances"]:
                wave = waves["waves"][appearance["wave"]]
                self.assertEqual(appearance["from"], wave["from"])
                self.assertEqual(appearance["to"], wave["to"])
                member = next(
                    member
                    for member in wave["members"]
                    if member["type"] == record["type"]
                    and member["identity"] == record["identity"]
                )
                self.assertEqual(
                    appearance["segments"], member["segments"]
                )
                self.assertEqual(appearance["added"], member["added"])
                self.assertEqual(appearance["removed"], member["removed"])
                self.assertEqual(appearance["changed"], member["changed"])


class LifetimeChurnRecurrencesGroupingTests(unittest.TestCase):
    @staticmethod
    def _member(type_name, identity, segments, added=0, removed=0,
                changed=0):
        return {
            "type": type_name,
            "identity": identity,
            "segments": segments,
            "added": added,
            "removed": removed,
            "changed": changed,
        }

    def test_builder_aggregates_across_waves_and_filters(self):
        waves = (
            {
                "from": 0, "to": 1,
                "members": (
                    self._member("node", "a", 1, changed=1),
                    self._member("node", "b", 1, added=1),
                ),
            },
            {
                "from": 2, "to": 3,
                "members": (
                    self._member("node", "a", 1, changed=1),
                    self._member("edge", "c", 1, removed=1),
                ),
            },
            {
                "from": 4, "to": 5,
                "members": (
                    self._member("node", "a", 2, changed=2),
                    self._member("node", "b", 1, removed=1),
                ),
            },
        )
        result = BranchStore._build_lifetime_churn_recurrences_result(
            waves, 2, 100
        )
        self.assertEqual(len(result["recurrences"]), 2)
        by_identity = {
            (r["type"], r["identity"]): r
            for r in result["recurrences"]
        }
        a = by_identity[("node", "a")]
        self.assertEqual(
            (a["waves"], a["first"], a["last"], a["span"]),
            (3, 0, 2, 3),
        )
        self.assertEqual(
            (a["segments"], a["added"], a["removed"], a["changed"]),
            (4, 0, 0, 4),
        )
        self.assertEqual(
            [entry["wave"] for entry in a["appearances"]], [0, 1, 2]
        )
        self.assertEqual(
            [(entry["from"], entry["to"]) for entry in a["appearances"]],
            [(0, 1), (2, 3), (4, 5)],
        )
        self.assertEqual(
            [entry["segments"] for entry in a["appearances"]], [1, 1, 2]
        )
        b = by_identity[("node", "b")]
        self.assertEqual(b["waves"], 2)
        self.assertEqual((b["first"], b["last"]), (0, 2))
        self.assertEqual((b["added"], b["removed"]), (1, 1))
        # c appears in only one wave and is dropped.
        self.assertNotIn(("edge", "c"), by_identity)
        self.assertEqual(
            result["totals"],
            {"recurrences": 2, "waves": 5, "segments": 6, "added": 1,
             "removed": 1, "changed": 4},
        )

    def test_builder_empty_waves_returns_zeros(self):
        result = BranchStore._build_lifetime_churn_recurrences_result(
            (), 1, 100
        )
        self.assertEqual(result["recurrences"], ())
        self.assertEqual(
            result["totals"],
            {"recurrences": 0, "waves": 0, "segments": 0, "added": 0,
             "removed": 0, "changed": 0},
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
        # Once the earlier parameters pass, recurrence_limit fails.
        with self.assertRaises(TypeError):
            self.call(min_waves=1, recurrence_limit="x")
        # Both still precede every state lookup and the token lookup.
        with self.assertRaises(ValueError):
            self.call(reference="ghost", min_waves=0)
        with self.assertRaises(ValueError):
            self.call(token="whatever", recurrence_limit=0)

    def test_recurrence_count_beyond_limit_raises_value_error(self):
        # Three identities recur across the two separated waves; cap two.
        with self.assertRaises(ValueError) as caught:
            self.call(
                windows=((0,), (1,), (1,), (2,)), recurrence_limit=2
            )
        self.assertIn("recurrence limit", str(caught.exception))
        result = self.call(
            windows=((0,), (1,), (1,), (2,)), recurrence_limit=3
        )
        self.assertEqual(len(result["recurrences"]), 3)
        # An empty result never trips the cap.
        self.assertEqual(
            self.call(recurrence_limit=1, min_waves=9)["recurrences"], ()
        )

    def test_wave_limit_still_fires_ahead_of_recurrence_cap(self):
        with self.assertRaises(ValueError) as caught:
            self.call(
                windows=((0,), (1,), (1,), (2,)),
                wave_limit=1,
                recurrence_limit=1,
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
        result = self.call(
            windows=((0,), (1,), (1,), (2,)),
            churn_limit=1,
            streak_limit=1,
        )
        self.assertEqual(len(result["recurrences"]), 3)

    def test_batch_caps_still_fire(self):
        with self.assertRaises(ValueError) as caught:
            self.call(total_diff_limit=5)
        self.assertIn("total diff", str(caught.exception))

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
            store, args, windows=((0,), (1,), (1,), (2,))
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
            store, args, windows=((0,), (1,), (1,), (2,))
        )
        pristine = copy.deepcopy(first)
        for record in first["recurrences"]:
            record["appearances"] += (
                {"wave": 9, "from": 9, "to": 10, "segments": 1,
                 "added": 0, "removed": 0, "changed": 1},
            )
            record["waves"] = 99
        self.assertEqual(
            recurrences_call(
                store, args, windows=((0,), (1,), (1,), (2,))
            ),
            pristine,
        )

    def test_success_and_failure_change_nothing(self):
        store, args = diamond_store()
        heads = dict(store._heads)
        appends = copy.deepcopy(store._appends)
        merges = copy.deepcopy(store._merges)
        audit = store.audit_log()

        recurrences_call(store, args)
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
            dict(wave_limit=0),
            dict(min_waves=0),
            dict(min_waves="x"),
            dict(recurrence_limit=0),
            dict(recurrence_limit="x"),
            dict(
                wave_limit=1,
                windows=((0,), (1,), (1,), (2,)),
            ),
            dict(
                recurrence_limit=1,
                windows=((0,), (1,), (1,), (2,)),
            ),
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

    def test_waves_query_is_unaffected(self):
        store, args = diamond_store()
        recurrences_call(store, args)
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
        self.call(token=token, windows=((0,), (1,), (1,), (2,)))
        with self.assertRaises(ValueError):
            self.call(
                token=token,
                wave_limit=1,
                windows=((0,), (1,), (1,), (2,)),
            )
        with self.assertRaises(ValueError):
            self.call(
                token=token,
                recurrence_limit=1,
                windows=((0,), (1,), (1,), (2,)),
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
