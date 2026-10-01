import copy
import inspect
import unittest

from city_twin.branches import BranchStore

from tests.test_lifetime_churn import (
    churn_call,
    churn_kwargs,
    diamond_store,
)


def streaks_kwargs(store_args, **overrides) -> dict:
    kwargs = churn_kwargs(store_args, **overrides)
    kwargs["streak_limit"] = overrides.pop("streak_limit", 50)
    return kwargs


def streaks_call(store, store_args, **overrides):
    return store.lifetime_churn_streaks(
        **streaks_kwargs(store_args, **overrides)
    )


def reference_streaks(evolution: dict[str, object]) -> dict[str, object]:
    """Independently group a lifetime_evolution result into streaks."""
    type_order = {"node": 0, "edge": 1, "gap": 2}
    per_identity: dict[tuple, list[tuple]] = {}
    encounter: dict[tuple, int] = {}
    next_encounter = 0
    for segment in evolution["segments"]:
        position = segment["left"]
        for change in segment["changes"]:
            type_name = change["kind"].split("_", 1)[0]
            key = (type_name, change["identity"])
            if key not in encounter:
                encounter[key] = next_encounter
                next_encounter += 1
            per_identity.setdefault(key, []).append((position, change))

    records: list[dict[str, object]] = []
    for key, entries in per_identity.items():
        type_name, identity = key
        run: list[tuple] = []

        def flush(run_entries: list[tuple]) -> None:
            added = removed = changed = 0
            evidence = []
            for run_position, change in run_entries:
                if change["kind"].endswith("_added"):
                    added += 1
                elif change["kind"].endswith("_removed"):
                    removed += 1
                else:
                    changed += 1
                evidence.append(
                    {
                        "from": run_position,
                        "to": run_position + 1,
                        "kind": change["kind"],
                        "before": change["before"],
                        "after": change["after"],
                    }
                )
            records.append(
                {
                    "type": type_name,
                    "identity": identity,
                    "from": run_entries[0][0],
                    "to": run_entries[-1][0] + 1,
                    "segments": len(run_entries),
                    "added": added,
                    "removed": removed,
                    "changed": changed,
                    "changes": tuple(evidence),
                }
            )

        for entry in entries:
            if run and entry[0] != run[-1][0] + 1:
                flush(run)
                run = []
            run.append(entry)
        if run:
            flush(run)

    records.sort(
        key=lambda record: (
            -(record["added"] + record["removed"] + record["changed"]),
            record["from"],
            record["to"],
            type_order[record["type"]],
            encounter[(record["type"], record["identity"])],
        )
    )
    return {
        "streaks": tuple(records),
        "totals": {
            "streaks": len(records),
            "identities": len(per_identity),
            "added": sum(r["added"] for r in records),
            "removed": sum(r["removed"] for r in records),
            "changed": sum(r["changed"] for r in records),
        },
    }


class LifetimeChurnStreaksSignatureTests(unittest.TestCase):
    def test_public_signature_is_churn_plus_streak_limit(self):
        parameters = list(
            inspect.signature(
                BranchStore.lifetime_churn_streaks
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
        result = streaks_call(store, args)
        self.assertIsInstance(result, dict)
        self.assertEqual(list(result), ["streaks", "totals"])
        self.assertIsInstance(result["streaks"], tuple)
        self.assertEqual(
            list(result["totals"]),
            ["streaks", "identities", "added", "removed", "changed"],
        )
        for record in result["streaks"]:
            self.assertIsInstance(record, dict)
            self.assertEqual(
                list(record),
                [
                    "type",
                    "identity",
                    "from",
                    "to",
                    "segments",
                    "added",
                    "removed",
                    "changed",
                    "changes",
                ],
            )
            self.assertIsInstance(record["changes"], tuple)
            for change in record["changes"]:
                self.assertEqual(
                    list(change), ["from", "to", "kind", "before", "after"]
                )


class LifetimeChurnStreaksResultTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        return streaks_call(self.store, self.args, **overrides)

    def evolution(self, **overrides):
        kwargs = churn_kwargs(self.args, **overrides)
        kwargs.pop("churn_limit")
        kwargs.pop("streak_limit", None)
        return self.store.lifetime_evolution(**kwargs)

    def test_matches_independent_grouping_of_evolution(self):
        for windows in (
            ((0,), (1,), (2,)),
            ((0, 2), (1,), (0, 1, 2)),
            ((2,), (0,)),
            ((1, 2), (0, 1)),
            ((0, 1), (0, 1), (0, 1)),
            ((0, 1, 2),),
        ):
            with self.subTest(windows=windows):
                self.assertEqual(
                    self.call(windows=windows),
                    reference_streaks(self.evolution(windows=windows)),
                )

    def test_diamond_fixture_every_identity_churns_all_three_segments(self):
        result = self.call()
        self.assertEqual(len(result["streaks"]), 3)
        for record in result["streaks"]:
            self.assertEqual((record["from"], record["to"]), (0, 2))
            self.assertEqual(record["segments"], 2)
            self.assertEqual(
                len(record["changes"]),
                record["added"] + record["removed"] + record["changed"],
            )
        by_identity = {
            (record["type"], record["identity"]): record
            for record in result["streaks"]
        }
        a0 = by_identity[("node", ("a0", "x"))]
        w = by_identity[("node", ("W", "y"))]
        edge = by_identity[("edge", (("a0", "x"), ("W", "y")))]
        self.assertEqual((a0["added"], a0["removed"], a0["changed"]), (0, 0, 2))
        self.assertEqual((w["added"], w["removed"], w["changed"]), (1, 0, 1))
        self.assertEqual(
            (edge["added"], edge["removed"], edge["changed"]), (1, 0, 1)
        )
        self.assertEqual(
            result["totals"],
            {"streaks": 3, "identities": 3, "added": 2, "removed": 0,
             "changed": 4},
        )

    def test_single_change_segment_is_a_streak(self):
        # Two windows -> one segment; every touched identity gets one
        # length-one streak regardless of kind.
        result = self.call(windows=((2,), (0,)))
        self.assertTrue(result["streaks"])
        for record in result["streaks"]:
            self.assertEqual(record["segments"], 1)
            self.assertEqual(record["to"], record["from"] + 1)

    def test_identities_present_but_unchanged_never_appear(self):
        result = self.call(windows=((0, 1), (0, 1), (0, 1)))
        self.assertEqual(result["streaks"], ())
        self.assertEqual(
            result["totals"],
            {"streaks": 0, "identities": 0, "added": 0, "removed": 0,
             "changed": 0},
        )

    def test_empty_windows_causes_and_single_window_return_zeros(self):
        for overrides in (
            {"windows": ()},
            {"causes": ()},
            {"windows": ((0, 1, 2),)},
        ):
            with self.subTest(overrides=overrides):
                result = self.call(**overrides)
                self.assertEqual(result["streaks"], ())
                self.assertEqual(
                    result["totals"],
                    {"streaks": 0, "identities": 0, "added": 0,
                     "removed": 0, "changed": 0},
                )

    def test_change_evidence_keeps_lifetime_diff_semantics(self):
        result = self.call()
        evolution = self.evolution()
        segment_changes = {
            (segment["left"], segment["right"]): segment["changes"]
            for segment in evolution["segments"]
        }
        for record in result["streaks"]:
            for change in record["changes"]:
                match = next(
                    candidate
                    for candidate in segment_changes[
                        (change["from"], change["to"])
                    ]
                    if candidate["identity"] == record["identity"]
                )
                self.assertEqual(change["kind"], match["kind"])
                self.assertEqual(change["before"], match["before"])
                self.assertEqual(change["after"], match["after"])
                if change["kind"].endswith("_added"):
                    self.assertIsNone(change["before"])
                elif change["kind"].endswith("_removed"):
                    self.assertIsNone(change["after"])

    def test_sort_score_desc_then_from_to_type_identity(self):
        result = self.call()
        keys = [
            (
                -(r["added"] + r["removed"] + r["changed"]),
                r["from"],
                r["to"],
                {"node": 0, "edge": 1, "gap": 2}[r["type"]],
            )
            for r in result["streaks"]
        ]
        self.assertEqual(keys, sorted(keys))


class LifetimeChurnStreakGroupingTests(unittest.TestCase):
    @staticmethod
    def _lifetime(identity):
        return {
            "type": "node",
            "identity": identity,
            "first_seen": 0,
            "last_seen": 0,
            "intervals": ((0, 0),),
            "transitions": (),
        }

    def _change(self, kind, identity, before, after):
        return {
            "kind": kind,
            "identity": identity,
            "before": (
                None if before is None else self._lifetime(before)
            ),
            "after": None if after is None else self._lifetime(after),
        }

    def test_gap_segment_cuts_and_removed_then_readded_continues(self):
        x = ("X", "k")
        y = ("Y", "k")
        edge = ("e",)
        segments = [
            (
                self._change("node_changed", x, x, x),
                self._change("node_removed", y, y, None),
                self._change("edge_added", edge, None, edge),
            ),
            (
                self._change("node_added", y, None, y),
                self._change("edge_changed", edge, edge, edge),
            ),
            # X is absent on the previous segment, so this reopens it.
            (self._change("node_added", x, None, x),),
        ]
        result = BranchStore._build_lifetime_churn_streaks_result(
            segments, 100
        )
        by_identity = {}
        for record in result["streaks"]:
            by_identity.setdefault(
                (record["type"], record["identity"]), []
            ).append(record)

        x_streaks = by_identity[("node", x)]
        self.assertEqual(len(x_streaks), 2)
        self.assertEqual(
            (x_streaks[0]["from"], x_streaks[0]["to"],
             x_streaks[0]["segments"], x_streaks[0]["changed"]),
            (0, 1, 1, 1),
        )
        self.assertEqual(
            (x_streaks[1]["from"], x_streaks[1]["to"],
             x_streaks[1]["segments"], x_streaks[1]["added"]),
            (2, 3, 1, 1),
        )

        # Removed then re-added on the next segment stays one streak.
        y_streaks = by_identity[("node", y)]
        self.assertEqual(len(y_streaks), 1)
        self.assertEqual(
            (y_streaks[0]["from"], y_streaks[0]["to"],
             y_streaks[0]["segments"]),
            (0, 2, 2),
        )
        self.assertEqual(
            (y_streaks[0]["added"], y_streaks[0]["removed"],
             y_streaks[0]["changed"]),
            (1, 1, 0),
        )

        edge_streaks = by_identity[("edge", edge)]
        self.assertEqual(len(edge_streaks), 1)
        self.assertEqual(
            (edge_streaks[0]["from"], edge_streaks[0]["to"]), (0, 2)
        )

        self.assertEqual(
            result["totals"],
            {"streaks": 4, "identities": 3, "added": 3, "removed": 1,
             "changed": 2},
        )
        order = [
            (r["identity"], r["type"],
             r["added"] + r["removed"] + r["changed"], r["from"])
            for r in result["streaks"]
        ]
        self.assertEqual(
            order,
            [(y, "node", 2, 0), (edge, "edge", 2, 0),
             (x, "node", 1, 0), (x, "node", 1, 2)],
        )


class LifetimeChurnStreaksLimitValidationTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        overrides.setdefault("causes", ("a0",))
        return streaks_call(self.store, self.args, **overrides)

    def test_streak_limit_must_be_a_non_bool_positive_int(self):
        for bad in (True, False, 1.0, "1", None):
            with self.subTest(streak_limit=bad):
                with self.assertRaises(TypeError):
                    self.call(streak_limit=bad)
        with self.assertRaises(ValueError):
            self.call(streak_limit=0)
        with self.assertRaises(ValueError):
            self.call(streak_limit=-4)

    def test_streak_limit_validated_last_among_caps(self):
        # churn_limit fails before streak_limit.
        with self.assertRaises(ValueError):
            self.call(churn_limit=0, streak_limit=0)
        with self.assertRaises(TypeError):
            self.call(churn_limit="x", streak_limit=0)
        # total_diff_limit fails first even when streak_limit is bad.
        with self.assertRaises(ValueError):
            self.call(total_diff_limit=0, streak_limit="x")
        # Once the earlier cap passes, streak_limit's own type error fires.
        with self.assertRaises(TypeError):
            self.call(total_diff_limit=1, streak_limit="x")
        # Still precedes every state lookup and the token lookup.
        with self.assertRaises(ValueError):
            self.call(reference="ghost", streak_limit=0)
        with self.assertRaises(ValueError):
            self.call(token="whatever", streak_limit=0)

    def test_record_count_beyond_streak_limit_raises_value_error(self):
        with self.assertRaises(ValueError) as caught:
            self.call(streak_limit=2)
        self.assertIn("streak limit", str(caught.exception))
        result = self.call(streak_limit=3)
        self.assertEqual(len(result["streaks"]), 3)

    def test_batch_caps_still_fire_ahead_of_streak_cap(self):
        # Six total changes over five trips the batch cap, not streak cap.
        with self.assertRaises(ValueError) as caught:
            self.call(total_diff_limit=5, streak_limit=1)
        self.assertIn("total diff", str(caught.exception))
        # Empty batches never trip the streak cap.
        self.assertEqual(
            self.call(windows=(), streak_limit=1)["streaks"], ()
        )

    def test_shared_validation_errors_match_churn(self):
        with self.assertRaises(TypeError):
            self.call(windows="x")
        with self.assertRaises(ValueError):
            self.call(windows=((-1,),))
        with self.assertRaises(ValueError):
            self.call(direction="sideways")
        with self.assertRaises(ValueError):
            self.call(window_limit=2, windows=((0,), (1,), (2,)))


class LifetimeChurnStreaksStateTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        overrides.setdefault("causes", ("a0",))
        return streaks_call(self.store, self.args, **overrides)

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


class LifetimeChurnStreaksIsolationTests(unittest.TestCase):
    def test_result_levels_are_mutually_unshared(self):
        store, args = diamond_store()
        result = streaks_call(store, args)
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
        first = streaks_call(store, args)
        pristine = copy.deepcopy(first)
        for record in first["streaks"]:
            record["changes"] += (
                {"from": 4, "to": 5, "kind": "evil",
                 "before": None, "after": None},
            )
            for change in record["changes"]:
                for side in ("before", "after"):
                    evidence = change[side]
                    if evidence is not None:
                        evidence["first_seen"] = 99
                        evidence["intervals"] += ((99, 99),)
        self.assertEqual(streaks_call(store, args), pristine)

    def test_success_and_failure_change_nothing(self):
        store, args = diamond_store()
        heads = dict(store._heads)
        appends = copy.deepcopy(store._appends)
        merges = copy.deepcopy(store._merges)
        audit = store.audit_log()

        streaks_call(store, args)
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
            dict(streak_limit="x"),
            dict(streak_limit=1),
            dict(causes=("zzz",)),
            dict(limit=1),
            dict(reference="ghost"),
        ]
        for overrides in failures:
            with self.subTest(overrides=overrides):
                with self.assertRaises(Exception):
                    streaks_call(store, args, causes=("a0",), **overrides)

        self.assertEqual(dict(store._heads), heads)
        self.assertEqual(copy.deepcopy(store._appends), appends)
        self.assertEqual(copy.deepcopy(store._merges), merges)
        self.assertEqual(store.audit_log(), audit)

    def test_existing_public_entries_are_unaffected(self):
        store, args = diamond_store()
        streaks_call(store, args)
        churn = churn_call(store, args)
        self.assertEqual(churn_call(store, args), churn)


class LifetimeChurnStreaksTokenTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, token=None, **overrides):
        return streaks_call(self.store, self.args, token=token, **overrides)

    def test_success_consumes_one_read_and_failure_refunds(self):
        token = self.store.create_snapshot(2)
        self.call(token=token)
        with self.assertRaises(ValueError):
            self.call(token=token, streak_limit=1)
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
