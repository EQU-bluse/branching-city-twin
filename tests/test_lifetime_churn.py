import copy
import inspect
import unittest

from city_twin.branches import BranchStore
from city_twin.event_graph import EventGraph


def diamond_store() -> tuple[BranchStore, tuple]:
    """Three aligned checkpoints; prefix 1 is a0 only, later prefixes add W.

    a0 reaches merge W over a cross-branch convergence edge; the third
    checkpoint changes nothing, so the last segment carries only
    position-shift changes.
    """
    graph = EventGraph()
    graph.add("root", 0, (), {})
    store = BranchStore(graph)
    store.create("main", "root")
    store.create("a", "root")
    store.append("main", "r0", 1, {})
    store.append("a", "a0", 2, {"x": 2})
    store.append("main", "r1", 3, {})
    store.create("b", "a0")
    store.append("b", "bu", 4, {})
    store.create("c", "a0")
    store.append("c", "cv", 4, {})
    store.merge("b", "c", "W", 5, {"y": 8})
    store.append("main", "r2", 6, {})
    series = {
        "a": (("r0", "a0"), ("r1", "a0"), ("r2", "a0")),
        "b": (("r0", "a0"), ("r1", "W"), ("r2", "W")),
    }
    scenario = {
        "weights": {"x": 1, "y": 1},
        "total_budget": 20,
        "key_budgets": {},
    }
    args = ("main", series, scenario, "total_budget", (0, 10, 20), 0, 2, (), (), 10)
    return store, args


def churn_kwargs(store_args, **overrides) -> dict:
    _, series, scenario, axis, values, mn, mx, req, excl, limit = store_args
    kwargs = dict(
        reference="main",
        series=series,
        base_scenario=scenario,
        axis=axis,
        values=values,
        min_size=mn,
        max_size=mx,
        required=req,
        exclusive_pairs=excl,
        limit=limit,
        windows=((0,), (1,), (2,)),
        causes=("a0",),
        direction="both",
        depth=99,
        node_limit=50,
        change_limit=50,
        lifetime_limit=50,
        diff_limit=50,
        window_limit=50,
        total_diff_limit=50,
        churn_limit=50,
    )
    kwargs.update(overrides)
    return kwargs


def churn_call(store, store_args, **overrides):
    return store.lifetime_churn(**churn_kwargs(store_args, **overrides))


def reference_churn(evolution: dict[str, object]) -> dict[str, object]:
    """Independently aggregate a lifetime_evolution result for comparison."""
    encounter: dict[tuple, int] = {}
    next_encounter = 0
    present: dict[tuple, list[int]] = {}
    for window in evolution["windows"]:
        position = window["position"]
        for record in window["lifetimes"]:
            key = (record["type"], record["identity"])
            if key not in encounter:
                encounter[key] = next_encounter
                next_encounter += 1
                present[key] = []
            present[key].append(position)

    added: dict[tuple, int] = {}
    removed: dict[tuple, int] = {}
    changed: dict[tuple, int] = {}
    identity_changes: dict[tuple, list[dict[str, object]]] = {}
    first_change_to: dict[tuple, int] = {}
    for segment in evolution["segments"]:
        for change in segment["changes"]:
            type_name = change["kind"].split("_", 1)[0]
            key = (type_name, change["identity"])
            if key not in encounter:
                encounter[key] = next_encounter
                next_encounter += 1
                first_change_to[key] = segment["right"]
            if change["kind"].endswith("_added"):
                added[key] = added.get(key, 0) + 1
            elif change["kind"].endswith("_removed"):
                removed[key] = removed.get(key, 0) + 1
            else:
                changed[key] = changed.get(key, 0) + 1
            identity_changes.setdefault(key, []).append(
                {
                    "from": segment["left"],
                    "to": segment["right"],
                    "kind": change["kind"],
                    "before": change["before"],
                    "after": change["after"],
                }
            )

    type_order = {"node": 0, "edge": 1, "gap": 2}
    ordered = sorted(
        encounter,
        key=lambda key: (
            -(added.get(key, 0) + removed.get(key, 0) + changed.get(key, 0)),
            present[key][0] if key in present else first_change_to[key],
            type_order[key[0]],
            encounter[key],
        ),
    )
    records = []
    total_added = total_removed = total_changed = 0
    for key in ordered:
        a, r, c = added.get(key, 0), removed.get(key, 0), changed.get(key, 0)
        total_added += a
        total_removed += r
        total_changed += c
        records.append(
            {
                "type": key[0],
                "identity": key[1],
                "present": tuple(present.get(key, ())),
                "added": a,
                "removed": r,
                "changed": c,
                "changes": tuple(identity_changes.get(key, ())),
            }
        )
    return {
        "identities": tuple(records),
        "totals": {
            "identities": len(records),
            "added": total_added,
            "removed": total_removed,
            "changed": total_changed,
        },
    }


class LifetimeChurnSignatureTests(unittest.TestCase):
    def test_public_signature_is_evolution_plus_churn_limit(self):
        parameters = list(
            inspect.signature(BranchStore.lifetime_churn).parameters.values()
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
        result = churn_call(store, args)
        self.assertIsInstance(result, dict)
        self.assertEqual(list(result), ["identities", "totals"])
        self.assertIsInstance(result["identities"], tuple)
        self.assertEqual(
            list(result["totals"]),
            ["identities", "added", "removed", "changed"],
        )
        for record in result["identities"]:
            self.assertIsInstance(record, dict)
            self.assertEqual(
                list(record),
                [
                    "type",
                    "identity",
                    "present",
                    "added",
                    "removed",
                    "changed",
                    "changes",
                ],
            )
            self.assertIsInstance(record["present"], tuple)
            self.assertIsInstance(record["changes"], tuple)
            for change in record["changes"]:
                self.assertEqual(
                    list(change), ["from", "to", "kind", "before", "after"]
                )


class LifetimeChurnResultTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        return churn_call(self.store, self.args, **overrides)

    def evolution(self, **overrides):
        kwargs = churn_kwargs(self.args, **overrides)
        kwargs.pop("churn_limit")
        return self.store.lifetime_evolution(**kwargs)

    def test_matches_independent_aggregation_of_evolution(self):
        for windows in (
            ((0,), (1,), (2,)),
            ((0, 2), (1,), (0, 1, 2)),
            ((2,), (0,)),
            ((1, 2), (0, 1)),
            ((0, 1), (0, 1), (0, 1)),
            ((0, 1, 2),),
        ):
            with self.subTest(windows=windows):
                result = self.call(windows=windows)
                self.assertEqual(
                    result, reference_churn(self.evolution(windows=windows))
                )

    def test_diamond_fixture_counters_and_presence(self):
        result = self.call()
        by_identity = {
            (record["type"], record["identity"]): record
            for record in result["identities"]
        }
        a0 = by_identity[("node", ("a0", "x"))]
        w = by_identity[("node", ("W", "y"))]
        edge = by_identity[("edge", (("a0", "x"), ("W", "y")))]
        self.assertEqual(a0["present"], (0, 1, 2))
        self.assertEqual((a0["added"], a0["removed"], a0["changed"]), (0, 0, 2))
        self.assertEqual(
            [(c["from"], c["to"], c["kind"]) for c in a0["changes"]],
            [(0, 1, "node_changed"), (1, 2, "node_changed")],
        )
        self.assertEqual(w["present"], (1, 2))
        self.assertEqual((w["added"], w["removed"], w["changed"]), (1, 0, 1))
        self.assertEqual(
            [(c["from"], c["to"], c["kind"]) for c in w["changes"]],
            [(0, 1, "node_added"), (1, 2, "node_changed")],
        )
        self.assertEqual(edge["present"], (1, 2))
        self.assertEqual(
            (edge["added"], edge["removed"], edge["changed"]), (1, 0, 1)
        )
        self.assertEqual(
            [(c["from"], c["to"], c["kind"]) for c in edge["changes"]],
            [(0, 1, "edge_added"), (1, 2, "edge_changed")],
        )

    def test_records_order_by_score_then_first_presence_then_type(self):
        result = self.call()
        records = result["identities"]
        # All three identities score 2; a0 is present from window 0, the
        # other two from window 1 where the node precedes the edge.
        self.assertEqual(
            [(r["type"], r["identity"]) for r in records],
            [
                ("node", ("a0", "x")),
                ("node", ("W", "y")),
                ("edge", (("a0", "x"), ("W", "y"))),
            ],
        )
        scores = [r["added"] + r["removed"] + r["changed"] for r in records]
        first_presence = [
            r["present"][0] if r["present"] else r["changes"][0]["to"]
            for r in records
        ]
        self.assertEqual(scores, sorted(scores, reverse=True))
        for earlier, later in zip(records, records[1:]):
            earlier_score = (
                earlier["added"] + earlier["removed"] + earlier["changed"]
            )
            later_score = later["added"] + later["removed"] + later["changed"]
            if earlier_score == later_score:
                earlier_first = (
                    earlier["present"][0]
                    if earlier["present"]
                    else earlier["changes"][0]["to"]
                )
                later_first = (
                    later["present"][0]
                    if later["present"]
                    else later["changes"][0]["to"]
                )
                self.assertLessEqual(earlier_first, later_first)

    def test_disjoint_windows_count_removals(self):
        result = self.call(windows=((2,), (0,)))
        kinds = {
            (r["type"], r["identity"]): (
                r["added"],
                r["removed"],
                r["changed"],
                r["present"],
            )
            for r in result["identities"]
        }
        self.assertEqual(kinds[("node", ("W", "y"))], (0, 1, 0, (0,)))
        self.assertEqual(
            kinds[("edge", (("a0", "x"), ("W", "y")))], (0, 1, 0, (0,))
        )
        self.assertEqual(kinds[("node", ("a0", "x"))][:3], (0, 0, 1))

    def test_totals_match_record_sums(self):
        result = self.call()
        records = result["identities"]
        self.assertEqual(
            result["totals"],
            {
                "identities": len(records),
                "added": sum(r["added"] for r in records),
                "removed": sum(r["removed"] for r in records),
                "changed": sum(r["changed"] for r in records),
            },
        )
        self.assertEqual(
            result["totals"],
            {"identities": 3, "added": 2, "removed": 0, "changed": 4},
        )

    def test_single_window_has_presence_but_no_changes(self):
        result = self.call(windows=((0, 1, 2),))
        self.assertEqual(len(result["identities"]), 3)
        for record in result["identities"]:
            self.assertEqual(record["present"], (0,))
            self.assertEqual(
                (record["added"], record["removed"], record["changed"]),
                (0, 0, 0),
            )
            self.assertEqual(record["changes"], ())
        self.assertEqual(
            result["totals"],
            {"identities": 3, "added": 0, "removed": 0, "changed": 0},
        )

    def test_identical_adjacent_windows_create_no_changes(self):
        result = self.call(windows=((0, 1), (0, 1), (0, 1)))
        self.assertEqual(len(result["identities"]), 3)
        for record in result["identities"]:
            self.assertEqual(record["present"], (0, 1, 2))
            self.assertEqual(
                (record["added"], record["removed"], record["changed"]),
                (0, 0, 0),
            )
            self.assertEqual(record["changes"], ())
        self.assertEqual(
            result["totals"],
            {"identities": 3, "added": 0, "removed": 0, "changed": 0},
        )

    def test_empty_windows_return_empty_identities_and_zero_totals(self):
        result = self.call(windows=())
        self.assertEqual(
            result,
            {
                "identities": (),
                "totals": {
                    "identities": 0,
                    "added": 0,
                    "removed": 0,
                    "changed": 0,
                },
            },
        )

    def test_empty_causes_return_empty_identities_and_zero_totals(self):
        result = self.call(causes=())
        self.assertEqual(result["identities"], ())
        self.assertEqual(
            result["totals"],
            {"identities": 0, "added": 0, "removed": 0, "changed": 0},
        )

    def test_change_evidence_keeps_lifetime_diff_semantics(self):
        result = self.call()
        evolution = self.evolution()
        segment_changes = {
            (segment["left"], segment["right"]): segment["changes"]
            for segment in evolution["segments"]
        }
        for record in result["identities"]:
            for change in record["changes"]:
                match = next(
                    candidate
                    for candidate in segment_changes[(change["from"], change["to"])]
                    if candidate["identity"] == record["identity"]
                )
                self.assertEqual(change["kind"], match["kind"])
                self.assertEqual(change["before"], match["before"])
                self.assertEqual(change["after"], match["after"])
                if change["kind"].endswith("_added"):
                    self.assertIsNone(change["before"])
                elif change["kind"].endswith("_removed"):
                    self.assertIsNone(change["after"])


class LifetimeChurnLimitValidationTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        overrides.setdefault("causes", ("a0",))
        return churn_call(self.store, self.args, **overrides)

    def test_churn_limit_must_be_a_non_bool_positive_int(self):
        for bad in (True, False, 1.0, "1", None):
            with self.subTest(churn_limit=bad):
                with self.assertRaises(TypeError):
                    self.call(churn_limit=bad)
        with self.assertRaises(ValueError):
            self.call(churn_limit=0)
        with self.assertRaises(ValueError):
            self.call(churn_limit=-4)

    def test_churn_limit_validated_after_total_diff_limit(self):
        # total_diff_limit fails first even when churn_limit is also bad.
        with self.assertRaises(ValueError):
            self.call(total_diff_limit=0, churn_limit="x")
        with self.assertRaises(TypeError):
            self.call(total_diff_limit=1, churn_limit="x")
        with self.assertRaises(ValueError):
            self.call(diff_limit=0, churn_limit=0)
        # And before the earlier caps would even matter.
        with self.assertRaises(ValueError):
            self.call(window_limit=0, churn_limit=0)

    def test_churn_limit_validated_before_state_lookups(self):
        with self.assertRaises(ValueError):
            self.call(reference="ghost", churn_limit=0)
        with self.assertRaises(TypeError):
            self.call(reference="ghost", churn_limit="x")

    def test_record_count_beyond_churn_limit_raises_value_error(self):
        with self.assertRaises(ValueError) as caught:
            self.call(churn_limit=2)
        self.assertIn("churn limit", str(caught.exception))
        result = self.call(churn_limit=3)
        self.assertEqual(len(result["identities"]), 3)

    def test_over_limit_returns_no_partial_results_and_runs_last(self):
        # Caps intrinsic to the batch still fire ahead of churn_limit:
        # total diffs here are six, over five.
        with self.assertRaises(ValueError):
            self.call(total_diff_limit=5, churn_limit=50)
        # With those satisfied, churn_limit=1 trips on the three records.
        with self.assertRaises(ValueError):
            self.call(churn_limit=1)
        # Empty batches never trip the churn cap.
        result = self.call(windows=(), churn_limit=1)
        self.assertEqual(result["identities"], ())

    def test_evolution_parity_validation_errors(self):
        # The shared validation surface behaves exactly like evolution.
        with self.assertRaises(TypeError):
            self.call(windows="x")
        with self.assertRaises(TypeError):
            self.call(windows=((0,), "x"))
        with self.assertRaises(TypeError):
            self.call(windows=((True,),))
        with self.assertRaises(ValueError):
            self.call(windows=((-1,),))
        with self.assertRaises(ValueError):
            self.call(windows=((9,),))
        with self.assertRaises(ValueError):
            self.call(windows=((1, 1),))
        with self.assertRaises(TypeError):
            self.call(causes=[1])
        with self.assertRaises(ValueError):
            self.call(direction="sideways")
        with self.assertRaises(ValueError):
            self.call(node_limit=0)
        with self.assertRaises(ValueError):
            self.call(window_limit=2, windows=((0,), (1,), (2,)))
        with self.assertRaises(ValueError):
            self.call(diff_limit=2, windows=((0,), (1,)))
        with self.assertRaises(ValueError):
            self.call(total_diff_limit=5)
        with self.assertRaises(ValueError):
            self.call(lifetime_limit=2, windows=((0, 1, 2),))


class LifetimeChurnStateCheckTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        overrides.setdefault("causes", ("a0",))
        return churn_call(self.store, self.args, **overrides)

    def test_unknown_branch_and_history_node_raise_key_error(self):
        with self.assertRaises(KeyError):
            self.call(reference="ghost")
        with self.assertRaises(KeyError):
            self.call(
                series={
                    "a": (("r0", "a0"), ("r1", "a0"), ("r2", "a0")),
                    "b": (("r0", "a0"), ("r1", "W"), ("r2", "nope2")),
                },
            )

    def test_absent_cause_raises_key_error(self):
        with self.assertRaises(KeyError) as caught:
            self.call(causes=("zzz",))
        self.assertEqual(caught.exception.args[0], "zzz")

    def test_resource_caps_raise_value_error(self):
        with self.assertRaises(ValueError):
            self.call(limit=1)
        with self.assertRaises(ValueError):
            self.call(direction="forward", node_limit=1)
        with self.assertRaises(ValueError):
            self.call(windows=((0, 1),), change_limit=1)

    def test_empty_windows_still_run_every_state_check(self):
        with self.assertRaises(KeyError):
            self.call(windows=(), reference="ghost")
        with self.assertRaises(ValueError):
            self.call(windows=(), limit=1)


class LifetimeChurnIsolationTests(unittest.TestCase):
    def test_result_levels_are_mutually_unshared(self):
        store, args = diamond_store()
        result = churn_call(store, args)
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
        first = churn_call(store, args)
        pristine = copy.deepcopy(first)
        for record in first["identities"]:
            record["present"] += (99,)
            record["changes"] += ({"from": 4, "to": 5, "kind": "evil",
                                   "before": None, "after": None},)
            for change in record["changes"]:
                for side in ("before", "after"):
                    evidence = change[side]
                    if evidence is not None:
                        evidence["first_seen"] = 99
                        evidence["intervals"] += ((99, 99),)
        second = churn_call(store, args)
        self.assertEqual(second, pristine)

    def test_repeated_calls_are_equal(self):
        store, args = diamond_store()
        self.assertEqual(churn_call(store, args), churn_call(store, args))


class LifetimeChurnReadOnlyTests(unittest.TestCase):
    def test_success_and_failure_change_nothing(self):
        store, args = diamond_store()
        heads = dict(store._heads)
        appends = copy.deepcopy(store._appends)
        merges = copy.deepcopy(store._merges)
        audit = store.audit_log()

        churn_call(store, args)
        failures = [
            dict(windows="x"),
            dict(windows=("x",)),
            dict(windows=((-1,),)),
            dict(windows=((9,),)),
            dict(windows=((1, 0),)),
            dict(windows=((1, 1),)),
            dict(windows=((True,),)),
            dict(causes=[1]),
            dict(direction="up"),
            dict(node_limit=0),
            dict(change_limit=0),
            dict(lifetime_limit=0),
            dict(diff_limit=0),
            dict(window_limit=0),
            dict(total_diff_limit=0),
            dict(churn_limit=0),
            dict(churn_limit="x"),
            dict(churn_limit=1),
            dict(causes=("zzz",)),
            dict(limit=1),
            dict(reference="ghost"),
        ]
        for overrides in failures:
            with self.subTest(overrides=overrides):
                with self.assertRaises(Exception):
                    churn_call(store, args, causes=("a0",), **overrides)

        self.assertEqual(dict(store._heads), heads)
        self.assertEqual(copy.deepcopy(store._appends), appends)
        self.assertEqual(copy.deepcopy(store._merges), merges)
        self.assertEqual(store.audit_log(), audit)

    def test_other_public_entries_are_unaffected(self):
        store, args = diamond_store()
        evolution_kwargs = {
            key: value
            for key, value in churn_kwargs(args).items()
            if key != "churn_limit"
        }
        evolution = store.lifetime_evolution(**evolution_kwargs)
        churn_call(store, args)
        self.assertEqual(store.lifetime_evolution(**evolution_kwargs), evolution)


class LifetimeChurnTokenTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, token=None, **overrides):
        return churn_call(self.store, self.args, token=token, **overrides)

    def test_non_str_token_raises_type_error(self):
        for bad in (123, True, b"x", (), 1.5):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    self.call(token=bad)

    def test_empty_token_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.call(token="")

    def test_unknown_and_released_tokens_raise_key_error(self):
        with self.assertRaises(KeyError):
            self.call(token="never-issued")
        token = self.store.create_snapshot(2)
        self.store.release_snapshot(token)
        with self.assertRaises(KeyError):
            self.call(token=token)

    def test_churn_limit_failure_precedes_token_lookup(self):
        with self.assertRaises(ValueError):
            self.call(token="whatever", churn_limit=0)

    def test_success_consumes_one_read_and_failure_refunds(self):
        token = self.store.create_snapshot(2)
        self.call(token=token)
        # A failing query refunds its reservation, so the second read
        # remains available after it.
        with self.assertRaises(ValueError):
            self.call(token=token, churn_limit=1)
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
        # The frozen view never learned the post-snapshot events: a
        # history-node lookup that succeeds live raises KeyError on it.
        series_new_node = {
            "a": (("r0", "a2"), ("r1", "a2"), ("r2", "a2")),
            "b": (("r0", "a0"), ("r1", "W"), ("r2", "W")),
        }
        self.call(series=series_new_node)
        with self.assertRaises(KeyError):
            self.call(token=token, series=series_new_node)


if __name__ == "__main__":
    unittest.main()
