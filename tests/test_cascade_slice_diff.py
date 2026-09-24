import copy
import inspect
import unittest

from city_twin.branches import BranchStore
from city_twin.event_graph import EventGraph


def diamond_store() -> tuple[BranchStore, tuple]:
    """Two aligned checkpoints; prefix 1 is a0 only, prefix 2 adds W.

    a0 reaches merge W over a cross-branch convergence edge.
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
    series = {
        "a": (("r0", "a0"), ("r1", "a0")),
        "b": (("r0", "a0"), ("r1", "W")),
    }
    scenario = {
        "weights": {"x": 1, "y": 1},
        "total_budget": 20,
        "key_budgets": {},
    }
    args = ("main", series, scenario, "total_budget", (0, 10, 20), 0, 2, (), (), 10)
    return store, args


def changed_store() -> tuple[BranchStore, dict, tuple]:
    """A node keeps its (cause, key) but its locating checkpoint moves.

    Prefix 1 attributes the interval-0 feasibility change at checkpoint 0
    with no affected events; prefix 2 first breaches at checkpoint 1, so the
    same cause node relocates and gains affected events -- a content change
    without any add or remove.
    """
    graph = EventGraph()
    graph.add("root", 0, (), {})
    store = BranchStore(graph)
    store.create("main", "root")
    store.create("a", "root")
    store.append("main", "r0", 1, {"x": 2})
    store.append("a", "a0", 2, {"x": 5})
    store.append("main", "r1", 3, {"x": -2})
    store.append("a", "a1", 4, {})
    store.append("main", "r2", 5, {})
    points = (("r0", "a0"), ("r1", "a1"), ("r2", "a1"))
    scenario = {"weights": {"x": 1}, "total_budget": 6, "key_budgets": {}}
    return store, scenario, points


def gap_store() -> tuple[BranchStore, dict, tuple]:
    """Prefix 2 introduces an empty-combo gap at interval 0."""
    graph = EventGraph()
    graph.add("root", 0, (), {})
    store = BranchStore(graph)
    store.create("main", "root")
    store.create("a", "root")
    store.append("main", "r1", 1, {})
    store.append("a", "p1", 2, {"k": 5})
    store.append("a", "c1", 3, {})
    store.create("q", "p1")
    store.append("q", "q1", 4, {})
    store.merge("a", "q", "D", 5, {"k": 7})
    store.append("a", "E", 6, {"k": 7})
    store.merge("main", "a", "H", 7, {})
    store.append("main", "r2", 8, {})
    points = (("r1", "c1"), ("r2", "E"))
    scenario = {"weights": {"k": 1}, "total_budget": 13, "key_budgets": {}}
    return store, scenario, points


def diff_kwargs(store_args, **overrides) -> dict:
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
        before=0,
        after=1,
        causes=("a0",),
        direction="both",
        depth=99,
        node_limit=50,
    )
    kwargs.update(overrides)
    return kwargs


def diff_call(store, store_args, **overrides):
    return store.cascade_slice_diff(**diff_kwargs(store_args, **overrides))


class CascadeSliceDiffSignatureTests(unittest.TestCase):
    def test_public_signature_inserts_before_after_before_causes(self) -> None:
        parameters = list(
            inspect.signature(BranchStore.cascade_slice_diff).parameters.values()
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
                "before",
                "after",
                "causes",
                "direction",
                "depth",
                "node_limit",
            ],
        )
        self.assertTrue(
            all(p.default is inspect.Parameter.empty for p in parameters)
        )

    def test_result_is_a_dict_with_before_after_changes(self) -> None:
        store, args = diamond_store()
        result = diff_call(store, args)
        self.assertIsInstance(result, dict)
        self.assertEqual(list(result), ["before", "after", "changes"])
        for side in ("before", "after"):
            self.assertIsInstance(result[side], dict)
            self.assertEqual(list(result[side]), ["nodes", "edges", "gaps"])
            for key in ("nodes", "edges", "gaps"):
                self.assertIsInstance(result[side][key], tuple)
        self.assertIsInstance(result["changes"], tuple)


class CascadeSliceDiffPrefixTests(unittest.TestCase):
    def test_each_side_equals_the_slice_on_its_checkpoint_prefix(self) -> None:
        store, args = diamond_store()
        _, series, scenario, axis, values, mn, mx, req, excl, limit = args
        slice_args_common = (scenario, axis, values, mn, mx, req, excl, limit)

        result = diff_call(
            store, args, before=0, after=1, causes=("a0",),
            direction="forward", depth=99, node_limit=50,
        )
        before_standalone = store.cascade_slice(
            "main", {k: v[:1] for k, v in series.items()},
            *slice_args_common,
            causes=("a0",), direction="forward", depth=99, node_limit=50,
        )
        after_standalone = store.cascade_slice(
            "main", {k: v[:2] for k, v in series.items()},
            *slice_args_common,
            causes=("a0",), direction="forward", depth=99, node_limit=50,
        )
        self.assertEqual(result["before"], before_standalone)
        self.assertEqual(result["after"], after_standalone)

    def test_sides_use_only_the_prefix_not_the_full_plan(self) -> None:
        # before = prefix of one point must not see W, which only exists once
        # the second checkpoint is part of the frozen cascade.
        store, args = diamond_store()
        result = diff_call(store, args, before=0, after=1, causes=("a0",))
        self.assertEqual(
            [(n["cause"], n["key"]) for n in result["before"]["nodes"]],
            [("a0", "x")],
        )
        self.assertEqual(result["before"]["edges"], ())
        self.assertEqual(
            [(n["cause"], n["key"]) for n in result["after"]["nodes"]],
            [("a0", "x"), ("W", "y")],
        )

    def test_equal_indices_yield_equal_sides_and_no_changes(self) -> None:
        store, args = diamond_store()
        result = diff_call(store, args, before=1, after=1, causes=("a0",))
        self.assertEqual(result["before"], result["after"])
        self.assertEqual(result["changes"], ())


class CascadeSliceDiffChangeTests(unittest.TestCase):
    def test_node_and_edge_added_follow_in_group_order(self) -> None:
        store, args = diamond_store()
        result = diff_call(store, args, before=0, after=1, causes=("a0",))
        kinds = [change["kind"] for change in result["changes"]]
        self.assertEqual(kinds, ["node_added", "edge_added"])
        node_added = result["changes"][0]
        self.assertEqual(node_added["identity"], ("W", "y"))
        self.assertIsNone(node_added["before"])
        self.assertEqual(
            (node_added["after"]["cause"], node_added["after"]["key"]),
            ("W", "y"),
        )
        edge_added = result["changes"][1]
        # Edge identity is endpoint node identities, never position numbers.
        self.assertEqual(
            edge_added["identity"],
            ((("a0", "x"), ("W", "y"))),
        )
        self.assertIsNone(edge_added["before"])
        self.assertEqual(
            edge_added["after"]["path"], ("a0", "bu", "W")
        )

    def test_node_changed_keeps_identity_and_shows_both_contents(self) -> None:
        store, scenario, points = changed_store()
        result = store.cascade_slice_diff(
            reference="main",
            series={"a": points},
            base_scenario=scenario,
            axis="total_budget",
            values=(1, 4, 10),
            min_size=0,
            max_size=1,
            required=(),
            exclusive_pairs=(),
            limit=10,
            before=0,
            after=1,
            causes=("a0", "r0"),
            direction="both",
            depth=99,
            node_limit=50,
        )
        self.assertEqual(
            [c["kind"] for c in result["changes"]],
            ["node_changed", "node_changed"],
        )
        first = result["changes"][0]
        self.assertEqual(first["identity"], ("a0", "x"))
        self.assertEqual(first["before"]["checkpoints"], (0,))
        self.assertEqual(first["before"]["affected"], ())
        self.assertEqual(first["after"]["checkpoints"], (1,))
        self.assertEqual(first["after"]["affected"], ("a1",))

    def test_gap_added_is_in_the_gap_group(self) -> None:
        store, scenario, points = gap_store()
        result = store.cascade_slice_diff(
            reference="main",
            series={"a": points},
            base_scenario=scenario,
            axis="total_budget",
            values=(0, 10, 13),
            min_size=0,
            max_size=1,
            required=(),
            exclusive_pairs=(),
            limit=10,
            before=0,
            after=1,
            causes=("p1",),
            direction="both",
            depth=0,
            node_limit=50,
        )
        self.assertEqual(
            [c["kind"] for c in result["changes"]], ["gap_added"]
        )
        change = result["changes"][0]
        self.assertEqual(change["identity"], (0, ()))
        self.assertIsNone(change["before"])
        self.assertEqual(
            change["after"], {"interval": 0, "members": ()}
        )


class SliceChangeClassificationTests(unittest.TestCase):
    """Deterministic coverage of the pure identity/ordering classification."""

    @staticmethod
    def node(cause, key, intervals=(0,), checkpoints=(0,),
             branches=("a",), affected=()):
        return {
            "cause": cause,
            "key": key,
            "intervals": tuple(intervals),
            "checkpoints": tuple(checkpoints),
            "branches": tuple(branches),
            "affected": tuple(affected),
        }

    @staticmethod
    def edge(source, target, path):
        return {"source": source, "target": target, "path": tuple(path)}

    @staticmethod
    def gap(interval, members):
        return {"interval": interval, "members": tuple(members)}

    def changes_for(self, before, after):
        return BranchStore._build_slice_changes(before, after)

    def test_groups_run_nodes_then_edges_then_gaps(self) -> None:
        before = {
            "nodes": (self.node("n1", "k"),),
            "edges": (),
            "gaps": (self.gap(0, ()),),
        }
        after = {"nodes": (), "edges": (), "gaps": ()}
        changes = self.changes_for(before, after)
        self.assertEqual(
            [c["kind"] for c in changes], ["node_removed", "gap_removed"]
        )

    def test_within_each_group_removed_added_changed_in_order(self) -> None:
        # Nodes: removed n0, added n2, changed n1; a/b/c are identity-stable
        # endpoint anchors present on both sides (at different positions).
        before_nodes = (
            self.node("n0", "k"),
            self.node("n1", "k", affected=("e1",)),
            self.node("a", "k"),
            self.node("b", "k"),
            self.node("c", "k"),
        )
        after_nodes = (
            self.node("n1", "k", affected=("e1", "e2")),
            self.node("n2", "k"),
            self.node("a", "k"),
            self.node("b", "k"),
            self.node("c", "k"),
        )
        before = {
            "nodes": before_nodes,
            # n1 is before-position 1; a=2, b=3, c=4.
            "edges": (
                self.edge(1, 2, ("n1", "a")),       # removed (n1 -> a)
                self.edge(1, 3, ("n1", "b", "p")),  # changed path
            ),
            "gaps": (
                self.gap(0, ()),      # removed
                self.gap(2, ("m",)),  # unchanged, anchors the interval
            ),
        }
        after = {
            "nodes": after_nodes,
            # n1 is after-position 0; a=2, b=3, c=4.
            "edges": (
                self.edge(0, 3, ("n1", "b", "Q")),  # changed path (n1 -> b)
                self.edge(0, 4, ("n1", "c")),       # added (n1 -> c)
            ),
            "gaps": (
                self.gap(2, ("m",)),  # unchanged
                self.gap(1, ()),      # added
            ),
        }
        changes = self.changes_for(before, after)
        self.assertEqual(
            [c["kind"] for c in changes],
            [
                "node_removed",
                "node_added",
                "node_changed",
                "edge_removed",
                "edge_added",
                "edge_changed",
                "gap_removed",
                "gap_added",
            ],
        )
        by_kind = {c["kind"]: c for c in changes}
        self.assertEqual(by_kind["node_removed"]["identity"], ("n0", "k"))
        self.assertEqual(by_kind["node_added"]["identity"], ("n2", "k"))
        self.assertEqual(by_kind["node_changed"]["identity"], ("n1", "k"))
        # Edge endpoints compared by node identity despite different slice
        # position numbers on the two sides (n1 is index 1 before, 0 after).
        self.assertEqual(
            by_kind["edge_removed"]["identity"],
            (("n1", "k"), ("a", "k")),
        )
        self.assertEqual(
            by_kind["edge_added"]["identity"],
            (("n1", "k"), ("c", "k")),
        )
        self.assertEqual(
            by_kind["edge_changed"]["identity"],
            (("n1", "k"), ("b", "k")),
        )
        self.assertEqual(
            by_kind["edge_changed"]["before"]["path"], ("n1", "b", "p")
        )
        self.assertEqual(
            by_kind["edge_changed"]["after"]["path"], ("n1", "b", "Q")
        )
        self.assertEqual(by_kind["gap_removed"]["identity"], (0, ()))
        self.assertEqual(by_kind["gap_added"]["identity"], (1, ()))

    def test_record_field_order_and_none_sides(self) -> None:
        before = {"nodes": (self.node("n0", "k"),), "edges": (), "gaps": ()}
        after = {"nodes": (), "edges": (), "gaps": ()}
        (removed,) = self.changes_for(before, after)
        self.assertEqual(list(removed), ["kind", "identity", "before", "after"])
        self.assertIsNotNone(removed["before"])
        self.assertIsNone(removed["after"])

        (added,) = self.changes_for(after, before)
        self.assertEqual(added["kind"], "node_added")
        self.assertIsNone(added["before"])
        self.assertIsNotNone(added["after"])

    def test_identical_slices_produce_no_changes(self) -> None:
        side = {
            "nodes": (self.node("n0", "k"),),
            "edges": (self.edge(0, 0, ("n0",)),),
            "gaps": (self.gap(0, ()),),
        }
        self.assertEqual(self.changes_for(side, side), ())

    def test_change_records_are_detached_from_input_slices(self) -> None:
        before = {
            "nodes": (self.node("n0", "k", affected=("e",)),),
            "edges": (self.edge(0, 0, ("n0",)),),
            "gaps": (self.gap(0, ("m",)),),
        }
        after = {
            "nodes": (
                self.node("n0", "k", affected=("e", "f")),
                self.node("n1", "k"),
            ),
            "edges": (
                self.edge(0, 0, ("n0", "X")),
                self.edge(0, 1, ("n0", "n1")),
            ),
            "gaps": (self.gap(0, ("m",)), self.gap(1, ())),
        }
        changes = self.changes_for(before, after)
        for change in changes:
            for side in ("before", "after"):
                record = change[side]
                if record is None:
                    continue
                # Mutating a record side must not touch the input slices.
                if "affected" in record:
                    record["affected"] += ("z",)
                elif "path" in record:
                    record["path"] += ("z",)
                else:
                    record["members"] += ("z",)
        self.assertEqual(before["nodes"][0]["affected"], ("e",))
        self.assertEqual(after["nodes"][0]["affected"], ("e", "f"))
        self.assertEqual(before["edges"][0]["path"], ("n0",))
        self.assertEqual(before["gaps"][0]["members"], ("m",))


class CascadeSliceDiffCauseTests(unittest.TestCase):
    def test_cause_present_after_but_absent_before_raises(self) -> None:
        store, args = diamond_store()
        # W is a node only on the after (two-point) prefix.
        with self.assertRaises(KeyError) as caught:
            diff_call(store, args, before=0, after=1, causes=("W",))
        self.assertEqual(caught.exception.args[0], "W")

    def test_each_cause_checks_before_side_then_after_side(self) -> None:
        store, args = diamond_store()
        # a0 exists on both sides; W exists only after, so the per-cause
        # before-then-after order reaches the after-side miss for W.
        with self.assertRaises(KeyError) as caught:
            diff_call(store, args, before=0, after=1, causes=("a0", "W"))
        self.assertEqual(caught.exception.args[0], "W")

    def test_first_absent_cause_in_input_order_wins(self) -> None:
        store, args = diamond_store()
        with self.assertRaises(KeyError) as caught:
            diff_call(
                store, args, before=0, after=1, causes=("zzz", "a0")
            )
        self.assertEqual(caught.exception.args[0], "zzz")

    def test_graph_event_that_is_not_a_slice_node_raises(self) -> None:
        store, args = diamond_store()
        # r0 is a real graph event but never becomes a cascade node.
        with self.assertRaises(KeyError):
            diff_call(store, args, before=0, after=1, causes=("r0",))


class CascadeSliceDiffEmptyCausesTests(unittest.TestCase):
    def test_empty_causes_return_two_empty_slices_and_empty_changes(self) -> None:
        store, args = diamond_store()
        result = diff_call(
            store, args, before=0, after=1, causes=(),
            direction="forward", depth=5, node_limit=1,
        )
        empty = {"nodes": (), "edges": (), "gaps": ()}
        self.assertEqual(result, {"before": empty, "after": empty, "changes": ()})

    def test_empty_causes_still_run_state_and_range_checks(self) -> None:
        store, args = diamond_store()
        with self.assertRaises(KeyError):
            diff_call(store, args, reference="ghost", causes=())
        with self.assertRaises(ValueError):
            diff_call(store, args, limit=1, causes=())
        with self.assertRaises(ValueError):
            diff_call(store, args, before=5, after=6, causes=())
        with self.assertRaises(ValueError):
            diff_call(store, args, node_limit=0, causes=())


class CascadeSliceDiffIndexValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store, self.args = diamond_store()  # two checkpoints

    def call(self, **overrides):
        return diff_call(self.store, self.args, causes=("a0",), **overrides)

    def test_indices_must_be_non_bool_ints(self) -> None:
        for bad in (True, False, 1.0, "1", None):
            with self.subTest(before=bad):
                with self.assertRaises(TypeError):
                    self.call(before=bad, after=1)
            with self.subTest(after=bad):
                with self.assertRaises(TypeError):
                    self.call(before=0, after=bad)

    def test_negative_indices_raise_value_error(self) -> None:
        with self.assertRaises(ValueError):
            self.call(before=-1, after=1)
        with self.assertRaises(ValueError):
            self.call(before=0, after=-1)

    def test_out_of_range_indices_raise_value_error(self) -> None:
        with self.assertRaises(ValueError):
            self.call(before=2, after=2)
        with self.assertRaises(ValueError):
            self.call(before=0, after=2)

    def test_before_greater_than_after_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            self.call(before=1, after=0)

    def test_equal_in_range_indices_are_allowed(self) -> None:
        self.assertEqual(self.call(before=0, after=0)["changes"], ())
        self.assertEqual(self.call(before=1, after=1)["changes"], ())

    def test_before_is_validated_before_after(self) -> None:
        # A bad/negative/out-of-range before beats any bad after.
        with self.assertRaises(TypeError):
            self.call(before="x", after="y")
        with self.assertRaises(ValueError):
            self.call(before=-1, after=-1)
        with self.assertRaises(ValueError):
            self.call(before=9, after=-1)

    def test_empty_plan_has_no_valid_index(self) -> None:
        _, args = self.store, self.args
        with self.assertRaises(ValueError):
            diff_call(
                self.store, args, series={}, max_size=0,
                before=0, after=0, causes=(), depth=0, node_limit=1,
            )


class CascadeSliceDiffValidationOrderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        overrides.setdefault("causes", ("a0",))
        return diff_call(self.store, self.args, **overrides)

    def test_ordinary_inputs_validated_first(self) -> None:
        with self.assertRaises(TypeError):
            self.call(reference=1, before="x", causes=[1])
        with self.assertRaises(ValueError):
            self.call(axis="nope", before=-1)
        with self.assertRaises(TypeError):
            self.call(values=[0, 1], before=0)

    def test_indices_validated_before_existing_slice_inputs(self) -> None:
        with self.assertRaises(TypeError):
            self.call(before="x", causes=[1])
        with self.assertRaises(ValueError):
            self.call(before=-1, causes=("a0", "a0"))
        with self.assertRaises(ValueError):
            self.call(before=9, direction="sideways")
        with self.assertRaises(TypeError):
            self.call(before=0, after="x", depth=True)
        # Valid indices, then the existing slice-input order applies.
        with self.assertRaises(TypeError):
            self.call(causes=[1])
        with self.assertRaises(ValueError):
            self.call(causes=("a0", "a0"))
        with self.assertRaises(TypeError):
            self.call(direction=1)
        with self.assertRaises(TypeError):
            self.call(depth=True)
        with self.assertRaises(ValueError):
            self.call(depth=-1)
        with self.assertRaises(ValueError):
            self.call(node_limit=0)

    def test_all_input_validation_precedes_state_lookups(self) -> None:
        with self.assertRaises(TypeError):
            self.call(reference="ghost", before="x")
        with self.assertRaises(ValueError):
            self.call(reference="ghost", before=-1)
        with self.assertRaises(KeyError):
            self.call(reference="ghost")

    def test_branch_and_node_lookup_contracts_are_unchanged(self) -> None:
        # Plans stay aligned on reference nodes; only a series node is
        # unknown, so the historical-node lookup raises KeyError.
        with self.assertRaises(KeyError):
            self.call(
                series={
                    "a": (("r0", "a0"), ("r1", "a0")),
                    "b": (("r0", "a0"), ("r1", "nope2")),
                },
            )

    def test_candidate_cap_still_breaches_with_no_partial_result(self) -> None:
        with self.assertRaises(ValueError):
            self.call(limit=1)


class CascadeSliceDiffNodeLimitTests(unittest.TestCase):
    def test_after_side_over_cap_raises_value_error(self) -> None:
        store, args = diamond_store()
        # before prefix selects one node, after prefix selects two.
        with self.assertRaises(ValueError):
            diff_call(
                store, args, before=0, after=1, causes=("a0",),
                direction="forward", depth=99, node_limit=1,
            )

    def test_limit_covering_the_larger_side_succeeds(self) -> None:
        store, args = diamond_store()
        result = diff_call(
            store, args, before=0, after=1, causes=("a0",),
            direction="forward", depth=99, node_limit=2,
        )
        self.assertEqual(len(result["after"]["nodes"]), 2)

    def test_over_cap_changes_nothing(self) -> None:
        store, args = diamond_store()
        heads = dict(store._heads)
        with self.assertRaises(ValueError):
            diff_call(
                store, args, before=0, after=1, causes=("a0",),
                node_limit=1,
            )
        self.assertEqual(dict(store._heads), heads)


class CascadeSliceDiffIsolationTests(unittest.TestCase):
    def test_result_levels_are_mutually_unshared(self) -> None:
        store, args = diamond_store()
        result = diff_call(store, args, before=0, after=1, causes=("a0",))
        all_records = (
            [id(n) for n in result["before"]["nodes"]]
            + [id(n) for n in result["after"]["nodes"]]
            + [id(e) for e in result["before"]["edges"]]
            + [id(e) for e in result["after"]["edges"]]
            + [id(g) for g in result["before"]["gaps"]]
            + [id(g) for g in result["after"]["gaps"]]
            + [id(c) for c in result["changes"]]
            + [
                id(c[side])
                for c in result["changes"]
                for side in ("before", "after")
                if c[side] is not None
            ]
        )
        self.assertEqual(len(all_records), len(set(all_records)))

    def test_change_sides_do_not_alias_the_slice_records(self) -> None:
        store, scenario, points = changed_store()
        result = store.cascade_slice_diff(
            reference="main",
            series={"a": points},
            base_scenario=scenario,
            axis="total_budget",
            values=(1, 4, 10),
            min_size=0,
            max_size=1,
            required=(),
            exclusive_pairs=(),
            limit=10,
            before=0,
            after=1,
            causes=("a0", "r0"),
            direction="both",
            depth=99,
            node_limit=50,
        )
        before_nodes = {id(n) for n in result["before"]["nodes"]}
        after_nodes = {id(n) for n in result["after"]["nodes"]}
        for change in result["changes"]:
            if change["before"] is not None:
                self.assertNotIn(id(change["before"]), before_nodes)
            if change["after"] is not None:
                self.assertNotIn(id(change["after"]), after_nodes)

    def test_mutating_result_never_affects_later_calls(self) -> None:
        store, args = diamond_store()
        first = diff_call(store, args, before=0, after=1, causes=("a0",))
        pristine = copy.deepcopy(first)
        for side in ("before", "after"):
            for node in first[side]["nodes"]:
                node["branches"] += ("evil",)
                node["affected"] += ("evil",)
            for edge in first[side]["edges"]:
                edge["path"] += ("evil",)
        for change in first["changes"]:
            for side in ("before", "after"):
                if change[side] is not None and "path" in change[side]:
                    change[side]["path"] += ("evil",)
        second = diff_call(store, args, before=0, after=1, causes=("a0",))
        self.assertEqual(second, pristine)


class CascadeSliceDiffReadOnlyTests(unittest.TestCase):
    def test_success_and_failure_change_nothing(self) -> None:
        store, args = diamond_store()
        heads = dict(store._heads)
        appends = copy.deepcopy(store._appends)
        merges = copy.deepcopy(store._merges)
        audit = store.audit_log()

        diff_call(store, args, before=0, after=1, causes=("a0",))
        failures = [
            dict(before="x"),
            dict(before=-1),
            dict(before=9),
            dict(before=1, after=0),
            dict(causes=[1]),
            dict(direction="up"),
            dict(depth=-1),
            dict(node_limit=0),
            dict(causes=("W",)),
            dict(limit=1),
            dict(reference="ghost"),
        ]
        for overrides in failures:
            with self.assertRaises(Exception):
                diff_call(store, args, causes=("a0",), **overrides)

        self.assertEqual(dict(store._heads), heads)
        self.assertEqual(copy.deepcopy(store._appends), appends)
        self.assertEqual(copy.deepcopy(store._merges), merges)
        self.assertEqual(store.audit_log(), audit)

    def test_existing_queries_are_unaffected(self) -> None:
        store, args = diamond_store()
        cascade = store.decision_cascade(*args)
        slice_result = store.cascade_slice(
            *args, causes=("a0",), direction="both", depth=99, node_limit=50
        )
        diff_call(store, args, before=0, after=1, causes=("a0",))
        self.assertEqual(store.decision_cascade(*args), cascade)
        self.assertEqual(
            store.cascade_slice(
                *args, causes=("a0",), direction="both", depth=99,
                node_limit=50,
            ),
            slice_result,
        )

    def test_repeated_calls_are_equal(self) -> None:
        store, args = diamond_store()
        first = diff_call(store, args, before=0, after=1, causes=("a0",))
        second = diff_call(store, args, before=0, after=1, causes=("a0",))
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
