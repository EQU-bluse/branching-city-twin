import copy
import inspect
import unittest

from city_twin.branches import BranchStore
from city_twin.event_graph import EventGraph

from test_cascade_slice import (
    STANDARD_SCENARIO,
    STANDARD_SERIES,
    STANDARD_VALUES,
    chain_store,
    diamond_store,
    standard_store,
)


STANDARD_ARGS = (
    "main",
    STANDARD_SERIES,
    STANDARD_SCENARIO,
    "total_budget",
    STANDARD_VALUES,
    0,
    2,
    (),
    (),
    10,
)


def prefix_series(series, checkpoint):
    return {name: points[: checkpoint + 1] for name, points in series.items()}


def diff_call(store, *args, **overrides):
    kwargs = dict(
        before=0,
        after=1,
        causes=(),
        direction="both",
        depth=0,
        node_limit=10,
    )
    kwargs.update(overrides)
    return store.cascade_slice_diff(*args, **kwargs)


def gap_store():
    """Interval 0 node exists at prefix 0; the empty-combo gap appears only
    once checkpoint 1 is available."""
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
    series = {"a": (("r1", "c1"), ("r2", "E"))}
    scenario = {"weights": {"k": 1}, "total_budget": 13, "key_budgets": {}}
    args = ("main", series, scenario, "total_budget", (0, 10, 13), 0, 1, (), (), 10)
    return store, args


class CascadeSliceDiffSignatureTests(unittest.TestCase):
    def test_signature_inserts_before_and_after_before_causes(self) -> None:
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
        store = standard_store()
        result = diff_call(store, *STANDARD_ARGS, causes=("m1", "s1"))
        self.assertIsInstance(result, dict)
        self.assertEqual(list(result), ["before", "after", "changes"])
        for name in ("before", "after"):
            self.assertEqual(list(result[name]), ["nodes", "edges", "gaps"])
            for section in result[name].values():
                self.assertIsInstance(section, tuple)
        self.assertIsInstance(result["changes"], tuple)


class CascadeSliceDiffSlicesTests(unittest.TestCase):
    def _assert_prefix_equal(self, store, args, causes, **slice_kwargs):
        reference, series, *rest = args
        scenario, axis, values = rest[0], rest[1], rest[2]
        tail = rest[3:]
        result = diff_call(
            store, *args, before=0, after=1, causes=causes, **slice_kwargs
        )
        for side, checkpoint in (("before", 0), ("after", 1)):
            expected = store.cascade_slice(
                reference,
                prefix_series(series, checkpoint),
                scenario,
                axis,
                values,
                *tail,
                causes=causes,
                **slice_kwargs,
            )
            self.assertEqual(result[side], expected, side)

    def test_each_side_equals_the_slice_on_its_prefix(self) -> None:
        store = standard_store()
        self._assert_prefix_equal(
            store,
            STANDARD_ARGS,
            ("m1", "s1"),
            direction="both",
            depth=2,
            node_limit=20,
        )

    def test_diamond_prefixes_match_and_edge_is_added(self) -> None:
        store, args = diamond_store()
        self._assert_prefix_equal(
            store,
            args,
            ("a0",),
            direction="forward",
            depth=99,
            node_limit=10,
        )

    def test_equal_checkpoints_yield_identical_slices_and_no_changes(self) -> None:
        store = standard_store()
        result = diff_call(
            store,
            *STANDARD_ARGS,
            before=1,
            after=1,
            causes=("m1", "s1", "f2"),
            direction="both",
            depth=1,
            node_limit=10,
        )
        self.assertEqual(result["before"], result["after"])
        self.assertEqual(result["changes"], ())

    def test_empty_causes_return_two_empty_slices(self) -> None:
        store = standard_store()
        result = diff_call(
            store, *STANDARD_ARGS, causes=(), depth=9, node_limit=1
        )
        empty = {"nodes": (), "edges": (), "gaps": ()}
        self.assertEqual(
            result, {"before": empty, "after": dict(empty), "changes": ()}
        )

    def test_empty_causes_still_run_state_checks(self) -> None:
        store = standard_store()
        with self.assertRaises(KeyError):
            store.cascade_slice_diff(
                "ghost",
                STANDARD_SERIES,
                STANDARD_SCENARIO,
                "total_budget",
                STANDARD_VALUES,
                0,
                2,
                (),
                (),
                10,
                0,
                1,
                (),
                "both",
                0,
                10,
            )
        # Candidate cap still runs on the state query.
        with self.assertRaises(ValueError):
            store.cascade_slice_diff(
                *STANDARD_ARGS[:-1], 3, 0, 1, (), "both", 0, 10
            )


class CascadeSliceDiffChangeTests(unittest.TestCase):
    def test_records_are_kind_identity_before_after(self) -> None:
        store = standard_store()
        result = diff_call(
            store,
            *STANDARD_ARGS,
            causes=("m1", "s1"),
            direction="both",
            depth=0,
            node_limit=10,
        )
        for record in result["changes"]:
            self.assertEqual(list(record), ["kind", "identity", "before", "after"])

    def test_node_addition_uses_cause_key_identity(self) -> None:
        # The reached node W appears only after checkpoint 1, while the
        # start cause a0 exists on both prefixes.
        store, args = diamond_store()
        result = diff_call(
            store,
            *args,
            causes=("a0",),
            direction="forward",
            depth=99,
            node_limit=10,
        )
        added = [c for c in result["changes"] if c["kind"] == "node_added"]
        self.assertEqual([c["identity"] for c in added], [("W", "y")])
        record = added[0]
        self.assertIsNone(record["before"])
        self.assertEqual(record["after"]["cause"], "W")
        self.assertEqual(record["after"]["key"], "y")

    def test_node_content_change_is_classified_changed(self) -> None:
        store = standard_store()
        result = diff_call(
            store,
            *STANDARD_ARGS,
            causes=("m1", "s1"),
            direction="both",
            depth=0,
            node_limit=10,
        )
        identities = sorted(
            c["identity"]
            for c in result["changes"]
            if c["kind"] == "node_changed"
        )
        self.assertEqual(identities, [("m1", "y"), ("s1", "y")])
        for record in result["changes"]:
            if record["kind"] == "node_changed":
                self.assertIsNotNone(record["before"])
                self.assertIsNotNone(record["after"])

    def test_edge_addition_identity_uses_node_identities_not_positions(
        self,
    ) -> None:
        store, args = diamond_store()
        result = diff_call(
            store,
            *args,
            causes=("a0",),
            direction="forward",
            depth=99,
            node_limit=10,
        )
        edges = [c for c in result["changes"] if c["kind"] == "edge_added"]
        self.assertEqual(len(edges), 1)
        record = edges[0]
        # Identity is the two endpoint (cause, key) pairings, never
        # slice position numbers.
        self.assertEqual(
            record["identity"],
            ((("a0", "x"), ("W", "y"))),
        )
        self.assertIsNone(record["before"])
        # The present side keeps its own position remapping and path.
        self.assertEqual(
            record["after"],
            {"source": 0, "target": 1, "path": ("a0", "bu", "W")},
        )

    def test_gap_addition_uses_interval_and_members_identity(self) -> None:
        store, args = gap_store()
        result = diff_call(
            store,
            *args,
            causes=("p1",),
            direction="both",
            depth=0,
            node_limit=10,
        )
        gaps = [c for c in result["changes"] if c["kind"] == "gap_added"]
        self.assertEqual([c["identity"] for c in gaps], [(0, ())])
        self.assertIsNone(gaps[0]["before"])
        self.assertEqual(gaps[0]["after"], {"interval": 0, "members": ()})

    def test_categories_are_ordered_nodes_edges_gaps(self) -> None:
        store, args = diamond_store()
        result = diff_call(
            store,
            *args,
            causes=("a0",),
            direction="forward",
            depth=99,
            node_limit=10,
        )
        order = ["node", "edge", "gap"]
        ranks = [order.index(c["kind"].split("_")[0]) for c in result["changes"]]
        self.assertEqual(ranks, sorted(ranks))

    def test_within_category_removed_then_added_then_changed(self) -> None:
        before = {
            "nodes": (
                {
                    "cause": "a",
                    "key": "x",
                    "intervals": (0,),
                    "checkpoints": (0,),
                    "branches": ("b",),
                    "affected": (),
                },
                {
                    "cause": "b",
                    "key": "x",
                    "intervals": (0,),
                    "checkpoints": (0,),
                    "branches": ("b",),
                    "affected": (),
                },
                {
                    "cause": "c",
                    "key": "x",
                    "intervals": (0,),
                    "checkpoints": (0,),
                    "branches": ("b",),
                    "affected": (),
                },
            ),
            "edges": (),
            "gaps": (),
        }
        after = copy.deepcopy(before)
        # 'a' is removed, 'd' added, 'c' content changed; 'b' stays.
        after["nodes"] = (
            dict(before["nodes"][1]),
            {
                "cause": "c",
                "key": "x",
                "intervals": (0, 1),
                "checkpoints": (0,),
                "branches": ("b",),
                "affected": (),
            },
            {
                "cause": "d",
                "key": "x",
                "intervals": (0,),
                "checkpoints": (0,),
                "branches": ("b",),
                "affected": (),
            },
        )
        store = standard_store()
        changes = store._diff_cascade_slices(before, after)
        self.assertEqual(
            [(c["kind"], c["identity"]) for c in changes],
            [
                ("node_removed", ("a", "x")),
                ("node_added", ("d", "x")),
                ("node_changed", ("c", "x")),
            ],
        )

    def test_edge_identity_survives_position_remap(self) -> None:
        # The same endpoint identities at different positions is a kept edge
        # whose content (positions) changed, not a remove/add pair.
        def node(cause):
            return {
                "cause": cause,
                "key": "x",
                "intervals": (0,),
                "checkpoints": (0,),
                "branches": ("b",),
                "affected": (),
            }

        before = {
            "nodes": (node("a"), node("b")),
            "edges": (
                {"source": 0, "target": 1, "path": ("a", "b")},
            ),
            "gaps": (),
        }
        after = {
            "nodes": (node("new"), node("a"), node("b")),
            "edges": (
                {"source": 1, "target": 2, "path": ("a", "b")},
            ),
            "gaps": (),
        }
        store = standard_store()
        changes = store._diff_cascade_slices(before, after)
        kinds = {c["kind"] for c in changes}
        self.assertIn("node_added", kinds)
        edge_changes = [c for c in changes if c["kind"].startswith("edge_")]
        self.assertEqual(
            [(c["kind"], c["identity"]) for c in edge_changes],
            [("edge_changed", (("a", "x"), ("b", "x")))],
        )
        self.assertEqual(
            edge_changes[0]["before"],
            {"source": 0, "target": 1, "path": ("a", "b")},
        )
        self.assertEqual(
            edge_changes[0]["after"],
            {"source": 1, "target": 2, "path": ("a", "b")},
        )


class CascadeSliceDiffCausePresenceTests(unittest.TestCase):
    def test_cause_missing_before_raises_even_when_present_after(self) -> None:
        store = standard_store()
        # f2 is a cascade node only after checkpoint 1.
        with self.assertRaises(KeyError) as caught:
            diff_call(
                store,
                *STANDARD_ARGS,
                causes=("f2",),
                direction="both",
                depth=0,
                node_limit=10,
            )
        self.assertEqual(caught.exception.args[0], "f2")

    def test_cause_order_checks_before_side_first_per_cause(self) -> None:
        store = standard_store()
        # 'f2' (input order first) is absent before -> it, not any
        # later cause, is reported.
        with self.assertRaises(KeyError) as caught:
            diff_call(
                store,
                *STANDARD_ARGS,
                causes=("f2", "m1"),
                direction="both",
                depth=0,
                node_limit=10,
            )
        self.assertEqual(caught.exception.args[0], "f2")

    def test_node_cap_breach_on_either_side_returns_nothing(self) -> None:
        store = standard_store()
        with self.assertRaises(ValueError):
            diff_call(
                store,
                *STANDARD_ARGS,
                causes=("m1", "s1"),
                direction="both",
                depth=0,
                node_limit=1,
            )

    def test_empty_causes_never_triplicate_node_cap(self) -> None:
        store = standard_store()
        result = diff_call(
            store, *STANDARD_ARGS, causes=(), node_limit=1
        )
        self.assertEqual(result["changes"], ())


class CascadeSliceDiffValidationTests(unittest.TestCase):
    def call(self, store, **overrides):
        kwargs = dict(
            before=0,
            after=1,
            causes=(),
            direction="both",
            depth=0,
            node_limit=10,
        )
        kwargs.update(overrides)
        return store.cascade_slice_diff(*STANDARD_ARGS, **kwargs)

    def test_indices_must_be_non_bool_ints(self) -> None:
        store = standard_store()
        for bad in (True, False, 1.0, "1", None):
            with self.subTest(before=bad):
                with self.assertRaises(TypeError):
                    self.call(store, before=bad)
            with self.subTest(after=bad):
                with self.assertRaises(TypeError):
                    self.call(store, after=bad)

    def test_negative_indices_raise_value_error(self) -> None:
        store = standard_store()
        with self.assertRaises(ValueError):
            self.call(store, before=-1, after=0)
        with self.assertRaises(ValueError):
            self.call(store, before=0, after=-1)

    def test_before_greater_than_after_raises(self) -> None:
        store = standard_store()
        with self.assertRaises(ValueError):
            self.call(store, before=1, after=0)

    def test_out_of_range_indices_raise(self) -> None:
        store = standard_store()  # two checkpoints: 0 and 1
        with self.assertRaises(ValueError):
            self.call(store, before=2, after=2)
        with self.assertRaises(ValueError):
            self.call(store, before=0, after=2)

    def test_ordinary_inputs_validated_before_indices(self) -> None:
        store = standard_store()
        with self.assertRaises(TypeError):
            store.cascade_slice_diff(
                1,
                STANDARD_SERIES,
                STANDARD_SCENARIO,
                "total_budget",
                STANDARD_VALUES,
                0,
                2,
                (),
                (),
                10,
                -1,
                -1,
                (),
                1,
                True,
                True,
            )

    def test_indices_validated_before_slice_inputs(self) -> None:
        store = standard_store()
        with self.assertRaises(ValueError):
            self.call(
                store,
                before=-1,
                after=-1,
                causes=(1,),
                direction=1,
                depth=True,
                node_limit=0,
            )

    def test_before_validated_before_after(self) -> None:
        store = standard_store()
        with self.assertRaises(TypeError):
            self.call(store, before=True, after=-1)
        with self.assertRaises(ValueError):
            self.call(store, before=-1, after=-1)

    def test_slice_range_keeps_cascade_slice_contracts(self) -> None:
        store = standard_store()
        with self.assertRaises(TypeError):
            self.call(store, causes=[1])
        with self.assertRaises(ValueError):
            self.call(store, causes=("m1", "m1"))
        with self.assertRaises(TypeError):
            self.call(store, causes=("m1",), direction=1)
        with self.assertRaises(ValueError):
            self.call(store, causes=("m1",), direction="sideways")
        with self.assertRaises(TypeError):
            self.call(store, causes=("m1",), depth=True)
        with self.assertRaises(ValueError):
            self.call(store, causes=("m1",), depth=-1)
        with self.assertRaises(TypeError):
            self.call(store, causes=("m1",), node_limit=True)
        with self.assertRaises(ValueError):
            self.call(store, causes=("m1",), node_limit=0)

    def test_unknown_branch_and_nodes_keep_key_error(self) -> None:
        store = standard_store()
        with self.assertRaises(KeyError):
            store.cascade_slice_diff(
                "ghost",
                STANDARD_SERIES,
                STANDARD_SCENARIO,
                "total_budget",
                STANDARD_VALUES,
                0,
                2,
                (),
                (),
                10,
                0,
                1,
                (),
                "both",
                0,
                10,
            )


class CascadeSliceDiffReadOnlyTests(unittest.TestCase):
    def test_success_and_failure_change_nothing(self) -> None:
        store, args = diamond_store()
        heads = dict(store._heads)
        appends = copy.deepcopy(store._appends)
        merges = copy.deepcopy(store._merges)
        audit = store.audit_log()

        diff_call(
            store,
            *args,
            causes=("a0",),
            direction="forward",
            depth=99,
            node_limit=10,
        )
        failures = [
            dict(before=-1, after=0),
            dict(before=0, after=9),
            dict(before=1, after=0),
            dict(causes=("zzz",)),
            dict(node_limit=0, causes=("a0",), depth=0),
            dict(reference="ghost"),
        ]
        for overrides in failures:
            kwargs = dict(
                before=0,
                after=1,
                causes=(),
                direction="both",
                depth=0,
                node_limit=10,
            )
            kwargs.update(overrides)
            with self.assertRaises(Exception):
                store.cascade_slice_diff(*args, **kwargs)

        self.assertEqual(dict(store._heads), heads)
        self.assertEqual(copy.deepcopy(store._appends), appends)
        self.assertEqual(copy.deepcopy(store._merges), merges)
        self.assertEqual(store.audit_log(), audit)

    def test_results_detached_unshared_and_stable(self) -> None:
        store, args = diamond_store()
        first = diff_call(
            store,
            *args,
            causes=("a0",),
            direction="forward",
            depth=99,
            node_limit=10,
        )
        pristine = copy.deepcopy(first)
        for side in ("before", "after"):
            for node in first[side]["nodes"]:
                node["cause"] = "evil"
                node["branches"] += ("evil",)
                node["intervals"] += (99,)
            for edge in first[side]["edges"]:
                edge["path"] += ("evil",)
        for record in first["changes"]:
            record["kind"] = "evil"
            for side in ("before", "after"):
                value = record[side]
                if isinstance(value, dict):
                    if "path" in value:
                        value["path"] += ("evil",)
                    elif "cause" in value:
                        value["cause"] = "evil"
        second = diff_call(
            store,
            *args,
            causes=("a0",),
            direction="forward",
            depth=99,
            node_limit=10,
        )
        self.assertEqual(second, pristine)

        # No two records inside one result share an identity-bearing object.
        ids_ = []
        for side in ("before", "after"):
            ids_ += [id(n) for n in second[side]["nodes"]]
            ids_ += [id(e) for e in second[side]["edges"]]
            ids_ += [id(g) for g in second[side]["gaps"]]
        for record in second["changes"]:
            ids_ += [id(record)]
            for side in ("before", "after"):
                if record[side] is not None:
                    ids_.append(id(record[side]))
        self.assertEqual(len(ids_), len(set(ids_)))

    def test_existing_queries_are_unaffected(self) -> None:
        store, args = chain_store()
        cascade = store.decision_cascade(*args)
        single = store.cascade_slice(
            *args,
            causes=("a0", "b1", "c2"),
            direction="both",
            depth=99,
            node_limit=10,
        )
        diff_call(
            store,
            *args,
            before=0,
            after=0,
            causes=("a0", "b1", "c2"),
            direction="both",
            depth=99,
            node_limit=10,
        )
        self.assertEqual(store.decision_cascade(*args), cascade)
        self.assertEqual(
            store.cascade_slice(
                *args,
                causes=("a0", "b1", "c2"),
                direction="both",
                depth=99,
                node_limit=10,
            ),
            single,
        )


if __name__ == "__main__":
    unittest.main()
