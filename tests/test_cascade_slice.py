import copy
import inspect
import unittest

from city_twin.branches import BranchStore
from city_twin.event_graph import EventGraph


def standard_store() -> BranchStore:
    graph = EventGraph()
    graph.add("root", 0, (), {})
    store = BranchStore(graph)
    store.create("main", "root")
    store.create("feature", "root")
    store.create("side", "root")
    store.append("main", "m1", 1, {"x": 1, "y": 2})
    store.append("feature", "f1", 2, {"x": 2})
    store.append("feature", "f2", 3, {"y": 5})
    store.append("main", "m2", 4, {"z": 7})
    store.append("side", "s1", 5, {"x": 3, "y": -1})
    store.append("side", "s2", 6, {"z": 4})
    return store


STANDARD_SERIES = {
    "feature": (("m1", "f1"), ("m2", "f2")),
    "side": (("m1", "s1"), ("m2", "s2")),
}

STANDARD_SCENARIO = {
    "weights": {"x": 2, "y": 3, "z": 1},
    "total_budget": 100,
    "key_budgets": {},
}
STANDARD_VALUES = (0, 15, 16, 17, 100)


def slice_call(store: BranchStore, **overrides) -> object:
    kwargs = dict(
        reference="main",
        series=STANDARD_SERIES,
        base_scenario=STANDARD_SCENARIO,
        axis="total_budget",
        values=STANDARD_VALUES,
        min_size=0,
        max_size=2,
        required=(),
        exclusive_pairs=(),
        limit=10,
        causes=(),
        direction="both",
        depth=0,
        node_limit=10,
    )
    kwargs.update(overrides)
    return store.cascade_slice(**kwargs)


def chain_store() -> tuple[BranchStore, tuple]:
    """a0 -> b1 -> c2 strict-descendant cause chain over three intervals."""
    graph = EventGraph()
    graph.add("root", 0, (), {})
    store = BranchStore(graph)
    store.create("main", "root")
    store.append("main", "r0", 1, {})
    store.create("a", "root")
    store.append("a", "a0", 2, {"x": 6})
    store.create("b", "a0")
    store.append("b", "b1", 3, {"y": 8})
    store.create("c", "b1")
    store.append("c", "c2", 4, {"z": 9})
    store.append("main", "r1", 5, {})
    series = {
        "a": (("r0", "a0"),),
        "b": (("r0", "b1"),),
        "c": (("r0", "c2"),),
    }
    scenario = {
        "weights": {"x": 1, "y": 1, "z": 1},
        "total_budget": 20,
        "key_budgets": {},
    }
    values = (0, 6, 14, 23)
    args = ("main", series, scenario, "total_budget", values, 0, 3, (), (), 10)
    return store, args


def diamond_store() -> tuple[BranchStore, tuple]:
    """a0 reaches merge W over a cross-branch convergence edge."""
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


class CascadeSliceSignatureTests(unittest.TestCase):
    def test_public_signature_extends_the_cascade_arguments(self) -> None:
        parameters = list(
            inspect.signature(BranchStore.cascade_slice).parameters.values()
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
                "causes",
                "direction",
                "depth",
                "node_limit",
            ],
        )
        self.assertTrue(
            all(p.default is inspect.Parameter.empty for p in parameters)
        )

    def test_result_is_a_dict_with_three_tuple_keys(self) -> None:
        result = slice_call(standard_store(), causes=("m1",))
        self.assertIsInstance(result, dict)
        self.assertEqual(list(result), ["nodes", "edges", "gaps"])
        for name in ("nodes", "edges", "gaps"):
            self.assertIsInstance(result[name], tuple)


class CascadeSliceTraversalTests(unittest.TestCase):
    def test_full_chain_is_a_three_node_two_edge_cascade(self) -> None:
        store, args = chain_store()
        full = store.decision_cascade(*args)
        self.assertEqual(
            [(n["cause"], n["key"]) for n in full["nodes"]],
            [("a0", "x"), ("b1", "y"), ("c2", "z")],
        )
        self.assertEqual(
            [(e["source"], e["target"], e["path"]) for e in full["edges"]],
            [(0, 1, ("a0", "b1")), (1, 2, ("b1", "c2"))],
        )

    def test_forward_expands_from_earlier_to_later(self) -> None:
        store, args = chain_store()

        def run(depth: int) -> list:
            result = store.cascade_slice(
                *args, causes=("a0",), direction="forward",
                depth=depth, node_limit=10,
            )
            return [n["cause"] for n in result["nodes"]]

        self.assertEqual(run(0), ["a0"])
        self.assertEqual(run(1), ["a0", "b1"])
        self.assertEqual(run(2), ["a0", "b1", "c2"])
        self.assertEqual(run(99), ["a0", "b1", "c2"])

    def test_backward_expands_from_later_to_earlier(self) -> None:
        store, args = chain_store()

        def run(depth: int) -> list:
            result = store.cascade_slice(
                *args, causes=("c2",), direction="backward",
                depth=depth, node_limit=10,
            )
            return [n["cause"] for n in result["nodes"]]

        self.assertEqual(run(0), ["c2"])
        self.assertEqual(run(1), ["b1", "c2"])
        self.assertEqual(run(2), ["a0", "b1", "c2"])
        self.assertEqual(run(99), ["a0", "b1", "c2"])

    def test_both_expands_in_two_directions(self) -> None:
        store, args = chain_store()
        result = store.cascade_slice(
            *args, causes=("b1",), direction="both", depth=1, node_limit=10
        )
        self.assertEqual(
            [n["cause"] for n in result["nodes"]], ["a0", "b1", "c2"]
        )
        far = store.cascade_slice(
            *args, causes=("b1",), direction="both", depth=99, node_limit=10
        )
        self.assertEqual(
            [n["cause"] for n in far["nodes"]], ["a0", "b1", "c2"]
        )

    def test_depth_zero_keeps_only_start_nodes(self) -> None:
        store, args = chain_store()
        result = store.cascade_slice(
            *args, causes=("b1",), direction="both", depth=0, node_limit=10
        )
        self.assertEqual([n["cause"] for n in result["nodes"]], ["b1"])
        self.assertEqual(result["edges"], ())

    def test_several_starts_keep_one_copy_each_in_cascade_order(self) -> None:
        store, args = chain_store()
        # Reversed input order does not change the output node order, and a
        # node reachable from both starts is selected exactly once.
        result = store.cascade_slice(
            *args, causes=("c2", "a0"), direction="both", depth=99,
            node_limit=10,
        )
        self.assertEqual(
            [n["cause"] for n in result["nodes"]], ["a0", "b1", "c2"]
        )

    def test_one_cause_event_with_several_keys_starts_every_node(self) -> None:
        graph = EventGraph()
        graph.add("root", 0, (), {})
        store = BranchStore(graph)
        store.create("main", "root")
        store.append("main", "r1", 1, {})
        store.create("a", "root")
        store.append("a", "p1", 2, {"k": 5, "j": 4})
        store.create("b", "p1")
        store.append("b", "bn", 3, {"j": 100})
        store.append("main", "r2", 4, {})
        series = {"a": (("r1", "p1"),), "b": (("r1", "bn"),)}
        scenario = {
            "weights": {"k": 1, "j": 1},
            "total_budget": 100,
            "key_budgets": {},
        }
        args = (
            "main", series, scenario, "total_budget", (0, 9, 104, 120),
            0, 2, (), (), 50,
        )
        result = store.cascade_slice(
            *args, causes=("p1",), direction="both", depth=0, node_limit=10
        )
        self.assertEqual(
            [(n["cause"], n["key"]) for n in result["nodes"]],
            [("p1", "k"), ("p1", "j")],
        )

    def test_cross_branch_convergence_edge_is_walkable_forward(self) -> None:
        store, args = diamond_store()
        full = store.decision_cascade(*args)
        self.assertEqual(
            [(e["source"], e["target"], e["path"]) for e in full["edges"]],
            [(0, 1, ("a0", "bu", "W"))],
        )
        start_only = store.cascade_slice(
            *args, causes=("a0",), direction="forward", depth=0, node_limit=10
        )
        self.assertEqual([n["cause"] for n in start_only["nodes"]], ["a0"])
        reached = store.cascade_slice(
            *args, causes=("a0",), direction="forward", depth=1, node_limit=10
        )
        self.assertEqual(
            [n["cause"] for n in reached["nodes"]], ["a0", "W"]
        )
        # The convergence path content is unchanged even though endpoints
        # remap to slice positions.
        self.assertEqual(
            reached["edges"],
            ({"source": 0, "target": 1, "path": ("a0", "bu", "W")},),
        )

    def test_forward_never_steps_to_unrelated_interval_nodes(self) -> None:
        # The standard fixture's causes all fork from the root, so the full
        # cascade has no edges: no expansion can leave the start node.
        store = standard_store()
        for direction in ("forward", "backward", "both"):
            result = slice_call(
                store, causes=("f2",), direction=direction, depth=99
            )
            self.assertEqual(
                [n["cause"] for n in result["nodes"]], ["f2"], direction
            )
            self.assertEqual(result["edges"], ())


class CascadeSliceContentTests(unittest.TestCase):
    def test_nodes_keep_cascade_order_and_field_shape(self) -> None:
        store, args = chain_store()
        result = store.cascade_slice(
            *args, causes=("c2", "a0"), direction="both", depth=99,
            node_limit=10,
        )
        for node in result["nodes"]:
            self.assertEqual(
                list(node),
                ["cause", "key", "intervals", "checkpoints",
                 "branches", "affected"],
            )
        self.assertEqual(
            [(n["cause"], n["key"]) for n in result["nodes"]],
            [("a0", "x"), ("b1", "y"), ("c2", "z")],
        )

    def test_edges_drop_endpoints_outside_the_slice_and_remap(self) -> None:
        store, args = chain_store()
        result = store.cascade_slice(
            *args, causes=("a0",), direction="forward", depth=1,
            node_limit=10,
        )
        # a0->b1 stays; b1->c2 is dropped because c2 is outside. The kept
        # edge keeps its path and the existing edge order.
        self.assertEqual(
            result["edges"],
            ({"source": 0, "target": 1, "path": ("a0", "b1")},),
        )
        back = store.cascade_slice(
            *args, causes=("c2",), direction="backward", depth=1,
            node_limit=10,
        )
        self.assertEqual(
            back["edges"],
            ({"source": 0, "target": 1, "path": ("b1", "c2")},),
        )

    def test_slice_edges_are_exactly_the_full_edges_with_both_ends_in(self) -> None:
        store, args = chain_store()
        for direction, cause in (
            ("forward", "a0"),
            ("backward", "c2"),
            ("both", "b1"),
        ):
            result = store.cascade_slice(
                *args, causes=(cause,), direction=direction, depth=99,
                node_limit=10,
            )
            full = store.decision_cascade(*args)
            selected = {n["cause"] for n in result["nodes"]}
            expected = [
                edge
                for edge in full["edges"]
                if full["nodes"][edge["source"]]["cause"] in selected
                and full["nodes"][edge["target"]]["cause"] in selected
            ]
            self.assertEqual(len(result["edges"]), len(expected), direction)

    def test_gap_on_a_selected_node_interval_is_kept_in_order(self) -> None:
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
        args = ("main", series, scenario, "total_budget", (0, 10, 13),
                0, 1, (), (), 10)
        full = store.decision_cascade(*args)
        # The p1 node appears at interval 0, the same interval as the gap.
        self.assertEqual(full["nodes"][0]["intervals"], (0,))
        self.assertEqual(
            [(g["interval"], g["members"]) for g in full["gaps"]], [(0, ())]
        )
        result = store.cascade_slice(
            *args, causes=("p1",), direction="both", depth=0, node_limit=10
        )
        self.assertEqual(
            [
                {"interval": g["interval"], "members": tuple(g["members"])}
                for g in result["gaps"]
            ],
            [{"interval": 0, "members": ()}],
        )
        for gap in result["gaps"]:
            self.assertEqual(list(gap), ["interval", "members"])

    def test_gap_on_an_unselected_interval_is_dropped(self) -> None:
        # Interval 0 holds only an empty-combo gap; the cause node lives at
        # interval 1, so slicing from that cause must carry no gap.
        graph = EventGraph()
        graph.add("root", 0, (), {})
        store = BranchStore(graph)
        store.create("main", "root")
        store.create("a", "root")
        store.create("b", "root")
        store.append("main", "r1", 1, {})
        store.append("a", "p1", 2, {"k": 5})
        store.append("a", "c1", 3, {})
        store.create("q", "p1")
        store.append("q", "q1", 4, {})
        store.merge("a", "q", "D", 5, {"k": 7})
        store.append("a", "E", 6, {"k": 7})
        store.append("b", "b1", 2, {"j": 4})
        store.append("b", "b2", 5, {"j": 3})
        store.merge("main", "a", "H", 7, {})
        store.append("main", "r2", 8, {})
        series = {"b": (("r1", "b1"), ("r2", "b2"))}
        scenario = {"weights": {"j": 1}, "total_budget": 13, "key_budgets": {}}
        args = ("main", series, scenario, "j", (0, 1, 17), 0, 1, (), (), 50)
        full = store.decision_cascade(*args)
        self.assertEqual(full["nodes"][0]["intervals"], (1,))
        self.assertEqual([g["interval"] for g in full["gaps"]], [0])
        result = store.cascade_slice(
            *args, causes=("b1",), direction="both", depth=0, node_limit=10
        )
        self.assertEqual([n["cause"] for n in result["nodes"]], ["b1"])
        self.assertEqual(result["gaps"], ())

    def test_gap_intervals_always_match_selected_node_intervals(self) -> None:
        store, args = chain_store()
        full = store.decision_cascade(*args)
        self.assertEqual(full["gaps"], ())
        for cause in ("a0", "b1", "c2"):
            for direction in ("forward", "backward", "both"):
                for depth in (0, 1, 2):
                    result = store.cascade_slice(
                        *args, causes=(cause,), direction=direction,
                        depth=depth, node_limit=10,
                    )
                    intervals = {
                        interval
                        for node in result["nodes"]
                        for interval in node["intervals"]
                    }
                    for gap in result["gaps"]:
                        self.assertIn(gap["interval"], intervals)


class CascadeSliceNodeLimitTests(unittest.TestCase):
    def test_exceeding_node_limit_raises_value_error(self) -> None:
        store, args = chain_store()
        with self.assertRaises(ValueError):
            store.cascade_slice(
                *args, causes=("a0",), direction="forward", depth=99,
                node_limit=2,
            )

    def test_node_limit_counts_all_multi_key_start_nodes(self) -> None:
        graph = EventGraph()
        graph.add("root", 0, (), {})
        store = BranchStore(graph)
        store.create("main", "root")
        store.append("main", "r1", 1, {})
        store.create("a", "root")
        store.append("a", "p1", 2, {"k": 5, "j": 4})
        store.create("b", "p1")
        store.append("b", "bn", 3, {"j": 100})
        store.append("main", "r2", 4, {})
        series = {"a": (("r1", "p1"),), "b": (("r1", "bn"),)}
        scenario = {
            "weights": {"k": 1, "j": 1},
            "total_budget": 100,
            "key_budgets": {},
        }
        args = (
            "main", series, scenario, "total_budget", (0, 9, 104, 120),
            0, 2, (), (), 50,
        )
        with self.assertRaises(ValueError):
            store.cascade_slice(
                *args, causes=("p1",), direction="both", depth=0,
                node_limit=1,
            )

    def test_limit_equal_to_selection_succeeds(self) -> None:
        store, args = chain_store()
        result = store.cascade_slice(
            *args, causes=("a0",), direction="forward", depth=1,
            node_limit=2,
        )
        self.assertEqual(len(result["nodes"]), 2)

    def test_over_cap_call_returns_nothing_and_changes_nothing(self) -> None:
        store, args = chain_store()
        heads = dict(store._heads)
        with self.assertRaises(ValueError):
            store.cascade_slice(
                *args, causes=("a0",), direction="forward", depth=99,
                node_limit=1,
            )
        self.assertEqual(dict(store._heads), heads)


class CascadeSliceEmptyCausesTests(unittest.TestCase):
    def test_empty_causes_return_three_empty_tuples(self) -> None:
        store, args = chain_store()
        result = store.cascade_slice(
            *args, causes=(), direction="forward", depth=5, node_limit=1
        )
        self.assertEqual(result, {"nodes": (), "edges": (), "gaps": ()})

    def test_empty_causes_do_not_trigger_node_limit_but_still_query_state(self) -> None:
        store = standard_store()
        # Unknown branch is still looked up after slice validation.
        with self.assertRaises(KeyError):
            slice_call(store, reference="ghost", causes=())
        # The candidate cap still runs on the state query.
        with self.assertRaises(ValueError):
            slice_call(store, causes=(), limit=3)
        # An invalid node limit is still validated for an empty set.
        with self.assertRaises(ValueError):
            slice_call(store, causes=(), node_limit=0)


class CascadeSliceCausePresenceTests(unittest.TestCase):
    def test_cause_absent_from_cascade_nodes_raises_key_error(self) -> None:
        store, args = chain_store()
        with self.assertRaises(KeyError):
            store.cascade_slice(
                *args, causes=("nope",), direction="forward", depth=1,
                node_limit=10,
            )

    def test_graph_event_that_is_not_a_cause_node_raises_key_error(self) -> None:
        # r0 exists in the graph but never becomes a cascade node.
        store, args = chain_store()
        with self.assertRaises(KeyError):
            store.cascade_slice(
                *args, causes=("r0",), direction="forward", depth=1,
                node_limit=10,
            )

    def test_first_absent_cause_in_input_order_raises(self) -> None:
        store, args = chain_store()
        with self.assertRaises(KeyError) as caught:
            store.cascade_slice(
                *args, causes=("zzz", "a0"), direction="forward", depth=1,
                node_limit=10,
            )
        self.assertEqual(caught.exception.args[0], "zzz")


class CascadeSliceValidationTests(unittest.TestCase):
    def test_causes_must_be_a_tuple(self) -> None:
        store = standard_store()
        for bad in (["m1"], {"m1"}, "m1"):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    slice_call(store, causes=bad)

    def test_cause_elements_must_be_nonempty_strings(self) -> None:
        store = standard_store()
        with self.assertRaises(TypeError):
            slice_call(store, causes=(1,))
        with self.assertRaises(TypeError):
            slice_call(store, causes=(None,))
        with self.assertRaises(ValueError):
            slice_call(store, causes=("",))

    def test_duplicate_cause_raises_value_error(self) -> None:
        store = standard_store()
        with self.assertRaises(ValueError):
            slice_call(store, causes=("m1", "m1"))

    def test_direction_type_and_allowed_values(self) -> None:
        store = standard_store()
        with self.assertRaises(TypeError):
            slice_call(store, causes=("m1",), direction=1)
        for bad in ("", "Forward", "FORWARD", "back", "both "):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    slice_call(store, causes=("m1",), direction=bad)
        for good in ("forward", "backward", "both"):
            slice_call(store, causes=("m1",), direction=good)

    def test_depth_must_be_non_negative_non_bool_int(self) -> None:
        store = standard_store()
        for bad in (True, False, 1.0, "1", None):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    slice_call(store, causes=("m1",), depth=bad)
        with self.assertRaises(ValueError):
            slice_call(store, causes=("m1",), depth=-1)

    def test_node_limit_must_be_positive_non_bool_int(self) -> None:
        store = standard_store()
        for bad in (True, False, 1.0, "1", None):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    slice_call(store, causes=("m1",), node_limit=bad)
        for bad in (0, -1):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    slice_call(store, causes=("m1",), node_limit=bad)

    def test_ordinary_inputs_are_validated_first(self) -> None:
        store = standard_store()
        # A bad ordinary input beats every bad slice input.
        with self.assertRaises(TypeError):
            slice_call(store, reference=1, causes=[1], direction=1,
                       depth=True, node_limit=True)
        with self.assertRaises(TypeError):
            slice_call(store, values=[0, 1], causes=("m1",))
        with self.assertRaises(ValueError):
            slice_call(store, axis="nope", causes=("m1",))

    def test_slice_inputs_are_validated_before_state_lookups(self) -> None:
        store = standard_store()
        # Unknown reference branch (a state lookup) must surface after each
        # slice-input error, but by itself still raise KeyError.
        with self.assertRaises(TypeError):
            slice_call(store, reference="ghost", causes=[1])
        with self.assertRaises(ValueError):
            slice_call(store, reference="ghost", causes=("m1", "m1"))
        with self.assertRaises(TypeError):
            slice_call(store, reference="ghost", direction=1)
        with self.assertRaises(ValueError):
            slice_call(store, reference="ghost", direction="sideways")
        with self.assertRaises(TypeError):
            slice_call(store, reference="ghost", depth=True)
        with self.assertRaises(ValueError):
            slice_call(store, reference="ghost", depth=-1)
        with self.assertRaises(TypeError):
            slice_call(store, reference="ghost", node_limit=True)
        with self.assertRaises(ValueError):
            slice_call(store, reference="ghost", node_limit=0)
        with self.assertRaises(KeyError):
            slice_call(store, reference="ghost", causes=("m1",))

    def test_slice_inputs_validated_in_signature_order(self) -> None:
        store = standard_store()
        # causes before direction: a non-str cause wins over a bad direction.
        with self.assertRaises(TypeError):
            slice_call(store, causes=(1,), direction=1)
        # direction before depth: empty direction wins over negative depth.
        with self.assertRaises(ValueError):
            slice_call(store, causes=("m1",), direction="", depth=-1)
        # depth before node_limit: bool depth wins over a zero limit.
        with self.assertRaises(TypeError):
            slice_call(
                store, causes=("m1",), direction="both",
                depth=True, node_limit=0,
            )
        with self.assertRaises(ValueError):
            slice_call(
                store, causes=("m1",), direction="both",
                depth=-1, node_limit=0,
            )

    def test_ordinary_cascade_errors_are_unchanged(self) -> None:
        store = standard_store()
        invalid = [
            dict(reference=1),
            dict(reference=""),
            dict(reference="ghost"),
            dict(series=()),
            dict(series={"": ()}),
            dict(base_scenario=()),
            dict(base_scenario={}),
            dict(axis=1),
            dict(axis=""),
            dict(axis="nope"),
            dict(values=[0, 1]),
            dict(values=(0,)),
            dict(values=(1, 0)),
            dict(values=(True, 1)),
            dict(min_size=True),
            dict(max_size=1.0),
            dict(required=["feature"]),
            dict(exclusive_pairs=((),)),
            dict(limit=0),
        ]
        for overrides in invalid:
            with self.subTest(**overrides):
                with self.assertRaises(Exception):
                    slice_call(store, **overrides)
        with self.assertRaises(KeyError):
            slice_call(
                store,
                series={"feature": (("nope1", "f1"),)},
                max_size=1,
                causes=("f1",),
            )
        with self.assertRaises(ValueError):
            slice_call(store, causes=("m1",), limit=3)


class CascadeSliceReadOnlyTests(unittest.TestCase):
    def test_success_and_failure_change_nothing(self) -> None:
        store = standard_store()
        heads = dict(store._heads)
        appends = copy.deepcopy(store._appends)
        merges = copy.deepcopy(store._merges)
        replay = store.replay("main")
        audit = store.audit_log()

        slice_call(store, causes=("m1", "f2"), direction="both", depth=4)
        failures = [
            dict(causes=[1]),
            dict(direction="up"),
            dict(depth=-1),
            dict(node_limit=0),
            dict(causes=("zzz",)),
            dict(limit=1),
            dict(reference="ghost", causes=("m1",)),
        ]
        for overrides in failures:
            with self.assertRaises(Exception):
                slice_call(store, **overrides)

        self.assertEqual(dict(store._heads), heads)
        self.assertEqual(copy.deepcopy(store._appends), appends)
        self.assertEqual(copy.deepcopy(store._merges), merges)
        self.assertEqual(store.replay("main"), replay)
        self.assertEqual(store.audit_log(), audit)

    def test_existing_queries_are_unaffected(self) -> None:
        store = standard_store()
        cascade = store.decision_cascade(
            "main", STANDARD_SERIES, STANDARD_SCENARIO, "total_budget",
            STANDARD_VALUES, 0, 2, (), (), 10,
        )
        breakpoints = store.frontier_breakpoints(
            "main", STANDARD_SERIES, STANDARD_SCENARIO, "total_budget",
            STANDARD_VALUES, 0, 2, (), (), 10,
        )
        explanation = store.explain_frontier_breakpoints(
            "main", STANDARD_SERIES, STANDARD_SCENARIO, "total_budget",
            STANDARD_VALUES, 0, 2, (), (), 10,
        )
        slice_call(store, causes=("m1",), direction="both", depth=3)
        self.assertEqual(
            store.decision_cascade(
                "main", STANDARD_SERIES, STANDARD_SCENARIO, "total_budget",
                STANDARD_VALUES, 0, 2, (), (), 10,
            ),
            cascade,
        )
        self.assertEqual(
            store.frontier_breakpoints(
                "main", STANDARD_SERIES, STANDARD_SCENARIO, "total_budget",
                STANDARD_VALUES, 0, 2, (), (), 10,
            ),
            breakpoints,
        )
        self.assertEqual(
            store.explain_frontier_breakpoints(
                "main", STANDARD_SERIES, STANDARD_SCENARIO, "total_budget",
                STANDARD_VALUES, 0, 2, (), (), 10,
            ),
            explanation,
        )

    def test_results_are_detached_and_unshared(self) -> None:
        store, args = chain_store()
        result = store.cascade_slice(
            *args, causes=("a0",), direction="both", depth=99, node_limit=10
        )
        pristine = copy.deepcopy(result)
        for node in result["nodes"]:
            node["cause"] = "evil"
            node["branches"] += ("evil",)
            node["affected"] += ("evil",)
            node["intervals"] += (99,)
        for edge in result["edges"]:
            edge["path"] += ("evil",)
        for gap in result["gaps"]:
            gap["members"] += ("evil",)
        again = store.cascade_slice(
            *args, causes=("a0",), direction="both", depth=99, node_limit=10
        )
        self.assertEqual(again, pristine)

        # No two dict positions inside one result share an object.
        single_ids = (
            [id(node) for node in again["nodes"]]
            + [id(edge) for edge in again["edges"]]
            + [id(gap) for gap in again["gaps"]]
        )
        self.assertEqual(len(single_ids), len(set(single_ids)))

        # A slice never aliases a fresh full-cascade result built first and
        # still alive.
        full = store.decision_cascade(*args)
        all_ids = (
            [id(node) for node in full["nodes"]]
            + [id(edge) for edge in full["edges"]]
            + [id(gap) for gap in full["gaps"]]
        )
        self.assertFalse(set(single_ids) & set(all_ids))

    def test_repeated_calls_are_equal(self) -> None:
        store, args = chain_store()
        first = store.cascade_slice(
            *args, causes=("b1",), direction="both", depth=1, node_limit=10
        )
        second = store.cascade_slice(
            *args, causes=("b1",), direction="both", depth=1, node_limit=10
        )
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
