import copy
import inspect
import unittest

from city_twin.branches import BranchStore
from city_twin.event_graph import EventGraph


def make_chain_store() -> tuple[BranchStore, dict, dict]:
    """Four-series chain: a0 -> b1 -> c2 -> d3 across four intervals."""
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
    store.create("d", "c2")
    store.append("d", "d3", 5, {"w": 11})
    store.append("main", "r1", 6, {})
    store.append("main", "r2", 7, {})
    store.append("main", "r3", 8, {})
    series = {
        "a": (("r0", "a0"), ("r1", "a0"), ("r2", "a0"), ("r3", "a0")),
        "b": (("r0", "b1"), ("r1", "b1"), ("r2", "b1"), ("r3", "b1")),
        "c": (("r0", "c2"), ("r1", "c2"), ("r2", "c2"), ("r3", "c2")),
        "d": (("r0", "d3"), ("r1", "d3"), ("r2", "d3"), ("r3", "d3")),
    }
    scenario = {
        "weights": {"x": 1, "y": 1, "z": 1, "w": 1},
        "total_budget": 40,
        "key_budgets": {},
    }
    return store, series, scenario


def chain_slice(
    store: BranchStore,
    series: dict,
    scenario: dict,
    causes: tuple,
    direction: str,
    depth: int,
    max_nodes: int = 10,
) -> dict:
    return store.cascade_slice(
        "main",
        series,
        scenario,
        "total_budget",
        (0, 10, 20, 30, 40),
        0,
        4,
        (),
        (),
        40,
        causes,
        direction,
        depth,
        max_nodes,
    )


def make_gap_store() -> tuple[BranchStore, dict, dict]:
    """Two intervals: a gap plus the p1 node at interval 0, j1 at 1."""
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
    store.create("j", "root")
    store.append("j", "j1", 9, {"x": 30})
    series = {"a": (("r1", "c1"), ("r2", "E")), "j": (("r1", "j1"), ("r2", "j1"))}
    scenario = {
        "weights": {"k": 1, "x": 1},
        "total_budget": 60,
        "key_budgets": {},
    }
    return store, series, scenario


def gap_slice(
    store: BranchStore,
    series: dict,
    scenario: dict,
    causes: tuple,
    direction: str = "both",
    depth: int = 5,
    max_nodes: int = 10,
) -> dict:
    return store.cascade_slice(
        "main",
        series,
        scenario,
        "total_budget",
        tuple(range(0, 61)),
        0,
        2,
        (),
        (),
        60,
        causes,
        direction,
        depth,
        max_nodes,
    )


def make_multikey_store() -> tuple[BranchStore, dict, dict]:
    """One cause event g1 named under both the x and the y state key."""
    graph = EventGraph()
    graph.add("root", 0, (), {})
    store = BranchStore(graph)
    store.create("main", "root")
    store.append("main", "r0", 1, {})
    store.create("g", "root")
    store.append("g", "g1", 2, {"x": 5, "y": 8})
    store.create("h", "g1")
    store.append("h", "h1", 3, {"x": 4})
    store.append("main", "r1", 4, {})
    series = {"g": (("r0", "g1"), ("r1", "g1")), "h": (("r0", "h1"), ("r1", "h1"))}
    scenario = {
        "weights": {"x": 1, "y": 1},
        "total_budget": 30,
        "key_budgets": {},
    }
    return store, series, scenario


class CascadeSliceSignatureTests(unittest.TestCase):
    def test_public_signature_appends_slice_parameters(self) -> None:
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
                "max_nodes",
            ],
        )
        self.assertTrue(
            all(p.default is inspect.Parameter.empty for p in parameters)
        )

    def test_result_is_a_dict_with_three_keys(self) -> None:
        store, series, scenario = make_chain_store()
        result = chain_slice(store, series, scenario, ("a0",), "forward", 1)
        self.assertIsInstance(result, dict)
        self.assertEqual(list(result), ["nodes", "edges", "gaps"])
        for name in ("nodes", "edges", "gaps"):
            self.assertIsInstance(result[name], tuple)


class CascadeSliceDirectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store, self.series, self.scenario = make_chain_store()

    def causes_of(self, result: dict) -> list[tuple[str, str]]:
        return [(node["cause"], node["key"]) for node in result["nodes"]]

    def test_forward_expands_toward_later_causes(self) -> None:
        result = chain_slice(
            self.store, self.series, self.scenario, ("b1",), "forward", 1
        )
        self.assertEqual(self.causes_of(result), [("b1", "y"), ("c2", "z")])

    def test_backward_expands_toward_earlier_causes(self) -> None:
        result = chain_slice(
            self.store, self.series, self.scenario, ("c2",), "backward", 1
        )
        self.assertEqual(
            self.causes_of(result), [("a0", "x"), ("b1", "y"), ("c2", "z")]
        )

    def test_both_expands_in_both_directions(self) -> None:
        result = chain_slice(
            self.store, self.series, self.scenario, ("b1",), "both", 1
        )
        self.assertEqual(
            self.causes_of(result), [("a0", "x"), ("b1", "y"), ("c2", "z")]
        )

    def test_forward_from_last_cause_keeps_only_the_start(self) -> None:
        result = chain_slice(
            self.store, self.series, self.scenario, ("d3",), "forward", 3
        )
        self.assertEqual(self.causes_of(result), [("d3", "w")])

    def test_depth_zero_keeps_only_start_points(self) -> None:
        result = chain_slice(
            self.store, self.series, self.scenario, ("a0",), "both", 0
        )
        self.assertEqual(self.causes_of(result), [("a0", "x")])
        self.assertEqual(result["edges"], ())

    def test_depth_counts_edges(self) -> None:
        # b1 -> c2 is one edge, b1 -> c2 -> d3 is two.
        one = chain_slice(
            self.store, self.series, self.scenario, ("b1",), "forward", 1
        )
        self.assertEqual(self.causes_of(one), [("b1", "y"), ("c2", "z")])
        two = chain_slice(
            self.store, self.series, self.scenario, ("b1",), "forward", 2
        )
        self.assertEqual(
            self.causes_of(two), [("b1", "y"), ("c2", "z"), ("d3", "w")]
        )

    def test_depth_limits_backward_expansion(self) -> None:
        # d3 has direct backward edges to a0 and c2; b1 is two edges away.
        one = chain_slice(
            self.store, self.series, self.scenario, ("d3",), "backward", 1
        )
        self.assertEqual(
            self.causes_of(one), [("a0", "x"), ("c2", "z"), ("d3", "w")]
        )
        two = chain_slice(
            self.store, self.series, self.scenario, ("d3",), "backward", 2
        )
        self.assertEqual(
            self.causes_of(two),
            [("a0", "x"), ("b1", "y"), ("c2", "z"), ("d3", "w")],
        )

    def test_nodes_keep_full_cascade_order(self) -> None:
        result = chain_slice(
            self.store, self.series, self.scenario, ("d3",), "backward", 2
        )
        self.assertEqual(
            self.causes_of(result),
            [("a0", "x"), ("b1", "y"), ("c2", "z"), ("d3", "w")],
        )

    def test_multiple_starts_reaching_one_node_keep_it_once(self) -> None:
        result = chain_slice(
            self.store, self.series, self.scenario,
            ("c2", "b1"), "backward", 1,
        )
        causes = self.causes_of(result)
        self.assertEqual(len(causes), len(set(causes)))
        self.assertEqual(causes, [("a0", "x"), ("b1", "y"), ("c2", "z")])


class CascadeSliceEdgeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store, self.series, self.scenario = make_chain_store()

    def test_edges_remapped_to_slice_positions(self) -> None:
        result = chain_slice(
            self.store, self.series, self.scenario, ("b1",), "forward", 2
        )
        self.assertEqual(
            [(node["cause"], node["key"]) for node in result["nodes"]],
            [("b1", "y"), ("c2", "z"), ("d3", "w")],
        )
        self.assertEqual(
            [(edge["source"], edge["target"]) for edge in result["edges"]],
            [(0, 1), (1, 2)],
        )

    def test_edge_paths_and_order_are_unchanged(self) -> None:
        full = self.store.decision_cascade(
            "main",
            self.series,
            self.scenario,
            "total_budget",
            (0, 10, 20, 30, 40),
            0,
            4,
            (),
            (),
            40,
        )
        result = chain_slice(
            self.store, self.series, self.scenario, ("d3",), "backward", 2
        )
        self.assertEqual(result["nodes"], full["nodes"])
        self.assertEqual(result["edges"], full["edges"])
        paths = [edge["path"] for edge in result["edges"]]
        self.assertEqual(
            paths,
            [
                ("a0", "b1"),
                ("a0", "b1", "c2"),
                ("a0", "b1", "c2", "d3"),
                ("b1", "c2"),
                ("c2", "d3"),
            ],
        )

    def test_edges_with_unselected_end_are_dropped(self) -> None:
        result = chain_slice(
            self.store, self.series, self.scenario, ("b1",), "forward", 1
        )
        # d3 is out of depth, so the c2 -> d3 edge cannot survive.
        self.assertEqual(
            [(edge["source"], edge["target"], edge["path"])
             for edge in result["edges"]],
            [(0, 1, ("b1", "c2"))],
        )

    def test_merge_convergence_edge_stays_valid(self) -> None:
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
        result = store.cascade_slice(
            "main", series, scenario, "total_budget", (0, 10, 20),
            0, 2, (), (), 10, ("W",), "backward", 1, 10,
        )
        self.assertEqual(
            [(node["cause"], node["key"]) for node in result["nodes"]],
            [("a0", "x"), ("W", "y")],
        )
        self.assertEqual(
            [(edge["source"], edge["target"], edge["path"])
             for edge in result["edges"]],
            [(0, 1, ("a0", "bu", "W"))],
        )


class CascadeSliceGapTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store, self.series, self.scenario = make_gap_store()

    def test_gap_kept_when_selected_node_shares_its_interval(self) -> None:
        result = gap_slice(self.store, self.series, self.scenario, ("p1",))
        self.assertEqual(
            [(node["cause"], node["key"]) for node in result["nodes"]],
            [("p1", "k")],
        )
        self.assertEqual(
            [(gap["interval"], gap["members"]) for gap in result["gaps"]],
            [(0, ())],
        )

    def test_gap_dropped_when_no_selected_node_shares_its_interval(self) -> None:
        result = gap_slice(self.store, self.series, self.scenario, ("j1",))
        self.assertEqual(
            [(node["cause"], node["key"]) for node in result["nodes"]],
            [("j1", "x")],
        )
        self.assertEqual(result["gaps"], ())

    def test_gap_dicts_are_fresh_copies(self) -> None:
        result = gap_slice(self.store, self.series, self.scenario, ("p1",))
        gap = result["gaps"][0]
        self.assertEqual(list(gap), ["interval", "members"])
        gap["members"] += ("evil",)
        again = gap_slice(self.store, self.series, self.scenario, ("p1",))
        self.assertEqual(again["gaps"][0]["members"], ())


class CascadeSliceMultiKeyTests(unittest.TestCase):
    def test_one_cause_event_starts_every_matching_node(self) -> None:
        store, series, scenario = make_multikey_store()
        result = store.cascade_slice(
            "main", series, scenario, "total_budget", tuple(range(0, 31)),
            0, 2, (), (), 60, ("g1",), "both", 0, 10,
        )
        # Depth zero still keeps every start: g1 is the cause of both the
        # x and the y evidence node.
        self.assertEqual(
            [(node["cause"], node["key"]) for node in result["nodes"]],
            [("g1", "y"), ("g1", "x")],
        )


class CascadeSliceNodeCapTests(unittest.TestCase):
    def test_node_cap_counts_every_selected_node(self) -> None:
        store, series, scenario = make_chain_store()
        # backward depth 1 from d3 selects a0, c2 and d3.
        result = chain_slice(
            store, series, scenario, ("d3",), "backward", 1, max_nodes=3
        )
        self.assertEqual(len(result["nodes"]), 3)
        with self.assertRaises(ValueError):
            chain_slice(
                store, series, scenario, ("d3",), "backward", 1, max_nodes=2
            )

    def test_node_cap_exceeded_returns_no_partial_result(self) -> None:
        store, series, scenario = make_chain_store()
        with self.assertRaises(ValueError):
            chain_slice(
                store, series, scenario, ("a0",), "forward", 3, max_nodes=3
            )

    def test_node_cap_not_triggered_by_empty_causes(self) -> None:
        store, series, scenario = make_chain_store()
        result = chain_slice(
            store, series, scenario, (), "forward", 3, max_nodes=1
        )
        self.assertEqual(result, {"nodes": (), "edges": (), "gaps": ()})


class CascadeSliceEmptyCauseTests(unittest.TestCase):
    def test_empty_causes_return_three_empty_tuples(self) -> None:
        store, series, scenario = make_chain_store()
        result = chain_slice(store, series, scenario, (), "both", 2)
        self.assertEqual(result, {"nodes": (), "edges": (), "gaps": ()})

    def test_empty_causes_still_run_state_queries(self) -> None:
        store, series, scenario = make_chain_store()
        # Unknown reference branch: the state lookup still happens.
        with self.assertRaises(KeyError):
            store.cascade_slice(
                "ghost", series, scenario, "total_budget",
                (0, 10, 20, 30, 40), 0, 4, (), (), 40,
                (), "forward", 1, 10,
            )
        # The candidate cap still applies.
        with self.assertRaises(ValueError):
            store.cascade_slice(
                "main", series, scenario, "total_budget",
                (0, 10, 20, 30, 40), 0, 4, (), (), 1,
                (), "forward", 1, 10,
            )


class CascadeSliceValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store, self.series, self.scenario = make_chain_store()

    def slice(self, **overrides) -> object:
        kwargs = dict(
            reference="main",
            series=self.series,
            base_scenario=self.scenario,
            axis="total_budget",
            values=(0, 10, 20, 30, 40),
            min_size=0,
            max_size=4,
            required=(),
            exclusive_pairs=(),
            limit=40,
            causes=("a0",),
            direction="forward",
            depth=1,
            max_nodes=10,
        )
        kwargs.update(overrides)
        return self.store.cascade_slice(**kwargs)

    def test_causes_container_and_element_types(self) -> None:
        with self.assertRaises(TypeError):
            self.slice(causes=["a0"])
        with self.assertRaises(TypeError):
            self.slice(causes="a0")
        with self.assertRaises(TypeError):
            self.slice(causes=(1,))
        with self.assertRaises(TypeError):
            self.slice(causes=(None,))

    def test_causes_empty_and_duplicate_names(self) -> None:
        with self.assertRaises(ValueError):
            self.slice(causes=("",))
        with self.assertRaises(ValueError):
            self.slice(causes=("a0", "a0"))

    def test_direction_type_and_value(self) -> None:
        with self.assertRaises(TypeError):
            self.slice(direction=1)
        with self.assertRaises(TypeError):
            self.slice(direction=None)
        with self.assertRaises(ValueError):
            self.slice(direction="")
        with self.assertRaises(ValueError):
            self.slice(direction="sideways")

    def test_direction_case_variants_are_rejected(self) -> None:
        for bad in ("Forward", "BACKWARD", "Both"):
            with self.subTest(direction=bad):
                with self.assertRaises(ValueError):
                    self.slice(direction=bad)

    def test_depth_and_max_nodes_types(self) -> None:
        with self.assertRaises(TypeError):
            self.slice(depth=True)
        with self.assertRaises(TypeError):
            self.slice(depth=1.0)
        with self.assertRaises(TypeError):
            self.slice(depth="1")
        with self.assertRaises(TypeError):
            self.slice(max_nodes=False)
        with self.assertRaises(TypeError):
            self.slice(max_nodes=2.0)

    def test_depth_and_max_nodes_ranges(self) -> None:
        with self.assertRaises(ValueError):
            self.slice(depth=-1)
        with self.assertRaises(ValueError):
            self.slice(max_nodes=0)
        with self.assertRaises(ValueError):
            self.slice(max_nodes=-2)
        # Boundary values are accepted.
        self.slice(depth=0, max_nodes=1)

    def test_unknown_cause_event_raises_key_error(self) -> None:
        with self.assertRaises(KeyError):
            self.slice(causes=("ghost",))
        # A real graph event that is no cascade cause is still unknown.
        with self.assertRaises(KeyError):
            self.slice(causes=("r0",))
        # One bad cause among good ones raises as well.
        with self.assertRaises(KeyError):
            self.slice(causes=("a0", "ghost"))

    def test_cascade_input_validation_matches_the_cascade_entry(self) -> None:
        invalid = [
            dict(reference=1),
            dict(reference=""),
            dict(reference="ghost"),
            dict(series=()),
            dict(series={"": ()}),
            dict(base_scenario=()),
            dict(axis="nope"),
            dict(values=[0, 1]),
            dict(values=(1, 0)),
            dict(min_size=True),
            dict(max_size=1.0),
            dict(required=["a"]),
            dict(exclusive_pairs=((),)),
            dict(limit=0),
        ]
        for overrides in invalid:
            with self.subTest(**overrides):
                with self.assertRaises(Exception) as slice_caught:
                    self.slice(**overrides)
                cascade_kwargs = dict(
                    reference="main",
                    series=self.series,
                    base_scenario=self.scenario,
                    axis="total_budget",
                    values=(0, 10, 20, 30, 40),
                    min_size=0,
                    max_size=4,
                    required=(),
                    exclusive_pairs=(),
                    limit=40,
                )
                cascade_kwargs.update(overrides)
                with self.assertRaises(type(slice_caught.exception)):
                    self.store.decision_cascade(**cascade_kwargs)

    def test_slice_inputs_validate_before_state_lookup(self) -> None:
        # The slice's own inputs validate after the ordinary cascade
        # inputs but before any branch or node is looked up, so a bad
        # slice input beats an unknown reference branch.
        with self.assertRaises(ValueError):
            self.slice(reference="ghost", causes=("a0", "a0"))
        with self.assertRaises(ValueError):
            self.slice(reference="ghost", direction="Forward")
        with self.assertRaises(ValueError):
            self.slice(reference="ghost", depth=-1)
        with self.assertRaises(ValueError):
            self.slice(reference="ghost", max_nodes=0)
        with self.assertRaises(TypeError):
            self.slice(reference="ghost", causes=["a0"])
        with self.assertRaises(TypeError):
            self.slice(reference="ghost", direction=1)


class CascadeSliceReadOnlyTests(unittest.TestCase):
    def test_query_is_read_only(self) -> None:
        store, series, scenario = make_chain_store()
        heads = dict(store._heads)
        appends = copy.deepcopy(store._appends)
        merges = copy.deepcopy(store._merges)
        replay_main = store.replay("main")
        audit = store.audit_log()
        chain_slice(store, series, scenario, ("b1",), "both", 2)
        with self.assertRaises(ValueError):
            chain_slice(store, series, scenario, ("a0",), "forward", 3,
                        max_nodes=1)
        with self.assertRaises(KeyError):
            chain_slice(store, series, scenario, ("ghost",), "forward", 1)
        with self.assertRaises(TypeError):
            chain_slice(store, series, scenario, ["a0"], "forward", 1)
        self.assertEqual(dict(store._heads), heads)
        self.assertEqual(copy.deepcopy(store._appends), appends)
        self.assertEqual(copy.deepcopy(store._merges), merges)
        self.assertEqual(store.replay("main"), replay_main)
        self.assertEqual(store.audit_log(), audit)

    def test_cascade_and_explanation_entries_are_unaffected(self) -> None:
        store, series, scenario = make_chain_store()
        cascade = store.decision_cascade(
            "main", series, scenario, "total_budget",
            (0, 10, 20, 30, 40), 0, 4, (), (), 40,
        )
        chain_slice(store, series, scenario, ("b1",), "both", 2)
        self.assertEqual(
            store.decision_cascade(
                "main", series, scenario, "total_budget",
                (0, 10, 20, 30, 40), 0, 4, (), (), 40,
            ),
            cascade,
        )

    def test_results_detached_and_unshared(self) -> None:
        store, series, scenario = make_chain_store()
        result = chain_slice(store, series, scenario, ("d3",), "backward", 2)
        pristine = copy.deepcopy(result)
        for node in result["nodes"]:
            node["cause"] = "evil"
            node["intervals"] += (99,)
            node["affected"] += ("evil",)
        for edge in result["edges"]:
            edge["path"] += ("evil",)
        self.assertEqual(
            chain_slice(store, series, scenario, ("d3",), "backward", 2),
            pristine,
        )

    def test_mutable_levels_do_not_share_objects(self) -> None:
        store, series, scenario = make_chain_store()
        result = chain_slice(store, series, scenario, ("d3",), "backward", 2)
        dict_ids = (
            [id(node) for node in result["nodes"]]
            + [id(edge) for edge in result["edges"]]
            + [id(gap) for gap in result["gaps"]]
        )
        self.assertEqual(len(dict_ids), len(set(dict_ids)))

    def test_slice_does_not_share_dicts_with_the_full_cascade(self) -> None:
        store, series, scenario = make_chain_store()
        cascade = store.decision_cascade(
            "main", series, scenario, "total_budget",
            (0, 10, 20, 30, 40), 0, 4, (), (), 40,
        )
        result = chain_slice(store, series, scenario, ("d3",), "backward", 2)
        cascade_ids = {id(node) for node in cascade["nodes"]} | {
            id(edge) for edge in cascade["edges"]
        }
        slice_ids = {id(node) for node in result["nodes"]} | {
            id(edge) for edge in result["edges"]
        }
        self.assertFalse(cascade_ids & slice_ids)

    def test_repeated_calls_are_equal(self) -> None:
        store, series, scenario = make_chain_store()
        self.assertEqual(
            chain_slice(store, series, scenario, ("b1",), "both", 2),
            chain_slice(store, series, scenario, ("b1",), "both", 2),
        )


if __name__ == "__main__":
    unittest.main()
