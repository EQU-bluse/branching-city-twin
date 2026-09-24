import copy
import inspect
import unittest

from city_twin.branches import BranchStore
from city_twin.event_graph import EventGraph


def make_store() -> BranchStore:
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


def make_series() -> dict:
    return {
        "feature": (("m1", "f1"), ("m2", "f2")),
        "side": (("m1", "s1"), ("m2", "s2")),
    }


WEIGHTS = {"x": 2, "y": 3, "z": 1}


def base_scenario(
    weights: dict | None = None,
    total_budget: int | float = 100,
    key_budgets: dict | None = None,
) -> dict:
    return {
        "weights": WEIGHTS if weights is None else weights,
        "total_budget": total_budget,
        "key_budgets": {} if key_budgets is None else key_budgets,
    }


def call(store: BranchStore, **overrides) -> object:
    kwargs = dict(
        reference="main",
        series=make_series(),
        base_scenario=base_scenario(),
        axis="total_budget",
        values=(0, 15, 16, 17, 100),
        min_size=0,
        max_size=2,
        required=(),
        exclusive_pairs=(),
        limit=10,
    )
    kwargs.update(overrides)
    return store.decision_cascade(**kwargs)


def make_chained_store() -> BranchStore:
    # Branch e continues branch a's history, so e's causes are strict
    # descendants of a's causes.
    graph = EventGraph()
    graph.add("root", 0, (), {})
    store = BranchStore(graph)
    store.create("main", "root")
    store.create("a", "root")
    store.append("main", "m1", 1, {})
    store.append("a", "a1", 2, {"x": 10})
    store.append("a", "a2", 3, {"y": 20})
    store.create("e", "a2")
    store.append("e", "e1", 4, {"v": 40})
    store.append("e", "e2", 5, {})
    return store


def chained_series() -> dict:
    return {
        "a": (("m1", "a1"), ("m1", "a2")),
        "e": (("m1", "e1"), ("m1", "e2")),
    }


class DecisionCascadeTests(unittest.TestCase):
    # --- Signature -------------------------------------------------------

    def test_public_signature_matches_breakpoint_queries(self) -> None:
        parameters = list(
            inspect.signature(BranchStore.decision_cascade).parameters.values()
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
            ],
        )
        self.assertTrue(
            all(p.default is inspect.Parameter.empty for p in parameters)
        )

    # --- Shape and key order ----------------------------------------------

    def test_result_shape_and_key_order(self) -> None:
        store = make_store()
        result = call(store)
        self.assertEqual(list(result), ["nodes", "edges", "gaps"])
        self.assertIsInstance(result["nodes"], tuple)
        self.assertIsInstance(result["edges"], tuple)
        self.assertIsInstance(result["gaps"], tuple)
        for node in result["nodes"]:
            self.assertEqual(
                list(node),
                [
                    "cause",
                    "key",
                    "checkpoint",
                    "affected",
                    "intervals",
                    "checkpoints",
                    "branches",
                ],
            )
            self.assertIsInstance(node["cause"], str)
            self.assertIsInstance(node["key"], str)
            self.assertIsInstance(node["affected"], tuple)
            self.assertIsInstance(node["intervals"], tuple)
            self.assertIsInstance(node["checkpoints"], tuple)
            self.assertIsInstance(node["branches"], tuple)
        for edge in result["edges"]:
            self.assertEqual(list(edge), ["source", "target", "path"])
            self.assertIsInstance(edge["source"], tuple)
            self.assertIsInstance(edge["target"], tuple)
            self.assertIsInstance(edge["path"], tuple)
        for gap in result["gaps"]:
            self.assertEqual(list(gap), ["interval", "members"])
            self.assertIsInstance(gap["members"], tuple)

    def test_no_turning_points_returns_three_empty_tuples(self) -> None:
        store = make_store()
        self.assertEqual(
            call(store, values=(16, 17)),
            {"nodes": (), "edges": (), "gaps": ()},
        )

    # --- Nodes -------------------------------------------------------------

    def test_nodes_from_the_standard_fixture(self) -> None:
        store = make_store()
        result = call(store)
        self.assertEqual(
            result["nodes"],
            (
                {
                    "cause": "s1",
                    "key": "y",
                    "checkpoint": 1,
                    "affected": ("s2",),
                    "intervals": (0, 1, 1),
                    "checkpoints": (1, 1, 0),
                    "branches": ("side",),
                },
                {
                    "cause": "f2",
                    "key": "y",
                    "checkpoint": 1,
                    "affected": (),
                    "intervals": (1,),
                    "checkpoints": (1,),
                    "branches": ("feature",),
                },
            ),
        )
        self.assertEqual(result["edges"], ())
        self.assertEqual(result["gaps"], ())

    def test_shared_cause_event_merges_across_branches_and_intervals(
        self,
    ) -> None:
        store = make_chained_store()
        # Weight axis x: (e,) flips on the first interval, (a,) on the
        # second; both attribute to the shared inherited event a1.
        result = store.decision_cascade(
            "main",
            chained_series(),
            {
                "weights": {"v": 1, "x": 1, "y": 1},
                "total_budget": 100,
                "key_budgets": {},
            },
            "x",
            (1, 5, 10),
            0,
            2,
            (),
            (),
            10,
        )
        (node,) = result["nodes"]
        self.assertEqual(node["cause"], "a1")
        self.assertEqual(node["key"], "x")
        self.assertEqual(node["branches"], ("a", "e"))
        self.assertEqual(node["intervals"], (0, 0, 0, 1))
        self.assertEqual(node["checkpoints"], (0, 0, 0, 1))
        # The affected range unites every occurrence's closure in replay
        # order: a2 and e1 are a1's descendants on the two branches.
        self.assertEqual(node["affected"], ("a2", "e1"))
        self.assertEqual(result["edges"], ())
        self.assertEqual(result["gaps"], ())

    def test_nodes_sorted_by_first_interval_checkpoint_key_and_cause(
        self,
    ) -> None:
        store = make_store()
        result = call(store)
        sort_keys = [
            (
                node["intervals"][0],
                node["checkpoint"] is None,
                node["checkpoint"] if node["checkpoint"] is not None else 0,
                node["key"],
                node["cause"],
            )
            for node in result["nodes"]
        ]
        self.assertEqual(sort_keys, sorted(sort_keys))

    # --- Edges -------------------------------------------------------------

    def test_descendant_causes_chain_across_adjacent_intervals(self) -> None:
        store = make_chained_store()
        # (a,) flips on (5, 30) with cause a1; (e,) flips on (30, 70) with
        # cause e1, a strict descendant of a1 through a2.
        result = store.decision_cascade(
            "main",
            chained_series(),
            {
                "weights": {"v": 1, "x": 1, "y": 1},
                "total_budget": 0,
                "key_budgets": {},
            },
            "total_budget",
            (5, 30, 70, 100),
            0,
            2,
            (),
            (),
            10,
        )
        self.assertEqual(
            [node["cause"] for node in result["nodes"]], ["a1", "e1"]
        )
        self.assertEqual(
            result["edges"],
            (
                {
                    "source": ("a1", "x"),
                    "target": ("e1", "v"),
                    "path": ("a1", "a2", "e1"),
                },
            ),
        )
        self.assertEqual(result["gaps"], ())

    def test_merge_event_forms_a_cross_branch_edge(self) -> None:
        graph = EventGraph()
        graph.add("root", 0, (), {})
        store = BranchStore(graph)
        store.create("main", "root")
        store.create("a", "root")
        store.create("b", "root")
        store.append("main", "m1", 1, {})
        store.append("a", "a1", 2, {"x": 10})
        store.append("a", "a2", 3, {"y": 20})
        store.append("b", "b1", 4, {"w": 5})
        store.append("b", "b2", 5, {"q": 8})
        store.merge("b", "a", "mg", 6, {"z": 1})
        store.append("b", "b3", 7, {"v": 50})
        store.append("main", "m2", 8, {})
        series = {
            "a": (("m1", "a1"), ("m2", "a2")),
            "b": (("m1", "b2"), ("m2", "b3")),
        }
        result = store.decision_cascade(
            "main",
            series,
            {
                "weights": {"q": 1, "v": 1, "w": 1, "x": 1, "y": 1, "z": 1},
                "total_budget": 0,
                "key_budgets": {},
            },
            "total_budget",
            (15, 30, 94, 100),
            0,
            2,
            (),
            (),
            20,
        )
        self.assertEqual(
            result["edges"],
            (
                {
                    "source": ("a2", "y"),
                    "target": ("b3", "v"),
                    "path": ("a2", "mg", "b3"),
                },
            ),
        )

    def test_unrelated_causes_stay_independent_chains(self) -> None:
        store = make_store()
        # s1 and f2 are on different branches with no ancestor relation,
        # so no edge is created between their nodes.
        result = call(store)
        causes = {node["cause"] for node in result["nodes"]}
        self.assertEqual(causes, {"s1", "f2"})
        self.assertEqual(result["edges"], ())

    def test_shortest_path_tie_breaks_by_id_tuple(self) -> None:
        # Diamond: p -> c1, p -> c2, merge -> q with parents (c1, c2).
        # Two shortest paths exist from p to q; the lexicographically
        # smallest id tuple wins.
        graph = EventGraph()
        graph.add("root", 0, (), {})
        store = BranchStore(graph)
        store.create("main", "root")
        store.create("one", "root")
        store.create("two", "root")
        store.append("main", "m1", 1, {})
        store.append("one", "p", 2, {"x": 10})
        store.append("one", "c1", 3, {})
        store.append("two", "c2", 4, {})
        store.create("joined", "c1")
        store.merge("joined", "two", "q", 5, {"y": 30})
        store.append("joined", "q2", 6, {})
        series = {
            "one": (("m1", "p"), ("m1", "c1")),
            "joined": (("m1", "q"), ("m1", "q2")),
        }
        # one: cp0/cp1 risk 10 (x via p); joined: cp0/cp1 risk 40 (x+y).
        # (one,) flips on (5, 10) with cause p; (joined,) flips on
        # (10, 40) with cause q.
        result = store.decision_cascade(
            "main",
            series,
            {
                "weights": {"x": 1, "y": 1},
                "total_budget": 0,
                "key_budgets": {},
            },
            "total_budget",
            (5, 10, 40, 100),
            0,
            2,
            (),
            (),
            20,
        )
        edges = [
            edge
            for edge in result["edges"]
            if edge["source"] == ("p", "x")
        ]
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0]["target"], ("q", "y"))
        self.assertEqual(edges[0]["path"], ("p", "c1", "q"))

    # --- Gaps --------------------------------------------------------------

    def test_missing_cause_key_records_a_gap(self) -> None:
        graph = EventGraph()
        graph.add("root", 0, (), {})
        store = BranchStore(graph)
        store.create("main", "root")
        store.create("a", "root")
        store.create("b", "root")
        store.append("main", "r1", 1, {})
        # a never touches y; its dominators move when b's y weight changes.
        store.append("a", "a1", 2, {"x": 10})
        store.append("b", "b1", 3, {"x": -4, "y": 10})
        series = {"a": (("r1", "a1"),), "b": (("r1", "b1"),)}
        result = store.decision_cascade(
            "main",
            series,
            {
                "weights": {"x": 1, "y": 0.3},
                "total_budget": 100.0,
                "key_budgets": {},
            },
            "y",
            (0.3, 0.5),
            0,
            2,
            (),
            (),
            10,
        )
        self.assertEqual(result["nodes"], ())
        self.assertEqual(result["edges"], ())
        self.assertEqual(
            result["gaps"], ({"interval": 0, "members": ("a",)},)
        )

    def test_missing_member_cause_records_a_gap(self) -> None:
        # The divergence on the cause key is driven by the reference side
        # (m2 touches y); the member side names no cause event, so the
        # change is a gap even though a cause key exists.
        graph = EventGraph()
        graph.add("root", 0, (), {})
        store = BranchStore(graph)
        store.create("main", "root")
        store.append("main", "m1", 1, {})
        store.append("main", "m2", 2, {"y": 5})
        store.create("a", "m1")
        store.append("a", "a1", 3, {"x": 10})
        result = store.decision_cascade(
            "main",
            {"a": (("m2", "a1"),)},
            {
                "weights": {"x": 1, "y": 1},
                "total_budget": 12,
                "key_budgets": {},
            },
            "y",
            (0.4, 0.5),
            0,
            1,
            (),
            (),
            10,
        )
        self.assertEqual(result["nodes"], ())
        self.assertEqual(result["edges"], ())
        self.assertEqual(
            result["gaps"], ({"interval": 0, "members": ("a",)},)
        )

    # --- Validation parity with frontier_breakpoints ----------------------

    def test_validation_errors_match_the_turning_point_query(self) -> None:
        store = make_store()
        series = make_series()
        scenario = base_scenario()

        def cascade(**overrides):
            return call(store, **overrides)

        def breakpoints(**overrides):
            kwargs = dict(
                reference="main",
                series=series,
                base_scenario=scenario,
                axis="total_budget",
                values=(0, 100),
                min_size=0,
                max_size=2,
                required=(),
                exclusive_pairs=(),
                limit=10,
            )
            kwargs.update(overrides)
            return store.frontier_breakpoints(**kwargs)

        # Every invalid input must raise the same exception type.
        invalid = [
            dict(reference=1),
            dict(reference=""),
            dict(reference="ghost"),
            dict(series=()),
            dict(series={"": ()}),
            dict(series={"ghost": ()}),
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
            dict(required=("ghost",)),
            dict(exclusive_pairs=((),)),
            dict(limit=0),
        ]
        for overrides in invalid:
            with self.subTest(**overrides):
                with self.assertRaises(Exception) as cascade_caught:
                    cascade(**overrides)
                with self.assertRaises(type(cascade_caught.exception)):
                    breakpoints(**overrides)

    def test_booleans_never_accepted_as_numbers(self) -> None:
        store = make_store()
        for bad_kwargs in (
            dict(values=(0, True)),
            dict(min_size=True),
            dict(max_size=False),
            dict(limit=True),
        ):
            with self.subTest(**bad_kwargs):
                with self.assertRaises(TypeError):
                    call(store, **bad_kwargs)

    def test_unknown_branch_and_node_raise_key_error(self) -> None:
        store = make_store()
        with self.assertRaises(KeyError):
            call(store, reference="ghost")
        with self.assertRaises(KeyError):
            call(
                store,
                series={"feature": (("nope1", "f1"),)},
                max_size=1,
            )
        with self.assertRaises(KeyError):
            call(
                store,
                series={"feature": (("f2", "f1"),)},
                max_size=1,
            )

    def test_candidate_limit_exceeded_raises_value_error(self) -> None:
        store = make_store()
        with self.assertRaises(ValueError):
            call(store, limit=3)

    def test_misaligned_checkpoints_raise_value_error(self) -> None:
        store = make_store()
        with self.assertRaises(ValueError):
            call(
                store,
                series={
                    "feature": (("m1", "f1"), ("m2", "f2")),
                    "side": (("m1", "s1"),),
                },
            )

    # --- Read-only and detachment -----------------------------------------

    def test_query_is_read_only(self) -> None:
        store = make_store()
        heads = dict(store._heads)
        appends = copy.deepcopy(store._appends)
        merges = copy.deepcopy(store._merges)
        replay_main = store.replay("main")
        audit = store.audit_log()
        series = make_series()
        scenario = base_scenario()
        explained_before = store.explain_frontier_breakpoints(
            "main", series, scenario, "total_budget",
            (0, 15, 16, 17, 100), 0, 2, (), (), 10,
        )
        call(store)
        # Failing calls leave state untouched as well.
        with self.assertRaises(ValueError):
            call(store, limit=1)
        with self.assertRaises(KeyError):
            call(
                store,
                series={"feature": (("m1", "nope"),)},
                max_size=1,
            )
        with self.assertRaises(TypeError):
            call(store, values=[0, 100])
        self.assertEqual(dict(store._heads), heads)
        self.assertEqual(copy.deepcopy(store._appends), appends)
        self.assertEqual(copy.deepcopy(store._merges), merges)
        self.assertEqual(store.replay("main"), replay_main)
        self.assertEqual(store.audit_log(), audit)
        # The existing explanation query is unaffected.
        self.assertEqual(
            store.explain_frontier_breakpoints(
                "main", series, scenario, "total_budget",
                (0, 15, 16, 17, 100), 0, 2, (), (), 10,
            ),
            explained_before,
        )

    def test_results_detached_and_unshared(self) -> None:
        store = make_store()
        result = call(store)
        pristine = copy.deepcopy(result)
        for node in result["nodes"]:
            node["key"] = "evil"
            node["affected"] += ("evil",)
            node["intervals"] += (99,)
            node["branches"] += ("evil",)
        for edge in result["edges"]:
            edge["path"] += ("evil",)
        for gap in result["gaps"]:
            gap["members"] += ("evil",)
        self.assertEqual(call(store), pristine)

    def test_repeated_calls_are_equal(self) -> None:
        store = make_store()
        first = call(store)
        second = call(store)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
