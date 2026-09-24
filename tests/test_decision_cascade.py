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


def explain_call(store: BranchStore, **overrides) -> object:
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
    return store.explain_frontier_breakpoints(**kwargs)


class DecisionCascadeSignatureTests(unittest.TestCase):
    def test_public_signature_matches_turning_point_explanation(self) -> None:
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

    def test_result_is_a_dict_with_three_keys(self) -> None:
        result = call(make_store())
        self.assertIsInstance(result, dict)
        self.assertEqual(list(result), ["nodes", "edges", "gaps"])
        for name in ("nodes", "edges", "gaps"):
            self.assertIsInstance(result[name], tuple)


class DecisionCascadeNodeTests(unittest.TestCase):
    def test_standard_fixture_nodes(self) -> None:
        store = make_store()
        result = call(store)
        nodes = result["nodes"]
        # Interval 0 (15->16): the side change names main's m1 and side's
        # s1 as the y causes. Interval 1 (17->100) adds feature's f2.
        by_key = {(node["cause"], node["key"]): node for node in nodes}
        self.assertEqual(
            set(by_key),
            {("m1", "y"), ("s1", "y"), ("f2", "y")},
        )
        m1 = by_key[("m1", "y")]
        self.assertEqual(
            list(m1),
            [
                "cause",
                "key",
                "intervals",
                "checkpoints",
                "branches",
                "affected",
            ],
        )
        self.assertEqual(m1["intervals"], (0, 1))
        self.assertEqual(m1["branches"], ("main",))
        # main's checkpoint-1 closure at m2 makes m2 the affected event.
        self.assertEqual(m1["affected"], ("m2",))
        s1 = by_key[("s1", "y")]
        self.assertEqual(s1["intervals"], (0, 1))
        self.assertEqual(s1["branches"], ("side",))
        self.assertEqual(s1["affected"], ("s2",))
        f2 = by_key[("f2", "y")]
        self.assertEqual(f2["intervals"], (1,))
        self.assertEqual(f2["branches"], ("feature",))
        self.assertEqual(f2["affected"], ())

    def test_checkpoints_aggregate_in_first_appearance_order(self) -> None:
        store = make_store()
        nodes = {
            (node["cause"], node["key"]): node
            for node in call(store)["nodes"]
        }
        # s1 drives the side change at checkpoint 1 (interval 0) and joins
        # the pair change at checkpoint 0 (interval 1): both checkpoints are
        # kept, in first-appearance order.
        self.assertEqual(nodes[("s1", "y")]["checkpoints"], (1, 0))
        self.assertEqual(nodes[("m1", "y")]["checkpoints"], (1, 0))

    def test_branches_sorted_by_unicode_code_point(self) -> None:
        graph = EventGraph()
        graph.add("root", 0, (), {})
        store = BranchStore(graph)
        store.create("main", "root")
        store.create("beta", "root")
        store.create("alpha", "root")
        store.append("main", "r0", 1, {})
        store.append("beta", "be0", 2, {"x": 5})
        store.append("alpha", "al0", 3, {"y": 5})
        store.append("main", "r1", 4, {})
        # Both series forking from r0 make the shared reference cause m...
        series = {
            "beta": (("r0", "be0"), ("r1", "be0")),
            "alpha": (("r0", "al0"), ("r1", "al0")),
        }
        scenario = {
            "weights": {"x": 1, "y": 1},
            "total_budget": 10,
            "key_budgets": {},
        }
        result = store.decision_cascade(
            "main", series, scenario, "total_budget", (0, 10), 0, 2, (), (), 10
        )
        for node in result["nodes"]:
            self.assertEqual(
                node["branches"], tuple(sorted(node["branches"]))
            )

    def test_same_cause_event_and_key_is_one_node_across_intervals(self) -> None:
        store = make_store()
        causes = [(n["cause"], n["key"]) for n in call(store)["nodes"]]
        self.assertEqual(len(causes), len(set(causes)))
        # s1/y drives turns in both intervals but is a single node.
        self.assertEqual(len([c for c in causes if c == ("s1", "y")]), 1)

    def test_node_ordering(self) -> None:
        store = make_store()
        result = call(store)
        nodes = result["nodes"]
        self.assertEqual(
            [(n["cause"], n["key"]) for n in nodes],
            [("m1", "y"), ("s1", "y"), ("f2", "y")],
        )
        # The node tuple's own order sorts: first interval, then checkpoint,
        # then key, then event id.
        def sort_key(node):
            checkpoint = node["checkpoints"][0] if node["checkpoints"] else None
            return (
                node["intervals"][0],
                checkpoint is not None,
                checkpoint if checkpoint is not None else 0,
                node["key"],
                node["cause"],
            )

        self.assertEqual(nodes, tuple(sorted(nodes, key=sort_key)))


class DecisionCascadeEdgeTests(unittest.TestCase):
    def _chain_store(self) -> tuple[BranchStore, dict, dict]:
        graph = EventGraph()
        graph.add("root", 0, (), {})
        store = BranchStore(graph)
        store.create("main", "root")
        store.append("main", "r0", 1, {})
        store.create("a", "root")
        store.append("a", "a0", 2, {"x": 6})
        # b forks from a0, so b1 is a strict descendant of a0.
        store.create("b", "a0")
        store.append("b", "b1", 3, {"y": 8})
        store.append("main", "r1", 4, {})
        series = {"a": (("r0", "a0"),), "b": (("r0", "b1"),)}
        scenario = {
            "weights": {"x": 1, "y": 1},
            "total_budget": 20,
            "key_budgets": {},
        }
        return store, series, scenario

    def test_edge_from_earlier_to_later_strict_descendant(self) -> None:
        store, series, scenario = self._chain_store()
        result = store.decision_cascade(
            "main",
            series,
            scenario,
            "total_budget",
            (0, 10, 20),
            0,
            2,
            (),
            (),
            10,
        )
        self.assertEqual(len(result["edges"]), 1)
        edge = result["edges"][0]
        self.assertEqual(
            list(edge), ["source", "target", "path"]
        )
        nodes = result["nodes"]
        source = nodes[edge["source"]]
        target = nodes[edge["target"]]
        self.assertEqual((source["cause"], source["key"]), ("a0", "x"))
        self.assertEqual((target["cause"], target["key"]), ("b1", "y"))
        # Shortest parent-to-child path contains both ends.
        self.assertEqual(edge["path"], ("a0", "b1"))

    def test_non_descendant_causes_stay_independent(self) -> None:
        # The standard fixture's later feature cause f2 is on a branch that
        # forks from root, so neither earlier cause (m1 or s1) is its
        # ancestor: no edges at all.
        result = call(make_store())
        self.assertEqual(result["edges"], ())

    def test_edges_only_between_adjacent_intervals(self) -> None:
        # Two intervals where the first and last causes are unrelated but
        # each shares its node across an intervening interval cannot create
        # a non-adjacent edge.
        store, series, scenario = self._chain_store()
        explained = explain_call(
            store,
            series=series,
            base_scenario=scenario,
            values=(0, 10, 20),
        )
        self.assertEqual(len(explained), 2)
        result = store.decision_cascade(
            "main",
            series,
            scenario,
            "total_budget",
            (0, 10, 20),
            0,
            2,
            (),
            (),
            10,
        )
        for edge in result["edges"]:
            source_interval = result["nodes"][edge["source"]]["intervals"][0]
            target_interval = result["nodes"][edge["target"]]["intervals"][0]
            self.assertEqual(target_interval - source_interval, 1)

    def _diamond_store(self) -> tuple[BranchStore, dict, dict]:
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
        # W merges c into b, reachable from a0 on two equal-length paths.
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
        return store, series, scenario

    def test_shortest_path_tie_break_picks_smallest_tuple(self) -> None:
        store, series, scenario = self._diamond_store()
        result = store.decision_cascade(
            "main",
            series,
            scenario,
            "total_budget",
            (0, 10, 20),
            0,
            2,
            (),
            (),
            10,
        )
        edges = result["edges"]
        self.assertEqual(len(edges), 1)
        # Two equal-length paths a0->bu->W and a0->cv->W; the complete id
        # tuple ordering selects the bu path.
        self.assertEqual(edges[0]["path"], ("a0", "bu", "W"))

    def test_merge_convergence_crosses_branches(self) -> None:
        store, series, scenario = self._diamond_store()
        result = store.decision_cascade(
            "main",
            series,
            scenario,
            "total_budget",
            (0, 10, 20),
            0,
            2,
            (),
            (),
            10,
        )
        edge = result["edges"][0]
        source = result["nodes"][edge["source"]]
        target = result["nodes"][edge["target"]]
        # The later cause W is a merge event on b; the earlier cause lives
        # on a and reaches it through the merge graph.
        self.assertEqual(target["cause"], "W")
        self.assertIn("a", source["branches"])

    def test_edge_indices_reference_node_order(self) -> None:
        store, series, scenario = self._diamond_store()
        result = store.decision_cascade(
            "main",
            series,
            scenario,
            "total_budget",
            (0, 10, 20),
            0,
            2,
            (),
            (),
            10,
        )
        for edge in result["edges"]:
            self.assertIsInstance(edge["source"], int)
            self.assertIsInstance(edge["target"], int)
            self.assertGreaterEqual(edge["source"], 0)
            self.assertGreaterEqual(edge["target"], 0)
            self.assertLess(edge["source"], len(result["nodes"]))
            self.assertLess(edge["target"], len(result["nodes"]))


class DecisionCascadeGapTests(unittest.TestCase):
    def test_missing_cause_key_becomes_gap_not_node(self) -> None:
        # A domination-only change for the empty combination has no
        # locating cause key; it must appear as a gap and never as a node.
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
        scenario = {
            "weights": {"k": 1},
            "total_budget": 13,
            "key_budgets": {},
        }
        explained = store.explain_frontier_breakpoints(
            "main", series, scenario, "total_budget", (0, 10, 13),
            0, 1, (), (), 10,
        )
        members_by_interval = {
            index: tuple(change["members"] for change in interval["changes"])
            for index, interval in enumerate(explained)
        }
        result = store.decision_cascade(
            "main", series, scenario, "total_budget", (0, 10, 13),
            0, 1, (), (), 10,
        )
        gap_members = [(gap["interval"], gap["members"]) for gap in result["gaps"]]
        self.assertIn((0, ()), gap_members)
        for gap in result["gaps"]:
            self.assertEqual(list(gap), ["interval", "members"])
            self.assertIsInstance(gap["interval"], int)
            self.assertIsInstance(gap["members"], tuple)
            self.assertIn(
                gap["members"], members_by_interval[gap["interval"]]
            )
        # No node was forged for the empty-combo domination change.
        self.assertTrue(all(node["key"] for node in result["nodes"]))

    def test_gaps_keep_interval_and_candidate_order(self) -> None:
        store = make_store()
        result = call(store)
        gaps = result["gaps"]
        intervals = [gap["interval"] for gap in gaps]
        self.assertEqual(intervals, sorted(intervals))
        for gap in gaps:
            self.assertEqual(
                gap["members"], tuple(sorted(gap["members"]))
            )


class DecisionCascadeEmptyTests(unittest.TestCase):
    def test_no_turning_intervals_returns_three_empty_tuples(self) -> None:
        store = make_store()
        result = call(store, values=(16, 17))
        self.assertEqual(result, {"nodes": (), "edges": (), "gaps": ()})

    def test_no_checkpoints_returns_three_empty_tuples(self) -> None:
        store = make_store()
        result = call(
            store,
            series={"feature": (), "side": ()},
            base_scenario=base_scenario(total_budget=0, key_budgets={"x": 0}),
            values=(0, 100),
        )
        self.assertEqual(result, {"nodes": (), "edges": (), "gaps": ()})


class DecisionCascadeAffectedOrderTests(unittest.TestCase):
    def test_affected_follows_historical_closure_replay_order(self) -> None:
        store = make_store()
        result = call(store)
        s1 = next(
            node for node in result["nodes"] if node["cause"] == "s1"
        )
        # The s1 strict descendants within the s2 closure in replay order.
        self.assertEqual(s1["affected"], ("s2",))
        m1 = next(
            node for node in result["nodes"] if node["cause"] == "m1"
        )
        self.assertEqual(m1["affected"], ("m2",))

    def test_affected_are_deduplicated_and_strict_descendants(self) -> None:
        store, series, scenario = self._store()
        result = store.decision_cascade(
            "main", series, scenario, "total_budget", (0, 10, 20),
            0, 2, (), (), 10,
        )
        for node in result["nodes"]:
            self.assertEqual(
                node["affected"], tuple(dict.fromkeys(node["affected"]))
            )
            self.assertNotIn(node["cause"], node["affected"])

    def _store(self):
        graph = EventGraph()
        graph.add("root", 0, (), {})
        store = BranchStore(graph)
        store.create("main", "root")
        store.append("main", "r0", 1, {})
        store.create("a", "root")
        store.append("a", "a0", 2, {"x": 6})
        store.create("b", "a0")
        store.append("b", "b1", 3, {"y": 8})
        store.append("main", "r1", 4, {})
        series = {"a": (("r0", "a0"),), "b": (("r0", "b1"),)}
        scenario = {
            "weights": {"x": 1, "y": 1},
            "total_budget": 20,
            "key_budgets": {},
        }
        return store, series, scenario


class DecisionCascadeValidationTests(unittest.TestCase):
    def test_validation_errors_match_the_explanation_entry(self) -> None:
        store = make_store()
        series = make_series()
        scenario = base_scenario()

        def cascade(**overrides):
            return call(store, **overrides)

        def explain(**overrides):
            return explain_call(store, **overrides)

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
                    explain(**overrides)

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

    def test_candidate_limit_raises_value_error_with_no_partial_result(self) -> None:
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


class DecisionCascadeReadOnlyTests(unittest.TestCase):
    def test_query_is_read_only(self) -> None:
        store = make_store()
        heads = dict(store._heads)
        appends = copy.deepcopy(store._appends)
        merges = copy.deepcopy(store._merges)
        replay_main = store.replay("main")
        audit = store.audit_log()
        call(store)
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

    def test_existing_query_results_are_unaffected(self) -> None:
        store = make_store()
        series = make_series()
        explanation = explain_call(store)
        breakpoints = store.frontier_breakpoints(
            "main",
            series,
            base_scenario(),
            "total_budget",
            (0, 15, 16, 17, 100),
            0,
            2,
            (),
            (),
            10,
        )
        call(store)
        self.assertEqual(explain_call(store), explanation)
        self.assertEqual(
            store.frontier_breakpoints(
                "main",
                series,
                base_scenario(),
                "total_budget",
                (0, 15, 16, 17, 100),
                0,
                2,
                (),
                (),
                10,
            ),
            breakpoints,
        )

    def test_results_detached_and_unshared(self) -> None:
        store = make_store()
        result = call(store)
        pristine = copy.deepcopy(result)
        for node in result["nodes"]:
            node["cause"] = "evil"
            node["branches"] += ("evil",)
            node["affected"] += ("evil",)
        for edge in result["edges"]:
            edge["path"] += ("evil",)
        for gap in result["gaps"]:
            gap["members"] += ("evil",)
        self.assertEqual(call(store), pristine)

    def test_mutable_levels_do_not_share_objects(self) -> None:
        store = make_store()
        result = call(store)
        dict_ids = (
            [id(node) for node in result["nodes"]]
            + [id(edge) for edge in result["edges"]]
            + [id(gap) for gap in result["gaps"]]
        )
        self.assertEqual(len(dict_ids), len(set(dict_ids)))

    def test_repeated_calls_are_equal(self) -> None:
        store = make_store()
        self.assertEqual(call(store), call(store))


if __name__ == "__main__":
    unittest.main()
