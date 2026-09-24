import copy
import inspect
import subprocess
import sys
import unittest

from city_twin.branches import BranchStore
from city_twin.event_graph import EventGraph


def diamond_store() -> tuple[BranchStore, tuple]:
    """Three aligned checkpoints; prefix 1 is a0 only, later prefixes add W.

    a0 reaches merge W over a cross-branch convergence edge; the third
    checkpoint changes nothing, so the last segment is empty.
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


def changed_store() -> tuple[BranchStore, dict, tuple]:
    """The same (cause, key) node changes content between the first prefixes.

    Prefix 1 attributes the interval-0 change at checkpoint 0 with no
    affected events; prefix 2 first locates it at checkpoint 1, so the same
    cause node relocates and gains affected events -- a content change with
    neither an add nor a remove; prefix 3 changes nothing further.
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
    """The second prefix introduces an empty-combo gap at interval 0."""
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


def lifetime_kwargs(store_args, **overrides) -> dict:
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
        indices=(0, 1, 2),
        causes=("a0",),
        direction="both",
        depth=99,
        node_limit=50,
        change_limit=50,
        lifetime_limit=50,
    )
    kwargs.update(overrides)
    return kwargs


def lifetime_call(store, store_args, **overrides):
    return store.cascade_slice_lifetimes(
        **lifetime_kwargs(store_args, **overrides)
    )


def changed_kwargs(scenario, points, **overrides) -> dict:
    kwargs = dict(
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
        indices=(0, 1, 2),
        causes=("a0", "r0"),
        direction="both",
        depth=99,
        node_limit=50,
        change_limit=50,
        lifetime_limit=50,
    )
    kwargs.update(overrides)
    return kwargs


def gap_kwargs(scenario, points, **overrides) -> dict:
    kwargs = dict(
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
        indices=(0, 1),
        causes=("p1",),
        direction="both",
        depth=0,
        node_limit=50,
        change_limit=50,
        lifetime_limit=50,
    )
    kwargs.update(overrides)
    return kwargs


class CascadeSliceLifetimesSignatureTests(unittest.TestCase):
    def test_public_signature_appends_lifetime_limit(self) -> None:
        parameters = list(
            inspect.signature(
                BranchStore.cascade_slice_lifetimes
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
                "indices",
                "causes",
                "direction",
                "depth",
                "node_limit",
                "change_limit",
                "lifetime_limit",
            ],
        )
        self.assertTrue(
            all(p.default is inspect.Parameter.empty for p in parameters)
        )

    def test_result_is_a_dict_with_only_lifetimes(self) -> None:
        store, args = diamond_store()
        result = lifetime_call(store, args)
        self.assertIsInstance(result, dict)
        self.assertEqual(list(result), ["lifetimes"])
        self.assertIsInstance(result["lifetimes"], tuple)
        for record in result["lifetimes"]:
            self.assertEqual(
                list(record),
                [
                    "type",
                    "identity",
                    "first_seen",
                    "last_seen",
                    "intervals",
                    "transitions",
                ],
            )
            self.assertIn(record["type"], ("node", "edge", "gap"))
            self.assertIsInstance(record["intervals"], tuple)
            self.assertIsInstance(record["transitions"], tuple)


class CascadeSliceLifetimesContentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        return lifetime_call(self.store, self.args, **overrides)

    def index(self, records, identity):
        matches = [
            position
            for position, record in enumerate(records)
            if record["identity"] == identity
        ]
        self.assertEqual(len(matches), 1, identity)
        return matches[0]

    def test_node_lifetimes_track_first_last_and_closed_intervals(self) -> None:
        records = self.call()["lifetimes"]
        a0 = records[self.index(records, ("a0", "x"))]
        self.assertEqual(
            (a0["type"], a0["first_seen"], a0["last_seen"]),
            ("node", 0, 2),
        )
        self.assertEqual(a0["intervals"], ((0, 2),))

    def test_first_selected_snapshot_evidence_has_no_added_transition(self) -> None:
        records = self.call()["lifetimes"]
        a0 = records[self.index(records, ("a0", "x"))]
        # a0 already exists at the first selected snapshot: its lifetime
        # counts, but no addition is fabricated.
        self.assertEqual(a0["transitions"], ())

    def test_added_node_records_one_addition_with_detached_evidence(self) -> None:
        records = self.call(indices=(0, 1, 2))["lifetimes"]
        w = records[self.index(records, ("W", "y"))]
        self.assertEqual(
            (w["type"], w["first_seen"], w["last_seen"]), ("node", 1, 2)
        )
        self.assertEqual(w["intervals"], ((1, 2),))
        self.assertEqual(len(w["transitions"]), 1)
        transition = w["transitions"][0]
        self.assertEqual(
            list(transition), ["kind", "before", "after"]
        )
        self.assertEqual(transition["kind"], "node_added")
        self.assertIsNone(transition["before"])
        # The after side is the detached evidence from the segment at 1.
        timeline_kwargs = lifetime_kwargs(self.args, indices=(0, 1, 2))
        timeline_kwargs.pop("lifetime_limit")
        timeline = self.store.cascade_slice_timeline(**timeline_kwargs)
        (segment,) = (
            segment
            for segment in timeline["segments"]
            if segment["before"] == 0 and segment["after"] == 1
        )
        added = next(
            change
            for change in segment["changes"]
            if change["identity"] == ("W", "y")
        )
        self.assertEqual(transition["after"], added["after"])

    def test_edge_identity_uses_endpoint_node_identities(self) -> None:
        records = self.call()["lifetimes"]
        position = self.index(records, (("a0", "x"), ("W", "y")))
        edge = records[position]
        self.assertEqual(edge["type"], "edge")
        self.assertEqual(
            (edge["first_seen"], edge["last_seen"]), (1, 2)
        )
        self.assertEqual(edge["intervals"], ((1, 2),))
        self.assertEqual(
            [t["kind"] for t in edge["transitions"]], ["edge_added"]
        )
        self.assertIsNone(edge["transitions"][0]["before"])
        self.assertEqual(
            edge["transitions"][0]["after"]["path"], ("a0", "bu", "W")
        )

    def test_records_run_nodes_then_edges_then_gaps(self) -> None:
        records = self.call()["lifetimes"]
        types = [record["type"] for record in records]
        self.assertEqual(types, sorted(types, key=("node", "edge", "gap").index))
        self.assertEqual(
            [r["identity"] for r in records if r["type"] == "node"],
            [("a0", "x"), ("W", "y")],
        )
        self.assertEqual(
            [r["identity"] for r in records if r["type"] == "edge"],
            [(("a0", "x"), ("W", "y"))],
        )
        self.assertEqual([r for r in records if r["type"] == "gap"], [])

    def test_identical_adjacent_snapshots_add_no_transitions(self) -> None:
        records = self.call(indices=(1, 2))["lifetimes"]
        # Both identities and the edge already exist at snapshot 1; segment
        # (1, 2) changes nothing, so every transition list is empty.
        self.assertEqual(
            [
                (r["identity"], r["first_seen"], r["intervals"])
                for r in records
            ],
            [
                (("a0", "x"), 1, ((1, 2),)),
                (("W", "y"), 1, ((1, 2),)),
                ((("a0", "x"), ("W", "y")), 1, ((1, 2),)),
            ],
        )
        self.assertTrue(all(r["transitions"] == () for r in records))

    def test_window_skipping_a_checkpoint_stays_one_interval(self) -> None:
        # Snapshots 0 and 2 are adjacent *selected* snapshots: a0 exists in
        # both, so its interval is the closed (0, 2), not two singletons.
        records = self.call(indices=(0, 2))["lifetimes"]
        a0 = records[self.index(records, ("a0", "x"))]
        self.assertEqual(a0["intervals"], ((0, 2),))
        w = records[self.index(records, ("W", "y"))]
        self.assertEqual((w["first_seen"], w["last_seen"]), (2, 2))
        self.assertEqual(w["intervals"], ((2, 2),))
        self.assertEqual(
            [t["kind"] for t in w["transitions"]], ["node_added"]
        )

    def test_only_identities_present_in_a_snapshot_are_listed(self) -> None:
        # With one snapshot there are no segments, hence no transitions at
        # all; every listed identity exists at that snapshot.
        records = self.call(indices=(1,))["lifetimes"]
        self.assertEqual(
            {r["identity"] for r in records},
            {("a0", "x"), ("W", "y"), (("a0", "x"), ("W", "y"))},
        )
        self.assertTrue(all(r["transitions"] == () for r in records))


class CascadeSliceLifetimesTransitionTests(unittest.TestCase):
    def test_node_content_change_keeps_both_evidence_sides(self) -> None:
        store, scenario, points = changed_store()
        result = store.cascade_slice_lifetimes(
            **changed_kwargs(scenario, points)
        )
        records = {r["identity"]: r for r in result["lifetimes"]}
        a0 = records[("a0", "x")]
        self.assertEqual(
            (a0["first_seen"], a0["last_seen"], a0["intervals"]),
            (0, 2, ((0, 2),)),
        )
        # Only segment (0, 1) changes content; segment (1, 2) adds nothing.
        self.assertEqual(
            [t["kind"] for t in a0["transitions"]], ["node_changed"]
        )
        transition = a0["transitions"][0]
        self.assertEqual(
            list(transition), ["kind", "before", "after"]
        )
        self.assertEqual(transition["before"]["checkpoints"], (0,))
        self.assertEqual(transition["before"]["affected"], ())
        self.assertEqual(transition["after"]["checkpoints"], (1,))
        self.assertEqual(transition["after"]["affected"], ("a1",))

    def test_gap_lifetime_uses_interval_and_member_identity(self) -> None:
        store, scenario, points = gap_store()
        result = store.cascade_slice_lifetimes(**gap_kwargs(scenario, points))
        records = {r["identity"]: r for r in result["lifetimes"]}
        by_type = {}
        for record in result["lifetimes"]:
            by_type.setdefault(record["type"], []).append(record)
        self.assertEqual(list(by_type), ["node", "gap"])
        node = records[("p1", "k")]
        self.assertEqual(node["intervals"], ((0, 1),))
        self.assertEqual(node["transitions"], ())
        gap_record = records[(0, ())]
        self.assertEqual(gap_record["type"], "gap")
        self.assertEqual(
            (
                gap_record["first_seen"],
                gap_record["last_seen"],
                gap_record["intervals"],
            ),
            (1, 1, ((1, 1),)),
        )
        self.assertEqual(
            [t["kind"] for t in gap_record["transitions"]], ["gap_added"]
        )
        transition = gap_record["transitions"][0]
        self.assertIsNone(transition["before"])
        self.assertEqual(transition["after"], {"interval": 0, "members": ()})

    def test_transitions_match_timeline_segments_filtered_per_identity(self) -> None:
        store, args = diamond_store()
        kwargs = lifetime_kwargs(args, indices=(0, 1, 2))
        timeline_kwargs = dict(kwargs)
        timeline_kwargs.pop("lifetime_limit")
        timeline = store.cascade_slice_timeline(**timeline_kwargs)
        lifetimes = store.cascade_slice_lifetimes(**kwargs)["lifetimes"]
        per_identity: dict = {}
        for segment in timeline["segments"]:
            for change in segment["changes"]:
                per_identity.setdefault(change["identity"], []).append(
                    {
                        "kind": change["kind"],
                        "before": change["before"],
                        "after": change["after"],
                    }
                )
        for record in lifetimes:
            self.assertEqual(
                record["transitions"],
                tuple(per_identity.get(record["identity"], ())),
            )


class SliceLifetimeAssemblyTests(unittest.TestCase):
    """Deterministic coverage of the pure lifetime assembly classification."""

    @staticmethod
    def node(cause, key, **overrides):
        record = {
            "cause": cause,
            "key": key,
            "intervals": (0,),
            "checkpoints": (0,),
            "branches": ("a",),
            "affected": (),
        }
        record.update(overrides)
        return record

    @staticmethod
    def edge(source, target, path):
        return {"source": source, "target": target, "path": tuple(path)}

    @staticmethod
    def gap(interval, members):
        return {"interval": interval, "members": tuple(members)}

    def build(self, index_values, slices):
        segments = [
            BranchStore._build_slice_changes(left, right)
            for left, right in zip(slices, slices[1:])
        ]
        return BranchStore._build_slice_lifetimes(
            index_values, slices, segments
        )

    def test_disappearance_and_reappearance_opens_a_new_interval(self) -> None:
        # Selected checkpoints 0, 2, 4: A exists, vanishes, returns; B
        # persists throughout.
        slices = [
            {
                "nodes": (self.node("A", "k"), self.node("B", "k")),
                "edges": (),
                "gaps": (),
            },
            {
                "nodes": (self.node("B", "k"),),
                "edges": (),
                "gaps": (),
            },
            {
                "nodes": (
                    self.node("A", "k", affected=("z",)),
                    self.node("B", "k"),
                ),
                "edges": (),
                "gaps": (),
            },
        ]
        records = self.build([0, 2, 4], slices)
        by_identity = {r["identity"]: r for r in records}
        self.assertEqual(by_identity[("A", "k")]["intervals"], ((0, 0), (4, 4)))
        self.assertEqual(
            (
                by_identity[("A", "k")]["first_seen"],
                by_identity[("A", "k")]["last_seen"],
            ),
            (0, 4),
        )
        self.assertEqual(by_identity[("B", "k")]["intervals"], ((0, 4),))
        self.assertEqual(
            [t["kind"] for t in by_identity[("A", "k")]["transitions"]],
            ["node_removed", "node_added"],
        )
        self.assertEqual(by_identity[("B", "k")]["transitions"], ())

    def test_edge_and_gap_lifetimes_split_like_nodes(self) -> None:
        slices = [
            {
                "nodes": (self.node("A", "k"), self.node("B", "k")),
                "edges": (self.edge(0, 1, ("A", "B")),),
                "gaps": (self.gap(0, ("m",)),),
            },
            {
                "nodes": (self.node("A", "k"), self.node("B", "k")),
                "edges": (),
                "gaps": (),
            },
            {
                "nodes": (self.node("A", "k"), self.node("B", "k")),
                "edges": (self.edge(0, 1, ("A", "B", "X")),),
                "gaps": (self.gap(0, ("m",)),),
            },
        ]
        records = self.build([0, 1, 2], slices)
        by_identity = {
            (r["type"], r["identity"]): r for r in records
        }
        edge_record = by_identity[("edge", (("A", "k"), ("B", "k")))]
        self.assertEqual(edge_record["intervals"], ((0, 0), (2, 2)))
        self.assertEqual(
            [t["kind"] for t in edge_record["transitions"]],
            ["edge_removed", "edge_added"],
        )
        gap_record = by_identity[("gap", (0, ("m",)))]
        self.assertEqual(gap_record["intervals"], ((0, 0), (2, 2)))
        self.assertEqual(
            [t["kind"] for t in gap_record["transitions"]],
            ["gap_removed", "gap_added"],
        )

    def test_edge_path_change_is_a_content_transition_without_interval_gap(self) -> None:
        slices = [
            {
                "nodes": (self.node("A", "k"), self.node("B", "k")),
                "edges": (self.edge(0, 1, ("A", "B", "p")),),
                "gaps": (),
            },
            {
                "nodes": (self.node("A", "k"), self.node("B", "k")),
                "edges": (self.edge(0, 1, ("A", "B", "q")),),
                "gaps": (),
            },
        ]
        records = self.build([3, 7], slices)
        edge_record = next(r for r in records if r["type"] == "edge")
        self.assertEqual(edge_record["intervals"], ((3, 7),))
        self.assertEqual(
            [t["kind"] for t in edge_record["transitions"]], ["edge_changed"]
        )
        transition = edge_record["transitions"][0]
        self.assertEqual(transition["before"]["path"], ("A", "B", "p"))
        self.assertEqual(transition["after"]["path"], ("A", "B", "q"))

    def test_ordering_uses_first_seen_then_existing_identity_order(self) -> None:
        # At checkpoint 0: nodes A, B in slice order; at checkpoint 1: C then
        # a returning A. Same first_seen keeps evidence order, and the later
        # first-seen node sorts after both.
        slices = [
            {
                "nodes": (self.node("A", "k"), self.node("B", "k")),
                "edges": (),
                "gaps": (),
            },
            {
                "nodes": (self.node("C", "k"), self.node("A", "k")),
                "edges": (),
                "gaps": (),
            },
        ]
        records = self.build([0, 1], slices)
        self.assertEqual(
            [(r["identity"], r["first_seen"]) for r in records],
            [(("A", "k"), 0), (("B", "k"), 0), (("C", "k"), 1)],
        )

    def test_intervals_are_closed_two_tuples_in_time_order(self) -> None:
        slices = [
            {"nodes": (self.node("A", "k"),), "edges": (), "gaps": ()},
            {"nodes": (), "edges": (), "gaps": ()},
            {"nodes": (self.node("A", "k"),), "edges": (), "gaps": ()},
            {"nodes": (), "edges": (), "gaps": ()},
            {"nodes": (self.node("A", "k"),), "edges": (), "gaps": ()},
        ]
        records = self.build([0, 2, 4, 6, 8], slices)
        node = next(iter(records))
        self.assertEqual(node["intervals"], ((0, 0), (4, 4), (8, 8)))
        self.assertTrue(all(len(interval) == 2 for interval in node["intervals"]))


class CascadeSliceLifetimesLimitValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        overrides.setdefault("causes", ("a0",))
        return lifetime_call(self.store, self.args, **overrides)

    def test_lifetime_limit_must_be_a_non_bool_int(self) -> None:
        for bad in (True, False, 1.0, "1", None):
            with self.subTest(lifetime_limit=bad):
                with self.assertRaises(TypeError):
                    self.call(lifetime_limit=bad)

    def test_lifetime_limit_below_one_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            self.call(lifetime_limit=0)
        with self.assertRaises(ValueError):
            self.call(lifetime_limit=-3)

    def test_too_many_records_raise_and_return_nothing(self) -> None:
        # The (0, 1, 2) window produces three lifetime records.
        self.assertEqual(len(self.call(lifetime_limit=3)["lifetimes"]), 3)
        with self.assertRaises(ValueError):
            self.call(lifetime_limit=2)
        with self.assertRaises(ValueError):
            self.call(lifetime_limit=1)

    def test_lifetime_limit_is_validated_last_of_the_inputs(self) -> None:
        # node_limit's input check precedes change_limit, which precedes
        # lifetime_limit.
        with self.assertRaises(ValueError):
            self.call(node_limit=0, change_limit=0, lifetime_limit=0)
        with self.assertRaises(ValueError):
            self.call(node_limit=1, change_limit=0, lifetime_limit=0)
        with self.assertRaises(TypeError):
            self.call(node_limit=1, change_limit="x", lifetime_limit=0)
        with self.assertRaises(TypeError):
            self.call(node_limit=1, change_limit=1, lifetime_limit="x")
        with self.assertRaises(ValueError):
            self.call(node_limit=1, change_limit=1, lifetime_limit=0)


class CascadeSliceLifetimesValidationOrderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        overrides.setdefault("causes", ("a0",))
        return lifetime_call(self.store, self.args, **overrides)

    def test_ordinary_inputs_validated_first(self) -> None:
        with self.assertRaises(TypeError):
            self.call(reference=1, indices="x", causes=[1])
        with self.assertRaises(ValueError):
            self.call(axis="nope", indices=(-1,))
        with self.assertRaises(TypeError):
            self.call(values=[0, 1], indices=(0,))

    def test_indices_validated_before_slice_and_lifetime_inputs(self) -> None:
        with self.assertRaises(TypeError):
            self.call(indices="x", causes=[1])
        with self.assertRaises(ValueError):
            self.call(indices=(-1,), causes=("a0", "a0"))
        with self.assertRaises(ValueError):
            self.call(indices=(9,), direction="sideways")
        with self.assertRaises(TypeError):
            self.call(indices=(0, True), depth=True)
        # indices checks win over a bad lifetime_limit as well.
        with self.assertRaises(ValueError):
            self.call(indices=(0, 0), lifetime_limit="x")

    def test_all_input_validation_precedes_state_lookups(self) -> None:
        with self.assertRaises(TypeError):
            self.call(reference="ghost", indices="x")
        with self.assertRaises(ValueError):
            self.call(reference="ghost", indices=(-1,))
        with self.assertRaises(KeyError):
            self.call(reference="ghost")

    def test_branch_and_node_lookup_contracts_are_unchanged(self) -> None:
        with self.assertRaises(KeyError):
            self.call(
                series={
                    "a": (("r0", "a0"), ("r1", "a0"), ("r2", "a0")),
                    "b": (("r0", "a0"), ("r1", "W"), ("r2", "nope2")),
                },
            )

    def test_candidate_node_and_change_caps_still_breach(self) -> None:
        with self.assertRaises(ValueError):
            self.call(limit=1)
        with self.assertRaises(ValueError):
            self.call(indices=(0, 1), node_limit=1)
        with self.assertRaises(ValueError):
            self.call(indices=(0, 1), change_limit=1)

    def test_empty_plan_has_no_valid_index(self) -> None:
        with self.assertRaises(ValueError):
            self.call(series={}, max_size=0, indices=(0,), causes=())


class CascadeSliceLifetimesCauseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        return lifetime_call(self.store, self.args, **overrides)

    def test_cause_absent_from_every_slice_raises(self) -> None:
        with self.assertRaises(KeyError) as caught:
            self.call(indices=(0, 1), causes=("zzz",))
        self.assertEqual(caught.exception.args[0], "zzz")

    def test_first_absent_cause_in_input_order_wins(self) -> None:
        with self.assertRaises(KeyError) as caught:
            self.call(indices=(0, 1), causes=("zzz", "a0"))
        self.assertEqual(caught.exception.args[0], "zzz")

    def test_cause_present_in_any_one_slice_is_accepted(self) -> None:
        result = self.call(indices=(0, 1), causes=("a0", "W"))
        self.assertIn("lifetimes", result)

    def test_graph_event_that_is_never_a_slice_node_raises(self) -> None:
        with self.assertRaises(KeyError):
            self.call(indices=(0, 1), causes=("r0",))


class CascadeSliceLifetimesEmptyInputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store, self.args = diamond_store()

    def test_empty_indices_return_an_empty_tuple(self) -> None:
        result = lifetime_call(self.store, self.args, indices=(), causes=("a0",))
        self.assertEqual(result, {"lifetimes": ()})

    def test_empty_indices_still_run_input_and_state_checks(self) -> None:
        with self.assertRaises(KeyError):
            lifetime_call(
                self.store, self.args, indices=(), reference="ghost", causes=()
            )
        with self.assertRaises(ValueError):
            lifetime_call(
                self.store, self.args, indices=(), limit=1, causes=()
            )
        with self.assertRaises(ValueError):
            lifetime_call(
                self.store, self.args, indices=(), node_limit=0, causes=()
            )
        with self.assertRaises(ValueError):
            lifetime_call(
                self.store, self.args, indices=(), change_limit=0, causes=()
            )
        with self.assertRaises(ValueError):
            lifetime_call(
                self.store, self.args, indices=(), lifetime_limit=0, causes=()
            )
        with self.assertRaises(TypeError):
            lifetime_call(
                self.store, self.args, indices=(), causes=[1]
            )

    def test_empty_causes_return_an_empty_tuple(self) -> None:
        result = lifetime_call(
            self.store, self.args, indices=(0, 1, 2), causes=()
        )
        self.assertEqual(result, {"lifetimes": ()})

    def test_empty_causes_still_run_state_and_range_checks(self) -> None:
        with self.assertRaises(KeyError):
            lifetime_call(self.store, self.args, reference="ghost", causes=())
        with self.assertRaises(ValueError):
            lifetime_call(self.store, self.args, limit=1, causes=())
        with self.assertRaises(ValueError):
            lifetime_call(
                self.store, self.args, indices=(5, 6), causes=()
            )
        with self.assertRaises(ValueError):
            lifetime_call(
                self.store, self.args, node_limit=0, causes=()
            )
        with self.assertRaises(ValueError):
            lifetime_call(
                self.store, self.args, lifetime_limit=0, causes=()
            )


class CascadeSliceLifetimesIsolationTests(unittest.TestCase):
    def test_result_levels_are_mutually_unshared(self) -> None:
        store, args = diamond_store()
        result = lifetime_call(store, args, indices=(0, 1, 2))
        record_ids = []
        for record in result["lifetimes"]:
            record_ids.append(id(record))
            record_ids.append(id(record["intervals"]))
            record_ids.append(id(record["transitions"]))
            for transition in record["transitions"]:
                record_ids.append(id(transition))
                for side in ("before", "after"):
                    evidence = transition[side]
                    if evidence is not None:
                        record_ids.append(id(evidence))
                        # Empty tuples are interned by Python itself; only
                        # non-empty evidence containers can be distinct.
                        record_ids.extend(
                            id(value)
                            for value in evidence.values()
                            if isinstance(value, tuple) and value
                        )
        self.assertEqual(len(record_ids), len(set(record_ids)))

    def test_mutating_result_never_affects_later_calls(self) -> None:
        store, args = diamond_store()
        first = lifetime_call(store, args, indices=(0, 1, 2))
        pristine = copy.deepcopy(first)
        for record in first["lifetimes"]:
            record["intervals"] += (9, 9)
            for transition in record["transitions"]:
                for side in ("before", "after"):
                    evidence = transition[side]
                    if evidence is None:
                        continue
                    if "path" in evidence:
                        evidence["path"] += ("evil",)
                    elif "affected" in evidence:
                        evidence["affected"] += ("evil",)
                    else:
                        evidence["members"] += ("evil",)
        second = lifetime_call(store, args, indices=(0, 1, 2))
        self.assertEqual(second, pristine)

    def test_repeated_calls_are_itemwise_equal(self) -> None:
        store, args = diamond_store()
        first = lifetime_call(store, args, indices=(0, 1, 2))
        second = lifetime_call(store, args, indices=(0, 1, 2))
        self.assertEqual(first, second)


class CascadeSliceLifetimesReadOnlyTests(unittest.TestCase):
    def test_success_and_failure_change_nothing(self) -> None:
        store, args = diamond_store()
        heads = dict(store._heads)
        appends = copy.deepcopy(store._appends)
        merges = copy.deepcopy(store._merges)
        audit = store.audit_log()

        lifetime_call(store, args, indices=(0, 1, 2))
        failures = [
            dict(indices="x"),
            dict(indices=(-1,)),
            dict(indices=(9,)),
            dict(indices=(1, 0)),
            dict(indices=(1, 1)),
            dict(causes=[1]),
            dict(direction="up"),
            dict(depth=-1),
            dict(node_limit=0),
            dict(change_limit=0),
            dict(change_limit=1),
            dict(lifetime_limit=0),
            dict(lifetime_limit="x"),
            dict(lifetime_limit=2),
            dict(causes=("zzz",)),
            dict(limit=1),
            dict(reference="ghost"),
        ]
        for overrides in failures:
            with self.subTest(overrides=overrides):
                with self.assertRaises(Exception):
                    lifetime_call(store, args, causes=("a0",), **overrides)

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
        diff_kwargs = lifetime_kwargs(args)
        diff_kwargs.pop("indices")
        diff_kwargs.pop("change_limit")
        diff_kwargs.pop("lifetime_limit")
        diff = store.cascade_slice_diff(before=0, after=1, **diff_kwargs)
        timeline_kwargs = lifetime_kwargs(args)
        timeline_kwargs.pop("lifetime_limit")
        timeline = store.cascade_slice_timeline(**timeline_kwargs)
        lifetime_call(store, args, indices=(0, 1, 2))
        self.assertEqual(store.decision_cascade(*args), cascade)
        self.assertEqual(
            store.cascade_slice(
                *args, causes=("a0",), direction="both", depth=99,
                node_limit=50,
            ),
            slice_result,
        )
        self.assertEqual(
            store.cascade_slice_diff(before=0, after=1, **diff_kwargs), diff
        )
        self.assertEqual(
            store.cascade_slice_timeline(**timeline_kwargs), timeline
        )

    def test_no_new_cli_command_is_added(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "city_twin", "lifetimes"],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        status = subprocess.run(
            [sys.executable, "-m", "city_twin", "status"],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("ready", status.stdout)


if __name__ == "__main__":
    unittest.main()
