import copy
import inspect
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


def lifetimes_kwargs(store_args, **overrides) -> dict:
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
        indices=(0, 1),
        causes=("a0",),
        direction="both",
        depth=99,
        node_limit=50,
        change_limit=50,
        lifetime_limit=50,
    )
    kwargs.update(overrides)
    return kwargs


def lifetimes_call(store, store_args, **overrides):
    return store.cascade_slice_lifetimes(
        **lifetimes_kwargs(store_args, **overrides)
    )


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
                "token",
            ],
        )
        self.assertEqual(
            [p.default for p in parameters[:-1]],
            [inspect.Parameter.empty] * (len(parameters) - 1),
        )
        self.assertIsNone(parameters[-1].default)

    def test_result_is_a_dict_with_only_lifetimes(self) -> None:
        store, args = diamond_store()
        result = lifetimes_call(store, args, indices=(0, 1, 2))
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


class CascadeSliceLifetimesRecordTests(unittest.TestCase):
    def test_diamond_lifetimes(self) -> None:
        store, args = diamond_store()
        result = lifetimes_call(store, args, indices=(0, 1, 2), causes=("a0",))
        records = result["lifetimes"]
        self.assertEqual(
            [(r["type"], r["identity"]) for r in records],
            [
                ("node", ("a0", "x")),
                ("node", ("W", "y")),
                ("edge", (("a0", "x"), ("W", "y"))),
            ],
        )
        a0, w, edge = records
        self.assertEqual((a0["first_seen"], a0["last_seen"]), (0, 2))
        self.assertEqual(a0["intervals"], ((0, 2),))
        self.assertEqual((w["first_seen"], w["last_seen"]), (1, 2))
        self.assertEqual(w["intervals"], ((1, 2),))
        self.assertEqual((edge["first_seen"], edge["last_seen"]), (1, 2))
        self.assertEqual(edge["intervals"], ((1, 2),))

    def test_first_snapshot_evidence_gains_no_fabricated_addition(self) -> None:
        store, args = diamond_store()
        result = lifetimes_call(store, args, indices=(0, 1, 2), causes=("a0",))
        a0 = result["lifetimes"][0]
        self.assertEqual(a0["identity"], ("a0", "x"))
        self.assertEqual(a0["transitions"], ())

    def test_transitions_keep_classification_and_both_sides(self) -> None:
        store, args = diamond_store()
        result = lifetimes_call(store, args, indices=(0, 1, 2), causes=("a0",))
        w = result["lifetimes"][1]
        (transition,) = w["transitions"]
        self.assertEqual(list(transition), ["kind", "before", "after"])
        self.assertEqual(transition["kind"], "node_added")
        self.assertIsNone(transition["before"])
        self.assertEqual(
            set(transition["after"]),
            {"cause", "key", "intervals", "checkpoints", "branches", "affected"},
        )
        edge = result["lifetimes"][2]
        (edge_transition,) = edge["transitions"]
        self.assertEqual(edge_transition["kind"], "edge_added")
        self.assertIsNone(edge_transition["before"])
        self.assertEqual(edge_transition["after"]["path"], ("a0", "bu", "W"))

    def test_lifetimes_match_the_timeline_segments(self) -> None:
        store, args = diamond_store()
        result = lifetimes_call(store, args, indices=(0, 1, 2), causes=("a0",))
        timeline_kwargs = lifetimes_kwargs(args, indices=(0, 1, 2))
        timeline_kwargs.pop("lifetime_limit")
        timeline = store.cascade_slice_timeline(**timeline_kwargs)
        changes_by_identity = {}
        for segment in timeline["segments"]:
            for change in segment["changes"]:
                changes_by_identity.setdefault(change["identity"], []).append(
                    {
                        "kind": change["kind"],
                        "before": change["before"],
                        "after": change["after"],
                    }
                )
        for record in result["lifetimes"]:
            self.assertEqual(
                list(record["transitions"]),
                changes_by_identity.get(record["identity"], []),
            )
        # Presence matches the timeline snapshots exactly.
        for record in result["lifetimes"]:
            seen = []
            for snapshot in timeline["snapshots"]:
                nodes = snapshot["slice"]["nodes"]
                present = set()
                if record["type"] == "node":
                    present = {(n["cause"], n["key"]) for n in nodes}
                elif record["type"] == "edge":
                    present = {
                        (
                            (nodes[e["source"]]["cause"], nodes[e["source"]]["key"]),
                            (nodes[e["target"]]["cause"], nodes[e["target"]]["key"]),
                        )
                        for e in snapshot["slice"]["edges"]
                    }
                else:
                    present = {
                        (g["interval"], tuple(g["members"]))
                        for g in snapshot["slice"]["gaps"]
                    }
                if record["identity"] in present:
                    seen.append(snapshot["index"])
            self.assertEqual(seen[0], record["first_seen"])
            self.assertEqual(seen[-1], record["last_seen"])

    def test_single_index_yields_records_without_transitions(self) -> None:
        store, args = diamond_store()
        result = lifetimes_call(store, args, indices=(1,), causes=("a0",))
        self.assertEqual(
            [(r["type"], r["identity"]) for r in result["lifetimes"]],
            [("node", ("a0", "x")), ("node", ("W", "y")),
             ("edge", (("a0", "x"), ("W", "y")))],
        )
        for record in result["lifetimes"]:
            self.assertEqual(record["first_seen"], 1)
            self.assertEqual(record["last_seen"], 1)
            self.assertEqual(record["intervals"], ((1, 1),))
            self.assertEqual(record["transitions"], ())

    def test_records_order_by_type_then_first_seen(self) -> None:
        store, args = diamond_store()
        result = lifetimes_call(store, args, indices=(0, 1, 2), causes=("a0",))
        types = [r["type"] for r in result["lifetimes"]]
        self.assertEqual(types, ["node", "node", "edge"])
        node_firsts = [
            r["first_seen"] for r in result["lifetimes"] if r["type"] == "node"
        ]
        self.assertEqual(node_firsts, sorted(node_firsts))


class CascadeSliceLifetimesAggregationTests(unittest.TestCase):
    """Direct checks of the aggregation over hand-built snapshots/segments."""

    def test_reappearance_opens_a_new_interval(self) -> None:
        node = {
            "cause": "a",
            "key": "x",
            "intervals": (),
            "checkpoints": (),
            "branches": (),
            "affected": (),
        }
        snapshots = (
            {"index": 0, "slice": {"nodes": (node,), "edges": (), "gaps": ()}},
            {"index": 1, "slice": {"nodes": (), "edges": (), "gaps": ()}},
            {"index": 2, "slice": {"nodes": (node,), "edges": (), "gaps": ()}},
        )
        segments = (
            {
                "before": 0,
                "after": 1,
                "changes": (
                    {
                        "kind": "node_removed",
                        "identity": ("a", "x"),
                        "before": node,
                        "after": None,
                    },
                ),
            },
            {
                "before": 1,
                "after": 2,
                "changes": (
                    {
                        "kind": "node_added",
                        "identity": ("a", "x"),
                        "before": None,
                        "after": node,
                    },
                ),
            },
        )
        (record,) = BranchStore._build_slice_lifetimes(snapshots, segments)
        self.assertEqual(record["type"], "node")
        self.assertEqual(record["identity"], ("a", "x"))
        self.assertEqual((record["first_seen"], record["last_seen"]), (0, 2))
        self.assertEqual(record["intervals"], ((0, 0), (2, 2)))
        self.assertEqual(
            [t["kind"] for t in record["transitions"]],
            ["node_removed", "node_added"],
        )

    def test_gap_and_edge_identities_and_type_ordering(self) -> None:
        node = {
            "cause": "a",
            "key": "x",
            "intervals": (),
            "checkpoints": (),
            "branches": (),
            "affected": (),
        }
        other = dict(node, cause="b", key="y")
        edge = {"source": 0, "target": 1, "path": ("a", "b")}
        gap = {"interval": 0, "members": ("m",)}
        snapshots = (
            {
                "index": 0,
                "slice": {
                    "nodes": (node, other),
                    "edges": (edge,),
                    "gaps": (gap,),
                },
            },
        )
        records = BranchStore._build_slice_lifetimes(snapshots, ())
        self.assertEqual(
            [(r["type"], r["identity"]) for r in records],
            [
                ("node", ("a", "x")),
                ("node", ("b", "y")),
                ("edge", (("a", "x"), ("b", "y"))),
                ("gap", (0, ("m",))),
            ],
        )
        for record in records:
            self.assertEqual(record["intervals"], ((0, 0),))
            self.assertEqual(record["transitions"], ())

    def test_empty_snapshots_and_segments_yield_no_records(self) -> None:
        self.assertEqual(BranchStore._build_slice_lifetimes((), ()), ())


class CascadeSliceLifetimesLifetimeLimitValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        overrides.setdefault("causes", ("a0",))
        return lifetimes_call(self.store, self.args, **overrides)

    def test_lifetime_limit_must_be_a_non_bool_int(self) -> None:
        for bad in (True, False, 1.0, "1", None):
            with self.subTest(lifetime_limit=bad):
                with self.assertRaises(TypeError):
                    self.call(lifetime_limit=bad)

    def test_lifetime_limit_below_one_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            self.call(lifetime_limit=0)
        with self.assertRaises(ValueError):
            self.call(lifetime_limit=-2)

    def test_records_beyond_lifetime_limit_raise_value_error(self) -> None:
        # The (0, 1, 2) window carries three lifetime records.
        with self.assertRaises(ValueError):
            self.call(indices=(0, 1, 2), lifetime_limit=2)
        with self.assertRaises(ValueError):
            self.call(indices=(0, 1, 2), lifetime_limit=1)

    def test_lifetime_limit_covering_the_records_succeeds(self) -> None:
        result = self.call(indices=(0, 1, 2), lifetime_limit=3)
        self.assertEqual(len(result["lifetimes"]), 3)

    def test_lifetime_limit_validated_after_change_limit(self) -> None:
        # change_limit is checked first, so its error wins over a bad
        # lifetime_limit; a valid change_limit then surfaces
        # lifetime_limit's.
        with self.assertRaises(ValueError):
            self.call(change_limit=0, lifetime_limit=0)
        with self.assertRaises(ValueError):
            self.call(change_limit=0, lifetime_limit="x")
        with self.assertRaises(TypeError):
            self.call(change_limit=1, lifetime_limit="x")

    def test_lifetime_limit_validated_after_slice_inputs(self) -> None:
        with self.assertRaises(ValueError):
            self.call(node_limit=0, lifetime_limit=0)
        with self.assertRaises(TypeError):
            self.call(depth=True, lifetime_limit=True)


class CascadeSliceLifetimesValidationOrderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        overrides.setdefault("causes", ("a0",))
        return lifetimes_call(self.store, self.args, **overrides)

    def test_ordinary_inputs_validated_first(self) -> None:
        with self.assertRaises(TypeError):
            self.call(reference=1, indices="x", causes=[1], lifetime_limit=0)
        with self.assertRaises(ValueError):
            self.call(axis="nope", indices=(-1,), lifetime_limit=0)
        with self.assertRaises(TypeError):
            self.call(values=[0, 1], indices=(0,), lifetime_limit=0)

    def test_indices_validated_before_slice_inputs_and_limits(self) -> None:
        with self.assertRaises(TypeError):
            self.call(indices="x", causes=[1], lifetime_limit=0)
        with self.assertRaises(ValueError):
            self.call(indices=(-1,), causes=("a0", "a0"), lifetime_limit=0)
        with self.assertRaises(ValueError):
            self.call(indices=(9,), direction="sideways", lifetime_limit=0)
        with self.assertRaises(TypeError):
            self.call(indices=(0, True), depth=True, lifetime_limit=0)

    def test_all_input_validation_precedes_state_lookups(self) -> None:
        with self.assertRaises(TypeError):
            self.call(reference="ghost", indices="x")
        with self.assertRaises(ValueError):
            self.call(reference="ghost", lifetime_limit=0)
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

    def test_candidate_cap_still_breaches_with_no_partial_result(self) -> None:
        with self.assertRaises(ValueError):
            self.call(limit=1)

    def test_node_and_change_caps_still_breach(self) -> None:
        with self.assertRaises(ValueError):
            self.call(indices=(0, 1), direction="forward", node_limit=1)
        with self.assertRaises(ValueError):
            self.call(indices=(0, 1), change_limit=1)


class CascadeSliceLifetimesCauseTests(unittest.TestCase):
    def test_cause_absent_from_every_slice_raises(self) -> None:
        store, args = diamond_store()
        with self.assertRaises(KeyError) as caught:
            lifetimes_call(store, args, indices=(0, 1), causes=("zzz",))
        self.assertEqual(caught.exception.args[0], "zzz")

    def test_first_absent_cause_in_input_order_wins(self) -> None:
        store, args = diamond_store()
        with self.assertRaises(KeyError) as caught:
            lifetimes_call(store, args, indices=(0, 1), causes=("zzz", "a0"))
        self.assertEqual(caught.exception.args[0], "zzz")

    def test_cause_present_in_any_one_slice_is_accepted(self) -> None:
        store, args = diamond_store()
        result = lifetimes_call(store, args, indices=(0, 1), causes=("a0", "W"))
        self.assertTrue(result["lifetimes"])

    def test_graph_event_that_is_never_a_slice_node_raises(self) -> None:
        store, args = diamond_store()
        with self.assertRaises(KeyError):
            lifetimes_call(store, args, indices=(0, 1), causes=("r0",))


class CascadeSliceLifetimesEmptyInputTests(unittest.TestCase):
    def test_empty_indices_return_empty_lifetimes(self) -> None:
        store, args = diamond_store()
        result = lifetimes_call(store, args, indices=(), causes=("a0",))
        self.assertEqual(result, {"lifetimes": ()})

    def test_empty_indices_still_run_input_and_state_checks(self) -> None:
        store, args = diamond_store()
        with self.assertRaises(KeyError):
            lifetimes_call(store, args, indices=(), reference="ghost", causes=())
        with self.assertRaises(ValueError):
            lifetimes_call(store, args, indices=(), limit=1, causes=())
        with self.assertRaises(ValueError):
            lifetimes_call(store, args, indices=(), node_limit=0, causes=())
        with self.assertRaises(ValueError):
            lifetimes_call(store, args, indices=(), change_limit=0, causes=())
        with self.assertRaises(ValueError):
            lifetimes_call(store, args, indices=(), lifetime_limit=0, causes=())
        with self.assertRaises(TypeError):
            lifetimes_call(store, args, indices=(), causes=[1])

    def test_empty_causes_return_empty_lifetimes(self) -> None:
        store, args = diamond_store()
        result = lifetimes_call(
            store, args, indices=(0, 1, 2), causes=(),
            direction="forward", depth=5, node_limit=1,
        )
        self.assertEqual(result, {"lifetimes": ()})

    def test_empty_causes_still_run_state_and_range_checks(self) -> None:
        store, args = diamond_store()
        with self.assertRaises(KeyError):
            lifetimes_call(store, args, reference="ghost", causes=())
        with self.assertRaises(ValueError):
            lifetimes_call(store, args, limit=1, causes=())
        with self.assertRaises(ValueError):
            lifetimes_call(store, args, indices=(5, 6), causes=())
        with self.assertRaises(ValueError):
            lifetimes_call(store, args, node_limit=0, causes=())


class CascadeSliceLifetimesIsolationTests(unittest.TestCase):
    def test_result_levels_are_mutually_unshared(self) -> None:
        store, args = diamond_store()
        result = lifetimes_call(store, args, indices=(0, 1, 2), causes=("a0",))
        record_ids = []
        for record in result["lifetimes"]:
            record_ids.append(id(record))
            record_ids.extend(id(t) for t in record["transitions"])
            record_ids.extend(
                id(t[side])
                for t in record["transitions"]
                for side in ("before", "after")
                if t[side] is not None
            )
        self.assertEqual(len(record_ids), len(set(record_ids)))

    def test_mutating_result_never_affects_later_calls(self) -> None:
        store, args = diamond_store()
        first = lifetimes_call(store, args, indices=(0, 1, 2), causes=("a0",))
        pristine = copy.deepcopy(first)
        for record in first["lifetimes"]:
            for transition in record["transitions"]:
                for side in ("before", "after"):
                    evidence = transition[side]
                    if evidence is None:
                        continue
                    if "path" in evidence:
                        evidence["path"] += ("evil",)
                    if "branches" in evidence:
                        evidence["branches"] += ("evil",)
        second = lifetimes_call(store, args, indices=(0, 1, 2), causes=("a0",))
        self.assertEqual(second, pristine)


class CascadeSliceLifetimesReadOnlyTests(unittest.TestCase):
    def test_success_and_failure_change_nothing(self) -> None:
        store, args = diamond_store()
        heads = dict(store._heads)
        appends = copy.deepcopy(store._appends)
        merges = copy.deepcopy(store._merges)
        audit = store.audit_log()

        lifetimes_call(store, args, indices=(0, 1, 2), causes=("a0",))
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
            dict(lifetime_limit=1),
            dict(causes=("zzz",)),
            dict(limit=1),
            dict(reference="ghost"),
        ]
        for overrides in failures:
            with self.subTest(overrides=overrides):
                with self.assertRaises(Exception):
                    lifetimes_call(store, args, causes=("a0",), **overrides)

        self.assertEqual(dict(store._heads), heads)
        self.assertEqual(copy.deepcopy(store._appends), appends)
        self.assertEqual(copy.deepcopy(store._merges), merges)
        self.assertEqual(store.audit_log(), audit)

    def test_existing_queries_are_unaffected(self) -> None:
        store, args = diamond_store()
        cascade = store.decision_cascade(*args)
        timeline_kwargs = lifetimes_kwargs(args, indices=(0, 1, 2))
        timeline_kwargs.pop("lifetime_limit")
        timeline = store.cascade_slice_timeline(**timeline_kwargs)
        lifetimes_call(store, args, indices=(0, 1, 2), causes=("a0",))
        self.assertEqual(store.decision_cascade(*args), cascade)
        self.assertEqual(
            store.cascade_slice_timeline(**timeline_kwargs), timeline
        )

    def test_repeated_calls_are_equal(self) -> None:
        store, args = diamond_store()
        first = lifetimes_call(store, args, indices=(0, 1, 2), causes=("a0",))
        second = lifetimes_call(store, args, indices=(0, 1, 2), causes=("a0",))
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
