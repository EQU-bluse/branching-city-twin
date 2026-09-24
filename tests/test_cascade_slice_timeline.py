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


def timeline_kwargs(store_args, **overrides) -> dict:
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
    )
    kwargs.update(overrides)
    return kwargs


def timeline_call(store, store_args, **overrides):
    return store.cascade_slice_timeline(
        **timeline_kwargs(store_args, **overrides)
    )


class CascadeSliceTimelineSignatureTests(unittest.TestCase):
    def test_public_signature_replaces_indices_and_appends_change_limit(self) -> None:
        parameters = list(
            inspect.signature(BranchStore.cascade_slice_timeline).parameters.values()
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
            ],
        )
        self.assertTrue(
            all(p.default is inspect.Parameter.empty for p in parameters)
        )

    def test_result_is_a_dict_with_snapshots_and_segments(self) -> None:
        store, args = diamond_store()
        result = timeline_call(store, args, indices=(0, 1, 2))
        self.assertIsInstance(result, dict)
        self.assertEqual(list(result), ["snapshots", "segments"])
        self.assertIsInstance(result["snapshots"], tuple)
        self.assertIsInstance(result["segments"], tuple)
        for snapshot in result["snapshots"]:
            self.assertEqual(list(snapshot), ["index", "slice"])
            self.assertEqual(list(snapshot["slice"]), ["nodes", "edges", "gaps"])
        for segment in result["segments"]:
            self.assertEqual(list(segment), ["before", "after", "changes"])
            self.assertIsInstance(segment["changes"], tuple)


class CascadeSliceTimelineSnapshotTests(unittest.TestCase):
    def test_each_snapshot_equals_the_slice_on_its_checkpoint_prefix(self) -> None:
        store, args = diamond_store()
        _, series, scenario, axis, values, mn, mx, req, excl, limit = args
        slice_args_common = (scenario, axis, values, mn, mx, req, excl, limit)

        result = timeline_call(
            store, args, indices=(0, 1, 2), causes=("a0",),
            direction="forward", depth=99, node_limit=50,
        )
        self.assertEqual(
            [snapshot["index"] for snapshot in result["snapshots"]],
            [0, 1, 2],
        )
        for index, snapshot in enumerate(result["snapshots"]):
            standalone = store.cascade_slice(
                "main", {k: v[: index + 1] for k, v in series.items()},
                *slice_args_common,
                causes=("a0",), direction="forward", depth=99, node_limit=50,
            )
            self.assertEqual(snapshot["slice"], standalone)

    def test_snapshots_use_only_the_prefix_not_the_full_plan(self) -> None:
        store, args = diamond_store()
        result = timeline_call(store, args, indices=(0, 1, 2), causes=("a0",))
        per_snapshot = [
            [(n["cause"], n["key"]) for n in snapshot["slice"]["nodes"]]
            for snapshot in result["snapshots"]
        ]
        self.assertEqual(
            per_snapshot,
            [[("a0", "x")], [("a0", "x"), ("W", "y")], [("a0", "x"), ("W", "y")]],
        )

    def test_single_index_yields_one_snapshot_and_no_segments(self) -> None:
        store, args = diamond_store()
        result = timeline_call(store, args, indices=(1,), causes=("a0",))
        self.assertEqual(len(result["snapshots"]), 1)
        self.assertEqual(result["snapshots"][0]["index"], 1)
        self.assertEqual(result["segments"], ())


class CascadeSliceTimelineSegmentTests(unittest.TestCase):
    def test_segments_compare_adjacent_indices_only(self) -> None:
        store, args = diamond_store()
        result = timeline_call(store, args, indices=(0, 1, 2), causes=("a0",))
        self.assertEqual(
            [(segment["before"], segment["after"]) for segment in result["segments"]],
            [(0, 1), (1, 2)],
        )

    def test_segment_changes_match_the_two_point_diff(self) -> None:
        store, args = diamond_store()
        result = timeline_call(store, args, indices=(0, 1, 2), causes=("a0",))
        # Compare against cascade_slice_diff for each adjacent pair.
        for segment in result["segments"]:
            kwargs = timeline_kwargs(args)
            kwargs.pop("indices")
            kwargs.pop("change_limit")
            diff = store.cascade_slice_diff(
                before=segment["before"], after=segment["after"], **kwargs
            )
            self.assertEqual(segment["changes"], diff["changes"])

    def test_change_identity_and_order_follow_the_diff_semantics(self) -> None:
        store, args = diamond_store()
        result = timeline_call(store, args, indices=(0, 1), causes=("a0",))
        (segment,) = result["segments"]
        kinds = [change["kind"] for change in segment["changes"]]
        self.assertEqual(kinds, ["node_added", "edge_added"])
        node_added = segment["changes"][0]
        self.assertEqual(list(node_added), ["kind", "identity", "before", "after"])
        self.assertEqual(node_added["identity"], ("W", "y"))
        self.assertIsNone(node_added["before"])
        edge_added = segment["changes"][1]
        self.assertEqual(edge_added["identity"], (("a0", "x"), ("W", "y")))
        self.assertEqual(edge_added["after"]["path"], ("a0", "bu", "W"))

    def test_unchanged_adjacent_positions_have_empty_changes(self) -> None:
        store, args = diamond_store()
        result = timeline_call(store, args, indices=(1, 2), causes=("a0",))
        (segment,) = result["segments"]
        self.assertEqual(segment["changes"], ())


class CascadeSliceTimelineIndicesValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store, self.args = diamond_store()  # three checkpoints

    def call(self, **overrides):
        overrides.setdefault("causes", ("a0",))
        return timeline_call(self.store, self.args, **overrides)

    def test_indices_must_be_a_tuple(self) -> None:
        for bad in ([0, 1], "01", 0, None):
            with self.subTest(indices=bad):
                with self.assertRaises(TypeError):
                    self.call(indices=bad)

    def test_elements_must_be_non_bool_ints(self) -> None:
        for bad in (True, False, 1.0, "1", None):
            with self.subTest(element=bad):
                with self.assertRaises(TypeError):
                    self.call(indices=(0, bad))

    def test_negative_indices_raise_value_error(self) -> None:
        with self.assertRaises(ValueError):
            self.call(indices=(-1,))
        with self.assertRaises(ValueError):
            self.call(indices=(0, -1))

    def test_out_of_range_indices_raise_value_error(self) -> None:
        with self.assertRaises(ValueError):
            self.call(indices=(3,))
        with self.assertRaises(ValueError):
            self.call(indices=(0, 3))

    def test_duplicate_indices_raise_value_error(self) -> None:
        with self.assertRaises(ValueError):
            self.call(indices=(1, 1))

    def test_not_strictly_increasing_indices_raise_value_error(self) -> None:
        with self.assertRaises(ValueError):
            self.call(indices=(1, 0))
        with self.assertRaises(ValueError):
            self.call(indices=(0, 2, 1))

    def test_empty_plan_has_no_valid_index(self) -> None:
        with self.assertRaises(ValueError):
            timeline_call(
                self.store, self.args, series={}, max_size=0,
                indices=(0,), causes=(), depth=0, node_limit=1,
            )


class CascadeSliceTimelineChangeLimitValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        overrides.setdefault("causes", ("a0",))
        return timeline_call(self.store, self.args, **overrides)

    def test_change_limit_must_be_a_non_bool_int(self) -> None:
        for bad in (True, False, 1.0, "1", None):
            with self.subTest(change_limit=bad):
                with self.assertRaises(TypeError):
                    self.call(change_limit=bad)

    def test_change_limit_below_one_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            self.call(change_limit=0)
        with self.assertRaises(ValueError):
            self.call(change_limit=-2)

    def test_total_changes_beyond_change_limit_raise_value_error(self) -> None:
        # The (0, 1) segment carries two change records.
        with self.assertRaises(ValueError):
            self.call(indices=(0, 1), change_limit=1)
        with self.assertRaises(ValueError):
            self.call(indices=(0, 1, 2), change_limit=1)

    def test_change_limit_covering_the_total_succeeds(self) -> None:
        result = self.call(indices=(0, 1, 2), change_limit=2)
        self.assertEqual(len(result["segments"]), 2)

    def test_change_limit_validated_after_node_limit(self) -> None:
        # node_limit is checked first, so its error wins over a bad
        # change_limit; a valid node_limit then surfaces change_limit's.
        with self.assertRaises(ValueError):
            self.call(node_limit=0, change_limit=0)
        with self.assertRaises(ValueError):
            self.call(node_limit=0, change_limit="x")
        with self.assertRaises(TypeError):
            self.call(node_limit=1, change_limit="x")


class CascadeSliceTimelineValidationOrderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        overrides.setdefault("causes", ("a0",))
        return timeline_call(self.store, self.args, **overrides)

    def test_ordinary_inputs_validated_first(self) -> None:
        with self.assertRaises(TypeError):
            self.call(reference=1, indices="x", causes=[1])
        with self.assertRaises(ValueError):
            self.call(axis="nope", indices=(-1,))
        with self.assertRaises(TypeError):
            self.call(values=[0, 1], indices=(0,))

    def test_indices_validated_before_existing_slice_inputs(self) -> None:
        with self.assertRaises(TypeError):
            self.call(indices="x", causes=[1])
        with self.assertRaises(ValueError):
            self.call(indices=(-1,), causes=("a0", "a0"))
        with self.assertRaises(ValueError):
            self.call(indices=(9,), direction="sideways")
        with self.assertRaises(TypeError):
            self.call(indices=(0, True), depth=True)
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

    def test_candidate_cap_still_breaches_with_no_partial_result(self) -> None:
        with self.assertRaises(ValueError):
            self.call(limit=1)


class CascadeSliceTimelineCauseTests(unittest.TestCase):
    def test_cause_absent_from_every_slice_raises(self) -> None:
        store, args = diamond_store()
        with self.assertRaises(KeyError) as caught:
            timeline_call(store, args, indices=(0, 1), causes=("zzz",))
        self.assertEqual(caught.exception.args[0], "zzz")

    def test_first_absent_cause_in_input_order_wins(self) -> None:
        store, args = diamond_store()
        with self.assertRaises(KeyError) as caught:
            timeline_call(store, args, indices=(0, 1), causes=("zzz", "a0"))
        self.assertEqual(caught.exception.args[0], "zzz")

    def test_cause_present_in_any_one_slice_is_accepted(self) -> None:
        store, args = diamond_store()
        # W is a node only from the second checkpoint on, so index 0 has an
        # empty start set for it, but the batch must not raise.
        result = timeline_call(store, args, indices=(0, 1), causes=("a0", "W"))
        self.assertEqual(len(result["snapshots"]), 2)

    def test_graph_event_that_is_never_a_slice_node_raises(self) -> None:
        store, args = diamond_store()
        with self.assertRaises(KeyError):
            timeline_call(store, args, indices=(0, 1), causes=("r0",))


class CascadeSliceTimelineEmptyInputTests(unittest.TestCase):
    def test_empty_indices_return_two_empty_tuples(self) -> None:
        store, args = diamond_store()
        result = timeline_call(store, args, indices=(), causes=("a0",))
        self.assertEqual(result, {"snapshots": (), "segments": ()})

    def test_empty_indices_still_run_input_and_state_checks(self) -> None:
        store, args = diamond_store()
        with self.assertRaises(KeyError):
            timeline_call(store, args, indices=(), reference="ghost", causes=())
        with self.assertRaises(ValueError):
            timeline_call(store, args, indices=(), limit=1, causes=())
        with self.assertRaises(ValueError):
            timeline_call(store, args, indices=(), node_limit=0, causes=())
        with self.assertRaises(ValueError):
            timeline_call(store, args, indices=(), change_limit=0, causes=())
        with self.assertRaises(TypeError):
            timeline_call(store, args, indices=(), causes=[1])

    def test_empty_causes_yield_empty_slices_and_empty_segments(self) -> None:
        store, args = diamond_store()
        result = timeline_call(
            store, args, indices=(0, 1, 2), causes=(),
            direction="forward", depth=5, node_limit=1,
        )
        empty = {"nodes": (), "edges": (), "gaps": ()}
        self.assertEqual(
            result,
            {
                "snapshots": (
                    {"index": 0, "slice": empty},
                    {"index": 1, "slice": empty},
                    {"index": 2, "slice": empty},
                ),
                "segments": (
                    {"before": 0, "after": 1, "changes": ()},
                    {"before": 1, "after": 2, "changes": ()},
                ),
            },
        )

    def test_empty_causes_still_run_state_and_range_checks(self) -> None:
        store, args = diamond_store()
        with self.assertRaises(KeyError):
            timeline_call(store, args, reference="ghost", causes=())
        with self.assertRaises(ValueError):
            timeline_call(store, args, limit=1, causes=())
        with self.assertRaises(ValueError):
            timeline_call(store, args, indices=(5, 6), causes=())
        with self.assertRaises(ValueError):
            timeline_call(store, args, node_limit=0, causes=())


class CascadeSliceTimelineNodeLimitTests(unittest.TestCase):
    def test_any_position_over_cap_raises_value_error(self) -> None:
        store, args = diamond_store()
        # The prefix-1 slice selects one node, later prefixes select two.
        with self.assertRaises(ValueError):
            timeline_call(
                store, args, indices=(0, 1), causes=("a0",),
                direction="forward", depth=99, node_limit=1,
            )

    def test_limit_covering_the_largest_position_succeeds(self) -> None:
        store, args = diamond_store()
        result = timeline_call(
            store, args, indices=(0, 1, 2), causes=("a0",),
            direction="forward", depth=99, node_limit=2,
        )
        self.assertEqual(len(result["snapshots"][2]["slice"]["nodes"]), 2)

    def test_over_cap_changes_nothing(self) -> None:
        store, args = diamond_store()
        heads = dict(store._heads)
        with self.assertRaises(ValueError):
            timeline_call(store, args, indices=(0, 1), node_limit=1)
        self.assertEqual(dict(store._heads), heads)


class CascadeSliceTimelineIsolationTests(unittest.TestCase):
    def test_result_levels_are_mutually_unshared(self) -> None:
        store, args = diamond_store()
        result = timeline_call(store, args, indices=(0, 1, 2), causes=("a0",))
        record_ids = []
        for snapshot in result["snapshots"]:
            record_ids.append(id(snapshot))
            record_ids.append(id(snapshot["slice"]))
            for group in ("nodes", "edges", "gaps"):
                record_ids.extend(id(r) for r in snapshot["slice"][group])
        for segment in result["segments"]:
            record_ids.append(id(segment))
            record_ids.extend(id(c) for c in segment["changes"])
            record_ids.extend(
                id(c[side])
                for c in segment["changes"]
                for side in ("before", "after")
                if c[side] is not None
            )
        self.assertEqual(len(record_ids), len(set(record_ids)))

    def test_mutating_result_never_affects_later_calls(self) -> None:
        store, args = diamond_store()
        first = timeline_call(store, args, indices=(0, 1, 2), causes=("a0",))
        pristine = copy.deepcopy(first)
        for snapshot in first["snapshots"]:
            for node in snapshot["slice"]["nodes"]:
                node["branches"] += ("evil",)
                node["affected"] += ("evil",)
            for edge in snapshot["slice"]["edges"]:
                edge["path"] += ("evil",)
        for segment in first["segments"]:
            for change in segment["changes"]:
                for side in ("before", "after"):
                    if change[side] is not None and "path" in change[side]:
                        change[side]["path"] += ("evil",)
        second = timeline_call(store, args, indices=(0, 1, 2), causes=("a0",))
        self.assertEqual(second, pristine)


class CascadeSliceTimelineReadOnlyTests(unittest.TestCase):
    def test_success_and_failure_change_nothing(self) -> None:
        store, args = diamond_store()
        heads = dict(store._heads)
        appends = copy.deepcopy(store._appends)
        merges = copy.deepcopy(store._merges)
        audit = store.audit_log()

        timeline_call(store, args, indices=(0, 1, 2), causes=("a0",))
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
            dict(causes=("zzz",)),
            dict(limit=1),
            dict(reference="ghost"),
        ]
        for overrides in failures:
            with self.subTest(overrides=overrides):
                with self.assertRaises(Exception):
                    timeline_call(store, args, causes=("a0",), **overrides)

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
        diff_kwargs = timeline_kwargs(args)
        diff_kwargs.pop("indices")
        diff_kwargs.pop("change_limit")
        diff = store.cascade_slice_diff(before=0, after=1, **diff_kwargs)
        timeline_call(store, args, indices=(0, 1, 2), causes=("a0",))
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

    def test_repeated_calls_are_equal(self) -> None:
        store, args = diamond_store()
        first = timeline_call(store, args, indices=(0, 1, 2), causes=("a0",))
        second = timeline_call(store, args, indices=(0, 1, 2), causes=("a0",))
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
