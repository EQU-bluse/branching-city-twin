import copy
import inspect
import unittest

from city_twin.branches import BranchStore
from city_twin.event_graph import EventGraph


def diamond_store() -> tuple[BranchStore, tuple]:
    """Three aligned checkpoints; prefix 0 is a0 only, later prefixes add W.

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


def compare_kwargs(store_args, **overrides) -> dict:
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
        left_indices=(0, 1),
        right_indices=(0, 1),
        causes=("a0",),
        direction="both",
        depth=99,
        node_limit=50,
        change_limit=50,
        lifetime_limit=50,
        diff_limit=50,
    )
    kwargs.update(overrides)
    return kwargs


def compare_call(store, store_args, **overrides):
    return store.compare_lifetimes(**compare_kwargs(store_args, **overrides))


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


def node_lifetime(cause, key, first, last, intervals, transitions=()):
    return {
        "type": "node",
        "identity": (cause, key),
        "first_seen": first,
        "last_seen": last,
        "intervals": tuple(intervals),
        "transitions": tuple(transitions),
    }


def edge_lifetime(source, target, first, last, intervals, transitions=()):
    return {
        "type": "edge",
        "identity": (
            (source[0], source[1]),
            (target[0], target[1]),
        ),
        "first_seen": first,
        "last_seen": last,
        "intervals": tuple(intervals),
        "transitions": tuple(transitions),
    }


def gap_lifetime(interval, members, first, last, intervals, transitions=()):
    return {
        "type": "gap",
        "identity": (interval, tuple(members)),
        "first_seen": first,
        "last_seen": last,
        "intervals": tuple(intervals),
        "transitions": tuple(transitions),
    }


class CompareLifetimesSignatureTests(unittest.TestCase):
    def test_public_signature_replaces_indices_and_appends_diff_limit(self) -> None:
        parameters = list(
            inspect.signature(BranchStore.compare_lifetimes).parameters.values()
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
                "left_indices",
                "right_indices",
                "causes",
                "direction",
                "depth",
                "node_limit",
                "change_limit",
                "lifetime_limit",
                "diff_limit",
            ],
        )
        self.assertTrue(
            all(p.default is inspect.Parameter.empty for p in parameters)
        )

    def test_result_is_a_dict_with_left_right_changes(self) -> None:
        store, args = diamond_store()
        result = compare_call(store, args, left_indices=(0, 1, 2), right_indices=(1, 2))
        self.assertIsInstance(result, dict)
        self.assertEqual(list(result), ["left", "right", "changes"])
        self.assertIsInstance(result["left"], tuple)
        self.assertIsInstance(result["right"], tuple)
        self.assertIsInstance(result["changes"], tuple)
        for record in result["changes"]:
            self.assertEqual(list(record), ["kind", "identity", "before", "after"])
            self.assertIn(
                record["kind"].split("_", 1)[0], ("node", "edge", "gap")
            )
            self.assertIn(
                record["kind"].split("_", 1)[1],
                ("removed", "added", "changed"),
            )

    def test_sides_match_the_standalone_lifetime_query(self) -> None:
        store, args = diamond_store()
        result = compare_call(
            store, args, left_indices=(0, 1, 2), right_indices=(0,)
        )
        left = store.cascade_slice_lifetimes(
            **lifetimes_kwargs(args, indices=(0, 1, 2))
        )["lifetimes"]
        right = store.cascade_slice_lifetimes(
            **lifetimes_kwargs(args, indices=(0,))
        )["lifetimes"]
        self.assertEqual(result["left"], left)
        self.assertEqual(result["right"], right)


class CompareLifetimesRecordTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        return compare_call(self.store, self.args, **overrides)

    def test_identical_windows_have_no_changes(self) -> None:
        for window in ((0,), (1,), (0, 1), (0, 1, 2)):
            with self.subTest(window=window):
                result = self.call(left_indices=window, right_indices=window)
                self.assertEqual(result["changes"], ())

    def test_right_only_identities_are_added(self) -> None:
        # The empty left window makes every right identity an addition.
        result = self.call(left_indices=(), right_indices=(0, 1, 2))
        self.assertEqual(result["left"], ())
        self.assertEqual(
            [(c["kind"], c["identity"]) for c in result["changes"]],
            [
                ("node_added", ("a0", "x")),
                ("node_added", ("W", "y")),
                ("edge_added", (("a0", "x"), ("W", "y"))),
            ],
        )
        for change in result["changes"]:
            self.assertIsNone(change["before"])
            self.assertIsNotNone(change["after"])

    def test_left_only_identities_are_removed(self) -> None:
        result = self.call(left_indices=(0, 1, 2), right_indices=())
        self.assertEqual(result["right"], ())
        self.assertEqual(
            [(c["kind"], c["identity"]) for c in result["changes"]],
            [
                ("node_removed", ("a0", "x")),
                ("node_removed", ("W", "y")),
                ("edge_removed", (("a0", "x"), ("W", "y"))),
            ],
        )
        for change in result["changes"]:
            self.assertIsNotNone(change["before"])
            self.assertIsNone(change["after"])

    def test_changed_intervals_and_first_last_positions(self) -> None:
        # Left sees a0 over checkpoints 0..2; the right window only at 0.
        result = self.call(left_indices=(0, 1, 2), right_indices=(0,))
        by_identity = {c["identity"]: c for c in result["changes"]}
        a0 = by_identity[("a0", "x")]
        self.assertEqual(a0["kind"], "node_changed")
        self.assertEqual(
            (a0["before"]["first_seen"], a0["before"]["last_seen"]), (0, 2)
        )
        self.assertEqual(a0["before"]["intervals"], ((0, 2),))
        self.assertEqual(
            (a0["after"]["first_seen"], a0["after"]["last_seen"]), (0, 0)
        )
        self.assertEqual(a0["after"]["intervals"], ((0, 0),))

    def test_classification_difference_counts_as_changed(self) -> None:
        # Window (0, 1) sees the W/edge additions as transitions inside
        # the window; window (1,) has the same presence interval but no
        # transition, since the evidence predates its first snapshot.
        result = self.call(left_indices=(0, 1), right_indices=(1,))
        kinds = {(c["kind"], c["identity"]) for c in result["changes"]}
        self.assertIn(("edge_changed", (("a0", "x"), ("W", "y"))), kinds)
        edge = next(
            c
            for c in result["changes"]
            if c["identity"] == (("a0", "x"), ("W", "y"))
        )
        self.assertEqual(edge["before"]["intervals"], ((1, 1),))
        self.assertEqual(edge["after"]["intervals"], ((1, 1),))
        self.assertEqual(
            [t["kind"] for t in edge["before"]["transitions"]], ["edge_added"]
        )
        self.assertEqual(edge["after"]["transitions"], ())

    def test_non_adjacent_window_positions_stay_one_interval(self) -> None:
        # Window (0, 2) has two adjacent selected snapshots that both
        # carry a0, so its lifetime is one interval spanning the
        # unselected checkpoint; versus the empty right window that
        # record is the removed one.
        result = self.call(left_indices=(0, 2), right_indices=())
        a0 = next(
            c for c in result["changes"] if c["identity"] == ("a0", "x")
        )
        self.assertEqual(a0["kind"], "node_removed")
        self.assertEqual(
            (a0["before"]["first_seen"], a0["before"]["last_seen"]), (0, 2)
        )
        self.assertEqual(a0["before"]["intervals"], ((0, 2),))
        self.assertEqual(a0["before"]["transitions"], ())

    def test_change_records_order_type_then_kind(self) -> None:
        # node: W removed (left order), a0 changed; edge: edge removed.
        result = self.call(left_indices=(0, 1, 2), right_indices=(0,))
        self.assertEqual(
            [(c["kind"], c["identity"]) for c in result["changes"]],
            [
                ("node_removed", ("W", "y")),
                ("node_changed", ("a0", "x")),
                ("edge_removed", (("a0", "x"), ("W", "y"))),
            ],
        )

    def test_added_follow_right_order_changed_follow_left(self) -> None:
        # Left window (0,) carries only a0; the right window (1,) also
        # carries W and the edge. W is added (right order); a0 merely
        # moves from (0, 0) to (1, 1), so it is changed (left order);
        # additions still come before changes within the node group.
        result = self.call(left_indices=(0,), right_indices=(1,))
        self.assertEqual(
            [(c["kind"], c["identity"]) for c in result["changes"]],
            [
                ("node_added", ("W", "y")),
                ("node_changed", ("a0", "x")),
                ("edge_added", (("a0", "x"), ("W", "y"))),
            ],
        )


class BuildLifetimeChangesDirectTests(unittest.TestCase):
    def test_grouping_and_ordering_on_hand_built_tuples(self) -> None:
        removed_node = node_lifetime("a", "x", 0, 0, ((0, 0),))
        changed_left = node_lifetime("b", "y", 0, 1, ((0, 1),))
        changed_right = node_lifetime("b", "y", 1, 1, ((1, 1),))
        identical = node_lifetime("same", "z", 0, 0, ((0, 0),))
        added_node = node_lifetime("c", "q", 2, 2, ((2, 2),))
        removed_edge = edge_lifetime(
            ("a", "x"), ("b", "y"), 0, 1, ((0, 1),)
        )
        added_gap = gap_lifetime(1, ("m",), 2, 2, ((2, 2),))
        changed_gap_left = gap_lifetime(
            0, ("n",), 0, 0, ((0, 0),),
            (
                {
                    "kind": "gap_removed",
                    "before": {"interval": 0, "members": ("n",)},
                    "after": None,
                },
            ),
        )
        changed_gap_right = gap_lifetime(0, ("n",), 0, 2, ((0, 2),))

        left = (
            removed_node,
            changed_left,
            identical,
            removed_edge,
            changed_gap_left,
        )
        right = (
            changed_right,
            identical,
            added_node,
            added_gap,
            changed_gap_right,
        )
        changes = BranchStore._build_lifetime_changes(left, right)
        self.assertEqual(
            [(c["kind"], c["identity"]) for c in changes],
            [
                ("node_removed", ("a", "x")),
                ("node_added", ("c", "q")),
                ("node_changed", ("b", "y")),
                ("edge_removed", (("a", "x"), ("b", "y"))),
                ("gap_added", (1, ("m",))),
                ("gap_changed", (0, ("n",))),
            ],
        )
        node_added = changes[1]
        self.assertIsNone(node_added["before"])
        self.assertEqual(node_added["after"], added_node)
        node_changed = changes[2]
        self.assertEqual(node_changed["before"], changed_left)
        self.assertEqual(node_changed["after"], changed_right)

    def test_identical_records_are_omitted(self) -> None:
        record = node_lifetime("a", "x", 0, 1, ((0, 1),))
        self.assertEqual(
            BranchStore._build_lifetime_changes((record,), (copy.deepcopy(record),)),
            (),
        )

    def test_transition_classification_difference_is_detected(self) -> None:
        before = node_lifetime(
            "a", "x", 0, 1, ((0, 1),),
            ({"kind": "node_removed", "before": None, "after": None},),
        )
        after = node_lifetime("a", "x", 0, 1, ((0, 1),))
        (change,) = BranchStore._build_lifetime_changes((before,), (after,))
        self.assertEqual(change["kind"], "node_changed")

    def test_change_sides_are_detached_copies(self) -> None:
        evidence = {
            "cause": "a",
            "key": "x",
            "intervals": (0,),
            "checkpoints": (0,),
            "branches": ("main",),
            "affected": (),
        }
        left_record = node_lifetime(
            "a", "x", 0, 0, ((0, 0),),
            ({"kind": "node_removed", "before": evidence, "after": None},),
        )
        right_record = node_lifetime("b", "y", 1, 1, ((1, 1),))
        changes = BranchStore._build_lifetime_changes(
            (left_record,), (right_record,)
        )
        ids_ = [id(c["before"]) for c in changes if c["before"] is not None]
        ids_ += [id(c["after"]) for c in changes if c["after"] is not None]
        ids_ += [
            id(transition[side])
            for c in changes
            for side_record in (c["before"], c["after"])
            if isinstance(side_record, dict)
            for transition in side_record["transitions"]
            for side in ("before", "after")
            if transition[side] is not None
        ]
        self.assertEqual(len(ids_), len(set(ids_)))
        self.assertNotIn(id(left_record), ids_)
        self.assertNotIn(id(evidence), ids_)

    def test_empty_sides(self) -> None:
        record = node_lifetime("a", "x", 0, 0, ((0, 0),))
        self.assertEqual(
            [c["kind"] for c in BranchStore._build_lifetime_changes((), (record,))],
            ["node_added"],
        )
        self.assertEqual(
            [c["kind"] for c in BranchStore._build_lifetime_changes((record,), ())],
            ["node_removed"],
        )
        self.assertEqual(BranchStore._build_lifetime_changes((), ()), ())


class CompareLifetimesWindowValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        overrides.setdefault("causes", ("a0",))
        return compare_call(self.store, self.args, **overrides)

    def test_window_must_be_a_tuple(self) -> None:
        for name in ("left_indices", "right_indices"):
            for bad in ("x", [0, 1], {0, 1}, 0):
                with self.subTest(name=name, bad=bad):
                    with self.assertRaises(TypeError):
                        self.call(**{name: bad})

    def test_window_elements_must_be_non_bool_ints(self) -> None:
        with self.assertRaises(TypeError):
            self.call(left_indices=(True,))
        with self.assertRaises(TypeError):
            self.call(right_indices=(False,))
        with self.assertRaises(TypeError):
            self.call(left_indices=(0, 1.0))
        with self.assertRaises(TypeError):
            self.call(right_indices=(0, "1"))

    def test_negative_out_of_range_duplicate_and_unordered_indices(self) -> None:
        with self.assertRaises(ValueError):
            self.call(left_indices=(-1,))
        with self.assertRaises(ValueError):
            self.call(left_indices=(3,))
        with self.assertRaises(ValueError):
            self.call(right_indices=(9,))
        with self.assertRaises(ValueError):
            self.call(left_indices=(1, 1))
        with self.assertRaises(ValueError):
            self.call(right_indices=(1, 0))

    def test_left_window_validated_before_right_window(self) -> None:
        with self.assertRaises(TypeError):
            self.call(left_indices="x", right_indices="y")
        with self.assertRaises(ValueError):
            self.call(left_indices=(-1,), right_indices="x")
        with self.assertRaises(TypeError):
            self.call(left_indices=(0,), right_indices="x")
        with self.assertRaises(ValueError):
            self.call(left_indices=(0,), right_indices=(-1,))

    def test_windows_validated_in_input_order(self) -> None:
        # The bad element is the second one, proving element-wise walking.
        with self.assertRaises(TypeError):
            self.call(left_indices=(0, True))
        with self.assertRaises(ValueError):
            self.call(left_indices=(0, 3))
        with self.assertRaises(TypeError):
            self.call(left_indices=(0,), right_indices=(0, True))

    def test_windows_validated_before_slice_inputs_and_limits(self) -> None:
        with self.assertRaises(TypeError):
            self.call(left_indices="x", causes=[1], diff_limit=0)
        with self.assertRaises(ValueError):
            self.call(left_indices=(-1,), causes=("a0", "a0"))
        with self.assertRaises(ValueError):
            self.call(right_indices=(9,), direction="sideways")


class CompareLifetimesLimitValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        overrides.setdefault("causes", ("a0",))
        return compare_call(self.store, self.args, **overrides)

    def test_diff_limit_must_be_a_non_bool_int(self) -> None:
        for bad in (True, False, 1.0, "1", None):
            with self.subTest(diff_limit=bad):
                with self.assertRaises(TypeError):
                    self.call(diff_limit=bad)

    def test_diff_limit_below_one_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            self.call(diff_limit=0)
        with self.assertRaises(ValueError):
            self.call(diff_limit=-3)

    def test_diff_limit_validated_last_among_limits(self) -> None:
        # change_limit's failure wins over a bad diff_limit; then
        # lifetime_limit's; only a valid earlier limit surfaces diff's.
        with self.assertRaises(ValueError):
            self.call(change_limit=0, lifetime_limit=0, diff_limit=0)
        with self.assertRaises(ValueError):
            self.call(change_limit=1, lifetime_limit=0, diff_limit="x")
        with self.assertRaises(TypeError):
            self.call(change_limit=1, lifetime_limit=1, diff_limit="x")

    def test_changes_beyond_diff_limit_raise_with_no_partial_result(self) -> None:
        # Left (0,1,2) vs right (0,) yields three changes.
        with self.assertRaises(ValueError):
            self.call(
                left_indices=(0, 1, 2), right_indices=(0,), diff_limit=2
            )
        result = self.call(
            left_indices=(0, 1, 2), right_indices=(0,), diff_limit=3
        )
        self.assertEqual(len(result["changes"]), 3)

    def test_lifetime_limit_caps_each_side(self) -> None:
        # An empty left side bypasses the left cap; the right window's
        # three records still breach a two-record cap.
        with self.assertRaises(ValueError):
            self.call(
                left_indices=(), right_indices=(0, 1, 2), lifetime_limit=2
            )
        result = self.call(
            left_indices=(), right_indices=(0, 1, 2), lifetime_limit=3
        )
        self.assertEqual(len(result["right"]), 3)

    def test_left_side_caps_breach_before_the_right_side(self) -> None:
        with self.assertRaises(ValueError):
            self.call(
                left_indices=(0, 1),
                right_indices=(0, 1),
                direction="forward",
                depth=99,
                node_limit=1,
            )
        with self.assertRaises(ValueError):
            self.call(left_indices=(0, 1), change_limit=1)


class CompareLifetimesValidationOrderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        overrides.setdefault("causes", ("a0",))
        return compare_call(self.store, self.args, **overrides)

    def test_ordinary_inputs_validated_first(self) -> None:
        with self.assertRaises(TypeError):
            self.call(reference=1, left_indices="x", causes=[1], diff_limit=0)
        with self.assertRaises(ValueError):
            self.call(axis="nope", left_indices=(-1,), diff_limit=0)
        with self.assertRaises(TypeError):
            self.call(values=[0, 1], left_indices=(0,), diff_limit=0)

    def test_all_input_validation_precedes_state_lookups(self) -> None:
        with self.assertRaises(TypeError):
            self.call(reference="ghost", left_indices="x")
        with self.assertRaises(ValueError):
            self.call(reference="ghost", diff_limit=0)
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

    def test_cause_absent_from_left_raises_even_when_right_has_it(self) -> None:
        # W exists in the right window's slices but not the left's.
        with self.assertRaises(KeyError) as caught:
            self.call(
                left_indices=(0,), right_indices=(1, 2), causes=("W",)
            )
        self.assertEqual(caught.exception.args[0], "W")

    def test_unknown_cause_raises_key_error(self) -> None:
        with self.assertRaises(KeyError) as caught:
            self.call(causes=("zzz",))
        self.assertEqual(caught.exception.args[0], "zzz")


class CompareLifetimesEmptyWindowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        return compare_call(self.store, self.args, **overrides)

    def test_both_windows_empty(self) -> None:
        result = self.call(left_indices=(), right_indices=(), causes=("a0",))
        self.assertEqual(result, {"left": (), "right": (), "changes": ()})

    def test_empty_windows_still_run_input_and_state_checks(self) -> None:
        with self.assertRaises(KeyError):
            self.call(left_indices=(), right_indices=(), reference="ghost",
                      causes=())
        with self.assertRaises(ValueError):
            self.call(left_indices=(), right_indices=(), limit=1, causes=())
        with self.assertRaises(ValueError):
            self.call(left_indices=(), right_indices=(), node_limit=0,
                      causes=())
        with self.assertRaises(ValueError):
            self.call(left_indices=(), right_indices=(), change_limit=0,
                      causes=())
        with self.assertRaises(ValueError):
            self.call(left_indices=(), right_indices=(), lifetime_limit=0,
                      causes=())
        with self.assertRaises(ValueError):
            self.call(left_indices=(), right_indices=(), diff_limit=0,
                      causes=())
        with self.assertRaises(TypeError):
            self.call(left_indices=(), right_indices=(), causes=[1])

    def test_one_empty_window_still_enforces_the_other_caps(self) -> None:
        with self.assertRaises(ValueError):
            self.call(
                left_indices=(0, 1),
                right_indices=(),
                change_limit=1,
            )


class CompareLifetimesIsolationTests(unittest.TestCase):
    def test_result_levels_are_mutually_unshared(self) -> None:
        store, args = diamond_store()
        result = compare_call(
            store, args, left_indices=(0, 1), right_indices=(1, 2)
        )
        ids_ = []
        for side in ("left", "right"):
            for record in result[side]:
                ids_.append(id(record))
        for change in result["changes"]:
            ids_.append(id(change))
            for side in ("before", "after"):
                record = change[side]
                if record is None:
                    continue
                ids_.append(id(record))
                for transition in record["transitions"]:
                    for evidence_side in ("before", "after"):
                        evidence = transition[evidence_side]
                        if evidence is not None:
                            ids_.append(id(evidence))
        self.assertEqual(len(ids_), len(set(ids_)))
        # The change copies are not the same objects as the side tuples'.
        side_ids = {
            id(record)
            for side in ("left", "right")
            for record in result[side]
        }
        self.assertTrue(
            side_ids.isdisjoint(
                id(change[s])
                for change in result["changes"]
                for s in ("before", "after")
                if change[s] is not None
            )
        )

    def test_mutating_result_never_affects_later_calls(self) -> None:
        store, args = diamond_store()
        first = compare_call(
            store, args, left_indices=(0, 1), right_indices=(1, 2)
        )
        pristine = copy.deepcopy(first)

        def corrupt(record) -> None:
            record["intervals"] = ((-9, -9),)
            for transition in record["transitions"]:
                for side in ("before", "after"):
                    evidence = transition[side]
                    if evidence is None:
                        continue
                    if "path" in evidence:
                        evidence["path"] += ("evil",)
                    elif "members" in evidence:
                        evidence["members"] += ("evil",)
                    elif "branches" in evidence:
                        evidence["branches"] += ("evil",)

        for side in ("left", "right"):
            for record in first[side]:
                corrupt(record)
        for change in first["changes"]:
            for side in ("before", "after"):
                if change[side] is not None:
                    corrupt(change[side])

        second = compare_call(
            store, args, left_indices=(0, 1), right_indices=(1, 2)
        )
        self.assertEqual(second, pristine)


class CompareLifetimesReadOnlyTests(unittest.TestCase):
    def test_success_and_failure_change_nothing(self) -> None:
        store, args = diamond_store()
        heads = dict(store._heads)
        appends = copy.deepcopy(store._appends)
        merges = copy.deepcopy(store._merges)
        audit = store.audit_log()

        compare_call(store, args, left_indices=(0, 1), right_indices=(1, 2))
        failures = [
            dict(left_indices="x"),
            dict(right_indices="x"),
            dict(left_indices=(-1,)),
            dict(right_indices=(9,)),
            dict(left_indices=(1, 0)),
            dict(left_indices=(1, 1)),
            dict(left_indices=(True,)),
            dict(causes=[1]),
            dict(direction="up"),
            dict(depth=-1),
            dict(node_limit=0),
            dict(change_limit=0),
            dict(lifetime_limit=0),
            dict(diff_limit=0),
            dict(diff_limit=True),
            dict(diff_limit=1),
            dict(causes=("zzz",)),
            dict(causes=("W",), left_indices=(0,), right_indices=(1, 2)),
            dict(limit=1),
            dict(reference="ghost"),
        ]
        for overrides in failures:
            with self.subTest(overrides=overrides):
                with self.assertRaises(Exception):
                    compare_call(
                        store,
                        args,
                        left_indices=(0, 1),
                        right_indices=(1, 2),
                        causes=("a0",),
                        **overrides,
                    )

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
        lifetimes = store.cascade_slice_lifetimes(
            **lifetimes_kwargs(args, indices=(0, 1, 2))
        )
        compare_call(store, args, left_indices=(0, 1), right_indices=(1, 2))
        self.assertEqual(store.decision_cascade(*args), cascade)
        self.assertEqual(store.cascade_slice_timeline(**timeline_kwargs), timeline)
        self.assertEqual(
            store.cascade_slice_lifetimes(
                **lifetimes_kwargs(args, indices=(0, 1, 2))
            ),
            lifetimes,
        )

    def test_repeated_calls_are_equal(self) -> None:
        store, args = diamond_store()
        first = compare_call(
            store, args, left_indices=(0, 1), right_indices=(1, 2)
        )
        second = compare_call(
            store, args, left_indices=(0, 1), right_indices=(1, 2)
        )
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
