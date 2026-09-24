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
        right_indices=(1, 2),
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


def lifetime_record(
    type_name,
    identity,
    first_seen=0,
    last_seen=0,
    intervals=((0, 0),),
    transitions=(),
):
    return {
        "type": type_name,
        "identity": identity,
        "first_seen": first_seen,
        "last_seen": last_seen,
        "intervals": intervals,
        "transitions": transitions,
    }


class CompareLifetimesSignatureTests(unittest.TestCase):
    def test_public_signature_replaces_indices_and_appends_diff_limit(self):
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

    def test_result_is_a_dict_with_left_right_changes(self):
        store, args = diamond_store()
        result = compare_call(store, args)
        self.assertIsInstance(result, dict)
        self.assertEqual(list(result), ["left", "right", "changes"])
        self.assertIsInstance(result["left"], tuple)
        self.assertIsInstance(result["right"], tuple)
        self.assertIsInstance(result["changes"], tuple)
        for record in result["left"] + result["right"]:
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
        for change in result["changes"]:
            self.assertEqual(list(change), ["kind", "identity", "before", "after"])


class CompareLifetimesDiffTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def test_shifted_windows_mark_shared_identities_changed(self):
        result = compare_call(self.store, self.args)
        self.assertEqual(
            [(c["kind"], c["identity"]) for c in result["changes"]],
            [
                ("node_changed", ("a0", "x")),
                ("node_changed", ("W", "y")),
                ("edge_changed", (("a0", "x"), ("W", "y"))),
            ],
        )
        a0 = result["changes"][0]
        self.assertEqual(a0["before"]["first_seen"], 0)
        self.assertEqual(a0["before"]["last_seen"], 1)
        self.assertEqual(a0["before"]["intervals"], ((0, 1),))
        self.assertEqual(a0["after"]["first_seen"], 1)
        self.assertEqual(a0["after"]["last_seen"], 2)
        self.assertEqual(a0["after"]["intervals"], ((1, 2),))

    def test_added_identities_carry_none_before(self):
        result = compare_call(
            self.store, self.args, left_indices=(0,), right_indices=(1,)
        )
        self.assertEqual(
            [(c["kind"], c["identity"]) for c in result["changes"]],
            [
                ("node_added", ("W", "y")),
                ("node_changed", ("a0", "x")),
                ("edge_added", (("a0", "x"), ("W", "y"))),
            ],
        )
        added_w = result["changes"][0]
        self.assertIsNone(added_w["before"])
        self.assertEqual(added_w["after"]["first_seen"], 1)
        self.assertEqual(added_w["after"]["intervals"], ((1, 1),))

    def test_removed_identities_carry_none_after(self):
        result = compare_call(
            self.store, self.args, left_indices=(1,), right_indices=(0,)
        )
        self.assertEqual(
            [(c["kind"], c["identity"]) for c in result["changes"]],
            [
                ("node_removed", ("W", "y")),
                ("node_changed", ("a0", "x")),
                ("edge_removed", (("a0", "x"), ("W", "y"))),
            ],
        )
        removed_w = result["changes"][0]
        self.assertIsNone(removed_w["after"])
        self.assertEqual(removed_w["before"]["first_seen"], 1)

    def test_identical_windows_yield_no_changes(self):
        result = compare_call(
            self.store, self.args, left_indices=(0, 1), right_indices=(0, 1)
        )
        self.assertEqual(result["changes"], ())
        self.assertEqual(len(result["left"]), 3)
        self.assertEqual(result["left"], result["right"])

    def test_left_and_right_match_the_lifetime_query(self):
        store, args = self.store, self.args
        result = compare_call(store, args)
        lifetimes_kwargs = compare_kwargs(args)
        lifetimes_kwargs.pop("diff_limit")
        left_window = lifetimes_kwargs.pop("left_indices")
        right_window = lifetimes_kwargs.pop("right_indices")
        for side, window in (("left", left_window), ("right", right_window)):
            window_kwargs = dict(lifetimes_kwargs, indices=window)
            expected = store.cascade_slice_lifetimes(**window_kwargs)
            self.assertEqual(result[side], expected["lifetimes"])

    def test_transition_difference_marks_changed(self):
        # left (0, 1) sees W's addition; right (1,) starts with W present.
        result = compare_call(
            self.store, self.args, left_indices=(0, 1), right_indices=(1,)
        )
        kinds = {c["identity"]: c["kind"] for c in result["changes"]}
        self.assertEqual(kinds[("W", "y")], "node_changed")
        w_change = next(
            c for c in result["changes"] if c["identity"] == ("W", "y")
        )
        self.assertEqual(len(w_change["before"]["transitions"]), 1)
        self.assertEqual(w_change["after"]["transitions"], ())


class CompareLifetimesOrderingTests(unittest.TestCase):
    """Direct checks of the window diff over hand-built lifetime records."""

    def test_type_groups_and_removed_added_changed_order(self):
        left = (
            lifetime_record("node", ("a", "x")),
            lifetime_record("node", ("b", "y")),
            lifetime_record("node", ("c", "z")),
            lifetime_record("edge", (("a", "x"), ("b", "y"))),
            lifetime_record("gap", (0, ("m",))),
        )
        right = (
            lifetime_record("node", ("d", "w")),
            lifetime_record("node", ("b", "y"), last_seen=1, intervals=((0, 1),)),
            lifetime_record("node", ("e", "v")),
            lifetime_record("edge", (("b", "y"), ("c", "z"))),
            lifetime_record("gap", (0, ("m",))),
        )
        changes = BranchStore._build_lifetime_changes(left, right)
        self.assertEqual(
            [(c["kind"], c["identity"]) for c in changes],
            [
                ("node_removed", ("a", "x")),
                ("node_removed", ("c", "z")),
                ("node_added", ("d", "w")),
                ("node_added", ("e", "v")),
                ("node_changed", ("b", "y")),
                ("edge_removed", (("a", "x"), ("b", "y"))),
                ("edge_added", (("b", "y"), ("c", "z"))),
            ],
        )

    def test_changed_detection_covers_each_summary_field(self):
        base = lifetime_record("node", ("a", "x"))
        variants = [
            lifetime_record("node", ("a", "x"), first_seen=1),
            lifetime_record("node", ("a", "x"), last_seen=2),
            lifetime_record("node", ("a", "x"), intervals=((0, 0), (2, 2))),
            lifetime_record(
                "node",
                ("a", "x"),
                transitions=(
                    {"kind": "node_added", "before": None, "after": None},
                ),
            ),
        ]
        for variant in variants:
            with self.subTest(variant=variant):
                (change,) = BranchStore._build_lifetime_changes((base,), (variant,))
                self.assertEqual(change["kind"], "node_changed")
        self.assertEqual(
            BranchStore._build_lifetime_changes((base,), (lifetime_record("node", ("a", "x")),)),
            (),
        )

    def test_empty_inputs_yield_no_changes(self):
        self.assertEqual(BranchStore._build_lifetime_changes((), ()), ())


class CompareLifetimesWindowValidationTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        overrides.setdefault("causes", ("a0",))
        return compare_call(self.store, self.args, **overrides)

    def test_window_must_be_a_tuple(self):
        for side in ("left_indices", "right_indices"):
            for bad in ("x", [0, 1], 1, None):
                with self.subTest(side=side, value=bad):
                    with self.assertRaises(TypeError):
                        self.call(**{side: bad})

    def test_window_elements_must_be_non_bool_ints(self):
        for side in ("left_indices", "right_indices"):
            for bad in (True, False, 1.0, "1", None):
                with self.subTest(side=side, value=bad):
                    with self.assertRaises(TypeError):
                        self.call(**{side: (0, bad)})

    def test_window_index_value_errors(self):
        for side in ("left_indices", "right_indices"):
            with self.subTest(side=side, case="negative"):
                with self.assertRaises(ValueError):
                    self.call(**{side: (-1,)})
            with self.subTest(side=side, case="out-of-range"):
                with self.assertRaises(ValueError):
                    self.call(**{side: (3,)})
            with self.subTest(side=side, case="duplicate"):
                with self.assertRaises(ValueError):
                    self.call(**{side: (1, 1)})
            with self.subTest(side=side, case="not-increasing"):
                with self.assertRaises(ValueError):
                    self.call(**{side: (2, 1)})

    def test_left_window_validated_before_right_window(self):
        with self.assertRaises(TypeError):
            self.call(left_indices="x", right_indices=(-1,))
        with self.assertRaises(ValueError):
            self.call(left_indices=(-1,), right_indices="x")
        with self.assertRaises(ValueError):
            self.call(left_indices=(0, 0), right_indices=(-1,))

    def test_windows_validated_before_slice_inputs_and_limits(self):
        with self.assertRaises(TypeError):
            self.call(left_indices="x", causes=[1], diff_limit=0)
        with self.assertRaises(ValueError):
            self.call(right_indices=(9,), direction="sideways", diff_limit=0)
        with self.assertRaises(TypeError):
            self.call(left_indices=(0, True), depth=True, diff_limit=0)

    def test_ordinary_inputs_validated_first(self):
        with self.assertRaises(TypeError):
            self.call(reference=1, left_indices="x", causes=[1], diff_limit=0)
        with self.assertRaises(ValueError):
            self.call(axis="nope", left_indices=(-1,), diff_limit=0)

    def test_all_input_validation_precedes_state_lookups(self):
        with self.assertRaises(TypeError):
            self.call(reference="ghost", left_indices="x")
        with self.assertRaises(ValueError):
            self.call(reference="ghost", diff_limit=0)
        with self.assertRaises(KeyError):
            self.call(reference="ghost")


class CompareLifetimesLimitValidationTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        overrides.setdefault("causes", ("a0",))
        return compare_call(self.store, self.args, **overrides)

    def test_diff_limit_must_be_a_non_bool_int(self):
        for bad in (True, False, 1.0, "1", None):
            with self.subTest(diff_limit=bad):
                with self.assertRaises(TypeError):
                    self.call(diff_limit=bad)

    def test_diff_limit_below_one_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.call(diff_limit=0)
        with self.assertRaises(ValueError):
            self.call(diff_limit=-2)

    def test_diff_limit_validated_after_lifetime_limit(self):
        with self.assertRaises(ValueError):
            self.call(change_limit=0, diff_limit=0)
        with self.assertRaises(ValueError):
            self.call(lifetime_limit=0, diff_limit="x")
        with self.assertRaises(TypeError):
            self.call(lifetime_limit=1, diff_limit="x")

    def test_changes_beyond_diff_limit_raise_value_error(self):
        # left (0,) vs right (1,) yields three change records.
        with self.assertRaises(ValueError):
            self.call(left_indices=(0,), right_indices=(1,), diff_limit=2)
        with self.assertRaises(ValueError):
            self.call(left_indices=(0,), right_indices=(1,), diff_limit=1)
        result = self.call(left_indices=(0,), right_indices=(1,), diff_limit=3)
        self.assertEqual(len(result["changes"]), 3)

    def test_lifetime_limit_caps_each_window(self):
        # The (0, 1, 2) window carries three lifetime records.
        with self.assertRaises(ValueError):
            self.call(left_indices=(0, 1, 2), right_indices=(0,), lifetime_limit=2)
        with self.assertRaises(ValueError):
            self.call(left_indices=(0,), right_indices=(0, 1, 2), lifetime_limit=2)
        result = self.call(
            left_indices=(0, 1, 2), right_indices=(0, 1, 2), lifetime_limit=3
        )
        self.assertEqual(len(result["left"]), 3)
        self.assertEqual(len(result["right"]), 3)


class CompareLifetimesStateCheckTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        overrides.setdefault("causes", ("a0",))
        return compare_call(self.store, self.args, **overrides)

    def test_unknown_branch_and_history_node_raise_key_error(self):
        with self.assertRaises(KeyError):
            self.call(reference="ghost")
        with self.assertRaises(KeyError):
            self.call(
                series={
                    "a": (("r0", "a0"), ("r1", "a0"), ("r2", "a0")),
                    "b": (("r0", "a0"), ("r1", "W"), ("r2", "nope2")),
                },
            )

    def test_absent_cause_raises_key_error(self):
        with self.assertRaises(KeyError) as caught:
            self.call(causes=("zzz",))
        self.assertEqual(caught.exception.args[0], "zzz")

    def test_candidate_node_and_change_caps_still_breach(self):
        with self.assertRaises(ValueError):
            self.call(limit=1)
        with self.assertRaises(ValueError):
            self.call(direction="forward", node_limit=1)
        with self.assertRaises(ValueError):
            self.call(change_limit=1)


class CompareLifetimesEmptyWindowTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        overrides.setdefault("causes", ("a0",))
        return compare_call(self.store, self.args, **overrides)

    def test_both_windows_empty_yield_three_empty_tuples(self):
        result = self.call(left_indices=(), right_indices=())
        self.assertEqual(result, {"left": (), "right": (), "changes": ()})

    def test_empty_left_window_adds_every_right_identity(self):
        result = self.call(left_indices=(), right_indices=(0,))
        self.assertEqual(result["left"], ())
        self.assertEqual(len(result["right"]), 1)
        self.assertEqual(
            [(c["kind"], c["identity"]) for c in result["changes"]],
            [("node_added", ("a0", "x"))],
        )

    def test_empty_right_window_removes_every_left_identity(self):
        result = self.call(left_indices=(0, 1), right_indices=())
        self.assertEqual(result["right"], ())
        self.assertEqual(
            [c["kind"] for c in result["changes"]],
            ["node_removed", "node_removed", "edge_removed"],
        )

    def test_empty_windows_still_run_input_and_state_checks(self):
        with self.assertRaises(KeyError):
            self.call(left_indices=(), right_indices=(), reference="ghost")
        with self.assertRaises(ValueError):
            self.call(left_indices=(), right_indices=(), limit=1)
        with self.assertRaises(ValueError):
            self.call(left_indices=(), right_indices=(), node_limit=0)
        with self.assertRaises(ValueError):
            self.call(left_indices=(), right_indices=(), change_limit=0)
        with self.assertRaises(ValueError):
            self.call(left_indices=(), right_indices=(), lifetime_limit=0)
        with self.assertRaises(ValueError):
            self.call(left_indices=(), right_indices=(), diff_limit=0)
        with self.assertRaises(TypeError):
            self.call(left_indices=(), right_indices=(), causes=[1])

    def test_empty_causes_yield_empty_lifetimes_on_both_sides(self):
        result = self.call(causes=())
        self.assertEqual(result, {"left": (), "right": (), "changes": ()})


class CompareLifetimesIsolationTests(unittest.TestCase):
    def test_result_levels_are_mutually_unshared(self):
        store, args = diamond_store()
        result = compare_call(store, args)
        seen_ids = []

        def collect(value):
            if isinstance(value, dict):
                seen_ids.append(id(value))
                for item in value.values():
                    collect(item)
            elif isinstance(value, tuple):
                for item in value:
                    collect(item)

        collect(result)
        self.assertEqual(len(seen_ids), len(set(seen_ids)))

    def test_mutating_result_never_affects_later_calls(self):
        store, args = diamond_store()
        first = compare_call(store, args)
        pristine = copy.deepcopy(first)
        for record in first["left"] + first["right"]:
            record["intervals"] += ((99, 99),)
            for transition in record["transitions"]:
                for side in ("before", "after"):
                    evidence = transition[side]
                    if evidence is not None and "branches" in evidence:
                        evidence["branches"] += ("evil",)
        for change in first["changes"]:
            for side in ("before", "after"):
                if change[side] is not None:
                    change[side]["first_seen"] = 99
        second = compare_call(store, args)
        self.assertEqual(second, pristine)


class CompareLifetimesReadOnlyTests(unittest.TestCase):
    def test_success_and_failure_change_nothing(self):
        store, args = diamond_store()
        heads = dict(store._heads)
        appends = copy.deepcopy(store._appends)
        merges = copy.deepcopy(store._merges)
        audit = store.audit_log()

        compare_call(store, args)
        failures = [
            dict(left_indices="x"),
            dict(right_indices="x"),
            dict(left_indices=(-1,)),
            dict(right_indices=(9,)),
            dict(left_indices=(1, 0)),
            dict(right_indices=(1, 1)),
            dict(causes=[1]),
            dict(direction="up"),
            dict(depth=-1),
            dict(node_limit=0),
            dict(change_limit=0),
            dict(change_limit=1),
            dict(lifetime_limit=0),
            dict(lifetime_limit=1),
            dict(diff_limit=0),
            dict(diff_limit=1),
            dict(causes=("zzz",)),
            dict(limit=1),
            dict(reference="ghost"),
        ]
        for overrides in failures:
            with self.subTest(overrides=overrides):
                with self.assertRaises(Exception):
                    compare_call(store, args, causes=("a0",), **overrides)

        self.assertEqual(dict(store._heads), heads)
        self.assertEqual(copy.deepcopy(store._appends), appends)
        self.assertEqual(copy.deepcopy(store._merges), merges)
        self.assertEqual(store.audit_log(), audit)

    def test_existing_queries_are_unaffected(self):
        store, args = diamond_store()
        lifetimes_kwargs = compare_kwargs(args, indices=(0, 1, 2))
        lifetimes_kwargs.pop("left_indices")
        lifetimes_kwargs.pop("right_indices")
        lifetimes_kwargs.pop("diff_limit")
        lifetimes = store.cascade_slice_lifetimes(**lifetimes_kwargs)
        compare_call(store, args)
        self.assertEqual(store.cascade_slice_lifetimes(**lifetimes_kwargs), lifetimes)

    def test_repeated_calls_are_equal(self):
        store, args = diamond_store()
        first = compare_call(store, args)
        second = compare_call(store, args)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
