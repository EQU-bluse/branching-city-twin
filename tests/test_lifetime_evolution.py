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


def evolution_kwargs(store_args, **overrides) -> dict:
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
        windows=((0,), (1,), (2,)),
        causes=("a0",),
        direction="both",
        depth=99,
        node_limit=50,
        change_limit=50,
        lifetime_limit=50,
        diff_limit=50,
        window_limit=50,
        total_diff_limit=50,
    )
    kwargs.update(overrides)
    return kwargs


def evolution_call(store, store_args, **overrides):
    return store.lifetime_evolution(**evolution_kwargs(store_args, **overrides))


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


class LifetimeEvolutionSignatureTests(unittest.TestCase):
    def test_public_signature_replaces_pair_and_appends_two_caps(self):
        parameters = list(
            inspect.signature(BranchStore.lifetime_evolution).parameters.values()
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
                "windows",
                "causes",
                "direction",
                "depth",
                "node_limit",
                "change_limit",
                "lifetime_limit",
                "diff_limit",
                "window_limit",
                "total_diff_limit",
                "token",
            ],
        )
        self.assertEqual(
            [p.default for p in parameters[:-1]],
            [inspect.Parameter.empty] * (len(parameters) - 1),
        )
        self.assertIsNone(parameters[-1].default)

    def test_result_is_a_dict_with_windows_and_segments(self):
        store, args = diamond_store()
        result = evolution_call(store, args)
        self.assertIsInstance(result, dict)
        self.assertEqual(list(result), ["windows", "segments"])
        self.assertIsInstance(result["windows"], tuple)
        self.assertIsInstance(result["segments"], tuple)
        for window in result["windows"]:
            self.assertEqual(list(window), ["position", "lifetimes"])
            self.assertIsInstance(window["lifetimes"], tuple)
            for record in window["lifetimes"]:
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
        for segment in result["segments"]:
            self.assertEqual(list(segment), ["left", "right", "changes"])
            self.assertIsInstance(segment["changes"], tuple)
            for change in segment["changes"]:
                self.assertEqual(list(change), ["kind", "identity", "before", "after"])


class LifetimeEvolutionResultTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        return evolution_call(self.store, self.args, **overrides)

    def test_window_records_carry_zero_based_positions(self):
        result = self.call()
        self.assertEqual(
            [window["position"] for window in result["windows"]], [0, 1, 2]
        )
        self.assertEqual(
            [(segment["left"], segment["right"]) for segment in result["segments"]],
            [(0, 1), (1, 2)],
        )

    def test_single_window_has_no_segments(self):
        result = self.call(windows=((0, 1, 2),))
        self.assertEqual(len(result["windows"]), 1)
        self.assertEqual(result["windows"][0]["position"], 0)
        self.assertEqual(len(result["windows"][0]["lifetimes"]), 3)
        self.assertEqual(result["segments"], ())

    def test_window_lifetime_counts_and_segment_changes_follow_the_fixture(self):
        result = self.call()
        self.assertEqual(
            [len(window["lifetimes"]) for window in result["windows"]],
            [1, 3, 3],
        )
        first, second = result["segments"]
        self.assertEqual(
            [(change["kind"], change["identity"]) for change in first["changes"]],
            [
                ("node_added", ("W", "y")),
                ("node_changed", ("a0", "x")),
                ("edge_added", (("a0", "x"), ("W", "y"))),
            ],
        )
        # The third checkpoint changes nothing structural: the second
        # segment only carries the position shifts of present identities.
        self.assertEqual(
            [change["kind"] for change in second["changes"]],
            ["node_changed", "node_changed", "edge_changed"],
        )

    def test_each_window_matches_the_single_window_lifetime_query(self):
        windows = ((0, 2), (1,), (0, 1, 2))
        result = self.call(windows=windows)
        base = evolution_kwargs(self.args, windows=windows)
        base.pop("windows")
        base.pop("diff_limit")
        base.pop("window_limit")
        base.pop("total_diff_limit")
        for window, indices in zip(result["windows"], windows):
            expected = self.store.cascade_slice_lifetimes(
                **dict(base, indices=indices)
            )
            self.assertEqual(window["lifetimes"], expected["lifetimes"])

    def test_each_segment_matches_the_two_window_diff_query(self):
        windows = ((0, 1), (1, 2))
        result = self.call(windows=windows)
        expected = self.store.compare_lifetimes(
            **compare_kwargs(
                self.args,
                left_indices=windows[0],
                right_indices=windows[1],
            )
        )
        self.assertEqual(len(result["segments"]), 1)
        segment = result["segments"][0]
        self.assertEqual((segment["left"], segment["right"]), (0, 1))
        self.assertEqual(segment["changes"], expected["changes"])
        self.assertEqual(result["windows"][0]["lifetimes"], expected["left"])
        self.assertEqual(result["windows"][1]["lifetimes"], expected["right"])

    def test_three_window_batch_matches_pairwise_diffs(self):
        windows = ((0,), (1,), (2,))
        result = self.call(windows=windows)
        for segment in result["segments"]:
            expected = self.store.compare_lifetimes(
                **compare_kwargs(
                    self.args,
                    left_indices=windows[segment["left"]],
                    right_indices=windows[segment["right"]],
                )
            )
            self.assertEqual(segment["changes"], expected["changes"])

    def test_identical_adjacent_windows_keep_an_empty_segment(self):
        result = self.call(windows=((0, 1), (0, 1), (0, 1)))
        self.assertEqual(len(result["windows"]), 3)
        self.assertEqual(
            [window["position"] for window in result["windows"]], [0, 1, 2]
        )
        self.assertEqual(len(result["segments"]), 2)
        for segment in result["segments"]:
            self.assertEqual(segment["changes"], ())
        self.assertEqual(
            result["windows"][0]["lifetimes"],
            result["windows"][1]["lifetimes"],
        )
        self.assertEqual(
            result["windows"][1]["lifetimes"],
            result["windows"][2]["lifetimes"],
        )

    def test_windows_may_share_indices_and_appear_out_of_order(self):
        # Index 1 is shared between the two windows and the windows run
        # backward in checkpoint order; uniqueness holds only per window.
        # Every identity exists at the shared checkpoint 1, so nothing is
        # purely added or removed -- only shifted/narrowed lifetimes.
        result = self.call(windows=((1, 2), (0, 1)))
        segment = result["segments"][0]
        self.assertEqual(
            [(change["kind"], change["identity"]) for change in segment["changes"]],
            [
                ("node_changed", ("a0", "x")),
                ("node_changed", ("W", "y")),
                ("edge_changed", (("a0", "x"), ("W", "y"))),
            ],
        )

    def test_disjoint_windows_keep_added_removed_classification(self):
        # Disjoint windows (2) then (0): W and the edge exist only in the
        # first window while a0 shifts, mirroring the two-window query.
        result = self.call(windows=((2,), (0,)))
        kinds = [
            (change["kind"], change["identity"])
            for change in result["segments"][0]["changes"]
        ]
        self.assertEqual(
            kinds,
            [
                ("node_removed", ("W", "y")),
                ("node_changed", ("a0", "x")),
                ("edge_removed", (("a0", "x"), ("W", "y"))),
            ],
        )

    def test_empty_windows_return_two_empty_tuples(self):
        result = self.call(windows=())
        self.assertEqual(result, {"windows": (), "segments": ()})

    def test_empty_causes_yield_empty_lifetimes_and_segments(self):
        result = self.call(causes=())
        self.assertTrue(all(window["lifetimes"] == () for window in result["windows"]))
        self.assertTrue(all(segment["changes"] == () for segment in result["segments"]))


class LifetimeEvolutionWindowsValidationTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        overrides.setdefault("causes", ("a0",))
        return evolution_call(self.store, self.args, **overrides)

    def test_windows_must_be_a_tuple(self):
        for bad in ("x", [()], 1, None, (() for _ in ())):
            with self.subTest(value=bad):
                with self.assertRaises(TypeError):
                    self.call(windows=bad)

    def test_each_window_must_be_a_tuple(self):
        for bad in ("x", [0], 1, None):
            with self.subTest(value=bad):
                with self.assertRaises(TypeError):
                    self.call(windows=((0,), bad))

    def test_window_elements_must_be_non_bool_ints(self):
        for bad in (True, False, 1.0, "1", None):
            with self.subTest(value=bad):
                with self.assertRaises(TypeError):
                    self.call(windows=((0, bad),))

    def test_index_value_errors(self):
        with self.assertRaises(ValueError):
            self.call(windows=((-1,),))
        with self.assertRaises(ValueError):
            self.call(windows=((3,),))
        with self.assertRaises(ValueError):
            self.call(windows=((1, 1),))
        with self.assertRaises(ValueError):
            self.call(windows=((2, 1),))

    def test_every_window_error_names_its_position_in_windows(self):
        cases = [
            ("type-window", ((0,), "x"), TypeError),
            ("type-element", ((0,), (0, True)), TypeError),
            ("negative", ((0,), (0,), (-1,)), ValueError),
            ("out-of-range", ((0,), (3,)), ValueError),
            ("duplicate", ((0,), (1,), (1, 1)), ValueError),
            ("not-increasing", ((0,), (1, 0)), ValueError),
        ]
        for label, windows, error in cases:
            with self.subTest(case=label):
                with self.assertRaises(error) as caught:
                    self.call(windows=windows)
                position = 2 if label in ("negative", "duplicate") else 1
                self.assertIn(f"windows[{position}]", str(caught.exception))

    def test_duplicate_index_message_keeps_window_position(self):
        with self.assertRaises(ValueError) as caught:
            self.call(windows=((0,), (0,), (2, 2)))
        self.assertIn("windows[2]", str(caught.exception))
        self.assertIn("duplicate", str(caught.exception))

    def test_identical_windows_are_not_duplicate_input_errors(self):
        # The same window twice, including adjacent placement, is fine.
        self.call(windows=((0, 1), (0, 1)))
        self.call(windows=((2,), (2,), (2,)))

    def test_windows_validated_in_input_order(self):
        with self.assertRaises(TypeError):
            self.call(windows=("x", (-1,)))
        with self.assertRaises(ValueError):
            self.call(windows=((-1,), "x"))
        with self.assertRaises(TypeError):
            self.call(windows=((0,), (True,), (-1,)))
        with self.assertRaises(ValueError):
            self.call(windows=((0,), (-1,), (True,)))

    def test_windows_validated_before_slice_inputs_and_limits(self):
        with self.assertRaises(TypeError):
            self.call(windows="x", causes=[1], window_limit=0)
        with self.assertRaises(ValueError):
            self.call(windows=((9,),), direction="sideways", window_limit=0)
        with self.assertRaises(TypeError):
            self.call(windows=((True,),), depth=True, window_limit=0)

    def test_ordinary_inputs_validated_first(self):
        with self.assertRaises(TypeError):
            self.call(reference=1, windows="x", causes=[1], window_limit=0)
        with self.assertRaises(ValueError):
            self.call(axis="nope", windows=((-1,),), window_limit=0)

    def test_all_input_validation_precedes_state_lookups(self):
        with self.assertRaises(TypeError):
            self.call(reference="ghost", windows="x")
        with self.assertRaises(ValueError):
            self.call(reference="ghost", window_limit=0)
        with self.assertRaises(KeyError):
            self.call(reference="ghost")


class LifetimeEvolutionLimitValidationTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        overrides.setdefault("causes", ("a0",))
        return evolution_call(self.store, self.args, **overrides)

    def test_window_limit_must_be_a_non_bool_positive_int(self):
        for bad in (True, False, 1.0, "1", None):
            with self.subTest(window_limit=bad):
                with self.assertRaises(TypeError):
                    self.call(window_limit=bad)
        with self.assertRaises(ValueError):
            self.call(window_limit=0)
        with self.assertRaises(ValueError):
            self.call(window_limit=-3)

    def test_total_diff_limit_must_be_a_non_bool_positive_int(self):
        for bad in (True, False, 1.0, "1", None):
            with self.subTest(total_diff_limit=bad):
                with self.assertRaises(TypeError):
                    self.call(total_diff_limit=bad)
        with self.assertRaises(ValueError):
            self.call(total_diff_limit=0)
        with self.assertRaises(ValueError):
            self.call(total_diff_limit=-3)

    def test_new_caps_validated_after_diff_limit_and_in_named_order(self):
        with self.assertRaises(ValueError):
            self.call(change_limit=0, window_limit=0)
        with self.assertRaises(ValueError):
            self.call(diff_limit=0, window_limit="x")
        with self.assertRaises(TypeError):
            self.call(diff_limit=1, window_limit="x")
        with self.assertRaises(ValueError):
            self.call(window_limit=0, total_diff_limit="x")
        with self.assertRaises(TypeError):
            self.call(window_limit=1, total_diff_limit="x")

    def test_window_count_beyond_window_limit_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.call(windows=((0,), (1,), (2,)), window_limit=2)
        result = self.call(windows=((0,), (1,), (2,)), window_limit=3)
        self.assertEqual(len(result["windows"]), 3)

    def test_window_count_cap_runs_before_state_lookups(self):
        with self.assertRaises(ValueError):
            self.call(
                reference="ghost", windows=((0,), (1,)), window_limit=1
            )

    def test_per_window_node_cap_breaches(self):
        # At checkpoint 1 the forward slice from a0 reaches W, so the
        # second window selects two nodes against a cap of one.
        with self.assertRaises(ValueError):
            self.call(
                windows=((0,), (1,), (2,)),
                direction="forward",
                node_limit=1,
            )

    def test_per_window_change_cap_breaches(self):
        with self.assertRaises(ValueError):
            self.call(windows=((0, 1),), change_limit=1)
        result = self.call(windows=((0, 1),), change_limit=3)
        self.assertEqual(len(result["windows"][0]["lifetimes"]), 3)

    def test_per_window_lifetime_cap_breaches_and_names_position(self):
        with self.assertRaises(ValueError) as caught:
            self.call(windows=((0,), (0, 1, 2)), lifetime_limit=2)
        self.assertIn("position 1", str(caught.exception))
        with self.assertRaises(ValueError):
            self.call(windows=((0, 1, 2),), lifetime_limit=2)
        result = self.call(windows=((0, 1, 2),), lifetime_limit=3)
        self.assertEqual(len(result["windows"][0]["lifetimes"]), 3)

    def test_segment_change_count_beyond_diff_limit_raises_value_error(self):
        # Window 0 -> 1 carries three changes.
        with self.assertRaises(ValueError):
            self.call(windows=((0,), (1,)), diff_limit=2)
        with self.assertRaises(ValueError):
            self.call(windows=((0,), (1,)), diff_limit=1)
        result = self.call(windows=((0,), (1,)), diff_limit=3)
        self.assertEqual(len(result["segments"][0]["changes"]), 3)

    def test_total_change_count_beyond_total_diff_limit_raises_value_error(self):
        # Two segments carry three changes each: six in total.
        with self.assertRaises(ValueError):
            self.call(total_diff_limit=5)
        with self.assertRaises(ValueError):
            self.call(total_diff_limit=1)
        result = self.call(total_diff_limit=6)
        self.assertEqual(
            sum(len(segment["changes"]) for segment in result["segments"]), 6
        )

    def test_identical_windows_spend_no_diff_budget(self):
        result = self.call(
            windows=((0, 1), (0, 1), (0, 1)), total_diff_limit=1
        )
        self.assertEqual(
            [len(segment["changes"]) for segment in result["segments"]], [0, 0]
        )


class LifetimeEvolutionStateCheckTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        overrides.setdefault("causes", ("a0",))
        return evolution_call(self.store, self.args, **overrides)

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

    def test_misaligned_series_points_raise_value_error(self):
        with self.assertRaises(ValueError):
            self.call(
                series={
                    "a": (("r0", "a0"), ("r1", "a0"), ("r2", "a0")),
                    "b": (("r1", "a0"), ("r1", "W"), ("r2", "W")),
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
            self.call(windows=((0, 1),), change_limit=1)

    def test_empty_windows_still_run_every_state_check(self):
        with self.assertRaises(KeyError):
            self.call(windows=(), reference="ghost")
        with self.assertRaises(ValueError):
            self.call(windows=(), limit=1)
        with self.assertRaises(KeyError):
            self.call(
                windows=(),
                series={
                    "a": (("r0", "a0"), ("r1", "a0"), ("r2", "a0")),
                    "b": (("r0", "a0"), ("r1", "W"), ("r2", "nope2")),
                },
            )

    def test_empty_windows_still_run_every_input_check(self):
        with self.assertRaises(ValueError):
            self.call(windows=(), node_limit=0)
        with self.assertRaises(ValueError):
            self.call(windows=(), change_limit=0)
        with self.assertRaises(ValueError):
            self.call(windows=(), lifetime_limit=0)
        with self.assertRaises(ValueError):
            self.call(windows=(), diff_limit=0)
        with self.assertRaises(ValueError):
            self.call(windows=(), window_limit=0)
        with self.assertRaises(ValueError):
            self.call(windows=(), total_diff_limit=0)
        with self.assertRaises(TypeError):
            self.call(windows=(), causes=[1])
        with self.assertRaises(TypeError):
            self.call(windows=(), window_limit=True)


class LifetimeEvolutionIsolationTests(unittest.TestCase):
    def test_result_levels_are_mutually_unshared(self):
        store, args = diamond_store()
        result = evolution_call(store, args)
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
        first = evolution_call(store, args)
        pristine = copy.deepcopy(first)
        for window in first["windows"]:
            for record in window["lifetimes"]:
                record["intervals"] += ((99, 99),)
                for transition in record["transitions"]:
                    for side in ("before", "after"):
                        evidence = transition[side]
                        if evidence is not None and "branches" in evidence:
                            evidence["branches"] += ("evil",)
        for segment in first["segments"]:
            for change in segment["changes"]:
                for side in ("before", "after"):
                    if change[side] is not None:
                        change[side]["first_seen"] = 99
        second = evolution_call(store, args)
        self.assertEqual(second, pristine)


class LifetimeEvolutionReadOnlyTests(unittest.TestCase):
    def test_success_and_failure_change_nothing(self):
        store, args = diamond_store()
        heads = dict(store._heads)
        appends = copy.deepcopy(store._appends)
        merges = copy.deepcopy(store._merges)
        audit = store.audit_log()

        evolution_call(store, args)
        failures = [
            dict(windows="x"),
            dict(windows=("x",)),
            dict(windows=((0,), "x")),
            dict(windows=((-1,),)),
            dict(windows=((9,),)),
            dict(windows=((1, 0),)),
            dict(windows=((1, 1),)),
            dict(windows=((True,),)),
            dict(causes=[1]),
            dict(direction="up"),
            dict(depth=-1),
            dict(node_limit=0),
            dict(change_limit=0),
            dict(windows=((0, 1),), change_limit=1),
            dict(lifetime_limit=0),
            dict(lifetime_limit=1),
            dict(diff_limit=0),
            dict(diff_limit=1),
            dict(window_limit=0),
            dict(total_diff_limit=0),
            dict(total_diff_limit=1),
            dict(causes=("zzz",)),
            dict(limit=1),
            dict(reference="ghost"),
        ]
        for overrides in failures:
            with self.subTest(overrides=overrides):
                with self.assertRaises(Exception):
                    evolution_call(store, args, causes=("a0",), **overrides)

        self.assertEqual(dict(store._heads), heads)
        self.assertEqual(copy.deepcopy(store._appends), appends)
        self.assertEqual(copy.deepcopy(store._merges), merges)
        self.assertEqual(store.audit_log(), audit)

    def test_existing_queries_are_unaffected(self):
        store, args = diamond_store()
        lifetimes_kwargs = compare_kwargs(args)
        lifetimes_kwargs.pop("left_indices")
        lifetimes_kwargs.pop("right_indices")
        lifetimes_kwargs.pop("diff_limit")
        single = store.cascade_slice_lifetimes(
            **dict(lifetimes_kwargs, indices=(0, 1, 2))
        )
        compared = store.compare_lifetimes(
            **compare_kwargs(args, left_indices=(0, 1), right_indices=(1, 2))
        )
        evolution_call(store, args)
        self.assertEqual(
            store.cascade_slice_lifetimes(
                **dict(lifetimes_kwargs, indices=(0, 1, 2))
            ),
            single,
        )
        self.assertEqual(
            store.compare_lifetimes(
                **compare_kwargs(args, left_indices=(0, 1), right_indices=(1, 2))
            ),
            compared,
        )

    def test_repeated_calls_are_equal(self):
        store, args = diamond_store()
        first = evolution_call(store, args)
        second = evolution_call(store, args)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
