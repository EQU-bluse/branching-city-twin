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
        windows=((0, 1), (1, 2)),
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


class LifetimeEvolutionSignatureTests(unittest.TestCase):
    def test_public_signature_replaces_window_pair_and_appends_two_caps(self):
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
            ],
        )
        self.assertTrue(
            all(p.default is inspect.Parameter.empty for p in parameters)
        )

    def test_result_is_a_dict_with_windows_segments(self):
        store, args = diamond_store()
        result = evolution_call(store, args)
        self.assertIsInstance(result, dict)
        self.assertEqual(list(result), ["windows", "segments"])
        self.assertIsInstance(result["windows"], tuple)
        self.assertIsInstance(result["segments"], tuple)
        self.assertEqual(len(result["windows"]), 2)
        self.assertEqual(len(result["segments"]), 1)
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


class LifetimeEvolutionSemanticsTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def test_window_records_carry_zero_based_positions_in_order(self):
        result = evolution_call(
            self.store, self.args, windows=((0,), (0, 1), (1, 2))
        )
        self.assertEqual(
            [window["position"] for window in result["windows"]], [0, 1, 2]
        )
        self.assertEqual(
            [(s["left"], s["right"]) for s in result["segments"]],
            [(0, 1), (1, 2)],
        )

    def test_each_window_matches_the_lifetime_query_on_its_indices(self):
        windows = ((0,), (0, 1), (1, 2), (0, 1, 2))
        result = evolution_call(self.store, self.args, windows=windows)
        base_kwargs = evolution_kwargs(self.args, windows=windows)
        for base in ("windows", "diff_limit", "window_limit", "total_diff_limit"):
            base_kwargs.pop(base)
        for position, index_window in enumerate(windows):
            expected = self.store.cascade_slice_lifetimes(
                **dict(base_kwargs, indices=index_window)
            )
            self.assertEqual(
                result["windows"][position]["lifetimes"],
                expected["lifetimes"],
            )

    def test_each_segment_matches_the_two_window_comparison(self):
        windows = ((0,), (1,), (0, 1))
        result = evolution_call(self.store, self.args, windows=windows)
        compare_kwargs = evolution_kwargs(self.args, windows=windows)
        for base in ("windows", "window_limit", "total_diff_limit"):
            compare_kwargs.pop(base)
        for position in range(len(windows) - 1):
            expected = self.store.compare_lifetimes(
                **dict(
                    compare_kwargs,
                    left_indices=windows[position],
                    right_indices=windows[position + 1],
                )
            )
            self.assertEqual(
                result["segments"][position]["changes"], expected["changes"]
            )

    def test_identical_adjacent_windows_keep_segment_with_empty_changes(self):
        result = evolution_call(
            self.store, self.args, windows=((0, 1), (0, 1), (1, 2))
        )
        self.assertEqual(len(result["segments"]), 2)
        identical, differing = result["segments"]
        self.assertEqual((identical["left"], identical["right"]), (0, 1))
        self.assertEqual(identical["changes"], ())
        self.assertEqual((differing["left"], differing["right"]), (1, 2))
        self.assertEqual(len(differing["changes"]), 3)
        self.assertEqual(
            result["windows"][0]["lifetimes"],
            result["windows"][1]["lifetimes"],
        )

    def test_single_window_has_no_segments(self):
        result = evolution_call(self.store, self.args, windows=((2,),))
        self.assertEqual(len(result["windows"]), 1)
        self.assertEqual(result["windows"][0]["position"], 0)
        self.assertEqual(result["segments"], ())

    def test_segment_change_classification_matches_two_window_query(self):
        result = evolution_call(
            self.store, self.args, windows=((0,), (1,))
        )
        (segment,) = result["segments"]
        self.assertEqual(
            [(c["kind"], c["identity"]) for c in segment["changes"]],
            [
                ("node_added", ("W", "y")),
                ("node_changed", ("a0", "x")),
                ("edge_added", (("a0", "x"), ("W", "y"))),
            ],
        )


class LifetimeEvolutionWindowsValidationTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        overrides.setdefault("causes", ("a0",))
        return evolution_call(self.store, self.args, **overrides)

    def test_windows_must_be_a_tuple(self):
        for bad in ("x", [()], 1, None, ((), [])):
            with self.subTest(value=bad):
                with self.assertRaises(TypeError):
                    self.call(windows=bad)

    def test_each_window_must_be_a_tuple_with_its_position(self):
        for bad in ("x", [0, 1], 1, None):
            with self.subTest(position=0, value=bad):
                with self.assertRaises(TypeError) as caught:
                    self.call(windows=(bad,))
                self.assertIn("windows[0]", str(caught.exception))
            with self.subTest(position=1, value=bad):
                with self.assertRaises(TypeError) as caught:
                    self.call(windows=((), bad))
                self.assertIn("windows[1]", str(caught.exception))

    def test_window_elements_must_be_non_bool_ints(self):
        for position, window in ((0, (True,)), (1, (0, False)), (2, (0, 1.0))):
            with self.subTest(window=window):
                with self.assertRaises(TypeError) as caught:
                    self.call(windows=((),) * position + (window,))
                self.assertIn(f"windows[{position}]", str(caught.exception))

    def test_window_index_value_errors_carry_window_position(self):
        cases = [
            ("negative", (-1,)),
            ("out-of-range", (3,)),
            ("duplicate", (1, 1)),
            ("not-increasing", (2, 1)),
        ]
        for position in (0, 2):
            for label, window in cases:
                with self.subTest(position=position, case=label):
                    with self.assertRaises(ValueError) as caught:
                        self.call(
                            windows=((),) * position
                            + (window,)
                            + ((),) * (2 - position)
                        )
                    self.assertIn(f"windows[{position}]", str(caught.exception))

    def test_duplicate_index_error_keeps_window_position_identifier(self):
        with self.assertRaises(ValueError) as caught:
            self.call(windows=((0,), (2, 2)))
        message = str(caught.exception)
        self.assertIn("windows[1]", message)
        self.assertIn("duplicate", message)

    def test_identical_windows_may_be_adjacent(self):
        result = self.call(windows=((0, 1), (0, 1), (0, 1)))
        self.assertEqual(len(result["windows"]), 3)
        self.assertEqual(
            [segment["changes"] for segment in result["segments"]],
            [(), ()],
        )

    def test_windows_validated_in_input_order(self):
        with self.assertRaises(TypeError):
            self.call(windows=("x", (-1,)))
        with self.assertRaises(ValueError):
            self.call(windows=((-1,), "x"))
        with self.assertRaises(ValueError):
            self.call(windows=((0, 0), (-1,)))
        with self.assertRaises(TypeError) as caught:
            self.call(windows=((), (True,)))
        self.assertIn("windows[1]", str(caught.exception))

    def test_windows_validated_before_slice_inputs_and_limits(self):
        with self.assertRaises(TypeError):
            self.call(windows="x", causes=[1], window_limit=0)
        with self.assertRaises(ValueError):
            self.call(windows=((9,),), direction="sideways", window_limit=0)
        with self.assertRaises(TypeError):
            self.call(windows=((0, True),), depth=True, window_limit=0)

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

    def test_new_caps_must_be_non_bool_ints(self):
        for name in ("window_limit", "total_diff_limit"):
            for bad in (True, False, 1.0, "1", None):
                with self.subTest(name=name, value=bad):
                    with self.assertRaises(TypeError):
                        self.call(**{name: bad})

    def test_new_caps_below_one_raise_value_error(self):
        for name in ("window_limit", "total_diff_limit"):
            for bad in (0, -2):
                with self.subTest(name=name, value=bad):
                    with self.assertRaises(ValueError):
                        self.call(**{name: bad})

    def test_new_caps_validated_after_existing_limits(self):
        # window_limit is checked after diff_limit, total_diff_limit last.
        with self.assertRaises(ValueError):
            self.call(diff_limit=0, window_limit="x")
        with self.assertRaises(ValueError):
            self.call(window_limit=0, total_diff_limit="x")
        with self.assertRaises(TypeError):
            self.call(window_limit=1, total_diff_limit="x")
        with self.assertRaises(TypeError):
            self.call(diff_limit=1, window_limit="x")

    def test_window_count_beyond_window_limit_raises(self):
        with self.assertRaises(ValueError) as caught:
            self.call(windows=((), (), (), ()), window_limit=3)
        self.assertIn("window limit", str(caught.exception))
        result = self.call(windows=((), (), ()), window_limit=3)
        self.assertEqual(len(result["windows"]), 3)

    def test_lifetime_limit_caps_each_window_in_window_order(self):
        # The (0, 1, 2) window carries three lifetime records.
        with self.assertRaises(ValueError) as caught:
            self.call(windows=((0, 1, 2), (0,)), lifetime_limit=2)
        message = str(caught.exception)
        self.assertIn("lifetime limit", message)
        self.assertIn("position 0", message)
        with self.assertRaises(ValueError) as caught:
            self.call(windows=((0,), (0, 1, 2)), lifetime_limit=2)
        self.assertIn("position 1", str(caught.exception))
        result = self.call(
            windows=((0, 1, 2), (0, 1, 2)), lifetime_limit=3
        )
        self.assertTrue(all(len(w["lifetimes"]) == 3 for w in result["windows"]))

    def test_existing_per_window_caps_still_breach(self):
        with self.assertRaises(ValueError):
            self.call(limit=1)
        # The second window trips the node cap, the first the change cap.
        with self.assertRaises(ValueError):
            self.call(
                windows=((0,), (0, 1)),
                direction="forward",
                node_limit=1,
            )
        with self.assertRaises(ValueError):
            self.call(windows=((0, 1), (0,)), change_limit=1)

    def test_segment_beyond_diff_limit_raises_with_position(self):
        # Three changes between (0, 1) and (1, 2).
        with self.assertRaises(ValueError) as caught:
            self.call(windows=((0, 1), (1, 2)), diff_limit=2)
        self.assertIn("position 0", str(caught.exception))
        # The first segment is empty; the second carries three changes.
        with self.assertRaises(ValueError) as caught:
            self.call(
                windows=((0, 1), (0, 1), (1, 2)), diff_limit=2
            )
        self.assertIn("position 1", str(caught.exception))
        result = self.call(windows=((0, 1), (1, 2)), diff_limit=3)
        self.assertEqual(len(result["segments"][0]["changes"]), 3)

    def test_batch_change_count_beyond_total_diff_limit_raises(self):
        # Two three-change segments: (0,) -> (1,) adds W, then (1,) -> (0,).
        windows = ((0,), (1,), (0,))
        with self.assertRaises(ValueError) as caught:
            self.call(windows=windows, total_diff_limit=5)
        message = str(caught.exception)
        self.assertIn("total diff limit", message)
        self.assertIn("6 change records", message)
        result = self.call(windows=windows, total_diff_limit=6)
        self.assertEqual(
            [len(s["changes"]) for s in result["segments"]], [3, 3]
        )

    def test_segment_and_batch_caps_return_no_partial_result(self):
        # Limits that fail must raise even when each individual window is
        # fine on its own.
        with self.assertRaises(ValueError):
            self.call(
                windows=((0,), (1,), (0,)),
                diff_limit=50,
                total_diff_limit=1,
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

    def test_misaligned_checkpoints_raise_value_error(self):
        with self.assertRaises(ValueError):
            self.call(
                series={
                    "a": (("r0", "a0"), ("r1", "a0"), ("r2", "a0")),
                    "b": (("r0", "a0"), ("r1", "W")),
                },
            )

    def test_absent_cause_raises_key_error(self):
        with self.assertRaises(KeyError) as caught:
            self.call(windows=((0,),), causes=("zzz",))
        self.assertEqual(caught.exception.args[0], "zzz")

    def test_candidate_cap_still_breaches(self):
        with self.assertRaises(ValueError):
            self.call(limit=1)


class LifetimeEvolutionEmptyTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        overrides.setdefault("causes", ("a0",))
        return evolution_call(self.store, self.args, **overrides)

    def test_empty_windows_yield_two_empty_tuples(self):
        result = self.call(windows=())
        self.assertEqual(result, {"windows": (), "segments": ()})

    def test_empty_windows_still_run_input_and_state_checks(self):
        with self.assertRaises(KeyError):
            self.call(windows=(), reference="ghost")
        with self.assertRaises(ValueError):
            self.call(windows=(), limit=1)
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
        # Zero plans leave no valid index, but no window needs one here.
        self.assertEqual(
            self.call(
                windows=(),
                series={},
                min_size=0,
                max_size=0,
            ),
            {"windows": (), "segments": ()},
        )

    def test_empty_windows_with_zero_plans_check_window_limit_against_count(self):
        # Zero windows never trip the window count cap.
        result = self.call(
            windows=(),
            series={},
            min_size=0,
            max_size=0,
            window_limit=1,
        )
        self.assertEqual(result, {"windows": (), "segments": ()})

    def test_empty_causes_yield_empty_lifetimes_and_segments(self):
        result = self.call(windows=((0,), (1,), (2,)), causes=())
        self.assertEqual(
            [window["lifetimes"] for window in result["windows"]],
            [(), (), ()],
        )
        self.assertEqual(len(result["segments"]), 2)
        self.assertTrue(
            all(segment["changes"] == () for segment in result["segments"])
        )
        self.assertEqual(
            [(s["left"], s["right"]) for s in result["segments"]],
            [(0, 1), (1, 2)],
        )

    def test_empty_inner_windows_are_allowed_and_compared(self):
        result = self.call(windows=((), (), (0,)))
        self.assertEqual(
            [window["lifetimes"] for window in result["windows"]],
            [(), (), result["windows"][2]["lifetimes"]],
        )
        self.assertEqual(result["segments"][0]["changes"], ())
        self.assertEqual(result["segments"][1]["left"], 1)
        self.assertEqual(result["segments"][1]["right"], 2)
        self.assertTrue(len(result["segments"][1]["changes"]) >= 1)


class LifetimeEvolutionIsolationTests(unittest.TestCase):
    def test_result_levels_are_mutually_unshared(self):
        store, args = diamond_store()
        result = evolution_call(
            store, args, windows=((0, 1), (1, 2), (0, 1))
        )
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

    def test_identical_windows_share_no_result_objects(self):
        store, args = diamond_store()
        result = evolution_call(store, args, windows=((0, 1), (0, 1)))
        first, second = result["windows"]
        self.assertIsNot(first, second)
        self.assertIsNot(first["lifetimes"], second["lifetimes"])
        self.assertEqual(first["lifetimes"], second["lifetimes"])
        for left_record, right_record in zip(
            first["lifetimes"], second["lifetimes"]
        ):
            self.assertIsNot(left_record, right_record)

    def test_mutating_result_never_affects_later_calls(self):
        store, args = diamond_store()
        first = evolution_call(store, args, windows=((0, 1), (1, 2)))
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
        second = evolution_call(store, args, windows=((0, 1), (1, 2)))
        self.assertEqual(second, pristine)


class LifetimeEvolutionReadOnlyTests(unittest.TestCase):
    def test_success_and_failure_change_nothing(self):
        store, args = diamond_store()
        heads = dict(store._heads)
        appends = copy.deepcopy(store._appends)
        merges = copy.deepcopy(store._merges)
        audit = store.audit_log()

        evolution_call(store, args, windows=((0, 1), (1, 2)))
        failures = [
            dict(windows="x"),
            dict(windows=([0],)),
            dict(windows=((True,),)),
            dict(windows=((-1,),)),
            dict(windows=((9,),)),
            dict(windows=((1, 0),)),
            dict(windows=((1, 1),)),
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
            dict(window_limit=0),
            dict(window_limit=1),
            dict(total_diff_limit=0),
            dict(total_diff_limit=1),
            dict(causes=("zzz",)),
            dict(limit=1),
            dict(reference="ghost"),
        ]
        for overrides in failures:
            with self.subTest(overrides=overrides):
                with self.assertRaises(Exception):
                    evolution_call(
                        store,
                        args,
                        causes=("a0",),
                        windows=((0, 1), (1, 2)),
                        **overrides,
                    )

        self.assertEqual(dict(store._heads), heads)
        self.assertEqual(copy.deepcopy(store._appends), appends)
        self.assertEqual(copy.deepcopy(store._merges), merges)
        self.assertEqual(store.audit_log(), audit)

    def test_existing_queries_are_unaffected(self):
        store, args = diamond_store()
        kwargs = evolution_kwargs(
            args, windows=((0, 1), (1, 2), (0,))
        )
        lifetimes_kwargs = dict(
            kwargs,
            indices=(0, 1, 2),
        )
        for key in (
            "windows",
            "diff_limit",
            "window_limit",
            "total_diff_limit",
        ):
            lifetimes_kwargs.pop(key)
        lifetimes = store.cascade_slice_lifetimes(**lifetimes_kwargs)
        evolution_call(store, args, windows=((0, 1), (1, 2), (0,)))
        self.assertEqual(
            store.cascade_slice_lifetimes(**lifetimes_kwargs), lifetimes
        )
        compare = store.compare_lifetimes(
            **dict(
                {
                    k: v
                    for k, v in kwargs.items()
                    if k not in ("windows", "window_limit", "total_diff_limit")
                },
                left_indices=(0, 1),
                right_indices=(1, 2),
            )
        )
        self.assertEqual(
            store.compare_lifetimes(
                **dict(
                    {
                        k: v
                        for k, v in kwargs.items()
                        if k
                        not in ("windows", "window_limit", "total_diff_limit")
                    },
                    left_indices=(0, 1),
                    right_indices=(1, 2),
                )
            ),
            compare,
        )

    def test_repeated_calls_are_equal(self):
        store, args = diamond_store()
        windows = ((0,), (0, 1), (1, 2), (0, 1))
        first = evolution_call(store, args, windows=windows)
        second = evolution_call(store, args, windows=windows)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
