import unittest

from city_twin.branches import BranchStore
from city_twin.event_graph import EventGraph


def make_store() -> BranchStore:
    graph = EventGraph()
    graph.add("root", 0, (), {"a": 1})
    store = BranchStore(graph)
    store.create("main", "root")
    store.create("feature", "root")
    store.append("main", "m1", 1, {"b": 2})
    store.append("feature", "f1", 2, {"c": 3})
    store.append("feature", "f2", 3, {"c": 0})
    store.merge("main", "feature", "merge1", 4, {"m": 5})
    return store


def make_diverging_store() -> BranchStore:
    graph = EventGraph()
    graph.add("root", 0, (), {})
    store = BranchStore(graph)
    store.create("main", "root")
    store.create("feature", "root")
    store.append("main", "m1", 1, {"x": 1})
    store.append("feature", "f1", 2, {"x": 2})
    return store


class AttributeDivergenceTests(unittest.TestCase):
    def test_result_keys_are_ordered(self) -> None:
        result = make_diverging_store().attribute_divergence(
            "main", "feature", "x"
        )
        self.assertEqual(list(result), ["key", "fork", "left", "right"])
        for side in (result["left"], result["right"]):
            self.assertEqual(list(side), ["value", "cause", "path", "affected"])

    def test_diverging_values_attribute_first_exclusive_touch(self) -> None:
        result = make_diverging_store().attribute_divergence(
            "main", "feature", "x"
        )
        self.assertEqual(result["key"], "x")
        self.assertEqual(result["fork"], "root")
        self.assertEqual(
            result["left"],
            {"value": 1, "cause": "m1", "path": ("m1",), "affected": ()},
        )
        self.assertEqual(
            result["right"],
            {"value": 2, "cause": "f1", "path": ("f1",), "affected": ()},
        )

    def test_side_without_exclusive_touch_has_none_cause(self) -> None:
        # "b" is appended only on main; feature's closure is a subset.
        result = make_store().attribute_divergence("main", "feature", "b")
        self.assertEqual(result["fork"], "f2")
        self.assertEqual(
            result["left"],
            {
                "value": 2,
                "cause": "m1",
                "path": ("m1", "merge1"),
                "affected": ("merge1",),
            },
        )
        self.assertEqual(
            result["right"],
            {"value": 0, "cause": None, "path": (), "affected": ()},
        )

    def test_equal_values_have_no_causes(self) -> None:
        # Both sides replay "c" to 3 (f1: +3, f2: +0).
        result = make_store().attribute_divergence("main", "feature", "c")
        self.assertEqual(result["fork"], "f2")
        self.assertEqual(
            result["left"],
            {"value": 3, "cause": None, "path": (), "affected": ()},
        )
        self.assertEqual(
            result["right"],
            {"value": 3, "cause": None, "path": (), "affected": ()},
        )

    def test_untouched_key_defaults_to_zero_on_both_sides(self) -> None:
        result = make_store().attribute_divergence("main", "feature", "zzz")
        self.assertEqual(result["left"]["value"], 0)
        self.assertEqual(result["right"]["value"], 0)
        self.assertIsNone(result["left"]["cause"])
        self.assertIsNone(result["right"]["cause"])

    def test_zero_delta_event_can_be_cause(self) -> None:
        store = make_store()
        store.append("feature", "f3", 5, {"b": 0})
        result = store.attribute_divergence("main", "feature", "b")
        # main replays "b" to 2, feature to 0; f3 touches "b" with a
        # zero delta and is the first feature-exclusive touch.
        self.assertEqual(result["right"]["cause"], "f3")
        self.assertEqual(result["right"]["path"], ("f3",))
        self.assertEqual(result["right"]["affected"], ())

    def test_path_and_affected_span_multiple_events(self) -> None:
        store = make_diverging_store()
        store.append("main", "m2", 3, {"y": 1})
        store.append("main", "m3", 4, {"z": 1})
        result = store.attribute_divergence("main", "feature", "x")
        self.assertEqual(result["left"]["cause"], "m1")
        self.assertEqual(result["left"]["path"], ("m1", "m2", "m3"))
        self.assertEqual(result["left"]["affected"], ("m2", "m3"))

    def test_shortest_path_tie_broken_by_id_tuple_order(self) -> None:
        graph = EventGraph()
        graph.add("root", 0, (), {})
        store = BranchStore(graph)
        store.create("main", "root")
        store.append("main", "m1", 1, {"x": 1})
        store.create("side", "m1")
        store.append("side", "s1", 2, {"y": 1})
        store.append("main", "m2", 3, {"z": 1})
        store.merge("main", "side", "merge1", 4, {})
        store.create("other", "root")
        store.append("other", "o1", 5, {"x": 9})
        result = store.attribute_divergence("main", "other", "x")
        # Two shortest cause-to-head paths exist; ("m1", "m2", "merge1")
        # is Unicode-smaller than ("m1", "s1", "merge1").
        self.assertEqual(result["left"]["cause"], "m1")
        self.assertEqual(result["left"]["path"], ("m1", "m2", "merge1"))
        self.assertEqual(
            result["left"]["affected"], ("s1", "m2", "merge1")
        )

    def test_affected_follows_side_replay_order(self) -> None:
        store = make_diverging_store()
        store.append("main", "zz", 2, {"y": 1})
        store.append("main", "aa", 3, {"z": 1})
        result = store.attribute_divergence("main", "feature", "x")
        # Replay order is (at, id), not id order.
        self.assertEqual(result["left"]["affected"], ("zz", "aa"))

    def test_no_common_events_gives_none_fork(self) -> None:
        graph = EventGraph()
        graph.add("r1", 0, (), {"x": 1})
        graph.add("r2", 0, (), {"x": 2})
        store = BranchStore(graph)
        store.create("one", "r1")
        store.create("two", "r2")
        result = store.attribute_divergence("one", "two", "x")
        self.assertIsNone(result["fork"])
        self.assertEqual(result["left"]["cause"], "r1")
        self.assertEqual(result["right"]["cause"], "r2")

    def test_result_is_detached_from_internal_state(self) -> None:
        store = make_diverging_store()
        result = store.attribute_divergence("main", "feature", "x")
        result["left"]["value"] = 999
        result["left"]["path"] += ("evil",)
        result["fork"] = "evil"
        fresh = store.attribute_divergence("main", "feature", "x")
        self.assertEqual(fresh["left"]["value"], 1)
        self.assertEqual(fresh["left"]["path"], ("m1",))
        self.assertEqual(fresh["fork"], "root")
        self.assertIsNot(result, fresh)
        self.assertIsNot(result["left"], fresh["left"])

    def test_non_str_arguments_raise_type_error(self) -> None:
        store = make_store()
        for bad in (None, 1, 1.5, b"main", ["main"]):
            with self.assertRaises(TypeError):
                store.attribute_divergence(bad, "feature", "c")  # type: ignore[arg-type]
            with self.assertRaises(TypeError):
                store.attribute_divergence("main", bad, "c")  # type: ignore[arg-type]
            with self.assertRaises(TypeError):
                store.attribute_divergence("main", "feature", bad)  # type: ignore[arg-type]

    def test_empty_arguments_raise_value_error(self) -> None:
        store = make_store()
        with self.assertRaises(ValueError):
            store.attribute_divergence("", "feature", "c")
        with self.assertRaises(ValueError):
            store.attribute_divergence("main", "", "c")
        with self.assertRaises(ValueError):
            store.attribute_divergence("main", "feature", "")

    def test_unknown_branch_raises_key_error(self) -> None:
        store = make_store()
        with self.assertRaises(KeyError):
            store.attribute_divergence("ghost", "feature", "c")
        with self.assertRaises(KeyError):
            store.attribute_divergence("main", "ghost", "c")

    def test_same_branch_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            make_store().attribute_divergence("main", "main", "c")

    def test_validation_precedes_branch_lookup(self) -> None:
        store = make_store()
        # Invalid key is rejected before unknown branches are consulted.
        with self.assertRaises(ValueError):
            store.attribute_divergence("ghost", "ghost", "")
        # Unknown branches raise KeyError before the same-name check.
        with self.assertRaises(KeyError):
            store.attribute_divergence("ghost", "ghost", "c")

    def test_query_is_read_only(self) -> None:
        store = make_store()
        before_main = store.replay("main")
        before_feature = store.replay("feature")
        store.attribute_divergence("main", "feature", "b")
        store.attribute_divergence("main", "feature", "zzz")
        self.assertEqual(store.replay("main"), before_main)
        self.assertEqual(store.replay("feature"), before_feature)
        self.assertEqual(store.head("main"), "merge1")
        self.assertEqual(store.head("feature"), "f2")


if __name__ == "__main__":
    unittest.main()
