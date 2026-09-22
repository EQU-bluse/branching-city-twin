import unittest

from city_twin.branches import BranchStore
from city_twin.event_graph import EventGraph


def make_graph() -> EventGraph:
    graph = EventGraph()
    graph.add("root", 0, (), {"a": 1})
    graph.add("e1", 2, ("root",), {"b": 2})
    graph.add("e2", 2, ("root",), {"a": 3})
    graph.add("e3", 5, ("e1", "e2"), {"a": -1, "b": -2})
    graph.add("e4", 9, ("e3",), {"c": 7})
    return graph


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


class EventGraphDiffAtTests(unittest.TestCase):
    def test_pairs_states_keyed_by_union_sorted(self) -> None:
        graph = make_graph()
        graph.add("other", 3, ("root",), {"Z": 4, "a": 10})
        diff = graph.diff_at("e4", "other", 5)
        self.assertEqual(
            diff,
            {"Z": (0, 4), "a": (3, 11), "b": (0, 0)},
        )
        self.assertEqual(list(diff), ["Z", "a", "b"])

    def test_only_events_up_to_at_count(self) -> None:
        graph = make_graph()
        self.assertEqual(graph.diff_at("e4", "root", 0), {"a": (1, 1)})
        self.assertEqual(
            graph.diff_at("e4", "root", 2),
            {"a": (4, 1), "b": (2, 0)},
        )
        self.assertEqual(
            graph.diff_at("e4", "root", 9),
            {"a": (3, 1), "b": (0, 0), "c": (7, 0)},
        )

    def test_keeps_zero_value_keys_on_both_sides(self) -> None:
        graph = make_graph()
        diff = graph.diff_at("e4", "e1", 5)
        self.assertEqual(diff["b"], (0, 2))

    def test_same_head_matches_replay_at_pairs(self) -> None:
        graph = make_graph()
        state = graph.replay_at("e4", 9)
        self.assertEqual(
            graph.diff_at("e4", "e4", 9),
            {key: (value, value) for key, value in state.items()},
        )

    def test_non_str_heads_raise_type_error(self) -> None:
        graph = make_graph()
        for bad in (None, 1, 1.5, b"e4", ["e4"]):
            with self.assertRaises(TypeError):
                graph.diff_at(bad, "e4", 5)  # type: ignore[arg-type]
            with self.assertRaises(TypeError):
                graph.diff_at("e4", bad, 5)  # type: ignore[arg-type]

    def test_empty_heads_raise_value_error(self) -> None:
        graph = make_graph()
        with self.assertRaises(ValueError):
            graph.diff_at("", "e4", 5)
        with self.assertRaises(ValueError):
            graph.diff_at("e4", "", 5)

    def test_non_int_at_raises_type_error(self) -> None:
        graph = make_graph()
        for bad in (None, "5", 1.5, True, False):
            with self.assertRaises(TypeError):
                graph.diff_at("e4", "root", bad)  # type: ignore[arg-type]

    def test_negative_at_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            make_graph().diff_at("e4", "root", -1)

    def test_arguments_validated_in_signature_order(self) -> None:
        graph = make_graph()
        with self.assertRaises(TypeError):
            graph.diff_at(None, "nope", True)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            graph.diff_at("", "nope", -1)
        with self.assertRaises(TypeError):
            graph.diff_at("e4", None, True)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            graph.diff_at("e4", "", -1)
        # at is checked before either head is looked up.
        with self.assertRaises(TypeError):
            graph.diff_at("nope-a", "nope-b", True)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            graph.diff_at("nope-a", "nope-b", -1)

    def test_unknown_heads_raise_key_error_after_validation(self) -> None:
        graph = make_graph()
        with self.assertRaises(KeyError):
            graph.diff_at("nope", "e4", 5)
        with self.assertRaises(KeyError):
            graph.diff_at("e4", "nope", 5)
        try:
            graph.diff_at("nope-a", "nope-b", 5)
        except KeyError as error:
            self.assertEqual(error.args, ("nope-a",))
        else:
            self.fail("expected KeyError")

    def test_query_is_read_only_and_detached(self) -> None:
        graph = make_graph()
        before = graph.replay("e4")
        diff = graph.diff_at("e4", "root", 5)
        diff["a"] = (999, 999)
        diff["b"] = (9, 9)
        self.assertEqual(graph.replay("e4"), before)
        again = graph.diff_at("e4", "root", 5)
        self.assertEqual(again, {"a": (3, 1), "b": (0, 0)})
        self.assertTrue(all(type(value) is tuple for value in again.values()))


class BranchStoreDiffAtTests(unittest.TestCase):
    def test_diffs_current_heads_as_of_at(self) -> None:
        store = make_store()
        self.assertEqual(store.diff_at("main", "feature", 0), {"a": (1, 1)})
        self.assertEqual(
            store.diff_at("main", "feature", 1),
            {"a": (1, 1), "b": (2, 0)},
        )
        self.assertEqual(
            store.diff_at("main", "feature", 4),
            {"a": (1, 1), "b": (2, 0), "c": (3, 3), "m": (5, 0)},
        )

    def test_non_str_names_raise_type_error(self) -> None:
        store = make_store()
        for bad in (None, 1, 1.5, b"main", ["main"]):
            with self.assertRaises(TypeError):
                store.diff_at(bad, "feature", 1)  # type: ignore[arg-type]
            with self.assertRaises(TypeError):
                store.diff_at("main", bad, 1)  # type: ignore[arg-type]

    def test_empty_names_raise_value_error(self) -> None:
        store = make_store()
        with self.assertRaises(ValueError):
            store.diff_at("", "feature", 1)
        with self.assertRaises(ValueError):
            store.diff_at("main", "", 1)

    def test_non_int_at_raises_type_error(self) -> None:
        store = make_store()
        for bad in (None, "1", 1.5, True, False):
            with self.assertRaises(TypeError):
                store.diff_at("main", "feature", bad)  # type: ignore[arg-type]

    def test_negative_at_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            make_store().diff_at("main", "feature", -1)

    def test_arguments_validated_in_signature_order(self) -> None:
        store = make_store()
        with self.assertRaises(TypeError):
            store.diff_at(None, "nope", True)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            store.diff_at("", "nope", -1)
        with self.assertRaises(TypeError):
            store.diff_at("main", None, True)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            store.diff_at("main", "", -1)
        with self.assertRaises(TypeError):
            store.diff_at("nope-a", "nope-b", True)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            store.diff_at("nope-a", "nope-b", -1)

    def test_unknown_branches_raise_key_error_after_validation(self) -> None:
        store = make_store()
        with self.assertRaises(KeyError):
            store.diff_at("nope", "feature", 1)
        with self.assertRaises(KeyError):
            store.diff_at("main", "nope", 1)
        try:
            store.diff_at("nope-a", "nope-b", 1)
        except KeyError as error:
            self.assertEqual(error.args, ("nope-a",))
        else:
            self.fail("expected KeyError")

    def test_query_leaves_store_unchanged_and_is_detached(self) -> None:
        store = make_store()
        heads_before = (store.head("main"), store.head("feature"))
        replay_before = store.replay("main")
        diff = store.diff_at("main", "feature", 4)
        diff["a"] = (9, 9)
        self.assertEqual((store.head("main"), store.head("feature")), heads_before)
        self.assertEqual(store.replay("main"), replay_before)
        self.assertEqual(store.audit_merge("merge1")["at"], 4)
        self.assertEqual(
            store.diff_at("main", "feature", 4)["a"], (1, 1)
        )


if __name__ == "__main__":
    unittest.main()
