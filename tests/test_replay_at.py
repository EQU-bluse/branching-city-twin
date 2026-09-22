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


class EventGraphReplayAtTests(unittest.TestCase):
    def test_replays_only_events_up_to_at(self) -> None:
        graph = make_graph()
        self.assertEqual(graph.replay_at("e4", 0), {"a": 1})
        self.assertEqual(graph.replay_at("e4", 2), {"a": 4, "b": 2})
        self.assertEqual(graph.replay_at("e4", 5), {"a": 3, "b": 0})
        self.assertEqual(graph.replay_at("e4", 8), {"a": 3, "b": 0})

    def test_boundary_time_includes_equal_events(self) -> None:
        graph = make_graph()
        self.assertEqual(graph.replay_at("e4", 9), graph.replay("e4"))
        self.assertEqual(graph.replay_at("e4", 100), graph.replay("e4"))

    def test_keeps_zero_value_keys(self) -> None:
        graph = make_graph()
        state = graph.replay_at("e4", 5)
        self.assertIn("b", state)
        self.assertEqual(state["b"], 0)

    def test_result_keys_sorted_by_code_point(self) -> None:
        graph = EventGraph()
        graph.add("r", 0, (), {"b": 1, "A": 2, "a": 3})
        self.assertEqual(list(graph.replay_at("r", 0)), ["A", "a", "b"])

    def test_only_ancestor_closure_is_considered(self) -> None:
        graph = make_graph()
        graph.add("other", 1, (), {"z": 9})
        self.assertEqual(graph.replay_at("e1", 10), {"a": 1, "b": 2})

    def test_non_str_head_raises_type_error(self) -> None:
        graph = make_graph()
        for bad in (None, 1, 1.5, b"e4", ["e4"]):
            with self.assertRaises(TypeError):
                graph.replay_at(bad, 5)  # type: ignore[arg-type]

    def test_empty_head_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            make_graph().replay_at("", 5)

    def test_unknown_head_raises_key_error(self) -> None:
        with self.assertRaises(KeyError):
            make_graph().replay_at("nope", 5)

    def test_non_int_at_raises_type_error(self) -> None:
        graph = make_graph()
        for bad in (None, "5", 1.5, True, False):
            with self.assertRaises(TypeError):
                graph.replay_at("e4", bad)  # type: ignore[arg-type]

    def test_negative_at_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            make_graph().replay_at("e4", -1)

    def test_head_validated_before_at(self) -> None:
        graph = make_graph()
        with self.assertRaises(TypeError):
            graph.replay_at(None, -1)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            graph.replay_at("", -1)

    def test_failure_leaves_graph_unchanged(self) -> None:
        graph = make_graph()
        before = graph.replay("e4")
        for call in (
            lambda: graph.replay_at(None, 1),  # type: ignore[arg-type]
            lambda: graph.replay_at("", 1),
            lambda: graph.replay_at("nope", 1),
            lambda: graph.replay_at("e4", True),
            lambda: graph.replay_at("e4", -1),
        ):
            with self.assertRaises((TypeError, ValueError, KeyError)):
                call()
        self.assertEqual(graph.replay("e4"), before)
        graph.add("e4", 9, ("e3",), {"c": 7})  # idempotent re-add still works

    def test_future_events_do_not_affect_result(self) -> None:
        graph = make_graph()
        snapshot = graph.replay_at("e4", 5)
        graph.add("late", 50, ("e4",), {"a": 100})
        self.assertEqual(graph.replay_at("e4", 5), snapshot)
        self.assertEqual(graph.replay_at("late", 5), snapshot)

    def test_result_is_detached(self) -> None:
        graph = make_graph()
        state = graph.replay_at("e4", 9)
        state["a"] = 999
        self.assertEqual(graph.replay_at("e4", 9)["a"], 3)


class BranchStoreReplayAtTests(unittest.TestCase):
    def test_replays_branch_head_as_of_at(self) -> None:
        store = make_store()
        self.assertEqual(store.replay_at("main", 0), {"a": 1})
        self.assertEqual(store.replay_at("main", 1), {"a": 1, "b": 2})
        self.assertEqual(
            store.replay_at("main", 4),
            {"a": 1, "b": 2, "c": 3, "m": 5},
        )
        self.assertEqual(store.replay_at("main", 4), store.replay("main"))

    def test_tracks_current_head(self) -> None:
        store = make_store()
        self.assertEqual(store.replay_at("feature", 2), {"a": 1, "c": 3})
        self.assertEqual(store.replay_at("feature", 3), {"a": 1, "c": 3})

    def test_non_str_name_raises_type_error(self) -> None:
        store = make_store()
        for bad in (None, 1, 1.5, b"main", ["main"]):
            with self.assertRaises(TypeError):
                store.replay_at(bad, 1)  # type: ignore[arg-type]

    def test_empty_name_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            make_store().replay_at("", 1)

    def test_non_int_at_raises_type_error(self) -> None:
        store = make_store()
        for bad in (None, "1", 1.5, True, False):
            with self.assertRaises(TypeError):
                store.replay_at("main", bad)  # type: ignore[arg-type]

    def test_negative_at_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            make_store().replay_at("main", -1)

    def test_at_validated_before_branch_lookup(self) -> None:
        store = make_store()
        with self.assertRaises(TypeError):
            store.replay_at("nope", True)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            store.replay_at("nope", -1)

    def test_unknown_branch_raises_key_error(self) -> None:
        with self.assertRaises(KeyError):
            make_store().replay_at("nope", 1)

    def test_failure_leaves_store_unchanged(self) -> None:
        store = make_store()
        heads_before = (store.head("main"), store.head("feature"))
        replay_before = store.replay("main")
        for call in (
            lambda: store.replay_at(None, 1),  # type: ignore[arg-type]
            lambda: store.replay_at("", 1),
            lambda: store.replay_at("main", True),
            lambda: store.replay_at("main", -1),
            lambda: store.replay_at("nope", 1),
        ):
            with self.assertRaises((TypeError, ValueError, KeyError)):
                call()
        self.assertEqual((store.head("main"), store.head("feature")), heads_before)
        self.assertEqual(store.replay("main"), replay_before)
        self.assertEqual(store.audit_merge("merge1")["at"], 4)


if __name__ == "__main__":
    unittest.main()
