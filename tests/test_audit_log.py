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


class AuditLogTests(unittest.TestCase):
    def test_empty_store_exports_nothing(self) -> None:
        graph = EventGraph()
        graph.add("root", 0, (), {"a": 1})
        store = BranchStore(graph)
        store.create("main", "root")
        self.assertEqual(store.audit_log(), ())

    def test_excludes_events_added_directly_to_graph(self) -> None:
        store = make_store()
        ids = [record["event_id"] for record in store.audit_log()]
        self.assertNotIn("root", ids)
        self.assertEqual(sorted(ids), ["f1", "f2", "m1", "merge1"])

    def test_sorted_by_at_then_event_id(self) -> None:
        graph = EventGraph()
        graph.add("root", 0, (), {})
        store = BranchStore(graph)
        store.create("main", "root")
        store.append("main", "b", 5, {"x": 1})
        store.append("main", "a", 5, {"x": 1})
        store.append("main", "c", 3, {"x": 1})
        ids = [record["event_id"] for record in store.audit_log()]
        self.assertEqual(ids, ["c", "a", "b"])

    def test_record_shape_for_append_and_merge(self) -> None:
        records = {r["event_id"]: r for r in make_store().audit_log()}
        for record in records.values():
            self.assertEqual(
                list(record),
                ["event_id", "kind", "owner", "peer", "parents", "at", "changes"],
            )
        append = records["m1"]
        self.assertEqual(append["kind"], "append")
        self.assertEqual(append["owner"], "main")
        self.assertIsNone(append["peer"])
        self.assertEqual(append["parents"], ("root",))
        self.assertEqual(append["at"], 1)
        self.assertEqual(append["changes"], {"b": 2})

        merge = records["merge1"]
        self.assertEqual(merge["kind"], "merge")
        self.assertEqual(merge["owner"], "main")
        self.assertEqual(merge["peer"], "feature")
        self.assertEqual(merge["parents"], ("m1", "f2"))
        self.assertEqual(merge["at"], 4)
        self.assertEqual(merge["changes"], {"m": 5})

    def test_changes_keys_sorted_by_code_point(self) -> None:
        graph = EventGraph()
        graph.add("root", 0, (), {})
        store = BranchStore(graph)
        store.create("main", "root")
        store.append("main", "e1", 1, {"z": 1, "a": 2, "É": 3, "m": 4})
        (record,) = store.audit_log()
        self.assertEqual(list(record["changes"]), ["a", "m", "z", "É"])

    def test_each_event_appears_once(self) -> None:
        store = make_store()
        # Idempotent repeats must not duplicate audit entries.
        store.append("main", "m1", 1, {"b": 2})
        store.merge("main", "feature", "merge1", 4, {"m": 5})
        ids = [record["event_id"] for record in store.audit_log()]
        self.assertEqual(len(ids), len(set(ids)))

    def test_result_is_detached_from_internal_records(self) -> None:
        store = make_store()
        first = store.audit_log()
        for record in first:
            record["changes"]["evil"] = 1
            record["owner"] = "other"
        second = store.audit_log()
        self.assertNotEqual(first, second)
        self.assertEqual(store.replay("main"), {"a": 1, "b": 2, "c": 3, "m": 5})
        self.assertEqual(store.head("main"), "merge1")
        # Idempotency records are intact: repeats are still no-ops.
        store.append("main", "m1", 1, {"b": 2})
        store.merge("main", "feature", "merge1", 4, {"m": 5})
        self.assertEqual(store.audit_log(), second)

    def test_repeated_calls_are_equal_and_fresh(self) -> None:
        store = make_store()
        first = store.audit_log()
        second = store.audit_log()
        self.assertEqual(first, second)
        self.assertIsNot(first, second)
        for left, right in zip(first, second):
            self.assertIsNot(left, right)
            self.assertIsNot(left["changes"], right["changes"])

    def test_read_only_leaves_behavior_unchanged(self) -> None:
        store = make_store()
        store.audit_log()
        self.assertEqual(store.head("main"), "merge1")
        self.assertEqual(store.head("feature"), "f2")
        self.assertEqual(store.trace("main", "c"), ("f1", "f2"))
        self.assertEqual(
            store.diff_at("main", "feature", 3),
            {"a": (1, 1), "b": (2, 0), "c": (3, 3)},
        )


if __name__ == "__main__":
    unittest.main()
