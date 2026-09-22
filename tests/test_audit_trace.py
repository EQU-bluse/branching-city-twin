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


class AuditMergeTests(unittest.TestCase):
    def test_returns_record_with_ordered_keys(self) -> None:
        record = make_store().audit_merge("merge1")
        self.assertEqual(
            list(record),
            ["event_id", "target", "source", "parents", "at", "changes"],
        )
        self.assertEqual(record["event_id"], "merge1")
        self.assertEqual(record["target"], "main")
        self.assertEqual(record["source"], "feature")
        self.assertEqual(record["parents"], ("m1", "f2"))
        self.assertEqual(record["at"], 4)
        self.assertEqual(record["changes"], {"m": 5})

    def test_result_is_detached_from_internal_records(self) -> None:
        store = make_store()
        record = store.audit_merge("merge1")
        record["changes"]["m"] = 999
        record["changes"]["evil"] = 1
        record["target"] = "other"
        fresh = store.audit_merge("merge1")
        self.assertEqual(fresh["changes"], {"m": 5})
        self.assertEqual(fresh["target"], "main")
        self.assertIsNot(record, fresh)
        self.assertIsNot(record["changes"], fresh["changes"])

    def test_non_str_id_raises_type_error(self) -> None:
        store = make_store()
        for bad in (None, 1, 1.5, b"merge1", ["merge1"]):
            with self.assertRaises(TypeError):
                store.audit_merge(bad)  # type: ignore[arg-type]

    def test_empty_id_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            make_store().audit_merge("")

    def test_unknown_id_raises_key_error(self) -> None:
        with self.assertRaises(KeyError):
            make_store().audit_merge("nope")

    def test_non_merge_event_id_raises_key_error(self) -> None:
        store = make_store()
        for event_id in ("root", "m1", "f1", "f2"):
            with self.assertRaises(KeyError):
                store.audit_merge(event_id)

    def test_failures_leave_state_unchanged(self) -> None:
        store = make_store()
        before = store.replay("main")
        for bad_call in (
            lambda: store.audit_merge(""),
            lambda: store.audit_merge("nope"),
            lambda: store.audit_merge("m1"),
        ):
            with self.assertRaises((KeyError, ValueError)):
                bad_call()
        self.assertEqual(store.replay("main"), before)
        self.assertEqual(store.head("main"), "merge1")


class TraceTests(unittest.TestCase):
    def test_traces_key_in_replay_order(self) -> None:
        store = make_store()
        store.append("main", "m2", 5, {"c": 10})
        self.assertEqual(store.trace("main", "c"), ("f1", "f2", "m2"))

    def test_includes_zero_delta_and_merge_events(self) -> None:
        store = make_store()
        # f2 carries a zero delta for "c"; merge1 touches "m".
        self.assertIn("f2", store.trace("main", "c"))
        self.assertEqual(store.trace("main", "m"), ("merge1",))

    def test_untouched_key_returns_empty_tuple(self) -> None:
        self.assertEqual(make_store().trace("main", "zzz"), ())

    def test_scoped_to_branch_head_closure(self) -> None:
        store = make_store()
        # "b" is only appended on main, not on feature.
        self.assertEqual(store.trace("feature", "b"), ())
        self.assertEqual(store.trace("main", "b"), ("m1",))

    def test_no_duplicate_ids(self) -> None:
        store = make_store()
        traced = store.trace("main", "c")
        self.assertEqual(len(traced), len(set(traced)))

    def test_invalid_name_and_key_raise(self) -> None:
        store = make_store()
        for bad in (None, 1, b"main"):
            with self.assertRaises(TypeError):
                store.trace(bad, "c")  # type: ignore[arg-type]
            with self.assertRaises(TypeError):
                store.trace("main", bad)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            store.trace("", "c")
        with self.assertRaises(ValueError):
            store.trace("main", "")

    def test_unknown_branch_raises_key_error(self) -> None:
        with self.assertRaises(KeyError):
            make_store().trace("ghost", "c")

    def test_failures_leave_state_unchanged(self) -> None:
        store = make_store()
        before = store.replay("main")
        for bad_call in (
            lambda: store.trace("", "c"),
            lambda: store.trace("main", ""),
            lambda: store.trace("ghost", "c"),
        ):
            with self.assertRaises((KeyError, ValueError)):
                bad_call()
        self.assertEqual(store.replay("main"), before)
        self.assertEqual(store.head("main"), "merge1")


if __name__ == "__main__":
    unittest.main()
