import hashlib
import json
import os
import tempfile
import threading
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


class CheckpointCase(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = os.path.join(self.dir.name, "checkpoint.json")

    def save(self, store: BranchStore, path: str | None = None) -> bytes:
        store.save_checkpoint(path or self.path)
        with open(path or self.path, "rb") as handle:
            return handle.read()

    def write(self, raw: bytes, path: str | None = None) -> str:
        target = path or self.path
        with open(target, "wb") as handle:
            handle.write(raw)
        return target

    def document_of(self, store: BranchStore) -> dict:
        return json.loads(self.save(store))

    def redump(self, document: dict) -> bytes:
        """Re-serialize a (possibly tampered) document canonically."""
        payload = document["payload"]
        payload_bytes = json.dumps(
            payload, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        document["checksum"] = hashlib.sha256(payload_bytes).hexdigest()
        return json.dumps(
            document, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")

    def tampered(self, store: BranchStore, mutate) -> bytes:
        """Save, apply ``mutate`` to the payload, re-checksum, re-serialize."""
        document = self.document_of(store)
        mutate(document["payload"])
        return self.redump(document)


class SaveFormatTests(CheckpointCase):
    def test_returns_none(self) -> None:
        self.assertIsNone(make_store().save_checkpoint(self.path))

    def test_top_level_key_order_and_version(self) -> None:
        document = self.document_of(make_store())
        self.assertEqual(
            list(document), ["schema_version", "payload", "checksum"]
        )
        self.assertEqual(document["schema_version"], 1)

    def test_payload_key_order(self) -> None:
        document = self.document_of(make_store())
        self.assertEqual(
            list(document["payload"]),
            ["events", "branches", "appends", "merges"],
        )

    def test_compact_utf8_without_bom_or_trailing_newline(self) -> None:
        raw = self.save(make_store())
        self.assertFalse(raw.startswith(b"\xef\xbb\xbf"))
        self.assertFalse(raw.endswith(b"\n"))
        self.assertNotIn(b": ", raw)
        self.assertNotIn(b", ", raw)
        raw.decode("utf-8")

    def test_non_ascii_identifiers_stay_unescaped(self) -> None:
        graph = EventGraph()
        graph.add("røøt", 0, (), {})
        store = BranchStore(graph)
        store.create("main", "røøt")
        raw = self.save(store)
        self.assertIn("røøt".encode("utf-8"), raw)
        self.assertEqual(
            BranchStore.load_checkpoint(self.path).audit_log(), ()
        )

    def test_checksum_matches_payload_canonical_bytes(self) -> None:
        document = self.document_of(make_store())
        payload_bytes = json.dumps(
            document["payload"], separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        self.assertEqual(
            document["checksum"],
            hashlib.sha256(payload_bytes).hexdigest(),
        )

    def test_identifiers_and_change_keys_sorted(self) -> None:
        graph = EventGraph()
        graph.add("root", 0, (), {"z": 1, "a": 1})
        store = BranchStore(graph)
        store.create("b-branch", "root")
        store.create("a-branch", "root")
        store.append("a-branch", "e2", 1, {"y": 1, "b": 2})
        store.append("a-branch", "e1", 2, {})
        document = self.document_of(store)
        payload = document["payload"]
        self.assertEqual(
            [event["id"] for event in payload["events"]],
            ["e1", "e2", "root"],
        )
        self.assertEqual(
            [branch["name"] for branch in payload["branches"]],
            ["a-branch", "b-branch"],
        )
        self.assertEqual(
            [record["event_id"] for record in payload["appends"]],
            ["e1", "e2"],
        )
        self.assertEqual(list(payload["events"][0]["changes"]), [])
        e2 = payload["events"][1]
        self.assertEqual(list(e2["changes"]), ["b", "y"])

    def test_parent_order_preserved(self) -> None:
        document = self.document_of(make_store())
        events = {e["id"]: e for e in document["payload"]["events"]}
        self.assertEqual(events["merge1"]["parents"], ["m1", "f2"])

    def test_same_state_produces_byte_identical_files(self) -> None:
        first = self.save(make_store())
        second_path = os.path.join(self.dir.name, "second.json")
        second = self.save(make_store(), second_path)
        self.assertEqual(first, second)

    def test_save_does_not_change_business_state(self) -> None:
        store = make_store()
        audit_before = store.audit_log()
        graph_before = store._graph.to_json()
        store.save_checkpoint(self.path)
        self.assertEqual(store.audit_log(), audit_before)
        self.assertEqual(store._graph.to_json(), graph_before)

    def test_empty_store_round_trip(self) -> None:
        store = BranchStore(EventGraph())
        self.save(store)
        loaded = BranchStore.load_checkpoint(self.path)
        self.assertEqual(loaded.audit_log(), ())


class SavePathAndAtomicityTests(CheckpointCase):
    def test_non_str_path_raises_type_error(self) -> None:
        store = make_store()
        for bad in (None, 1, 1.5, b"path", ["x"]):
            with self.assertRaises(TypeError):
                store.save_checkpoint(bad)
            with self.assertRaises(TypeError):
                BranchStore.load_checkpoint(bad)

    def test_empty_path_raises_value_error(self) -> None:
        store = make_store()
        with self.assertRaises(ValueError):
            store.save_checkpoint("")
        with self.assertRaises(ValueError):
            BranchStore.load_checkpoint("")

    def test_missing_directory_raises_os_error(self) -> None:
        store = make_store()
        missing = os.path.join(self.dir.name, "nope", "cp.json")
        with self.assertRaises(OSError):
            store.save_checkpoint(missing)

    def test_load_missing_file_raises_os_error(self) -> None:
        with self.assertRaises(OSError):
            BranchStore.load_checkpoint(
                os.path.join(self.dir.name, "missing.json")
            )

    def test_failed_save_keeps_old_target_and_cleans_temp(self) -> None:
        store = make_store()
        original = self.save(store)
        # A directory at the target path makes the atomic replace fail.
        blocker = os.path.join(self.dir.name, "blocked")
        os.mkdir(blocker)
        with self.assertRaises(OSError):
            store.save_checkpoint(blocker)
        self.assertTrue(os.path.isdir(blocker))
        # No temporary files leaked into the directory.
        self.assertEqual(
            sorted(os.listdir(self.dir.name)),
            ["blocked", "checkpoint.json"],
        )
        # The earlier good checkpoint is still intact and loadable.
        with open(self.path, "rb") as handle:
            self.assertEqual(handle.read(), original)
        BranchStore.load_checkpoint(self.path)

    def test_temp_file_cleaned_up_on_success(self) -> None:
        self.save(make_store())
        self.assertEqual(os.listdir(self.dir.name), ["checkpoint.json"])

    def test_concurrent_mutations_keep_checkpoints_consistent(self) -> None:
        graph = EventGraph()
        graph.add("root", 0, (), {})
        store = BranchStore(graph)
        store.create("main", "root")
        errors = []
        paths = []

        def writer() -> None:
            for index in range(50):
                path = os.path.join(self.dir.name, f"cp-{index}.json")
                try:
                    store.save_checkpoint(path)
                    paths.append(path)
                except OSError as exc:  # pragma: no cover - unexpected
                    errors.append(exc)

        def appender() -> None:
            for index in range(50):
                store.append("main", f"e{index}", index + 1, {"x": 1})

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=appender),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(errors, [])
        self.assertTrue(paths)
        for path in paths:
            loaded = BranchStore.load_checkpoint(path)
            # Graph and records agree: every recorded append resolves.
            for record in loaded.audit_log():
                self.assertIn(record["event_id"], loaded._graph._at)


class LoadValidationTests(CheckpointCase):
    def test_invalid_utf8_raises_value_error(self) -> None:
        self.write(b'{"schema_version": 1, "payload": "\xff"}')
        with self.assertRaises(ValueError):
            BranchStore.load_checkpoint(self.path)

    def test_invalid_json_raises_value_error(self) -> None:
        self.write(b"not json")
        with self.assertRaises(ValueError):
            BranchStore.load_checkpoint(self.path)

    def test_bom_raises_value_error(self) -> None:
        self.write(b"\xef\xbb\xbf" + self.save(make_store()))
        with self.assertRaises(ValueError):
            BranchStore.load_checkpoint(self.path)

    def test_checksum_mismatch_raises_value_error(self) -> None:
        document = self.document_of(make_store())
        document["checksum"] = "0" * 64
        self.write(
            json.dumps(document, separators=(",", ":")).encode("utf-8")
        )
        with self.assertRaises(ValueError):
            BranchStore.load_checkpoint(self.path)

    def test_tampered_payload_detected(self) -> None:
        raw = bytearray(self.save(make_store()))
        # Flip one payload byte without updating the checksum.
        index = raw.index(b'"at":1')
        raw[index + 5] = ord("9")
        self.write(bytes(raw))
        with self.assertRaises(ValueError):
            BranchStore.load_checkpoint(self.path)

    def test_unknown_schema_version_raises_value_error(self) -> None:
        store = make_store()
        for bad in (0, 2, "1", 1.0, True, None):
            document = self.document_of(store)
            document["schema_version"] = bad
            self.write(self.redump(document))
            with self.assertRaises(ValueError, msg=f"version {bad!r}"):
                BranchStore.load_checkpoint(self.path)

    def test_top_level_structure_violations(self) -> None:
        toplevel = self.document_of(make_store())
        missing_checksum = {
            "schema_version": 1,
            "payload": toplevel["payload"],
        }
        variants = [
            json.dumps(missing_checksum).encode("utf-8"),
            self.redump(dict(toplevel, extra=1)),
            json.dumps([]).encode("utf-8"),
            json.dumps("text").encode("utf-8"),
            json.dumps(42).encode("utf-8"),
        ]
        for raw in variants:
            self.write(raw)
            with self.assertRaises(ValueError, msg=repr(raw[:60])):
                BranchStore.load_checkpoint(self.path)

    def test_payload_structure_violations(self) -> None:
        store = make_store()

        def drop_key(payload: dict) -> None:
            del payload["merges"]

        def add_key(payload: dict) -> None:
            payload["extra"] = []

        def wrong_type(payload: dict) -> None:
            payload["events"] = {}

        for mutate in (drop_key, add_key, wrong_type):
            self.write(self.tampered(store, mutate))
            with self.assertRaises(ValueError):
                BranchStore.load_checkpoint(self.path)

    def test_event_violations(self) -> None:
        store = make_store()

        cases = []

        def missing_field(payload):
            del payload["events"][0]["at"]

        def extra_field(payload):
            payload["events"][0]["note"] = 1

        def negative_at(payload):
            payload["events"][0]["at"] = -1

        def bool_at(payload):
            payload["events"][0]["at"] = True

        def empty_id(payload):
            payload["events"][0]["id"] = ""

        def duplicate_event(payload):
            payload["events"].append(dict(payload["events"][0]))

        def unknown_parent(payload):
            payload["events"][0]["parents"] = ["ghost"]

        def duplicate_parent(payload):
            for event in payload["events"]:
                if event["id"] == "merge1":
                    event["parents"] = ["m1", "m1"]

        def bool_change(payload):
            payload["events"][0]["changes"] = {"x": False}

        cases = [
            missing_field,
            extra_field,
            negative_at,
            bool_at,
            empty_id,
            duplicate_event,
            unknown_parent,
            duplicate_parent,
            bool_change,
        ]
        for mutate in cases:
            self.write(self.tampered(store, mutate))
            with self.assertRaises(
                ValueError, msg=getattr(mutate, "__name__", "?")
            ):
                BranchStore.load_checkpoint(self.path)

    def test_cycle_detected(self) -> None:
        graph = EventGraph()
        graph.add("a", 0, (), {})
        store = BranchStore(graph)
        store.create("main", "a")

        def make_cycle(payload):
            payload["events"].append(
                {"id": "b", "at": 1, "parents": ["c"], "changes": {}}
            )
            payload["events"].append(
                {"id": "c", "at": 2, "parents": ["b"], "changes": {}}
            )

        self.write(self.tampered(store, make_cycle))
        with self.assertRaises(ValueError):
            BranchStore.load_checkpoint(self.path)

    def test_branch_violations(self) -> None:
        store = make_store()

        def missing_field(payload):
            del payload["branches"][0]["head"]

        def extra_field(payload):
            payload["branches"][0]["tag"] = "x"

        def empty_name(payload):
            payload["branches"][0]["name"] = ""

        def duplicate_name(payload):
            payload["branches"].append(dict(payload["branches"][0]))

        def unknown_head(payload):
            payload["branches"][0]["head"] = "ghost"

        def unknown_source(payload):
            payload["branches"][0]["source"] = "ghost"

        def source_not_ancestor(payload):
            for branch in payload["branches"]:
                if branch["name"] == "main":
                    branch["source"] = "f1"

        for mutate in (
            missing_field,
            extra_field,
            empty_name,
            duplicate_name,
            unknown_head,
            unknown_source,
            source_not_ancestor,
        ):
            self.write(self.tampered(store, mutate))
            with self.assertRaises(
                ValueError, msg=getattr(mutate, "__name__", "?")
            ):
                BranchStore.load_checkpoint(self.path)

    def test_append_record_violations(self) -> None:
        store = make_store()

        def missing_field(payload):
            del payload["appends"][0]["at"]

        def extra_field(payload):
            payload["appends"][0]["peer"] = None

        def unknown_event(payload):
            payload["appends"][0]["event_id"] = "ghost"

        def unknown_branch(payload):
            payload["appends"][0]["branch"] = "ghost"

        def duplicate_record(payload):
            payload["appends"].append(dict(payload["appends"][0]))

        def at_mismatch(payload):
            payload["appends"][0]["at"] += 100

        def changes_mismatch(payload):
            payload["appends"][0]["changes"] = {"other": 1}

        def merge_event_as_append(payload):
            payload["appends"].append(
                {
                    "event_id": "merge1",
                    "branch": "main",
                    "at": 4,
                    "changes": {"m": 5},
                }
            )

        def off_chain_parent(payload):
            # f1's recorded parent chain no longer reaches feature's head.
            payload["appends"].append(
                {
                    "event_id": "f1",
                    "branch": "main",
                    "at": 2,
                    "changes": {"c": 3},
                }
            )

        for mutate in (
            missing_field,
            extra_field,
            unknown_event,
            unknown_branch,
            duplicate_record,
            at_mismatch,
            changes_mismatch,
            merge_event_as_append,
            off_chain_parent,
        ):
            self.write(self.tampered(store, mutate))
            with self.assertRaises(
                ValueError, msg=getattr(mutate, "__name__", "?")
            ):
                BranchStore.load_checkpoint(self.path)

    def test_merge_record_violations(self) -> None:
        store = make_store()

        def missing_field(payload):
            del payload["merges"][0]["target"]

        def extra_field(payload):
            payload["merges"][0]["kind"] = "merge"

        def unknown_event(payload):
            payload["merges"][0]["event_id"] = "ghost"

        def unknown_target(payload):
            payload["merges"][0]["target"] = "ghost"

        def same_target_source(payload):
            payload["merges"][0]["source"] = payload["merges"][0]["target"]

        def duplicate_record(payload):
            payload["merges"].append(dict(payload["merges"][0]))

        def at_mismatch(payload):
            payload["merges"][0]["at"] += 100

        def changes_mismatch(payload):
            payload["merges"][0]["changes"] = {"m": 6}

        def append_event_as_merge(payload):
            payload["merges"].append(
                {
                    "event_id": "m1",
                    "target": "main",
                    "source": "feature",
                    "at": 1,
                    "changes": {"b": 2},
                }
            )

        def swapped_parents(payload):
            # Parent order no longer matches (target, source).
            payload["merges"][0]["target"] = "feature"
            payload["merges"][0]["source"] = "main"

        for mutate in (
            missing_field,
            extra_field,
            unknown_event,
            unknown_target,
            same_target_source,
            duplicate_record,
            at_mismatch,
            changes_mismatch,
            append_event_as_merge,
            swapped_parents,
        ):
            self.write(self.tampered(store, mutate))
            with self.assertRaises(
                ValueError, msg=getattr(mutate, "__name__", "?")
            ):
                BranchStore.load_checkpoint(self.path)

    def test_failed_load_does_not_rewrite_file(self) -> None:
        document = self.document_of(make_store())
        document["checksum"] = "0" * 64
        raw = json.dumps(
            document, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        self.write(raw)
        with self.assertRaises(ValueError):
            BranchStore.load_checkpoint(self.path)
        with open(self.path, "rb") as handle:
            self.assertEqual(handle.read(), raw)


class RoundTripTests(CheckpointCase):
    def test_replay_audit_and_trace_match(self) -> None:
        store = make_store()
        self.save(store)
        loaded = BranchStore.load_checkpoint(self.path)
        self.assertEqual(loaded.audit_log(), store.audit_log())
        self.assertEqual(
            loaded.audit_merge("merge1"), store.audit_merge("merge1")
        )
        for name in ("main", "feature"):
            for key in ("a", "b", "c", "m"):
                self.assertEqual(
                    loaded.trace(name, key), store.trace(name, key)
                )
        self.assertEqual(
            loaded.compare("main", "feature"),
            store.compare("main", "feature"),
        )
        self.assertEqual(
            loaded.diff_at("main", "feature", 3),
            store.diff_at("main", "feature", 3),
        )

    def test_idempotency_and_conflicts_preserved(self) -> None:
        store = make_store()
        self.save(store)
        loaded = BranchStore.load_checkpoint(self.path)
        # Exact repeats are no-ops.
        self.assertIsNone(loaded.create("main", "root"))
        self.assertIsNone(loaded.append("main", "m1", 1, {"b": 2}))
        self.assertIsNone(
            loaded.merge("main", "feature", "merge1", 4, {"m": 5})
        )
        # Diverging repeats conflict exactly as on the original store.
        with self.assertRaises(ValueError):
            loaded.create("main", "f1")
        with self.assertRaises(ValueError):
            loaded.append("main", "m1", 1, {"b": 3})
        with self.assertRaises(ValueError):
            loaded.merge("main", "feature", "merge1", 4, {"m": 6})
        with self.assertRaises(ValueError):
            store.append("main", "m1", 1, {"b": 3})

    def test_loaded_store_continues_growing(self) -> None:
        store = make_store()
        self.save(store)
        loaded = BranchStore.load_checkpoint(self.path)
        loaded.append("main", "m2", 5, {"n": 1})
        loaded.create("hotfix", "f2")
        loaded.append("hotfix", "h1", 6, {"h": 1})
        self.assertEqual(loaded._graph.replay("m2"), {"a": 1, "b": 2, "c": 3, "m": 5, "n": 1})
        # The original store is untouched by the loaded store's writes.
        self.assertNotIn("m2", store._graph._at)
        self.assertNotIn("hotfix", store._heads)

    def test_snapshot_tokens_not_persisted(self) -> None:
        store = make_store()
        token = store.create_snapshot(3)
        self.save(store)
        loaded = BranchStore.load_checkpoint(self.path)
        self.assertEqual(loaded._snapshots, {})
        with self.assertRaises(KeyError):
            loaded.release_snapshot(token)
        # The original store's token is unaffected by the save.
        store.release_snapshot(token)

    def test_loaded_store_shares_no_mutable_state(self) -> None:
        store = make_store()
        self.save(store)
        loaded = BranchStore.load_checkpoint(self.path)
        loaded._graph._changes["m1"]["b"] = 99
        loaded._appends["main"]["m1"][1]["b"] = 99
        self.assertEqual(store._graph._changes["m1"]["b"], 2)
        self.assertEqual(store._appends["main"]["m1"][1]["b"], 2)

    def test_double_round_trip_is_stable(self) -> None:
        store = make_store()
        first = self.save(store)
        loaded = BranchStore.load_checkpoint(self.path)
        second_path = os.path.join(self.dir.name, "second.json")
        second = self.save(loaded, second_path)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
