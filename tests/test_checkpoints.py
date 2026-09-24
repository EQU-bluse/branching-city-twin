import hashlib
import json
import os
import tempfile
import threading
import unittest
from unittest import mock

from city_twin.branches import BranchStore
from city_twin.event_graph import EventGraph


def make_store() -> BranchStore:
    graph = EventGraph()
    graph.add("root", 0, (), {"seed": 7})
    store = BranchStore(graph)
    store.create("main", "root")
    store.create("feature", "root")
    store.append("main", "m1", 1, {"a": 1})
    store.append("feature", "f1", 2, {"b": 2})
    store.append("feature", "f2", 3, {"b": 0})
    store.merge("main", "feature", "M", 4, {"c": 3})
    store.append("main", "m2", 5, {"a": 2})
    return store


class CheckpointRoundTripTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "cp.json")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_save_returns_none_load_returns_store(self) -> None:
        store = make_store()
        self.assertIsNone(store.save_checkpoint(self.path))
        restored = BranchStore.load_checkpoint(self.path)
        self.assertIsInstance(restored, BranchStore)
        self.assertIsNot(restored, store)

    def test_replays_match(self) -> None:
        store = make_store()
        store.save_checkpoint(self.path)
        restored = BranchStore.load_checkpoint(self.path)
        for name in ("main", "feature"):
            self.assertEqual(restored.replay(name), store.replay(name))
            self.assertEqual(
                restored.replay_at(name, 3), store.replay_at(name, 3)
            )
            self.assertEqual(restored.head(name), store.head(name))
        self.assertEqual(
            restored.diff_at("main", "feature", 5),
            store.diff_at("main", "feature", 5),
        )

    def test_audit_log_matches(self) -> None:
        store = make_store()
        store.save_checkpoint(self.path)
        restored = BranchStore.load_checkpoint(self.path)
        self.assertEqual(restored.audit_log(), store.audit_log())
        self.assertEqual(
            restored.audit_merge("M"), store.audit_merge("M")
        )

    def test_restored_idempotency_matches(self) -> None:
        store = make_store()
        store.save_checkpoint(self.path)
        restored = BranchStore.load_checkpoint(self.path)
        # Repeating recorded append/merge calls stays a no-op.
        restored.append("main", "m1", 1, {"a": 1})
        restored.append("feature", "f2", 3, {"b": 0})
        restored.merge("main", "feature", "M", 4, {"c": 3})
        self.assertEqual(restored.audit_log(), store.audit_log())
        # A recorded id with different inputs is still a conflict.
        with self.assertRaises(ValueError):
            restored.append("main", "m1", 1, {"a": 9})
        with self.assertRaises(ValueError):
            restored.merge("main", "feature", "M", 4, {"c": 9})
        # New writes work and merge conflict detection survives restore.
        restored.append("feature", "f3", 6, {"z": 1})
        restored.append("main", "m3", 6, {"z": 2})
        with self.assertRaises(ValueError):
            restored.merge("main", "feature", "M2", 7, {})

    def test_store_with_only_graph_round_trips(self) -> None:
        graph = EventGraph()
        graph.add("root", 0, (), {})
        store = BranchStore(graph)
        store.save_checkpoint(self.path)
        restored = BranchStore.load_checkpoint(self.path)
        self.assertEqual(restored.audit_log(), ())
        with self.assertRaises(KeyError):
            restored.head("main")

    def test_empty_changes_and_unicode_keys_sorted(self) -> None:
        graph = EventGraph()
        graph.add("root", 0, (), {})
        store = BranchStore(graph)
        store.create("支", "root")
        store.append("支", "é", 1, {"z": 1, "é": 2, "a": 3})
        store.append("支", "a", 2, {})
        store.save_checkpoint(self.path)
        raw = _read(self.path)
        # Non-ASCII stays literal, no \u escapes.
        self.assertIn("支".encode("utf-8"), raw)
        self.assertIn("é".encode("utf-8"), raw)
        doc = json.loads(raw)
        events = doc["payload"]["events"]
        ids = [e["id"] for e in events]
        self.assertEqual(ids, sorted(ids))
        changes_keys = list(events[-2]["changes"])
        self.assertEqual(changes_keys, sorted(changes_keys))
        restored = BranchStore.load_checkpoint(self.path)
        self.assertEqual(
            restored.replay("支"), store.replay("支")
        )

    def test_file_format_is_compact_utf8_no_bom_no_trailing_newline(self) -> None:
        store = make_store()
        store.save_checkpoint(self.path)
        raw = _read(self.path)
        self.assertFalse(raw.startswith(b"\xef\xbb\xbf"))
        raw.decode("utf-8")  # raises if not UTF-8
        self.assertFalse(raw.endswith(b"\n"))
        # Compact separators: no whitespace outside strings; fixture keys
        # and values contain none of their own.
        self.assertNotIn(b" ", raw)
        self.assertNotIn(b"\n", raw)
        doc = json.loads(raw)
        self.assertEqual(
            list(doc), ["schema_version", "payload", "checksum"]
        )
        self.assertEqual(doc["schema_version"], 1)
        self.assertEqual(
            list(doc["payload"]),
            ["events", "branches", "appends", "merges"],
        )
        self.assertEqual(len(doc["checksum"]), 64)
        self.assertEqual(doc["checksum"], doc["checksum"].lower())
        for event in doc["payload"]["events"]:
            self.assertEqual(
                list(event), ["id", "at", "parents", "changes"]
            )
        for branch in doc["payload"]["branches"]:
            self.assertEqual(list(branch), ["name", "source", "head"])

    def test_checksum_is_sha256_of_canonical_payload(self) -> None:
        make_store().save_checkpoint(self.path)
        raw = _read(self.path)
        doc = json.loads(raw)
        canonical = json.dumps(
            doc["payload"],
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        self.assertEqual(
            doc["checksum"], hashlib.sha256(canonical).hexdigest()
        )

    def test_pretty_file_with_matching_checksum_loads(self) -> None:
        # The checksum covers canonical payload bytes, not the raw file,
        # so re-indented JSON with a recomputed checksum still verifies.
        store = make_store()
        store.save_checkpoint(self.path)
        doc = _document(self.path)
        other = os.path.join(self.tmp.name, "pretty.json")
        with open(other, "w", encoding="utf-8") as handle:
            json.dump(doc, handle, indent=2)
        restored = BranchStore.load_checkpoint(other)
        self.assertEqual(restored.audit_log(), store.audit_log())

    def test_equal_state_produces_byte_identical_files(self) -> None:
        path_b = os.path.join(self.tmp.name, "cp2.json")
        make_store().save_checkpoint(self.path)
        make_store().save_checkpoint(path_b)
        self.assertEqual(
            _read(self.path), _read(path_b)
        )

    def test_snapshot_tokens_are_not_persisted(self) -> None:
        store = make_store()
        token = store.create_snapshot(3)
        store.save_checkpoint(self.path)
        restored = BranchStore.load_checkpoint(self.path)
        # The token table starts empty: the live token is unknown.
        with self.assertRaises(KeyError):
            restored.release_snapshot(token)
        # A fresh token table works with a fresh read allowance.
        new_token = restored.create_snapshot(1)
        self.assertNotEqual(new_token, token)
        view = restored._reserve_snapshot_read(new_token)
        self.assertEqual(view.audit_log(), restored.audit_log())
        with self.assertRaises(RuntimeError):
            restored._reserve_snapshot_read(new_token)
        restored.release_snapshot(new_token)

    def test_restored_state_is_isolated(self) -> None:
        store = make_store()
        store.save_checkpoint(self.path)
        restored = BranchStore.load_checkpoint(self.path)
        restored.append("main", "new", 9, {"q": 1})
        self.assertNotIn("new", [r["event_id"] for r in store.audit_log()])
        self.assertEqual(store.head("main"), "m2")
        # Mutating returned audit data never reaches the store.
        records = restored.audit_log()
        records[0]["changes"]["hacked"] = 100
        again = restored.audit_log()
        self.assertNotIn("hacked", again[0]["changes"])


class PathValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_save_rejects_non_str(self) -> None:
        store = make_store()
        for bad in (1, 1.0, b"x", None, ["a"], object()):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    store.save_checkpoint(bad)

    def test_load_rejects_non_str(self) -> None:
        for bad in (1, 1.0, b"x", None):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    BranchStore.load_checkpoint(bad)

    def test_save_rejects_empty_str(self) -> None:
        with self.assertRaises(ValueError):
            make_store().save_checkpoint("")

    def test_load_rejects_empty_str(self) -> None:
        with self.assertRaises(ValueError):
            BranchStore.load_checkpoint("")

    def test_path_checks_precede_state_and_filesystem(self) -> None:
        # TypeError beats ValueError beats any state/file consideration.
        with self.assertRaises(TypeError):
            BranchStore.load_checkpoint(None)
        # Empty path is rejected even though no file exists there.
        with self.assertRaises(ValueError):
            BranchStore.load_checkpoint("")

    def test_load_missing_file_raises_os_error(self) -> None:
        missing = os.path.join(self.tmp.name, "does-not-exist.json")
        with self.assertRaises(OSError):
            BranchStore.load_checkpoint(missing)


class SaveFailureAtomicityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "cp.json")
        make_store().save_checkpoint(self.path)
        self.original_bytes = _read(self.path)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _assert_untouched(self) -> None:
        self.assertEqual(
            _read(self.path), self.original_bytes
        )
        leftovers = [
            name
            for name in os.listdir(self.tmp.name)
            if name.startswith(".checkpoint-")
        ]
        self.assertEqual(leftovers, [])

    def test_nonexistent_directory_raises_os_error(self) -> None:
        bad = os.path.join(self.tmp.name, "no-such-dir", "cp.json")
        with self.assertRaises(OSError):
            make_store().save_checkpoint(bad)

    def test_target_is_an_existing_directory(self) -> None:
        with self.assertRaises(OSError):
            make_store().save_checkpoint(self.tmp.name)
        self._assert_untouched()

    def test_replace_failure_leaves_old_target_and_removes_tmp(self) -> None:
        with mock.patch(
            "city_twin.branches.os.replace", side_effect=OSError("boom")
        ):
            with self.assertRaises(OSError):
                make_store().save_checkpoint(self.path)
        self._assert_untouched()

    def test_fsync_failure_leaves_old_target_and_removes_tmp(self) -> None:
        with mock.patch(
            "city_twin.branches.os.fsync", side_effect=OSError("boom")
        ):
            with self.assertRaises(OSError):
                make_store().save_checkpoint(self.path)
        self._assert_untouched()

    def test_write_failure_leaves_old_target_and_removes_tmp(self) -> None:
        # Redirect the freshly opened temp file to /dev/full, whose writes
        # fail with ENOSPC at flush time.
        def fail_to_dev_full(fd: int, mode: str = "wb"):  # type: ignore[no-untyped-def]
            os.close(fd)
            return open("/dev/full", mode)

        with mock.patch.object(
            os, "fdopen", side_effect=fail_to_dev_full
        ):
            with self.assertRaises(OSError):
                make_store().save_checkpoint(self.path)
        self._assert_untouched()

    def test_save_never_changes_business_state(self) -> None:
        store = make_store()
        before = store.audit_log()
        store.save_checkpoint(self.path)
        store.save_checkpoint(self.path)
        self.assertEqual(store.audit_log(), before)


def _document(path: str) -> dict:
    with open(path, "rb") as handle:
        return json.loads(handle.read())


def _write_checkpoint(path: str, doc: dict) -> None:
    """Write a (possibly tampered) document as compact UTF-8 bytes."""
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(doc, separators=(",", ":"), ensure_ascii=False))


def _read(path: str) -> bytes:
    with open(path, "rb") as handle:
        return handle.read()


def _canonical_payload_bytes(doc: dict) -> bytes:
    return json.dumps(
        doc["payload"],
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


class CorruptCheckpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "cp.json")
        make_store().save_checkpoint(self.path)
        self.good = _document(self.path)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _assert_value_error(self, doc: dict) -> None:
        tampered_path = os.path.join(self.tmp.name, "tampered.json")
        _write_checkpoint(tampered_path, doc)
        with self.assertRaises(ValueError):
            BranchStore.load_checkpoint(tampered_path)

    def test_raw_bytes_tampering_detected_by_checksum(self) -> None:
        raw = _read(self.path)
        tampered = raw.replace(b'"a":1', b'"a":8', 1)
        tampered_path = os.path.join(self.tmp.name, "raw.json")
        with open(tampered_path, "wb") as handle:
            handle.write(tampered)
        with self.assertRaises(ValueError):
            BranchStore.load_checkpoint(tampered_path)

    def test_wrong_checksum_value(self) -> None:
        doc = json.loads(json.dumps(self.good))
        doc["checksum"] = "0" * 64
        self._assert_value_error(doc)

    def test_malformed_checksum(self) -> None:
        for bad in ("", "abc", "Z" * 64, 1, None):
            doc = json.loads(json.dumps(self.good))
            doc["checksum"] = bad
            self._assert_value_error(doc)

    def test_unsupported_version(self) -> None:
        doc = json.loads(json.dumps(self.good))
        doc["schema_version"] = 2
        # Checksum was computed over version-1 payload; recomputing against
        # the same payload keeps it matching so the version check is reached.
        self._assert_value_error(doc)

    def test_bool_version_rejected(self) -> None:
        doc = json.loads(json.dumps(self.good))
        doc["schema_version"] = True
        self._assert_value_error(doc)

    def test_missing_and_extra_top_level_keys(self) -> None:
        for build in (
            lambda d: {k: d[k] for k in ("schema_version", "payload")},
            lambda d: dict(d, extra=1),
        ):
            doc = build(json.loads(json.dumps(self.good)))
            self._assert_value_error(doc)

    def test_payload_not_object(self) -> None:
        doc = json.loads(json.dumps(self.good))
        doc["payload"] = [1, 2, 3]
        doc["checksum"] = hashlib.sha256(
            json.dumps(
                [1, 2, 3], separators=(",", ":"), ensure_ascii=False
            ).encode()
        ).hexdigest()
        self._assert_value_error(doc)

    def test_missing_payload_section(self) -> None:
        doc = json.loads(json.dumps(self.good))
        del doc["payload"]["merges"]
        doc["checksum"] = hashlib.sha256(
            _canonical_payload_bytes(doc)
        ).hexdigest()
        self._assert_value_error(doc)

    def test_extra_event_field(self) -> None:
        doc = json.loads(json.dumps(self.good))
        doc["payload"]["events"][0]["extra"] = 1
        doc["checksum"] = hashlib.sha256(
            _canonical_payload_bytes(doc)
        ).hexdigest()
        self._assert_value_error(doc)

    def test_bool_where_int_required(self) -> None:
        doc = json.loads(json.dumps(self.good))
        doc["payload"]["events"][1]["at"] = False
        doc["checksum"] = hashlib.sha256(
            _canonical_payload_bytes(doc)
        ).hexdigest()
        self._assert_value_error(doc)

    def test_negative_timestamp(self) -> None:
        doc = json.loads(json.dumps(self.good))
        doc["payload"]["events"][1]["at"] = -1
        doc["checksum"] = hashlib.sha256(
            _canonical_payload_bytes(doc)
        ).hexdigest()
        self._assert_value_error(doc)

    def test_empty_event_id(self) -> None:
        doc = json.loads(json.dumps(self.good))
        doc["payload"]["events"][0]["id"] = ""
        doc["checksum"] = hashlib.sha256(
            _canonical_payload_bytes(doc)
        ).hexdigest()
        self._assert_value_error(doc)

    def test_duplicate_event(self) -> None:
        doc = json.loads(json.dumps(self.good))
        events = doc["payload"]["events"]
        events.append(json.loads(json.dumps(events[0])))
        doc["checksum"] = hashlib.sha256(
            _canonical_payload_bytes(doc)
        ).hexdigest()
        self._assert_value_error(doc)

    def test_duplicate_parent(self) -> None:
        doc = json.loads(json.dumps(self.good))
        merge_event = next(
            e for e in doc["payload"]["events"] if e["id"] == "M"
        )
        merge_event["parents"] = ["m1", "m1"]
        doc["checksum"] = hashlib.sha256(
            _canonical_payload_bytes(doc)
        ).hexdigest()
        self._assert_value_error(doc)

    def test_unknown_parent(self) -> None:
        doc = json.loads(json.dumps(self.good))
        for event in doc["payload"]["events"]:
            if event["id"] == "m1":
                event["parents"] = ["ghost"]
        doc["checksum"] = hashlib.sha256(
            _canonical_payload_bytes(doc)
        ).hexdigest()
        self._assert_value_error(doc)

    def test_cycle(self) -> None:
        doc = {
            "schema_version": 1,
            "payload": {
                "events": [
                    {"id": "a", "at": 0, "parents": ["b"], "changes": {}},
                    {"id": "b", "at": 0, "parents": ["a"], "changes": {}},
                ],
                "branches": [],
                "appends": {},
                "merges": {},
            },
            "checksum": "",
        }
        doc["checksum"] = hashlib.sha256(
            _canonical_payload_bytes(doc)
        ).hexdigest()
        self._assert_value_error(doc)

    def test_unknown_branch_head_and_source(self) -> None:
        for field in ("head", "source"):
            doc = json.loads(json.dumps(self.good))
            main = next(
                b for b in doc["payload"]["branches"] if b["name"] == "main"
            )
            main[field] = "ghost"
            doc["checksum"] = hashlib.sha256(
                _canonical_payload_bytes(doc)
            ).hexdigest()
            self._assert_value_error(doc)

    def test_duplicate_branch(self) -> None:
        doc = json.loads(json.dumps(self.good))
        branches = doc["payload"]["branches"]
        branches.append(json.loads(json.dumps(branches[0])))
        doc["checksum"] = hashlib.sha256(
            _canonical_payload_bytes(doc)
        ).hexdigest()
        self._assert_value_error(doc)

    def test_append_record_unknown_branch(self) -> None:
        doc = json.loads(json.dumps(self.good))
        doc["payload"]["appends"]["ghost"] = {
            "m1": {"at": 1, "changes": {"a": 1}}
        }
        doc["checksum"] = hashlib.sha256(
            _canonical_payload_bytes(doc)
        ).hexdigest()
        self._assert_value_error(doc)

    def test_append_record_without_matching_event(self) -> None:
        doc = json.loads(json.dumps(self.good))
        doc["payload"]["appends"]["main"]["phantom"] = {
            "at": 8, "changes": {}
        }
        doc["checksum"] = hashlib.sha256(
            _canonical_payload_bytes(doc)
        ).hexdigest()
        self._assert_value_error(doc)

    def test_append_record_mismatched_at_and_changes(self) -> None:
        for mutate in (
            lambda d: d["payload"]["appends"]["main"]["m1"].__setitem__(
                "at", 9
            ),
            lambda d: d["payload"]["appends"]["main"]["m1"]["changes"].__setitem__(
                "a", 9
            ),
        ):
            doc = json.loads(json.dumps(self.good))
            mutate(doc)
            doc["checksum"] = hashlib.sha256(
                _canonical_payload_bytes(doc)
            ).hexdigest()
            self._assert_value_error(doc)

    def test_append_record_on_two_parent_event(self) -> None:
        doc = json.loads(json.dumps(self.good))
        doc["payload"]["appends"]["main"]["M"] = {"at": 4, "changes": {"c": 3}}
        doc["payload"]["merges"].pop("M")
        doc["checksum"] = hashlib.sha256(
            _canonical_payload_bytes(doc)
        ).hexdigest()
        self._assert_value_error(doc)

    def test_branch_head_not_reachable_through_records(self) -> None:
        doc = json.loads(json.dumps(self.good))
        # feature claims f2 as head while its append record is missing.
        doc["payload"]["appends"]["feature"].pop("f2")
        doc["checksum"] = hashlib.sha256(
            _canonical_payload_bytes(doc)
        ).hexdigest()
        self._assert_value_error(doc)

    def test_record_off_the_source_head_chain(self) -> None:
        doc = json.loads(json.dumps(self.good))
        # main claims an append whose parent chain never reaches its head.
        doc["payload"]["appends"]["main"]["f1"] = {
            "at": 2, "changes": {"b": 2}
        }
        doc["checksum"] = hashlib.sha256(
            _canonical_payload_bytes(doc)
        ).hexdigest()
        self._assert_value_error(doc)

    def test_merge_parents_swapped(self) -> None:
        doc = json.loads(json.dumps(self.good))
        merge_event = next(
            e for e in doc["payload"]["events"] if e["id"] == "M"
        )
        merge_event["parents"] = ["f1", "m1"]
        doc["checksum"] = hashlib.sha256(
            _canonical_payload_bytes(doc)
        ).hexdigest()
        self._assert_value_error(doc)

    def test_merge_unknown_source_branch(self) -> None:
        doc = json.loads(json.dumps(self.good))
        doc["payload"]["merges"]["M"]["source"] = "ghost"
        doc["checksum"] = hashlib.sha256(
            _canonical_payload_bytes(doc)
        ).hexdigest()
        self._assert_value_error(doc)

    def test_merge_source_head_had_advanced(self) -> None:
        # feature: root -> f1 -> f2 -> f3. Branch Y starts at f2 and is
        # merged into main first (GY, second parent f2), which puts f2 in
        # main's chain. M then records source parent f1 although feature's
        # head had already advanced to f3 -- and f2 (later on feature's
        # chain) is provably in M's ancestor closure through GY.
        doc = {
            "schema_version": 1,
            "payload": {
                "events": [
                    {"id": "GY", "at": 3, "parents": ["root", "f2"],
                     "changes": {"y": 1}},
                    {"id": "M", "at": 5, "parents": ["GY", "f1"],
                     "changes": {"c": 3}},
                    {"id": "f1", "at": 1, "parents": ["root"],
                     "changes": {"b": 2}},
                    {"id": "f2", "at": 2, "parents": ["f1"],
                     "changes": {"b": 0}},
                    {"id": "f3", "at": 4, "parents": ["f2"],
                     "changes": {"b": 5}},
                    {"id": "root", "at": 0, "parents": [], "changes": {}},
                ],
                "branches": [
                    {"name": "Y", "source": "f2", "head": "f2"},
                    {"name": "feature", "source": "root", "head": "f3"},
                    {"name": "main", "source": "root", "head": "M"},
                ],
                "appends": {
                    "feature": {
                        "f1": {"at": 1, "changes": {"b": 2}},
                        "f2": {"at": 2, "changes": {"b": 0}},
                        "f3": {"at": 4, "changes": {"b": 5}},
                    }
                },
                "merges": {
                    "GY": {"target": "main", "source": "Y", "at": 3,
                           "changes": {"y": 1}},
                    "M": {"target": "main", "source": "feature", "at": 5,
                          "changes": {"c": 3}},
                },
            },
            "checksum": "",
        }
        doc["checksum"] = hashlib.sha256(
            _canonical_payload_bytes(doc)
        ).hexdigest()
        self._assert_value_error(doc)

    def test_merge_overlapping_exclusive_keys_rejected(self) -> None:
        # Two branches each touched key "k" exclusively before the merge --
        # exactly the conflict merge() itself refuses.
        doc = {
            "schema_version": 1,
            "payload": {
                "events": [
                    {"id": "M", "at": 2, "parents": ["t", "u"],
                     "changes": {}},
                    {"id": "r", "at": 0, "parents": [], "changes": {}},
                    {"id": "t", "at": 1, "parents": ["r"],
                     "changes": {"k": 1}},
                    {"id": "u", "at": 1, "parents": ["r"],
                     "changes": {"k": 2}},
                ],
                "branches": [
                    {"name": "main", "source": "r", "head": "M"},
                    {"name": "side", "source": "r", "head": "u"},
                ],
                "appends": {
                    "main": {"t": {"at": 1, "changes": {"k": 1}}},
                    "side": {"u": {"at": 1, "changes": {"k": 2}}},
                },
                "merges": {
                    "M": {"target": "main", "source": "side", "at": 2,
                          "changes": {}},
                },
            },
            "checksum": "",
        }
        doc["checksum"] = hashlib.sha256(
            _canonical_payload_bytes(doc)
        ).hexdigest()
        self._assert_value_error(doc)

    def test_invalid_utf8(self) -> None:
        bad = os.path.join(self.tmp.name, "bad-utf8.json")
        with open(bad, "wb") as handle:
            handle.write(b"\xff\xfe" + _read(self.path))
        with self.assertRaises(ValueError):
            BranchStore.load_checkpoint(bad)

    def test_utf8_bom_rejected(self) -> None:
        bad = os.path.join(self.tmp.name, "bom.json")
        with open(bad, "wb") as handle:
            handle.write(b"\xef\xbb\xbf" + _read(self.path))
        with self.assertRaises(ValueError):
            BranchStore.load_checkpoint(bad)

    def test_malformed_json(self) -> None:
        bad = os.path.join(self.tmp.name, "bad-json.json")
        with open(bad, "w", encoding="utf-8") as handle:
            handle.write("{not json")
        with self.assertRaises(ValueError):
            BranchStore.load_checkpoint(bad)

    def test_json_top_level_array(self) -> None:
        bad = os.path.join(self.tmp.name, "array.json")
        with open(bad, "w", encoding="utf-8") as handle:
            handle.write("[1,2,3]")
        with self.assertRaises(ValueError):
            BranchStore.load_checkpoint(bad)

    def test_duplicate_json_key_rejected(self) -> None:
        bad = os.path.join(self.tmp.name, "dup.json")
        with open(bad, "w", encoding="utf-8") as handle:
            handle.write(
                '{"schema_version":1,"schema_version":1,'
                '"payload":{"events":[],"branches":[],"appends":{},'
                '"merges":{}},"checksum":"'
                + hashlib.sha256(
                    json.dumps(
                        {
                            "events": [],
                            "branches": [],
                            "appends": {},
                            "merges": {},
                        },
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest()
                + '"}'
            )
        with self.assertRaises(ValueError):
            BranchStore.load_checkpoint(bad)

    def test_failure_does_not_return_partial_or_modify_file(self) -> None:
        raw_before = _read(self.path)
        doc = json.loads(json.dumps(self.good))
        doc["payload"]["events"][1]["at"] = -1
        doc["checksum"] = hashlib.sha256(
            _canonical_payload_bytes(doc)
        ).hexdigest()
        bad = os.path.join(self.tmp.name, "bad.json")
        _write_checkpoint(bad, doc)
        bad_bytes_before = _read(bad)
        with self.assertRaises(ValueError):
            BranchStore.load_checkpoint(bad)
        self.assertEqual(_read(bad), bad_bytes_before)
        self.assertEqual(_read(self.path), raw_before)


class ConcurrentCheckpointTests(unittest.TestCase):
    def test_every_concurrent_checkpoint_is_consistent(self) -> None:
        graph = EventGraph()
        graph.add("root", 0, (), {})
        store = BranchStore(graph)
        store.create("main", "root")
        for worker in range(4):
            store.create(f"w{worker}", "root")

        stop = threading.Event()
        errors: list[BaseException] = []
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        counter = {"n": 0}
        counter_lock = threading.Lock()

        def worker(wid: int) -> None:
            try:
                head = "root"
                for step in range(20):
                    eid = f"w{wid}-{step}"
                    store.append(f"w{wid}", eid, step, {f"k{wid}": 1})
                    head = eid
                    if step % 3 == 2:
                        store.merge(
                            "main",
                            f"w{wid}",
                            f"mg{wid}-{step}",
                            step,
                            {f"km{wid}": 1},
                        )
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        def saver() -> None:
            try:
                while not stop.is_set():
                    with counter_lock:
                        counter["n"] += 1
                        n = counter["n"]
                    path = os.path.join(tmp.name, f"live-{n}.json")
                    store.save_checkpoint(path)
                    # Every captured instant must validate and restore.
                    restored = BranchStore.load_checkpoint(path)
                    for name in ("main", "w0", "w1", "w2", "w3"):
                        restored.replay(name)
                    restored.audit_log()
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        savers = [threading.Thread(target=saver) for _ in range(2)]
        workers = [
            threading.Thread(target=worker, args=(wid,)) for wid in range(4)
        ]
        for thread in savers + workers:
            thread.start()
        for thread in workers:
            thread.join()
        stop.set()
        for thread in savers:
            thread.join()

        self.assertEqual(errors, [])
        final_path = os.path.join(tmp.name, "final.json")
        store.save_checkpoint(final_path)
        restored = BranchStore.load_checkpoint(final_path)
        self.assertEqual(restored.audit_log(), store.audit_log())
        for name in ("main", "w0", "w1", "w2", "w3"):
            self.assertEqual(
                restored.replay(name), store.replay(name)
            )
        # No temporary files leaked.
        leftovers = [
            name
            for name in os.listdir(tmp.name)
            if name.startswith(".checkpoint-")
        ]
        self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()
