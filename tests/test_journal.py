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
    return store


def _read(path: str) -> bytes:
    with open(path, "rb") as handle:
        return handle.read()


def _document(path: str) -> dict:
    return json.loads(_read(path))


def _canonical(value) -> bytes:
    return json.dumps(
        value, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _frame_digest(frame: dict) -> str:
    return hashlib.sha256(_canonical(frame)).hexdigest()


class JournalTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.cp = os.path.join(self.tmp.name, "cp.json")
        self.journal = os.path.join(self.tmp.name, "journal.json")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def make_checkpointed_store(self) -> BranchStore:
        store = make_store()
        store.save_checkpoint(self.cp)
        return store

    def make_journaled_store(self) -> BranchStore:
        store = self.make_checkpointed_store()
        store.enable_journal(self.cp, self.journal)
        return store


class EnableJournalTests(JournalTestBase):
    def test_enable_returns_none_and_creates_empty_journal(self) -> None:
        store = self.make_checkpointed_store()
        self.assertIsNone(store.enable_journal(self.cp, self.journal))
        doc = _document(self.journal)
        self.assertEqual(
            list(doc), ["schema_version", "checkpoint", "frames"]
        )
        self.assertEqual(doc["schema_version"], 2)
        self.assertEqual(doc["frames"], [])
        self.assertEqual(doc["checkpoint"], _document(self.cp)["checksum"])

    def test_journal_file_is_compact_utf8_no_bom_no_trailing_newline(
        self,
    ) -> None:
        store = self.make_journaled_store()
        store.append("main", "m2", 2, {"支": 1})
        raw = _read(self.journal)
        self.assertFalse(raw.startswith(b"\xef\xbb\xbf"))
        raw.decode("utf-8")
        self.assertFalse(raw.endswith(b"\n"))
        self.assertNotIn(b" ", raw)
        self.assertIn("支".encode("utf-8"), raw)

    def test_rejects_non_str_paths(self) -> None:
        store = self.make_checkpointed_store()
        for bad in (1, 1.0, b"x", None, ["a"]):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    store.enable_journal(bad, self.journal)
                with self.assertRaises(TypeError):
                    store.enable_journal(self.cp, bad)

    def test_rejects_empty_str_paths(self) -> None:
        store = self.make_checkpointed_store()
        with self.assertRaises(ValueError):
            store.enable_journal("", self.journal)
        with self.assertRaises(ValueError):
            store.enable_journal(self.cp, "")

    def test_missing_checkpoint_raises_os_error(self) -> None:
        store = make_store()
        missing = os.path.join(self.tmp.name, "missing.json")
        with self.assertRaises(OSError):
            store.enable_journal(missing, self.journal)

    def test_corrupt_checkpoint_raises_value_error(self) -> None:
        store = self.make_checkpointed_store()
        with open(self.cp, "ab") as handle:
            handle.write(b"garbage")
        with self.assertRaises(ValueError):
            store.enable_journal(self.cp, self.journal)

    def test_mismatched_checkpoint_raises_value_error(self) -> None:
        store = self.make_checkpointed_store()
        # The checkpoint no longer describes the store's state.
        store.append("main", "m2", 2, {"a": 5})
        with self.assertRaises(ValueError):
            store.enable_journal(self.cp, self.journal)
        self.assertFalse(os.path.exists(self.journal))

    def test_other_store_checkpoint_raises_value_error(self) -> None:
        store = self.make_checkpointed_store()
        other = os.path.join(self.tmp.name, "other.json")
        make_store().save_checkpoint(other)
        # Same history is the same checkpoint, so change the other one.
        changed = make_store()
        changed.append("main", "extra", 9, {"z": 1})
        changed.save_checkpoint(other)
        with self.assertRaises(ValueError):
            store.enable_journal(other, self.journal)

    def test_double_enable_raises_runtime_error(self) -> None:
        store = self.make_journaled_store()
        other = os.path.join(self.tmp.name, "journal2.json")
        with self.assertRaises(RuntimeError):
            store.enable_journal(self.cp, other)
        self.assertFalse(os.path.exists(other))

    def test_existing_journal_target_raises_os_error(self) -> None:
        store = self.make_checkpointed_store()
        with open(self.journal, "wb") as handle:
            handle.write(b"occupied")
        with self.assertRaises(OSError):
            store.enable_journal(self.cp, self.journal)
        self.assertEqual(_read(self.journal), b"occupied")

    def test_journal_path_in_missing_directory_raises_os_error(self) -> None:
        store = self.make_checkpointed_store()
        bad = os.path.join(self.tmp.name, "no-such-dir", "j.json")
        with self.assertRaises(OSError):
            store.enable_journal(self.cp, bad)

    def test_failed_enable_leaves_store_unattached(self) -> None:
        store = self.make_checkpointed_store()
        store.append("main", "m2", 2, {"a": 5})
        with self.assertRaises(ValueError):
            store.enable_journal(self.cp, self.journal)
        # A later valid enable still works.
        store.save_checkpoint(self.cp)
        store.enable_journal(self.cp, self.journal)
        store.append("main", "m3", 3, {"a": 1})
        self.assertEqual(len(_document(self.journal)["frames"]), 1)


class JournalFrameTests(JournalTestBase):
    def test_each_commit_writes_one_frame_in_order(self) -> None:
        store = self.make_journaled_store()
        store.append("main", "m2", 2, {"a": 1})
        store.create("side", "m1")
        store.append("side", "s1", 3, {"s": 1})
        store.merge("main", "side", "M2", 4, {"c": 1})
        doc = _document(self.journal)
        self.assertEqual(
            [(f["seq"], f["op"]) for f in doc["frames"]],
            [(1, "append"), (2, "create"), (3, "append"), (4, "merge")],
        )
        self.assertEqual(
            doc["frames"][0]["params"],
            {"name": "main", "id": "m2", "at": 2, "changes": {"a": 1}},
        )
        self.assertEqual(
            doc["frames"][1]["params"],
            {"name": "side", "from_event": "m1"},
        )
        self.assertEqual(
            doc["frames"][3]["params"],
            {
                "target": "main",
                "source": "side",
                "id": "M2",
                "at": 4,
                "changes": {"c": 1},
            },
        )

    def test_frames_form_a_sha256_chain_rooted_in_the_checkpoint(self) -> None:
        store = self.make_journaled_store()
        store.append("main", "m2", 2, {"a": 1})
        store.append("main", "m3", 3, {"a": 2})
        doc = _document(self.journal)
        first, second = doc["frames"]
        self.assertEqual(first["prev"], doc["checkpoint"])
        self.assertEqual(second["prev"], _frame_digest(first))
        for frame in doc["frames"]:
            self.assertEqual(
                list(frame),
                ["seq", "op", "params", "before", "after", "prev"],
            )
            self.assertEqual(len(frame["prev"]), 64)
            self.assertEqual(frame["prev"], frame["prev"].lower())
            self.assertEqual(len(frame["before"]), 64)
            self.assertEqual(len(frame["after"]), 64)
        # The first frame leaves the bound checkpoint state, and the
        # summaries chain the states independently of the prev links.
        self.assertEqual(first["before"], doc["checkpoint"])
        self.assertEqual(second["before"], first["after"])
        # The summaries describe the checkpoint payload format.
        for frame in doc["frames"]:
            for field in ("before", "after"):
                self.assertEqual(
                    frame[field], frame[field].lower()
                )
                self.assertTrue(
                    all(c in "0123456789abcdef" for c in frame[field])
                )

    def test_idempotent_retries_write_no_frames(self) -> None:
        store = self.make_journaled_store()
        store.append("main", "m2", 2, {"a": 1})
        store.create("side", "m1")
        store.append("side", "s1", 3, {"s": 1})
        store.merge("main", "side", "M2", 4, {"c": 1})
        before = _read(self.journal)
        # Exact retries of recorded operations are no-ops.
        store.append("main", "m2", 2, {"a": 1})
        store.create("side", "m1")
        store.append("side", "s1", 3, {"s": 1})
        store.merge("main", "side", "M2", 4, {"c": 1})
        self.assertEqual(_read(self.journal), before)

    def test_failed_operations_write_no_frames(self) -> None:
        store = self.make_journaled_store()
        store.append("main", "m2", 2, {"a": 1})
        before = _read(self.journal)
        with self.assertRaises(ValueError):
            store.append("main", "m2", 2, {"a": 9})
        with self.assertRaises(KeyError):
            store.append("ghost", "g1", 3, {})
        # Existing name with a different source is a create conflict.
        with self.assertRaises(ValueError):
            store.create("main", "m1")
        with self.assertRaises(ValueError):
            store.merge("main", "main", "MM", 4, {})
        self.assertEqual(_read(self.journal), before)

    def test_same_history_produces_identical_bytes(self) -> None:
        def build(journal_path: str, cp_path: str) -> None:
            store = make_store()
            store.save_checkpoint(cp_path)
            store.enable_journal(cp_path, journal_path)
            store.append("main", "m2", 2, {"z": 1, "a": 2})
            store.merge("main", "feature", "M2", 3, {"c": 1})

        cp2 = os.path.join(self.tmp.name, "cp2.json")
        journal2 = os.path.join(self.tmp.name, "journal2.json")
        build(self.journal, self.cp)
        build(journal2, cp2)
        self.assertEqual(_read(self.journal), _read(journal2))


class JournalCommitFailureTests(JournalTestBase):
    def test_replace_failure_raises_and_preserves_state_and_journal(
        self,
    ) -> None:
        store = self.make_journaled_store()
        store.append("main", "m2", 2, {"a": 1})
        journal_before = _read(self.journal)
        audit_before = store.audit_log()
        head_before = store.head("main")
        with mock.patch(
            "city_twin.branches.os.replace", side_effect=OSError("boom")
        ):
            with self.assertRaises(OSError):
                store.append("main", "m3", 3, {"a": 2})
        self.assertEqual(_read(self.journal), journal_before)
        self.assertEqual(store.audit_log(), audit_before)
        self.assertEqual(store.head("main"), head_before)
        # The rolled-back event is fully gone from the graph, so the same
        # inputs can be committed cleanly by a later call.
        self.assertNotIn("m3", store._graph._at)
        store.append("main", "m3", 3, {"a": 2})
        self.assertEqual(store.head("main"), "m3")
        # The retry above journaled exactly one new frame.
        self.assertEqual(len(_document(self.journal)["frames"]), 2)

    def test_fsync_failure_rolls_back_merge(self) -> None:
        store = self.make_journaled_store()
        store.append("feature", "f2", 3, {"b": 0})
        with mock.patch(
            "city_twin.branches.os.fsync", side_effect=OSError("boom")
        ):
            with self.assertRaises(OSError):
                store.merge("main", "feature", "M", 4, {"c": 3})
        self.assertEqual(store.head("main"), "m1")
        self.assertNotIn("M", store._merges)
        self.assertNotIn("M", store._graph._at)
        # The earlier feature append was committed; the failed merge was not.
        frames = _document(self.journal)["frames"]
        self.assertEqual([f["op"] for f in frames], ["append"])
        # No temporary files leaked.
        leftovers = [
            name
            for name in os.listdir(self.tmp.name)
            if name.startswith(".journal-")
        ]
        self.assertEqual(leftovers, [])

    def test_detached_store_needs_no_journal(self) -> None:
        store = make_store()
        store.append("main", "m2", 2, {"a": 1})
        self.assertFalse(os.path.exists(self.journal))


class LoadJournalTests(JournalTestBase):
    def build_history(self) -> BranchStore:
        store = self.make_journaled_store()
        store.append("main", "m2", 2, {"a": 1})
        store.create("side", "m1")
        store.append("side", "s1", 3, {"s": 1})
        store.merge("main", "side", "M2", 4, {"c": 1})
        store.append("main", "m3", 5, {"a": 2})
        return store

    def test_full_recovery_matches_original(self) -> None:
        store = self.build_history()
        restored = BranchStore.load_journal(self.cp, self.journal, None)
        self.assertIsInstance(restored, BranchStore)
        self.assertEqual(restored.audit_log(), store.audit_log())
        for name in ("main", "feature", "side"):
            self.assertEqual(restored.replay(name), store.replay(name))
            self.assertEqual(restored.head(name), store.head(name))
        self.assertEqual(
            restored.audit_merge("M2"), store.audit_merge("M2")
        )
        # Idempotency and conflict semantics survive the replay.
        restored.append("main", "m3", 5, {"a": 2})
        with self.assertRaises(ValueError):
            restored.append("main", "m3", 5, {"a": 9})
        with self.assertRaises(ValueError):
            restored.merge("main", "side", "M2", 4, {"c": 9})

    def test_through_zero_restores_only_the_checkpoint(self) -> None:
        store = self.build_history()
        restored = BranchStore.load_journal(self.cp, self.journal, 0)
        baseline = BranchStore.load_checkpoint(self.cp)
        self.assertEqual(restored.audit_log(), baseline.audit_log())
        self.assertEqual(restored.head("main"), "m1")
        with self.assertRaises(KeyError):
            restored.head("side")

    def test_through_replays_a_prefix(self) -> None:
        self.build_history()
        restored = BranchStore.load_journal(self.cp, self.journal, 3)
        # Frames 1-3: main's m2, create side, side's s1; the merge (4)
        # and m3 (5) are not replayed.
        self.assertEqual(restored.head("main"), "m2")
        self.assertEqual(restored.head("side"), "s1")
        self.assertEqual(
            restored.replay("side"), {"seed": 7, "a": 1, "s": 1}
        )
        with self.assertRaises(KeyError):
            restored.audit_merge("M2")
        self.assertEqual(restored.head("feature"), "f1")

    def test_through_at_last_frame_is_a_full_recovery(self) -> None:
        self.build_history()
        last = len(_document(self.journal)["frames"])
        restored = BranchStore.load_journal(self.cp, self.journal, last)
        # Fully recovered: still attached, new commits extend the chain.
        restored.append("main", "m4", 6, {"a": 3})
        doc = _document(self.journal)
        self.assertEqual(len(doc["frames"]), last + 1)
        self.assertEqual(doc["frames"][-1]["seq"], last + 1)
        self.assertEqual(
            doc["frames"][-1]["prev"], _frame_digest(doc["frames"][-2])
        )

    def test_full_recovery_stays_attached_and_appends_safely(self) -> None:
        store = self.build_history()
        restored = BranchStore.load_journal(self.cp, self.journal, None)
        restored.append("main", "m4", 6, {"a": 3})
        again = BranchStore.load_journal(self.cp, self.journal, None)
        self.assertEqual(again.head("main"), "m4")
        self.assertEqual(again.replay("main"), restored.replay("main"))
        self.assertEqual(again.audit_log(), restored.audit_log())
        # The original store object is unaffected by the recovery.
        self.assertEqual(store.head("main"), "m3")

    def test_historical_recovery_is_detached(self) -> None:
        self.build_history()
        journal_before = _read(self.journal)
        restored = BranchStore.load_journal(self.cp, self.journal, 2)
        restored.append("main", "m9", 9, {"a": 9})
        self.assertEqual(_read(self.journal), journal_before)

    def test_through_type_validation(self) -> None:
        self.build_history()
        for bad in (True, False, "1", 1.0, b"0", [1]):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    BranchStore.load_journal(self.cp, self.journal, bad)

    def test_through_range_validation(self) -> None:
        self.build_history()
        last = len(_document(self.journal)["frames"])
        with self.assertRaises(ValueError):
            BranchStore.load_journal(self.cp, self.journal, -1)
        with self.assertRaises(ValueError):
            BranchStore.load_journal(self.cp, self.journal, last + 1)

    def test_path_validation(self) -> None:
        self.build_history()
        for bad in (1, None, b"x"):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    BranchStore.load_journal(bad, self.journal, None)
                with self.assertRaises(TypeError):
                    BranchStore.load_journal(self.cp, bad, None)
        with self.assertRaises(ValueError):
            BranchStore.load_journal("", self.journal, None)
        with self.assertRaises(ValueError):
            BranchStore.load_journal(self.cp, "", None)
        with self.assertRaises(OSError):
            BranchStore.load_journal(
                os.path.join(self.tmp.name, "missing.json"),
                self.journal,
                None,
            )
        with self.assertRaises(OSError):
            BranchStore.load_journal(
                self.cp,
                os.path.join(self.tmp.name, "missing.json"),
                None,
            )

    def test_recovery_does_not_modify_input_files(self) -> None:
        self.build_history()
        cp_before = _read(self.cp)
        journal_before = _read(self.journal)
        BranchStore.load_journal(self.cp, self.journal, None)
        BranchStore.load_journal(self.cp, self.journal, 2)
        self.assertEqual(_read(self.cp), cp_before)
        self.assertEqual(_read(self.journal), journal_before)

    def test_empty_journal_recovers_checkpoint_state(self) -> None:
        store = self.make_journaled_store()
        restored = BranchStore.load_journal(self.cp, self.journal, None)
        self.assertEqual(restored.audit_log(), store.audit_log())
        # A full recovery of an empty journal is attached.
        restored.append("main", "m2", 2, {"a": 1})
        self.assertEqual(len(_document(self.journal)["frames"]), 1)

    def test_snapshot_tokens_are_not_persisted(self) -> None:
        store = self.build_history()
        token = store.create_snapshot(2)
        restored = BranchStore.load_journal(self.cp, self.journal, None)
        with self.assertRaises(KeyError):
            restored.release_snapshot(token)


class CorruptJournalTests(JournalTestBase):
    def setUp(self) -> None:
        super().setUp()
        self.store = self.make_journaled_store()
        self.store.append("main", "m2", 2, {"a": 1})
        self.store.append("main", "m3", 3, {"a": 2})
        self.good = _document(self.journal)

    def write_journal(self, doc) -> None:
        with open(self.journal, "w", encoding="utf-8", newline="") as handle:
            handle.write(
                json.dumps(doc, separators=(",", ":"), ensure_ascii=False)
            )

    def rewrite_chain(self, doc) -> None:
        """Recompute the prev links so only structural defects remain."""
        prev = doc["checkpoint"]
        for frame in doc["frames"]:
            frame["prev"] = prev
            prev = _frame_digest(frame)

    def assert_bad(self, doc) -> None:
        self.write_journal(doc)
        with self.assertRaises(ValueError):
            BranchStore.load_journal(self.cp, self.journal, None)

    def test_utf8_bom_rejected(self) -> None:
        with open(self.journal, "wb") as handle:
            handle.write(b"\xef\xbb\xbf" + _read(self.journal))
        with self.assertRaises(ValueError):
            BranchStore.load_journal(self.cp, self.journal, None)

    def test_invalid_utf8(self) -> None:
        with open(self.journal, "wb") as handle:
            handle.write(b"\xff\xfe" + _read(self.journal))
        with self.assertRaises(ValueError):
            BranchStore.load_journal(self.cp, self.journal, None)

    def test_malformed_json(self) -> None:
        with open(self.journal, "w", encoding="utf-8") as handle:
            handle.write("{not json")
        with self.assertRaises(ValueError):
            BranchStore.load_journal(self.cp, self.journal, None)

    def test_truncated_journal(self) -> None:
        raw = _read(self.journal)
        with open(self.journal, "wb") as handle:
            handle.write(raw[: len(raw) // 2])
        with self.assertRaises(ValueError):
            BranchStore.load_journal(self.cp, self.journal, None)

    def test_duplicate_json_key_rejected(self) -> None:
        raw = _read(self.journal).decode("utf-8")
        dup = raw.replace(
            '"schema_version":2', '"schema_version":2,"schema_version":2', 1
        )
        with open(self.journal, "w", encoding="utf-8") as handle:
            handle.write(dup)
        with self.assertRaises(ValueError):
            BranchStore.load_journal(self.cp, self.journal, None)

    def test_top_level_not_object(self) -> None:
        self.assert_bad([1, 2, 3])

    def test_missing_and_extra_top_level_keys(self) -> None:
        doc = json.loads(json.dumps(self.good))
        del doc["frames"]
        self.assert_bad(doc)
        doc = json.loads(json.dumps(self.good))
        doc["extra"] = 1
        self.assert_bad(doc)

    def test_unsupported_version(self) -> None:
        for bad in (3, True, "1"):
            doc = json.loads(json.dumps(self.good))
            doc["schema_version"] = bad
            self.assert_bad(doc)

    def test_malformed_checkpoint_binding(self) -> None:
        for bad in ("", "abc", "Z" * 64, 1, None):
            doc = json.loads(json.dumps(self.good))
            doc["checkpoint"] = bad
            self.assert_bad(doc)

    def test_wrong_checkpoint_binding(self) -> None:
        doc = json.loads(json.dumps(self.good))
        doc["checkpoint"] = "0" * 64
        self.rewrite_chain(doc)
        self.assert_bad(doc)

    def test_frames_not_an_array(self) -> None:
        doc = json.loads(json.dumps(self.good))
        doc["frames"] = {}
        self.assert_bad(doc)

    def test_frame_not_an_object(self) -> None:
        doc = json.loads(json.dumps(self.good))
        doc["frames"][0] = "frame"
        self.assert_bad(doc)

    def test_frame_missing_and_extra_keys(self) -> None:
        doc = json.loads(json.dumps(self.good))
        del doc["frames"][0]["prev"]
        self.assert_bad(doc)
        doc = json.loads(json.dumps(self.good))
        doc["frames"][0]["extra"] = 1
        self.rewrite_chain(doc)
        self.assert_bad(doc)

    def test_non_consecutive_seq(self) -> None:
        doc = json.loads(json.dumps(self.good))
        doc["frames"][1]["seq"] = 3
        self.rewrite_chain(doc)
        self.assert_bad(doc)

    def test_bool_and_non_int_seq(self) -> None:
        for bad in (True, "1"):
            doc = json.loads(json.dumps(self.good))
            doc["frames"][0]["seq"] = bad
            self.rewrite_chain(doc)
            self.assert_bad(doc)

    def test_broken_hash_chain(self) -> None:
        doc = json.loads(json.dumps(self.good))
        doc["frames"][1]["prev"] = "0" * 64
        self.assert_bad(doc)

    def test_first_frame_not_rooted_in_checkpoint(self) -> None:
        doc = json.loads(json.dumps(self.good))
        doc["frames"][0]["prev"] = "0" * 64
        self.assert_bad(doc)

    def test_out_of_order_frames(self) -> None:
        doc = json.loads(json.dumps(self.good))
        doc["frames"] = [doc["frames"][1], doc["frames"][0]]
        self.rewrite_chain(doc)
        self.assert_bad(doc)

    def test_duplicate_frame(self) -> None:
        doc = json.loads(json.dumps(self.good))
        first = json.loads(json.dumps(doc["frames"][0]))
        doc["frames"].insert(0, first)
        doc["frames"][1]["seq"] = 2
        doc["frames"][2]["seq"] = 3
        self.rewrite_chain(doc)
        # The duplicated append is recorded already: replay conflict.
        self.assert_bad(doc)

    def test_unknown_op(self) -> None:
        doc = json.loads(json.dumps(self.good))
        doc["frames"][0]["op"] = "delete"
        self.rewrite_chain(doc)
        self.assert_bad(doc)

    def test_bad_params(self) -> None:
        for mutate in (
            lambda f: f["params"].__setitem__("at", -1),
            lambda f: f["params"].__setitem__("at", True),
            lambda f: f["params"].__setitem__("name", ""),
            lambda f: f["params"].__setitem__("extra", 1),
            lambda f: f["params"].pop("id"),
            lambda f: f["params"]["changes"].__setitem__("a", True),
        ):
            doc = json.loads(json.dumps(self.good))
            mutate(doc["frames"][0])
            self.rewrite_chain(doc)
            self.assert_bad(doc)

    def test_replay_conflict(self) -> None:
        # A frame whose event id clashes with checkpoint state.
        doc = json.loads(json.dumps(self.good))
        doc["frames"][0]["params"] = {
            "name": "main",
            "id": "m1",
            "at": 9,
            "changes": {"a": 9},
        }
        self.rewrite_chain(doc)
        self.assert_bad(doc)

    def test_frame_referencing_unknown_branch(self) -> None:
        doc = json.loads(json.dumps(self.good))
        doc["frames"][0]["params"]["name"] = "ghost"
        self.rewrite_chain(doc)
        self.assert_bad(doc)

    def test_failed_load_modifies_nothing(self) -> None:
        cp_before = _read(self.cp)
        doc = json.loads(json.dumps(self.good))
        doc["frames"][1]["prev"] = "0" * 64
        self.write_journal(doc)
        journal_before = _read(self.journal)
        with self.assertRaises(ValueError):
            BranchStore.load_journal(self.cp, self.journal, None)
        self.assertEqual(_read(self.cp), cp_before)
        self.assertEqual(_read(self.journal), journal_before)


class VersionOneJournalTests(JournalTestBase):
    """Old version-1 journals (frames without state summaries) stay
    readable, and a fully recovered one keeps appending in version 1."""

    def build_v1_history(self) -> BranchStore:
        store = self.make_journaled_store()
        store.append("main", "m2", 2, {"a": 1})
        store.create("side", "m1")
        store.append("side", "s1", 3, {"s": 1})
        store.merge("main", "side", "M2", 4, {"c": 1})
        # Rewrite the version-2 journal as version 1, stripping the
        # before/after summaries and rebuilding only the prev chain.
        doc = _document(self.journal)
        v1_frames = []
        prev = doc["checkpoint"]
        for index, frame in enumerate(doc["frames"]):
            stripped = {
                "seq": index + 1,
                "op": frame["op"],
                "params": frame["params"],
                "prev": prev,
            }
            v1_frames.append(stripped)
            prev = _frame_digest(stripped)
        v1 = {
            "schema_version": 1,
            "checkpoint": doc["checkpoint"],
            "frames": v1_frames,
        }
        with open(self.journal, "wb") as handle:
            handle.write(_canonical(v1))
        return store

    def test_v1_journal_recovers(self) -> None:
        store = self.build_v1_history()
        restored = BranchStore.load_journal(self.cp, self.journal, None)
        self.assertEqual(restored.audit_log(), store.audit_log())
        for name in ("main", "feature", "side"):
            self.assertEqual(restored.replay(name), store.replay(name))

    def test_v1_prefix_recovery(self) -> None:
        self.build_v1_history()
        restored = BranchStore.load_journal(self.cp, self.journal, 1)
        self.assertEqual(restored.head("main"), "m2")
        with self.assertRaises(KeyError):
            restored.head("side")

    def test_v1_recovery_keeps_appending_in_version_1(self) -> None:
        self.build_v1_history()
        restored = BranchStore.load_journal(self.cp, self.journal, None)
        restored.append("main", "m3", 5, {"a": 2})
        doc = _document(self.journal)
        self.assertEqual(doc["schema_version"], 1)
        self.assertEqual(
            list(doc["frames"][-1]), ["seq", "op", "params", "prev"]
        )
        self.assertEqual(doc["frames"][-1]["seq"], 5)
        # The extended version-1 journal still recovers.
        again = BranchStore.load_journal(self.cp, self.journal, None)
        self.assertEqual(again.head("main"), "m3")

    def test_v1_frame_with_summary_field_is_rejected(self) -> None:
        self.build_v1_history()
        doc = _document(self.journal)
        doc["frames"][0]["before"] = "0" * 64
        prev = doc["checkpoint"]
        for index, frame in enumerate(doc["frames"]):
            frame["seq"] = index + 1
            frame["prev"] = prev
            prev = _frame_digest(frame)
        with open(self.journal, "wb") as handle:
            handle.write(_canonical(doc))
        with self.assertRaises(ValueError):
            BranchStore.load_journal(self.cp, self.journal, None)

    def test_v2_frame_missing_summary_is_rejected(self) -> None:
        self.make_journaled_store().append("main", "m2", 2, {"a": 1})
        doc = _document(self.journal)
        del doc["frames"][0]["after"]
        with open(self.journal, "wb") as handle:
            handle.write(_canonical(doc))
        with self.assertRaises(ValueError):
            BranchStore.load_journal(self.cp, self.journal, None)


class OrderingProofTests(JournalTestBase):
    """Version 2 must reject swapped/duplicated/reordered frames even
    when the attacker renumbers and rebuilds both hash chains."""

    def setUp(self) -> None:
        super().setUp()
        store = self.make_journaled_store()
        store.append("main", "m2", 2, {"a": 1})
        store.append("main", "m3", 3, {"a": 2})
        store.create("side", "m1")
        store.append("side", "s1", 4, {"s": 1})
        self.checkpoint = _document(self.cp)["checksum"]

    def write(self, doc) -> None:
        with open(self.journal, "wb") as handle:
            handle.write(_canonical(doc))

    def rebuild_chains(self, doc) -> None:
        """Renumber, rebuild the prev frame chain AND thread the
        per-frame before/after summaries as if the attacker relinked
        them; each frame keeps its own recorded summary values."""
        prev = doc["checkpoint"]
        state = doc["checkpoint"]
        for index, frame in enumerate(doc["frames"]):
            frame["seq"] = index + 1
            frame["prev"] = prev
            frame["before"] = state
            state = frame["after"]
            prev = _frame_digest(frame)

    def assert_recovery_rejected(self, doc) -> None:
        self.write(doc)
        with self.assertRaises(ValueError):
            BranchStore.load_journal(self.cp, self.journal, None)

    def test_swapped_frames_rejected(self) -> None:
        doc = _document(self.journal)
        doc["frames"][0], doc["frames"][1] = (
            doc["frames"][1],
            doc["frames"][0],
        )
        self.rebuild_chains(doc)
        self.assert_recovery_rejected(doc)

    def test_reordered_frames_rejected(self) -> None:
        doc = _document(self.journal)
        doc["frames"] = [
            doc["frames"][3],
            doc["frames"][0],
            doc["frames"][2],
            doc["frames"][1],
        ]
        self.rebuild_chains(doc)
        self.assert_recovery_rejected(doc)

    def test_duplicated_frame_rejected(self) -> None:
        doc = _document(self.journal)
        duplicate = json.loads(json.dumps(doc["frames"][0]))
        doc["frames"].insert(1, duplicate)
        self.rebuild_chains(doc)
        self.assert_recovery_rejected(doc)

    def test_renumbered_but_summaries_untouched_rejected(self) -> None:
        # Swap with only the prev chain rebuilt; summaries stay frozen.
        doc = _document(self.journal)
        doc["frames"][0], doc["frames"][1] = (
            doc["frames"][1],
            doc["frames"][0],
        )
        prev = doc["checkpoint"]
        for index, frame in enumerate(doc["frames"]):
            frame["seq"] = index + 1
            frame["prev"] = prev
            prev = _frame_digest(frame)
        self.assert_recovery_rejected(doc)

    def test_forged_after_summary_rejected(self) -> None:
        doc = _document(self.journal)
        doc["frames"][0]["after"] = "f" * 64
        self.rebuild_chains(doc)
        self.assert_recovery_rejected(doc)

    def test_forged_before_summary_rejected(self) -> None:
        doc = _document(self.journal)
        doc["frames"][1]["before"] = "a" * 64
        # No rebuild: the forged before must be caught on its own (it
        # also breaks the downstream prev chain).
        self.assert_recovery_rejected(doc)

    def test_summary_disagreement_with_params_rejected(self) -> None:
        # Keep the summary chain consistent but change an operation
        # parameter, then rebuild the prev chain only.
        doc = _document(self.journal)
        doc["frames"][0]["params"]["at"] = 8
        prev = doc["checkpoint"]
        for index, frame in enumerate(doc["frames"]):
            frame["seq"] = index + 1
            frame["prev"] = prev
            prev = _frame_digest(frame)
        self.assert_recovery_rejected(doc)

    def test_malformed_summary_fields_rejected(self) -> None:
        for bad in ("", "abc", "Z" * 64, 1, None):
            doc = _document(self.journal)
            doc["frames"][0]["before"] = bad
            self.write(doc)
            with self.assertRaises(ValueError):
                BranchStore.load_journal(self.cp, self.journal, None)


class ConcurrentJournalTests(unittest.TestCase):
    def test_journal_order_matches_commit_order(self) -> None:
        graph = EventGraph()
        graph.add("root", 0, (), {})
        store = BranchStore(graph)
        store.create("main", "root")
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        cp = os.path.join(tmp.name, "cp.json")
        journal = os.path.join(tmp.name, "journal.json")
        for worker in range(4):
            store.create(f"w{worker}", "root")
        store.save_checkpoint(cp)
        store.enable_journal(cp, journal)

        errors: list[BaseException] = []

        def worker(wid: int) -> None:
            try:
                for step in range(15):
                    store.append(f"w{wid}", f"w{wid}-{step}", step, {"k": 1})
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [
            threading.Thread(target=worker, args=(wid,)) for wid in range(4)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(errors, [])
        doc = _document(journal)
        self.assertEqual(len(doc["frames"]), 60)
        self.assertEqual(
            [frame["seq"] for frame in doc["frames"]], list(range(1, 61))
        )
        # The journaled history restores exactly the final state.
        restored = BranchStore.load_journal(cp, journal, None)
        self.assertEqual(restored.audit_log(), store.audit_log())
        for wid in range(4):
            self.assertEqual(
                restored.replay(f"w{wid}"), store.replay(f"w{wid}")
            )
        leftovers = [
            name
            for name in os.listdir(tmp.name)
            if name.startswith(".journal-")
        ]
        self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()
