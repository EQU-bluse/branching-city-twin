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


def _state_summary_of(store: BranchStore) -> str:
    payload = store._checkpoint_payload(store._snapshot_state())
    return hashlib.sha256(_canonical(payload)).hexdigest()


class RotateTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.cp = os.path.join(self.tmp.name, "cp.json")
        self.journal = os.path.join(self.tmp.name, "journal.json")
        self.cp2 = os.path.join(self.tmp.name, "cp2.json")
        self.journal2 = os.path.join(self.tmp.name, "journal2.json")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def make_journaled_store(self) -> BranchStore:
        store = make_store()
        store.save_checkpoint(self.cp)
        store.enable_journal(self.cp, self.journal)
        return store

    def build_history(self) -> BranchStore:
        store = self.make_journaled_store()
        store.append("main", "m2", 2, {"a": 1})
        store.create("side", "m1")
        store.append("side", "s1", 3, {"s": 1})
        store.merge("main", "side", "M2", 4, {"c": 1})
        return store


class RotateValidationTests(RotateTestBase):
    def test_rejects_non_str_paths_in_signature_order(self) -> None:
        store = self.make_journaled_store()
        for bad in (1, 1.0, b"x", None, ["a"], (1,)):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    store.rotate_journal(bad, self.journal2)
                with self.assertRaises(TypeError):
                    store.rotate_journal(self.cp2, bad)

    def test_rejects_empty_str_paths(self) -> None:
        store = self.make_journaled_store()
        with self.assertRaises(ValueError):
            store.rotate_journal("", self.journal2)
        with self.assertRaises(ValueError):
            store.rotate_journal(self.cp2, "")

    def test_detached_store_raises_runtime_error(self) -> None:
        store = make_store()
        with self.assertRaises(RuntimeError):
            store.rotate_journal(self.cp2, self.journal2)
        self.assertFalse(os.path.exists(self.cp2))
        self.assertFalse(os.path.exists(self.journal2))

    def test_historical_recovery_is_detached_and_cannot_rotate(self) -> None:
        self.build_history()
        restored = BranchStore.load_journal(self.cp, self.journal, 2)
        with self.assertRaises(RuntimeError):
            restored.rotate_journal(self.cp2, self.journal2)

    def test_validation_precedes_attachment_check(self) -> None:
        store = make_store()
        with self.assertRaises(TypeError):
            store.rotate_journal(1, self.journal2)
        with self.assertRaises(ValueError):
            store.rotate_journal("", self.journal2)

    def test_existing_checkpoint_target_raises_os_error(self) -> None:
        store = self.make_journaled_store()
        with open(self.cp2, "wb") as handle:
            handle.write(b"occupied")
        with self.assertRaises(OSError):
            store.rotate_journal(self.cp2, self.journal2)
        self.assertEqual(_read(self.cp2), b"occupied")
        self.assertFalse(os.path.exists(self.journal2))

    def test_existing_journal_target_raises_os_error(self) -> None:
        store = self.make_journaled_store()
        with open(self.journal2, "wb") as handle:
            handle.write(b"occupied")
        with self.assertRaises(OSError):
            store.rotate_journal(self.cp2, self.journal2)
        self.assertEqual(_read(self.journal2), b"occupied")
        self.assertFalse(os.path.exists(self.cp2))

    def test_old_files_as_targets_raise_os_error(self) -> None:
        store = self.make_journaled_store()
        cp_before = _read(self.cp)
        journal_before = _read(self.journal)
        with self.assertRaises(OSError):
            store.rotate_journal(self.cp, self.journal2)
        with self.assertRaises(OSError):
            store.rotate_journal(self.cp2, self.journal)
        self.assertEqual(_read(self.cp), cp_before)
        self.assertEqual(_read(self.journal), journal_before)

    def test_missing_directory_raises_os_error(self) -> None:
        store = self.make_journaled_store()
        bad = os.path.join(self.tmp.name, "no-such-dir", "cp.json")
        with self.assertRaises(OSError):
            store.rotate_journal(bad, self.journal2)
        with self.assertRaises(OSError):
            store.rotate_journal(self.cp2, bad)

    def test_failed_rotation_leaves_store_attached_to_old_journal(self) -> None:
        store = self.make_journaled_store()
        store.append("main", "m2", 2, {"a": 1})
        with open(self.journal2, "wb") as handle:
            handle.write(b"occupied")
        with self.assertRaises(OSError):
            store.rotate_journal(self.cp2, self.journal2)
        # The old journal still receives new commits.
        store.append("main", "m3", 3, {"a": 2})
        frames = _document(self.journal)["frames"]
        self.assertEqual([f["seq"] for f in frames], [1, 2])
        self.assertEqual(store.head("main"), "m3")


class RotateSuccessTests(RotateTestBase):
    def test_rotate_returns_none_and_creates_bound_pair(self) -> None:
        store = self.build_history()
        self.assertIsNone(store.rotate_journal(self.cp2, self.journal2))
        cp_doc = _document(self.cp2)
        self.assertEqual(
            list(cp_doc), ["schema_version", "payload", "checksum"]
        )
        self.assertEqual(cp_doc["schema_version"], 1)
        journal_doc = _document(self.journal2)
        self.assertEqual(
            list(journal_doc), ["schema_version", "checkpoint", "frames"]
        )
        self.assertEqual(journal_doc["schema_version"], 2)
        self.assertEqual(journal_doc["frames"], [])
        self.assertEqual(journal_doc["checkpoint"], cp_doc["checksum"])

    def test_new_checkpoint_restores_rotation_instant(self) -> None:
        store = self.build_history()
        store.rotate_journal(self.cp2, self.journal2)
        restored = BranchStore.load_checkpoint(self.cp2)
        self.assertEqual(restored.audit_log(), store.audit_log())
        for name in ("main", "feature", "side"):
            self.assertEqual(restored.replay(name), store.replay(name))
            self.assertEqual(restored.head(name), store.head(name))
        self.assertEqual(
            restored.audit_merge("M2"), store.audit_merge("M2")
        )
        # Idempotency and conflict semantics match the rotation instant.
        restored.append("main", "m2", 2, {"a": 1})
        with self.assertRaises(ValueError):
            restored.append("main", "m2", 2, {"a": 9})
        with self.assertRaises(ValueError):
            restored.merge("main", "side", "M2", 4, {"c": 9})

    def test_old_files_stay_byte_for_byte_and_recover_as_archive(self) -> None:
        store = self.build_history()
        cp_before = _read(self.cp)
        journal_before = _read(self.journal)
        store.rotate_journal(self.cp2, self.journal2)
        self.assertEqual(_read(self.cp), cp_before)
        self.assertEqual(_read(self.journal), journal_before)
        # The archive pair still recovers the rotation-instant state.
        archived = BranchStore.load_journal(self.cp, self.journal, None)
        self.assertEqual(archived.audit_log(), store.audit_log())
        self.assertEqual(archived.head("main"), store.head("main"))
        # Historical positions of the archive still work too.
        older = BranchStore.load_journal(self.cp, self.journal, 1)
        self.assertEqual(older.head("main"), "m2")
        with self.assertRaises(KeyError):
            older.head("side")

    def test_new_commits_go_to_new_journal_from_seq_one(self) -> None:
        store = self.build_history()
        old_journal_before = _read(self.journal)
        store.rotate_journal(self.cp2, self.journal2)
        store.append("main", "m3", 5, {"a": 2})
        store.append("feature", "f2", 6, {"b": 3})
        doc = _document(self.journal2)
        self.assertEqual(doc["schema_version"], 2)
        self.assertEqual(
            [(f["seq"], f["op"]) for f in doc["frames"]],
            [(1, "append"), (2, "append")],
        )
        # The old journal is untouched by post-rotation commits.
        self.assertEqual(_read(self.journal), old_journal_before)

    def test_full_recovery_from_rotated_pair_continues_chain(self) -> None:
        store = self.build_history()
        store.rotate_journal(self.cp2, self.journal2)
        store.append("main", "m3", 5, {"a": 2})
        restored = BranchStore.load_journal(self.cp2, self.journal2, None)
        self.assertEqual(restored.audit_log(), store.audit_log())
        self.assertEqual(restored.head("main"), "m3")
        # Still attached: new commits extend the rotated journal.
        restored.append("main", "m4", 6, {"a": 3})
        doc = _document(self.journal2)
        self.assertEqual(len(doc["frames"]), 2)
        self.assertEqual(doc["frames"][-1]["seq"], 2)
        self.assertEqual(
            doc["frames"][-1]["prev"], _frame_digest(doc["frames"][-2])
        )

    def test_double_rotation_chains(self) -> None:
        store = self.build_history()
        store.rotate_journal(self.cp2, self.journal2)
        store.append("main", "m3", 5, {"a": 2})
        cp3 = os.path.join(self.tmp.name, "cp3.json")
        journal3 = os.path.join(self.tmp.name, "journal3.json")
        store.rotate_journal(cp3, journal3)
        store.create("late", "m3")
        restored = BranchStore.load_journal(cp3, journal3, None)
        self.assertEqual(restored.audit_log(), store.audit_log())
        self.assertEqual(restored.head("late"), "m3")
        # Each archived pair still recovers its own instant.
        mid = BranchStore.load_journal(self.cp2, self.journal2, None)
        self.assertEqual(mid.head("main"), "m3")
        with self.assertRaises(KeyError):
            mid.head("late")

    def test_rotate_empty_journal(self) -> None:
        store = self.make_journaled_store()
        store.rotate_journal(self.cp2, self.journal2)
        restored = BranchStore.load_journal(self.cp2, self.journal2, None)
        self.assertEqual(restored.audit_log(), store.audit_log())

    def test_rotated_journal_file_is_compact_utf8(self) -> None:
        store = self.make_journaled_store()
        store.rotate_journal(self.cp2, self.journal2)
        store.append("main", "m2", 2, {"支": 1})
        raw = _read(self.journal2)
        self.assertFalse(raw.startswith(b"\xef\xbb\xbf"))
        raw.decode("utf-8")
        self.assertFalse(raw.endswith(b"\n"))
        self.assertNotIn(b" ", raw)


class Version2FrameTests(RotateTestBase):
    def make_v2_store(self) -> BranchStore:
        store = self.build_history()
        store.rotate_journal(self.cp2, self.journal2)
        return store

    def test_frames_carry_state_summaries(self) -> None:
        store = self.make_v2_store()
        store.append("main", "m3", 5, {"a": 2})
        store.create("late", "m2")
        doc = _document(self.journal2)
        first, second = doc["frames"]
        for frame in (first, second):
            self.assertEqual(
                list(frame),
                ["seq", "op", "params", "before", "after", "prev"],
            )
            for field in ("before", "after", "prev"):
                self.assertEqual(len(frame[field]), 64)
                self.assertEqual(frame[field], frame[field].lower())
        # The first frame's before-summary is the bound checkpoint's
        # checksum; summaries chain frame to frame.
        self.assertEqual(first["before"], doc["checkpoint"])
        self.assertEqual(second["before"], first["after"])
        # The final after-summary is the live state's canonical summary.
        self.assertEqual(second["after"], _state_summary_of(store))
        # The prev hash chain is intact on top of the summaries.
        self.assertEqual(first["prev"], doc["checkpoint"])
        self.assertEqual(second["prev"], _frame_digest(first))

    def test_merge_frame_summaries(self) -> None:
        store = self.make_v2_store()
        store.append("feature", "f2", 5, {"b": 0})
        store.merge("main", "feature", "M9", 6, {"c": 3})
        doc = _document(self.journal2)
        self.assertEqual(
            [f["op"] for f in doc["frames"]], ["append", "merge"]
        )
        self.assertEqual(doc["frames"][1]["after"], _state_summary_of(store))

    def test_v2_recovery_matches_original(self) -> None:
        store = self.make_v2_store()
        store.append("main", "m3", 5, {"a": 2})
        store.create("late", "m2")
        store.append("late", "l1", 6, {"z": 1})
        store.merge("main", "late", "M3", 7, {"c": 1})
        restored = BranchStore.load_journal(self.cp2, self.journal2, None)
        self.assertEqual(restored.audit_log(), store.audit_log())
        for name in ("main", "feature", "side", "late"):
            self.assertEqual(restored.replay(name), store.replay(name))
            self.assertEqual(restored.head(name), store.head(name))
        self.assertEqual(
            restored.audit_merge("M3"), store.audit_merge("M3")
        )

    def test_v2_through_replays_a_verified_prefix(self) -> None:
        store = self.make_v2_store()
        store.append("main", "m3", 5, {"a": 2})
        store.append("main", "m4", 6, {"a": 3})
        restored = BranchStore.load_journal(self.cp2, self.journal2, 1)
        self.assertEqual(restored.head("main"), "m3")
        # A historical recovery is detached and cannot rotate.
        with self.assertRaises(RuntimeError):
            restored.rotate_journal(
                os.path.join(self.tmp.name, "x.json"),
                os.path.join(self.tmp.name, "y.json"),
            )

    def test_v1_journals_still_load(self) -> None:
        store = self.build_history()
        restored = BranchStore.load_journal(self.cp, self.journal, None)
        self.assertEqual(restored.audit_log(), store.audit_log())
        # A v1-attached store keeps writing v1 frames.
        restored.append("main", "m3", 5, {"a": 2})
        frame = _document(self.journal)["frames"][-1]
        self.assertEqual(list(frame), ["seq", "op", "params", "prev"])

    def test_same_history_after_rotation_produces_identical_bytes(self) -> None:
        def build(tag: str) -> tuple[str, str]:
            graph = EventGraph()
            graph.add("root", 0, (), {"seed": 7})
            store = BranchStore(graph)
            store.create("main", "root")
            store.create("feature", "root")
            store.append("main", "m1", 1, {"a": 1})
            store.append("feature", "f1", 2, {"b": 2})
            base_cp = os.path.join(self.tmp.name, f"{tag}-base-cp.json")
            base_j = os.path.join(self.tmp.name, f"{tag}-base-j.json")
            store.save_checkpoint(base_cp)
            store.enable_journal(base_cp, base_j)
            store.append("main", "m2", 2, {"a": 1})
            store.create("side", "m1")
            store.append("side", "s1", 3, {"s": 1})
            store.merge("main", "side", "M2", 4, {"c": 1})
            cp_path = os.path.join(self.tmp.name, f"{tag}-cp.json")
            j_path = os.path.join(self.tmp.name, f"{tag}-j.json")
            store.rotate_journal(cp_path, j_path)
            store.append("main", "m3", 5, {"z": 1, "a": 2})
            return cp_path, j_path

        cp_a, j_a = build("a")
        cp_b, j_b = build("b")
        self.assertEqual(_read(cp_a), _read(cp_b))
        self.assertEqual(_read(j_a), _read(j_b))


class CorruptV2JournalTests(RotateTestBase):
    def setUp(self) -> None:
        super().setUp()
        self.store = self.build_history()
        self.store.rotate_journal(self.cp2, self.journal2)
        self.store.append("main", "m3", 5, {"a": 2})
        self.store.append("main", "m4", 6, {"a": 3})
        self.good = _document(self.journal2)

    def write_journal(self, doc) -> None:
        with open(self.journal2, "w", encoding="utf-8", newline="") as handle:
            handle.write(
                json.dumps(doc, separators=(",", ":"), ensure_ascii=False)
            )

    def rewrite_chain(self, doc) -> None:
        """Recompute prev links and state summaries so only the targeted
        defect remains."""
        prev = doc["checkpoint"]
        for frame in doc["frames"]:
            frame["prev"] = prev
            prev = _frame_digest(frame)

    def assert_bad(self, doc) -> None:
        self.write_journal(doc)
        with self.assertRaises(ValueError):
            BranchStore.load_journal(self.cp2, self.journal2, None)

    def test_version_one_and_two_accepted_three_rejected(self) -> None:
        doc = json.loads(json.dumps(self.good))
        doc["schema_version"] = 3
        self.assert_bad(doc)
        doc = json.loads(json.dumps(self.good))
        doc["schema_version"] = True
        self.assert_bad(doc)

    def test_frame_missing_or_extra_summary_fields(self) -> None:
        doc = json.loads(json.dumps(self.good))
        del doc["frames"][0]["before"]
        self.assert_bad(doc)
        doc = json.loads(json.dumps(self.good))
        del doc["frames"][0]["after"]
        self.assert_bad(doc)
        doc = json.loads(json.dumps(self.good))
        doc["frames"][0]["extra"] = 1
        self.rewrite_chain(doc)
        self.assert_bad(doc)

    def test_malformed_summary_fields(self) -> None:
        for bad in ("", "abc", "Z" * 64, 1, None, True):
            doc = json.loads(json.dumps(self.good))
            doc["frames"][0]["before"] = bad
            self.rewrite_chain(doc)
            self.assert_bad(doc)
            doc = json.loads(json.dumps(self.good))
            doc["frames"][1]["after"] = bad
            self.rewrite_chain(doc)
            self.assert_bad(doc)

    def test_wrong_checkpoint_binding(self) -> None:
        doc = json.loads(json.dumps(self.good))
        doc["checkpoint"] = "0" * 64
        self.rewrite_chain(doc)
        self.assert_bad(doc)

    def test_swapped_frames_renumbered_and_rechained_still_fail(self) -> None:
        doc = json.loads(json.dumps(self.good))
        frames = doc["frames"]
        frames[0], frames[1] = frames[1], frames[0]
        for index, frame in enumerate(frames):
            frame["seq"] = index + 1
        self.rewrite_chain(doc)
        self.assert_bad(doc)

    def test_duplicated_frame_renumbered_and_rechained_still_fail(self) -> None:
        doc = json.loads(json.dumps(self.good))
        first = json.loads(json.dumps(doc["frames"][0]))
        doc["frames"].insert(0, first)
        for index, frame in enumerate(doc["frames"]):
            frame["seq"] = index + 1
        self.rewrite_chain(doc)
        self.assert_bad(doc)

    def test_reordered_with_rebuilt_summaries_still_fails(self) -> None:
        # Even recomputing the summaries over the *reordered* sequence
        # cannot help: the reordered replay either conflicts or produces
        # states whose summaries do not match the recomputed ones.
        doc = json.loads(json.dumps(self.good))
        frames = doc["frames"]
        frames[0], frames[1] = frames[1], frames[0]
        for index, frame in enumerate(frames):
            frame["seq"] = index + 1
        self.rewrite_chain(doc)
        self.assert_bad(doc)

    def test_tampered_before_summary(self) -> None:
        doc = json.loads(json.dumps(self.good))
        doc["frames"][1]["before"] = doc["frames"][1]["after"]
        self.rewrite_chain(doc)
        self.assert_bad(doc)

    def test_tampered_after_summary(self) -> None:
        doc = json.loads(json.dumps(self.good))
        doc["frames"][0]["after"] = doc["frames"][0]["before"]
        self.rewrite_chain(doc)
        self.assert_bad(doc)

    def test_tampered_params_caught_by_summary(self) -> None:
        doc = json.loads(json.dumps(self.good))
        doc["frames"][0]["params"]["changes"] = {"a": 99}
        self.rewrite_chain(doc)
        self.assert_bad(doc)

    def test_broken_hash_chain(self) -> None:
        doc = json.loads(json.dumps(self.good))
        doc["frames"][1]["prev"] = "0" * 64
        self.assert_bad(doc)

    def test_failed_load_modifies_nothing(self) -> None:
        cp_before = _read(self.cp2)
        doc = json.loads(json.dumps(self.good))
        doc["frames"][1]["before"] = "0" * 64
        self.write_journal(doc)
        journal_before = _read(self.journal2)
        with self.assertRaises(ValueError):
            BranchStore.load_journal(self.cp2, self.journal2, None)
        self.assertEqual(_read(self.cp2), cp_before)
        self.assertEqual(_read(self.journal2), journal_before)


class RotateFailureAtomicityTests(RotateTestBase):
    def test_journal_create_failure_removes_new_checkpoint(self) -> None:
        store = self.build_history()
        journal_before = _read(self.journal)
        audit_before = store.audit_log()
        real_open = os.open

        def failing_open(path, flags, mode=0o777):
            if os.path.abspath(str(path)) == os.path.abspath(self.journal2):
                raise OSError("boom")
            return real_open(path, flags, mode)

        with mock.patch("city_twin.branches.os.open", failing_open):
            with self.assertRaises(OSError):
                store.rotate_journal(self.cp2, self.journal2)
        # The half-created checkpoint is cleaned up; nothing switched.
        self.assertFalse(os.path.exists(self.cp2))
        self.assertFalse(os.path.exists(self.journal2))
        self.assertEqual(_read(self.journal), journal_before)
        self.assertEqual(store.audit_log(), audit_before)
        store.append("main", "m3", 5, {"a": 2})
        self.assertEqual(len(_document(self.journal)["frames"]), 5)

    def test_checkpoint_write_failure_creates_nothing(self) -> None:
        store = self.build_history()
        with mock.patch(
            "city_twin.branches.os.fsync", side_effect=OSError("boom")
        ):
            with self.assertRaises(OSError):
                store.rotate_journal(self.cp2, self.journal2)
        self.assertFalse(os.path.exists(self.cp2))
        self.assertFalse(os.path.exists(self.journal2))
        # Business state is item-for-item unchanged and journaling works.
        self.assertEqual(store.head("main"), "M2")
        store.append("main", "m3", 5, {"a": 2})
        self.assertEqual(store.head("main"), "m3")

    def test_failed_rotation_preserves_idempotency_records(self) -> None:
        store = self.build_history()
        with open(self.cp2, "wb") as handle:
            handle.write(b"occupied")
        with self.assertRaises(OSError):
            store.rotate_journal(self.cp2, self.journal2)
        # Recorded operations are still idempotent no-ops, and conflicts
        # still conflict -- the failure consumed nothing.
        store.append("main", "m2", 2, {"a": 1})
        with self.assertRaises(ValueError):
            store.append("main", "m2", 2, {"a": 9})
        with self.assertRaises(ValueError):
            store.merge("main", "side", "M2", 4, {"c": 9})
        # A later valid rotation still succeeds.
        os.remove(self.cp2)
        store.rotate_journal(self.cp2, self.journal2)
        self.assertEqual(_document(self.journal2)["frames"], [])


class ConcurrentRotateTests(RotateTestBase):
    def test_rotation_blocks_and_orders_with_commits(self) -> None:
        graph = EventGraph()
        graph.add("root", 0, (), {})
        store = BranchStore(graph)
        store.create("main", "root")
        for worker in range(3):
            store.create(f"w{worker}", "root")
        store.save_checkpoint(self.cp)
        store.enable_journal(self.cp, self.journal)

        errors: list[BaseException] = []

        def worker(wid: int) -> None:
            try:
                for step in range(10):
                    store.append(f"w{wid}", f"w{wid}-{step}", step, {"k": 1})
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [
            threading.Thread(target=worker, args=(wid,)) for wid in range(3)
        ]
        for thread in threads:
            thread.start()
        store.rotate_journal(self.cp2, self.journal2)
        for thread in threads:
            thread.join()

        self.assertEqual(errors, [])
        # Every commit landed either in the old or the new journal, and
        # the rotated pair alone recovers the final state exactly.
        restored = BranchStore.load_journal(self.cp2, self.journal2, None)
        self.assertEqual(restored.audit_log(), store.audit_log())
        for wid in range(3):
            self.assertEqual(
                restored.replay(f"w{wid}"), store.replay(f"w{wid}")
            )
        old_frames = _document(self.journal)["frames"]
        new_frames = _document(self.journal2)["frames"]
        self.assertEqual(len(old_frames) + len(new_frames), 30)
        self.assertEqual(
            [f["seq"] for f in new_frames],
            list(range(1, len(new_frames) + 1)),
        )


if __name__ == "__main__":
    unittest.main()
