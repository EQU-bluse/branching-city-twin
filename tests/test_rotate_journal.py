import hashlib
import json
import os
import tempfile
import threading
import unittest
from unittest import mock

from city_twin.branches import BranchStore
from city_twin.event_graph import EventGraph


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


def make_history() -> BranchStore:
    graph = EventGraph()
    graph.add("root", 0, (), {"seed": 7})
    store = BranchStore(graph)
    store.create("main", "root")
    store.create("feature", "root")
    store.append("main", "m1", 1, {"a": 1})
    store.append("feature", "f1", 2, {"b": 2})
    store.append("main", "m2", 3, {"a": 2})
    store.merge("main", "feature", "M1", 4, {"c": 1})
    return store


class RotateJournalTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.cp = os.path.join(self.tmp.name, "cp.json")
        self.journal = os.path.join(self.tmp.name, "journal.json")
        self.cp2 = os.path.join(self.tmp.name, "cp2.json")
        self.journal2 = os.path.join(self.tmp.name, "journal2.json")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def make_journaled_store(self) -> BranchStore:
        store = make_history()
        store.save_checkpoint(self.cp)
        store.enable_journal(self.cp, self.journal)
        store.append("main", "m3", 5, {"a": 3})
        store.create("side", "m2")
        store.append("side", "s1", 6, {"s": 1})
        return store


class RotateJournalTests(RotateJournalTestBase):
    def test_rotate_returns_none_and_creates_files(self) -> None:
        store = self.make_journaled_store()
        self.assertIsNone(store.rotate_journal(self.cp2, self.journal2))
        doc = _document(self.journal2)
        self.assertEqual(doc["schema_version"], 2)
        self.assertEqual(doc["frames"], [])
        self.assertEqual(doc["checkpoint"], _document(self.cp2)["checksum"])

    def test_new_checkpoint_restores_rotation_instant_state(self) -> None:
        store = self.make_journaled_store()
        store.rotate_journal(self.cp2, self.journal2)
        restored = BranchStore.load_checkpoint(self.cp2)
        self.assertEqual(restored.audit_log(), store.audit_log())
        for name in ("main", "feature", "side"):
            self.assertEqual(restored.replay(name), store.replay(name))
            self.assertEqual(restored.head(name), store.head(name))
        # Audit and idempotency/conflict behavior survive the rotation.
        self.assertEqual(
            restored.audit_merge("M1"), store.audit_merge("M1")
        )
        restored.append("main", "m3", 5, {"a": 3})
        with self.assertRaises(ValueError):
            restored.append("main", "m3", 5, {"a": 9})

    def test_new_journal_replays_from_one(self) -> None:
        store = self.make_journaled_store()
        store.rotate_journal(self.cp2, self.journal2)
        store.append("main", "m4", 7, {"a": 4})
        doc = _document(self.journal2)
        self.assertEqual(len(doc["frames"]), 1)
        self.assertEqual(doc["frames"][0]["seq"], 1)
        self.assertEqual(doc["frames"][0]["prev"], doc["checkpoint"])
        self.assertEqual(doc["frames"][0]["before"], doc["checkpoint"])
        self.assertEqual(doc["frames"][0]["op"], "append")
        # The old journal receives no new frames.
        self.assertEqual(len(_document(self.journal)["frames"]), 3)
        # The rotated pair fully recovers without the old files.
        restored = BranchStore.load_journal(self.cp2, self.journal2, None)
        self.assertEqual(restored.head("main"), "m4")
        self.assertEqual(restored.audit_log(), store.audit_log())

    def test_old_files_remain_byte_for_byte_unchanged(self) -> None:
        store = self.make_journaled_store()
        old_cp = _read(self.cp)
        old_journal = _read(self.journal)
        store.rotate_journal(self.cp2, self.journal2)
        store.append("main", "m4", 7, {"a": 4})
        self.assertEqual(_read(self.cp), old_cp)
        self.assertEqual(_read(self.journal), old_journal)
        # The old pair is independently recoverable as an archive.
        archived = BranchStore.load_journal(self.cp, self.journal, None)
        self.assertEqual(archived.head("side"), "s1")

    def test_rotate_empty_journal(self) -> None:
        store = make_history()
        store.save_checkpoint(self.cp)
        store.enable_journal(self.cp, self.journal)
        store.rotate_journal(self.cp2, self.journal2)
        restored = BranchStore.load_journal(self.cp2, self.journal2, None)
        self.assertEqual(restored.audit_log(), store.audit_log())
        self.assertEqual(restored.head("main"), store.head("main"))

    def test_repeated_rotation_chains(self) -> None:
        store = self.make_journaled_store()
        cp3 = os.path.join(self.tmp.name, "cp3.json")
        journal3 = os.path.join(self.tmp.name, "journal3.json")
        store.rotate_journal(self.cp2, self.journal2)
        store.append("main", "m4", 7, {"a": 4})
        store.rotate_journal(cp3, journal3)
        store.append("main", "m5", 8, {"a": 5})
        self.assertEqual(
            [f["seq"] for f in _document(journal3)["frames"]], [1]
        )
        restored = BranchStore.load_journal(cp3, journal3, None)
        self.assertEqual(restored.head("main"), "m5")
        self.assertEqual(restored.audit_log(), store.audit_log())

    def test_same_history_produces_identical_bytes(self) -> None:
        first = self.make_journaled_store()
        first.rotate_journal(self.cp2, self.journal2)

        second_cp = os.path.join(self.tmp.name, "bcp.json")
        second_journal = os.path.join(self.tmp.name, "bjournal.json")
        second_cp2 = os.path.join(self.tmp.name, "bcp2.json")
        second_journal2 = os.path.join(self.tmp.name, "bjournal2.json")
        second = make_history()
        second.save_checkpoint(second_cp)
        second.enable_journal(second_cp, second_journal)
        second.append("main", "m3", 5, {"a": 3})
        second.create("side", "m2")
        second.append("side", "s1", 6, {"s": 1})
        second.rotate_journal(second_cp2, second_journal2)
        self.assertEqual(_read(self.cp2), _read(second_cp2))
        self.assertEqual(_read(self.journal2), _read(second_journal2))


class RotateJournalValidationTests(RotateJournalTestBase):
    def test_path_validation_order(self) -> None:
        store = self.make_journaled_store()
        for bad in (1, 1.0, b"x", None, ["a"]):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    store.rotate_journal(bad, self.journal2)
                with self.assertRaises(TypeError):
                    store.rotate_journal(self.cp2, bad)
        with self.assertRaises(ValueError):
            store.rotate_journal("", self.journal2)
        with self.assertRaises(ValueError):
            store.rotate_journal(self.cp2, "")

    def test_same_path_for_both_targets_rejected(self) -> None:
        store = self.make_journaled_store()
        with self.assertRaises(ValueError):
            store.rotate_journal(self.cp2, self.cp2)

    def test_unattached_store_raises_runtime_error(self) -> None:
        store = make_history()
        store.save_checkpoint(self.cp)
        with self.assertRaises(RuntimeError):
            store.rotate_journal(self.cp2, self.journal2)
        # A historical (detached) prefix recovery also cannot rotate.
        store.enable_journal(self.cp, self.journal)
        store.append("main", "m3", 5, {"a": 3})
        detached = BranchStore.load_journal(self.cp, self.journal, 0)
        with self.assertRaises(RuntimeError):
            detached.rotate_journal(self.cp2, self.journal2)

    def test_runtime_error_precedes_target_existence(self) -> None:
        # Detached instance and both targets present: RuntimeError wins.
        store = make_history()
        store.save_checkpoint(self.cp)
        for target in (self.cp2, self.journal2):
            with open(target, "wb") as handle:
                handle.write(b"occupied")
        with self.assertRaises(RuntimeError):
            store.rotate_journal(self.cp2, self.journal2)

    def test_existing_targets_raise_os_error_and_are_not_overwritten(
        self,
    ) -> None:
        store = self.make_journaled_store()
        with open(self.cp2, "wb") as handle:
            handle.write(b"occupied-cp")
        with self.assertRaises(OSError):
            store.rotate_journal(self.cp2, self.journal2)
        self.assertEqual(_read(self.cp2), b"occupied-cp")
        self.assertFalse(os.path.exists(self.journal2))

        os.remove(self.cp2)
        with open(self.journal2, "wb") as handle:
            handle.write(b"occupied-journal")
        with self.assertRaises(OSError):
            store.rotate_journal(self.cp2, self.journal2)
        self.assertEqual(_read(self.journal2), b"occupied-journal")
        self.assertFalse(os.path.exists(self.cp2))
        # Store stays on the old journal.
        store.append("main", "m4", 7, {"a": 4})
        self.assertEqual(len(_document(self.journal)["frames"]), 4)

    def test_targets_in_missing_directory_raise_os_error(self) -> None:
        store = self.make_journaled_store()
        bad_cp = os.path.join(self.tmp.name, "no-such-dir", "cp.json")
        with self.assertRaises(OSError):
            store.rotate_journal(bad_cp, self.journal2)
        self.assertFalse(os.path.exists(self.journal2))
        bad_journal = os.path.join(self.tmp.name, "no-such-dir", "j.json")
        with self.assertRaises(OSError):
            store.rotate_journal(self.cp2, bad_journal)
        self.assertFalse(os.path.exists(self.cp2))

    def test_checkpoint_replaced_first_failure_unpublishes_it(self) -> None:
        store = self.make_journaled_store()
        journal_before = _read(self.journal)
        audit_before = store.audit_log()
        # The checkpoint publish succeeds; the journal publish fails, so
        # the already-published checkpoint must be removed again.
        real_link = os.link

        def flaky(src, dst, *args, **kwargs):
            if os.path.abspath(dst) == os.path.abspath(self.journal2):
                raise OSError("boom")
            return real_link(src, dst, *args, **kwargs)

        with mock.patch(
            "city_twin.branches.os.link", side_effect=flaky
        ):
            with self.assertRaises(OSError):
                store.rotate_journal(self.cp2, self.journal2)
        self.assertFalse(os.path.exists(self.cp2))
        self.assertFalse(os.path.exists(self.journal2))
        # Business state and the old attachment are untouched.
        self.assertEqual(store.audit_log(), audit_before)
        self.assertEqual(_read(self.journal), journal_before)
        store.append("main", "m4", 7, {"a": 4})
        self.assertEqual(len(_document(self.journal)["frames"]), 4)
        # A clean retry afterwards succeeds.
        store.rotate_journal(self.cp2, self.journal2)
        self.assertTrue(os.path.exists(self.cp2))
        self.assertTrue(os.path.exists(self.journal2))

    def test_target_occupied_mid_publish_is_never_overwritten(self) -> None:
        store = self.make_journaled_store()
        journal_before = _read(self.journal)
        # A racing writer creates the checkpoint target after the lstat
        # check but before the publish; the no-clobber publish must fail
        # with OSError and leave the racing content untouched.
        real_link = os.link

        def racy(src, dst, *args, **kwargs):
            if os.path.abspath(dst) == os.path.abspath(self.cp2):
                with open(dst, "wb") as handle:
                    handle.write(b"racing-writer")
            return real_link(src, dst, *args, **kwargs)

        with mock.patch(
            "city_twin.branches.os.link", side_effect=racy
        ):
            with self.assertRaises(OSError):
                store.rotate_journal(self.cp2, self.journal2)
        self.assertEqual(_read(self.cp2), b"racing-writer")
        self.assertFalse(os.path.exists(self.journal2))
        # The store stays on the old journal, byte-for-byte intact.
        self.assertEqual(_read(self.journal), journal_before)
        store.append("main", "m4", 7, {"a": 4})
        self.assertEqual(len(_document(self.journal)["frames"]), 4)

    def test_directory_fsync_failure_aborts_and_cleans_up(self) -> None:
        store = self.make_journaled_store()
        with mock.patch(
            "city_twin.branches.os.fsync", side_effect=OSError("boom")
        ):
            with self.assertRaises(OSError):
                store.rotate_journal(self.cp2, self.journal2)
        self.assertFalse(os.path.exists(self.cp2))
        self.assertFalse(os.path.exists(self.journal2))
        leftovers = [
            name
            for name in os.listdir(self.tmp.name)
            if name.endswith(".tmp")
        ]
        self.assertEqual(leftovers, [])
        # Still attached to the old journal.
        store.append("main", "m4", 7, {"a": 4})
        self.assertEqual(len(_document(self.journal)["frames"]), 4)

    def test_live_state_diverging_from_files_aborts(self) -> None:
        store = self.make_journaled_store()
        # Corrupt the bound checkpoint on disk after it was bound; the
        # rotation must detect that the files no longer describe the live
        # state and refuse (ValueError) without creating targets.
        divergent = make_history()
        divergent.append("main", "zzz", 99, {"z": 9})
        divergent.save_checkpoint(self.cp)
        with self.assertRaises(ValueError):
            store.rotate_journal(self.cp2, self.journal2)
        self.assertFalse(os.path.exists(self.cp2))
        self.assertFalse(os.path.exists(self.journal2))


class RotatedV1JournalTests(RotateJournalTestBase):
    def test_rotation_of_recovered_v1_emits_v2(self) -> None:
        store = self.make_journaled_store()
        # Convert the journal to version 1 by stripping summaries.
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
        with open(self.journal, "wb") as handle:
            handle.write(
                _canonical(
                    {
                        "schema_version": 1,
                        "checkpoint": doc["checkpoint"],
                        "frames": v1_frames,
                    }
                )
            )
        recovered = BranchStore.load_journal(self.cp, self.journal, None)
        recovered.rotate_journal(self.cp2, self.journal2)
        self.assertEqual(_document(self.journal2)["schema_version"], 2)
        recovered.append("main", "m4", 7, {"a": 4})
        doc2 = _document(self.journal2)
        self.assertEqual(
            list(doc2["frames"][0]),
            ["seq", "op", "params", "before", "after", "prev"],
        )


class RotateConcurrencyTests(RotateJournalTestBase):
    def test_rotation_blocks_concurrent_commits(self) -> None:
        store = self.make_journaled_store()

        # Block the rotation (which holds the state lock) at the first
        # fsync of the checkpoint temp file, and prove a concurrent
        # append cannot make progress until the rotation finishes.
        inside = threading.Event()
        proceed = threading.Event()
        real_fsync = os.fsync
        calls = {"n": 0}

        def barrier(fd) -> None:
            calls["n"] += 1
            if calls["n"] == 1:
                inside.set()
                proceed.wait(timeout=5)
            return real_fsync(fd)

        done = threading.Event()

        def do_rotate() -> None:
            store.rotate_journal(self.cp2, self.journal2)
            done.set()

        with mock.patch(
            "city_twin.branches.os.fsync", side_effect=barrier
        ):
            rotator = threading.Thread(target=do_rotate)
            rotator.start()
            self.assertTrue(inside.wait(timeout=5))

            appended = threading.Event()

            def do_append() -> None:
                store.append("main", "late", 90, {"a": 1})
                appended.set()

            committer = threading.Thread(target=do_append)
            committer.start()
            # The append cannot commit while rotation holds the lock.
            self.assertFalse(appended.wait(timeout=0.3))
            self.assertFalse(done.is_set())

            proceed.set()
            rotator.join(timeout=5)
            committer.join(timeout=5)

        self.assertTrue(done.is_set())
        self.assertTrue(appended.is_set())
        # The append that queued during rotation landed on the NEW
        # journal, with a fresh sequence starting at one.
        self.assertEqual(
            [f["op"] for f in _document(self.journal2)["frames"]],
            ["append"],
        )
        self.assertEqual(
            [f["seq"] for f in _document(self.journal2)["frames"]], [1]
        )
        restored = BranchStore.load_journal(self.cp2, self.journal2, None)
        self.assertEqual(restored.head("main"), "late")


if __name__ == "__main__":
    unittest.main()
