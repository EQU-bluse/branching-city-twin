"""Tests for cross-process recovery of interrupted dual-cache publishes.

The segmented build and the batch range-diff entry gain an optional
``recovery_path``. These tests cover:

* omitted ``recovery_path`` keeping the baseline call, result and
  authentication semantics (no record file is ever created);
* argument validation order and type/value errors, including the
  same-directory rule;
* a successful build leaving no record, backup or temp residue;
* an interrupted publish (index replaced, record at ``prepared``)
  being rolled back byte-for-byte by the next call;
* an interrupted publish with both targets replaced being completed
  (old backups and record removed) by the next call;
* an unparseable record, a checksum mismatch, a phase/target
  contradiction and a corrupt backup raising ``ValueError`` and
  preserving all forensic material;
* batch diffs with a recovery path matching the baseline results.
"""

import json
import os
import tempfile
import unittest
from unittest import mock

from city_twin.branches import BranchStore
from city_twin.event_graph import EventGraph


def _read(path: str) -> bytes:
    with open(path, "rb") as handle:
        return handle.read()


def make_store() -> BranchStore:
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


class RecoveryPublishTestBase(unittest.TestCase):
    SEGMENT_SIZE = 2

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = os.path.join(self.tmp.name, "gens")
        os.mkdir(self.root)
        self.cp = os.path.join(self.tmp.name, "cp.json")
        self.journal = os.path.join(self.tmp.name, "journal.json")
        self.chain = os.path.join(self.tmp.name, "chain.json")
        self.index = os.path.join(self.tmp.name, "index.json")
        self.progress = os.path.join(self.tmp.name, "progress.json")
        self.recovery = os.path.join(self.tmp.name, "recovery.json")
        store = make_store()
        store.save_checkpoint(self.cp)
        store.enable_journal(self.cp, self.journal)
        store.append("main", "m3", 5, {"a": 3})
        store.rotate_generation(self.root)
        self.store = store
        self.head: str | None = None

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def append_frame(self, previous: "str | None") -> str:
        return BranchStore.append_recovery_audit_chain(
            self.root, self.chain, previous
        )

    def grow_chain(self, count: int, start: int = 0) -> str:
        head = self.head
        for i in range(count):
            head = self.append_frame(head)
            if i + 1 < count:
                self.store.append(
                    "main", f"g{start + i}", 100 + start + i,
                    {"a": start + i},
                )
                self.store.rotate_generation(self.root)
        self.head = head
        return head

    def build(self, *args: object) -> str:
        return BranchStore.build_recovery_audit_segments(  # type: ignore[return-value]
            self.chain, self.index, self.SEGMENT_SIZE, *args
        )

    def diff(self, ranges, *args: object):
        return BranchStore.diff_recovery_audit_ranges(
            self.chain,
            self.head,
            tuple(ranges),
            self.index,
            self.SEGMENT_SIZE,
            *args,
        )

    def residue(self) -> list:
        return [
            name
            for name in os.listdir(self.tmp.name)
            if name.startswith(".recovery-")
        ]

    def root_snapshot(self) -> dict:
        captured = {}
        for dirpath, _dirnames, filenames in os.walk(self.root):
            for filename in filenames:
                path = os.path.join(dirpath, filename)
                captured[path] = _read(path)
        return captured

    def crash_next_build(self, fail_phases):
        """Patch the record writer so the next build dies (like a
        process kill) when publishing the given phases, and patch the
        inline recovery away so the interrupted state stays on disk."""
        real_write = BranchStore._write_recovery_record_locked

        def flaky(cls, recovery_path, phase, targets):
            if phase in fail_phases:
                raise OSError("simulated crash")
            return real_write(recovery_path, phase, targets)

        writer = mock.patch.object(
            BranchStore,
            "_write_recovery_record_locked",
            classmethod(flaky),
        )
        recover = mock.patch.object(
            BranchStore,
            "_recover_cache_publish_locked",
            classmethod(lambda cls, *args: None),
        )
        return writer, recover


class ValidationTests(RecoveryPublishTestBase):
    def test_non_str_recovery_path_raises_type_error(self) -> None:
        self.grow_chain(4)
        with self.assertRaises(TypeError):
            self.build(self.progress, 123)
        with self.assertRaises(TypeError):
            self.diff([(1, 2)], self.progress, 1.5)

    def test_empty_recovery_path_raises_value_error(self) -> None:
        self.grow_chain(4)
        with self.assertRaises(ValueError):
            self.build(self.progress, "")
        with self.assertRaises(ValueError):
            self.diff([(1, 2)], self.progress, "")

    def test_recovery_path_requires_progress_path(self) -> None:
        self.grow_chain(4)
        with self.assertRaises(ValueError):
            self.build(None, self.recovery)
        with self.assertRaises(ValueError):
            self.diff([(1, 2)], None, self.recovery)

    def test_recovery_path_must_not_alias_chain_index_or_progress(
        self,
    ) -> None:
        self.grow_chain(4)
        for forbidden in (self.chain, self.index, self.progress):
            with self.assertRaises(ValueError):
                self.build(self.progress, forbidden)
            with self.assertRaises(ValueError):
                self.diff([(1, 2)], self.progress, forbidden)

    def test_recovery_path_must_share_the_cache_directory(self) -> None:
        self.grow_chain(4)
        other = os.path.join(self.tmp.name, "elsewhere")
        os.mkdir(other)
        remote = os.path.join(other, "recovery.json")
        with self.assertRaises(ValueError):
            self.build(self.progress, remote)
        with self.assertRaises(ValueError):
            self.diff([(1, 2)], self.progress, remote)
        remote_index = os.path.join(other, "index.json")
        with self.assertRaises(ValueError):
            BranchStore.build_recovery_audit_segments(
                self.chain,
                remote_index,
                self.SEGMENT_SIZE,
                self.progress,
                self.recovery,
            )

    def test_recovery_validated_after_existing_arguments(self) -> None:
        self.grow_chain(4)
        # A bad segment size is rejected before the recovery type.
        with self.assertRaises(TypeError):
            BranchStore.build_recovery_audit_segments(
                self.chain, self.index, "x", self.progress, 123
            )
        with self.assertRaises(ValueError):
            BranchStore.build_recovery_audit_segments(
                self.chain, self.index, 0, self.progress, 123
            )
        # A bad progress path is rejected before the recovery path.
        with self.assertRaises(TypeError):
            BranchStore.build_recovery_audit_segments(
                self.chain, self.index, self.SEGMENT_SIZE, 7, 123
            )


class SuccessfulPublishTests(RecoveryPublishTestBase):
    def test_build_with_recovery_leaves_no_residue(self) -> None:
        head = self.grow_chain(5)
        before = self.root_snapshot()
        result = self.build(self.progress, self.recovery)
        self.assertEqual(result, head)
        self.assertFalse(os.path.lexists(self.recovery))
        self.assertEqual(self.residue(), [])
        self.assertEqual(self.root_snapshot(), before)
        # The caches are the same bytes a recovery-less build produces.
        index_bytes = _read(self.index)
        progress_bytes = _read(self.progress)
        os.remove(self.index)
        os.remove(self.progress)
        self.assertEqual(self.build(self.progress), head)
        self.assertEqual(_read(self.index), index_bytes)
        self.assertEqual(_read(self.progress), progress_bytes)

    def test_omitted_recovery_never_creates_a_record(self) -> None:
        head = self.grow_chain(5)
        self.assertEqual(self.build(self.progress), head)
        self.assertFalse(os.path.lexists(self.recovery))
        self.assertEqual(self.residue(), [])

    def test_resume_and_diff_with_recovery(self) -> None:
        self.grow_chain(4)
        self.build(self.progress, self.recovery)
        head = self.grow_chain(4, start=4)
        # The appended chain resumes across the authenticated prefix.
        self.assertEqual(self.build(self.progress, self.recovery), head)
        self.assertEqual(self.residue(), [])
        ranges = [(1, 3), (2, 8), (5, 5)]
        with_recovery = self.diff(ranges, self.progress, self.recovery)
        baseline = self.diff(ranges, self.progress)
        self.assertEqual(with_recovery, baseline)
        self.assertEqual(len(with_recovery), 3)
        self.assertEqual(with_recovery[2]["changes"], ())
        self.assertEqual(self.residue(), [])
        # Result dictionaries are detached between calls.
        again = self.diff(ranges, self.progress, self.recovery)
        self.assertEqual(again, with_recovery)
        self.assertIsNot(again[0], with_recovery[0])
        self.assertIsNot(again[0]["changes"], with_recovery[0]["changes"])


class InterruptedPublishTests(RecoveryPublishTestBase):
    def test_crash_before_index_record_rolls_back(self) -> None:
        self.grow_chain(4)
        self.build(self.progress, self.recovery)
        old_index = _read(self.index)
        old_progress = _read(self.progress)
        head = self.grow_chain(2, start=4)
        writer, recover = self.crash_next_build({"index"})
        with writer, recover:
            with self.assertRaises(OSError):
                self.build(self.progress, self.recovery)
        # The interrupted publish left the new index, the old progress,
        # durable backups and a record at 'prepared'.
        self.assertNotEqual(_read(self.index), old_index)
        self.assertEqual(_read(self.progress), old_progress)
        record = json.loads(_read(self.recovery).decode("utf-8"))
        self.assertEqual(record["phase"], "prepared")
        backups = [
            name
            for name in os.listdir(self.tmp.name)
            if name.startswith(".recovery-cache-backup-")
        ]
        self.assertEqual(len(backups), 2)
        # The next call rolls the index back byte-for-byte, then
        # rebuilds and publishes the new version cleanly.
        self.assertEqual(self.build(self.progress, self.recovery), head)
        self.assertFalse(os.path.lexists(self.recovery))
        self.assertEqual(self.residue(), [])

    def test_rollback_restores_old_version_byte_for_byte(self) -> None:
        self.grow_chain(4)
        self.build(self.progress, self.recovery)
        old_index = _read(self.index)
        old_progress = _read(self.progress)
        self.grow_chain(2, start=4)
        writer, recover = self.crash_next_build({"index"})
        with writer, recover:
            with self.assertRaises(OSError):
                self.build(self.progress, self.recovery)
        BranchStore._recover_cache_publish_locked(
            self.index, self.progress, self.recovery
        )
        self.assertEqual(_read(self.index), old_index)
        self.assertEqual(_read(self.progress), old_progress)
        self.assertFalse(os.path.lexists(self.recovery))
        self.assertEqual(self.residue(), [])

    def test_crash_after_both_replaced_completes(self) -> None:
        self.grow_chain(4)
        self.build(self.progress, self.recovery)
        head = self.grow_chain(2, start=4)
        writer, recover = self.crash_next_build({"progress"})
        with writer, recover:
            with self.assertRaises(OSError):
                self.build(self.progress, self.recovery)
        record = json.loads(_read(self.recovery).decode("utf-8"))
        self.assertEqual(record["phase"], "index")
        new_index = _read(self.index)
        new_progress = _read(self.progress)
        # Both targets already hold the new bytes: the next call
        # completes the publish, dropping the backups and the record.
        BranchStore._recover_cache_publish_locked(
            self.index, self.progress, self.recovery
        )
        self.assertEqual(_read(self.index), new_index)
        self.assertEqual(_read(self.progress), new_progress)
        self.assertFalse(os.path.lexists(self.recovery))
        self.assertEqual(self.residue(), [])
        # The completed caches are fresh, so a query answers without
        # any further publication.
        result = self.diff([(1, 6)], self.progress, self.recovery)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["start"], 1)
        self.assertEqual(result[0]["end"], 6)
        self.assertEqual(self.residue(), [])

    def test_failed_publish_recovers_inline(self) -> None:
        self.grow_chain(4)
        self.build(self.progress, self.recovery)
        old_index = _read(self.index)
        self.grow_chain(2, start=4)
        # A genuine I/O failure mid-publish (recovery not patched out)
        # rolls the interrupted publish back before surfacing.
        real_write = BranchStore._write_recovery_record_locked

        def flaky(cls, recovery_path, phase, targets):
            if phase == "index":
                raise OSError("simulated failure")
            return real_write(recovery_path, phase, targets)

        with mock.patch.object(
            BranchStore,
            "_write_recovery_record_locked",
            classmethod(flaky),
        ):
            with self.assertRaises(OSError):
                self.build(self.progress, self.recovery)
        self.assertEqual(_read(self.index), old_index)
        self.assertFalse(os.path.lexists(self.recovery))
        self.assertEqual(self.residue(), [])


class ForensicPreservationTests(RecoveryPublishTestBase):
    def test_unparseable_record_raises_and_is_preserved(self) -> None:
        self.grow_chain(4)
        self.build(self.progress, self.recovery)
        with open(self.recovery, "wb") as handle:
            handle.write(b'{"format":')
        with self.assertRaises(ValueError):
            self.build(self.progress, self.recovery)
        self.assertEqual(_read(self.recovery), b'{"format":')
        with self.assertRaises(ValueError):
            self.diff([(1, 2)], self.progress, self.recovery)
        self.assertEqual(_read(self.recovery), b'{"format":')

    def test_checksum_mismatch_raises_and_is_preserved(self) -> None:
        self.grow_chain(4)
        self.build(self.progress, self.recovery)
        targets = {
            name: {
                "old_exists": False,
                "old": None,
                "new_exists": True,
                "new": "0" * 64,
                "backup": None,
            }
            for name in ("index", "progress")
        }
        data = bytearray(
            BranchStore._recovery_record_bytes("prepared", targets)
        )
        data[-3] = ord("1") if data[-3] != ord("1") else ord("0")
        with open(self.recovery, "wb") as handle:
            handle.write(bytes(data))
        with self.assertRaises(ValueError):
            self.build(self.progress, self.recovery)
        self.assertEqual(_read(self.recovery), bytes(data))

    def test_contradictory_phase_raises_and_is_preserved(self) -> None:
        self.grow_chain(4)
        self.build(self.progress, self.recovery)
        # A record claiming the publish finished while both targets
        # still hold the old bytes contradicts every recoverable phase.
        targets = {
            name: {
                "old_exists": True,
                "old": BranchStore._sha256_file(getattr(self, name)),
                "new_exists": True,
                "new": "1" * 64,
                "backup": "missing-backup.tmp",
            }
            for name in ("index", "progress")
        }
        data = BranchStore._recovery_record_bytes("progress", targets)
        with open(self.recovery, "wb") as handle:
            handle.write(data)
        with self.assertRaises(ValueError):
            self.build(self.progress, self.recovery)
        self.assertEqual(_read(self.recovery), data)

    def test_corrupt_backup_raises_and_is_preserved(self) -> None:
        self.grow_chain(4)
        self.build(self.progress, self.recovery)
        self.grow_chain(2, start=4)
        writer, recover = self.crash_next_build({"index"})
        with writer, recover:
            with self.assertRaises(OSError):
                self.build(self.progress, self.recovery)
        record = json.loads(_read(self.recovery).decode("utf-8"))
        backup = os.path.join(
            self.tmp.name, record["targets"]["index"]["backup"]
        )
        corrupted = b"x" * 64
        with open(backup, "wb") as handle:
            handle.write(corrupted)
        with self.assertRaises(ValueError):
            self.build(self.progress, self.recovery)
        # Everything needed for a later diagnosis is still on disk.
        self.assertEqual(_read(backup), corrupted)
        self.assertTrue(os.path.lexists(self.recovery))


if __name__ == "__main__":
    unittest.main()
