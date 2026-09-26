"""Tests for cross-process recovery of interrupted two-cache publications.

When a segmented build or batch range diff is given a ``recovery_path``
in addition to its progress cursor, the index and progress targets are
published together under a durable crash protocol: complete backups of
both old targets, a checksummed recovery record made durable before the
first replacement and advanced after each replacement, and cleanup of
the backups and record only after the complete new version is on disk.

These tests cover:

* omitting ``recovery_path`` keeping the baseline call, result and
  authentication semantics, and a successful publication leaving no
  record or backup behind;
* a process dying at every phase (record prepared, index replaced,
  progress replaced, cleanup) being finished or rolled back by the next
  entry-point call under the chain lock, with old targets restored
  byte-for-byte (or to recorded absence) and the new version committed
  only when both targets match its digests;
* no success and no incremental scan ever starting from a mixed
  index/progress version;
* malformed records, digest/phase contradictions and corrupt backups
  raising :class:`ValueError` with all evidence retained;
* unreachable phase/digest combinations -- a ``prepared`` record
  coexisting with new-version progress bytes -- never committing or
  rolling back, and missing or corrupt required backups raising
  :class:`ValueError` with record, targets, backups and temps kept;
* commit and rollback cleanup removing only protocol-owned temp files
  and never other directory entries;
* identical republications under a lagging record phase recovering
  cleanly, and a disturbed target without a durable record raising
  :class:`ValueError`;
* missing directories and failed restore/cleanup/sync steps raising
  :class:`OSError` without swallowing files later recovery needs;
* argument validation order, type/value errors, aliasing and the
  shared-directory rule;
* batch results staying item-for-item identical to the single-range
  interface with detached per-layer copies;
* concurrent callers observing only complete versions and never the
  canonical chain or generation state.
"""

import glob
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


def _sha(path: str) -> str:
    return hashlib.sha256(_read(path)).hexdigest()


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


class RecoveryPublicationTestBase(unittest.TestCase):
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

    def build(self, recovery: object = None) -> str:
        if recovery is None:
            recovery = self.recovery
        return BranchStore.build_recovery_audit_segments(  # type: ignore[return-value]
            self.chain,
            self.index,
            self.SEGMENT_SIZE,
            self.progress,
            recovery,
        )

    def diff(self, ranges, recovery: object = None):
        if recovery is None:
            recovery = self.recovery
        return BranchStore.diff_recovery_audit_ranges(
            self.chain,
            self.head,
            tuple(ranges),
            self.index,
            self.SEGMENT_SIZE,
            self.progress,
            recovery,
        )

    def residue(self) -> list:
        return sorted(
            name
            for name in os.listdir(self.tmp.name)
            if name.startswith(".recovery-")
        )

    def root_snapshot(self) -> dict:
        captured = {}
        for dirpath, _dirnames, filenames in os.walk(self.root):
            for filename in filenames:
                path = os.path.join(dirpath, filename)
                captured[path] = _read(path)
        return captured

    def kill_publication_at(self, point: str) -> None:
        """Mimic a process that dies during publication.

        The publication raises :class:`OSError` at ``point`` (one of
        ``index_replace``, ``progress_replace``, ``record_after_index``,
        ``record_after_progress`` or ``cleanup``) and the failure
        handler's convergence is killed as well, so the durable recovery
        record and backups are left exactly as a dead process would
        leave them. Orphaned build temps -- which a real kill also
        leaves and are not recovery material -- are removed afterwards.
        """
        real_replace = os.replace
        real_remove = os.remove
        real_recover = BranchStore._recover_cache_publication_locked
        real_write_record = BranchStore._write_cache_recovery_record

        def flaky_replace(src, dst):
            if (
                point == "index_replace"
                and os.path.realpath(dst) == os.path.realpath(self.index)
            ):
                raise OSError("simulated death at index replace")
            if (
                point == "progress_replace"
                and os.path.realpath(dst) == os.path.realpath(self.progress)
            ):
                raise OSError("simulated death at progress replace")
            return real_replace(src, dst)

        def flaky_remove(path):
            if (
                point == "cleanup"
                and os.path.basename(path).startswith(
                    ".recovery-cache-backup-"
                )
            ):
                raise OSError("simulated death during backup cleanup")
            return real_remove(path)

        def flaky_write_record(recovery_path, data):
            phase = json.loads(data.decode("utf-8"))["phase"]
            if (
                point == "record_after_index" and phase == "index"
            ) or (
                point == "record_after_progress" and phase == "progress"
            ):
                raise OSError("simulated death before the phase record")
            return real_write_record(recovery_path, data)

        def dead_recover(index_path, progress_path, recovery_path):
            if os.path.exists(recovery_path):
                raise OSError("simulated process death before convergence")
            return real_recover(index_path, progress_path, recovery_path)

        with mock.patch.object(
            BranchStore,
            "_recover_cache_publication_locked",
            staticmethod(dead_recover),
        ), mock.patch(
            "city_twin.branches.os.replace", flaky_replace
        ), mock.patch(
            "city_twin.branches.os.remove", flaky_remove
        ), mock.patch.object(
            BranchStore,
            "_write_cache_recovery_record",
            staticmethod(flaky_write_record),
        ):
            with self.assertRaises(OSError):
                self.build()
        for pattern in (
            ".recovery-chain-segments-*.tmp",
            ".recovery-chain-progress-*.tmp",
            ".recovery-cache-record-*.tmp",
        ):
            for leftover in glob.glob(os.path.join(self.tmp.name, pattern)):
                os.remove(leftover)

    def recover_now(self) -> None:
        BranchStore._recover_cache_publication_locked(
            self.index, self.progress, self.recovery
        )

    def record_document(self) -> dict:
        return json.loads(_read(self.recovery).decode("utf-8"))


class SuccessfulPublicationTests(RecoveryPublicationTestBase):
    def test_successful_build_leaves_no_recovery_material(self) -> None:
        head = self.grow_chain(5)
        self.assertEqual(self.build(), head)
        self.assertTrue(os.path.exists(self.index))
        self.assertTrue(os.path.exists(self.progress))
        self.assertFalse(os.path.exists(self.recovery))
        self.assertEqual(self.residue(), [])

    def test_repeated_builds_are_stable(self) -> None:
        head = self.grow_chain(6)
        self.assertEqual(self.build(), head)
        index_before = _read(self.index)
        progress_before = _read(self.progress)
        # A query with a recovery path authenticates everything and
        # leaves the caches untouched.
        self.assertEqual(self.diff(((5, 6),)), self.diff(((5, 6),)))
        self.assertEqual(_read(self.index), index_before)
        self.assertEqual(_read(self.progress), progress_before)
        self.assertFalse(os.path.exists(self.recovery))
        self.assertEqual(self.residue(), [])

    def test_batch_results_match_single_range_interface(self) -> None:
        self.grow_chain(8)
        self.build()
        head = self.grow_chain(3, start=10)
        ranges = ((7, 10), (8, 9), (10, 10), (7, 7))
        results = self.diff(ranges)
        self.assertEqual(
            [(r["start"], r["end"]) for r in results], list(ranges)
        )
        for result, (start, end) in zip(results, ranges):
            expected = BranchStore.diff_recovery_audit_range(
                self.chain, head, start, end
            )
            self.assertEqual(result["changes"], expected)

    def test_batch_results_are_detached_across_calls(self) -> None:
        self.grow_chain(8)
        self.build()
        first = self.diff(((1, 8),))
        second = self.diff(((1, 8),))
        self.assertEqual(first, second)
        self.assertIsNot(first, second)
        self.assertIsNot(first[0], second[0])

    def test_record_shape_is_compact_and_checksummed(self) -> None:
        self.grow_chain(4)
        self.build()
        self.grow_chain(3, start=10)
        captured: dict[str, bytes] = {}
        orig_write = BranchStore._write_cache_recovery_record

        def spy(recovery_path, data):
            captured.setdefault("data", bytes(data))
            return orig_write(recovery_path, data)

        with mock.patch.object(
            BranchStore,
            "_write_cache_recovery_record",
            staticmethod(spy),
        ):
            self.build()
        raw = captured["data"]
        self.assertFalse(raw.startswith(b"\xef\xbb\xbf"))
        self.assertFalse(raw.endswith(b"\n"))
        self.assertNotIn(b"\n", raw)
        self.assertNotIn(b" ", raw)
        document = json.loads(raw.decode("utf-8"))
        self.assertEqual(
            set(document),
            {"format", "version", "phase", "index", "progress", "checksum"},
        )
        self.assertEqual(document["format"], "branching-city-twin/recovery-cache-publication")
        self.assertEqual(document["version"], 1)
        self.assertEqual(document["phase"], "prepared")
        checksum = document.pop("checksum")
        expected = hashlib.sha256(
            BranchStore._canonical_json(document).encode("utf-8")
        ).hexdigest()
        self.assertEqual(checksum, expected)
        for name in ("index", "progress"):
            target = document[name]
            self.assertEqual(
                set(target),
                {
                    "old_exists",
                    "old_digest",
                    "new_exists",
                    "new_digest",
                    "backup",
                },
            )
            self.assertTrue(target["old_exists"])
            self.assertTrue(target["new_exists"])
            self.assertEqual(len(target["old_digest"]), 64)
            self.assertEqual(len(target["new_digest"]), 64)
            self.assertTrue(target["backup"].startswith(".recovery-cache-backup-"))
            self.assertNotIn("/", target["backup"])


class OmittedRecoveryPathTests(RecoveryPublicationTestBase):
    def test_none_keeps_baseline_behavior(self) -> None:
        head = self.grow_chain(4)
        result = BranchStore.build_recovery_audit_segments(
            self.chain, self.index, self.SEGMENT_SIZE, self.progress, None
        )
        self.assertEqual(result, head)
        self.assertFalse(os.path.exists(self.recovery))
        self.assertEqual(
            BranchStore.diff_recovery_audit_ranges(
                self.chain,
                head,
                (),
                self.index,
                self.SEGMENT_SIZE,
                self.progress,
                None,
            ),
            (),
        )
        self.assertFalse(os.path.exists(self.recovery))


class InterruptedPublicationTests(RecoveryPublicationTestBase):
    def setUp(self) -> None:
        super().setUp()
        self.grow_chain(5)
        self.build()
        self.old_index = _read(self.index)
        self.old_progress = _read(self.progress)
        self.grow_chain(3, start=20)
        self.new_head = self.head

    def _assert_old_version(self) -> None:
        self.assertEqual(_read(self.index), self.old_index)
        self.assertEqual(_read(self.progress), self.old_progress)

    def test_death_at_index_replace_prepared_phase(self) -> None:
        self.kill_publication_at("index_replace")
        self.assertTrue(os.path.exists(self.recovery))
        self.assertEqual(self.record_document()["phase"], "prepared")
        # Nothing was replaced yet.
        self._assert_old_version()
        backups = [
            name
            for name in os.listdir(self.tmp.name)
            if name.startswith(".recovery-cache-backup-")
        ]
        self.assertEqual(len(backups), 2)
        self.recover_now()
        self._assert_old_version()
        self.assertFalse(os.path.exists(self.recovery))
        self.assertEqual(self.residue(), [])
        # The next entry-point call rebuilds the new version normally.
        self.assertEqual(self.build(), self.new_head)

    def test_death_at_progress_replace_rolls_new_index_back(self) -> None:
        self.kill_publication_at("progress_replace")
        self.assertEqual(self.record_document()["phase"], "index")
        # Mixed versions on disk before recovery.
        self.assertNotEqual(_read(self.index), self.old_index)
        self.assertEqual(_read(self.progress), self.old_progress)
        self.recover_now()
        self._assert_old_version()
        self.assertFalse(os.path.exists(self.recovery))
        self.assertEqual(self.residue(), [])
        self.assertEqual(self.build(), self.new_head)
        self.assertEqual(self.residue(), [])

    def test_death_during_cleanup_commits_new_version(self) -> None:
        self.kill_publication_at("cleanup")
        self.assertEqual(self.record_document()["phase"], "progress")
        new_index = _read(self.index)
        new_progress = _read(self.progress)
        self.assertNotEqual(new_index, self.old_index)
        self.assertNotEqual(new_progress, self.old_progress)
        backups = [
            name
            for name in os.listdir(self.tmp.name)
            if name.startswith(".recovery-cache-backup-")
        ]
        self.assertEqual(len(backups), 2)
        # Both targets carry the recorded new version: finish it.
        self.recover_now()
        self.assertEqual(_read(self.index), new_index)
        self.assertEqual(_read(self.progress), new_progress)
        self.assertFalse(os.path.exists(self.recovery))
        self.assertEqual(self.residue(), [])

    def test_death_between_index_replace_and_its_record(self) -> None:
        # The index is new and the record still says "prepared": the
        # digest evidence -- not the lagging phase -- decides rollback.
        self.kill_publication_at("record_after_index")
        self.assertEqual(self.record_document()["phase"], "prepared")
        self.assertNotEqual(_read(self.index), self.old_index)
        self.assertEqual(_read(self.progress), self.old_progress)
        self.recover_now()
        self._assert_old_version()
        self.assertFalse(os.path.exists(self.recovery))
        self.assertEqual(self.residue(), [])
        self.assertEqual(self.build(), self.new_head)

    def test_death_between_progress_replace_and_its_record(self) -> None:
        # Both targets are new while the record still says "index": the
        # next process commits the complete new version.
        self.kill_publication_at("record_after_progress")
        self.assertEqual(self.record_document()["phase"], "index")
        new_index = _read(self.index)
        new_progress = _read(self.progress)
        self.recover_now()
        self.assertEqual(_read(self.index), new_index)
        self.assertEqual(_read(self.progress), new_progress)
        self.assertFalse(os.path.exists(self.recovery))
        self.assertEqual(self.residue(), [])

    def test_interrupted_rollback_is_finished_to_old_version(self) -> None:
        # Start from a complete new publication that died before
        # cleanup, then restore just the index from its backup as a
        # previous recovery attempt might have done before dying: the
        # progress target is new while the index is already old.
        self.kill_publication_at("record_after_progress")
        record = self.record_document()
        index_backup = os.path.join(
            self.tmp.name, record["index"]["backup"]
        )
        old_index_via_backup = _read(index_backup)
        os.replace(index_backup, self.index)
        self._assert_old_index_after_partial_rollback(old_index_via_backup)
        self.recover_now()
        self._assert_old_version()
        self.assertFalse(os.path.exists(self.recovery))
        self.assertEqual(self.residue(), [])

    def _assert_old_index_after_partial_rollback(self, old_index: bytes) -> None:
        self.assertEqual(_read(self.index), old_index)
        self.assertNotEqual(_read(self.progress), self.old_progress)

    def test_entry_point_auto_recovers_then_serves_tail_diffs(self) -> None:
        self.kill_publication_at("progress_replace")
        # The next process enters through the batch interface; recovery
        # runs under the lock before any cache is read, then the appended
        # tail is scanned once and the answers match the single-range API.
        calls = {"resume": 0, "full": 0}
        resume_orig = BranchStore._scan_chain_resuming
        full_orig = BranchStore._scan_recovery_chain

        def resume_wrap(*args, **kwargs):
            calls["resume"] += 1
            return resume_orig(*args, **kwargs)

        def full_wrap(*args, **kwargs):
            calls["full"] += 1
            return full_orig(*args, **kwargs)

        with mock.patch.object(
            BranchStore, "_scan_chain_resuming", staticmethod(resume_wrap)
        ), mock.patch.object(
            BranchStore, "_scan_recovery_chain", staticmethod(full_wrap)
        ):
            results = self.diff(((7, 8),))
        self.assertEqual(calls["resume"], 1)
        self.assertEqual(calls["full"], 0)
        expected = BranchStore.diff_recovery_audit_range(
            self.chain, self.new_head, 7, 8
        )
        self.assertEqual(results[0]["changes"], expected)
        self.assertFalse(os.path.exists(self.recovery))
        self.assertEqual(self.residue(), [])

    def test_mixed_version_never_serves_prefix_query_from_cache(self) -> None:
        self.kill_publication_at("progress_replace")
        calls = {"resume": 0, "full": 0}
        resume_orig = BranchStore._scan_chain_resuming
        full_orig = BranchStore._scan_recovery_chain

        def resume_wrap(*args, **kwargs):
            calls["resume"] += 1
            return resume_orig(*args, **kwargs)

        def full_wrap(*args, **kwargs):
            calls["full"] += 1
            return full_orig(*args, **kwargs)

        with mock.patch.object(
            BranchStore, "_scan_chain_resuming", staticmethod(resume_wrap)
        ), mock.patch.object(
            BranchStore, "_scan_recovery_chain", staticmethod(full_wrap)
        ):
            results = self.diff(((2, 8),))
        # Endpoint 2 sits inside the recorded boundary: after rollback a
        # full authenticated scan is required, no incremental answer can
        # come from the mixed version that crashed.
        self.assertGreaterEqual(calls["full"], 1)
        self.assertEqual(calls["resume"], 0)
        expected = BranchStore.diff_recovery_audit_range(
            self.chain, self.new_head, 2, 8
        )
        self.assertEqual(results[0]["changes"], expected)
        self.assertFalse(os.path.exists(self.recovery))

    def test_recovery_preserves_generations_and_chain(self) -> None:
        snapshot = self.root_snapshot()
        chain_before = _read(self.chain)
        self.kill_publication_at("progress_replace")
        self.recover_now()
        self.build()
        self.diff(((7, 8),))
        self.assertEqual(_read(self.chain), chain_before)
        self.assertEqual(self.root_snapshot(), snapshot)
        self.assertEqual(self.residue(), [])


class FirstPublicationAbsenceTests(RecoveryPublicationTestBase):
    def test_absent_old_targets_are_restored_to_absence(self) -> None:
        self.grow_chain(4)
        # No caches exist yet; the first publication has both targets
        # recorded as previously absent.
        self.kill_publication_at("progress_replace")
        record = self.record_document()
        self.assertFalse(record["index"]["old_exists"])
        self.assertFalse(record["progress"]["old_exists"])
        # The index replacement landed before the crash.
        self.assertTrue(os.path.exists(self.index))
        self.recover_now()
        self.assertFalse(os.path.exists(self.index))
        self.assertFalse(os.path.exists(self.progress))
        self.assertFalse(os.path.exists(self.recovery))
        self.assertEqual(self.residue(), [])
        # A fresh process can now perform the initial build.
        head = self.head
        self.assertEqual(self.build(), head)
        self.assertTrue(os.path.exists(self.index))
        self.assertTrue(os.path.exists(self.progress))


class RecordAndEvidenceFailureTests(RecoveryPublicationTestBase):
    def setUp(self) -> None:
        super().setUp()
        self.grow_chain(5)
        self.build()
        self.grow_chain(3, start=20)
        self.kill_publication_at("progress_replace")
        self.good_record = _read(self.recovery)

    def _rebuild_record(self, mutate) -> None:
        document = json.loads(self.good_record.decode("utf-8"))
        mutate(document)
        body = {
            key: value
            for key, value in document.items()
            if key != "checksum"
        }
        document["checksum"] = hashlib.sha256(
            BranchStore._canonical_json(body).encode("utf-8")
        ).hexdigest()
        with open(self.recovery, "wb") as handle:
            handle.write(BranchStore._canonical_json(document).encode("utf-8"))

    def _assert_value_error_keeps_evidence(self) -> None:
        with self.assertRaises(ValueError):
            self.recover_now()
        # Record, targets and backups all remain for forensics.
        self.assertTrue(os.path.exists(self.recovery))
        backups = [
            name
            for name in os.listdir(self.tmp.name)
            if name.startswith(".recovery-cache-backup-")
        ]
        self.assertTrue(backups)

    def test_unparseable_record(self) -> None:
        for raw in (
            b"",
            b"garbage",
            b"\xef\xbb\xbf" + _read(self.recovery),
            _read(self.recovery) + b"\n",
        ):
            with self.subTest(raw=raw[:12]):
                with open(self.recovery, "wb") as handle:
                    handle.write(raw)
                self._assert_value_error_keeps_evidence()

    def test_bad_checksum(self) -> None:
        raw = _read(self.recovery)
        document = json.loads(raw.decode("utf-8"))
        document["checksum"] = "1" * 64
        with open(self.recovery, "wb") as handle:
            handle.write(BranchStore._canonical_json(document).encode("utf-8"))
        self._assert_value_error_keeps_evidence()

    def test_bad_fields_and_values(self) -> None:
        def change_format(doc):
            doc["format"] = "something-else"

        def change_version(doc):
            doc["version"] = 2

        def change_phase(doc):
            doc["phase"] = "halfway"

        def drop_key(doc):
            del doc["progress"]

        for mutate in (change_format, change_version, change_phase, drop_key):
            with self.subTest(mutate=mutate.__name__):
                self._rebuild_record(mutate)
                self._assert_value_error_keeps_evidence()
                # Restore the original interrupted record; targets and
                # backups are untouched by a failed recovery.
                with open(self.recovery, "wb") as handle:
                    handle.write(self.good_record)

    def test_target_with_unknown_bytes_is_contradiction(self) -> None:
        # The progress target currently holds its old bytes; replace it
        # with bytes matching neither recorded digest.
        with open(self.progress, "wb") as handle:
            handle.write(b"neither old nor new content")
        self._assert_value_error_keeps_evidence()
        self.assertEqual(_read(self.progress), b"neither old nor new content")

    def test_recorded_existing_target_going_missing_is_contradiction(self) -> None:
        os.remove(self.progress)
        self._assert_value_error_keeps_evidence()

    def test_corrupt_backup_content(self) -> None:
        record = self.record_document()
        backup_name = record["index"]["backup"]
        with open(os.path.join(self.tmp.name, backup_name), "wb") as handle:
            handle.write(b"corrupt backup bytes")
        self._assert_value_error_keeps_evidence()

    def test_rollback_failure_raises_oserror_and_keeps_record(self) -> None:
        real_replace = os.replace

        def fail_restore(src, dst):
            if os.path.basename(src).startswith(".recovery-cache-backup-"):
                raise OSError("simulated restore failure")
            return real_replace(src, dst)

        with mock.patch("city_twin.branches.os.replace", fail_restore):
            with self.assertRaises(OSError):
                self.recover_now()
        self.assertTrue(os.path.exists(self.recovery))

    def test_record_cleanup_failure_raises_oserror(self) -> None:
        # The shared setUp left an index-phase crash; resolve it through
        # a normal entry, grow again, then die during cleanup so both
        # targets carry the complete new version (phase "progress").
        self.build()
        self.grow_chain(3, start=40)
        self.kill_publication_at("cleanup")
        record_before = _read(self.recovery)
        real_remove = os.remove

        def fail_record_remove(path):
            if os.path.realpath(path) == os.path.realpath(self.recovery):
                raise OSError("simulated record unlink failure")
            return real_remove(path)

        with mock.patch("city_twin.branches.os.remove", fail_record_remove):
            with self.assertRaises(OSError):
                self.recover_now()
        # The record is the file a later recovery still needs; the new
        # targets stay complete and another attempt finishes cleanly.
        self.assertEqual(_read(self.recovery), record_before)
        self.recover_now()
        self.assertFalse(os.path.exists(self.recovery))
        self.assertEqual(self.residue(), [])


class RecoveryStateMachineClosureTests(RecoveryPublicationTestBase):
    def setUp(self) -> None:
        super().setUp()
        self.grow_chain(5)
        self.build()
        self.old_index = _read(self.index)
        self.old_progress = _read(self.progress)
        self.grow_chain(3, start=20)

    def _rewrite_record_phase(self, phase: str) -> None:
        document = self.record_document()
        raw = BranchStore._cache_recovery_record_bytes(
            phase, document["index"], document["progress"]
        )
        with open(self.recovery, "wb") as handle:
            handle.write(raw)

    def _backups(self) -> list:
        return [
            name
            for name in os.listdir(self.tmp.name)
            if name.startswith(".recovery-cache-backup-")
        ]

    def test_prepared_record_with_new_targets_is_contradiction(self) -> None:
        # Both targets carry the new version but the record says
        # "prepared" -- a combination the protocol cannot produce, so
        # the superficial agreement must not delete the evidence.
        self.kill_publication_at("record_after_progress")
        self._rewrite_record_phase("prepared")
        new_index = _read(self.index)
        new_progress = _read(self.progress)
        backups = self._backups()
        self.assertEqual(len(backups), 2)
        with self.assertRaises(ValueError):
            self.recover_now()
        self.assertEqual(self.record_document()["phase"], "prepared")
        self.assertEqual(_read(self.index), new_index)
        self.assertEqual(_read(self.progress), new_progress)
        self.assertEqual(self._backups(), backups)

    def test_prepared_record_with_new_progress_only_is_contradiction(
        self,
    ) -> None:
        self.kill_publication_at("record_after_progress")
        record = self.record_document()
        os.replace(
            os.path.join(self.tmp.name, record["index"]["backup"]),
            self.index,
        )
        self._rewrite_record_phase("prepared")
        with self.assertRaises(ValueError):
            self.recover_now()
        self.assertTrue(os.path.exists(self.recovery))
        self.assertEqual(_read(self.index), self.old_index)
        self.assertNotEqual(_read(self.progress), self.old_progress)

    def test_missing_required_backup_is_value_error(self) -> None:
        self.kill_publication_at("progress_replace")
        record = self.record_document()
        os.remove(os.path.join(self.tmp.name, record["index"]["backup"]))
        new_index = _read(self.index)
        with self.assertRaises(ValueError):
            self.recover_now()
        # Everything still needed for forensics is retained.
        self.assertTrue(os.path.exists(self.recovery))
        self.assertEqual(_read(self.index), new_index)
        self.assertEqual(_read(self.progress), self.old_progress)
        self.assertEqual(len(self._backups()), 1)

    def test_commit_with_corrupt_backup_is_value_error(self) -> None:
        self.kill_publication_at("cleanup")
        record = self.record_document()
        backup_path = os.path.join(
            self.tmp.name, record["index"]["backup"]
        )
        with open(backup_path, "wb") as handle:
            handle.write(b"corrupt backup bytes")
        orphan = os.path.join(self.tmp.name, ".recovery-cache-record-zz.tmp")
        with open(orphan, "wb") as handle:
            handle.write(b"orphan")
        new_index = _read(self.index)
        with self.assertRaises(ValueError):
            self.recover_now()
        # The corrupt backup, the orphan temp and the record all stay.
        self.assertEqual(_read(backup_path), b"corrupt backup bytes")
        self.assertEqual(_read(orphan), b"orphan")
        self.assertTrue(os.path.exists(self.recovery))
        self.assertEqual(_read(self.index), new_index)

    def test_commit_cleanup_removes_only_protocol_temps(self) -> None:
        self.kill_publication_at("cleanup")
        orphan = os.path.join(self.tmp.name, ".recovery-cache-record-zz.tmp")
        with open(orphan, "wb") as handle:
            handle.write(b"orphan")
        unrelated = os.path.join(self.tmp.name, "unrelated.txt")
        with open(unrelated, "wb") as handle:
            handle.write(b"keep")
        new_index = _read(self.index)
        new_progress = _read(self.progress)
        self.recover_now()
        self.assertEqual(_read(self.index), new_index)
        self.assertEqual(_read(self.progress), new_progress)
        self.assertFalse(os.path.exists(orphan))
        self.assertEqual(_read(unrelated), b"keep")
        self.assertFalse(os.path.exists(self.recovery))
        self.assertEqual(self.residue(), [])

    def test_rollback_cleanup_removes_only_protocol_temps(self) -> None:
        self.kill_publication_at("progress_replace")
        orphan = os.path.join(self.tmp.name, ".recovery-cache-record-zz.tmp")
        with open(orphan, "wb") as handle:
            handle.write(b"orphan")
        unrelated = os.path.join(self.tmp.name, "unrelated.txt")
        with open(unrelated, "wb") as handle:
            handle.write(b"keep")
        self.recover_now()
        self.assertEqual(_read(self.index), self.old_index)
        self.assertEqual(_read(self.progress), self.old_progress)
        self.assertFalse(os.path.exists(orphan))
        self.assertEqual(_read(unrelated), b"keep")
        self.assertFalse(os.path.exists(self.recovery))
        self.assertEqual(self.residue(), [])

    def test_identical_republication_recovers_cleanly(self) -> None:
        # Rebuilding an unchanged chain republishes identical bytes; a
        # crash leaving a "prepared" record is an ordinary reachable
        # state, not a contradiction.
        self.assertEqual(self.build(), self.head)
        same_index = _read(self.index)
        same_progress = _read(self.progress)
        self.kill_publication_at("index_replace")
        self.assertEqual(self.record_document()["phase"], "prepared")
        self.recover_now()
        self.assertEqual(_read(self.index), same_index)
        self.assertEqual(_read(self.progress), same_progress)
        self.assertFalse(os.path.exists(self.recovery))
        self.assertEqual(self.residue(), [])
        self.assertEqual(self.build(), self.head)

    def test_disturbed_target_without_record_is_value_error(self) -> None:
        def disturbing_write(recovery_path, data):
            # The record never becomes durable, yet a target moves.
            with open(self.index, "ab") as handle:
                handle.write(b"x")
            raise OSError("simulated record write failure")

        with mock.patch.object(
            BranchStore,
            "_write_cache_recovery_record",
            staticmethod(disturbing_write),
        ):
            with self.assertRaises(ValueError):
                self.build()
        self.assertFalse(os.path.exists(self.recovery))
        self.assertNotEqual(_read(self.index), self.old_index)
        self.assertEqual(_read(self.progress), self.old_progress)
        # The backups of the old version are retained as evidence.
        self.assertTrue(self._backups())


class PreparationFailureTests(RecoveryPublicationTestBase):
    def test_backup_failure_before_record_leaves_old_version(self) -> None:
        self.grow_chain(5)
        self.build()
        index_before = _read(self.index)
        progress_before = _read(self.progress)
        self.grow_chain(3, start=20)

        def fail_backup(target):
            raise OSError("simulated backup failure")

        with mock.patch.object(
            BranchStore, "_backup_cache_file", staticmethod(fail_backup)
        ):
            with self.assertRaises(OSError):
                self.build()
        self.assertFalse(os.path.exists(self.recovery))
        self.assertEqual(_read(self.index), index_before)
        self.assertEqual(_read(self.progress), progress_before)
        self.assertEqual(self.residue(), [])


class ArgumentValidationTests(RecoveryPublicationTestBase):
    def test_recovery_type_validation(self) -> None:
        self.grow_chain(2)
        for bad in (1, 1.0, b"x", ["r"], {"r": 1}, object()):
            with self.subTest(bad=type(bad).__name__):
                with self.assertRaises(TypeError):
                    self.build(bad)  # type: ignore[arg-type]
                with self.assertRaises(TypeError):
                    BranchStore.diff_recovery_audit_ranges(
                        self.chain,
                        self.head,
                        (),
                        self.index,
                        self.SEGMENT_SIZE,
                        self.progress,
                        bad,
                    )

    def test_recovery_empty_string(self) -> None:
        self.grow_chain(2)
        with self.assertRaises(ValueError):
            self.build("")
        with self.assertRaises(ValueError):
            BranchStore.diff_recovery_audit_ranges(
                self.chain,
                self.head,
                (),
                self.index,
                self.SEGMENT_SIZE,
                self.progress,
                "",
            )

    def test_recovery_requires_progress(self) -> None:
        self.grow_chain(2)
        with self.assertRaises(ValueError):
            BranchStore.build_recovery_audit_segments(
                self.chain,
                self.index,
                self.SEGMENT_SIZE,
                None,
                self.recovery,
            )

    def test_recovery_must_not_alias_chain_index_or_progress(self) -> None:
        self.grow_chain(2)
        for same in (self.chain, self.index, self.progress):
            with self.subTest(same=os.path.basename(same)):
                with self.assertRaises(ValueError):
                    self.build(same)
        link = os.path.join(self.tmp.name, "recovery-link")
        os.symlink(self.chain, link)
        with self.assertRaises(ValueError):
            self.build(link)

    def test_three_cache_files_must_share_one_directory(self) -> None:
        self.grow_chain(2)
        sub = os.path.join(self.tmp.name, "sub")
        os.mkdir(sub)
        other_index = os.path.join(sub, "index.json")
        other_recovery = os.path.join(sub, "recovery.json")
        # Index elsewhere, progress and recovery here.
        with self.assertRaises(ValueError):
            BranchStore.build_recovery_audit_segments(
                self.chain,
                other_index,
                self.SEGMENT_SIZE,
                self.progress,
                self.recovery,
            )
        # Recovery elsewhere while both caches share a directory.
        with self.assertRaises(ValueError):
            BranchStore.build_recovery_audit_segments(
                self.chain,
                self.index,
                self.SEGMENT_SIZE,
                self.progress,
                other_recovery,
            )

    def test_validation_order_is_after_existing_arguments(self) -> None:
        self.grow_chain(1)
        # A bad segment size is reported before the recovery value.
        with self.assertRaises(TypeError):
            BranchStore.build_recovery_audit_segments(
                self.chain, self.index, 1.5, self.progress, 123
            )
        with self.assertRaises(TypeError):
            BranchStore.build_recovery_audit_segments(
                42, self.index, 2, self.progress, 123
            )
        # Range validation precedes recovery validation in the batch.
        with self.assertRaises(TypeError):
            BranchStore.diff_recovery_audit_ranges(
                self.chain,
                self.head,
                [(1, 2)],
                self.index,
                self.SEGMENT_SIZE,
                self.progress,
                123,
            )

    def test_none_recovery_keeps_other_value_errors(self) -> None:
        self.grow_chain(2)
        # Head mismatch surfaces through the batch even with recovery on.
        with self.assertRaises(ValueError):
            BranchStore.diff_recovery_audit_ranges(
                self.chain,
                "0" * 64,
                (),
                self.index,
                self.SEGMENT_SIZE,
                self.progress,
                self.recovery,
            )


class OSErrorSurfaceTests(RecoveryPublicationTestBase):
    def test_missing_cache_directory_raises_oserror(self) -> None:
        self.grow_chain(2)
        missing_dir = os.path.join(self.tmp.name, "missing")
        with self.assertRaises(OSError):
            BranchStore.build_recovery_audit_segments(
                self.chain,
                os.path.join(missing_dir, "index.json"),
                self.SEGMENT_SIZE,
                os.path.join(missing_dir, "progress.json"),
                os.path.join(missing_dir, "recovery.json"),
            )
        self.assertFalse(os.path.exists(missing_dir))

    def test_corrupt_chain_still_raises_value_error_with_record_gone(self) -> None:
        self.grow_chain(4)
        self.build()
        document = json.loads(_read(self.chain).decode("utf-8"))
        good_chain = _read(self.chain)
        document["frames"] = document["frames"][:2]
        with open(self.chain, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(document, separators=(",", ":")))
        with self.assertRaises(ValueError):
            self.build()
        self.assertFalse(os.path.exists(self.recovery))
        self.assertEqual(self.residue(), [])
        with open(self.chain, "wb") as handle:
            handle.write(good_chain)


class ConcurrencyTests(RecoveryPublicationTestBase):
    def test_concurrent_entries_observe_only_full_versions(self) -> None:
        head = self.grow_chain(4)
        self.build()
        errors: list[BaseException] = []
        lock = threading.Lock()

        def builder() -> None:
            try:
                for _ in range(4):
                    BranchStore.build_recovery_audit_segments(
                        self.chain,
                        self.index,
                        self.SEGMENT_SIZE,
                        self.progress,
                        self.recovery,
                    )
            except BaseException as exc:  # pragma: no cover - diagnostic
                with lock:
                    errors.append(exc)

        def differ() -> None:
            try:
                for _ in range(4):
                    results = BranchStore.diff_recovery_audit_ranges(
                        self.chain,
                        head,
                        ((1, 4),),
                        self.index,
                        self.SEGMENT_SIZE,
                        self.progress,
                        self.recovery,
                    )
                    expected = BranchStore.diff_recovery_audit_range(
                        self.chain, head, 1, 4
                    )
                    assert results[0]["changes"] == expected
            except BaseException as exc:  # pragma: no cover - diagnostic
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=builder) for _ in range(3)]
        threads += [threading.Thread(target=differ) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(errors, [])
        self.assertFalse(os.path.exists(self.recovery))
        self.assertEqual(self.residue(), [])
        final = self.build()
        self.assertTrue(
            BranchStore.verify_recovery_audit_chain(
                self.chain, final, self.index
            )
        )


if __name__ == "__main__":
    unittest.main()
