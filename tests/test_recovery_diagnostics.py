"""Tests for the read-only cross-restart recovery diagnostics.

``BranchStore.recovery_diagnostic`` observes the existing cross-process
cache-publication recovery state machine without running it and returns
a sealed compact JSON string; ``BranchStore.verify_recovery_diagnostics``
authenticates a consecutive run of those strings. No recovery is
performed and no file is modified.

These tests cover:

* the idle report and the canonical, checksummed, deterministic shape;
* pending commit and pending rollback judgments matching the state
  machine's reachable phase/target combinations, including the
  index-only first publication and recorded absence;
* committed and rolled-back continuations after the recovery record
  disappears, still anchored to the same transaction identity;
* every rejection category -- damaged record, phase contradiction,
  target digest mismatch/missing, missing or corrupt backup, evidence
  regression and a switched transaction;
* a damaged record's identity being the raw-byte digest and no
  material body ever being echoed;
* diagnostic-run authentication: tuple/element typing, per-record
  legality and predecessor, transaction and state-transition
  continuity;
* path and predecessor argument validation;
* OSError for a missing or unreadable cache on a first observation;
* strict read-only isolation of every target, backup, record, chain
  and generation file.
"""

import glob
import hashlib
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


class RecoveryDiagnosticTestBase(unittest.TestCase):
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

    def build(self) -> str:
        return BranchStore.build_recovery_audit_segments(  # type: ignore[return-value]
            self.chain,
            self.index,
            self.SEGMENT_SIZE,
            self.progress,
            self.recovery,
        )

    def diagnostic(self, previous: "str | None" = None) -> str:
        return BranchStore.recovery_diagnostic(
            self.index, self.progress, self.recovery, previous
        )

    def diagnostic_document(self, previous: "str | None" = None) -> dict:
        return json.loads(self.diagnostic(previous))

    def recover_now(self) -> None:
        BranchStore._recover_cache_publication_locked(
            self.index, self.progress, self.recovery
        )

    def record_document(self) -> dict:
        return json.loads(_read(self.recovery).decode("utf-8"))

    def kill_publication_at(self, point: str) -> None:
        """Mimic a process dying during publication, leaving the durable
        record and backups exactly as a dead process would; orphaned
        build temps are removed afterwards."""
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

    def residue(self) -> list:
        return sorted(
            name
            for name in os.listdir(self.tmp.name)
            if name.startswith(".recovery-")
        )

    def directory_snapshot(self) -> dict:
        captured = {}
        for name in os.listdir(self.tmp.name):
            path = os.path.join(self.tmp.name, name)
            if os.path.isfile(path):
                captured[name] = _read(path)
        for dirpath, _dirnames, filenames in os.walk(self.root):
            for filename in filenames:
                path = os.path.join(dirpath, filename)
                captured[path] = _read(path)
        return captured

    def reseal_record(self, mutate) -> None:
        document = self.record_document()
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
            handle.write(
                BranchStore._canonical_json(document).encode("utf-8")
            )


class IdleDiagnosticTests(RecoveryDiagnosticTestBase):
    def test_idle_shape_when_no_record(self) -> None:
        self.grow_chain(5)
        self.build()
        self.assertFalse(os.path.exists(self.recovery))
        text = self.diagnostic()
        raw = text.encode("utf-8")
        self.assertFalse(raw.startswith(b"\xef\xbb\xbf"))
        self.assertFalse(raw.endswith(b"\n"))
        self.assertNotIn(b"\n", raw)
        self.assertNotIn(b" ", raw)
        document = json.loads(text)
        self.assertEqual(
            set(document),
            {
                "format",
                "version",
                "transaction",
                "previous",
                "state",
                "phase",
                "index",
                "progress",
                "backups",
                "disposition",
                "reason",
                "checksum",
            },
        )
        self.assertEqual(
            document["format"], "branching-city-twin/recovery-diagnostic"
        )
        self.assertEqual(document["version"], 1)
        self.assertIsNone(document["transaction"])
        self.assertEqual(document["previous"], "")
        self.assertEqual(document["state"], "idle")
        self.assertIsNone(document["phase"])
        self.assertEqual(document["disposition"], "none")
        self.assertEqual(document["reason"], "")
        self.assertEqual(document["backups"], [])
        for name in ("index", "progress"):
            view = document[name]
            self.assertEqual(
                set(view), {"exists", "state", "old_digest", "new_digest"}
            )
            self.assertTrue(view["exists"])
            self.assertIsNone(view["state"])
            self.assertIsNone(view["old_digest"])
            self.assertIsNone(view["new_digest"])
        checksum = document.pop("checksum")
        expected = hashlib.sha256(
            BranchStore._canonical_json(document).encode("utf-8")
        ).hexdigest()
        self.assertEqual(checksum, expected)
        self.assertEqual(
            text, BranchStore._canonical_json(json.loads(text))
        )

    def test_identical_evidence_is_byte_identical(self) -> None:
        self.grow_chain(4)
        self.build()
        first = self.diagnostic()
        second = self.diagnostic()
        self.assertEqual(first, second)

    def test_empty_tuple_verifies(self) -> None:
        self.assertIs(BranchStore.verify_recovery_diagnostics(()), True)

    def test_single_idle_diagnostic_verifies(self) -> None:
        self.grow_chain(3)
        self.build()
        self.assertIs(
            BranchStore.verify_recovery_diagnostics((self.diagnostic(),)),
            True,
        )


class PendingCommitDiagnosticTests(RecoveryDiagnosticTestBase):
    def setUp(self) -> None:
        super().setUp()
        self.grow_chain(5)
        self.build()
        self.grow_chain(3, start=20)
        self.kill_publication_at("cleanup")

    def test_pending_commit_judgment(self) -> None:
        document = self.diagnostic_document()
        self.assertEqual(document["state"], "pending_commit")
        self.assertEqual(document["disposition"], "commit")
        self.assertEqual(document["reason"], "")
        self.assertEqual(document["phase"], "progress")
        record = self.record_document()
        self.assertEqual(document["transaction"], record["checksum"])
        for name in ("index", "progress"):
            self.assertEqual(document[name]["state"], "new")
            self.assertTrue(document[name]["exists"])
            self.assertEqual(
                document[name]["old_digest"], record[name]["old_digest"]
            )
            self.assertEqual(
                document[name]["new_digest"], record[name]["new_digest"]
            )
        self.assertEqual(len(document["backups"]), 2)
        self.assertTrue(
            all(entry["state"] == "valid" for entry in document["backups"])
        )

    def test_committed_after_record_cleanup(self) -> None:
        first = self.diagnostic()
        self.recover_now()
        second = self.diagnostic(first)
        document = json.loads(second)
        self.assertEqual(document["state"], "committed")
        self.assertEqual(document["disposition"], "none")
        self.assertIsNone(document["phase"])
        self.assertEqual(
            document["transaction"], json.loads(first)["transaction"]
        )
        self.assertEqual(document["previous"], json.loads(first)["checksum"])
        self.assertTrue(
            BranchStore.verify_recovery_diagnostics((first, second))
        )

    def test_pending_commit_then_committed_is_stable(self) -> None:
        first = self.diagnostic()
        first_again = self.diagnostic(first)
        self.recover_now()
        second = self.diagnostic(first_again)
        self.assertTrue(
            BranchStore.verify_recovery_diagnostics(
                (first, first_again, second)
            )
        )

    def test_lagging_phase_still_commits_when_both_targets_new(self) -> None:
        self.recover_now()
        self.grow_chain(2, start=40)
        self.kill_publication_at("record_after_progress")
        document = self.diagnostic_document()
        self.assertEqual(document["phase"], "index")
        self.assertEqual(document["state"], "pending_commit")


class PendingRollbackDiagnosticTests(RecoveryDiagnosticTestBase):
    def setUp(self) -> None:
        super().setUp()
        self.grow_chain(5)
        self.build()
        self.grow_chain(3, start=20)
        self.kill_publication_at("progress_replace")

    def test_pending_rollback_judgment(self) -> None:
        document = self.diagnostic_document()
        self.assertEqual(document["state"], "pending_rollback")
        self.assertEqual(document["disposition"], "rollback")
        self.assertEqual(document["reason"], "")
        self.assertEqual(document["phase"], "index")
        self.assertEqual(document["index"]["state"], "new")
        self.assertEqual(document["progress"]["state"], "old")
        for entry in document["backups"]:
            self.assertEqual(entry["state"], "valid")

    def test_rollback_chain_converges(self) -> None:
        first = self.diagnostic()
        first_again = self.diagnostic(first)
        self.recover_now()
        second = self.diagnostic(first_again)
        document = json.loads(second)
        self.assertEqual(document["state"], "rolled_back")
        self.assertEqual(document["disposition"], "none")
        self.assertIsNone(document["phase"])
        self.assertTrue(
            BranchStore.verify_recovery_diagnostics(
                (first, first_again, second)
            )
        )

    def test_reprocessing_after_settlement_is_regression(self) -> None:
        first = self.diagnostic()
        self.recover_now()
        settled = self.diagnostic(first)
        self.assertEqual(json.loads(settled)["state"], "rolled_back")
        # A fresh interrupted publication with the caches republished is
        # new activity after the disposition settled.
        self.grow_chain(2, start=40)
        self.kill_publication_at("progress_replace")
        reprocessed = self.diagnostic(settled)
        document = json.loads(reprocessed)
        self.assertEqual(document["state"], "rejected")
        self.assertEqual(document["reason"], "evidence_regression")
        self.assertFalse(
            BranchStore.verify_recovery_diagnostics(
                (first, settled, reprocessed)
            )
        )

    def test_evidence_degradation_fails_the_run(self) -> None:
        # A sound pending observation followed by evidence that can no
        # longer be recovered (a required backup goes corrupt) yields a
        # valid rejection diagnostic but the run no longer verifies.
        first = self.diagnostic()
        self.assertEqual(json.loads(first)["state"], "pending_rollback")
        record = self.record_document()
        with open(
            os.path.join(self.tmp.name, record["index"]["backup"]), "wb"
        ) as handle:
            handle.write(b"corrupt backup bytes")
        degraded = self.diagnostic(first)
        self.assertEqual(json.loads(degraded)["state"], "rejected")
        self.assertEqual(
            json.loads(degraded)["reason"], "backup_invalid"
        )
        self.assertFalse(
            BranchStore.verify_recovery_diagnostics((first, degraded))
        )

    def test_settled_observations_repeat_with_the_run_intact(self) -> None:
        # After the disposition settles, further read-only observations
        # with no new activity simply restate it and keep authenticating.
        first = self.diagnostic()
        self.recover_now()
        settled = self.diagnostic(first)
        again = self.diagnostic(settled)
        once_more = self.diagnostic(again)
        for text in (settled, again, once_more):
            self.assertEqual(json.loads(text)["state"], "rolled_back")
        self.assertTrue(
            BranchStore.verify_recovery_diagnostics(
                (first, settled, again, once_more)
            )
        )


class IndexOnlyFirstPublicationTests(RecoveryDiagnosticTestBase):
    def setUp(self) -> None:
        super().setUp()
        self.grow_chain(1)
        self.kill_publication_at("index_replace")

    def test_absent_targets_pending_rollback(self) -> None:
        document = self.diagnostic_document()
        self.assertEqual(document["state"], "pending_rollback")
        self.assertEqual(document["phase"], "prepared")
        for name in ("index", "progress"):
            self.assertFalse(document[name]["exists"])
            self.assertEqual(document[name]["state"], "old")
            self.assertIsNone(document[name]["old_digest"])
        self.assertEqual(document["backups"], [])
        self.assertTrue(
            BranchStore.verify_recovery_diagnostics((self.diagnostic(),))
        )

    def test_continuation_tolerates_restored_absence(self) -> None:
        first = self.diagnostic()
        self.recover_now()
        self.assertFalse(os.path.exists(self.index))
        # The predecessor explains the absence, so no OSError is raised.
        second = self.diagnostic(first)
        self.assertEqual(json.loads(second)["state"], "rolled_back")
        self.assertTrue(
            BranchStore.verify_recovery_diagnostics((first, second))
        )


class RejectionDiagnosticTests(RecoveryDiagnosticTestBase):
    def setUp(self) -> None:
        super().setUp()
        self.grow_chain(5)
        self.build()
        self.grow_chain(3, start=20)
        self.kill_publication_at("progress_replace")

    def test_damaged_record_uses_raw_byte_digest(self) -> None:
        for raw in (
            b"garbage",
            b"\xef\xbb\xbf" + _read(self.recovery),
            _read(self.recovery) + b"\n",
        ):
            with self.subTest(raw=raw[:12]):
                with open(self.recovery, "wb") as handle:
                    handle.write(raw)
                document = self.diagnostic_document()
                self.assertEqual(document["state"], "rejected")
                self.assertEqual(document["disposition"], "refuse")
                self.assertEqual(document["reason"], "invalid_record")
                self.assertIsNone(document["phase"])
                self.assertEqual(
                    document["transaction"],
                    hashlib.sha256(raw).hexdigest(),
                )
                self.assertNotIn(raw, self.diagnostic().encode("utf-8"))
        # An empty record is damaged too; its identity is the empty
        # input's SHA-256.
        with open(self.recovery, "wb"):
            pass
        document = self.diagnostic_document()
        self.assertEqual(document["state"], "rejected")
        self.assertEqual(document["reason"], "invalid_record")
        self.assertEqual(
            document["transaction"], hashlib.sha256(b"").hexdigest()
        )

    def test_material_record_body_is_never_echoed(self) -> None:
        text = self.diagnostic()
        self.assertNotIn(
            "branching-city-twin/recovery-cache-publication", text
        )

    def test_phase_contradiction(self) -> None:
        # Both targets new while the record claims nothing was replaced.
        self.recover_now()
        self.grow_chain(2, start=40)
        self.kill_publication_at("cleanup")
        self.reseal_record(lambda doc: doc.__setitem__("phase", "prepared"))
        document = self.diagnostic_document()
        self.assertEqual(document["state"], "rejected")
        self.assertEqual(document["reason"], "phase_contradiction")

    def test_target_unknown_bytes(self) -> None:
        with open(self.progress, "wb") as handle:
            handle.write(b"neither old nor new content")
        document = self.diagnostic_document()
        self.assertEqual(document["state"], "rejected")
        self.assertEqual(document["reason"], "target_contradiction")
        self.assertEqual(document["progress"]["state"], "unknown")

    def test_recorded_target_going_missing(self) -> None:
        os.remove(self.progress)
        document = self.diagnostic_document()
        self.assertEqual(document["state"], "rejected")
        self.assertEqual(document["reason"], "target_contradiction")
        self.assertFalse(document["progress"]["exists"])
        self.assertEqual(document["progress"]["state"], "missing")

    def test_corrupt_backup(self) -> None:
        record = self.record_document()
        backup_name = record["index"]["backup"]
        with open(os.path.join(self.tmp.name, backup_name), "wb") as handle:
            handle.write(b"corrupt backup bytes")
        document = self.diagnostic_document()
        self.assertEqual(document["state"], "rejected")
        self.assertEqual(document["reason"], "backup_invalid")
        statuses = {
            entry["name"]: entry["state"] for entry in document["backups"]
        }
        self.assertEqual(statuses[backup_name], "corrupt")

    def test_missing_backup_needed_for_restore(self) -> None:
        record = self.record_document()
        backup_name = record["index"]["backup"]
        os.remove(os.path.join(self.tmp.name, backup_name))
        document = self.diagnostic_document()
        self.assertEqual(document["state"], "rejected")
        self.assertEqual(document["reason"], "backup_invalid")
        statuses = {
            entry["name"]: entry["state"] for entry in document["backups"]
        }
        self.assertEqual(statuses[backup_name], "missing")

    def test_rejected_persistence_verifies_with_same_reason(self) -> None:
        with open(self.recovery, "wb") as handle:
            handle.write(b"garbage")
        first = self.diagnostic()
        second = self.diagnostic(first)
        self.assertEqual(json.loads(second)["state"], "rejected")
        self.assertEqual(json.loads(second)["reason"], "invalid_record")
        self.assertTrue(
            BranchStore.verify_recovery_diagnostics((first, second))
        )


class ContinuationDiagnosticTests(RecoveryDiagnosticTestBase):
    def setUp(self) -> None:
        super().setUp()
        self.grow_chain(5)
        self.build()
        self.grow_chain(3, start=20)
        self.kill_publication_at("progress_replace")

    def test_transaction_switch_is_rejected(self) -> None:
        # The first observation is still pending; a later, different
        # valid recovery record is a mid-flight transaction switch.
        first = self.diagnostic()
        first_txn = json.loads(first)["transaction"]
        self.recover_now()
        self.grow_chain(2, start=40)
        self.kill_publication_at("progress_replace")
        switched = self.diagnostic(first)
        document = json.loads(switched)
        self.assertEqual(document["state"], "rejected")
        self.assertEqual(document["reason"], "transaction_changed")
        self.assertNotEqual(document["transaction"], first_txn)
        self.assertFalse(
            BranchStore.verify_recovery_diagnostics((first, switched))
        )

    def test_reason_change_inside_one_transaction_breaks_run(self) -> None:
        # Same record/checksum: first a corrupt backup, then unknown
        # target bytes -- the rejection reason changes identity.
        record = self.record_document()
        backup_path = os.path.join(
            self.tmp.name, record["index"]["backup"]
        )
        original_backup = _read(backup_path)
        with open(backup_path, "wb") as handle:
            handle.write(b"corrupt backup bytes")
        first = self.diagnostic()
        self.assertEqual(json.loads(first)["reason"], "backup_invalid")
        with open(backup_path, "wb") as handle:
            handle.write(original_backup)
        with open(self.progress, "wb") as handle:
            handle.write(b"neither old nor new content")
        second = self.diagnostic(first)
        self.assertEqual(json.loads(second)["reason"], "target_contradiction")
        self.assertFalse(
            BranchStore.verify_recovery_diagnostics((first, second))
        )


class VerifyDiagnosticsValidationTests(RecoveryDiagnosticTestBase):
    def setUp(self) -> None:
        super().setUp()
        self.grow_chain(4)
        self.build()

    def test_container_type_must_be_tuple(self) -> None:
        text = self.diagnostic()
        for bad in ([text], {text}, (x for x in (text,))):
            with self.subTest(bad=type(bad).__name__):
                with self.assertRaises(TypeError):
                    BranchStore.verify_recovery_diagnostics(bad)

    def test_element_type_must_be_str(self) -> None:
        with self.assertRaises(TypeError):
            BranchStore.verify_recovery_diagnostics((1,))
        with self.assertRaises(TypeError):
            BranchStore.verify_recovery_diagnostics((self.diagnostic(), None))

    def test_individually_illegal_records_raise_value_error(self) -> None:
        text = self.diagnostic()

        def mutate_seal(mutate) -> str:
            doc = json.loads(text)
            mutate(doc)
            body = {
                key: value
                for key, value in doc.items()
                if key != "checksum"
            }
            doc["checksum"] = hashlib.sha256(
                BranchStore._canonical_json(body).encode("utf-8")
            ).hexdigest()
            return BranchStore._canonical_json(doc)

        with self.assertRaises(ValueError):
            BranchStore.verify_recovery_diagnostics(("",))
        with self.assertRaises(ValueError):
            BranchStore.verify_recovery_diagnostics(("not json",))
        with self.assertRaises(ValueError):
            BranchStore.verify_recovery_diagnostics(
                ('{"a":1,"a":2}',)
            )
        with self.assertRaises(ValueError):
            BranchStore.verify_recovery_diagnostics((text + "\n",))
        with self.assertRaises(ValueError):
            BranchStore.verify_recovery_diagnostics(
                (mutate_seal(lambda d: d.__setitem__("version", 2)),)
            )
        with self.assertRaises(ValueError):
            BranchStore.verify_recovery_diagnostics(
                (mutate_seal(lambda d: d.pop("phase")),)
            )
        bad_checksum = json.loads(text)
        bad_checksum["checksum"] = "0" * 64
        with self.assertRaises(ValueError):
            BranchStore.verify_recovery_diagnostics(
                (BranchStore._canonical_json(bad_checksum),)
            )

    def test_run_must_start_without_predecessor(self) -> None:
        first = self.diagnostic()
        second = self.diagnostic(first)
        self.assertFalse(
            BranchStore.verify_recovery_diagnostics((second,))
        )

    def test_broken_predecessor_link_returns_false(self) -> None:
        first = self.diagnostic()
        # A fresh observation with no predecessor starts its own run.
        other = self.diagnostic()
        self.assertFalse(
            BranchStore.verify_recovery_diagnostics((first, other))
        )

    def test_idle_run_is_continuous(self) -> None:
        first = self.diagnostic()
        second = self.diagnostic(first)
        self.assertTrue(
            BranchStore.verify_recovery_diagnostics((first, second))
        )


class ArgumentValidationTests(RecoveryDiagnosticTestBase):
    def setUp(self) -> None:
        super().setUp()
        self.grow_chain(3)
        self.build()

    def test_path_type_validation(self) -> None:
        for bad in (1, 1.0, b"x", ["x"], {"x": 1}, object()):
            with self.subTest(bad=type(bad).__name__):
                with self.assertRaises(TypeError):
                    BranchStore.recovery_diagnostic(
                        bad, self.progress, self.recovery
                    )
                with self.assertRaises(TypeError):
                    BranchStore.recovery_diagnostic(
                        self.index, bad, self.recovery
                    )
                with self.assertRaises(TypeError):
                    BranchStore.recovery_diagnostic(
                        self.index, self.progress, bad
                    )

    def test_empty_paths_raise_value_error(self) -> None:
        with self.assertRaises(ValueError):
            BranchStore.recovery_diagnostic("", self.progress, self.recovery)
        with self.assertRaises(ValueError):
            BranchStore.recovery_diagnostic(self.index, "", self.recovery)
        with self.assertRaises(ValueError):
            BranchStore.recovery_diagnostic(self.index, self.progress, "")

    def test_paths_must_not_alias(self) -> None:
        with self.assertRaises(ValueError):
            BranchStore.recovery_diagnostic(
                self.index, self.index, self.recovery
            )
        with self.assertRaises(ValueError):
            BranchStore.recovery_diagnostic(
                self.index, self.recovery, self.recovery
            )
        link = os.path.join(self.tmp.name, "recovery-link")
        os.symlink(self.index, link)
        with self.assertRaises(ValueError):
            BranchStore.recovery_diagnostic(
                self.index, self.progress, link
            )

    def test_paths_must_share_one_directory(self) -> None:
        sub = os.path.join(self.tmp.name, "sub")
        os.mkdir(sub)
        with self.assertRaises(ValueError):
            BranchStore.recovery_diagnostic(
                os.path.join(sub, "index.json"),
                self.progress,
                self.recovery,
            )
        with self.assertRaises(ValueError):
            BranchStore.recovery_diagnostic(
                self.index,
                self.progress,
                os.path.join(sub, "recovery.json"),
            )

    def test_previous_type_validation(self) -> None:
        for bad in (1, 1.0, b"x", [], {}, object()):
            with self.subTest(bad=type(bad).__name__):
                with self.assertRaises(TypeError):
                    self.diagnostic(bad)  # type: ignore[arg-type]

    def test_previous_value_validation(self) -> None:
        for bad in (
            "",
            "not json",
            '{"a":1,"a":2}',
            "{}",
            self.diagnostic() + " ",
        ):
            with self.subTest(bad=bad[:12]):
                with self.assertRaises(ValueError):
                    self.diagnostic(bad)

    def test_idle_diagnostic_chains_to_idle(self) -> None:
        first = self.diagnostic()
        second = self.diagnostic(first)
        self.assertEqual(json.loads(second)["state"], "idle")
        self.assertTrue(
            BranchStore.verify_recovery_diagnostics((first, second))
        )


class OSErrorSurfaceTests(RecoveryDiagnosticTestBase):
    def test_missing_index_raises_oserror(self) -> None:
        self.grow_chain(3)
        self.build()
        os.remove(self.index)
        with self.assertRaises(OSError):
            self.diagnostic()

    def test_missing_progress_raises_oserror(self) -> None:
        self.grow_chain(3)
        self.build()
        os.remove(self.progress)
        with self.assertRaises(OSError):
            self.diagnostic()

    def test_damaged_record_with_missing_cache_still_reports_rejection(self):
        self.grow_chain(3)
        self.build()
        with open(self.recovery, "wb") as handle:
            handle.write(b"garbage")
        os.remove(self.progress)
        document = self.diagnostic_document()
        self.assertEqual(document["state"], "rejected")
        self.assertEqual(document["reason"], "invalid_record")


class ReadOnlyIsolationTests(RecoveryDiagnosticTestBase):
    def _assert_isolation(self) -> None:
        before = self.directory_snapshot()
        names_before = set(os.listdir(self.tmp.name))
        text = self.diagnostic()
        # Calling again with the result as predecessor exercises the
        # continuation path without changing anything either.
        self.diagnostic(text)
        self.assertEqual(set(os.listdir(self.tmp.name)), names_before)
        self.assertEqual(self.directory_snapshot(), before)

    def test_idle_observation_is_read_only(self) -> None:
        self.grow_chain(3)
        self.build()
        self._assert_isolation()

    def test_pending_observation_is_read_only(self) -> None:
        self.grow_chain(5)
        self.build()
        self.grow_chain(3, start=20)
        self.kill_publication_at("progress_replace")
        record_before = _read(self.recovery)
        backups_before = {
            name: _read(os.path.join(self.tmp.name, name))
            for name in os.listdir(self.tmp.name)
            if name.startswith(".recovery-cache-backup-")
        }
        self._assert_isolation()
        self.assertEqual(_read(self.recovery), record_before)
        for name, data in backups_before.items():
            self.assertEqual(_read(os.path.join(self.tmp.name, name)), data)

    def test_rejected_observation_is_read_only(self) -> None:
        self.grow_chain(4)
        self.build()
        with open(self.recovery, "wb") as handle:
            handle.write(b"garbage")
        self._assert_isolation()
        self.assertEqual(_read(self.recovery), b"garbage")

    def test_existing_recovery_workflow_unchanged(self) -> None:
        head = self.grow_chain(5)
        self.assertEqual(self.build(), head)
        self.assertFalse(os.path.exists(self.recovery))
        # Observations do not perturb subsequent recovery or queries.
        self.diagnostic()
        self.assertEqual(self.build(), head)
        results = BranchStore.diff_recovery_audit_ranges(
            self.chain,
            head,
            ((1, 5),),
            self.index,
            self.SEGMENT_SIZE,
            self.progress,
            self.recovery,
        )
        self.assertEqual(
            results[0]["changes"],
            BranchStore.diff_recovery_audit_range(self.chain, head, 1, 5),
        )
        self.assertFalse(os.path.exists(self.recovery))
        self.assertEqual(self.residue(), [])


if __name__ == "__main__":
    unittest.main()
