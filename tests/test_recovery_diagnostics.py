"""Tests for the read-only cross-restart recovery diagnostics.

``BranchStore.recovery_diagnostic`` observes the durable evidence of an
interrupted two-cache publication (segment index, progress cursor and
checksummed recovery record) without performing any recovery and
without modifying anything; ``BranchStore.verify_recovery_diagnostics``
authenticates a consecutive sequence of those observations.

These tests cover:

* the idle observation when no record and no predecessor exist, and
  the compact, BOM-less, newline-free, checksummed, deterministic
  envelope shape;
* ``pending_commit`` / ``pending_rollback`` observations at each crash
  point, including per-target old/new/same verdicts and backup status;
* continuing one transaction after its record was cleaned up: a
  pending observation settles to ``committed`` or ``rolled_back`` from
  the predecessor's summary and terminal observations carry forward;
* every stable rejection reason -- a corrupt record (identified by the
  SHA-256 of its raw bytes, never echoing material), a target matching
  neither recorded digest, a phase/digest contradiction, a missing or
  corrupt required backup, a disposed transaction's record being
  reprocessed, and a rejected predecessor whose record vanished;
* sequence authentication: genesis/prev links, constant transaction
  identity, no evidence regression or disposition reversal, malformed
  records raising ``ValueError`` and container/element type errors
  raising ``TypeError``;
* path and predecessor argument validation;
* missing/unreadable index or progress raising ``OSError`` while
  corrupt recovery material is expressed as a rejected diagnostic;
* strict read-only isolation: targets, backups, record, canonical
  chain, generation directories and business state stay byte-for-byte
  untouched.
"""

import hashlib
import json
import os
import unittest
from unittest import mock

from city_twin.branches import BranchStore

from tests.test_recovery_audit_recovery import (
    RecoveryPublicationTestBase,
)


def reseal(document: dict) -> str:
    """Re-serialize a (possibly tampered) diagnostic document with a
    fresh checksum over every other field."""
    document = dict(document)
    body = {
        key: value
        for key, value in document.items()
        if key != "checksum"
    }
    document["checksum"] = hashlib.sha256(
        BranchStore._canonical_json(body).encode("utf-8")
    ).hexdigest()
    return BranchStore._canonical_json(document)


class IdleDiagnosticTests(RecoveryPublicationTestBase):
    def test_no_record_no_predecessor_is_idle(self) -> None:
        self.grow_chain(5)
        self.build()
        self.assertFalse(os.path.exists(self.recovery))
        text = BranchStore.recovery_diagnostic(
            self.index, self.progress, self.recovery
        )
        document = json.loads(text)
        self.assertEqual(
            document["disposition"], "idle"
        )
        self.assertEqual(document["reason"], "idle")
        self.assertEqual(document["transaction"], "")
        self.assertEqual(document["phase"], "")
        self.assertEqual(document["prev"], "0" * 64)
        self.assertEqual(
            document["index"], {"state": "present", "matches": ""}
        )
        self.assertEqual(
            document["progress"], {"state": "present", "matches": ""}
        )
        self.assertEqual(
            document["backups"],
            {
                "index": {"present": False, "intact": False},
                "progress": {"present": False, "intact": False},
            },
        )

    def test_missing_targets_raise_oserror_not_an_idle_report(self) -> None:
        # No build ever ran: both caches are absent, which is an
        # observation failure, never an idle verdict.
        self.grow_chain(2)
        with self.assertRaises(OSError):
            BranchStore.recovery_diagnostic(
                self.index, self.progress, self.recovery
            )

    def test_envelope_is_compact_checksummed_and_deterministic(self) -> None:
        self.grow_chain(3)
        self.build()
        first = BranchStore.recovery_diagnostic(
            self.index, self.progress, self.recovery
        )
        self.assertFalse(first.startswith("﻿"))
        self.assertFalse(first.endswith("\n"))
        self.assertNotIn("\n", first)
        self.assertNotIn(" ", first)
        document = json.loads(first)
        self.assertEqual(
            set(document),
            {
                "format",
                "version",
                "transaction",
                "prev",
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
            document["format"],
            "branching-city-twin/recovery-diagnostic",
        )
        self.assertEqual(document["version"], 1)
        body = {key: value for key, value in document.items() if key != "checksum"}
        self.assertEqual(
            document["checksum"],
            hashlib.sha256(
                BranchStore._canonical_json(body).encode("utf-8")
            ).hexdigest(),
        )
        # Same evidence, byte-for-byte identical output.
        self.assertEqual(
            first,
            BranchStore.recovery_diagnostic(
                self.index, self.progress, self.recovery
            ),
        )
        self.assertTrue(
            BranchStore.verify_recovery_diagnostics((first,))
        )


class PendingDiagnosticTests(RecoveryPublicationTestBase):
    def setUp(self) -> None:
        super().setUp()
        self.grow_chain(5)
        self.build()
        self.grow_chain(3, start=20)

    def test_mixed_version_is_pending_rollback(self) -> None:
        self.kill_publication_at("progress_replace")
        text = BranchStore.recovery_diagnostic(
            self.index, self.progress, self.recovery
        )
        document = json.loads(text)
        self.assertEqual(document["disposition"], "pending_rollback")
        self.assertEqual(document["reason"], "pending_rollback")
        self.assertEqual(document["phase"], "index")
        self.assertEqual(
            document["transaction"], self.record_document()["checksum"]
        )
        self.assertEqual(
            document["index"], {"state": "new", "matches": "new"}
        )
        self.assertEqual(
            document["progress"], {"state": "old", "matches": "old"}
        )
        for name in ("index", "progress"):
            self.assertEqual(
                document["backups"][name],
                {"present": True, "intact": True},
            )
        self.assertTrue(
            BranchStore.verify_recovery_diagnostics((text,))
        )

    def test_complete_new_version_is_pending_commit(self) -> None:
        self.kill_publication_at("cleanup")
        document = json.loads(
            BranchStore.recovery_diagnostic(
                self.index, self.progress, self.recovery
            )
        )
        self.assertEqual(document["disposition"], "pending_commit")
        self.assertEqual(document["reason"], "pending_commit")
        self.assertEqual(document["phase"], "progress")
        self.assertEqual(
            document["index"], {"state": "new", "matches": "new"}
        )
        self.assertEqual(
            document["progress"], {"state": "new", "matches": "new"}
        )

    def test_prepared_phase_observation(self) -> None:
        self.kill_publication_at("index_replace")
        document = json.loads(
            BranchStore.recovery_diagnostic(
                self.index, self.progress, self.recovery
            )
        )
        self.assertEqual(document["disposition"], "pending_rollback")
        self.assertEqual(document["phase"], "prepared")
        self.assertEqual(
            document["index"], {"state": "old", "matches": "old"}
        )
        self.assertEqual(
            document["progress"], {"state": "old", "matches": "old"}
        )

    def test_same_evidence_observed_twice_stays_linked(self) -> None:
        self.kill_publication_at("index_replace")
        first = BranchStore.recovery_diagnostic(
            self.index, self.progress, self.recovery
        )
        # Nothing changes between two read-only observations: the
        # evidence and transaction stay identical and the records link.
        second = BranchStore.recovery_diagnostic(
            self.index, self.progress, self.recovery, first
        )
        self.assertEqual(
            json.loads(second)["transaction"],
            json.loads(first)["transaction"],
        )
        self.assertEqual(
            json.loads(second)["disposition"], "pending_rollback"
        )
        self.assertTrue(
            BranchStore.verify_recovery_diagnostics((first, second))
        )


class DispositionContinuationTests(RecoveryPublicationTestBase):
    def test_pending_rollback_settles_after_record_cleanup(self) -> None:
        self.grow_chain(5)
        self.build()
        self.grow_chain(3, start=20)
        self.kill_publication_at("progress_replace")
        pending = BranchStore.recovery_diagnostic(
            self.index, self.progress, self.recovery
        )
        self.assertEqual(
            json.loads(pending)["disposition"], "pending_rollback"
        )
        transaction = json.loads(pending)["transaction"]
        self.recover_now()
        self.assertFalse(os.path.exists(self.recovery))
        settled = BranchStore.recovery_diagnostic(
            self.index, self.progress, self.recovery, pending
        )
        document = json.loads(settled)
        self.assertEqual(document["disposition"], "rolled_back")
        self.assertEqual(document["reason"], "rolled_back")
        self.assertEqual(document["transaction"], transaction)
        self.assertEqual(document["phase"], "")
        self.assertEqual(
            document["index"], {"state": "present", "matches": ""}
        )
        self.assertTrue(
            BranchStore.verify_recovery_diagnostics(
                (pending, settled)
            )
        )
        # Terminal observations carry forward across repeated calls
        # after the record is gone.
        carried = BranchStore.recovery_diagnostic(
            self.index, self.progress, self.recovery, settled
        )
        self.assertEqual(
            json.loads(carried)["disposition"], "rolled_back"
        )
        self.assertTrue(
            BranchStore.verify_recovery_diagnostics(
                (pending, settled, carried)
            )
        )

    def test_pending_commit_settles_after_record_cleanup(self) -> None:
        self.grow_chain(5)
        self.build()
        self.grow_chain(3, start=20)
        self.kill_publication_at("cleanup")
        pending = BranchStore.recovery_diagnostic(
            self.index, self.progress, self.recovery
        )
        self.assertEqual(
            json.loads(pending)["disposition"], "pending_commit"
        )
        self.recover_now()
        settled = BranchStore.recovery_diagnostic(
            self.index, self.progress, self.recovery, pending
        )
        self.assertEqual(json.loads(settled)["disposition"], "committed")
        self.assertTrue(
            BranchStore.verify_recovery_diagnostics(
                (pending, settled)
            )
        )

    def test_idle_chain_opens_a_transaction(self) -> None:
        self.grow_chain(5)
        self.build()
        idle = BranchStore.recovery_diagnostic(
            self.index, self.progress, self.recovery
        )
        self.assertEqual(json.loads(idle)["disposition"], "idle")
        self.grow_chain(3, start=20)
        self.kill_publication_at("progress_replace")
        pending = BranchStore.recovery_diagnostic(
            self.index, self.progress, self.recovery, idle
        )
        self.assertEqual(
            json.loads(pending)["disposition"], "pending_rollback"
        )
        self.assertTrue(
            BranchStore.verify_recovery_diagnostics((idle, pending))
        )

    def test_repeated_idle_observations_stay_idle(self) -> None:
        self.grow_chain(5)
        self.build()
        first = BranchStore.recovery_diagnostic(
            self.index, self.progress, self.recovery
        )
        second = BranchStore.recovery_diagnostic(
            self.index, self.progress, self.recovery, first
        )
        self.assertEqual(json.loads(second)["disposition"], "idle")
        self.assertEqual(
            json.loads(second)["transaction"], ""
        )
        self.assertTrue(
            BranchStore.verify_recovery_diagnostics((first, second))
        )

    def test_idle_observation_then_corrupt_record_is_continuous(self) -> None:
        self.grow_chain(2)
        # No build: create the two cache files directly so the
        # observation can run.
        with open(self.index, "wb") as handle:
            handle.write(b"{}")
        with open(self.progress, "wb") as handle:
            handle.write(b"{}")
        idle = BranchStore.recovery_diagnostic(
            self.index, self.progress, self.recovery
        )
        with open(self.recovery, "wb") as handle:
            handle.write(b"garbage")
        rejected = BranchStore.recovery_diagnostic(
            self.index, self.progress, self.recovery, idle
        )
        self.assertEqual(
            json.loads(rejected)["reason"], "record_corrupt"
        )
        self.assertTrue(
            BranchStore.verify_recovery_diagnostics((idle, rejected))
        )


class RejectionReasonTests(RecoveryPublicationTestBase):
    def setUp(self) -> None:
        super().setUp()
        self.grow_chain(5)
        self.build()
        self.grow_chain(3, start=20)
        self.kill_publication_at("progress_replace")

    def diagnostic(self, previous=None) -> dict:
        return json.loads(
            BranchStore.recovery_diagnostic(
                self.index,
                self.progress,
                self.recovery,
                previous,
            )
        )

    def test_corrupt_record_uses_raw_byte_digest_and_no_echo(self) -> None:
        for raw, label in (
            (b"", "empty"),
            (b"garbage", "garbage"),
            (b"\xef\xbb\xbf" + b"{}", "bom"),
        ):
            with self.subTest(label=label):
                with open(self.recovery, "wb") as handle:
                    handle.write(raw)
                document = self.diagnostic()
                self.assertEqual(
                    document["disposition"], "rejected"
                )
                self.assertEqual(document["reason"], "record_corrupt")
                self.assertEqual(
                    document["transaction"],
                    hashlib.sha256(raw).hexdigest(),
                )
                self.assertEqual(document["phase"], "")
                # A record that cannot be parsed yields no recorded
                # old/new digests, so targets are only observed.
                self.assertEqual(
                    document["index"],
                    {"state": "present", "matches": ""},
                )
                self.assertEqual(
                    document["progress"],
                    {"state": "present", "matches": ""},
                )
                self.assertEqual(
                    document["backups"],
                    {
                        "index": {"present": False, "intact": False},
                        "progress": {"present": False, "intact": False},
                    },
                )
                # The record's bytes never appear anywhere in the
                # diagnostic string.
                text = BranchStore.recovery_diagnostic(
                    self.index, self.progress, self.recovery
                )
                self.assertNotIn("garbage", text)

    def test_target_with_unknown_bytes_is_target_mismatch(self) -> None:
        with open(self.progress, "wb") as handle:
            handle.write(b"neither recorded version")
        document = self.diagnostic()
        self.assertEqual(document["disposition"], "rejected")
        self.assertEqual(document["reason"], "target_mismatch")
        self.assertEqual(
            document["progress"], {"state": "unknown", "matches": ""}
        )
        self.assertEqual(
            document["index"], {"state": "new", "matches": "new"}
        )

    def test_phase_contradiction_is_reported(self) -> None:
        # Clear the setUp crash, then capture the prepared record of a
        # fresh publication that dies with both targets already
        # carrying the new bytes, and reinstall the lagging
        # prepared-phase record.
        self.recover_now()
        writes: dict[str, bytes] = {}
        real_write = BranchStore._write_cache_recovery_record

        def spy(recovery_path, data):
            phase = json.loads(data.decode("utf-8"))["phase"]
            writes.setdefault(phase, bytes(data))
            return real_write(recovery_path, data)

        with mock.patch.object(
            BranchStore,
            "_write_cache_recovery_record",
            staticmethod(spy),
        ):
            self.kill_publication_at("cleanup")
        self.assertIn("prepared", writes)
        with open(self.recovery, "wb") as handle:
            handle.write(writes["prepared"])
        document = self.diagnostic()
        self.assertEqual(document["disposition"], "rejected")
        self.assertEqual(document["reason"], "phase_contradiction")
        self.assertEqual(document["phase"], "prepared")
        self.assertEqual(
            document["index"], {"state": "new", "matches": "new"}
        )
        self.assertEqual(
            document["progress"], {"state": "new", "matches": "new"}
        )

    def test_missing_required_backup_is_reported(self) -> None:
        record = self.record_document()
        os.remove(
            os.path.join(self.tmp.name, record["index"]["backup"])
        )
        document = self.diagnostic()
        self.assertEqual(document["disposition"], "rejected")
        self.assertEqual(document["reason"], "backup_missing")
        self.assertEqual(
            document["backups"]["index"],
            {"present": False, "intact": False},
        )

    def test_corrupt_backup_is_reported(self) -> None:
        record = self.record_document()
        with open(
            os.path.join(self.tmp.name, record["index"]["backup"]), "wb"
        ) as handle:
            handle.write(b"corrupt backup bytes")
        document = self.diagnostic()
        self.assertEqual(document["disposition"], "rejected")
        self.assertEqual(document["reason"], "backup_corrupt")
        self.assertEqual(
            document["backups"]["index"],
            {"present": True, "intact": False},
        )

    def test_reprocessed_transaction_is_rejected(self) -> None:
        pending = BranchStore.recovery_diagnostic(
            self.index, self.progress, self.recovery
        )
        self.assertEqual(
            json.loads(pending)["disposition"], "pending_rollback"
        )
        record_bytes = self._read_bytes(self.recovery)
        self.recover_now()
        settled = BranchStore.recovery_diagnostic(
            self.index, self.progress, self.recovery, pending
        )
        self.assertEqual(
            json.loads(settled)["disposition"], "rolled_back"
        )
        # The same durable record reappears after disposition.
        with open(self.recovery, "wb") as handle:
            handle.write(record_bytes)
        reprocessed = BranchStore.recovery_diagnostic(
            self.index, self.progress, self.recovery, settled
        )
        document = json.loads(reprocessed)
        self.assertEqual(document["disposition"], "rejected")
        self.assertEqual(document["reason"], "reprocessed")
        # The reprocessed observation links its predecessor, but a
        # disposed transaction being reprocessed must never authenticate
        # as a continuous sequence.
        self.assertTrue(
            BranchStore.verify_recovery_diagnostics(
                (pending, settled)
            )
        )
        self.assertFalse(
            BranchStore.verify_recovery_diagnostics(
                (pending, settled, reprocessed)
            )
        )

    def test_rejected_predecessor_losing_record_contradicts(self) -> None:
        with open(self.recovery, "wb") as handle:
            handle.write(b"garbage")
        rejected = BranchStore.recovery_diagnostic(
            self.index, self.progress, self.recovery
        )
        self.assertEqual(
            json.loads(rejected)["reason"], "record_corrupt"
        )
        # The protocol never removes a rejected record; its
        # disappearance contradicts the predecessor.
        os.remove(self.recovery)
        contradicted = BranchStore.recovery_diagnostic(
            self.index, self.progress, self.recovery, rejected
        )
        document = json.loads(contradicted)
        self.assertEqual(document["disposition"], "rejected")
        self.assertEqual(
            document["reason"], "transaction_contradiction"
        )
        self.assertTrue(
            BranchStore.verify_recovery_diagnostics(
                (rejected, contradicted)
            )
        )

    @staticmethod
    def _read_bytes(path: str) -> bytes:
        with open(path, "rb") as handle:
            return handle.read()


class VerifyRecoveryDiagnosticsTests(unittest.TestCase):
    def _document(self, **overrides) -> dict:
        document = {
            "format": "branching-city-twin/recovery-diagnostic",
            "version": 1,
            "transaction": "a" * 64,
            "prev": "0" * 64,
            "phase": "index",
            "index": {"state": "new", "matches": "new"},
            "progress": {"state": "old", "matches": "old"},
            "backups": {
                "index": {"present": True, "intact": True},
                "progress": {"present": True, "intact": True},
            },
            "disposition": "pending_rollback",
            "reason": "pending_rollback",
            "checksum": "",
        }
        document.update(overrides)
        return document

    def test_empty_tuple_is_true(self) -> None:
        self.assertTrue(
            BranchStore.verify_recovery_diagnostics(())
        )

    def test_container_and_element_types(self) -> None:
        with self.assertRaises(TypeError):
            BranchStore.verify_recovery_diagnostics([])  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            BranchStore.verify_recovery_diagnostics(("ok", 1))  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            BranchStore.verify_recovery_diagnostics(None)  # type: ignore[arg-type]

    def test_first_record_must_root_at_genesis(self) -> None:
        text = reseal(self._document(prev="1" * 64))
        self.assertFalse(
            BranchStore.verify_recovery_diagnostics((text,))
        )

    def test_first_record_cannot_be_terminal(self) -> None:
        terminal_shape = {
            "phase": "",
            "index": {"state": "present", "matches": ""},
            "progress": {"state": "present", "matches": ""},
            "backups": {
                "index": {"present": False, "intact": False},
                "progress": {"present": False, "intact": False},
            },
        }
        text = reseal(
            self._document(
                disposition="committed",
                reason="committed",
                **terminal_shape,
            )
        )
        self.assertFalse(
            BranchStore.verify_recovery_diagnostics((text,))
        )

    def test_unreachable_pending_observation_is_invalid(self) -> None:
        # Both targets on old bytes can never be pending_commit.
        text = reseal(
            self._document(
                phase="progress",
                index={"state": "old", "matches": "old"},
                progress={"state": "old", "matches": "old"},
                disposition="pending_commit",
                reason="pending_commit",
            )
        )
        with self.assertRaises(ValueError):
            BranchStore.verify_recovery_diagnostics((text,))
        # A new target that a rollback would restore needs its backup.
        text = reseal(
            self._document(
                backups={
                    "index": {"present": False, "intact": False},
                    "progress": {"present": True, "intact": True},
                },
            )
        )
        with self.assertRaises(ValueError):
            BranchStore.verify_recovery_diagnostics((text,))

    def test_broken_prev_link_returns_false(self) -> None:
        first_doc = json.loads(reseal(self._document()))
        terminal_shape = {
            "phase": "",
            "index": {"state": "present", "matches": ""},
            "progress": {"state": "present", "matches": ""},
            "backups": {
                "index": {"present": False, "intact": False},
                "progress": {"present": False, "intact": False},
            },
        }
        second_text = reseal(
            self._document(
                prev="f" * 64,
                disposition="rolled_back",
                reason="rolled_back",
                **terminal_shape,
            )
        )
        self.assertFalse(
            BranchStore.verify_recovery_diagnostics(
                (
                    BranchStore._canonical_json(first_doc),
                    second_text,
                )
            )
        )

    def test_transaction_mutation_returns_false(self) -> None:
        first = reseal(self._document())
        second = json.loads(first)
        second["transaction"] = "b" * 64
        second["prev"] = json.loads(first)["checksum"]
        self.assertFalse(
            BranchStore.verify_recovery_diagnostics(
                (first, reseal(second))
            )
        )

    def test_forbidden_transitions_return_false(self) -> None:
        first = reseal(self._document())
        chain_head = json.loads(first)["checksum"]

        def successor(**overrides) -> str:
            return reseal(self._document(prev=chain_head, **overrides))

        terminal_shape = {
            "phase": "",
            "index": {"state": "present", "matches": ""},
            "progress": {"state": "present", "matches": ""},
            "backups": {
                "index": {"present": False, "intact": False},
                "progress": {"present": False, "intact": False},
            },
        }
        # pending_rollback cannot become committed or stay evidence-new
        self.assertFalse(
            BranchStore.verify_recovery_diagnostics(
                (
                    first,
                    successor(
                        disposition="committed",
                        reason="committed",
                        **terminal_shape,
                    ),
                )
            )
        )
        self.assertFalse(
            BranchStore.verify_recovery_diagnostics(
                (
                    first,
                    successor(
                        disposition="pending_commit",
                        reason="pending_commit",
                        phase="progress",
                        index={"state": "new", "matches": "new"},
                        progress={"state": "new", "matches": "new"},
                    ),
                )
            )
        )
        # rolled_back cannot go back to pending.
        rolled_back = successor(
            disposition="rolled_back",
            reason="rolled_back",
            **terminal_shape,
        )
        rolled_doc = json.loads(rolled_back)
        again = reseal(
            self._document(
                prev=rolled_doc["checksum"],
                disposition="pending_rollback",
                reason="pending_rollback",
            )
        )
        self.assertFalse(
            BranchStore.verify_recovery_diagnostics(
                (first, rolled_back, again)
            )
        )

    def test_terminal_state_carries_forward(self) -> None:
        first = reseal(self._document())
        head = json.loads(first)["checksum"]
        terminal_shape = {
            "phase": "",
            "index": {"state": "present", "matches": ""},
            "progress": {"state": "present", "matches": ""},
            "backups": {
                "index": {"present": False, "intact": False},
                "progress": {"present": False, "intact": False},
            },
        }
        settled = reseal(
            self._document(
                prev=head,
                disposition="rolled_back",
                reason="rolled_back",
                **terminal_shape,
            )
        )
        carried = reseal(
            self._document(
                prev=json.loads(settled)["checksum"],
                disposition="rolled_back",
                reason="rolled_back",
                **terminal_shape,
            )
        )
        self.assertTrue(
            BranchStore.verify_recovery_diagnostics(
                (first, settled, carried)
            )
        )

    def test_phase_regression_returns_false(self) -> None:
        # Same pending disposition, but the observed phase moves from
        # "index" back to "prepared": the evidence regressed.
        first = reseal(self._document(phase="index"))
        regressed = reseal(
            self._document(
                prev=json.loads(first)["checksum"],
                phase="prepared",
                index={"state": "old", "matches": "old"},
                progress={"state": "old", "matches": "old"},
            )
        )
        self.assertFalse(
            BranchStore.verify_recovery_diagnostics(
                (first, regressed)
            )
        )

    def test_malformed_records_raise_value_error(self) -> None:
        good = json.loads(reseal(self._document()))
        cases = []
        # Non-canonical JSON (whitespace).
        cases.append(json.dumps(good, indent=2))
        # Duplicate key.
        cases.append(
            BranchStore._canonical_json(good)[:-1]
            + ',"reason":"pending_rollback"}'
        )
        # Unsupported version (checksum resealed).
        cases.append(reseal(self._document(version=2)))
        # Extra key.
        cases.append(reseal(self._document(extra=1)))
        # Unknown disposition.
        cases.append(
            reseal(self._document(disposition="halfway"))
        )
        # State/matches disagreement.
        cases.append(
            reseal(
                self._document(
                    index={"state": "new", "matches": "old"}
                )
            )
        )
        # Intact backup that is absent.
        cases.append(
            reseal(
                self._document(
                    backups={
                        "index": {"present": False, "intact": True},
                        "progress": {"present": True, "intact": True},
                    }
                )
            )
        )
        # Bad checksum (no reseal).
        tampered = self._document()
        tampered["checksum"] = "f" * 64
        cases.append(
            BranchStore._canonical_json(tampered)
        )
        for case in cases:
            with self.subTest(case=case[:32]):
                with self.assertRaises(ValueError):
                    BranchStore.verify_recovery_diagnostics((case,))


class ArgumentValidationTests(RecoveryPublicationTestBase):
    def setUp(self) -> None:
        super().setUp()
        self.grow_chain(2)

    def test_path_types(self) -> None:
        for bad in (1, 1.5, b"x", ["x"], {"x": 1}, object(), None):
            with self.subTest(bad=type(bad).__name__):
                with self.assertRaises(TypeError):
                    BranchStore.recovery_diagnostic(
                        bad, self.progress, self.recovery  # type: ignore[arg-type]
                    )
                with self.assertRaises(TypeError):
                    BranchStore.recovery_diagnostic(
                        self.index, bad, self.recovery  # type: ignore[arg-type]
                    )
                with self.assertRaises(TypeError):
                    BranchStore.recovery_diagnostic(
                        self.index, self.progress, bad  # type: ignore[arg-type]
                    )

    def test_empty_paths(self) -> None:
        with self.assertRaises(ValueError):
            BranchStore.recovery_diagnostic(
                "", self.progress, self.recovery
            )
        with self.assertRaises(ValueError):
            BranchStore.recovery_diagnostic(
                self.index, "", self.recovery
            )
        with self.assertRaises(ValueError):
            BranchStore.recovery_diagnostic(
                self.index, self.progress, ""
            )

    def test_paths_must_be_distinct(self) -> None:
        with self.assertRaises(ValueError):
            BranchStore.recovery_diagnostic(
                self.index, self.index, self.recovery
            )
        link = os.path.join(self.tmp.name, "index-link")
        os.symlink(self.index, link)
        with self.assertRaises(ValueError):
            BranchStore.recovery_diagnostic(
                self.index,
                link,
                os.path.join(self.tmp.name, "r.json"),
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

    def test_previous_types(self) -> None:
        for bad in (1, 1.5, b"x", [], {}, object()):
            with self.subTest(bad=type(bad).__name__):
                with self.assertRaises(TypeError):
                    BranchStore.recovery_diagnostic(
                        self.index,
                        self.progress,
                        self.recovery,
                        bad,  # type: ignore[arg-type]
                    )

    def test_previous_empty_string(self) -> None:
        with self.assertRaises(ValueError):
            BranchStore.recovery_diagnostic(
                self.index, self.progress, self.recovery, ""
            )

    def test_previous_malformed(self) -> None:
        for bad in (
            "garbage",
            "{}",
            '{"format":"branching-city-twin/recovery-diagnostic"}',
        ):
            with self.subTest(bad=bad[:16]):
                with self.assertRaises(ValueError):
                    BranchStore.recovery_diagnostic(
                        self.index,
                        self.progress,
                        self.recovery,
                        bad,
                    )
        # A well-formed envelope with a wrong checksum.
        document = {
            "format": "branching-city-twin/recovery-diagnostic",
            "version": 1,
            "transaction": "",
            "prev": "0" * 64,
            "phase": "",
            "index": {"state": "absent", "matches": ""},
            "progress": {"state": "absent", "matches": ""},
            "backups": {
                "index": {"present": False, "intact": False},
                "progress": {"present": False, "intact": False},
            },
            "disposition": "idle",
            "reason": "idle",
            "checksum": "0" * 64,
        }
        with self.assertRaises(ValueError):
            BranchStore.recovery_diagnostic(
                self.index,
                self.progress,
                self.recovery,
                BranchStore._canonical_json(document),
            )


class OSErrorSurfaceTests(RecoveryPublicationTestBase):
    def test_missing_index_raises_oserror(self) -> None:
        self.grow_chain(2)
        with self.assertRaises(OSError):
            BranchStore.recovery_diagnostic(
                os.path.join(self.tmp.name, "missing-index.json"),
                self.progress,
                self.recovery,
            )

    def test_missing_progress_raises_oserror(self) -> None:
        self.grow_chain(2)
        with self.assertRaises(OSError):
            BranchStore.recovery_diagnostic(
                self.index,
                os.path.join(self.tmp.name, "missing-progress.json"),
                self.recovery,
            )

    def test_directory_as_target_raises_oserror(self) -> None:
        self.grow_chain(2)
        sub = os.path.join(self.tmp.name, "adir")
        os.mkdir(sub)
        with self.assertRaises(OSError):
            BranchStore.recovery_diagnostic(
                sub, self.progress, self.recovery
            )

    def test_recovery_record_being_a_directory_raises_oserror(self) -> None:
        self.grow_chain(2)
        with open(self.index, "wb") as handle:
            handle.write(b"{}")
        with open(self.progress, "wb") as handle:
            handle.write(b"{}")
        record_dir = os.path.join(self.tmp.name, "record-dir")
        os.mkdir(record_dir)
        with self.assertRaises(OSError):
            BranchStore.recovery_diagnostic(
                self.index, self.progress, record_dir
            )


class ReadOnlyIsolationTests(RecoveryPublicationTestBase):
    def setUp(self) -> None:
        super().setUp()
        self.grow_chain(5)
        self.build()
        self.grow_chain(3, start=20)
        self.kill_publication_at("progress_replace")

    def _snapshot(self) -> dict[str, bytes]:
        captured = {}
        for name in os.listdir(self.tmp.name):
            path = os.path.join(self.tmp.name, name)
            if os.path.isfile(path) or os.path.islink(path):
                with open(path, "rb") as handle:
                    captured[name] = handle.read()
        return captured

    def test_diagnostics_modify_nothing(self) -> None:
        before = self._snapshot()
        generations = self.root_snapshot()
        first = BranchStore.recovery_diagnostic(
            self.index, self.progress, self.recovery
        )
        BranchStore.recovery_diagnostic(
            self.index, self.progress, self.recovery, first
        )
        # A corrupt record is also read-only.
        with open(self.recovery, "wb") as handle:
            handle.write(b"garbage")
        before_corrupt = self._snapshot()
        BranchStore.recovery_diagnostic(
            self.index, self.progress, self.recovery
        )
        self.assertEqual(self._snapshot(), before_corrupt)
        # Restore the interrupted record and confirm everything else
        # was untouched by the earlier observations.
        with open(self.recovery, "wb") as handle:
            handle.write(before[os.path.basename(self.recovery)])
        self.assertEqual(self._snapshot(), before)
        self.assertEqual(self.root_snapshot(), generations)

    def test_diagnostic_does_not_read_the_chain(self) -> None:
        # The canonical chain is irrelevant to the observation: even a
        # missing chain file changes nothing and raises nothing.
        chain_bytes = self._read_bytes(self.chain)
        os.remove(self.chain)
        try:
            document = json.loads(
                BranchStore.recovery_diagnostic(
                    self.index, self.progress, self.recovery
                )
            )
            self.assertEqual(
                document["disposition"], "pending_rollback"
            )
            self.assertFalse(os.path.exists(self.chain))
        finally:
            with open(self.chain, "wb") as handle:
                handle.write(chain_bytes)

    def test_no_recovery_is_performed(self) -> None:
        # After a pure observation the real recovery entry still sees
        # the exact interrupted publication.
        record_before = self._read_bytes(self.recovery)
        index_before = self._read_bytes(self.index)
        progress_before = self._read_bytes(self.progress)
        BranchStore.recovery_diagnostic(
            self.index, self.progress, self.recovery
        )
        self.assertEqual(
            self._read_bytes(self.recovery), record_before
        )
        self.assertEqual(self._read_bytes(self.index), index_before)
        self.assertEqual(
            self._read_bytes(self.progress), progress_before
        )
        # The state machine still has work to do.
        self.recover_now()
        self.assertFalse(os.path.exists(self.recovery))

    def test_no_leftover_temp_files(self) -> None:
        # The interrupted publication legitimately leaves its backups;
        # the diagnostic must neither add nor remove any directory
        # entry.
        before = set(os.listdir(self.tmp.name))
        BranchStore.recovery_diagnostic(
            self.index, self.progress, self.recovery
        )
        after = set(os.listdir(self.tmp.name))
        self.assertEqual(after, before)

    @staticmethod
    def _read_bytes(path: str) -> bytes:
        with open(path, "rb") as handle:
            return handle.read()


if __name__ == "__main__":
    unittest.main()
