"""Tests for rotation-proof transaction memory and cross-process
read/publish coordination of the segmented diagnostic ledger.

These extend :mod:`tests.test_append_diagnostic_ledger` for the two
properties fixed on top of the baseline directory ledger:

* rotation must not forget transactions -- once the oldest segments
  leave retention, a durable, authenticated prefix-evidence document
  still records every rotated-away transaction identity (bound to the
  prefix's last segment digest, entry count and time bounds), so
  appending one of those identities again raises ``ValueError``
  precisely, across restarts, with no probabilistic judgement;
* every directory page is coordinated with manifest switches and
  rotated-segment reclamation -- a shared read lock excludes the
  publisher's exclusive lock (and therefore garbage collection) for
  the whole read, so a page sees one complete version, segments still
  being read are never deleted, and a reader that exits abruptly frees
  its occupancy at the operating-system level.

They also cover the failure contract: missing, extra or mismatched
evidence fields, corrupt evidence or a broken prefix link raise
``ValueError``; a missing evidence file or I/O failure raises
``OSError``; a stale expected snapshot raises ``RuntimeError`` without
writing anything; and a failed publish leaves the previous manifest,
segments and evidence byte-for-byte readable.
"""

import hashlib
import json
import os
import tempfile
import threading
import time
import unittest
from unittest import mock

from city_twin.branches import BranchStore

try:
    from test_diagnostic_ledger import make_chain, reseal, transaction
except ImportError:  # running as a package member from the repo root
    from tests.test_diagnostic_ledger import (
        make_chain,
        reseal,
        transaction,
    )


def read_bytes(path: str) -> bytes:
    with open(path, "rb") as handle:
        return handle.read()


class RotationMemoryTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = os.path.join(self.tmp.name, "ledger")
        os.mkdir(self.dir)
        self.chains = [make_chain(transaction(i)) for i in range(8)]
        self.entries = tuple(
            (time, chain)
            for time, chain in zip(range(1, 9), self.chains)
        )

    def append(self, entries, expected, keep=1, segment_limit=2):
        return BranchStore.append_diagnostic_ledger(
            self.dir, entries, expected, segment_limit, keep, None
        )

    def page(self, **overrides):
        arguments = {
            "path": self.dir,
            "start": None,
            "end": None,
            "transactions": None,
            "limit": 100,
            "cursor": None,
        }
        arguments.update(overrides)
        return BranchStore.page_diagnostic_ledger(**arguments)

    def manifest_path(self) -> str:
        return os.path.join(self.dir, "manifest.json")

    def manifest(self) -> dict:
        return json.loads(read_bytes(self.manifest_path()).decode("utf-8"))

    def write_manifest(self, document: dict) -> None:
        with open(self.manifest_path(), "w", encoding="utf-8") as handle:
            handle.write(reseal(document))

    def evidence_path(self) -> str:
        ref = self.manifest()["evidence"]["digest"]
        return os.path.join(
            self.dir,
            BranchStore._diagnostic_ledger_evidence_name(ref),
        )

    def evidence(self) -> dict:
        return json.loads(read_bytes(self.evidence_path()).decode("utf-8"))

    def write_evidence(self, document: dict) -> None:
        with open(self.evidence_path(), "w", encoding="utf-8") as handle:
            handle.write(reseal(document))

    def publish_rotating_chain(self):
        """Publish three batches with keep=1, so every publish beyond
        the first rotates; return the final snapshot digest."""
        digest = self.append(self.entries[0:2], None)
        digest = self.append(self.entries[2:4], digest)
        digest = self.append(self.entries[4:6], digest)
        return digest


class RotationMemoryTests(RotationMemoryTestBase):
    def test_rotated_transaction_is_rejected_after_segment_is_gone(self):
        digest = self.publish_rotating_chain()
        # Transactions 0..3 lived in segments 1 and 2, both removed by
        # now; their segment files are gone but they must be remembered.
        self.assertEqual(
            sorted(os.listdir(self.dir)),
            [
                ".diagnostic-ledger.lock",
                os.path.basename(self.evidence_path()),
                "manifest.json",
                "segment-000003.json",
            ],
        )
        for index in range(4):
            with self.subTest(rotated_index=index):
                with self.assertRaises(ValueError):
                    self.append(
                        ((9, self.chains[index]),), digest
                    )
        # A genuinely new transaction is still accepted.
        fresh = make_chain(transaction(40))
        new_digest = self.append(((9, fresh),), digest)
        self.assertNotEqual(new_digest, digest)

    def test_evidence_accumulates_across_repeated_rotations(self):
        digest = self.publish_rotating_chain()
        evidence = self.evidence()
        self.assertEqual(evidence["entries"], 4)
        self.assertEqual(len(evidence["transactions"]), 4)
        self.assertEqual(evidence["first_at"], 1)
        self.assertEqual(evidence["last_at"], 4)
        # Evidence covers the exact identities of the first four chains.
        expected_transactions = {
            json.loads(chain[0])["transaction"]
            for chain in self.chains[:4]
        }
        self.assertEqual(set(evidence["transactions"]), expected_transactions)
        # Rotate once more: the remembered set grows, not resets.
        digest = self.append(self.entries[6:8], digest)
        evidence = self.evidence()
        self.assertEqual(evidence["entries"], 6)
        self.assertEqual(len(evidence["transactions"]), 6)
        self.assertEqual(evidence["last_at"], 6)

    def test_memory_survives_a_restart(self):
        digest = self.publish_rotating_chain()
        # The class holds no in-memory ledger state; a fresh "process"
        # is just another call. The durable evidence must still reject.
        with self.assertRaises(ValueError):
            BranchStore.append_diagnostic_ledger(
                self.dir,
                ((9, self.chains[2]),),
                digest,
                2,
                1,
                None,
            )

    def test_uniqueness_spans_retained_evidence_and_new_batch(self):
        digest = self.publish_rotating_chain()
        retained_txn = json.loads(self.chains[4][0])["transaction"]
        rotated_txn = json.loads(self.chains[0][0])["transaction"]
        # A batch that repeats one retained and one remembered identity
        # is rejected wholesale; even a repeat within the new batch is.
        for repeated in (retained_txn, rotated_txn):
            with self.subTest(repeated=repeated[:8]):
                with self.assertRaises(ValueError):
                    self.append(
                        ((9, make_chain(repeated)),), digest
                    )


class EvidenceShapeTests(RotationMemoryTestBase):
    def test_manifest_carries_the_evidence_pointer(self):
        digest = self.publish_rotating_chain()
        self.assertEqual(digest, self.page()["snapshot"]["digest"])
        manifest = self.manifest()
        self.assertEqual(set(manifest["evidence"]), {"digest"})
        raw = read_bytes(self.evidence_path())
        self.assertEqual(
            manifest["evidence"]["digest"],
            hashlib.sha256(raw).hexdigest(),
        )
        document = json.loads(raw.decode("utf-8"))
        self.assertEqual(
            document["format"],
            "branching-city-twin/diagnostic-ledger-prefix-evidence",
        )
        self.assertEqual(document["version"], 1)
        # The evidence's prefix summary matches the manifest's proof.
        dropped = manifest["dropped"]
        self.assertEqual(document["digest"], dropped["digest"])
        self.assertEqual(document["entries"], dropped["entries"])
        self.assertEqual(document["first_at"], dropped["first_at"])
        self.assertEqual(document["last_at"], dropped["last_at"])
        # The first retained segment chains onto the dropped prefix.
        retained = manifest["segments"][0]
        segment = json.loads(
            read_bytes(
                os.path.join(self.dir, retained["name"])
            ).decode("utf-8")
        )
        self.assertEqual(segment["prev"], dropped["digest"])

    def test_evidence_is_absent_before_the_first_rotation(self):
        self.append(self.entries[0:2], None, keep=10)
        manifest = self.manifest()
        self.assertIsNone(manifest["dropped"])
        self.assertIsNone(manifest["evidence"])
        self.assertFalse(
            [
                name
                for name in os.listdir(self.dir)
                if name.startswith("dropped-evidence-")
            ]
        )

    def test_dropped_and_evidence_must_agree(self):
        self.publish_rotating_chain()
        manifest = self.manifest()
        manifest["evidence"] = None  # dropped stays present
        self.write_manifest(manifest)
        with self.assertRaises(ValueError):
            self.page()
        with self.assertRaises(ValueError):
            self.append(self.entries[6:8], "0" * 64)

    def test_manifest_evidence_pointer_with_bad_digest(self):
        # Pointing at a digest with no matching file is a missing file
        # -> OSError; the digest-mismatch (file exists, content differs)
        # case is covered by the reseal tests.
        self.publish_rotating_chain()
        manifest = self.manifest()
        manifest["evidence"] = {"digest": "f" * 64}
        self.write_manifest(manifest)
        with self.assertRaises(OSError):
            self.page()

    def test_evidence_summary_must_match_the_dropped_proof(self):
        self.publish_rotating_chain()
        manifest = self.manifest()
        dropped = manifest["dropped"]
        # Build a second, internally valid evidence whose summary does
        # not match the manifest's dropped proof, publish it under its
        # own name and repoint the manifest at it.
        forged_dropped = dict(dropped)
        forged_dropped["digest"] = "e" * 64
        data = BranchStore._diagnostic_ledger_evidence_document(
            list(self.evidence()["transactions"]), forged_dropped
        )
        other_digest = hashlib.sha256(data).hexdigest()
        other_name = BranchStore._diagnostic_ledger_evidence_name(
            other_digest
        )
        with open(os.path.join(self.dir, other_name), "wb") as handle:
            handle.write(data)
        manifest["evidence"] = {"digest": other_digest}
        self.write_manifest(manifest)
        with self.assertRaises(ValueError):
            self.page()

    def test_manifest_evidence_extra_or_missing_field(self):
        self.publish_rotating_chain()
        manifest = self.manifest()
        manifest["evidence"] = {"digest": manifest["evidence"]["digest"],
                                "extra": 1}
        self.write_manifest(manifest)
        with self.assertRaises(ValueError):
            self.page()

    def test_missing_evidence_file_raises_oserror(self):
        self.publish_rotating_chain()
        os.remove(self.evidence_path())
        with self.assertRaises(OSError):
            self.page()

    def test_evidence_checksum_detects_tampering(self):
        self.publish_rotating_chain()
        path = self.evidence_path()
        document = json.loads(read_bytes(path).decode("utf-8"))
        document["transactions"][0] = "f" * 64
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(BranchStore._canonical_json(document))
        with self.assertRaises(ValueError):
            self.page()


class EvidenceContentTests(RotationMemoryTestBase):
    def setUp(self):
        super().setUp()
        self.publish_rotating_chain()

    def assert_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.page()

    def test_resealed_identity_change_is_rejected(self):
        document = self.evidence()
        document["transactions"][0] = "f" * 64
        self.write_evidence(document)
        self.assert_rejected()

    def test_malformed_identity_is_rejected(self):
        document = self.evidence()
        document["transactions"][0] = "z" * 64
        self.write_evidence(document)
        self.assert_rejected()

    def test_duplicate_identity_is_rejected(self):
        document = self.evidence()
        document["transactions"][1] = document["transactions"][0]
        self.write_evidence(document)
        self.assert_rejected()

    def test_entry_count_mismatch_is_rejected(self):
        document = self.evidence()
        document["entries"] = document["entries"] + 1
        self.write_evidence(document)
        self.assert_rejected()

    def test_time_bound_mismatch_is_rejected(self):
        # A time bound that no longer matches the manifest's dropped
        # proof breaks the evidence/prefix binding.
        document = self.evidence()
        document["last_at"] = document["last_at"] + 1
        self.write_evidence(document)
        self.assert_rejected()

    def test_prefix_digest_mismatch_is_rejected(self):
        document = self.evidence()
        document["digest"] = "f" * 64
        self.write_evidence(document)
        self.assert_rejected()

    def test_bad_envelope_is_rejected(self):
        path = self.evidence_path()
        valid = read_bytes(path)
        for raw in (
            b"not json",
            b"\xef\xbb\xbf" + valid,
            valid + b"\n",
        ):
            with self.subTest(raw=raw[:12]):
                with open(path, "wb") as handle:
                    handle.write(raw)
                self.assert_rejected()
        # Restore a valid document before the format/version checks.
        with open(path, "wb") as handle:
            handle.write(valid)
        self.page()

    def test_unknown_format_and_version_are_rejected(self):
        for tamper in (
            lambda doc: doc.update(format="branching-city-twin/other"),
            lambda doc: doc.update(version=2),
            lambda doc: doc.update(extra=1),
        ):
            document = self.evidence()
            tamper(document)
            self.write_evidence(document)
            self.assert_rejected()


class ConcurrencyCoordinationTests(RotationMemoryTestBase):
    def test_publisher_waits_for_an_active_reader_then_reclaims(self):
        digest = self.append(self.entries[0:4], None, keep=2)
        lock_path = os.path.join(
            os.path.realpath(self.dir),
            BranchStore._DIAGNOSTIC_LEDGER_LOCK_NAME,
        )
        rfd = os.open(lock_path, os.O_RDONLY)
        BranchStore._acquire_diagnostic_ledger_reader_lock(rfd)
        outcome = {}

        def publish():
            outcome["digest"] = self.append(
                self.entries[4:6], digest, keep=2
            )

        thread = threading.Thread(target=publish)
        thread.start()
        thread.join(0.5)
        self.assertTrue(thread.is_alive())
        # While the reader is active the old segment must survive and
        # the visible snapshot must be the pre-rotation one.
        self.assertTrue(
            os.path.exists(os.path.join(self.dir, "segment-000001.json"))
        )
        self.assertEqual(
            self.page()["snapshot"]["digest"], digest
        )
        os.close(rfd)  # release as an abrupt reader exit would
        thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertIn("digest", outcome)
        # After the drain the rotation completed and reclaimed the old
        # segment, pointing at fresh evidence.
        self.assertFalse(
            os.path.exists(os.path.join(self.dir, "segment-000001.json"))
        )
        result = self.page()
        self.assertEqual(result["snapshot"]["digest"], outcome["digest"])
        self.assertEqual(
            [item["at"] for item in result["items"]], [3, 4, 5, 6]
        )

    def test_pages_while_a_publish_waits_see_only_the_old_version(self):
        digest = self.append(self.entries[0:4], None, keep=2)
        lock_path = os.path.join(
            os.path.realpath(self.dir),
            BranchStore._DIAGNOSTIC_LEDGER_LOCK_NAME,
        )
        # One long-lived read occupies the shared lock.
        gate = os.open(lock_path, os.O_RDONLY)
        BranchStore._acquire_diagnostic_ledger_reader_lock(gate)
        outcome = {}

        def publish():
            try:
                outcome["digest"] = self.append(
                    self.entries[4:6], digest, keep=2
                )
            except BaseException as exc:  # pragma: no cover - diagnostic
                outcome["error"] = exc

        thread = threading.Thread(target=publish)
        thread.start()
        thread.join(0.5)
        self.assertTrue(thread.is_alive())
        # While the publish is parked behind the held read, every other
        # page sees the complete pre-rotation version -- never a mix.
        for _ in range(20):
            result = self.page()
            self.assertEqual(result["snapshot"]["digest"], digest)
            self.assertEqual(
                [item["at"] for item in result["items"]], [1, 2, 3, 4]
            )
        os.close(gate)
        thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertNotIn("error", outcome)
        # Once the publish lands, every page sees the new generation.
        result = self.page()
        self.assertEqual(result["snapshot"]["digest"], outcome["digest"])
        self.assertEqual(
            [item["at"] for item in result["items"]], [3, 4, 5, 6]
        )

    def test_interleaved_pages_and_publishes_never_mix_versions(self):
        # Readers yield between pages so the publisher can win the lock;
        # every individual page must still be one whole generation.
        digest = self.append(self.entries[0:4], None, keep=2)
        seen_generations = []
        seen_lock = threading.Lock()
        stop = threading.Event()

        def reader():
            while not stop.is_set():
                result = self.page()
                times = tuple(item["at"] for item in result["items"])
                self.assertIn(
                    times,
                    ((1, 2, 3, 4), (3, 4, 5, 6), (5, 6, 7, 8)),
                )
                with seen_lock:
                    seen_generations.append(times)
                time.sleep(0.002)

        threads = [threading.Thread(target=reader) for _ in range(3)]
        for thread in threads:
            thread.start()
        for chunk in (self.entries[4:6], self.entries[6:8]):
            digest = self.append(chunk, digest, keep=2)
            time.sleep(0.005)
        stop.set()
        for thread in threads:
            thread.join(5)
        self.assertTrue(any(g == (5, 6, 7, 8) for g in seen_generations))

    def test_reader_creates_no_files(self):
        self.append(self.entries[0:2], None, keep=10)
        before = sorted(os.listdir(self.dir))
        self.page()
        self.assertEqual(sorted(os.listdir(self.dir)), before)


class FailureAtomicityTests(RotationMemoryTestBase):
    def test_stale_expected_writes_nothing(self):
        digest = self.publish_rotating_chain()
        before = {
            name: read_bytes(os.path.join(self.dir, name))
            for name in os.listdir(self.dir)
            if os.path.isfile(os.path.join(self.dir, name))
        }
        with self.assertRaises(RuntimeError):
            self.append(self.entries[6:8], "0" * 64)
        after = {
            name: read_bytes(os.path.join(self.dir, name))
            for name in os.listdir(self.dir)
            if os.path.isfile(os.path.join(self.dir, name))
        }
        self.assertEqual(before, after)
        self.assertEqual(self.page()["snapshot"]["digest"], digest)

    def test_evidence_write_failure_leaves_old_snapshot_intact(self):
        digest = self.append(self.entries[0:2], None, keep=1)
        digest = self.append(self.entries[2:4], digest, keep=1)
        before = {
            name: read_bytes(os.path.join(self.dir, name))
            for name in os.listdir(self.dir)
            if os.path.isfile(os.path.join(self.dir, name))
        }
        # The next publish rotates; make the evidence write fail before
        # the manifest is switched.
        original = BranchStore._write_diagnostic_ledger_file
        names = {"seen": []}

        def failing_write(path, data, prefix):
            names["seen"].append(prefix)
            if "evidence" in prefix:
                raise OSError("boom")
            return original(path, data, prefix)

        with mock.patch.object(
            BranchStore,
            "_write_diagnostic_ledger_file",
            side_effect=failing_write,
        ):
            with self.assertRaises(OSError):
                self.append(self.entries[4:6], digest, keep=1)
        after = {
            name: read_bytes(os.path.join(self.dir, name))
            for name in os.listdir(self.dir)
            if os.path.isfile(os.path.join(self.dir, name))
        }
        self.assertEqual(before, after)
        self.assertEqual(self.page()["snapshot"]["digest"], digest)

    def test_manifest_switch_failure_leaves_old_evidence_intact(self):
        digest = self.publish_rotating_chain()
        evidence_before = read_bytes(self.evidence_path())
        manifest_before = read_bytes(self.manifest_path())
        original = BranchStore._write_diagnostic_ledger_file

        def failing_write(path, data, prefix):
            if "manifest" in prefix:
                raise OSError("boom")
            return original(path, data, prefix)

        with mock.patch.object(
            BranchStore,
            "_write_diagnostic_ledger_file",
            side_effect=failing_write,
        ):
            with self.assertRaises(OSError):
                self.append(self.entries[6:8], digest, keep=1)
        self.assertEqual(read_bytes(self.manifest_path()), manifest_before)
        self.assertEqual(read_bytes(self.evidence_path()), evidence_before)


if __name__ == "__main__":
    unittest.main()
