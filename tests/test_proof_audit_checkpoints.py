"""Tests for resumable, incrementally continuing offline proof audits.

``BranchStore.create_proof_audit_checkpoint`` seals an offline audit of
a tuple of canonical transaction-mode diagnostic ledger proofs into a
cross-process checkpoint string, and
``BranchStore.resume_proof_audit`` continues it with a new batch. The
incremental audit is item-for-item identical to handing every proof to
``BranchStore.audit_diagnostic_ledger_proof_sequence`` in one call.

These tests cover:

* argument validation and exception types for both entry points,
  including the validate-inputs / authenticate-checkpoint /
  authenticate-new-proofs order and the absence of partial results;
* the initial empty-tuple checkpoint and first-batch identity
  settlement;
* byte-for-byte deterministic canonical JSON with no trailing newline
  and a SHA-256 checksum over every state field;
* equivalence of every batch split with the one-shot audit, including
  cross-batch rotations, for continuous and every failed classification;
* failed terminal sealing: ``RuntimeError`` on non-empty continuation
  and read-only sealed retrieval (same checkpoint bytes) on empty;
* strict checkpoint authentication: encoding, canonical form, duplicate
  keys, format, version, structure, checksum and internal
  self-consistency;
* identity-set disagreement on continuation;
* deep isolation of every returned level and no ledger or state access.
"""

import hashlib
import json
import os
import shutil
import tempfile
import unittest

from city_twin.branches import BranchStore

try:
    from test_diagnostic_ledger import make_chain, reseal, transaction
except ImportError:  # running as a package member from the repo root
    from tests.test_diagnostic_ledger import (
        make_chain,
        reseal,
        transaction,
    )


def canonical(document: dict) -> str:
    return BranchStore._canonical_json(document)


class CheckpointTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.chains = [make_chain(transaction(i)) for i in range(12)]
        self.entries = tuple(
            (time, chain)
            for time, chain in zip(range(1, 13), self.chains)
        )
        self.identities = tuple(transaction(i) for i in range(12))

    def ledger(self, name: str) -> str:
        directory = os.path.join(self.tmp.name, name)
        os.mkdir(directory)
        return directory

    def append(self, directory, entries, expected, keep=10, limit=2):
        return BranchStore.append_diagnostic_ledger(
            directory, entries, expected, limit, keep, None
        )

    def export(self, directory, identities=None):
        if identities is None:
            identities = self.identities
        return BranchStore.export_diagnostic_ledger_proof(
            directory, None, None, identities
        )

    def sequence(self, proofs):
        return BranchStore.audit_diagnostic_ledger_proof_sequence(proofs)

    def create(self, proofs):
        return BranchStore.create_proof_audit_checkpoint(proofs)

    def resume(self, checkpoint, proofs):
        return BranchStore.resume_proof_audit(checkpoint, proofs)

    def digest_of(self, directory) -> str:
        return BranchStore.page_diagnostic_ledger(
            directory, None, None, None, 100, None
        )["snapshot"]["digest"]

    @staticmethod
    def _directory_files(directory) -> dict:
        contents = {}
        for name in os.listdir(directory):
            with open(os.path.join(directory, name), "rb") as handle:
                contents[name] = handle.read()
        return contents

    def reseal_proof(self, document: dict) -> str:
        """Re-seal every layer of a (possibly tampered) proof so it
        still verifies on its own."""
        document["evidence"] = json.loads(reseal(document["evidence"]))
        evidence_digest = hashlib.sha256(
            canonical(document["evidence"]).encode("utf-8")
        ).hexdigest()
        document["manifest"]["evidence"] = {"digest": evidence_digest}
        document["manifest"] = json.loads(reseal(document["manifest"]))
        manifest_raw = canonical(document["manifest"]).encode("utf-8")
        document["snapshot"] = (
            BranchStore._diagnostic_ledger_proof_snapshot(
                manifest_raw,
                document["manifest"],
                document["manifest"]["segments"],
                evidence_digest,
            )
        )
        body = {
            key: value
            for key, value in document.items()
            if key != "checksum"
        }
        document["checksum"] = hashlib.sha256(
            canonical(body).encode("utf-8")
        ).hexdigest()
        text = canonical(document)
        BranchStore.verify_diagnostic_ledger_proof(text)
        return text

    @staticmethod
    def reseal_checkpoint(document: dict) -> str:
        """Re-serialize a (possibly tampered) checkpoint with a fresh
        checksum over its format, version and state."""
        body = {
            "format": document["format"],
            "version": document["version"],
            "state": document["state"],
        }
        document["checksum"] = hashlib.sha256(
            canonical(body).encode("utf-8")
        ).hexdigest()
        return canonical(document)

    def grow(self, keep=10):
        """Build four successive proofs p0..p3 of one appending ledger."""
        directory = self.ledger("ledger")
        digest = self.append(directory, self.entries[:4], None, keep=keep)
        p0 = self.export(directory)
        digest = self.append(
            directory, self.entries[4:6], digest, keep=keep
        )
        p1 = self.export(directory)
        digest = self.append(
            directory, self.entries[6:8], digest, keep=keep
        )
        p2 = self.export(directory)
        digest = self.append(
            directory, self.entries[8:10], digest, keep=keep
        )
        p3 = self.export(directory)
        return directory, (p0, p1, p2, p3)

    def fork_pair(self):
        base = self.ledger("base")
        digest = self.append(base, self.entries[:4], None)
        left = self.ledger("left")
        right = self.ledger("right")
        for target in (left, right):
            shutil.copytree(base, target, dirs_exist_ok=True)
        self.append(left, self.entries[4:6], digest)
        other = tuple(
            (time, make_chain(transaction(100 + time)))
            for time in (5, 6)
        )
        self.append(right, other, digest)
        return self.export(left), self.export(right)


class ArgumentValidationTests(CheckpointTestBase):
    def setUp(self) -> None:
        super().setUp()
        self.directory = self.ledger("ledger")
        self.append(self.directory, self.entries[:2], None)
        self.proof = self.export(self.directory)

    def test_create_requires_a_tuple(self) -> None:
        for bad in ([self.proof], {self.proof}, None, 1, object()):
            with self.subTest(bad=type(bad).__name__):
                with self.assertRaises(TypeError):
                    self.create(bad)

    def test_create_rejects_non_string_elements(self) -> None:
        for bad in (1, 1.5, b"x", ["x"], {"x": 1}, object(), None):
            with self.subTest(bad=type(bad).__name__):
                with self.assertRaises(TypeError):
                    self.create((self.proof, bad))

    def test_create_authenticates_every_proof(self) -> None:
        with self.assertRaises(ValueError):
            self.create(("",))
        with self.assertRaises(ValueError):
            self.create((self.proof, ""))
        with self.assertRaises(ValueError):
            self.create((self.proof, "not a proof"))
        pretty = json.dumps(json.loads(self.proof), indent=2)
        with self.assertRaises(ValueError):
            self.create((self.proof, pretty))

    def test_create_rejects_time_mode_and_identity_mismatch(self) -> None:
        timed = BranchStore.export_diagnostic_ledger_proof(
            self.directory, 1, 2, None
        )
        with self.assertRaises(ValueError):
            self.create((timed,))
        subset = self.export(self.directory, (transaction(0),))
        with self.assertRaises(ValueError):
            self.create((self.proof, subset))

    def test_resume_checkpoint_type_checked_first(self) -> None:
        for bad in (None, 1, b"x", [], object()):
            with self.subTest(bad=type(bad).__name__):
                with self.assertRaises(TypeError):
                    self.resume(bad, ())
                with self.assertRaises(TypeError):
                    self.resume(bad, [self.proof])

    def test_resume_proofs_container_must_be_tuple(self) -> None:
        checkpoint = self.create(())
        for bad in ([], None, 1, {self.proof}):
            with self.subTest(bad=type(bad).__name__):
                with self.assertRaises(TypeError):
                    self.resume(checkpoint, bad)

    def test_resume_elements_must_be_str(self) -> None:
        checkpoint = self.create(())
        for bad in (1, b"x", [self.proof], None):
            with self.subTest(bad=type(bad).__name__):
                with self.assertRaises(TypeError):
                    self.resume(checkpoint, (bad,))

    def test_resume_ordinary_inputs_checked_before_checkpoint(self) -> None:
        # A non-tuple container and non-str element surface as TypeError
        # before the (empty) checkpoint is authenticated.
        with self.assertRaises(TypeError):
            self.resume("", [])
        with self.assertRaises(TypeError):
            self.resume("", (1,))

    def test_resume_empty_checkpoint_is_value_error(self) -> None:
        with self.assertRaises(ValueError):
            self.resume("", ())

    def test_resume_invalid_new_proof_leaves_no_partial_result(self) -> None:
        checkpoint = self.create((self.proof,))
        with self.assertRaises(ValueError):
            self.resume(checkpoint, ("",))
        with self.assertRaises(ValueError):
            self.resume(checkpoint, ("not a proof",))
        # The earlier checkpoint is untouched and still resumable.
        result = self.resume(checkpoint, ())
        self.assertEqual(result["checkpoint"], checkpoint)
        self.assertEqual(result["audit"]["status"], "continuous")

    def test_resume_identity_set_disagreement(self) -> None:
        checkpoint = self.create((self.proof,))
        other_directory = self.ledger("other")
        self.append(other_directory, self.entries[:2], None)
        reordered = self.export(
            other_directory, (transaction(1), transaction(0))
        )
        with self.assertRaises(ValueError):
            self.resume(checkpoint, (reordered,))
        # Old checkpoint remains valid.
        result = self.resume(checkpoint, ())
        self.assertEqual(result["checkpoint"], checkpoint)


class InitialCheckpointTests(CheckpointTestBase):
    def test_empty_tuple_creates_initial_checkpoint(self) -> None:
        checkpoint = self.create(())
        self.assertIsInstance(checkpoint, str)
        self.assertTrue(checkpoint)
        self.assertFalse(checkpoint.endswith("\n"))
        self.assertNotIn("\n", checkpoint)
        self.assertNotIn(", ", checkpoint)
        self.assertNotIn(": ", checkpoint)
        document = json.loads(checkpoint)
        self.assertEqual(
            canonical(document),
            checkpoint,
        )

    def test_initial_checkpoint_is_deterministic(self) -> None:
        self.assertEqual(self.create(()), self.create(()))

    def test_empty_resume_returns_equal_empty_audit_same_bytes(self) -> None:
        checkpoint = self.create(())
        result = self.resume(checkpoint, ())
        self.assertEqual(list(result), ["audit", "checkpoint"])
        self.assertEqual(result["checkpoint"], checkpoint)
        audit = result["audit"]
        self.assertEqual(
            list(audit), ["status", "common", "hops", "failure"]
        )
        self.assertEqual(audit, self.sequence(()))

    def test_first_batch_after_initial_checkpoint(self) -> None:
        _directory, proofs = self.grow()
        checkpoint = self.create(())
        result = self.resume(checkpoint, (proofs[0], proofs[1]))
        self.assertEqual(
            result["audit"], self.sequence((proofs[0], proofs[1]))
        )
        result = self.resume(
            result["checkpoint"], (proofs[2], proofs[3])
        )
        self.assertEqual(result["audit"], self.sequence(proofs))

    def test_failed_first_batch_seals(self) -> None:
        before, after = self.fork_pair()
        checkpoint = self.create(())
        result = self.resume(checkpoint, (before, after))
        self.assertEqual(
            result["audit"], self.sequence((before, after))
        )
        self.assertEqual(result["audit"]["status"], "failed")
        with self.assertRaises(RuntimeError):
            self.resume(result["checkpoint"], (before,))


class IncrementalEquivalenceTests(CheckpointTestBase):
    def _assert_split_matches(self, proofs, split_points) -> None:
        expected = self.sequence(proofs)
        checkpoint = self.create(())
        start = 0
        produced = None
        for end in split_points:
            produced = self.resume(
                checkpoint, proofs[start:end]
            )
            checkpoint = produced["checkpoint"]
            start = end
        self.assertEqual(start, len(proofs))
        self.assertEqual(produced["audit"], expected)

    def test_every_split_of_a_continuous_sequence(self) -> None:
        _directory, proofs = self.grow(keep=1)
        for split in (1, 2, 3):
            with self.subTest(split=split):
                self._assert_split_matches(proofs, (split, 4))
        self._assert_split_matches(proofs, (1, 2, 3, 4))

    def test_create_then_resume_matches_create_over_all(self) -> None:
        _directory, proofs = self.grow(keep=1)
        checkpoint_a = self.create(proofs)
        checkpoint_b = self.create((proofs[0],))
        checkpoint_b = self.resume(
            checkpoint_b, tuple(proofs[1:])
        )["checkpoint"]
        self.assertEqual(checkpoint_a, checkpoint_b)

    def test_cross_batch_hop_indices_are_continuous(self) -> None:
        _directory, proofs = self.grow()
        checkpoint = self.create((proofs[0],))
        result = self.resume(checkpoint, (proofs[1],))
        self.assertEqual(
            [hop["index"] for hop in result["audit"]["hops"]], [1]
        )
        result = self.resume(
            result["checkpoint"], (proofs[2], proofs[3])
        )
        self.assertEqual(
            [hop["index"] for hop in result["audit"]["hops"]],
            [1, 2, 3],
        )
        self.assertEqual(result["audit"]["status"], "continuous")

    def test_empty_resume_adds_no_hop_and_keeps_bytes(self) -> None:
        _directory, proofs = self.grow()
        checkpoint = self.create((proofs[0], proofs[1]))
        before = self.resume(checkpoint, ())
        self.assertEqual(before["checkpoint"], checkpoint)
        self.assertEqual(
            before["audit"], self.sequence((proofs[0], proofs[1]))
        )
        after = self.resume(checkpoint, ())
        self.assertEqual(after["audit"], before["audit"])
        self.assertEqual(after["checkpoint"], checkpoint)

    def test_continuous_audit_covers_rotation_across_batches(self) -> None:
        _directory, proofs = self.grow(keep=1)
        checkpoint = self.create((proofs[0],))
        result = self.resume(checkpoint, (proofs[1],))
        hop = result["audit"]["hops"][0]
        expected_hop = self.sequence((proofs[0], proofs[1]))["hops"][0]
        self.assertEqual(hop["relation"], "continued")
        self.assertTrue(hop["rotated"])
        self.assertEqual(hop["rotated"], expected_hop["rotated"])

    def test_single_proof_checkpoint(self) -> None:
        directory = self.ledger("one")
        self.append(directory, self.entries[:4], None)
        proof = self.export(directory)
        checkpoint = self.create((proof,))
        result = self.resume(checkpoint, ())
        self.assertEqual(result["audit"], self.sequence((proof,)))
        self.assertEqual(result["audit"]["common"]["index"], 2)


class FailedContinuationTests(CheckpointTestBase):
    def test_fork_across_batch_boundary(self) -> None:
        _directory, proofs = self.grow()
        before, after = self.fork_pair()
        all_proofs = (proofs[0], before, after, proofs[0])
        checkpoint = self.create((proofs[0],))
        result = self.resume(checkpoint, (before, after, proofs[0]))
        self.assertEqual(result["audit"], self.sequence(all_proofs))
        self.assertEqual(
            result["audit"]["failure"],
            {"index": 2, "classification": "forked"},
        )

    def test_failure_seals_and_empty_continuation_reads_it(self) -> None:
        before, after = self.fork_pair()
        checkpoint = self.create((before,))
        failed = self.resume(checkpoint, (after,))
        self.assertEqual(failed["audit"]["status"], "failed")
        with self.assertRaises(RuntimeError):
            self.resume(failed["checkpoint"], (before,))
        sealed = self.resume(failed["checkpoint"], ())
        self.assertEqual(sealed["audit"], failed["audit"])
        self.assertEqual(sealed["checkpoint"], failed["checkpoint"])

    def test_create_failed_checkpoint_is_sealed(self) -> None:
        before, after = self.fork_pair()
        checkpoint = self.create((before, after))
        document = json.loads(checkpoint)
        self.assertEqual(document["state"]["status"], "failed")
        with self.assertRaises(RuntimeError):
            self.resume(checkpoint, (before,))
        self.assertEqual(
            self.resume(checkpoint, ())["audit"],
            self.sequence((before, after)),
        )

    def test_rollback_across_batches(self) -> None:
        _directory, proofs = self.grow()
        checkpoint = self.create((proofs[2],))
        result = self.resume(checkpoint, (proofs[1],))
        self.assertEqual(
            result["audit"], self.sequence((proofs[2], proofs[1]))
        )
        self.assertEqual(
            result["audit"]["failure"]["classification"], "rollback"
        )

    def test_splice_across_batches(self) -> None:
        first_dir = self.ledger("first")
        self.append(first_dir, self.entries[:2], None)
        second_dir = self.ledger("second")
        other = tuple(
            (time, make_chain(transaction(50 + time)))
            for time in (1, 2)
        )
        self.append(second_dir, other, None)
        first_proof = self.export(first_dir)
        second_proof = self.export(second_dir)
        checkpoint = self.create((first_proof,))
        result = self.resume(checkpoint, (second_proof,))
        self.assertEqual(
            result["audit"],
            self.sequence((first_proof, second_proof)),
        )
        self.assertEqual(
            result["audit"]["failure"]["classification"], "spliced"
        )

    def test_missing_transition_across_batches(self) -> None:
        directory = self.ledger("ledger")
        digest = self.append(directory, self.entries[:4], None, keep=2)
        before = self.export(directory)
        digest = self.append(
            directory, self.entries[4:8], digest, keep=2
        )
        self.append(directory, self.entries[8:10], digest, keep=2)
        after = self.export(directory)
        checkpoint = self.create((before,))
        result = self.resume(checkpoint, (after,))
        self.assertEqual(
            result["audit"], self.sequence((before, after))
        )
        self.assertEqual(
            result["audit"]["failure"]["classification"],
            "missing_transition",
        )

    def test_rewritten_across_batches(self) -> None:
        directory = self.ledger("ledger")
        digest = self.append(directory, self.entries[:4], None, keep=1)
        before = self.export(directory)
        self.append(directory, self.entries[4:6], digest, keep=1)
        document = json.loads(self.export(directory))
        forged = "f" * 64
        document["manifest"]["dropped"]["digest"] = forged
        document["evidence"]["digest"] = forged
        segment = document["segments"][0]["document"]
        segment["prev"] = forged
        document["segments"][0]["document"] = json.loads(
            reseal(segment)
        )
        segment_digest = hashlib.sha256(
            canonical(
                document["segments"][0]["document"]
            ).encode("utf-8")
        ).hexdigest()
        document["manifest"]["segments"][0]["digest"] = segment_digest
        tampered = self.reseal_proof(document)
        checkpoint = self.create((before,))
        result = self.resume(checkpoint, (tampered,))
        self.assertEqual(
            result["audit"], self.sequence((before, tampered))
        )
        self.assertEqual(
            result["audit"]["failure"]["classification"], "rewritten"
        )


class CheckpointAuthenticationTests(CheckpointTestBase):
    def setUp(self) -> None:
        super().setUp()
        self.directory = self.ledger("ledger")
        self.append(self.directory, self.entries[:4], None)
        self.proof = self.export(self.directory)
        self.checkpoint = self.create((self.proof,))

    def test_non_canonical_json_rejected(self) -> None:
        pretty = json.dumps(json.loads(self.checkpoint), indent=2)
        with self.assertRaises(ValueError):
            self.resume(pretty, ())

    def test_duplicate_keys_rejected(self) -> None:
        tampered = self.checkpoint.replace(
            '"failure":null',
            '"failure":null,"zzz":1,"zzz":2',
        )
        self.assertIn("zzz", tampered)
        with self.assertRaises(ValueError):
            self.resume(tampered, ())

    def test_bad_format_version_and_keys_rejected(self) -> None:
        document = json.loads(self.checkpoint)
        bad_format = dict(document)
        bad_format["format"] = "other"
        with self.assertRaises(ValueError):
            self.resume(canonical(bad_format), ())
        bad_version = dict(document)
        bad_version["version"] = 2
        with self.assertRaises(ValueError):
            self.resume(canonical(bad_version), ())
        bad_keys = dict(document)
        del bad_keys["state"]
        with self.assertRaises(ValueError):
            self.resume(canonical(bad_keys), ())

    def test_checksum_tampering_rejected(self) -> None:
        flipped = (
            self.checkpoint[:-1]
            + ("0" if self.checkpoint[-1] != "0" else "1")
        )
        with self.assertRaises(ValueError):
            self.resume(flipped, ())
        # Tamper a state byte without updating the checksum.
        tampered = self.checkpoint.replace('"n":1', '"n":2', 1)
        with self.assertRaises(ValueError):
            self.resume(tampered, ())

    def test_inconsistent_state_rejected_even_resealed(self) -> None:
        document = json.loads(self.checkpoint)
        document["state"]["n"] = 2
        with self.assertRaises(ValueError):
            self.resume(self.reseal_checkpoint(document), ())

    def test_inconsistent_status_rejected_even_resealed(self) -> None:
        document = json.loads(self.checkpoint)
        document["state"]["status"] = "bogus"
        with self.assertRaises(ValueError):
            self.resume(self.reseal_checkpoint(document), ())

    def test_failed_state_mismatched_failure_rejected(self) -> None:
        before, after = self.fork_pair()
        checkpoint = self.create((before, after))
        document = json.loads(checkpoint)
        document["state"]["failure"]["classification"] = "rollback"
        with self.assertRaises(ValueError):
            self.resume(self.reseal_checkpoint(document), ())

    def test_tampered_last_view_digest_rejected(self) -> None:
        document = json.loads(self.checkpoint)
        segment = document["state"]["last"]["segments"][0]
        segment["digest"] = "f" * 64
        with self.assertRaises(ValueError):
            self.resume(self.reseal_checkpoint(document), ())

    def test_tampered_snapshot_rejected_when_resealed(self) -> None:
        for field, value in (
            ("bytes", 999999),
            ("entries", 99),
            ("segments", 9),
        ):
            with self.subTest(field=field):
                document = json.loads(self.checkpoint)
                document["state"]["last"]["snapshot"][field] = value
                with self.assertRaises(ValueError):
                    self.resume(
                        self.reseal_checkpoint(document), ()
                    )

    def test_tampered_snapshot_digest_rejected_when_resealed(self) -> None:
        document = json.loads(self.checkpoint)
        document["state"]["last"]["snapshot"]["digest"] = "f" * 64
        with self.assertRaises(ValueError):
            self.resume(self.reseal_checkpoint(document), ())

    def test_tampered_chain_record_rejected_when_resealed(self) -> None:
        document = json.loads(self.checkpoint)
        entry = document["state"]["last"]["segments"][0]["body"][0]
        entry["records"] = entry["records"][:-1]
        entry["summary"]["count"] -= 1
        with self.assertRaises(ValueError):
            self.resume(self.reseal_checkpoint(document), ())

    def test_foreign_document_rejected(self) -> None:
        foreign = canonical({"hello": "world"})
        with self.assertRaises(ValueError):
            self.resume(foreign, ())

    def test_checkpoint_checksum_covers_every_field(self) -> None:
        document = json.loads(self.checkpoint)
        # Mutating any state field and resealing authenticates the
        # checksum layer, so the inner structural checks are what must
        # reject the change rather than the parse failing silently.
        document["state"]["b0"] = {"index": 1, "digest": "a" * 64,
                                   "entries": 4}
        with self.assertRaises(ValueError):
            self.resume(self.reseal_checkpoint(document), ())


class IsolationAndPurityTests(CheckpointTestBase):
    def test_returned_levels_are_isolated(self) -> None:
        _directory, proofs = self.grow(keep=1)
        checkpoint = self.create((proofs[0],))
        result = self.resume(checkpoint, (proofs[1], proofs[2]))
        expected = self.sequence((proofs[0], proofs[1], proofs[2]))
        self.assertEqual(result["audit"], expected)
        audit = result["audit"]
        audit["common"]["digest"] = "mutated"
        audit["hops"][0]["common"]["digest"] = "mutated"
        audit["hops"][0]["after_only"][0]["digest"] = "mutated"
        audit["hops"][0]["after_only"][0]["entries"][0]["summary"][
            "transaction"
        ] = "mutated"
        list(audit["hops"][0]["rotated"]).append("mutated")
        again = self.resume(checkpoint, (proofs[1], proofs[2]))
        self.assertNotEqual(
            again["audit"]["common"]["digest"], "mutated"
        )
        self.assertNotIn(
            "mutated",
            again["audit"]["hops"][0]["after_only"][0]["digest"],
        )
        self.assertEqual(again["audit"], expected)
        for hop in again["audit"]["hops"]:
            self.assertIsInstance(hop["after_only"], tuple)
            for segment_item in hop["after_only"]:
                self.assertIsInstance(segment_item["entries"], tuple)
                for entry in segment_item["entries"]:
                    self.assertIsInstance(entry["records"], tuple)

    def test_failed_levels_are_isolated(self) -> None:
        before, after = self.fork_pair()
        checkpoint = self.create((before,))
        first = self.resume(checkpoint, (after,))
        first["audit"]["hops"][0]["before_only"][0]["digest"] = "x"
        second = self.resume(checkpoint, (after,))
        self.assertEqual(second["audit"], self.sequence((before, after)))
        self.assertNotEqual(
            second["audit"]["hops"][0]["before_only"][0]["digest"], "x"
        )

    def test_chained_resumes_share_no_state(self) -> None:
        _directory, proofs = self.grow()
        first = self.resume(
            self.create((proofs[0],)), (proofs[1],)
        )
        second = self.resume(
            first["checkpoint"], (proofs[2],)
        )
        # Mutating the first result does not affect the second chain.
        first["audit"]["hops"][0]["relation"] = "forked"
        third = self.resume(
            first["checkpoint"], (proofs[2],)
        )
        self.assertEqual(third["audit"], second["audit"])

    def test_no_ledger_files_are_read_or_written(self) -> None:
        directory = self.ledger("ledger")
        digest = self.append(directory, self.entries[:4], None)
        first = self.export(directory)
        digest = self.append(
            directory, self.entries[4:6], digest, keep=1
        )
        second = self.export(directory)
        files = self._directory_files(directory)
        checkpoint = self.create((first,))
        self.resume(checkpoint, (second,))
        self.create((first, second))
        self.assertEqual(files, self._directory_files(directory))
        self.assertEqual(self.digest_of(directory), digest)

    def test_create_and_resume_do_not_mutate_proofs(self) -> None:
        _directory, proofs = self.grow()
        saved = tuple(proofs)
        checkpoint = self.create(proofs)
        self.resume(checkpoint, ())
        self.assertEqual(proofs, saved)


if __name__ == "__main__":
    unittest.main()
