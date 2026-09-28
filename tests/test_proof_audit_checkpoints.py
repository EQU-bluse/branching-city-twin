"""Tests for incremental checkpoints of the offline diagnostic ledger
proof-sequence audit.

``BranchStore.create_proof_audit_checkpoint`` bootstraps or seals an
audit checkpoint from a tuple of canonical transaction-mode proof
strings -- the empty tuple is allowed and defers the ordered
transaction identity set to the first resumed batch -- and
``BranchStore.resume_proof_audit`` continues a sealed audit with a new
tuple of proofs, entirely offline. Both reuse the authentication,
relation and exception semantics of
:meth:`BranchStore.audit_diagnostic_ledger_proof_sequence`, so a
sequence audited in arbitrary batches is item-for-item identical to the
one-shot audit.

These tests cover:

* argument validation and its order: :class:`TypeError` for non-tuple
  containers and non-str elements, :class:`ValueError` for empty,
  non-canonical, time-range or non-authenticating proofs and mismatched
  ordered identity sets;
* the canonical checkpoint document: compact UTF-8 JSON without a BOM
  or trailing newline, fixed key order, byte-for-byte determinism for
  the same sequence and a SHA-256 checksum covering every other field;
* checkpoint authentication: an empty string, non-canonical JSON,
  duplicate keys, a bad version or structure, a checksum mismatch or an
  internal inconsistency between the boundary, hops, terminal state and
  the re-authenticated last-proof view all raise :class:`ValueError`;
* equivalence: every batching of a sequence -- including the empty
  bootstrap, empty (read-only) continuations, cross-batch rotations and
  each failing classification -- reproduces the one-shot audit, with
  hop indices growing continuously;
* terminal sealing: a ``"failed"`` checkpoint answers read-only empty
  continuations forever but raises :class:`RuntimeError` for a
  non-empty one; proofs past the terminal hop never enter the
  checkpoint;
* isolation and purity: returned levels never share mutable state with
  later calls and no proof, checkpoint, ledger file, event graph,
  business audit or idempotency state is modified.
"""

import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
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

    def digest_of(self, directory) -> str:
        return BranchStore.page_diagnostic_ledger(
            directory, None, None, None, 100, None
        )["snapshot"]["digest"]

    def grow(self, name, chunks, keep=10):
        """Build a ledger and return the exported proof after each
        chunk in ``chunks`` (an iterable of entry counts)."""
        directory = self.ledger(name)
        proofs = []
        cursor = 0
        expected = None
        for size in chunks:
            expected = self.append(
                directory,
                self.entries[cursor : cursor + size],
                expected,
                keep=keep,
            )
            cursor += size
            proofs.append(self.export(directory))
        return directory, proofs

    @staticmethod
    def reseal_checkpoint(document: dict) -> str:
        """Re-serialize a (tampered) checkpoint with a fresh checksum
        over every other field."""
        body = {
            key: value
            for key, value in document.items()
            if key != "checksum"
        }
        document["checksum"] = hashlib.sha256(
            canonical(body).encode("utf-8")
        ).hexdigest()
        return canonical(document)

    def reseal_proof(self, document: dict) -> str:
        """Re-seal every layer of a (possibly tampered) proof so it
        still verifies on its own."""
        document["evidence"] = json.loads(reseal(document["evidence"]))
        evidence_digest = hashlib.sha256(
            canonical(document["evidence"]).encode("utf-8")
        ).hexdigest()
        document["manifest"]["evidence"] = {
            "digest": evidence_digest
        }
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
    def _directory_files(directory) -> dict:
        contents = {}
        for name in os.listdir(directory):
            with open(os.path.join(directory, name), "rb") as handle:
                contents[name] = handle.read()
        return contents

    def assert_batches_equal_one_shot(
        self, proofs, batches, seed=None
    ) -> str:
        """Resume ``batches`` in order from the empty bootstrap (or a
        supplied seed checkpoint string), assert the sealed audit
        equals the one-shot sequence audit, and return the final
        checkpoint string."""
        expected = (
            BranchStore.audit_diagnostic_ledger_proof_sequence(proofs)
        )
        if seed is None:
            checkpoint = (
                BranchStore.create_proof_audit_checkpoint(())
            )
        else:
            checkpoint = seed
        result = None
        for batch in batches:
            result = BranchStore.resume_proof_audit(checkpoint, batch)
            checkpoint = result["checkpoint"]
        assert result is not None
        self.assertEqual(result["audit"], expected)
        return checkpoint


class CreateValidationTests(CheckpointTestBase):
    def setUp(self) -> None:
        super().setUp()
        self.directory = self.ledger("ledger")
        self.append(self.directory, self.entries[:2], None)
        self.proof = self.export(self.directory)

    def test_container_must_be_a_tuple(self) -> None:
        for bad in ([self.proof], {self.proof}, None, 1, object()):
            with self.subTest(bad=type(bad).__name__):
                with self.assertRaises(TypeError):
                    BranchStore.create_proof_audit_checkpoint(bad)

    def test_non_string_elements_raise_type_error(self) -> None:
        for bad in (1, 1.5, b"x", ["x"], {"x": 1}, object(), None):
            with self.subTest(bad=type(bad).__name__):
                with self.assertRaises(TypeError):
                    BranchStore.create_proof_audit_checkpoint(
                        (self.proof, bad)
                    )

    def test_invalid_proofs_raise_value_error(self) -> None:
        with self.assertRaises(ValueError):
            BranchStore.create_proof_audit_checkpoint(("",))
        with self.assertRaises(ValueError):
            BranchStore.create_proof_audit_checkpoint(
                (self.proof, "")
            )
        with self.assertRaises(ValueError):
            BranchStore.create_proof_audit_checkpoint(
                (self.proof, "not a proof")
            )
        pretty = json.dumps(json.loads(self.proof), indent=2)
        with self.assertRaises(ValueError):
            BranchStore.create_proof_audit_checkpoint(
                (self.proof, pretty)
            )

    def test_time_range_proofs_are_rejected(self) -> None:
        timed = BranchStore.export_diagnostic_ledger_proof(
            self.directory, 1, 2, None
        )
        with self.assertRaises(ValueError):
            BranchStore.create_proof_audit_checkpoint(
                (timed, self.proof)
            )

    def test_identity_sets_must_match_in_order(self) -> None:
        subset = self.export(
            self.directory, (transaction(0),)
        )
        reordered = self.export(
            self.directory, (transaction(1), transaction(0))
        )
        with self.assertRaises(ValueError):
            BranchStore.create_proof_audit_checkpoint(
                (subset, self.proof)
            )
        with self.assertRaises(ValueError):
            BranchStore.create_proof_audit_checkpoint(
                (self.proof, reordered)
            )

    def test_all_proofs_authenticate_before_any_analysis(self) -> None:
        # A forked pair followed by a non-authenticating string must
        # surface the authentication fault, never a sealed failure.
        base = self.ledger("base")
        self.append(base, self.entries[:4], None)
        left = self.ledger("left")
        right = self.ledger("right")
        for target in (left, right):
            shutil.copytree(base, target, dirs_exist_ok=True)
        base_digest = self.digest_of(base)
        self.append(left, self.entries[4:6], base_digest)
        other = tuple(
            (time, make_chain(transaction(100 + time)))
            for time in (5, 6)
        )
        self.append(right, other, base_digest)
        with self.assertRaises(ValueError):
            BranchStore.create_proof_audit_checkpoint(
                (
                    self.export(left),
                    self.export(right),
                    "not a proof",
                )
            )

    def test_failure_does_not_seal_proofs_past_the_terminal_hop(
        self,
    ) -> None:
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
        before, after = self.export(left), self.export(right)
        sealed_at_fork = (
            BranchStore.create_proof_audit_checkpoint(
                (before, after)
            )
        )
        # The third proof is pre-authenticated but the fork terminates
        # inference; it must not enter the sealed state.
        sealed_with_extra = (
            BranchStore.create_proof_audit_checkpoint(
                (before, after, before)
            )
        )
        self.assertEqual(sealed_at_fork, sealed_with_extra)


class CheckpointEncodingTests(CheckpointTestBase):
    DOCUMENT_KEYS = (
        "format",
        "version",
        "count",
        "identities",
        "common",
        "status",
        "failure",
        "hops",
        "last",
        "checksum",
    )

    def test_empty_tuple_bootstraps_an_empty_checkpoint(self) -> None:
        checkpoint = (
            BranchStore.create_proof_audit_checkpoint(())
        )
        self.assertIsInstance(checkpoint, str)
        self.assertTrue(checkpoint)
        self.assertFalse(checkpoint.endswith("\n"))
        self.assertFalse(checkpoint.endswith("\r"))
        self.assertFalse(checkpoint.startswith("\ufeff"))
        document = json.loads(checkpoint)
        self.assertEqual(list(document), list(self.DOCUMENT_KEYS))
        self.assertEqual(document["version"], 1)
        self.assertEqual(document["count"], 0)
        self.assertEqual(document["identities"], [])
        self.assertIsNone(document["common"])
        self.assertEqual(document["status"], "empty")
        self.assertIsNone(document["failure"])
        self.assertEqual(document["hops"], [])
        self.assertIsNone(document["last"])
        # Deterministic across calls.
        self.assertEqual(
            checkpoint,
            BranchStore.create_proof_audit_checkpoint(()),
        )

    def test_non_empty_continuous_document_shape(self) -> None:
        _directory, (proof,) = self.grow("ledger", (4,))
        checkpoint = (
            BranchStore.create_proof_audit_checkpoint((proof,))
        )
        document = json.loads(checkpoint)
        self.assertEqual(list(document), list(self.DOCUMENT_KEYS))
        self.assertEqual(document["count"], 1)
        self.assertEqual(
            document["identities"], list(self.identities)
        )
        self.assertEqual(document["status"], "continuous")
        self.assertIsNone(document["failure"])
        self.assertEqual(document["hops"], [])
        self.assertEqual(
            set(document["common"]), {"index", "digest", "entries"}
        )
        self.assertEqual(document["common"]["index"], 2)
        self.assertEqual(document["common"]["entries"], 4)
        self.assertEqual(
            set(document["last"]),
            {"snapshot", "filter", "view"},
        )
        self.assertEqual(
            document["last"]["filter"]["transactions"],
            list(self.identities),
        )

    def test_same_sequence_is_byte_for_byte_identical(self) -> None:
        _directory, (first, second) = self.grow(
            "ledger", (4, 2)
        )
        proofs = (first, second)
        encoded_a = (
            BranchStore.create_proof_audit_checkpoint(proofs)
        )
        encoded_b = (
            BranchStore.create_proof_audit_checkpoint(tuple(proofs))
        )
        self.assertEqual(encoded_a, encoded_b)
        # Resuming into the same cumulative state produces the exact
        # bytes of sealing that state in one call.
        seeded = (
            BranchStore.create_proof_audit_checkpoint((first,))
        )
        resumed = BranchStore.resume_proof_audit(
            seeded, (second,)
        )
        self.assertEqual(resumed["checkpoint"], encoded_a)

    def test_checksum_covers_every_other_field(self) -> None:
        _directory, (first, second) = self.grow(
            "ledger", (4, 2)
        )
        checkpoint = (
            BranchStore.create_proof_audit_checkpoint((first, second))
        )
        document = json.loads(checkpoint)
        body = {
            key: value
            for key, value in document.items()
            if key != "checksum"
        }
        expected = hashlib.sha256(
            canonical(body).encode("utf-8")
        ).hexdigest()
        self.assertEqual(document["checksum"], expected)

    def test_failed_document_shape(self) -> None:
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
        checkpoint = (
            BranchStore.create_proof_audit_checkpoint(
                (self.export(left), self.export(right))
            )
        )
        document = json.loads(checkpoint)
        self.assertEqual(document["status"], "failed")
        self.assertEqual(document["count"], 2)
        self.assertEqual(len(document["hops"]), 1)
        self.assertEqual(
            document["failure"],
            {"index": 1, "classification": "forked"},
        )
        self.assertEqual(
            document["hops"][0]["reason"], "forked"
        )
        # The stored last view is the right-hand (forked) proof; with
        # no rotation it retains every segment of the 6-entry ledger.
        view = document["last"]["view"]
        self.assertEqual(
            [meta["index"] for meta in view["segments"]], [1, 2, 3]
        )
        # Only the fork's own increment is recorded as after_only.
        self.assertEqual(
            [item["index"] for item in document["hops"][0]["after_only"]],
            [3],
        )


class ResumeValidationTests(CheckpointTestBase):
    def setUp(self) -> None:
        super().setUp()
        self.directory = self.ledger("ledger")
        self.append(self.directory, self.entries[:2], None)
        self.proof = self.export(self.directory)
        self.checkpoint = (
            BranchStore.create_proof_audit_checkpoint((self.proof,))
        )

    def test_checkpoint_must_be_str(self) -> None:
        for bad in (1, 1.5, b"x", ["x"], {"x": 1}, None, object()):
            with self.subTest(bad=type(bad).__name__):
                with self.assertRaises(TypeError):
                    BranchStore.resume_proof_audit(bad, ())

    def test_proofs_must_be_a_tuple(self) -> None:
        for bad in ([self.proof], {self.proof}, None, 1, object()):
            with self.subTest(bad=type(bad).__name__):
                with self.assertRaises(TypeError):
                    BranchStore.resume_proof_audit(
                        self.checkpoint, bad
                    )

    def test_proof_elements_must_be_str(self) -> None:
        for bad in (1, 1.5, b"x", ["x"], {"x": 1}, None, object()):
            with self.subTest(bad=type(bad).__name__):
                with self.assertRaises(TypeError):
                    BranchStore.resume_proof_audit(
                        self.checkpoint, (bad,)
                    )

    def test_ordinary_inputs_are_checked_before_the_checkpoint(
        self,
    ) -> None:
        # An element type fault surfaces before an (also invalid)
        # checkpoint is even parsed.
        with self.assertRaises(TypeError):
            BranchStore.resume_proof_audit("", (1,))

    def test_checkpoint_is_authenticated_before_new_proofs(self) -> None:
        with self.assertRaises(ValueError):
            BranchStore.resume_proof_audit("", ())
        with self.assertRaises(ValueError):
            BranchStore.resume_proof_audit("not json", ())

    def test_non_canonical_duplicate_and_bom_checkpoints_rejected(
        self,
    ) -> None:
        pretty = json.dumps(
            json.loads(self.checkpoint), indent=2
        )
        with self.assertRaises(ValueError):
            BranchStore.resume_proof_audit(pretty, ())
        with self.assertRaises(ValueError):
            BranchStore.resume_proof_audit(
                self.checkpoint + "\n", ()
            )
        with self.assertRaises(ValueError):
            BranchStore.resume_proof_audit(
                "\ufeff" + self.checkpoint, ()
            )
        duplicate = self.checkpoint.replace(
            '"version":1,', '"version":1,"version":1,', 1
        )
        with self.assertRaises(ValueError):
            BranchStore.resume_proof_audit(duplicate, ())
        not_json = self.checkpoint.replace(
            '"count":1,', '"count":NaN,', 1
        )
        with self.assertRaises(ValueError):
            BranchStore.resume_proof_audit(not_json, ())

    def test_bad_structure_version_and_checksum_rejected(self) -> None:
        def resealed(document):
            return self.reseal_checkpoint(document)

        document = json.loads(self.checkpoint)
        wrong_format = dict(document)
        wrong_format["format"] = "other"
        with self.assertRaises(ValueError):
            BranchStore.resume_proof_audit(
                resealed(wrong_format), ()
            )
        wrong_version = dict(document)
        wrong_version["version"] = 2
        with self.assertRaises(ValueError):
            BranchStore.resume_proof_audit(
                resealed(wrong_version), ()
            )
        extra = dict(document)
        extra["extra"] = 1
        with self.assertRaises(ValueError):
            BranchStore.resume_proof_audit(resealed(extra), ())
        missing = {
            key: value
            for key, value in document.items()
            if key != "failure"
        }
        with self.assertRaises(ValueError):
            BranchStore.resume_proof_audit(resealed(missing), ())
        # A raw bit flip with no re-seal fails the checksum.
        tampered = self.checkpoint.replace(
            '"count":1,', '"count":2,', 1
        )
        with self.assertRaises(ValueError):
            BranchStore.resume_proof_audit(tampered, ())
        # A wrong checksum string is itself malformed.
        bad_checksum = dict(document)
        bad_checksum["checksum"] = "zz"
        with self.assertRaises(ValueError):
            BranchStore.resume_proof_audit(
                canonical(bad_checksum), ()
            )

    def test_invalid_new_proofs_raise_value_error(self) -> None:
        with self.assertRaises(ValueError):
            BranchStore.resume_proof_audit(
                self.checkpoint, ("",)
            )
        with self.assertRaises(ValueError):
            BranchStore.resume_proof_audit(
                self.checkpoint, (self.proof, "not a proof")
            )
        pretty = json.dumps(json.loads(self.proof), indent=2)
        with self.assertRaises(ValueError):
            BranchStore.resume_proof_audit(
                self.checkpoint, (pretty,)
            )

    def test_time_range_new_proofs_are_rejected(self) -> None:
        timed = BranchStore.export_diagnostic_ledger_proof(
            self.directory, 1, 2, None
        )
        with self.assertRaises(ValueError):
            BranchStore.resume_proof_audit(
                self.checkpoint, (timed,)
            )

    def test_new_proof_identity_set_must_match_checkpoint(self) -> None:
        other = self.export(
            self.directory, (transaction(50),)
        )
        with self.assertRaises(ValueError):
            BranchStore.resume_proof_audit(
                self.checkpoint, (other,)
            )
        reordered = self.export(
            self.directory, (transaction(1), transaction(0))
        )
        with self.assertRaises(ValueError):
            BranchStore.resume_proof_audit(
                self.checkpoint, (reordered,)
            )

    def test_all_new_proofs_authenticate_before_any_hop(self) -> None:
        _directory, (first, second) = self.grow(
            "growing", (2, 2)
        )
        # second would continue first legally, but the trailing string
        # is invalid: the authentication fault wins.
        with self.assertRaises(ValueError):
            BranchStore.resume_proof_audit(
                self.checkpoint, (second, "not a proof")
            )

    def test_internally_inconsistent_checkpoints_rejected(self) -> None:
        _directory, (first, second) = self.grow(
            "growing", (2, 2)
        )
        checkpoint = (
            BranchStore.create_proof_audit_checkpoint(
                (first, second)
            )
        )

        def resealed(document):
            return self.reseal_checkpoint(document)

        # Cumulative count disagreeing with the hop count.
        document = json.loads(checkpoint)
        document["count"] = 5
        with self.assertRaises(ValueError):
            BranchStore.resume_proof_audit(
                resealed(document), ()
            )
        # Common boundary not following from the first hop.
        document = json.loads(checkpoint)
        document["common"]["entries"] = 99
        with self.assertRaises(ValueError):
            BranchStore.resume_proof_audit(
                resealed(document), ()
            )
        # Non-consecutive hop indices.
        document = json.loads(checkpoint)
        document["hops"][0]["index"] = 7
        with self.assertRaises(ValueError):
            BranchStore.resume_proof_audit(
                resealed(document), ()
            )
        # The last-proof view that no longer authenticates.
        document = json.loads(checkpoint)
        document["last"]["view"]["segments"][0]["digest"] = (
            "a" * 64
        )
        with self.assertRaises(ValueError):
            BranchStore.resume_proof_audit(
                resealed(document), ()
            )
        # Identities reordered relative to the stored last filter.
        document = json.loads(checkpoint)
        document["identities"] = list(
            reversed(document["identities"])
        )
        with self.assertRaises(ValueError):
            BranchStore.resume_proof_audit(
                resealed(document), ()
            )
        # Terminal failure whose classification contradicts the hop.
        base = self.ledger("base")
        digest = self.append(base, self.entries[:2], None)
        digest = self.append(base, self.entries[2:4], digest)
        middle = self.export(base)
        digest = self.append(base, self.entries[4:6], digest)
        newest = self.export(base)
        failed = (
            BranchStore.create_proof_audit_checkpoint(
                (self.proof, middle, newest, middle)
            )
        )
        document = json.loads(failed)
        document["failure"]["classification"] = "spliced"
        with self.assertRaises(ValueError):
            BranchStore.resume_proof_audit(
                resealed(document), ()
            )


class EmptyContinuationTests(CheckpointTestBase):
    def test_empty_bootstrap_then_first_batch_sets_identities(
        self,
    ) -> None:
        _directory, (first, second) = self.grow(
            "ledger", (4, 2)
        )
        bootstrap = (
            BranchStore.create_proof_audit_checkpoint(())
        )
        result = BranchStore.resume_proof_audit(
            bootstrap, (first, second)
        )
        one_shot = (
            BranchStore.audit_diagnostic_ledger_proof_sequence(
                (first, second)
            )
        )
        self.assertEqual(result["audit"], one_shot)
        document = json.loads(result["checkpoint"])
        self.assertEqual(
            document["identities"], list(self.identities)
        )

    def test_empty_continuation_is_a_pure_read_only_query(self) -> None:
        _directory, (first, second) = self.grow(
            "ledger", (4, 2)
        )
        checkpoint = (
            BranchStore.create_proof_audit_checkpoint((first, second))
        )
        result = BranchStore.resume_proof_audit(checkpoint, ())
        # Same bytes, isolated levels, no new hop.
        self.assertEqual(result["checkpoint"], checkpoint)
        one_shot = (
            BranchStore.audit_diagnostic_ledger_proof_sequence(
                (first, second)
            )
        )
        self.assertEqual(result["audit"], one_shot)
        self.assertEqual(
            [hop["index"] for hop in result["audit"]["hops"]], [1]
        )
        # Repeated queries stay byte-for-byte stable.
        again = BranchStore.resume_proof_audit(
            result["checkpoint"], ()
        )
        self.assertEqual(again["checkpoint"], checkpoint)

    def test_failed_checkpoint_answers_empty_but_rejects_more(
        self,
    ) -> None:
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
        before, after = self.export(left), self.export(right)
        sealed = (
            BranchStore.create_proof_audit_checkpoint(
                (before, after)
            )
        )
        read_only = BranchStore.resume_proof_audit(sealed, ())
        self.assertEqual(read_only["checkpoint"], sealed)
        self.assertEqual(
            read_only["audit"],
            BranchStore.audit_diagnostic_ledger_proof_sequence(
                (before, after)
            ),
        )
        with self.assertRaises(RuntimeError):
            BranchStore.resume_proof_audit(sealed, (before,))
        # The rejected continuation does not change the bytes.
        self.assertEqual(
            BranchStore.resume_proof_audit(sealed, ())[
                "checkpoint"
            ],
            sealed,
        )


class ResumeEquivalenceTests(CheckpointTestBase):
    def _growing_proofs(self, name, sizes, keep=10):
        _directory, proofs = self.grow(name, sizes, keep=keep)
        return proofs

    def test_single_proof_seed_then_appends(self) -> None:
        proofs = self._growing_proofs("ledger", (4, 4, 2))
        seed = (
            BranchStore.create_proof_audit_checkpoint((proofs[0],))
        )
        final = self.assert_batches_equal_one_shot(
            tuple(proofs),
            [(proofs[1],), (proofs[2],)],
            seed=seed,
        )
        self.assertEqual(
            [hop["index"] for hop in json.loads(final)["hops"]],
            [1, 2],
        )

    def test_empty_bootstrap_arbitrary_batching(self) -> None:
        proofs = self._growing_proofs("ledger", (2, 2, 2, 2, 2))
        for batches in (
            [(proofs[0],), (proofs[1],), (proofs[2],),
             (proofs[3],), (proofs[4],)],
            [tuple(proofs)],
            [(proofs[0], proofs[1]), (proofs[2], proofs[3], proofs[4])],
            [tuple(proofs[:3]), tuple(proofs[3:])],
        ):
            with self.subTest(batches=len(batches)):
                self.assert_batches_equal_one_shot(
                    tuple(proofs), batches
                )

    def test_empty_batches_between_real_batches_add_no_hop(self) -> None:
        proofs = self._growing_proofs("ledger", (4, 2, 2))
        bootstrap = (
            BranchStore.create_proof_audit_checkpoint(())
        )
        first = BranchStore.resume_proof_audit(
            bootstrap, (proofs[0],)
        )
        idle = BranchStore.resume_proof_audit(
            first["checkpoint"], ()
        )
        self.assertEqual(idle["checkpoint"], first["checkpoint"])
        second = BranchStore.resume_proof_audit(
            idle["checkpoint"], (proofs[1], proofs[2])
        )
        self.assertEqual(
            second["audit"],
            BranchStore.audit_diagnostic_ledger_proof_sequence(
                tuple(proofs)
            ),
        )
        self.assertEqual(
            [hop["index"] for hop in second["audit"]["hops"]],
            [1, 2],
        )

    def test_same_hop_across_the_batch_boundary(self) -> None:
        proofs = self._growing_proofs("ledger", (4,))
        seed = (
            BranchStore.create_proof_audit_checkpoint((proofs[0],))
        )
        result = BranchStore.resume_proof_audit(
            seed, (proofs[0], proofs[0])
        )
        self.assertEqual(
            result["audit"],
            BranchStore.audit_diagnostic_ledger_proof_sequence(
                (proofs[0], proofs[0], proofs[0])
            ),
        )
        self.assertEqual(
            [hop["relation"] for hop in result["audit"]["hops"]],
            ["same", "same"],
        )

    def test_rotation_across_the_batch_boundary(self) -> None:
        directory = self.ledger("ledger")
        digest = self.append(directory, self.entries[:4], None)
        identities = (
            transaction(2), transaction(0), transaction(9)
        )
        before = self.export(directory, identities)
        self.append(
            directory, self.entries[4:6], digest, keep=1
        )
        after = self.export(directory, identities)
        seed = (
            BranchStore.create_proof_audit_checkpoint((before,))
        )
        result = BranchStore.resume_proof_audit(seed, (after,))
        self.assertEqual(
            result["audit"],
            BranchStore.audit_diagnostic_ledger_proof_sequence(
                (before, after)
            ),
        )
        self.assertEqual(
            result["audit"]["hops"][0]["rotated"],
            (transaction(2), transaction(0)),
        )

    def test_empty_ledger_first_proof_then_history(self) -> None:
        empty_dir = self.ledger("empty")
        self.append(empty_dir, (), None)
        empty = self.export(empty_dir)
        directory = self.ledger("ledger")
        self.append(directory, self.entries[:4], None, keep=1)
        self.append(
            directory,
            self.entries[4:6],
            self.digest_of(directory),
            keep=1,
        )
        grown = self.export(directory)
        seed = (
            BranchStore.create_proof_audit_checkpoint((empty,))
        )
        result = BranchStore.resume_proof_audit(seed, (grown,))
        self.assertEqual(
            result["audit"],
            BranchStore.audit_diagnostic_ledger_proof_sequence(
                (empty, grown)
            ),
        )
        self.assertIsNone(result["audit"]["common"])

    def _fork_pair(self):
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

    def test_fork_on_the_cross_batch_hop_seals(self) -> None:
        before, after = self._fork_pair()
        seed = (
            BranchStore.create_proof_audit_checkpoint((before,))
        )
        result = BranchStore.resume_proof_audit(seed, (after,))
        self.assertEqual(
            result["audit"],
            BranchStore.audit_diagnostic_ledger_proof_sequence(
                (before, after)
            ),
        )
        self.assertEqual(result["audit"]["status"], "failed")
        self.assertEqual(
            result["audit"]["failure"],
            {"index": 1, "classification": "forked"},
        )

    def test_successful_then_fork_then_sealed_across_batches(self) -> None:
        directory = self.ledger("ledger")
        digest = self.append(directory, self.entries[:2], None)
        first = self.export(directory)
        digest = self.append(directory, self.entries[2:4], digest)
        before, after = self._fork_pair()
        seed = (
            BranchStore.create_proof_audit_checkpoint((first,))
        )
        result = BranchStore.resume_proof_audit(
            seed, (before, after)
        )
        self.assertEqual(
            result["audit"],
            BranchStore.audit_diagnostic_ledger_proof_sequence(
                (first, before, after)
            ),
        )
        self.assertEqual(
            [hop["relation"] for hop in result["audit"]["hops"]],
            ["continued", "forked"],
        )
        self.assertEqual(
            result["audit"]["failure"],
            {"index": 2, "classification": "forked"},
        )

    def test_rollback_on_the_cross_batch_hop(self) -> None:
        directory = self.ledger("ledger")
        digest = self.append(directory, self.entries[:2], None)
        oldest = self.export(directory)
        digest = self.append(directory, self.entries[2:4], digest)
        middle = self.export(directory)
        digest = self.append(directory, self.entries[4:6], digest)
        newest = self.export(directory)
        seed = (
            BranchStore.create_proof_audit_checkpoint(
                (oldest, middle, newest)
            )
        )
        result = BranchStore.resume_proof_audit(seed, (middle,))
        self.assertEqual(
            result["audit"],
            BranchStore.audit_diagnostic_ledger_proof_sequence(
                (oldest, middle, newest, middle)
            ),
        )
        self.assertEqual(
            result["audit"]["failure"]["classification"],
            "rollback",
        )
        self.assertEqual(
            [hop["index"] for hop in result["audit"]["hops"]],
            [1, 2, 3],
        )

    def test_splice_on_the_cross_batch_hop(self) -> None:
        first_dir = self.ledger("first")
        self.append(first_dir, self.entries[:2], None)
        first = self.export(first_dir)
        second_dir = self.ledger("second")
        other = tuple(
            (time, make_chain(transaction(50 + time)))
            for time in (1, 2)
        )
        self.append(second_dir, other, None)
        second = self.export(second_dir)
        seed = (
            BranchStore.create_proof_audit_checkpoint((first,))
        )
        result = BranchStore.resume_proof_audit(seed, (second,))
        self.assertEqual(
            result["audit"],
            BranchStore.audit_diagnostic_ledger_proof_sequence(
                (first, second)
            ),
        )
        self.assertEqual(
            result["audit"]["failure"]["classification"],
            "spliced",
        )

    def test_missing_transition_on_the_cross_batch_hop(self) -> None:
        directory = self.ledger("ledger")
        digest = self.append(
            directory, self.entries[:4], None, keep=2
        )
        before = self.export(directory)
        digest = self.append(
            directory, self.entries[4:8], digest, keep=2
        )
        self.append(
            directory, self.entries[8:10], digest, keep=2
        )
        after = self.export(directory)
        seed = (
            BranchStore.create_proof_audit_checkpoint((before,))
        )
        result = BranchStore.resume_proof_audit(seed, (after,))
        self.assertEqual(
            result["audit"],
            BranchStore.audit_diagnostic_ledger_proof_sequence(
                (before, after)
            ),
        )
        self.assertEqual(
            result["audit"]["failure"]["classification"],
            "missing_transition",
        )

    def test_rewrite_on_the_cross_batch_hop(self) -> None:
        directory = self.ledger("ledger")
        digest = self.append(
            directory, self.entries[:4], None, keep=1
        )
        before = self.export(directory)
        self.append(
            directory, self.entries[4:6], digest, keep=1
        )
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
        document["manifest"]["segments"][0]["digest"] = (
            segment_digest
        )
        tampered = self.reseal_proof(document)
        seed = (
            BranchStore.create_proof_audit_checkpoint((before,))
        )
        result = BranchStore.resume_proof_audit(seed, (tampered,))
        self.assertEqual(
            result["audit"],
            BranchStore.audit_diagnostic_ledger_proof_sequence(
                (before, tampered)
            ),
        )
        self.assertEqual(
            result["audit"]["failure"]["classification"],
            "rewritten",
        )

    def test_failure_inside_a_later_batch_seals_rest_of_batch(
        self,
    ) -> None:
        directory = self.ledger("ledger")
        digest = self.append(directory, self.entries[:2], None)
        oldest = self.export(directory)
        digest = self.append(directory, self.entries[2:4], digest)
        middle = self.export(directory)
        digest = self.append(directory, self.entries[4:6], digest)
        newest = self.export(directory)
        seed = (
            BranchStore.create_proof_audit_checkpoint(
                (oldest, middle)
            )
        )
        # newest continues, middle rolls back: the rollback inside the
        # batch seals and later proofs never get analysed.
        result = BranchStore.resume_proof_audit(
            seed, (newest, middle, newest, middle)
        )
        self.assertEqual(
            result["audit"],
            BranchStore.audit_diagnostic_ledger_proof_sequence(
                (oldest, middle, newest, middle)
            ),
        )
        self.assertEqual(
            [hop["relation"] for hop in result["audit"]["hops"]],
            ["continued", "continued", "rollback"],
        )

    def test_checkpoint_round_trips_through_another_process(self) -> None:
        _directory, (first, second) = self.grow(
            "ledger", (4, 2)
        )
        checkpoint = (
            BranchStore.create_proof_audit_checkpoint((first,))
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = os.getcwd()
        script = (
            "import os, json, pickle, sys\n"
            "from city_twin.branches import BranchStore\n"
            "checkpoint, proof = pickle.loads("
            "bytes.fromhex(os.environ['CP_ARGS']))\n"
            "result = BranchStore.resume_proof_audit("
            "checkpoint, (proof,))\n"
            "sys.stdout.write(json.dumps({"
            "'status': result['audit']['status'], "
            "'checkpoint': result['checkpoint']}))\n"
        )
        payload = (
            pickle.dumps((checkpoint, second)).hex()
        )
        env["CP_ARGS"] = payload
        completed = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        remote = json.loads(completed.stdout)
        local = BranchStore.resume_proof_audit(
            checkpoint, (second,)
        )
        self.assertEqual(remote["status"], "continuous")
        self.assertEqual(remote["checkpoint"], local["checkpoint"])


class IsolationAndPurityTests(CheckpointTestBase):
    def test_returned_levels_never_contaminate_later_calls(self) -> None:
        _directory, (first, second, third) = self.grow(
            "ledger", (4, 2, 2), keep=1
        )
        checkpoint = (
            BranchStore.create_proof_audit_checkpoint((first,))
        )
        first_result = BranchStore.resume_proof_audit(
            checkpoint, (second,)
        )
        first_result["audit"]["common"]["digest"] = "mutated"
        first_result["audit"]["hops"][0]["common"]["digest"] = (
            "mutated"
        )
        first_result["audit"]["hops"][0]["after_only"][0][
            "digest"
        ] = "mutated"
        first_result["audit"]["hops"][0]["after_only"][0][
            "entries"
        ][0]["summary"]["transaction"] = "mutated"
        mutable = list(first_result["audit"]["hops"][0]["rotated"])
        mutable.append("mutated")
        second_result = BranchStore.resume_proof_audit(
            checkpoint, (second,)
        )
        self.assertNotEqual(
            second_result["audit"]["common"]["digest"], "mutated"
        )
        hop = second_result["audit"]["hops"][0]
        self.assertNotEqual(
            hop["common"]["digest"], "mutated"
        )
        self.assertNotEqual(
            hop["after_only"][0]["digest"], "mutated"
        )
        self.assertNotIn("mutated", hop["rotated"])
        # Chaining off the mutated return's checkpoint is impossible
        # (strings), but the sealed bytes still resume cleanly.
        chained = BranchStore.resume_proof_audit(
            second_result["checkpoint"], (third,)
        )
        self.assertEqual(
            chained["audit"],
            BranchStore.audit_diagnostic_ledger_proof_sequence(
                (first, second, third)
            ),
        )

    def test_empty_continuation_result_is_detached(self) -> None:
        _directory, (first, second) = self.grow(
            "ledger", (4, 2)
        )
        checkpoint = (
            BranchStore.create_proof_audit_checkpoint((first, second))
        )
        read = BranchStore.resume_proof_audit(checkpoint, ())
        read["audit"]["common"]["digest"] = "mutated"
        again = BranchStore.resume_proof_audit(checkpoint, ())
        self.assertNotEqual(
            again["audit"]["common"]["digest"], "mutated"
        )

    def test_nested_tuple_shapes_match_the_one_shot_audit(self) -> None:
        _directory, (first, second) = self.grow(
            "ledger", (4, 2)
        )
        result = BranchStore.resume_proof_audit(
            BranchStore.create_proof_audit_checkpoint((first,)),
            (second,),
        )
        self.assertIsInstance(result["audit"]["hops"], tuple)
        for hop in result["audit"]["hops"]:
            self.assertIsInstance(hop["after_only"], tuple)
            self.assertIsInstance(hop["before_only"], tuple)
            self.assertIsInstance(hop["rotated"], tuple)
            for segment in hop["after_only"]:
                self.assertIsInstance(segment["entries"], tuple)
                for entry in segment["entries"]:
                    self.assertIsInstance(entry["records"], tuple)

    def test_success_and_failure_never_touch_the_ledger(self) -> None:
        directory, (first, second) = self.grow(
            "ledger", (4, 2), keep=1
        )
        snapshot = self._directory_files(directory)
        digest = self.digest_of(directory)
        BranchStore.create_proof_audit_checkpoint((first,))
        BranchStore.create_proof_audit_checkpoint((first, second))
        seed = (
            BranchStore.create_proof_audit_checkpoint((first,))
        )
        BranchStore.resume_proof_audit(seed, (second,))
        BranchStore.resume_proof_audit(seed, ())
        with self.assertRaises(ValueError):
            BranchStore.create_proof_audit_checkpoint(
                (first, "not a proof")
            )
        with self.assertRaises(ValueError):
            BranchStore.resume_proof_audit(seed, ("not a proof",))
        with self.assertRaises(ValueError):
            BranchStore.resume_proof_audit("not a checkpoint", ())
        self.assertEqual(snapshot, self._directory_files(directory))
        self.assertEqual(self.digest_of(directory), digest)
        self.assertEqual(
            self.export(directory), second
        )

    def test_result_dict_has_audit_and_checkpoint_in_order(self) -> None:
        _directory, (first,) = self.grow("ledger", (4,))
        result = BranchStore.resume_proof_audit(
            BranchStore.create_proof_audit_checkpoint(()),
            (first,),
        )
        self.assertEqual(list(result), ["audit", "checkpoint"])
        idle = BranchStore.resume_proof_audit(
            BranchStore.create_proof_audit_checkpoint((first,)), ()
        )
        self.assertEqual(list(idle), ["audit", "checkpoint"])


class CheckpointTamperAnchorTests(CheckpointTestBase):
    """Every historical fact the last proof authenticates is anchored,
    so re-sealing a mutated field never produces a checkpoint that
    parses."""

    def resealed(self, document: dict) -> str:
        return self.reseal_checkpoint(document)

    def assert_rejected(self, document: dict) -> None:
        with self.assertRaises(ValueError):
            BranchStore.resume_proof_audit(
                self.resealed(document), ()
            )

    def _rotation_proofs(self, keep=1):
        directory = self.ledger("rotating")
        digest = self.append(
            directory, self.entries[:4], None, keep=10
        )
        first = self.export(directory)
        digest = self.append(
            directory, self.entries[4:6], digest, keep=keep
        )
        second = self.export(directory)
        digest = self.append(
            directory, self.entries[6:8], digest, keep=keep
        )
        third = self.export(directory)
        return first, second, third

    def test_tampering_a_hop_rotated_identity_is_rejected(self) -> None:
        _first, second, third = self._rotation_proofs()
        checkpoint = (
            BranchStore.create_proof_audit_checkpoint(
                (second, third)
            )
        )
        document = json.loads(checkpoint)
        document["hops"][0]["rotated"][0] = transaction(99)
        self.assert_rejected(document)

    def test_rotated_identity_out_of_order_is_rejected(self) -> None:
        _first, second, third = self._rotation_proofs()
        checkpoint = (
            BranchStore.create_proof_audit_checkpoint(
                (second, third)
            )
        )
        rotated = json.loads(checkpoint)["hops"][0]["rotated"]
        if len(rotated) >= 2:
            document = json.loads(checkpoint)
            document["hops"][0]["rotated"] = list(
                reversed(rotated)
            )
            self.assert_rejected(document)

    def test_rotating_an_identity_the_last_proof_retains_is_rejected(
        self,
    ) -> None:
        _first, second, third = self._rotation_proofs()
        checkpoint = (
            BranchStore.create_proof_audit_checkpoint(
                (second, third)
            )
        )
        document = json.loads(checkpoint)
        retained = transaction(6)  # in third's retained segments
        document["hops"][0]["rotated"].append(retained)
        self.assert_rejected(document)

    def test_tampering_a_carried_segment_record_is_rejected(self) -> None:
        _directory, (first, second, third) = self.grow(
            "growing", (4, 2, 2)
        )
        checkpoint = (
            BranchStore.create_proof_audit_checkpoint(
                (second, third)
            )
        )
        item = json.loads(checkpoint)["hops"][0]["after_only"][0]
        # Segment index swap breaks the contiguous chain.
        document = json.loads(checkpoint)
        document["hops"][0]["after_only"][0]["index"] = (
            item["index"] + 5
        )
        self.assert_rejected(document)
        # A recorded diagnostic record body swap breaks the recomputed
        # segment digest.
        document = json.loads(checkpoint)
        entry = document["hops"][0]["after_only"][0]["entries"][0]
        entry["records"] = list(entry["records"])
        entry["records"][0] = entry["records"][0] + "X"
        self.assert_rejected(document)
        # A stored summary swap breaks summary re-derivation.
        document = json.loads(checkpoint)
        document["hops"][0]["after_only"][0]["entries"][0][
            "summary"
        ]["count"] = 99
        self.assert_rejected(document)

    def test_tampering_a_forks_abandoned_leg_is_rejected(self) -> None:
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
        checkpoint = (
            BranchStore.create_proof_audit_checkpoint(
                (self.export(left), self.export(right))
            )
        )
        leg = json.loads(checkpoint)["hops"][0]["before_only"]
        self.assertTrue(leg)
        document = json.loads(checkpoint)
        document["hops"][0]["before_only"][0]["digest"] = (
            "f" * 64
        )
        self.assert_rejected(document)
        document = json.loads(checkpoint)
        document["hops"][0]["before_only"][0]["entries"][0][
            "records"
        ][0] += "X"
        self.assert_rejected(document)

    def test_tampering_a_rollback_leg_across_rotation_is_rejected(
        self,
    ) -> None:
        directory = self.ledger("rotating")
        digest = self.append(
            directory, self.entries[:4], None, keep=10
        )
        oldest = self.export(directory)
        digest = self.append(
            directory, self.entries[4:6], digest, keep=1
        )
        middle = self.export(directory)
        digest = self.append(
            directory, self.entries[6:8], digest, keep=1
        )
        newest = self.export(directory)
        digest = self.append(
            directory, self.entries[8:10], digest, keep=1
        )
        latest = self.export(directory)
        checkpoint = (
            BranchStore.create_proof_audit_checkpoint(
                (oldest, middle, newest, latest, middle)
            )
        )
        document = json.loads(checkpoint)
        rollback_hop = document["hops"][-1]
        rollback_hop["before_only"][0]["digest"] = "f" * 64
        self.assert_rejected(document)


if __name__ == "__main__":
    unittest.main()
