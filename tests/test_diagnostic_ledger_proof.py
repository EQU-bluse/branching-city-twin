"""Tests for offline-verifiable proofs exported from the segmented,
possibly rotated diagnostic ledger.

``BranchStore.export_diagnostic_ledger_proof`` exports a canonical,
checksummed proof string bound to one published directory-ledger
snapshot -- its manifest, the required complete segment bodies, the
rotated-prefix evidence and the filter -- under the same shared reader
lock directory pages use; ``BranchStore.verify_diagnostic_ledger_proof``
authenticates that string with no access to the original ledger and
returns an isolated ``snapshot``/``items``/``rotated``/``absent``
mapping.

These tests cover:

* argument validation for both entry points (directory, proof string,
  mutually exclusive time-range and transaction-identity modes, bounds
  and identity shapes);
* transaction mode: every requested identity classified exactly once
  as retained, rotated or never seen, items in ledger order, the extra
  classes in request order;
* time mode: inclusive bounds, only intersecting segments carried,
  complete in-range retained entries, and refusal (ValueError) when the
  range reaches into the rotated prefix whose per-entry times are not
  recoverable;
* offline verification with the ledger deleted afterwards and strict
  result isolation;
* tampering: checksum, manifest, segment bodies and links, missing or
  extra segments, cross-snapshot splicing, filter and snapshot
  rewrites, range/coverage gaps and one identity landing in multiple
  classes, all rejected with ``ValueError``;
* coordination: the export takes the shared read lock on the append
  coordination file, observes whole snapshots and creates no files;
* the single-file ledger, append, paging and the ``status`` command are
  unchanged.
"""

import hashlib
import json
import os
import tempfile
import threading
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


def read_bytes(path: str) -> bytes:
    with open(path, "rb") as handle:
        return handle.read()


def canonical(document: dict) -> str:
    return BranchStore._canonical_json(document)


def reseal_proof(document: dict) -> str:
    """Re-seal a (possibly tampered) proof with a fresh package
    checksum over every other field."""
    document = dict(document)
    body = {
        key: value for key, value in document.items() if key != "checksum"
    }
    document["checksum"] = hashlib.sha256(
        canonical(body).encode("utf-8")
    ).hexdigest()
    return canonical(document)


class ProofTestBase(unittest.TestCase):
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

    def publish_rotated(self, keep=1, segment_limit=2):
        """Publish three batches so the first four entries are rotated
        away and only the last batch is retained; return the digest."""
        digest = self.append(
            self.entries[0:2], None, keep=keep, segment_limit=segment_limit
        )
        digest = self.append(
            self.entries[2:4], digest, keep=keep,
            segment_limit=segment_limit,
        )
        digest = self.append(
            self.entries[4:6], digest, keep=keep,
            segment_limit=segment_limit,
        )
        return digest

    def export(self, start=..., end=..., transactions=...):
        if start is ...:
            start = None
        if end is ...:
            end = None
        if transactions is ...:
            transactions = (
                transaction(0),
                transaction(4),
                transaction(5),
                transaction(99),
            )
        return BranchStore.export_diagnostic_ledger_proof(
            self.dir, start, end, transactions
        )

    def verify(self, proof=None):
        if proof is None:
            proof = self.export()
        return BranchStore.verify_diagnostic_ledger_proof(proof)

    def manifest(self) -> dict:
        return json.loads(
            read_bytes(os.path.join(self.dir, "manifest.json")).decode(
                "utf-8"
            )
        )

    def lock_path(self) -> str:
        return os.path.join(
            os.path.realpath(self.dir),
            BranchStore._DIAGNOSTIC_LEDGER_LOCK_NAME,
        )


class ExportValidationTests(ProofTestBase):
    def test_directory_validation(self) -> None:
        for bad in (1, 1.5, b"x", ["x"], {"x": 1}, object(), None):
            with self.subTest(bad=type(bad).__name__):
                with self.assertRaises(TypeError):
                    BranchStore.export_diagnostic_ledger_proof(
                        bad, None, None, (transaction(0),)
                    )
        with self.assertRaises(ValueError):
            BranchStore.export_diagnostic_ledger_proof(
                "", None, None, (transaction(0),)
            )

    def test_missing_directory_raises_oserror(self) -> None:
        missing = os.path.join(self.tmp.name, "nope")
        with self.assertRaises(OSError):
            BranchStore.export_diagnostic_ledger_proof(
                missing, None, None, (transaction(0),)
            )

    def test_single_file_ledger_is_not_a_proof_directory(self) -> None:
        path = os.path.join(self.tmp.name, "ledger.json")
        BranchStore.save_diagnostic_ledger(path, self.entries[:2])
        with self.assertRaises(OSError):
            BranchStore.export_diagnostic_ledger_proof(
                path, None, None, (transaction(0),)
            )

    def test_directory_without_manifest_raises_oserror(self) -> None:
        with self.assertRaises(OSError):
            BranchStore.export_diagnostic_ledger_proof(
                self.dir, None, None, (transaction(0),)
            )

    def test_bound_types(self) -> None:
        self.publish_rotated()
        for name in ("start", "end"):
            for bad in (True, False, 1.5, "1", b"1"):
                with self.subTest(name=name, bad=bad):
                    with self.assertRaises(TypeError):
                        kwargs = {name: bad}
                        BranchStore.export_diagnostic_ledger_proof(
                            self.dir,
                            kwargs.get("start", 1),
                            kwargs.get("end", 1),
                            None,
                        )

    def test_negative_bounds_are_value_errors(self) -> None:
        self.publish_rotated()
        with self.assertRaises(ValueError):
            BranchStore.export_diagnostic_ledger_proof(
                self.dir, -1, 4, None
            )
        with self.assertRaises(ValueError):
            BranchStore.export_diagnostic_ledger_proof(
                self.dir, 1, -1, None
            )

    def test_bounds_must_be_both_none_or_both_int(self) -> None:
        self.publish_rotated()
        with self.assertRaises(ValueError):
            BranchStore.export_diagnostic_ledger_proof(
                self.dir, 1, None, None
            )
        with self.assertRaises(ValueError):
            BranchStore.export_diagnostic_ledger_proof(
                self.dir, None, 4, None
            )

    def test_reversed_range_is_value_error(self) -> None:
        self.publish_rotated()
        with self.assertRaises(ValueError):
            BranchStore.export_diagnostic_ledger_proof(
                self.dir, 6, 5, None
            )

    def test_transactions_container_validation(self) -> None:
        self.publish_rotated()
        for bad in (["a"], "a", 1, {"a"}, {"a": 1}):
            with self.subTest(bad=type(bad).__name__):
                with self.assertRaises(TypeError):
                    BranchStore.export_diagnostic_ledger_proof(
                        self.dir, None, None, bad
                    )
        with self.assertRaises(TypeError):
            BranchStore.export_diagnostic_ledger_proof(
                self.dir, None, None, (transaction(0), 1)
            )

    def test_transactions_value_validation(self) -> None:
        self.publish_rotated()
        with self.assertRaises(ValueError):
            BranchStore.export_diagnostic_ledger_proof(
                self.dir, None, None, ()
            )
        with self.assertRaises(ValueError):
            BranchStore.export_diagnostic_ledger_proof(
                self.dir, None, None, ("",)
            )
        with self.assertRaises(ValueError):
            BranchStore.export_diagnostic_ledger_proof(
                self.dir, None, None,
                (transaction(0), transaction(0)),
            )

    def test_modes_are_mutually_exclusive_and_required(self) -> None:
        self.publish_rotated()
        # Neither mode.
        with self.assertRaises(ValueError):
            BranchStore.export_diagnostic_ledger_proof(
                self.dir, None, None, None
            )
        # Both modes.
        with self.assertRaises(ValueError):
            BranchStore.export_diagnostic_ledger_proof(
                self.dir, 1, 4, (transaction(0),)
            )

    def test_proof_argument_validation(self) -> None:
        for bad in (1, 1.5, b"x", ["x"], {"x": 1}, object(), None):
            with self.subTest(bad=type(bad).__name__):
                with self.assertRaises(TypeError):
                    BranchStore.verify_diagnostic_ledger_proof(bad)
        with self.assertRaises(ValueError):
            BranchStore.verify_diagnostic_ledger_proof("")

    def test_missing_segment_file_raises_oserror(self) -> None:
        self.publish_rotated()
        os.remove(os.path.join(self.dir, "segment-000003.json"))
        with self.assertRaises(OSError):
            self.export()


class TransactionModeProofTests(ProofTestBase):
    def setUp(self) -> None:
        super().setUp()
        self.digest = self.publish_rotated()

    def test_result_shape_and_key_order(self) -> None:
        result = self.verify()
        self.assertEqual(
            list(result), ["snapshot", "items", "rotated", "absent"]
        )
        snapshot = result["snapshot"]
        self.assertEqual(
            list(snapshot),
            ["digest", "bytes", "segments", "entries", "dropped",
             "evidence"],
        )
        manifest = self.manifest()
        self.assertEqual(snapshot["digest"], self.digest)
        raw = read_bytes(os.path.join(self.dir, "manifest.json"))
        self.assertEqual(snapshot["bytes"], len(raw))
        self.assertEqual(snapshot["segments"], 1)
        self.assertEqual(snapshot["entries"], 2)
        self.assertEqual(snapshot["dropped"], manifest["dropped"])
        self.assertEqual(
            snapshot["evidence"], manifest["evidence"]["digest"]
        )

    def test_identities_are_classified_retained_rotated_absent(self):
        result = self.verify()
        retained = tuple(
            item["summary"]["transaction"] for item in result["items"]
        )
        self.assertEqual(
            retained,
            tuple(sorted((transaction(4), transaction(5)))),
        )
        self.assertEqual(result["rotated"], (transaction(0),))
        self.assertEqual(result["absent"], (transaction(99),))
        for item, at in zip(result["items"], (5, 6)):
            self.assertEqual(list(item), ["at", "summary", "records"])
            self.assertIsInstance(item["records"], tuple)
            self.assertEqual(item["at"], at)
            self.assertTrue(item["summary"]["terminal"])

    def test_extra_classes_follow_the_request_order(self) -> None:
        identities = (
            transaction(99),
            transaction(3),
            transaction(5),
            transaction(0),
            transaction(98),
        )
        proof = BranchStore.export_diagnostic_ledger_proof(
            self.dir, None, None, identities
        )
        result = BranchStore.verify_diagnostic_ledger_proof(proof)
        self.assertEqual(
            result["rotated"], (transaction(3), transaction(0))
        )
        self.assertEqual(
            result["absent"], (transaction(99), transaction(98))
        )
        self.assertEqual(len(result["items"]), 1)

    def test_all_retained_all_rotated_all_absent(self) -> None:
        retained = self.verify(
            BranchStore.export_diagnostic_ledger_proof(
                self.dir, None, None,
                (transaction(4), transaction(5)),
            )
        )
        self.assertEqual(len(retained["items"]), 2)
        self.assertEqual(retained["rotated"], ())
        self.assertEqual(retained["absent"], ())
        rotated = self.verify(
            BranchStore.export_diagnostic_ledger_proof(
                self.dir, None, None,
                tuple(transaction(i) for i in range(4)),
            )
        )
        self.assertEqual(rotated["items"], ())
        self.assertEqual(
            rotated["rotated"], tuple(transaction(i) for i in range(4))
        )
        absent = self.verify(
            BranchStore.export_diagnostic_ledger_proof(
                self.dir, None, None, (transaction(40), transaction(41))
            )
        )
        self.assertEqual(absent["items"], ())
        self.assertEqual(absent["rotated"], ())
        self.assertEqual(
            absent["absent"], (transaction(40), transaction(41))
        )

    def test_items_follow_ledger_time_and_transaction_order(self) -> None:
        # No rotation: retain all eight entries, ask for them out of
        # order; items still come back in (time, transaction) order.
        directory = os.path.join(self.tmp.name, "full")
        os.mkdir(directory)
        digest = BranchStore.append_diagnostic_ledger(
            directory, self.entries, None, 3, 10, None
        )
        identities = tuple(
            sorted(
                (transaction(i) for i in range(8)),
                key=lambda value: (-sum(map(ord, value[:4])), value),
            )
        )
        proof = BranchStore.export_diagnostic_ledger_proof(
            directory, None, None, identities
        )
        result = BranchStore.verify_diagnostic_ledger_proof(proof)
        self.assertEqual(
            [item["at"] for item in result["items"]],
            [1, 2, 3, 4, 5, 6, 7, 8],
        )
        self.assertEqual(result["snapshot"]["digest"], digest)

    def test_transaction_mode_carries_every_retained_segment(self) -> None:
        directory = os.path.join(self.tmp.name, "full")
        os.mkdir(directory)
        BranchStore.append_diagnostic_ledger(
            directory, self.entries, None, 2, 10, None
        )
        # Even though only one identity is requested, all four retained
        # segment bodies must be bound by the proof.
        proof = BranchStore.export_diagnostic_ledger_proof(
            directory, None, None, (transaction(7),)
        )
        document = json.loads(proof)
        self.assertEqual(
            [segment["name"] for segment in document["segments"]],
            [f"segment-{index:06d}.json" for index in range(1, 5)],
        )
        result = BranchStore.verify_diagnostic_ledger_proof(proof)
        self.assertEqual(
            [item["at"] for item in result["items"]], [8]
        )

    def test_proof_envelope_shape(self) -> None:
        proof = self.export()
        self.assertFalse(proof.startswith("\ufeff"))
        self.assertFalse(proof.endswith("\n"))
        self.assertNotIn("\n", proof)
        self.assertNotIn(" ", proof)
        document = json.loads(proof)
        self.assertEqual(
            document["format"],
            "branching-city-twin/diagnostic-ledger-proof",
        )
        self.assertEqual(document["version"], 1)
        self.assertEqual(
            canonical(document), proof
        )
        body = {
            key: value
            for key, value in document.items()
            if key != "checksum"
        }
        self.assertEqual(
            document["checksum"],
            hashlib.sha256(
                canonical(body).encode("utf-8")
            ).hexdigest(),
        )

    def test_export_is_deterministic(self) -> None:
        identities = (transaction(2), transaction(5))
        first = BranchStore.export_diagnostic_ledger_proof(
            self.dir, None, None, identities
        )
        second = BranchStore.export_diagnostic_ledger_proof(
            self.dir, None, None, identities
        )
        self.assertEqual(first, second)

    def test_verification_works_with_the_ledger_deleted(self) -> None:
        proof = self.export()
        import shutil

        shutil.rmtree(self.dir)
        result = BranchStore.verify_diagnostic_ledger_proof(proof)
        self.assertEqual(
            [item["at"] for item in result["items"]], [5, 6]
        )
        self.assertEqual(result["rotated"], (transaction(0),))
        self.assertEqual(result["absent"], (transaction(99),))

    def test_returned_levels_are_isolated(self) -> None:
        first = self.verify()
        first["items"][0]["summary"]["transaction"] = "mutated"
        first["snapshot"]["digest"] = "mutated"
        mutable_rotated = list(first["rotated"])
        mutable_rotated.append("mutated")
        second = self.verify()
        self.assertNotEqual(
            second["items"][0]["summary"]["transaction"], "mutated"
        )
        self.assertNotEqual(second["snapshot"]["digest"], "mutated")
        self.assertNotIn("mutated", second["rotated"])
        self.assertIsInstance(second["rotated"], tuple)

    def test_without_rotation_rotated_class_is_always_empty(self):
        directory = os.path.join(self.tmp.name, "full")
        os.mkdir(directory)
        BranchStore.append_diagnostic_ledger(
            directory, self.entries[:4], None, 2, 10, None
        )
        proof = BranchStore.export_diagnostic_ledger_proof(
            directory, None, None,
            (transaction(0), transaction(40)),
        )
        document = json.loads(proof)
        self.assertIsNone(document["evidence"])
        self.assertIsNone(document["snapshot"]["dropped"])
        result = BranchStore.verify_diagnostic_ledger_proof(proof)
        self.assertEqual(result["rotated"], ())
        self.assertEqual(result["absent"], (transaction(40),))


class TimeModeProofTests(ProofTestBase):
    def setUp(self) -> None:
        super().setUp()
        # Four retained segments over times 1..8, nothing rotated:
        # range filtering can omit segments by manifest bounds.
        self.directory = os.path.join(self.tmp.name, "full")
        os.mkdir(self.directory)
        self.digest = BranchStore.append_diagnostic_ledger(
            self.directory, self.entries, None, 2, 10, None
        )

    def export_range(self, start, end):
        return BranchStore.export_diagnostic_ledger_proof(
            self.directory, start, end, None
        )

    def test_inclusive_range_matches_only_intersecting_entries(self):
        result = BranchStore.verify_diagnostic_ledger_proof(
            self.export_range(3, 6)
        )
        self.assertEqual(
            [item["at"] for item in result["items"]], [3, 4, 5, 6]
        )
        self.assertEqual(result["rotated"], ())
        self.assertEqual(result["absent"], ())

    def test_only_intersecting_segments_are_carried(self) -> None:
        proof = self.export_range(5, 6)
        document = json.loads(proof)
        self.assertEqual(
            [segment["name"] for segment in document["segments"]],
            ["segment-000003.json"],
        )
        result = BranchStore.verify_diagnostic_ledger_proof(proof)
        self.assertEqual([item["at"] for item in result["items"]], [5, 6])

    def test_range_partially_overlapping_a_segment_is_complete(self):
        proof = self.export_range(4, 5)
        document = json.loads(proof)
        # Segment 2 covers times 3-4 and segment 3 times 5-6: both
        # intersect and both must be carried.
        self.assertEqual(
            [segment["name"] for segment in document["segments"]],
            ["segment-000002.json", "segment-000003.json"],
        )
        result = BranchStore.verify_diagnostic_ledger_proof(proof)
        self.assertEqual([item["at"] for item in result["items"]], [4, 5])

    def test_range_without_any_entry_carries_no_segment(self) -> None:
        proof = self.export_range(40, 60)
        document = json.loads(proof)
        self.assertEqual(document["segments"], [])
        result = BranchStore.verify_diagnostic_ledger_proof(proof)
        self.assertEqual(result["items"], ())
        self.assertEqual(result["snapshot"]["digest"], self.digest)

    def test_full_range_carries_every_segment(self) -> None:
        proof = self.export_range(0, 100)
        document = json.loads(proof)
        self.assertEqual(
            len(document["segments"]), 4
        )
        result = BranchStore.verify_diagnostic_ledger_proof(proof)
        self.assertEqual(
            [item["at"] for item in result["items"]],
            [1, 2, 3, 4, 5, 6, 7, 8],
        )

    def test_range_touching_rotated_boundary_is_refused(self) -> None:
        rotated_dir = os.path.join(self.tmp.name, "rotated")
        os.mkdir(rotated_dir)
        digest = BranchStore.append_diagnostic_ledger(
            rotated_dir, self.entries[0:2], None, 2, 1, None
        )
        digest = BranchStore.append_diagnostic_ledger(
            rotated_dir, self.entries[2:4], digest, 2, 1, None
        )
        BranchStore.append_diagnostic_ledger(
            rotated_dir, self.entries[4:6], digest, 2, 1, None
        )
        # Prefix covers times 1..4; any intersection must be refused.
        for start, end in ((1, 3), (4, 4), (0, 100), (2, 5)):
            with self.subTest(start=start, end=end):
                with self.assertRaises(ValueError):
                    BranchStore.export_diagnostic_ledger_proof(
                        rotated_dir, start, end, None
                    )
        # A wholly retained range is still fine.
        proof = BranchStore.export_diagnostic_ledger_proof(
            rotated_dir, 5, 6, None
        )
        result = BranchStore.verify_diagnostic_ledger_proof(proof)
        self.assertEqual([item["at"] for item in result["items"]], [5, 6])

    def test_range_starting_after_prefix_is_completely_covered(self):
        # Times 1..4 rotated, retained starts at 5: a range that begins
        # at 5 does not intersect the prefix bounds [1, 4].
        rotated_dir = os.path.join(self.tmp.name, "rotated2")
        os.mkdir(rotated_dir)
        digest = BranchStore.append_diagnostic_ledger(
            rotated_dir, self.entries[0:2], None, 2, 1, None
        )
        digest = BranchStore.append_diagnostic_ledger(
            rotated_dir, self.entries[2:4], digest, 2, 1, None
        )
        BranchStore.append_diagnostic_ledger(
            rotated_dir, self.entries[4:6], digest, 2, 1, None
        )
        proof = BranchStore.export_diagnostic_ledger_proof(
            rotated_dir, 5, 100, None
        )
        result = BranchStore.verify_diagnostic_ledger_proof(proof)
        self.assertEqual([item["at"] for item in result["items"]], [5, 6])


class ProofTamperingTests(ProofTestBase):
    def setUp(self) -> None:
        super().setUp()
        self.digest = self.publish_rotated()
        self.proof_text = self.export()
        self.document = json.loads(self.proof_text)

    def assert_rejected(self, document=None) -> None:
        if document is None:
            document = self.document
        with self.assertRaises(ValueError):
            BranchStore.verify_diagnostic_ledger_proof(canonical(document))

    def test_not_json(self) -> None:
        with self.assertRaises(ValueError):
            BranchStore.verify_diagnostic_ledger_proof("not a proof")

    def test_bom_and_trailing_newline(self) -> None:
        with self.assertRaises(ValueError):
            BranchStore.verify_diagnostic_ledger_proof(
                "\ufeff" + self.proof_text
            )
        with self.assertRaises(ValueError):
            BranchStore.verify_diagnostic_ledger_proof(
                self.proof_text + "\n"
            )

    def test_non_canonical_encoding(self) -> None:
        with self.assertRaises(ValueError):
            BranchStore.verify_diagnostic_ledger_proof(
                json.dumps(self.document, indent=2)
            )

    def test_top_checksum_tampering(self) -> None:
        document = json.loads(self.proof_text)
        document["checksum"] = "f" * 64
        with self.assertRaises(ValueError):
            BranchStore.verify_diagnostic_ledger_proof(canonical(document))

    def test_unknown_format_and_version(self) -> None:
        for tamper in (
            lambda doc: doc.update(format="branching-city-twin/other"),
            lambda doc: doc.update(version=2),
            lambda doc: doc.update(extra=1),
        ):
            document = json.loads(self.proof_text)
            tamper(document)
            self.assert_rejected(document)

    def test_manifest_body_tampering_is_rejected_even_when_resealed(self):
        document = json.loads(self.proof_text)
        document["manifest"]["version"] = 2
        # A forger can only re-seal the outer package; the manifest's
        # own checksum still fails.
        text = reseal_proof(document)
        with self.assertRaises(ValueError):
            BranchStore.verify_diagnostic_ledger_proof(text)

    def test_segment_body_tampering_breaks_the_manifest_binding(self):
        document = json.loads(self.proof_text)
        segment = document["segments"][0]
        raw = canonical(segment["document"]).encode("utf-8")
        parsed = json.loads(raw.decode("utf-8"))
        parsed["index"] = 7
        # Re-seal the segment's own checksum so that check passes: its
        # digest then differs from the manifest digest.
        segment["document"] = json.loads(reseal(parsed))
        with self.assertRaises(ValueError):
            BranchStore.verify_diagnostic_ledger_proof(
                reseal_proof(document)
            )

    def test_broken_segment_predecessor_link_is_rejected(self) -> None:
        # Build a second rotated ledger whose retained segment chains
        # from a different prefix, then graft it into the proof.
        other_dir = os.path.join(self.tmp.name, "other")
        os.mkdir(other_dir)
        other_chains = [make_chain(transaction(100 + i)) for i in range(6)]
        other_entries = tuple(
            (time, chain)
            for time, chain in zip(range(1, 7), other_chains)
        )
        digest = BranchStore.append_diagnostic_ledger(
            other_dir, other_entries[0:2], None, 2, 1, None
        )
        digest = BranchStore.append_diagnostic_ledger(
            other_dir, other_entries[2:4], digest, 2, 1, None
        )
        BranchStore.append_diagnostic_ledger(
            other_dir, other_entries[4:6], digest, 2, 1, None
        )
        other_proof = json.loads(
            BranchStore.export_diagnostic_ledger_proof(
                other_dir, None, None,
                (transaction(104), transaction(0)),
            )
        )
        document = json.loads(self.proof_text)
        document["segments"] = other_proof["segments"]
        with self.assertRaises(ValueError):
            BranchStore.verify_diagnostic_ledger_proof(
                reseal_proof(document)
            )

    def test_missing_required_segment_is_rejected(self) -> None:
        document = json.loads(self.proof_text)
        document["segments"] = []
        with self.assertRaises(ValueError):
            BranchStore.verify_diagnostic_ledger_proof(
                reseal_proof(document)
            )

    def test_extra_segment_is_rejected(self) -> None:
        # Time-mode proof that legitimately omits segments; graft the
        # omitted segment body from a full-range proof and reseal.
        directory = os.path.join(self.tmp.name, "full")
        os.mkdir(directory)
        BranchStore.append_diagnostic_ledger(
            directory, self.entries, None, 2, 10, None
        )
        narrow = json.loads(
            BranchStore.export_diagnostic_ledger_proof(
                directory, 7, 8, None
            )
        )
        broad = json.loads(
            BranchStore.export_diagnostic_ledger_proof(
                directory, 1, 8, None
            )
        )
        self.assertEqual(
            [s["name"] for s in narrow["segments"]],
            ["segment-000004.json"],
        )
        narrow["segments"] = list(narrow["segments"]) + [
            segment
            for segment in broad["segments"]
            if segment["name"] == "segment-000001.json"
        ]
        with self.assertRaises(ValueError):
            BranchStore.verify_diagnostic_ledger_proof(
                reseal_proof(narrow)
            )

    def test_cross_snapshot_splice_is_rejected(self) -> None:
        before = json.loads(self.proof_text)
        self.append(self.entries[6:8], self.digest)
        after = json.loads(self.export())
        # Manifest and snapshot from the new generation, segment bodies
        # and evidence from the old one.
        before["manifest"] = after["manifest"]
        before["snapshot"] = after["snapshot"]
        with self.assertRaises(ValueError):
            BranchStore.verify_diagnostic_ledger_proof(
                reseal_proof(before)
            )

    def test_snapshot_rewrite_is_rejected(self) -> None:
        document = json.loads(self.proof_text)
        document["snapshot"]["entries"] = 99
        with self.assertRaises(ValueError):
            BranchStore.verify_diagnostic_ledger_proof(
                reseal_proof(document)
            )

    def test_filter_widening_into_the_rotated_prefix_is_rejected(self):
        # A time-mode proof exported wholly inside the retained range;
        # widen the forged filter into the prefix and reseal.
        directory = os.path.join(self.tmp.name, "timed")
        os.mkdir(directory)
        digest = BranchStore.append_diagnostic_ledger(
            directory, self.entries[0:2], None, 2, 1, None
        )
        digest = BranchStore.append_diagnostic_ledger(
            directory, self.entries[2:4], digest, 2, 1, None
        )
        BranchStore.append_diagnostic_ledger(
            directory, self.entries[4:6], digest, 2, 1, None
        )
        document = json.loads(
            BranchStore.export_diagnostic_ledger_proof(
                directory, 5, 6, None
            )
        )
        document["filter"] = {"mode": "time", "start": 1, "end": 6}
        with self.assertRaises(ValueError):
            BranchStore.verify_diagnostic_ledger_proof(
                reseal_proof(document)
            )

    def test_filter_mode_switch_is_rejected(self) -> None:
        # The retained segment covers times 5-6; switching the filter to
        # a time range inside the rotated prefix demands segments the
        # transaction-mode proof does not carry.
        document = json.loads(self.proof_text)
        document["filter"] = {"mode": "time", "start": 1, "end": 2}
        with self.assertRaises(ValueError):
            BranchStore.verify_diagnostic_ledger_proof(
                reseal_proof(document)
            )

    def test_unknown_filter_mode_is_rejected(self) -> None:
        document = json.loads(self.proof_text)
        document["filter"] = {"mode": "mystery"}
        with self.assertRaises(ValueError):
            BranchStore.verify_diagnostic_ledger_proof(
                reseal_proof(document)
            )

    def test_identity_in_two_classes_is_rejected(self) -> None:
        # Forge the prefix evidence to also remember a retained
        # identity, then rewire the evidence pointer through the
        # manifest and snapshot and re-seal every layer. Verification is
        # offline, so only the embedded documents need to agree.
        document = json.loads(self.proof_text)
        retained_identity = transaction(4)
        evidence = document["evidence"]
        evidence["transactions"] = list(evidence["transactions"]) + [
            retained_identity
        ]
        evidence["entries"] += 1
        evidence_text = reseal(evidence)
        evidence_obj = json.loads(evidence_text)
        evidence_digest = hashlib.sha256(
            evidence_text.encode("utf-8")
        ).hexdigest()
        manifest = document["manifest"]
        manifest["evidence"] = {"digest": evidence_digest}
        manifest_text = reseal(manifest)
        manifest_obj = json.loads(manifest_text)
        document["evidence"] = evidence_obj
        document["manifest"] = manifest_obj
        document["snapshot"] = BranchStore._diagnostic_ledger_proof_snapshot(
            manifest_text.encode("utf-8"),
            manifest_obj,
            manifest_obj["segments"],
            evidence_digest,
        )
        # The proof requests the retained-but-forged-rotated identity.
        document["filter"] = {
            "mode": "transactions",
            "transactions": [retained_identity],
        }
        with self.assertRaises(ValueError):
            BranchStore.verify_diagnostic_ledger_proof(
                reseal_proof(document)
            )

    def test_duplicate_filter_identity_is_rejected(self) -> None:
        document = json.loads(self.proof_text)
        document["filter"] = {
            "mode": "transactions",
            "transactions": [transaction(0), transaction(0)],
        }
        with self.assertRaises(ValueError):
            BranchStore.verify_diagnostic_ledger_proof(
                reseal_proof(document)
            )

    def test_evidence_without_dropped_prefix_is_rejected(self) -> None:
        directory = os.path.join(self.tmp.name, "plain")
        os.mkdir(directory)
        BranchStore.append_diagnostic_ledger(
            directory, self.entries[:2], None, 2, 10, None
        )
        document = json.loads(
            BranchStore.export_diagnostic_ledger_proof(
                directory, None, None, (transaction(0),)
            )
        )
        self.assertIsNone(document["evidence"])
        document["evidence"] = json.loads(self.proof_text)["evidence"]
        with self.assertRaises(ValueError):
            BranchStore.verify_diagnostic_ledger_proof(
                reseal_proof(document)
            )


class ProofLockingTests(ProofTestBase):
    def test_export_waits_for_an_exclusive_publish_lock(self) -> None:
        digest = self.publish_rotated()
        lock_fd = os.open(self.lock_path(), os.O_RDWR)
        BranchStore._acquire_chain_lock(lock_fd, None)
        outcome = {}

        def export():
            try:
                outcome["proof"] = self.export()
            except BaseException as exc:  # pragma: no cover - diagnostic
                outcome["error"] = exc

        thread = threading.Thread(target=export)
        thread.start()
        thread.join(0.5)
        self.assertTrue(thread.is_alive())
        BranchStore._release_chain_lock(lock_fd)
        os.close(lock_fd)
        thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertNotIn("error", outcome)
        result = BranchStore.verify_diagnostic_ledger_proof(
            outcome["proof"]
        )
        self.assertEqual(result["snapshot"]["digest"], digest)

    def test_publisher_waits_for_an_active_export(self) -> None:
        digest = self.append(self.entries[0:4], None, keep=2)
        lock_fd = os.open(self.lock_path(), os.O_RDONLY)
        BranchStore._acquire_diagnostic_ledger_reader_lock(lock_fd)
        outcome = {}

        def publish():
            outcome["digest"] = self.append(
                self.entries[4:6], digest, keep=2
            )

        thread = threading.Thread(target=publish)
        thread.start()
        thread.join(0.5)
        self.assertTrue(thread.is_alive())
        # While the export's read lock is held, the rotating publish has
        # not happened: the old segment survives and a parallel export
        # still sees the pre-publish snapshot.
        self.assertTrue(
            os.path.exists(os.path.join(self.dir, "segment-000001.json"))
        )
        result = BranchStore.verify_diagnostic_ledger_proof(self.export())
        self.assertEqual(result["snapshot"]["digest"], digest)
        os.close(lock_fd)
        thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertIn("digest", outcome)

    def test_export_creates_no_files(self) -> None:
        self.publish_rotated()
        before = sorted(os.listdir(self.dir))
        self.export()
        self.assertEqual(sorted(os.listdir(self.dir)), before)

    def test_export_is_strictly_read_only_across_repeats(self) -> None:
        digest = self.publish_rotated()
        before = {
            name: read_bytes(os.path.join(self.dir, name))
            for name in os.listdir(self.dir)
            if os.path.isfile(os.path.join(self.dir, name))
        }
        for _ in range(3):
            self.export()
            self.export(start=5, end=6, transactions=None)
        after = {
            name: read_bytes(os.path.join(self.dir, name))
            for name in os.listdir(self.dir)
            if os.path.isfile(os.path.join(self.dir, name))
        }
        self.assertEqual(before, after)
        page = BranchStore.page_diagnostic_ledger(
            self.dir, None, None, None, 100, None
        )
        self.assertEqual(page["snapshot"]["digest"], digest)


class BaselineBehaviourTests(ProofTestBase):
    def test_paging_and_append_still_work(self) -> None:
        digest = self.publish_rotated()
        page = BranchStore.page_diagnostic_ledger(
            self.dir, None, None, None, 100, None
        )
        self.assertEqual(page["snapshot"]["digest"], digest)
        self.assertEqual(
            [item["at"] for item in page["items"]], [5, 6]
        )
        fresh = make_chain(transaction(40))
        new_digest = self.append(((9, fresh),), digest)
        self.assertNotEqual(new_digest, digest)

    def test_single_file_ledger_paging_is_unchanged(self) -> None:
        path = os.path.join(self.tmp.name, "ledger.json")
        BranchStore.save_diagnostic_ledger(path, self.entries[:3])
        page = BranchStore.page_diagnostic_ledger(
            path, None, None, None, 100, None
        )
        self.assertEqual(
            [item["at"] for item in page["items"]], [1, 2, 3]
        )


if __name__ == "__main__":
    unittest.main()
