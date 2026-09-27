"""Tests for offline-verifiable diagnostic ledger proofs.

``BranchStore.export_diagnostic_ledger_proof`` exports a canonical,
checksummed proof string from a segmented directory diagnostic ledger,
and ``BranchStore.verify_diagnostic_ledger_proof`` authenticates that
string with no access to the original ledger directory.

These tests cover:

* selection validation: exactly one of an inclusive non-negative
  non-``bool`` int time range and a non-empty tuple of distinct
  non-empty transaction identities may be used; bad types, reversed or
  negative bounds, empty/duplicate identities and mixed/empty modes
  raise as specified;
* transaction mode: every requested identity is classified exactly as
  retained, rotated away (prefix evidence) or never seen, identities
  follow the request order while items follow ledger order;
* time mode: inclusive filtering, only intersecting retained segments
  are bound, a range disjoint from a rotated prefix still verifies, and
  a range intersecting the prefix's remembered time bounds raises
  ``ValueError`` because per-entry prefix times are not retained;
* offline verification: deleting the ledger directory after export
  changes nothing, and the returned mapping is isolated;
* tampering: checksum edits, resealed manifest/snapshot/selection/
  item/classification edits, missing or extra segments, cross-snapshot
  splices and forged prefix evidence all raise ``ValueError``;
* coordination: exports take the same shared reader lock directory
  pages use, so they block an exclusive publish and observe only a
  complete pre-/post-publish snapshot, and create no files.
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


def identity(chain) -> str:
    return json.loads(chain[0])["transaction"]


class ProofTestBase(unittest.TestCase):
    keep = 2

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = os.path.join(self.tmp.name, "ledger")
        os.mkdir(self.dir)
        self.chains = [make_chain(transaction(i)) for i in range(8)]
        self.entries = tuple(
            (at, chain)
            for at, chain in zip(range(1, 9), self.chains)
        )
        self.identities = [identity(chain) for chain in self.chains]
        digest = None
        for chunk in (
            self.entries[0:2],
            self.entries[2:4],
            self.entries[4:6],
            self.entries[6:8],
        ):
            digest = BranchStore.append_diagnostic_ledger(
                self.dir, chunk, digest, 2, self.keep, None
            )
        self.digest = digest

    def export(self, start=None, end=None, transactions=None):
        return BranchStore.export_diagnostic_ledger_proof(
            self.dir, start, end, transactions
        )

    def proof_document(self, proof=None):
        if proof is None:
            proof = self.export(transactions=tuple(self.identities))
        return json.loads(proof)

    def reseal_proof(self, document: dict) -> str:
        return reseal(document)


class NoRotationProofTestBase(ProofTestBase):
    keep = 10


class ExportValidationTests(ProofTestBase):
    def test_directory_validation(self) -> None:
        ids = (self.identities[0],)
        for bad in (1, 1.5, b"x", ["x"], {"x": 1}, object(), None):
            with self.subTest(bad=type(bad).__name__):
                with self.assertRaises(TypeError):
                    BranchStore.export_diagnostic_ledger_proof(
                        bad, None, None, ids
                    )
        with self.assertRaises(ValueError):
            BranchStore.export_diagnostic_ledger_proof("", None, None, ids)

    def test_missing_directory_raises_oserror(self) -> None:
        missing = os.path.join(self.tmp.name, "nope")
        with self.assertRaises(OSError):
            BranchStore.export_diagnostic_ledger_proof(
                missing, None, None, (self.identities[0],)
            )

    def test_directory_without_lock_or_manifest_raises_oserror(self) -> None:
        empty = os.path.join(self.tmp.name, "empty")
        os.mkdir(empty)
        with self.assertRaises(OSError):
            BranchStore.export_diagnostic_ledger_proof(
                empty, None, None, (self.identities[0],)
            )

    def test_bound_types(self) -> None:
        for bad in (True, False, 1.5, "1", b"1", []):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    self.export(start=bad, end=10)
                with self.assertRaises(TypeError):
                    self.export(start=0, end=bad)

    def test_bounds_must_be_pair_or_both_none(self) -> None:
        with self.assertRaises(ValueError):
            self.export(start=1, end=None)
        with self.assertRaises(ValueError):
            self.export(start=None, end=1)

    def test_bounds_must_be_ordered_and_non_negative(self) -> None:
        with self.assertRaises(ValueError):
            self.export(start=4, end=3)
        with self.assertRaises(ValueError):
            self.export(start=-1, end=3)
        with self.assertRaises(ValueError):
            self.export(start=0, end=-1)

    def test_transactions_container_and_elements(self) -> None:
        for bad in (["x"], {self.identities[0]}, "x", 1, b"x"):
            with self.subTest(bad=type(bad).__name__):
                with self.assertRaises(TypeError):
                    self.export(transactions=bad)
        with self.assertRaises(TypeError):
            self.export(transactions=(1,))
        with self.assertRaises(ValueError):
            self.export(transactions=())
        with self.assertRaises(ValueError):
            self.export(transactions=("",))
        repeated = self.identities[0]
        with self.assertRaises(ValueError):
            self.export(transactions=(repeated, repeated))

    def test_modes_are_exclusive_and_required(self) -> None:
        with self.assertRaises(ValueError):
            self.export(start=1, end=2,
                        transactions=(self.identities[0],))
        with self.assertRaises(ValueError):
            self.export(start=None, end=None, transactions=None)

    def test_verify_argument_validation(self) -> None:
        proof = self.export(start=5, end=8)
        for bad in (1, 1.5, b"x", [], {}, None, object()):
            with self.subTest(bad=type(bad).__name__):
                with self.assertRaises(TypeError):
                    BranchStore.verify_diagnostic_ledger_proof(bad)
        with self.assertRaises(ValueError):
            BranchStore.verify_diagnostic_ledger_proof("")
        # A genuine proof verifies.
        result = BranchStore.verify_diagnostic_ledger_proof(proof)
        self.assertEqual([item["at"] for item in result["items"]], [5, 6, 7, 8])


class TransactionModeTests(ProofTestBase):
    def test_three_way_classification_in_request_order(self) -> None:
        # keep=2: retained identities are chains 4..7; rotated 0..3.
        requested = (
            self.identities[6],  # retained
            self.identities[1],  # rotated
            self.identities[0],  # rotated
            "never-seen-identity",  # absent
            self.identities[4],  # retained
        )
        proof = self.export(transactions=requested)
        result = BranchStore.verify_diagnostic_ledger_proof(proof)
        self.assertEqual(list(result), ["snapshot", "items", "rotated", "absent"])
        self.assertEqual(
            result["rotated"],
            (self.identities[1], self.identities[0]),
        )
        self.assertEqual(result["absent"], ("never-seen-identity",))
        # Items follow the ledger's (time, transaction) order, not the
        # request order: identity 4 at time 5 precedes identity 6 at 7.
        self.assertEqual(
            [item["at"] for item in result["items"]], [5, 7]
        )
        self.assertEqual(
            [item["summary"]["transaction"] for item in result["items"]],
            [self.identities[4], self.identities[6]],
        )

    def test_every_requested_identity_falls_into_exactly_one_class(self) -> None:
        requested = tuple(reversed(self.identities)) + ("missing",)
        result = BranchStore.verify_diagnostic_ledger_proof(
            self.export(transactions=requested)
        )
        self.assertEqual(
            result["rotated"], tuple(reversed(self.identities[:4]))
        )
        self.assertEqual(result["absent"], ("missing",))
        classified = (
            {item["summary"]["transaction"] for item in result["items"]}
            | set(result["rotated"])
            | set(result["absent"])
        )
        self.assertEqual(classified, set(requested))

    def test_transaction_mode_binds_every_retained_segment(self) -> None:
        document = self.proof_document()
        self.assertEqual(
            [block["name"] for block in document["segments"]],
            ["segment-000003.json", "segment-000004.json"],
        )

    def test_proof_carries_evidence_and_snapshot_summary(self) -> None:
        document = self.proof_document()
        self.assertIsNotNone(document["evidence"])
        snapshot = document["snapshot"]
        self.assertEqual(
            list(snapshot),
            [
                "digest",
                "entries",
                "bytes",
                "dropped_entries",
                "dropped_first_at",
                "dropped_last_at",
                "evidence_digest",
            ],
        )
        self.assertEqual(snapshot["dropped_entries"], 4)
        self.assertEqual(
            (snapshot["dropped_first_at"], snapshot["dropped_last_at"]),
            (1, 4),
        )
        self.assertEqual(snapshot["entries"], 4)
        self.assertEqual(
            snapshot["digest"],
            hashlib.sha256(
                read_bytes(os.path.join(self.dir, "manifest.json"))
            ).hexdigest(),
        )

    def test_identities_requested_but_absent_need_no_entries(self) -> None:
        result = BranchStore.verify_diagnostic_ledger_proof(
            self.export(transactions=("ghost",))
        )
        self.assertEqual(result["items"], ())
        self.assertEqual(result["rotated"], ())
        self.assertEqual(result["absent"], ("ghost",))


class TransactionModeNoRotationTests(NoRotationProofTestBase):
    def test_no_rotation_means_no_evidence_block(self) -> None:
        document = self.proof_document()
        self.assertIsNone(document["evidence"])
        self.assertIsNone(document["snapshot"]["dropped_entries"])
        self.assertIsNone(document["snapshot"]["evidence_digest"])

    def test_all_retained_classification(self) -> None:
        requested = (self.identities[2], self.identities[7], "ghost")
        result = BranchStore.verify_diagnostic_ledger_proof(
            self.export(transactions=requested)
        )
        self.assertEqual(result["rotated"], ())
        self.assertEqual(result["absent"], ("ghost",))
        self.assertEqual(
            [item["at"] for item in result["items"]], [3, 8]
        )

    def test_empty_ledger_classifies_everything_absent(self) -> None:
        empty_dir = os.path.join(self.tmp.name, "empty-ledger")
        os.mkdir(empty_dir)
        BranchStore.append_diagnostic_ledger(
            empty_dir, (), None, 2, 10, None
        )
        proof = BranchStore.export_diagnostic_ledger_proof(
            empty_dir, None, None, ("ghost",)
        )
        result = BranchStore.verify_diagnostic_ledger_proof(proof)
        self.assertEqual(result["items"], ())
        self.assertEqual(result["rotated"], ())
        self.assertEqual(result["absent"], ("ghost",))
        self.assertEqual(result["snapshot"]["entries"], 0)


class TimeModeTests(ProofTestBase):
    def test_inclusive_filtering_and_result_shape(self) -> None:
        result = BranchStore.verify_diagnostic_ledger_proof(
            self.export(start=5, end=7)
        )
        self.assertEqual([item["at"] for item in result["items"]], [5, 6, 7])
        self.assertEqual(result["rotated"], ())
        self.assertEqual(result["absent"], ())
        for item in result["items"]:
            self.assertEqual(list(item), ["at", "summary", "records"])
            self.assertIsInstance(item["records"], tuple)

    def test_only_intersecting_segments_are_bound(self) -> None:
        document = json.loads(self.export(start=8, end=8))
        self.assertEqual(
            [block["name"] for block in document["segments"]],
            ["segment-000004.json"],
        )
        document = json.loads(self.export(start=5, end=5))
        self.assertEqual(
            [block["name"] for block in document["segments"]],
            ["segment-000003.json"],
        )
        document = json.loads(self.export(start=6, end=7))
        self.assertEqual(
            [block["name"] for block in document["segments"]],
            ["segment-000003.json", "segment-000004.json"],
        )

    def test_range_without_intersecting_segments_binds_none(self) -> None:
        document = json.loads(self.export(start=100, end=200))
        self.assertEqual(document["segments"], [])
        self.assertEqual(document["items"], [])
        # Still verifies offline: no retained record can sit in [100,200].
        result = BranchStore.verify_diagnostic_ledger_proof(
            self.reseal_proof(document)
        )
        self.assertEqual(result["items"], ())

    def test_range_strictly_before_a_rotated_prefix_is_complete(self) -> None:
        # Dropped prefix spans times 1..4; [0,0] is disjoint and empty.
        result = BranchStore.verify_diagnostic_ledger_proof(
            self.export(start=0, end=0)
        )
        self.assertEqual(result["items"], ())

    def test_range_strictly_after_a_rotated_prefix_is_complete(self) -> None:
        # start = last_at + 1 is the first provably disjoint point.
        result = BranchStore.verify_diagnostic_ledger_proof(
            self.export(start=5, end=100)
        )
        self.assertEqual(
            [item["at"] for item in result["items"]], [5, 6, 7, 8]
        )

    def test_range_touching_prefix_bounds_is_rejected(self) -> None:
        # The intersection is inclusive: end == prefix first_at and
        # start == prefix last_at both intersect.
        with self.assertRaises(ValueError):
            self.export(start=1, end=4)
        with self.assertRaises(ValueError):
            self.export(start=4, end=4)
        with self.assertRaises(ValueError):
            self.export(start=4, end=10)


class TimeModeNoRotationTests(NoRotationProofTestBase):
    def test_middle_range_binds_only_middle_segments(self) -> None:
        document = json.loads(self.export(start=3, end=6))
        self.assertEqual(
            [block["name"] for block in document["segments"]],
            ["segment-000002.json", "segment-000003.json"],
        )
        result = BranchStore.verify_diagnostic_ledger_proof(
            self.reseal_proof(document)
        )
        self.assertEqual(
            [item["at"] for item in result["items"]], [3, 4, 5, 6]
        )

    def test_full_range_binds_every_segment(self) -> None:
        document = json.loads(self.export(start=0, end=100))
        self.assertEqual(
            [block["name"] for block in document["segments"]],
            [f"segment-{index:06d}.json" for index in range(1, 5)],
        )


class OfflineAndIsolationTests(ProofTestBase):
    def test_verify_never_opens_the_ledger_directory(self) -> None:
        proof = self.export(
            transactions=(self.identities[0], self.identities[4], "ghost")
        )
        import shutil

        shutil.rmtree(self.dir)
        result = BranchStore.verify_diagnostic_ledger_proof(proof)
        self.assertEqual(
            [item["summary"]["transaction"] for item in result["items"]],
            [self.identities[4]],
        )
        self.assertEqual(result["rotated"], (self.identities[0],))
        self.assertEqual(result["absent"], ("ghost",))

    def test_returned_levels_are_isolated(self) -> None:
        proof = self.export(transactions=tuple(self.identities))
        first = BranchStore.verify_diagnostic_ledger_proof(proof)
        rotated_before = first["rotated"]
        first["items"][0]["summary"]["transaction"] = "mutated"
        first["snapshot"]["digest"] = "mutated"
        # The classification levels are immutable tuples.
        with self.assertRaises(AttributeError):
            first["rotated"].__setitem__(0, "mutated")  # type: ignore[attr-defined]
        second = BranchStore.verify_diagnostic_ledger_proof(proof)
        self.assertNotEqual(
            second["items"][0]["summary"]["transaction"], "mutated"
        )
        self.assertNotEqual(second["snapshot"]["digest"], "mutated")
        self.assertEqual(second["rotated"], rotated_before)

    def test_proof_is_canonical_compact_json(self) -> None:
        proof = self.export(start=5, end=8)
        self.assertFalse(proof.startswith("﻿"))
        self.assertFalse(proof.endswith("\n"))
        document = json.loads(proof)
        self.assertEqual(
            BranchStore._canonical_json(document), proof
        )

    def test_export_is_read_only(self) -> None:
        before = {
            name: read_bytes(os.path.join(self.dir, name))
            for name in os.listdir(self.dir)
            if os.path.isfile(os.path.join(self.dir, name))
        }
        self.export(transactions=tuple(self.identities))
        self.export(start=5, end=8)
        after = {
            name: read_bytes(os.path.join(self.dir, name))
            for name in os.listdir(self.dir)
            if os.path.isfile(os.path.join(self.dir, name))
        }
        self.assertEqual(before, after)


class ProofEnvelopeTests(ProofTestBase):
    def test_raw_envelope_defects(self) -> None:
        proof = self.export(start=5, end=8)
        for bad in (
            "not json",
            "﻿" + proof,
            proof + "\n",
        ):
            with self.subTest(bad=bad[:12]):
                with self.assertRaises(ValueError):
                    BranchStore.verify_diagnostic_ledger_proof(bad)

    def test_format_and_version(self) -> None:
        document = self.proof_document(
            self.export(transactions=(self.identities[4],))
        )
        self.assertEqual(
            document["format"],
            "branching-city-twin/diagnostic-ledger-proof",
        )
        self.assertEqual(document["version"], 1)
        for tamper in (
            lambda doc: doc.update(format="branching-city-twin/other"),
            lambda doc: doc.update(version=2),
            lambda doc: doc.update(extra=1),
        ):
            tampered = json.loads(json.dumps(document))
            tamper(tampered)
            with self.assertRaises(ValueError):
                BranchStore.verify_diagnostic_ledger_proof(
                    self.reseal_proof(tampered)
                )

    def test_checksum_bit_flip_is_detected(self) -> None:
        proof = self.export(start=5, end=8)
        document = json.loads(proof)
        document["checksum"] = "f" * 64
        with self.assertRaises(ValueError):
            BranchStore.verify_diagnostic_ledger_proof(
                BranchStore._canonical_json(document)
            )


class ProofTamperTests(ProofTestBase):
    def setUp(self) -> None:
        super().setUp()
        self.proof = self.export(
            transactions=tuple(self.identities) + ("ghost",)
        )
        self.document = json.loads(self.proof)

    def verify_resealed(self, document=None) -> None:
        document = self.document if document is None else document
        with self.assertRaises(ValueError):
            BranchStore.verify_diagnostic_ledger_proof(
                self.reseal_proof(document)
            )

    def test_snapshot_digest_change_is_detected(self) -> None:
        self.document["snapshot"]["digest"] = "f" * 64
        self.verify_resealed()

    def test_snapshot_prefix_summary_change_is_detected(self) -> None:
        self.document["snapshot"]["dropped_entries"] = 99
        self.verify_resealed()

    def test_selection_edit_changes_needed_segments(self) -> None:
        # Claiming an empty time selection while holding transaction
        # mode's all-segments bundle breaks the binding.
        self.document["selection"]["mode"] = "time"
        self.document["selection"]["start"] = 8
        self.document["selection"]["end"] = 8
        self.document["selection"]["transactions"] = None
        self.verify_resealed()

    def test_manifest_checksum_tamper_is_detected(self) -> None:
        self.document["manifest"]["version"] = 2
        # Not even re-sealing the manifest: its own checksum fails.
        with self.assertRaises(ValueError):
            BranchStore.verify_diagnostic_ledger_proof(
                self.reseal_proof(self.document)
            )

    def test_items_list_omission_is_detected(self) -> None:
        del self.document["items"][0]
        self.verify_resealed()

    def test_extra_item_is_detected(self) -> None:
        forged = dict(self.document["items"][0])
        forged["at"] = forged["at"] + 100
        self.document["items"].append(forged)
        self.verify_resealed()

    def test_item_record_tamper_is_detected(self) -> None:
        chain = self.document["items"][0]["records"]
        record = json.loads(chain[0])
        record["reason"] = "forged"
        chain[0] = reseal(record)
        self.verify_resealed()

    def test_item_summary_tamper_is_detected(self) -> None:
        self.document["items"][0]["summary"]["count"] = 999
        self.verify_resealed()

    def test_reordered_items_are_detected(self) -> None:
        items = self.document["items"]
        if len(items) >= 2:
            items[0], items[1] = items[1], items[0]
            self.verify_resealed()

    def test_rotated_classification_edit_is_detected(self) -> None:
        rotated = self.document["rotated"]
        self.assertTrue(rotated)
        rotated[0], rotated[1] = rotated[1], rotated[0]
        self.verify_resealed()

    def test_absent_classification_edit_is_detected(self) -> None:
        self.document["absent"] = ["someone-else"]
        self.verify_resealed()

    def test_identity_in_two_classes_is_detected(self) -> None:
        retained = self.document["items"][0]["summary"]["transaction"]
        self.document["rotated"].append(retained)
        self.verify_resealed()

    def test_missing_needed_segment_is_detected(self) -> None:
        names = [block["name"] for block in self.document["segments"]]
        self.assertEqual(len(names), 2)
        del self.document["segments"][0]
        self.verify_resealed()

    def test_extra_segment_block_is_detected(self) -> None:
        duplicate = json.loads(json.dumps(self.document["segments"][0]))
        self.document["segments"].append(duplicate)
        self.verify_resealed()

    def test_segment_checksum_tamper_is_detected(self) -> None:
        self.document["segments"][0]["body"]["checksum"] = "f" * 64
        self.verify_resealed()

    def test_segment_predecessor_reseal_is_detected(self) -> None:
        # A sophisticated forger recomputes the segment checksum after
        # rewiring the predecessor; the manifest digest binding catches
        # the resulting content digest.
        body = self.document["segments"][0]["body"]
        body["prev"] = "f" * 64
        self.document["segments"][0]["body"] = json.loads(reseal(body))
        self.verify_resealed()

    def test_evidence_identity_tamper_is_detected(self) -> None:
        evidence_body = self.document["evidence"]["body"]
        evidence_body["transactions"][0] = "f" * 64
        self.verify_resealed()

    def test_evidence_block_dropped_is_detected(self) -> None:
        self.document["evidence"] = None
        self.verify_resealed()

    def test_extra_evidence_without_prefix_is_detected(self) -> None:
        # Time-mode proof of a prefix-free point still carries the
        # evidence block, which must not be claimable as prefix-free.
        time_document = json.loads(self.export(start=5, end=8))
        self.assertIsNotNone(time_document["evidence"])
        time_document["snapshot"]["dropped_entries"] = None
        time_document["snapshot"]["dropped_first_at"] = None
        time_document["snapshot"]["dropped_last_at"] = None
        time_document["snapshot"]["evidence_digest"] = None
        with self.assertRaises(ValueError):
            BranchStore.verify_diagnostic_ledger_proof(
                self.reseal_proof(time_document)
            )


class CrossSnapshotTests(ProofTestBase):
    def test_segment_spliced_from_another_snapshot_is_detected(self) -> None:
        other_dir = os.path.join(self.tmp.name, "other")
        os.mkdir(other_dir)
        other_chains = [make_chain(transaction(100 + i)) for i in range(8)]
        other_entries = tuple(
            (at, chain)
            for at, chain in zip(range(1, 9), other_chains)
        )
        digest = None
        for chunk in (
            other_entries[0:2],
            other_entries[2:4],
            other_entries[4:6],
            other_entries[6:8],
        ):
            digest = BranchStore.append_diagnostic_ledger(
                other_dir, chunk, digest, 2, 2, None
            )
        document_a = json.loads(
            self.export(transactions=tuple(self.identities))
        )
        document_b = json.loads(
            BranchStore.export_diagnostic_ledger_proof(
                other_dir, None, None,
                tuple(json.loads(c[0])["transaction"] for c in other_chains),
            )
        )
        # Same segment names exist in both manifests, but the bytes are
        # bound to different digests: splice one across.
        spliced = json.loads(json.dumps(document_a))
        spliced["segments"][0] = document_b["segments"][0]
        with self.assertRaises(ValueError):
            BranchStore.verify_diagnostic_ledger_proof(
                self.reseal_proof(spliced)
            )

    def test_manifest_spliced_from_another_snapshot_is_detected(self) -> None:
        # Capture a proof against the current snapshot, then append so
        # the visible manifest moves to a new digest.
        earlier_document = json.loads(
            self.export(transactions=tuple(self.identities))
        )
        new_chain = make_chain(transaction(200))
        BranchStore.append_diagnostic_ledger(
            self.dir, ((9, new_chain),), self.digest, 2, 2, None
        )
        later_document = json.loads(
            self.export(
                transactions=tuple(self.identities) + (transaction(200),)
            )
        )
        # Splice the later manifest into the earlier proof bundle: the
        # snapshot digest and segment digests no longer agree.
        earlier_document["manifest"] = later_document["manifest"]
        with self.assertRaises(ValueError):
            BranchStore.verify_diagnostic_ledger_proof(
                self.reseal_proof(earlier_document)
            )


class ProofCoordinationTests(ProofTestBase):
    def test_export_blocks_waits_for_an_active_exporter_writer(self) -> None:
        lock_path = os.path.join(
            os.path.realpath(self.dir),
            BranchStore._DIAGNOSTIC_LEDGER_LOCK_NAME,
        )
        rfd = os.open(lock_path, os.O_RDONLY)
        BranchStore._acquire_diagnostic_ledger_reader_lock(rfd)
        outcome: dict[str, object] = {}

        def publish() -> None:
            try:
                new_chain = make_chain(transaction(99))
                outcome["digest"] = BranchStore.append_diagnostic_ledger(
                    self.dir, ((9, new_chain),), self.digest, 2, 2, None
                )
            except BaseException as exc:  # pragma: no cover - diagnostic
                outcome["error"] = exc

        thread = threading.Thread(target=publish)
        thread.start()
        thread.join(0.5)
        self.assertTrue(thread.is_alive())
        # While the export-style shared lock is held, an export itself
        # still completes (shared locks are compatible) and sees the old
        # snapshot.
        proof = self.export(start=5, end=8)
        result = BranchStore.verify_diagnostic_ledger_proof(proof)
        self.assertEqual(
            [item["at"] for item in result["items"]], [5, 6, 7, 8]
        )
        os.close(rfd)
        thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertIn("digest", outcome)

    def test_export_waits_behind_an_exclusive_publish_lock(self) -> None:
        lock_path = os.path.join(
            os.path.realpath(self.dir),
            BranchStore._DIAGNOSTIC_LEDGER_LOCK_NAME,
        )
        fd = os.open(lock_path, os.O_RDWR)
        self.assertTrue(BranchStore._try_acquire_chain_lock(fd))
        outcome: dict[str, object] = {}

        def export() -> None:
            try:
                outcome["proof"] = self.export(start=5, end=8)
            except BaseException as exc:  # pragma: no cover - diagnostic
                outcome["error"] = exc

        thread = threading.Thread(target=export)
        thread.start()
        thread.join(0.5)
        self.assertTrue(thread.is_alive())
        BranchStore._release_chain_lock(fd)
        os.close(fd)
        thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertIn("proof", outcome)

    def test_export_observes_one_complete_snapshot(self) -> None:
        # Hammer concurrent exports and publishes; every proof must
        # verify on its own and report one coherent generation.
        stop = threading.Event()
        proofs: list[str] = []
        proofs_lock = threading.Lock()

        def exporter() -> None:
            while not stop.is_set():
                proof = self.export(
                    transactions=(self.identities[4], self.identities[6])
                )
                with proofs_lock:
                    proofs.append(proof)

        threads = [threading.Thread(target=exporter) for _ in range(2)]
        for thread in threads:
            thread.start()
        digest = self.digest
        for tag in range(3):
            chain = make_chain(transaction(300 + tag))
            digest = BranchStore.append_diagnostic_ledger(
                self.dir, ((10 + tag, chain),), digest, 2, 2, None
            )
        stop.set()
        for thread in threads:
            thread.join(5)
        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertTrue(proofs)
        for proof in proofs:
            BranchStore.verify_diagnostic_ledger_proof(proof)


if __name__ == "__main__":
    unittest.main()
