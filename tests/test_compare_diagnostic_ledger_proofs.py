"""Tests for offline continuity and fork analysis of diagnostic
ledger proofs.

``BranchStore.compare_diagnostic_ledger_proofs`` takes two canonical
proof strings exported by
:meth:`BranchStore.export_diagnostic_ledger_proof` -- never the ledger
itself -- and reports how the second authenticated history relates to
the first: ``"same"`` snapshots, a ``"continued"`` append (possibly
across a verified rotation) or a ``"forked"`` pair of legal successors
after the last digest-identical segment, refusing rollbacks,
cross-ledger splices, rotations that lost their transition segments
and rewritten shared history with ``ValueError``.

These tests cover:

* argument validation: non-string proofs (``TypeError``), empty or
  non-canonical proofs, time-range filters and mismatched ordered
  identity sets (``ValueError``), in before/after order;
* ``same``: identical snapshots, empty changes, the tail segment as
  the common boundary, empty ledgers with a ``None`` boundary;
* ``continued``: plain appends, rotation exactly up to the before
  tail, rotation of a proper prefix, pure rotation without new
  entries, continuation from the empty ledger, and the ``rotated``
  identities newly entering the prefix in request order;
* ``forked``: divergent successors with and without rotation, each
  side's unique segments in chain order;
* refusal: an ``after`` proof that is a strict historical prefix
  (rollback), two ledgers with no common segment (splice), a rotation
  past the before tail with the transition segments missing, and
  resealed proofs whose rotated-prefix digest, entry count, time
  bounds or evidence identities contradict the before history
  (rewrite);
* strict isolation of every returned level and no modification of
  any proof, ledger or other state.
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


class CompareTestBase(unittest.TestCase):
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

    def compare(self, before, after):
        return BranchStore.compare_diagnostic_ledger_proofs(before, after)

    def digest_of(self, directory) -> str:
        return BranchStore.page_diagnostic_ledger(
            directory, None, None, None, 100, None
        )["snapshot"]["digest"]

    def reseal_proof(self, document: dict) -> str:
        """Re-seal every layer of a (possibly tampered) proof so it
        still verifies on its own: evidence, manifest, snapshot and
        the package checksum."""
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
        # The tampered proof must still authenticate in isolation.
        BranchStore.verify_diagnostic_ledger_proof(text)
        return text


class ArgumentValidationTests(CompareTestBase):
    def setUp(self) -> None:
        super().setUp()
        self.directory = self.ledger("ledger")
        self.append(self.directory, self.entries[:2], None)
        self.proof = self.export(self.directory)

    def test_non_string_arguments_raise_type_error(self) -> None:
        for bad in (1, 1.5, b"x", ["x"], {"x": 1}, object(), None):
            with self.subTest(bad=type(bad).__name__):
                with self.assertRaises(TypeError):
                    self.compare(bad, self.proof)
                with self.assertRaises(TypeError):
                    self.compare(self.proof, bad)

    def test_empty_and_non_canonical_proofs_raise_value_error(self):
        with self.assertRaises(ValueError):
            self.compare("", self.proof)
        with self.assertRaises(ValueError):
            self.compare(self.proof, "")
        document = json.loads(self.proof)
        with self.assertRaises(ValueError):
            self.compare(json.dumps(document, indent=2), self.proof)
        with self.assertRaises(ValueError):
            self.compare(self.proof, "not a proof")

    def test_before_is_validated_first(self) -> None:
        with self.assertRaises(ValueError):
            self.compare("not a proof", 1)
        with self.assertRaises(TypeError):
            self.compare(1, "not a proof")

    def test_time_range_proofs_are_rejected(self) -> None:
        timed = BranchStore.export_diagnostic_ledger_proof(
            self.directory, 1, 2, None
        )
        with self.assertRaises(ValueError):
            self.compare(timed, self.proof)
        with self.assertRaises(ValueError):
            self.compare(self.proof, timed)
        with self.assertRaises(ValueError):
            self.compare(timed, timed)

    def test_identity_sets_must_match_in_order(self) -> None:
        subset = self.export(self.directory, (transaction(0),))
        with self.assertRaises(ValueError):
            self.compare(subset, self.proof)
        reordered = self.export(
            self.directory, (transaction(1), transaction(0))
        )
        narrowed = self.export(
            self.directory, (transaction(0), transaction(1))
        )
        with self.assertRaises(ValueError):
            self.compare(narrowed, reordered)

    def test_authentication_failure_raises_value_error(self) -> None:
        document = json.loads(self.proof)
        document["snapshot"]["entries"] = 99
        body = {
            key: value
            for key, value in document.items()
            if key != "checksum"
        }
        document["checksum"] = hashlib.sha256(
            canonical(body).encode("utf-8")
        ).hexdigest()
        with self.assertRaises(ValueError):
            self.compare(canonical(document), self.proof)


class SameRelationTests(CompareTestBase):
    def test_identical_snapshots_are_same(self) -> None:
        directory = self.ledger("ledger")
        digest = self.append(directory, self.entries[:6], None)
        result = self.compare(self.export(directory), self.export(directory))
        self.assertEqual(
            list(result),
            ["relation", "common", "before_only", "after_only",
             "rotated"],
        )
        self.assertEqual(result["relation"], "same")
        self.assertEqual(result["before_only"], ())
        self.assertEqual(result["after_only"], ())
        self.assertEqual(result["rotated"], ())
        common = result["common"]
        self.assertEqual(list(common), ["index", "digest", "entries"])
        self.assertEqual(common["index"], 3)
        self.assertEqual(common["entries"], 6)
        with open(
            os.path.join(directory, "manifest.json"), "rb"
        ) as handle:
            manifest = json.loads(handle.read().decode("utf-8"))
        self.assertEqual(
            common["digest"], manifest["segments"][-1]["digest"]
        )
        self.assertEqual(self.digest_of(directory), digest)

    def test_same_with_a_rotated_prefix_counts_every_entry(self) -> None:
        directory = self.ledger("ledger")
        digest = self.append(directory, self.entries[:4], None, keep=1)
        digest = self.append(directory, self.entries[4:6], digest, keep=1)
        result = self.compare(self.export(directory), self.export(directory))
        self.assertEqual(result["relation"], "same")
        self.assertEqual(result["common"]["index"], 3)
        self.assertEqual(result["common"]["entries"], 6)

    def test_empty_ledgers_are_same_with_no_boundary(self) -> None:
        first = self.ledger("first")
        second = self.ledger("second")
        self.append(first, (), None)
        self.append(second, (), None)
        result = self.compare(self.export(first), self.export(second))
        self.assertEqual(result["relation"], "same")
        self.assertIsNone(result["common"])
        self.assertEqual(result["before_only"], ())
        self.assertEqual(result["after_only"], ())
        self.assertEqual(result["rotated"], ())


class ContinuedRelationTests(CompareTestBase):
    def test_plain_append_continues(self) -> None:
        directory = self.ledger("ledger")
        digest = self.append(directory, self.entries[:4], None)
        before = self.export(directory)
        digest = self.append(directory, self.entries[4:8], digest)
        after = self.export(directory)
        result = self.compare(before, after)
        self.assertEqual(result["relation"], "continued")
        self.assertEqual(result["common"]["index"], 2)
        self.assertEqual(result["common"]["entries"], 4)
        self.assertEqual(result["before_only"], ())
        self.assertEqual(result["rotated"], ())
        increment = result["after_only"]
        self.assertEqual([item["index"] for item in increment], [3, 4])
        for item in increment:
            self.assertEqual(list(item), ["index", "digest", "entries"])
        self.assertEqual(
            [entry["at"] for entry in increment[0]["entries"]], [5, 6]
        )
        self.assertEqual(
            [entry["at"] for entry in increment[1]["entries"]], [7, 8]
        )
        self.assertEqual(
            increment[0]["entries"][0]["summary"]["transaction"],
            transaction(4),
        )
        self.assertIsInstance(increment[0]["entries"], tuple)
        self.assertIsInstance(increment[0]["entries"][0]["records"], tuple)

    def test_continuation_from_the_empty_ledger_has_no_boundary(self):
        directory = self.ledger("ledger")
        digest = self.append(directory, (), None)
        before = self.export(directory)
        digest = self.append(directory, self.entries[:4], digest)
        after = self.export(directory)
        result = self.compare(before, after)
        self.assertEqual(result["relation"], "continued")
        self.assertIsNone(result["common"])
        self.assertEqual(result["before_only"], ())
        self.assertEqual(result["rotated"], ())
        self.assertEqual(
            [item["index"] for item in result["after_only"]], [1, 2]
        )

    def test_continuation_from_empty_across_a_rotation(self) -> None:
        empty = self.ledger("empty")
        self.append(empty, (), None)
        before = self.export(empty)
        directory = self.ledger("ledger")
        digest = self.append(directory, self.entries[:4], None, keep=1)
        self.append(directory, self.entries[4:6], digest, keep=1)
        result = self.compare(before, self.export(directory))
        self.assertEqual(result["relation"], "continued")
        self.assertIsNone(result["common"])
        self.assertEqual(
            [item["index"] for item in result["after_only"]], [3]
        )
        self.assertEqual(
            result["rotated"], tuple(transaction(i) for i in range(4))
        )

    def test_rotation_exactly_up_to_the_before_tail(self) -> None:
        directory = self.ledger("ledger")
        digest = self.append(directory, self.entries[:4], None)
        before = self.export(directory)
        # Segments 1-2 rotated away, segment 3 appended, keep=1.
        digest = self.append(directory, self.entries[4:6], digest, keep=1)
        result = self.compare(before, self.export(directory))
        self.assertEqual(result["relation"], "continued")
        self.assertEqual(result["common"]["index"], 2)
        self.assertEqual(result["common"]["entries"], 4)
        self.assertEqual(
            [item["index"] for item in result["after_only"]], [3]
        )
        self.assertEqual(
            result["rotated"], tuple(transaction(i) for i in range(4))
        )

    def test_rotation_of_a_proper_prefix_then_append(self) -> None:
        directory = self.ledger("ledger")
        digest = self.append(directory, self.entries[:6], None)
        before = self.export(directory)
        # Segments 1-2 rotated away, segments 3-4 retained, 5 appended.
        digest = self.append(directory, self.entries[6:8], digest, keep=2)
        digest = self.append(directory, self.entries[8:10], digest, keep=2)
        result = self.compare(before, self.export(directory))
        self.assertEqual(result["relation"], "continued")
        self.assertEqual(result["common"]["index"], 3)
        self.assertEqual(result["common"]["entries"], 6)
        self.assertEqual(
            [item["index"] for item in result["after_only"]], [4, 5]
        )
        self.assertEqual(
            result["rotated"], tuple(transaction(i) for i in range(6))
        )

    def test_pure_rotation_without_new_entries(self) -> None:
        directory = self.ledger("ledger")
        digest = self.append(directory, self.entries[:6], None, keep=2)
        digest = self.append(directory, self.entries[6:8], digest, keep=2)
        before = self.export(directory)
        # An empty batch with a tighter retention rotates one segment.
        self.append(directory, (), digest, keep=1)
        result = self.compare(before, self.export(directory))
        self.assertEqual(result["relation"], "continued")
        self.assertEqual(result["before_only"], ())
        self.assertEqual(result["after_only"], ())
        self.assertEqual(result["common"]["index"], 4)
        self.assertEqual(result["common"]["entries"], 8)
        self.assertEqual(
            result["rotated"], (transaction(4), transaction(5))
        )

    def test_rotated_follows_the_request_order(self) -> None:
        directory = self.ledger("ledger")
        digest = self.append(directory, self.entries[:4], None)
        identities = (transaction(2), transaction(0), transaction(9))
        before = self.export(directory, identities)
        self.append(directory, self.entries[4:6], digest, keep=1)
        after = self.export(directory, identities)
        result = self.compare(before, after)
        self.assertEqual(result["relation"], "continued")
        self.assertEqual(
            result["rotated"], (transaction(2), transaction(0))
        )

    def test_rotation_after_an_already_rotated_before(self) -> None:
        directory = self.ledger("ledger")
        digest = self.append(directory, self.entries[:4], None, keep=2)
        digest = self.append(directory, self.entries[4:6], digest, keep=2)
        before = self.export(directory)
        self.assertEqual(
            json.loads(before)["manifest"]["dropped"]["entries"], 2
        )
        digest = self.append(directory, self.entries[6:8], digest, keep=2)
        digest = self.append(directory, self.entries[8:10], digest, keep=2)
        result = self.compare(before, self.export(directory))
        self.assertEqual(result["relation"], "continued")
        self.assertEqual(result["common"]["index"], 3)
        self.assertEqual(result["common"]["entries"], 6)
        self.assertEqual(
            [item["index"] for item in result["after_only"]], [4, 5]
        )
        self.assertEqual(
            result["rotated"], tuple(transaction(i) for i in range(2, 6))
        )


class ForkedRelationTests(CompareTestBase):
    def fork(self, keep=10):
        """Two copies of one ledger appended with different batches:
        return (before_proof, after_proof)."""
        base = self.ledger("base")
        digest = self.append(base, self.entries[:4], None, keep=keep)
        left = self.ledger("left")
        right = self.ledger("right")
        for source, target in ((base, left), (base, right)):
            shutil.copytree(source, target, dirs_exist_ok=True)
        left_entries = self.entries[4:6]
        right_entries = tuple(
            (time, make_chain(transaction(100 + time)))
            for time in (5, 6)
        )
        self.append(left, left_entries, digest, keep=keep)
        self.append(right, right_entries, digest, keep=keep)
        return self.export(left), self.export(right)

    def test_divergent_successors_fork(self) -> None:
        before, after = self.fork()
        result = self.compare(before, after)
        self.assertEqual(result["relation"], "forked")
        self.assertEqual(result["common"]["index"], 2)
        self.assertEqual(result["common"]["entries"], 4)
        self.assertEqual(result["rotated"], ())
        self.assertEqual(
            [item["index"] for item in result["before_only"]], [3]
        )
        self.assertEqual(
            [item["index"] for item in result["after_only"]], [3]
        )
        self.assertEqual(
            result["before_only"][0]["entries"][0]["summary"][
                "transaction"
            ],
            transaction(4),
        )
        self.assertEqual(
            result["after_only"][0]["entries"][0]["summary"][
                "transaction"
            ],
            transaction(105),
        )

    def test_fork_after_a_shared_rotation(self) -> None:
        before, after = self.fork(keep=1)
        result = self.compare(before, after)
        self.assertEqual(result["relation"], "forked")
        # Both sides rotated segments 1-2 away identically, then
        # diverged at segment 3: the boundary is the rotated prefix's
        # last segment.
        self.assertEqual(result["common"]["index"], 2)
        self.assertEqual(result["common"]["entries"], 4)
        self.assertEqual(
            [item["index"] for item in result["before_only"]], [3]
        )
        self.assertEqual(
            [item["index"] for item in result["after_only"]], [3]
        )
        self.assertEqual(result["rotated"], ())

    def test_fork_when_only_after_rotated(self) -> None:
        base = self.ledger("base")
        self.append(base, self.entries[:6], None)
        before = self.export(base)
        # A copy of the same lineage diverges at segment 3 and then
        # rotates the shared prefix away.
        forked = self.ledger("forked")
        early = self.ledger("early")
        self.append(early, self.entries[:4], None)
        shutil.copytree(early, forked, dirs_exist_ok=True)
        right_entries = tuple(
            (time, make_chain(transaction(100 + time)))
            for time in (5, 6)
        )
        self.append(
            forked, right_entries, self.digest_of(forked), keep=1
        )
        result = self.compare(before, self.export(forked))
        self.assertEqual(result["relation"], "forked")
        self.assertEqual(result["common"]["index"], 2)
        self.assertEqual(result["common"]["entries"], 4)
        self.assertEqual(
            [item["index"] for item in result["before_only"]], [3]
        )
        self.assertEqual(
            [item["index"] for item in result["after_only"]], [3]
        )


class RefusalTests(CompareTestBase):
    def test_rollback_is_refused(self) -> None:
        directory = self.ledger("ledger")
        digest = self.append(directory, self.entries[:2], None)
        before = self.export(directory)
        self.append(directory, self.entries[2:4], digest)
        after = self.export(directory)
        # after is the earlier snapshot: comparing newer against older
        # is a rollback, not a continuation.
        with self.assertRaises(ValueError):
            self.compare(after, before)

    def test_empty_after_a_non_empty_before_is_a_rollback(self) -> None:
        empty = self.ledger("empty")
        self.append(empty, (), None)
        directory = self.ledger("ledger")
        self.append(directory, self.entries[:2], None)
        with self.assertRaises(ValueError):
            self.compare(self.export(directory), self.export(empty))

    def test_cross_ledger_splice_is_refused(self) -> None:
        first = self.ledger("first")
        second = self.ledger("second")
        self.append(first, self.entries[:2], None)
        other = tuple(
            (time, make_chain(transaction(50 + time)))
            for time in (1, 2)
        )
        self.append(second, other, None)
        with self.assertRaises(ValueError):
            self.compare(self.export(first), self.export(second))

    def test_rotation_past_the_before_tail_is_refused(self) -> None:
        directory = self.ledger("ledger")
        digest = self.append(directory, self.entries[:4], None, keep=2)
        before = self.export(directory)
        # The ledger advances past the before tail and then rotates
        # the connecting segment away: no proof carries it any more.
        digest = self.append(directory, self.entries[4:8], digest, keep=2)
        self.append(directory, self.entries[8:10], digest, keep=2)
        after = self.export(directory)
        manifest = json.loads(after)["manifest"]
        self.assertEqual(manifest["dropped"]["entries"], 6)
        with self.assertRaises(ValueError):
            self.compare(before, after)

    def test_rewrite_of_shared_prefix_digest_is_refused(self) -> None:
        directory = self.ledger("ledger")
        digest = self.append(directory, self.entries[:4], None, keep=1)
        before = self.export(directory)
        self.append(directory, self.entries[4:6], digest, keep=1)
        document = json.loads(self.export(directory))
        forged = "f" * 64
        document["manifest"]["dropped"]["digest"] = forged
        document["evidence"]["digest"] = forged
        # The forged digest also becomes segment 3's predecessor link.
        segment = document["segments"][0]["document"]
        segment["prev"] = forged
        document["segments"][0]["document"] = json.loads(reseal(segment))
        segment_digest = hashlib.sha256(
            canonical(document["segments"][0]["document"]).encode(
                "utf-8"
            )
        ).hexdigest()
        document["manifest"]["segments"][0]["digest"] = segment_digest
        tampered = self.reseal_proof(document)
        with self.assertRaises(ValueError):
            self.compare(before, tampered)

    def test_rewrite_of_shared_prefix_count_is_refused(self) -> None:
        directory = self.ledger("ledger")
        digest = self.append(directory, self.entries[:4], None, keep=1)
        before = self.export(directory)
        self.append(directory, self.entries[4:6], digest, keep=1)
        document = json.loads(self.export(directory))
        document["manifest"]["dropped"]["entries"] = 3
        document["evidence"]["entries"] = 3
        document["evidence"]["transactions"] = document["evidence"][
            "transactions"
        ][:3]
        tampered = self.reseal_proof(document)
        with self.assertRaises(ValueError):
            self.compare(before, tampered)

    def test_rewrite_of_shared_prefix_time_bounds_is_refused(self):
        directory = self.ledger("ledger")
        digest = self.append(directory, self.entries[:4], None, keep=1)
        before = self.export(directory)
        self.append(directory, self.entries[4:6], digest, keep=1)
        for key, value in (("first_at", 0), ("last_at", 99)):
            document = json.loads(self.export(directory))
            document["manifest"]["dropped"][key] = value
            document["evidence"][key] = value
            tampered = self.reseal_proof(document)
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    self.compare(before, tampered)

    def test_rewrite_of_shared_prefix_identities_is_refused(self):
        directory = self.ledger("ledger")
        digest = self.append(directory, self.entries[:4], None, keep=1)
        before = self.export(directory)
        self.append(directory, self.entries[4:6], digest, keep=1)
        document = json.loads(self.export(directory))
        # Swap the remembered identity order: the evidence no longer
        # inherits the before history exactly.
        identities = list(document["evidence"]["transactions"])
        identities[0], identities[-1] = identities[-1], identities[0]
        document["evidence"]["transactions"] = identities
        tampered = self.reseal_proof(document)
        with self.assertRaises(ValueError):
            self.compare(before, tampered)

    def test_shared_rotated_prefixes_must_agree(self) -> None:
        directory = self.ledger("ledger")
        digest = self.append(directory, self.entries[:4], None, keep=1)
        digest = self.append(directory, self.entries[4:6], digest, keep=1)
        before = self.export(directory)
        document = json.loads(self.export(directory))
        document["manifest"]["dropped"]["last_at"] = 99
        document["evidence"]["last_at"] = 99
        tampered = self.reseal_proof(document)
        with self.assertRaises(ValueError):
            self.compare(before, tampered)

    def test_after_without_the_established_prefix_is_refused(self):
        directory = self.ledger("ledger")
        digest = self.append(directory, self.entries[:4], None, keep=1)
        digest = self.append(directory, self.entries[4:6], digest, keep=1)
        before = self.export(directory)
        # A different ledger that never rotated shares nothing with
        # the rotated before proof.
        plain = self.ledger("plain")
        self.append(plain, self.entries[:2], None)
        with self.assertRaises(ValueError):
            self.compare(before, self.export(plain))


class IsolationTests(CompareTestBase):
    def test_returned_levels_are_isolated_between_calls(self) -> None:
        directory = self.ledger("ledger")
        digest = self.append(directory, self.entries[:4], None)
        before = self.export(directory)
        self.append(directory, self.entries[4:6], digest, keep=1)
        after = self.export(directory)
        first = self.compare(before, after)
        first["common"]["digest"] = "mutated"
        first["after_only"][0]["entries"][0]["summary"][
            "transaction"
        ] = "mutated"
        first["after_only"][0]["digest"] = "mutated"
        mutable = list(first["rotated"])
        mutable.append("mutated")
        second = self.compare(before, after)
        self.assertNotEqual(second["common"]["digest"], "mutated")
        self.assertNotEqual(
            second["after_only"][0]["entries"][0]["summary"][
                "transaction"
            ],
            "mutated",
        )
        self.assertNotEqual(second["after_only"][0]["digest"], "mutated")
        self.assertNotIn("mutated", second["rotated"])
        self.assertIsInstance(second["rotated"], tuple)

    def test_comparison_does_not_modify_the_proofs_or_the_ledger(self):
        directory = self.ledger("ledger")
        digest = self.append(directory, self.entries[:4], None)
        before = self.export(directory)
        digest = self.append(directory, self.entries[4:6], digest, keep=1)
        after = self.export(directory)
        files = {
            name: open(os.path.join(directory, name), "rb").read()
            for name in os.listdir(directory)
        }
        self.compare(before, after)
        with self.assertRaises(ValueError):
            self.compare(after, before)
        self.assertEqual(
            files,
            {
                name: open(os.path.join(directory, name), "rb").read()
                for name in os.listdir(directory)
            },
        )
        # Re-exporting the unchanged snapshot is deterministic.
        self.assertEqual(after, self.export(directory))
        page = BranchStore.page_diagnostic_ledger(
            directory, None, None, None, 100, None
        )
        self.assertEqual(page["snapshot"]["digest"], digest)


if __name__ == "__main__":
    unittest.main()
