"""Tests for multi-hop offline history auditing of diagnostic
ledger proof sequences.

``BranchStore.audit_diagnostic_ledger_proof_sequence`` takes a tuple of
canonical proof strings exported by
:meth:`BranchStore.export_diagnostic_ledger_proof` -- never a ledger --
authenticates every proof before analysing any relation, and reports
the hops between adjacent proofs: ``"same"`` snapshots, a
``"continued"`` append or fully evidenced rotation, and the first
``"forked"`` successor pair (which keeps both sides' unique histories
and terminates the audit). Individually legal proofs that nevertheless
contradict one another are reported rather than refused: a strict
historical prefix is ``"rollback"``, a shared-position digest, count,
time-bound or body contradiction is ``"rewritten"``, a rotation missing
its connecting segments is ``"missing_transition"``, and two non-empty
histories with no authenticatable common segment are ``"spliced"``
(except when the first proof authenticates the empty ledger).

These tests cover:

* argument validation: non-tuple containers (``TypeError``); non-str
  elements, empty, non-canonical or non-authenticating proofs,
  time-range filters and mismatched ordered identity sets
  (``ValueError``); full authentication of every proof before any
  relation analysis, so a late invalid element never leaks a partial
  audit;
* the deterministic ``"empty"`` result for the empty tuple;
* ``"continuous"`` audits: a single proof, repeated ``same`` hops,
  plain appends and evidenced rotations across several hops;
* ``"failed"`` audits: a first fork that terminates inference,
  rollback, splice, missing transition and rewrite classifications,
  each with its stable hop reason and first-failure record;
* strict isolation of every nested level and no modification of any
  proof, ledger or other state.
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


class SequenceAuditTestBase(unittest.TestCase):
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

    def audit(self, proofs):
        return BranchStore.audit_diagnostic_ledger_proof_sequence(proofs)

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
        still verifies on its own: evidence, manifest, snapshot and the
        package checksum."""
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


class ArgumentValidationTests(SequenceAuditTestBase):
    def setUp(self) -> None:
        super().setUp()
        self.directory = self.ledger("ledger")
        self.append(self.directory, self.entries[:2], None)
        self.proof = self.export(self.directory)

    def test_argument_must_be_a_tuple(self) -> None:
        for bad in ([self.proof], {self.proof}, None, 1, object()):
            with self.subTest(bad=type(bad).__name__):
                with self.assertRaises(TypeError):
                    self.audit(bad)

    def test_empty_tuple_is_the_empty_status(self) -> None:
        result = self.audit(())
        self.assertEqual(
            list(result), ["status", "common", "hops", "failure"]
        )
        self.assertEqual(result["status"], "empty")
        self.assertIsNone(result["common"])
        self.assertEqual(result["hops"], ())
        self.assertIsNone(result["failure"])
        # Deterministic and detached.
        self.assertEqual(result, self.audit(()))

    def test_non_string_elements_raise_type_error(self) -> None:
        for bad in (1, 1.5, b"x", ["x"], {"x": 1}, object(), None):
            with self.subTest(bad=type(bad).__name__):
                with self.assertRaises(TypeError):
                    self.audit((self.proof, bad))

    def test_elements_are_checked_in_order(self) -> None:
        # The type fault at index 0 surfaces before the value fault at
        # index 1.
        with self.assertRaises(TypeError):
            self.audit((1, ""))
        with self.assertRaises(ValueError):
            self.audit(("", 1))

    def test_empty_non_canonical_and_unsigned_proofs_raise(self) -> None:
        with self.assertRaises(ValueError):
            self.audit((self.proof, ""))
        with self.assertRaises(ValueError):
            self.audit(("", self.proof))
        pretty = json.dumps(json.loads(self.proof), indent=2)
        with self.assertRaises(ValueError):
            self.audit((self.proof, pretty))
        with self.assertRaises(ValueError):
            self.audit((self.proof, "not a proof"))

    def test_all_proofs_authenticate_before_any_analysis(self) -> None:
        # Two valid proofs whose first hop is a rollback, followed by a
        # non-authenticating string: the late authentication fault must
        # surface as ValueError instead of a partial failed audit.
        self.append(
            self.directory,
            self.entries[2:4],
            self.digest_of(self.directory),
        )
        newer = self.export(self.directory)
        older_dir = self.ledger("older")
        self.append(older_dir, self.entries[:2], None)
        older = self.export(older_dir)
        with self.assertRaises(ValueError):
            self.audit((newer, older, "not a proof"))
        # A forked first hop does not rescue a later invalid proof.
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
            self.audit((self.export(left), self.export(right), ""))

    def test_time_range_proofs_are_rejected(self) -> None:
        timed = BranchStore.export_diagnostic_ledger_proof(
            self.directory, 1, 2, None
        )
        with self.assertRaises(ValueError):
            self.audit((timed, self.proof))
        with self.assertRaises(ValueError):
            self.audit((self.proof, timed))

    def test_identity_sets_must_match_in_order(self) -> None:
        subset = self.export(self.directory, (transaction(0),))
        reordered = self.export(
            self.directory, (transaction(1), transaction(0))
        )
        with self.assertRaises(ValueError):
            self.audit((subset, self.proof))
        with self.assertRaises(ValueError):
            self.audit((self.proof, reordered))


class ContinuousAuditTests(SequenceAuditTestBase):
    def test_single_proof_ends_on_its_own_boundary(self) -> None:
        directory = self.ledger("ledger")
        self.append(directory, self.entries[:4], None)
        result = self.audit((self.export(directory),))
        self.assertEqual(result["status"], "continuous")
        self.assertIsNone(result["failure"])
        self.assertEqual(result["hops"], ())
        self.assertEqual(result["common"]["index"], 2)
        self.assertEqual(result["common"]["entries"], 4)

    def test_single_empty_ledger_proof_has_no_boundary(self) -> None:
        directory = self.ledger("empty")
        self.append(directory, (), None)
        result = self.audit((self.export(directory),))
        self.assertEqual(result["status"], "continuous")
        self.assertIsNone(result["common"])

    def test_repeated_identical_snapshots_are_same_hops(self) -> None:
        directory = self.ledger("ledger")
        self.append(directory, self.entries[:4], None)
        proof = self.export(directory)
        result = self.audit((proof, proof, proof))
        self.assertEqual(result["status"], "continuous")
        self.assertEqual(len(result["hops"]), 2)
        for hop in result["hops"]:
            self.assertEqual(
                list(hop),
                [
                    "index",
                    "relation",
                    "common",
                    "before_only",
                    "after_only",
                    "rotated",
                ],
            )
            self.assertEqual(hop["relation"], "same")
            self.assertEqual(hop["common"]["index"], 2)
            self.assertEqual(hop["before_only"], ())
            self.assertEqual(hop["after_only"], ())
            self.assertEqual(hop["rotated"], ())
        self.assertEqual([hop["index"] for hop in result["hops"]], [1, 2])
        self.assertEqual(result["common"]["index"], 2)

    def test_plain_appends_continue_across_hops(self) -> None:
        directory = self.ledger("ledger")
        digest = self.append(directory, self.entries[:4], None)
        first = self.export(directory)
        digest = self.append(directory, self.entries[4:8], digest)
        second = self.export(directory)
        digest = self.append(directory, self.entries[8:10], digest)
        third = self.export(directory)
        result = self.audit((first, second, third))
        self.assertEqual(result["status"], "continuous")
        self.assertIsNone(result["failure"])
        self.assertEqual(
            [hop["relation"] for hop in result["hops"]],
            ["continued", "continued"],
        )
        self.assertEqual(
            [
                segment["index"]
                for segment in result["hops"][0]["after_only"]
            ],
            [3, 4],
        )
        self.assertEqual(
            [
                segment["index"]
                for segment in result["hops"][1]["after_only"]
            ],
            [5],
        )
        for hop in result["hops"]:
            self.assertEqual(hop["before_only"], ())
            self.assertEqual(hop["rotated"], ())
            self.assertNotIn("reason", hop)
        # The common boundary stays on the first proof's tail: later
        # appends extend history but do not move the intersection.
        self.assertEqual(result["common"]["index"], 2)
        self.assertEqual(result["common"]["entries"], 4)

    def test_rotation_hop_carries_new_identities_in_request_order(
        self,
    ) -> None:
        directory = self.ledger("ledger")
        digest = self.append(directory, self.entries[:4], None)
        identities = (transaction(2), transaction(0), transaction(9))
        before = self.export(directory, identities)
        self.append(directory, self.entries[4:6], digest, keep=1)
        after = self.export(directory, identities)
        result = self.audit((before, after))
        self.assertEqual(result["status"], "continuous")
        hop = result["hops"][0]
        self.assertEqual(hop["relation"], "continued")
        self.assertEqual(hop["common"]["index"], 2)
        self.assertEqual(
            [segment["index"] for segment in hop["after_only"]], [3]
        )
        self.assertEqual(
            hop["rotated"], (transaction(2), transaction(0))
        )

    def test_empty_first_proof_continues_into_history(self) -> None:
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
        result = self.audit((empty, self.export(directory)))
        self.assertEqual(result["status"], "continuous")
        self.assertIsNone(result["common"])
        hop = result["hops"][0]
        self.assertEqual(hop["relation"], "continued")
        self.assertIsNone(hop["common"])
        self.assertEqual(
            [segment["index"] for segment in hop["after_only"]], [3]
        )
        self.assertEqual(
            hop["rotated"], tuple(transaction(i) for i in range(4))
        )


class FailedAuditTests(SequenceAuditTestBase):
    def fork_proofs(self, keep=10):
        base = self.ledger("base")
        digest = self.append(base, self.entries[:4], None, keep=keep)
        left = self.ledger("left")
        right = self.ledger("right")
        for target in (left, right):
            shutil.copytree(base, target, dirs_exist_ok=True)
        self.append(left, self.entries[4:6], digest, keep=keep)
        other = tuple(
            (time, make_chain(transaction(100 + time)))
            for time in (5, 6)
        )
        self.append(right, other, digest, keep=keep)
        return self.export(left), self.export(right)

    def test_first_fork_keeps_both_histories_and_failure_record(self):
        before, after = self.fork_proofs()
        result = self.audit((before, after))
        self.assertEqual(result["status"], "failed")
        self.assertEqual(
            list(result), ["status", "common", "hops", "failure"]
        )
        self.assertEqual(len(result["hops"]), 1)
        hop = result["hops"][0]
        self.assertEqual(
            list(hop),
            [
                "index",
                "relation",
                "common",
                "before_only",
                "after_only",
                "rotated",
                "reason",
            ],
        )
        self.assertEqual(hop["index"], 1)
        self.assertEqual(hop["relation"], "forked")
        self.assertEqual(hop["reason"], "forked")
        self.assertEqual(hop["common"]["index"], 2)
        self.assertEqual(hop["common"]["entries"], 4)
        self.assertEqual(
            [segment["index"] for segment in hop["before_only"]], [3]
        )
        self.assertEqual(
            [segment["index"] for segment in hop["after_only"]], [3]
        )
        self.assertEqual(
            hop["before_only"][0]["entries"][0]["summary"][
                "transaction"
            ],
            transaction(4),
        )
        self.assertEqual(
            hop["after_only"][0]["entries"][0]["summary"][
                "transaction"
            ],
            transaction(105),
        )
        self.assertEqual(hop["rotated"], ())
        self.assertEqual(
            result["failure"],
            {"index": 1, "classification": "forked"},
        )
        self.assertEqual(result["common"], hop["common"])

    def test_a_fork_terminates_later_inference(self) -> None:
        # A successful hop, then a fork; the third proof (which would
        # itself be a rollback relative to the right fork tip) is never
        # analysed: exactly two hops are recorded.
        directory = self.ledger("ledger")
        digest = self.append(directory, self.entries[:2], None)
        first = self.export(directory)
        digest = self.append(directory, self.entries[2:4], digest)
        base = self.ledger("base")
        shutil.copytree(directory, base, dirs_exist_ok=True)
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
        # An earlier snapshot than the right fork tip.
        result = self.audit(
            (first, self.export(left), self.export(right), first)
        )
        self.assertEqual(result["status"], "failed")
        self.assertEqual(
            [hop["relation"] for hop in result["hops"]],
            ["continued", "forked"],
        )
        self.assertEqual(
            result["failure"],
            {"index": 2, "classification": "forked"},
        )
        self.assertEqual(result["common"]["index"], 1)

    def test_rollback_after_successful_hops(self) -> None:
        directory = self.ledger("ledger")
        digest = self.append(directory, self.entries[:2], None)
        oldest = self.export(directory)
        digest = self.append(directory, self.entries[2:4], digest)
        middle = self.export(directory)
        digest = self.append(directory, self.entries[4:6], digest)
        newest = self.export(directory)
        result = self.audit((oldest, middle, newest, middle))
        self.assertEqual(result["status"], "failed")
        self.assertEqual(
            [hop["relation"] for hop in result["hops"]],
            ["continued", "continued", "rollback"],
        )
        failing = result["hops"][2]
        self.assertEqual(failing["index"], 3)
        self.assertEqual(failing["relation"], "rollback")
        self.assertEqual(failing["reason"], "rollback")
        self.assertEqual(failing["common"]["index"], 2)
        self.assertEqual(failing["common"]["entries"], 4)
        self.assertEqual(
            [segment["index"] for segment in failing["before_only"]],
            [3],
        )
        self.assertEqual(failing["after_only"], ())
        self.assertEqual(
            result["failure"],
            {"index": 3, "classification": "rollback"},
        )
        # The common boundary never moves past the first proof's tail.
        self.assertEqual(result["common"]["index"], 1)
        self.assertEqual(result["common"]["entries"], 2)

    def test_rollback_to_the_empty_ledger(self) -> None:
        empty_dir = self.ledger("empty")
        self.append(empty_dir, (), None)
        empty = self.export(empty_dir)
        directory = self.ledger("ledger")
        self.append(directory, self.entries[:2], None)
        result = self.audit((self.export(directory), empty))
        self.assertEqual(result["status"], "failed")
        hop = result["hops"][0]
        self.assertEqual(hop["relation"], "rollback")
        self.assertIsNone(hop["common"])
        self.assertEqual(hop["after_only"], ())
        self.assertEqual(
            [segment["index"] for segment in hop["before_only"]], [1]
        )
        self.assertIsNone(result["common"])
        self.assertEqual(
            result["failure"],
            {"index": 1, "classification": "rollback"},
        )

    def test_splice_between_two_non_empty_ledgers(self) -> None:
        first = self.ledger("first")
        second = self.ledger("second")
        self.append(first, self.entries[:2], None)
        other = tuple(
            (time, make_chain(transaction(50 + time)))
            for time in (1, 2)
        )
        self.append(second, other, None)
        result = self.audit((self.export(first), self.export(second)))
        self.assertEqual(result["status"], "failed")
        hop = result["hops"][0]
        self.assertEqual(hop["relation"], "spliced")
        self.assertEqual(hop["reason"], "spliced")
        self.assertIsNone(hop["common"])
        self.assertEqual(
            [segment["index"] for segment in hop["before_only"]], [1]
        )
        self.assertEqual(
            [segment["index"] for segment in hop["after_only"]], [1]
        )
        self.assertIsNone(result["common"])
        self.assertEqual(
            result["failure"],
            {"index": 1, "classification": "spliced"},
        )

    def test_missing_transition_segments(self) -> None:
        directory = self.ledger("ledger")
        digest = self.append(directory, self.entries[:4], None, keep=2)
        before = self.export(directory)
        digest = self.append(directory, self.entries[4:8], digest, keep=2)
        self.append(directory, self.entries[8:10], digest, keep=2)
        result = self.audit((before, self.export(directory)))
        self.assertEqual(result["status"], "failed")
        hop = result["hops"][0]
        self.assertEqual(hop["relation"], "missing_transition")
        self.assertEqual(hop["reason"], "missing_transition")
        # The before tail (segment 2) is the last boundary both sides
        # can authenticate without the missing transition segments.
        self.assertEqual(hop["common"]["index"], 2)
        self.assertEqual(hop["common"]["entries"], 4)
        self.assertEqual(
            result["failure"],
            {"index": 1, "classification": "missing_transition"},
        )

    def test_rewritten_rotated_prefix_digest(self) -> None:
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
            canonical(document["segments"][0]["document"]).encode(
                "utf-8"
            )
        ).hexdigest()
        document["manifest"]["segments"][0]["digest"] = segment_digest
        tampered = self.reseal_proof(document)
        result = self.audit((before, tampered))
        self.assertEqual(result["status"], "failed")
        hop = result["hops"][0]
        self.assertEqual(hop["relation"], "rewritten")
        self.assertEqual(hop["reason"], "rewritten")
        # Segment 1 was the last shared segment before the forged
        # boundary; the before side still uniquely carries segment 2.
        self.assertEqual(hop["common"]["index"], 1)
        self.assertEqual(hop["common"]["entries"], 2)
        self.assertEqual(
            [segment["index"] for segment in hop["before_only"]], [2]
        )
        self.assertEqual(
            result["failure"],
            {"index": 1, "classification": "rewritten"},
        )

    def test_rewritten_rotated_prefix_count(self) -> None:
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
        result = self.audit((before, tampered))
        self.assertEqual(result["status"], "failed")
        self.assertEqual(
            result["failure"]["classification"], "rewritten"
        )
        self.assertEqual(
            result["hops"][0]["reason"], "rewritten"
        )

    def test_rewritten_shared_prefix_time_bounds(self) -> None:
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
                result = self.audit((before, tampered))
                self.assertEqual(result["status"], "failed")
                self.assertEqual(
                    result["failure"]["classification"], "rewritten"
                )


class IsolationAndPurityTests(SequenceAuditTestBase):
    def test_nested_levels_are_isolated_between_calls(self) -> None:
        directory = self.ledger("ledger")
        digest = self.append(directory, self.entries[:4], None)
        first = self.export(directory)
        digest = self.append(directory, self.entries[4:6], digest, keep=1)
        second = self.export(directory)
        digest = self.append(directory, self.entries[6:8], digest, keep=1)
        third = self.export(directory)
        proofs = (first, second, third)
        result = self.audit(proofs)
        result["common"]["digest"] = "mutated"
        result["hops"][0]["common"]["digest"] = "mutated"
        result["hops"][0]["after_only"][0]["digest"] = "mutated"
        result["hops"][0]["after_only"][0]["entries"][0]["summary"][
            "transaction"
        ] = "mutated"
        mutable = list(result["hops"][0]["rotated"])
        mutable.append("mutated")
        again = self.audit(proofs)
        self.assertNotEqual(again["common"]["digest"], "mutated")
        self.assertNotEqual(
            again["hops"][0]["common"]["digest"], "mutated"
        )
        self.assertNotEqual(
            again["hops"][0]["after_only"][0]["digest"], "mutated"
        )
        self.assertNotEqual(
            again["hops"][0]["after_only"][0]["entries"][0]["summary"][
                "transaction"
            ],
            "mutated",
        )
        self.assertNotIn("mutated", again["hops"][0]["rotated"])
        for hop in again["hops"]:
            self.assertIsInstance(hop["after_only"], tuple)
            for segment in hop["after_only"]:
                self.assertIsInstance(segment["entries"], tuple)
                for entry in segment["entries"]:
                    self.assertIsInstance(entry["records"], tuple)

    def test_failure_levels_are_isolated_between_calls(self) -> None:
        before, after = self._fork_pair()
        proofs = (before, after)
        first = self.audit(proofs)
        first["hops"][0]["before_only"][0]["digest"] = "mutated"
        first["hops"][0]["common"]["digest"] = "mutated"
        second = self.audit(proofs)
        self.assertNotEqual(
            second["hops"][0]["before_only"][0]["digest"], "mutated"
        )
        self.assertNotEqual(
            second["hops"][0]["common"]["digest"], "mutated"
        )

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

    def test_audit_does_not_modify_proofs_or_the_ledger(self) -> None:
        directory = self.ledger("ledger")
        digest = self.append(directory, self.entries[:4], None)
        before = self.export(directory)
        digest = self.append(directory, self.entries[4:6], digest, keep=1)
        after = self.export(directory)
        files = self._directory_files(directory)
        self.audit((before, after))
        self.audit((after, before))
        self.assertEqual(files, self._directory_files(directory))
        self.assertEqual(after, self.export(directory))
        self.assertEqual(
            BranchStore.page_diagnostic_ledger(
                directory, None, None, None, 100, None
            )["snapshot"]["digest"],
            digest,
        )


if __name__ == "__main__":
    unittest.main()
