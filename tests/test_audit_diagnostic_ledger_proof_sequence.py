"""Tests for offline multi-hop auditing of diagnostic ledger proofs.

``BranchStore.audit_diagnostic_ledger_proof_sequence`` takes a tuple of
canonical proof strings exported by
:meth:`BranchStore.export_diagnostic_ledger_proof` -- never the ledgers
themselves -- and audits the ordered history they represent with no
filesystem access:

* argument validation: a non-tuple raises ``TypeError``; the empty
  tuple returns the deterministic ``"empty"`` result; elements are
  checked strictly in order (non-:class:`str`` -> ``TypeError``, empty,
  non-canonical or unauthenticatable proof -> ``ValueError``); every
  proof is authenticated before any relationship is examined, then all
  proofs must use the transaction-identity mode and request the same
  ordered identity set;
* continuous histories: ``"same"`` and ``"continued"`` hops (plain
  appends and verified rotations, with the newly rotated identities);
* findings recorded rather than raised: the first ``"forked"`` hop with
  its shared boundary and both unique histories ends further inference;
  ``"rollback"`` is a strict historical prefix, never a fork;
  ``"rewritten"`` shared history; ``"missing_transition"`` rotations
  whose connecting segments were lost; and ``"spliced"`` non-empty
  histories with no common segment (an empty first proof is exempt);
* result shape: ``status``/``common``/``hops``/``failure`` key order,
  hop indices, boundaries, unique segments, stable reasons and the
  failing right index; and strict isolation of every returned level with
  no modification of the proofs, ledgers or any other state.
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
    from test_compare_diagnostic_ledger_proofs import CompareTestBase
except ImportError:  # running as a package member from the repo root
    from tests.test_diagnostic_ledger import (
        make_chain,
        reseal,
        transaction,
    )
    from tests.test_compare_diagnostic_ledger_proofs import CompareTestBase


class SequenceTestBase(CompareTestBase):
    def audit(self, proofs):
        return BranchStore.audit_diagnostic_ledger_proof_sequence(proofs)

    def fork_pair(self, keep=10):
        """Two ledgers that share a prefix and diverge at segment 3:
        return (root_proof, left_proof, right_proof)."""
        base = self.ledger("fp_base")
        digest = self.append(base, self.entries[:4], None, keep=keep)
        left = self.ledger("fp_left")
        right = self.ledger("fp_right")
        shutil.copytree(base, left, dirs_exist_ok=True)
        shutil.copytree(base, right, dirs_exist_ok=True)
        left_entries = self.entries[4:6]
        right_entries = tuple(
            (time, make_chain(transaction(100 + time)))
            for time in (5, 6)
        )
        self.append(left, left_entries, digest, keep=keep)
        self.append(right, right_entries, digest, keep=keep)
        return self.export(base), self.export(left), self.export(right)

    def tampered_reseal(self, directory):
        """Export ``directory`` and forge a shared-prefix digest rewrite,
        re-sealing every layer (including the first retained segment's
        new predecessor link and digest) so the proof still authenticates
        on its own -- the exact recipe a cross-proof rewrite needs."""
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
            BranchStore._canonical_json(
                document["segments"][0]["document"]
            ).encode("utf-8")
        ).hexdigest()
        document["manifest"]["segments"][0]["digest"] = segment_digest
        return self.reseal_proof(document)


class EmptyAndValidationTests(SequenceTestBase):
    def test_empty_tuple_is_deterministic(self) -> None:
        result = self.audit(())
        self.assertEqual(
            list(result), ["status", "common", "hops", "failure"]
        )
        self.assertEqual(result["status"], "empty")
        self.assertIsNone(result["common"])
        self.assertEqual(result["hops"], ())
        self.assertIsNone(result["failure"])
        # Repeated empty audits never share mutable state.
        self.assertIsNot(self.audit(()), result)

    def test_non_tuple_raises_type_error(self) -> None:
        directory = self.ledger("ledger")
        self.append(directory, self.entries[:2], None)
        proof = self.export(directory)
        for bad in ([proof], {proof}, proof, (proof for _ in (0,))):
            with self.subTest(kind=type(bad).__name__):
                with self.assertRaises(TypeError):
                    self.audit(bad)

    def test_non_str_element_raises_type_error(self) -> None:
        directory = self.ledger("ledger")
        self.append(directory, self.entries[:2], None)
        proof = self.export(directory)
        for bad in (1, 1.5, b"x", ["x"], {"x": 1}, object(), None):
            with self.subTest(kind=type(bad).__name__):
                with self.assertRaises(TypeError):
                    self.audit((bad,))
                with self.assertRaises(TypeError):
                    self.audit((proof, bad))

    def test_empty_and_non_canonical_elements_raise_value_error(self):
        directory = self.ledger("ledger")
        self.append(directory, self.entries[:2], None)
        proof = self.export(directory)
        with self.assertRaises(ValueError):
            self.audit(("",))
        with self.assertRaises(ValueError):
            self.audit((proof, ""))
        with self.assertRaises(ValueError):
            self.audit(("not a proof",))
        with self.assertRaises(ValueError):
            self.audit((proof, json.dumps(json.loads(proof), indent=2)))

    def test_elements_are_checked_in_order(self) -> None:
        directory = self.ledger("ledger")
        self.append(directory, self.entries[:2], None)
        proof = self.export(directory)
        # A str defect (ValueError) at index 0 outranks a type defect
        # later; a type defect at index 0 outranks a str defect later.
        with self.assertRaises(ValueError):
            self.audit(("not a proof", 1))
        with self.assertRaises(TypeError):
            self.audit((1, "not a proof"))
        self.assertEqual(proof, proof)

    def test_all_proofs_authenticate_before_any_analysis(self) -> None:
        directory = self.ledger("ledger")
        digest = self.append(directory, self.entries[:2], None)
        older = self.export(directory)
        self.append(directory, self.entries[2:4], digest)
        newer = self.export(directory)
        # The first hop would already be a rollback finding, but the
        # malformed third element must surface first as ValueError -- no
        # partial (failed) audit is returned.
        with self.assertRaises(ValueError):
            self.audit((newer, older, "garbage"))
        with self.assertRaises(TypeError):
            self.audit((newer, older, 1))

    def test_time_range_proofs_are_rejected(self) -> None:
        directory = self.ledger("ledger")
        self.append(directory, self.entries[:2], None)
        proof = self.export(directory)
        timed = BranchStore.export_diagnostic_ledger_proof(
            directory, 1, 2, None
        )
        with self.assertRaises(ValueError):
            self.audit((timed,))
        with self.assertRaises(ValueError):
            self.audit((proof, timed))

    def test_identity_sets_must_match_in_order(self) -> None:
        directory = self.ledger("ledger")
        self.append(directory, self.entries[:2], None)
        full = self.export(directory)
        subset = self.export(directory, (transaction(0),))
        reordered = self.export(
            directory, (transaction(1), transaction(0))
        )
        with self.assertRaises(ValueError):
            self.audit((subset, full))
        with self.assertRaises(ValueError):
            self.audit((full, reordered))

    def test_single_proof_is_continuous_with_no_hops(self) -> None:
        directory = self.ledger("ledger")
        self.append(directory, self.entries[:4], None)
        result = self.audit((self.export(directory),))
        self.assertEqual(result["status"], "continuous")
        self.assertEqual(result["hops"], ())
        self.assertIsNone(result["failure"])
        self.assertEqual(result["common"]["index"], 2)
        self.assertEqual(result["common"]["entries"], 4)


class ContinuousSequenceTests(SequenceTestBase):
    def test_plain_append_sequence(self) -> None:
        directory = self.ledger("ledger")
        digest = self.append(directory, self.entries[:4], None)
        first = self.export(directory)
        digest = self.append(directory, self.entries[4:8], digest)
        second = self.export(directory)
        self.append(directory, self.entries[8:10], digest)
        third = self.export(directory)
        result = self.audit((first, second, third))
        self.assertEqual(result["status"], "continuous")
        self.assertIsNone(result["failure"])
        # The boundary every proof jointly authenticates is the oldest
        # proof's tail; later proofs carry it inside their own chains.
        self.assertEqual(result["common"]["index"], 2)
        self.assertEqual(result["common"]["entries"], 4)
        hops = result["hops"]
        self.assertEqual([hop["index"] for hop in hops], [1, 2])
        for hop in hops:
            self.assertEqual(hop["relation"], "continued")
            self.assertEqual(
                list(hop),
                ["index", "relation", "common", "before_only",
                 "after_only", "rotated"],
            )
            self.assertEqual(hop["before_only"], ())
            self.assertEqual(hop["rotated"], ())
        self.assertEqual(
            [item["index"] for item in hops[0]["after_only"]], [3, 4]
        )
        self.assertEqual(
            [item["index"] for item in hops[1]["after_only"]], [5]
        )
        self.assertEqual(hops[0]["common"]["index"], 2)
        self.assertEqual(hops[1]["common"]["index"], 4)
        self.assertIsInstance(hops, tuple)

    def test_same_snapshot_hop_is_continuous(self) -> None:
        directory = self.ledger("ledger")
        self.append(directory, self.entries[:4], None)
        proof = self.export(directory)
        result = self.audit((proof, proof, proof))
        self.assertEqual(result["status"], "continuous")
        self.assertEqual(
            [hop["relation"] for hop in result["hops"]],
            ["same", "same"],
        )
        for hop in result["hops"]:
            self.assertEqual(hop["before_only"], ())
            self.assertEqual(hop["after_only"], ())
            self.assertEqual(hop["rotated"], ())
            self.assertEqual(hop["common"]["index"], 2)

    def test_rotation_hop_carries_new_rotated_identities(self) -> None:
        directory = self.ledger("ledger")
        digest = self.append(directory, self.entries[:4], None)
        before = self.export(directory)
        self.append(directory, self.entries[4:6], digest, keep=1)
        after = self.export(directory)
        result = self.audit((before, after))
        self.assertEqual(result["status"], "continuous")
        hop = result["hops"][0]
        self.assertEqual(hop["relation"], "continued")
        self.assertEqual(
            hop["rotated"], tuple(transaction(i) for i in range(4))
        )
        self.assertEqual(
            [item["index"] for item in hop["after_only"]], [3]
        )
        self.assertEqual(hop["common"]["index"], 2)

    def test_continuation_starts_from_an_empty_first_proof(self) -> None:
        empty_dir = self.ledger("empty")
        self.append(empty_dir, (), None)
        empty = self.export(empty_dir)
        directory = self.ledger("ledger")
        self.append(directory, self.entries[:4], None)
        result = self.audit((empty, self.export(directory)))
        self.assertEqual(result["status"], "continuous")
        hop = result["hops"][0]
        self.assertEqual(hop["relation"], "continued")
        self.assertIsNone(hop["common"])
        self.assertIsNone(result["common"])
        self.assertEqual(
            [item["index"] for item in hop["after_only"]], [1, 2]
        )


class FindingSequenceTests(SequenceTestBase):
    def test_first_fork_terminates_later_inference(self) -> None:
        root, left, right = self.fork_pair()
        # A fourth proof that would relate to ``right`` is carried along
        # but never analysed once the fork at index 2 ends inference.
        decoy = root
        result = self.audit((root, left, right, decoy))
        self.assertEqual(result["status"], "failed")
        self.assertEqual(len(result["hops"]), 2)
        first = result["hops"][0]
        self.assertEqual(first["index"], 1)
        self.assertEqual(first["relation"], "continued")
        fork = result["hops"][1]
        self.assertEqual(
            list(fork),
            ["index", "relation", "common", "before_only",
             "after_only", "rotated", "reason"],
        )
        self.assertEqual(fork["index"], 2)
        self.assertEqual(fork["relation"], "forked")
        self.assertEqual(fork["reason"], "forked")
        self.assertEqual(fork["common"]["index"], 2)
        self.assertEqual(fork["common"]["entries"], 4)
        self.assertEqual(
            [item["index"] for item in fork["before_only"]], [3]
        )
        self.assertEqual(
            [item["index"] for item in fork["after_only"]], [3]
        )
        self.assertEqual(
            fork["before_only"][0]["entries"][0]["summary"][
                "transaction"
            ],
            transaction(4),
        )
        self.assertEqual(
            fork["after_only"][0]["entries"][0]["summary"][
                "transaction"
            ],
            transaction(105),
        )
        self.assertEqual(fork["rotated"], ())
        self.assertEqual(
            result["failure"], {"index": 2, "relation": "forked"}
        )
        self.assertEqual(list(result["failure"]), ["index", "relation"])
        self.assertEqual(result["common"]["index"], 2)

    def test_fork_after_a_shared_rotation(self) -> None:
        root, left, right = self.fork_pair(keep=1)
        result = self.audit((root, left, right))
        self.assertEqual(result["status"], "failed")
        fork = result["hops"][1]
        self.assertEqual(fork["relation"], "forked")
        # Both sides rotated segments 1-2 away identically, then
        # diverged at segment 3: the boundary is the prefix tail.
        self.assertEqual(fork["common"]["index"], 2)
        self.assertEqual(fork["common"]["entries"], 4)
        self.assertEqual(
            [item["index"] for item in fork["before_only"]], [3]
        )
        self.assertEqual(
            [item["index"] for item in fork["after_only"]], [3]
        )
        self.assertEqual(result["failure"]["relation"], "forked")
        self.assertEqual(result["common"], fork["common"])
        self.assertIsNot(result["common"], fork["common"])

    def test_rollback_after_a_successful_hop(self) -> None:
        directory = self.ledger("ledger")
        digest = self.append(directory, self.entries[:2], None)
        first = self.export(directory)
        digest = self.append(directory, self.entries[2:4], digest)
        second = self.export(directory)
        self.append(directory, self.entries[4:6], digest)
        third = self.export(directory)
        # Grows, grows, then jumps back to the first snapshot.
        result = self.audit((first, second, third, first))
        self.assertEqual(result["status"], "failed")
        self.assertEqual(len(result["hops"]), 3)
        self.assertEqual(
            [hop["relation"] for hop in result["hops"]],
            ["continued", "continued", "rollback"],
        )
        rollback = result["hops"][2]
        self.assertEqual(rollback["reason"], "rollback")
        self.assertEqual(rollback["common"]["index"], 1)
        self.assertEqual(rollback["common"]["entries"], 2)
        self.assertEqual(
            [item["index"] for item in rollback["before_only"]],
            [2, 3],
        )
        self.assertEqual(result["failure"]["index"], 3)
        # The boundary every proof jointly authenticates is bounded back
        # to the rollback target: the first proof's tail, index 1.
        self.assertEqual(result["common"]["index"], 1)

    def test_rollback_is_a_finding_not_a_fork(self) -> None:
        directory = self.ledger("ledger")
        digest = self.append(directory, self.entries[:4], None)
        before = self.export(directory)
        self.append(directory, self.entries[4:6], digest)
        after = self.export(directory)
        # Newer history audited before the older snapshot: rollback.
        result = self.audit((after, before))
        self.assertEqual(result["status"], "failed")
        hop = result["hops"][0]
        self.assertEqual(hop["index"], 1)
        self.assertEqual(hop["relation"], "rollback")
        self.assertEqual(hop["reason"], "rollback")
        self.assertEqual(hop["common"]["index"], 2)
        self.assertEqual(hop["common"]["entries"], 4)
        self.assertEqual(
            [item["index"] for item in hop["before_only"]], [3]
        )
        self.assertEqual(hop["after_only"], ())
        self.assertEqual(
            result["failure"], {"index": 1, "relation": "rollback"}
        )

    def test_rollback_all_the_way_to_empty(self) -> None:
        empty_dir = self.ledger("empty")
        self.append(empty_dir, (), None)
        empty = self.export(empty_dir)
        directory = self.ledger("ledger")
        self.append(directory, self.entries[:4], None)
        grown = self.export(directory)
        result = self.audit((grown, empty))
        self.assertEqual(result["status"], "failed")
        hop = result["hops"][0]
        self.assertEqual(hop["relation"], "rollback")
        self.assertIsNone(hop["common"])
        self.assertEqual(
            [item["index"] for item in hop["before_only"]], [1, 2]
        )
        self.assertEqual(result["failure"]["relation"], "rollback")

    def test_rewritten_shared_prefix_is_a_finding(self) -> None:
        directory = self.ledger("ledger")
        digest = self.append(directory, self.entries[:4], None, keep=1)
        before = self.export(directory)
        self.append(directory, self.entries[4:6], digest, keep=1)
        tampered = self.tampered_reseal(directory)
        # The tampered proof still authenticates on its own.
        BranchStore.verify_diagnostic_ledger_proof(tampered)
        result = self.audit((before, tampered))
        self.assertEqual(result["status"], "failed")
        hop = result["hops"][0]
        self.assertEqual(hop["relation"], "rewritten")
        self.assertEqual(hop["reason"], "rewritten")
        self.assertEqual(
            result["failure"], {"index": 1, "relation": "rewritten"}
        )

    def test_missing_transition_is_a_finding(self) -> None:
        directory = self.ledger("ledger")
        digest = self.append(directory, self.entries[:4], None, keep=2)
        before = self.export(directory)
        digest = self.append(directory, self.entries[4:8], digest, keep=2)
        self.append(directory, self.entries[8:10], digest, keep=2)
        after = self.export(directory)
        self.assertEqual(
            json.loads(after)["manifest"]["dropped"]["entries"], 6
        )
        result = self.audit((before, after))
        self.assertEqual(result["status"], "failed")
        hop = result["hops"][0]
        self.assertEqual(hop["relation"], "missing_transition")
        self.assertEqual(hop["reason"], "missing_transition")
        self.assertEqual(
            result["failure"],
            {"index": 1, "relation": "missing_transition"},
        )

    def test_cross_ledger_splice_is_a_finding(self) -> None:
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
        self.assertIsNone(result["common"])
        self.assertEqual(
            [item["index"] for item in hop["before_only"]], [1]
        )
        self.assertEqual(
            [item["index"] for item in hop["after_only"]], [1]
        )
        self.assertEqual(
            result["failure"], {"index": 1, "relation": "spliced"}
        )

    def test_failure_after_successful_hops_keeps_the_shared_boundary(
        self,
    ) -> None:
        directory = self.ledger("ledger")
        digest = self.append(directory, self.entries[:4], None)
        first = self.export(directory)
        digest = self.append(directory, self.entries[4:6], digest)
        second = self.export(directory)
        # An unrelated ledger's proof as the third history splices with
        # the second; the first hop stays recorded as continued.
        other_dir = self.ledger("other")
        self.append(
            other_dir,
            tuple(
                (time, make_chain(transaction(70 + time)))
                for time in (1, 2)
            ),
            None,
        )
        third = self.export(other_dir)
        result = self.audit((first, second, third))
        self.assertEqual(result["status"], "failed")
        self.assertEqual(len(result["hops"]), 2)
        self.assertEqual(result["hops"][0]["relation"], "continued")
        self.assertEqual(result["hops"][1]["relation"], "spliced")
        self.assertEqual(result["failure"]["index"], 2)
        # A splice leaves no boundary every proof jointly authenticates.
        self.assertIsNone(result["common"])


class IsolationAndPurityTests(SequenceTestBase):
    def test_returned_levels_are_detached_between_audits(self) -> None:
        directory = self.ledger("ledger")
        digest = self.append(directory, self.entries[:4], None)
        first = self.export(directory)
        self.append(directory, self.entries[4:6], digest, keep=1)
        second = self.export(directory)
        proofs = (first, second)
        result = self.audit(proofs)
        result["common"]["digest"] = "mutated"
        hop = result["hops"][0]
        hop["after_only"][0]["entries"][0]["summary"][
            "transaction"
        ] = "mutated"
        hop["common"]["digest"] = "mutated"
        list(hop["rotated"]).append("mutated")
        again = self.audit(proofs)
        self.assertNotEqual(again["common"]["digest"], "mutated")
        self.assertNotEqual(
            again["hops"][0]["after_only"][0]["entries"][0]["summary"][
                "transaction"
            ],
            "mutated",
        )
        self.assertNotEqual(
            again["hops"][0]["common"]["digest"], "mutated"
        )
        self.assertNotIn("mutated", again["hops"][0]["rotated"])
        self.assertIsInstance(again["hops"], tuple)
        self.assertIsInstance(again["hops"][0]["after_only"], tuple)
        self.assertIsInstance(
            again["hops"][0]["after_only"][0]["entries"], tuple
        )

    def test_fork_levels_are_detached_between_audits(self) -> None:
        root, left, right = self.fork_pair()
        result = self.audit((root, left, right))
        fork = result["hops"][1]
        fork["common"]["digest"] = "mutated"
        fork["before_only"][0]["digest"] = "mutated"
        again = self.audit((root, left, right))
        fork_again = again["hops"][1]
        self.assertNotEqual(fork_again["common"]["digest"], "mutated")
        self.assertNotEqual(
            fork_again["before_only"][0]["digest"], "mutated"
        )

    def test_audit_does_not_modify_proofs_ledger_or_state(self) -> None:
        directory = self.ledger("ledger")
        digest = self.append(directory, self.entries[:4], None)
        first = self.export(directory)
        digest = self.append(directory, self.entries[4:6], digest, keep=1)
        second = self.export(directory)

        def snapshot_files():
            contents = {}
            for name in os.listdir(directory):
                with open(os.path.join(directory, name), "rb") as handle:
                    contents[name] = handle.read()
            return contents

        files = snapshot_files()
        proofs = (first, second)
        self.audit(proofs)
        # A failing audit (rollback ordering) must leave state untouched.
        self.audit((second, first))
        self.assertEqual(files, snapshot_files())
        self.assertEqual((first, second), proofs)
        self.assertEqual(second, self.export(directory))
        page = BranchStore.page_diagnostic_ledger(
            directory, None, None, None, 100, None
        )
        self.assertEqual(page["snapshot"]["digest"], digest)


if __name__ == "__main__":
    unittest.main()
