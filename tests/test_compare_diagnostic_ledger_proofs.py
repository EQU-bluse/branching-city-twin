"""Tests for offline continuity and fork analysis between two
diagnostic ledger proofs.

``BranchStore.compare_diagnostic_ledger_proofs(before, after)`` takes
two canonical transaction-mode proof strings and, with no access to the
original ledgers, classifies their histories:

* ``same`` -- equivalent authenticated snapshots;
* ``continued`` -- ``after`` appends to ``before``, possibly across a
  rotation proved by a concrete shared transition segment, reporting the
  newly appended segments (``after_only``) and the identities that newly
  enter the prefix evidence (``rotated``);
* ``forked`` -- a concretely shared segment followed by two different
  valid successors, each side's exclusive history retained separately.

These tests additionally pin the rejection rules -- rollback (after a
strict prefix), cross-ledger splice (no authenticable shared segment),
a rotation that jumps past ``before``'s tail with only an aggregate
prefix digest connecting the ends, history rewrite contradictions and
all argument/filter validation -- the ``common`` boundary contents,
the empty-ledger genesis case, full result isolation and the absence of
filesystem side effects.
"""

import hashlib
import json
import os
import shutil
import tempfile
import unittest

from city_twin.branches import BranchStore

try:
    from test_diagnostic_ledger import make_chain, transaction
except ImportError:  # running as a package member from the repo root
    from tests.test_diagnostic_ledger import make_chain, transaction


def canonical(document: dict) -> str:
    return BranchStore._canonical_json(document)


def read_bytes(path: str) -> bytes:
    with open(path, "rb") as handle:
        return handle.read()


def reseal(document: dict) -> str:
    """Re-serialize a document with a fresh checksum over every other
    field."""
    document = dict(document)
    body = {
        key: value for key, value in document.items() if key != "checksum"
    }
    document["checksum"] = hashlib.sha256(
        canonical(body).encode("utf-8")
    ).hexdigest()
    return canonical(document)


def reseal_proof(document: dict) -> str:
    return reseal(document)


class CompareTestBase(unittest.TestCase):
    #: identities 0..7 with times 1..8
    COUNT = 8

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.chains = [
            make_chain(transaction(i)) for i in range(self.COUNT)
        ]
        self.entries = tuple(
            (time + 1, chain)
            for time, chain in enumerate(self.chains)
        )
        self.identities = tuple(
            transaction(i) for i in range(self.COUNT)
        )

    def make_dir(self, name: str) -> str:
        path = os.path.join(self.tmp.name, name)
        os.mkdir(path)
        return path

    def append(self, directory, batch, expected, keep=10, limit=2):
        return BranchStore.append_diagnostic_ledger(
            directory, batch, expected, limit, keep, None
        )

    def proof(self, directory, identities=None):
        if identities is None:
            identities = self.identities
        return BranchStore.export_diagnostic_ledger_proof(
            directory, None, None, tuple(identities)
        )

    def time_proof(self, directory, start, end):
        return BranchStore.export_diagnostic_ledger_proof(
            directory, start, end, None
        )

    def compare(self, before, after):
        return BranchStore.compare_diagnostic_ledger_proofs(before, after)

    def clone(self, name: str) -> str:
        source = os.path.join(self.tmp.name, name)
        target = os.path.join(self.tmp.name, name + "-clone")
        shutil.copytree(source, target)
        return target

    def publish(self, directory, batches, keep=10, limit=2):
        """Publish successive batches from ``self.entries``."""
        digest = None
        position = 0
        for size in batches:
            digest = self.append(
                directory,
                self.entries[position : position + size],
                digest,
                keep=keep,
                limit=limit,
            )
            position += size
        return digest

    def snapshot(self, directory):
        return BranchStore.page_diagnostic_ledger(
            directory, None, None, None, 1000, None
        )["snapshot"]


class ResultShapeTests(CompareTestBase):
    def test_key_order_and_types(self) -> None:
        before_dir = self.make_dir("before")
        self.publish(before_dir, [4], keep=10)
        after_dir = self.make_dir("after")
        self.publish(after_dir, [6], keep=10)
        result = self.compare(
            self.proof(before_dir), self.proof(after_dir)
        )
        self.assertEqual(
            list(result),
            ["relation", "common", "before_only", "after_only", "rotated"],
        )
        self.assertIsInstance(result["before_only"], tuple)
        self.assertIsInstance(result["after_only"], tuple)
        self.assertIsInstance(result["rotated"], tuple)


class SameTests(CompareTestBase):
    def test_same_snapshot(self) -> None:
        directory = self.make_dir("ledger")
        self.publish(directory, [4], keep=10)
        result = self.compare(
            self.proof(directory), self.proof(directory)
        )
        self.assertEqual(result["relation"], "same")
        self.assertEqual(result["common"]["index"], 2)
        self.assertEqual(result["common"]["entries"], 4)
        self.assertEqual(
            result["common"]["digest"],
            json.loads(self.proof(directory))["manifest"]["segments"][-1][
                "digest"
            ],
        )
        self.assertEqual(result["before_only"], ())
        self.assertEqual(result["after_only"], ())
        self.assertEqual(result["rotated"], ())

    def test_same_rotated_snapshot(self) -> None:
        directory = self.make_dir("ledger")
        self.publish(directory, [2, 2, 2], keep=1)
        result = self.compare(
            self.proof(directory), self.proof(directory)
        )
        self.assertEqual(result["relation"], "same")
        # Only segment 3 is retained; it is the common boundary with all
        # six entries cumulative.
        self.assertEqual(result["common"]["index"], 3)
        self.assertEqual(result["common"]["entries"], 6)

    def test_two_empty_snapshots_are_same(self) -> None:
        directory = self.make_dir("empty")
        self.append(directory, (), None)
        result = self.compare(
            self.proof(directory), self.proof(directory)
        )
        self.assertEqual(result["relation"], "same")
        self.assertIsNone(result["common"])
        self.assertEqual(result["before_only"], ())
        self.assertEqual(result["after_only"], ())


class ContinuedTests(CompareTestBase):
    def test_append_without_rotation(self) -> None:
        directory = self.make_dir("ledger")
        digest = self.publish(directory, [4], keep=10)
        before = self.proof(directory)
        self.append(directory, self.entries[4:6], digest, keep=10)
        after = self.proof(directory)
        result = self.compare(before, after)
        self.assertEqual(result["relation"], "continued")
        self.assertEqual(result["common"]["index"], 2)
        self.assertEqual(result["common"]["entries"], 4)
        self.assertEqual(result["before_only"], ())
        self.assertEqual(
            [segment["index"] for segment in result["after_only"]],
            [3],
        )
        segment = result["after_only"][0]
        self.assertEqual(list(segment), ["index", "digest", "entries"])
        self.assertEqual(
            [entry["at"] for entry in segment["entries"]], [5, 6]
        )
        for entry in segment["entries"]:
            self.assertEqual(list(entry), ["at", "summary", "records"])
            self.assertIsInstance(entry["records"], tuple)
        self.assertEqual(result["rotated"], ())

    def test_empty_before_starts_any_history(self) -> None:
        empty = self.make_dir("empty")
        self.append(empty, (), None)
        ledger = self.make_dir("ledger")
        self.publish(ledger, [4], keep=10)
        result = self.compare(
            self.proof(empty), self.proof(ledger)
        )
        self.assertEqual(result["relation"], "continued")
        self.assertIsNone(result["common"])
        self.assertEqual(result["before_only"], ())
        self.assertEqual(
            [segment["index"] for segment in result["after_only"]],
            [1, 2],
        )
        self.assertEqual(
            [
                entry["at"]
                for segment in result["after_only"]
                for entry in segment["entries"]
            ],
            [1, 2, 3, 4],
        )
        self.assertEqual(result["rotated"], ())

    def test_rotation_with_overlapping_transition_segment(self) -> None:
        # before keeps two segments (1,2); after appends one segment and
        # still keeps two, retaining (2,3): segment 2 is the concrete
        # transition, segment 1 rotates into the evidence.
        directory = self.make_dir("ledger")
        digest = self.publish(directory, [4], keep=2)
        before = self.proof(directory)
        self.append(directory, self.entries[4:6], digest, keep=2)
        after = self.proof(directory)
        result = self.compare(before, after)
        self.assertEqual(result["relation"], "continued")
        self.assertEqual(result["common"]["index"], 2)
        self.assertEqual(result["common"]["entries"], 4)
        self.assertEqual(
            [segment["index"] for segment in result["after_only"]],
            [3],
        )
        # Identities of segment 1 newly entered after's prefix evidence,
        # in request order.
        self.assertEqual(
            result["rotated"],
            (transaction(0), transaction(1)),
        )

    def test_rotation_swallowing_before_window_uses_concrete_tail(self):
        # before keeps one segment after six entries (segment 3); after
        # appends two entries keeping one, retaining segment 4 with
        # segments 1..3 rotated. before's concrete tail (segment 3) is
        # exactly after's prefix-end transition segment.
        directory = self.make_dir("ledger")
        digest = self.publish(directory, [2, 2, 2], keep=1)
        before = self.proof(directory)
        self.append(directory, self.entries[6:8], digest, keep=1)
        after = self.proof(directory)
        result = self.compare(before, after)
        self.assertEqual(result["relation"], "continued")
        self.assertEqual(result["common"]["index"], 3)
        self.assertEqual(result["common"]["entries"], 6)
        self.assertEqual(
            [segment["index"] for segment in result["after_only"]],
            [4],
        )
        self.assertEqual(
            [entry["at"] for entry in result["after_only"][0]["entries"]],
            [7, 8],
        )
        # Segments 3's identities were retained in before and newly
        # rotate into after's evidence; segments 1..2 were already in
        # before's evidence.
        self.assertEqual(
            result["rotated"],
            (transaction(4), transaction(5)),
        )

    def test_rotated_follows_request_order(self) -> None:
        directory = self.make_dir("ledger")
        digest = self.publish(directory, [4], keep=2)
        requested = tuple(
            transaction(i) for i in (7, 1, 0, 6, 99)
        )
        before = self.proof(directory, requested)
        self.append(directory, self.entries[4:6], digest, keep=2)
        after = self.proof(directory, requested)
        result = self.compare(before, after)
        self.assertEqual(result["relation"], "continued")
        self.assertEqual(result["rotated"], (transaction(1), transaction(0)))

    def test_adjacent_rotation_links_through_the_concrete_tail(self):
        # keep=1, two entries per append: before retains segment 1,
        # after appends segment 2 and rotates segment 1 into its prefix
        # evidence. after's prefix-end anchor is exactly before's
        # concrete tail, so the tail itself is the transition segment.
        directory = self.make_dir("ledger")
        digest = self.append(
            directory, self.entries[0:2], None, keep=1
        )
        before = self.proof(directory)
        digest = self.append(
            directory, self.entries[2:4], digest, keep=1
        )
        after = self.proof(directory)
        result = self.compare(before, after)
        self.assertEqual(result["relation"], "continued")
        self.assertEqual(result["common"]["index"], 1)
        self.assertEqual(result["common"]["entries"], 2)
        self.assertEqual(
            [segment["index"] for segment in result["after_only"]],
            [2],
        )
        self.assertEqual(
            result["rotated"], (transaction(0), transaction(1))
        )

    def test_skipping_over_the_transition_segment_is_rejected(self):
        # Same shape, but compare non-adjacent snapshots: before proves
        # only segment 1, after retains segment 3 with its prefix ending
        # on segment 2 (which before never proves). No concrete
        # transition connects the ends.
        directory = self.make_dir("ledger")
        digest = self.append(
            directory, self.entries[0:2], None, keep=1
        )
        before = self.proof(directory)
        digest = self.append(
            directory, self.entries[2:4], digest, keep=1
        )
        self.append(directory, self.entries[4:6], digest, keep=1)
        after = self.proof(directory)
        with self.assertRaises(ValueError):
            self.compare(before, after)

    def test_continued_across_two_rotations_with_overlap(self) -> None:
        # keep=2 so adjacent snapshots always share one concrete
        # transition segment even while rotating.
        directory = self.make_dir("ledger")
        digest = self.publish(directory, [4], keep=2)
        before = self.proof(directory)
        digest = self.append(
            directory, self.entries[4:6], digest, keep=2
        )
        middle = self.proof(directory)
        self.append(directory, self.entries[6:8], digest, keep=2)
        after = self.proof(directory)
        first = self.compare(before, middle)
        self.assertEqual(first["relation"], "continued")
        self.assertEqual(first["common"]["index"], 2)
        second = self.compare(middle, after)
        self.assertEqual(second["relation"], "continued")
        self.assertEqual(second["common"]["index"], 3)
        self.assertEqual(
            second["common"]["entries"], 6
        )
        self.assertEqual(
            second["rotated"], (transaction(2), transaction(3))
        )

    def test_exclusive_entries_are_fully_isolated_across_calls(self):
        side_a, side_b, identities = ForkTests._fork_pair(self)
        p_before = self.proof(side_a, identities)
        p_after = self.proof(side_b, identities)
        first = self.compare(p_before, p_after)
        # Mutate every nested level of one result.
        first["before_only"][0]["entries"][0]["records"] = ("x",)
        first["after_only"][0]["entries"][0]["summary"]["reason"] = "x"
        first["common"]["entries"] = -1
        again = self.compare(p_before, p_after)
        self.assertNotEqual(
            again["before_only"][0]["entries"][0]["records"], ("x",)
        )
        self.assertNotEqual(
            again["after_only"][0]["entries"][0]["summary"]["reason"], "x"
        )
        self.assertEqual(again["common"]["entries"], 2)


class ForkTests(CompareTestBase):
    def _fork_pair(self, keep_a=10, keep_b=10):
        """Two ledgers sharing entries 1..2 then appending different
        transaction identities; return the two directories and the
        union of requested identities."""
        base = self.make_dir("base")
        digest = self.append(
            base, self.entries[0:2], None, keep=10
        )
        side_a = os.path.join(self.tmp.name, "side-a")
        side_b = os.path.join(self.tmp.name, "side-b")
        shutil.copytree(base, side_a)
        shutil.copytree(base, side_b)
        self.append(
            side_a, self.entries[2:4], digest, keep=keep_a
        )
        alternate = tuple(
            (time, chain)
            for time, chain in zip(
                [3, 4],
                [make_chain(transaction(40)), make_chain(transaction(41))],
            )
        )
        self.append(side_b, alternate, digest, keep=keep_b)
        identities = tuple(
            transaction(i) for i in (0, 1, 2, 3, 40, 41)
        )
        return side_a, side_b, identities

    def test_fork_after_shared_segment(self) -> None:
        side_a, side_b, identities = self._fork_pair()
        result = self.compare(
            self.proof(side_a, identities),
            self.proof(side_b, identities),
        )
        self.assertEqual(result["relation"], "forked")
        self.assertEqual(result["common"]["index"], 1)
        self.assertEqual(result["common"]["entries"], 2)
        self.assertEqual(
            [segment["index"] for segment in result["before_only"]],
            [2],
        )
        self.assertEqual(
            [segment["index"] for segment in result["after_only"]],
            [2],
        )
        self.assertNotEqual(
            result["before_only"][0]["digest"],
            result["after_only"][0]["digest"],
        )
        self.assertEqual(
            [
                entry["summary"]["transaction"]
                for entry in result["before_only"][0]["entries"]
            ],
            [transaction(2), transaction(3)],
        )
        self.assertEqual(
            [
                entry["summary"]["transaction"]
                for entry in result["after_only"][0]["entries"]
            ],
            [transaction(40), transaction(41)],
        )
        self.assertEqual(result["rotated"], ())

    def test_fork_with_one_side_rotated(self) -> None:
        # Side A keeps everything; side B rotates the shared segment 1
        # away (keep=1, retains the divergent segment 2 only). The
        # predecessor segment 1 is still concrete on side A, so the fork
        # is provable.
        side_a, side_b, identities = self._fork_pair(
            keep_a=10, keep_b=1
        )
        result = self.compare(
            self.proof(side_a, identities),
            self.proof(side_b, identities),
        )
        self.assertEqual(result["relation"], "forked")
        self.assertEqual(result["common"]["index"], 1)
        self.assertEqual(result["common"]["entries"], 2)
        self.assertEqual(
            [segment["index"] for segment in result["before_only"]],
            [2],
        )
        self.assertEqual(
            [segment["index"] for segment in result["after_only"]],
            [2],
        )


class RejectionTests(CompareTestBase):
    def test_rollback_after_strict_prefix_is_value_error(self) -> None:
        # Two genuine snapshots of the same ledger at different points:
        # comparing the later snapshot as ``before`` against the earlier
        # as ``after`` is a rollback over the concrete shared prefix.
        directory = self.make_dir("ledger")
        digest = self.publish(directory, [4], keep=10)
        earlier = self.proof(directory)
        self.append(directory, self.entries[4:6], digest, keep=10)
        later = self.proof(directory)
        with self.assertRaises(ValueError):
            self.compare(later, earlier)
        # The forward direction is continued.
        forward = self.compare(earlier, later)
        self.assertEqual(forward["relation"], "continued")

    def test_rollback_across_rotation_is_value_error(self) -> None:
        directory = self.make_dir("ledger")
        digest = self.publish(directory, [2, 2, 2], keep=1)
        after = self.proof(directory)
        earlier_dir = self.make_dir("earlier-rotated")
        self.publish(earlier_dir, [2, 2], keep=1)
        before = self.proof(earlier_dir)
        with self.assertRaises(ValueError):
            self.compare(after, before)

    def test_independent_ledgers_are_cross_ledger_splices(self) -> None:
        first = self.make_dir("first")
        second = self.make_dir("second")
        # Different transaction identities from genesis.
        self.append(first, self.entries[0:2], None)
        other = tuple(
            (time, chain)
            for time, chain in zip(
                [1, 2],
                [make_chain(transaction(20)), make_chain(transaction(21))],
            )
        )
        self.append(second, other, None)
        identities = tuple(
            transaction(i) for i in (0, 1, 20, 21)
        )
        with self.assertRaises(ValueError):
            self.compare(
                self.proof(first, identities),
                self.proof(second, identities),
            )

    def test_rotation_past_tail_without_transition_is_rejected(self):
        # before proves only segment 1; after rotates past it and
        # retains segment 3 (prefix ends on segment 2, which before
        # never proves): no concrete transition segment exists.
        directory = self.make_dir("ledger")
        digest = self.append(
            directory, self.entries[0:2], None, keep=1
        )
        before = self.proof(directory)
        digest = self.append(
            directory, self.entries[2:4], digest, keep=1
        )
        self.append(directory, self.entries[4:6], digest, keep=1)
        after = self.proof(directory)
        with self.assertRaises(ValueError):
            self.compare(before, after)

    def test_shared_segment_only_via_aggregate_evidence_is_rejected(self):
        # Two ledgers share segments 1 and 2, then append divergent
        # segment 3, each with keep=1 so both shared segments rotate
        # away and only the divergent segment is retained. The alleged
        # common predecessor (segment 2) survives solely as an aggregate
        # evidence digest on both sides, never as a concrete segment.
        base = self.make_dir("base")
        digest = self.append(base, self.entries[0:2], None)
        side_a = os.path.join(self.tmp.name, "agg-a")
        side_b = os.path.join(self.tmp.name, "agg-b")
        shutil.copytree(base, side_a)
        shutil.copytree(base, side_b)
        # The common segment 2, retained briefly with keep=2.
        digest_a = self.append(side_a, self.entries[2:4], digest, keep=2)
        digest_b = self.append(side_b, self.entries[2:4], digest, keep=2)
        # Divergent segment 3 on each side at times 5..6, with keep=1:
        # both shared segments rotate away, leaving only segment 3.
        fork_a = tuple(
            (time, chain)
            for time, chain in zip(
                [5, 6],
                [make_chain(transaction(50)), make_chain(transaction(51))],
            )
        )
        fork_b = tuple(
            (time, chain)
            for time, chain in zip(
                [5, 6],
                [make_chain(transaction(40)), make_chain(transaction(41))],
            )
        )
        self.append(side_a, fork_a, digest_a, keep=1)
        self.append(side_b, fork_b, digest_b, keep=1)
        identities = tuple(
            transaction(i)
            for i in (0, 1, 2, 3, 40, 41, 50, 51)
        )
        with self.assertRaises(ValueError):
            self.compare(
                self.proof(side_a, identities),
                self.proof(side_b, identities),
            )

    def test_tampered_proof_fails_authentication(self) -> None:
        directory = self.make_dir("ledger")
        self.publish(directory, [4], keep=10)
        good = json.loads(self.proof(directory))
        tampered = json.loads(self.proof(directory))
        tampered["segments"][0]["document"]["index"] = 9
        with self.assertRaises(ValueError):
            self.compare(
                canonical(tampered), reseal_proof(good)
            )

    def test_history_rewrite_inside_shared_history_is_rejected(self):
        # Two independently authenticated proofs whose windows overlap
        # on a segment index but whose rotated-prefix anchors disagree at
        # that same position cannot be a continuous history. Build both
        # with keep=1 retaining the same segment index over disjoint
        # contents: each only proves segment 3 and its aggregate prefix.
        first = self.make_dir("first-chain")
        self.publish(first, [2, 2, 2], keep=1)
        second = self.make_dir("second-chain")
        other_entries = tuple(
            (time, chain)
            for time, chain in zip(
                range(1, 7),
                [make_chain(transaction(60 + i)) for i in range(6)],
            )
        )
        position = 0
        digest = None
        for size in (2, 2, 2):
            digest = self.append(
                second,
                other_entries[position : position + size],
                digest,
                keep=1,
            )
            position += size
        identities = tuple(
            transaction(i) for i in (*range(8), *range(60, 66))
        )
        with self.assertRaises(ValueError):
            self.compare(
                self.proof(first, identities),
                self.proof(second, identities),
            )


class ValidationTests(CompareTestBase):
    def _good_proofs(self):
        before_dir = self.make_dir("before")
        self.publish(before_dir, [2], keep=10)
        after_dir = self.make_dir("after")
        self.publish(after_dir, [4], keep=10)
        return self.proof(before_dir), self.proof(after_dir)

    def test_before_type_validation(self) -> None:
        before, after = self._good_proofs()
        for bad in (1, 1.5, b"x", ["x"], {"x": 1}, None, object()):
            with self.subTest(bad=type(bad).__name__):
                with self.assertRaises(TypeError):
                    self.compare(bad, after)

    def test_after_type_validation(self) -> None:
        before, _after = self._good_proofs()
        for bad in (1, 1.5, b"x", ["x"], {"x": 1}, None, object()):
            with self.subTest(bad=type(bad).__name__):
                with self.assertRaises(TypeError):
                    self.compare(before, bad)

    def test_arguments_validated_in_order(self) -> None:
        _before, after = self._good_proofs()
        with self.assertRaises(TypeError):
            self.compare(123, after)
        with self.assertRaises(TypeError):
            self.compare(123, 456)

    def test_empty_strings_are_value_errors(self) -> None:
        before, after = self._good_proofs()
        with self.assertRaises(ValueError):
            self.compare("", after)
        with self.assertRaises(ValueError):
            self.compare(before, "")

    def test_non_canonical_proof_is_value_error(self) -> None:
        before, after = self._good_proofs()
        with self.assertRaises(ValueError):
            self.compare(json.dumps(json.loads(before), indent=2), after)
        with self.assertRaises(ValueError):
            self.compare(before, "not a proof")

    def test_time_mode_proofs_are_rejected(self) -> None:
        before_dir = self.make_dir("before")
        self.publish(before_dir, [4], keep=10)
        after_dir = self.make_dir("after")
        self.publish(after_dir, [6], keep=10)
        time_before = self.time_proof(before_dir, 1, 4)
        time_after = self.time_proof(after_dir, 1, 6)
        txn_after = self.proof(after_dir)
        with self.assertRaises(ValueError):
            self.compare(time_before, txn_after)
        txn_before = self.proof(before_dir)
        with self.assertRaises(ValueError):
            self.compare(txn_before, time_after)
        with self.assertRaises(ValueError):
            self.compare(time_before, time_after)

    def test_identity_sets_must_match_in_content_and_order(self) -> None:
        before_dir = self.make_dir("before")
        self.publish(before_dir, [4], keep=10)
        after_dir = self.make_dir("after")
        self.publish(after_dir, [6], keep=10)
        before = self.proof(
            before_dir,
            (transaction(0), transaction(1), transaction(2)),
        )
        different = self.proof(
            after_dir,
            (transaction(0), transaction(1), transaction(99)),
        )
        with self.assertRaises(ValueError):
            self.compare(before, different)
        reordered = self.proof(
            after_dir,
            (transaction(2), transaction(1), transaction(0)),
        )
        with self.assertRaises(ValueError):
            self.compare(before, reordered)


class IsolationAndSideEffectTests(CompareTestBase):
    def test_returned_levels_are_isolated(self) -> None:
        before_dir = self.make_dir("before")
        digest = self.publish(before_dir, [4], keep=2)
        before = self.proof(before_dir)
        self.append(
            before_dir, self.entries[4:6], digest, keep=2
        )
        after = self.proof(before_dir)
        first = self.compare(before, after)
        segment = first["after_only"][0]
        segment["index"] = 999
        segment["entries"][0]["at"] = 999
        segment["entries"][0]["summary"]["transaction"] = "mutated"
        records = list(segment["entries"][0]["records"])
        records.append("mutated")
        common = first["common"]
        common["digest"] = "mutated"
        rotated = list(first["rotated"])
        rotated.append("mutated")

        second = self.compare(before, after)
        fresh_segment = second["after_only"][0]
        self.assertEqual(fresh_segment["index"], 3)
        self.assertEqual(fresh_segment["entries"][0]["at"], 5)
        self.assertNotEqual(
            fresh_segment["entries"][0]["summary"]["transaction"],
            "mutated",
        )
        self.assertNotIn("mutated", fresh_segment["entries"][0]["records"])
        self.assertNotEqual(second["common"]["digest"], "mutated")
        self.assertNotIn("mutated", second["rotated"])
        self.assertIsInstance(second["rotated"], tuple)

    def test_comparison_reads_nothing_from_a_foreign_identity(self):
        # The two proofs must be sufficient on their own: deleting the
        # ledgers before comparing must not change the answer.
        before_dir = self.make_dir("before")
        digest = self.publish(before_dir, [4], keep=2)
        before = self.proof(before_dir)
        self.append(
            before_dir, self.entries[4:6], digest, keep=2
        )
        after = self.proof(before_dir)
        expected = self.compare(before, after)
        shutil.rmtree(before_dir)
        offline = self.compare(before, after)
        self.assertEqual(offline, expected)

    def test_comparison_modifies_no_files(self) -> None:
        before_dir = self.make_dir("before")
        digest = self.publish(before_dir, [4], keep=2)
        self.append(
            before_dir, self.entries[4:6], digest, keep=2
        )

        def snapshot_files():
            return {
                name: read_bytes(os.path.join(before_dir, name))
                for name in os.listdir(before_dir)
                if os.path.isfile(os.path.join(before_dir, name))
            }

        listing_before = sorted(os.listdir(before_dir))
        snapshot_before = snapshot_files()
        for _ in range(3):
            self.compare(self.proof(before_dir), self.proof(before_dir))
        self.assertEqual(sorted(os.listdir(before_dir)), listing_before)
        self.assertEqual(snapshot_files(), snapshot_before)

    def test_proof_strings_are_not_modified(self) -> None:
        before_dir = self.make_dir("before")
        digest = self.publish(before_dir, [4], keep=10)
        before = self.proof(before_dir)
        self.append(
            before_dir, self.entries[4:6], digest, keep=10
        )
        after = self.proof(before_dir)
        before_copy = before
        after_copy = after
        self.compare(before, after)
        self.assertEqual(before, before_copy)
        self.assertEqual(after, after_copy)


class BaselineBehaviourTests(CompareTestBase):
    def test_single_proof_verify_unchanged(self) -> None:
        directory = self.make_dir("ledger")
        self.publish(directory, [2, 2, 2], keep=1)
        proof = self.proof(directory)
        result = BranchStore.verify_diagnostic_ledger_proof(proof)
        self.assertEqual(
            list(result), ["snapshot", "items", "rotated", "absent"]
        )

    def test_append_and_paging_unchanged(self) -> None:
        directory = self.make_dir("ledger")
        digest = self.publish(directory, [4], keep=3)
        digest = self.append(
            directory, self.entries[4:6], digest, keep=3
        )
        page = BranchStore.page_diagnostic_ledger(
            directory, None, None, None, 100, None
        )
        self.assertEqual(page["snapshot"]["digest"], digest)
        self.assertEqual(
            [item["at"] for item in page["items"]], [1, 2, 3, 4, 5, 6]
        )


if __name__ == "__main__":
    unittest.main()
