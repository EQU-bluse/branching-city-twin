"""Tests for the rebuildable bounded-memory recovery-audit chain index."""

import json
import os
import tempfile
import unittest

from city_twin.branches import BranchStore
from city_twin.event_graph import EventGraph


def _read(path: str) -> bytes:
    with open(path, "rb") as handle:
        return handle.read()


def make_history() -> BranchStore:
    graph = EventGraph()
    graph.add("root", 0, (), {"seed": 7})
    store = BranchStore(graph)
    store.create("main", "root")
    store.create("feature", "root")
    store.append("main", "m1", 1, {"a": 1})
    store.append("feature", "f1", 2, {"b": 2})
    store.append("main", "m2", 3, {"a": 2})
    store.merge("main", "feature", "M1", 4, {"c": 1})
    return store


class RecoveryAuditIndexTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = os.path.join(self.tmp.name, "gens")
        os.mkdir(self.root)
        self.cp = os.path.join(self.tmp.name, "cp.json")
        self.journal = os.path.join(self.tmp.name, "journal.json")
        self.chain = os.path.join(self.tmp.name, "chain.json")
        self.index = os.path.join(self.tmp.name, "index.json")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def make_store(self) -> BranchStore:
        store = make_history()
        store.save_checkpoint(self.cp)
        store.enable_journal(self.cp, self.journal)
        store.append("main", "m3", 5, {"a": 3})
        return store

    def rotate_three(self) -> BranchStore:
        store = self.make_store()
        store.rotate_generation(self.root)
        store.append("main", "m4", 6, {"a": 4})
        store.rotate_generation(self.root)
        store.append("main", "m5", 7, {"a": 5})
        store.rotate_generation(self.root)
        return store

    def append_frame(self, previous: "str | None") -> str:
        return BranchStore.append_recovery_audit_chain(
            self.root, self.chain, previous
        )

    def chain_with_frames(self, count: int) -> str:
        """A chain of ``count`` frames over changing evidence; returns
        the head digest."""
        store = self.rotate_three()
        head: "str | None" = None
        for frame in range(count):
            head = self.append_frame(head)
            if frame + 1 < count:
                store.append("main", f"m{6 + frame}", 8 + frame, {"a": 6 + frame})
                store.rotate_generation(self.root)
        assert head is not None
        return head

    def build_index(self) -> str:
        return BranchStore.build_recovery_audit_index(self.chain, self.index)

    def index_document(self) -> dict:
        return json.loads(_read(self.index).decode("utf-8"))

    def root_snapshot(self) -> dict:
        captured = {}
        for dirpath, _dirnames, filenames in os.walk(self.root):
            for filename in filenames:
                path = os.path.join(dirpath, filename)
                captured[path] = _read(path)
        return captured

    def tmp_files(self) -> list:
        return [
            name
            for name in os.listdir(self.tmp.name)
            if name.startswith(".recovery-")
        ]


class BuildRecoveryAuditIndexTests(RecoveryAuditIndexTestBase):
    def test_build_returns_head_and_publishes_index(self) -> None:
        head = self.chain_with_frames(3)
        result = self.build_index()
        self.assertEqual(result, head)
        self.assertTrue(os.path.exists(self.index))
        document = self.index_document()
        self.assertEqual(
            list(document),
            ["format", "version", "entries", "chain_version", "frames", "head"],
        )
        self.assertEqual(
            document["format"],
            "branching-city-twin/recovery-audit-chain-index",
        )
        self.assertEqual(document["version"], 1)
        self.assertEqual(document["chain_version"], 1)
        self.assertEqual(document["frames"], 3)
        self.assertEqual(document["head"], head)

    def test_index_encoding_is_canonical(self) -> None:
        self.chain_with_frames(2)
        self.build_index()
        raw = _read(self.index)
        self.assertFalse(raw.startswith(b"\xef\xbb\xbf"))
        self.assertFalse(raw.endswith(b"\n"))
        self.assertNotIn(b" ", raw)
        self.assertNotIn(b"\n", raw)

    def test_index_entries_locate_frames(self) -> None:
        self.chain_with_frames(4)
        self.build_index()
        chain_bytes = _read(self.chain)
        chain_document = json.loads(chain_bytes.decode("utf-8"))
        entries = self.index_document()["entries"]
        self.assertEqual(len(entries), 4)
        for position, entry in enumerate(entries, start=1):
            self.assertEqual(
                list(entry), ["seq", "offset", "length", "digest"]
            )
            self.assertEqual(entry["seq"], position)
            fragment = chain_bytes[
                entry["offset"]: entry["offset"] + entry["length"]
            ]
            frame = json.loads(fragment.decode("utf-8"))
            self.assertEqual(frame, chain_document["frames"][position - 1])
            self.assertEqual(entry["digest"], frame["digest"])
        self.assertEqual(entries[-1]["digest"], self.index_document()["head"])

    def test_same_chain_yields_identical_bytes(self) -> None:
        self.chain_with_frames(3)
        self.build_index()
        first = _read(self.index)
        os.remove(self.index)
        self.assertEqual(self.build_index(), self.index_document()["head"])
        self.assertEqual(_read(self.index), first)
        # Rebuilding over an existing index is byte-identical too.
        self.build_index()
        self.assertEqual(_read(self.index), first)

    def test_build_leaves_chain_and_generations_untouched(self) -> None:
        self.chain_with_frames(3)
        chain_before = _read(self.chain)
        snapshot = self.root_snapshot()
        self.build_index()
        self.assertEqual(_read(self.chain), chain_before)
        self.assertEqual(self.root_snapshot(), snapshot)
        self.assertEqual(self.tmp_files(), [])

    def test_build_argument_validation(self) -> None:
        self.chain_with_frames(1)
        for bad in (1, 1.0, b"x", None, ["a"]):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    BranchStore.build_recovery_audit_index(bad, self.index)
                with self.assertRaises(TypeError):
                    BranchStore.build_recovery_audit_index(self.chain, bad)
        with self.assertRaises(ValueError):
            BranchStore.build_recovery_audit_index("", self.index)
        with self.assertRaises(ValueError):
            BranchStore.build_recovery_audit_index(self.chain, "")
        # The index must never name the chain file itself.
        with self.assertRaises(ValueError):
            BranchStore.build_recovery_audit_index(self.chain, self.chain)
        self.assertFalse(os.path.exists(self.index))

    def test_build_missing_chain_raises_oserror(self) -> None:
        with self.assertRaises(OSError):
            self.build_index()
        self.assertFalse(os.path.exists(self.index))

    def test_build_missing_index_directory_raises_oserror(self) -> None:
        self.chain_with_frames(1)
        missing = os.path.join(self.tmp.name, "nope", "index.json")
        with self.assertRaises(OSError):
            BranchStore.build_recovery_audit_index(self.chain, missing)

    def test_build_rejects_defective_chain_and_publishes_nothing(self) -> None:
        head = self.chain_with_frames(3)
        document = json.loads(_read(self.chain).decode("utf-8"))
        document["frames"][1]["prev"] = "0" * 64
        with open(self.chain, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(document, separators=(",", ":")))
        with self.assertRaises(ValueError):
            self.build_index()
        self.assertFalse(os.path.exists(self.index))
        self.assertEqual(self.tmp_files(), [])
        # The defective chain is left byte-for-byte as it was.
        self.assertEqual(json.loads(_read(self.chain)), document)
        self.assertIsInstance(head, str)


class VerifyWithIndexTests(RecoveryAuditIndexTestBase):
    def test_verify_matches_no_index_behaviour(self) -> None:
        head = self.chain_with_frames(3)
        self.build_index()
        self.assertIs(
            BranchStore.verify_recovery_audit_chain(self.chain, head, self.index),
            True,
        )
        self.assertIs(
            BranchStore.verify_recovery_audit_chain(
                self.chain, "0" * 64, self.index
            ),
            False,
        )
        self.assertIs(
            BranchStore.verify_recovery_audit_chain(
                self.chain, head, index_path=self.index
            ),
            True,
        )

    def test_verify_rebuilds_missing_index(self) -> None:
        head = self.chain_with_frames(2)
        self.assertFalse(os.path.exists(self.index))
        self.assertIs(
            BranchStore.verify_recovery_audit_chain(self.chain, head, self.index),
            True,
        )
        self.assertEqual(self.index_document()["head"], head)

    def test_verify_rebuilds_stale_index(self) -> None:
        head = self.chain_with_frames(2)
        self.build_index()
        stale_bytes = _read(self.index)
        head = self.append_frame(head)
        self.assertIs(
            BranchStore.verify_recovery_audit_chain(self.chain, head, self.index),
            True,
        )
        document = self.index_document()
        self.assertEqual(document["head"], head)
        self.assertEqual(document["frames"], 3)
        self.assertNotEqual(_read(self.index), stale_bytes)

    def test_verify_rebuilds_corrupt_index(self) -> None:
        head = self.chain_with_frames(2)
        self.build_index()
        for corrupt in (
            b"",
            b"garbage",
            b"\xef\xbb\xbf" + _read(self.index),
            _read(self.index) + b"\n",
            _read(self.index)[: len(_read(self.index)) // 2],
        ):
            with self.subTest(corrupt=corrupt[:12]):
                with open(self.index, "wb") as handle:
                    handle.write(corrupt)
                self.assertIs(
                    BranchStore.verify_recovery_audit_chain(
                        self.chain, head, self.index
                    ),
                    True,
                )
                self.assertEqual(self.index_document()["head"], head)

    def test_verify_does_not_trust_index_over_chain(self) -> None:
        head = self.chain_with_frames(3)
        self.build_index()
        # Corrupt an embedded record's checksum; the index still binds
        # the old head, but the chain defect must surface.
        document = json.loads(_read(self.chain).decode("utf-8"))
        record = json.loads(document["frames"][1]["record"])
        record["checksum"] = "0" * 64
        document["frames"][1]["record"] = json.dumps(
            record, separators=(",", ":")
        )
        with open(self.chain, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(document, separators=(",", ":")))
        with self.assertRaises(ValueError):
            BranchStore.verify_recovery_audit_chain(
                self.chain, head, self.index
            )

    def test_verify_with_index_leaves_everything_untouched(self) -> None:
        head = self.chain_with_frames(2)
        self.build_index()
        chain_before = _read(self.chain)
        index_before = _read(self.index)
        snapshot = self.root_snapshot()
        BranchStore.verify_recovery_audit_chain(self.chain, head, self.index)
        self.assertEqual(_read(self.chain), chain_before)
        self.assertEqual(_read(self.index), index_before)
        self.assertEqual(self.root_snapshot(), snapshot)
        self.assertEqual(self.tmp_files(), [])

    def test_verify_index_argument_validation(self) -> None:
        head = self.chain_with_frames(1)
        for bad in (1, 1.0, b"x", ["a"]):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    BranchStore.verify_recovery_audit_chain(
                        self.chain, head, bad
                    )
        with self.assertRaises(ValueError):
            BranchStore.verify_recovery_audit_chain(self.chain, head, "")
        with self.assertRaises(ValueError):
            BranchStore.verify_recovery_audit_chain(
                self.chain, head, self.chain
            )
        # None keeps the no-index behaviour.
        self.assertIs(
            BranchStore.verify_recovery_audit_chain(self.chain, head, None),
            True,
        )

    def test_verify_with_index_missing_chain_raises_oserror(self) -> None:
        with self.assertRaises(OSError):
            BranchStore.verify_recovery_audit_chain(
                self.chain, "0" * 64, self.index
            )


class DiffWithIndexTests(RecoveryAuditIndexTestBase):
    def test_diff_matches_no_index_result(self) -> None:
        head = self.chain_with_frames(4)
        self.build_index()
        for start, end in ((1, 4), (1, 2), (2, 3), (3, 4), (1, 1), (4, 4)):
            with self.subTest(start=start, end=end):
                expected = BranchStore.diff_recovery_audit_range(
                    self.chain, head, start, end
                )
                indexed = BranchStore.diff_recovery_audit_range(
                    self.chain, head, start, end, self.index
                )
                self.assertEqual(indexed, expected)
        # A range over genuinely different evidence is non-empty.
        self.assertNotEqual(
            BranchStore.diff_recovery_audit_range(
                self.chain, head, 1, 4, self.index
            ),
            (),
        )

    def test_diff_result_is_detached(self) -> None:
        head = self.chain_with_frames(3)
        first = BranchStore.diff_recovery_audit_range(
            self.chain, head, 1, 3, self.index
        )
        again = BranchStore.diff_recovery_audit_range(
            self.chain, head, 1, 3, self.index
        )
        self.assertEqual(first, again)
        self.assertIsNot(first, again)

    def test_diff_rebuilds_missing_or_stale_index(self) -> None:
        head = self.chain_with_frames(2)
        expected = BranchStore.diff_recovery_audit_range(
            self.chain, head, 1, 2
        )
        self.assertFalse(os.path.exists(self.index))
        self.assertEqual(
            BranchStore.diff_recovery_audit_range(
                self.chain, head, 1, 2, self.index
            ),
            expected,
        )
        self.assertEqual(self.index_document()["frames"], 2)
        # Append, then the stale index is rebuilt during the next diff.
        head = self.append_frame(head)
        result = BranchStore.diff_recovery_audit_range(
            self.chain, head, 1, 3, self.index
        )
        self.assertEqual(
            result,
            BranchStore.diff_recovery_audit_range(self.chain, head, 1, 3),
        )
        self.assertEqual(self.index_document()["frames"], 3)

    def test_diff_with_index_head_mismatch_raises(self) -> None:
        self.chain_with_frames(2)
        self.build_index()
        with self.assertRaises(ValueError):
            BranchStore.diff_recovery_audit_range(
                self.chain, "0" * 64, 1, 2, self.index
            )

    def test_diff_with_index_range_validation(self) -> None:
        head = self.chain_with_frames(3)
        self.build_index()
        for start, end in ((0, 1), (1, 0), (-2, -1), (3, 2), (1, 4), (2, 9)):
            with self.subTest(start=start, end=end):
                with self.assertRaises(ValueError):
                    BranchStore.diff_recovery_audit_range(
                        self.chain, head, start, end, self.index
                    )
        for bad in (True, 1.0, "1", None, b"1"):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    BranchStore.diff_recovery_audit_range(
                        self.chain, head, bad, 2, self.index
                    )
                with self.assertRaises(TypeError):
                    BranchStore.diff_recovery_audit_range(
                        self.chain, head, 1, bad, self.index
                    )

    def test_diff_does_not_trust_index_over_chain(self) -> None:
        head = self.chain_with_frames(3)
        self.build_index()
        # Break the predecessor link of the last frame; the fresh index
        # must not mask the defect.
        document = json.loads(_read(self.chain).decode("utf-8"))
        document["frames"][2]["prev"] = "0" * 64
        with open(self.chain, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(document, separators=(",", ":")))
        with self.assertRaises(ValueError):
            BranchStore.diff_recovery_audit_range(
                self.chain, head, 1, 2, self.index
            )

    def test_diff_with_index_argument_validation(self) -> None:
        head = self.chain_with_frames(1)
        for bad in (1, 1.0, b"x", ["a"]):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    BranchStore.diff_recovery_audit_range(
                        self.chain, head, 1, 1, bad
                    )
        with self.assertRaises(ValueError):
            BranchStore.diff_recovery_audit_range(
                self.chain, head, 1, 1, ""
            )
        with self.assertRaises(ValueError):
            BranchStore.diff_recovery_audit_range(
                self.chain, head, 1, 1, self.chain
            )
        # None keeps the no-index behaviour.
        self.assertEqual(
            BranchStore.diff_recovery_audit_range(
                self.chain, head, 1, 1, None
            ),
            (),
        )


class IndexedQueryDefectTests(RecoveryAuditIndexTestBase):
    """Chain defects must surface identically through the indexed path."""

    def assert_both_reject(self, head: str) -> None:
        with self.assertRaises(ValueError):
            BranchStore.verify_recovery_audit_chain(
                self.chain, head, self.index
            )
        with self.assertRaises(ValueError):
            BranchStore.diff_recovery_audit_range(
                self.chain, head, 1, 1, self.index
            )

    def rewrite_chain(self, document: dict) -> None:
        with open(self.chain, "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(document, separators=(",", ":"), ensure_ascii=False)
            )

    def test_truncated_chain(self) -> None:
        head = self.chain_with_frames(3)
        self.build_index()
        raw = _read(self.chain)
        with open(self.chain, "wb") as handle:
            handle.write(raw[: len(raw) // 2])
        self.assert_both_reject(head)

    def test_non_canonical_chain(self) -> None:
        head = self.chain_with_frames(2)
        self.build_index()
        raw = _read(self.chain).replace(b'"version":1', b'"version": 1', 1)
        with open(self.chain, "wb") as handle:
            handle.write(raw)
        self.assert_both_reject(head)

    def test_duplicate_key(self) -> None:
        head = self.chain_with_frames(2)
        self.build_index()
        raw = _read(self.chain).replace(
            b'"version":1', b'"version":1,"version":1', 1
        )
        with open(self.chain, "wb") as handle:
            handle.write(raw)
        self.assert_both_reject(head)

    def test_frame_digest_error(self) -> None:
        head = self.chain_with_frames(2)
        self.build_index()
        document = json.loads(_read(self.chain).decode("utf-8"))
        document["frames"][0]["digest"] = "f" * 64
        self.rewrite_chain(document)
        self.assert_both_reject(head)

    def test_bom_bearing_chain(self) -> None:
        head = self.chain_with_frames(1)
        self.build_index()
        with open(self.chain, "wb") as handle:
            handle.write(b"\xef\xbb\xbf" + _read(self.chain))
        self.assert_both_reject(head)


class IndexedQuerySideEffectTests(RecoveryAuditIndexTestBase):
    def test_queries_leave_chain_generations_and_logs_untouched(self) -> None:
        head = self.chain_with_frames(3)
        chain_before = _read(self.chain)
        snapshot = self.root_snapshot()
        BranchStore.verify_recovery_audit_chain(self.chain, head, self.index)
        BranchStore.diff_recovery_audit_range(self.chain, head, 1, 3, self.index)
        self.assertEqual(_read(self.chain), chain_before)
        self.assertEqual(self.root_snapshot(), snapshot)
        self.assertEqual(self.tmp_files(), [])

    def test_failed_rebuild_propagates_and_keeps_chain(self) -> None:
        head = self.chain_with_frames(2)
        chain_before = _read(self.chain)
        # The index's parent directory does not exist, so the rebuild
        # cannot publish and must surface OSError.
        missing = os.path.join(self.tmp.name, "nope", "index.json")
        with self.assertRaises(OSError):
            BranchStore.verify_recovery_audit_chain(self.chain, head, missing)
        self.assertEqual(_read(self.chain), chain_before)

    def test_lock_file_is_the_only_residue(self) -> None:
        head = self.chain_with_frames(1)
        BranchStore.verify_recovery_audit_chain(self.chain, head, self.index)
        self.assertTrue(
            os.path.exists(self.chain + ".lock")
            or os.path.exists(os.path.realpath(self.chain) + ".lock")
        )
        self.assertEqual(self.tmp_files(), [])


class IndexedQueryLongChainTests(RecoveryAuditIndexTestBase):
    def test_long_chain_queries(self) -> None:
        head = self.chain_with_frames(40)
        self.build_index()
        self.assertIs(
            BranchStore.verify_recovery_audit_chain(self.chain, head, self.index),
            True,
        )
        for start, end in ((1, 40), (5, 35), (20, 21), (40, 40)):
            with self.subTest(start=start, end=end):
                self.assertEqual(
                    BranchStore.diff_recovery_audit_range(
                        self.chain, head, start, end, self.index
                    ),
                    BranchStore.diff_recovery_audit_range(
                        self.chain, head, start, end
                    ),
                )
        document = self.index_document()
        self.assertEqual(document["frames"], 40)
        self.assertEqual(len(document["entries"]), 40)


if __name__ == "__main__":
    unittest.main()
