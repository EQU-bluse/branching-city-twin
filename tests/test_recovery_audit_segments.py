"""Tests for the segmented recovery-audit chain index and the batched
range-diff entry point."""

import hashlib
import inspect
import json
import os
import unittest
import unittest.mock

from city_twin.branches import BranchStore

from test_recovery_audit_index import (
    RecoveryAuditIndexTestBase,
    _read,
)


def _canonical(value) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def _frame_bytes(frame: dict) -> bytes:
    return _canonical(
        {
            "seq": frame["seq"],
            "prev": frame["prev"],
            "record": frame["record"],
            "digest": frame["digest"],
        }
    ).encode("utf-8")


def _segment_digest(frames: list) -> str:
    hasher = hashlib.sha256()
    for frame in frames:
        hasher.update(_frame_bytes(frame))
    return hasher.hexdigest()


class RecoveryAuditSegmentsTestBase(RecoveryAuditIndexTestBase):
    def setUp(self) -> None:
        super().setUp()
        self.segments = os.path.join(self.tmp.name, "segments.json")

    def build_segments(self, segment_size: int = 2) -> str:
        return BranchStore.build_recovery_audit_segments(
            self.chain, self.segments, segment_size
        )

    def segments_document(self) -> dict:
        return json.loads(_read(self.segments).decode("utf-8"))

    def chain_frames(self) -> list:
        document = json.loads(_read(self.chain).decode("utf-8"))
        return document["frames"]

    def expected_segments(self, segment_size: int) -> list:
        frames = self.chain_frames()
        expected = []
        for start in range(0, len(frames), segment_size):
            group = frames[start : start + segment_size]
            expected.append(
                {
                    "first": group[0]["seq"],
                    "last": group[-1]["seq"],
                    "first_digest": group[0]["digest"],
                    "last_digest": group[-1]["digest"],
                    "digest": _segment_digest(group),
                }
            )
        return expected


class BuildRecoveryAuditSegmentsTests(RecoveryAuditSegmentsTestBase):
    def test_build_returns_head_and_publishes_segments(self) -> None:
        head = self.chain_with_frames(5)
        result = self.build_segments(2)
        self.assertEqual(result, head)
        document = self.segments_document()
        self.assertEqual(
            list(document),
            [
                "format",
                "version",
                "segment_size",
                "segments",
                "chain_version",
                "frames",
                "head",
            ],
        )
        self.assertEqual(
            document["format"],
            "branching-city-twin/recovery-audit-chain-segments",
        )
        self.assertEqual(document["version"], 1)
        self.assertEqual(document["segment_size"], 2)
        self.assertEqual(document["chain_version"], 1)
        self.assertEqual(document["frames"], 5)
        self.assertEqual(document["head"], head)

    def test_segments_group_consecutive_frames(self) -> None:
        self.chain_with_frames(7)
        self.build_segments(3)
        segments = self.segments_document()["segments"]
        self.assertEqual(segments, self.expected_segments(3))
        self.assertEqual(
            [(s["first"], s["last"]) for s in segments],
            [(1, 3), (4, 6), (7, 7)],
        )
        # The last segment may be short; every segment digest covers
        # exactly its frames' canonical bytes.
        frames = self.chain_frames()
        self.assertEqual(
            segments[-1]["digest"], _segment_digest(frames[6:])
        )

    def test_segment_size_one_and_larger_than_chain(self) -> None:
        self.chain_with_frames(3)
        self.build_segments(1)
        segments = self.segments_document()["segments"]
        self.assertEqual(len(segments), 3)
        self.assertEqual(
            [(s["first"], s["last"]) for s in segments],
            [(1, 1), (2, 2), (3, 3)],
        )
        self.build_segments(100)
        segments = self.segments_document()["segments"]
        self.assertEqual(
            [(s["first"], s["last"]) for s in segments], [(1, 3)]
        )

    def test_segments_encoding_is_canonical(self) -> None:
        self.chain_with_frames(2)
        self.build_segments(2)
        raw = _read(self.segments)
        self.assertFalse(raw.startswith(b"\xef\xbb\xbf"))
        self.assertFalse(raw.endswith(b"\n"))
        self.assertNotIn(b" ", raw)
        self.assertNotIn(b"\n", raw)

    def test_same_chain_yields_identical_bytes(self) -> None:
        self.chain_with_frames(4)
        self.build_segments(2)
        first = _read(self.segments)
        os.remove(self.segments)
        self.assertEqual(self.build_segments(2), self.segments_document()["head"])
        self.assertEqual(_read(self.segments), first)
        self.build_segments(2)
        self.assertEqual(_read(self.segments), first)
        # A different segment size yields a different document.
        self.build_segments(3)
        self.assertNotEqual(_read(self.segments), first)

    def test_build_leaves_chain_and_generations_untouched(self) -> None:
        self.chain_with_frames(3)
        chain_before = _read(self.chain)
        snapshot = self.root_snapshot()
        self.build_segments(2)
        self.assertEqual(_read(self.chain), chain_before)
        self.assertEqual(self.root_snapshot(), snapshot)
        self.assertEqual(self.tmp_files(), [])

    def test_build_argument_validation(self) -> None:
        self.chain_with_frames(1)
        for bad in (1, 1.0, b"x", None, ["a"]):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    BranchStore.build_recovery_audit_segments(
                        bad, self.segments, 2
                    )
                with self.assertRaises(TypeError):
                    BranchStore.build_recovery_audit_segments(
                        self.chain, bad, 2
                    )
        with self.assertRaises(ValueError):
            BranchStore.build_recovery_audit_segments("", self.segments, 2)
        with self.assertRaises(ValueError):
            BranchStore.build_recovery_audit_segments(self.chain, "", 2)
        with self.assertRaises(ValueError):
            BranchStore.build_recovery_audit_segments(
                self.chain, self.chain, 2
            )
        for bad_size in (True, 1.0, "2", None, b"2"):
            with self.subTest(bad_size=bad_size):
                with self.assertRaises(TypeError):
                    BranchStore.build_recovery_audit_segments(
                        self.chain, self.segments, bad_size
                    )
        for small in (0, -1, -100):
            with self.subTest(small=small):
                with self.assertRaises(ValueError):
                    BranchStore.build_recovery_audit_segments(
                        self.chain, self.segments, small
                    )
        self.assertFalse(os.path.exists(self.segments))

    def test_build_has_no_default_arguments(self) -> None:
        signature = inspect.signature(
            BranchStore.build_recovery_audit_segments
        )
        for parameter in signature.parameters.values():
            self.assertIs(parameter.default, inspect.Parameter.empty)

    def test_build_missing_chain_raises_oserror(self) -> None:
        with self.assertRaises(OSError):
            self.build_segments(2)
        self.assertFalse(os.path.exists(self.segments))

    def test_build_missing_index_directory_raises_oserror(self) -> None:
        self.chain_with_frames(1)
        missing = os.path.join(self.tmp.name, "nope", "segments.json")
        with self.assertRaises(OSError):
            BranchStore.build_recovery_audit_segments(
                self.chain, missing, 2
            )

    def test_build_rejects_defective_chain_and_publishes_nothing(self) -> None:
        self.chain_with_frames(3)
        document = json.loads(_read(self.chain).decode("utf-8"))
        document["frames"][1]["prev"] = "0" * 64
        with open(self.chain, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(document, separators=(",", ":")))
        with self.assertRaises(ValueError):
            self.build_segments(2)
        self.assertFalse(os.path.exists(self.segments))
        self.assertEqual(self.tmp_files(), [])
        self.assertEqual(json.loads(_read(self.chain)), document)


class DiffRecoveryAuditRangesTests(RecoveryAuditSegmentsTestBase):
    def diff_ranges(self, head, ranges, segment_size=2):
        return BranchStore.diff_recovery_audit_ranges(
            self.chain, head, ranges, self.segments, segment_size
        )

    def test_results_match_single_range_results_in_order(self) -> None:
        head = self.chain_with_frames(4)
        ranges = ((3, 4), (1, 4), (1, 2), (2, 2), (4, 4), (1, 1))
        results = self.diff_ranges(head, ranges)
        self.assertIsInstance(results, tuple)
        self.assertEqual(len(results), len(ranges))
        for item, (start, end) in zip(results, ranges):
            self.assertEqual(list(item), ["start", "end", "changes"])
            self.assertEqual(item["start"], start)
            self.assertEqual(item["end"], end)
            self.assertEqual(
                item["changes"],
                BranchStore.diff_recovery_audit_range(
                    self.chain, head, start, end
                ),
            )
        # Genuinely different evidence yields non-empty changes.
        self.assertNotEqual(results[1]["changes"], ())
        # Equal endpoints yield empty changes.
        self.assertEqual(results[3]["changes"], ())

    def test_results_are_detached(self) -> None:
        head = self.chain_with_frames(3)
        first = self.diff_ranges(head, ((1, 3), (2, 2)))
        again = self.diff_ranges(head, ((1, 3), (2, 2)))
        self.assertEqual(first, again)
        self.assertIsNot(first, again)
        self.assertIsNot(first[0], again[0])
        self.assertIsNot(first[0]["changes"], again[0]["changes"])
        first[0]["changes"] = "tampered"
        self.assertNotEqual(
            self.diff_ranges(head, ((1, 3), (2, 2)))[0]["changes"],
            "tampered",
        )

    def test_empty_ranges_still_authenticates(self) -> None:
        head = self.chain_with_frames(2)
        self.assertFalse(os.path.exists(self.segments))
        self.assertEqual(self.diff_ranges(head, ()), ())
        # The segment index was built and authenticated anyway.
        self.assertEqual(self.segments_document()["head"], head)
        # A bad head is rejected even with nothing to diff.
        with self.assertRaises(ValueError):
            self.diff_ranges("0" * 64, ())
        # A chain defect is rejected even with nothing to diff.
        document = json.loads(_read(self.chain).decode("utf-8"))
        document["frames"][0]["digest"] = "f" * 64
        with open(self.chain, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(document, separators=(",", ":")))
        with self.assertRaises(ValueError):
            self.diff_ranges(head, ())

    def test_ranges_argument_validation(self) -> None:
        head = self.chain_with_frames(3)
        for bad in ([(1, 2)], "x", 1, None, {(1, 2)}):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    self.diff_ranges(head, bad)
        for bad_item in ([1, 2], (1,), (1, 2, 3), "12", 5, None):
            with self.subTest(bad_item=bad_item):
                with self.assertRaises(TypeError):
                    self.diff_ranges(head, (bad_item,))
        for bad_endpoint in (True, 1.0, "1", None, b"1"):
            with self.subTest(bad_endpoint=bad_endpoint):
                with self.assertRaises(TypeError):
                    self.diff_ranges(head, ((bad_endpoint, 2),))
                with self.assertRaises(TypeError):
                    self.diff_ranges(head, ((1, bad_endpoint),))
        for bad_range in ((0, 1), (1, 0), (-2, -1), (3, 2), (1, 4), (2, 9)):
            with self.subTest(bad_range=bad_range):
                with self.assertRaises(ValueError):
                    self.diff_ranges(head, (bad_range,))
        with self.assertRaises(ValueError):
            self.diff_ranges(head, ((1, 2), (1, 2)))
        with self.assertRaises(ValueError):
            self.diff_ranges(head, ((2, 3), (1, 1), (2, 3)))
        # Overlapping but distinct ranges are not duplicates.
        results = self.diff_ranges(head, ((1, 3), (2, 3), (2, 2)))
        self.assertEqual(len(results), 3)

    def test_head_and_path_validation(self) -> None:
        head = self.chain_with_frames(2)
        for bad in (1, 1.0, b"x", None, ["a"]):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    BranchStore.diff_recovery_audit_ranges(
                        bad, head, (), self.segments, 2
                    )
                with self.assertRaises(TypeError):
                    BranchStore.diff_recovery_audit_ranges(
                        self.chain, head, (), bad, 2
                    )
                with self.assertRaises(TypeError):
                    BranchStore.diff_recovery_audit_ranges(
                        self.chain, bad, (), self.segments, 2
                    )
        with self.assertRaises(ValueError):
            BranchStore.diff_recovery_audit_ranges(
                "", head, (), self.segments, 2
            )
        with self.assertRaises(ValueError):
            BranchStore.diff_recovery_audit_ranges(
                self.chain, head, (), "", 2
            )
        with self.assertRaises(ValueError):
            BranchStore.diff_recovery_audit_ranges(
                self.chain, head, (), self.chain, 2
            )
        for bad_head in ("0" * 63, "0" * 65, "g" * 64, "A" * 64, ""):
            with self.subTest(bad_head=bad_head):
                with self.assertRaises(ValueError):
                    BranchStore.diff_recovery_audit_ranges(
                        self.chain, bad_head, (), self.segments, 2
                    )
        for bad_size in (True, 1.0, "2", None):
            with self.subTest(bad_size=bad_size):
                with self.assertRaises(TypeError):
                    BranchStore.diff_recovery_audit_ranges(
                        self.chain, head, (), self.segments, bad_size
                    )
        with self.assertRaises(ValueError):
            BranchStore.diff_recovery_audit_ranges(
                self.chain, head, (), self.segments, 0
            )

    def test_diff_ranges_has_no_default_arguments(self) -> None:
        signature = inspect.signature(
            BranchStore.diff_recovery_audit_ranges
        )
        for parameter in signature.parameters.values():
            self.assertIs(parameter.default, inspect.Parameter.empty)

    def test_head_mismatch_raises(self) -> None:
        self.chain_with_frames(2)
        head = "0" * 64
        with self.assertRaises(ValueError):
            self.diff_ranges(head, ((1, 2),))

    def test_missing_chain_raises_oserror(self) -> None:
        with self.assertRaises(OSError):
            self.diff_ranges("0" * 64, ((1, 2),))

    def test_missing_and_stale_index_are_rebuilt(self) -> None:
        head = self.chain_with_frames(2)
        self.assertFalse(os.path.exists(self.segments))
        self.diff_ranges(head, ((1, 2),))
        self.assertEqual(self.segments_document()["frames"], 2)
        # Append, then the stale segment index is rebuilt on the next call.
        head = self.append_frame(head)
        self.diff_ranges(head, ((1, 3),))
        document = self.segments_document()
        self.assertEqual(document["frames"], 3)
        self.assertEqual(document["head"], head)
        # A segment index built with another segment size is stale too.
        self.diff_ranges(head, ((1, 3),), segment_size=5)
        self.assertEqual(self.segments_document()["segment_size"], 5)

    def test_corrupt_index_is_rebuilt(self) -> None:
        head = self.chain_with_frames(3)
        self.build_segments(2)
        good = _read(self.segments)
        for corrupt in (
            b"",
            b"garbage",
            b"\xef\xbb\xbf" + good,
            good + b"\n",
            good[: len(good) // 2],
        ):
            with self.subTest(corrupt=corrupt[:12]):
                with open(self.segments, "wb") as handle:
                    handle.write(corrupt)
                self.diff_ranges(head, ((1, 3),))
                self.assertEqual(_read(self.segments), good)

    def test_tampered_segment_is_rebuilt(self) -> None:
        head = self.chain_with_frames(3)
        self.build_segments(2)
        good = _read(self.segments)
        document = self.segments_document()
        # A structurally valid segment whose digest evidence does not
        # match the authenticated chain marks the whole index corrupt.
        document["segments"][0]["digest"] = "0" * 64
        with open(self.segments, "wb") as handle:
            handle.write(
                json.dumps(document, separators=(",", ":")).encode("utf-8")
            )
        self.diff_ranges(head, ((1, 3),))
        self.assertEqual(_read(self.segments), good)
        # Same for tampered boundary digests and bounds.
        document = self.segments_document()
        document["segments"][0]["last"] = 1
        with open(self.segments, "wb") as handle:
            handle.write(
                json.dumps(document, separators=(",", ":")).encode("utf-8")
            )
        self.diff_ranges(head, ((1, 3),))
        self.assertEqual(_read(self.segments), good)

    def test_still_mismatched_after_one_rebuild_raises(self) -> None:
        head = self.chain_with_frames(2)
        calls = []
        original = BranchStore._build_recovery_audit_segments_locked

        def poisoned(cls, path, index_path, segment_size):
            calls.append(1)
            result = original(path, index_path, segment_size)
            document = json.loads(_read(index_path).decode("utf-8"))
            document["segments"][0]["first_digest"] = "0" * 64
            with open(index_path, "wb") as handle:
                handle.write(
                    json.dumps(document, separators=(",", ":")).encode(
                        "utf-8"
                    )
                )
            return result

        with unittest.mock.patch.object(
            BranchStore,
            "_build_recovery_audit_segments_locked",
            classmethod(poisoned),
        ):
            with self.assertRaises(ValueError):
                self.diff_ranges(head, ((1, 2),))
        self.assertEqual(len(calls), 1)

    def test_index_does_not_mask_chain_defects(self) -> None:
        head = self.chain_with_frames(3)
        self.build_segments(2)
        document = json.loads(_read(self.chain).decode("utf-8"))
        record = json.loads(document["frames"][1]["record"])
        record["checksum"] = "0" * 64
        document["frames"][1]["record"] = json.dumps(
            record, separators=(",", ":")
        )
        with open(self.chain, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(document, separators=(",", ":")))
        with self.assertRaises(ValueError):
            self.diff_ranges(head, ((1, 2),))

    def test_queries_leave_everything_untouched(self) -> None:
        head = self.chain_with_frames(3)
        self.build_segments(2)
        chain_before = _read(self.chain)
        segments_before = _read(self.segments)
        snapshot = self.root_snapshot()
        self.diff_ranges(head, ((1, 3), (2, 2)))
        self.assertEqual(_read(self.chain), chain_before)
        self.assertEqual(_read(self.segments), segments_before)
        self.assertEqual(self.root_snapshot(), snapshot)
        self.assertEqual(self.tmp_files(), [])

    def test_failed_rebuild_propagates_and_keeps_chain(self) -> None:
        head = self.chain_with_frames(2)
        chain_before = _read(self.chain)
        missing = os.path.join(self.tmp.name, "nope", "segments.json")
        with self.assertRaises(OSError):
            BranchStore.diff_recovery_audit_ranges(
                self.chain, head, ((1, 2),), missing, 2
            )
        self.assertEqual(_read(self.chain), chain_before)

    def test_long_chain_batch(self) -> None:
        head = self.chain_with_frames(40)
        ranges = ((1, 40), (5, 35), (20, 21), (40, 40), (1, 1))
        results = self.diff_ranges(head, ranges, segment_size=7)
        for item, (start, end) in zip(results, ranges):
            self.assertEqual(
                item["changes"],
                BranchStore.diff_recovery_audit_range(
                    self.chain, head, start, end
                ),
            )
        document = self.segments_document()
        self.assertEqual(document["frames"], 40)
        self.assertEqual(len(document["segments"]), 6)


if __name__ == "__main__":
    unittest.main()
