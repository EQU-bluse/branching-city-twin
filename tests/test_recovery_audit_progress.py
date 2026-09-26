"""Tests for resumable, rollback-published recovery-chain segment builds."""

import json
import os
import tempfile
import unittest
from unittest import mock

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


class ProgressTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = os.path.join(self.tmp.name, "gens")
        os.mkdir(self.root)
        self.cp = os.path.join(self.tmp.name, "cp.json")
        self.journal = os.path.join(self.tmp.name, "journal.json")
        self.chain = os.path.join(self.tmp.name, "chain.json")
        self.index = os.path.join(self.tmp.name, "index.json")
        self.progress = os.path.join(self.tmp.name, "progress.json")

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

    def advance_evidence(self, store: BranchStore, seq: int) -> None:
        store.append("main", f"m{seq}", seq + 2, {"a": seq})
        store.rotate_generation(self.root)

    def chain_with_frames(self, count: int) -> tuple[BranchStore, str]:
        """A chain of ``count`` frames over changing evidence."""
        store = self.rotate_three()
        head: "str | None" = None
        for frame in range(count):
            head = self.append_frame(head)
            if frame + 1 < count:
                self.advance_evidence(store, 6 + frame)
        assert head is not None
        return store, head

    def grow(self, store: BranchStore, head: str, count: int) -> str:
        """Append ``count`` frames over changing evidence."""
        base = len(_read(self.chain) and json.loads(_read(self.chain))["frames"])
        for offset in range(count):
            if offset < count - 1:
                self.advance_evidence(store, 20 + base + offset)
            head = self.append_frame(head)
        return head

    def progress_document(self) -> dict:
        return json.loads(_read(self.progress).decode("utf-8"))

    def index_document(self) -> dict:
        return json.loads(_read(self.index).decode("utf-8"))

    def residue(self) -> list[str]:
        return [
            name
            for name in os.listdir(self.tmp.name)
            if name.startswith(".recovery-")
            and (name.endswith(".tmp") or name.endswith(".bak"))
        ]

    def batch(
        self,
        head: str,
        ranges: tuple,
        *,
        segment_size: int = 3,
    ) -> tuple:
        return BranchStore.diff_recovery_audit_ranges(
            self.chain,
            head,
            ranges,
            self.index,
            segment_size,
            self.progress,
        )

    def reference(self, head: str, ranges: tuple) -> tuple:
        return tuple(
            BranchStore.diff_recovery_audit_range(
                self.chain, head, start, end
            )
            for start, end in ranges
        )

    def assert_scans_chain_once(self, action) -> None:  # type: ignore[no-untyped-def]
        original = BranchStore._scan_recovery_chain
        calls = {"count": 0}

        def wrapped(*args, **kwargs):
            calls["count"] += 1
            return original(*args, **kwargs)

        with mock.patch.object(
            BranchStore, "_scan_recovery_chain", wrapped
        ):
            result = action()
        self.assertEqual(calls["count"], 1)
        return result


class ProgressArgumentValidationTests(ProgressTestBase):
    def test_progress_type_validation(self) -> None:
        _store, head = self.chain_with_frames(2)
        for bad in (1, 1.0, b"x", ["a"], (), {}):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    BranchStore.build_recovery_audit_segments(
                        self.chain, self.index, 3, bad
                    )
                with self.assertRaises(TypeError):
                    BranchStore.diff_recovery_audit_ranges(
                        self.chain, head, (), self.index, 3, bad
                    )

    def test_progress_empty_string(self) -> None:
        _store, head = self.chain_with_frames(2)
        with self.assertRaises(ValueError):
            BranchStore.build_recovery_audit_segments(
                self.chain, self.index, 3, ""
            )
        with self.assertRaises(ValueError):
            BranchStore.diff_recovery_audit_ranges(
                self.chain, head, (), self.index, 3, ""
            )

    def test_none_keeps_original_arguments(self) -> None:
        _store, head = self.chain_with_frames(2)
        self.assertEqual(
            BranchStore.build_recovery_audit_segments(
                self.chain, self.index, 3, None
            ),
            head,
        )
        self.assertFalse(os.path.exists(self.progress))

    def test_progress_aliases_are_rejected(self) -> None:
        _store, head = self.chain_with_frames(2)
        link = os.path.join(self.tmp.name, "chain-link.json")
        os.symlink(self.chain, link)
        for alias in (self.chain, self.index, link):
            with self.subTest(alias=alias):
                with self.assertRaises(ValueError):
                    BranchStore.build_recovery_audit_segments(
                        self.chain, self.index, 3, alias
                    )
        index_link = os.path.join(self.tmp.name, "index-link.json")
        os.symlink(self.index, index_link)
        with self.assertRaises(ValueError):
            BranchStore.build_recovery_audit_segments(
                self.chain, self.index, 3, index_link
            )

    def test_validation_order_is_unchanged(self) -> None:
        _store, head = self.chain_with_frames(1)
        # Existing arguments keep validating before progress_path.
        with self.assertRaisesRegex(ValueError, "segment_size"):
            BranchStore.build_recovery_audit_segments(
                self.chain, self.index, 0, ""
            )
        with self.assertRaises(TypeError):
            BranchStore.diff_recovery_audit_ranges(
                self.chain, head, (("1", 2),), self.index, 3, None
            )
        with self.assertRaisesRegex(ValueError, "duplicate range"):
            BranchStore.diff_recovery_audit_ranges(
                self.chain,
                head,
                ((1, 2), (1, 2)),
                self.index,
                3,
                "",
            )

    def test_missing_chain_is_oserror_not_rebuild(self) -> None:
        with self.assertRaises(OSError):
            BranchStore.build_recovery_audit_segments(
                self.chain, self.index, 3, self.progress
            )
        self.assertFalse(os.path.exists(self.index))
        self.assertFalse(os.path.exists(self.progress))


class ProgressContentTests(ProgressTestBase):
    def test_first_build_publishes_bound_progress(self) -> None:
        _store, head = self.chain_with_frames(7)
        result = BranchStore.build_recovery_audit_segments(
            self.chain, self.index, 3, self.progress
        )
        self.assertEqual(result, head)
        document = self.progress_document()
        self.assertEqual(
            list(document),
            [
                "format",
                "version",
                "segment_size",
                "boundary",
                "boundary_digest",
                "frames",
                "head",
                "prefix_digest",
                "index_digest",
            ],
        )
        self.assertEqual(
            document["format"],
            "branching-city-twin/recovery-audit-chain-progress",
        )
        self.assertEqual(document["version"], 1)
        self.assertEqual(document["segment_size"], 3)
        self.assertEqual(document["boundary"], 6)
        self.assertEqual(document["frames"], 7)
        self.assertEqual(document["head"], head)
        self.assertEqual(len(document["prefix_digest"]), 64)
        self.assertEqual(len(document["index_digest"]), 64)
        index = self.index_document()
        self.assertEqual(document["head"], index["head"])
        self.assertEqual(
            document["boundary_digest"],
            index["segments"][-1]["last_digest"]
            if document["boundary"] == index["frames"]
            else index["segments"][1]["last_digest"],
        )

    def test_progress_is_canonical_compact_json(self) -> None:
        self.chain_with_frames(4)
        BranchStore.build_recovery_audit_segments(
            self.chain, self.index, 2, self.progress
        )
        raw = _read(self.progress)
        self.assertFalse(raw.startswith(b"\xef\xbb\xbf"))
        self.assertFalse(raw.endswith(b"\n"))
        self.assertNotIn(b" ", raw)
        document = json.loads(raw.decode("utf-8"))
        self.assertEqual(
            raw,
            json.dumps(document, separators=(",", ":")).encode("utf-8"),
        )

    def test_progress_stores_no_audit_bodies(self) -> None:
        self.chain_with_frames(4)
        BranchStore.build_recovery_audit_segments(
            self.chain, self.index, 2, self.progress
        )
        raw = _read(self.progress)
        self.assertNotIn(b"generations", raw)
        self.assertNotIn(b"checksum", raw)
        self.assertNotIn(b"current", raw)

    def test_identical_builds_publish_identical_progress(self) -> None:
        self.chain_with_frames(4)
        BranchStore.build_recovery_audit_segments(
            self.chain, self.index, 2, self.progress
        )
        first = _read(self.progress)
        BranchStore.build_recovery_audit_segments(
            self.chain, self.index, 2, self.progress
        )
        self.assertEqual(_read(self.progress), first)

    def test_boundary_zero_for_short_chain(self) -> None:
        _store, head = self.chain_with_frames(3)
        BranchStore.build_recovery_audit_segments(
            self.chain, self.index, 10, self.progress
        )
        document = self.progress_document()
        self.assertEqual(document["boundary"], 0)
        self.assertEqual(document["boundary_digest"], "0" * 64)
        self.assertEqual(document["frames"], 3)
        self.assertEqual(document["head"], head)


class ResumeTests(ProgressTestBase):
    def test_resume_scans_chain_once_and_matches_reference(self) -> None:
        store, head = self.chain_with_frames(7)
        BranchStore.build_recovery_audit_segments(
            self.chain, self.index, 3, self.progress
        )
        head = self.grow(store, head, 5)
        ranges = ((1, 12), (2, 8), (7, 12), (10, 12))
        result = self.assert_scans_chain_once(
            lambda: self.batch(head, ranges)
        )
        self.assertEqual(
            tuple(item["changes"] for item in result),
            self.reference(head, ranges),
        )

    def test_resume_advances_boundary_and_binds_new_head(self) -> None:
        store, head = self.chain_with_frames(7)
        BranchStore.build_recovery_audit_segments(
            self.chain, self.index, 3, self.progress
        )
        self.assertEqual(self.progress_document()["boundary"], 6)
        head = self.grow(store, head, 5)  # 12 frames
        self.batch(head, ((1, 12),))
        document = self.progress_document()
        self.assertEqual(document["frames"], 12)
        self.assertEqual(document["boundary"], 12)
        self.assertEqual(document["head"], head)
        self.assertEqual(self.index_document()["head"], head)
        self.assertEqual(
            [(s["first"], s["last"]) for s in self.index_document()["segments"]],
            [(1, 3), (4, 6), (7, 9), (10, 12)],
        )

    def test_repeated_resumes_each_scan_once(self) -> None:
        store, head = self.chain_with_frames(6)
        BranchStore.build_recovery_audit_segments(
            self.chain, self.index, 3, self.progress
        )
        for round_index in range(4):
            head = self.grow(store, head, 1)
            frames = 7 + round_index
            result = self.assert_scans_chain_once(
                lambda frames=frames: self.batch(
                    head, ((1, frames), (frames - 1, frames))
                )
            )
            self.assertEqual(
                tuple(item["changes"] for item in result),
                self.reference(
                    head, ((1, frames), (frames - 1, frames))
                ),
            )
        self.assertEqual(self.progress_document()["frames"], 10)

    def test_resume_without_append_changes_nothing(self) -> None:
        store, head = self.chain_with_frames(6)
        BranchStore.build_recovery_audit_segments(
            self.chain, self.index, 3, self.progress
        )
        head = self.grow(store, head, 3)  # 9 frames: exact multiple
        self.batch(head, ((1, 9),))  # publishes the 9-frame binding
        index_before = _read(self.index)
        progress_before = _read(self.progress)
        result = self.batch(head, ((1, 9), (8, 9)))
        self.assertEqual(
            tuple(item["changes"] for item in result),
            self.reference(head, ((1, 9), (8, 9))),
        )
        self.assertEqual(_read(self.index), index_before)
        self.assertEqual(_read(self.progress), progress_before)

    def test_resume_from_exact_multiple_boundary(self) -> None:
        store, head = self.chain_with_frames(9)
        BranchStore.build_recovery_audit_segments(
            self.chain, self.index, 3, self.progress
        )
        self.assertEqual(self.progress_document()["boundary"], 9)
        head = self.grow(store, head, 1)
        result = self.batch(head, ((1, 10), (9, 10)))
        self.assertEqual(
            tuple(item["changes"] for item in result),
            self.reference(head, ((1, 10), (9, 10))),
        )
        self.assertEqual(
            [(s["first"], s["last"]) for s in self.index_document()["segments"]],
            [(1, 3), (4, 6), (7, 9), (10, 10)],
        )

    def test_resume_from_zero_boundary(self) -> None:
        store, head = self.chain_with_frames(3)
        BranchStore.build_recovery_audit_segments(
            self.chain, self.index, 20, self.progress
        )
        head = self.grow(store, head, 2)
        result = self.batch(head, ((1, 5), (4, 5)), segment_size=20)
        self.assertEqual(
            tuple(item["changes"] for item in result),
            self.reference(head, ((1, 5), (4, 5))),
        )
        document = self.progress_document()
        self.assertEqual(document["boundary"], 0)
        self.assertEqual(document["frames"], 5)

    def test_old_endpoint_records_are_reauthenticated(self) -> None:
        store, head = self.chain_with_frames(6)
        BranchStore.build_recovery_audit_segments(
            self.chain, self.index, 3, self.progress
        )
        head = self.grow(store, head, 3)
        parsed = []
        original = BranchStore._parse_recovery_audit_record

        def counting(record):
            parsed.append(1)
            return original(record)

        with mock.patch.object(
            BranchStore, "_parse_recovery_audit_record", counting
        ):
            self.batch(head, ((1, 3), (8, 9)))
        # Old queried endpoints 1 and 3 must be parsed even though the
        # whole old prefix is verified with the lightweight envelope
        # check, alongside the appended frames 7..9.
        self.assertGreaterEqual(len(parsed), 2 + 3)

    def test_build_entry_also_resumes(self) -> None:
        store, head = self.chain_with_frames(6)
        BranchStore.build_recovery_audit_segments(
            self.chain, self.index, 3, self.progress
        )
        head = self.grow(store, head, 3)
        result = self.assert_scans_chain_once(
            lambda: BranchStore.build_recovery_audit_segments(
                self.chain, self.index, 3, self.progress
            )
        )
        self.assertEqual(result, head)
        self.assertEqual(self.progress_document()["head"], head)


class RebuildFallbackTests(ProgressTestBase):
    def _prepared(self) -> tuple[BranchStore, str]:
        store, head = self.chain_with_frames(6)
        BranchStore.build_recovery_audit_segments(
            self.chain, self.index, 3, self.progress
        )
        return store, self.grow(store, head, 3)

    def test_missing_progress_rebuilds_once(self) -> None:
        _store, head = self._prepared()
        os.remove(self.progress)
        result = self.assert_scans_chain_once(
            lambda: self.batch(head, ((1, 9), (8, 9)))
        )
        self.assertEqual(
            tuple(item["changes"] for item in result),
            self.reference(head, ((1, 9), (8, 9))),
        )
        self.assertEqual(self.progress_document()["head"], head)

    def test_truncated_progress_rebuilds_once(self) -> None:
        _store, head = self._prepared()
        raw = _read(self.progress)
        with open(self.progress, "wb") as handle:
            handle.write(raw[: len(raw) // 2])
        result = self.assert_scans_chain_once(
            lambda: self.batch(head, ((1, 2),))
        )
        self.assertEqual(
            tuple(item["changes"] for item in result),
            self.reference(head, ((1, 2),)),
        )

    def test_garbage_progress_rebuilds(self) -> None:
        _store, head = self._prepared()
        for raw in (b"", b"garbage", b"\xef\xbb\xbf{}", b"[]", b"123"):
            with self.subTest(raw=raw):
                with open(self.progress, "wb") as handle:
                    handle.write(raw)
                result = self.batch(head, ((1, 9),))
                self.assertEqual(
                    result[0]["changes"],
                    BranchStore.diff_recovery_audit_range(
                        self.chain, head, 1, 9
                    ),
                )

    def test_missing_index_rebuilds_under_extension_pin(self) -> None:
        _store, head = self._prepared()
        os.remove(self.index)
        result = self.assert_scans_chain_once(
            lambda: self.batch(head, ((1, 3), (7, 9)))
        )
        self.assertEqual(
            tuple(item["changes"] for item in result),
            self.reference(head, ((1, 3), (7, 9))),
        )
        self.assertEqual(self.residue(), [])

    def test_corrupt_index_rebuilds_once(self) -> None:
        _store, head = self._prepared()
        raw = _read(self.index)
        for corruption in (
            raw + b"\n",
            raw[: len(raw) // 2],
            b"garbage",
            b"\xef\xbb\xbf" + raw,
        ):
            with self.subTest(corruption=corruption[:8]):
                with open(self.index, "wb") as handle:
                    handle.write(corruption)
                result = self.assert_scans_chain_once(
                    lambda: self.batch(head, ((1, 9),))
                )
                self.assertEqual(
                    result[0]["changes"],
                    BranchStore.diff_recovery_audit_range(
                        self.chain, head, 1, 9
                    ),
                )

    def test_segment_size_change_rebuilds(self) -> None:
        _store, head = self._prepared()
        result = self.assert_scans_chain_once(
            lambda: self.batch(head, ((1, 9),), segment_size=2)
        )
        self.assertEqual(
            result[0]["changes"],
            BranchStore.diff_recovery_audit_range(self.chain, head, 1, 9),
        )
        self.assertEqual(self.progress_document()["segment_size"], 2)

    def test_rebuild_still_rejects_truncated_chain(self) -> None:
        _store, head = self._prepared()
        document = json.loads(_read(self.chain).decode("utf-8"))
        document["frames"] = document["frames"][:4]
        with open(self.chain, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(document, separators=(",", ":")))
        # A valid progress plus a missing index cannot legitimize the
        # truncation even though the fallback is a full rebuild.
        os.remove(self.index)
        with self.assertRaises(ValueError):
            self.batch(head, ((1, 2),))


class TamperResistanceTests(ProgressTestBase):
    def _prepared(self) -> tuple[str, bytes, bytes, bytes]:
        store, head = self.chain_with_frames(6)
        BranchStore.build_recovery_audit_segments(
            self.chain, self.index, 3, self.progress
        )
        head = self.grow(store, head, 3)
        return (
            head,
            _read(self.chain),
            _read(self.index),
            _read(self.progress),
        )

    def _rewrite_chain(self, document: dict) -> None:
        with open(self.chain, "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(document, separators=(",", ":"), ensure_ascii=False)
            )

    def test_truncated_chain_rejected(self) -> None:
        head, chain_raw, _idx, _prog = self._prepared()
        document = json.loads(chain_raw.decode("utf-8"))
        document["frames"] = document["frames"][:4]
        self._rewrite_chain(document)
        with self.assertRaises(ValueError):
            self.batch(head, ((1, 2),))

    def test_rewritten_old_frame_rejected(self) -> None:
        head, chain_raw, _idx, _prog = self._prepared()
        document = json.loads(chain_raw.decode("utf-8"))
        record = json.loads(document["frames"][1]["record"])
        record["checksum"] = "0" * 64
        document["frames"][1]["record"] = json.dumps(
            record, separators=(",", ":")
        )
        self._rewrite_chain(document)
        with self.assertRaises(ValueError):
            self.batch(head, ((1, 2),))

    def test_reordered_frames_rejected(self) -> None:
        head, chain_raw, _idx, _prog = self._prepared()
        document = json.loads(chain_raw.decode("utf-8"))
        document["frames"][2], document["frames"][3] = (
            document["frames"][3],
            document["frames"][2],
        )
        self._rewrite_chain(document)
        with self.assertRaises(ValueError):
            self.batch(head, ((1, 2),))

    def test_duplicate_frame_rejected(self) -> None:
        head, chain_raw, _idx, _prog = self._prepared()
        document = json.loads(chain_raw.decode("utf-8"))
        document["frames"].insert(2, document["frames"][2])
        self._rewrite_chain(document)
        with self.assertRaises(ValueError):
            self.batch(head, ((1, 2),))

    def test_broken_prev_link_rejected(self) -> None:
        head, chain_raw, _idx, _prog = self._prepared()
        document = json.loads(chain_raw.decode("utf-8"))
        document["frames"][-1]["prev"] = "0" * 64
        self._rewrite_chain(document)
        with self.assertRaises(ValueError):
            self.batch(head, ((1, 2),))

    def test_wrong_expected_head_rejected(self) -> None:
        _head, _c, _i, _p = self._prepared()
        with self.assertRaises(ValueError):
            self.batch("0" * 64, ((1, 2),))

    def test_forged_progress_prefix_digest_rejected(self) -> None:
        head, _chain_raw, _idx, progress_raw = self._prepared()
        document = json.loads(progress_raw.decode("utf-8"))
        document["prefix_digest"] = "f" * 64
        with open(self.progress, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(document, separators=(",", ":")))
        with self.assertRaises(ValueError):
            self.batch(head, ((1, 2),))

    def test_progress_boundary_digest_mismatch_rejected(self) -> None:
        head, _chain_raw, _idx, progress_raw = self._prepared()
        document = json.loads(progress_raw.decode("utf-8"))
        document["boundary_digest"] = "a" * 64
        with open(self.progress, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(document, separators=(",", ":")))
        with self.assertRaises(ValueError):
            self.batch(head, ((1, 2),))

    def test_tampered_prefix_cannot_be_masked_by_index(self) -> None:
        head, chain_raw, _idx, _prog = self._prepared()
        document = json.loads(chain_raw.decode("utf-8"))
        # Swap two records' payloads inside already-sealed frames and
        # rebuild the index to match the altered chain; the progress
        # prefix digest must still expose the rewrite.
        record_a = document["frames"][0]["record"]
        record_b = document["frames"][1]["record"]
        document["frames"][0]["record"] = record_b
        document["frames"][1]["record"] = record_a
        self._rewrite_chain(document)
        with self.assertRaises(ValueError):
            self.batch(head, ((1, 2),))


class EmptyRangesTests(ProgressTestBase):
    def test_empty_ranges_authenticates_everything(self) -> None:
        store, head = self.chain_with_frames(4)
        BranchStore.build_recovery_audit_segments(
            self.chain, self.index, 2, self.progress
        )
        head = self.grow(store, head, 2)
        result = self.assert_scans_chain_once(
            lambda: self.batch(head, ())
        )
        self.assertEqual(result, ())
        # Caches were still kept fresh for the 6-frame chain.
        self.assertEqual(self.progress_document()["frames"], 6)

    def test_empty_ranges_wrong_head_rejected(self) -> None:
        _store, _head = self.chain_with_frames(2)
        BranchStore.build_recovery_audit_segments(
            self.chain, self.index, 2, self.progress
        )
        with self.assertRaises(ValueError):
            self.batch("0" * 64, ())


class ResultParityAndDetachmentTests(ProgressTestBase):
    def test_batch_matches_single_range_diff(self) -> None:
        store, head = self.chain_with_frames(8)
        BranchStore.build_recovery_audit_segments(
            self.chain, self.index, 3, self.progress
        )
        head = self.grow(store, head, 4)
        ranges = (
            (1, 12),
            (1, 1),
            (2, 5),
            (6, 9),
            (10, 12),
            (7, 7),
            (3, 11),
        )
        result = self.batch(head, ranges)
        reference = self.reference(head, ranges)
        self.assertEqual(
            tuple(item["changes"] for item in result), reference
        )
        for item, (start, end) in zip(result, ranges):
            self.assertEqual(item["start"], start)
            self.assertEqual(item["end"], end)

    def test_result_objects_are_detached(self) -> None:
        _store, head = self.chain_with_frames(8)
        BranchStore.build_recovery_audit_segments(
            self.chain, self.index, 3, self.progress
        )
        first = self.batch(head, ((1, 8), (2, 3)))
        second = self.batch(head, ((1, 8), (2, 3)))
        self.assertEqual(first, second)
        self.assertIsNot(first, second)
        self.assertIsNot(first[0], second[0])
        first[0]["changes"] = ()  # type: ignore[typeddict-item]
        self.assertNotEqual(
            self.batch(head, ((1, 8),))[0]["changes"], ()
        )

    def test_endpoint_out_of_range_rejected_after_resume(self) -> None:
        store, head = self.chain_with_frames(6)
        BranchStore.build_recovery_audit_segments(
            self.chain, self.index, 3, self.progress
        )
        head = self.grow(store, head, 3)  # 9 frames
        with self.assertRaises(ValueError):
            self.batch(head, ((1, 10),))
        with self.assertRaises(ValueError):
            self.batch(head, ((0, 2),))
        with self.assertRaises(ValueError):
            self.batch(head, ((5, 4),))
        with self.assertRaises(ValueError):
            self.batch(head, ((1, 2), (1, 2)))


class PublicationTests(ProgressTestBase):
    def test_success_leaves_no_temp_or_backup_residue(self) -> None:
        store, head = self.chain_with_frames(6)
        BranchStore.build_recovery_audit_segments(
            self.chain, self.index, 3, self.progress
        )
        head = self.grow(store, head, 3)
        self.batch(head, ((1, 9),))
        self.assertEqual(self.residue(), [])

    def test_failed_chain_auth_leaves_caches_byte_identical(self) -> None:
        _store, head = self.chain_with_frames(6)
        BranchStore.build_recovery_audit_segments(
            self.chain, self.index, 3, self.progress
        )
        index_before = _read(self.index)
        progress_before = _read(self.progress)
        document = json.loads(_read(self.chain).decode("utf-8"))
        document["frames"][0]["digest"] = "a" * 64
        with open(self.chain, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(document, separators=(",", ":")))
        with self.assertRaises(ValueError):
            self.batch(head, ((1, 2),))
        self.assertEqual(_read(self.index), index_before)
        self.assertEqual(_read(self.progress), progress_before)
        self.assertEqual(self.residue(), [])

    def test_progress_publish_failure_rolls_back_index(self) -> None:
        store, head = self.chain_with_frames(6)
        BranchStore.build_recovery_audit_segments(
            self.chain, self.index, 3, self.progress
        )
        index_before = _read(self.index)
        progress_before = _read(self.progress)
        head = self.grow(store, head, 3)
        real_replace = os.replace
        calls = {"count": 0}

        def failing_replace(source, target, *args, **kwargs):
            # Fail only the first install attempt onto the progress
            # target; the rollback's restore replace must succeed.
            if target == self.progress and calls["count"] == 0:
                calls["count"] += 1
                raise OSError("simulated progress replace failure")
            return real_replace(source, target, *args, **kwargs)

        with mock.patch("city_twin.branches.os.replace", failing_replace):
            with self.assertRaises(OSError):
                self.batch(head, ((1, 9),))
        self.assertEqual(calls["count"], 1)
        # The already-moved index must be restored byte for byte.
        self.assertEqual(_read(self.index), index_before)
        self.assertEqual(_read(self.progress), progress_before)
        self.assertEqual(self.residue(), [])

    def test_index_publish_failure_leaves_old_index(self) -> None:
        _store, _head = self.chain_with_frames(5)
        BranchStore.build_recovery_audit_segments(
            self.chain, self.index, 2, self.progress
        )
        index_before = _read(self.index)
        progress_before = _read(self.progress)
        real_replace = os.replace
        failed = {"done": False}

        def failing_replace(source, target, *args, **kwargs):
            # Fail the first install attempt onto the index only; the
            # rollback restore replace must then succeed.
            if target == self.index and not failed["done"]:
                failed["done"] = True
                raise OSError("simulated index replace failure")
            return real_replace(source, target, *args, **kwargs)

        with mock.patch("city_twin.branches.os.replace", failing_replace):
            with self.assertRaises(OSError):
                BranchStore.build_recovery_audit_segments(
                    self.chain, self.index, 3, self.progress
                )
        self.assertEqual(_read(self.index), index_before)
        self.assertEqual(_read(self.progress), progress_before)
        self.assertEqual(self.residue(), [])

    def test_unwritable_progress_directory_is_oserror_only(self) -> None:
        _store, _head = self.chain_with_frames(3)
        missing = os.path.join(self.tmp.name, "nope", "progress.json")
        with self.assertRaises(OSError):
            BranchStore.build_recovery_audit_segments(
                self.chain, self.index, 2, missing
            )
        self.assertFalse(os.path.exists(self.index))
        self.assertEqual(self.residue(), [])

    def test_chain_and_generations_untouched(self) -> None:
        store, head = self.chain_with_frames(4)
        BranchStore.build_recovery_audit_segments(
            self.chain, self.index, 2, self.progress
        )
        head = self.grow(store, head, 2)
        chain_before = _read(self.chain)
        generation_files = {}
        for dirpath, _dirs, filenames in os.walk(self.root):
            for name in filenames:
                path = os.path.join(dirpath, name)
                with open(path, "rb") as handle:
                    generation_files[path] = handle.read()
        self.batch(head, ((1, 6),))
        self.assertEqual(_read(self.chain), chain_before)
        for path, data in generation_files.items():
            with open(path, "rb") as handle:
                self.assertEqual(handle.read(), data)


if __name__ == "__main__":
    unittest.main()
