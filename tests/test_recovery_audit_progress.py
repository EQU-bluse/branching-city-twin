"""Tests for resumable, rollback-safe recovery-chain segment builds.

The segmented build and the batch range-diff entry gain an optional
progress cursor. These tests cover:

* omitted ``progress_path`` keeping the baseline call, result and
  authentication semantics;
* a sound cursor letting a batch scan only the appended frames (the
  canonical chain is scanned at most once, not per range);
* missing, truncated or mismatching cursor/index triggering exactly
  one full rebuild;
* truncated, rewritten, reordered or duplicated chains being rejected
  even when a readable cursor exists;
* deterministic compact cursor bytes that bind the boundary, prefix
  and index evidence without ever storing an audit body;
* rollback-safe publication restoring the old index byte-for-byte when
  publishing progress fails;
* argument validation order and type/value errors.
"""

import json
import os
import tempfile
import threading
import unittest
from unittest import mock

from city_twin.branches import BranchStore
from city_twin.event_graph import EventGraph


def _read(path: str) -> bytes:
    with open(path, "rb") as handle:
        return handle.read()


def make_store() -> BranchStore:
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


class RecoveryProgressTestBase(unittest.TestCase):
    SEGMENT_SIZE = 2

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = os.path.join(self.tmp.name, "gens")
        os.mkdir(self.root)
        self.cp = os.path.join(self.tmp.name, "cp.json")
        self.journal = os.path.join(self.tmp.name, "journal.json")
        self.chain = os.path.join(self.tmp.name, "chain.json")
        self.index = os.path.join(self.tmp.name, "index.json")
        self.progress = os.path.join(self.tmp.name, "progress.json")
        store = make_store()
        store.save_checkpoint(self.cp)
        store.enable_journal(self.cp, self.journal)
        store.append("main", "m3", 5, {"a": 3})
        store.rotate_generation(self.root)
        self.store = store
        self.head: str | None = None

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def append_frame(self, previous: "str | None") -> str:
        return BranchStore.append_recovery_audit_chain(
            self.root, self.chain, previous
        )

    def grow_chain(self, count: int, start: int = 0) -> str:
        head = self.head
        for i in range(count):
            head = self.append_frame(head)
            if i + 1 < count:
                self.store.append(
                    "main", f"g{start + i}", 100 + start + i,
                    {"a": start + i},
                )
                self.store.rotate_generation(self.root)
        self.head = head
        return head

    def build(self, *args: object) -> str:
        return BranchStore.build_recovery_audit_segments(  # type: ignore[return-value]
            self.chain, self.index, self.SEGMENT_SIZE, *args
        )

    def diff(self, ranges, *args: object):
        return BranchStore.diff_recovery_audit_ranges(
            self.chain,
            self.head,
            tuple(ranges),
            self.index,
            self.SEGMENT_SIZE,
            *args,
        )

    def progress_document(self) -> dict:
        return json.loads(_read(self.progress).decode("utf-8"))

    def index_document(self) -> dict:
        return json.loads(_read(self.index).decode("utf-8"))

    def chain_document(self) -> dict:
        return json.loads(_read(self.chain).decode("utf-8"))

    def write_chain(self, document: dict) -> None:
        with open(self.chain, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(document, separators=(",", ":")))

    def root_snapshot(self) -> dict:
        captured = {}
        for dirpath, _dirnames, filenames in os.walk(self.root):
            for filename in filenames:
                path = os.path.join(dirpath, filename)
                captured[path] = _read(path)
        return captured

    def residue(self) -> list:
        return [
            name
            for name in os.listdir(self.tmp.name)
            if name.startswith(".recovery-")
        ]


class BuildWithProgressTests(RecoveryProgressTestBase):
    def test_initial_build_publishes_index_and_progress(self) -> None:
        head = self.grow_chain(5)
        result = self.build(self.progress)
        self.assertEqual(result, head)
        progress = self.progress_document()
        self.assertEqual(
            list(progress),
            [
                "format",
                "version",
                "segment_size",
                "frames",
                "head",
                "boundary",
                "prefix",
                "index",
            ],
        )
        self.assertEqual(
            progress["format"],
            "branching-city-twin/recovery-audit-chain-progress",
        )
        self.assertEqual(progress["version"], 1)
        self.assertEqual(progress["segment_size"], self.SEGMENT_SIZE)
        self.assertEqual(progress["frames"], 5)
        self.assertEqual(progress["head"], head)
        # The boundary is the last frame completing a segment.
        self.assertEqual(progress["boundary"]["seq"], 4)
        self.assertEqual(len(progress["boundary"]["digest"]), 64)
        self.assertEqual(len(progress["prefix"]), 64)
        self.assertEqual(len(progress["index"]), 64)
        # The index binds the same frame count and head.
        self.assertEqual(self.index_document()["frames"], 5)
        self.assertEqual(self.index_document()["head"], head)

    def test_progress_bytes_are_canonical_compact(self) -> None:
        self.grow_chain(3)
        self.build(self.progress)
        raw = _read(self.progress)
        self.assertFalse(raw.startswith(b"\xef\xbb\xbf"))
        self.assertFalse(raw.endswith(b"\n"))
        self.assertNotIn(b" ", raw)
        self.assertNotIn(b"\n", raw)
        # Deterministic: rebuilding the same chain writes the same bytes.
        first = _read(self.progress)
        self.build(self.progress)
        self.assertEqual(_read(self.progress), first)

    def test_progress_never_stores_an_audit_body(self) -> None:
        head = self.grow_chain(4)
        self.build(self.progress)
        raw = _read(self.progress)
        record = self.chain_document()["frames"][0]["record"]
        # The embedded audit record is long and distinctive; none of its
        # content may appear in the progress cursor.
        self.assertGreater(len(record), 64)
        self.assertNotIn(record.encode("utf-8"), raw)
        for token in (b"current", b"selected", b"generations", b"pointer"):
            self.assertNotIn(token, raw)
        self.assertEqual(self.progress_document()["head"], head)

    def test_no_progress_file_without_argument(self) -> None:
        self.grow_chain(3)
        self.build()
        self.assertTrue(os.path.exists(self.index))
        self.assertFalse(os.path.exists(self.progress))

    def test_chain_shorter_than_a_segment_publishes_no_progress(self) -> None:
        self.grow_chain(1)
        result = self.build(self.progress)
        self.assertEqual(result, self.head)
        self.assertTrue(os.path.exists(self.index))
        self.assertFalse(os.path.exists(self.progress))

    def test_progress_is_independent_of_index_caches(self) -> None:
        head = self.grow_chain(4)
        self.build(self.progress)
        os.remove(self.index)
        os.remove(self.progress)
        # A fresh process state rebuilds purely from the canonical chain.
        self.assertEqual(self.build(self.progress), head)
        self.assertEqual(self.progress_document()["head"], head)


class ResumeTests(RecoveryProgressTestBase):
    def _patch_scanners(self) -> dict:
        calls = {"resume": 0, "full": 0}
        resume_orig = BranchStore._scan_chain_resuming
        full_orig = BranchStore._scan_recovery_chain

        def resume_wrap(*args, **kwargs):
            calls["resume"] += 1
            return resume_orig(*args, **kwargs)

        def full_wrap(*args, **kwargs):
            calls["full"] += 1
            return full_orig(*args, **kwargs)

        resume_patch = mock.patch.object(
            BranchStore, "_scan_chain_resuming", staticmethod(resume_wrap)
        )
        full_patch = mock.patch.object(
            BranchStore, "_scan_recovery_chain", staticmethod(full_wrap)
        )
        resume_patch.start()
        full_patch.start()
        self.addCleanup(resume_patch.stop)
        self.addCleanup(full_patch.stop)
        return calls

    def test_append_resumes_from_boundary(self) -> None:
        self.grow_chain(5)
        self.build(self.progress)
        calls = self._patch_scanners()
        head = self.grow_chain(3, start=10)
        result = self.build(self.progress)
        self.assertEqual(result, head)
        self.assertEqual(calls, {"resume": 1, "full": 0})
        progress = self.progress_document()
        self.assertEqual(progress["frames"], 8)
        self.assertEqual(progress["head"], head)
        # 8 frames with size 2 -> boundary is frame 8.
        self.assertEqual(progress["boundary"]["seq"], 8)

    def test_partial_trailing_segment_then_append(self) -> None:
        # 4 frames complete two segments; a fifth frame starts a short
        # trailing segment, whose frames are *not* treated as a boundary.
        self.grow_chain(5)
        self.build(self.progress)
        self.assertEqual(self.progress_document()["boundary"]["seq"], 4)
        self.grow_chain(3, start=20)
        result = self.build(self.progress)
        self.assertEqual(result, self.head)
        self.assertEqual(self.progress_document()["boundary"]["seq"], 8)

    def test_tail_spanning_multiple_segments(self) -> None:
        self.SEGMENT_SIZE = 3
        self.grow_chain(3)
        self.build(self.progress)
        self.grow_chain(7, start=60)
        result = self.build(self.progress)
        self.assertEqual(result, self.head)
        progress = self.progress_document()
        # 10 frames with segment size 3 -> complete boundary is 9.
        self.assertEqual(progress["frames"], 10)
        self.assertEqual(progress["boundary"]["seq"], 9)
        frame9 = self.chain_document()["frames"][8]
        self.assertEqual(progress["boundary"]["digest"], frame9["digest"])
        # The recorded byte span must locate frame 9 exactly.
        raw = _read(self.chain)
        boundary = progress["boundary"]
        fragment = json.loads(
            raw[boundary["offset"]: boundary["offset"] + boundary["length"]]
        )
        self.assertEqual(fragment, frame9)
        # The recorded prefix is the SHA-256 of every raw chain byte
        # through frame 9 (the boundary offset plus its length).
        import hashlib

        expected_prefix = hashlib.sha256(
            raw[: boundary["offset"] + boundary["length"]]
        ).hexdigest()
        self.assertEqual(progress["prefix"], expected_prefix)
        # Appending across the partial segment resumes again cleanly.
        self.grow_chain(2, start=80)
        self.build(self.progress)
        self.assertEqual(self.progress_document()["boundary"]["seq"], 12)

    def test_batch_tail_ranges_scan_chain_once_and_match(self) -> None:
        self.grow_chain(6)
        self.build(self.progress)
        calls = self._patch_scanners()
        self.grow_chain(4, start=30)
        ranges = ((7, 10), (8, 9), (10, 10), (7, 7))
        results = self.diff(ranges, self.progress)
        self.assertEqual(calls, {"resume": 1, "full": 0})
        self.assertEqual([(r["start"], r["end"]) for r in results], list(ranges))
        for result, (start, end) in zip(results, ranges):
            expected = BranchStore.diff_recovery_audit_range(
                self.chain, self.head, start, end
            )
            self.assertEqual(result["changes"], expected)

    def test_batch_results_are_detached(self) -> None:
        self.grow_chain(8)
        self.build(self.progress)
        first = self.diff(((1, 8),), self.progress)
        second = self.diff(((1, 8),), self.progress)
        self.assertEqual(first, second)
        self.assertIsNot(first, second)
        self.assertIsNot(first[0], second[0])

    def test_prefix_query_forces_one_full_rebuild(self) -> None:
        self.grow_chain(4)
        self.build(self.progress)
        calls = self._patch_scanners()
        self.grow_chain(4, start=40)
        # Endpoint 2 lies inside the recorded boundary.
        results = self.diff(((2, 8),), self.progress)
        self.assertGreaterEqual(calls["full"], 1)
        self.assertEqual(calls["resume"], 0)
        expected = BranchStore.diff_recovery_audit_range(
            self.chain, self.head, 2, 8
        )
        self.assertEqual(results[0]["changes"], expected)
        # A second tail-only query resumes again from the new cursor.
        calls.update(resume=0, full=0)
        self.grow_chain(2, start=50)
        self.diff(((9, 10),), self.progress)
        self.assertEqual(calls["resume"], 1)
        self.assertEqual(calls["full"], 0)

    def test_unchanged_chain_query_leaves_caches_untouched(self) -> None:
        self.grow_chain(6)
        self.build(self.progress)
        index_before = _read(self.index)
        progress_before = _read(self.progress)
        self.diff(((5, 6),), self.progress)
        self.assertEqual(_read(self.index), index_before)
        self.assertEqual(_read(self.progress), progress_before)
        self.assertEqual(self.residue(), [])


class RebuildFallbackTests(RecoveryProgressTestBase):
    def test_corrupt_progress_forces_full_rebuild(self) -> None:
        head = self.grow_chain(4)
        self.build(self.progress)
        for corrupt in (
            b"",
            b"garbage",
            b"\xef\xbb\xbf" + _read(self.progress),
            _read(self.progress)[: len(_read(self.progress)) // 2],
            _read(self.progress) + b"\n",
        ):
            with self.subTest(corrupt=corrupt[:12]):
                with open(self.progress, "wb") as handle:
                    handle.write(corrupt)
                self.assertEqual(self.build(self.progress), head)
                self.assertEqual(self.progress_document()["head"], head)

    def test_progress_with_wrong_field_forces_full_rebuild(self) -> None:
        self.grow_chain(4)
        self.build(self.progress)
        document = self.progress_document()
        document["frames"] = 3
        with open(self.progress, "w", encoding="utf-8") as handle:
            json.dump(document, handle, separators=(",", ":"))
        # The chain was not actually truncated, so the full rebuild
        # repopulates a correct cursor.
        self.assertEqual(self.build(self.progress), self.head)
        self.assertEqual(self.progress_document()["frames"], 4)

    def test_corrupt_index_forces_full_rebuild(self) -> None:
        head = self.grow_chain(4)
        self.build(self.progress)
        with open(self.index, "wb") as handle:
            handle.write(_read(self.index)[:30])
        self.assertEqual(self.build(self.progress), head)
        self.assertEqual(self.index_document()["head"], head)

    def test_segment_size_change_forces_full_rebuild(self) -> None:
        head = self.grow_chain(6)
        self.build(self.progress)
        BranchStore.build_recovery_audit_segments(
            self.chain, self.index, 3, self.progress
        )
        self.assertEqual(self.progress_document()["segment_size"], 3)
        self.assertEqual(
            BranchStore.build_recovery_audit_segments(
                self.chain, self.index, 3, self.progress
            ),
            head,
        )

    def test_missing_progress_and_index_build_from_chain(self) -> None:
        head = self.grow_chain(4)
        self.assertEqual(self.build(self.progress), head)
        self.assertTrue(os.path.exists(self.index))
        self.assertTrue(os.path.exists(self.progress))


class TamperRejectionTests(RecoveryProgressTestBase):
    def setUp(self) -> None:
        super().setUp()
        self.grow_chain(6)
        self.build(self.progress)
        self.good_chain = _read(self.chain)
        self.good_index = _read(self.index)
        self.good_progress = _read(self.progress)

    def restore(self) -> None:
        with open(self.chain, "wb") as handle:
            handle.write(self.good_chain)

    def assert_rejected_and_untouched(self) -> None:
        with self.assertRaises(ValueError):
            self.build(self.progress)
        self.assertEqual(_read(self.index), self.good_index)
        self.assertEqual(_read(self.progress), self.good_progress)
        self.assertEqual(self.residue(), [])

    def test_truncated_chain(self) -> None:
        document = self.chain_document()
        self.write_chain({**document, "frames": document["frames"][:3]})
        self.assert_rejected_and_untouched()
        self.restore()

    def test_rewritten_prefix_frame(self) -> None:
        document = self.chain_document()
        document["frames"][1]["record"] = document["frames"][2]["record"]
        self.write_chain(document)
        self.assert_rejected_and_untouched()
        self.restore()

    def test_reordered_frames(self) -> None:
        document = self.chain_document()
        frames = document["frames"]
        frames[2], frames[3] = frames[3], frames[2]
        self.write_chain(document)
        self.assert_rejected_and_untouched()
        self.restore()

    def test_duplicated_frame(self) -> None:
        document = self.chain_document()
        frames = document["frames"]
        # Repeat frame 4 in frame 5's slot; the digest link breaks.
        frames[4] = dict(frames[3])
        self.write_chain(document)
        self.assert_rejected_and_untouched()
        self.restore()

    def test_non_canonical_chain(self) -> None:
        raw = _read(self.chain).replace(b'"version":1', b'"version": 1', 1)
        with open(self.chain, "wb") as handle:
            handle.write(raw)
        self.assert_rejected_and_untouched()
        self.restore()

    def test_sound_fork_at_boundary_rejected_after_rebuild(self) -> None:
        # Rewrite the boundary frame's *digest* while keeping the JSON
        # structurally parseable up to that point; the resume gate fails
        # on the prefix digest and the full rebuild rejects the chain.
        document = self.chain_document()
        document["frames"][3]["digest"] = "f" * 64
        self.write_chain(document)
        self.assert_rejected_and_untouched()
        self.restore()

    def test_head_mismatch_in_batch_diff(self) -> None:
        with self.assertRaises(ValueError):
            BranchStore.diff_recovery_audit_ranges(
                self.chain,
                "0" * 64,
                ((5, 6),),
                self.index,
                self.SEGMENT_SIZE,
                self.progress,
            )

    def test_generations_untouched_on_failure(self) -> None:
        snapshot = self.root_snapshot()
        document = self.chain_document()
        self.write_chain({**document, "frames": document["frames"][:2]})
        with self.assertRaises(ValueError):
            self.build(self.progress)
        self.assertEqual(self.root_snapshot(), snapshot)
        self.restore()


class ArgumentValidationTests(RecoveryProgressTestBase):
    def test_progress_type_validation(self) -> None:
        self.grow_chain(2)
        for bad in (1, 1.0, b"x", ["p"], {"p": 1}, object()):
            with self.subTest(bad=type(bad).__name__):
                with self.assertRaises(TypeError):
                    self.build(bad)
                with self.assertRaises(TypeError):
                    BranchStore.diff_recovery_audit_ranges(
                        self.chain,
                        self.head,
                        (),
                        self.index,
                        self.SEGMENT_SIZE,
                        bad,
                    )

    def test_progress_empty_string(self) -> None:
        self.grow_chain(2)
        with self.assertRaises(ValueError):
            self.build("")
        with self.assertRaises(ValueError):
            BranchStore.diff_recovery_audit_ranges(
                self.chain,
                self.head,
                (),
                self.index,
                self.SEGMENT_SIZE,
                "",
            )

    def test_progress_must_not_alias_chain_or_index(self) -> None:
        self.grow_chain(2)
        for same in (self.chain, self.index):
            with self.subTest(same=os.path.basename(same)):
                with self.assertRaises(ValueError):
                    self.build(same)
        link = os.path.join(self.tmp.name, "chain-link")
        os.symlink(self.chain, link)
        with self.assertRaises(ValueError):
            self.build(link)

    def test_none_keeps_baseline_signature(self) -> None:
        self.grow_chain(2)
        self.build(None)
        self.assertFalse(os.path.exists(self.progress))
        self.assertEqual(self.diff((), None), ())

    def test_existing_validation_order_is_preserved(self) -> None:
        # A bad segment size and bad progress are independent; the
        # segment size is checked before the progress path.
        self.grow_chain(1)
        with self.assertRaises(TypeError):
            BranchStore.build_recovery_audit_segments(
                self.chain, self.index, 1.5, self.progress
            )
        with self.assertRaises(TypeError):
            BranchStore.build_recovery_audit_segments(
                42, self.index, 2, self.progress
            )

    def test_batch_container_and_endpoint_types(self) -> None:
        self.grow_chain(4)
        self.build(self.progress)
        with self.assertRaises(TypeError):
            BranchStore.diff_recovery_audit_ranges(
                self.chain,
                self.head,
                [(1, 2)],  # list, not tuple
                self.index,
                self.SEGMENT_SIZE,
                self.progress,
            )
        with self.assertRaises(TypeError):
            BranchStore.diff_recovery_audit_ranges(
                self.chain,
                self.head,
                ((1.0, 2),),
                self.index,
                self.SEGMENT_SIZE,
                self.progress,
            )
        for bad_ranges in (((0, 2),), ((2, 1),), ((1, 1), (1, 1)), ((1, 99),)):
            with self.assertRaises(ValueError):
                self.diff(bad_ranges, self.progress)


class EmptyRangesTests(RecoveryProgressTestBase):
    def test_empty_ranges_authenticate_everything(self) -> None:
        head = self.grow_chain(4)
        self.build(self.progress)
        self.assertEqual(self.diff(()), ())
        self.assertEqual(self.diff((), self.progress), ())
        # A wrong head still surfaces, proving the chain was checked.
        with self.assertRaises(ValueError):
            BranchStore.diff_recovery_audit_ranges(
                self.chain,
                "0" * 64,
                (),
                self.index,
                self.SEGMENT_SIZE,
                self.progress,
            )
        # And a tampered chain is rejected even for an empty batch.
        document = self.chain_document()
        document["frames"][0]["digest"] = "a" * 64
        self.write_chain(document)
        with self.assertRaises(ValueError):
            BranchStore.diff_recovery_audit_ranges(
                self.chain,
                head,
                (),
                self.index,
                self.SEGMENT_SIZE,
                self.progress,
            )


class OSErrorAndRollbackTests(RecoveryProgressTestBase):
    def test_missing_chain_raises_oserror(self) -> None:
        with self.assertRaises(OSError):
            self.build(self.progress)
        self.assertFalse(os.path.exists(self.index))
        self.assertFalse(os.path.exists(self.progress))

    def test_missing_progress_directory_raises_oserror(self) -> None:
        self.grow_chain(2)
        missing = os.path.join(self.tmp.name, "nope", "progress.json")
        with self.assertRaises(OSError):
            self.build(missing)

    def test_failed_progress_publish_restores_index_bytes(self) -> None:
        self.grow_chain(4)
        self.build(self.progress)
        index_before = _read(self.index)
        progress_before = _read(self.progress)
        self.grow_chain(2, start=70)

        real_replace = os.replace

        def flaky_replace(src, dst):
            # Fail only the forward publication of the progress temp;
            # the rollback restore of the index from its byte-for-byte
            # backup (a different temp name) must still succeed.
            if os.path.basename(src).startswith(
                ".recovery-chain-progress-"
            ):
                raise OSError("simulated progress publish failure")
            return real_replace(src, dst)

        with mock.patch("city_twin.branches.os.replace", flaky_replace):
            with self.assertRaises(OSError):
                self.build(self.progress)
        # The index published in step one is rolled back byte-for-byte.
        self.assertEqual(_read(self.index), index_before)
        self.assertEqual(_read(self.progress), progress_before)
        self.assertEqual(self.residue(), [])

    def test_failed_index_publish_leaves_old_index(self) -> None:
        self.grow_chain(2)
        self.build(self.progress)
        index_before = _read(self.index)

        real_replace = os.replace

        def fail_first(src, dst):
            if os.path.realpath(dst) == os.path.realpath(self.index):
                raise OSError("simulated index publish failure")
            return real_replace(src, dst)

        self.grow_chain(2, start=80)
        with mock.patch("city_twin.branches.os.replace", fail_first):
            with self.assertRaises(OSError):
                self.build(self.progress)
        self.assertEqual(_read(self.index), index_before)
        self.assertEqual(self.residue(), [])

    def test_chain_never_modified_by_a_build(self) -> None:
        self.grow_chain(4)
        snapshot = self.root_snapshot()
        self.build(self.progress)
        # Appending a frame (even over identical evidence) only touches
        # the chain file; a following build must leave both it and the
        # generation directory untouched.
        self.head = self.append_frame(self.head)
        chain_after_append = _read(self.chain)
        self.build(self.progress)
        self.assertEqual(_read(self.chain), chain_after_append)
        self.assertEqual(self.root_snapshot(), snapshot)
        self.assertEqual(self.residue(), [])


class ConcurrencyTests(RecoveryProgressTestBase):
    def test_concurrent_builds_and_appends_observe_full_versions(self) -> None:
        head = self.grow_chain(2)
        self.build(self.progress)
        errors: list[BaseException] = []
        built_heads: list[str] = []
        lock = threading.Lock()

        def builder() -> None:
            try:
                for _ in range(5):
                    result = BranchStore.build_recovery_audit_segments(
                        self.chain,
                        self.index,
                        self.SEGMENT_SIZE,
                        self.progress,
                    )
                    with lock:
                        built_heads.append(result)
            except BaseException as exc:  # pragma: no cover - diagnostic
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=builder) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(errors, [])
        # Every build observed a complete chain; a final build and verify
        # converges on the append-produced head.
        final = BranchStore.build_recovery_audit_segments(
            self.chain, self.index, self.SEGMENT_SIZE, self.progress
        )
        self.assertTrue(
            BranchStore.verify_recovery_audit_chain(
                self.chain, final, self.index
            )
        )
        self.assertIn(final, built_heads + [head, final])
        self.assertEqual(self.residue(), [])


if __name__ == "__main__":
    unittest.main()
