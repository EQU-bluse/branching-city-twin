"""Tests for the persistent, tamper-evident recovery-audit chain."""

import hashlib
import json
import multiprocessing
import os
import shutil
import tempfile
import threading
import unittest

from city_twin.branches import BranchStore
from city_twin.event_graph import EventGraph


def _read(path: str) -> bytes:
    with open(path, "rb") as handle:
        return handle.read()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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


class RecoveryAuditChainTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = os.path.join(self.tmp.name, "gens")
        os.mkdir(self.root)
        self.cp = os.path.join(self.tmp.name, "cp.json")
        self.journal = os.path.join(self.tmp.name, "journal.json")
        self.chain = os.path.join(self.tmp.name, "chain.json")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def make_store(self) -> BranchStore:
        store = make_history()
        store.save_checkpoint(self.cp)
        store.enable_journal(self.cp, self.journal)
        store.append("main", "m3", 5, {"a": 3})
        return store

    def generation_dir(self, number: int) -> str:
        return os.path.join(self.root, f"generation-{number:016d}")

    def rotate_three(self) -> BranchStore:
        store = self.make_store()
        store.rotate_generation(self.root)
        store.append("main", "m4", 6, {"a": 4})
        store.rotate_generation(self.root)
        store.append("main", "m5", 7, {"a": 5})
        store.rotate_generation(self.root)
        return store

    def snapshot(self) -> dict[str, bytes]:
        captured: dict[str, bytes] = {}
        for dirpath, _dirnames, filenames in os.walk(self.root):
            for filename in filenames:
                path = os.path.join(dirpath, filename)
                captured[path] = _read(path)
        return captured

    def append_frame(self, previous: str | None) -> str:
        return BranchStore.append_recovery_audit_chain(
            self.root, self.chain, previous
        )

    def chain_document(self) -> dict[str, object]:
        return json.loads(_read(self.chain).decode("utf-8"))


class AppendRecoveryAuditChainTests(RecoveryAuditChainTestBase):
    def test_first_frame_shape_and_encoding(self) -> None:
        self.rotate_three()
        head = self.append_frame(None)
        self.assertIsInstance(head, str)
        self.assertEqual(len(head), 64)
        self.assertEqual(set(head), set("0123456789abcdef"))
        raw = _read(self.chain)
        self.assertFalse(raw.startswith(b"\xef\xbb\xbf"))
        self.assertFalse(raw.endswith(b"\n"))
        self.assertNotIn(b" ", raw)
        self.assertNotIn(b"\n", raw)
        document = json.loads(raw)
        self.assertEqual(list(document), ["format", "version", "frames"])
        self.assertEqual(
            document["format"],
            "branching-city-twin/recovery-audit-chain",
        )
        self.assertEqual(document["version"], 1)
        frames = document["frames"]
        self.assertEqual(len(frames), 1)
        frame = frames[0]
        self.assertEqual(list(frame), ["seq", "prev", "record", "digest"])
        self.assertEqual(frame["seq"], 1)
        self.assertEqual(frame["prev"], "0" * 64)
        self.assertEqual(
            frame["record"],
            BranchStore.export_recovery_audit(self.root),
        )
        signed = json.dumps(
            {
                "seq": 1,
                "prev": "0" * 64,
                "record": frame["record"],
            },
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        self.assertEqual(frame["digest"], _sha256(signed.encode("utf-8")))
        self.assertEqual(frame["digest"], head)

    def test_frames_link_consecutively(self) -> None:
        self.rotate_three()
        heads: list[str] = []
        for _ in range(4):
            heads.append(self.append_frame(heads[-1] if heads else None))
        document = self.chain_document()
        frames = document["frames"]
        self.assertEqual([f["seq"] for f in frames], [1, 2, 3, 4])
        self.assertEqual(frames[0]["prev"], "0" * 64)
        for prior, frame in zip(frames, frames[1:]):
            self.assertEqual(frame["prev"], prior["digest"])
        self.assertEqual([f["digest"] for f in frames], heads)
        self.assertEqual(frames[-1]["digest"], heads[-1])

    def test_unchanged_evidence_still_appends_a_frame(self) -> None:
        self.rotate_three()
        first = self.append_frame(None)
        second = self.append_frame(first)
        self.assertNotEqual(first, second)
        frames = self.chain_document()["frames"]
        self.assertEqual(len(frames), 2)
        # The sampled record is identical ...
        self.assertEqual(frames[0]["record"], frames[1]["record"])
        # ... but the sequence and predecessor link make a distinct frame.
        self.assertNotEqual(frames[0]["digest"], frames[1]["digest"])
        self.assertEqual(frames[1]["prev"], frames[0]["digest"])

    def test_record_reflects_new_evidence(self) -> None:
        store = self.rotate_three()
        first = self.append_frame(None)
        store.append("main", "m6", 8, {"a": 6})
        store.rotate_generation(self.root)
        second = self.append_frame(first)
        frames = self.chain_document()["frames"]
        self.assertNotEqual(frames[0]["record"], frames[1]["record"])
        second_record = json.loads(frames[1]["record"])
        self.assertEqual(
            second_record["selected"],
            "generation-0000000000000004",
        )
        self.assertEqual(frames[1]["digest"], second)

    def test_genesis_requires_none(self) -> None:
        self.rotate_three()
        # A digest supplied when no chain exists is rejected and creates
        # no file.
        with self.assertRaises(RuntimeError):
            self.append_frame("0" * 64)
        self.assertFalse(os.path.exists(self.chain))
        head = self.append_frame(None)
        # Once the chain exists, None is rejected and the file is
        # untouched.
        before = _read(self.chain)
        with self.assertRaises(RuntimeError):
            self.append_frame(None)
        self.assertEqual(_read(self.chain), before)
        # The wrong digest is likewise rejected without touching the file.
        with self.assertRaises(RuntimeError):
            self.append_frame("f" * 64)
        self.assertEqual(_read(self.chain), before)
        # Extend once so an earlier head is genuinely stale.
        second = self.append_frame(head)
        before_two = _read(self.chain)
        # An earlier head (a fork/rollback to a previous tip) is rejected.
        with self.assertRaises(RuntimeError):
            self.append_frame(head)
        self.assertEqual(_read(self.chain), before_two)
        self.assertIsInstance(second, str)

    def test_previous_head_validation(self) -> None:
        self.rotate_three()
        for bad in (1, 1.0, b"x", ["x"], {}):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    BranchStore.append_recovery_audit_chain(
                        self.root, self.chain, bad
                    )
        for bad in ("", "a" * 63, "A" * 64, "g" * 64, "0" * 63):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    BranchStore.append_recovery_audit_chain(
                        self.root, self.chain, bad
                    )

    def test_root_validation_precedes_path_and_head(self) -> None:
        self.rotate_three()
        for bad in (1, 1.0, b"x", None, ["a"]):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    BranchStore.append_recovery_audit_chain(
                        bad, self.chain, None
                    )
        with self.assertRaises(ValueError):
            BranchStore.append_recovery_audit_chain("", self.chain, None)
        with self.assertRaises(OSError):
            BranchStore.append_recovery_audit_chain(
                os.path.join(self.tmp.name, "nope"), self.chain, None
            )

    def test_path_validation(self) -> None:
        self.rotate_three()
        for bad in (1, 1.0, b"x", None, ["a"]):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    BranchStore.append_recovery_audit_chain(
                        self.root, bad, None
                    )
        with self.assertRaises(ValueError):
            BranchStore.append_recovery_audit_chain(self.root, "", None)

    def test_path_in_missing_directory_raises_oserror(self) -> None:
        self.rotate_three()
        path = os.path.join(self.tmp.name, "missing-dir", "chain.json")
        with self.assertRaises(OSError):
            BranchStore.append_recovery_audit_chain(self.root, path, None)

    def test_no_valid_generation_does_not_create_chain(self) -> None:
        with self.assertRaises(ValueError):
            BranchStore.append_recovery_audit_chain(
                self.root, self.chain, None
            )
        self.assertFalse(os.path.exists(self.chain))

    def test_append_is_read_only_on_generation_root(self) -> None:
        self.rotate_three()
        before = self.snapshot()
        first = self.append_frame(None)
        self.append_frame(first)
        self.assertEqual(self.snapshot(), before)

    def test_failed_write_leaves_old_chain_untouched(self) -> None:
        self.rotate_three()
        head = self.append_frame(None)
        before = _read(self.chain)
        readonly_dir = os.path.join(self.tmp.name, "readonly")
        os.mkdir(readonly_dir)
        target = os.path.join(readonly_dir, "chain.json")
        shutil.copyfile(self.chain, target)
        os.chmod(readonly_dir, 0o555)
        try:
            if os.geteuid() != 0:
                with self.assertRaises(OSError):
                    BranchStore.append_recovery_audit_chain(
                        self.root, target, head
                    )
                self.assertEqual(_read(target), before)
                # No temporary leftovers are visible in the directory.
                self.assertEqual(
                    [n for n in os.listdir(readonly_dir) if n != "."],
                    ["chain.json"],
                )
        finally:
            os.chmod(readonly_dir, 0o755)
        # The original chain still appends normally afterwards.
        self.assertIsInstance(self.append_frame(head), str)


class VerifyRecoveryAuditChainTests(RecoveryAuditChainTestBase):
    def build_chain(self, frames: int = 3) -> list[str]:
        self.rotate_three()
        heads: list[str] = []
        for _ in range(frames):
            heads.append(
                BranchStore.append_recovery_audit_chain(
                    self.root,
                    self.chain,
                    heads[-1] if heads else None,
                )
            )
        return heads

    def reserialize(self, document: object) -> str:
        return json.dumps(
            document, separators=(",", ":"), ensure_ascii=False
        )

    def test_matching_head_verifies(self) -> None:
        heads = self.build_chain()
        self.assertTrue(
            BranchStore.verify_recovery_audit_chain(self.chain, heads[-1])
        )

    def test_verify_is_idempotent_and_read_only(self) -> None:
        heads = self.build_chain()
        before = _read(self.chain)
        root_before = self.snapshot()
        for _ in range(3):
            self.assertTrue(
                BranchStore.verify_recovery_audit_chain(
                    self.chain, heads[-1]
                )
            )
        self.assertEqual(_read(self.chain), before)
        self.assertEqual(self.snapshot(), root_before)

    def test_rolled_back_head_returns_false(self) -> None:
        heads = self.build_chain(3)
        # The head of the first two frames is a valid internal digest but
        # not the current tip: the chain detects truncation/rollback.
        self.assertFalse(
            BranchStore.verify_recovery_audit_chain(self.chain, heads[0])
        )
        self.assertFalse(
            BranchStore.verify_recovery_audit_chain(self.chain, "f" * 64)
        )

    def test_missing_file_raises_oserror(self) -> None:
        with self.assertRaises(OSError):
            BranchStore.verify_recovery_audit_chain(self.chain, "f" * 64)

    def test_empty_documents_raise_valueerror(self) -> None:
        self.rotate_three()
        for content in (b"", self.reserialize(
            {
                "format": "branching-city-twin/recovery-audit-chain",
                "version": 1,
                "frames": [],
            }
        ).encode("utf-8")):
            with open(self.chain, "wb") as handle:
                handle.write(content)
            with self.subTest(content=content[:1]):
                with self.assertRaises(ValueError):
                    BranchStore.verify_recovery_audit_chain(
                        self.chain, "f" * 64
                    )

    def test_encoding_violations_raise_valueerror(self) -> None:
        heads = self.build_chain(1)
        raw = _read(self.chain)
        for corrupted in (
            b"\xef\xbb\xbf" + raw,
            raw + b"\n",
            b"{",
            b"not json",
        ):
            with open(self.chain, "wb") as handle:
                handle.write(corrupted)
            with self.subTest(corrupted=corrupted[:8]):
                with self.assertRaises(ValueError):
                    BranchStore.verify_recovery_audit_chain(
                        self.chain, heads[-1]
                    )

    def test_duplicate_keys_raise_valueerror(self) -> None:
        self.build_chain(1)
        raw = _read(self.chain).decode("utf-8")
        corrupted = raw.replace(
            '"frames":[', '"frames":[], "frames":[', 1
        )
        with open(self.chain, "wb") as handle:
            handle.write(corrupted.encode("utf-8"))
        with self.assertRaises(ValueError):
            BranchStore.verify_recovery_audit_chain(self.chain, "f" * 64)

    def test_non_canonical_encoding_raises_valueerror(self) -> None:
        self.build_chain(1)
        document = json.loads(_read(self.chain))
        with open(self.chain, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2)
        with self.assertRaises(ValueError):
            BranchStore.verify_recovery_audit_chain(self.chain, "f" * 64)

    def test_structure_violations_raise_valueerror(self) -> None:
        self.build_chain(1)
        good = json.loads(_read(self.chain))
        candidates = (
            [],
            None,
            {"version": 1, "frames": []},
            {
                "format": "other",
                "version": 1,
                "frames": good["frames"],
            },
            {
                "format": good["format"],
                "version": 2,
                "frames": good["frames"],
            },
            {
                "format": good["format"],
                "version": 1,
                "frames": {},
            },
        )
        for candidate in candidates:
            with open(self.chain, "wb") as handle:
                handle.write(self.reserialize(candidate).encode("utf-8"))
            with self.subTest(candidate=candidate):
                with self.assertRaises(ValueError):
                    BranchStore.verify_recovery_audit_chain(
                        self.chain, "f" * 64
                    )

    def test_tampered_frames_raise_valueerror(self) -> None:
        # Three frames over genuinely different evidence, so swapping two
        # of them cannot collapse back into the original head even after
        # honest re-sealing.
        store = self.rotate_three()
        heads = [
            BranchStore.append_recovery_audit_chain(
                self.root, self.chain, None
            )
        ]
        store.append("main", "m6", 8, {"a": 6})
        store.rotate_generation(self.root)
        heads.append(
            BranchStore.append_recovery_audit_chain(
                self.root, self.chain, heads[-1]
            )
        )
        store.append("main", "m7", 9, {"a": 7})
        store.rotate_generation(self.root)
        heads.append(
            BranchStore.append_recovery_audit_chain(
                self.root, self.chain, heads[-1]
            )
        )
        good = json.loads(_read(self.chain))
        self.assertEqual(
            {f["record"] for f in good["frames"]}
            and len({f["record"] for f in good["frames"]}),
            3,
        )

        def frame_digest(frame: dict[str, object]) -> str:
            signed = self.reserialize(
                {
                    "seq": frame["seq"],
                    "prev": frame["prev"],
                    "record": frame["record"],
                }
            )
            return _sha256(signed.encode("utf-8"))

        value_error_cases: list[tuple[str, object]] = []
        head_mismatch_cases: list[tuple[str, object]] = []

        # 1. Reordered (swap frames 1 and 2, rebuild links honestly): an
        #    internally sound fork whose head no longer matches the
        #    externally retained digest, so verification returns False.
        swapped = json.loads(json.dumps(good))
        frames = swapped["frames"]
        frames[0], frames[1] = frames[1], frames[0]
        frames[0]["seq"] = 1
        frames[0]["prev"] = "0" * 64
        frames[0]["digest"] = frame_digest(frames[0])
        frames[1]["seq"] = 2
        frames[1]["prev"] = frames[0]["digest"]
        frames[1]["digest"] = frame_digest(frames[1])
        frames[2]["prev"] = frames[1]["digest"]
        frames[2]["digest"] = frame_digest(frames[2])
        head_mismatch_cases.append(("reorder", swapped))

        # 2. Truncated to the first two frames: a sound prefix whose head
        #    is an older digest -> rollback reported as False.
        truncated = json.loads(json.dumps(good))
        truncated["frames"] = truncated["frames"][:2]
        head_mismatch_cases.append(("truncate", truncated))

        # 3. Duplicate the last frame bumping its sequence without
        #    rebuilding the frame digest: the digest no longer covers the
        #    claimed seq, so validation raises ValueError.
        duplicated = json.loads(json.dumps(good))
        dup = json.loads(json.dumps(duplicated["frames"][-1]))
        dup["seq"] = 4
        duplicated["frames"].append(dup)
        value_error_cases.append(("duplicate", duplicated))

        # 4. Delete the middle frame and renumber/re-link honestly: the
        #    chain is internally consistent but seals frame 3's record at
        #    position 2 and ends at a different digest -> False.
        deleted = json.loads(json.dumps(good))
        remaining = [deleted["frames"][0], deleted["frames"][2]]
        remaining[1]["seq"] = 2
        remaining[1]["prev"] = remaining[0]["digest"]
        remaining[1]["digest"] = frame_digest(remaining[1])
        deleted["frames"] = remaining
        head_mismatch_cases.append(("delete-renumber", deleted))

        for label, candidate in value_error_cases:
            with open(self.chain, "wb") as handle:
                handle.write(
                    self.reserialize(candidate).encode("utf-8")
                )
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    BranchStore.verify_recovery_audit_chain(
                        self.chain, heads[-1]
                    )
        for label, candidate in head_mismatch_cases:
            with open(self.chain, "wb") as handle:
                handle.write(
                    self.reserialize(candidate).encode("utf-8")
                )
            with self.subTest(label=label):
                self.assertFalse(
                    BranchStore.verify_recovery_audit_chain(
                        self.chain, heads[-1]
                    )
                )

    def test_broken_digest_links_raise_valueerror(self) -> None:
        heads = self.build_chain(2)
        good = json.loads(_read(self.chain))

        # Corrupt a frame digest directly.
        corrupted = json.loads(json.dumps(good))
        corrupted["frames"][0]["digest"] = "a" * 64
        with open(self.chain, "wb") as handle:
            handle.write(self.reserialize(corrupted).encode("utf-8"))
        with self.assertRaises(ValueError):
            BranchStore.verify_recovery_audit_chain(self.chain, "a" * 64)

        # Corrupt the predecessor link directly.
        corrupted = json.loads(json.dumps(good))
        corrupted["frames"][1]["prev"] = "b" * 64
        with open(self.chain, "wb") as handle:
            handle.write(self.reserialize(corrupted).encode("utf-8"))
        with self.assertRaises(ValueError):
            BranchStore.verify_recovery_audit_chain(self.chain, heads[-1])

    def test_rewritten_record_raises_valueerror(self) -> None:
        heads = self.build_chain(1)
        document = json.loads(_read(self.chain))
        record = json.loads(document["frames"][0]["record"])
        # Flip the selection but keep the audit record's own checksum
        # stale: the embedded record must fail its checksum validation.
        record["selected"] = "generation-0000000000000001"
        document["frames"][0]["record"] = self.reserialize(record)
        with open(self.chain, "wb") as handle:
            handle.write(self.reserialize(document).encode("utf-8"))
        with self.assertRaises(ValueError):
            BranchStore.verify_recovery_audit_chain(self.chain, heads[-1])

    def test_argument_validation(self) -> None:
        self.build_chain(1)
        for bad in (1, 1.0, b"x", None, ["x"], {}):
            with self.subTest(where="path", bad=bad):
                with self.assertRaises(TypeError):
                    BranchStore.verify_recovery_audit_chain(bad, "f" * 64)
            with self.subTest(where="head", bad=bad):
                if bad is None:
                    with self.assertRaises(TypeError):
                        BranchStore.verify_recovery_audit_chain(
                            self.chain, bad
                        )
                else:
                    with self.assertRaises(TypeError):
                        BranchStore.verify_recovery_audit_chain(
                            self.chain, bad
                        )
        with self.assertRaises(ValueError):
            BranchStore.verify_recovery_audit_chain(self.chain, "")
        with self.assertRaises(ValueError):
            BranchStore.verify_recovery_audit_chain(self.chain, "F" * 64)
        with self.assertRaises(ValueError):
            BranchStore.verify_recovery_audit_chain("", "f" * 64)


class DiffRecoveryAuditRangeTests(RecoveryAuditChainTestBase):
    def build_with_change(self) -> tuple[list[str], BranchStore]:
        store = self.rotate_three()
        heads = [
            BranchStore.append_recovery_audit_chain(
                self.root, self.chain, None
            )
        ]
        # A no-change sample sits between: range diffs must still work
        # when endpoints skip over it.
        heads.append(
            BranchStore.append_recovery_audit_chain(
                self.root, self.chain, heads[-1]
            )
        )
        store.append("main", "m6", 8, {"a": 6})
        store.rotate_generation(self.root)
        heads.append(
            BranchStore.append_recovery_audit_chain(
                self.root, self.chain, heads[-1]
            )
        )
        return heads, store

    def test_same_sequence_returns_empty_tuple(self) -> None:
        heads, _store = self.build_with_change()
        result = BranchStore.diff_recovery_audit_range(
            self.chain, heads[-1], 2, 2
        )
        self.assertIsInstance(result, tuple)
        self.assertEqual(result, ())

    def test_unchanged_range_returns_empty_tuple(self) -> None:
        heads, _store = self.build_with_change()
        self.assertEqual(
            BranchStore.diff_recovery_audit_range(
                self.chain, heads[-1], 1, 2
            ),
            (),
        )

    def test_changed_range_matches_single_record_diff(self) -> None:
        heads, _store = self.build_with_change()
        result = BranchStore.diff_recovery_audit_range(
            self.chain, heads[-1], 1, 3
        )
        document = self.chain_document()
        first_record = document["frames"][0]["record"]
        expected = BranchStore.diff_recovery_audit(
            self.root, first_record
        )
        self.assertEqual(result, expected)
        # Stable order and the familiar five-field tuple shape.
        self.assertTrue(result)
        for change in result:
            self.assertEqual(len(change), 5)
            category, generation, kind, before, after = change
            self.assertIn(
                category,
                ("pointer", "chain", "checkpoint", "journal", "replay"),
            )
            self.assertIn(kind, ("added", "removed", "changed"))
        # The new generation shows up as added.
        self.assertIn(
            ("chain", "generation-0000000000000004", "added"),
            {(c[0], c[1], c[2]) for c in result},
        )

    def test_subrange_skipping_unchanged_frame(self) -> None:
        heads, _store = self.build_with_change()
        # Endpoint 2 was a no-op sample; 2 -> 3 sees the same changes.
        via_two = BranchStore.diff_recovery_audit_range(
            self.chain, heads[-1], 2, 3
        )
        via_one = BranchStore.diff_recovery_audit_range(
            self.chain, heads[-1], 1, 3
        )
        self.assertEqual(via_two, via_one)

    def test_head_mismatch_raises_valueerror(self) -> None:
        heads, _store = self.build_with_change()
        with self.assertRaises(ValueError):
            BranchStore.diff_recovery_audit_range(
                self.chain, "f" * 64, 1, 3
            )
        # An earlier, internally valid head is also not authorization.
        with self.assertRaises(ValueError):
            BranchStore.diff_recovery_audit_range(
                self.chain, heads[0], 1, 3
            )

    def test_corrupt_chain_raises_valueerror(self) -> None:
        heads, _store = self.build_with_change()
        raw = _read(self.chain)
        with open(self.chain, "wb") as handle:
            handle.write(raw + b" ")
        with self.assertRaises(ValueError):
            BranchStore.diff_recovery_audit_range(
                self.chain, heads[-1], 1, 3
            )

    def test_sequence_validation(self) -> None:
        heads, _store = self.build_with_change()
        # Non-int (and bool) sequences are TypeErrors.
        for bad in (1.0, "1", [1], None, True, False):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    BranchStore.diff_recovery_audit_range(
                        self.chain, heads[-1], bad, 3
                    )
                with self.assertRaises(TypeError):
                    BranchStore.diff_recovery_audit_range(
                        self.chain, heads[-1], 1, bad
                    )
        # Domain violations are ValueErrors.
        for start, end in ((0, 3), (1, 0), (3, 1), (1, 4), (4, 4)):
            with self.subTest(start=start, end=end):
                with self.assertRaises(ValueError):
                    BranchStore.diff_recovery_audit_range(
                        self.chain, heads[-1], start, end
                    )

    def test_path_and_head_validation(self) -> None:
        heads, _store = self.build_with_change()
        for bad in (1, 1.0, b"x", []):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    BranchStore.diff_recovery_audit_range(
                        bad, heads[-1], 1, 3
                    )
        with self.assertRaises(ValueError):
            BranchStore.diff_recovery_audit_range(
                self.chain, "not-a-digest", 1, 3
            )
        with self.assertRaises(OSError):
            BranchStore.diff_recovery_audit_range(
                os.path.join(self.tmp.name, "nope"), heads[-1], 1, 3
            )

    def test_diff_is_read_only_and_detached(self) -> None:
        heads, _store = self.build_with_change()
        before = _read(self.chain)
        root_before = self.snapshot()
        result = BranchStore.diff_recovery_audit_range(
            self.chain, heads[-1], 1, 3
        )
        # Mutating returned tuples/dicts cannot affect later calls.
        for change in result:
            if isinstance(change[3], dict):
                change[3].clear()
        again = BranchStore.diff_recovery_audit_range(
            self.chain, heads[-1], 1, 3
        )
        self.assertTrue(all(c[3] is not None for c in again if c[2] != "added"))
        self.assertEqual(_read(self.chain), before)
        self.assertEqual(self.snapshot(), root_before)


def _hold_lock_process(
    lock_path: str, ready, release: object, held: object
) -> None:
    """Child process: hold the OS lock on ``lock_path`` until told to
    release it, proving the lock is released by the operating system
    when the process exits."""
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        BranchStore._lock_file_exclusive(fd)
        ready.set()
        release.wait(30)
    finally:
        os.close(fd)
    held.set()


class AppendRecoveryAuditChainLockTests(RecoveryAuditChainTestBase):
    def lock_path(self) -> str:
        return os.path.realpath(self.chain) + ".lock"

    def test_timeout_validation(self) -> None:
        self.rotate_three()
        for bad in (True, False, "1", b"1", [1], (1,), {}):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    BranchStore.append_recovery_audit_chain(
                        self.root, self.chain, None, timeout=bad
                    )
        for bad in (-1, -0.5, float("nan"), float("inf"), float("-inf")):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    BranchStore.append_recovery_audit_chain(
                        self.root, self.chain, None, timeout=bad
                    )
        self.assertFalse(os.path.exists(self.chain))

    def test_timeout_validated_after_root_path_and_previous(self) -> None:
        self.rotate_three()
        # A bad root still wins over a bad timeout.
        with self.assertRaises(TypeError):
            BranchStore.append_recovery_audit_chain(
                1, self.chain, None, timeout="x"
            )
        # A bad path still wins over a bad timeout.
        with self.assertRaises(TypeError):
            BranchStore.append_recovery_audit_chain(
                self.root, 1, None, timeout="x"
            )
        # A bad head digest still wins over a bad timeout.
        with self.assertRaises(TypeError):
            BranchStore.append_recovery_audit_chain(
                self.root, self.chain, 1, timeout="x"
            )
        with self.assertRaises(ValueError):
            BranchStore.append_recovery_audit_chain(
                self.root, self.chain, "g" * 64, timeout=-1
            )

    def test_timeout_accepts_int_and_float(self) -> None:
        self.rotate_three()
        head = BranchStore.append_recovery_audit_chain(
            self.root, self.chain, None, timeout=5
        )
        self.assertIsInstance(head, str)
        second = BranchStore.append_recovery_audit_chain(
            self.root, self.chain, head, timeout=2.5
        )
        self.assertIsInstance(second, str)
        third = BranchStore.append_recovery_audit_chain(
            self.root, self.chain, second, timeout=0
        )
        self.assertIsInstance(third, str)
        self.assertEqual(
            len(self.chain_document()["frames"]), 3
        )

    def test_contended_lock_times_out_without_touching_chain(self) -> None:
        self.rotate_three()
        head = self.append_frame(None)
        before = _read(self.chain)
        root_before = self.snapshot()
        with BranchStore._recovery_chain_lock(self.chain, None):
            with self.assertRaises(TimeoutError):
                BranchStore.append_recovery_audit_chain(
                    self.root, self.chain, head, timeout=0.3
                )
        self.assertEqual(_read(self.chain), before)
        self.assertEqual(self.snapshot(), root_before)
        # The chain still appends normally once the lock is free.
        self.assertIsInstance(self.append_frame(head), str)

    def test_zero_timeout_fails_fast_when_locked(self) -> None:
        self.rotate_three()
        head = self.append_frame(None)
        before = _read(self.chain)
        with BranchStore._recovery_chain_lock(self.chain, None):
            with self.assertRaises(TimeoutError):
                BranchStore.append_recovery_audit_chain(
                    self.root, self.chain, head, timeout=0
                )
        self.assertEqual(_read(self.chain), before)

    def test_equivalent_paths_share_one_lock(self) -> None:
        self.rotate_three()
        head = self.append_frame(None)
        dotted = os.path.join(self.tmp.name, ".", "chain.json")
        self.assertNotEqual(dotted, self.chain)
        with BranchStore._recovery_chain_lock(dotted, None):
            with self.assertRaises(TimeoutError):
                BranchStore.append_recovery_audit_chain(
                    self.root, self.chain, head, timeout=0.2
                )

    def test_lock_file_is_fixed_empty_and_not_temp_residue(self) -> None:
        self.rotate_three()
        head = self.append_frame(None)
        self.append_frame(head)
        lock_path = self.lock_path()
        self.assertTrue(os.path.exists(lock_path))
        # The coordination file never carries audit content.
        self.assertEqual(_read(lock_path), b"")
        # No temporary chain files remain after success or failure.
        leftovers = [
            name
            for name in os.listdir(self.tmp.name)
            if name.startswith(".recovery-chain-")
        ]
        self.assertEqual(leftovers, [])
        with self.assertRaises(RuntimeError):
            self.append_frame(head)
        self.assertEqual(
            [
                name
                for name in os.listdir(self.tmp.name)
                if name.startswith(".recovery-chain-")
            ],
            [],
        )

    def test_concurrent_same_head_appends_one_winner(self) -> None:
        self.rotate_three()
        head = self.append_frame(None)
        before = self.snapshot()
        results: list[object] = []
        barrier = threading.Barrier(8)

        def contend() -> None:
            barrier.wait()
            try:
                results.append(
                    BranchStore.append_recovery_audit_chain(
                        self.root, self.chain, head, timeout=10
                    )
                )
            except RuntimeError as exc:
                results.append(exc)

        threads = [
            threading.Thread(target=contend) for _ in range(8)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        winners = [r for r in results if isinstance(r, str)]
        losers = [r for r in results if isinstance(r, RuntimeError)]
        self.assertEqual(len(winners), 1)
        self.assertEqual(len(losers), 7)
        # Exactly one frame was appended, sealed by the winner.
        frames = self.chain_document()["frames"]
        self.assertEqual(len(frames), 2)
        self.assertEqual(frames[-1]["digest"], winners[0])
        self.assertEqual(frames[-1]["prev"], head)
        # The losers did not overwrite the winning frame.
        self.assertTrue(
            BranchStore.verify_recovery_audit_chain(
                self.chain, winners[0]
            )
        )
        # Nothing under the generation root changed.
        self.assertEqual(self.snapshot(), before)

    def test_concurrent_first_creation_one_winner(self) -> None:
        self.rotate_three()
        results: list[object] = []
        barrier = threading.Barrier(6)

        def contend() -> None:
            barrier.wait()
            try:
                results.append(
                    BranchStore.append_recovery_audit_chain(
                        self.root, self.chain, None, timeout=10
                    )
                )
            except RuntimeError as exc:
                results.append(exc)

        threads = [
            threading.Thread(target=contend) for _ in range(6)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        winners = [r for r in results if isinstance(r, str)]
        self.assertEqual(len(winners), 1)
        self.assertEqual(len(results), 6)
        frames = self.chain_document()["frames"]
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0]["digest"], winners[0])

    def test_concurrent_verify_reads_whole_documents(self) -> None:
        self.rotate_three()
        head = self.append_frame(None)
        stop = threading.Event()
        failures: list[object] = []

        def read_repeatedly() -> None:
            while not stop.is_set():
                try:
                    # Every read is a complete canonical document, so
                    # parsing and full authentication never fail; the
                    # head comparison itself may legitimately be False
                    # for a stale expected head.
                    document = json.loads(_read(self.chain))
                    digest = document["frames"][-1]["digest"]
                    BranchStore.verify_recovery_audit_chain(
                        self.chain, digest
                    )
                    try:
                        BranchStore.diff_recovery_audit_range(
                            self.chain,
                            digest,
                            1,
                            len(document["frames"]),
                        )
                    except ValueError as exc:
                        # The only acceptable race: the chain advanced
                        # between the read and the diff, so the
                        # authenticated head no longer matches. A
                        # partial document would surface as a different
                        # validation failure.
                        if "head does not match" not in str(exc):
                            raise
                except Exception as exc:  # pragma: no cover
                    failures.append(exc)
                    return

        reader = threading.Thread(target=read_repeatedly)
        reader.start()
        for _ in range(10):
            head = self.append_frame(head)
        stop.set()
        reader.join()
        self.assertEqual(failures, [])
        self.assertEqual(
            len(self.chain_document()["frames"]), 11
        )

    def test_lock_released_after_process_exit(self) -> None:
        self.rotate_three()
        head = self.append_frame(None)
        ctx = multiprocessing.get_context("spawn")
        ready = ctx.Event()
        release = ctx.Event()
        held = ctx.Event()
        child = ctx.Process(
            target=_hold_lock_process,
            args=(self.lock_path(), ready, release, held),
        )
        child.start()
        try:
            self.assertTrue(ready.wait(30))
            before = _read(self.chain)
            with self.assertRaises(TimeoutError):
                BranchStore.append_recovery_audit_chain(
                    self.root, self.chain, head, timeout=0.3
                )
            self.assertEqual(_read(self.chain), before)
        finally:
            release.set()
            child.join(30)
        self.assertTrue(held.is_set())
        # Once the other process let go, the append proceeds.
        self.assertIsInstance(self.append_frame(head), str)


if __name__ == "__main__":
    unittest.main()