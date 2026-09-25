"""Tests for the bounded-memory recovery-audit chain index."""

import hashlib
import json
import os
import tempfile
import tracemalloc
import unittest
from unittest import mock

import city_twin.branches as branches_module
from city_twin.branches import BranchStore
from city_twin.event_graph import EventGraph


def _read(path: str) -> bytes:
    with open(path, "rb") as handle:
        return handle.read()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class RecoveryAuditIndexTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = os.path.join(self.tmp.name, "gens")
        os.mkdir(self.root)
        self.cp = os.path.join(self.tmp.name, "cp.json")
        self.journal = os.path.join(self.tmp.name, "journal.json")
        self.chain = os.path.join(self.tmp.name, "chain.json")
        self.index = os.path.join(self.tmp.name, "chain.index.json")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def make_store(self) -> BranchStore:
        graph = EventGraph()
        graph.add("root", 0, (), {"seed": 7})
        store = BranchStore(graph)
        store.create("main", "root")
        store.append("main", "m1", 1, {"a": 1})
        store.save_checkpoint(self.cp)
        store.enable_journal(self.cp, self.journal)
        store.rotate_generation(self.root)
        return store

    def generation_dir(self, number: int) -> str:
        return os.path.join(self.root, f"generation-{number:016d}")

    def snapshot(self) -> dict[str, bytes]:
        captured: dict[str, bytes] = {}
        for dirpath, _dirnames, filenames in os.walk(self.root):
            for filename in filenames:
                path = os.path.join(dirpath, filename)
                captured[path] = _read(path)
        return captured

    def append_frame(self, store: BranchStore, previous: str | None) -> str:
        return BranchStore.append_recovery_audit_chain(
            self.root, self.chain, previous
        )

    def rotate_with_change(
        self, store: BranchStore, at: int, value: int
    ) -> None:
        store.append("main", f"m{at}", at, {"a": value})
        store.rotate_generation(self.root)

    def build_chain(
        self, frames: int, *, changing: bool = False
    ) -> tuple[list[str], BranchStore]:
        store = self.make_store()
        heads: list[str] = []
        for seq in range(frames):
            if changing and seq:
                self.rotate_with_change(store, at=10 + seq, value=seq)
            heads.append(self.append_frame(store, heads[-1] if heads else None))
        return heads, store


class BuildRecoveryAuditIndexTests(RecoveryAuditIndexTestBase):
    def test_build_returns_head_and_writes_canonical_json(self) -> None:
        heads, _store = self.build_chain(3)
        head = BranchStore.build_recovery_audit_index(self.chain, self.index)
        self.assertEqual(head, heads[-1])
        raw = _read(self.index)
        # UTF-8, no BOM, compact, no trailing newline.
        self.assertFalse(raw.startswith(b"\xef\xbb\xbf"))
        self.assertFalse(raw.endswith(b"\n"))
        self.assertNotIn(b"\n", raw)
        self.assertNotIn(b" ", raw)
        document = json.loads(raw)
        self.assertEqual(
            list(document),
            ["format", "version", "chain_version", "frames", "length",
             "head", "chain_digest"],
        )
        self.assertEqual(
            document["format"],
            "branching-city-twin/recovery-audit-chain-index",
        )
        self.assertEqual(document["version"], 1)
        self.assertEqual(document["chain_version"], 1)
        self.assertEqual(document["length"], 3)
        self.assertEqual(document["head"], heads[-1])
        self.assertEqual(
            document["chain_digest"], _sha256(_read(self.chain))
        )

    def test_entries_carry_offsets_links_and_digests(self) -> None:
        heads, _store = self.build_chain(3)
        BranchStore.build_recovery_audit_index(self.chain, self.index)
        chain_raw = _read(self.chain)
        chain_doc = json.loads(chain_raw)
        index_doc = json.loads(_read(self.index))
        entries = index_doc["frames"]
        self.assertEqual([e["seq"] for e in entries], [1, 2, 3])
        last_offset = -1
        for position, (entry, frame) in enumerate(
            zip(entries, chain_doc["frames"]), start=1
        ):
            self.assertEqual(
                list(entry), ["seq", "offset", "prev", "digest"]
            )
            self.assertEqual(entry["seq"], frame["seq"])
            self.assertEqual(entry["prev"], frame["prev"])
            self.assertEqual(entry["digest"], frame["digest"])
            # The offset locates exactly this frame object in the chain.
            self.assertGreater(entry["offset"], last_offset)
            self.assertEqual(
                chain_raw[entry["offset"]:entry["offset"] + 1], b"{"
            )
            # The frame at the recorded offset embeds the matching digest.
            located = chain_raw.index(
                frame["digest"].encode("ascii"), entry["offset"]
            )
            self.assertGreater(located, entry["offset"])
            last_offset = entry["offset"]
        self.assertEqual(entries[0]["prev"], "0" * 64)

    def test_same_chain_produces_identical_bytes(self) -> None:
        heads, _store = self.build_chain(4)
        first = self.index
        second = os.path.join(self.tmp.name, "other.index.json")
        returned_first = BranchStore.build_recovery_audit_index(
            self.chain, first
        )
        returned_second = BranchStore.build_recovery_audit_index(
            self.chain, second
        )
        self.assertEqual(returned_first, returned_second)
        self.assertEqual(_read(first), _read(second))
        # Rebuilding over an existing index emits the same bytes again.
        BranchStore.build_recovery_audit_index(self.chain, first)
        self.assertEqual(
            _read(first), _read(second)
        )

    def test_build_is_read_only_on_chain_and_generation_root(self) -> None:
        heads, _store = self.build_chain(2)
        chain_before = _read(self.chain)
        root_before = self.snapshot()
        BranchStore.build_recovery_audit_index(self.chain, self.index)
        # Rebuilding (index already present) also leaves the chain intact.
        BranchStore.build_recovery_audit_index(self.chain, self.index)
        self.assertEqual(_read(self.chain), chain_before)
        self.assertEqual(self.snapshot(), root_before)
        self.assertEqual(heads[-1], heads[-1])

    def test_build_requires_chain_to_exist(self) -> None:
        with self.assertRaises(OSError):
            BranchStore.build_recovery_audit_index(
                os.path.join(self.tmp.name, "missing.json"), self.index
            )
        self.assertFalse(os.path.exists(self.index))

    def test_build_on_corrupt_chain_raises_valueerror_and_no_index(self) -> None:
        self.build_chain(1)
        BranchStore.build_recovery_audit_index(self.chain, self.index)
        good = _read(self.chain)
        with open(self.chain, "wb") as handle:
            handle.write(good + b" ")
        with self.assertRaises(ValueError):
            BranchStore.build_recovery_audit_index(self.chain, self.index)
        # The previous index remains the valid pre-existing file; the
        # chain is restored to prove that index still authenticates.
        with open(self.chain, "wb") as handle:
            handle.write(good)
        index_doc = json.loads(_read(self.index))
        self.assertTrue(
            BranchStore.verify_recovery_audit_chain(
                self.chain, index_doc["head"], self.index
            )
        )

    def test_failed_publish_leaves_old_index_and_no_temp_files(self) -> None:
        self.build_chain(1)
        BranchStore.build_recovery_audit_index(self.chain, self.index)
        before = _read(self.index)
        readonly_dir = os.path.join(self.tmp.name, "readonly")
        os.mkdir(readonly_dir)
        target = os.path.join(readonly_dir, "index.json")
        os.chmod(readonly_dir, 0o555)
        try:
            if os.geteuid() != 0:
                with self.assertRaises(OSError):
                    BranchStore.build_recovery_audit_index(self.chain, target)
                self.assertEqual(
                    [n for n in os.listdir(readonly_dir) if n != "."],
                    [],
                )
        finally:
            os.chmod(readonly_dir, 0o755)
        # The previously published index is byte-for-byte untouched.
        self.assertEqual(_read(self.index), before)

    def test_path_validation(self) -> None:
        self.build_chain(1)
        for bad in (1, 1.0, b"x", [], {}):
            with self.subTest(where="chain", bad=bad):
                with self.assertRaises(TypeError):
                    BranchStore.build_recovery_audit_index(bad, self.index)
            with self.subTest(where="index", bad=bad):
                with self.assertRaises(TypeError):
                    BranchStore.build_recovery_audit_index(self.chain, bad)
        # None is not a legal build index path.
        with self.assertRaises(TypeError):
            BranchStore.build_recovery_audit_index(self.chain, None)
        with self.assertRaises(ValueError):
            BranchStore.build_recovery_audit_index("", self.index)
        with self.assertRaises(ValueError):
            BranchStore.build_recovery_audit_index(self.chain, "")

    def test_index_path_must_differ_from_chain_path(self) -> None:
        self.build_chain(1)
        with self.assertRaises(ValueError):
            BranchStore.build_recovery_audit_index(self.chain, self.chain)
        alias = os.path.join(self.tmp.name, "chain-link.json")
        os.symlink(self.chain, alias)
        with self.assertRaises(ValueError):
            BranchStore.build_recovery_audit_index(self.chain, alias)


class IndexedQueryEquivalenceTests(RecoveryAuditIndexTestBase):
    def test_verify_matches_with_and_without_index(self) -> None:
        heads, _store = self.build_chain(4)
        BranchStore.build_recovery_audit_index(self.chain, self.index)
        for head, expected in (
            (heads[-1], True),
            (heads[0], False),       # rolled-back prefix head
            ("f" * 64, False),       # unrelated digest
        ):
            with self.subTest(head=head[:4]):
                self.assertEqual(
                    BranchStore.verify_recovery_audit_chain(
                        self.chain, head, self.index
                    ),
                    BranchStore.verify_recovery_audit_chain(self.chain, head),
                )
                self.assertIs(
                    BranchStore.verify_recovery_audit_chain(
                        self.chain, head, self.index
                    ),
                    expected,
                )

    def test_diff_matches_item_by_item(self) -> None:
        heads, _store = self.build_chain(5, changing=True)
        BranchStore.build_recovery_audit_index(self.chain, self.index)
        cases = [(1, 1), (1, 2), (1, 5), (2, 4), (3, 5), (4, 5)]
        for start, end in cases:
            with self.subTest(start=start, end=end):
                indexed = BranchStore.diff_recovery_audit_range(
                    self.chain, heads[-1], start, end, self.index
                )
                plain = BranchStore.diff_recovery_audit_range(
                    self.chain, heads[-1], start, end
                )
                self.assertEqual(indexed, plain)
                self.assertIsInstance(indexed, tuple)
                for change in indexed:
                    self.assertEqual(len(change), 5)

    def test_indexed_diff_is_detached(self) -> None:
        heads, _store = self.build_chain(3, changing=True)
        BranchStore.build_recovery_audit_index(self.chain, self.index)
        result = BranchStore.diff_recovery_audit_range(
            self.chain, heads[-1], 1, 3, self.index
        )
        for change in result:
            if isinstance(change[3], dict):
                change[3].clear()
        again = BranchStore.diff_recovery_audit_range(
            self.chain, heads[-1], 1, 3, self.index
        )
        self.assertEqual(
            again,
            BranchStore.diff_recovery_audit_range(
                self.chain, heads[-1], 1, 3
            ),
        )

    def test_queries_are_read_only(self) -> None:
        heads, _store = self.build_chain(3)
        BranchStore.build_recovery_audit_index(self.chain, self.index)
        chain_before = _read(self.chain)
        index_before = _read(self.index)
        root_before = self.snapshot()
        BranchStore.verify_recovery_audit_chain(
            self.chain, heads[-1], self.index
        )
        BranchStore.diff_recovery_audit_range(
            self.chain, heads[-1], 1, 3, self.index
        )
        self.assertEqual(_read(self.chain), chain_before)
        self.assertEqual(_read(self.index), index_before)
        self.assertEqual(self.snapshot(), root_before)


class IndexRebuildTests(RecoveryAuditIndexTestBase):
    def _good_index_bytes(self, heads: list[str]) -> bytes:
        BranchStore.build_recovery_audit_index(self.chain, self.index)
        return _read(self.index)

    def test_missing_index_is_built_then_used(self) -> None:
        heads, _store = self.build_chain(2)
        self.assertFalse(os.path.exists(self.index))
        self.assertTrue(
            BranchStore.verify_recovery_audit_chain(
                self.chain, heads[-1], self.index
            )
        )
        # The query rebuilt the cache from the canonical chain.
        self.assertTrue(os.path.exists(self.index))
        self.assertEqual(
            json.loads(_read(self.index))["head"], heads[-1]
        )

    def test_stale_index_after_append_is_rebuilt_once(self) -> None:
        heads, store = self.build_chain(2, changing=True)
        good = self._good_index_bytes(heads)
        self.rotate_with_change(store, at=99, value=42)
        new_head = self.append_frame(store, heads[-1])
        self.assertTrue(
            BranchStore.verify_recovery_audit_chain(
                self.chain, new_head, self.index
            )
        )
        rebuilt = _read(self.index)
        self.assertNotEqual(rebuilt, good)
        self.assertEqual(json.loads(rebuilt)["head"], new_head)
        # Diff over the whole range works against the rebuilt index.
        self.assertEqual(
            BranchStore.diff_recovery_audit_range(
                self.chain, new_head, 1, 3, self.index
            ),
            BranchStore.diff_recovery_audit_range(self.chain, new_head, 1, 3),
        )

    def test_truncated_index_is_rebuilt(self) -> None:
        heads, _store = self.build_chain(3)
        good = self._good_index_bytes(heads)
        with open(self.index, "wb") as handle:
            handle.write(good[: len(good) // 2])
        self.assertTrue(
            BranchStore.verify_recovery_audit_chain(
                self.chain, heads[-1], self.index
            )
        )
        self.assertEqual(_read(self.index), good)

    def test_corrupted_index_is_rebuilt(self) -> None:
        heads, _store = self.build_chain(3)
        good = bytearray(self._good_index_bytes(heads))
        good[120] ^= 0xFF
        with open(self.index, "wb") as handle:
            handle.write(bytes(good))
        self.assertTrue(
            BranchStore.verify_recovery_audit_chain(
                self.chain, heads[-1], self.index
            )
        )
        self.assertEqual(
            json.loads(_read(self.index))["head"], heads[-1]
        )

    def test_forged_index_summary_is_rebuilt_not_trusted(self) -> None:
        heads, _store = self.build_chain(2)
        good = self._good_index_bytes(heads)
        forged = good.replace(heads[-1].encode("ascii"), b"f" * 64)
        self.assertNotEqual(forged, good)
        with open(self.index, "wb") as handle:
            handle.write(forged)
        # The real head still verifies: the forged cache is discarded and
        # rebuilt from the canonical chain.
        self.assertTrue(
            BranchStore.verify_recovery_audit_chain(
                self.chain, heads[-1], self.index
            )
        )

    def test_forged_index_entry_is_rebuilt(self) -> None:
        heads, _store = self.build_chain(2)
        good = self._good_index_bytes(heads)
        document = json.loads(good)
        document["frames"][0]["digest"] = "a" * 64
        tampered = json.dumps(
            document, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        with open(self.index, "wb") as handle:
            handle.write(tampered)
        self.assertTrue(
            BranchStore.verify_recovery_audit_chain(
                self.chain, heads[-1], self.index
            )
        )

    def test_index_rebuild_does_not_mask_a_bad_chain(self) -> None:
        heads, _store = self.build_chain(2)
        self._good_index_bytes(heads)
        good_chain = _read(self.chain)
        for corruption in (
            good_chain + b" ",
            good_chain + b"\n",
            b"\xef\xbb\xbf" + good_chain,
        ):
            with open(self.chain, "wb") as handle:
                handle.write(corruption)
            with self.subTest(corruption=corruption[:3]):
                with self.assertRaises(ValueError):
                    BranchStore.verify_recovery_audit_chain(
                        self.chain, heads[-1], self.index
                    )
            with open(self.chain, "wb") as handle:
                handle.write(good_chain)

    def test_tampered_frame_raises_valueerror_with_index(self) -> None:
        heads, _store = self.build_chain(2)
        self._good_index_bytes(heads)
        document = json.loads(_read(self.chain))
        document["frames"][0]["digest"] = "a" * 64
        with open(self.chain, "wb") as handle:
            handle.write(
                json.dumps(
                    document, separators=(",", ":"), ensure_ascii=False
                ).encode("utf-8")
            )
        with self.assertRaises(ValueError):
            BranchStore.verify_recovery_audit_chain(
                self.chain, heads[-1], self.index
            )


class IndexedValidationTests(RecoveryAuditIndexTestBase):
    def test_verify_argument_validation(self) -> None:
        heads, _store = self.build_chain(1)
        for bad in (1, 1.0, b"x", [], {}):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    BranchStore.verify_recovery_audit_chain(
                        bad, heads[-1], self.index
                    )
                with self.assertRaises(TypeError):
                    BranchStore.verify_recovery_audit_chain(
                        self.chain, bad, self.index
                    )
                with self.assertRaises(TypeError):
                    BranchStore.verify_recovery_audit_chain(
                        self.chain, heads[-1], bad
                    )
        with self.assertRaises(ValueError):
            BranchStore.verify_recovery_audit_chain(
                self.chain, heads[-1], ""
            )
        with self.assertRaises(ValueError):
            BranchStore.verify_recovery_audit_chain(
                "", heads[-1], self.index
            )

    def test_diff_sequence_validation_with_index(self) -> None:
        heads, _store = self.build_chain(3)
        BranchStore.build_recovery_audit_index(self.chain, self.index)
        for bad in (1.0, "1", [1], None, True, False):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    BranchStore.diff_recovery_audit_range(
                        self.chain, heads[-1], bad, 3, self.index
                    )
                with self.assertRaises(TypeError):
                    BranchStore.diff_recovery_audit_range(
                        self.chain, heads[-1], 1, bad, self.index
                    )
        for start, end in ((0, 3), (1, 0), (3, 1), (1, 4), (4, 4)):
            with self.subTest(start=start, end=end):
                with self.assertRaises(ValueError):
                    BranchStore.diff_recovery_audit_range(
                        self.chain, heads[-1], start, end, self.index
                    )

    def test_head_mismatch_raises_valueerror_with_index(self) -> None:
        heads, _store = self.build_chain(3)
        BranchStore.build_recovery_audit_index(self.chain, self.index)
        with self.assertRaises(ValueError):
            BranchStore.diff_recovery_audit_range(
                self.chain, "f" * 64, 1, 3, self.index
            )
        with self.assertRaises(ValueError):
            BranchStore.diff_recovery_audit_range(
                self.chain, heads[0], 1, 3, self.index
            )

    def test_missing_chain_raises_oserror_with_index(self) -> None:
        missing = os.path.join(self.tmp.name, "nope.json")
        with self.assertRaises(OSError):
            BranchStore.verify_recovery_audit_chain(
                missing, "f" * 64, self.index
            )
        with self.assertRaises(OSError):
            BranchStore.diff_recovery_audit_range(
                missing, "f" * 64, 1, 2, self.index
            )

    def test_omitting_index_preserves_index_free_behaviour(self) -> None:
        heads, _store = self.build_chain(2)
        # No index is created when the argument is omitted.
        self.assertTrue(
            BranchStore.verify_recovery_audit_chain(self.chain, heads[-1])
        )
        self.assertFalse(os.path.exists(self.index))
        self.assertIsInstance(
            BranchStore.diff_recovery_audit_range(
                self.chain, heads[-1], 1, 2
            ),
            tuple,
        )
        self.assertFalse(os.path.exists(self.index))


class CanonicalParityTests(RecoveryAuditIndexTestBase):
    """The indexed streaming path must accept or reject a chain exactly
    like the index-free whole-document loader."""

    def _write(self, document: object) -> None:
        with open(self.chain, "wb") as handle:
            handle.write(
                json.dumps(
                    document,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
            )

    def _plain_verifies(self, head: str) -> bool:
        try:
            return BranchStore.verify_recovery_audit_chain(self.chain, head)
        except ValueError:
            return False

    def _index_verifies(self, head: str) -> bool:
        try:
            BranchStore.build_recovery_audit_index(self.chain, self.index)
            return BranchStore.verify_recovery_audit_chain(
                self.chain, head, self.index
            )
        except ValueError:
            return False

    def test_reordered_frame_and_envelope_keys_accepted_like_loader(
        self,
    ) -> None:
        heads, _store = self.build_chain(1)
        raw = _read(self.chain)
        document = json.loads(raw)
        frame = document["frames"][0]
        # Reorder the frame's keys; the canonical loader accepts the
        # document because its compact re-encoding round-trips.
        reordered_frame = (
            '{"prev":'
            + json.dumps(frame["prev"])
            + ',"seq":1,"record":'
            + json.dumps(frame["record"])
            + ',"digest":'
            + json.dumps(frame["digest"])
            + "}"
        )
        canonical_frame = json.dumps(frame, separators=(",", ":"))
        chain_text = json.dumps(document, separators=(",", ":"))
        reordered = chain_text.replace(
            canonical_frame, reordered_frame, 1
        ).encode("utf-8")
        with open(self.chain, "wb") as handle:
            handle.write(reordered)
        self.assertTrue(self._plain_verifies(heads[-1]))
        self.assertTrue(self._index_verifies(heads[-1]))

        # Reorder the envelope keys (frames first) as well.
        self._write(
            {
                "frames": document["frames"],
                "format": document["format"],
                "version": document["version"],
            }
        )
        self.assertTrue(self._plain_verifies(heads[-1]))
        self.assertTrue(self._index_verifies(heads[-1]))

    def test_compact_whitespace_rejected_like_loader(self) -> None:
        heads, _store = self.build_chain(1)
        raw = _read(self.chain)
        for corrupted in (
            raw.replace(b'"seq":1', b'"seq": 1', 1),
            raw.replace(b'"frames":[', b'"frames" : [', 1),
            raw.replace(b'"frames":[{', b'"frames":[ {', 1),
            raw + b"\n",
        ):
            with open(self.chain, "wb") as handle:
                handle.write(corrupted)
            with self.subTest(case=corrupted[:12]):
                self.assertFalse(self._plain_verifies(heads[-1]))
                self.assertFalse(self._index_verifies(heads[-1]))

    def test_duplicate_and_extra_keys_rejected_like_loader(self) -> None:
        heads, _store = self.build_chain(1)
        raw = _read(self.chain)
        text = raw.decode("utf-8")
        cases = (
            text.replace('"version":1', '"version":1,"version":1', 1),
            text.replace('"format"', '"formatx":0,"format"', 1),
            text.replace('{"seq":1', '{"seq":1,"seq":1', 1),
        )
        for case in cases:
            with open(self.chain, "wb") as handle:
                handle.write(case.encode("utf-8"))
            with self.subTest(case=case[20:40]):
                self.assertFalse(self._plain_verifies(heads[-1]))
                self.assertFalse(self._index_verifies(heads[-1]))


class BoundedMemoryTests(RecoveryAuditIndexTestBase):
    def setUp(self) -> None:
        super().setUp()
        # A small fixed streaming buffer makes any per-frame or
        # per-distance growth obvious against the constant baseline.
        self._original_buffer = branches_module._RECOVERY_STREAM_BUFFER_SIZE
        branches_module._RECOVERY_STREAM_BUFFER_SIZE = 256
        branches_module._JsonWindowReader.CHUNK = 256

    def tearDown(self) -> None:
        branches_module._RECOVERY_STREAM_BUFFER_SIZE = self._original_buffer
        branches_module._JsonWindowReader.CHUNK = self._original_buffer
        super().tearDown()

    def _peak_kb(self, operation) -> float:
        tracemalloc.start()
        try:
            operation()
            _current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        return peak / 1024.0

    def _build_on(
        self, chain_path: str, frames: int
    ) -> list[str]:
        graph = EventGraph()
        graph.add("root", 0, (), {"seed": 7})
        store = BranchStore(graph)
        store.create("main", "root")
        store.append("main", "m1", 1, {"a": 1})
        cp = chain_path + ".cp.json"
        journal = chain_path + ".journal.json"
        store.save_checkpoint(cp)
        store.enable_journal(cp, journal)
        store.rotate_generation(self.root)
        heads: list[str] = []
        for _ in range(frames):
            heads.append(
                BranchStore.append_recovery_audit_chain(
                    self.root, chain_path, heads[-1] if heads else None
                )
            )
        return heads

    def test_build_memory_does_not_grow_with_frame_count(self) -> None:
        small_chain = os.path.join(self.tmp.name, "small.json")
        small_index = os.path.join(self.tmp.name, "small.idx")
        large_chain = os.path.join(self.tmp.name, "large.json")
        large_index = os.path.join(self.tmp.name, "large.idx")
        self._build_on(small_chain, 8)
        self._build_on(large_chain, 64)
        small_peak = self._peak_kb(
            lambda: BranchStore.build_recovery_audit_index(
                small_chain, small_index
            )
        )
        large_peak = self._peak_kb(
            lambda: BranchStore.build_recovery_audit_index(
                large_chain, large_index
            )
        )
        # Eight times as many frames must not materially grow memory;
        # allow a modest constant slack for interpreter/runtime noise.
        self.assertLess(large_peak, small_peak * 3 + 64)
        self.assertGreater(
            os.path.getsize(large_chain), os.path.getsize(small_chain) * 4
        )

    def test_query_memory_is_independent_of_chain_length(self) -> None:
        chains = {}
        for frames in (8, 64):
            chain_path = os.path.join(self.tmp.name, f"c{frames}.json")
            index_path = os.path.join(self.tmp.name, f"i{frames}.json")
            heads = self._build_on(chain_path, frames)
            BranchStore.build_recovery_audit_index(chain_path, index_path)
            chains[frames] = (chain_path, index_path, heads)
        c8, i8, h8 = chains[8]
        c64, i64, h64 = chains[64]
        baseline = self._peak_kb(
            lambda: BranchStore.verify_recovery_audit_chain(
                c8, h8[-1], i8
            )
        )
        verify_peak = self._peak_kb(
            lambda: BranchStore.verify_recovery_audit_chain(
                c64, h64[-1], i64
            )
        )
        # A range spanning the entire 64-frame chain stays bounded.
        diff_peak = self._peak_kb(
            lambda: BranchStore.diff_recovery_audit_range(
                c64, h64[-1], 1, 64, i64
            )
        )
        self.assertLess(verify_peak, baseline * 3 + 64)
        self.assertLess(diff_peak, baseline * 3 + 64)
        # Sanity: the large chain file really is many frames larger.
        self.assertGreater(
            os.path.getsize(c64), os.path.getsize(c8) * 4
        )


class LockCoordinationTests(RecoveryAuditIndexTestBase):
    def test_build_competes_for_the_same_chain_lock(self) -> None:
        heads, _store = self.build_chain(1)
        real_acquire = BranchStore._acquire_chain_lock
        lock_names: list[str] = []

        def recording_acquire(fd, timeout):  # type: ignore[no-untyped-def]
            try:
                link = os.readlink(f"/proc/self/fd/{fd}")
            except OSError:
                link = ""
            lock_names.append(link)
            return real_acquire(fd, timeout)

        with mock.patch.object(
            BranchStore,
            "_acquire_chain_lock",
            side_effect=recording_acquire,
        ):
            BranchStore.build_recovery_audit_index(self.chain, self.index)
            BranchStore.verify_recovery_audit_chain(
                self.chain, heads[-1], self.index
            )

        canonical = os.path.realpath(self.chain) + ".lock"
        self.assertTrue(lock_names)
        for name in lock_names:
            self.assertEqual(name, canonical)

    def test_concurrent_appends_builds_and_queries_never_tear(self) -> None:
        import subprocess
        import sys

        heads, store = self.build_chain(1, changing=True)
        # Keep one valid head per worker by reading the current tip is
        # not possible across processes; instead workers only verify the
        # chain's internal soundness by rebuilding and checking the
        # returned head round-trips through a fresh verify. Appenders
        # coordinate by always rebuilding the head from the chain file.
        worker = (
            "import sys, json, os\n"
            "from city_twin.branches import BranchStore\n"
            "root, chain, index, mode, rounds = sys.argv[1:6]\n"
            "rounds = int(rounds)\n"
            "for _ in range(rounds):\n"
            "    if mode == 'build':\n"
            "        head = BranchStore.build_recovery_audit_index(chain, index)\n"
            "        assert BranchStore.verify_recovery_audit_chain(chain, head, index)\n"
            "    elif mode == 'query':\n"
            "        # Authenticate whatever complete tip the index/chain hold.\n"
            "        head = BranchStore.build_recovery_audit_index(chain, index)\n"
            "        BranchStore.diff_recovery_audit_range(chain, head, 1, 1, index)\n"
            "    else:\n"
            "        # Appender: read the current tip under the lock,\n"
            "        # then append; retry when another appender won.\n"
            "        while True:\n"
            "            head = BranchStore.build_recovery_audit_index(chain, index)\n"
            "            try:\n"
            "                new = BranchStore.append_recovery_audit_chain(root, chain, head)\n"
            "                break\n"
            "            except RuntimeError:\n"
            "                continue\n"
            "        assert new != head\n"
            "print('ok')\n"
        )
        env = dict(os.environ, PYTHONPATH=os.path.dirname(os.path.dirname(__file__)))
        processes = []
        for mode, rounds, count in (
            ("append", 3, 3),
            ("build", 4, 2),
            ("query", 4, 2),
        ):
            for _ in range(count):
                processes.append(
                    subprocess.Popen(
                        [sys.executable, "-c", worker, self.root,
                         self.chain, self.index, mode, str(rounds)],
                        env=env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )
                )
        failures = []
        for process in processes:
            out, err = process.communicate(timeout=60)
            if process.returncode != 0:
                failures.append(err.decode("utf-8", "replace"))
        self.assertEqual(failures, [], "".join(failures))
        # The chain and a fresh index are mutually consistent and sound.
        final_head = BranchStore.build_recovery_audit_index(
            self.chain, self.index
        )
        self.assertTrue(
            BranchStore.verify_recovery_audit_chain(
                self.chain, final_head, self.index
            )
        )
        # Exactly the initial frame plus three appenders x three rounds.
        document = json.loads(_read(self.chain))
        self.assertEqual(len(document["frames"]), 1 + 3 * 3)


if __name__ == "__main__":
    unittest.main()
