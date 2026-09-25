import hashlib
import json
import os
import tempfile
import unittest

from city_twin.branches import BranchStore
from city_twin.event_graph import EventGraph


def _read(path: str) -> bytes:
    with open(path, "rb") as handle:
        return handle.read()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value) -> str:
    return json.dumps(
        value, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def make_history() -> BranchStore:
    graph = EventGraph()
    graph.add("root", 0, (), {"seed": 7})
    store = BranchStore(graph)
    store.create("main", "root")
    store.append("main", "m1", 1, {"a": 1})
    store.append("main", "m2", 3, {"a": 2})
    return store


def frame_digest(frame: dict) -> str:
    return _sha256(
        _canonical(
            {
                "seq": frame["seq"],
                "previous": frame["previous"],
                "record": frame["record"],
            }
        ).encode("utf-8")
    )


class RecoveryAuditChainTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = os.path.join(self.tmp.name, "gens")
        os.mkdir(self.root)
        self.chain = os.path.join(self.tmp.name, "chain.json")
        self.cp = os.path.join(self.tmp.name, "cp.json")
        self.journal = os.path.join(self.tmp.name, "journal.json")

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

    def snapshot_root(self) -> dict[str, bytes]:
        captured: dict[str, bytes] = {}
        for dirpath, _dirnames, filenames in os.walk(self.root):
            for filename in filenames:
                path = os.path.join(dirpath, filename)
                captured[path] = _read(path)
        return captured

    def append(self, previous_head=None) -> str:
        return BranchStore.append_recovery_audit_chain(
            self.root, self.chain, previous_head
        )

    def read_chain(self) -> dict:
        return json.loads(_read(self.chain).decode("utf-8"))


class AppendRecoveryAuditChainTests(RecoveryAuditChainTestBase):
    def test_first_append_creates_canonical_chain(self) -> None:
        self.rotate_three()
        head = self.append()
        self.assertIsInstance(head, str)
        self.assertEqual(len(head), 64)
        raw = _read(self.chain)
        self.assertFalse(raw.startswith(b"\xef\xbb\xbf"))
        text = raw.decode("utf-8")
        self.assertFalse(text.endswith("\n"))
        self.assertNotIn(" ", text)
        document = json.loads(text)
        self.assertEqual(list(document), ["format", "version", "frames"])
        self.assertEqual(
            document["format"], "branching-city-twin/recovery-audit-chain"
        )
        self.assertEqual(document["version"], 1)
        self.assertEqual(len(document["frames"]), 1)
        frame = document["frames"][0]
        self.assertEqual(list(frame), ["seq", "previous", "record", "digest"])
        self.assertEqual(frame["seq"], 1)
        self.assertEqual(frame["previous"], "0" * 64)
        self.assertEqual(frame["digest"], head)
        self.assertEqual(frame["digest"], frame_digest(frame))

    def test_frame_record_is_the_exported_audit_record(self) -> None:
        self.rotate_three()
        self.append()
        frame = self.read_chain()["frames"][0]
        self.assertEqual(
            frame["record"], BranchStore.export_recovery_audit(self.root)
        )

    def test_append_extends_and_chains(self) -> None:
        self.rotate_three()
        head_one = self.append()
        head_two = self.append(head_one)
        self.assertNotEqual(head_one, head_two)
        frames = self.read_chain()["frames"]
        self.assertEqual([f["seq"] for f in frames], [1, 2])
        self.assertEqual(frames[1]["previous"], head_one)
        self.assertEqual(frames[1]["digest"], head_two)
        # The evidence did not change: the record is identical, but the
        # sampling fact is still preserved as a new frame.
        self.assertEqual(frames[0]["record"], frames[1]["record"])

    def test_append_after_evidence_change(self) -> None:
        store = self.rotate_three()
        head_one = self.append()
        store.append("main", "m6", 8, {"a": 6})
        store.rotate_generation(self.root)
        head_two = self.append(head_one)
        frames = self.read_chain()["frames"]
        self.assertEqual(len(frames), 2)
        self.assertNotEqual(frames[0]["record"], frames[1]["record"])
        self.assertEqual(
            json.loads(frames[1]["record"])["selected"],
            "generation-0000000000000004",
        )
        self.assertTrue(
            BranchStore.verify_recovery_audit_chain(self.chain, head_two)
        )

    def test_first_append_requires_none_head(self) -> None:
        self.rotate_three()
        with self.assertRaises(RuntimeError):
            self.append("0" * 64)
        self.assertFalse(os.path.exists(self.chain))

    def test_append_with_stale_head_raises_and_preserves_file(self) -> None:
        self.rotate_three()
        head = self.append()
        before = _read(self.chain)
        with self.assertRaises(RuntimeError):
            self.append("1" * 64)
        with self.assertRaises(RuntimeError):
            self.append(None)
        self.assertEqual(_read(self.chain), before)
        # The chain still verifies against its true head.
        self.assertTrue(
            BranchStore.verify_recovery_audit_chain(self.chain, head)
        )

    def test_append_rejects_corrupt_existing_chain(self) -> None:
        self.rotate_three()
        head = self.append()
        with open(self.chain, "wb") as handle:
            handle.write(b'{"format":"x"}')
        with self.assertRaises(ValueError):
            self.append(head)
        self.assertEqual(_read(self.chain), b'{"format":"x"}')

    def test_append_without_recoverable_generation_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.append()
        self.assertFalse(os.path.exists(self.chain))

    def test_append_does_not_touch_generation_root(self) -> None:
        self.rotate_three()
        before = self.snapshot_root()
        head = self.append()
        self.append(head)
        self.assertEqual(self.snapshot_root(), before)

    def test_root_validation(self) -> None:
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
        with self.assertRaises(OSError):
            BranchStore.append_recovery_audit_chain(
                self.root,
                os.path.join(self.tmp.name, "missing-dir", "chain.json"),
                None,
            )

    def test_head_validation(self) -> None:
        self.rotate_three()
        for bad in (1, 1.0, b"x", ["a"], True):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    BranchStore.append_recovery_audit_chain(
                        self.root, self.chain, bad
                    )
        for bad in ("", "0" * 63, "0" * 65, "g" * 64, "A" * 64):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    BranchStore.append_recovery_audit_chain(
                        self.root, self.chain, bad
                    )
        self.assertFalse(os.path.exists(self.chain))


class VerifyRecoveryAuditChainTests(RecoveryAuditChainTestBase):
    def test_verify_roundtrip(self) -> None:
        self.rotate_three()
        head_one = self.append()
        self.assertTrue(
            BranchStore.verify_recovery_audit_chain(self.chain, head_one)
        )
        head_two = self.append(head_one)
        self.assertTrue(
            BranchStore.verify_recovery_audit_chain(self.chain, head_two)
        )
        # The old head no longer matches the extended chain.
        self.assertFalse(
            BranchStore.verify_recovery_audit_chain(self.chain, head_one)
        )

    def test_verify_with_other_digest_returns_false(self) -> None:
        self.rotate_three()
        self.append()
        self.assertFalse(
            BranchStore.verify_recovery_audit_chain(self.chain, "1" * 64)
        )

    def test_verify_is_read_only(self) -> None:
        self.rotate_three()
        head = self.append()
        before_file = _read(self.chain)
        before_root = self.snapshot_root()
        self.assertTrue(
            BranchStore.verify_recovery_audit_chain(self.chain, head)
        )
        self.assertEqual(_read(self.chain), before_file)
        self.assertEqual(self.snapshot_root(), before_root)

    def test_missing_chain_file_raises_oserror(self) -> None:
        with self.assertRaises(OSError):
            BranchStore.verify_recovery_audit_chain(self.chain, "0" * 64)

    def test_tampered_frames_rejected(self) -> None:
        store = self.rotate_three()
        head_one = self.append()
        # Rotate once more so the two frames seal different records.
        store.append("main", "m6", 8, {"a": 6})
        store.rotate_generation(self.root)
        self.append(head_one)
        document = self.read_chain()
        head = document["frames"][-1]["digest"]
        good = _read(self.chain)

        def rewritten(mutate) -> None:
            candidate = json.loads(good.decode("utf-8"))
            mutate(candidate)
            with open(self.chain, "wb") as handle:
                handle.write(_canonical(candidate).encode("utf-8"))
            try:
                with self.assertRaises(ValueError):
                    BranchStore.verify_recovery_audit_chain(self.chain, head)
            finally:
                with open(self.chain, "wb") as handle:
                    handle.write(good)

        # Truncating the last frame leaves a well-formed shorter chain:
        # it is caught by the external head digest, not the structure.
        candidate = json.loads(good.decode("utf-8"))
        candidate["frames"].pop()
        with open(self.chain, "wb") as handle:
            handle.write(_canonical(candidate).encode("utf-8"))
        self.assertFalse(
            BranchStore.verify_recovery_audit_chain(self.chain, head)
        )
        with open(self.chain, "wb") as handle:
            handle.write(good)
        # Duplicated frame.
        rewritten(lambda d: d["frames"].append(d["frames"][0]))
        # Swapped frames.
        rewritten(lambda d: d["frames"].reverse())
        # Renumbered without re-digesting.
        rewritten(lambda d: d["frames"][1].update(seq=3))
        # Rewritten record.
        rewritten(
            lambda d: d["frames"][0].update(
                record=d["frames"][1]["record"]
            )
        )
        # Empty chain document.
        rewritten(lambda d: d["frames"].clear())

    def test_encoding_defects_rejected(self) -> None:
        self.rotate_three()
        head = self.append()
        good = _read(self.chain)
        for raw in (
            b"\xef\xbb\xbf" + good,
            good + b"\n",
            good[:-1],
            b"not json",
            b"\xff\xfe",
            json.dumps(self.read_chain(), indent=2).encode("utf-8"),
        ):
            with self.subTest(raw=raw[:16]):
                with open(self.chain, "wb") as handle:
                    handle.write(raw)
                with self.assertRaises(ValueError):
                    BranchStore.verify_recovery_audit_chain(self.chain, head)
        with open(self.chain, "wb") as handle:
            handle.write(good)
        self.assertTrue(
            BranchStore.verify_recovery_audit_chain(self.chain, head)
        )

    def test_record_checksum_inside_chain_enforced(self) -> None:
        self.rotate_three()
        head = self.append()
        document = self.read_chain()
        record = json.loads(document["frames"][0]["record"])
        record["selected"] = "generation-0000000000000001"
        document["frames"][0]["record"] = _canonical(record)
        document["frames"][0]["digest"] = frame_digest(document["frames"][0])
        with open(self.chain, "wb") as handle:
            handle.write(_canonical(document).encode("utf-8"))
        with self.assertRaises(ValueError):
            BranchStore.verify_recovery_audit_chain(self.chain, head)

    def test_argument_validation(self) -> None:
        self.rotate_three()
        head = self.append()
        for bad in (1, 1.0, b"x", None, ["a"]):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    BranchStore.verify_recovery_audit_chain(bad, head)
        with self.assertRaises(ValueError):
            BranchStore.verify_recovery_audit_chain("", head)
        for bad in (1, 1.0, b"x", ["a"], True):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    BranchStore.verify_recovery_audit_chain(self.chain, bad)
        for bad in ("", "0" * 63, "zz"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    BranchStore.verify_recovery_audit_chain(self.chain, bad)


class DiffRecoveryAuditRangeTests(RecoveryAuditChainTestBase):
    def test_same_sequence_returns_empty_tuple(self) -> None:
        self.rotate_three()
        head = self.append()
        self.assertEqual(
            BranchStore.diff_recovery_audit_range(self.chain, head, 1, 1), ()
        )

    def test_unchanged_frames_return_empty_tuple(self) -> None:
        self.rotate_three()
        head_one = self.append()
        head_two = self.append(head_one)
        self.assertEqual(
            BranchStore.diff_recovery_audit_range(self.chain, head_two, 1, 2),
            (),
        )

    def test_range_diff_matches_live_diff(self) -> None:
        store = self.rotate_three()
        head_one = self.append()
        record_one = BranchStore.export_recovery_audit(self.root)
        store.append("main", "m6", 8, {"a": 6})
        store.rotate_generation(self.root)
        head_two = self.append(head_one)
        changes = BranchStore.diff_recovery_audit_range(
            self.chain, head_two, 1, 2
        )
        self.assertIsInstance(changes, tuple)
        self.assertNotEqual(changes, ())
        # Identical to diffing the sealed first record against live
        # evidence now.
        self.assertEqual(
            changes,
            BranchStore.diff_recovery_audit(self.root, record_one),
        )
        categories = [entry[0] for entry in changes]
        self.assertEqual(categories[0], "pointer")
        # Reversed direction flips before/after.
        reverse = BranchStore.diff_recovery_audit_range(
            self.chain, head_two, 1, 1
        )
        self.assertEqual(reverse, ())

    def test_subrange_over_three_frames(self) -> None:
        store = self.rotate_three()
        head_one = self.append()
        store.append("main", "m6", 8, {"a": 6})
        store.rotate_generation(self.root)
        head_two = self.append(head_one)
        store.append("main", "m7", 9, {"a": 7})
        store.rotate_generation(self.root)
        head_three = self.append(head_two)
        changes = BranchStore.diff_recovery_audit_range(
            self.chain, head_three, 1, 3
        )
        selections = [
            entry for entry in changes
            if entry[0] == "chain" and entry[1] is None
        ]
        self.assertEqual(len(selections), 1)
        self.assertEqual(
            selections[0][3], "generation-0000000000000003"
        )
        self.assertEqual(
            selections[0][4], "generation-0000000000000005"
        )
        # The middle sub-range sees only the first rotation.
        mid = BranchStore.diff_recovery_audit_range(
            self.chain, head_three, 2, 3
        )
        mid_selections = [
            entry for entry in mid
            if entry[0] == "chain" and entry[1] is None
        ]
        self.assertEqual(
            mid_selections[0][3], "generation-0000000000000004"
        )

    def test_head_mismatch_raises_valueerror(self) -> None:
        self.rotate_three()
        self.append()
        with self.assertRaises(ValueError):
            BranchStore.diff_recovery_audit_range(
                self.chain, "1" * 64, 1, 1
            )
        with self.assertRaises(ValueError):
            BranchStore.diff_recovery_audit_range(self.chain, None, 1, 1)

    def test_sequence_validation(self) -> None:
        self.rotate_three()
        head_one = self.append()
        head = self.append(head_one)
        for bad in (True, 1.0, "1", None, [1]):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    BranchStore.diff_recovery_audit_range(
                        self.chain, head, bad, 1
                    )
                with self.assertRaises(TypeError):
                    BranchStore.diff_recovery_audit_range(
                        self.chain, head, 1, bad
                    )
        for start, end in ((0, 1), (1, 0), (-2, -1), (2, 1), (1, 3), (3, 3)):
            with self.subTest(start=start, end=end):
                with self.assertRaises(ValueError):
                    BranchStore.diff_recovery_audit_range(
                        self.chain, head, start, end
                    )

    def test_tampered_chain_raises_valueerror(self) -> None:
        self.rotate_three()
        head = self.append()
        document = self.read_chain()
        document["frames"][0]["seq"] = 7
        with open(self.chain, "wb") as handle:
            handle.write(_canonical(document).encode("utf-8"))
        with self.assertRaises(ValueError):
            BranchStore.diff_recovery_audit_range(self.chain, head, 1, 1)

    def test_diff_is_read_only(self) -> None:
        self.rotate_three()
        head_one = self.append()
        head_two = self.append(head_one)
        before_file = _read(self.chain)
        before_root = self.snapshot_root()
        BranchStore.diff_recovery_audit_range(self.chain, head_two, 1, 2)
        self.assertEqual(_read(self.chain), before_file)
        self.assertEqual(self.snapshot_root(), before_root)

    def test_argument_validation_order(self) -> None:
        self.rotate_three()
        head = self.append()
        for bad in (1, 1.0, b"x", None):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    BranchStore.diff_recovery_audit_range(bad, head, 1, 1)
        with self.assertRaises(ValueError):
            BranchStore.diff_recovery_audit_range("", head, 1, 1)
        with self.assertRaises(OSError):
            BranchStore.diff_recovery_audit_range(
                os.path.join(self.tmp.name, "nope.json"), head, 1, 1
            )


if __name__ == "__main__":
    unittest.main()
