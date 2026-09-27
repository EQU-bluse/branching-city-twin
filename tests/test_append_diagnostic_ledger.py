"""Tests for the segmented, append-only diagnostic ledger directory.

``BranchStore.append_diagnostic_ledger`` publishes time-stamped
recovery diagnostic chains into a ledger directory as immutable,
digest-chained segment files behind an atomically switched manifest,
with compare-and-swap snapshot guards, cross-process locking, bounded
segment sizes and rotation that keeps continuity evidence in the
manifest. ``BranchStore.page_diagnostic_ledger`` pages a ledger
directory's retained segments with the same filters, ordering, return
structure and snapshot-bound cursors as a single-file ledger.

These tests cover:

* argument validation -- directory, entries shape and authentication,
  expected snapshot, segment limit, retain count and lock timeout;
* the compare-and-swap publish protocol: an empty expectation for the
  first publish, the current snapshot digest afterwards, and at most
  one winner among callers racing with the same old snapshot;
* segment layout: canonical immutable segment files bounded by the
  segment limit, linked by digest, described exactly by the manifest;
* rotation: the oldest segments are removed beyond the retain count
  while the manifest preserves the removed prefix's last digest,
  entry count and time bounds, and the first retained segment still
  links to them;
* directory paging: only retained segments are visible, ordering and
  filtering match the single-file ledger, and cursors raise
  ``RuntimeError`` after an append, rotation or truncation;
* durability: a failed publish leaves the previous snapshot
  byte-for-byte readable and leaks no partial result;
* corruption: a defective manifest, segment encoding or digest chain
  raises ``ValueError`` with no partial page.
"""

import hashlib
import json
import os
import tempfile
import threading
import unittest
from unittest import mock

from city_twin.branches import BranchStore


def reseal(document: dict) -> str:
    """Re-serialize a (possibly tampered) document with a fresh
    checksum over every other field."""
    document = dict(document)
    body = {
        key: value for key, value in document.items() if key != "checksum"
    }
    document["checksum"] = hashlib.sha256(
        BranchStore._canonical_json(body).encode("utf-8")
    ).hexdigest()
    return BranchStore._canonical_json(document)


_PENDING_SHAPES = {
    "pending_rollback": {
        "phase": "index",
        "index": {"state": "new", "matches": "new"},
        "progress": {"state": "old", "matches": "old"},
        "backups": {
            "index": {"present": True, "intact": True},
            "progress": {"present": True, "intact": True},
        },
    },
    "pending_commit": {
        "phase": "progress",
        "index": {"state": "new", "matches": "new"},
        "progress": {"state": "new", "matches": "new"},
        "backups": {
            "index": {"present": True, "intact": True},
            "progress": {"present": True, "intact": True},
        },
    },
}

_TERMINAL_SHAPE = {
    "phase": "",
    "index": {"state": "present", "matches": ""},
    "progress": {"state": "present", "matches": ""},
    "backups": {
        "index": {"present": False, "intact": False},
        "progress": {"present": False, "intact": False},
    },
}


def make_chain(
    transaction: str,
    dispositions: tuple[str, ...] = ("pending_rollback", "rolled_back"),
) -> tuple[str, ...]:
    """Build one authentic diagnostic chain for ``transaction``."""
    records = []
    prev = "0" * 64
    for index, disposition in enumerate(dispositions):
        shape = (
            _PENDING_SHAPES[disposition] if index == 0 else _TERMINAL_SHAPE
        )
        document = {
            "format": "branching-city-twin/recovery-diagnostic",
            "version": 1,
            "transaction": transaction,
            "prev": prev,
            **shape,
            "disposition": disposition,
            "reason": disposition,
            "checksum": "",
        }
        text = reseal(document)
        prev = json.loads(text)["checksum"]
        records.append(text)
    chain = tuple(records)
    assert BranchStore.verify_recovery_diagnostics(chain)
    return chain


def transaction(tag: int) -> str:
    """One distinct 64-hex transaction identity per tag."""
    return hashlib.sha256(f"transaction-{tag}".encode("utf-8")).hexdigest()


class AppendTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = os.path.join(self.tmp.name, "ledger")
        os.mkdir(self.dir)
        self.chains = [make_chain(transaction(i)) for i in range(10)]

    def append(self, **overrides):
        arguments = {
            "directory": self.dir,
            "entries": (),
            "expected": None,
            "segment_limit": 2,
            "retain": 10,
            "timeout": None,
        }
        arguments.update(overrides)
        return BranchStore.append_diagnostic_ledger(**arguments)

    def page(self, **overrides):
        arguments = {
            "path": self.dir,
            "start": None,
            "end": None,
            "transactions": None,
            "limit": 100,
            "cursor": None,
        }
        arguments.update(overrides)
        return BranchStore.page_diagnostic_ledger(**arguments)

    def manifest_path(self) -> str:
        return os.path.join(self.dir, "manifest.json")

    def read_manifest_raw(self) -> bytes:
        with open(self.manifest_path(), "rb") as handle:
            return handle.read()

    def read_manifest(self) -> dict:
        return json.loads(self.read_manifest_raw().decode("utf-8"))

    def read_segment_raw(self, name: str) -> bytes:
        with open(os.path.join(self.dir, name), "rb") as handle:
            return handle.read()

    def segment_names(self) -> list[str]:
        return sorted(
            name
            for name in os.listdir(self.dir)
            if name.startswith("segment-")
        )


class AppendValidationTests(AppendTestBase):
    def test_directory_validation(self) -> None:
        for bad in (1, 1.5, b"x", ["x"], {"x": 1}, object(), None):
            with self.subTest(bad=type(bad).__name__):
                with self.assertRaises(TypeError):
                    self.append(directory=bad)
        with self.assertRaises(ValueError):
            self.append(directory="")

    def test_missing_directory_raises_oserror(self) -> None:
        missing = os.path.join(self.tmp.name, "nope")
        with self.assertRaises(OSError):
            self.append(directory=missing)
        # The directory is never created implicitly, and no lock-file
        # residue is left beside it.
        self.assertFalse(os.path.exists(missing))
        self.assertFalse(os.path.exists(missing + ".lock"))
        # An existing non-directory is not a ledger directory either.
        file_path = os.path.join(self.tmp.name, "file")
        with open(file_path, "w", encoding="utf-8") as handle:
            handle.write("x")
        with self.assertRaises(OSError):
            self.append(directory=file_path)

    def test_entries_container_and_shape(self) -> None:
        for bad in ([], "x", 1, None, {"a": 1}):
            with self.subTest(bad=type(bad).__name__):
                with self.assertRaises(TypeError):
                    self.append(entries=bad)
        chain = self.chains[0]
        for bad_entry in ((1,), (1, chain, chain), [1, chain], "x1", 1):
            with self.subTest(bad=bad_entry):
                with self.assertRaises(TypeError):
                    self.append(entries=(bad_entry,))

    def test_time_type_and_value(self) -> None:
        chain = self.chains[0]
        for bad_time in (True, False, 1.5, "1", None, b"1"):
            with self.subTest(bad=bad_time):
                with self.assertRaises(TypeError):
                    self.append(entries=((bad_time, chain),))
        with self.assertRaises(ValueError):
            self.append(entries=((-1, chain),))
        with self.assertRaises(ValueError):
            self.append(
                entries=((2, self.chains[0]), (1, self.chains[1]))
            )

    def test_chain_validation(self) -> None:
        # Non-tuple chain container and non-str records.
        with self.assertRaises(TypeError):
            self.append(entries=((1, list(self.chains[0])),))
        with self.assertRaises(TypeError):
            self.append(entries=((1, (self.chains[0][0], 1)),))
        # Empty chain.
        with self.assertRaises(ValueError):
            self.append(entries=((1, ()),))
        # Broken prev link.
        broken = list(self.chains[0])
        document = json.loads(broken[1])
        document["prev"] = "f" * 64
        broken[1] = reseal(document)
        with self.assertRaises(ValueError):
            self.append(entries=((1, tuple(broken)),))
        # Malformed record.
        with self.assertRaises(ValueError):
            self.append(entries=((1, ("not json",)),))
        # Duplicated transaction identity within the new entries.
        with self.assertRaises(ValueError):
            self.append(
                entries=((1, self.chains[0]), (2, self.chains[0]))
            )

    def test_expected_validation(self) -> None:
        for bad in (1, 1.5, b"x", ["x"], {"x": 1}, object()):
            with self.subTest(bad=type(bad).__name__):
                with self.assertRaises(TypeError):
                    self.append(expected=bad)
        for bad in ("", "xyz", "f" * 63, "F" * 64):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    self.append(expected=bad)

    def test_segment_limit_and_retain_validation(self) -> None:
        for name in ("segment_limit", "retain"):
            for bad in (True, False, 1.5, "1", None, b"1"):
                with self.subTest(name=name, bad=bad):
                    with self.assertRaises(TypeError):
                        self.append(**{name: bad})
            for bad in (0, -1):
                with self.subTest(name=name, bad=bad):
                    with self.assertRaises(ValueError):
                        self.append(**{name: bad})

    def test_timeout_validation(self) -> None:
        for bad in (True, False, "1", b"1", object()):
            with self.subTest(bad=type(bad).__name__):
                with self.assertRaises(TypeError):
                    self.append(timeout=bad)
        for bad in (-1, -0.5, float("nan"), float("inf"), float("-inf")):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    self.append(timeout=bad)
        # None, ints and floats are accepted.
        snapshot = None
        for timeout in (None, 0, 0.5):
            snapshot = self.append(
                entries=((len(self.segment_names()) + 1,
                          self.chains[len(self.segment_names())]),),
                expected=snapshot,
                segment_limit=1,
                timeout=timeout,
            )
        self.assertEqual(len(self.segment_names()), 3)

    def test_lock_timeout_raises_timeouterror(self) -> None:
        lock_path = os.path.realpath(self.dir) + ".lock"
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            self.assertTrue(BranchStore._try_acquire_chain_lock(fd))
            with self.assertRaises(TimeoutError):
                self.append(timeout=0.2)
        finally:
            BranchStore._release_chain_lock(fd)
            os.close(fd)
        # Once released, the append goes through.
        self.append(timeout=1.0)
        self.assertTrue(os.path.exists(self.manifest_path()))


class AppendPublishTests(AppendTestBase):
    def test_first_publish_requires_empty_expected(self) -> None:
        with self.assertRaises(RuntimeError):
            self.append(
                entries=((1, self.chains[0]),), expected="0" * 64
            )
        self.assertFalse(os.path.exists(self.manifest_path()))

    def test_first_publish_and_returned_snapshot(self) -> None:
        snapshot = self.append(
            entries=((1, self.chains[0]), (2, self.chains[1])),
            segment_limit=1,
        )
        self.assertIsInstance(snapshot, str)
        self.assertEqual(len(snapshot), 64)
        manifest = self.read_manifest()
        self.assertEqual(manifest["generation"], 1)
        self.assertEqual(
            [segment["name"] for segment in manifest["segments"]],
            ["segment-000001.json", "segment-000002.json"],
        )
        self.assertIsNone(manifest["rotated"])
        self.assertEqual(
            snapshot, hashlib.sha256(self.read_manifest_raw()).hexdigest()
        )
        result = self.page()
        self.assertEqual(result["snapshot"]["digest"], snapshot)
        self.assertEqual(result["snapshot"]["entries"], 2)
        self.assertEqual(
            result["snapshot"]["size"], len(self.read_manifest_raw())
        )
        self.assertEqual(
            [item["at"] for item in result["items"]], [1, 2]
        )

    def test_empty_entries_seal_an_empty_directory_ledger(self) -> None:
        snapshot = self.append(entries=())
        manifest = self.read_manifest()
        self.assertEqual(manifest["segments"], [])
        self.assertIsNone(manifest["rotated"])
        result = self.page()
        self.assertEqual(result["items"], ())
        self.assertIsNone(result["next_cursor"])
        self.assertEqual(result["snapshot"]["entries"], 0)
        self.assertEqual(result["snapshot"]["digest"], snapshot)

    def test_subsequent_publish_requires_the_current_snapshot(self) -> None:
        snapshot = self.append(entries=((1, self.chains[0]),))
        # Neither an empty expectation nor a stale digest publishes.
        with self.assertRaises(RuntimeError):
            self.append(entries=((2, self.chains[1]),), expected=None)
        with self.assertRaises(RuntimeError):
            self.append(
                entries=((2, self.chains[1]),), expected="0" * 64
            )
        followup = self.append(
            entries=((2, self.chains[1]),), expected=snapshot
        )
        self.assertNotEqual(followup, snapshot)
        self.assertEqual(self.read_manifest()["generation"], 2)
        result = self.page()
        self.assertEqual(result["snapshot"]["digest"], followup)
        self.assertEqual(
            [item["at"] for item in result["items"]], [1, 2]
        )

    def test_same_old_snapshot_allows_at_most_one_winner(self) -> None:
        first = self.append(entries=((1, self.chains[0]),))
        winner = self.append(
            entries=((2, self.chains[1]),), expected=first
        )
        # A second caller reusing the same old snapshot loses, and the
        # winner's data is not overwritten.
        with self.assertRaises(RuntimeError):
            self.append(
                entries=((3, self.chains[2]),), expected=first
            )
        result = self.page()
        self.assertEqual(result["snapshot"]["digest"], winner)
        self.assertEqual(
            [item["at"] for item in result["items"]], [1, 2]
        )

    def test_new_times_must_not_precede_the_ledger_last(self) -> None:
        snapshot = self.append(entries=((5, self.chains[0]),))
        with self.assertRaises(ValueError):
            self.append(
                entries=((4, self.chains[1]),), expected=snapshot
            )
        # An equal time is not earlier than the ledger's last item.
        self.append(entries=((5, self.chains[1]),), expected=snapshot)
        self.assertEqual(
            [item["at"] for item in self.page()["items"]], [5, 5]
        )

    def test_transaction_must_not_duplicate_a_retained_one(self) -> None:
        snapshot = self.append(entries=((1, self.chains[0]),))
        with self.assertRaises(ValueError):
            self.append(
                entries=((2, self.chains[0]),), expected=snapshot
            )

    def test_equivalent_directory_paths_publish_the_same_ledger(self) -> None:
        alias = os.path.join(self.tmp.name, "alias")
        os.symlink(self.dir, alias)
        snapshot = self.append(entries=((1, self.chains[0]),))
        followup = self.append(
            directory=alias,
            entries=((2, self.chains[1]),),
            expected=snapshot,
        )
        self.assertTrue(os.path.exists(self.manifest_path()))
        result = self.page()
        self.assertEqual(result["snapshot"]["digest"], followup)
        self.assertEqual(result["snapshot"]["entries"], 2)

    def test_concurrent_publishers_with_one_snapshot_one_wins(self) -> None:
        snapshot = self.append(entries=((1, self.chains[0]),))
        barrier = threading.Barrier(6)
        outcomes = []
        outcomes_lock = threading.Lock()

        def publish(tag: int) -> None:
            barrier.wait()
            try:
                result = self.append(
                    entries=((tag + 1, self.chains[tag]),),
                    expected=snapshot,
                )
            except RuntimeError:
                result = None
            with outcomes_lock:
                outcomes.append(result)

        threads = [
            threading.Thread(target=publish, args=(tag,))
            for tag in range(1, 6)
        ]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join()
        # Exactly one racer published; everyone else lost the snapshot
        # race, and the winner's entry is the only one added.
        winners = [result for result in outcomes if result is not None]
        self.assertEqual(len(winners), 1)
        result = self.page()
        self.assertEqual(result["snapshot"]["digest"], winners[0])
        self.assertEqual(result["snapshot"]["entries"], 2)


class SegmentLayoutTests(AppendTestBase):
    def test_segments_respect_the_limit_and_stay_immutable(self) -> None:
        snapshot = self.append(
            entries=tuple(
                (index + 1, chain)
                for index, chain in enumerate(self.chains[:5])
            ),
            segment_limit=2,
        )
        manifest = self.read_manifest()
        sizes = [segment["entries"] for segment in manifest["segments"]]
        self.assertEqual(sizes, [2, 2, 1])
        before = {
            name: self.read_segment_raw(name)
            for name in self.segment_names()
        }
        self.append(
            entries=((6, self.chains[5]), (7, self.chains[6])),
            expected=snapshot,
            segment_limit=2,
        )
        # The first publish's segments are untouched byte-for-byte;
        # the new entries landed in new segments only.
        for name, raw in before.items():
            self.assertEqual(raw, self.read_segment_raw(name))
        self.assertEqual(
            self.segment_names(),
            [
                "segment-000001.json",
                "segment-000002.json",
                "segment-000003.json",
                "segment-000004.json",
            ],
        )
        manifest = self.read_manifest()
        self.assertEqual(
            [segment["entries"] for segment in manifest["segments"]],
            [2, 2, 1, 2],
        )

    def test_segments_chain_by_digest_and_match_the_manifest(self) -> None:
        self.append(
            entries=tuple(
                (index + 1, chain)
                for index, chain in enumerate(self.chains[:3])
            ),
            segment_limit=1,
        )
        manifest = self.read_manifest()
        previous = "0" * 64
        for position, descriptor in enumerate(manifest["segments"]):
            raw = self.read_segment_raw(descriptor["name"])
            digest = hashlib.sha256(raw).hexdigest()
            self.assertEqual(descriptor["digest"], digest)
            segment = json.loads(raw.decode("utf-8"))
            self.assertEqual(segment["index"], position + 1)
            self.assertEqual(segment["prev"], previous)
            self.assertEqual(
                descriptor["entries"], len(segment["entries"])
            )
            self.assertEqual(
                descriptor["first"], segment["entries"][0]["at"]
            )
            self.assertEqual(
                descriptor["last"], segment["entries"][-1]["at"]
            )
            previous = digest

    def test_entries_are_stored_in_canonical_order(self) -> None:
        # Equal-time entries given out of transaction order are stored
        # sorted by (time, transaction).
        ordered = sorted(
            range(3),
            key=lambda index: BranchStore.summarize_recovery_diagnostics(
                (self.chains[index],)
            )[0]["transaction"],
        )
        entries = tuple((2, self.chains[index]) for index in ordered)
        shuffled = (entries[1], entries[2], entries[0])
        self.append(entries=shuffled, segment_limit=10)
        segment = json.loads(
            self.read_segment_raw("segment-000001.json").decode("utf-8")
        )
        summaries = BranchStore.summarize_recovery_diagnostics(
            tuple(tuple(entry["chain"]) for entry in segment["entries"])
        )
        keys = [
            (entry["at"], summary["transaction"])
            for entry, summary in zip(segment["entries"], summaries)
        ]
        self.assertEqual(keys, sorted(keys))


class RotationTests(AppendTestBase):
    def test_rotation_removes_oldest_segments_and_keeps_evidence(
        self,
    ) -> None:
        snapshot = None
        segment_bytes = {}
        for index in range(4):
            snapshot = self.append(
                entries=((index + 1, self.chains[index]),),
                expected=snapshot,
                segment_limit=1,
                retain=2,
            )
            for name in self.segment_names():
                segment_bytes[name] = self.read_segment_raw(name)
        # Four one-entry segments were published, two retained.
        self.assertEqual(
            self.segment_names(),
            ["segment-000003.json", "segment-000004.json"],
        )
        manifest = self.read_manifest()
        self.assertEqual(
            [segment["name"] for segment in manifest["segments"]],
            ["segment-000003.json", "segment-000004.json"],
        )
        rotated = manifest["rotated"]
        self.assertEqual(
            rotated["digest"],
            hashlib.sha256(segment_bytes["segment-000002.json"]).hexdigest(),
        )
        self.assertEqual(rotated["entries"], 2)
        self.assertEqual(rotated["first"], 1)
        self.assertEqual(rotated["last"], 2)
        # The first retained segment still links to the removed prefix.
        segment = json.loads(
            self.read_segment_raw("segment-000003.json").decode("utf-8")
        )
        self.assertEqual(segment["prev"], rotated["digest"])
        # Paging sees only the retained entries.
        result = self.page()
        self.assertEqual(
            [item["at"] for item in result["items"]], [3, 4]
        )
        self.assertEqual(result["snapshot"]["entries"], 2)

    def test_rotation_accumulates_the_removed_prefix(self) -> None:
        snapshot = self.append(
            entries=tuple(
                (index + 1, chain)
                for index, chain in enumerate(self.chains[:4])
            ),
            segment_limit=2,
            retain=2,
        )
        # One append of four entries in two segments: nothing rotated.
        self.assertIsNone(self.read_manifest()["rotated"])
        snapshot = self.append(
            entries=((5, self.chains[4]), (6, self.chains[5])),
            expected=snapshot,
            segment_limit=2,
            retain=2,
        )
        manifest = self.read_manifest()
        self.assertEqual(
            [segment["name"] for segment in manifest["segments"]],
            ["segment-000002.json", "segment-000003.json"],
        )
        rotated = manifest["rotated"]
        self.assertEqual(rotated["entries"], 2)
        self.assertEqual((rotated["first"], rotated["last"]), (1, 2))
        snapshot = self.append(
            entries=((7, self.chains[6]), (8, self.chains[7])),
            expected=snapshot,
            segment_limit=2,
            retain=2,
        )
        manifest = self.read_manifest()
        rotated = manifest["rotated"]
        # The evidence now covers both removed segments.
        self.assertEqual(rotated["entries"], 4)
        self.assertEqual((rotated["first"], rotated["last"]), (1, 4))
        self.assertEqual(
            [item["at"] for item in self.page()["items"]], [5, 6, 7, 8]
        )


class DirectoryPageTests(AppendTestBase):
    def setUp(self) -> None:
        super().setUp()
        self.entries = tuple(
            (time, chain)
            for time, chain in zip((1, 2, 2, 3, 4), self.chains)
        )
        self.snapshot = self.append(
            entries=self.entries, segment_limit=2
        )
        self.summaries = BranchStore.summarize_recovery_diagnostics(
            tuple(chain for _at, chain in self.entries)
        )
        self.expect_order = sorted(
            range(5),
            key=lambda index: (
                self.entries[index][0],
                self.summaries[index]["transaction"],
            ),
        )

    def expect_item(self, index: int) -> dict:
        return {
            "at": self.entries[index][0],
            "summary": dict(self.summaries[index]),
            "records": self.entries[index][1],
        }

    def test_result_shape_and_full_page(self) -> None:
        result = self.page()
        self.assertEqual(
            list(result), ["snapshot", "items", "next_cursor"]
        )
        self.assertIsNone(result["next_cursor"])
        self.assertEqual(
            list(result["items"]),
            [self.expect_item(index) for index in self.expect_order],
        )
        for item in result["items"]:
            self.assertEqual(list(item), ["at", "summary", "records"])
            self.assertIsInstance(item["records"], tuple)

    def test_filters_match_the_single_file_ledger(self) -> None:
        result = self.page(start=2, end=3)
        self.assertEqual(
            [item["at"] for item in result["items"]], [2, 2, 3]
        )
        chosen = tuple(
            sorted(
                self.summaries[index]["transaction"]
                for index in (0, 4)
            )
        )
        result = self.page(transactions=chosen)
        self.assertEqual(
            [item["at"] for item in result["items"]], [1, 4]
        )
        empty = self.page(transactions=())
        self.assertEqual(empty["items"], ())
        self.assertEqual(empty["snapshot"]["entries"], 5)

    def test_pagination_walks_every_retained_entry_once(self) -> None:
        seen = []
        cursor = None
        pages = 0
        while True:
            result = self.page(limit=2, cursor=cursor)
            pages += 1
            seen.extend(result["items"])
            cursor = result["next_cursor"]
            if cursor is None:
                break
            self.assertIsInstance(cursor, str)
        self.assertEqual(pages, 3)
        self.assertEqual(
            seen, [self.expect_item(index) for index in self.expect_order]
        )

    def test_cursor_survives_a_restart(self) -> None:
        first = self.page(limit=4)
        stashed = json.loads(first["next_cursor"])
        resumed = self.page(
            limit=4, cursor=BranchStore._canonical_json(stashed)
        )
        self.assertEqual(len(resumed["items"]), 1)
        self.assertIsNone(resumed["next_cursor"])

    def test_append_invalidates_outstanding_cursors(self) -> None:
        cursor = self.page(limit=2)["next_cursor"]
        self.append(
            entries=((5, self.chains[5]),), expected=self.snapshot
        )
        with self.assertRaises(RuntimeError):
            self.page(limit=2, cursor=cursor)

    def test_rotation_invalidates_outstanding_cursors(self) -> None:
        cursor = self.page(limit=2)["next_cursor"]
        snapshot = self.snapshot
        for index in range(5, 9):
            snapshot = self.append(
                entries=((index, self.chains[index]),),
                expected=snapshot,
                segment_limit=1,
                retain=1,
            )
        with self.assertRaises(RuntimeError):
            self.page(limit=2, cursor=cursor)

    def test_truncated_manifest_invalidates_outstanding_cursors(
        self,
    ) -> None:
        cursor = self.page(limit=2)["next_cursor"]
        raw = self.read_manifest_raw()
        with open(self.manifest_path(), "wb") as handle:
            handle.write(raw[:-10])
        with self.assertRaises(RuntimeError):
            self.page(limit=2, cursor=cursor)

    def test_missing_directory_or_manifest_raises_oserror(self) -> None:
        with self.assertRaises(OSError):
            self.page(path=os.path.join(self.tmp.name, "missing"))
        empty = os.path.join(self.tmp.name, "empty")
        os.mkdir(empty)
        with self.assertRaises(OSError):
            self.page(path=empty)

    def test_returned_levels_share_nothing(self) -> None:
        first = self.page()
        first["items"][0]["summary"]["transaction"] = "mutated"
        first["snapshot"]["digest"] = "mutated"
        second = self.page()
        self.assertNotEqual(
            second["items"][0]["summary"]["transaction"], "mutated"
        )
        self.assertNotEqual(second["snapshot"]["digest"], "mutated")


class AppendFailureTests(AppendTestBase):
    def test_failed_publish_keeps_the_old_snapshot_readable(self) -> None:
        snapshot = self.append(
            entries=((1, self.chains[0]), (2, self.chains[1])),
            segment_limit=1,
        )
        before = self.read_manifest_raw()
        segments_before = self.segment_names()
        with mock.patch.object(
            BranchStore, "_replace_durable", side_effect=OSError("boom")
        ):
            with self.assertRaises(OSError):
                self.append(
                    entries=((3, self.chains[2]),), expected=snapshot
                )
        # The old manifest is byte-for-byte intact, no partial segment
        # leaked, and the ledger still pages and appends.
        self.assertEqual(before, self.read_manifest_raw())
        self.assertEqual(segments_before, self.segment_names())
        self.assertEqual(self.page()["snapshot"]["digest"], snapshot)
        followup = self.append(
            entries=((3, self.chains[2]),), expected=snapshot
        )
        self.assertEqual(self.page()["snapshot"]["digest"], followup)

    def test_failed_manifest_switch_keeps_the_old_snapshot(self) -> None:
        snapshot = self.append(entries=((1, self.chains[0]),))
        before = self.read_manifest_raw()
        real_replace = os.replace
        manifest_path = self.manifest_path()

        def failing_replace(source, target):
            if os.path.realpath(target) == os.path.realpath(
                manifest_path
            ):
                raise OSError("boom")
            return real_replace(source, target)

        with mock.patch.object(os, "replace", failing_replace):
            with self.assertRaises(OSError):
                self.append(
                    entries=((2, self.chains[1]),), expected=snapshot
                )
        self.assertEqual(before, self.read_manifest_raw())
        self.assertEqual(self.page()["snapshot"]["digest"], snapshot)
        # The failed publish's segment was cleaned up, so the retry
        # reuses its sequence index.
        self.append(entries=((2, self.chains[1]),), expected=snapshot)
        self.assertEqual(
            self.segment_names(),
            ["segment-000001.json", "segment-000002.json"],
        )

    def test_failed_validation_changes_nothing(self) -> None:
        snapshot = self.append(entries=((1, self.chains[0]),))
        before = self.read_manifest_raw()
        with self.assertRaises(ValueError):
            self.append(entries=((2, ()),), expected=snapshot)
        with self.assertRaises(RuntimeError):
            self.append(entries=((2, self.chains[1]),), expected=None)
        self.assertEqual(before, self.read_manifest_raw())
        self.assertEqual(self.segment_names(), ["segment-000001.json"])


class CorruptDirectoryTests(AppendTestBase):
    def setUp(self) -> None:
        super().setUp()
        self.snapshot = self.append(
            entries=tuple(
                (index + 1, chain)
                for index, chain in enumerate(self.chains[:4])
            ),
            segment_limit=2,
        )

    def assert_broken(self) -> None:
        with self.assertRaises(ValueError):
            self.page()
        # Appending re-authenticates the on-disk state under the lock
        # and fails the same way instead of building on corruption.
        with self.assertRaises(ValueError):
            self.append(
                entries=((5, self.chains[4]),), expected=self.snapshot
            )

    def tamper_manifest(self, **overrides) -> None:
        manifest = self.read_manifest()
        manifest.update(overrides)
        with open(self.manifest_path(), "w", encoding="utf-8") as handle:
            handle.write(reseal(manifest))

    def test_manifest_checksum(self) -> None:
        manifest = self.read_manifest()
        manifest["checksum"] = "f" * 64
        with open(self.manifest_path(), "w", encoding="utf-8") as handle:
            handle.write(BranchStore._canonical_json(manifest))
        self.assert_broken()

    def test_manifest_format_and_version(self) -> None:
        self.tamper_manifest(format="branching-city-twin/other")
        self.assert_broken()
        self.tamper_manifest(version=2)
        self.assert_broken()

    def test_manifest_descriptor_tampering(self) -> None:
        manifest = self.read_manifest()
        manifest["segments"][0]["entries"] += 1
        self.tamper_manifest(segments=manifest["segments"])
        self.assert_broken()

    def test_segment_bytes_do_not_match_the_manifest(self) -> None:
        raw = bytearray(self.read_segment_raw("segment-000001.json"))
        raw[-1] = ord("}") if raw[-1] != ord("}") else ord("{")
        with open(
            os.path.join(self.dir, "segment-000001.json"), "wb"
        ) as handle:
            handle.write(bytes(raw))
        self.assert_broken()

    def test_segment_checksum(self) -> None:
        segment = json.loads(
            self.read_segment_raw("segment-000001.json").decode("utf-8")
        )
        segment["checksum"] = "f" * 64
        raw = BranchStore._canonical_json(segment).encode("utf-8")
        with open(
            os.path.join(self.dir, "segment-000001.json"), "wb"
        ) as handle:
            handle.write(raw)
        # Re-point the manifest at the tampered bytes so only the
        # segment's own checksum defect remains.
        manifest = self.read_manifest()
        manifest["segments"][0]["digest"] = hashlib.sha256(
            raw
        ).hexdigest()
        self.tamper_manifest(segments=manifest["segments"])
        self.assert_broken()

    def test_broken_digest_chain(self) -> None:
        # Re-link the second segment at a foreign predecessor, fixing
        # its checksum and manifest digest so only the chain break
        # remains.
        segment = json.loads(
            self.read_segment_raw("segment-000002.json").decode("utf-8")
        )
        segment["prev"] = "f" * 64
        raw = reseal(segment).encode("utf-8")
        with open(
            os.path.join(self.dir, "segment-000002.json"), "wb"
        ) as handle:
            handle.write(raw)
        manifest = self.read_manifest()
        manifest["segments"][1]["digest"] = hashlib.sha256(
            raw
        ).hexdigest()
        self.tamper_manifest(segments=manifest["segments"])
        self.assert_broken()

    def test_segment_embedded_chain_authentication(self) -> None:
        segment = json.loads(
            self.read_segment_raw("segment-000001.json").decode("utf-8")
        )
        record = json.loads(segment["entries"][0]["chain"][1])
        record["prev"] = "f" * 64
        segment["entries"][0]["chain"][1] = reseal(record)
        raw = reseal(segment).encode("utf-8")
        with open(
            os.path.join(self.dir, "segment-000001.json"), "wb"
        ) as handle:
            handle.write(raw)
        manifest = self.read_manifest()
        manifest["segments"][0]["digest"] = hashlib.sha256(
            raw
        ).hexdigest()
        self.tamper_manifest(segments=manifest["segments"])
        self.assert_broken()

    def test_missing_segment_raises_oserror(self) -> None:
        os.remove(os.path.join(self.dir, "segment-000001.json"))
        with self.assertRaises(OSError):
            self.page()


class SingleFileLedgerUnchangedTests(AppendTestBase):
    def test_single_file_ledger_still_pages_as_before(self) -> None:
        path = os.path.join(self.tmp.name, "ledger.json")
        entries = tuple(
            (index + 1, chain)
            for index, chain in enumerate(self.chains[:3])
        )
        BranchStore.save_diagnostic_ledger(path, entries)
        result = BranchStore.page_diagnostic_ledger(
            path, None, None, None, 10, None
        )
        self.assertEqual(result["snapshot"]["entries"], 3)
        self.assertEqual(
            [item["at"] for item in result["items"]], [1, 2, 3]
        )
        # A directory and a file ledger keep independent snapshots.
        self.append(entries=((1, self.chains[3]),))
        again = BranchStore.page_diagnostic_ledger(
            path, None, None, None, 10, None
        )
        self.assertEqual(again["snapshot"], result["snapshot"])


if __name__ == "__main__":
    unittest.main()
