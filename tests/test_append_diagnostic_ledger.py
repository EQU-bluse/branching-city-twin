"""Tests for the segmented, incrementally published diagnostic ledger.

``BranchStore.append_diagnostic_ledger`` appends time-stamped recovery
diagnostic chains to a directory ledger: entries are sealed into
immutable digest-chained segments bounded by a per-segment limit, the
oldest segments are rotated away beyond a retention count (with the
manifest keeping the removed prefix's continuity proof), and every
publish is an optimistic-concurrency, cross-process-locked, durable
atomic manifest switch. ``BranchStore.page_diagnostic_ledger`` pages a
directory ledger's currently retained segments with the same filters,
ordering, cursors and return structure as a single-file ledger.

These tests cover:

* argument validation: directory, entries, expected snapshot, segment
  limit, retention count and lock timeout;
* first-publish and optimistic-concurrency rules: only an empty
  expected snapshot publishes first, afterwards the expected snapshot
  must equal the current one, and concurrent publishers sharing one
  old snapshot see exactly one winner;
* segmenting: canonical (time, transaction) order, per-segment limit,
  digest chaining and manifest metadata;
* rotation: oldest segments removed, continuity proof (last digest,
  entry count, time bounds) retained in the manifest;
* directory paging: filters, ordering, pagination, cursors bound to
  the manifest snapshot, ``RuntimeError`` after append, rotation or
  truncation, and snapshot consistency under concurrent publishes;
* corrupt manifests, segments and digest chains raising
  ``ValueError``, missing directories and I/O failures raising
  ``OSError``, and lock waits raising ``TimeoutError``;
* durability: a failed publish leaves the old snapshot byte-for-byte
  readable.
"""

import hashlib
import json
import os
import tempfile
import threading
import unittest
from unittest import mock

from city_twin.branches import BranchStore

try:
    from test_diagnostic_ledger import make_chain, reseal, transaction
except ImportError:  # running as a package member from the repo root
    from tests.test_diagnostic_ledger import (
        make_chain,
        reseal,
        transaction,
    )


def read_bytes(path: str) -> bytes:
    with open(path, "rb") as handle:
        return handle.read()


class AppendTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = os.path.join(self.tmp.name, "ledger")
        os.mkdir(self.dir)
        self.chains = [make_chain(transaction(i)) for i in range(8)]
        self.entries = tuple(
            (time, chain)
            for time, chain in zip((1, 2, 3, 4, 5, 6, 7, 8), self.chains)
        )

    def append(self, **overrides):
        arguments = {
            "directory": self.dir,
            "entries": self.entries[:2],
            "expected": None,
            "segment_limit": 2,
            "keep": 10,
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

    def manifest(self) -> dict:
        return json.loads(read_bytes(self.manifest_path()).decode("utf-8"))

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
        self.assertFalse(os.path.exists(missing))

    def test_directory_that_is_a_file_raises_oserror(self) -> None:
        file_path = os.path.join(self.tmp.name, "file")
        with open(file_path, "w", encoding="utf-8") as handle:
            handle.write("x")
        with self.assertRaises(OSError):
            self.append(directory=file_path)

    def test_entries_validation(self) -> None:
        for bad in ([], "x", 1, None, {"a": 1}):
            with self.subTest(bad=type(bad).__name__):
                with self.assertRaises(TypeError):
                    self.append(entries=bad)
        chain = self.chains[0]
        for bad_entry in ((1,), (1, chain, chain), [1, chain], "x1", 1):
            with self.subTest(bad=bad_entry):
                with self.assertRaises(TypeError):
                    self.append(entries=(bad_entry,))
        for bad_time in (True, False, 1.5, "1", None, b"1"):
            with self.subTest(bad=bad_time):
                with self.assertRaises(TypeError):
                    self.append(entries=((bad_time, chain),))
        with self.assertRaises(ValueError):
            self.append(entries=((-1, chain),))
        with self.assertRaises(ValueError):
            self.append(entries=((2, self.chains[0]), (1, self.chains[1])))
        with self.assertRaises(TypeError):
            self.append(entries=((1, list(chain)),))
        with self.assertRaises(ValueError):
            self.append(entries=((1, ()),))
        with self.assertRaises(ValueError):
            self.append(entries=((1, ("not json",)),))
        with self.assertRaises(ValueError):
            self.append(
                entries=((1, self.chains[0]), (2, self.chains[0]))
            )

    def test_expected_validation(self) -> None:
        for bad in (1, 1.5, b"x", ["x"], {"x": 1}, object(), True):
            with self.subTest(bad=type(bad).__name__):
                with self.assertRaises(TypeError):
                    self.append(expected=bad)
        for bad in ("", "0" * 63, "0" * 65, "z" * 64, "0" * 63 + "G"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    self.append(expected=bad)

    def test_segment_limit_and_keep_validation(self) -> None:
        for name in ("segment_limit", "keep"):
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
        self.append(timeout=None)
        self.append(entries=self.entries[2:4], expected=self.current())
        self.append(
            entries=self.entries[4:6],
            expected=self.current(),
            timeout=1.5,
        )

    def current(self) -> str:
        return hashlib.sha256(
            read_bytes(self.manifest_path())
        ).hexdigest()


class AppendPublishTests(AppendTestBase):
    def test_first_publish_requires_empty_expected(self) -> None:
        with self.assertRaises(RuntimeError):
            self.append(expected="0" * 64)
        self.assertFalse(os.path.exists(self.manifest_path()))
        self.assertEqual(self.segment_names(), [])

    def test_first_publish_creates_manifest_and_segments(self) -> None:
        digest = self.append()
        self.assertEqual(
            digest,
            hashlib.sha256(read_bytes(self.manifest_path())).hexdigest(),
        )
        manifest = self.manifest()
        self.assertEqual(
            manifest["format"],
            "branching-city-twin/diagnostic-ledger-manifest",
        )
        self.assertEqual(manifest["version"], 1)
        self.assertIsNone(manifest["dropped"])
        self.assertEqual(self.segment_names(), ["segment-000001.json"])
        self.assertEqual(len(manifest["segments"]), 1)
        meta = manifest["segments"][0]
        self.assertEqual(meta["name"], "segment-000001.json")
        self.assertEqual(meta["entries"], 2)
        self.assertEqual((meta["first_at"], meta["last_at"]), (1, 2))
        segment = json.loads(
            read_bytes(os.path.join(self.dir, meta["name"])).decode(
                "utf-8"
            )
        )
        self.assertEqual(segment["index"], 1)
        self.assertEqual(segment["prev"], "0" * 64)
        self.assertEqual(
            meta["digest"],
            hashlib.sha256(
                read_bytes(os.path.join(self.dir, meta["name"]))
            ).hexdigest(),
        )

    def test_empty_entries_publish_an_empty_ledger(self) -> None:
        digest = self.append(entries=())
        manifest = self.manifest()
        self.assertEqual(manifest["segments"], [])
        result = self.page()
        self.assertEqual(result["items"], ())
        self.assertEqual(result["snapshot"]["entries"], 0)
        self.assertEqual(result["snapshot"]["digest"], digest)

    def test_entries_are_chunked_by_segment_limit(self) -> None:
        self.append(entries=self.entries[:5], segment_limit=2)
        self.assertEqual(
            self.segment_names(),
            [
                "segment-000001.json",
                "segment-000002.json",
                "segment-000003.json",
            ],
        )
        manifest = self.manifest()
        counts = [meta["entries"] for meta in manifest["segments"]]
        self.assertEqual(counts, [2, 2, 1])
        # The segments form a digest chain.
        previous = "0" * 64
        for meta in manifest["segments"]:
            segment = json.loads(
                read_bytes(os.path.join(self.dir, meta["name"])).decode(
                    "utf-8"
                )
            )
            self.assertEqual(segment["prev"], previous)
            previous = meta["digest"]

    def test_entries_are_stored_in_canonical_order(self) -> None:
        # Two entries share time 2: their input order does not matter.
        entries = (
            (1, self.chains[0]),
            (2, self.chains[2]),
            (2, self.chains[1]),
            (3, self.chains[3]),
        )
        self.append(entries=entries)
        result = self.page()
        keys = [
            (item["at"], item["summary"]["transaction"])
            for item in result["items"]
        ]
        self.assertEqual(keys, sorted(keys))
        self.assertEqual(
            [item["at"] for item in result["items"]], [1, 2, 2, 3]
        )

    def test_second_publish_requires_the_current_snapshot(self) -> None:
        first = self.append()
        # Neither an empty expected nor a stale one publishes again.
        with self.assertRaises(RuntimeError):
            self.append(entries=self.entries[2:4], expected=None)
        with self.assertRaises(RuntimeError):
            self.append(entries=self.entries[2:4], expected="0" * 64)
        second = self.append(entries=self.entries[2:4], expected=first)
        self.assertNotEqual(first, second)
        self.assertEqual(second, self.current_digest())
        result = self.page()
        self.assertEqual(
            [item["at"] for item in result["items"]], [1, 2, 3, 4]
        )
        self.assertEqual(result["snapshot"]["digest"], second)

    def current_digest(self) -> str:
        return hashlib.sha256(
            read_bytes(self.manifest_path())
        ).hexdigest()

    def test_appended_time_must_not_regress_the_ledger_tail(self) -> None:
        first = self.append(entries=self.entries[:2])  # times 1, 2
        with self.assertRaises(ValueError):
            self.append(entries=((1, self.chains[2]),), expected=first)
        # Equal to the tail time is accepted.
        self.append(entries=((2, self.chains[2]),), expected=first)
        self.assertEqual(
            [item["at"] for item in self.page()["items"]], [1, 2, 2]
        )

    def test_appended_transaction_must_not_duplicate_the_ledger(self) -> None:
        first = self.append(entries=self.entries[:2])
        with self.assertRaises(ValueError):
            self.append(entries=((3, self.chains[0]),), expected=first)
        # The failed append changed nothing.
        self.assertEqual(first, self.current_digest())
        self.assertEqual(len(self.page()["items"]), 2)

    def test_equivalent_directory_spellings_share_one_ledger(self) -> None:
        first = self.append()
        spelling = os.path.join(self.dir, ".", "")
        with self.assertRaises(RuntimeError):
            self.append(
                directory=spelling,
                entries=self.entries[2:4],
                expected="0" * 64,
            )
        second = self.append(
            directory=spelling,
            entries=self.entries[2:4],
            expected=first,
        )
        self.assertEqual(second, self.current_digest())
        self.assertEqual(len(self.page()["items"]), 4)

    def test_concurrent_publishers_admit_exactly_one_winner(self) -> None:
        first = self.append()
        outcomes: list[tuple[str, object]] = []
        barrier = threading.Barrier(2)

        def publish(entries, tag):
            barrier.wait()
            try:
                digest = BranchStore.append_diagnostic_ledger(
                    self.dir, entries, first, 2, 10, None
                )
                outcomes.append((tag, digest))
            except RuntimeError as exc:
                outcomes.append((tag, exc))

        left = threading.Thread(
            target=publish, args=(self.entries[2:4], "left")
        )
        right = threading.Thread(
            target=publish, args=(self.entries[4:6], "right")
        )
        left.start()
        right.start()
        left.join()
        right.join()
        winners = [
            outcome for outcome in outcomes if isinstance(outcome[1], str)
        ]
        losers = [
            outcome
            for outcome in outcomes
            if isinstance(outcome[1], RuntimeError)
        ]
        self.assertEqual(len(winners), 1)
        self.assertEqual(len(losers), 1)
        # The winning publish is intact and the loser's data never
        # landed.
        self.assertEqual(winners[0][1], self.current_digest())
        times = [item["at"] for item in self.page()["items"]]
        if winners[0][0] == "left":
            self.assertEqual(times, [1, 2, 3, 4])
        else:
            self.assertEqual(times, [1, 2, 5, 6])

    def test_failed_publish_leaves_the_old_snapshot_readable(self) -> None:
        first = self.append()
        before = read_bytes(self.manifest_path())
        with mock.patch.object(
            BranchStore,
            "_write_diagnostic_ledger_file",
            side_effect=OSError("boom"),
        ):
            with self.assertRaises(OSError):
                self.append(entries=self.entries[2:4], expected=first)
        self.assertEqual(before, read_bytes(self.manifest_path()))
        result = self.page()
        self.assertEqual(result["snapshot"]["digest"], first)
        self.assertEqual([item["at"] for item in result["items"]], [1, 2])
        # No temporary residue was left behind.
        self.assertEqual(
            sorted(os.listdir(self.dir)),
            [".diagnostic-ledger.lock", "manifest.json", "segment-000001.json"],
        )


class AppendRotationTests(AppendTestBase):
    def publish(self, entries, keep=2):
        expected = getattr(self, "_expected", None)
        digest = BranchStore.append_diagnostic_ledger(
            self.dir, entries, expected, 2, keep, None
        )
        self._expected = digest
        return digest

    def test_rotation_removes_oldest_segments_and_keeps_proof(self) -> None:
        self.publish(self.entries[0:2])  # segment 1, times 1-2
        self.publish(self.entries[2:4])  # segment 2, times 3-4
        self.assertIsNone(self.manifest()["dropped"])
        self.publish(self.entries[4:6])  # segment 3, times 5-6
        self.assertEqual(
            self.segment_names(),
            ["segment-000002.json", "segment-000003.json"],
        )
        manifest = self.manifest()
        dropped = manifest["dropped"]
        self.assertEqual(dropped["entries"], 2)
        self.assertEqual((dropped["first_at"], dropped["last_at"]), (1, 2))
        # The continuity proof links the retained chain to the removed
        # prefix: the first retained segment cites the dropped digest.
        first_retained = json.loads(
            read_bytes(
                os.path.join(self.dir, manifest["segments"][0]["name"])
            ).decode("utf-8")
        )
        self.assertEqual(first_retained["prev"], dropped["digest"])
        # Only the retained segments are paged.
        result = self.page()
        self.assertEqual(
            [item["at"] for item in result["items"]], [3, 4, 5, 6]
        )
        self.assertEqual(result["snapshot"]["entries"], 4)

    def test_rotation_accumulates_the_dropped_prefix(self) -> None:
        self.publish(self.entries[0:2])
        self.publish(self.entries[2:4])
        self.publish(self.entries[4:6])
        self.publish(self.entries[6:8])  # drops segment 2 as well
        manifest = self.manifest()
        dropped = manifest["dropped"]
        self.assertEqual(dropped["entries"], 4)
        self.assertEqual((dropped["first_at"], dropped["last_at"]), (1, 4))
        self.assertEqual(
            [meta["name"] for meta in manifest["segments"]],
            ["segment-000003.json", "segment-000004.json"],
        )
        first_retained = json.loads(
            read_bytes(
                os.path.join(self.dir, "segment-000003.json")
            ).decode("utf-8")
        )
        self.assertEqual(first_retained["prev"], dropped["digest"])
        self.assertEqual(
            [item["at"] for item in self.page()["items"]], [5, 6, 7, 8]
        )

    def test_keep_one_retains_only_the_newest_segment(self) -> None:
        self.publish(self.entries[0:2], keep=1)
        self.publish(self.entries[2:4], keep=1)
        self.assertEqual(self.segment_names(), ["segment-000002.json"])
        manifest = self.manifest()
        self.assertEqual(manifest["dropped"]["entries"], 2)
        self.assertEqual(
            [item["at"] for item in self.page()["items"]], [3, 4]
        )


class DirectoryPageTests(AppendTestBase):
    def setUp(self) -> None:
        super().setUp()
        self.digest = BranchStore.append_diagnostic_ledger(
            self.dir, self.entries[:5], None, 2, 10, None
        )
        self.summaries = BranchStore.summarize_recovery_diagnostics(
            tuple(self.chains)
        )

    def test_page_shape_and_snapshot(self) -> None:
        result = self.page()
        self.assertEqual(
            list(result), ["snapshot", "items", "next_cursor"]
        )
        raw = read_bytes(self.manifest_path())
        self.assertEqual(
            result["snapshot"]["digest"], hashlib.sha256(raw).hexdigest()
        )
        self.assertEqual(result["snapshot"]["entries"], 5)
        self.assertEqual(result["snapshot"]["bytes"], len(raw))
        self.assertIsNone(result["next_cursor"])
        self.assertEqual(
            [item["at"] for item in result["items"]], [1, 2, 3, 4, 5]
        )
        for item in result["items"]:
            self.assertEqual(list(item), ["at", "summary", "records"])
            self.assertIsInstance(item["records"], tuple)

    def test_filters_bounds_and_pagination(self) -> None:
        result = self.page(start=2, end=4)
        self.assertEqual(
            [item["at"] for item in result["items"]], [2, 3, 4]
        )
        chosen = tuple(
            sorted(self.summaries[index]["transaction"] for index in (0, 4))
        )
        result = self.page(transactions=chosen)
        self.assertEqual(
            [item["at"] for item in result["items"]], [1, 5]
        )
        empty = self.page(transactions=())
        self.assertEqual(empty["items"], ())
        self.assertEqual(empty["snapshot"]["entries"], 5)
        # Walk every entry once through cursors.
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
        self.assertEqual(pages, 3)
        self.assertEqual([item["at"] for item in seen], [1, 2, 3, 4, 5])

    def test_page_sorts_across_segment_boundaries(self) -> None:
        # A later append at the tail time with an earlier-sorting
        # transaction still pages in canonical (time, transaction)
        # order.
        low, high = sorted(
            (transaction(100), transaction(101))
        )
        chains = {low: make_chain(low), high: make_chain(high)}
        digest = BranchStore.append_diagnostic_ledger(
            self.dir, ((5, chains[high]),), self.digest, 2, 10, None
        )
        BranchStore.append_diagnostic_ledger(
            self.dir, ((5, chains[low]),), digest, 2, 10, None
        )
        result = self.page(transactions=(low, high))
        keys = [
            (item["at"], item["summary"]["transaction"])
            for item in result["items"]
        ]
        self.assertEqual(keys, [(5, low), (5, high)])

    def test_append_between_pages_invalidates_the_cursor(self) -> None:
        cursor = self.page(limit=2)["next_cursor"]
        BranchStore.append_diagnostic_ledger(
            self.dir, self.entries[5:7], self.digest, 2, 10, None
        )
        with self.assertRaises(RuntimeError):
            self.page(limit=2, cursor=cursor)

    def test_rotation_between_pages_invalidates_the_cursor(self) -> None:
        cursor = self.page(limit=2)["next_cursor"]
        digest = self.digest
        for chunk in (self.entries[5:7], self.entries[7:8]):
            digest = BranchStore.append_diagnostic_ledger(
                self.dir, chunk, digest, 2, 2, None
            )
        with self.assertRaises(RuntimeError):
            self.page(limit=2, cursor=cursor)

    def test_truncated_manifest_invalidates_the_cursor(self) -> None:
        cursor = self.page(limit=2)["next_cursor"]
        raw = read_bytes(self.manifest_path())
        with open(self.manifest_path(), "wb") as handle:
            handle.write(raw[:-10])
        with self.assertRaises(RuntimeError):
            self.page(limit=2, cursor=cursor)

    def test_cursor_still_binds_the_filters(self) -> None:
        cursor = self.page(limit=2)["next_cursor"]
        with self.assertRaises(ValueError):
            self.page(limit=2, cursor=cursor, start=2)
        with self.assertRaises(ValueError):
            self.page(limit=2, cursor=cursor, transactions=())
        resumed = self.page(limit=3, cursor=cursor)
        self.assertEqual(len(resumed["items"]), 3)

    def test_directory_without_manifest_raises_oserror(self) -> None:
        empty = os.path.join(self.tmp.name, "empty")
        os.mkdir(empty)
        with self.assertRaises(OSError):
            self.page(path=empty)

    def test_paging_is_read_only(self) -> None:
        before = sorted(os.listdir(self.dir))
        snapshot = read_bytes(self.manifest_path())
        cursor = None
        while True:
            result = self.page(limit=1, cursor=cursor)
            cursor = result["next_cursor"]
            if cursor is None:
                break
        self.assertEqual(sorted(os.listdir(self.dir)), before)
        self.assertEqual(snapshot, read_bytes(self.manifest_path()))

    def test_returned_levels_share_nothing(self) -> None:
        first = self.page()
        first["items"][0]["summary"]["transaction"] = "mutated"
        first["snapshot"]["digest"] = "mutated"
        second = self.page()
        self.assertNotEqual(
            second["items"][0]["summary"]["transaction"], "mutated"
        )
        self.assertNotEqual(second["snapshot"]["digest"], "mutated")


class DirectoryCorruptionTests(AppendTestBase):
    def setUp(self) -> None:
        super().setUp()
        BranchStore.append_diagnostic_ledger(
            self.dir, self.entries[:4], None, 2, 10, None
        )

    def write_manifest(self, document: dict) -> None:
        with open(self.manifest_path(), "w", encoding="utf-8") as handle:
            handle.write(reseal(document))

    def tamper_manifest(self, **overrides) -> None:
        document = self.manifest()
        document.update(overrides)
        self.write_manifest(document)

    def test_manifest_envelope_defects(self) -> None:
        raw = read_bytes(self.manifest_path())
        with open(self.manifest_path(), "wb") as handle:
            handle.write(b"\xef\xbb\xbf" + raw)
        with self.assertRaises(ValueError):
            self.page()
        with open(self.manifest_path(), "wb") as handle:
            handle.write(raw + b"\n")
        with self.assertRaises(ValueError):
            self.page()
        with open(self.manifest_path(), "wb") as handle:
            handle.write(b"not json")
        with self.assertRaises(ValueError):
            self.page()

    def test_manifest_checksum_format_and_version(self) -> None:
        document = self.manifest()
        document["checksum"] = "f" * 64
        with open(self.manifest_path(), "w", encoding="utf-8") as handle:
            handle.write(BranchStore._canonical_json(document))
        with self.assertRaises(ValueError):
            self.page()
        self.tamper_manifest(format="branching-city-twin/other")
        with self.assertRaises(ValueError):
            self.page()
        self.tamper_manifest(version=2)
        with self.assertRaises(ValueError):
            self.page()

    def test_manifest_segment_metadata_mismatch(self) -> None:
        manifest = self.manifest()
        meta = dict(manifest["segments"][0])
        meta["digest"] = "f" * 64
        segments = [meta, *manifest["segments"][1:]]
        self.tamper_manifest(segments=segments)
        with self.assertRaises(ValueError):
            self.page()

    def test_segment_checksum_defect(self) -> None:
        path = os.path.join(self.dir, "segment-000001.json")
        document = json.loads(read_bytes(path).decode("utf-8"))
        document["checksum"] = "f" * 64
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(BranchStore._canonical_json(document))
        with self.assertRaises(ValueError):
            self.page()

    def test_segment_digest_chain_break(self) -> None:
        # Rewriting a segment's predecessor link (resealed, so only the
        # chain defect remains) is detected against the manifest.
        path = os.path.join(self.dir, "segment-000002.json")
        document = json.loads(read_bytes(path).decode("utf-8"))
        document["prev"] = "f" * 64
        body = {
            key: value
            for key, value in document.items()
            if key != "checksum"
        }
        document["checksum"] = hashlib.sha256(
            BranchStore._canonical_json(body).encode("utf-8")
        ).hexdigest()
        data = BranchStore._canonical_json(document).encode("utf-8")
        manifest = self.manifest()
        manifest["segments"][1]["digest"] = hashlib.sha256(
            data
        ).hexdigest()
        self.tamper_manifest(segments=manifest["segments"])
        with open(path, "wb") as handle:
            handle.write(data)
        with self.assertRaises(ValueError):
            self.page()

    def test_segment_chain_authentication_defect(self) -> None:
        path = os.path.join(self.dir, "segment-000001.json")
        document = json.loads(read_bytes(path).decode("utf-8"))
        chain = list(document["entries"][0]["chain"])
        record = json.loads(chain[1])
        record["prev"] = "f" * 64
        chain[1] = reseal(record)
        document["entries"][0]["chain"] = chain
        body = {
            key: value
            for key, value in document.items()
            if key != "checksum"
        }
        document["checksum"] = hashlib.sha256(
            BranchStore._canonical_json(body).encode("utf-8")
        ).hexdigest()
        data = BranchStore._canonical_json(document).encode("utf-8")
        manifest = self.manifest()
        manifest["segments"][0]["digest"] = hashlib.sha256(
            data
        ).hexdigest()
        self.tamper_manifest(segments=manifest["segments"])
        with open(path, "wb") as handle:
            handle.write(data)
        with self.assertRaises(ValueError):
            self.page()

    def test_missing_segment_raises_oserror(self) -> None:
        os.remove(os.path.join(self.dir, "segment-000002.json"))
        with self.assertRaises(OSError):
            self.page()

    def test_corrupt_manifest_blocks_append_with_value_error(self) -> None:
        self.tamper_manifest(version=2)
        with self.assertRaises(ValueError):
            BranchStore.append_diagnostic_ledger(
                self.dir, self.entries[4:6], "0" * 64, 2, 10, None
            )


class AppendLockTests(AppendTestBase):
    def test_lock_wait_times_out(self) -> None:
        lock_path = os.path.join(
            os.path.realpath(self.dir),
            BranchStore._DIAGNOSTIC_LEDGER_LOCK_NAME,
        )
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            self.assertTrue(BranchStore._try_acquire_chain_lock(fd))
            with self.assertRaises(TimeoutError):
                self.append(timeout=0.2)
        finally:
            BranchStore._release_chain_lock(fd)
            os.close(fd)
        # The timed-out call published nothing.
        self.assertFalse(os.path.exists(self.manifest_path()))
        self.assertEqual(self.append(), self.page()["snapshot"]["digest"])


if __name__ == "__main__":
    unittest.main()
