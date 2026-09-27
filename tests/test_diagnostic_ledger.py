"""Tests for the persistent diagnostic ledger.

``BranchStore.save_diagnostic_ledger`` seals a snapshot of
time-stamped recovery diagnostic chains into a compact, checksummed,
atomically replaced JSON ledger; ``BranchStore.page_diagnostic_ledger``
reads it back one snapshot-consistent page at a time through
resumption cursors that survive restarts.

These tests cover:

* ledger envelope shape: compact canonical UTF-8, no BOM, no trailing
  newline, SHA-256 checksum, entries stored in (time, transaction)
  order;
* argument validation for both entry points -- path, entries shape,
  entry times, chain authentication, time bounds, transaction filter,
  limit and cursor types;
* durable atomic replacement: failures raise ``OSError`` and leave a
  previous ledger byte-for-byte untouched;
* pagination: inclusive time bounds, transaction filtering (unknown
  identities never error, an empty tuple matches nothing), stable
  (time, transaction) ordering, multi-page iteration and the terminal
  ``None`` cursor;
* cursor semantics: cross-restart resume, filter binding, and
  ``RuntimeError`` when the ledger is replaced, truncated or appended
  to between pages;
* corrupt ledgers (encoding, checksum, structure or embedded chain
  authentication) raising ``ValueError`` with no partial page;
* strict read-only isolation of queries and fresh, unshared return
  levels.
"""

import hashlib
import json
import os
import tempfile
import unittest
from unittest import mock

from city_twin.branches import BranchStore


def reseal(document: dict) -> str:
    """Re-serialize a (possibly tampered) diagnostic document with a
    fresh checksum over every other field."""
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
    """Build one authentic diagnostic chain for ``transaction``
    following the given disposition sequence (a pending observation
    first, terminal observations carried forward)."""
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


class LedgerTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "ledger.json")
        self.chains = [make_chain(transaction(i)) for i in range(5)]
        self.entries = tuple(
            (time, chain)
            for time, chain in zip((1, 2, 2, 3, 4), self.chains)
        )

    def save(self, entries=None) -> None:
        BranchStore.save_diagnostic_ledger(
            self.path, self.entries if entries is None else entries
        )

    def page(self, **overrides):
        arguments = {
            "path": self.path,
            "start": None,
            "end": None,
            "transactions": None,
            "limit": 10,
            "cursor": None,
        }
        arguments.update(overrides)
        return BranchStore.page_diagnostic_ledger(**arguments)

    def read_raw(self) -> bytes:
        with open(self.path, "rb") as handle:
            return handle.read()


class SaveEnvelopeTests(LedgerTestBase):
    def test_envelope_is_compact_checksummed_and_sorted(self) -> None:
        self.save()
        raw = self.read_raw()
        text = raw.decode("utf-8")
        self.assertFalse(text.startswith("\ufeff"))
        self.assertFalse(text.endswith("\n"))
        self.assertNotIn("\n", text)
        self.assertNotIn(" ", text)
        document = json.loads(text)
        self.assertEqual(
            set(document), {"format", "version", "entries", "checksum"}
        )
        self.assertEqual(
            document["format"], "branching-city-twin/diagnostic-ledger"
        )
        self.assertEqual(document["version"], 1)
        body = {
            key: value
            for key, value in document.items()
            if key != "checksum"
        }
        self.assertEqual(
            document["checksum"],
            hashlib.sha256(
                BranchStore._canonical_json(body).encode("utf-8")
            ).hexdigest(),
        )
        # Canonical bytes reproduce exactly.
        self.assertEqual(
            text, BranchStore._canonical_json(document)
        )
        # Entries are stored in (time, transaction) order.
        summaries = BranchStore.summarize_recovery_diagnostics(
            tuple(self.chains)
        )
        expect = sorted(
            zip((1, 2, 2, 3, 4), summaries),
            key=lambda pair: (pair[0], pair[1]["transaction"]),
        )
        stored = document["entries"]
        self.assertEqual(
            [entry["at"] for entry in stored],
            [time for time, _summary in expect],
        )
        for entry, (time, summary) in zip(stored, expect):
            self.assertEqual(entry["at"], time)
            self.assertEqual(
                BranchStore.summarize_recovery_diagnostics(
                    (tuple(entry["chain"]),)
                )[0]["transaction"],
                summary["transaction"],
            )

    def test_save_is_deterministic_regardless_of_equal_time_order(
        self,
    ) -> None:
        # Two entries share time 2: swapping their input order seals
        # the same canonical bytes because storage is sorted.
        swapped = (
            self.entries[0],
            self.entries[2],
            self.entries[1],
            self.entries[3],
            self.entries[4],
        )
        self.save()
        first = self.read_raw()
        self.save(swapped)
        self.assertEqual(first, self.read_raw())

    def test_empty_entries_seal_an_empty_ledger(self) -> None:
        self.save(())
        document = json.loads(self.read_raw().decode("utf-8"))
        self.assertEqual(document["entries"], [])
        result = self.page()
        self.assertEqual(result["items"], ())
        self.assertIsNone(result["next_cursor"])
        self.assertEqual(result["snapshot"]["entries"], 0)

    def test_save_replaces_a_previous_ledger_wholesale(self) -> None:
        self.save()
        replacement = ((9, make_chain(transaction(99))),)
        self.save(replacement)
        result = self.page()
        self.assertEqual(len(result["items"]), 1)
        self.assertEqual(result["items"][0]["at"], 9)
        self.assertEqual(result["snapshot"]["entries"], 1)


class SaveValidationTests(LedgerTestBase):
    def test_path_validation(self) -> None:
        for bad in (1, 1.5, b"x", ["x"], {"x": 1}, object(), None):
            with self.subTest(bad=type(bad).__name__):
                with self.assertRaises(TypeError):
                    BranchStore.save_diagnostic_ledger(bad, ())
        with self.assertRaises(ValueError):
            BranchStore.save_diagnostic_ledger("", ())

    def test_entries_container_and_shape(self) -> None:
        for bad in ([], "x", 1, None, {"a": 1}):
            with self.subTest(bad=type(bad).__name__):
                with self.assertRaises(TypeError):
                    BranchStore.save_diagnostic_ledger(self.path, bad)
        chain = self.chains[0]
        for bad_entry in (
            (1,),
            (1, chain, chain),
            [1, chain],
            "x1",
            1,
        ):
            with self.subTest(bad=bad_entry):
                with self.assertRaises(TypeError):
                    BranchStore.save_diagnostic_ledger(
                        self.path, (bad_entry,)
                    )

    def test_time_type_and_value(self) -> None:
        chain = self.chains[0]
        for bad_time in (True, False, 1.5, "1", None, b"1"):
            with self.subTest(bad=bad_time):
                with self.assertRaises(TypeError):
                    BranchStore.save_diagnostic_ledger(
                        self.path, ((bad_time, chain),)
                    )
        with self.assertRaises(ValueError):
            BranchStore.save_diagnostic_ledger(self.path, ((-1, chain),))

    def test_times_must_not_regress(self) -> None:
        with self.assertRaises(ValueError):
            BranchStore.save_diagnostic_ledger(
                self.path,
                ((2, self.chains[0]), (1, self.chains[1])),
            )
        # Equal times are accepted.
        BranchStore.save_diagnostic_ledger(
            self.path, ((2, self.chains[0]), (2, self.chains[1]))
        )

    def test_chain_container_and_element_types(self) -> None:
        with self.assertRaises(TypeError):
            BranchStore.save_diagnostic_ledger(
                self.path, ((1, list(self.chains[0])),)
            )
        with self.assertRaises(TypeError):
            BranchStore.save_diagnostic_ledger(
                self.path, ((1, (self.chains[0][0], 1)),)
            )

    def test_chain_authentication_rules(self) -> None:
        # Empty chain.
        with self.assertRaises(ValueError):
            BranchStore.save_diagnostic_ledger(self.path, ((1, ()),))
        # Broken prev link.
        broken = list(self.chains[0])
        document = json.loads(broken[1])
        document["prev"] = "f" * 64
        broken[1] = reseal(document)
        with self.assertRaises(ValueError):
            BranchStore.save_diagnostic_ledger(
                self.path, ((1, tuple(broken)),)
            )
        # Malformed record.
        with self.assertRaises(ValueError):
            BranchStore.save_diagnostic_ledger(
                self.path, ((1, ("not json",)),)
            )
        # Chain that never names a transaction.
        idle = reseal(
            {
                "format": "branching-city-twin/recovery-diagnostic",
                "version": 1,
                "transaction": "",
                "prev": "0" * 64,
                "phase": "",
                "index": {"state": "present", "matches": ""},
                "progress": {"state": "present", "matches": ""},
                "backups": {
                    "index": {"present": False, "intact": False},
                    "progress": {"present": False, "intact": False},
                },
                "disposition": "idle",
                "reason": "idle",
                "checksum": "",
            }
        )
        with self.assertRaises(ValueError):
            BranchStore.save_diagnostic_ledger(self.path, ((1, (idle,)),))
        # Duplicated transaction identity.
        with self.assertRaises(ValueError):
            BranchStore.save_diagnostic_ledger(
                self.path,
                ((1, self.chains[0]), (2, self.chains[0])),
            )

    def test_missing_directory_raises_oserror(self) -> None:
        missing = os.path.join(self.tmp.name, "nope", "ledger.json")
        with self.assertRaises(OSError):
            BranchStore.save_diagnostic_ledger(missing, self.entries)
        self.assertFalse(os.path.exists(missing))

    def test_failure_leaves_previous_ledger_untouched(self) -> None:
        self.save()
        before = self.read_raw()
        # A failing validation changes nothing.
        with self.assertRaises(ValueError):
            BranchStore.save_diagnostic_ledger(self.path, ((1, ()),))
        self.assertEqual(before, self.read_raw())
        # A failing replace changes nothing either.
        with mock.patch.object(
            BranchStore, "_replace_durable", side_effect=OSError("boom")
        ):
            with self.assertRaises(OSError):
                BranchStore.save_diagnostic_ledger(self.path, self.entries)
        self.assertEqual(before, self.read_raw())


class PageValidationTests(LedgerTestBase):
    def setUp(self) -> None:
        super().setUp()
        self.save()

    def test_path_validation(self) -> None:
        for bad in (1, 1.5, b"x", ["x"], {"x": 1}, object(), None):
            with self.subTest(bad=type(bad).__name__):
                with self.assertRaises(TypeError):
                    self.page(path=bad)
        with self.assertRaises(ValueError):
            self.page(path="")

    def test_missing_or_unreadable_ledger_raises_oserror(self) -> None:
        with self.assertRaises(OSError):
            self.page(path=os.path.join(self.tmp.name, "missing.json"))
        with self.assertRaises(OSError):
            self.page(path=self.tmp.name)

    def test_bound_validation(self) -> None:
        for name in ("start", "end"):
            for bad in (True, False, 1.5, "1", -1, b"1"):
                with self.subTest(name=name, bad=bad):
                    with self.assertRaises(TypeError):
                        self.page(**{name: bad})
        with self.assertRaises(ValueError):
            self.page(start=3, end=2)
        # Boundary values are fine.
        self.page(start=0, end=0)

    def test_transactions_validation(self) -> None:
        for bad in (["a"], "a", 1, {"a"}, {"a": 1}):
            with self.subTest(bad=type(bad).__name__):
                with self.assertRaises(TypeError):
                    self.page(transactions=bad)
        with self.assertRaises(TypeError):
            self.page(transactions=(transaction(0), 1))
        with self.assertRaises(ValueError):
            self.page(transactions=(transaction(0), ""))
        with self.assertRaises(ValueError):
            self.page(
                transactions=(transaction(0), transaction(0))
            )

    def test_limit_validation(self) -> None:
        for bad in (True, False, 1.5, "1", None, b"1"):
            with self.subTest(bad=type(bad).__name__):
                with self.assertRaises(TypeError):
                    self.page(limit=bad)
        for bad in (0, -1):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    self.page(limit=bad)

    def test_cursor_type_validation(self) -> None:
        for bad in (1, 1.5, b"x", ["x"], {"x": 1}, object()):
            with self.subTest(bad=type(bad).__name__):
                with self.assertRaises(TypeError):
                    self.page(cursor=bad)

    def test_cursor_value_validation(self) -> None:
        first = self.page(limit=2)
        cursor = first["next_cursor"]
        self.assertIsNotNone(cursor)
        # Empty string.
        with self.assertRaises(ValueError):
            self.page(cursor="")
        # Not JSON.
        with self.assertRaises(ValueError):
            self.page(cursor="not a cursor")
        # Non-canonical (pretty-printed).
        with self.assertRaises(ValueError):
            self.page(cursor=json.dumps(json.loads(cursor), indent=2))
        # Tampered checksum.
        document = json.loads(cursor)
        document["checksum"] = "f" * 64
        with self.assertRaises(ValueError):
            self.page(cursor=BranchStore._canonical_json(document))
        # Tampered body, resealed: offset beyond the ledger still
        # parses, but a foreign format string does not.
        document = json.loads(cursor)
        document["format"] = "branching-city-twin/other"
        body = {
            key: value
            for key, value in document.items()
            if key != "checksum"
        }
        document["checksum"] = hashlib.sha256(
            BranchStore._canonical_json(body).encode("utf-8")
        ).hexdigest()
        with self.assertRaises(ValueError):
            self.page(cursor=BranchStore._canonical_json(document))

    def test_cursor_binds_the_filters(self) -> None:
        first = self.page(limit=2)
        cursor = first["next_cursor"]
        with self.assertRaises(ValueError):
            self.page(limit=2, cursor=cursor, start=2)
        with self.assertRaises(ValueError):
            self.page(limit=2, cursor=cursor, transactions=())
        # A different limit with the same filters is fine.
        resumed = self.page(limit=3, cursor=cursor)
        self.assertEqual(len(resumed["items"]), 3)


class PageContentTests(LedgerTestBase):
    def setUp(self) -> None:
        super().setUp()
        self.save()
        self.summaries = BranchStore.summarize_recovery_diagnostics(
            tuple(self.chains)
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
            "records": self.chains[index],
        }

    def test_result_shape_and_full_page(self) -> None:
        result = self.page()
        self.assertEqual(
            list(result), ["snapshot", "items", "next_cursor"]
        )
        snapshot = result["snapshot"]
        raw = self.read_raw()
        self.assertEqual(
            snapshot["digest"], hashlib.sha256(raw).hexdigest()
        )
        self.assertEqual(snapshot["entries"], 5)
        self.assertEqual(snapshot["size"], len(raw))
        self.assertIsNone(result["next_cursor"])
        self.assertEqual(
            list(result["items"]),
            [self.expect_item(index) for index in self.expect_order],
        )
        for item in result["items"]:
            self.assertEqual(list(item), ["at", "summary", "records"])
            self.assertIsInstance(item["records"], tuple)

    def test_items_are_sorted_by_time_then_transaction(self) -> None:
        result = self.page()
        keys = [
            (item["at"], item["summary"]["transaction"])
            for item in result["items"]
        ]
        self.assertEqual(keys, sorted(keys))

    def test_inclusive_time_bounds(self) -> None:
        result = self.page(start=2, end=3)
        self.assertEqual(
            [item["at"] for item in result["items"]], [2, 2, 3]
        )
        point = self.page(start=1, end=1)
        self.assertEqual([item["at"] for item in point["items"]], [1])
        none = self.page(start=5, end=9)
        self.assertEqual(none["items"], ())
        self.assertIsNone(none["next_cursor"])
        self.assertEqual(none["snapshot"]["entries"], 5)

    def test_transaction_filter(self) -> None:
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
        # Unknown identities are simply never matched.
        unknown = self.page(transactions=("0" * 64,))
        self.assertEqual(unknown["items"], ())
        self.assertEqual(unknown["snapshot"]["entries"], 5)
        # An empty tuple matches nothing but still reports a snapshot.
        empty = self.page(transactions=())
        self.assertEqual(empty["items"], ())
        self.assertIsNone(empty["next_cursor"])
        self.assertEqual(empty["snapshot"]["entries"], 5)

    def test_pagination_walks_every_entry_once(self) -> None:
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

    def test_cursor_resumes_across_process_boundaries(self) -> None:
        # The cursor is a plain string: stashing it (as a restart
        # would) and resuming in a fresh call continues the walk.
        first = self.page(limit=4)
        stashed = json.loads(first["next_cursor"])
        resumed = self.page(
            limit=4, cursor=BranchStore._canonical_json(stashed)
        )
        self.assertEqual(len(resumed["items"]), 1)
        self.assertIsNone(resumed["next_cursor"])

    def test_filtered_pagination(self) -> None:
        result = self.page(start=2, end=4, limit=2)
        self.assertEqual(
            [item["at"] for item in result["items"]], [2, 2]
        )
        resumed = self.page(
            start=2, end=4, limit=2, cursor=result["next_cursor"]
        )
        self.assertEqual(
            [item["at"] for item in resumed["items"]], [3, 4]
        )
        self.assertIsNone(resumed["next_cursor"])

    def test_returned_levels_share_nothing(self) -> None:
        first = self.page()
        first["items"][0]["summary"]["transaction"] = "mutated"
        first["snapshot"]["digest"] = "mutated"
        second = self.page()
        self.assertNotEqual(
            second["items"][0]["summary"]["transaction"], "mutated"
        )
        self.assertNotEqual(second["snapshot"]["digest"], "mutated")
        self.assertEqual(first["items"][0]["at"], second["items"][0]["at"])


class PageConcurrencyTests(LedgerTestBase):
    def setUp(self) -> None:
        super().setUp()
        self.save()
        self.cursor = self.page(limit=2)["next_cursor"]

    def resume(self):
        return self.page(limit=2, cursor=self.cursor)

    def test_replaced_ledger_raises_runtimeerror(self) -> None:
        self.save(((7, make_chain(transaction(7))),))
        with self.assertRaises(RuntimeError):
            self.resume()

    def test_resealed_same_content_still_counts_as_replacement(
        self,
    ) -> None:
        # Atomic replacement publishes a new file identity even when
        # the bytes are identical.
        self.save()
        with self.assertRaises(RuntimeError):
            self.resume()

    def test_truncated_ledger_raises_runtimeerror(self) -> None:
        raw = self.read_raw()
        with open(self.path, "wb") as handle:
            handle.write(raw[:-10])
        with self.assertRaises(RuntimeError):
            self.resume()

    def test_appended_ledger_raises_runtimeerror(self) -> None:
        with open(self.path, "ab") as handle:
            handle.write(b" ")
        with self.assertRaises(RuntimeError):
            self.resume()

    def test_deleted_ledger_raises_oserror(self) -> None:
        os.remove(self.path)
        with self.assertRaises(OSError):
            self.resume()

    def test_unchanged_ledger_resumes(self) -> None:
        result = self.resume()
        self.assertEqual(len(result["items"]), 2)


class CorruptLedgerTests(LedgerTestBase):
    def setUp(self) -> None:
        super().setUp()
        self.save()

    def write_text(self, text: str) -> None:
        with open(self.path, "w", encoding="utf-8") as handle:
            handle.write(text)

    def tamper(self, **overrides) -> None:
        """Rewrite the ledger with one document-level change, resealing
        the checksum so only the named defect remains."""
        document = json.loads(self.read_raw().decode("utf-8"))
        document.update(overrides)
        body = {
            key: value
            for key, value in document.items()
            if key != "checksum"
        }
        document["checksum"] = hashlib.sha256(
            BranchStore._canonical_json(body).encode("utf-8")
        ).hexdigest()
        self.write_text(BranchStore._canonical_json(document))

    def assert_page_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            self.page()

    def test_bom(self) -> None:
        self.write_text("\ufeff" + self.read_raw().decode("utf-8"))
        self.assert_page_raises_value_error()

    def test_trailing_newline(self) -> None:
        self.write_text(self.read_raw().decode("utf-8") + "\n")
        self.assert_page_raises_value_error()

    def test_invalid_utf8(self) -> None:
        with open(self.path, "wb") as handle:
            handle.write(b"\xff\xfe")
        self.assert_page_raises_value_error()

    def test_not_json(self) -> None:
        self.write_text("not json")
        self.assert_page_raises_value_error()

    def test_non_canonical(self) -> None:
        document = json.loads(self.read_raw().decode("utf-8"))
        self.write_text(json.dumps(document, indent=2))
        self.assert_page_raises_value_error()

    def test_bad_checksum(self) -> None:
        document = json.loads(self.read_raw().decode("utf-8"))
        document["checksum"] = "f" * 64
        self.write_text(BranchStore._canonical_json(document))
        self.assert_page_raises_value_error()

    def test_unknown_format_and_version(self) -> None:
        self.tamper(format="branching-city-twin/other")
        self.assert_page_raises_value_error()
        self.tamper(version=2)
        self.assert_page_raises_value_error()

    def test_missing_and_extra_keys(self) -> None:
        document = json.loads(self.read_raw().decode("utf-8"))
        del document["entries"]
        body = {
            key: value
            for key, value in document.items()
            if key != "checksum"
        }
        document["checksum"] = hashlib.sha256(
            BranchStore._canonical_json(body).encode("utf-8")
        ).hexdigest()
        self.write_text(BranchStore._canonical_json(document))
        self.assert_page_raises_value_error()
        self.tamper(extra=1)
        self.assert_page_raises_value_error()

    def test_unsorted_entries(self) -> None:
        document = json.loads(self.read_raw().decode("utf-8"))
        entries = document["entries"]
        entries[0], entries[1] = entries[1], entries[0]
        self.tamper(entries=entries)
        self.assert_page_raises_value_error()

    def test_embedded_chain_authentication(self) -> None:
        document = json.loads(self.read_raw().decode("utf-8"))
        entries = document["entries"]
        # Break one record's prev link inside the ledger.
        chain = list(entries[1]["chain"])
        record = json.loads(chain[1])
        record["prev"] = "f" * 64
        chain[1] = reseal(record)
        entries[1]["chain"] = chain
        self.tamper(entries=entries)
        self.assert_page_raises_value_error()
        # Duplicate a whole entry: the transaction repeats.
        document = json.loads(self.read_raw().decode("utf-8"))
        entries = document["entries"]
        entries.append(dict(entries[-1]))
        self.tamper(entries=entries)
        self.assert_page_raises_value_error()

    def test_no_partial_page_from_a_corrupt_ledger(self) -> None:
        self.tamper(version=2)
        try:
            self.page()
        except ValueError:
            pass
        else:  # pragma: no cover - the assert above covers this
            self.fail("expected ValueError")


class ReadOnlyIsolationTests(LedgerTestBase):
    def test_paging_never_modifies_the_ledger(self) -> None:
        self.save()
        before = self.read_raw()
        cursor = None
        while True:
            result = self.page(limit=1, cursor=cursor)
            cursor = result["next_cursor"]
            if cursor is None:
                break
        self.assertEqual(before, self.read_raw())
        # Saving with invalid arguments also leaves the file alone.
        with self.assertRaises(TypeError):
            BranchStore.save_diagnostic_ledger(self.path, "x")
        self.assertEqual(before, self.read_raw())


if __name__ == "__main__":
    unittest.main()
