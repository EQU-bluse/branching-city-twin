"""Tests for the persistent diagnostic ledger: save_diagnostic_ledger
and page_diagnostic_ledger."""

import hashlib
import json
import os
import unittest
from unittest import mock

from city_twin.branches import BranchStore

from tests.test_recovery_audit_recovery import (
    RecoveryPublicationTestBase,
)


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


class DiagnosticLedgerTestBase(RecoveryPublicationTestBase):
    def setUp(self) -> None:
        super().setUp()
        self.grow_chain(5)
        self.build()
        self.ledger = os.path.join(self.tmp.name, "ledger.json")
        self.counter = 20

    def make_chain(self) -> tuple:
        """Interrupt one publication so a fresh transaction appears,
        observe it pending, recover, and observe it settled: one
        authenticating two-record chain naming a unique transaction."""
        self.grow_chain(3, start=self.counter)
        self.counter += 10
        self.kill_publication_at("index_replace")
        pending = BranchStore.recovery_diagnostic(
            self.index, self.progress, self.recovery
        )
        self.recover_now()
        settled = BranchStore.recovery_diagnostic(
            self.index, self.progress, self.recovery, pending
        )
        self.assertTrue(
            BranchStore.verify_recovery_diagnostics((pending, settled))
        )
        return (pending, settled)

    def save_default(self, count: int = 3) -> list:
        chains = [self.make_chain() for _ in range(count)]
        entries = tuple((i * 10, chain) for i, chain in enumerate(chains))
        BranchStore.save_diagnostic_ledger(self.ledger, entries)
        return chains

    def page(self, **overrides):
        arguments = {
            "path": self.ledger,
            "start": None,
            "end": None,
            "transactions": None,
            "limit": 100,
            "cursor": None,
        }
        arguments.update(overrides)
        return BranchStore.page_diagnostic_ledger(**arguments)


class SaveDiagnosticLedgerValidationTests(DiagnosticLedgerTestBase):
    def test_path_type_and_emptiness(self) -> None:
        chain = self.make_chain()
        for bad in (None, 7, b"x", (self.ledger,)):
            with self.assertRaises(TypeError):
                BranchStore.save_diagnostic_ledger(bad, ((0, chain),))
        with self.assertRaises(ValueError):
            BranchStore.save_diagnostic_ledger("", ((0, chain),))

    def test_entries_shape_and_type(self) -> None:
        chain = self.make_chain()
        with self.assertRaises(TypeError):
            BranchStore.save_diagnostic_ledger(self.ledger, [(0, chain)])
        with self.assertRaises(TypeError):
            BranchStore.save_diagnostic_ledger(self.ledger, ([0, chain],))
        with self.assertRaises(TypeError):
            BranchStore.save_diagnostic_ledger(self.ledger, ((0, chain, 1),))
        with self.assertRaises(TypeError):
            BranchStore.save_diagnostic_ledger(self.ledger, ((0,),))

    def test_time_domain(self) -> None:
        chain = self.make_chain()
        for bad in (True, 1.5, "0"):
            with self.assertRaises(TypeError):
                BranchStore.save_diagnostic_ledger(
                    self.ledger, ((bad, chain),)
                )
        with self.assertRaises(ValueError):
            BranchStore.save_diagnostic_ledger(self.ledger, ((-1, chain),))

    def test_times_must_not_go_backwards(self) -> None:
        first, second = self.make_chain(), self.make_chain()
        with self.assertRaises(ValueError):
            BranchStore.save_diagnostic_ledger(
                self.ledger, ((10, first), (5, second))
            )
        # Equal times are allowed (non-decreasing).
        BranchStore.save_diagnostic_ledger(
            self.ledger, ((10, first), (10, second))
        )

    def test_chain_validation(self) -> None:
        chain = self.make_chain()
        with self.assertRaises(TypeError):
            BranchStore.save_diagnostic_ledger(self.ledger, ((0, list(chain)),))
        with self.assertRaises(TypeError):
            BranchStore.save_diagnostic_ledger(
                self.ledger, ((0, (chain[0], 7)),)
            )
        with self.assertRaises(ValueError):
            BranchStore.save_diagnostic_ledger(self.ledger, ((0, ()),))
        # A record that does not parse fails authentication.
        with self.assertRaises(ValueError):
            BranchStore.save_diagnostic_ledger(
                self.ledger, ((0, ("not a diagnostic",)),)
            )
        # A chain that parses but does not authenticate.
        document = json.loads(chain[0])
        document["disposition"] = "committed"
        tampered = (reseal(document),)
        with self.assertRaises(ValueError):
            BranchStore.save_diagnostic_ledger(self.ledger, ((0, tampered),))

    def test_duplicate_transaction_rejected(self) -> None:
        chain = self.make_chain()
        with self.assertRaises(ValueError):
            BranchStore.save_diagnostic_ledger(
                self.ledger, ((0, chain), (1, chain))
            )

    def test_missing_directory_raises_oserror(self) -> None:
        chain = self.make_chain()
        with self.assertRaises(OSError):
            BranchStore.save_diagnostic_ledger(
                os.path.join(self.tmp.name, "nope", "ledger.json"),
                ((0, chain),),
            )


class SaveDiagnosticLedgerFileTests(DiagnosticLedgerTestBase):
    def test_file_is_compact_checksummed_and_deterministic(self) -> None:
        chains = self.save_default(2)
        with open(self.ledger, "rb") as handle:
            raw = handle.read()
        self.assertFalse(raw.startswith(b"\xef\xbb\xbf"))
        self.assertFalse(raw.endswith(b"\n"))
        self.assertNotIn(b"\n", raw)
        self.assertNotIn(b" ", raw)
        document = json.loads(raw.decode("utf-8"))
        self.assertEqual(
            list(document), ["format", "version", "entries", "checksum"]
        )
        self.assertEqual(
            document["format"], "branching-city-twin/diagnostic-ledger"
        )
        self.assertEqual(document["version"], 1)
        body = {
            key: value for key, value in document.items() if key != "checksum"
        }
        self.assertEqual(
            document["checksum"],
            hashlib.sha256(
                BranchStore._canonical_json(body).encode("utf-8")
            ).hexdigest(),
        )
        self.assertEqual(
            document["entries"],
            [
                {"time": i * 10, "records": list(chain)}
                for i, chain in enumerate(chains)
            ],
        )
        # Equal input seals byte-for-byte identical files.
        other = os.path.join(self.tmp.name, "other.json")
        BranchStore.save_diagnostic_ledger(
            other,
            tuple((i * 10, chain) for i, chain in enumerate(chains)),
        )
        with open(other, "rb") as handle:
            self.assertEqual(handle.read(), raw)

    def test_empty_entries_seal_an_empty_ledger(self) -> None:
        BranchStore.save_diagnostic_ledger(self.ledger, ())
        result = self.page()
        self.assertEqual(result["items"], [])
        self.assertIsNone(result["next_cursor"])
        self.assertTrue(result["snapshot"])

    def test_failed_save_keeps_the_old_ledger(self) -> None:
        self.save_default(1)
        with open(self.ledger, "rb") as handle:
            before = handle.read()
        chain = self.make_chain()
        real_replace = os.replace

        def flaky_replace(src, dst):
            if os.path.realpath(dst) == os.path.realpath(self.ledger):
                raise OSError("simulated replace failure")
            return real_replace(src, dst)

        with mock.patch(
            "city_twin.branches.os.replace", flaky_replace
        ):
            with self.assertRaises(OSError):
                BranchStore.save_diagnostic_ledger(
                    self.ledger, ((0, chain),)
                )
        with open(self.ledger, "rb") as handle:
            self.assertEqual(handle.read(), before)
        # No temporary file is stranded.
        self.assertEqual(
            [
                name
                for name in os.listdir(self.tmp.name)
                if name.startswith(".diagnostic-ledger-")
            ],
            [],
        )


class PageDiagnosticLedgerValidationTests(DiagnosticLedgerTestBase):
    def setUp(self) -> None:
        super().setUp()
        self.save_default(3)

    def test_path_and_missing_file(self) -> None:
        for bad in (None, 7, b"x"):
            with self.assertRaises(TypeError):
                self.page(path=bad)
        with self.assertRaises(ValueError):
            self.page(path="")
        with self.assertRaises(OSError):
            self.page(path=os.path.join(self.tmp.name, "missing.json"))

    def test_time_bounds(self) -> None:
        for bad in (True, -1, 1.5, "0"):
            with self.assertRaises(TypeError):
                self.page(start=bad)
            with self.assertRaises(TypeError):
                self.page(end=bad)
        with self.assertRaises(ValueError):
            self.page(start=10, end=5)
        # Reversed bounds fail before the file is even consulted.
        with self.assertRaises(ValueError):
            self.page(
                path=os.path.join(self.tmp.name, "missing.json"),
                start=10,
                end=5,
            )

    def test_transactions_filter_validation(self) -> None:
        with self.assertRaises(TypeError):
            self.page(transactions=["a"])
        with self.assertRaises(TypeError):
            self.page(transactions=("a", 7))
        with self.assertRaises(ValueError):
            self.page(transactions=("a", ""))
        with self.assertRaises(ValueError):
            self.page(transactions=("a", "a"))

    def test_limit_validation(self) -> None:
        for bad in (True, "2", 2.0):
            with self.assertRaises(TypeError):
                self.page(limit=bad)
        for bad in (0, -3):
            with self.assertRaises(ValueError):
                self.page(limit=bad)

    def test_cursor_validation(self) -> None:
        for bad in (7, b"x", ("x",)):
            with self.assertRaises(TypeError):
                self.page(cursor=bad)
        with self.assertRaises(ValueError):
            self.page(cursor="")
        with self.assertRaises(ValueError):
            self.page(cursor="not a cursor")
        # A pretty-printed (non-canonical) cursor is rejected.
        first = self.page(limit=1)
        document = json.loads(first["next_cursor"])
        with self.assertRaises(ValueError):
            self.page(limit=1, cursor=json.dumps(document, indent=2))
        # A checksum mismatch is rejected.
        document["offset"] = 0
        with self.assertRaises(ValueError):
            self.page(
                limit=1, cursor=BranchStore._canonical_json(document)
            )

    def test_cursor_binds_the_filter(self) -> None:
        first = self.page(limit=1)
        cursor = first["next_cursor"]
        with self.assertRaises(ValueError):
            self.page(limit=1, cursor=cursor, start=1)
        with self.assertRaises(ValueError):
            self.page(limit=2, cursor=cursor)
        with self.assertRaises(ValueError):
            self.page(limit=1, cursor=cursor, transactions=("x",))
        # The matching continuation is accepted.
        self.page(limit=1, cursor=cursor)


class PageDiagnosticLedgerContentTests(DiagnosticLedgerTestBase):
    def test_corrupt_ledgers_raise_valueerror(self) -> None:
        self.save_default(1)
        with open(self.ledger, "rb") as handle:
            raw = handle.read()
        document = json.loads(raw.decode("utf-8"))

        variants = []
        # BOM and trailing newline.
        variants.append(b"\xef\xbb\xbf" + raw)
        variants.append(raw + b"\n")
        # Checksum mismatch.
        variants.append(raw[:-70] + b"0" * 64 + raw[-6:])
        # Bad top-level shape.
        variants.append(b"[]")
        # Ledger-level checksum valid but embedded chain broken.
        record = json.loads(document["entries"][0]["records"][0])
        record["disposition"] = "committed"
        document["entries"][0]["records"][0] = reseal(record)
        variants.append(reseal(document).encode("utf-8"))
        # Decreasing times, resealed.
        healthy = json.loads(raw.decode("utf-8"))
        healthy["entries"].append(
            {"time": -1 + healthy["entries"][0]["time"], "records": healthy["entries"][0]["records"]}
        )
        variants.append(reseal(healthy).encode("utf-8"))

        for variant in variants:
            with open(self.ledger, "wb") as handle:
                handle.write(variant)
            with self.assertRaises(ValueError):
                self.page()

    def test_pages_cover_every_entry_exactly_once(self) -> None:
        chains = self.save_default(5)
        seen = []
        cursor = None
        snapshots = set()
        pages = 0
        while True:
            result = self.page(limit=2, cursor=cursor)
            self.assertEqual(
                list(result), ["snapshot", "items", "next_cursor"]
            )
            snapshots.add(result["snapshot"])
            seen.extend(result["items"])
            pages += 1
            cursor = result["next_cursor"]
            if cursor is None:
                break
        self.assertEqual(pages, 3)
        self.assertEqual(len(snapshots), 1)
        self.assertEqual([item["time"] for item in seen], [0, 10, 20, 30, 40])
        for item, chain in zip(seen, chains):
            self.assertEqual(item["records"], chain)
            self.assertEqual(
                item["summary"],
                BranchStore.summarize_recovery_diagnostics((chain,))[0],
            )

    def test_items_sort_by_time_then_transaction(self) -> None:
        chains = [self.make_chain() for _ in range(3)]
        BranchStore.save_diagnostic_ledger(
            self.ledger,
            tuple((0, chain) for chain in chains),
        )
        result = self.page()
        keys = [
            (item["time"], item["summary"]["transaction"])
            for item in result["items"]
        ]
        self.assertEqual(keys, sorted(keys))

    def test_time_bounds_are_inclusive(self) -> None:
        self.save_default(3)  # times 0, 10, 20
        result = self.page(start=10, end=20)
        self.assertEqual(
            [item["time"] for item in result["items"]], [10, 20]
        )
        self.assertIsNone(result["next_cursor"])
        self.assertEqual(self.page(start=0, end=0)["items"][0]["time"], 0)
        self.assertEqual(self.page(start=21)["items"], [])

    def test_transactions_filter(self) -> None:
        chains = self.save_default(3)
        summaries = BranchStore.summarize_recovery_diagnostics(
            tuple(chains)
        )
        wanted = (summaries[0]["transaction"], summaries[2]["transaction"])
        result = self.page(transactions=wanted)
        self.assertEqual(
            [item["summary"]["transaction"] for item in result["items"]],
            [wanted[0], wanted[1]],
        )
        # Unknown identities match nothing and never raise.
        result = self.page(transactions=("0" * 64,))
        self.assertEqual(result["items"], [])
        self.assertTrue(result["snapshot"])
        # The empty tuple matches nothing.
        result = self.page(transactions=())
        self.assertEqual(result["items"], [])
        self.assertIsNone(result["next_cursor"])
        self.assertTrue(result["snapshot"])

    def test_continuation_across_versions_raises_runtimeerror(self) -> None:
        self.save_default(3)
        first = self.page(limit=1)
        cursor = first["next_cursor"]
        # Replacement with a fresh ledger.
        self.save_default(3)
        with self.assertRaises(RuntimeError):
            self.page(limit=1, cursor=cursor)
        # Append.
        self.save_default(3)
        cursor = self.page(limit=1)["next_cursor"]
        with open(self.ledger, "ab") as handle:
            handle.write(b" ")
        with self.assertRaises(RuntimeError):
            self.page(limit=1, cursor=cursor)
        # Truncation.
        self.save_default(3)
        cursor = self.page(limit=1)["next_cursor"]
        with open(self.ledger, "r+b") as handle:
            handle.truncate(20)
        with self.assertRaises(RuntimeError):
            self.page(limit=1, cursor=cursor)

    def test_query_is_read_only_and_results_do_not_alias(self) -> None:
        self.save_default(2)
        with open(self.ledger, "rb") as handle:
            before = handle.read()
        first = self.page(limit=1)
        first["items"][0]["summary"]["transaction"] = "mutated"
        first["items"][0]["records"] += ("extra",)
        second = self.page(limit=1)
        self.assertNotEqual(
            second["items"][0]["summary"]["transaction"], "mutated"
        )
        self.assertEqual(len(second["items"][0]["records"]), 2)
        with open(self.ledger, "rb") as handle:
            self.assertEqual(handle.read(), before)


if __name__ == "__main__":
    unittest.main()
