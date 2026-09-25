import hashlib
import json
import os
import shutil
import tempfile
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


class RecoveryAuditTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = os.path.join(self.tmp.name, "gens")
        os.mkdir(self.root)
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

    def snapshot(self) -> dict[str, bytes]:
        captured: dict[str, bytes] = {}
        for dirpath, _dirnames, filenames in os.walk(self.root):
            for filename in filenames:
                path = os.path.join(dirpath, filename)
                captured[path] = _read(path)
        return captured


class ExportRecoveryAuditTests(RecoveryAuditTestBase):
    def test_record_shape_and_canonical_encoding(self) -> None:
        self.rotate_three()
        record = BranchStore.export_recovery_audit(self.root)
        self.assertIsInstance(record, str)
        self.assertFalse(record.endswith("\n"))
        # Compact JSON: no structural whitespace anywhere.
        self.assertNotIn(" ", record)
        self.assertNotIn("\n", record)
        self.assertNotIn("\t", record)
        # UTF-8 decodes and parses with the exact envelope key order.
        document = json.loads(record)
        self.assertEqual(
            list(document),
            [
                "format",
                "version",
                "current",
                "selected",
                "generations",
                "ignored",
                "checksum",
            ],
        )
        self.assertEqual(document["format"], "branching-city-twin/recovery-audit")
        self.assertEqual(document["version"], 2)
        self.assertEqual(document["selected"], "generation-0000000000000003")
        self.assertEqual(
            document["current"],
            {
                "state": "ok",
                "value": "generation-0000000000000003",
                "digest": None,
            },
        )
        self.assertEqual(document["ignored"], [])
        self.assertIsInstance(document["checksum"], str)
        self.assertEqual(len(document["checksum"]), 64)

    def test_generation_records_carry_full_evidence(self) -> None:
        self.rotate_three()
        document = json.loads(BranchStore.export_recovery_audit(self.root))
        generations = document["generations"]
        self.assertEqual([g["number"] for g in generations], [1, 2, 3])
        self.assertEqual(
            [g["name"] for g in generations],
            [
                "generation-0000000000000001",
                "generation-0000000000000002",
                "generation-0000000000000003",
            ],
        )
        self.assertTrue(all(g["valid"] for g in generations))
        self.assertEqual(
            [g["dependency"] for g in generations],
            ["root", "linked", "linked"],
        )
        self.assertIsNone(generations[0]["previous"])
        # Each link cites the prior manifest's actual digest.
        for prior, later in zip(generations, generations[1:]):
            self.assertEqual(later["previous"], prior["manifest"])
        # The file digests match the bytes on disk and replay is clean.
        for generation in generations:
            directory = self.generation_dir(generation["number"])
            self.assertEqual(
                generation["manifest"],
                _sha256(_read(os.path.join(directory, "manifest.json"))),
            )
            self.assertEqual(
                generation["files"]["checkpoint"],
                _sha256(_read(os.path.join(directory, "checkpoint.json"))),
            )
            self.assertEqual(
                generation["files"]["journal"],
                _sha256(_read(os.path.join(directory, "journal.json"))),
            )
            self.assertEqual(generation["replay"], "ok")
            self.assertIsNone(generation["reason"])

    def test_checksum_covers_decision_and_evidence(self) -> None:
        self.rotate_three()
        record = BranchStore.export_recovery_audit(self.root)
        document = json.loads(record)
        body = {
            key: document[key]
            for key in (
                "format",
                "version",
                "current",
                "selected",
                "generations",
                "ignored",
            )
        }
        expected = _sha256(
            json.dumps(
                body,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        )
        self.assertEqual(document["checksum"], expected)

    def test_same_evidence_is_byte_identical(self) -> None:
        self.rotate_three()
        first = BranchStore.export_recovery_audit(self.root)
        second = BranchStore.export_recovery_audit(self.root)
        self.assertEqual(first, second)

    def test_identical_roots_emit_identical_records(self) -> None:
        self.rotate_three()
        other = os.path.join(self.tmp.name, "other")
        shutil.copytree(self.root, other)
        self.assertEqual(
            BranchStore.export_recovery_audit(self.root),
            BranchStore.export_recovery_audit(other),
        )

    def test_missing_pointer_state(self) -> None:
        self.rotate_three()
        os.remove(os.path.join(self.root, "CURRENT"))
        document = json.loads(BranchStore.export_recovery_audit(self.root))
        self.assertEqual(
            document["current"],
            {"state": "missing", "value": None, "digest": None},
        )
        self.assertEqual(document["selected"], "generation-0000000000000003")

    def test_stale_legal_pointer_reported_verbatim(self) -> None:
        self.rotate_three()
        with open(os.path.join(self.root, "CURRENT"), "wb") as handle:
            handle.write(b"generation-0000000000000001")
        document = json.loads(BranchStore.export_recovery_audit(self.root))
        self.assertEqual(
            document["current"],
            {
                "state": "ok",
                "value": "generation-0000000000000001",
                "digest": None,
            },
        )
        # The stale hint does not steer the evidence-based selection.
        self.assertEqual(document["selected"], "generation-0000000000000003")

    def test_corrupt_pointers_are_digested_not_echoed(self) -> None:
        self.rotate_three()
        # Each raw pointer carries a distinctive token that must never
        # appear in the audit record; only the "corrupt" summary and
        # the SHA-256 digest of the raw bytes do.
        cases = (
            b"\xff\xfe secret-pointer-one",
            b"",
            b"  ",
            b" secret-pointer-two",
            b"secret-pointer-three\n",
            b"secret-pointer-four",
        )
        digests = set()
        for raw in cases:
            with open(os.path.join(self.root, "CURRENT"), "wb") as handle:
                handle.write(raw)
            record = BranchStore.export_recovery_audit(self.root)
            document = json.loads(record)
            self.assertEqual(
                document["current"],
                {
                    "state": "corrupt",
                    "value": None,
                    "digest": _sha256(raw),
                },
            )
            digests.add(document["current"]["digest"])
            for token in (
                "secret-pointer-one",
                "secret-pointer-two",
                "secret-pointer-three",
                "secret-pointer-four",
            ):
                self.assertNotIn(token, record)
        # Distinct corrupt contents yield distinct digests, so two
        # different corruptions are no longer indistinguishable.
        self.assertEqual(len(digests), len(cases))

    def test_invalid_generations_recorded_with_reasons_and_dependencies(
        self,
    ) -> None:
        self.rotate_three()
        # Corrupt generation 2's checkpoint: it and dependent 3 fail.
        with open(
            os.path.join(self.generation_dir(2), "checkpoint.json"), "ab"
        ) as handle:
            handle.write(b" ")
        document = json.loads(BranchStore.export_recovery_audit(self.root))
        self.assertEqual(document["selected"], "generation-0000000000000001")
        by_number = {g["number"]: g for g in document["generations"]}
        self.assertTrue(by_number[1]["valid"])
        self.assertFalse(by_number[2]["valid"])
        self.assertEqual(by_number[2]["dependency"], "linked")
        self.assertIn("checkpoint", by_number[2]["reason"])
        self.assertNotEqual(by_number[2]["replay"], "ok")
        self.assertFalse(by_number[3]["valid"])
        self.assertEqual(by_number[3]["dependency"], "invalid")
        self.assertIn("invalid", by_number[3]["reason"])
        # The dependent's own file evidence is still captured read-only.
        self.assertIsNotNone(by_number[3]["files"]["checkpoint"])
        self.assertEqual(
            [entry["name"] for entry in document["ignored"]],
            ["generation-0000000000000002", "generation-0000000000000003"],
        )

    def test_export_without_valid_generation_raises(self) -> None:
        with self.assertRaises(ValueError):
            BranchStore.export_recovery_audit(self.root)

    def test_export_is_read_only(self) -> None:
        self.rotate_three()
        before = self.snapshot()
        BranchStore.export_recovery_audit(self.root)
        self.assertEqual(self.snapshot(), before)

    def test_root_validation(self) -> None:
        for bad in (1, 1.0, b"x", None, ["a"]):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    BranchStore.export_recovery_audit(bad)
        with self.assertRaises(ValueError):
            BranchStore.export_recovery_audit("")
        with self.assertRaises(OSError):
            BranchStore.export_recovery_audit(os.path.join(self.tmp.name, "nope"))
        with self.assertRaises(OSError):
            BranchStore.export_recovery_audit(self.cp)


class VerifyRecoveryAuditTests(RecoveryAuditTestBase):
    def test_verify_matches_fresh_evidence(self) -> None:
        self.rotate_three()
        record = BranchStore.export_recovery_audit(self.root)
        self.assertTrue(
            BranchStore.verify_recovery_audit(self.root, record)
        )

    def test_verify_is_idempotent_and_read_only(self) -> None:
        self.rotate_three()
        record = BranchStore.export_recovery_audit(self.root)
        before = self.snapshot()
        self.assertTrue(
            BranchStore.verify_recovery_audit(self.root, record)
        )
        self.assertTrue(
            BranchStore.verify_recovery_audit(self.root, record)
        )
        self.assertEqual(self.snapshot(), before)

    def test_pointer_change_is_evidence_change(self) -> None:
        self.rotate_three()
        record = BranchStore.export_recovery_audit(self.root)
        with open(os.path.join(self.root, "CURRENT"), "wb") as handle:
            handle.write(b"generation-0000000000000001")
        self.assertFalse(
            BranchStore.verify_recovery_audit(self.root, record)
        )

    def test_pointer_removal_is_evidence_change(self) -> None:
        self.rotate_three()
        record = BranchStore.export_recovery_audit(self.root)
        os.remove(os.path.join(self.root, "CURRENT"))
        self.assertFalse(
            BranchStore.verify_recovery_audit(self.root, record)
        )

    def test_generation_file_change_is_evidence_change(self) -> None:
        self.rotate_three()
        record = BranchStore.export_recovery_audit(self.root)
        with open(
            os.path.join(self.generation_dir(3), "journal.json"), "ab"
        ) as handle:
            handle.write(b" ")
        self.assertFalse(
            BranchStore.verify_recovery_audit(self.root, record)
        )

    def test_losing_all_generations_is_evidence_change(self) -> None:
        self.rotate_three()
        record = BranchStore.export_recovery_audit(self.root)
        empty = os.path.join(self.tmp.name, "empty")
        os.mkdir(empty)
        self.assertFalse(
            BranchStore.verify_recovery_audit(empty, record)
        )

    def test_non_canonical_encoding_rejected(self) -> None:
        self.rotate_three()
        record = BranchStore.export_recovery_audit(self.root)
        document = json.loads(record)
        pretty = json.dumps(document, indent=2)
        with self.assertRaises(ValueError):
            BranchStore.verify_recovery_audit(self.root, pretty)
        with self.assertRaises(ValueError):
            BranchStore.verify_recovery_audit(self.root, record + "\n")

    def test_bad_checksum_rejected(self) -> None:
        self.rotate_three()
        document = json.loads(BranchStore.export_recovery_audit(self.root))
        document["checksum"] = "0" * 64
        with self.assertRaises(ValueError):
            BranchStore.verify_recovery_audit(
                self.root, json.dumps(document, separators=(",", ":"))
            )

    def test_root_validation_precedes_record_validation(self) -> None:
        for bad in (1, 1.0, b"x", None, ["a"]):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    BranchStore.verify_recovery_audit(bad, "{}")
        with self.assertRaises(ValueError):
            BranchStore.verify_recovery_audit("", "{}")
        with self.assertRaises(OSError):
            BranchStore.verify_recovery_audit(
                os.path.join(self.tmp.name, "nope"), "{}"
            )

    def test_record_validation(self) -> None:
        self.rotate_three()
        for bad in (1, 1.0, b"x", None, ["x"]):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    BranchStore.verify_recovery_audit(self.root, bad)
        for bad in (
            "",
            "{",
            "not json",
            '{"a":1,"a":2}',
            "[]",
            "null",
            "{}",
        ):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    BranchStore.verify_recovery_audit(self.root, bad)

    def test_field_violations_rejected(self) -> None:
        self.rotate_three()
        record = BranchStore.export_recovery_audit(self.root)

        # Structural violations carry a recomputed checksum, so the
        # failure is a field ValueError rather than a checksum mismatch.
        for change in (
            {"version": 3},
            {"format": "other"},
            {"selected": "not-a-legal-generation-name"},
        ):
            document = json.loads(record)
            document.update(change)
            document["checksum"] = self.checksum_of(document)
            encoded = json.dumps(
                document, separators=(",", ":"), ensure_ascii=False
            )
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    BranchStore.verify_recovery_audit(self.root, encoded)

        # A structurally legal but wrong decision (a plausible, existing
        # generation number that the evidence did not select) instead
        # verifies the checksum fine and reports an evidence mismatch.
        document = json.loads(record)
        document["selected"] = "generation-0000000000000001"
        document["checksum"] = self.checksum_of(document)
        encoded = json.dumps(document, separators=(",", ":"), ensure_ascii=False)
        self.assertFalse(
            BranchStore.verify_recovery_audit(self.root, encoded)
        )
        # The unmodified envelope still verifies.
        self.assertTrue(
            BranchStore.verify_recovery_audit(self.root, record)
        )

    def test_version_one_records_still_verify(self) -> None:
        self.rotate_three()
        document = json.loads(BranchStore.export_recovery_audit(self.root))
        # Downgrade the sealed record to version 1: drop the pointer
        # digest and re-seal with a fresh checksum.
        document["version"] = 1
        document["current"] = {
            "state": document["current"]["state"],
            "value": document["current"]["value"],
        }
        document["checksum"] = self.checksum_of(document)
        record = json.dumps(document, separators=(",", ":"), ensure_ascii=False)
        self.assertTrue(BranchStore.verify_recovery_audit(self.root, record))
        # A pointer change is still an evidence change under version 1.
        with open(os.path.join(self.root, "CURRENT"), "wb") as handle:
            handle.write(b"generation-0000000000000001")
        self.assertFalse(BranchStore.verify_recovery_audit(self.root, record))

    def test_version_two_detects_corrupt_pointer_replacement(self) -> None:
        self.rotate_three()
        pointer = os.path.join(self.root, "CURRENT")
        with open(pointer, "wb") as handle:
            handle.write(b"corrupt-one")
        record = BranchStore.export_recovery_audit(self.root)
        self.assertTrue(BranchStore.verify_recovery_audit(self.root, record))
        # Same "corrupt" state, different raw bytes: version 2 records
        # pin the digest, so the replacement is an evidence change.
        with open(pointer, "wb") as handle:
            handle.write(b"corrupt-two")
        self.assertFalse(BranchStore.verify_recovery_audit(self.root, record))

    def test_version_one_ignores_corrupt_pointer_content(self) -> None:
        self.rotate_three()
        pointer = os.path.join(self.root, "CURRENT")
        with open(pointer, "wb") as handle:
            handle.write(b"corrupt-one")
        document = json.loads(BranchStore.export_recovery_audit(self.root))
        document["version"] = 1
        document["current"] = {
            "state": "corrupt",
            "value": None,
        }
        document["checksum"] = self.checksum_of(document)
        record = json.dumps(document, separators=(",", ":"), ensure_ascii=False)
        self.assertTrue(BranchStore.verify_recovery_audit(self.root, record))
        # Version 1 records carry no digest, so a content swap between
        # two corruptions remains invisible to them.
        with open(pointer, "wb") as handle:
            handle.write(b"corrupt-two")
        self.assertTrue(BranchStore.verify_recovery_audit(self.root, record))

    @staticmethod
    def checksum_of(document: dict[str, object]) -> str:
        body = {
            key: document[key]
            for key in (
                "format",
                "version",
                "current",
                "selected",
                "generations",
                "ignored",
            )
        }
        return _sha256(
            json.dumps(
                body,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        )


class DiffRecoveryAuditTests(RecoveryAuditTestBase):
    def seal(self) -> str:
        return BranchStore.export_recovery_audit(self.root)

    def test_no_changes_returns_empty_tuple(self) -> None:
        self.rotate_three()
        record = self.seal()
        changes = BranchStore.diff_recovery_audit(self.root, record)
        self.assertIsInstance(changes, tuple)
        self.assertEqual(changes, ())

    def test_pointer_state_change_reported_first(self) -> None:
        self.rotate_three()
        record = self.seal()
        os.remove(os.path.join(self.root, "CURRENT"))
        changes = BranchStore.diff_recovery_audit(self.root, record)
        self.assertEqual(len(changes), 1)
        category, generation, change, before, after = changes[0]
        self.assertEqual(category, "pointer")
        self.assertIsNone(generation)
        self.assertEqual(change, "changed")
        self.assertEqual(
            before,
            {
                "state": "ok",
                "value": "generation-0000000000000003",
                "digest": None,
            },
        )
        self.assertEqual(
            after,
            {"state": "missing", "value": None, "digest": None},
        )

    def test_corrupt_pointer_replacement_classified_by_digest(self) -> None:
        self.rotate_three()
        pointer = os.path.join(self.root, "CURRENT")
        with open(pointer, "wb") as handle:
            handle.write(b"corrupt-one")
        record = self.seal()
        with open(pointer, "wb") as handle:
            handle.write(b"corrupt-two")
        changes = BranchStore.diff_recovery_audit(self.root, record)
        self.assertEqual(len(changes), 1)
        category, generation, change, before, after = changes[0]
        self.assertEqual((category, generation, change), ("pointer", None, "changed"))
        self.assertEqual(
            before,
            {"state": "corrupt", "value": None, "digest": _sha256(b"corrupt-one")},
        )
        self.assertEqual(
            after,
            {"state": "corrupt", "value": None, "digest": _sha256(b"corrupt-two")},
        )
        # Raw pointer bytes never appear in the diff.
        for change_tuple in changes:
            self.assertNotIn(b"corrupt-one", repr(change_tuple).encode())
            self.assertNotIn(b"corrupt-two", repr(change_tuple).encode())

    def test_pointer_change_against_version_one_record(self) -> None:
        self.rotate_three()
        document = json.loads(self.seal())
        document["version"] = 1
        document["current"] = {
            "state": document["current"]["state"],
            "value": document["current"]["value"],
        }
        document["checksum"] = VerifyRecoveryAuditTests.checksum_of(document)
        record = json.dumps(document, separators=(",", ":"), ensure_ascii=False)
        self.assertEqual(BranchStore.diff_recovery_audit(self.root, record), ())
        with open(os.path.join(self.root, "CURRENT"), "wb") as handle:
            handle.write(b"generation-0000000000000001")
        changes = BranchStore.diff_recovery_audit(self.root, record)
        self.assertEqual(len(changes), 1)
        category, generation, change, before, after = changes[0]
        self.assertEqual((category, generation, change), ("pointer", None, "changed"))
        # A version 1 record compares state and value only: no digest.
        self.assertEqual(
            before,
            {"state": "ok", "value": "generation-0000000000000003"},
        )
        self.assertEqual(
            after,
            {"state": "ok", "value": "generation-0000000000000001"},
        )

    def test_generation_file_change_classified(self) -> None:
        self.rotate_three()
        record = self.seal()
        journal = os.path.join(self.generation_dir(3), "journal.json")
        original = _sha256(_read(journal))
        with open(journal, "ab") as handle:
            handle.write(b" ")
        changes = BranchStore.diff_recovery_audit(self.root, record)
        by_category = {}
        for category, generation, change, before, after in changes:
            by_category.setdefault(category, []).append(
                (generation, change, before, after)
            )
        # The tampered journal digest changed for generation 3.
        self.assertEqual(len(by_category["journal"]), 1)
        generation, change, before, after = by_category["journal"][0]
        self.assertEqual(generation, "generation-0000000000000003")
        self.assertEqual(change, "changed")
        self.assertEqual(before, original)
        self.assertEqual(after, _sha256(_read(journal)))
        # Its validity, replay conclusion and the selection changed too.
        self.assertIn(
            ("generation-0000000000000003", "changed"),
            [(g, c) for g, c, _b, _a in by_category["chain"]],
        )
        self.assertIn(
            ("generation-0000000000000003", "changed"),
            [(g, c) for g, c, _b, _a in by_category["replay"]],
        )
        self.assertIn(
            (None, "changed"),
            [(g, c) for g, c, _b, _a in by_category["chain"]],
        )
        # Ordering: pointer (none here), then selection, then
        # per-generation chain, checkpoint, journal, replay in
        # ascending generation number order.
        categories = [entry[0] for entry in changes]
        generations = [entry[1] for entry in changes]
        self.assertEqual(generations[0], None)
        tail = [
            (g, c)
            for g, c in zip(generations[1:], categories[1:])
        ]
        numbers = [
            int(g.rsplit("-", 1)[1]) if g is not None else -1
            for g, _c in tail
        ]
        self.assertEqual(numbers, sorted(numbers))

    def test_added_and_removed_generations(self) -> None:
        store = self.rotate_three()
        record = self.seal()
        store.append("main", "m6", 8, {"a": 6})
        store.rotate_generation(self.root)
        added = BranchStore.diff_recovery_audit(self.root, record)
        added_names = {
            entry[1]
            for entry in added
            if entry[1] == "generation-0000000000000004"
        }
        self.assertEqual(added_names, {"generation-0000000000000004"})
        kinds = {entry[0] for entry in added if entry[1] is not None}
        self.assertEqual(kinds, {"chain", "checkpoint", "journal", "replay"})
        for entry in added:
            if entry[1] == "generation-0000000000000004":
                self.assertEqual(entry[2], "added")
                self.assertIsNone(entry[3])
                self.assertIsNotNone(entry[4])
        # Removing the newest generation again flips the direction.
        record_two = self.seal()
        shutil.rmtree(self.generation_dir(4))
        with open(os.path.join(self.root, "CURRENT"), "wb") as handle:
            handle.write(b"generation-0000000000000003")
        removed = BranchStore.diff_recovery_audit(self.root, record_two)
        for entry in removed:
            if entry[1] == "generation-0000000000000004":
                self.assertEqual(entry[2], "removed")
                self.assertIsNotNone(entry[3])
                self.assertIsNone(entry[4])

    def test_losing_all_generations_reports_changes_not_error(self) -> None:
        self.rotate_three()
        record = self.seal()
        empty = os.path.join(self.tmp.name, "empty")
        os.mkdir(empty)
        changes = BranchStore.diff_recovery_audit(empty, record)
        self.assertNotEqual(changes, ())
        categories = [entry[0] for entry in changes]
        self.assertEqual(categories[0], "pointer")
        # The selection and every sealed generation are gone.
        selection = [entry for entry in changes if entry[1] is None and entry[0] == "chain"]
        self.assertEqual(len(selection), 1)
        self.assertEqual(selection[0][3], "generation-0000000000000003")
        self.assertIsNone(selection[0][4])
        removed = [entry for entry in changes if entry[2] == "removed"]
        self.assertTrue(removed)
        self.assertFalse(
            any(entry[2] == "added" for entry in changes)
        )

    def test_diff_is_read_only(self) -> None:
        self.rotate_three()
        record = self.seal()
        with open(os.path.join(self.root, "CURRENT"), "wb") as handle:
            handle.write(b"generation-0000000000000001")
        before = self.snapshot()
        BranchStore.diff_recovery_audit(self.root, record)
        self.assertEqual(self.snapshot(), before)

    def test_root_validation_precedes_record_validation(self) -> None:
        for bad in (1, 1.0, b"x", None, ["a"]):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    BranchStore.diff_recovery_audit(bad, "{}")
        with self.assertRaises(ValueError):
            BranchStore.diff_recovery_audit("", "{}")
        with self.assertRaises(OSError):
            BranchStore.diff_recovery_audit(
                os.path.join(self.tmp.name, "nope"), "{}"
            )

    def test_record_validation(self) -> None:
        self.rotate_three()
        for bad in (1, 1.0, b"x", None, ["x"]):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    BranchStore.diff_recovery_audit(self.root, bad)
        for bad in (
            "",
            "{",
            "not json",
            '{"a":1,"a":2}',
            "[]",
            "null",
            "{}",
        ):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    BranchStore.diff_recovery_audit(self.root, bad)
        # A tampered checksum is rejected, not reported as a diff.
        document = json.loads(self.seal())
        document["checksum"] = "0" * 64
        with self.assertRaises(ValueError):
            BranchStore.diff_recovery_audit(
                self.root, json.dumps(document, separators=(",", ":"))
            )


if __name__ == "__main__":
    unittest.main()
