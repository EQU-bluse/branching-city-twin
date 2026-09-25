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
        # The nested pointer object keeps a fixed key order.
        self.assertEqual(
            list(document["current"]), ["state", "value", "digest"]
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

    def test_corrupt_pointers_are_summarized_not_echoed(self) -> None:
        self.rotate_three()
        # Each raw pointer carries a distinctive token that must never
        # appear in the audit record; only the "corrupt" state and the
        # SHA-256 digest of the raw bytes do.
        cases = (
            b"\xff\xfe secret-pointer-one",
            b"",
            b"  ",
            b" secret-pointer-two",
            b"secret-pointer-three\n",
            b"secret-pointer-four",
        )
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
            for token in (
                "secret-pointer-one",
                "secret-pointer-two",
                "secret-pointer-three",
                "secret-pointer-four",
            ):
                self.assertNotIn(token, record)

    def test_distinct_corrupt_pointers_are_distinguishable(self) -> None:
        self.rotate_three()
        with open(os.path.join(self.root, "CURRENT"), "wb") as handle:
            handle.write(b"first-corruption")
        first = BranchStore.export_recovery_audit(self.root)
        with open(os.path.join(self.root, "CURRENT"), "wb") as handle:
            handle.write(b"second-corruption")
        second = BranchStore.export_recovery_audit(self.root)
        self.assertNotEqual(first, second)
        self.assertFalse(
            BranchStore.verify_recovery_audit(self.root, first)
        )

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


def _reencode(document: dict[str, object]) -> str:
    return json.dumps(
        document, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


class VersionCompatibilityTests(RecoveryAuditTestBase):
    def v1_record(self, record: str) -> str:
        """Rewrite a version 2 record as a signed version 1 one."""
        document = json.loads(record)
        document["version"] = 1
        current = document["current"]
        assert set(current) == {"state", "value", "digest"}
        document["current"] = {"state": current["state"], "value": current["value"]}
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
        document["checksum"] = _sha256(_reencode(body).encode("utf-8"))
        return _reencode(document)

    def test_version_one_records_are_accepted(self) -> None:
        self.rotate_three()
        record = self.v1_record(
            BranchStore.export_recovery_audit(self.root)
        )
        self.assertEqual(json.loads(record)["version"], 1)
        self.assertTrue(
            BranchStore.verify_recovery_audit(self.root, record)
        )
        self.assertEqual(
            BranchStore.diff_recovery_audit(self.root, record), ()
        )

    def test_version_two_missing_digest_rejected(self) -> None:
        self.rotate_three()
        document = json.loads(
            BranchStore.export_recovery_audit(self.root)
        )
        document["current"] = {
            "state": document["current"]["state"],
            "value": document["current"]["value"],
        }
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
        document["checksum"] = _sha256(
            _reencode(body).encode("utf-8")
        )
        with self.assertRaises(ValueError):
            BranchStore.verify_recovery_audit(
                self.root, _reencode(document)
            )

    def test_version_one_is_blind_to_corrupt_pointer_replacement(self) -> None:
        self.rotate_three()
        with open(os.path.join(self.root, "CURRENT"), "wb") as handle:
            handle.write(b"first-corruption")
        record = self.v1_record(
            BranchStore.export_recovery_audit(self.root)
        )
        with open(os.path.join(self.root, "CURRENT"), "wb") as handle:
            handle.write(b"second-corruption")
        # Version 1 keeps only the corrupt state, so this is invisible.
        self.assertTrue(
            BranchStore.verify_recovery_audit(self.root, record)
        )
        self.assertEqual(
            BranchStore.diff_recovery_audit(self.root, record), ()
        )

    def test_version_two_detects_corrupt_pointer_replacement(self) -> None:
        self.rotate_three()
        with open(os.path.join(self.root, "CURRENT"), "wb") as handle:
            handle.write(b"first-corruption")
        record = BranchStore.export_recovery_audit(self.root)
        with open(os.path.join(self.root, "CURRENT"), "wb") as handle:
            handle.write(b"second-corruption")
        self.assertFalse(
            BranchStore.verify_recovery_audit(self.root, record)
        )
        changes = BranchStore.diff_recovery_audit(self.root, record)
        self.assertEqual(len(changes), 1)
        category, generation, change, before, after = changes[0]
        self.assertEqual(
            (category, generation, change),
            ("pointer", None, "changed"),
        )
        self.assertEqual(before["state"], "corrupt")
        self.assertIsNone(before["value"])
        self.assertEqual(
            before["digest"], _sha256(b"first-corruption")
        )
        self.assertEqual(
            after["digest"], _sha256(b"second-corruption")
        )
        for side in (before, after):
            rendered = json.dumps(side)
            self.assertNotIn("first-corruption", rendered)
            self.assertNotIn("second-corruption", rendered)


class DiffRecoveryAuditTests(RecoveryAuditTestBase):
    def names(self, changes: tuple) -> list[tuple[str, str | None, str]]:
        return [
            (category, generation, change)
            for category, generation, change, _before, _after in changes
        ]

    def test_no_change_is_empty_tuple(self) -> None:
        self.rotate_three()
        record = BranchStore.export_recovery_audit(self.root)
        changes = BranchStore.diff_recovery_audit(self.root, record)
        self.assertEqual(changes, ())
        self.assertIsInstance(changes, tuple)

    def test_result_entries_are_five_tuples_with_immutable_values(self) -> None:
        self.rotate_three()
        record = BranchStore.export_recovery_audit(self.root)
        with open(os.path.join(self.root, "CURRENT"), "wb") as handle:
            handle.write(b"generation-0000000000000001")
        changes = BranchStore.diff_recovery_audit(self.root, record)
        self.assertEqual(len(changes), 1)
        category, generation, change, before, after = changes[0]
        self.assertEqual(
            (category, generation, change),
            ("pointer", None, "changed"),
        )
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
            {
                "state": "ok",
                "value": "generation-0000000000000001",
                "digest": None,
            },
        )

    def test_corrupted_checkpoint_classifies_every_category(self) -> None:
        self.rotate_three()
        record = BranchStore.export_recovery_audit(self.root)
        with open(
            os.path.join(self.generation_dir(2), "checkpoint.json"), "ab"
        ) as handle:
            handle.write(b" ")
        changes = BranchStore.diff_recovery_audit(self.root, record)
        g2 = "generation-0000000000000002"
        g3 = "generation-0000000000000003"
        # Pointer first (unchanged here), then chain, checkpoint,
        # journal, replay; per-generation groups ascend by number.
        self.assertEqual(
            self.names(changes),
            [
                ("chain", None, "changed"),
                ("chain", g2, "changed"),
                ("chain", g3, "changed"),
                ("checkpoint", g2, "changed"),
                ("replay", g2, "changed"),
                ("replay", g3, "changed"),
            ],
        )
        entries = {
            (category, generation): (change, before, after)
            for category, generation, change, before, after in changes
        }
        change, before, after = entries[("chain", None)]
        self.assertEqual(before, g3)
        self.assertEqual(after, "generation-0000000000000001")
        change, before, after = entries[("chain", g2)]
        self.assertTrue(before["valid"])
        self.assertIsNone(before["reason"])
        self.assertFalse(after["valid"])
        self.assertIsNotNone(after["reason"])
        self.assertEqual(before["manifest"], after["manifest"])
        change, before, after = entries[("chain", g3)]
        self.assertEqual(before["dependency"], "linked")
        self.assertEqual(after["dependency"], "invalid")
        change, before, after = entries[("checkpoint", g2)]
        self.assertIsInstance(before, str)
        self.assertIsInstance(after, str)
        self.assertNotEqual(before, after)
        change, before, after = entries[("replay", g2)]
        self.assertEqual(before, "ok")
        self.assertEqual(after, "not-run")

    def test_generation_removal_is_reported_per_category(self) -> None:
        self.rotate_three()
        record = BranchStore.export_recovery_audit(self.root)
        shutil.rmtree(self.generation_dir(3))
        changes = BranchStore.diff_recovery_audit(self.root, record)
        g3 = "generation-0000000000000003"
        self.assertEqual(
            self.names(changes),
            [
                ("chain", None, "changed"),
                ("chain", g3, "removed"),
                ("checkpoint", g3, "removed"),
                ("journal", g3, "removed"),
                ("replay", g3, "removed"),
            ],
        )
        entries = {
            (category, generation): (change, before, after)
            for category, generation, change, before, after in changes
        }
        for category in ("checkpoint", "journal", "replay"):
            change, before, after = entries[(category, g3)]
            self.assertIsNotNone(before)
            self.assertIsNone(after)
        change, before, after = entries[("chain", g3)]
        self.assertTrue(before["valid"])
        self.assertIsNone(after)

    def test_generation_addition_is_reported_per_category(self) -> None:
        store = self.make_store()
        store.rotate_generation(self.root)
        record = BranchStore.export_recovery_audit(self.root)
        store.append("main", "m4", 6, {"a": 4})
        store.rotate_generation(self.root)
        changes = BranchStore.diff_recovery_audit(self.root, record)
        g1 = "generation-0000000000000001"
        g2 = "generation-0000000000000002"
        # The append before rotation commits into gen 1's journal and
        # re-points gen 1's manifest; rotation adds gen 2 and moves the
        # pointer and selection.
        self.assertEqual(
            self.names(changes),
            [
                ("pointer", None, "changed"),
                ("chain", None, "changed"),
                ("chain", g1, "changed"),
                ("chain", g2, "added"),
                ("checkpoint", g2, "added"),
                ("journal", g1, "changed"),
                ("journal", g2, "added"),
                ("replay", g2, "added"),
            ],
        )
        entries = {
            (category, generation): (change, before, after)
            for category, generation, change, before, after in changes
        }
        change, before, after = entries[("chain", g1)]
        self.assertTrue(before["valid"])
        self.assertTrue(after["valid"])
        self.assertNotEqual(before["manifest"], after["manifest"])
        change, before, after = entries[("journal", g1)]
        self.assertNotEqual(before, after)
        for category in ("checkpoint", "journal", "replay"):
            change, before, after = entries[(category, g2)]
            self.assertIsNone(before)
            self.assertIsNotNone(after)

    def test_losing_every_generation_returns_changes_not_error(self) -> None:
        self.rotate_three()
        record = BranchStore.export_recovery_audit(self.root)
        empty = os.path.join(self.tmp.name, "empty")
        os.mkdir(empty)
        changes = BranchStore.diff_recovery_audit(empty, record)
        categories = [entry[0] for entry in changes]
        # The pointer changed too (ok -> missing), selection changed,
        # and all three generations vanished from every category.
        self.assertEqual(categories[0], "pointer")
        self.assertIn(("chain", None, "changed"), self.names(changes))
        for number in (1, 2, 3):
            name = f"generation-{number:016d}"
            for category in ("chain", "checkpoint", "journal", "replay"):
                self.assertIn(
                    (category, name, "removed"), self.names(changes)
                )

    def test_journal_change_reports_journal_and_replay_only(self) -> None:
        self.rotate_three()
        record = BranchStore.export_recovery_audit(self.root)
        # Gen 1 sits behind two still-valid successors; tamper with gen
        # 3's journal, which invalidates the tip.
        with open(
            os.path.join(self.generation_dir(3), "journal.json"), "ab"
        ) as handle:
            handle.write(b" ")
        changes = BranchStore.diff_recovery_audit(self.root, record)
        names = self.names(changes)
        self.assertIn(
            ("journal", "generation-0000000000000003", "changed"), names
        )
        self.assertIn(
            ("replay", "generation-0000000000000003", "changed"), names
        )
        self.assertFalse(
            any(category == "checkpoint" for category, _g, _c in names)
        )

    def test_pointer_sides_never_carry_raw_bytes(self) -> None:
        self.rotate_three()
        with open(os.path.join(self.root, "CURRENT"), "wb") as handle:
            handle.write(b"\xff secret-pointer-alpha")
        record = BranchStore.export_recovery_audit(self.root)
        with open(os.path.join(self.root, "CURRENT"), "wb") as handle:
            handle.write(b"generation-0000000000000002")
        changes = BranchStore.diff_recovery_audit(self.root, record)
        rendered = repr(changes)
        self.assertNotIn("secret-pointer-alpha", rendered)
        pointer = [
            entry for entry in changes if entry[0] == "pointer"
        ]
        self.assertEqual(len(pointer), 1)
        _category, generation, change, before, after = pointer[0]
        self.assertIsNone(generation)
        self.assertEqual(change, "changed")
        self.assertEqual(before["value"], None)
        self.assertEqual(
            before["digest"], _sha256(b"\xff secret-pointer-alpha")
        )
        self.assertEqual(after["value"], "generation-0000000000000002")
        self.assertIsNone(after["digest"])

    def test_diff_is_read_only(self) -> None:
        self.rotate_three()
        record = BranchStore.export_recovery_audit(self.root)
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

    def test_bad_checksum_rejected(self) -> None:
        self.rotate_three()
        document = json.loads(
            BranchStore.export_recovery_audit(self.root)
        )
        document["checksum"] = "0" * 64
        with self.assertRaises(ValueError):
            BranchStore.diff_recovery_audit(
                self.root, _reencode(document)
            )


if __name__ == "__main__":
    unittest.main()
