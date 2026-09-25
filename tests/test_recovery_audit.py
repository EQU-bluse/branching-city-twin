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

    def rotate_three(self) -> BranchStore:
        store = make_history()
        store.save_checkpoint(self.cp)
        store.enable_journal(self.cp, self.journal)
        store.append("main", "m3", 5, {"a": 3})
        store.rotate_generation(self.root)
        store.append("main", "m4", 6, {"a": 4})
        store.rotate_generation(self.root)
        store.append("main", "m5", 7, {"a": 5})
        store.rotate_generation(self.root)
        return store

    def generation_dir(self, number: int) -> str:
        return os.path.join(self.root, f"generation-{number:016d}")

    def snapshot_tree(self) -> dict:
        before = {}
        for dirpath, _dirnames, filenames in os.walk(self.root):
            for filename in filenames:
                path = os.path.join(dirpath, filename)
                before[path] = _read(path)
        return before


class ExportRecoveryAuditTests(RecoveryAuditTestBase):
    def test_record_shape_and_canonical_form(self) -> None:
        self.rotate_three()
        record = BranchStore.export_recovery_audit(self.root)
        self.assertIsInstance(record, str)
        self.assertEqual(record, record.strip())
        self.assertNotIn(" ", record)
        document = json.loads(record)
        self.assertEqual(
            list(document),
            ["version", "selected", "pointer", "generations", "checksum"],
        )
        self.assertEqual(document["version"], 1)
        self.assertEqual(
            document["selected"], "generation-0000000000000003"
        )
        self.assertEqual(
            document["pointer"],
            {"status": "ok", "name": "generation-0000000000000003"},
        )
        generations = document["generations"]
        self.assertEqual(
            [entry["name"] for entry in generations],
            [
                "generation-0000000000000001",
                "generation-0000000000000002",
                "generation-0000000000000003",
            ],
        )
        self.assertEqual(
            [entry["depends_on"] for entry in generations],
            [
                None,
                "generation-0000000000000001",
                "generation-0000000000000002",
            ],
        )
        for number, entry in enumerate(generations, start=1):
            directory = self.generation_dir(number)
            self.assertEqual(entry["number"], number)
            self.assertTrue(entry["valid"])
            self.assertIsNone(entry["reason"])
            self.assertEqual(entry["replay"], "ok")
            for field, filename in (
                ("manifest", "manifest.json"),
                ("checkpoint", "checkpoint.json"),
                ("journal", "journal.json"),
            ):
                self.assertEqual(
                    entry[field],
                    _sha256(_read(os.path.join(directory, filename))),
                )
        payload = {key: document[key] for key in list(document)[:-1]}
        canonical = json.dumps(
            payload, separators=(",", ":"), ensure_ascii=False
        )
        self.assertEqual(document["checksum"], _sha256(canonical.encode()))

    def test_export_is_deterministic(self) -> None:
        self.rotate_three()
        first = BranchStore.export_recovery_audit(self.root)
        second = BranchStore.export_recovery_audit(self.root)
        self.assertEqual(first, second)

    def test_export_is_read_only(self) -> None:
        self.rotate_three()
        before = self.snapshot_tree()
        BranchStore.export_recovery_audit(self.root)
        self.assertEqual(self.snapshot_tree(), before)

    def test_missing_pointer_records_status(self) -> None:
        self.rotate_three()
        os.remove(os.path.join(self.root, "CURRENT"))
        document = json.loads(BranchStore.export_recovery_audit(self.root))
        self.assertEqual(document["pointer"], {"status": "missing"})
        self.assertEqual(
            document["selected"], "generation-0000000000000003"
        )

    def test_stale_pointer_records_verbatim_name(self) -> None:
        self.rotate_three()
        with open(os.path.join(self.root, "CURRENT"), "wb") as handle:
            handle.write(b"generation-0000000000000099")
        document = json.loads(BranchStore.export_recovery_audit(self.root))
        self.assertEqual(
            document["pointer"],
            {"status": "ok", "name": "generation-0000000000000099"},
        )

    def test_corrupt_pointer_records_only_digest(self) -> None:
        self.rotate_three()
        raw = b"\xff\xfe not a name"
        with open(os.path.join(self.root, "CURRENT"), "wb") as handle:
            handle.write(raw)
        record = BranchStore.export_recovery_audit(self.root)
        document = json.loads(record)
        self.assertEqual(
            document["pointer"],
            {"status": "corrupt", "sha256": _sha256(raw)},
        )
        # The raw pointer content is never echoed into the record.
        self.assertNotIn("not a name", record)

    def test_whitespace_padded_pointer_is_corrupt(self) -> None:
        self.rotate_three()
        raw = b"generation-0000000000000003\n"
        with open(os.path.join(self.root, "CURRENT"), "wb") as handle:
            handle.write(raw)
        document = json.loads(BranchStore.export_recovery_audit(self.root))
        self.assertEqual(
            document["pointer"],
            {"status": "corrupt", "sha256": _sha256(raw)},
        )
        _recovered, report = BranchStore.load_latest_generation(self.root)
        self.assertIsNone(report["current"])

    def test_illegal_pointer_name_is_corrupt(self) -> None:
        self.rotate_three()
        with open(os.path.join(self.root, "CURRENT"), "wb") as handle:
            handle.write(b"generation-1")
        document = json.loads(BranchStore.export_recovery_audit(self.root))
        self.assertEqual(document["pointer"]["status"], "corrupt")
        _recovered, report = BranchStore.load_latest_generation(self.root)
        self.assertIsNone(report["current"])

    def test_invalid_predecessor_propagates_in_record(self) -> None:
        self.rotate_three()
        with open(
            os.path.join(self.generation_dir(2), "checkpoint.json"), "ab"
        ) as handle:
            handle.write(b" ")
        document = json.loads(BranchStore.export_recovery_audit(self.root))
        self.assertEqual(
            document["selected"], "generation-0000000000000001"
        )
        generations = document["generations"]
        self.assertEqual(
            [entry["valid"] for entry in generations],
            [True, False, False],
        )
        self.assertEqual(
            [entry["replay"] for entry in generations],
            ["ok", "failed", "skipped"],
        )
        self.assertIsNone(generations[0]["reason"])
        self.assertIn("digest", generations[1]["reason"])
        self.assertIn("chain", generations[2]["reason"])
        # The corrupt checkpoint's actual digest is recorded as evidence.
        self.assertEqual(
            generations[1]["checkpoint"],
            _sha256(
                _read(
                    os.path.join(
                        self.generation_dir(2), "checkpoint.json"
                    )
                )
            ),
        )

    def test_no_valid_generation_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            BranchStore.export_recovery_audit(self.root)

    def test_all_invalid_raises_value_error(self) -> None:
        self.rotate_three()
        os.remove(os.path.join(self.generation_dir(1), "checkpoint.json"))
        os.remove(os.path.join(self.generation_dir(2), "checkpoint.json"))
        os.remove(os.path.join(self.generation_dir(3), "checkpoint.json"))
        with self.assertRaises(ValueError):
            BranchStore.export_recovery_audit(self.root)

    def test_root_validation(self) -> None:
        for bad in (1, 1.0, b"x", None, ["a"]):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    BranchStore.export_recovery_audit(bad)
        with self.assertRaises(ValueError):
            BranchStore.export_recovery_audit("")
        with self.assertRaises(OSError):
            BranchStore.export_recovery_audit(
                os.path.join(self.tmp.name, "nope")
            )
        with self.assertRaises(OSError):
            BranchStore.export_recovery_audit(self.cp)


class VerifyRecoveryAuditTests(RecoveryAuditTestBase):
    def test_roundtrip_true(self) -> None:
        self.rotate_three()
        record = BranchStore.export_recovery_audit(self.root)
        self.assertTrue(
            BranchStore.verify_recovery_audit(self.root, record)
        )

    def test_verify_is_read_only(self) -> None:
        self.rotate_three()
        record = BranchStore.export_recovery_audit(self.root)
        before = self.snapshot_tree()
        BranchStore.verify_recovery_audit(self.root, record)
        self.assertEqual(self.snapshot_tree(), before)

    def test_changed_checkpoint_returns_false(self) -> None:
        self.rotate_three()
        record = BranchStore.export_recovery_audit(self.root)
        with open(
            os.path.join(self.generation_dir(3), "checkpoint.json"), "ab"
        ) as handle:
            handle.write(b" ")
        self.assertFalse(
            BranchStore.verify_recovery_audit(self.root, record)
        )

    def test_changed_pointer_returns_false(self) -> None:
        self.rotate_three()
        record = BranchStore.export_recovery_audit(self.root)
        os.remove(os.path.join(self.root, "CURRENT"))
        self.assertFalse(
            BranchStore.verify_recovery_audit(self.root, record)
        )

    def test_removed_generation_returns_false(self) -> None:
        self.rotate_three()
        record = BranchStore.export_recovery_audit(self.root)
        os.remove(os.path.join(self.generation_dir(2), "manifest.json"))
        self.assertFalse(
            BranchStore.verify_recovery_audit(self.root, record)
        )

    def test_newer_rotation_returns_false(self) -> None:
        store = self.rotate_three()
        record = BranchStore.export_recovery_audit(self.root)
        store.append("main", "m6", 8, {"a": 6})
        store.rotate_generation(self.root)
        self.assertFalse(
            BranchStore.verify_recovery_audit(self.root, record)
        )

    def test_no_recoverable_generation_returns_false(self) -> None:
        self.rotate_three()
        record = BranchStore.export_recovery_audit(self.root)
        for number in (1, 2, 3):
            os.remove(
                os.path.join(self.generation_dir(number), "checkpoint.json")
            )
        self.assertFalse(
            BranchStore.verify_recovery_audit(self.root, record)
        )

    def test_record_type_validation(self) -> None:
        self.rotate_three()
        for bad in (1, 1.0, b"x", None, ["a"], {"a": 1}):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    BranchStore.verify_recovery_audit(self.root, bad)

    def test_root_validated_before_record(self) -> None:
        for bad in (1, None):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    BranchStore.verify_recovery_audit(bad, "")
        with self.assertRaises(ValueError):
            BranchStore.verify_recovery_audit("", "not json")
        with self.assertRaises(OSError):
            BranchStore.verify_recovery_audit(
                os.path.join(self.tmp.name, "nope"), "not json"
            )

    def test_malformed_records_raise_value_error(self) -> None:
        self.rotate_three()
        record = BranchStore.export_recovery_audit(self.root)
        document = json.loads(record)
        bad_records = {
            "empty": "",
            "not json": "{nope",
            "not an object": json.dumps([1, 2, 3]),
            "duplicate keys": record.replace(
                '"version":1', '"version":1,"version":1', 1
            ),
            "wrong version": json.dumps(
                {**document, "version": 2}, separators=(",", ":")
            ),
            "missing key": json.dumps(
                {
                    key: value
                    for key, value in document.items()
                    if key != "selected"
                },
                separators=(",", ":"),
            ),
            "non-canonical spacing": record.replace(
                '"version":1', '"version": 1', 1
            ),
            "trailing newline": record + "\n",
            "checksum mismatch": record[:-65] + (
                "0" if record[-65] != "0" else "1"
            ) + record[-64:],
        }
        for label, bad in bad_records.items():
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    BranchStore.verify_recovery_audit(self.root, bad)

    def test_field_violations_raise_value_error(self) -> None:
        self.rotate_three()
        record = BranchStore.export_recovery_audit(self.root)
        document = json.loads(record)

        def mutated(**changes) -> str:
            return json.dumps(
                {**document, **changes}, separators=(",", ":")
            )

        bad_records = {
            "selected not a name": mutated(selected="generation-1"),
            "selected not newest valid": mutated(
                selected="generation-0000000000000001"
            ),
            "pointer bad status": mutated(pointer={"status": "weird"}),
            "pointer corrupt without digest": mutated(
                pointer={"status": "corrupt"}
            ),
            "generations not a list": mutated(generations={}),
        }
        generations = document["generations"]
        bad_entry = dict(generations[0])
        bad_entry["valid"] = False
        bad_records["valid without reason"] = mutated(
            generations=[bad_entry, *generations[1:]]
        )
        bad_entry = dict(generations[0])
        bad_entry["replay"] = "skipped"
        bad_records["replay disagrees with valid"] = mutated(
            generations=[bad_entry, *generations[1:]]
        )
        bad_entry = dict(generations[1])
        bad_entry["depends_on"] = None
        bad_records["depends_on not chained"] = mutated(
            generations=[generations[0], bad_entry, generations[2]]
        )
        for label, bad in bad_records.items():
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    BranchStore.verify_recovery_audit(self.root, bad)

    def test_foreign_directory_record_returns_false(self) -> None:
        self.rotate_three()
        record = BranchStore.export_recovery_audit(self.root)
        other_root = os.path.join(self.tmp.name, "other")
        os.mkdir(other_root)
        store = make_history()
        store.save_checkpoint(os.path.join(self.tmp.name, "cp2.json"))
        store.enable_journal(
            os.path.join(self.tmp.name, "cp2.json"),
            os.path.join(self.tmp.name, "journal2.json"),
        )
        store.append("main", "m3", 5, {"a": 3})
        store.rotate_generation(other_root)
        # Same history, but only one generation: the record's decision
        # and evidence do not match.
        self.assertFalse(
            BranchStore.verify_recovery_audit(other_root, record)
        )


if __name__ == "__main__":
    unittest.main()
