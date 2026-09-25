import hashlib
import json
import os
import tempfile
import threading
import unittest
from unittest import mock

from city_twin.branches import BranchStore
from city_twin.event_graph import EventGraph


def _read(path: str) -> bytes:
    with open(path, "rb") as handle:
        return handle.read()


def _document(path: str) -> dict:
    return json.loads(_read(path))


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


class RotateGenerationTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = os.path.join(self.tmp.name, "generations")
        os.mkdir(self.root)
        self.cp = os.path.join(self.tmp.name, "cp.json")
        self.journal = os.path.join(self.tmp.name, "journal.json")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def make_journaled_store(self) -> BranchStore:
        store = make_history()
        store.save_checkpoint(self.cp)
        store.enable_journal(self.cp, self.journal)
        store.append("main", "m3", 5, {"a": 3})
        store.create("side", "m2")
        store.append("side", "s1", 6, {"s": 1})
        return store

    def generation_dir(self, number: int) -> str:
        return os.path.join(self.root, f"generation-{number:016d}")

    def manifest(self, number: int) -> dict:
        return _document(
            os.path.join(self.generation_dir(number), "manifest.json")
        )


class RotateGenerationTests(RotateGenerationTestBase):
    def test_returns_name_and_publishes_complete_generation(self) -> None:
        store = self.make_journaled_store()
        name = store.rotate_generation(self.root)
        self.assertEqual(name, "generation-0000000000000001")
        generation = self.generation_dir(1)
        self.assertEqual(
            sorted(os.listdir(generation)),
            ["checkpoint.json", "journal.json", "manifest.json"],
        )
        manifest = self.manifest(1)
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["generation"], 1)
        self.assertIsNone(manifest["previous"])
        self.assertIs(manifest["complete"], True)
        checkpoint_bytes = _read(os.path.join(generation, "checkpoint.json"))
        journal_bytes = _read(os.path.join(generation, "journal.json"))
        self.assertEqual(
            manifest["checkpoint"],
            hashlib.sha256(checkpoint_bytes).hexdigest(),
        )
        self.assertEqual(
            manifest["journal"], hashlib.sha256(journal_bytes).hexdigest()
        )
        journal_doc = json.loads(journal_bytes)
        self.assertEqual(journal_doc["schema_version"], 2)
        self.assertEqual(journal_doc["frames"], [])
        self.assertEqual(
            journal_doc["checkpoint"],
            json.loads(checkpoint_bytes)["checksum"],
        )
        # The CURRENT pointer names the new generation.
        self.assertEqual(
            _read(os.path.join(self.root, "CURRENT")), name.encode("utf-8")
        )

    def test_numbers_increment_and_manifests_chain(self) -> None:
        store = self.make_journaled_store()
        first = store.rotate_generation(self.root)
        store.append("main", "m4", 7, {"a": 4})
        second = store.rotate_generation(self.root)
        self.assertEqual(first, "generation-0000000000000001")
        self.assertEqual(second, "generation-0000000000000002")
        first_manifest_bytes = _read(
            os.path.join(self.generation_dir(1), "manifest.json")
        )
        self.assertEqual(
            self.manifest(2)["previous"],
            hashlib.sha256(first_manifest_bytes).hexdigest(),
        )
        self.assertEqual(
            _read(os.path.join(self.root, "CURRENT")),
            second.encode("utf-8"),
        )

    def test_new_journal_replays_from_one_and_old_bytes_unchanged(
        self,
    ) -> None:
        store = self.make_journaled_store()
        first = store.rotate_generation(self.root)
        first_files = {
            name: _read(os.path.join(self.generation_dir(1), name))
            for name in ("checkpoint.json", "journal.json", "manifest.json")
        }
        store.append("main", "m4", 7, {"a": 4})
        journal_doc = _document(
            os.path.join(self.generation_dir(1), "journal.json")
        )
        self.assertEqual([frame["seq"] for frame in journal_doc["frames"]], [1])
        self.assertEqual(
            journal_doc["frames"][0]["prev"], journal_doc["checkpoint"]
        )
        second = store.rotate_generation(self.root)
        store.append("main", "m5", 8, {"a": 5})
        # The first generation's checkpoint and manifest never change.
        for name, content in first_files.items():
            if name == "journal.json":
                continue
            self.assertEqual(
                _read(os.path.join(self.generation_dir(1), name)), content
            )
        # The second generation recovers the state at its own rotation
        # instant plus everything committed to its journal since.
        restored, report = BranchStore.load_latest_generation(self.root)
        self.assertEqual(report["selected"], second)
        self.assertEqual(restored.head("main"), "m5")
        self.assertEqual(restored.audit_log(), store.audit_log())
        self.assertEqual(first, "generation-0000000000000001")

    def test_recovered_store_continues_committing(self) -> None:
        store = self.make_journaled_store()
        name = store.rotate_generation(self.root)
        store.append("main", "m4", 7, {"a": 4})
        restored, report = BranchStore.load_latest_generation(self.root)
        self.assertEqual(report["selected"], name)
        self.assertEqual(report["current"], name)
        self.assertEqual(report["ignored"], [])
        self.assertEqual(restored.head("main"), "m4")
        # The recovered store is attached to the selected journal.
        restored.append("main", "m5", 8, {"a": 5})
        journal_doc = _document(
            os.path.join(self.generation_dir(1), "journal.json")
        )
        self.assertEqual(
            [frame["seq"] for frame in journal_doc["frames"]], [1, 2]
        )
        self.assertEqual(restored.head("main"), "m5")
        # The pair also recovers through the plain journal entry point.
        via_pair = BranchStore.load_journal(
            os.path.join(self.generation_dir(1), "checkpoint.json"),
            os.path.join(self.generation_dir(1), "journal.json"),
            None,
        )
        self.assertEqual(via_pair.head("main"), "m5")

    def test_same_history_produces_identical_generation_bytes(self) -> None:
        first = self.make_journaled_store()
        first.rotate_generation(self.root)

        other_root = os.path.join(self.tmp.name, "other")
        os.mkdir(other_root)
        second_cp = os.path.join(self.tmp.name, "cp-b.json")
        second_journal = os.path.join(self.tmp.name, "journal-b.json")
        second = make_history()
        second.save_checkpoint(second_cp)
        second.enable_journal(second_cp, second_journal)
        second.append("main", "m3", 5, {"a": 3})
        second.create("side", "m2")
        second.append("side", "s1", 6, {"s": 1})
        second.rotate_generation(other_root)

        for name in ("checkpoint.json", "journal.json", "manifest.json"):
            self.assertEqual(
                _read(os.path.join(self.generation_dir(1), name)),
                _read(
                    os.path.join(
                        other_root, "generation-0000000000000001", name
                    )
                ),
            )


class RotateGenerationValidationTests(RotateGenerationTestBase):
    def test_root_validation(self) -> None:
        store = self.make_journaled_store()
        for bad in (1, 1.0, b"x", None, ["a"]):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    store.rotate_generation(bad)
                with self.assertRaises(TypeError):
                    BranchStore.load_latest_generation(bad)
        with self.assertRaises(ValueError):
            store.rotate_generation("")
        with self.assertRaises(ValueError):
            BranchStore.load_latest_generation("")

    def test_missing_or_non_directory_root_raises_os_error(self) -> None:
        store = self.make_journaled_store()
        missing = os.path.join(self.tmp.name, "no-such-dir")
        with self.assertRaises(OSError):
            store.rotate_generation(missing)
        with self.assertRaises(OSError):
            BranchStore.load_latest_generation(missing)
        file_root = os.path.join(self.tmp.name, "a-file")
        with open(file_root, "wb") as handle:
            handle.write(b"not a directory")
        with self.assertRaises(OSError):
            store.rotate_generation(file_root)
        with self.assertRaises(OSError):
            BranchStore.load_latest_generation(file_root)

    def test_unattached_store_raises_runtime_error(self) -> None:
        store = make_history()
        store.save_checkpoint(self.cp)
        with self.assertRaises(RuntimeError):
            store.rotate_generation(self.root)
        store.enable_journal(self.cp, self.journal)
        store.append("main", "m3", 5, {"a": 3})
        detached = BranchStore.load_journal(self.cp, self.journal, 0)
        with self.assertRaises(RuntimeError):
            detached.rotate_generation(self.root)
        self.assertEqual(os.listdir(self.root), [])

    def test_half_published_directory_is_never_overwritten(self) -> None:
        store = self.make_journaled_store()
        # Forensic material from an interrupted first rotation: no
        # manifest, so it is not a published generation, but the name
        # is occupied and the next rotation must not clobber it.
        leftover = self.generation_dir(1)
        os.mkdir(leftover)
        with open(os.path.join(leftover, "checkpoint.json"), "wb") as handle:
            handle.write(b"forensic")
        with self.assertRaises(OSError):
            store.rotate_generation(self.root)
        self.assertEqual(
            _read(os.path.join(leftover, "checkpoint.json")), b"forensic"
        )
        self.assertFalse(os.path.exists(os.path.join(self.root, "CURRENT")))
        # The store is still attached to its old journal.
        store.append("main", "m4", 7, {"a": 4})
        self.assertEqual(len(_document(self.journal)["frames"]), 4)

    def test_sync_failure_leaves_no_generation_and_old_attachment(
        self,
    ) -> None:
        store = self.make_journaled_store()
        journal_before = _read(self.journal)
        with mock.patch(
            "city_twin.branches.os.fsync", side_effect=OSError("boom")
        ):
            with self.assertRaises(OSError):
                store.rotate_generation(self.root)
        self.assertEqual(os.listdir(self.root), [])
        self.assertEqual(_read(self.journal), journal_before)
        store.append("main", "m4", 7, {"a": 4})
        self.assertEqual(len(_document(self.journal)["frames"]), 4)
        # A clean retry afterwards succeeds.
        name = store.rotate_generation(self.root)
        self.assertEqual(name, "generation-0000000000000001")

    def test_current_update_failure_keeps_old_attachment(self) -> None:
        store = self.make_journaled_store()
        journal_before = _read(self.journal)
        real_replace = os.replace

        def flaky(src, dst, *args, **kwargs):
            if os.path.basename(dst) == "CURRENT":
                raise OSError("boom")
            return real_replace(src, dst, *args, **kwargs)

        with mock.patch(
            "city_twin.branches.os.replace", side_effect=flaky
        ):
            with self.assertRaises(OSError):
                store.rotate_generation(self.root)
        # The store keeps writing the old journal.
        store.append("main", "m4", 7, {"a": 4})
        self.assertEqual(len(_document(self.journal)["frames"]), 4)
        # The fully published generation stays behind as valid evidence
        # and is selected even without a CURRENT pointer.
        self.assertFalse(os.path.exists(os.path.join(self.root, "CURRENT")))
        restored, report = BranchStore.load_latest_generation(self.root)
        self.assertEqual(report["selected"], "generation-0000000000000001")
        self.assertIsNone(report["current"])
        self.assertEqual(restored.head("side"), "s1")


class LoadLatestGenerationTests(RotateGenerationTestBase):
    def rotate_three(self) -> BranchStore:
        store = self.make_journaled_store()
        store.rotate_generation(self.root)
        store.append("main", "m4", 7, {"a": 4})
        store.rotate_generation(self.root)
        store.append("main", "m5", 8, {"a": 5})
        store.rotate_generation(self.root)
        return store

    def test_empty_root_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            BranchStore.load_latest_generation(self.root)

    def test_only_invalid_generations_raise_value_error(self) -> None:
        self.make_journaled_store()
        # A half-published generation alone is not enough.
        os.mkdir(self.generation_dir(1))
        with self.assertRaises(ValueError):
            BranchStore.load_latest_generation(self.root)

    def test_report_key_order(self) -> None:
        store = self.make_journaled_store()
        store.rotate_generation(self.root)
        _restored, report = BranchStore.load_latest_generation(self.root)
        self.assertEqual(list(report), ["selected", "current", "ignored"])

    def test_missing_current_pointer_falls_back_to_evidence(self) -> None:
        store = self.make_journaled_store()
        name = store.rotate_generation(self.root)
        os.remove(os.path.join(self.root, "CURRENT"))
        restored, report = BranchStore.load_latest_generation(self.root)
        self.assertEqual(report["selected"], name)
        self.assertIsNone(report["current"])
        self.assertEqual(restored.head("side"), "s1")

    def test_stale_current_pointer_falls_back_to_evidence(self) -> None:
        store = self.rotate_three()
        # Point CURRENT at a generation that does not exist.
        with open(os.path.join(self.root, "CURRENT"), "wb") as handle:
            handle.write(b"generation-0000000000000009")
        restored, report = BranchStore.load_latest_generation(self.root)
        self.assertEqual(report["selected"], "generation-0000000000000003")
        self.assertEqual(report["current"], "generation-0000000000000009")
        self.assertEqual(restored.head("main"), store.head("main"))

    def test_corrupt_current_pointer_reports_none(self) -> None:
        store = self.make_journaled_store()
        name = store.rotate_generation(self.root)
        with open(os.path.join(self.root, "CURRENT"), "wb") as handle:
            handle.write(b"\xff\xfe not a generation")
        restored, report = BranchStore.load_latest_generation(self.root)
        self.assertEqual(report["selected"], name)
        self.assertIsNone(report["current"])
        self.assertEqual(restored.head("side"), "s1")

    def test_corrupt_latest_generation_falls_back_to_previous(self) -> None:
        store = self.rotate_three()
        expected_audit = store.audit_log()
        # Corrupt the newest generation's checkpoint; its manifest
        # digest no longer matches the file.
        checkpoint_path = os.path.join(
            self.generation_dir(3), "checkpoint.json"
        )
        with open(checkpoint_path, "wb") as handle:
            handle.write(b"corrupted")
        restored, report = BranchStore.load_latest_generation(self.root)
        self.assertEqual(report["selected"], "generation-0000000000000002")
        self.assertEqual(report["current"], "generation-0000000000000003")
        self.assertEqual(
            [entry["name"] for entry in report["ignored"]],
            ["generation-0000000000000003"],
        )
        self.assertEqual(
            report["ignored"][0]["reason"],
            "checkpoint digest does not match the manifest",
        )
        # Generation 2's journal had already received m5, so the
        # fallback recovers the full history.
        self.assertEqual(restored.head("main"), "m5")
        self.assertEqual(restored.audit_log(), expected_audit)

    def test_broken_chain_invalidates_later_generations(self) -> None:
        self.rotate_three()
        # Rewrite generation 2's manifest with a tampered chain link;
        # its own digest changes, so generation 3's chain breaks too.
        manifest_path = os.path.join(
            self.generation_dir(2), "manifest.json"
        )
        manifest = _document(manifest_path)
        manifest["previous"] = "0" * 64
        with open(manifest_path, "wb") as handle:
            handle.write(
                json.dumps(manifest, separators=(",", ":")).encode("utf-8")
            )
        restored, report = BranchStore.load_latest_generation(self.root)
        self.assertEqual(report["selected"], "generation-0000000000000001")
        self.assertEqual(
            [entry["name"] for entry in report["ignored"]],
            ["generation-0000000000000002", "generation-0000000000000003"],
        )
        self.assertEqual(
            [entry["reason"] for entry in report["ignored"]],
            ["digest chain is broken", "digest chain is broken"],
        )
        self.assertEqual(restored.head("main"), "m4")

    def test_half_published_latest_generation_is_ignored(self) -> None:
        self.rotate_three()
        # A crashed fourth rotation: directory and checkpoint, no
        # manifest.
        leftover = self.generation_dir(4)
        os.mkdir(leftover)
        with open(os.path.join(leftover, "checkpoint.json"), "wb") as handle:
            handle.write(b"half-written")
        restored, report = BranchStore.load_latest_generation(self.root)
        self.assertEqual(report["selected"], "generation-0000000000000003")
        self.assertEqual(
            report["ignored"],
            [
                {
                    "name": "generation-0000000000000004",
                    "reason": "missing manifest",
                }
            ],
        )
        self.assertEqual(restored.head("main"), "m5")

    def test_missing_data_file_marks_generation_invalid(self) -> None:
        self.rotate_three()
        os.remove(os.path.join(self.generation_dir(3), "journal.json"))
        restored, report = BranchStore.load_latest_generation(self.root)
        self.assertEqual(report["selected"], "generation-0000000000000002")
        self.assertEqual(
            report["ignored"],
            [
                {
                    "name": "generation-0000000000000003",
                    "reason": "missing journal",
                }
            ],
        )
        self.assertEqual(restored.head("main"), "m5")

    def test_unparseable_content_marks_generation_invalid(self) -> None:
        self.rotate_three()
        # Rewrite generation 3's journal with garbage and fix the
        # manifest digest to match, so only the content parse can fail.
        journal_path = os.path.join(self.generation_dir(3), "journal.json")
        with open(journal_path, "wb") as handle:
            handle.write(b"not json at all")
        manifest_path = os.path.join(
            self.generation_dir(3), "manifest.json"
        )
        manifest = _document(manifest_path)
        manifest["journal"] = hashlib.sha256(b"not json at all").hexdigest()
        with open(manifest_path, "wb") as handle:
            handle.write(
                json.dumps(manifest, separators=(",", ":")).encode("utf-8")
            )
        restored, report = BranchStore.load_latest_generation(self.root)
        self.assertEqual(report["selected"], "generation-0000000000000002")
        self.assertEqual(
            report["ignored"],
            [
                {
                    "name": "generation-0000000000000003",
                    "reason": "journal is not parseable",
                }
            ],
        )
        self.assertEqual(restored.head("main"), "m5")

    def test_load_is_read_only(self) -> None:
        self.rotate_three()
        before = {
            name: _read(os.path.join(dirpath, name))
            for dirpath, _dirs, files in os.walk(self.root)
            for name in files
        }
        before[os.path.join(self.root, "CURRENT")] = _read(
            os.path.join(self.root, "CURRENT")
        )
        BranchStore.load_latest_generation(self.root)
        after = {
            name: _read(os.path.join(dirpath, name))
            for dirpath, _dirs, files in os.walk(self.root)
            for name in files
        }
        after[os.path.join(self.root, "CURRENT")] = _read(
            os.path.join(self.root, "CURRENT")
        )
        self.assertEqual(before, after)

    def test_non_generation_entries_are_ignored(self) -> None:
        store = self.make_journaled_store()
        name = store.rotate_generation(self.root)
        # Stray files and lookalike names play no role.
        with open(os.path.join(self.root, "notes.txt"), "wb") as handle:
            handle.write(b"hello")
        os.mkdir(os.path.join(self.root, "generation-1"))
        os.mkdir(os.path.join(self.root, "generation-000000000000000x"))
        with open(
            os.path.join(self.root, "generation-0000000000000099"), "wb"
        ) as handle:
            handle.write(b"a file, not a directory")
        restored, report = BranchStore.load_latest_generation(self.root)
        self.assertEqual(report["selected"], name)
        self.assertEqual(report["ignored"], [])
        self.assertEqual(restored.head("side"), "s1")


class RotateGenerationConcurrencyTests(RotateGenerationTestBase):
    def test_rotation_blocks_concurrent_commits(self) -> None:
        store = self.make_journaled_store()

        inside = threading.Event()
        proceed = threading.Event()
        real_fsync = os.fsync
        calls = {"n": 0}

        def barrier(fd) -> None:
            calls["n"] += 1
            if calls["n"] == 1:
                inside.set()
                proceed.wait(timeout=5)
            return real_fsync(fd)

        done = threading.Event()

        def do_rotate() -> None:
            store.rotate_generation(self.root)
            done.set()

        with mock.patch(
            "city_twin.branches.os.fsync", side_effect=barrier
        ):
            rotator = threading.Thread(target=do_rotate)
            rotator.start()
            self.assertTrue(inside.wait(timeout=5))

            appended = threading.Event()

            def do_append() -> None:
                store.append("main", "late", 90, {"a": 1})
                appended.set()

            committer = threading.Thread(target=do_append)
            committer.start()
            self.assertFalse(appended.wait(timeout=0.3))
            self.assertFalse(done.is_set())

            proceed.set()
            rotator.join(timeout=5)
            committer.join(timeout=5)

        self.assertTrue(done.is_set())
        self.assertTrue(appended.is_set())
        # The queued commit landed on the new generation's journal,
        # starting at sequence one.
        journal_doc = _document(
            os.path.join(self.generation_dir(1), "journal.json")
        )
        self.assertEqual(
            [frame["seq"] for frame in journal_doc["frames"]], [1]
        )
        restored, report = BranchStore.load_latest_generation(self.root)
        self.assertEqual(report["selected"], "generation-0000000000000001")
        self.assertEqual(restored.head("main"), "late")


if __name__ == "__main__":
    unittest.main()
