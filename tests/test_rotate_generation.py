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


class GenerationTestBase(unittest.TestCase):
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

    def files_of(self, number: int) -> dict:
        directory = self.generation_dir(number)
        return {
            name: _read(os.path.join(directory, name))
            for name in ("checkpoint.json", "journal.json", "manifest.json")
        }


class RotateGenerationTests(GenerationTestBase):
    def test_first_rotation_layout_and_return(self) -> None:
        store = self.make_store()
        name = store.rotate_generation(self.root)
        self.assertEqual(name, "generation-0000000000000001")
        directory = self.generation_dir(1)
        self.assertEqual(
            sorted(os.listdir(directory)),
            ["checkpoint.json", "journal.json", "manifest.json"],
        )
        self.assertEqual(
            _read(os.path.join(self.root, "CURRENT")), name.encode("utf-8")
        )
        checkpoint = _document(os.path.join(directory, "checkpoint.json"))
        journal = _document(os.path.join(directory, "journal.json"))
        manifest = _document(os.path.join(directory, "manifest.json"))
        self.assertEqual(checkpoint["schema_version"], 1)
        self.assertEqual(journal["schema_version"], 2)
        self.assertEqual(journal["frames"], [])
        self.assertEqual(journal["checkpoint"], checkpoint["checksum"])
        self.assertEqual(
            manifest,
            {
                "schema_version": 1,
                "generation": name,
                "number": 1,
                "previous": None,
                "checkpoint": _sha256(
                    _read(os.path.join(directory, "checkpoint.json"))
                ),
                "journal": _sha256(
                    _read(os.path.join(directory, "journal.json"))
                ),
                "complete": True,
            },
        )

    def test_numbers_increment_and_manifests_chain(self) -> None:
        store = self.make_store()
        first = store.rotate_generation(self.root)
        store.append("main", "m4", 6, {"a": 4})
        second = store.rotate_generation(self.root)
        self.assertEqual(second, "generation-0000000000000002")
        manifest1 = self.files_of(1)["manifest.json"]
        manifest2 = _document(
            os.path.join(self.generation_dir(2), "manifest.json")
        )
        self.assertEqual(manifest2["previous"], _sha256(manifest1))
        self.assertEqual(manifest2["number"], 2)
        self.assertEqual(manifest2["generation"], second)
        self.assertEqual(
            _read(os.path.join(self.root, "CURRENT")),
            second.encode("utf-8"),
        )
        self.assertEqual(first, "generation-0000000000000001")

    def test_new_journal_continues_from_sequence_one(self) -> None:
        store = self.make_store()
        journal_frames_before = len(_document(self.journal)["frames"])
        name = store.rotate_generation(self.root)
        store.append("main", "m4", 6, {"a": 4})
        store.create("side", "m2")
        journal = _document(
            os.path.join(self.generation_dir(1), "journal.json")
        )
        self.assertEqual([f["seq"] for f in journal["frames"]], [1, 2])
        self.assertEqual(journal["frames"][0]["prev"], journal["checkpoint"])
        # The pre-rotation journal receives no further frames.
        self.assertEqual(
            len(_document(self.journal)["frames"]), journal_frames_before
        )
        # The manifest follows the growing journal.
        manifest = _document(
            os.path.join(self.generation_dir(1), "manifest.json")
        )
        self.assertEqual(
            manifest["journal"],
            _sha256(
                _read(os.path.join(self.generation_dir(1), "journal.json"))
            ),
        )
        self.assertEqual(name, "generation-0000000000000001")

    def test_old_generations_stay_byte_for_byte(self) -> None:
        store = self.make_store()
        store.rotate_generation(self.root)
        store.append("main", "m4", 6, {"a": 4})
        store.rotate_generation(self.root)
        frozen = self.files_of(1)
        pointer = _read(os.path.join(self.root, "CURRENT"))
        store.append("main", "m5", 7, {"a": 5})
        store.rotate_generation(self.root)
        self.assertEqual(self.files_of(1), frozen)
        # The second generation is untouched by the third rotation too.
        frozen2 = self.files_of(2)
        store.append("main", "m6", 8, {"a": 6})
        self.assertEqual(self.files_of(2), frozen2)
        self.assertNotEqual(
            _read(os.path.join(self.root, "CURRENT")), pointer
        )

    def test_recovery_matches_rotation_instant(self) -> None:
        store = self.make_store()
        store.rotate_generation(self.root)
        recovered, report = BranchStore.load_latest_generation(self.root)
        self.assertEqual(recovered.audit_log(), store.audit_log())
        for name in ("main", "feature"):
            self.assertEqual(recovered.replay(name), store.replay(name))
            self.assertEqual(recovered.head(name), store.head(name))
        self.assertEqual(
            recovered.audit_merge("M1"), store.audit_merge("M1")
        )

    def test_rotation_without_prior_journal(self) -> None:
        store = make_history()
        name = store.rotate_generation(self.root)
        self.assertEqual(name, "generation-0000000000000001")
        store.append("main", "m4", 6, {"a": 4})
        journal = _document(
            os.path.join(self.generation_dir(1), "journal.json")
        )
        self.assertEqual([f["seq"] for f in journal["frames"]], [1])

    def test_same_history_produces_identical_bytes(self) -> None:
        first = self.make_store()
        first.rotate_generation(self.root)
        other_root = os.path.join(self.tmp.name, "other")
        os.mkdir(other_root)
        second = make_history()
        second.save_checkpoint(os.path.join(self.tmp.name, "cp2.json"))
        second.enable_journal(
            os.path.join(self.tmp.name, "cp2.json"),
            os.path.join(self.tmp.name, "journal2.json"),
        )
        second.append("main", "m3", 5, {"a": 3})
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


class RotateGenerationValidationTests(GenerationTestBase):
    def test_path_validation(self) -> None:
        store = self.make_store()
        for bad in (1, 1.0, b"x", None, ["a"]):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    store.rotate_generation(bad)
        with self.assertRaises(ValueError):
            store.rotate_generation("")

    def test_missing_root_raises_os_error(self) -> None:
        store = self.make_store()
        with self.assertRaises(OSError):
            store.rotate_generation(os.path.join(self.tmp.name, "nope"))

    def test_file_root_raises_os_error(self) -> None:
        store = self.make_store()
        with self.assertRaises(OSError):
            store.rotate_generation(self.cp)

    @unittest.skipIf(
        hasattr(os, "geteuid") and os.geteuid() == 0,
        "root ignores permission bits",
    )
    def test_unwritable_root_raises_os_error(self) -> None:
        store = self.make_store()
        os.chmod(self.root, 0o555)
        try:
            with self.assertRaises(OSError):
                store.rotate_generation(self.root)
        finally:
            os.chmod(self.root, 0o755)

    def test_occupied_generation_target_raises_and_preserves(self) -> None:
        store = self.make_store()
        # Forensic material from an interrupted attempt occupies the
        # next number; it must never be overwritten.
        leftover = self.generation_dir(1)
        os.mkdir(leftover)
        marker = os.path.join(leftover, "notes.txt")
        with open(marker, "wb") as handle:
            handle.write(b"forensics")
        with self.assertRaises(OSError):
            store.rotate_generation(self.root)
        self.assertEqual(_read(marker), b"forensics")
        self.assertFalse(
            os.path.exists(os.path.join(self.root, "CURRENT"))
        )
        # The store keeps its previous attachment.
        store.append("main", "m4", 6, {"a": 4})
        self.assertEqual(len(_document(self.journal)["frames"]), 2)

    def test_half_published_generation_does_not_skip_numbers(self) -> None:
        store = self.make_store()
        # A half-published generation-2 (no manifest) does not count
        # towards the numbering; the next rotation wants number 2 and
        # refuses to overwrite the leftover.
        store.rotate_generation(self.root)
        os.mkdir(self.generation_dir(2))
        with self.assertRaises(OSError):
            store.rotate_generation(self.root)
        self.assertEqual(
            _read(os.path.join(self.root, "CURRENT")),
            b"generation-0000000000000001",
        )

    def test_publish_failure_cleans_up_and_keeps_attachment(self) -> None:
        store = self.make_store()
        journal_before = _read(self.journal)
        audit_before = store.audit_log()
        real_link = os.link

        def flaky(src, dst, *args, **kwargs):
            if dst.endswith("journal.json"):
                raise OSError("boom")
            return real_link(src, dst, *args, **kwargs)

        with mock.patch(
            "city_twin.branches.os.link", side_effect=flaky
        ):
            with self.assertRaises(OSError):
                store.rotate_generation(self.root)
        self.assertEqual(os.listdir(self.root), [])
        self.assertEqual(_read(self.journal), journal_before)
        self.assertEqual(store.audit_log(), audit_before)
        store.append("main", "m4", 6, {"a": 4})
        self.assertEqual(len(_document(self.journal)["frames"]), 2)
        # A clean retry succeeds and reuses the number.
        self.assertEqual(
            store.rotate_generation(self.root),
            "generation-0000000000000001",
        )

    def test_pointer_lands_last(self) -> None:
        store = self.make_store()
        real_fsync = os.fsync
        calls = {"n": 0}

        def flaky(fd):
            calls["n"] += 1
            if calls["n"] == 2:
                raise OSError("boom")
            return real_fsync(fd)

        with mock.patch(
            "city_twin.branches.os.fsync", side_effect=flaky
        ):
            with self.assertRaises(OSError):
                store.rotate_generation(self.root)
        self.assertFalse(
            os.path.exists(os.path.join(self.root, "CURRENT"))
        )
        self.assertEqual(os.listdir(self.root), [])


class RotateGenerationConcurrencyTests(GenerationTestBase):
    def test_rotation_blocks_concurrent_commits(self) -> None:
        store = self.make_store()
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
        journal = _document(
            os.path.join(self.generation_dir(1), "journal.json")
        )
        self.assertEqual([f["seq"] for f in journal["frames"]], [1])
        self.assertEqual(journal["frames"][0]["op"], "append")


class LoadLatestGenerationTests(GenerationTestBase):
    def rotate_three(self) -> BranchStore:
        store = self.make_store()
        store.rotate_generation(self.root)
        store.append("main", "m4", 6, {"a": 4})
        store.rotate_generation(self.root)
        store.append("main", "m5", 7, {"a": 5})
        store.rotate_generation(self.root)
        return store

    def test_roundtrip_and_report_shape(self) -> None:
        store = self.rotate_three()
        recovered, report = BranchStore.load_latest_generation(self.root)
        self.assertEqual(
            list(report), ["selected", "current", "ignored"]
        )
        self.assertEqual(report["selected"], "generation-0000000000000003")
        self.assertEqual(report["current"], "generation-0000000000000003")
        self.assertEqual(report["ignored"], [])
        self.assertEqual(recovered.audit_log(), store.audit_log())
        self.assertEqual(recovered.head("main"), store.head("main"))

    def test_recovered_store_continues_committing(self) -> None:
        self.rotate_three()
        recovered, _ = BranchStore.load_latest_generation(self.root)
        recovered.append("main", "m6", 8, {"a": 6})
        journal_path = os.path.join(
            self.generation_dir(3), "journal.json"
        )
        self.assertEqual(
            [f["seq"] for f in _document(journal_path)["frames"]], [1]
        )
        # The manifest tracks the commit, so a later load recovers it.
        manifest = _document(os.path.join(self.generation_dir(3),
                                          "manifest.json"))
        self.assertEqual(manifest["journal"], _sha256(_read(journal_path)))
        again, report = BranchStore.load_latest_generation(self.root)
        self.assertEqual(report["selected"], "generation-0000000000000003")
        self.assertEqual(again.head("main"), "m6")
        self.assertEqual(again.audit_log(), recovered.audit_log())

    def test_load_is_read_only(self) -> None:
        self.rotate_three()
        before = {}
        for dirpath, _dirnames, filenames in os.walk(self.root):
            for filename in filenames:
                path = os.path.join(dirpath, filename)
                before[path] = _read(path)
        BranchStore.load_latest_generation(self.root)
        after = {}
        for dirpath, _dirnames, filenames in os.walk(self.root):
            for filename in filenames:
                path = os.path.join(dirpath, filename)
                after[path] = _read(path)
        self.assertEqual(before, after)

    def test_missing_current_pointer_falls_back_to_evidence(self) -> None:
        self.rotate_three()
        os.remove(os.path.join(self.root, "CURRENT"))
        recovered, report = BranchStore.load_latest_generation(self.root)
        self.assertEqual(report["selected"], "generation-0000000000000003")
        self.assertIsNone(report["current"])
        self.assertEqual(recovered.head("main"), "m5")

    def test_stale_current_pointer_falls_back_to_evidence(self) -> None:
        self.rotate_three()
        with open(os.path.join(self.root, "CURRENT"), "wb") as handle:
            handle.write(b"generation-0000000000000099")
        recovered, report = BranchStore.load_latest_generation(self.root)
        self.assertEqual(report["selected"], "generation-0000000000000003")
        self.assertEqual(report["current"], "generation-0000000000000099")
        self.assertEqual(recovered.head("main"), "m5")

    def test_corrupt_current_pointer_falls_back_to_evidence(self) -> None:
        self.rotate_three()
        with open(os.path.join(self.root, "CURRENT"), "wb") as handle:
            handle.write(b"\xff\xfe not a name")
        _recovered, report = BranchStore.load_latest_generation(self.root)
        self.assertEqual(report["selected"], "generation-0000000000000003")
        self.assertIsNone(report["current"])

    def test_broken_chain_invalidates_dependents(self) -> None:
        self.rotate_three()
        # Tamper with generation 2's manifest: its own digest check
        # breaks, and generation 3's chain link no longer matches.
        manifest_path = os.path.join(
            self.generation_dir(2), "manifest.json"
        )
        manifest = _document(manifest_path)
        manifest["checkpoint"] = "0" * 64
        with open(manifest_path, "wb") as handle:
            handle.write(
                json.dumps(manifest, separators=(",", ":")).encode("utf-8")
            )
        recovered, report = BranchStore.load_latest_generation(self.root)
        self.assertEqual(report["selected"], "generation-0000000000000001")
        self.assertEqual(
            [entry["name"] for entry in report["ignored"]],
            ["generation-0000000000000002", "generation-0000000000000003"],
        )
        reasons = [entry["reason"] for entry in report["ignored"]]
        self.assertIn("digest", reasons[0])
        self.assertIn("chain", reasons[1])
        # Generation 1's journal holds the m4 commit that followed it.
        self.assertEqual(recovered.head("main"), "m4")

    def test_corrupt_data_file_invalidates_every_dependent(self) -> None:
        self.rotate_three()
        # Corrupt generation 2's checkpoint bytes. Generation 2 is
        # invalid, and the invalidity propagates by number: generation 3
        # cannot chain over an invalid predecessor even though its own
        # manifest still quotes generation 2's (old) manifest digest, so
        # recovery stops at the last valid generation, number 1.
        checkpoint_path = os.path.join(
            self.generation_dir(2), "checkpoint.json"
        )
        with open(checkpoint_path, "ab") as handle:
            handle.write(b" ")
        recovered, report = BranchStore.load_latest_generation(self.root)
        self.assertEqual(report["selected"], "generation-0000000000000001")
        self.assertEqual(
            [entry["name"] for entry in report["ignored"]],
            ["generation-0000000000000002", "generation-0000000000000003"],
        )
        reasons = [entry["reason"] for entry in report["ignored"]]
        self.assertIn("checkpoint", reasons[0])
        self.assertIn("invalid", reasons[1])
        # Generation 1's journal holds the m4 commit that followed it.
        self.assertEqual(recovered.head("main"), "m4")

    def test_quoted_digest_of_invalid_predecessor_does_not_chain(
        self,
    ) -> None:
        self.rotate_three()
        # Replace generation 2's checkpoint but leave its manifest bytes
        # untouched, so generation 3's manifest still quotes the exact
        # digest of an invalid generation 2 manifest's predecessor chain.
        # Generation 3 must still be refused: quoting a digest never
        # legitimizes a predecessor whose own evidence is broken.
        checkpoint_path = os.path.join(
            self.generation_dir(2), "checkpoint.json"
        )
        with open(checkpoint_path, "wb") as handle:
            handle.write(b"{}")
        recovered, report = BranchStore.load_latest_generation(self.root)
        self.assertEqual(report["selected"], "generation-0000000000000001")
        self.assertEqual(
            [entry["name"] for entry in report["ignored"]],
            ["generation-0000000000000002", "generation-0000000000000003"],
        )
        self.assertEqual(recovered.head("main"), "m4")

    def test_gap_in_numbering_cannot_be_jumped(self) -> None:
        self.rotate_three()
        # Remove generation 2 entirely: generation 3 must not be reached
        # by jumping the gap, even if its digest quote is untouched.
        import shutil

        shutil.rmtree(self.generation_dir(2))
        recovered, report = BranchStore.load_latest_generation(self.root)
        self.assertEqual(report["selected"], "generation-0000000000000001")
        self.assertEqual(
            [entry["name"] for entry in report["ignored"]],
            ["generation-0000000000000003"],
        )
        self.assertIn("chain", report["ignored"][0]["reason"])

    def test_missing_files_and_half_finished_are_invalid(self) -> None:
        self.rotate_three()
        os.remove(os.path.join(self.generation_dir(3), "journal.json"))
        # A half-published generation 4: files but no manifest.
        os.mkdir(self.generation_dir(4))
        for name in ("checkpoint.json", "journal.json"):
            with open(
                os.path.join(self.generation_dir(4), name), "wb"
            ) as handle:
                handle.write(b"{}")
        recovered, report = BranchStore.load_latest_generation(self.root)
        self.assertEqual(report["selected"], "generation-0000000000000002")
        self.assertEqual(
            [entry["name"] for entry in report["ignored"]],
            ["generation-0000000000000003", "generation-0000000000000004"],
        )
        # Generation 2's journal holds the m5 commit that followed it.
        self.assertEqual(recovered.head("main"), "m5")

    def test_unparsable_manifest_is_invalid(self) -> None:
        self.rotate_three()
        manifest_path = os.path.join(
            self.generation_dir(3), "manifest.json"
        )
        with open(manifest_path, "wb") as handle:
            handle.write(b"{not json")
        recovered, report = BranchStore.load_latest_generation(self.root)
        self.assertEqual(report["selected"], "generation-0000000000000002")
        self.assertEqual(
            [entry["name"] for entry in report["ignored"]],
            ["generation-0000000000000003"],
        )
        self.assertEqual(recovered.head("main"), "m5")

    def test_incomplete_marker_is_invalid(self) -> None:
        store = self.make_store()
        store.rotate_generation(self.root)
        manifest_path = os.path.join(
            self.generation_dir(1), "manifest.json"
        )
        manifest = _document(manifest_path)
        manifest["complete"] = False
        with open(manifest_path, "wb") as handle:
            handle.write(
                json.dumps(manifest, separators=(",", ":")).encode("utf-8")
            )
        with self.assertRaises(ValueError):
            BranchStore.load_latest_generation(self.root)

    def test_zero_numbered_forensic_entry_does_not_poison_chain(
        self,
    ) -> None:
        self.rotate_three()
        # A directory matching the pattern but numbered zero is forensic
        # junk: it is reported invalid but never occupies the chain that
        # begins at generation one.
        os.mkdir(
            os.path.join(self.root, "generation-0000000000000000")
        )
        recovered, report = BranchStore.load_latest_generation(self.root)
        self.assertEqual(report["selected"], "generation-0000000000000003")
        self.assertEqual(
            [entry["name"] for entry in report["ignored"]],
            ["generation-0000000000000000"],
        )
        self.assertEqual(recovered.head("main"), "m5")

    def test_no_generations_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            BranchStore.load_latest_generation(self.root)

    def test_all_invalid_raises_value_error(self) -> None:
        store = self.make_store()
        store.rotate_generation(self.root)
        os.remove(os.path.join(self.generation_dir(1), "checkpoint.json"))
        with self.assertRaises(ValueError):
            BranchStore.load_latest_generation(self.root)

    def test_load_validation(self) -> None:
        for bad in (1, 1.0, b"x", None, ["a"]):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    BranchStore.load_latest_generation(bad)
        with self.assertRaises(ValueError):
            BranchStore.load_latest_generation("")
        with self.assertRaises(OSError):
            BranchStore.load_latest_generation(
                os.path.join(self.tmp.name, "nope")
            )
        with self.assertRaises(OSError):
            BranchStore.load_latest_generation(self.cp)

    def test_failed_load_changes_no_state(self) -> None:
        store = self.make_store()
        audit_before = store.audit_log()
        with self.assertRaises(ValueError):
            BranchStore.load_latest_generation(self.root)
        self.assertEqual(store.audit_log(), audit_before)
        store.append("main", "m4", 6, {"a": 4})
        self.assertEqual(store.head("main"), "m4")


if __name__ == "__main__":
    unittest.main()
