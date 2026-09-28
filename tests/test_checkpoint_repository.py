"""Tests for the concurrency-safe, durable checkpoint repository.

``CheckpointRepository`` persists one offline proof-audit checkpoint
string plus request evidence in a single UTF-8 compact JSON file and
adds optimistic concurrency (``expected`` digest), request-id
idempotency across restarts, cross-process file locking and an atomic
durable commit around the existing purely-offline
``create_proof_audit_checkpoint``/``resume_proof_audit`` contract.

These tests cover:

* construction and argument validation (``path`` and ``timeout``),
  :meth:`load`'s missing-:class:`KeyError` and read-only nature;
* :meth:`save`/:meth:`resume` argument and content validation, the
  ``expected`` optimistic-concurrency token and the resumed-audit
  equivalence with the one-shot sequence audit;
* idempotency: the same request id with the same inputs replays the
  original result (in threads and across a simulated restart) without a
  second continuation, while different inputs or a different operation
  kind raise :class:`ValueError`;
* concurrency: concurrent commits based on the same old digest have at
  most one winner in threads and in separate processes, and an elapsed
  lock wait raises :class:`TimeoutError`;
* crash safety: failed flush/replace/sync raises :class:`OSError` and
  leaves the old repository complete with no partial result or temp
  residue, interrupted temp files never participate in reads and can
  never mask a corrupt main file;
* repository authentication and evidence consistency;
* terminal sealing through the repository and business-state purity.
"""

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

from city_twin.branches import BranchStore, CheckpointRepository

try:
    from test_diagnostic_ledger import make_chain, transaction
except ImportError:  # running as a package member from the repo root
    from tests.test_diagnostic_ledger import make_chain, transaction


class RepositoryTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.identities = tuple(transaction(i) for i in range(8))
        self.entries = tuple(
            (index + 1, make_chain(transaction(index)))
            for index in range(8)
        )
        self._proofs = {}

    def path(self, name: str = "repository.json") -> str:
        return os.path.join(self.tmp.name, name)

    def repository(self, name: str = "repository.json", timeout=None):
        return CheckpointRepository(self.path(name), timeout)

    def ledger(self, name: str = "ledger") -> str:
        directory = os.path.join(self.tmp.name, name)
        os.mkdir(directory)
        return directory

    def _append(self, directory, entries, expected, keep=10):
        return BranchStore.append_diagnostic_ledger(
            directory, entries, expected, 10, keep, None
        )

    def _export(self, directory, identities=None):
        return BranchStore.export_diagnostic_ledger_proof(
            directory, None, None, identities or self.identities
        )

    def proofs(self, *sizes):
        """Grow one ledger in successive chunks and return the exported
        proof after each chunk; every proof uses the full identity set
        so they form one audit."""
        directory = self.ledger()
        proofs = []
        cursor = 0
        expected = None
        for size in sizes:
            expected = self._append(
                directory,
                self.entries[cursor : cursor + size],
                expected,
            )
            cursor += size
            proofs.append(self._export(directory))
        return proofs

    def bootstrap(self):
        return BranchStore.create_proof_audit_checkpoint(())

    @staticmethod
    def digest_of(checkpoint: str) -> str:
        return hashlib.sha256(checkpoint.encode("utf-8")).hexdigest()

    def raw_document(self, name: str = "repository.json"):
        with open(self.path(name), "rb") as handle:
            return handle.read()

    def directory_files(self, name: str = "ledger") -> dict:
        directory = os.path.join(self.tmp.name, name)
        contents = {}
        if not os.path.isdir(directory):
            return contents
        for entry in os.listdir(directory):
            with open(os.path.join(directory, entry), "rb") as handle:
                contents[entry] = handle.read()
        return contents


class ConstructionTests(RepositoryTestBase):
    def test_path_must_be_non_empty_str(self) -> None:
        for bad in (1, 1.5, b"x", ["x"], {"x": 1}, None, object()):
            with self.subTest(bad=type(bad).__name__):
                with self.assertRaises(TypeError):
                    CheckpointRepository(bad, None)
        with self.assertRaises(ValueError):
            CheckpointRepository("", None)

    def test_missing_parent_directory_raises_os_error(self) -> None:
        missing = os.path.join(self.tmp.name, "no-such-dir", "repo.json")
        with self.assertRaises(OSError):
            CheckpointRepository(missing, None)

    def test_timeout_validation(self) -> None:
        path = self.path()
        for bad in (True, "1", object(), (), []):
            with self.subTest(bad=type(bad).__name__):
                with self.assertRaises(TypeError):
                    CheckpointRepository(path, bad)
        for bad in (-1, -0.1, math.nan, math.inf, -math.inf):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    CheckpointRepository(path, bad)
        # Zero, None and positive finite numbers are accepted.
        for value in (None, 0, 0.0, 1, 2.5):
            CheckpointRepository(path, value)

    def test_construction_does_not_create_the_repository(self) -> None:
        path = self.path()
        CheckpointRepository(path, None)
        self.assertFalse(os.path.exists(path))

    def test_equivalent_path_spellings_share_the_lock(self) -> None:
        # A relative spelling and an absolute one resolve to the same
        # real lock path.
        rel = os.path.relpath(self.path())
        absolute = os.path.realpath(rel)
        one = CheckpointRepository(rel, None)
        other = CheckpointRepository(absolute, None)
        self.assertEqual(one._lock_path, other._lock_path)


class LoadTests(RepositoryTestBase):
    def test_load_before_first_commit_raises_key_error(self) -> None:
        with self.assertRaises(KeyError):
            self.repository().load()

    def test_load_returns_checkpoint_and_sha256_digest(self) -> None:
        repo = self.repository()
        checkpoint = self.bootstrap()
        digest = repo.save(None, checkpoint, "seed")
        loaded_checkpoint, loaded_digest = repo.load()
        self.assertEqual(loaded_checkpoint, checkpoint)
        self.assertEqual(loaded_digest, digest)
        self.assertEqual(loaded_digest, self.digest_of(checkpoint))

    def test_load_is_read_only(self) -> None:
        repo = self.repository()
        checkpoint = self.bootstrap()
        repo.save(None, checkpoint, "seed")
        before = set(os.listdir(self.tmp.name))
        for _ in range(3):
            loaded = repo.load()
            self.assertEqual(loaded[0], checkpoint)
        self.assertEqual(set(os.listdir(self.tmp.name)), before)


class SaveValidationTests(RepositoryTestBase):
    def setUp(self) -> None:
        super().setUp()
        self.checkpoint = self.bootstrap()

    def test_expected_type_and_shape(self) -> None:
        repo = self.repository()
        for bad in (1, 1.5, b"x", [], {}, object()):
            with self.subTest(bad=type(bad).__name__):
                with self.assertRaises(TypeError):
                    repo.save(bad, self.checkpoint, "seed")
        for bad in ("", "abc", "z" * 64, "0" * 63):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    repo.save(bad, self.checkpoint, "seed")

    def test_checkpoint_must_be_authentic_non_empty_str(self) -> None:
        repo = self.repository()
        for bad in (1, 1.5, b"x", [], {}, None, object()):
            with self.subTest(bad=type(bad).__name__):
                with self.assertRaises(TypeError):
                    repo.save(None, bad, "seed")
        with self.assertRaises(ValueError):
            repo.save(None, "", "seed")
        with self.assertRaises(ValueError):
            repo.save(None, "not a checkpoint", "seed")

    def test_request_id_must_be_non_empty_str(self) -> None:
        repo = self.repository()
        for bad in (1, 1.5, b"x", [], {}, None, object()):
            with self.subTest(bad=type(bad).__name__):
                with self.assertRaises(TypeError):
                    repo.save(None, self.checkpoint, bad)
        with self.assertRaises(ValueError):
            repo.save(None, self.checkpoint, "")

    def test_invalid_checkpoint_is_rejected_before_the_lock(self) -> None:
        # With an unwritable coordination location the lock itself would
        # fail; an invalid checkpoint must surface as ValueError first.
        repo = self.repository()
        with self.assertRaises(ValueError):
            repo.save(None, "not a checkpoint", "seed")
        self.assertFalse(os.path.exists(self.path()))

    def test_none_after_the_first_commit_is_runtime_error(self) -> None:
        repo = self.repository()
        repo.save(None, self.checkpoint, "seed")
        with self.assertRaises(RuntimeError):
            repo.save(None, self.checkpoint, "again")

    def test_wrong_digest_is_runtime_error(self) -> None:
        repo = self.repository()
        repo.save(None, self.checkpoint, "seed")
        with self.assertRaises(RuntimeError):
            repo.save("0" * 64, self.checkpoint, "again")

    def test_expected_must_be_none_when_no_repository_exists(self) -> None:
        repo = self.repository()
        with self.assertRaises(RuntimeError):
            repo.save(self.digest_of(self.checkpoint), self.checkpoint, "x")


class OptimisticConcurrencyTests(RepositoryTestBase):
    def test_concurrent_first_commits_have_one_winner(self) -> None:
        checkpoint = self.bootstrap()
        repo = self.repository()

        def attempt(index: int):
            try:
                repo.save(None, checkpoint, f"seed-{index}")
                return "ok"
            except RuntimeError:
                return "lost"

        with ThreadPoolExecutor(max_workers=8) as pool:
            outcomes = list(pool.map(attempt, range(8)))
        self.assertEqual(outcomes.count("ok"), 1)
        self.assertEqual(outcomes.count("lost"), 7)
        self.assertEqual(repo.load()[0], checkpoint)

    def test_concurrent_stale_digest_commits_have_one_winner(self) -> None:
        (first, second) = self.proofs(4, 2)
        repo = self.repository()
        seed = BranchStore.create_proof_audit_checkpoint(())
        base = repo.save(None, seed, "seed")

        candidates = []
        for count in range(1, 9):
            cp = BranchStore.create_proof_audit_checkpoint(
                (first,) * count
            )
            candidates.append(cp)

        def attempt(index: int):
            try:
                digest = repo.save(
                    base, candidates[index], f"save-{index}"
                )
                return ("ok", digest)
            except RuntimeError:
                return ("lost", None)

        with ThreadPoolExecutor(max_workers=8) as pool:
            outcomes = list(pool.map(attempt, range(8)))
        winners = [out for out in outcomes if out[0] == "ok"]
        self.assertEqual(len(winners), 1)
        winner_digest = winners[0][1]
        winner_index = next(
            index
            for index, candidate in enumerate(candidates)
            if self.digest_of(candidate) == winner_digest
        )
        checkpoint, digest = repo.load()
        self.assertEqual(digest, winner_digest)
        self.assertEqual(checkpoint, candidates[winner_index])
        del second  # one growing ledger is enough for this race


class ResumeTests(RepositoryTestBase):
    def test_resume_requires_an_existing_repository(self) -> None:
        with self.assertRaises(RuntimeError):
            self.repository().resume(None, (), "r")

    def test_proofs_follow_the_tuple_contract(self) -> None:
        repo = self.repository()
        repo.save(None, self.bootstrap(), "seed")
        checkpoint, digest = repo.load()
        for bad in ([], {}, "x", 1, None):
            with self.subTest(bad=type(bad).__name__):
                with self.assertRaises(TypeError):
                    repo.resume(digest, bad, "r")
        (proof,) = self.proofs(4)
        for bad in (1, b"x", [proof], None):
            with self.subTest(bad=type(bad).__name__):
                with self.assertRaises(TypeError):
                    repo.resume(digest, (bad,), "r")

    def test_resume_matches_the_one_shot_audit_and_persists(self) -> None:
        proofs = self.proofs(4, 2, 2)
        repo = self.repository()
        digest = repo.save(None, self.bootstrap(), "seed")
        audit, checkpoint, new_digest = repo.resume(
            digest, (proofs[0],), "r1"
        )
        self.assertEqual(
            audit,
            BranchStore.audit_diagnostic_ledger_proof_sequence(
                (proofs[0],)
            ),
        )
        self.assertEqual(new_digest, self.digest_of(checkpoint))
        audit, checkpoint, digest = repo.resume(
            new_digest, (proofs[1], proofs[2]), "r2"
        )
        self.assertEqual(
            audit,
            BranchStore.audit_diagnostic_ledger_proof_sequence(
                tuple(proofs)
            ),
        )
        self.assertEqual(
            [hop["index"] for hop in audit["hops"]], [1, 2]
        )
        loaded, loaded_digest = repo.load()
        self.assertEqual(loaded, checkpoint)
        self.assertEqual(loaded_digest, digest)

    def test_invalid_proof_raises_value_error_without_writing(self) -> None:
        proofs = self.proofs(4)
        repo = self.repository()
        digest = repo.save(None, self.bootstrap(), "seed")
        raw_before = self.raw_document()
        with self.assertRaises(ValueError):
            repo.resume(digest, ("not a proof",), "r1")
        with self.assertRaises(ValueError):
            repo.resume(digest, (proofs[0], "not a proof"), "r1")
        self.assertEqual(self.raw_document(), raw_before)
        self.assertEqual(repo.load()[1], digest)

    def test_empty_continuation_is_replayable_and_equivalent(self) -> None:
        (proof,) = self.proofs(4)
        repo = self.repository()
        digest = repo.save(None, self.bootstrap(), "seed")
        _audit, checkpoint, digest = repo.resume(
            digest, (proof,), "r1"
        )
        idle, idle_checkpoint, idle_digest = repo.resume(
            digest, (), "idle"
        )
        self.assertEqual(idle_checkpoint, checkpoint)
        self.assertEqual(idle_digest, digest)
        self.assertEqual(
            idle,
            BranchStore.audit_diagnostic_ledger_proof_sequence(
                (proof,)
            ),
        )


class IdempotencyTests(RepositoryTestBase):
    def test_save_retry_returns_original_digest_without_recommit(self) -> None:
        (proof,) = self.proofs(4)
        checkpoint = BranchStore.create_proof_audit_checkpoint(
            (proof,)
        )
        repo = self.repository()
        digest = repo.save(None, self.bootstrap(), "seed")

        first = repo.save(digest, checkpoint, "save-1")
        raw_after_first = self.raw_document()
        # Stale expected and a fresh repository handle: still a replay.
        reopened = self.repository()
        second = reopened.save(digest, checkpoint, "save-1")
        self.assertEqual(second, first)
        self.assertEqual(self.raw_document(), raw_after_first)

    def test_same_save_id_with_different_checkpoint_is_value_error(
        self,
    ) -> None:
        (first, second) = self.proofs(4, 2)
        cp1 = BranchStore.create_proof_audit_checkpoint((first,))
        cp2 = BranchStore.create_proof_audit_checkpoint(
            (first, second)
        )
        repo = self.repository()
        digest = repo.save(None, self.bootstrap(), "seed")
        repo.save(digest, cp1, "save-1")
        with self.assertRaises(ValueError):
            repo.save(digest, cp2, "save-1")

    def test_resume_retry_after_restart_returns_original_result(self) -> None:
        proofs = self.proofs(4, 2)
        repo = self.repository()
        digest = repo.save(None, self.bootstrap(), "seed")
        audit, checkpoint, new_digest = repo.resume(
            digest, (proofs[0],), "r1"
        )
        raw_after = self.raw_document()

        reopened = self.repository()
        # A stale expected still replays the completed request.
        replay_audit, replay_checkpoint, replay_digest = reopened.resume(
            digest, (proofs[0],), "r1"
        )
        self.assertEqual(replay_audit, audit)
        self.assertEqual(replay_checkpoint, checkpoint)
        self.assertEqual(replay_digest, new_digest)
        self.assertEqual(self.raw_document(), raw_after)

    def test_concurrent_same_id_resume_commits_once(self) -> None:
        (proof,) = self.proofs(4)
        repo = self.repository()
        digest = repo.save(None, self.bootstrap(), "seed")

        def attempt(_index: int):
            return repo.resume(digest, (proof,), "r1")

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(attempt, range(8)))
        first = results[0]
        for result in results[1:]:
            self.assertEqual(result, first)
        # Exactly one evidence record, so no continuation was doubled.
        document = json.loads(self.raw_document())
        ids = [request["id"] for request in document["requests"]]
        self.assertEqual(ids.count("r1"), 1)
        audit, _cp, _digest = first
        # One proof from the bootstrap yields no hop; a doubled
        # continuation would have produced an extra "same" hop.
        self.assertEqual(audit["hops"], ())

    def test_same_resume_id_with_different_proofs_is_value_error(
        self,
    ) -> None:
        proofs = self.proofs(4, 2)
        repo = self.repository()
        digest = repo.save(None, self.bootstrap(), "seed")
        repo.resume(digest, (proofs[0],), "r1")
        with self.assertRaises(ValueError):
            repo.resume(digest, (proofs[0], proofs[1]), "r1")

    def test_request_id_cannot_cross_operation_kinds(self) -> None:
        (proof,) = self.proofs(4)
        checkpoint = BranchStore.create_proof_audit_checkpoint((proof,))
        repo = self.repository()
        digest = repo.save(None, self.bootstrap(), "shared")
        # "shared" already recorded as a save; using it on a resume fails.
        with self.assertRaises(ValueError):
            repo.resume(digest, (proof,), "shared")


class TerminalStateRepositoryTests(RepositoryTestBase):
    def _fork_pair(self):
        base = self.ledger("base")
        digest = self._append(base, self.entries[:4], None)
        left = self.ledger("left")
        right = self.ledger("right")
        shutil.copytree(base, left, dirs_exist_ok=True)
        shutil.copytree(base, right, dirs_exist_ok=True)
        self._append(left, self.entries[4:6], digest)
        other = tuple(
            (time, make_chain(transaction(100 + index)))
            for index, time in enumerate((5, 6))
        )
        self._append(right, other, digest)
        return (
            self._export(left),
            self._export(right),
        )

    def test_failed_checkpoint_rejects_non_empty_before_reading_proofs(
        self,
    ) -> None:
        (first,) = self.proofs(4)
        before, after = self._fork_pair()
        repo = self.repository()
        digest = repo.save(None, self.bootstrap(), "seed")
        _a, checkpoint, digest = repo.resume(
            digest, (first,), "good"
        )
        audit, failed_checkpoint, failed_digest = repo.resume(
            digest, (before, after), "fork"
        )
        self.assertEqual(audit["status"], "failed")
        self.assertEqual(failed_digest, self.digest_of(failed_checkpoint))
        raw_before = self.raw_document()
        # A non-empty continuation is RuntimeError immediately, even
        # though the only "proof" in the batch could never authenticate;
        # no new evidence is written.
        with self.assertRaises(RuntimeError):
            repo.resume(
                failed_digest, ("not even read as a proof",), "more"
            )
        self.assertEqual(self.raw_document(), raw_before)
        # The empty read-only continuation still answers.
        idle, idle_cp, idle_digest = repo.resume(
            failed_digest, (), "read"
        )
        self.assertEqual(idle_cp, failed_checkpoint)
        self.assertEqual(idle_digest, failed_digest)
        self.assertEqual(idle["status"], "failed")


class RepositoryEncodingTests(RepositoryTestBase):
    def test_document_shape_and_compact_utf8_json(self) -> None:
        (proof,) = self.proofs(4)
        checkpoint = BranchStore.create_proof_audit_checkpoint((proof,))
        repo = self.repository()
        repo.save(None, self.bootstrap(), "seed")
        digest = repo.load()[1]
        repo.resume(digest, (proof,), "r1")
        raw = self.raw_document()
        self.assertFalse(raw.endswith(b"\n"))
        self.assertFalse(raw.startswith(b"\xef\xbb\xbf"))
        self.assertNotIn(b": ", raw)
        self.assertNotIn(b", ", raw)
        raw.decode("utf-8")  # valid UTF-8
        text = raw.decode("utf-8")
        # Canonical: re-serialization reproduces the bytes.
        document = json.loads(text)
        self.assertEqual(
            list(document),
            [
                "format",
                "version",
                "checkpoint",
                "digest",
                "requests",
                "checksum",
            ],
        )
        self.assertEqual(document["version"], 1)
        self.assertEqual(document["checkpoint"], checkpoint)
        self.assertEqual(
            document["digest"], self.digest_of(checkpoint)
        )
        self.assertEqual(len(document["requests"]), 2)
        self.assertEqual(
            list(document["requests"][0]),
            [
                "id",
                "kind",
                "input_digest",
                "result_checkpoint",
                "result_digest",
            ],
        )
        self.assertEqual(document["requests"][0]["kind"], "save")
        self.assertEqual(document["requests"][1]["kind"], "resume")
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
        # Byte-for-byte deterministic for the same history.
        sibling = self.repository("sibling.json")
        sibling.save(None, self.bootstrap(), "seed")
        sibling.resume(sibling.load()[1], (proof,), "r1")
        self.assertEqual(self.raw_document("sibling.json"), raw)


class DurabilityTests(RepositoryTestBase):
    def _seeded_repository(self):
        (proof,) = self.proofs(4)
        checkpoint = BranchStore.create_proof_audit_checkpoint((proof,))
        repo = self.repository()
        repo.save(None, self.bootstrap(), "seed")
        return repo, checkpoint

    def test_replace_failure_keeps_old_repository_complete(self) -> None:
        repo, checkpoint = self._seeded_repository()
        old_checkpoint, old_digest = repo.load()
        raw_before = self.raw_document()
        with mock.patch(
            "city_twin.branches.os.replace", side_effect=OSError("boom")
        ):
            with self.assertRaises(OSError):
                repo.save(old_digest, checkpoint, "save-1")
        # Old repository intact, no temp residue, no partial result.
        self.assertEqual(self.raw_document(), raw_before)
        self.assertEqual(repo.load(), (old_checkpoint, old_digest))
        leftovers = [
            name
            for name in os.listdir(self.tmp.name)
            if name.endswith(".tmp")
        ]
        self.assertEqual(leftovers, [])
        # The commit can now succeed for real.
        new_digest = repo.save(old_digest, checkpoint, "save-1")
        self.assertEqual(repo.load(), (checkpoint, new_digest))

    def test_fsync_failure_keeps_old_repository_complete(self) -> None:
        repo, checkpoint = self._seeded_repository()
        old_digest = repo.load()[1]
        raw_before = self.raw_document()
        with mock.patch(
            "city_twin.branches.os.fsync", side_effect=OSError("boom")
        ):
            with self.assertRaises(OSError):
                repo.save(old_digest, checkpoint, "save-1")
        self.assertEqual(self.raw_document(), raw_before)
        self.assertEqual(
            [n for n in os.listdir(self.tmp.name) if n.endswith(".tmp")],
            [],
        )

    def test_interrupted_temp_is_removed_after_main_authenticates(
        self,
    ) -> None:
        repo, checkpoint = self._seeded_repository()
        old_digest = repo.load()[1]
        handle = repo._path  # noqa: SLF001 - bind the real path
        prefix = ".proof-audit-checkpoint-" + os.path.basename(handle) + "-"
        leftover = os.path.join(
            self.tmp.name, prefix + "interrupted.tmp"
        )
        with open(leftover, "wb") as crashed:
            crashed.write(b'{"half": true')
        self.assertTrue(os.path.exists(leftover))
        repo.save(old_digest, checkpoint, "save-1")
        self.assertFalse(os.path.exists(leftover))

    def test_temp_file_never_masks_a_corrupt_main_file(self) -> None:
        repo, _checkpoint = self._seeded_repository()
        # A perfectly valid *other* repository sits in an abandoned temp
        # file; the corrupted main file must still be what is read.
        decoy = self.bootstrap()
        decoy_document = self._repository_document_bytes(decoy, [])
        handle = repo._path  # noqa: SLF001
        prefix = ".proof-audit-checkpoint-" + os.path.basename(handle) + "-"
        decoy_path = os.path.join(
            self.tmp.name, prefix + "decoy.tmp"
        )
        with open(decoy_path, "wb") as crashed:
            crashed.write(decoy_document)
        with open(self.path(), "wb") as damaged:
            damaged.write(b"{not valid json")
        with self.assertRaises(ValueError):
            repo.load()
        with self.assertRaises(ValueError):
            repo.save(self.bootstrap(), decoy, "x")
        with self.assertRaises(ValueError):
            repo.resume(None, (), "x")
        # The decoy is left in place because the main file never
        # authenticated and therefore never authorized cleanup.
        self.assertTrue(os.path.exists(decoy_path))

    @staticmethod
    def _repository_document_bytes(checkpoint, requests) -> bytes:
        digest = hashlib.sha256(
            checkpoint.encode("utf-8")
        ).hexdigest()
        body = {
            "format": CheckpointRepository._FORMAT,
            "version": CheckpointRepository._VERSION,
            "checkpoint": checkpoint,
            "digest": digest,
            "requests": requests,
        }
        document = dict(body)
        document["checksum"] = hashlib.sha256(
            BranchStore._canonical_json(body).encode("utf-8")
        ).hexdigest()
        return BranchStore._canonical_json(document).encode("utf-8")


class RepositoryAuthenticationTests(RepositoryTestBase):
    def setUp(self) -> None:
        super().setUp()
        (proof,) = self.proofs(4)
        self.checkpoint = BranchStore.create_proof_audit_checkpoint(
            (proof,)
        )
        self.repo = self.repository()
        self.repo.save(None, self.bootstrap(), "seed")
        digest = self.repo.load()[1]
        self.repo.resume(digest, (proof,), "r1")

    def _reseal(self, mutate):
        document = json.loads(self.raw_document())
        mutate(document)
        body = {
            key: value
            for key, value in document.items()
            if key != "checksum"
        }
        document["checksum"] = hashlib.sha256(
            BranchStore._canonical_json(body).encode("utf-8")
        ).hexdigest()
        with open(self.path(), "wb") as handle:
            handle.write(
                BranchStore._canonical_json(document).encode("utf-8")
            )

    def test_raw_bit_flip_is_rejected(self) -> None:
        raw = self.raw_document()
        with open(self.path(), "wb") as handle:
            handle.write(raw[:20] + b"X" + raw[21:])
        with self.assertRaises(ValueError):
            self.repo.load()

    def test_digest_disagreeing_with_checkpoint_is_rejected(self) -> None:
        def mutate(document):
            document["digest"] = "0" * 64
        self._reseal(mutate)
        with self.assertRaises(ValueError):
            self.repo.load()

    def test_checkpoint_that_no_longer_authenticates_is_rejected(
        self,
    ) -> None:
        def mutate(document):
            # Swap the stored checkpoint for a bootstrap; the repository
            # digest no longer agrees and the replacement is detected.
            document["checkpoint"] = self.bootstrap()
        self._reseal(mutate)
        with self.assertRaises(ValueError):
            self.repo.load()

    def test_inconsistent_request_evidence_is_rejected(self) -> None:
        def mutate(document):
            # The resume record's result checkpoint is the non-empty
            # self.checkpoint; replace it with the bootstrap so the
            # recorded digest no longer agrees.
            resume_record = next(
                request
                for request in document["requests"]
                if request["kind"] == "resume"
            )
            resume_record["result_checkpoint"] = self.bootstrap()
        self._reseal(mutate)
        with self.assertRaises(ValueError):
            self.repo.load()

    def test_duplicate_request_id_in_evidence_is_rejected(self) -> None:
        def mutate(document):
            document["requests"].append(dict(document["requests"][0]))
        self._reseal(mutate)
        with self.assertRaises(ValueError):
            self.repo.load()

    def test_unknown_format_and_version_rejected(self) -> None:
        def change_format(document):
            document["format"] = "other"
        self._reseal(change_format)
        with self.assertRaises(ValueError):
            self.repo.load()

        # A corrupt main file is never silently overwritten, so seed a
        # fresh file for the version mutation.
        os.remove(self.path())
        self.repo = self.repository()
        self.repo.save(None, self.bootstrap(), "seed")

        def change_version(document):
            document["version"] = 2
        self._reseal(change_version)
        with self.assertRaises(ValueError):
            self.repo.load()


class CrossProcessTests(RepositoryTestBase):
    SCRIPT = r"""
import os, sys
from city_twin.branches import CheckpointRepository
path, checkpoint_file, request_id, timeout = (
    sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
)
with open(checkpoint_file, encoding="utf-8") as handle:
    checkpoint = handle.read()
repo = CheckpointRepository(path, float(timeout))
try:
    digest = repo.save(None, checkpoint, request_id)
except RuntimeError as exc:
    print("LOST")
except TimeoutError as exc:
    print("TIMEOUT")
else:
    print("WIN " + digest[:8])
"""

    def _write_checkpoint_file(self, name, checkpoint):
        path = os.path.join(self.tmp.name, name)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(checkpoint)
        return path

    def test_concurrent_process_first_commits_have_one_winner(self) -> None:
        checkpoint = self.bootstrap()
        checkpoint_file = self._write_checkpoint_file(
            "checkpoint.txt", checkpoint
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = os.getcwd()
        processes = [
            subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    self.SCRIPT,
                    self.path(),
                    checkpoint_file,
                    f"seed-{index}",
                    "30",
                ],
                env=env,
                stdout=subprocess.PIPE,
                text=True,
            )
            for index in range(6)
        ]
        outcomes = [process.communicate()[0].strip() for process in processes]
        wins = [out for out in outcomes if out.startswith("WIN")]
        self.assertEqual(len(wins), 1, outcomes)
        self.assertEqual(outcomes.count("LOST"), 5)
        # The one winner's checkpoint is what survives.
        loaded, _digest = self.repository().load()
        self.assertEqual(loaded, checkpoint)

    HOLDER_SCRIPT = r"""
import os, sys, time
import fcntl
lock_path = sys.argv[1]
fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
fcntl.flock(fd, fcntl.LOCK_EX)
sys.stdout.write("LOCKED")
sys.stdout.flush()
time.sleep(5)
"""

    def test_lock_wait_timeout_raises_timeout_error(self) -> None:
        # First create the repository so load() has data to return once
        # the lock is free; the holder only blocks the coordination file.
        repo = self.repository()
        repo.save(None, self.bootstrap(), "seed")
        # Capture the current digest before the coordination file is
        # held, so nothing inside the locked window blocks on the
        # unbounded ``repo`` handle.
        current_digest = repo.load()[1]
        env = dict(os.environ)
        env["PYTHONPATH"] = os.getcwd()
        holder = subprocess.Popen(
            [
                sys.executable,
                "-c",
                self.HOLDER_SCRIPT,
                self.path() + ".lock",
            ],
            env=env,
            stdout=subprocess.PIPE,
            text=True,
        )
        try:
            self.assertEqual(holder.stdout.read(6), "LOCKED")
            bounded = self.repository(timeout=0.2)
            with self.assertRaises(TimeoutError):
                bounded.load()
            with self.assertRaises(TimeoutError):
                bounded.save(
                    current_digest, self.bootstrap(), "late"
                )
        finally:
            holder.terminate()
            holder.wait()
            holder.stdout.close()


class PurityTests(RepositoryTestBase):
    def test_repository_operations_never_touch_the_ledger(self) -> None:
        proofs = self.proofs(4, 2)
        snapshot = self.directory_files()
        digest_before = BranchStore.page_diagnostic_ledger(
            os.path.join(self.tmp.name, "ledger"), None, None, None, 100, None
        )["snapshot"]["digest"]
        repo = self.repository()
        seed = repo.save(None, self.bootstrap(), "seed")
        repo.resume(seed, (proofs[0],), "r1")
        digest = repo.load()[1]
        with self.assertRaises(ValueError):
            repo.resume(digest, ("not a proof",), "bad")
        with self.assertRaises(RuntimeError):
            repo.save("0" * 64, self.bootstrap(), "stale")
        with self.assertRaises(ValueError):
            repo.save(digest, "not a checkpoint", "badcp")
        self.assertEqual(self.directory_files(), snapshot)
        digest_after = BranchStore.page_diagnostic_ledger(
            os.path.join(self.tmp.name, "ledger"), None, None, None, 100, None
        )["snapshot"]["digest"]
        self.assertEqual(digest_after, digest_before)


if __name__ == "__main__":
    unittest.main()
