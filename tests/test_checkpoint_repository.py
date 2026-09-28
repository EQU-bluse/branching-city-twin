"""Tests for the concurrent, crash-safe persistent
``CheckpointRepository`` for incremental offline proof-audit
checkpoints, and for the ``resume_proof_audit`` sealing fix.

The repository binds one compact UTF-8 JSON document per path and
serializes commits with a cross-process file lock. These tests cover:

* constructor validation: ``path`` must be a non-empty :class:`str`,
  ``timeout`` must be ``None`` or a non-:class:`bool` finite
  non-negative number;
* :meth:`load`: :class:`KeyError` when no repository exists, the
  checkpoint plus its SHA-256 digest otherwise, :class:`ValueError` on
  any tampered document and :class:`TimeoutError` when the lock cannot
  be acquired;
* compare-and-swap: the first commit takes ``expected=None``; later
  commits must name the current digest or raise :class:`RuntimeError`,
  and concurrent commits of the same old digest have exactly one
  winner across threads and processes;
* request idempotency: the same ``request_id`` with the same inputs
  returns the original result without a second continuation, even with
  a stale ``expected`` and even after a process restart; the same id
  with different inputs raises :class:`ValueError`;
* durable shape and crash safety: compact canonical JSON, temp files
  from interrupted commits never participate in reads and are reclaimed
  only after the main document authenticates, so they can never mask a
  damaged main file;
* the failed-terminal sealing fix: after checkpoint authentication a
  failed state rejects a non-empty batch with :class:`RuntimeError`
  before any new proof is authenticated, while the empty read-only
  continuation still answers forever;
* purity: the ledger directory is byte-for-byte untouched by every
  repository operation.
"""

import fcntl
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest

from city_twin.branches import BranchStore, CheckpointRepository

try:
    from test_diagnostic_ledger import make_chain, transaction
except ImportError:  # running as a package member from the repo root
    from tests.test_diagnostic_ledger import (
        make_chain,
        transaction,
    )


def canonical(document: dict) -> str:
    return BranchStore._canonical_json(document)


class RepositoryTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.chains = [make_chain(transaction(i)) for i in range(12)]
        self.entries = tuple(
            (time_index, chain)
            for time_index, chain in zip(range(1, 13), self.chains)
        )
        self.identities = tuple(transaction(i) for i in range(12))

    def path(self, name: str = "repository.json") -> str:
        return os.path.join(self.tmp.name, name)

    def repo(self, name: str = "repository.json", timeout=None):
        return CheckpointRepository(self.path(name), timeout)

    def ledger(self, name: str) -> str:
        directory = os.path.join(self.tmp.name, name)
        os.mkdir(directory)
        return directory

    def append(self, directory, entries, expected, keep=10, limit=2):
        return BranchStore.append_diagnostic_ledger(
            directory, entries, expected, limit, keep, None
        )

    def export(self, directory, identities=None):
        if identities is None:
            identities = self.identities
        return BranchStore.export_diagnostic_ledger_proof(
            directory, None, None, identities
        )

    @staticmethod
    def digest_of(directory) -> str:
        return BranchStore.page_diagnostic_ledger(
            directory, None, None, None, 100, None
        )["snapshot"]["digest"]

    def grow(self, name, chunks, keep=10):
        directory = self.ledger(name)
        proofs = []
        cursor = 0
        expected = None
        for size in chunks:
            expected = self.append(
                directory,
                self.entries[cursor : cursor + size],
                expected,
                keep=keep,
            )
            cursor += size
            proofs.append(self.export(directory))
        return directory, proofs

    @staticmethod
    def directory_files(directory) -> dict:
        contents = {}
        for name in os.listdir(directory):
            with open(os.path.join(directory, name), "rb") as handle:
                contents[name] = handle.read()
        return contents


class ConstructorValidationTests(RepositoryTestBase):
    def test_path_must_be_nonempty_str(self) -> None:
        for bad in (1, 1.5, b"x", ["x"], {"x": 1}, None, object()):
            with self.subTest(bad=type(bad).__name__):
                with self.assertRaises(TypeError):
                    CheckpointRepository(bad, None)
        with self.assertRaises(ValueError):
            CheckpointRepository("", None)

    def test_path_is_validated_before_timeout(self) -> None:
        # Both invalid: the path fault surfaces first.
        with self.assertRaises(TypeError):
            CheckpointRepository(1, True)
        with self.assertRaises(ValueError):
            CheckpointRepository("", -1)

    def test_timeout_rejects_wrong_types(self) -> None:
        path = self.path()
        for bad in (True, False, "1", b"1", 1j, (1,), None):
            if bad is None:
                continue
            with self.subTest(bad=type(bad).__name__):
                with self.assertRaises(TypeError):
                    CheckpointRepository(path, bad)

    def test_timeout_rejects_illegal_values(self) -> None:
        path = self.path()
        for bad in (-1, -0.1, float("nan"), float("inf"),
                    float("-inf")):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    CheckpointRepository(path, bad)

    def test_timeout_accepts_none_zero_and_positive_finite(self) -> None:
        path = self.path()
        for value in (None, 0, 0.0, 1, 2.5):
            CheckpointRepository(path, value)

    def test_constructor_touches_no_files(self) -> None:
        self.repo("not-yet.json")
        self.assertEqual(os.listdir(self.tmp.name), [])

    def test_missing_parent_directory_raises_os_error(self) -> None:
        nested = os.path.join(self.tmp.name, "missing", "repository.json")
        repository = CheckpointRepository(nested, None)
        with self.assertRaises(OSError):
            repository.load()
        _directory, (proof,) = self.grow("ledger", (4,))
        checkpoint = BranchStore.create_proof_audit_checkpoint((proof,))
        with self.assertRaises(OSError):
            repository.save(None, checkpoint, "seed")
        # The failed operations created no directory or file.
        self.assertFalse(os.path.exists(os.path.join(self.tmp.name, "missing")))


class LoadTests(RepositoryTestBase):
    def test_missing_repository_raises_key_error(self) -> None:
        with self.assertRaises(KeyError):
            self.repo("missing.json").load()
        # The lookup creates no repository file.
        self.assertFalse(os.path.exists(self.path("missing.json")))

    def test_load_returns_checkpoint_and_digest(self) -> None:
        _directory, (proof,) = self.grow("ledger", (4,))
        checkpoint = BranchStore.create_proof_audit_checkpoint((proof,))
        repository = self.repo()
        digest = repository.save(None, checkpoint, "seed")
        loaded = repository.load()
        self.assertEqual(list(loaded), ["checkpoint", "digest"])
        self.assertEqual(loaded["checkpoint"], checkpoint)
        self.assertEqual(
            loaded["digest"],
            hashlib.sha256(checkpoint.encode("utf-8")).hexdigest(),
        )
        self.assertEqual(loaded["digest"], digest)

    def test_load_authenticates_every_field(self) -> None:
        _directory, (proof,) = self.grow("ledger", (4,))
        checkpoint = BranchStore.create_proof_audit_checkpoint((proof,))
        path = self.path()
        repository = self.repo()
        repository.save(None, checkpoint, "seed")
        with open(path, "rb") as handle:
            sound = handle.read()

        def rewrite(mutate):
            document = json.loads(sound)
            mutate(document)
            with open(path, "wb") as handle:
                handle.write(canonical(document).encode("utf-8"))

        def bad_format(document):
            document["format"] = "other"

        def bad_version(document):
            document["version"] = 2

        def bad_digest(document):
            document["digest"] = "f" * 64

        def bad_checkpoint(document):
            document["checkpoint"] = "not a checkpoint"

        def bad_seal(document):
            document["request"]["seal"] = "0" * 64

        def bad_proofs(document):
            document["request"]["proofs"] = ["tampered"]

        def extra_key(document):
            document["extra"] = 1

        for mutate in (
            bad_format,
            bad_version,
            bad_digest,
            bad_checkpoint,
            bad_seal,
            bad_proofs,
            extra_key,
        ):
            rewrite(mutate)
            with self.subTest(mutate=mutate.__name__):
                with self.assertRaises(ValueError):
                    repository.load()
            # Restore the sound byte-for-byte document each time.
            with open(path, "wb") as handle:
                handle.write(sound)

    def test_non_canonical_and_bom_rejected(self) -> None:
        _directory, (proof,) = self.grow("ledger", (4,))
        checkpoint = BranchStore.create_proof_audit_checkpoint((proof,))
        path = self.path()
        repository = self.repo()
        repository.save(None, checkpoint, "seed")
        with open(path, "rb") as handle:
            raw = handle.read()
        for mutated in (
            b"\xef\xbb\xbf" + raw,
            raw + b"\n",
        ):
            with open(path, "wb") as handle:
                handle.write(mutated)
            with self.assertRaises(ValueError):
                repository.load()

    def test_lock_timeout_raises_timeout_error(self) -> None:
        _directory, (proof,) = self.grow("ledger", (4,))
        checkpoint = BranchStore.create_proof_audit_checkpoint((proof,))
        repository = self.repo(timeout=None)
        repository.save(None, checkpoint, "seed")
        held = threading.Event()

        def hold() -> None:
            locker = self.repo(timeout=None)
            fd = os.open(locker._lock_path, os.O_RDWR | os.O_CREAT, 0o644)
            fcntl.flock(fd, fcntl.LOCK_EX)
            held.set()
            time.sleep(1.0)
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

        thread = threading.Thread(target=hold)
        thread.start()
        held.wait()
        try:
            start = time.monotonic()
            with self.assertRaises(TimeoutError):
                self.repo(timeout=0.1).load()
            self.assertLess(time.monotonic() - start, 0.9)
        finally:
            thread.join()


class SaveTests(RepositoryTestBase):
    def _seed_checkpoint(self):
        _directory, (proof,) = self.grow("ledger", (4,))
        return BranchStore.create_proof_audit_checkpoint((proof,))

    def test_first_save_then_round_trip(self) -> None:
        checkpoint = self._seed_checkpoint()
        repository = self.repo()
        digest = repository.save(None, checkpoint, "seed")
        self.assertEqual(
            digest, hashlib.sha256(checkpoint.encode("utf-8")).hexdigest()
        )
        self.assertEqual(repository.load()["checkpoint"], checkpoint)

    def test_expected_validation(self) -> None:
        checkpoint = self._seed_checkpoint()
        repository = self.repo()
        for bad in (1, b"x", (), [], True):
            with self.subTest(bad=type(bad).__name__):
                with self.assertRaises(TypeError):
                    repository.save(bad, checkpoint, "seed")
        for bad in ("", "abc", "z" * 64, "0" * 63):
            with self.subTest(bad=bad[:4]):
                with self.assertRaises(ValueError):
                    repository.save(bad, checkpoint, "seed")

    def test_checkpoint_and_request_id_must_be_nonempty_str(self) -> None:
        checkpoint = self._seed_checkpoint()
        repository = self.repo()
        for bad in (1, b"x", (), None, object()):
            with self.subTest(bad=type(bad).__name__):
                with self.assertRaises(TypeError):
                    repository.save(None, bad, "seed")
                with self.assertRaises(TypeError):
                    repository.save(None, checkpoint, bad)
        with self.assertRaises(ValueError):
            repository.save(None, "", "seed")
        with self.assertRaises(ValueError):
            repository.save(None, checkpoint, "")

    def test_invalid_checkpoint_raises_value_error(self) -> None:
        repository = self.repo()
        with self.assertRaises(ValueError):
            repository.save(None, "not a checkpoint", "seed")
        self.assertFalse(os.path.exists(self.path()))

    def test_second_save_requires_current_digest(self) -> None:
        checkpoint = self._seed_checkpoint()
        repository = self.repo()
        digest = repository.save(None, checkpoint, "seed")
        with self.assertRaises(RuntimeError):
            repository.save(None, checkpoint, "second")
        with self.assertRaises(RuntimeError):
            repository.save("0" * 64, checkpoint, "second")
        # The rejected saves changed nothing.
        self.assertEqual(repository.load()["digest"], digest)
        # The correct digest commits a no-op rewrite.
        again = repository.save(digest, checkpoint, "second")
        self.assertEqual(again, digest)

    def test_same_request_id_same_input_is_idempotent(self) -> None:
        checkpoint = self._seed_checkpoint()
        path = self.path()
        first = CheckpointRepository(path, None)
        digest = first.save(None, checkpoint, "seed")
        # Retried in a fresh instance (a process restart) with a stale
        # digest expectation: the committed request is identified before
        # the compare-and-swap, so the original digest comes back.
        second = CheckpointRepository(path, None)
        repeated = second.save("0" * 64, checkpoint, "seed")
        self.assertEqual(repeated, digest)
        repeated_none = second.save(None, checkpoint, "seed")
        self.assertEqual(repeated_none, digest)
        self.assertEqual(second.load()["digest"], digest)

    def test_same_request_id_different_checkpoint_rejected(self) -> None:
        _directory, (first, _second) = self.grow("ledger", (4, 2))
        one = BranchStore.create_proof_audit_checkpoint((first,))
        two = BranchStore.create_proof_audit_checkpoint((first, _second))
        repository = self.repo()
        digest = repository.save(None, one, "seed")
        with self.assertRaises(ValueError):
            repository.save(digest, two, "seed")
        # The conflict changes nothing.
        loaded = repository.load()
        self.assertEqual(loaded["checkpoint"], one)
        self.assertEqual(loaded["digest"], digest)

    def test_document_is_canonical_compact_utf8(self) -> None:
        checkpoint = self._seed_checkpoint()
        path = self.path()
        self.repo().save(None, checkpoint, "seed-réq")
        with open(path, "rb") as handle:
            raw = handle.read()
        self.assertFalse(raw.startswith(b"\xef\xbb\xbf"))
        self.assertFalse(raw.endswith(b"\n"))
        # Compact canonical JSON: non-ASCII emitted literally as UTF-8
        # rather than \u-escaped, and the bytes re-serialize identically.
        self.assertIn("réq".encode("utf-8"), raw)
        self.assertNotIn(b"\\u", raw)
        document = json.loads(raw)
        self.assertEqual(
            list(document),
            ["format", "version", "checkpoint", "digest", "request"],
        )
        self.assertEqual(document["version"], 1)
        self.assertEqual(canonical(document).encode("utf-8"), raw)
        self.assertEqual(
            list(document["request"]),
            ["id", "kind", "prev", "proofs", "checkpoint", "digest",
             "seal"],
        )
        self.assertIsNone(document["request"]["prev"])
        self.assertIsNone(document["request"]["proofs"])


class ResumeTests(RepositoryTestBase):
    def test_first_resume_starts_from_empty_bootstrap(self) -> None:
        _directory, (first, second) = self.grow("ledger", (4, 2))
        repository = self.repo()
        result = repository.resume(None, (first, second), "batch-1")
        self.assertEqual(
            list(result), ["audit", "checkpoint", "digest"]
        )
        expected = (
            BranchStore.audit_diagnostic_ledger_proof_sequence(
                (first, second)
            )
        )
        self.assertEqual(result["audit"], expected)
        sealed = (
            BranchStore.create_proof_audit_checkpoint((first, second))
        )
        self.assertEqual(result["checkpoint"], sealed)
        self.assertEqual(
            result["digest"],
            hashlib.sha256(sealed.encode("utf-8")).hexdigest(),
        )

    def test_chained_resumes_equal_one_shot_audit(self) -> None:
        _directory, proofs = self.grow(
            "ledger", (2, 2, 2, 2, 2)
        )
        repository = self.repo()
        result = repository.resume(None, (proofs[0],), "b0")
        for index, proof in enumerate(proofs[1:], start=1):
            result = repository.resume(
                result["digest"], (proof,), f"b{index}"
            )
        one_shot = (
            BranchStore.audit_diagnostic_ledger_proof_sequence(
                tuple(proofs)
            )
        )
        self.assertEqual(result["audit"], one_shot)
        self.assertEqual(
            result["checkpoint"],
            BranchStore.create_proof_audit_checkpoint(tuple(proofs)),
        )

    def test_resume_mixed_with_save(self) -> None:
        _directory, (first, second) = self.grow("ledger", (4, 2))
        repository = self.repo()
        result = repository.resume(None, (first,), "b0")
        digest = repository.save(
            result["digest"], result["checkpoint"], "snapshot"
        )
        self.assertEqual(digest, result["digest"])
        again = repository.resume(digest, (second,), "b1")
        self.assertEqual(
            again["audit"],
            BranchStore.audit_diagnostic_ledger_proof_sequence(
                (first, second)
            ),
        )

    def test_empty_batch_is_read_only_and_writes_nothing(self) -> None:
        _directory, (first, second) = self.grow("ledger", (4, 2))
        repository = self.repo()
        # Read-only against the unborn repository answers the empty
        # bootstrap and creates no file.
        unborn = repository.resume(None, (), "peek")
        self.assertEqual(unborn["audit"]["status"], "empty")
        self.assertEqual(
            unborn["checkpoint"],
            BranchStore.create_proof_audit_checkpoint(()),
        )
        self.assertFalse(os.path.exists(self.path()))
        result = repository.resume(None, (first, second), "batch")
        with open(self.path(), "rb") as handle:
            snapshot = handle.read()
        read = repository.resume(result["digest"], (), "query")
        self.assertEqual(read["audit"], result["audit"])
        self.assertEqual(read["checkpoint"], result["checkpoint"])
        self.assertEqual(read["digest"], result["digest"])
        # No write happened.
        with open(self.path(), "rb") as handle:
            self.assertEqual(handle.read(), snapshot)

    def test_read_only_still_checks_the_digest(self) -> None:
        _directory, (first,) = self.grow("ledger", (4,))
        repository = self.repo()
        result = repository.resume(None, (first,), "batch")
        with self.assertRaises(RuntimeError):
            repository.resume("0" * 64, (), "query")
        with self.assertRaises(RuntimeError):
            repository.resume(None, (), "query")
        with self.assertRaises(KeyError):
            self.repo("other.json").resume("0" * 64, (), "query")

    def test_read_only_reusing_a_committed_non_empty_id_rejected(
        self,
    ) -> None:
        _directory, (first,) = self.grow("ledger", (4,))
        repository = self.repo()
        result = repository.resume(None, (first,), "batch")
        # The id committed a non-empty batch; an empty continuation is a
        # different input and must not masquerade as that request.
        with self.assertRaises(ValueError):
            repository.resume(result["digest"], (), "batch")
        # An unrelated id is an ordinary read-only query.
        read = repository.resume(result["digest"], (), "query")
        self.assertEqual(read["digest"], result["digest"])

    def test_proof_arguments_follow_the_continuation_contract(self) -> None:
        _directory, (first,) = self.grow("ledger", (4,))
        repository = self.repo()
        repository.resume(None, (first,), "b0")
        loaded = repository.load()
        for bad in ([first], {first}, None, 1, object()):
            with self.subTest(bad=type(bad).__name__):
                with self.assertRaises(TypeError):
                    repository.resume(loaded["digest"], bad, "id")
        for bad in (1, 1.5, b"x", None, object()):
            with self.subTest(bad=type(bad).__name__):
                with self.assertRaises(TypeError):
                    repository.resume(
                        loaded["digest"], (bad,), "id"
                    )
        with self.assertRaises(ValueError):
            repository.resume(loaded["digest"], ("",), "id")
        with self.assertRaises(ValueError):
            repository.resume(
                loaded["digest"], ("not a proof",), "id"
            )

    def test_failed_terminal_rejects_non_empty_batch(self) -> None:
        before, after = self._fork_pair()
        repository = self.repo()
        result = repository.resume(None, (before, after), "fork")
        self.assertEqual(result["audit"]["status"], "failed")
        # Even an unauthenticatable string never gets read: the sealed
        # terminal state raises RuntimeError first.
        for batch in ((after,), ("",), ("not a proof",)):
            with self.subTest(batch=batch[:1]):
                with self.assertRaises(RuntimeError):
                    repository.resume(
                        result["digest"], batch, "more"
                    )
        # The empty continuation still answers, unchanged.
        read = repository.resume(result["digest"], (), "query")
        self.assertEqual(read["audit"], result["audit"])
        self.assertEqual(read["checkpoint"], result["checkpoint"])

    def test_failed_terminal_inside_first_batch_persists(self) -> None:
        before, after = self._fork_pair()
        repository = self.repo()
        result = repository.resume(None, (before, after), "fork")
        # A fresh instance (simulated restart) sees the sealed state.
        restarted = self.repo()
        self.assertEqual(restarted.load()["digest"], result["digest"])
        with self.assertRaises(RuntimeError):
            restarted.resume(result["digest"], (after,), "next")

    def test_same_request_id_same_batch_is_idempotent_across_restart(
        self,
    ) -> None:
        _directory, (first, second) = self.grow("ledger", (4, 2))
        path = self.path()
        first_repo = CheckpointRepository(path, None)
        result = first_repo.resume(None, (first,), "batch")
        # Restart and retry the identical request with the *new* digest
        # (the normal case) and with a stale digest (the crash case):
        # both return the original result without a second hop.
        restarted = CheckpointRepository(path, None)
        repeated = restarted.resume(
            result["digest"], (first,), "batch"
        )
        self.assertEqual(repeated, result)
        stale = restarted.resume("0" * 64, (first,), "batch")
        self.assertEqual(stale, result)
        # Continuing with the genuine next batch under a new request id.
        continued = restarted.resume(
            result["digest"], (second,), "next"
        )
        self.assertEqual(
            continued["audit"],
            BranchStore.audit_diagnostic_ledger_proof_sequence(
                (first, second)
            ),
        )
        # Only the last committed request's evidence is retained, so the
        # old id can no longer be replayed after "next"; repeating it
        # becomes a genuine new continuation from the current state.
        genuine = restarted.resume(
            continued["digest"], (first,), "batch"
        )
        self.assertEqual(
            genuine["audit"],
            BranchStore.audit_diagnostic_ledger_proof_sequence(
                (first, second, first)
            ),
        )
        self.assertEqual(genuine["audit"]["status"], "failed")

    def test_same_request_id_different_batch_rejected(self) -> None:
        _directory, (first, second) = self.grow("ledger", (4, 2))
        repository = self.repo()
        result = repository.resume(None, (first,), "batch")
        with self.assertRaises(ValueError):
            repository.resume(result["digest"], (second,), "batch")
        with self.assertRaises(ValueError):
            repository.save(result["digest"], result["checkpoint"], "batch")
        # Nothing moved.
        self.assertEqual(repository.load()["digest"], result["digest"])

    def test_stale_digest_after_another_commit_raises_runtime_error(
        self,
    ) -> None:
        _directory, (first, second) = self.grow("ledger", (4, 2))
        repository = self.repo()
        first_result = repository.resume(None, (first,), "b0")
        second_result = repository.resume(
            first_result["digest"], (second,), "b1"
        )
        # A different request id reusing the old digest loses the CAS.
        with self.assertRaises(RuntimeError):
            repository.resume(
                first_result["digest"], (second,), "late"
            )
        self.assertEqual(
            repository.load()["digest"], second_result["digest"]
        )

    def _fork_pair(self):
        base = self.ledger("base")
        digest = self.append(base, self.entries[:4], None)
        left = self.ledger("left")
        right = self.ledger("right")
        shutil.copytree(base, left, dirs_exist_ok=True)
        shutil.copytree(base, right, dirs_exist_ok=True)
        self.append(left, self.entries[4:6], digest)
        other = tuple(
            (time_index, make_chain(transaction(100 + time_index)))
            for time_index in (5, 6)
        )
        self.append(right, other, digest)
        return (
            self.export(left),
            self.export(right, self.identities),
        )


class ConcurrencyTests(RepositoryTestBase):
    def _two_proofs(self):
        _directory, (first, second) = self.grow("ledger", (4, 2))
        return first, second

    def test_parallel_thread_commits_have_one_winner(self) -> None:
        first, second = self._two_proofs()
        path = self.path()
        bootstrap = BranchStore.create_proof_audit_checkpoint(())
        CheckpointRepository(path, None).save(None, bootstrap, "seed")
        seed_digest = CheckpointRepository(path, None).load()["digest"]
        outcomes = []

        def worker(index: int) -> None:
            repository = CheckpointRepository(path, 10)
            try:
                result = repository.resume(
                    seed_digest, (first,), f"worker-{index}"
                )
                outcomes.append(("ok", result["digest"]))
            except RuntimeError:
                outcomes.append(("lost", None))
            except BaseException as exc:  # pragma: no cover - report
                outcomes.append(("error", repr(exc)))

        threads = [
            threading.Thread(target=worker, args=(index,))
            for index in range(8)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        winners = [outcome for outcome in outcomes if outcome[0] == "ok"]
        losers = [outcome for outcome in outcomes if outcome[0] == "lost"]
        errors = [outcome for outcome in outcomes if outcome[0] == "error"]
        self.assertEqual(errors, [])
        self.assertEqual(len(winners), 1)
        self.assertEqual(len(losers), 7)
        self.assertEqual(len({digest for _, digest in winners}), 1)

    def test_parallel_retries_of_one_request_all_get_original(self) -> None:
        first, _second = self._two_proofs()
        path = self.path()
        bootstrap = BranchStore.create_proof_audit_checkpoint(())
        CheckpointRepository(path, None).save(None, bootstrap, "seed")
        seed_digest = CheckpointRepository(path, None).load()["digest"]
        outcomes = []

        def worker() -> None:
            try:
                result = CheckpointRepository(path, 10).resume(
                    seed_digest, (first,), "the-only-request"
                )
                outcomes.append(("ok", result["digest"]))
            except BaseException as exc:  # pragma: no cover - report
                outcomes.append(("error", repr(exc)))

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        # One continuation is committed and every racing retry is served
        # the same original result; none raises.
        self.assertEqual(
            [outcome[0] for outcome in outcomes], ["ok"] * 8
        )
        digests = {digest for _, digest in outcomes}
        self.assertEqual(len(digests), 1)
        self.assertEqual(
            CheckpointRepository(path, None).load()["digest"],
            next(iter(digests)),
        )

    def test_parallel_process_commits_have_one_winner(self) -> None:
        first, _second = self._two_proofs()
        path = self.path()
        bootstrap = BranchStore.create_proof_audit_checkpoint(())
        CheckpointRepository(path, None).save(None, bootstrap, "seed")
        seed_digest = CheckpointRepository(path, None).load()["digest"]
        env = dict(os.environ)
        env["PYTHONPATH"] = os.getcwd()
        payload = pickle.dumps((path, seed_digest, first)).hex()
        script = (
            "import os, pickle, sys\n"
            "from city_twin.branches import CheckpointRepository\n"
            "path, seed, proof = pickle.loads("
            "bytes.fromhex(os.environ['REPO_ARGS']))\n"
            "identity = os.environ['WORKER_ID']\n"
            "try:\n"
            "    result = CheckpointRepository(path, 30).resume(\n"
            "        seed, (proof,), 'worker-' + identity)\n"
            "    sys.stdout.write('OK ' + result['digest'])\n"
            "except RuntimeError:\n"
            "    sys.stdout.write('LOST')\n"
        )
        processes = []
        for index in range(4):
            worker_env = dict(env)
            worker_env["REPO_ARGS"] = payload
            worker_env["WORKER_ID"] = str(index)
            processes.append(
                subprocess.Popen(
                    [sys.executable, "-c", script],
                    env=worker_env,
                    stdout=subprocess.PIPE,
                    text=True,
                )
            )
        outputs = []
        for process in processes:
            stdout, _stderr = process.communicate()
            self.assertEqual(process.returncode, 0)
            outputs.append(stdout.strip())
        winners = {out for out in outputs if out.startswith("OK")}
        losers = [out for out in outputs if out == "LOST"]
        self.assertEqual(len(winners), 1)
        self.assertEqual(len(losers), 3)
        # The stored document is exactly the one winner's commit.
        self.assertEqual(
            CheckpointRepository(path, None).load()["digest"],
            next(iter(winners)).split(" ", 1)[1],
        )


class CrashSafetyTests(RepositoryTestBase):
    def _checkpoint(self):
        _directory, (proof,) = self.grow("ledger", (4,))
        return BranchStore.create_proof_audit_checkpoint((proof,))

    def test_abandoned_temp_files_are_reclaimed_under_lock(self) -> None:
        checkpoint = self._checkpoint()
        path = self.path()
        repository = self.repo()
        repository.save(None, checkpoint, "seed")
        leftovers = [
            ".checkpoint-repository-abandoned-1.tmp",
            ".checkpoint-repository-abandoned-2.tmp",
        ]
        for name in leftovers:
            with open(os.path.join(self.tmp.name, name), "wb") as h:
                h.write(b"interrupted bytes")
        loaded = repository.load()
        self.assertEqual(loaded["checkpoint"], checkpoint)
        for name in leftovers:
            self.assertFalse(
                os.path.exists(os.path.join(self.tmp.name, name))
            )

    def test_temp_file_never_masks_a_damaged_main_file(self) -> None:
        checkpoint = self._checkpoint()
        path = self.path()
        repository = self.repo()
        repository.save(None, checkpoint, "seed")
        # A plausible temp file exists (an interrupted next commit) and
        # the main file is corrupt: the corruption must surface, and the
        # temp must not be reclaimed as if the main file were sound.
        temp_path = os.path.join(
            self.tmp.name, ".checkpoint-repository-interrupted.tmp"
        )
        with open(temp_path, "wb") as handle:
            handle.write(b'{"format": "fake"}')
        with open(path, "wb") as handle:
            handle.write(b"{broken")
        with self.assertRaises(ValueError):
            repository.load()
        self.assertTrue(os.path.exists(temp_path))

    def test_unborn_repository_reclaims_first_commit_leftover(self) -> None:
        path = self.path()
        leftover = os.path.join(
            self.tmp.name, ".checkpoint-repository-first.tmp"
        )
        with open(leftover, "wb") as handle:
            handle.write(b"partial")
        with self.assertRaises(KeyError):
            self.repo().load()
        self.assertFalse(os.path.exists(leftover))
        self.assertFalse(os.path.exists(path))

    def test_interrupted_commit_leaves_previous_document_intact(self) -> None:
        first = self._checkpoint()
        second_document = json.loads(first)
        # A distinct, still authentic checkpoint (tamper the identities
        # then re-seal through the encoder is overkill; use a real one).
        _directory, (proof,) = self.grow("other", (4,))
        second = BranchStore.create_proof_audit_checkpoint((proof,))
        path = self.path()
        repository = self.repo()
        first_digest = repository.save(None, first, "seed")
        # Simulate a crash after the temp file was written but before
        # the atomic replace: only an orphan temp exists.
        temp_path = os.path.join(
            self.tmp.name, ".checkpoint-repository-orphan.tmp"
        )
        with open(temp_path, "wb") as handle:
            handle.write(second.encode("utf-8"))
        loaded = repository.load()
        self.assertEqual(loaded["checkpoint"], first)
        self.assertEqual(loaded["digest"], first_digest)
        self.assertFalse(os.path.exists(temp_path))

    def test_failed_write_cleans_temp_and_keeps_repository(self) -> None:
        checkpoint = self._checkpoint()
        path = self.path()
        repository = self.repo()
        digest = repository.save(None, checkpoint, "seed")
        # Point a repository instance at a directory whose temp-file
        # creation is impossible after the main file exists, by making
        # the parent directory non-writable.
        os.chmod(self.tmp.name, 0o500)
        try:
            with self.assertRaises(OSError):
                CheckpointRepository(path, None).save(
                    digest, checkpoint, "second"
                )
        finally:
            os.chmod(self.tmp.name, 0o700)
        # The old repository is whole and readable.
        self.assertEqual(self.repo().load()["digest"], digest)
        temps = [
            name
            for name in os.listdir(self.tmp.name)
            if name.startswith(".checkpoint-repository-")
            and name.endswith(".tmp")
        ]
        self.assertEqual(temps, [])


class PurityTests(RepositoryTestBase):
    def test_repository_never_touches_the_ledger(self) -> None:
        directory, (first, second) = self.grow("ledger", (4, 2), keep=1)
        snapshot = self.directory_files(directory)
        digest = self.digest_of(directory)
        repository = self.repo()
        result = repository.resume(None, (first,), "b0")
        # Idempotent retry of the still-current request changes nothing.
        repository.resume(result["digest"], (first,), "b0")
        advanced = repository.resume(
            result["digest"], (second,), "b1"
        )
        repository.resume(advanced["digest"], (second,), "b1")
        repository.resume(advanced["digest"], (), "query")
        with self.assertRaises(ValueError):
            # Same request id with a different batch is a conflict and
            # must not write.
            repository.resume(advanced["digest"], (first,), "b1")
        repository.load()
        self.assertEqual(self.directory_files(directory), snapshot)
        self.assertEqual(self.digest_of(directory), digest)
        self.assertEqual(self.export(directory), second)


class ResumeSealingFixTests(RepositoryTestBase):
    """The failed-terminal RuntimeError must be raised right after the
    checkpoint authenticates, before any new proof is read."""

    def _failed_checkpoint(self):
        base = self.ledger("base")
        digest = self.append(base, self.entries[:4], None)
        left = self.ledger("left")
        right = self.ledger("right")
        shutil.copytree(base, left, dirs_exist_ok=True)
        shutil.copytree(base, right, dirs_exist_ok=True)
        self.append(left, self.entries[4:6], digest)
        other = tuple(
            (time_index, make_chain(transaction(100 + time_index)))
            for time_index in (5, 6)
        )
        self.append(right, other, digest)
        before = self.export(left)
        after = self.export(right, self.identities)
        sealed = BranchStore.create_proof_audit_checkpoint(
            (before, after)
        )
        return sealed, before

    def test_non_empty_batch_raises_runtime_error_without_reading_proofs(
        self,
    ) -> None:
        sealed, before = self._failed_checkpoint()
        # These would raise ValueError if they were authenticated first.
        for batch in (("",), ("not json",), (before + " ",)):
            with self.subTest(batch=batch[:1]):
                with self.assertRaises(RuntimeError):
                    BranchStore.resume_proof_audit(sealed, batch)

    def test_plain_type_errors_still_surface_before_the_checkpoint(
        self,
    ) -> None:
        sealed, _before = self._failed_checkpoint()
        with self.assertRaises(TypeError):
            BranchStore.resume_proof_audit(sealed, [])
        with self.assertRaises(TypeError):
            BranchStore.resume_proof_audit(sealed, (1,))
        with self.assertRaises(TypeError):
            BranchStore.resume_proof_audit(1, ())

    def test_empty_continuation_still_read_only(self) -> None:
        sealed, _before = self._failed_checkpoint()
        read = BranchStore.resume_proof_audit(sealed, ())
        self.assertEqual(read["checkpoint"], sealed)
        self.assertEqual(read["audit"]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
