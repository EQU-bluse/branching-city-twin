"""Named branches over a shared event graph."""

from __future__ import annotations

import contextlib
import errno
import hashlib
import heapq
import hmac
import itertools
import json
import math
import os
import stat
import tempfile
import threading
import time
import uuid
from typing import Any

from city_twin.event_graph import EventGraph

if os.name == "nt":
    import msvcrt
else:
    import fcntl


class _SnapshotEntry:
    """One snapshot token's frozen view and remaining read allowance.

    ``store`` is a detached :class:`BranchStore` holding deep copies of the
    creating store's event graph, heads and records at creation time; it is
    never mutated afterwards. ``remaining`` counts successful queries only
    and reaches zero exactly when the token expires. Read accounting and
    expiry are decided under the owning store's snapshot lock.
    """

    __slots__ = ("store", "remaining")

    def __init__(self, store: "BranchStore", max_reads: int) -> None:
        self.store = store
        self.remaining = max_reads


class _CanonicalJsonStream:
    """Incremental tokenizer for canonical compact JSON documents.

    Wraps a binary file object and accepts exactly the byte streams
    :meth:`BranchStore._canonical_json` produces: no whitespace anywhere,
    strings escaped exactly as ``json.dumps(..., ensure_ascii=False)``
    emits them and numbers in their canonical re-serialized form. Every
    deviation raises :class:`ValueError`, so an accepted document equals
    its own canonical re-serialization byte for byte. Only a fixed read
    buffer plus the current token are ever held in memory -- never the
    whole document -- and :attr:`offset` tracks the absolute byte
    position so callers can record where each parsed value starts and
    ends.
    """

    #: Bytes read from the underlying file per refill.
    _CHUNK_SIZE = 65536

    __slots__ = ("_file", "_buffer", "_start", "_pos", "_eof", "_consume")

    def __init__(self, fileobj: Any, consume_hasher: Any = None) -> None:
        self._file = fileobj
        self._buffer = b""
        self._start = 0
        self._pos = 0
        self._eof = False
        # Optional hasher fed the raw bytes strictly in the order they
        # are consumed (parsed), never the look-ahead still buffered, so
        # a caller can snapshot the digest of any parsed prefix without
        # holding the chain -- the hasher always contains exactly the
        # bytes from the start through :attr:`offset` once buffered but
        # unflushed bytes are included via :meth:`consumed_digest`.
        self._consume = consume_hasher

    @property
    def offset(self) -> int:
        """Absolute byte offset of the next unparsed byte."""
        return self._start + self._pos

    def _fill(self) -> None:
        if self._eof:
            return
        if self._pos:
            if self._consume is not None:
                self._consume.update(self._buffer[: self._pos])
            self._buffer = self._buffer[self._pos:]
            self._start += self._pos
            self._pos = 0
        chunk = self._file.read(self._CHUNK_SIZE)
        if chunk:
            self._buffer += chunk
        else:
            self._eof = True

    def consumed_digest(self) -> str:
        """Hex digest of every raw byte consumed up to the current
        :attr:`offset`, including bytes already parsed but still held in
        the refill buffer. The running hasher is not modified."""
        if self._consume is None:
            raise ValueError("no consume hasher is attached")
        snapshot = self._consume.copy()
        if self._pos:
            snapshot.update(self._buffer[: self._pos])
        return snapshot.hexdigest()

    def peek(self) -> int | None:
        """The next byte without consuming it, or ``None`` at the end."""
        while self._pos >= len(self._buffer) and not self._eof:
            self._fill()
        if self._pos >= len(self._buffer):
            return None
        return self._buffer[self._pos]

    def take(self) -> int:
        """Consume and return the next byte; the end of the input raises
        :class:`ValueError`."""
        byte = self.peek()
        if byte is None:
            raise ValueError(
                "recovery chain is not valid JSON: unexpected end of "
                "document"
            )
        self._pos += 1
        return byte

    def at_end(self) -> bool:
        """Whether every byte of the input has been consumed."""
        return self.peek() is None

    def skip_raw(self, size: int) -> None:
        """Advance exactly ``size`` raw bytes without tokenizing them.

        Only used to cross an already hash-bound authenticated prefix;
        the skipped bytes flow through the consume hasher exactly like
        parsed bytes (flushed on refill or counted by
        :meth:`consumed_digest` while buffered), leaving the stream at a
        known token boundary. Running past the end raises
        :class:`ValueError`; only the fixed refill buffer is held."""
        remaining = size
        while remaining:
            while self._pos >= len(self._buffer) and not self._eof:
                self._fill()
            available = len(self._buffer) - self._pos
            if not available:
                raise ValueError(
                    "recovery chain is not valid JSON: unexpected end of "
                    "document"
                )
            step = min(remaining, available)
            self._pos += step
            remaining -= step

    def expect(self, byte: int) -> None:
        """Consume one byte that must equal ``byte``."""
        if self.take() != byte:
            raise ValueError(
                "recovery chain is not valid JSON: unexpected content"
            )

    def expect_literal(self, word: bytes) -> None:
        """Consume exactly the bytes of ``word``."""
        for expected in word:
            if self.take() != expected:
                raise ValueError(
                    "recovery chain is not valid JSON: invalid literal"
                )

    def parse_string(self) -> str:
        """Parse one string token (the current byte must be the opening
        quote), validating its UTF-8 content and its canonical escaping,
        and return the decoded value."""
        raw = bytearray()
        raw.append(self.take())
        while True:
            byte = self.take()
            raw.append(byte)
            if byte == 0x22:  # '"'
                break
            if byte == 0x5C:  # '\\'
                raw.append(self.take())
        try:
            text = bytes(raw).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"recovery chain is not valid UTF-8: {exc}"
            ) from exc
        try:
            value = json.loads(text)
        except ValueError as exc:
            raise ValueError(
                f"recovery chain is not valid JSON: {exc}"
            ) from exc
        if json.dumps(value, ensure_ascii=False, allow_nan=False) != text:
            raise ValueError(
                "recovery chain is not canonical compact JSON"
            )
        return value

    def parse_number(self) -> int | float:
        """Parse one number token, validating its canonical form, and
        return the decoded :class:`int` or :class:`float`."""
        raw = bytearray()
        while True:
            byte = self.peek()
            if byte is None or not (
                0x30 <= byte <= 0x39 or byte in b"+-.eE"
            ):
                break
            raw.append(self.take())
        if not raw:
            raise ValueError(
                "recovery chain is not valid JSON: expected a number"
            )
        text = bytes(raw).decode("ascii")
        try:
            value = json.loads(text)
        except ValueError as exc:
            raise ValueError(
                f"recovery chain is not valid JSON: {exc}"
            ) from exc
        if json.dumps(value, ensure_ascii=False, allow_nan=False) != text:
            raise ValueError(
                "recovery chain is not canonical compact JSON"
            )
        return value

    def skip_value(self) -> None:
        """Parse and discard one arbitrary JSON value, still enforcing
        canonical encoding and rejecting duplicate object keys, without
        materializing the value."""
        byte = self.peek()
        if byte == 0x22:  # '"'
            self.parse_string()
            return
        if byte == 0x7B:  # '{'
            self.take()
            if self.peek() == 0x7D:  # '}'
                self.take()
                return
            seen: set[str] = set()
            while True:
                if self.peek() != 0x22:
                    raise ValueError(
                        "recovery chain is not valid JSON: expected an "
                        "object key"
                    )
                key = self.parse_string()
                if key in seen:
                    raise ValueError(
                        f"duplicate key {key!r} in JSON object"
                    )
                seen.add(key)
                self.expect(0x3A)  # ':'
                self.skip_value()
                byte = self.take()
                if byte == 0x2C:  # ','
                    continue
                if byte == 0x7D:  # '}'
                    return
                raise ValueError(
                    "recovery chain is not valid JSON: unexpected content"
                )
        if byte == 0x5B:  # '['
            self.take()
            if self.peek() == 0x5D:  # ']'
                self.take()
                return
            while True:
                self.skip_value()
                byte = self.take()
                if byte == 0x2C:  # ','
                    continue
                if byte == 0x5D:  # ']'
                    return
                raise ValueError(
                    "recovery chain is not valid JSON: unexpected content"
                )
        if byte == 0x74:  # 't'
            self.expect_literal(b"true")
            return
        if byte == 0x66:  # 'f'
            self.expect_literal(b"false")
            return
        if byte == 0x6E:  # 'n'
            self.expect_literal(b"null")
            return
        if byte is not None and (0x30 <= byte <= 0x39 or byte == 0x2D):
            self.parse_number()
            return
        raise ValueError(
            "recovery chain is not valid JSON: unexpected content"
        )


class _ChainSegmenter:
    """Fold an authenticated recovery-chain frame stream into segments.

    Frames arrive in sequence order via :meth:`on_frame`; every time
    ``segment_size`` consecutive frames have accumulated, one segment
    record is emitted through the ``on_segment`` callback and the
    accumulator resets, so only one frame's contribution is ever held.
    A segment record carries the first and last sequence numbers, the
    boundary frame digests and the SHA-256 of the segment's canonical
    frame bytes (the concatenation of each frame's canonical JSON, in
    order). :meth:`finish` emits the final, possibly short, segment.
    """

    __slots__ = (
        "_size",
        "_canonical",
        "_on_segment",
        "_count",
        "_first",
        "_first_digest",
        "_last",
        "_last_digest",
        "_hasher",
    )

    def __init__(
        self, segment_size: int, canonical: Any, on_segment: Any
    ) -> None:
        self._size = segment_size
        self._canonical = canonical
        self._on_segment = on_segment
        self._count = 0
        self._first = 0
        self._first_digest = ""
        self._last = 0
        self._last_digest = ""
        self._hasher = hashlib.sha256()

    def _reset(self) -> None:
        self._count = 0
        self._hasher = hashlib.sha256()

    def _emit(self) -> None:
        self._on_segment(
            {
                "first": self._first,
                "last": self._last,
                "first_digest": self._first_digest,
                "last_digest": self._last_digest,
                "digest": self._hasher.hexdigest(),
            }
        )
        self._reset()

    def on_frame(self, frame: dict[str, Any]) -> None:
        """Fold one authenticated frame into the current segment."""
        if self._count == self._size:
            self._emit()
        if not self._count:
            self._first = frame["seq"]
            self._first_digest = frame["digest"]
        self._last = frame["seq"]
        self._last_digest = frame["digest"]
        self._hasher.update(
            self._canonical(
                {
                    "seq": frame["seq"],
                    "prev": frame["prev"],
                    "record": frame["record"],
                    "digest": frame["digest"],
                }
            ).encode("utf-8")
        )
        self._count += 1

    def finish(self) -> None:
        """Emit the trailing segment when one is partially filled."""
        if self._count:
            self._emit()


class _RecoveryResumeMismatch(ValueError):
    """Internal signal: a resumable build's chain prefix parses but
    does not match the authenticated boundary recorded in progress.

    A genuinely malformed chain (a broken digest link, a non-canonical
    document or truncation before the boundary) raises plain
    :class:`ValueError`; this marker means the chain is internally
    sound yet forks from the recorded prefix, so the caller performs
    its one permitted full rebuild and re-checks the anchor before
    rejecting it.
    """


class BranchStore:
    """Named heads into a shared :class:`EventGraph`.

    A branch is created from an existing event and grows only by appending
    events whose sole parent is the branch's current head, so branches
    never move each other's heads. Appends are idempotent: the store
    records the ``(at, changes)`` each branch appended under an event id,
    and repeating the same call is a no-op. All validation finishes before
    any state is touched, so a failed call leaves the graph, branch heads
    and idempotency records unchanged.
    """

    #: Maximum number of live (unreleased) snapshot tokens per store.
    MAX_SNAPSHOTS = 32

    def __init__(self, graph: EventGraph) -> None:
        if not isinstance(graph, EventGraph):
            raise TypeError(
                f"graph must be an EventGraph, got {type(graph).__name__}"
            )
        self._graph = graph
        self._heads: dict[str, str] = {}
        self._sources: dict[str, str] = {}
        self._appends: dict[str, dict[str, tuple[int, dict[str, int]]]] = {}
        self._merges: dict[str, tuple[str, str, int, dict[str, int]]] = {}
        # token -> SnapshotEntry; only create_snapshot/release_snapshot and
        # the snapshot-aware lifetime queries touch it.
        self._snapshots: dict[str, "_SnapshotEntry"] = {}
        # Guards only the token registry and each entry's read accounting, so
        # a read reservation and the expiry decision stay one atomic step.
        self._snapshot_lock = threading.Lock()
        # Guards the mutable business state -- graph, heads, sources and the
        # append/merge records. create/append/merge run entirely under it and
        # checkpoint/snapshot capture copies under it, so a captured view can
        # never see the graph and records mid-commit. Journal commits also
        # run under it, so the journal's frame order is exactly the order in
        # which heads, audit and idempotency records commit.
        self._state_lock = threading.Lock()
        # Journal attachment: the path is None until enable_journal attaches
        # a journal (or a complete load_journal recovery re-attaches one).
        # The checkpoint checksum and frame list mirror the journal file's
        # binding and frame sequence exactly; both are meaningless while
        # the path is None. The version is the attached file's schema
        # version: new attachments use the current format, an attachment
        # recovered from an old journal keeps writing that old format.
        self._journal_path: str | None = None
        self._journal_checkpoint: str | None = None
        self._journal_checkpoint_path: str | None = None
        self._journal_version: int = self._JOURNAL_VERSION
        self._journal_frames: list[dict[str, Any]] = []
        # Generation context: set only while the attachment lives inside
        # a rotation generation (rotate_generation or a
        # load_latest_generation recovery), so every committed frame also
        # re-points the generation's manifest at the new journal digest.
        self._journal_manifest_path: str | None = None
        self._journal_manifest: dict[str, Any] | None = None

    @staticmethod
    def _require_nonempty_str(value: Any, name: str) -> None:
        # Same contract as EventGraph.add's string parameters.
        EventGraph._require_nonempty_str(value, name)

    def _require_known_branch(self, name: str) -> None:
        if name not in self._heads:
            raise KeyError(name)

    def create(self, name: str, from_event: str) -> None:
        """Create a branch named ``name`` starting at ``from_event``.

        Re-creating with the same name and source is a no-op; the same
        name with a different source is a conflict.
        """
        self._require_nonempty_str(name, "name")
        self._require_nonempty_str(from_event, "from_event")

        with self._state_lock:
            if name in self._heads:
                if self._sources[name] != from_event:
                    raise ValueError(
                        f"branch {name!r} already exists from "
                        f"{self._sources[name]!r}, not {from_event!r}"
                    )
                return
            if from_event not in self._graph._at:
                raise KeyError(from_event)

            # Apply the internal state change first so the frame's after
            # summary captures it, then commit durably; on a write failure
            # roll the internal change back. Everything happens under the
            # state lock, so the frame and the state become visible
            # together or not at all.
            before_state = (
                self._snapshot_state()
                if self._journal_path is not None
                else None
            )
            self._heads[name] = from_event
            self._sources[name] = from_event
            self._appends[name] = {}
            if self._journal_path is not None:
                after_state = self._snapshot_state()
                try:
                    self._commit_journal_frame(
                        "create",
                        {"name": name, "from_event": from_event},
                        before_state,
                        after_state,
                    )
                except BaseException:
                    del self._heads[name]
                    del self._sources[name]
                    del self._appends[name]
                    raise

    def append(
        self,
        name: str,
        id: str,
        at: int,
        changes: dict[str, int],
    ) -> None:
        """Append an event whose sole parent is the branch's current head.

        The head moves only if :meth:`EventGraph.add` succeeds. Repeating
        an append with the same ``name``/``id``/``at``/``changes`` is a
        no-op; reusing ``id`` with different inputs is a conflict, here
        and via the graph across branches.
        """
        # --- Full validation happens before any state is consulted. ---
        self._require_nonempty_str(name, "name")
        self._require_nonempty_str(id, "id")
        # The at/changes contract is exactly EventGraph.add's.
        at_value = EventGraph._require_int(at, "at")
        if at_value < 0:
            raise ValueError("at must be a non-negative int")
        if not isinstance(changes, dict):
            raise TypeError(
                f"changes must be a dict, got {type(changes).__name__}"
            )
        for key, value in changes.items():
            EventGraph._require_nonempty_str(key, "change key")
            EventGraph._require_int(value, f"change value for {key!r}")
        # Copy caller-owned input before it can be recorded anywhere.
        changes = dict(changes)

        with self._state_lock:
            self._require_known_branch(name)

            recorded = self._appends[name].get(id)
            if recorded is not None:
                if recorded == (at_value, changes):
                    return
                raise ValueError(
                    f"event {id!r} already appended on branch {name!r} "
                    f"with different inputs"
                )

            head = self._heads[name]
            # Snapshot the pre-commit state before the graph sees the new
            # event so the frame's before/after summaries bracket exactly
            # this operation; an unattached store writes no frame.
            before_state = (
                self._snapshot_state()
                if self._journal_path is not None
                else None
            )
            added_to_graph = id not in self._graph._at
            self._graph.add(id, at_value, (head,), changes)
            previous_head = self._heads[name]
            self._heads[name] = id
            self._appends[name][id] = (at_value, changes)

            # --- Commit: the journal frame is durably written while the
            # state changes are already in place; a write failure rolls the
            # graph add, head move and idempotency record back so business
            # state and the old journal stay consistent. Everything runs
            # under the state lock, so nothing observes the interim state.
            if self._journal_path is not None:
                after_state = self._snapshot_state()
                try:
                    self._commit_journal_frame(
                        "append",
                        {
                            "name": name,
                            "id": id,
                            "at": at_value,
                            "changes": {
                                key: changes[key] for key in sorted(changes)
                            },
                        },
                        before_state,
                        after_state,
                    )
                except BaseException:
                    if added_to_graph:
                        del self._graph._at[id]
                        del self._graph._parents[id]
                        del self._graph._changes[id]
                    self._heads[name] = previous_head
                    del self._appends[name][id]
                    raise

    def merge(
        self,
        target: str,
        source: str,
        id: str,
        at: int,
        changes: dict[str, int],
    ) -> None:
        """Merge ``source`` into ``target`` under a new two-parent event.

        The event's parents are ordered ``(target head, source head)``;
        only the target head moves, the source head is untouched. Calls
        are idempotent on the exact five-tuple, and reusing ``id`` with
        different inputs (here or in the graph) is a conflict. A merge is
        rejected when events exclusive to either side touch overlapping
        change keys. All validation finishes before any state is touched,
        so a failed call leaves the graph, heads and records unchanged.
        """
        # --- Full validation happens before any state is consulted. ---
        self._require_nonempty_str(target, "target")
        self._require_nonempty_str(source, "source")
        self._require_nonempty_str(id, "id")
        at_value = EventGraph._require_int(at, "at")
        if at_value < 0:
            raise ValueError("at must be a non-negative int")
        if not isinstance(changes, dict):
            raise TypeError(
                f"changes must be a dict, got {type(changes).__name__}"
            )
        for key, value in changes.items():
            EventGraph._require_nonempty_str(key, "change key")
            EventGraph._require_int(value, f"change value for {key!r}")
        # Copy caller-owned input before it can be recorded anywhere.
        changes = dict(changes)

        with self._state_lock:
            self._require_known_branch(target)
            self._require_known_branch(source)
            if target == source:
                raise ValueError(f"cannot merge branch {target!r} into itself")

            recorded = self._merges.get(id)
            if recorded is not None:
                if recorded == (target, source, at_value, changes):
                    return
                raise ValueError(
                    f"merge event {id!r} already recorded with different inputs"
                )

            target_head = self._heads[target]
            source_head = self._heads[source]

            target_closure = set(self._graph._ordered_ancestors(target_head))
            source_closure = set(self._graph._ordered_ancestors(source_head))
            common = target_closure & source_closure

            target_keys: set[str] = set()
            for event_id in target_closure - common:
                target_keys.update(self._graph._changes[event_id])
            source_keys: set[str] = set()
            for event_id in source_closure - common:
                source_keys.update(self._graph._changes[event_id])
            overlap = target_keys & source_keys
            if overlap:
                raise ValueError(
                    "conflicting changes on keys: "
                    + ", ".join(sorted(overlap))
                )

            # Snapshot the pre-commit state before the graph sees the new
            # event so the frame's before/after summaries bracket exactly
            # this operation; an unattached store writes no frame.
            before_state = (
                self._snapshot_state()
                if self._journal_path is not None
                else None
            )
            added_to_graph = id not in self._graph._at
            self._graph.add(id, at_value, (target_head, source_head), changes)
            previous_target_head = self._heads[target]
            self._heads[target] = id
            self._merges[id] = (target, source, at_value, changes)

            # --- Commit: the journal frame is durably written while the
            # state changes are already in place; a write failure rolls the
            # graph add, head move and merge record back so business state
            # and the old journal stay consistent. Everything runs under
            # the state lock, so nothing observes the interim state.
            if self._journal_path is not None:
                after_state = self._snapshot_state()
                try:
                    self._commit_journal_frame(
                        "merge",
                        {
                            "target": target,
                            "source": source,
                            "id": id,
                            "at": at_value,
                            "changes": {
                                key: changes[key] for key in sorted(changes)
                            },
                        },
                        before_state,
                        after_state,
                    )
                except BaseException:
                    if added_to_graph:
                        del self._graph._at[id]
                        del self._graph._parents[id]
                        del self._graph._changes[id]
                    self._heads[target] = previous_target_head
                    del self._merges[id]
                    raise

    def create_snapshot(self, max_reads: int) -> str:
        """Freeze the current graph, heads and records into a snapshot token.

        The token names a detached copy of everything the read-only
        lifetime queries consult: the full event graph (timestamps, parent
        tuples and per-event changes), every branch head, the append and
        merge records that form the audit log and idempotency keys. Later
        appends, merges or branch creation on this store never reach the
        token's view, and the snapshot never shares a mutable object with
        live state, so neither side can mutate the other.

        ``max_reads`` must be a non-``bool`` :class:`int` (else
        :class:`TypeError`) of at least one (else :class:`ValueError`);
        it bounds how many token-bearing lifetime queries may succeed.
        At most :attr:`MAX_SNAPSHOTS` (32) unreleased tokens -- expired
        ones included -- may exist at once; reaching the cap raises
        :class:`RuntimeError` without evicting an older token or
        disturbing any existing one. Snapshot creation is read-only:
        success or failure never modifies the event graph, branch heads,
        audit or idempotency records.
        """
        if isinstance(max_reads, bool) or not isinstance(max_reads, int):
            raise TypeError(
                f"max_reads must be an int, got {type(max_reads).__name__}"
            )
        if max_reads < 1:
            raise ValueError("max_reads must be >= 1")

        with self._snapshot_lock:
            # The cap is enforced before anything is copied, and expired
            # tokens are still unreleased, so they keep their slot until
            # release_snapshot frees it.
            if len(self._snapshots) >= self.MAX_SNAPSHOTS:
                raise RuntimeError(
                    f"snapshot token limit reached: at most "
                    f"{self.MAX_SNAPSHOTS} unreleased tokens per store"
                )
            with self._state_lock:
                state = self._snapshot_state()
            frozen = self._store_from_snapshot(state)
            while True:
                token = uuid.uuid4().hex
                if token not in self._snapshots:
                    break
            self._snapshots[token] = _SnapshotEntry(frozen, max_reads)
        return token

    def release_snapshot(self, token: str) -> None:
        """Release a snapshot token, valid or expired, and free its slot.

        ``token`` must be a non-empty :class:`str` (a non-``str`` raises
        :class:`TypeError`, an empty string :class:`ValueError`); an
        unknown token, one already released or one owned by another store
        raises :class:`KeyError`. Releasing a live token and releasing an
        expired token both succeed and return ``None``; the slot becomes
        reusable immediately and the token can never be restored. The
        release never modifies the event graph, branch heads, audit or
        idempotency records.
        """
        self._require_token_str(token)
        with self._snapshot_lock:
            entry = self._snapshots.pop(token, None)
        if entry is None:
            raise KeyError(token)
        return None

    @staticmethod
    def _require_token_str(token: Any) -> None:
        """Validate the token parameter's type and emptiness."""
        if not isinstance(token, str):
            raise TypeError(
                f"token must be a str, got {type(token).__name__}"
            )
        if not token:
            raise ValueError("token must be a non-empty str")

    def _reserve_snapshot_read(self, token: str) -> "BranchStore":
        """Validate ``token`` and atomically reserve one successful read.

        Runs only after every ordinary query parameter has been validated.
        The type/emptiness checks precede the registry lookup: a non-``str``
        token raises :class:`TypeError`, an empty string raises
        :class:`ValueError`, and an unknown, released or foreign token
        raises :class:`KeyError`. A token whose allowance is exhausted is
        expired: the query raises :class:`RuntimeError` without consuming
        anything. Otherwise the allowance is decremented and the expiry
        decision is made in the same locked step, so at most ``max_reads``
        concurrent queries can ever succeed. Returns the token's frozen,
        detached :class:`BranchStore` view.
        """
        self._require_token_str(token)
        with self._snapshot_lock:
            entry = self._snapshots.get(token)
            if entry is None:
                raise KeyError(token)
            if entry.remaining <= 0:
                raise RuntimeError(
                    "snapshot token has expired: read allowance exhausted"
                )
            entry.remaining -= 1
            return entry.store

    def _refund_snapshot_read(self, token: str) -> None:
        """Give a failed query's reserved read back to its token.

        Query failures never consume the read allowance. A token released
        in the meantime has no entry left and needs no refund; an expired
        allowance restored here makes the token usable again, exactly as
        if the failing query had never run.
        """
        with self._snapshot_lock:
            entry = self._snapshots.get(token)
            if entry is not None:
                entry.remaining += 1

    def _snapshot_state(self) -> dict[str, object]:
        """Deep-copy the mutable business state as one consistent instant.

        Must run while holding :attr:`_state_lock`, which serializes the
        copying against create/append/merge commits, so the event graph,
        branch heads and sources and the append/merge records can never be
        observed mid-commit. Every timestamp, parent tuple, change dict,
        head, source and record is copied; the result shares no mutable
        object with live state.
        """
        at: dict[str, int] = dict(self._graph._at)
        parents: dict[str, tuple[str, ...]] = dict(self._graph._parents)
        changes: dict[str, dict[str, int]] = {
            event_id: dict(event_changes)
            for event_id, event_changes in self._graph._changes.items()
        }
        return {
            "at": at,
            "parents": parents,
            "changes": changes,
            "heads": dict(self._heads),
            "sources": dict(self._sources),
            "appends": {
                name: {
                    event_id: (at_value, dict(event_changes))
                    for event_id, (at_value, event_changes) in appends.items()
                }
                for name, appends in self._appends.items()
            },
            "merges": {
                event_id: (target, source, at_value, dict(event_changes))
                for event_id, (
                    target,
                    source,
                    at_value,
                    event_changes,
                ) in self._merges.items()
            },
        }

    @staticmethod
    def _store_from_snapshot(state: dict[str, object]) -> "BranchStore":
        """Build a detached store from a snapshot captured by
        :meth:`_snapshot_state` (or equivalent validated restore data).

        The result shares no mutable object with ``state`` either, so
        later mutations of either side stay isolated.
        """
        graph = EventGraph()
        at = state["at"]
        parents = state["parents"]
        changes = state["changes"]
        for event_id in at:
            graph._at[event_id] = at[event_id]
            graph._parents[event_id] = tuple(parents[event_id])
            graph._changes[event_id] = dict(changes[event_id])

        frozen = BranchStore(graph)
        frozen._heads = dict(state["heads"])
        frozen._sources = dict(state["sources"])
        frozen._appends = {
            name: {
                event_id: (at_value, dict(event_changes))
                for event_id, (at_value, event_changes) in appends.items()
            }
            for name, appends in state["appends"].items()
        }
        frozen._merges = {
            event_id: (target, source, at_value, dict(event_changes))
            for event_id, (
                target,
                source,
                at_value,
                event_changes,
            ) in state["merges"].items()
        }
        return frozen

    # ------------------------------------------------------------------
    # Versioned, checksummed persistence
    # ------------------------------------------------------------------

    #: Checkpoint format version emitted by :meth:`save_checkpoint`.
    _CHECKPOINT_VERSION = 1
    _DOCUMENT_KEYS = ("schema_version", "payload", "checksum")
    _PAYLOAD_KEYS = ("events", "branches", "appends", "merges")
    _EVENT_KEYS = ("id", "at", "parents", "changes")
    _BRANCH_KEYS = ("name", "source", "head")
    _APPEND_RECORD_KEYS = ("at", "changes")
    _MERGE_RECORD_KEYS = ("target", "source", "at", "changes")
    _HEX_DIGITS = frozenset("0123456789abcdef")

    def save_checkpoint(self, path: Any) -> None:
        """Persist the store's business state to a versioned checkpoint file.

        The file is UTF-8 (no BOM), compact JSON with no trailing newline;
        its top-level keys are ordered ``schema_version, payload,
        checksum`` with ``schema_version`` equal to 1. The payload stores,
        in order, the event graph, the branch table (source and head), and
        the append and merge idempotency/audit records. Event, branch and
        record identifiers and change keys are sorted by Unicode code
        point; parent ids keep their recorded order. The checksum is the
        lowercase hex SHA-256 of the payload's canonical JSON bytes. Equal
        business state always produces a byte-for-byte identical file.

        ``path`` must be a non-empty :class:`str`: a non-``str`` raises
        :class:`TypeError` and an empty string :class:`ValueError`, both
        before any state is consulted. The state is copied as one
        consistent instant under the same lock that serializes
        create/append/merge, so concurrent mutations cannot leave the
        graph and records misaligned in the captured view, and neither
        success nor failure changes any business state; snapshot tokens,
        read allowances and release status are never persisted.

        The bytes are written to a temporary file in the same directory,
        flushed and ``fsync``-ed, then atomically moved onto ``path``; if
        anything fails before the move, the previous target (if any) is
        left byte-for-byte untouched and the temporary file is removed.
        Open, read/write, flush or replace failures raise
        :class:`OSError`.
        """
        if not isinstance(path, str):
            raise TypeError(
                f"path must be a str, got {type(path).__name__}"
            )
        if not path:
            raise ValueError("path must be a non-empty str")

        with self._state_lock:
            state = self._snapshot_state()
        payload = self._checkpoint_payload(state)
        payload_bytes = self._canonical_json(payload).encode("utf-8")
        checksum = hashlib.sha256(payload_bytes).hexdigest()
        document_bytes = self._canonical_json(
            {
                "schema_version": self._CHECKPOINT_VERSION,
                "payload": payload,
                "checksum": checksum,
            }
        ).encode("utf-8")

        directory = os.path.dirname(os.path.abspath(path))
        fd, tmp_path = tempfile.mkstemp(
            prefix=".checkpoint-", suffix=".tmp", dir=directory
        )
        replaced = False
        try:
            try:
                handle = os.fdopen(fd, "wb")
            except BaseException:
                # fdopen only reaches here without taking ownership of fd.
                with contextlib.suppress(OSError):
                    os.close(fd)
                raise
            with handle:
                handle.write(document_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, path)
            replaced = True
        finally:
            if not replaced:
                with contextlib.suppress(OSError):
                    os.remove(tmp_path)
        return None

    @classmethod
    def load_checkpoint(cls, path: Any) -> "BranchStore":
        """Restore a :class:`BranchStore` from a file written by
        :meth:`save_checkpoint`.

        ``path`` must be a non-empty :class:`str`: a non-``str`` raises
        :class:`TypeError` and an empty string :class:`ValueError`, both
        before the file is opened; filesystem failures (missing file,
        permissions, read errors) propagate as :class:`OSError`. Every
        other defect -- invalid UTF-8 or BOM, malformed JSON, wrong
        top-level shape, a missing/extra field, a wrong type, an empty
        identifier, a ``bool`` posing as an integer, a negative
        timestamp, an unsupported :class:`schema_version`, a checksum
        mismatch, duplicate events or parents, unknown parents, cycles,
        unknown branch heads or sources, append/merge records that do not
        match the events, parent order or branch lineage -- raises
        :class:`ValueError`. The checksum is verified against the
        payload before any object is built.

        On failure nothing partial is returned and the file is never
        modified; on success the returned store is a fresh instance whose
        replay, audit, conflict and idempotency behavior is identical to
        the saved one, but which shares no mutable state with the saved
        instance or the parsed JSON. Snapshot tokens are not persisted,
        so the restored store's token table is always empty and its read
        allowances start fresh.
        """
        if not isinstance(path, str):
            raise TypeError(
                f"path must be a str, got {type(path).__name__}"
            )
        if not path:
            raise ValueError("path must be a non-empty str")

        state, _checksum = cls._read_checkpoint_state(path)
        return cls._store_from_snapshot(state)

    @classmethod
    def _read_checkpoint_state(
        cls, path: str
    ) -> tuple[dict[str, object], str]:
        """Read, parse and fully validate a checkpoint file.

        Returns the validated business state and the document's verified
        checksum. Filesystem failures propagate as :class:`OSError`; every
        content defect raises :class:`ValueError`. The caller validates
        ``path``'s type and emptiness.
        """
        with open(path, "rb") as handle:
            raw = handle.read()

        if raw.startswith(b"\xef\xbb\xbf"):
            raise ValueError("checkpoint must be UTF-8 without a BOM")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"checkpoint is not valid UTF-8: {exc}") from exc
        # The hook raises ValueError (not JSONDecodeError) on duplicate keys;
        # it propagates directly, so both parse failures are ValueError.
        try:
            document = json.loads(
                text, object_pairs_hook=cls._reject_duplicate_json_keys
            )
        except json.JSONDecodeError as exc:
            raise ValueError(f"checkpoint is not valid JSON: {exc}") from exc

        state = cls._parse_checkpoint_document(document)
        return state, document["checksum"]

    @staticmethod
    def _canonical_json(value: Any) -> str:
        """Compact, deterministic JSON: no whitespace, non-ASCII literal,
        no non-finite floats and no trailing newline."""
        return json.dumps(
            value,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )

    @staticmethod
    def _reject_duplicate_json_keys(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        """``object_pairs_hook`` treating duplicate JSON keys as invalid."""
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key!r} in JSON object")
            result[key] = value
        return result

    @classmethod
    def _checkpoint_payload(cls, state: dict[str, object]) -> dict[str, object]:
        """Build the canonical, deterministically ordered payload object."""
        at_values: dict[str, int] = state["at"]  # type: ignore[assignment]
        parent_map: dict[str, tuple[str, ...]] = state["parents"]  # type: ignore[assignment]
        change_map: dict[str, dict[str, int]] = state["changes"]  # type: ignore[assignment]
        events = [
            {
                "id": event_id,
                "at": at_values[event_id],
                "parents": list(parent_map[event_id]),
                "changes": {
                    key: change_map[event_id][key]
                    for key in sorted(change_map[event_id])
                },
            }
            for event_id in sorted(at_values)
        ]

        heads: dict[str, str] = state["heads"]  # type: ignore[assignment]
        sources: dict[str, str] = state["sources"]  # type: ignore[assignment]
        branches = [
            {
                "name": name,
                "source": sources[name],
                "head": heads[name],
            }
            for name in sorted(heads)
        ]

        appends_map: dict[str, dict[str, tuple[int, dict[str, int]]]] = state[  # type: ignore[assignment]
            "appends"
        ]
        appends = {
            name: {
                event_id: {
                    "at": at_value,
                    "changes": {
                        key: event_changes[key]
                        for key in sorted(event_changes)
                    },
                }
                for event_id, (
                    at_value,
                    event_changes,
                ) in sorted(appends_map[name].items())
            }
            for name in sorted(appends_map)
        }

        merges_map: dict[str, tuple[str, str, int, dict[str, int]]] = state[  # type: ignore[assignment]
            "merges"
        ]
        merges = {}
        for event_id in sorted(merges_map):
            target, source, at_value, event_changes = merges_map[event_id]
            merges[event_id] = {
                "target": target,
                "source": source,
                "at": at_value,
                "changes": {
                    key: event_changes[key] for key in sorted(event_changes)
                },
            }

        return {
            "events": events,
            "branches": branches,
            "appends": appends,
            "merges": merges,
        }

    @classmethod
    def _parse_checkpoint_document(cls, document: Any) -> dict[str, object]:
        """Validate the envelope, verify the checksum, parse and validate
        the payload into fresh local structures."""
        if not isinstance(document, dict):
            raise ValueError("top-level JSON value must be an object")
        if set(document.keys()) != set(cls._DOCUMENT_KEYS):
            raise ValueError(
                "top-level object must contain exactly the keys "
                "'schema_version', 'payload' and 'checksum'"
            )

        version = document["schema_version"]
        if isinstance(version, bool) or not isinstance(version, int):
            raise ValueError("'schema_version' must be an int")
        if version != cls._CHECKPOINT_VERSION:
            raise ValueError(f"unsupported schema_version {version!r}")

        checksum = document["checksum"]
        if not isinstance(checksum, str) or len(checksum) != 64 or (
            set(checksum) - cls._HEX_DIGITS
        ):
            raise ValueError(
                "'checksum' must be a 64-character lowercase hex string"
            )

        payload = document["payload"]
        if not isinstance(payload, dict):
            raise ValueError("'payload' must be an object")
        if set(payload.keys()) != set(cls._PAYLOAD_KEYS):
            raise ValueError(
                "payload must contain exactly the keys 'events', "
                "'branches', 'appends' and 'merges'"
            )

        # Re-serializing the parsed payload reproduces the writer's
        # canonical bytes; the checksum must match before anything is built.
        try:
            payload_text = cls._canonical_json(payload)
        except ValueError as exc:
            raise ValueError(f"payload is not canonicalizable JSON: {exc}")
        actual = hashlib.sha256(payload_text.encode("utf-8")).hexdigest()
        if not hmac.compare_digest(actual, checksum):
            raise ValueError("checksum does not match payload")

        state = cls._parse_checkpoint_payload(payload)
        cls._validate_checkpoint_state(state)
        return state

    @classmethod
    def _parse_checkpoint_payload(cls, payload: dict[str, Any]) -> dict[str, object]:
        """Strict structural parse of the payload into local state."""
        raw_events = payload["events"]
        if not isinstance(raw_events, list):
            raise ValueError("'events' must be an array")

        at_values: dict[str, int] = {}
        parent_map: dict[str, tuple[str, ...]] = {}
        change_map: dict[str, dict[str, int]] = {}
        for index, event in enumerate(raw_events):
            if not isinstance(event, dict):
                raise ValueError(f"event at index {index} must be an object")
            if set(event.keys()) != set(cls._EVENT_KEYS):
                raise ValueError(
                    f"event at index {index} must contain exactly the keys "
                    "'id', 'at', 'parents' and 'changes'"
                )
            event_id = event["id"]
            if not isinstance(event_id, str) or not event_id:
                raise ValueError(
                    f"event at index {index}: id must be a non-empty str"
                )
            if event_id in at_values:
                raise ValueError(f"duplicate event id {event_id!r}")
            at_value = event["at"]
            if isinstance(at_value, bool) or not isinstance(at_value, int):
                raise ValueError(f"event {event_id!r}: at must be an int")
            if at_value < 0:
                raise ValueError(f"event {event_id!r}: at must be non-negative")
            raw_parents = event["parents"]
            if not isinstance(raw_parents, list):
                raise ValueError(f"event {event_id!r}: parents must be an array")
            event_parents: list[str] = []
            seen_parents: set[str] = set()
            for parent in raw_parents:
                if not isinstance(parent, str) or not parent:
                    raise ValueError(
                        f"event {event_id!r}: every parent must be a "
                        "non-empty str"
                    )
                if parent in seen_parents:
                    raise ValueError(
                        f"event {event_id!r}: duplicate parent {parent!r}"
                    )
                seen_parents.add(parent)
                event_parents.append(parent)
            event_changes = cls._parse_change_object(
                event["changes"], f"event {event_id!r}"
            )
            at_values[event_id] = at_value
            parent_map[event_id] = tuple(event_parents)
            change_map[event_id] = event_changes

        raw_branches = payload["branches"]
        if not isinstance(raw_branches, list):
            raise ValueError("'branches' must be an array")
        heads: dict[str, str] = {}
        sources: dict[str, str] = {}
        for index, branch in enumerate(raw_branches):
            if not isinstance(branch, dict):
                raise ValueError(f"branch at index {index} must be an object")
            if set(branch.keys()) != set(cls._BRANCH_KEYS):
                raise ValueError(
                    f"branch at index {index} must contain exactly the keys "
                    "'name', 'source' and 'head'"
                )
            name = branch["name"]
            source = branch["source"]
            head = branch["head"]
            for label, value in (
                ("name", name),
                ("source", source),
                ("head", head),
            ):
                if not isinstance(value, str) or not value:
                    raise ValueError(
                        f"branch at index {index}: {label} must be a "
                        "non-empty str"
                    )
            if name in heads:
                raise ValueError(f"duplicate branch name {name!r}")
            heads[name] = head
            sources[name] = source

        raw_appends = payload["appends"]
        if not isinstance(raw_appends, dict):
            raise ValueError("'appends' must be an object")
        appends: dict[str, dict[str, tuple[int, dict[str, int]]]] = {}
        for name, raw_records in raw_appends.items():
            if not isinstance(name, str) or not name:
                raise ValueError("every appends group key must be a non-empty str")
            if not isinstance(raw_records, dict):
                raise ValueError(
                    f"append records for branch {name!r} must be an object"
                )
            branch_appends: dict[str, tuple[int, dict[str, int]]] = {}
            for event_id, record in raw_records.items():
                if not event_id:
                    raise ValueError(
                        f"branch {name!r}: append event id must be a non-empty str"
                    )
                at_value, event_changes = cls._parse_record(
                    record,
                    cls._APPEND_RECORD_KEYS,
                    f"append record {event_id!r}",
                )
                if event_id in branch_appends:
                    raise ValueError(
                        f"duplicate append record for event {event_id!r} "
                        f"on branch {name!r}"
                    )
                branch_appends[event_id] = (at_value, event_changes)
            appends[name] = branch_appends

        raw_merges = payload["merges"]
        if not isinstance(raw_merges, dict):
            raise ValueError("'merges' must be an object")
        merges: dict[str, tuple[str, str, int, dict[str, int]]] = {}
        for event_id, record in raw_merges.items():
            if not event_id:
                raise ValueError("merge event id must be a non-empty str")
            if not isinstance(record, dict):
                raise ValueError(f"merge record {event_id!r} must be an object")
            if set(record.keys()) != set(cls._MERGE_RECORD_KEYS):
                raise ValueError(
                    f"merge record {event_id!r} must contain exactly the keys "
                    "'target', 'source', 'at' and 'changes'"
                )
            target = record["target"]
            source = record["source"]
            if not isinstance(target, str) or not target:
                raise ValueError(
                    f"merge record {event_id!r}: target must be a non-empty str"
                )
            if not isinstance(source, str) or not source:
                raise ValueError(
                    f"merge record {event_id!r}: source must be a non-empty str"
                )
            at_value = record["at"]
            if isinstance(at_value, bool) or not isinstance(at_value, int):
                raise ValueError(f"merge record {event_id!r}: at must be an int")
            if at_value < 0:
                raise ValueError(
                    f"merge record {event_id!r}: at must be non-negative"
                )
            event_changes = cls._parse_change_object(
                record["changes"], f"merge record {event_id!r}"
            )
            if event_id in merges:
                raise ValueError(f"duplicate merge record for event {event_id!r}")
            merges[event_id] = (target, source, at_value, event_changes)

        return {
            "at": at_values,
            "parents": parent_map,
            "changes": change_map,
            "heads": heads,
            "sources": sources,
            "appends": appends,
            "merges": merges,
        }

    @classmethod
    def _parse_record(
        cls, record: Any, expected_keys: tuple[str, ...], label: str
    ) -> tuple[int, dict[str, int]]:
        """Parse an ``{"at", "changes"}`` style record object."""
        if not isinstance(record, dict):
            raise ValueError(f"{label} must be an object")
        if set(record.keys()) != set(expected_keys):
            raise ValueError(f"{label} has an unexpected set of keys")
        at_value = record["at"]
        if isinstance(at_value, bool) or not isinstance(at_value, int):
            raise ValueError(f"{label}: at must be an int")
        if at_value < 0:
            raise ValueError(f"{label}: at must be non-negative")
        return at_value, cls._parse_change_object(record["changes"], label)

    @staticmethod
    def _parse_change_object(raw: Any, label: str) -> dict[str, int]:
        """Parse and copy a ``{non-empty str: int-not-bool}`` object."""
        if not isinstance(raw, dict):
            raise ValueError(f"{label}: changes must be an object")
        event_changes: dict[str, int] = {}
        for key, value in raw.items():
            if not isinstance(key, str) or not key:
                raise ValueError(
                    f"{label}: every change key must be a non-empty str"
                )
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(
                    f"{label}: change value for {key!r} must be an int"
                )
            event_changes[key] = value
        return event_changes

    @classmethod
    def _validate_checkpoint_state(cls, state: dict[str, object]) -> None:
        """Cross-field validation: graph integrity, branch lineage and
        record/event agreement. Raises ValueError on the first defect."""
        at_values: dict[str, int] = state["at"]  # type: ignore[assignment]
        parent_map: dict[str, tuple[str, ...]] = state["parents"]  # type: ignore[assignment]
        change_map: dict[str, dict[str, int]] = state["changes"]  # type: ignore[assignment]
        heads: dict[str, str] = state["heads"]  # type: ignore[assignment]
        sources: dict[str, str] = state["sources"]  # type: ignore[assignment]
        appends: dict[str, dict[str, tuple[int, dict[str, int]]]] = state[  # type: ignore[assignment]
            "appends"
        ]
        merges: dict[str, tuple[str, str, int, dict[str, int]]] = state[  # type: ignore[assignment]
            "merges"
        ]

        # Append/merge records may only be grouped under known branches and
        # an event is recorded at most once across all records.
        recorded_events: set[str] = set()
        for name, branch_appends in appends.items():
            if name not in heads:
                raise ValueError(f"append record for unknown branch {name!r}")
            for event_id in branch_appends:
                if event_id in recorded_events:
                    raise ValueError(
                        f"event {event_id!r} is recorded more than once"
                    )
                recorded_events.add(event_id)
        for event_id, (target, source, _, _) in merges.items():
            if target not in heads:
                raise ValueError(
                    f"merge record {event_id!r}: unknown target branch "
                    f"{target!r}"
                )
            if source not in heads:
                raise ValueError(
                    f"merge record {event_id!r}: unknown source branch "
                    f"{source!r}"
                )
            if target == source:
                raise ValueError(
                    f"merge record {event_id!r}: target and source must "
                    "differ"
                )
            if event_id in recorded_events:
                raise ValueError(
                    f"event {event_id!r} is recorded more than once"
                )
            recorded_events.add(event_id)

        # Graph: every parent is known.
        for event_id, event_parents in parent_map.items():
            for parent in event_parents:
                if parent not in at_values:
                    raise ValueError(
                        f"event {event_id!r}: unknown parent {parent!r}"
                    )

        # Graph: cycle detection via iterative child -> parent DFS.
        white, gray, black = 0, 1, 2
        color = {event_id: white for event_id in at_values}
        for start in at_values:
            if color[start] != white:
                continue
            color[start] = gray
            stack: list[tuple[str, int]] = [(start, 0)]
            while stack:
                node, cursor = stack[-1]
                node_parents = parent_map[node]
                if cursor < len(node_parents):
                    nxt = node_parents[cursor]
                    stack[-1] = (node, cursor + 1)
                    if color[nxt] == gray:
                        raise ValueError(
                            f"cycle detected involving event {nxt!r}"
                        )
                    if color[nxt] == white:
                        color[nxt] = gray
                        stack.append((nxt, 0))
                else:
                    color[node] = black
                    stack.pop()

        # Recorded events must agree with graph events, timestamps, changes
        # and parent arity.
        for name, branch_appends in appends.items():
            for event_id, (recorded_at, recorded_changes) in (
                branch_appends.items()
            ):
                if event_id not in at_values:
                    raise ValueError(
                        f"append record {event_id!r} on branch {name!r} "
                        "has no matching event"
                    )
                if at_values[event_id] != recorded_at:
                    raise ValueError(
                        f"append record {event_id!r}: at does not match event"
                    )
                if change_map[event_id] != recorded_changes:
                    raise ValueError(
                        f"append record {event_id!r}: changes do not match event"
                    )
                if len(parent_map[event_id]) != 1:
                    raise ValueError(
                        f"append record {event_id!r}: event must have exactly "
                        "one parent"
                    )
        merge_by_target: dict[str, list[str]] = {name: [] for name in heads}
        for event_id, (target, source, recorded_at, recorded_changes) in (
            merges.items()
        ):
            if event_id not in at_values:
                raise ValueError(
                    f"merge record {event_id!r} has no matching event"
                )
            if at_values[event_id] != recorded_at:
                raise ValueError(
                    f"merge record {event_id!r}: at does not match event"
                )
            if change_map[event_id] != recorded_changes:
                raise ValueError(
                    f"merge record {event_id!r}: changes do not match event"
                )
            event_parents = parent_map[event_id]
            if len(event_parents) != 2:
                raise ValueError(
                    f"merge record {event_id!r}: event must have exactly "
                    "two parents"
                )
            merge_by_target[target].append(event_id)

        def ancestors(head: str) -> set[str]:
            closure = {head}
            pending = [head]
            while pending:
                current = pending.pop()
                for parent in parent_map[current]:
                    if parent not in closure:
                        closure.add(parent)
                        pending.append(parent)
            return closure

        # Branch lineage: source -> appends and inbound merges -> head forms
        # one linear chain, consuming every record attributed to the branch.
        chain_events: dict[str, list[str]] = {}
        for name in heads:
            source = sources[name]
            if source not in at_values:
                raise ValueError(
                    f"branch {name!r}: unknown source event {source!r}"
                )
            head = heads[name]
            if head not in at_values:
                raise ValueError(
                    f"branch {name!r}: unknown head event {head!r}"
                )

            advances = dict(appends.get(name, {}))
            for event_id in merge_by_target[name]:
                advances[event_id] = True
            # first-parent -> advancing event; the linear head chain allows
            # at most one such successor per event.
            successors: dict[str, str] = {}
            for event_id in advances:
                first_parent = parent_map[event_id][0]
                if first_parent in successors:
                    raise ValueError(
                        f"branch {name!r}: events {successors[first_parent]!r} "
                        f"and {event_id!r} both advance head {first_parent!r}"
                    )
                successors[first_parent] = event_id

            chain: list[str] = []
            current = source
            remaining = set(advances)
            while current != head:
                nxt = successors.get(current)
                if nxt is None:
                    raise ValueError(
                        f"branch {name!r}: head {head!r} is not reachable "
                        f"from source {source!r} through its records"
                    )
                chain.append(nxt)
                remaining.remove(nxt)
                current = nxt
            if remaining:
                raise ValueError(
                    f"branch {name!r}: records "
                    f"{sorted(remaining)} are not on its source-to-head chain"
                )
            chain_events[name] = chain

        # Merge source parents: (target_head, source_head) order. The source
        # parent must be the last point of the source branch's chain present
        # in the merge event's ancestor closure.
        for event_id, (target, source, _, _) in merges.items():
            target_parent, source_parent = parent_map[event_id]

            # Reproduce merge()'s conflict check at the recorded heads: the
            # exclusive closures may not touch overlapping change keys.
            target_closure = ancestors(target_parent)
            source_closure = ancestors(source_parent)
            shared = target_closure & source_closure
            target_keys: set[str] = set()
            for exclusive in target_closure - shared:
                target_keys.update(change_map[exclusive])
            source_keys: set[str] = set()
            for exclusive in source_closure - shared:
                source_keys.update(change_map[exclusive])
            overlapping = target_keys & source_keys
            if overlapping:
                raise ValueError(
                    f"merge record {event_id!r}: conflicting changes on keys: "
                    + ", ".join(sorted(overlapping))
                )

            source_chain = [sources[source]] + chain_events[source]
            if source_parent not in source_chain:
                raise ValueError(
                    f"merge record {event_id!r}: source parent "
                    f"{source_parent!r} is not on branch {source!r}'s chain"
                )
            position = source_chain.index(source_parent)
            if position + 1 < len(source_chain):
                later = source_chain[position + 1]
                if later in ancestors(event_id):
                    raise ValueError(
                        f"merge record {event_id!r}: source branch "
                        f"{source!r} had advanced past {source_parent!r} "
                        "before the merge"
                    )
            # Target-side agreement is implied by the target chain walk, but
            # assert the recorded first parent actually precedes the merge.
            target_chain = [sources[target]] + chain_events[target]
            if event_id not in target_chain:
                raise ValueError(
                    f"merge record {event_id!r}: not on target branch "
                    f"{target!r}'s chain"
                )
            target_position = target_chain.index(event_id)
            if target_chain[target_position - 1] != target_parent:
                raise ValueError(
                    f"merge record {event_id!r}: first parent does not match "
                    f"target branch {target!r}'s head"
                )

    # ------------------------------------------------------------------
    # Crash-recoverable operation journal
    # ------------------------------------------------------------------

    #: Journal format version emitted by newly enabled and rotated
    #: journals and by every commit they receive. Version 2 frames carry
    #: canonical before/after business-state digests that make a swapped,
    #: duplicated or reordered history fail recovery even after the
    #: sequence numbers and ``prev`` chain have been rebuilt; version 1
    #: journals remain readable, and an attachment recovered from one
    #: keeps appending in that file's version.
    _JOURNAL_VERSION = 2
    _JOURNAL_READABLE_VERSIONS = (1, 2)
    _JOURNAL_DOCUMENT_KEYS = ("schema_version", "checkpoint", "frames")
    _JOURNAL_FRAME_KEYS_V1 = ("seq", "op", "params", "prev")
    _JOURNAL_FRAME_KEYS_V2 = ("seq", "op", "params", "before", "after", "prev")
    _JOURNAL_CREATE_PARAMS = ("name", "from_event")
    _JOURNAL_APPEND_PARAMS = ("name", "id", "at", "changes")
    _JOURNAL_MERGE_PARAMS = ("target", "source", "id", "at", "changes")

    def enable_journal(self, checkpoint_path: Any, journal_path: Any) -> None:
        """Start journaling every committed operation to ``journal_path``.

        ``checkpoint_path`` must name a checkpoint file that describes
        exactly this store's current business state: the file is parsed
        and verified like :meth:`load_checkpoint` does, and its payload
        checksum must equal the checksum of the store's live state. Both
        paths must be non-empty :class:`str` values: a non-``str`` raises
        :class:`TypeError` and an empty string :class:`ValueError`, both
        before any state or file is consulted. A corrupt checkpoint
        raises :class:`ValueError`, a valid checkpoint that does not
        match the current state raises :class:`ValueError`, filesystem
        failures (missing checkpoint, unwritable location) raise
        :class:`OSError`, and enabling a store that already has a journal
        raises :class:`RuntimeError`. If ``journal_path`` already exists
        it is never overwritten: the call raises :class:`OSError`
        (:class:`FileExistsError`).

        On success the journal file is created as UTF-8 (no BOM), compact
        JSON with no trailing newline, holding the format version, the
        bound checkpoint's checksum and an empty frame sequence, and the
        store is attached to it: from then on every non-idempotent
        successful create, append or merge commits exactly one frame
        before its state becomes visible. On any failure the store stays
        unattached and its business state is untouched.
        """
        for value, name in (
            (checkpoint_path, "checkpoint_path"),
            (journal_path, "journal_path"),
        ):
            if not isinstance(value, str):
                raise TypeError(
                    f"{name} must be a str, got {type(value).__name__}"
                )
            if not value:
                raise ValueError(f"{name} must be a non-empty str")

        with self._state_lock:
            if self._journal_path is not None:
                raise RuntimeError(
                    "a journal is already enabled on this store"
                )
            state = self._snapshot_state()
            _checkpoint_state, checksum = self._read_checkpoint_state(
                checkpoint_path
            )
            payload = self._checkpoint_payload(state)
            current = hashlib.sha256(
                self._canonical_json(payload).encode("utf-8")
            ).hexdigest()
            if not hmac.compare_digest(current, checksum):
                raise ValueError(
                    "checkpoint does not describe the store's current state"
                )

            document = {
                "schema_version": self._JOURNAL_VERSION,
                "checkpoint": checksum,
                "frames": [],
            }
            data = self._canonical_json(document).encode("utf-8")
            # O_EXCL: an existing journal target is never overwritten.
            fd = os.open(
                journal_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666
            )
            try:
                handle = os.fdopen(fd, "wb")
            except BaseException:
                # fdopen only reaches here without taking ownership of fd.
                with contextlib.suppress(OSError):
                    os.close(fd)
                raise
            with handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())

            self._journal_checkpoint = checksum
            self._journal_checkpoint_path = checkpoint_path
            self._journal_version = self._JOURNAL_VERSION
            self._journal_frames = []
            self._journal_path = journal_path
        return None

    @classmethod
    def load_journal(
        cls, checkpoint_path: Any, journal_path: Any, through: Any
    ) -> "BranchStore":
        """Restore a store from a checkpoint plus its operation journal.

        ``checkpoint_path`` and ``journal_path`` must be non-empty
        :class:`str` values (non-``str`` raises :class:`TypeError`, empty
        strings :class:`ValueError`); ``through`` must be ``None`` or a
        non-``bool`` :class:`int` (anything else raises
        :class:`TypeError`). Filesystem failures on either path raise
        :class:`OSError`. The checkpoint is parsed and verified exactly
        like :meth:`load_checkpoint`; the journal must then be UTF-8
        without a BOM, valid JSON without duplicate keys, carry a
        supported format version (1 or 2), be bound to the checkpoint's
        checksum and hold frames with the exact expected fields,
        consecutive sequence numbers starting at one and an unbroken
        SHA-256 chain. Version 2 frames additionally carry canonical
        before/after business-state summaries; recovery recomputes the
        state summary around every replayed frame, so a swapped,
        duplicated or reordered history is rejected with
        :class:`ValueError` even after the sequence numbers and the
        ``prev`` chain were rebuilt. Invalid UTF-8 or JSON, truncation,
        duplicate or out-of-order frames, a broken chain or summary
        link, a wrong binding or a frame that conflicts with the
        replayed state all raise :class:`ValueError`; version 1 journals
        without the summary fields remain fully readable.

        Only frames whose sequence number is at most ``through`` are
        replayed: ``None`` replays the whole journal, ``0`` restores only
        the base checkpoint, a negative value or one beyond the journal's
        last sequence number raises :class:`ValueError`. A store restored
        to the journal's end stays attached to ``journal_path`` and keeps
        journaling new commits; a store restored to an earlier position
        is detached and never touches the file. Neither input file is
        ever modified. The restored store replays queries, audit,
        conflict and idempotency semantics exactly like the original;
        snapshot tokens are not persisted, so its token table is empty.
        """
        for value, name in (
            (checkpoint_path, "checkpoint_path"),
            (journal_path, "journal_path"),
        ):
            if not isinstance(value, str):
                raise TypeError(
                    f"{name} must be a str, got {type(value).__name__}"
                )
            if not value:
                raise ValueError(f"{name} must be a non-empty str")
        if through is not None and (
            isinstance(through, bool) or not isinstance(through, int)
        ):
            raise TypeError(
                f"through must be None or an int, got "
                f"{type(through).__name__}"
            )

        state, checksum = cls._read_checkpoint_state(checkpoint_path)
        frames, bound, version = cls._read_journal_frames(journal_path)
        if not hmac.compare_digest(bound, checksum):
            raise ValueError(
                "journal is not bound to this checkpoint"
            )

        last = len(frames)
        if through is None:
            upto = last
        else:
            if through < 0:
                raise ValueError("through must be non-negative")
            if through > last:
                raise ValueError(
                    f"through {through} exceeds the journal's last "
                    f"sequence number {last}"
                )
            upto = through

        store = cls._store_from_snapshot(state)
        store._replay_journal_frames(frames[:upto], checksum)
        if upto == last:
            # A complete recovery re-attaches the journal so later commits
            # extend the same chain; the parsed frames are fresh local
            # objects, so nothing is shared with the caller. The recovered
            # attachment keeps the file's schema version.
            store._journal_checkpoint = checksum
            store._journal_checkpoint_path = checkpoint_path
            store._journal_version = version
            store._journal_frames = frames
            store._journal_path = journal_path
        return store

    def rotate_journal(self, checkpoint_path: Any, journal_path: Any) -> None:
        """Freeze the current state into a fresh checkpoint and empty journal.

        Rotation is only allowed on a store that is attached to a journal
        after a complete recovery to that journal's end (the state produced
        by :meth:`enable_journal` or a full :meth:`load_journal`); calling
        it on a detached store raises :class:`RuntimeError`. The current
        business state is fixed as the new checkpoint, and a new, empty
        version-2 journal is created bound to that checkpoint's checksum;
        from then on commits are written only to the new journal, whose
        frame sequence starts again at one. The method returns ``None``.

        ``checkpoint_path`` and ``journal_path`` are validated in that
        signature order: a non-``str`` raises :class:`TypeError`, an empty
        string :class:`ValueError`, and two paths naming the same file
        raise :class:`ValueError`; either target already existing raises
        :class:`OSError` and neither is overwritten. Every open, write,
        flush, directory-sync or replace failure likewise raises
        :class:`OSError` and leaves both targets absent.

        Rotation runs under the same lock that serializes create, append
        and merge, so it observes one consistent instant -- checkpoint
        binding, audit and idempotency records and the current journal
        end -- and blocks every concurrent commit. The bound checkpoint
        and journal on disk are re-read and replayed to prove they end in
        exactly the live state before anything is published; a mismatch
        raises :class:`ValueError` and the attached journal is not
        rotated. Both new files are fully flushed and fsync-ed (their
        directories synced) before the in-memory attachment switches; on
        any failure the store keeps writing the old journal, its business
        state is item-wise unchanged, and the old checkpoint and journal
        stay byte-for-byte intact for archive recovery. A store restored
        from the new checkpoint alone behaves identically -- replay,
        branch heads, audit, conflicts and idempotency -- to the store at
        the rotation instant.
        """
        for value, name in (
            (checkpoint_path, "checkpoint_path"),
            (journal_path, "journal_path"),
        ):
            if not isinstance(value, str):
                raise TypeError(
                    f"{name} must be a str, got {type(value).__name__}"
                )
            if not value:
                raise ValueError(f"{name} must be a non-empty str")
        if os.path.abspath(checkpoint_path) == os.path.abspath(journal_path):
            raise ValueError(
                "checkpoint_path and journal_path must name different files"
            )

        with self._state_lock:
            if self._journal_path is None:
                raise RuntimeError(
                    "no journal is attached to this store; rotation "
                    "requires enable_journal or a complete load_journal"
                )

            # Refuse to touch either target that is already there; lstat
            # also catches dangling symlinks. Both checks pass before any
            # temporary file is created, so an existing target is never
            # overwritten.
            for target, name in (
                (checkpoint_path, "checkpoint_path"),
                (journal_path, "journal_path"),
            ):
                try:
                    os.lstat(target)
                except FileNotFoundError:
                    pass
                except OSError:
                    raise
                else:
                    raise FileExistsError(
                        f"{name} already exists: {target!r}"
                    )
            # --- Build and verify the rotation instant as one view. ---
            state = self._snapshot_state()
            payload = self._checkpoint_payload(state)
            new_checksum = hashlib.sha256(
                self._canonical_json(payload).encode("utf-8")
            ).hexdigest()

            # Re-read and replay the bound checkpoint plus the on-disk
            # journal to prove checkpoint, audit/idempotency records and
            # the journal end describe exactly the live state.
            bound_state, bound_checksum = self._read_checkpoint_state(
                self._journal_checkpoint_path
            )
            if not hmac.compare_digest(
                bound_checksum, self._journal_checkpoint
            ):
                raise ValueError(
                    "bound checkpoint no longer matches the attached journal"
                )
            disk_frames, disk_bound, disk_version = (
                self._read_journal_frames(self._journal_path)
            )
            if not hmac.compare_digest(disk_bound, bound_checksum):
                raise ValueError(
                    "journal on disk is not bound to its checkpoint"
                )
            if disk_version != self._journal_version:
                raise ValueError(
                    "journal version on disk no longer matches the "
                    "attached journal"
                )
            if disk_frames != self._journal_frames:
                raise ValueError(
                    "journal on disk no longer matches the recovered "
                    "journal end"
                )
            verifier = self._store_from_snapshot(bound_state)
            verifier._replay_journal_frames(disk_frames, bound_checksum)
            with verifier._state_lock:
                end_summary = self._state_summary(verifier._snapshot_state())
            if not hmac.compare_digest(end_summary, new_checksum):
                raise ValueError(
                    "live state does not match the checkpoint and journal "
                    "end; refusing to rotate"
                )

            checkpoint_document = self._canonical_json(
                {
                    "schema_version": self._CHECKPOINT_VERSION,
                    "payload": payload,
                    "checksum": new_checksum,
                }
            ).encode("utf-8")
            journal_document = self._canonical_json(
                {
                    "schema_version": self._JOURNAL_VERSION,
                    "checkpoint": new_checksum,
                    "frames": [],
                }
            ).encode("utf-8")

            # --- Publish both files durably before switching anything.
            # Each goes to a temp file in its target directory, fully
            # written, flushed and fsync-ed, then atomically published
            # without overwriting: a hard link onto the target fails with
            # FileExistsError if a concurrent writer occupied it after
            # the lstat checks above, so an occupied target is never
            # overwritten on Linux or Windows (NTFS hard links); the temp
            # name is unlinked once the link lands. The directories are
            # fsync-ed only after both publishes succeed. Any failure
            # unpublishes a file that already landed and removes the
            # temporary files, so neither target survives a failed
            # rotation.
            cp_directory = os.path.dirname(os.path.abspath(checkpoint_path))
            journal_directory = os.path.dirname(
                os.path.abspath(journal_path)
            )
            cp_tmp: str | None = None
            journal_tmp: str | None = None
            cp_published = False
            journal_published = False
            directories_synced = False

            def remove_tmp(path: str | None) -> None:
                if path is not None:
                    with contextlib.suppress(OSError):
                        os.remove(path)

            try:
                cp_tmp = self._write_durable_temp(
                    cp_directory,
                    ".checkpoint-",
                    checkpoint_document,
                )
                journal_tmp = self._write_durable_temp(
                    journal_directory,
                    ".journal-",
                    journal_document,
                )
                os.link(cp_tmp, checkpoint_path)
                cp_published = True
                os.unlink(cp_tmp)
                cp_tmp = None
                os.link(journal_tmp, journal_path)
                journal_published = True
                os.unlink(journal_tmp)
                journal_tmp = None
                self._fsync_directory(cp_directory)
                self._fsync_directory(journal_directory)
                directories_synced = True
            except BaseException:
                if not directories_synced:
                    if journal_published:
                        with contextlib.suppress(OSError):
                            os.remove(journal_path)
                    if cp_published:
                        with contextlib.suppress(OSError):
                            os.remove(checkpoint_path)
                remove_tmp(cp_tmp)
                remove_tmp(journal_tmp)
                # Make the unpublishing itself durable; secondary failures
                # must not mask the original error.
                if cp_published or journal_published:
                    with contextlib.suppress(OSError):
                        self._fsync_directory(journal_directory)
                    with contextlib.suppress(OSError):
                        self._fsync_directory(cp_directory)
                raise

            # Both files are durable. Switch the attachment only now; the
            # business state itself is intentionally untouched. The new
            # journal lives outside any generation, so no manifest is
            # re-pointed on later commits.
            self._journal_checkpoint = new_checksum
            self._journal_checkpoint_path = checkpoint_path
            self._journal_version = self._JOURNAL_VERSION
            self._journal_frames = []
            self._journal_path = journal_path
            self._journal_manifest_path = None
            self._journal_manifest = None
        return None

    @staticmethod
    def _write_durable_temp(
        directory: str, prefix: str, data: bytes
    ) -> str:
        """Create a temp file in ``directory`` holding ``data``, flushed
        and fsync-ed; return its path. The caller replaces it onto the
        target and removes it on any failure."""
        fd, tmp_path = tempfile.mkstemp(
            prefix=prefix, suffix=".tmp", dir=directory
        )
        try:
            try:
                handle = os.fdopen(fd, "wb")
            except BaseException:
                # fdopen only reaches here without taking ownership of fd.
                with contextlib.suppress(OSError):
                    os.close(fd)
                with contextlib.suppress(OSError):
                    os.remove(tmp_path)
                raise
            with handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            with contextlib.suppress(OSError):
                os.remove(tmp_path)
            raise
        return tmp_path

    @staticmethod
    def _fsync_directory(directory: str) -> None:
        """Fsync a directory so a freshly published directory entry is
        durable across a crash; raises :class:`OSError` on failure.

        Windows cannot fsync a directory through :func:`os.open`; the
        durable commit the platform offers there is the file-level fsync
        already performed on every published file, so the directory sync
        is skipped and each platform uses its available durable commit."""
        if os.name == "nt":
            return
        flags = os.O_RDONLY
        directory_flag = getattr(os, "O_DIRECTORY", 0)
        fd = os.open(directory, flags | directory_flag)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _commit_journal_frame(
        self,
        op: str,
        params: dict[str, object],
        before_state: dict[str, object] | None,
        after_state: dict[str, object] | None,
    ) -> None:
        """Append one frame to the attached journal, if any.

        Must run under :attr:`_state_lock`, after the operation's
        validation (and any graph-level conflict check) has succeeded.
        The caller applies the operation's internal state changes first,
        passes the pre/post business-state snapshots bracketing them, and
        rolls every internal change back if this raises; the durable
        frame therefore either lands together with the matching state or
        neither does. ``before_state``/``after_state`` are only needed
        while a journal is attached. Version 2 frames additionally record
        the canonical before/after state summaries, so a history whose
        frames were swapped, duplicated or reordered cannot be made to
        recover by renumbering and rebuilding the ``prev`` chain.

        The full journal document -- version, bound checkpoint checksum
        and the frame sequence including the new frame -- is written to a
        temporary file in the journal's directory, flushed, fsync-ed and
        atomically moved onto the journal path; only then is the frame
        recorded in memory. Any open, write, flush or replace failure
        raises :class:`OSError`, leaves the previous journal file
        byte-for-byte untouched and records nothing, so the caller can
        roll business state back to match the old journal. Because the
        write happens under the state lock, the journal's frame order is
        exactly the commit order of heads, audit and idempotency records.
        """
        if self._journal_path is None:
            return
        if self._journal_frames:
            prev = self._journal_frame_digest(self._journal_frames[-1])
        else:
            prev = self._journal_checkpoint
        frame: dict[str, object] = {
            "seq": len(self._journal_frames) + 1,
            "op": op,
            "params": params,
        }
        if self._journal_version >= 2:
            frame["before"] = self._state_summary(before_state)
            frame["after"] = self._state_summary(after_state)
        frame["prev"] = prev
        journal_data = self._write_journal_document(
            self._journal_frames + [frame]
        )
        if self._journal_manifest_path is not None:
            self._commit_generation_manifest(journal_data)
        self._journal_frames.append(frame)

    @classmethod
    def _state_summary(cls, state: dict[str, object]) -> str:
        """SHA-256 over a business-state snapshot's canonical checkpoint
        payload.

        The digest is taken over exactly the bytes :meth:`save_checkpoint`
        checksummed, so equal business states always summarize alike and
        a frame's before/after pair pins both endpoints of its transition
        independently of its operation parameters.
        """
        payload = cls._checkpoint_payload(state)
        return hashlib.sha256(
            cls._canonical_json(payload).encode("utf-8")
        ).hexdigest()

    def _write_journal_document(
        self, frames: list[dict[str, object]]
    ) -> bytes:
        """Serialize the journal document and atomically replace the file.

        The bytes are UTF-8 (no BOM), compact JSON with no trailing
        newline, written to a temporary file in the journal's directory,
        flushed and fsync-ed, then atomically moved onto the journal
        path; on any failure the previous file is left byte-for-byte
        untouched and the temporary file is removed. Returns the exact
        bytes published, so a generation manifest can be re-pointed at
        the new journal digest.
        """
        document = {
            "schema_version": self._journal_version,
            "checkpoint": self._journal_checkpoint,
            "frames": frames,
        }
        data = self._canonical_json(document).encode("utf-8")
        self._replace_durable(self._journal_path, data, ".journal-")
        return data

    @staticmethod
    def _replace_durable(path: str, data: bytes, prefix: str) -> None:
        """Write ``data`` to a temp file beside ``path``, flush and
        fsync it, then atomically replace ``path``; on any failure the
        previous file is left byte-for-byte untouched and the temporary
        file is removed."""
        directory = os.path.dirname(os.path.abspath(path))
        fd, tmp_path = tempfile.mkstemp(
            prefix=prefix, suffix=".tmp", dir=directory
        )
        replaced = False
        try:
            try:
                handle = os.fdopen(fd, "wb")
            except BaseException:
                # fdopen only reaches here without taking ownership of fd.
                with contextlib.suppress(OSError):
                    os.close(fd)
                raise
            with handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, path)
            replaced = True
        finally:
            if not replaced:
                with contextlib.suppress(OSError):
                    os.remove(tmp_path)

    def _commit_generation_manifest(self, journal_data: bytes) -> None:
        """Re-point the attached generation's manifest at the journal
        document just committed.

        Runs under :attr:`_state_lock` right after the journal file was
        durably replaced. The manifest is rewritten (temp file, fsync,
        atomic replace) with the new journal digest; every other field --
        including the previous-generation digest chain -- is carried over
        unchanged, so the manifest always vouches for exactly the journal
        on disk. If the manifest cannot be committed, the previous
        journal bytes are restored best-effort so the generation stays
        consistent and the caller can roll the business state back; a
        crash between the journal and manifest commits leaves a digest
        mismatch that :meth:`load_latest_generation` reports as an
        invalid, half-committed generation instead of trusting it.
        """
        old_journal_data = self._canonical_json(
            {
                "schema_version": self._journal_version,
                "checkpoint": self._journal_checkpoint,
                "frames": self._journal_frames,
            }
        ).encode("utf-8")
        manifest = dict(self._journal_manifest)
        manifest["journal"] = hashlib.sha256(journal_data).hexdigest()
        data = self._canonical_json(manifest).encode("utf-8")
        try:
            self._replace_durable(
                self._journal_manifest_path, data, ".manifest-"
            )
        except BaseException:
            with contextlib.suppress(OSError):
                self._replace_durable(
                    self._journal_path, old_journal_data, ".journal-"
                )
            raise
        self._journal_manifest = manifest

    @classmethod
    def _journal_frame_digest(cls, frame: dict[str, object]) -> str:
        """Lowercase hex SHA-256 of a frame's canonical JSON bytes."""
        canonical = cls._canonical_json(frame)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @classmethod
    def _read_journal_frames(
        cls, path: str
    ) -> tuple[list[dict[str, object]], str, int]:
        """Read, parse and fully validate a journal file.

        Returns the canonical frame list, the bound checkpoint checksum
        and the document's schema version. Filesystem failures propagate
        as :class:`OSError`; every content defect raises
        :class:`ValueError`. The caller validates ``path``'s type and
        emptiness.
        """
        with open(path, "rb") as handle:
            raw = handle.read()

        if raw.startswith(b"\xef\xbb\xbf"):
            raise ValueError("journal must be UTF-8 without a BOM")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"journal is not valid UTF-8: {exc}") from exc
        try:
            document = json.loads(
                text, object_pairs_hook=cls._reject_duplicate_json_keys
            )
        except json.JSONDecodeError as exc:
            raise ValueError(f"journal is not valid JSON: {exc}") from exc
        return cls._parse_journal_document(document)

    @classmethod
    def _parse_journal_document(
        cls, document: Any
    ) -> tuple[list[dict[str, object]], str, int]:
        """Validate the journal envelope, frame fields, consecutive
        numbering, the hash chain and (version 2) the state-summary
        links; return canonical frames, the bound checkpoint checksum
        and the schema version."""
        if not isinstance(document, dict):
            raise ValueError("journal top-level JSON value must be an object")
        if set(document.keys()) != set(cls._JOURNAL_DOCUMENT_KEYS):
            raise ValueError(
                "journal must contain exactly the keys 'schema_version', "
                "'checkpoint' and 'frames'"
            )

        version = document["schema_version"]
        if isinstance(version, bool) or not isinstance(version, int):
            raise ValueError("journal 'schema_version' must be an int")
        if version not in cls._JOURNAL_READABLE_VERSIONS:
            raise ValueError(f"unsupported journal schema_version {version!r}")

        checksum = document["checkpoint"]
        if not isinstance(checksum, str) or len(checksum) != 64 or (
            set(checksum) - cls._HEX_DIGITS
        ):
            raise ValueError(
                "journal 'checkpoint' must be a 64-character lowercase "
                "hex string"
            )

        raw_frames = document["frames"]
        if not isinstance(raw_frames, list):
            raise ValueError("journal 'frames' must be an array")

        frames: list[dict[str, object]] = []
        prev = checksum
        # Version 2 additionally threads the expected state summary: the
        # first frame must leave the checkpoint state, and each frame's
        # before summary must equal the previous frame's after summary.
        expected_state = checksum
        for index, raw_frame in enumerate(raw_frames):
            frame = cls._parse_journal_frame(raw_frame, index, version)
            expected_seq = index + 1
            if frame["seq"] != expected_seq:
                raise ValueError(
                    f"journal frame at index {index}: seq must be "
                    f"{expected_seq}"
                )
            if not hmac.compare_digest(frame["prev"], prev):
                raise ValueError(
                    f"journal frame {expected_seq}: hash chain is broken"
                )
            if version >= 2:
                if not hmac.compare_digest(
                    frame["before"], expected_state
                ):
                    raise ValueError(
                        f"journal frame {expected_seq}: before-state "
                        "summary does not extend the preceding history"
                    )
                expected_state = frame["after"]
            prev = cls._journal_frame_digest(frame)
            frames.append(frame)
        return frames, checksum, version

    @classmethod
    def _parse_journal_frame(
        cls, raw: Any, index: int, version: int
    ) -> dict[str, object]:
        """Strict structural parse of one journal frame into its canonical
        form (fixed key order, change keys sorted)."""
        label = f"journal frame at index {index}"
        if not isinstance(raw, dict):
            raise ValueError(f"{label} must be an object")
        expected_keys = (
            cls._JOURNAL_FRAME_KEYS_V2
            if version >= 2
            else cls._JOURNAL_FRAME_KEYS_V1
        )
        if set(raw.keys()) != set(expected_keys):
            if version >= 2:
                raise ValueError(
                    f"{label} must contain exactly the keys 'seq', 'op', "
                    "'params', 'before', 'after' and 'prev'"
                )
            raise ValueError(
                f"{label} must contain exactly the keys 'seq', 'op', "
                "'params' and 'prev'"
            )

        seq = raw["seq"]
        if isinstance(seq, bool) or not isinstance(seq, int):
            raise ValueError(f"{label}: seq must be an int")
        if seq < 1:
            raise ValueError(f"{label}: seq must be positive")

        prev = raw["prev"]
        if not isinstance(prev, str) or len(prev) != 64 or (
            set(prev) - cls._HEX_DIGITS
        ):
            raise ValueError(
                f"{label}: prev must be a 64-character lowercase hex string"
            )

        before: str | None = None
        after: str | None = None
        if version >= 2:
            for field_name, value in (
                ("before", raw["before"]),
                ("after", raw["after"]),
            ):
                if not isinstance(value, str) or len(value) != 64 or (
                    set(value) - cls._HEX_DIGITS
                ):
                    raise ValueError(
                        f"{label}: {field_name} must be a 64-character "
                        "lowercase hex string"
                    )
            before = raw["before"]
            after = raw["after"]

        op = raw["op"]
        if op not in ("create", "append", "merge"):
            raise ValueError(f"{label}: unknown op {op!r}")

        params = raw["params"]
        if not isinstance(params, dict):
            raise ValueError(f"{label}: params must be an object")

        def require_name(value: Any, field: str) -> str:
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"{label}: {field} must be a non-empty str"
                )
            return value

        def require_at(value: Any) -> int:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{label}: at must be an int")
            if value < 0:
                raise ValueError(f"{label}: at must be non-negative")
            return value

        if op == "create":
            if set(params.keys()) != set(cls._JOURNAL_CREATE_PARAMS):
                raise ValueError(
                    f"{label}: create params must contain exactly the "
                    "keys 'name' and 'from_event'"
                )
            canonical: dict[str, object] = {
                "name": require_name(params["name"], "name"),
                "from_event": require_name(
                    params["from_event"], "from_event"
                ),
            }
        elif op == "append":
            if set(params.keys()) != set(cls._JOURNAL_APPEND_PARAMS):
                raise ValueError(
                    f"{label}: append params must contain exactly the "
                    "keys 'name', 'id', 'at' and 'changes'"
                )
            changes = cls._parse_change_object(params["changes"], label)
            canonical = {
                "name": require_name(params["name"], "name"),
                "id": require_name(params["id"], "id"),
                "at": require_at(params["at"]),
                "changes": {key: changes[key] for key in sorted(changes)},
            }
        else:
            if set(params.keys()) != set(cls._JOURNAL_MERGE_PARAMS):
                raise ValueError(
                    f"{label}: merge params must contain exactly the "
                    "keys 'target', 'source', 'id', 'at' and 'changes'"
                )
            changes = cls._parse_change_object(params["changes"], label)
            canonical = {
                "target": require_name(params["target"], "target"),
                "source": require_name(params["source"], "source"),
                "id": require_name(params["id"], "id"),
                "at": require_at(params["at"]),
                "changes": {key: changes[key] for key in sorted(changes)},
            }

        frame = {"seq": seq, "op": op, "params": canonical}
        if version >= 2:
            frame["before"] = before
            frame["after"] = after
        frame["prev"] = prev
        return frame

    def _replay_journal_frames(
        self, frames: list[dict[str, object]], base_checksum: str
    ) -> None:
        """Replay validated journal frames onto this store.

        Each frame is applied through the public create/append/merge
        path, so the replayed store rebuilds the graph, heads, audit and
        idempotency records with their ordinary semantics. A frame that
        would be a no-op or a conflict against the replayed state can
        never come from a well-formed journal, so it raises
        :class:`ValueError`.

        Version 2 frames are independently re-checked frame by frame: the
        running store's canonical state summary must equal each frame's
        ``before`` summary immediately before its operation and its
        ``after`` summary immediately afterwards, starting from the bound
        checkpoint's checksum. Renumbering frames and rebuilding their
        ``prev`` chain cannot repair a swapped, duplicated or reordered
        history, because the recomputed summaries still disagree.
        """
        current_summary = base_checksum
        for frame in frames:
            op = frame["op"]
            params = frame["params"]
            if "before" in frame and not hmac.compare_digest(
                frame["before"], current_summary
            ):
                raise ValueError(
                    f"journal frame {frame['seq']}: recorded before-state "
                    "summary does not match the replayed state"
                )
            try:
                if op == "create":
                    if params["name"] in self._heads:
                        raise ValueError(
                            f"branch {params['name']!r} already exists"
                        )
                    self.create(params["name"], params["from_event"])
                elif op == "append":
                    if params["id"] in self._appends.get(params["name"], {}):
                        raise ValueError(
                            f"event {params['id']!r} is already recorded"
                        )
                    self.append(
                        params["name"],
                        params["id"],
                        params["at"],
                        dict(params["changes"]),
                    )
                else:
                    if params["id"] in self._merges:
                        raise ValueError(
                            f"merge {params['id']!r} is already recorded"
                        )
                    self.merge(
                        params["target"],
                        params["source"],
                        params["id"],
                        params["at"],
                        dict(params["changes"]),
                    )
            except (KeyError, ValueError) as exc:
                raise ValueError(
                    f"journal frame {frame['seq']} does not replay "
                    f"cleanly: {exc}"
                ) from exc
            if "after" in frame:
                current_summary = self._state_summary(self._snapshot_state())
                if not hmac.compare_digest(
                    frame["after"], current_summary
                ):
                    raise ValueError(
                        f"journal frame {frame['seq']}: recorded after-state "
                        "summary does not match the replayed state"
                    )

    # ------------------------------------------------------------------
    # Auditable rotation generations
    # ------------------------------------------------------------------

    #: On-disk layout of a generation root: a ``CURRENT`` pointer file
    #: names the newest completely published generation, and each
    #: directory named ``generation-`` plus sixteen decimal digits holds
    #: exactly a checkpoint, an empty journal and a manifest chaining the
    #: previous generation's manifest digest and vouching for both data
    #: files' digests.
    _GENERATION_PREFIX = "generation-"
    _GENERATION_DIGITS = 16
    _GENERATION_CHECKPOINT = "checkpoint.json"
    _GENERATION_JOURNAL = "journal.json"
    _GENERATION_MANIFEST = "manifest.json"
    _GENERATION_POINTER = "CURRENT"
    _GENERATION_MANIFEST_VERSION = 1
    _GENERATION_MANIFEST_KEYS = (
        "schema_version",
        "generation",
        "number",
        "previous",
        "checkpoint",
        "journal",
        "complete",
    )

    #: Recovery audit envelope (see :meth:`export_recovery_audit`).
    _RECOVERY_AUDIT_FORMAT = "branching-city-twin/recovery-audit"
    #: Version written by :meth:`export_recovery_audit`. Version 2
    #: records a corrupt pointer's SHA-256 digest so two different
    #: corruptions are distinguishable; version 1 records only the
    #: pointer state and remains accepted on verification and diff.
    _RECOVERY_AUDIT_VERSION = 2
    _RECOVERY_AUDIT_SUPPORTED_VERSIONS = (1, 2)
    _RECOVERY_AUDIT_KEYS = (
        "format",
        "version",
        "current",
        "selected",
        "generations",
        "ignored",
        "checksum",
    )
    _POINTER_STATES = ("missing", "ok", "corrupt")
    _DEPENDENCY_STATES = ("root", "linked", "broken", "invalid")

    #: Persistent anti-rollback chain envelope (see
    #: :meth:`append_recovery_audit_chain`). The document stores an
    #: ordered list of sealed frames at a fixed schema version; every
    #: frame's digest links its sequence number, its predecessor and the
    #: canonical recovery-audit record sampled at append time.
    _RECOVERY_CHAIN_FORMAT = (
        "branching-city-twin/recovery-audit-chain"
    )
    _RECOVERY_CHAIN_VERSION = 1
    _RECOVERY_CHAIN_SUPPORTED_VERSIONS = (1,)
    _RECOVERY_CHAIN_KEYS = ("format", "version", "frames")
    _RECOVERY_CHAIN_FRAME_KEYS = ("seq", "prev", "record", "digest")
    #: The first frame roots the chain at the all-zero predecessor
    #: digest, which is never the digest of a real frame.
    _RECOVERY_CHAIN_GENESIS_PREV = "0" * 64
    #: Suffix of the fixed coordination lock file that serializes
    #: appends across processes; it lives next to the chain file named
    #: by its canonical path and never carries audit content.
    _RECOVERY_CHAIN_LOCK_SUFFIX = ".lock"
    #: Poll interval in seconds while waiting for the chain lock.
    _RECOVERY_CHAIN_LOCK_POLL = 0.05

    #: Disposable, rebuildable index over the recovery-audit chain (see
    #: :meth:`build_recovery_audit_index`). The index binds the chain's
    #: schema version, frame count and head digest and records, for every
    #: frame, the byte boundaries needed to locate it in the chain file
    #: and the frame digest as digest evidence. It is only ever a cache:
    #: the canonical chain remains the sole trusted source.
    _RECOVERY_CHAIN_INDEX_FORMAT = (
        "branching-city-twin/recovery-audit-chain-index"
    )
    _RECOVERY_CHAIN_INDEX_VERSION = 1
    _RECOVERY_CHAIN_INDEX_SUPPORTED_VERSIONS = (1,)
    _RECOVERY_CHAIN_INDEX_KEYS = (
        "format",
        "version",
        "entries",
        "chain_version",
        "frames",
        "head",
    )
    _RECOVERY_CHAIN_INDEX_ENTRY_KEYS = (
        "seq",
        "offset",
        "length",
        "digest",
    )

    #: Disposable, rebuildable segmented index over the recovery-audit
    #: chain (see :meth:`build_recovery_audit_segments`). Consecutive
    #: frames are grouped into fixed-size segments (the last one may be
    #: short); every segment records its first and last sequence
    #: numbers, the boundary frame digests and the SHA-256 of the
    #: segment's canonical frame bytes. The document binds the chain's
    #: schema version, frame count, head digest and segment size. It is
    #: only ever a cache: the canonical chain remains the sole trusted
    #: source.
    _RECOVERY_CHAIN_SEGMENTS_FORMAT = (
        "branching-city-twin/recovery-audit-chain-segments"
    )
    _RECOVERY_CHAIN_SEGMENTS_VERSION = 1
    _RECOVERY_CHAIN_SEGMENTS_SUPPORTED_VERSIONS = (1,)
    _RECOVERY_CHAIN_SEGMENTS_KEYS = (
        "format",
        "version",
        "segment_size",
        "segments",
        "chain_version",
        "frames",
        "head",
    )
    _RECOVERY_CHAIN_SEGMENT_KEYS = (
        "first",
        "last",
        "first_digest",
        "last_digest",
        "digest",
    )

    #: Resumable build progress for the segmented index (see
    #: :meth:`build_recovery_audit_segments`). The document is a small,
    #: fixed-size cursor -- no audit bodies -- binding the segment size,
    #: the authenticated boundary (the last frame completing a segment,
    #: its byte span and digest), the frame count and head at the last
    #: successful build, the SHA-256 of the canonical chain prefix
    #: through that boundary, and the fingerprint of the segment index
    #: evidence. It is only ever a cache: on any mismatch the canonical
    #: chain is rebuilt from once.
    _RECOVERY_CHAIN_PROGRESS_FORMAT = (
        "branching-city-twin/recovery-audit-chain-progress"
    )
    _RECOVERY_CHAIN_PROGRESS_VERSION = 1
    _RECOVERY_CHAIN_PROGRESS_SUPPORTED_VERSIONS = (1,)
    _RECOVERY_CHAIN_PROGRESS_KEYS = (
        "format",
        "version",
        "segment_size",
        "frames",
        "head",
        "boundary",
        "prefix",
        "index",
    )
    _RECOVERY_CHAIN_PROGRESS_BOUNDARY_KEYS = (
        "seq",
        "offset",
        "length",
        "digest",
    )

    #: Cross-process journal for one durable two-cache publication (see
    #: :meth:`build_recovery_audit_segments`). When a segmented build or
    #: batch diff publishes with a ``recovery_path``, this record is made
    #: durable in the caches' directory *before* either target is replaced
    #: and updated (and synced) after each replacement, so a process
    #: dying mid-publication lets the next entry-point call finish the
    #: new version or roll both targets back to their old bytes. The
    #: record is UTF-8 without a BOM, compact JSON without a trailing
    #: newline; it binds its own version, the publication phase, each
    #: target's old/new existence and SHA-256 digest and the backup file
    #: names, and carries a checksum over every other field.
    _RECOVERY_CACHE_RECORD_FORMAT = (
        "branching-city-twin/recovery-cache-publication"
    )
    _RECOVERY_CACHE_RECORD_VERSION = 1
    _RECOVERY_CACHE_RECORD_SUPPORTED_VERSIONS = (1,)
    _RECOVERY_CACHE_RECORD_KEYS = (
        "format",
        "version",
        "phase",
        "index",
        "progress",
        "checksum",
    )
    _RECOVERY_CACHE_TARGET_KEYS = (
        "old_exists",
        "old_digest",
        "new_exists",
        "new_digest",
        "backup",
    )
    #: Phase recorded before either target is replaced.
    _RECOVERY_CACHE_PHASE_PREPARED = "prepared"
    #: Phase recorded after the index target was replaced.
    _RECOVERY_CACHE_PHASE_INDEX = "index"
    #: Phase recorded after the progress target was replaced.
    _RECOVERY_CACHE_PHASE_PROGRESS = "progress"
    _RECOVERY_CACHE_PHASES = (
        _RECOVERY_CACHE_PHASE_PREPARED,
        _RECOVERY_CACHE_PHASE_INDEX,
        _RECOVERY_CACHE_PHASE_PROGRESS,
    )

    #: Read-only cross-restart recovery diagnostics (see
    #: :meth:`recovery_diagnostic`). A diagnostic is a compact,
    #: checksummed observation of the durable evidence an interrupted
    #: two-cache publication left behind -- it never performs any
    #: recovery and modifies nothing. It binds a format/version, the
    #: transaction identity (the valid recovery record's checksum, or
    #: the SHA-256 of a corrupt record's raw bytes), the predecessor
    #: diagnostic's digest, the observed phase, each target's verdict
    #: against the recorded old/new digests, the recorded backups'
    #: status, and the resulting disposition with a stable reason
    #: category. Consecutive diagnostics authenticate one another
    #: through :meth:`verify_recovery_diagnostics`.
    _RECOVERY_DIAGNOSTIC_FORMAT = (
        "branching-city-twin/recovery-diagnostic"
    )
    _RECOVERY_DIAGNOSTIC_VERSION = 1
    _RECOVERY_DIAGNOSTIC_SUPPORTED_VERSIONS = (1,)
    _RECOVERY_DIAGNOSTIC_KEYS = (
        "format",
        "version",
        "transaction",
        "prev",
        "phase",
        "index",
        "progress",
        "backups",
        "disposition",
        "reason",
        "checksum",
    )
    _RECOVERY_DIAGNOSTIC_TARGET_KEYS = ("state", "matches")
    _RECOVERY_DIAGNOSTIC_BACKUPS_KEYS = ("index", "progress")
    _RECOVERY_DIAGNOSTIC_BACKUP_KEYS = ("present", "intact")
    #: The first observation in a diagnostic sequence roots the
    #: predecessor link at the all-zero digest.
    _RECOVERY_DIAGNOSTIC_GENESIS_PREV = "0" * 64
    #: A target judged without recorded digests (no parseable record);
    #: the entry requires both cache files to exist, so an observation
    #: without a parseable record always sees present files.
    _RECOVERY_DIAGNOSTIC_TARGET_PRESENT = "present"
    #: A target judged against a valid record.
    _RECOVERY_DIAGNOSTIC_TARGET_NEW = "new"
    _RECOVERY_DIAGNOSTIC_TARGET_OLD = "old"
    _RECOVERY_DIAGNOSTIC_TARGET_SAME = "same"
    _RECOVERY_DIAGNOSTIC_TARGET_UNKNOWN = "unknown"
    _RECOVERY_DIAGNOSTIC_TARGET_STATES = (
        _RECOVERY_DIAGNOSTIC_TARGET_PRESENT,
        _RECOVERY_DIAGNOSTIC_TARGET_NEW,
        _RECOVERY_DIAGNOSTIC_TARGET_OLD,
        _RECOVERY_DIAGNOSTIC_TARGET_SAME,
        _RECOVERY_DIAGNOSTIC_TARGET_UNKNOWN,
    )
    #: Which recorded version a target's observed bytes stand for;
    #: ``"both"`` only occurs for the
    #: :attr:`_RECOVERY_DIAGNOSTIC_TARGET_SAME` verdict (a publication
    #: that recorded identical old and new digests).
    _RECOVERY_DIAGNOSTIC_MATCH_OLD = "old"
    _RECOVERY_DIAGNOSTIC_MATCH_NEW = "new"
    _RECOVERY_DIAGNOSTIC_MATCH_BOTH = "both"
    _RECOVERY_DIAGNOSTIC_DISPOSITION_IDLE = "idle"
    _RECOVERY_DIAGNOSTIC_DISPOSITION_PENDING_COMMIT = "pending_commit"
    _RECOVERY_DIAGNOSTIC_DISPOSITION_PENDING_ROLLBACK = "pending_rollback"
    _RECOVERY_DIAGNOSTIC_DISPOSITION_COMMITTED = "committed"
    _RECOVERY_DIAGNOSTIC_DISPOSITION_ROLLED_BACK = "rolled_back"
    _RECOVERY_DIAGNOSTIC_DISPOSITION_REJECTED = "rejected"
    _RECOVERY_DIAGNOSTIC_DISPOSITIONS = (
        _RECOVERY_DIAGNOSTIC_DISPOSITION_IDLE,
        _RECOVERY_DIAGNOSTIC_DISPOSITION_PENDING_COMMIT,
        _RECOVERY_DIAGNOSTIC_DISPOSITION_PENDING_ROLLBACK,
        _RECOVERY_DIAGNOSTIC_DISPOSITION_COMMITTED,
        _RECOVERY_DIAGNOSTIC_DISPOSITION_ROLLED_BACK,
        _RECOVERY_DIAGNOSTIC_DISPOSITION_REJECTED,
    )
    _RECOVERY_DIAGNOSTIC_PHASES = (
        "",
        _RECOVERY_CACHE_PHASE_PREPARED,
        _RECOVERY_CACHE_PHASE_INDEX,
        _RECOVERY_CACHE_PHASE_PROGRESS,
    )
    _RECOVERY_DIAGNOSTIC_REASON_IDLE = "idle"
    _RECOVERY_DIAGNOSTIC_REASON_PENDING_COMMIT = "pending_commit"
    _RECOVERY_DIAGNOSTIC_REASON_PENDING_ROLLBACK = "pending_rollback"
    _RECOVERY_DIAGNOSTIC_REASON_COMMITTED = "committed"
    _RECOVERY_DIAGNOSTIC_REASON_ROLLED_BACK = "rolled_back"
    _RECOVERY_DIAGNOSTIC_REASON_RECORD_CORRUPT = "record_corrupt"
    _RECOVERY_DIAGNOSTIC_REASON_PHASE_CONTRADICTION = (
        "phase_contradiction"
    )
    _RECOVERY_DIAGNOSTIC_REASON_TARGET_MISMATCH = "target_mismatch"
    _RECOVERY_DIAGNOSTIC_REASON_BACKUP_MISSING = "backup_missing"
    _RECOVERY_DIAGNOSTIC_REASON_BACKUP_CORRUPT = "backup_corrupt"
    _RECOVERY_DIAGNOSTIC_REASON_TRANSACTION_CONTRADICTION = (
        "transaction_contradiction"
    )
    #: A durable record for an already disposed transaction reappeared.
    _RECOVERY_DIAGNOSTIC_REASON_REPROCESSED = "reprocessed"
    _RECOVERY_DIAGNOSTIC_REASONS = (
        _RECOVERY_DIAGNOSTIC_REASON_IDLE,
        _RECOVERY_DIAGNOSTIC_REASON_PENDING_COMMIT,
        _RECOVERY_DIAGNOSTIC_REASON_PENDING_ROLLBACK,
        _RECOVERY_DIAGNOSTIC_REASON_COMMITTED,
        _RECOVERY_DIAGNOSTIC_REASON_ROLLED_BACK,
        _RECOVERY_DIAGNOSTIC_REASON_RECORD_CORRUPT,
        _RECOVERY_DIAGNOSTIC_REASON_PHASE_CONTRADICTION,
        _RECOVERY_DIAGNOSTIC_REASON_TARGET_MISMATCH,
        _RECOVERY_DIAGNOSTIC_REASON_BACKUP_MISSING,
        _RECOVERY_DIAGNOSTIC_REASON_BACKUP_CORRUPT,
        _RECOVERY_DIAGNOSTIC_REASON_TRANSACTION_CONTRADICTION,
        _RECOVERY_DIAGNOSTIC_REASON_REPROCESSED,
    )

    @classmethod
    def _generation_number(cls, name: str) -> int | None:
        """Number of a legal generation directory name, else ``None``."""
        if not name.startswith(cls._GENERATION_PREFIX):
            return None
        digits = name[len(cls._GENERATION_PREFIX):]
        if len(digits) != cls._GENERATION_DIGITS or (
            set(digits) - set("0123456789")
        ):
            return None
        return int(digits)

    @staticmethod
    def _require_generation_root(root: str, need_write: bool) -> None:
        """Validate that ``root`` names an existing, usable directory.

        A missing root, a non-directory and (for rotation) a directory
        that is not readable and writable all raise :class:`OSError`.
        """
        st = os.stat(root)
        if not stat.S_ISDIR(st.st_mode):
            raise NotADirectoryError(
                f"generation root is not a directory: {root!r}"
            )
        if need_write and not os.access(root, os.R_OK | os.W_OK):
            raise PermissionError(
                f"generation root is not readable and writable: {root!r}"
            )

    def rotate_generation(self, root: Any) -> str:
        """Publish the current state as a new generation under ``root``.

        ``root`` must be a non-empty :class:`str` (a non-``str`` raises
        :class:`TypeError`, an empty string :class:`ValueError`) naming
        an existing directory that is readable and writable; a missing
        root, a non-directory, insufficient permissions or a failed
        directory sync raise :class:`OSError`.

        The new generation is numbered one above the highest completely
        published generation found under ``root`` (one if there is none)
        and named ``generation-`` plus sixteen decimal digits. It holds
        exactly three files: a checkpoint of the live business state, an
        empty version-2 journal bound to that checkpoint, and a manifest
        recording the previous generation's manifest digest (``null``
        for the first generation), both data files' SHA-256 digests and
        the completion marker. Half-finished directories left by an
        interrupted rotation are forensic material: they never count
        towards the numbering and are never reused, cleaned up or
        overwritten -- if the computed target is occupied, the call
        raises :class:`OSError` and overwrites nothing.

        Everything runs under the same lock that serializes create,
        append and merge, so the checkpoint captures one consistent
        instant and concurrent commits block. The generation directory
        and its files are written, flushed, fsync-ed and published
        without overwriting (hard links, as in :meth:`rotate_journal`),
        the generation directory is synced, and only then is the
        ``CURRENT`` pointer atomically re-pointed and the root directory
        synced. A crash midway may leave forensic material behind, but
        it never switches this instance's attachment and never makes a
        half-finished generation valid; on a raised failure the
        half-built generation is removed best-effort and this store
        keeps its current attachment and business state.

        On success the store attaches to the new generation's journal,
        whose frames start again at one, and every later commit also
        re-points the manifest at the new journal digest; older
        generations stay byte-for-byte untouched. Returns the new
        generation's name.
        """
        if not isinstance(root, str):
            raise TypeError(
                f"root must be a str, got {type(root).__name__}"
            )
        if not root:
            raise ValueError("root must be a non-empty str")

        with self._state_lock:
            self._require_generation_root(root, need_write=True)
            entries = os.listdir(root)

            # Numbering follows the highest completely published
            # generation (a parseable manifest carrying the completion
            # marker); anything half-finished is forensic material and
            # neither counts nor is reused.
            best = 0
            best_digest: str | None = None
            for entry in entries:
                number = self._generation_number(entry)
                if number is None or number <= best:
                    continue
                if not os.path.isdir(os.path.join(root, entry)):
                    continue
                manifest_path = os.path.join(
                    root, entry, self._GENERATION_MANIFEST
                )
                try:
                    with open(manifest_path, "rb") as handle:
                        manifest_bytes = handle.read()
                except OSError:
                    continue
                try:
                    document = json.loads(manifest_bytes.decode("utf-8"))
                except (UnicodeDecodeError, ValueError):
                    continue
                if not isinstance(document, dict) or (
                    document.get("complete") is not True
                ):
                    continue
                best = number
                best_digest = hashlib.sha256(manifest_bytes).hexdigest()

            number = best + 1
            name = (
                f"{self._GENERATION_PREFIX}"
                f"{number:0{self._GENERATION_DIGITS}d}"
            )
            generation_dir = os.path.join(root, name)
            checkpoint_path = os.path.join(
                generation_dir, self._GENERATION_CHECKPOINT
            )
            journal_path = os.path.join(
                generation_dir, self._GENERATION_JOURNAL
            )
            manifest_path = os.path.join(
                generation_dir, self._GENERATION_MANIFEST
            )

            state = self._snapshot_state()
            payload = self._checkpoint_payload(state)
            checksum = hashlib.sha256(
                self._canonical_json(payload).encode("utf-8")
            ).hexdigest()
            checkpoint_data = self._canonical_json(
                {
                    "schema_version": self._CHECKPOINT_VERSION,
                    "payload": payload,
                    "checksum": checksum,
                }
            ).encode("utf-8")
            journal_data = self._canonical_json(
                {
                    "schema_version": self._JOURNAL_VERSION,
                    "checkpoint": checksum,
                    "frames": [],
                }
            ).encode("utf-8")
            manifest: dict[str, Any] = {
                "schema_version": self._GENERATION_MANIFEST_VERSION,
                "generation": name,
                "number": number,
                "previous": best_digest,
                "checkpoint": hashlib.sha256(checkpoint_data).hexdigest(),
                "journal": hashlib.sha256(journal_data).hexdigest(),
                "complete": True,
            }
            manifest_data = self._canonical_json(manifest).encode("utf-8")

            # Durable, no-overwrite publish: the fresh directory and each
            # file appear exactly once (a concurrent occupant raises
            # OSError and is never overwritten), the generation directory
            # is synced, and only then is the CURRENT pointer atomically
            # re-pointed and the root directory synced.
            os.mkdir(generation_dir)
            published: list[str] = []
            tmps: list[str] = []
            pointer_updated = False
            try:
                for prefix, target, data in (
                    (".checkpoint-", checkpoint_path, checkpoint_data),
                    (".journal-", journal_path, journal_data),
                    (".manifest-", manifest_path, manifest_data),
                ):
                    tmp = self._write_durable_temp(
                        generation_dir, prefix, data
                    )
                    tmps.append(tmp)
                    os.link(tmp, target)
                    published.append(target)
                    os.unlink(tmp)
                    tmps.pop()
                self._fsync_directory(generation_dir)
                pointer_tmp = self._write_durable_temp(
                    root, ".current-", name.encode("utf-8")
                )
                tmps.append(pointer_tmp)
                os.replace(
                    pointer_tmp,
                    os.path.join(root, self._GENERATION_POINTER),
                )
                tmps.pop()
                pointer_updated = True
                self._fsync_directory(root)
            except BaseException:
                for tmp in tmps:
                    with contextlib.suppress(OSError):
                        os.remove(tmp)
                if not pointer_updated:
                    for target in reversed(published):
                        with contextlib.suppress(OSError):
                            os.remove(target)
                    with contextlib.suppress(OSError):
                        os.rmdir(generation_dir)
                raise

            # Everything is durable and the pointer has landed; only now
            # does this instance switch to the new generation's journal.
            self._journal_checkpoint = checksum
            self._journal_checkpoint_path = checkpoint_path
            self._journal_version = self._JOURNAL_VERSION
            self._journal_frames = []
            self._journal_path = journal_path
            self._journal_manifest_path = manifest_path
            self._journal_manifest = manifest
            return name

    @classmethod
    def _validate_recovery_root(cls, root: Any) -> None:
        """Validate a recovery/audit ``root`` argument the same way for
        :meth:`load_latest_generation`, :meth:`export_recovery_audit`,
        :meth:`verify_recovery_audit` and :meth:`diff_recovery_audit`:
        a non-``str`` raises :class:`TypeError`, an empty string
        :class:`ValueError`, and a missing or non-directory path
        :class:`OSError`."""
        if not isinstance(root, str):
            raise TypeError(
                f"root must be a str, got {type(root).__name__}"
            )
        if not root:
            raise ValueError("root must be a non-empty str")
        cls._require_generation_root(root, need_write=False)

    @classmethod
    def load_latest_generation(
        cls, root: Any
    ) -> tuple["BranchStore", dict[str, object]]:
        """Recover a store from the newest valid generation under ``root``.

        ``root`` must be a non-empty :class:`str` (a non-``str`` raises
        :class:`TypeError`, an empty string :class:`ValueError`) naming
        an existing directory; a missing root, a non-directory, a failed
        directory enumeration or a failed file read raise
        :class:`OSError`. The check is strictly read-only: the ``CURRENT``
        pointer and every legally named ``generation-`` directory are
        inspected, and nothing is modified, cleaned up or removed.

        A generation is only valid when *all* of its evidence holds: its
        manifest parses and carries the completion marker, the manifest
        digest it cites as ``previous`` is exactly the immediately
        preceding generation's manifest, its checkpoint and journal
        match the manifest's digests, the journal binds to the
        checkpoint, and the journal replays to its end. Validity runs by
        number from generation one with no gaps: a generation is only an
        acceptable predecessor when its own chain is complete, so once a
        generation is invalid every later generation is reported as
        invalid for depending on a broken chain -- a successor that
        merely quotes the invalid predecessor's manifest digest, or one
        reached across a missing number, never chains over the break. The
        highest-numbered generation of the complete valid prefix is
        selected. The ``CURRENT`` pointer is only a hint and never steers
        selection: a missing pointer reports ``None``, a stale but legal
        generation name is reported verbatim, and an undecodable,
        empty, whitespace-bearing or illegally named pointer is
        normalized to ``None``. If no generation is valid,
        :class:`ValueError` is raised and no partial store is returned.

        Returns ``(store, report)``: the recovered store, attached to
        the selected generation's journal so it can keep committing
        (each commit re-points the manifest), and a report whose keys
        are ordered ``selected``, ``current``, ``ignored`` -- the
        selected generation's name, the pointer's normalized content
        (``None`` if missing, corrupt or not a legal generation name),
        and the invalid generations in ascending number order, each a
        dict with ``name`` and the definitive ``reason``. A failed load
        changes no business, audit, idempotency or query state anywhere.
        """
        cls._validate_recovery_root(root)
        evidence = cls._gather_recovery_evidence(root)
        selected = evidence["selected"]
        if selected is None:
            raise ValueError(
                f"no valid generation found under {root!r}"
            )

        chosen = evidence["selected_evidence"]
        generation_dir = chosen["dir"]
        store = chosen["store"]
        store._journal_checkpoint = chosen["checksum"]
        store._journal_checkpoint_path = os.path.join(
            generation_dir, cls._GENERATION_CHECKPOINT
        )
        store._journal_version = chosen["version"]
        store._journal_frames = chosen["frames"]
        store._journal_path = os.path.join(
            generation_dir, cls._GENERATION_JOURNAL
        )
        store._journal_manifest_path = os.path.join(
            generation_dir, cls._GENERATION_MANIFEST
        )
        store._journal_manifest = chosen["manifest"]
        report: dict[str, object] = {
            "selected": selected,
            "current": evidence["current"],
            "ignored": evidence["ignored"],
        }
        return store, report

    @classmethod
    def _read_recovery_pointer(
        cls, root: str
    ) -> tuple[str | None, str, str | None]:
        """Read the ``CURRENT`` hint strictly and read-only.

        Returns ``(value, state, digest)`` where state is ``"missing"``
        (no pointer file -- value and digest ``None``), ``"ok"`` (value
        is exactly a legal generation name, reported verbatim, however
        stale -- digest ``None``) or ``"corrupt"`` (undecodable bytes,
        an empty file, surrounding whitespace, or anything that is not
        a legal generation name -- value ``None`` and digest the
        SHA-256 of the raw bytes). Only the digest is retained for
        corrupt material: corrupt pointer bytes are never echoed, so
        two different corruptions are distinguishable without ever
        exposing their content.
        """
        try:
            with open(
                os.path.join(root, cls._GENERATION_POINTER), "rb"
            ) as handle:
                raw = handle.read()
        except FileNotFoundError:
            return None, "missing", None
        digest = hashlib.sha256(raw).hexdigest()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return None, "corrupt", digest
        if text == "" or text != text.strip():
            return None, "corrupt", digest
        number = cls._generation_number(text)
        if number is None or number < 1:
            return None, "corrupt", digest
        return text, "ok", None

    @classmethod
    def _gather_recovery_evidence(cls, root: str) -> dict[str, Any]:
        """Scan ``root`` read-only and decide recovery for every
        generation. Shared verbatim by loading and by recovery-audit
        export/verification, so all three act on identical evidence.

        Returns the normalized pointer value, state and corrupt digest,
        one ordered audit record per legally named generation entry, the
        ascending-number ``ignored`` list, the selected generation name
        (``None`` when the valid prefix is empty) and everything needed
        to attach a recovered store to it. Raises only :class:`OSError`
        for enumeration or read failures; invalid content is reported,
        never raised.
        """
        current, pointer_state, pointer_digest = (
            cls._read_recovery_pointer(root)
        )

        candidates: list[tuple[int, str, bool]] = []
        for entry in os.listdir(root):
            number = cls._generation_number(entry)
            if number is None:
                continue
            is_dir = os.path.isdir(os.path.join(root, entry))
            candidates.append((number, entry, is_dir))
        candidates.sort()

        records: list[dict[str, Any]] = []
        ignored: list[dict[str, str]] = []
        selected: str | None = None
        selected_evidence: dict[str, Any] | None = None
        # The chain is a contiguous, fully valid prefix starting at
        # generation one. ``chain_valid`` flips permanently at the first
        # break and every later generation is doomed regardless of its
        # own evidence; ``chain_digest`` is the manifest digest the next
        # generation must cite (None only before generation one).
        chain_valid = True
        chain_digest: str | None = None
        expected_number = 1
        for number, name, is_dir in candidates:
            if number < 1:
                # A zero-numbered entry matches the directory pattern but
                # is no generation: it is forensic junk, reported as
                # invalid without ever occupying a chain position, so it
                # cannot poison the real chain that starts at one.
                records.append(
                    {
                        "name": name,
                        "number": number,
                        "valid": False,
                        "dependency": "broken",
                        "previous": None,
                        "manifest": None,
                        "files": {
                            "checkpoint": None,
                            "journal": None,
                        },
                        "replay": "not-run",
                        "reason": (
                            "generation number zero is not a valid "
                            "generation"
                        ),
                    }
                )
                ignored.append(
                    {
                        "name": name,
                        "reason": (
                            "generation number zero is not a valid "
                            "generation"
                        ),
                    }
                )
                continue

            generation_dir = os.path.join(root, name)
            manifest: dict[str, Any] | None = None
            manifest_bytes: bytes | None = None
            manifest_digest: str | None = None
            claimed_previous: str | None = None
            replay = "not-run"

            # Raw, read-only evidence collection. File digests are
            # captured even for a generation that is doomed by its
            # predecessor, so the audit records what is on disk without
            # ever trusting it.
            structural_reason: str | None = None
            if not is_dir:
                structural_reason = "generation path is not a directory"
            else:
                manifest_path = os.path.join(
                    generation_dir, cls._GENERATION_MANIFEST
                )
                try:
                    with open(manifest_path, "rb") as handle:
                        manifest_bytes = handle.read()
                except FileNotFoundError:
                    structural_reason = "manifest is missing"
            if manifest_bytes is not None:
                manifest_digest = hashlib.sha256(
                    manifest_bytes
                ).hexdigest()
                parse_reason, parsed = cls._parse_generation_manifest(
                    manifest_bytes, name, number
                )
                if parse_reason is not None:
                    if structural_reason is None:
                        structural_reason = parse_reason
                else:
                    manifest = parsed
                    claimed_previous = manifest["previous"]
                    if manifest["complete"] is not True:
                        structural_reason = (
                            "manifest is not marked complete"
                        )
            files = (
                cls._generation_file_digests(generation_dir)
                if is_dir
                else {"checkpoint": None, "journal": None}
            )

            # Independent dependency verdict relative to the numbered
            # predecessor chain.
            if not chain_valid:
                dependency = "invalid"
            elif number == 1:
                dependency = (
                    "root" if claimed_previous is None else "broken"
                )
            elif number != expected_number or claimed_previous is None or (
                not hmac.compare_digest(claimed_previous, chain_digest)
            ):
                dependency = "broken"
            else:
                dependency = "linked"

            # Definitive ignore reason, in precedence order: an invalid
            # predecessor condemns the generation before any of its own
            # evidence is consulted, so a break can never be jumped.
            recovered: dict[str, Any] | None = None
            if not chain_valid:
                reason = (
                    "dependency chain is invalid: a previous "
                    "generation is invalid"
                )
            elif not is_dir:
                reason = structural_reason
            elif number != expected_number:
                reason = (
                    "dependency chain is broken: generation "
                    f"{expected_number} is missing"
                )
            elif structural_reason is not None:
                reason = structural_reason
            elif number == 1 and claimed_previous is not None:
                reason = (
                    "dependency chain is broken: the first generation "
                    "must cite no previous manifest"
                )
            elif number > 1 and claimed_previous is None:
                reason = (
                    "dependency chain is broken: the manifest cites no "
                    "previous manifest"
                )
            elif (
                number > 1
                and not hmac.compare_digest(
                    claimed_previous, chain_digest
                )
            ):
                reason = (
                    "dependency chain is broken: the cited previous "
                    "manifest digest does not match"
                )
            else:
                reason, recovered, replay = (
                    cls._verify_generation_evidence(
                        generation_dir, manifest
                    )
                )

            valid = reason is None
            record = {
                "name": name,
                "number": number,
                "valid": valid,
                "dependency": dependency,
                "previous": claimed_previous,
                "manifest": manifest_digest,
                "files": files,
                "replay": replay,
                "reason": reason,
            }
            records.append(record)

            if not valid:
                ignored.append({"name": name, "reason": reason})
                # Nothing downstream may chain over this break.
                chain_valid = False
                chain_digest = None
            else:
                selected = name
                selected_evidence = {
                    "dir": generation_dir,
                    "store": recovered["store"],
                    "frames": recovered["frames"],
                    "version": recovered["version"],
                    "checksum": recovered["checksum"],
                    "manifest": manifest,
                }
                chain_digest = manifest_digest
            expected_number = number + 1

        return {
            "current": current,
            "pointer_state": pointer_state,
            "pointer_digest": pointer_digest,
            "records": records,
            "ignored": ignored,
            "selected": selected,
            "selected_evidence": selected_evidence,
        }

    @classmethod
    def _generation_file_digests(
        cls, generation_dir: str
    ) -> dict[str, str | None]:
        """Read-only SHA-256 of a generation's data files.

        Captures on-disk evidence without judging it: a missing file
        contributes ``None`` while every readable file contributes its
        digest, even for a generation already doomed by its predecessor.
        A read failure other than absence propagates as :class:`OSError`.
        """
        digests: dict[str, str | None] = {
            "checkpoint": None,
            "journal": None,
        }
        for field, filename in (
            ("checkpoint", cls._GENERATION_CHECKPOINT),
            ("journal", cls._GENERATION_JOURNAL),
        ):
            path = os.path.join(generation_dir, filename)
            try:
                with open(path, "rb") as handle:
                    data = handle.read()
            except FileNotFoundError:
                continue
            digests[field] = hashlib.sha256(data).hexdigest()
        return digests

    @classmethod
    def _verify_generation_evidence(
        cls, generation_dir: str, manifest: dict[str, Any]
    ) -> tuple[str | None, dict[str, Any] | None, str]:
        """Verify a structurally accepted generation's data files, log
        binding and terminal replay.

        Applies the definitive ordering: checkpoint presence and digest,
        journal presence and digest, content validation, journal-to-
        checkpoint binding, and an actual replay of every frame. Returns
        ``(None, evidence, "ok")`` on success or ``(reason, None,
        replay)`` naming the first defect; ``replay`` is the terminal
        replay conclusion (``"ok"``, ``"not-run"`` when an earlier check
        stopped replay, or the replay failure message). Missing files are
        defects; other read failures raise :class:`OSError`.
        """
        checkpoint_path = os.path.join(
            generation_dir, cls._GENERATION_CHECKPOINT
        )
        journal_path = os.path.join(
            generation_dir, cls._GENERATION_JOURNAL
        )
        try:
            with open(checkpoint_path, "rb") as handle:
                checkpoint_bytes = handle.read()
        except FileNotFoundError:
            return "checkpoint file is missing", None, "not-run"
        if not hmac.compare_digest(
            hashlib.sha256(checkpoint_bytes).hexdigest(),
            manifest["checkpoint"],
        ):
            return "checkpoint digest mismatch", None, "not-run"
        try:
            with open(journal_path, "rb") as handle:
                journal_bytes = handle.read()
        except FileNotFoundError:
            return "journal file is missing", None, "not-run"
        if not hmac.compare_digest(
            hashlib.sha256(journal_bytes).hexdigest(),
            manifest["journal"],
        ):
            return "journal digest mismatch", None, "not-run"
        try:
            state, checksum = cls._read_checkpoint_state(checkpoint_path)
        except ValueError as exc:
            return (
                f"checkpoint is invalid: {exc}",
                None,
                "not-run",
            )
        try:
            frames, bound, version = cls._read_journal_frames(
                journal_path
            )
        except ValueError as exc:
            return (
                f"journal is invalid: {exc}",
                None,
                "not-run",
            )
        if not hmac.compare_digest(bound, checksum):
            return (
                "journal is not bound to the checkpoint",
                None,
                "not-run",
            )
        store = cls._store_from_snapshot(state)
        try:
            store._replay_journal_frames(frames, checksum)
        except ValueError as exc:
            message = f"journal does not replay cleanly: {exc}"
            return message, None, message
        return (
            None,
            {
                "store": store,
                "frames": frames,
                "version": version,
                "checksum": checksum,
            },
            "ok",
        )

    @classmethod
    def _recovery_audit_body(
        cls, evidence: dict[str, Any], version: int
    ) -> dict[str, Any]:
        """Assemble the signed recovery-audit body from gathered
        evidence at the given record version. Only summaries of corrupt
        material are kept; no raw pointer bytes or absolute paths
        appear, so identical evidence serializes identically anywhere.
        Version 2 adds the corrupt pointer's SHA-256 digest to the
        pointer summary (``None`` for a missing or readable pointer);
        version 1 carries only the pointer state and value."""
        current: dict[str, Any] = {
            "state": evidence["pointer_state"],
            "value": evidence["current"],
        }
        if version >= 2:
            current["digest"] = evidence["pointer_digest"]
        return {
            "format": cls._RECOVERY_AUDIT_FORMAT,
            "version": version,
            "current": current,
            "selected": evidence["selected"],
            "generations": [
                {
                    "name": record["name"],
                    "number": record["number"],
                    "valid": record["valid"],
                    "dependency": record["dependency"],
                    "previous": record["previous"],
                    "manifest": record["manifest"],
                    "files": {
                        "checkpoint": record["files"]["checkpoint"],
                        "journal": record["files"]["journal"],
                    },
                    "replay": record["replay"],
                    "reason": record["reason"],
                }
                for record in evidence["records"]
            ],
            "ignored": [
                {"name": entry["name"], "reason": entry["reason"]}
                for entry in evidence["ignored"]
            ],
        }

    @classmethod
    def export_recovery_audit(cls, root: Any) -> str:
        """Return a canonical, signed JSON audit of recovery evidence.

        ``root`` is validated exactly as for
        :meth:`load_latest_generation`: a non-``str`` raises
        :class:`TypeError`, an empty string :class:`ValueError`, and a
        missing root, non-directory, enumeration failure or read failure
        :class:`OSError`. The scan is strictly read-only.

        The record captures the pointer state (``missing``, ``ok`` or
        ``corrupt`` -- a corrupt pointer is summarized by the SHA-256
        digest of its raw bytes, never echoed), the selected generation,
        every generation's validity and dependency verdict with its
        claimed predecessor, manifest and file digests and
        terminal-replay conclusion, and the ascending-number ignore
        list with definitive reasons. It carries a format tag, schema
        version 2 and a SHA-256 checksum over the recovery decision and
        all evidence. Serialization is compact UTF-8 JSON with no extra
        whitespace or trailing newline, so identical evidence produces
        a byte-identical record.

        Exporting when no generation is recoverable raises
        :class:`ValueError`. No disk material -- the pointer,
        generations, business state, audits or idempotency records -- is
        modified.
        """
        cls._validate_recovery_root(root)
        evidence = cls._gather_recovery_evidence(root)
        if evidence["selected"] is None:
            raise ValueError(
                f"no valid generation found under {root!r}"
            )
        body = cls._recovery_audit_body(
            evidence, cls._RECOVERY_AUDIT_VERSION
        )
        body_text = cls._canonical_json(body)
        checksum = hashlib.sha256(
            body_text.encode("utf-8")
        ).hexdigest()
        envelope = dict(body)
        envelope["checksum"] = checksum
        return cls._canonical_json(envelope)

    @classmethod
    def verify_recovery_audit(cls, root: Any, record: Any) -> bool:
        """Re-scan ``root`` read-only and check it against an audit
        record.

        ``root`` is validated first as for the other recovery entry
        points (non-``str`` :class:`TypeError`, empty :class:`ValueError`,
        missing/non-directory/enumeration or read failure
        :class:`OSError`). ``record`` is validated next: a non-``str``
        raises :class:`TypeError`; an empty string, unparsable JSON,
        duplicate JSON keys, a non-canonical encoding, a structural or
        version violation, or a checksum that does not protect the
        record raises :class:`ValueError`. Both record versions are
        accepted: a version 1 record is checked against the pointer
        state and value alone, while a version 2 record also pins the
        corrupt pointer's SHA-256 digest, so replacing one corrupt
        pointer's content with another is an evidence change.

        A valid record is then compared against fresh evidence: the same
        recovery decision and the same file evidence return ``True``; any
        change under the directory -- including losing every recoverable
        generation -- returns ``False``. The verification is strictly
        read-only.
        """
        cls._validate_recovery_root(root)
        body = cls._parse_recovery_audit_record(record)
        evidence = cls._gather_recovery_evidence(root)
        expected_body = cls._recovery_audit_body(
            evidence, body["version"]
        )
        return hmac.compare_digest(
            cls._canonical_json(body),
            cls._canonical_json(expected_body),
        )

    @classmethod
    def diff_recovery_audit(
        cls, root: Any, record: Any
    ) -> tuple[tuple[Any, ...], ...]:
        """Classify how ``root``'s recovery evidence diverges from a
        sealed audit record, strictly read-only.

        ``root`` and ``record`` are validated exactly as for
        :meth:`verify_recovery_audit` (``root``: non-``str``
        :class:`TypeError`, empty :class:`ValueError`, missing,
        non-directory or unreadable :class:`OSError`; ``record``:
        non-``str`` :class:`TypeError`, empty, unparsable,
        duplicate-key, non-canonical, structural, version or checksum
        violations :class:`ValueError`). Both record versions are
        accepted, and the comparison uses the record's own version: a
        version 1 record compares the pointer by state and value, a
        version 2 record also by the corrupt pointer's digest.

        Returns a tuple of change tuples, empty when the fresh evidence
        matches the record exactly. Unlike
        :meth:`export_recovery_audit`, a directory with no recoverable
        generation left is not an error: the lost evidence is reported
        like any other change instead of raising :class:`ValueError`.
        Each change tuple is ``(category, generation, change, before,
        after)`` where ``before`` is the sealed value and ``after`` the
        current one, a missing side is ``None``, ``category`` is one of
        ``"pointer"``, ``"chain"``, ``"checkpoint"``, ``"journal"`` or
        ``"replay"`` and ``change`` is one of ``"added"``,
        ``"removed"`` or ``"changed"``.

        * ``pointer`` (generation ``None``) compares the pointer state,
          the legal pointer value and -- for version 2 records -- the
          corrupt pointer's SHA-256 digest; raw pointer bytes never
          appear in either side.
        * ``chain`` compares the selection result (generation ``None``)
          and, per generation, the generation's existence, dependency
          verdict, manifest digest, validity and ignore reason.
        * ``checkpoint`` and ``journal`` compare each generation's file
          digest, and ``replay`` its terminal replay conclusion.

        Changes are ordered with the pointer first, then the selection,
        then per generation in ascending number order the chain,
        checkpoint, journal and replay changes. Nothing under ``root``
        -- the pointer, generations, business state, audits or
        idempotency records -- is modified.
        """
        cls._validate_recovery_root(root)
        body = cls._parse_recovery_audit_record(record)
        evidence = cls._gather_recovery_evidence(root)
        current_body = cls._recovery_audit_body(
            evidence, body["version"]
        )
        return cls._diff_recovery_bodies(body, current_body)

    @classmethod
    def _validate_chain_path(cls, path: Any) -> None:
        """Validate a chain file ``path`` argument: a non-``str`` raises
        :class:`TypeError`, an empty string :class:`ValueError`."""
        if not isinstance(path, str):
            raise TypeError(
                f"path must be a str, got {type(path).__name__}"
            )
        if not path:
            raise ValueError("path must be a non-empty str")

    @classmethod
    def _validate_head_digest(
        cls, value: Any, *, allow_none: bool
    ) -> None:
        """Validate a head digest argument. With ``allow_none`` the
        genesis marker ``None`` is accepted (the append entry); every
        other use requires a 64-character lowercase hexadecimal string.
        A wrong type raises :class:`TypeError` and a malformed string
        :class:`ValueError`."""
        if value is None:
            if allow_none:
                return
            raise TypeError("expected head digest must be a str, got NoneType")
        if not isinstance(value, str):
            raise TypeError(
                "head digest must be a str, got "
                f"{type(value).__name__}"
            )
        if len(value) != 64 or set(value) - cls._HEX_DIGITS:
            raise ValueError(
                "head digest must be a 64-character lowercase hex string"
            )

    @classmethod
    def _recovery_chain_frame_digest(
        cls, seq: int, prev: str, record: str
    ) -> str:
        """Compute a frame digest over exactly its sequence number, its
        predecessor link and the embedded canonical audit record, so a
        missing, duplicated, swapped, reordered, truncated or rewritten
        frame changes the digest the caller compares against."""
        signed = cls._canonical_json(
            {"seq": seq, "prev": prev, "record": record}
        )
        return hashlib.sha256(signed.encode("utf-8")).hexdigest()

    @classmethod
    def _load_recovery_chain(
        cls, path: str
    ) -> list[dict[str, Any]]:
        """Read, decode and fully validate a chain file, returning its
        detached ordered frames, each carrying the parsed (but not
        caller-shared) audit record body.

        Filesystem failures -- including a missing file, which
        propagates as :class:`FileNotFoundError` -- propagate as
        :class:`OSError`; an empty document and every structural,
        encoding, sequence or digest-link defect raise
        :class:`ValueError`. The caller validates ``path``'s type and
        emptiness.
        """
        with open(path, "rb") as handle:
            raw = handle.read()
        if raw.startswith(b"\xef\xbb\xbf"):
            raise ValueError("recovery chain must be UTF-8 without a BOM")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"recovery chain is not valid UTF-8: {exc}"
            ) from exc
        if not text:
            raise ValueError("recovery chain document is empty")
        try:
            document = json.loads(
                text, object_pairs_hook=cls._reject_duplicate_json_keys
            )
        except ValueError as exc:
            raise ValueError(
                f"recovery chain is not valid JSON: {exc}"
            ) from exc
        if cls._canonical_json(document) != text:
            raise ValueError(
                "recovery chain is not canonical compact JSON"
            )
        return cls._parse_recovery_chain_document(document)

    @classmethod
    def _parse_recovery_chain_document(
        cls, document: Any
    ) -> list[dict[str, Any]]:
        """Validate a parsed chain envelope and every frame it seals,
        returning detached frames ordered by sequence number. Any
        structural, field-domain, sequence or hash-link deviation raises
        :class:`ValueError`."""
        if not isinstance(document, dict):
            raise ValueError(
                "recovery chain top-level JSON value must be an object"
            )
        if set(document.keys()) != set(cls._RECOVERY_CHAIN_KEYS):
            raise ValueError(
                "recovery chain must contain exactly the keys 'format', "
                "'version' and 'frames'"
            )
        if document["format"] != cls._RECOVERY_CHAIN_FORMAT:
            raise ValueError("recovery chain has an unknown format")
        version = document["version"]
        if isinstance(version, bool) or not isinstance(version, int):
            raise ValueError("recovery chain 'version' must be an int")
        if version not in cls._RECOVERY_CHAIN_SUPPORTED_VERSIONS:
            raise ValueError(
                f"unsupported recovery chain version {version!r}"
            )
        frames_value = document["frames"]
        if not isinstance(frames_value, list):
            raise ValueError("recovery chain 'frames' must be an array")
        if not frames_value:
            raise ValueError("recovery chain document has no frames")
        frames: list[dict[str, Any]] = []
        previous_digest = cls._RECOVERY_CHAIN_GENESIS_PREV
        for position, frame in enumerate(frames_value, start=1):
            if not isinstance(frame, dict) or set(frame.keys()) != set(
                cls._RECOVERY_CHAIN_FRAME_KEYS
            ):
                raise ValueError(
                    "each recovery chain frame must contain exactly the "
                    "keys 'seq', 'prev', 'record' and 'digest'"
                )
            seq = frame["seq"]
            if isinstance(seq, bool) or not isinstance(seq, int):
                raise ValueError(
                    f"recovery chain frame {position}: 'seq' must be an int"
                )
            if seq != position:
                raise ValueError(
                    f"recovery chain frame {position}: sequence numbers "
                    "must be consecutive starting at 1"
                )
            prev = frame["prev"]
            if not isinstance(prev, str) or len(prev) != 64 or (
                set(prev) - cls._HEX_DIGITS
            ):
                raise ValueError(
                    f"recovery chain frame {seq}: 'prev' must be a "
                    "64-character lowercase hex string"
                )
            if not hmac.compare_digest(prev, previous_digest):
                raise ValueError(
                    f"recovery chain frame {seq}: predecessor digest "
                    "does not link to the previous frame"
                )
            record = frame["record"]
            if not isinstance(record, str) or not record:
                raise ValueError(
                    f"recovery chain frame {seq}: 'record' must be a "
                    "non-empty str"
                )
            # The embedded audit record must itself be a structurally
            # sound, canonical, checksummed recovery-audit envelope.
            body = cls._parse_recovery_audit_record(record)
            digest = frame["digest"]
            if not isinstance(digest, str) or len(digest) != 64 or (
                set(digest) - cls._HEX_DIGITS
            ):
                raise ValueError(
                    f"recovery chain frame {seq}: 'digest' must be a "
                    "64-character lowercase hex string"
                )
            expected_digest = cls._recovery_chain_frame_digest(
                seq, prev, record
            )
            if not hmac.compare_digest(digest, expected_digest):
                raise ValueError(
                    f"recovery chain frame {seq}: frame digest is invalid"
                )
            frames.append(
                {
                    "seq": seq,
                    "prev": prev,
                    "record": record,
                    "digest": digest,
                    "body": body,
                }
            )
            previous_digest = digest
        return frames

    @classmethod
    def _write_recovery_chain(
        cls, path: str, frames: list[dict[str, Any]]
    ) -> None:
        """Serialize the complete ordered chain and durably atomically
        replace ``path``.

        The bytes are UTF-8 (no BOM), compact JSON with no trailing
        newline; they are written to a temporary file in the same
        directory, flushed, fsync-ed and atomically moved onto ``path``
        after the directory entry of the temp file is synced. On any
        failure the previous chain file is left byte-for-byte untouched
        and the temporary file is removed; no partial frame set ever
        becomes visible.
        """
        document = {
            "format": cls._RECOVERY_CHAIN_FORMAT,
            "version": cls._RECOVERY_CHAIN_VERSION,
            "frames": [
                {
                    "seq": frame["seq"],
                    "prev": frame["prev"],
                    "record": frame["record"],
                    "digest": frame["digest"],
                }
                for frame in frames
            ],
        }
        data = cls._canonical_json(document).encode("utf-8")
        directory = os.path.dirname(os.path.abspath(path))
        fd, tmp_path = tempfile.mkstemp(
            prefix=".recovery-chain-", suffix=".tmp", dir=directory
        )
        replaced = False
        try:
            try:
                handle = os.fdopen(fd, "wb")
            except BaseException:
                # fdopen only reaches here without taking ownership of fd.
                with contextlib.suppress(OSError):
                    os.close(fd)
                raise
            with handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            cls._fsync_directory(directory)
            os.replace(tmp_path, path)
            replaced = True
            cls._fsync_directory(directory)
        finally:
            if not replaced:
                with contextlib.suppress(OSError):
                    os.remove(tmp_path)

    @classmethod
    def _validate_chain_timeout(cls, timeout: Any) -> None:
        """Validate an append ``timeout`` argument: ``None`` (wait
        indefinitely) or a non-``bool`` :class:`int`/:class:`float`
        number of seconds. Any other type raises :class:`TypeError`; a
        negative value, NaN or either infinity raises
        :class:`ValueError`."""
        if timeout is None:
            return
        if isinstance(timeout, bool) or not isinstance(
            timeout, (int, float)
        ):
            raise TypeError(
                f"timeout must be None, an int or a float, got "
                f"{type(timeout).__name__}"
            )
        if isinstance(timeout, float) and (
            math.isnan(timeout) or math.isinf(timeout)
        ):
            raise ValueError("timeout must be a finite number of seconds")
        if timeout < 0:
            raise ValueError("timeout must not be negative")

    @classmethod
    def _recovery_chain_lock_path(cls, path: str) -> str:
        """Canonical coordination lock file for a chain file. Because
        the lock name derives from the chain file's canonical (symlink-
        and alias-resolved) path, every equivalent spelling of the same
        chain file competes for the same operating-system file lock."""
        return os.path.realpath(path) + cls._RECOVERY_CHAIN_LOCK_SUFFIX

    @classmethod
    def _try_acquire_chain_lock(cls, fd: int) -> bool:
        """One non-blocking attempt at the exclusive lock on ``fd``;
        returns whether it was acquired. A genuine system failure (as
        opposed to the lock being held) propagates as
        :class:`OSError`."""
        if os.name == "nt":
            try:
                # Lock one byte at position zero; locking beyond the end
                # of an empty file is legal on Windows.
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                if exc.errno in (errno.EACCES, errno.EDEADLK):
                    return False
                raise
            return True
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        return True

    @classmethod
    def _acquire_chain_lock(cls, fd: int, timeout: float | None) -> None:
        """Acquire the exclusive chain lock on ``fd``, polling until it
        is granted. ``None`` waits indefinitely (the historical
        behaviour); a finite ``timeout`` bounds the wait and raises
        :class:`TimeoutError` when it elapses. Locking system failures
        propagate as :class:`OSError`."""
        if timeout is None:
            deadline: float | None = None
        else:
            try:
                seconds = float(timeout)
            except OverflowError:
                # An absurdly large integer outlasts any clock: treat it
                # as an unbounded wait rather than failing conversion.
                seconds = math.inf
            deadline = time.monotonic() + seconds
        while True:
            if cls._try_acquire_chain_lock(fd):
                return
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        "timed out waiting for the recovery audit chain "
                        "lock"
                    )
                time.sleep(min(cls._RECOVERY_CHAIN_LOCK_POLL, remaining))
            else:
                time.sleep(cls._RECOVERY_CHAIN_LOCK_POLL)

    @classmethod
    def _release_chain_lock(cls, fd: int) -> None:
        """Release the exclusive chain lock on ``fd``; closing the
        descriptor (or the process exiting, however abruptly) also
        releases it at the operating-system level."""
        if os.name == "nt":
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(fd, fcntl.LOCK_UN)

    @classmethod
    def _acquire_diagnostic_ledger_reader_lock(cls, fd: int) -> None:
        """Acquire a *shared* lock on the diagnostic-ledger coordination
        file for the duration of one directory page. Any number of
        readers may hold it concurrently, but it excludes an exclusive
        publisher lock for the whole publish -- including rotated
        segment and superseded-evidence garbage collection -- so a page
        can never observe a half-switched snapshot and a publisher can
        never delete a segment an in-progress reader may still open.
        Acquisition blocks until granted; a locking system failure
        raises :class:`OSError`. Closing ``fd`` (or the process exiting,
        however abruptly) releases the lock at the operating-system
        level, so a reader that crashes leaves no occupancy a later
        publisher must wait on or explicitly reclaim."""
        if os.name == "nt":
            os.lseek(fd, 0, os.SEEK_SET)
            while True:
                try:
                    # Shared (read) lock on one byte at position zero.
                    msvcrt.locking(fd, msvcrt.LK_NBRLCK, 1)
                    return
                except OSError as exc:
                    if exc.errno not in (errno.EACCES, errno.EDEADLK):
                        raise
                    time.sleep(cls._RECOVERY_CHAIN_LOCK_POLL)
        else:
            fcntl.flock(fd, fcntl.LOCK_SH)

    @classmethod
    def _release_diagnostic_ledger_reader_lock(cls, fd: int) -> None:
        """Release a shared directory-page reader lock on ``fd``."""
        if os.name == "nt":
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(fd, fcntl.LOCK_UN)

    @classmethod
    def _validate_index_path(cls, index_path: Any) -> None:
        """Validate an index file ``index_path`` argument: a non-``str``
        raises :class:`TypeError`, an empty string :class:`ValueError`."""
        if not isinstance(index_path, str):
            raise TypeError(
                f"index_path must be a str, got "
                f"{type(index_path).__name__}"
            )
        if not index_path:
            raise ValueError("index_path must be a non-empty str")

    @classmethod
    def _validate_index_target(cls, path: str, index_path: str) -> None:
        """The index must never name the chain file itself, even through
        aliases, so publishing the index can never overwrite the chain."""
        if os.path.realpath(path) == os.path.realpath(index_path):
            raise ValueError(
                "index_path must not name the recovery chain file"
            )

    @classmethod
    def _validate_progress_path(cls, progress_path: Any) -> None:
        """Validate a ``progress_path`` argument: ``None`` is accepted
        (progress disabled); otherwise it must be a non-empty :class:`str`
        -- a non-``str`` raises :class:`TypeError`, an empty string
        :class:`ValueError`."""
        if progress_path is None:
            return
        if not isinstance(progress_path, str):
            raise TypeError(
                "progress_path must be None or a str, got "
                f"{type(progress_path).__name__}"
            )
        if not progress_path:
            raise ValueError("progress_path must be a non-empty str")

    @classmethod
    def _validate_progress_target(
        cls, path: str, index_path: str, progress_path: str
    ) -> None:
        """The progress file must never name the chain or its index, even
        through aliases, so publishing progress can never overwrite
        either."""
        progress_real = os.path.realpath(progress_path)
        if progress_real == os.path.realpath(path):
            raise ValueError(
                "progress_path must not name the recovery chain file"
            )
        if progress_real == os.path.realpath(index_path):
            raise ValueError(
                "progress_path must not name the recovery chain index file"
            )

    @classmethod
    def _validate_recovery_path(cls, recovery_path: Any) -> None:
        """Validate a ``recovery_path`` argument: ``None`` is accepted
        (cross-process publication recovery disabled); otherwise it must
        be a non-empty :class:`str` -- a non-``str`` raises
        :class:`TypeError`, an empty string :class:`ValueError`."""
        if recovery_path is None:
            return
        if not isinstance(recovery_path, str):
            raise TypeError(
                "recovery_path must be None or a str, got "
                f"{type(recovery_path).__name__}"
            )
        if not recovery_path:
            raise ValueError("recovery_path must be a non-empty str")

    @classmethod
    def _validate_recovery_target(
        cls,
        path: str,
        index_path: str,
        progress_path: str | None,
        recovery_path: str,
    ) -> None:
        """The recovery record only coordinates a publication of both
        caches, so a progress cursor is required, and the record must
        never name the chain, index or progress file (even through an
        alias); the index, progress and record must all live in one
        directory so every backup and atomic replace shares a single
        synced directory. Any violation raises :class:`ValueError`."""
        if progress_path is None:
            raise ValueError(
                "recovery_path requires a progress_path so both caches "
                "are published together"
            )
        recovery_real = os.path.realpath(recovery_path)
        for other, label in (
            (path, "recovery chain file"),
            (index_path, "recovery chain index file"),
            (progress_path, "recovery chain progress file"),
        ):
            if recovery_real == os.path.realpath(other):
                raise ValueError(
                    f"recovery_path must not name the {label}"
                )
        directories = {
            os.path.dirname(os.path.realpath(index_path)),
            os.path.dirname(os.path.realpath(progress_path)),
            os.path.dirname(os.path.realpath(recovery_path)),
        }
        if len(directories) != 1:
            raise ValueError(
                "the index, progress and recovery files must all live in "
                "the same directory"
            )

    @classmethod
    def _open_recovery_chain_locked(cls, path: str) -> Any:
        """Open the chain file for reading while the caller holds the
        chain lock, returning a binary file object positioned at the
        start.

        The file identity is re-checked inside the lock: the path is
        statted, opened and the open descriptor statted again; if the
        directory entry was swapped in between (a writer not honouring
        the lock), the open is retried, and a file that keeps racing
        raises :class:`OSError`. Because appends replace the chain
        atomically under the same lock, the returned descriptor only
        ever exposes the complete chain from before or after an append.
        A missing or unreadable file raises :class:`OSError`.
        """
        for _attempt in range(3):
            identity = os.stat(path)
            fileobj = open(path, "rb")
            current = os.fstat(fileobj.fileno())
            if (
                current.st_dev == identity.st_dev
                and current.st_ino == identity.st_ino
            ):
                return fileobj
            fileobj.close()
        raise OSError(f"recovery chain file is unstable: {path!r}")

    @classmethod
    def _parse_chain_frame_stream(
        cls, stream: _CanonicalJsonStream
    ) -> dict[str, Any]:
        """Parse one frame object from ``stream`` and return its raw
        field values. Structural and encoding deviations raise
        :class:`ValueError`; the field-domain, sequence and digest-link
        checks happen in :meth:`_validate_chain_frame_fields`."""
        stream.expect(0x7B)  # '{'
        fields: dict[str, Any] = {}
        if stream.peek() == 0x7D:  # '}'
            stream.take()
        else:
            while True:
                if stream.peek() != 0x22:  # '"'
                    raise ValueError(
                        "recovery chain is not valid JSON: expected a "
                        "frame key"
                    )
                key = stream.parse_string()
                if key in fields:
                    raise ValueError(
                        f"duplicate key {key!r} in JSON object"
                    )
                stream.expect(0x3A)  # ':'
                if key == "seq":
                    fields[key] = stream.parse_number()
                elif key in ("prev", "record", "digest"):
                    if stream.peek() != 0x22:  # '"'
                        raise ValueError(
                            f"recovery chain frame field {key!r} must be "
                            "a string"
                        )
                    fields[key] = stream.parse_string()
                else:
                    stream.skip_value()
                    fields[key] = None
                byte = stream.take()
                if byte == 0x2C:  # ','
                    continue
                if byte == 0x7D:  # '}'
                    break
                raise ValueError(
                    "recovery chain is not valid JSON: unexpected content"
                )
        if set(fields.keys()) != set(cls._RECOVERY_CHAIN_FRAME_KEYS):
            raise ValueError(
                "each recovery chain frame must contain exactly the "
                "keys 'seq', 'prev', 'record' and 'digest'"
            )
        return fields

    @classmethod
    def _validate_chain_frame_fields(
        cls, fields: dict[str, Any], position: int, previous_digest: str
    ) -> tuple[dict[str, Any], str]:
        """Validate one parsed frame against the running chain state --
        consecutive numbering from one, the predecessor link, the
        embedded record's checksum and the frame digest -- and return
        its detached audit body and its digest."""
        seq = fields["seq"]
        if isinstance(seq, bool) or not isinstance(seq, int):
            raise ValueError(
                f"recovery chain frame {position}: 'seq' must be an int"
            )
        if seq != position:
            raise ValueError(
                f"recovery chain frame {position}: sequence numbers "
                "must be consecutive starting at 1"
            )
        prev = fields["prev"]
        if len(prev) != 64 or set(prev) - cls._HEX_DIGITS:
            raise ValueError(
                f"recovery chain frame {seq}: 'prev' must be a "
                "64-character lowercase hex string"
            )
        if not hmac.compare_digest(prev, previous_digest):
            raise ValueError(
                f"recovery chain frame {seq}: predecessor digest "
                "does not link to the previous frame"
            )
        record = fields["record"]
        if not record:
            raise ValueError(
                f"recovery chain frame {seq}: 'record' must be a "
                "non-empty str"
            )
        body = cls._parse_recovery_audit_record(record)
        digest = fields["digest"]
        if len(digest) != 64 or set(digest) - cls._HEX_DIGITS:
            raise ValueError(
                f"recovery chain frame {seq}: 'digest' must be a "
                "64-character lowercase hex string"
            )
        expected_digest = cls._recovery_chain_frame_digest(
            seq, prev, record
        )
        if not hmac.compare_digest(digest, expected_digest):
            raise ValueError(
                f"recovery chain frame {seq}: frame digest is invalid"
            )
        return body, digest

    @classmethod
    def _scan_recovery_chain(
        cls,
        fileobj: Any,
        on_frame: Any,
        raw_hasher: Any = None,
        on_frame_authenticated: Any = None,
    ) -> tuple[int, int, str]:
        """Stream the open chain file and fully authenticate it --
        encoding, canonical form, duplicate keys, structure, consecutive
        numbering, predecessor links, record checksums and frame
        digests -- while holding only one frame in memory.

        ``on_frame`` is invoked once per frame, in order, with a mapping
        carrying ``seq``, ``prev``, ``record``, ``digest``, ``body``
        (the parsed, detached audit body) and ``offset``/``length``
        (the frame's byte boundaries in the file); what the callback
        keeps is what bounds the caller's memory. Returns the chain's
        schema version, frame count and head digest. Every content
        defect raises :class:`ValueError`; read failures propagate as
        :class:`OSError`.

        When ``raw_hasher`` is given it is fed the chain's raw bytes in
        strict consumption order, and ``on_frame_authenticated`` -- if
        given -- is called with the stream immediately after each
        frame's full authentication, so a resuming builder can snapshot
        the digest of the raw authenticated prefix through a segment
        boundary without holding the chain or scanning it twice.
        """
        stream = _CanonicalJsonStream(fileobj, raw_hasher)
        if stream.peek() is None:
            raise ValueError("recovery chain document is empty")
        if stream.peek() == 0xEF:
            bom = bytes((stream.take(), stream.take(), stream.take()))
            if bom == b"\xef\xbb\xbf":
                raise ValueError(
                    "recovery chain must be UTF-8 without a BOM"
                )
            raise ValueError(
                "recovery chain is not valid JSON: unexpected content"
            )
        stream.expect(0x7B)  # '{'
        seen: set[str] = set()
        format_value: Any = None
        version_value: Any = None
        count = 0
        previous_digest = cls._RECOVERY_CHAIN_GENESIS_PREV
        head = ""
        if stream.peek() == 0x7D:  # '}'
            stream.take()
        else:
            while True:
                if stream.peek() != 0x22:  # '"'
                    raise ValueError(
                        "recovery chain is not valid JSON: expected an "
                        "object key"
                    )
                key = stream.parse_string()
                if key in seen:
                    raise ValueError(
                        f"duplicate key {key!r} in JSON object"
                    )
                seen.add(key)
                stream.expect(0x3A)  # ':'
                if key == "frames":
                    stream.expect(0x5B)  # '['
                    if stream.peek() == 0x5D:  # ']'
                        stream.take()
                    else:
                        while True:
                            frame_offset = stream.offset
                            fields = cls._parse_chain_frame_stream(stream)
                            frame_length = stream.offset - frame_offset
                            count += 1
                            body, digest = (
                                cls._validate_chain_frame_fields(
                                    fields, count, previous_digest
                                )
                            )
                            previous_digest = digest
                            head = digest
                            on_frame(
                                {
                                    "seq": count,
                                    "prev": fields["prev"],
                                    "record": fields["record"],
                                    "digest": digest,
                                    "body": body,
                                    "offset": frame_offset,
                                    "length": frame_length,
                                }
                            )
                            if on_frame_authenticated is not None:
                                on_frame_authenticated(stream)
                            byte = stream.take()
                            if byte == 0x2C:  # ','
                                continue
                            if byte == 0x5D:  # ']'
                                break
                            raise ValueError(
                                "recovery chain is not valid JSON: "
                                "unexpected content"
                            )
                elif key == "format":
                    if stream.peek() == 0x22:  # '"'
                        format_value = stream.parse_string()
                    else:
                        stream.skip_value()
                elif key == "version":
                    version_value = stream.parse_number()
                else:
                    stream.skip_value()
                byte = stream.take()
                if byte == 0x2C:  # ','
                    continue
                if byte == 0x7D:  # '}'
                    break
                raise ValueError(
                    "recovery chain is not valid JSON: unexpected content"
                )
        if not stream.at_end():
            raise ValueError(
                "recovery chain is not valid JSON: trailing data"
            )
        if seen != set(cls._RECOVERY_CHAIN_KEYS):
            raise ValueError(
                "recovery chain must contain exactly the keys 'format', "
                "'version' and 'frames'"
            )
        if format_value != cls._RECOVERY_CHAIN_FORMAT:
            raise ValueError("recovery chain has an unknown format")
        if isinstance(version_value, bool) or not isinstance(
            version_value, int
        ):
            raise ValueError("recovery chain 'version' must be an int")
        if version_value not in cls._RECOVERY_CHAIN_SUPPORTED_VERSIONS:
            raise ValueError(
                f"unsupported recovery chain version {version_value!r}"
            )
        if not count:
            raise ValueError("recovery chain document has no frames")
        return version_value, count, head

    @classmethod
    def _build_recovery_audit_index_locked(
        cls, path: str, index_path: str
    ) -> str:
        """Stream-authenticate the chain and durably publish its index,
        returning the chain head digest; the caller holds the chain
        lock.

        Index entries are written as the frames stream past, so no more
        than one frame or audit body is ever materialized. The index
        bytes are UTF-8 without a BOM, compact JSON without a trailing
        newline, and identical for identical chains. They are written
        to a temporary file in the index's directory, flushed, fsync-ed
        and atomically moved onto ``index_path`` with directory syncs;
        on any failure the temporary file is removed and neither the
        chain file nor a previous index is modified.
        """
        chain_file = cls._open_recovery_chain_locked(path)
        try:
            directory = os.path.dirname(os.path.abspath(index_path))
            fd, tmp_path = tempfile.mkstemp(
                prefix=".recovery-chain-index-",
                suffix=".tmp",
                dir=directory,
            )
            published = False
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(
                        b'{"format":"'
                        + cls._RECOVERY_CHAIN_INDEX_FORMAT.encode("ascii")
                        + b'","version":'
                        + str(cls._RECOVERY_CHAIN_INDEX_VERSION).encode(
                            "ascii"
                        )
                        + b',"entries":['
                    )
                    first = True

                    def on_frame(frame: dict[str, Any]) -> None:
                        nonlocal first
                        entry = {
                            "seq": frame["seq"],
                            "offset": frame["offset"],
                            "length": frame["length"],
                            "digest": frame["digest"],
                        }
                        if not first:
                            handle.write(b",")
                        handle.write(
                            cls._canonical_json(entry).encode("utf-8")
                        )
                        first = False

                    version, count, head = cls._scan_recovery_chain(
                        chain_file, on_frame
                    )
                    handle.write(
                        (
                            '],"chain_version":%d,"frames":%d,"head":"%s"}'
                            % (version, count, head)
                        ).encode("ascii")
                    )
                    handle.flush()
                    os.fsync(handle.fileno())
                cls._fsync_directory(directory)
                os.replace(tmp_path, index_path)
                published = True
                cls._fsync_directory(directory)
            finally:
                if not published:
                    with contextlib.suppress(OSError):
                        os.remove(tmp_path)
        finally:
            chain_file.close()
        return head

    @classmethod
    def _scan_index_entries(
        cls, stream: _CanonicalJsonStream, fingerprint: Any
    ) -> tuple[int, str | None]:
        """Stream the index's ``entries`` array, validating each entry's
        shape, consecutive numbering, byte boundaries and digest
        evidence, and return the entry count and the last entry's
        digest. ``fingerprint`` is a SHA-256 hasher updated with each
        entry's canonical bytes, so the caller can compare the entries
        item for item against the authenticated chain frames. Any defect
        raises :class:`ValueError`, which the caller treats as a corrupt
        (rebuildable) index."""
        stream.expect(0x5B)  # '['
        count = 0
        last_digest: str | None = None
        if stream.peek() == 0x5D:  # ']'
            stream.take()
            return count, last_digest
        while True:
            stream.expect(0x7B)  # '{'
            fields: dict[str, Any] = {}
            if stream.peek() == 0x7D:  # '}'
                stream.take()
            else:
                while True:
                    if stream.peek() != 0x22:  # '"'
                        raise ValueError("expected an entry key")
                    key = stream.parse_string()
                    if key in fields:
                        raise ValueError(
                            f"duplicate key {key!r} in JSON object"
                        )
                    stream.expect(0x3A)  # ':'
                    if key in ("seq", "offset", "length"):
                        fields[key] = stream.parse_number()
                    elif key == "digest":
                        if stream.peek() != 0x22:  # '"'
                            raise ValueError("expected a string")
                        fields[key] = stream.parse_string()
                    else:
                        stream.skip_value()
                        fields[key] = None
                    byte = stream.take()
                    if byte == 0x2C:  # ','
                        continue
                    if byte == 0x7D:  # '}'
                        break
                    raise ValueError("unexpected content")
            if set(fields.keys()) != set(
                cls._RECOVERY_CHAIN_INDEX_ENTRY_KEYS
            ):
                raise ValueError("bad entry keys")
            count += 1
            seq = fields["seq"]
            if (
                isinstance(seq, bool)
                or not isinstance(seq, int)
                or seq != count
            ):
                raise ValueError("bad entry sequence")
            for name in ("offset", "length"):
                value = fields[name]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                ):
                    raise ValueError("bad entry boundary")
            digest = fields["digest"]
            if len(digest) != 64 or set(digest) - cls._HEX_DIGITS:
                raise ValueError("bad entry digest")
            fingerprint.update(
                cls._canonical_json(
                    {
                        "seq": seq,
                        "offset": fields["offset"],
                        "length": fields["length"],
                        "digest": digest,
                    }
                ).encode("utf-8")
            )
            last_digest = digest
            byte = stream.take()
            if byte == 0x2C:  # ','
                continue
            if byte == 0x5D:  # ']'
                break
            raise ValueError("unexpected content")
        return count, last_digest

    @classmethod
    def _read_recovery_chain_index(
        cls, index_path: str
    ) -> tuple[int, int, str, str] | None:
        """Read and fully validate the index, returning its binding --
        ``(chain_version, frames, head, entries_fingerprint)`` -- or
        ``None`` when the index is missing, unreadable or defective in
        any way. The fingerprint is the SHA-256 over every entry's
        canonical bytes in order, so the caller can check that each
        entry's sequence number, byte offset, length and digest evidence
        matches the corresponding authenticated chain frame item for
        item.

        The index is a disposable cache, so no index problem ever raises
        here: it only marks the index for a rebuild from the canonical
        chain. Only a fixed amount of memory is used regardless of the
        entry count.
        """
        try:
            with open(index_path, "rb") as fileobj:
                stream = _CanonicalJsonStream(fileobj)
                if stream.peek() is None or stream.peek() == 0xEF:
                    return None
                stream.expect(0x7B)  # '{'
                seen: set[str] = set()
                format_value: Any = None
                version_value: Any = None
                chain_version: Any = None
                declared_frames: Any = None
                head: Any = None
                count = 0
                last_digest: str | None = None
                fingerprint = hashlib.sha256()
                if stream.peek() == 0x7D:  # '}'
                    stream.take()
                else:
                    while True:
                        if stream.peek() != 0x22:  # '"'
                            raise ValueError("expected an object key")
                        key = stream.parse_string()
                        if key in seen:
                            raise ValueError(
                                f"duplicate key {key!r} in JSON object"
                            )
                        seen.add(key)
                        stream.expect(0x3A)  # ':'
                        if key == "entries":
                            count, last_digest = cls._scan_index_entries(
                                stream, fingerprint
                            )
                        elif key in ("format", "head"):
                            if stream.peek() != 0x22:  # '"'
                                raise ValueError("expected a string")
                            value = stream.parse_string()
                            if key == "format":
                                format_value = value
                            else:
                                head = value
                        elif key in ("version", "chain_version", "frames"):
                            value = stream.parse_number()
                            if key == "version":
                                version_value = value
                            elif key == "chain_version":
                                chain_version = value
                            else:
                                declared_frames = value
                        else:
                            stream.skip_value()
                        byte = stream.take()
                        if byte == 0x2C:  # ','
                            continue
                        if byte == 0x7D:  # '}'
                            break
                        raise ValueError("unexpected content")
                if not stream.at_end():
                    return None
                if seen != set(cls._RECOVERY_CHAIN_INDEX_KEYS):
                    return None
                if format_value != cls._RECOVERY_CHAIN_INDEX_FORMAT:
                    return None
                if (
                    isinstance(version_value, bool)
                    or not isinstance(version_value, int)
                    or version_value
                    not in cls._RECOVERY_CHAIN_INDEX_SUPPORTED_VERSIONS
                ):
                    return None
                if (
                    isinstance(chain_version, bool)
                    or not isinstance(chain_version, int)
                    or chain_version
                    not in cls._RECOVERY_CHAIN_SUPPORTED_VERSIONS
                ):
                    return None
                if (
                    isinstance(declared_frames, bool)
                    or not isinstance(declared_frames, int)
                    or declared_frames != count
                    or count < 1
                ):
                    return None
                if (
                    not isinstance(head, str)
                    or len(head) != 64
                    or set(head) - cls._HEX_DIGITS
                    or head != last_digest
                ):
                    return None
                return (
                    chain_version,
                    declared_frames,
                    head,
                    fingerprint.hexdigest(),
                )
        except (OSError, ValueError, RecursionError):
            return None

    @classmethod
    def _indexed_chain_scan(
        cls,
        path: str,
        index_path: str,
        wanted: set[int] | None,
    ) -> tuple[int, str, dict[int, dict[str, Any]]]:
        """Authenticate the chain with bounded memory under the chain
        lock, keep the index fresh and return the frame count, the head
        digest and the detached audit bodies of the ``wanted`` sequence
        numbers.

        The chain is streamed and fully authenticated to the same
        standard as :meth:`_load_recovery_chain`, but only one frame is
        ever materialized, plus the bodies the caller asked for, so
        memory grows only with the largest single frame and fixed
        buffers -- never with the total frame count or a queried range
        span. The index is then re-read and every entry's sequence
        number, byte offset, length and digest evidence is checked
        against the corresponding authenticated chain frame (via a
        rolling SHA-256 fingerprint, so no entry list is materialized);
        any entry that does not point at its canonical frame marks the
        index corrupt. A missing, stale or corrupt index is rebuilt once
        from the just-authenticated canonical chain and re-checked; if
        it still does not match, :class:`ValueError` is raised. The
        chain remains the only trusted source and the index never
        answers a query by itself, so no index content can mask a chain
        defect.
        """
        lock_path = cls._recovery_chain_lock_path(path)
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            cls._acquire_chain_lock(fd, None)
            try:
                captured: dict[int, dict[str, Any]] = {}
                fingerprint = hashlib.sha256()

                def on_frame(frame: dict[str, Any]) -> None:
                    if wanted is not None and frame["seq"] in wanted:
                        captured[frame["seq"]] = frame["body"]
                    fingerprint.update(
                        cls._canonical_json(
                            {
                                "seq": frame["seq"],
                                "offset": frame["offset"],
                                "length": frame["length"],
                                "digest": frame["digest"],
                            }
                        ).encode("utf-8")
                    )

                chain_file = cls._open_recovery_chain_locked(path)
                try:
                    version, count, head = cls._scan_recovery_chain(
                        chain_file, on_frame
                    )
                finally:
                    chain_file.close()
                expected = (version, count, head, fingerprint.hexdigest())
                binding = cls._read_recovery_chain_index(index_path)
                if binding != expected:
                    cls._build_recovery_audit_index_locked(
                        path, index_path
                    )
                    binding = cls._read_recovery_chain_index(index_path)
                    if binding != expected:
                        raise ValueError(
                            "recovery chain index does not match the "
                            "authenticated chain"
                        )
                return count, head, captured
            finally:
                cls._release_chain_lock(fd)
        finally:
            os.close(fd)

    @classmethod
    def append_recovery_audit_chain(
        cls,
        root: Any,
        path: Any,
        previous: Any,
        timeout: Any = None,
    ) -> str:
        """Append one fresh recovery-audit sample to a persistent,
        tamper-evident chain, returning the new head digest.

        ``root`` is validated exactly as for
        :meth:`export_recovery_audit`: a non-``str`` raises
        :class:`TypeError`, an empty string :class:`ValueError`, and a
        missing root, non-directory, enumeration or read failure
        :class:`OSError`; the scan is strictly read-only. ``path`` is
        validated next as a non-empty :class:`str` (non-``str``
        :class:`TypeError`, empty :class:`ValueError`); a failed chain
        read or write propagates :class:`OSError`. ``previous`` is the
        caller-saved previous head digest: it must be ``None`` or a
        64-character lowercase hexadecimal string, otherwise
        :class:`TypeError` for the wrong type and :class:`ValueError` for
        an illegal format. ``timeout`` is validated last: it must be
        ``None`` or a non-``bool`` :class:`int`/:class:`float` number of
        seconds (anything else raises :class:`TypeError`; a negative
        value, NaN or either infinity raises :class:`ValueError`).

        Appends are serialized across processes by an exclusive
        operating-system file lock on a fixed coordination file named
        after the chain file's canonical path, so equivalent spellings
        of the same chain file compete for the same lock on Windows and
        POSIX alike. Failing to create, open or lock that file raises
        :class:`OSError` before the chain file or any generation
        evidence is touched. ``timeout`` bounds only the wait for the
        lock: ``None`` (the default) blocks until the lock is granted,
        and a finite timeout that elapses first raises
        :class:`TimeoutError`. While waiting, no business, audit or
        idempotency state is held or modified, and a process dying
        mid-append releases the lock at the operating-system level, so
        the next caller proceeds.

        The first frame may only be created with ``previous`` equal to
        ``None``; once the chain exists, ``previous`` must equal the
        current last frame's digest. The existence check and the head
        comparison happen under the lock against a freshly re-read and
        fully re-authenticated chain, never against state observed
        before the lock was taken, so concurrent callers holding the
        same valid old head see at most one append succeed: every other
        caller acquires the lock afterwards and raises
        :class:`RuntimeError` without overwriting the winning frame, and
        concurrent first-time creations likewise admit exactly one
        chain. A mismatch raises :class:`RuntimeError` and leaves the
        file byte-for-byte unchanged.

        Each append first exports the current recovery evidence as a
        canonical, signed audit record and then fully re-verifies the
        existing chain (structure, canonical encoding, every embedded
        record's checksum, consecutive numbering and the digest links)
        before sealing one more frame -- all inside the same lock hold,
        so the export, the authentication, the new frame and the durable
        atomic replace (including the final directory sync) are one
        serialized step. A frame is added even when the audit record is
        identical to the previous sample, so the fact that a sample was
        taken survives. Frames are numbered from one; the first frame
        roots the chain at an all-zero predecessor digest, and every
        later frame cites the prior frame's digest. Each frame digest
        covers its sequence number, predecessor link and embedded
        record, so deletion, duplication, swapping, reordering,
        truncation or rewriting is detectable against the externally
        retained head digest.

        The whole chain is rewritten in the same directory and durably
        atomically replaced; a write, flush, replace or sync failure
        raises :class:`OSError`, leaves the old chain byte-for-byte
        untouched and exposes no partial frame, so concurrent
        :meth:`verify_recovery_audit_chain` and
        :meth:`diff_recovery_audit_range` readers only ever observe the
        complete document from before or after the append. Temporary
        chain files created by the call are removed on every path; the
        fixed coordination lock file is not temporary residue and never
        carries audit content. Nothing under ``root`` -- the pointer,
        generations, business state, audits or idempotency records -- is
        modified, whether the call succeeds, times out or loses the
        head comparison.
        """
        cls._validate_recovery_root(root)
        cls._validate_chain_path(path)
        cls._validate_head_digest(previous, allow_none=True)
        cls._validate_chain_timeout(timeout)

        lock_path = cls._recovery_chain_lock_path(path)
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            cls._acquire_chain_lock(fd, timeout)
            try:
                # Export first so a directory with no recoverable
                # generation raises ValueError before the chain file is
                # consulted; the export, authentication, framing and
                # durable replace all run inside this one lock hold.
                record = cls.export_recovery_audit(root)

                try:
                    frames = cls._load_recovery_chain(path)
                except FileNotFoundError:
                    frames = []
                if not frames:
                    if previous is not None:
                        raise RuntimeError(
                            "no recovery chain exists yet, so the "
                            "previous head digest must be None"
                        )
                    prev = cls._RECOVERY_CHAIN_GENESIS_PREV
                else:
                    if previous is None:
                        raise RuntimeError(
                            "recovery chain already exists, so the "
                            "previous head digest is required"
                        )
                    if not hmac.compare_digest(
                        frames[-1]["digest"], previous
                    ):
                        raise RuntimeError(
                            "previous head digest does not match the "
                            "current end of the recovery chain"
                        )
                    prev = frames[-1]["digest"]
                seq = len(frames) + 1
                digest = cls._recovery_chain_frame_digest(seq, prev, record)
                frames.append(
                    {
                        "seq": seq,
                        "prev": prev,
                        "record": record,
                        "digest": digest,
                    }
                )
                cls._write_recovery_chain(path, frames)
            finally:
                cls._release_chain_lock(fd)
        finally:
            os.close(fd)
        return digest

    @classmethod
    def verify_recovery_audit_chain(
        cls, path: Any, expected_head: Any, index_path: Any = None
    ) -> bool:
        """Authenticate a persistent recovery-audit chain strictly
        read-only.

        ``path`` must be a non-empty :class:`str` (a non-``str`` raises
        :class:`TypeError`, an empty string :class:`ValueError`); a
        missing or unreadable file raises :class:`OSError`. Any content
        defect -- an empty document, a non-UTF-8 or BOM-bearing file,
        malformed JSON, duplicate keys, a non-canonical encoding, a
        structural or version violation, a malformed embedded audit
        record, a non-consecutive sequence number or a broken
        predecessor/frame digest link -- raises :class:`ValueError`.

        When the chain is internally sound, the last frame's digest is
        compared against ``expected_head`` (a 64-character lowercase
        hexadecimal string, validated like the append argument): exact
        equality returns ``True``, any difference -- including a rolled
        back, truncated or forked chain -- returns ``False``. The check
        never modifies the chain, the generation directory or any
        business, audit or idempotency state, and its result is
        detached from the parsed document.

        ``index_path`` is optional; ``None`` (the default) keeps the
        behaviour above exactly. When given, it must be a non-empty
        :class:`str` (validated like ``path``) naming the chain's
        disposable index, and the call competes with appends for the
        same chain lock, re-checks the chain file's identity inside the
        lock and streams the whole chain with the same full
        authentication, so memory grows only with the largest single
        frame and fixed buffers -- never with the total frame count.
        The external head is still checked against the authenticated
        chain, never against the index: the index cannot mask any chain
        defect. Afterwards the index is re-read and every entry's
        sequence number, offset, length and digest evidence is checked
        against the corresponding authenticated chain frame; a missing,
        stale, truncated or corrupt index is rebuilt once from the
        canonical chain and durably atomically published, and an index
        that still does not match after the rebuild raises
        :class:`ValueError`. A failed rebuild propagates
        :class:`OSError` and leaves any previous index and the chain
        untouched.
        """
        if index_path is not None:
            cls._validate_chain_path(path)
            cls._validate_index_path(index_path)
            cls._validate_index_target(path, index_path)
            cls._validate_head_digest(expected_head, allow_none=False)
            _count, head, _captured = cls._indexed_chain_scan(
                path, index_path, None
            )
            return hmac.compare_digest(head, expected_head)
        cls._validate_chain_path(path)
        cls._validate_head_digest(expected_head, allow_none=False)
        frames = cls._load_recovery_chain(path)
        if not frames:
            raise ValueError("recovery chain document has no frames")
        return hmac.compare_digest(frames[-1]["digest"], expected_head)

    @classmethod
    def diff_recovery_audit_range(
        cls,
        path: Any,
        expected_head: Any,
        start: Any,
        end: Any,
        index_path: Any = None,
    ) -> tuple[tuple[Any, ...], ...]:
        """Compare the audit records sealed at two chain positions,
        strictly read-only.

        ``path`` and ``expected_head`` are authenticated exactly as for
        :meth:`verify_recovery_audit_chain`: an invalid path raises
        :class:`TypeError`/:class:`ValueError`, a missing or unreadable
        file :class:`OSError`, any chain defect :class:`ValueError`, and a
        sound chain whose last frame does not match ``expected_head``
        raises :class:`ValueError` rather than reporting a diff over
        unauthenticated material.

        ``start`` and ``end`` must be non-``bool`` integers, otherwise
        :class:`TypeError`; a value below one, a start later than end,
        or a value past the chain's last frame raises
        :class:`ValueError`. Both endpoints' embedded audit records are
        compared with the same classification, identities and stable
        ordering as :meth:`diff_recovery_audit`; equal sequence numbers
        yield an empty tuple. Nothing on disk or any business, audit or
        idempotency state is modified, and the returned tuples are
        detached from the parsed chain.

        ``index_path`` is optional; ``None`` (the default) keeps the
        behaviour above exactly. When given, it is validated like
        ``path`` and the call runs under the same chain lock that
        serializes appends, streaming the fully authenticated chain so
        memory grows only with the largest single frame and fixed
        buffers -- never with the total frame count or the range span.
        The two endpoint records are still taken from the authenticated
        canonical chain and the result is item-for-item identical to
        the no-index result; the index never answers by itself and
        cannot mask a chain defect. Every index entry's sequence number,
        offset, length and digest evidence is checked against the
        corresponding authenticated chain frame; a missing, stale,
        truncated or corrupt index is rebuilt once from the canonical
        chain and durably atomically published, and an index that still
        does not match after the rebuild raises :class:`ValueError`. A
        failed rebuild propagates :class:`OSError`.
        """
        cls._validate_chain_path(path)
        if index_path is not None:
            cls._validate_index_path(index_path)
            cls._validate_index_target(path, index_path)
        cls._validate_head_digest(expected_head, allow_none=False)
        for name, value in (("start", start), ("end", end)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"{name} must be an int, got {type(value).__name__}"
                )
        if index_path is None:
            frames = cls._load_recovery_chain(path)
            if not frames:
                raise ValueError("recovery chain document has no frames")
            if not hmac.compare_digest(frames[-1]["digest"], expected_head):
                raise ValueError(
                    "recovery chain head does not match the expected head "
                    "digest"
                )
            last = frames[-1]["seq"]
            if start < 1 or end < 1:
                raise ValueError(
                    "range endpoints must be greater than or equal to 1"
                )
            if start > end:
                raise ValueError("range start must not be later than end")
            if end > last:
                raise ValueError(
                    f"range end {end} is past the last frame {last}"
                )
            if start == end:
                return ()
            before = frames[start - 1]["body"]
            after = frames[end - 1]["body"]
            return cls._diff_recovery_bodies(before, after)
        last, head, captured = cls._indexed_chain_scan(
            path, index_path, {start, end}
        )
        if not hmac.compare_digest(head, expected_head):
            raise ValueError(
                "recovery chain head does not match the expected head "
                "digest"
            )
        if start < 1 or end < 1:
            raise ValueError(
                "range endpoints must be greater than or equal to 1"
            )
        if start > end:
            raise ValueError("range start must not be later than end")
        if end > last:
            raise ValueError(
                f"range end {end} is past the last frame {last}"
            )
        if start == end:
            return ()
        return cls._diff_recovery_bodies(
            captured[start], captured[end]
        )

    @classmethod
    def build_recovery_audit_index(
        cls, path: Any, index_path: Any
    ) -> str:
        """Build the disposable bounded-memory index over a persistent
        recovery-audit chain and return the chain's head digest.

        ``path`` and ``index_path`` must be non-empty :class:`str`
        values (a non-``str`` raises :class:`TypeError`, an empty string
        :class:`ValueError`), and ``index_path`` must not name the chain
        file itself, even through an alias (:class:`ValueError`). A
        missing or unreadable chain file, a missing index directory or
        any failed open, flush, replace or sync raises
        :class:`OSError` and publishes nothing.

        The build competes with appends for the same chain lock and
        re-checks the chain file's identity inside the lock, so it only
        ever observes the complete chain from before or after an
        append. While holding the lock it streams the chain and fully
        authenticates it -- encoding, canonical form, duplicate keys,
        frame order, predecessor links, embedded record checksums and
        frame digests -- without ever materializing all frames or audit
        bodies at once; any chain defect raises :class:`ValueError` and
        publishes nothing. It then writes an index bound to the chain's
        schema version, frame count and last-frame digest, recording
        for every frame the byte boundaries needed to locate it and its
        digest evidence. The index bytes are UTF-8 without a BOM,
        compact JSON without a trailing newline, and identical for
        identical chains; they are written to a temporary file in the
        index's directory, flushed, fsync-ed and atomically moved onto
        ``index_path`` with directory syncs, so concurrent readers see
        either the old index or the new one, never a partial document.

        The index is a disposable cache: the canonical chain stays the
        only trusted source, and :meth:`verify_recovery_audit_chain`
        and :meth:`diff_recovery_audit_range` rebuild it whenever it is
        missing, stale, truncated or corrupt. A crashed or failed build
        never rewrites the chain file, leaves any previous index valid
        or recognizably stale, and removes its temporary file. Whether
        the build succeeds or fails, nothing under the generation
        directory -- business state, audits, idempotency records or the
        journal attachment -- is modified.
        """
        cls._validate_chain_path(path)
        cls._validate_index_path(index_path)
        cls._validate_index_target(path, index_path)
        lock_path = cls._recovery_chain_lock_path(path)
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            cls._acquire_chain_lock(fd, None)
            try:
                return cls._build_recovery_audit_index_locked(
                    path, index_path
                )
            finally:
                cls._release_chain_lock(fd)
        finally:
            os.close(fd)

    @classmethod
    def _validate_segment_size(cls, segment_size: Any) -> None:
        """Validate a ``segment_size`` argument: a non-``bool``
        :class:`int`, otherwise :class:`TypeError`; a value below one
        raises :class:`ValueError`."""
        if isinstance(segment_size, bool) or not isinstance(
            segment_size, int
        ):
            raise TypeError(
                f"segment_size must be an int, got "
                f"{type(segment_size).__name__}"
            )
        if segment_size < 1:
            raise ValueError(
                "segment_size must be greater than or equal to 1"
            )

    #: Fixed chunk size for streaming a byte-for-byte backup copy.
    _RECOVERY_CACHE_BACKUP_CHUNK = 65536

    @classmethod
    def _segments_envelope_prefix(cls, segment_size: int) -> bytes:
        """Opening bytes of a segmented-index document through its
        ``segments`` array opening; segment records follow."""
        return (
            b'{"format":"'
            + cls._RECOVERY_CHAIN_SEGMENTS_FORMAT.encode("ascii")
            + b'","version":'
            + str(cls._RECOVERY_CHAIN_SEGMENTS_VERSION).encode("ascii")
            + b',"segment_size":'
            + str(segment_size).encode("ascii")
            + b',"segments":['
        )

    @staticmethod
    def _segments_trailer(version: int, count: int, head: str) -> bytes:
        """Closing bytes of a segmented-index document."""
        return (
            '],"chain_version":%d,"frames":%d,"head":"%s"}'
            % (version, count, head)
        ).encode("ascii")

    @classmethod
    def _progress_document_bytes(cls, payload: dict[str, Any]) -> bytes:
        """Deterministic compact JSON bytes for a normalized progress
        cursor; the document is a fixed-shape object with no audit
        content."""
        boundary = payload["boundary"]
        document = {
            "format": cls._RECOVERY_CHAIN_PROGRESS_FORMAT,
            "version": cls._RECOVERY_CHAIN_PROGRESS_VERSION,
            "segment_size": payload["segment_size"],
            "frames": payload["frames"],
            "head": payload["head"],
            "boundary": {
                "seq": boundary["seq"],
                "offset": boundary["offset"],
                "length": boundary["length"],
                "digest": boundary["digest"],
            },
            "prefix": payload["prefix"],
            "index": payload["index"],
        }
        return cls._canonical_json(document).encode("utf-8")

    @classmethod
    def _backup_cache_file(cls, target: str) -> str:
        """Copy the existing target's bytes to a durable temp file in the
        same directory, returning the backup path. The copy is streamed
        in fixed chunks and fsync-ed; any failure removes the partial
        backup and propagates :class:`OSError`."""
        directory = os.path.dirname(os.path.abspath(target))
        fd, backup_path = tempfile.mkstemp(
            prefix=".recovery-cache-backup-", suffix=".tmp", dir=directory
        )
        finished = False
        try:
            try:
                handle = os.fdopen(fd, "wb")
            except BaseException:
                with contextlib.suppress(OSError):
                    os.close(fd)
                with contextlib.suppress(OSError):
                    os.remove(backup_path)
                raise
            with handle:
                with open(target, "rb") as source:
                    while True:
                        chunk = source.read(cls._RECOVERY_CACHE_BACKUP_CHUNK)
                        if not chunk:
                            break
                        handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
            finished = True
            return backup_path
        finally:
            if not finished:
                with contextlib.suppress(OSError):
                    os.remove(backup_path)

    @classmethod
    def _publish_cache_files(
        cls, publications: list[tuple[str, str]]
    ) -> None:
        """Atomically publish freshly built cache files under a
        rollback-safe protocol.

        Each entry pairs a target path with an already-flushed,
        fsync-ed temp file holding its new bytes. Every pre-existing
        target is first streamed to a durable byte-for-byte backup in
        the same directory (no target moves during preparation, so
        readers always see the old file); the targets are then replaced
        one at a time, each preceded and followed by a directory sync.
        If any backup, replace or sync step fails, every target already
        replaced is restored byte-for-byte from its backup, targets not
        yet replaced keep their old bytes, and every temp and backup
        from this call is removed; :class:`OSError` is raised and no
        partial publication remains. On success every backup is
        removed. The canonical chain and all generation state are
        untouched either way.
        """
        prepared: list[dict[str, Any]] = []
        try:
            for target, tmp_path in publications:
                directory = os.path.dirname(os.path.abspath(target))
                backup_path: str | None = None
                if os.path.lexists(target):
                    backup_path = cls._backup_cache_file(target)
                prepared.append(
                    {
                        "target": target,
                        "tmp": tmp_path,
                        "directory": directory,
                        "backup": backup_path,
                        "replaced": False,
                    }
                )
            for entry in prepared:
                cls._fsync_directory(entry["directory"])
                os.replace(entry["tmp"], entry["target"])
                entry["replaced"] = True
                cls._fsync_directory(entry["directory"])
        except BaseException:
            for entry in prepared:
                if entry["replaced"]:
                    if entry["backup"] is not None:
                        with contextlib.suppress(OSError):
                            os.replace(entry["backup"], entry["target"])
                        entry["backup"] = None
                    else:
                        # The target did not exist before; remove the
                        # file this call published to restore absence.
                        with contextlib.suppress(OSError):
                            os.remove(entry["target"])
                    with contextlib.suppress(OSError):
                        cls._fsync_directory(entry["directory"])
                else:
                    with contextlib.suppress(OSError):
                        os.remove(entry["tmp"])
            for entry in prepared:
                if entry["backup"] is not None:
                    with contextlib.suppress(OSError):
                        os.remove(entry["backup"])
            raise
        for entry in prepared:
            if entry["backup"] is not None:
                with contextlib.suppress(OSError):
                    os.remove(entry["backup"])
                with contextlib.suppress(OSError):
                    cls._fsync_directory(entry["directory"])

    @classmethod
    def _sha256_file(cls, path: str) -> str:
        """Hex SHA-256 of a file's bytes, streamed in fixed chunks."""
        hasher = hashlib.sha256()
        with open(path, "rb") as handle:
            while True:
                chunk = handle.read(cls._RECOVERY_CACHE_BACKUP_CHUNK)
                if not chunk:
                    break
                hasher.update(chunk)
        return hasher.hexdigest()

    @classmethod
    def _cache_recovery_record_bytes(
        cls,
        phase: str,
        index_target: dict[str, Any],
        progress_target: dict[str, Any],
    ) -> bytes:
        """Deterministic compact JSON bytes for a cache-publication
        recovery record: the checksum is the SHA-256 over the canonical
        bytes of every other field and is serialized last."""
        body = {
            "format": cls._RECOVERY_CACHE_RECORD_FORMAT,
            "version": cls._RECOVERY_CACHE_RECORD_VERSION,
            "phase": phase,
            "index": index_target,
            "progress": progress_target,
        }
        document = dict(body)
        document["checksum"] = hashlib.sha256(
            cls._canonical_json(body).encode("utf-8")
        ).hexdigest()
        return cls._canonical_json(document).encode("utf-8")

    @classmethod
    def _parse_cache_recovery_record(cls, raw: bytes) -> dict[str, Any]:
        """Strictly parse and authenticate a cache-publication recovery
        record, returning its normalized mapping. Any encoding, JSON,
        canonical-form, field, type, version, phase or checksum defect
        raises :class:`ValueError`; the caller preserves all evidence."""
        if raw.startswith(b"\xef\xbb\xbf"):
            raise ValueError("recovery record must be UTF-8 without a BOM")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"recovery record is not valid UTF-8: {exc}")
        if not text:
            raise ValueError("recovery record is empty")
        try:
            document = json.loads(
                text, object_pairs_hook=cls._reject_duplicate_json_keys
            )
        except ValueError as exc:
            raise ValueError(f"recovery record is not valid JSON: {exc}")
        if cls._canonical_json(document) != text:
            raise ValueError(
                "recovery record is not canonical compact JSON"
            )
        if not isinstance(document, dict) or set(document) != set(
            cls._RECOVERY_CACHE_RECORD_KEYS
        ):
            raise ValueError("recovery record has bad top-level keys")
        if document["format"] != cls._RECOVERY_CACHE_RECORD_FORMAT:
            raise ValueError("recovery record has an unknown format")
        version = document["version"]
        if (
            isinstance(version, bool)
            or not isinstance(version, int)
            or version not in cls._RECOVERY_CACHE_RECORD_SUPPORTED_VERSIONS
        ):
            raise ValueError("recovery record has an unsupported version")
        phase = document["phase"]
        if not isinstance(phase, str) or phase not in (
            cls._RECOVERY_CACHE_PHASES
        ):
            raise ValueError("recovery record has an unknown phase")
        checksum = document["checksum"]
        if (
            not isinstance(checksum, str)
            or len(checksum) != 64
            or set(checksum) - cls._HEX_DIGITS
        ):
            raise ValueError("recovery record checksum is malformed")
        body = {
            "format": document["format"],
            "version": version,
            "phase": phase,
            "index": document["index"],
            "progress": document["progress"],
        }
        expected = hashlib.sha256(
            cls._canonical_json(body).encode("utf-8")
        ).hexdigest()
        if not hmac.compare_digest(expected, checksum):
            raise ValueError("recovery record checksum does not match")

        def parse_target(value: Any, label: str) -> dict[str, Any]:
            if not isinstance(value, dict) or set(value) != set(
                cls._RECOVERY_CACHE_TARGET_KEYS
            ):
                raise ValueError(f"recovery record {label} has bad keys")
            old_exists = value["old_exists"]
            new_exists = value["new_exists"]
            if not isinstance(old_exists, bool) or not isinstance(
                new_exists, bool
            ):
                raise ValueError(
                    f"recovery record {label} existence flags must be bools"
                )
            if not new_exists:
                raise ValueError(
                    f"recovery record {label} must describe the published "
                    "new version"
                )
            old_digest = value["old_digest"]
            new_digest = value["new_digest"]
            for name, digest in (
                ("old_digest", old_digest),
                ("new_digest", new_digest),
            ):
                if not isinstance(digest, str):
                    raise ValueError(
                        f"recovery record {label} {name} must be a str"
                    )
                if digest and (
                    len(digest) != 64 or set(digest) - cls._HEX_DIGITS
                ):
                    raise ValueError(
                        f"recovery record {label} {name} is malformed"
                    )
            if old_exists != bool(old_digest):
                raise ValueError(
                    f"recovery record {label} old existence and digest "
                    "disagree"
                )
            backup = value["backup"]
            if not isinstance(backup, str):
                raise ValueError(
                    f"recovery record {label} backup must be a str"
                )
            if old_exists:
                if not backup or backup in (".", "..") or (
                    "/" in backup or "\\" in backup or "\x00" in backup
                ):
                    raise ValueError(
                        f"recovery record {label} backup name is malformed"
                    )
            elif backup:
                raise ValueError(
                    f"recovery record {label} must not name a backup for a "
                    "target that did not previously exist"
                )
            return {
                "old_exists": old_exists,
                "old_digest": old_digest,
                "new_exists": True,
                "new_digest": new_digest,
                "backup": backup,
            }

        return {
            "format": document["format"],
            "version": version,
            "phase": phase,
            "index": parse_target(document["index"], "index"),
            "progress": parse_target(document["progress"], "progress"),
            "checksum": checksum,
        }

    @classmethod
    def _write_cache_recovery_record(
        cls, recovery_path: str, data: bytes
    ) -> None:
        """Durably publish recovery-record bytes: temp file in the
        record's directory, flush, fsync, atomic replace, directory
        syncs. Any failure removes the temp and propagates
        :class:`OSError`; a previous record file is only displaced by
        the atomic replace itself."""
        directory = os.path.dirname(os.path.abspath(recovery_path))
        fd, tmp_path = tempfile.mkstemp(
            prefix=".recovery-cache-record-", suffix=".tmp", dir=directory
        )
        replaced = False
        try:
            try:
                handle = os.fdopen(fd, "wb")
            except BaseException:
                with contextlib.suppress(OSError):
                    os.close(fd)
                with contextlib.suppress(OSError):
                    os.remove(tmp_path)
                raise
            with handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            cls._fsync_directory(directory)
            os.replace(tmp_path, recovery_path)
            replaced = True
            cls._fsync_directory(directory)
        finally:
            if not replaced:
                with contextlib.suppress(OSError):
                    os.remove(tmp_path)

    @classmethod
    def _cache_target_state(
        cls, target: str, described: dict[str, Any]
    ) -> str:
        """Classify a target's on-disk bytes against a recovery record:
        ``"new"`` when it holds the published bytes, ``"old"`` when it
        holds the recorded previous bytes (or is absent as recorded),
        ``"same"`` when the recorded old and new digests coincide so the
        bytes are both versions at once (a publication that changed
        nothing, or a target the publication never replaced), anything
        else -- missing-while-recorded or unknown third bytes -- raises
        :class:`ValueError`, because such a state cannot be completed or
        rolled back safely."""
        exists = os.path.lexists(target)
        if exists:
            digest = cls._sha256_file(target)
            matches_new = hmac.compare_digest(
                digest, described["new_digest"]
            )
            matches_old = described["old_exists"] and hmac.compare_digest(
                digest, described["old_digest"]
            )
            if matches_new and matches_old:
                return "same"
            if matches_new:
                return "new"
            if matches_old:
                return "old"
            raise ValueError(
                "recovery target content matches neither the recorded old "
                "nor new digest"
            )
        if not described["old_exists"]:
            return "old"
        raise ValueError(
            "recovery target is missing but the record says it previously "
            "existed"
        )

    @staticmethod
    def _cache_state_options(state: str) -> tuple[str, ...]:
        """The concrete versions a classified target state can stand
        for: ``"same"`` bytes are the recorded old and the recorded new
        version at once, every other state is exactly one."""
        if state == "same":
            return ("old", "new")
        return (state,)

    @classmethod
    def _reject_mixed_caches_without_record(
        cls, index_path: str, progress_path: str
    ) -> None:
        """Reject a mixed cache version that no valid recovery record
        explains. The caller holds the chain lock and no record exists.

        Every two-cache publication removes its record only after the
        complete new version of both targets is on disk, and an
        index-only publication (no complete segment yet, so no progress
        cursor is written) never leaves the progress cursor ahead of the
        segment index. Without a record the caches are therefore a
        complete version exactly when the cursor is absent, lags the
        index only across a segment-size change, or pins the same frame
        count and head. A cursor without an index, a cursor ahead of the
        index, the two caches pinning different heads at the same frame
        count, or a cursor lagging the index at the same segment size is
        a mixed version the protocol could never have left behind and
        raises :class:`ValueError`. A cache too defective to parse is
        not provably mixed: it is left to the usual single rebuild."""
        index_exists = os.path.lexists(index_path)
        progress_exists = os.path.lexists(progress_path)
        if not progress_exists:
            return
        if not index_exists:
            raise ValueError(
                "recovery cache progress cursor exists without its "
                "segment index and no recovery record explains the "
                "mixed version"
            )
        binding = cls._read_recovery_chain_segments(index_path)
        payload = cls._read_recovery_progress(progress_path)
        if binding is None or payload is None:
            return
        index_size, _version, index_frames, index_head, _fingerprint = (
            binding
        )
        progress_frames = payload["frames"]
        mixed = (
            progress_frames > index_frames
            or (
                progress_frames == index_frames
                and not hmac.compare_digest(payload["head"], index_head)
            )
            or (
                progress_frames < index_frames
                and payload["segment_size"] == index_size
            )
        )
        if mixed:
            raise ValueError(
                "recovery caches hold mixed versions and no recovery "
                "record explains the divergence"
            )

    @classmethod
    def _recover_cache_publication_locked(
        cls,
        index_path: str,
        progress_path: str,
        recovery_path: str,
    ) -> None:
        """Finish an interrupted two-cache publication from a previous
        process. The caller holds the chain lock and invokes this before
        any cache, index or progress is used.

        With no recovery record the caches must form one complete
        version (see :meth:`_reject_mixed_caches_without_record`); a
        mixed version no valid record explains raises
        :class:`ValueError`. Otherwise the record is parsed and
        checksummed strictly and its phase is cross-checked against both
        targets' current bytes: only a state combination the
        publication protocol itself can have left behind is converged.
        When both targets hold the recorded new bytes and the phase
        allows committing (a replacement's record was still due, or
        cleanup was interrupted) the new version is completed and the
        old-version backups are removed. Every other reachable state
        rolls both targets back: a target still holding the new bytes is
        either restored byte-for-byte from its named backup or, when it
        did not exist before, removed; a target already holding the old
        bytes is left alone (an earlier recovery attempt may have
        crashed after restoring it). Backups and the directory are
        synced as each restoration lands, and the record is removed only
        after every restoration, cleanup and directory sync has
        succeeded.

        A phase that contradicts the target digests (both targets new
        while the record says nothing was replaced, or any target back
        on the old version after the final phase was recorded) is
        unreachable and raises :class:`ValueError` -- the targets
        superficially agreeing never justifies deleting the record. An
        unparseable record, bad fields/checksum, target bytes from an
        unknown version or a missing or corrupt required backup also
        raises :class:`ValueError` and leaves every target, backup and
        the record untouched as evidence; a missing parent directory or
        any open, read, write, flush, replace, delete or sync failure
        raises :class:`OSError`, also with all recovery material
        retained.
        """
        if not os.path.lexists(recovery_path):
            cls._reject_mixed_caches_without_record(
                index_path, progress_path
            )
            return
        directory = os.path.dirname(os.path.abspath(recovery_path))
        with open(recovery_path, "rb") as handle:
            raw = handle.read()
        record = cls._parse_cache_recovery_record(raw)

        targets = (
            ("index", index_path, record["index"]),
            ("progress", progress_path, record["progress"]),
        )
        states = {
            label: cls._cache_target_state(target, described)
            for label, target, described in targets
        }
        phase = record["phase"]
        # Committing is only possible once the record shows the index
        # replacement landed (a later replacement or the cleanup may
        # still have been in flight) and both targets hold the recorded
        # new bytes.
        commit_allowed = phase in (
            cls._RECOVERY_CACHE_PHASE_INDEX,
            cls._RECOVERY_CACHE_PHASE_PROGRESS,
        ) and all(
            state in ("new", "same") for state in states.values()
        )
        # The (index, progress) version combinations each phase can
        # have left behind, including the residuals of a recovery
        # attempt that was itself interrupted mid-rollback. The final
        # phase is never rolled back, so it admits no rollback state.
        rollback_states = {
            cls._RECOVERY_CACHE_PHASE_PREPARED: (
                ("old", "old"),
                ("new", "old"),
            ),
            cls._RECOVERY_CACHE_PHASE_INDEX: (
                ("old", "old"),
                ("new", "old"),
                ("old", "new"),
                ("new", "new"),
            ),
        }
        rollback_allowed = phase in rollback_states and any(
            (index_state, progress_state) in rollback_states[phase]
            for index_state in cls._cache_state_options(states["index"])
            for progress_state in cls._cache_state_options(
                states["progress"]
            )
        )
        if not commit_allowed and not rollback_allowed:
            raise ValueError(
                "recovery record phase contradicts the cache target "
                "contents"
            )

        if commit_allowed:
            # Both replacements (and their directory entries) survived:
            # complete the new version by discarding the old bytes. An
            # earlier attempt may already have removed some backups
            # before dying short of the record unlink, so their absence
            # here is expected.
            for _label, _target, described in targets:
                if described["old_exists"]:
                    backup_path = os.path.join(
                        directory, described["backup"]
                    )
                    if os.path.lexists(backup_path):
                        os.remove(backup_path)
            cls._fsync_directory(directory)
            os.remove(recovery_path)
            cls._fsync_directory(directory)
            return

        # Rollback path: verify every backup before any target is
        # restored or any backup unlinked. A target still carrying the
        # new bytes requires its backup to be present and byte-for-byte
        # the recorded old bytes; a target already on the old bytes has
        # no need for its backup, but a surviving backup that is corrupt
        # can never be cleaned up quietly either.
        for label, _target, described in targets:
            if not described["old_exists"]:
                continue
            backup_path = os.path.join(directory, described["backup"])
            if states[label] == "new":
                if not os.path.lexists(backup_path):
                    raise ValueError(
                        "recovery backup needed to restore a target is "
                        "missing"
                    )
                backup_digest = cls._sha256_file(backup_path)
                if not hmac.compare_digest(
                    backup_digest, described["old_digest"]
                ):
                    raise ValueError(
                        "recovery backup content does not match the "
                        "recorded old digest"
                    )
            elif os.path.lexists(backup_path):
                backup_digest = cls._sha256_file(backup_path)
                if not hmac.compare_digest(
                    backup_digest, described["old_digest"]
                ):
                    raise ValueError(
                        "recovery backup content does not match the "
                        "recorded old digest"
                    )

        # Every target still holding the new bytes is restored to its
        # recorded old bytes; the backups were all verified above.
        for _label, target, described in targets:
            if states[_label] != "new":
                continue
            if described["old_exists"]:
                os.replace(
                    os.path.join(directory, described["backup"]), target
                )
            else:
                os.remove(target)
            cls._fsync_directory(directory)
        # Every target is back on the old version now; discard the
        # backups that are still present and, only last, the record.
        for _label, _target, described in targets:
            if described["old_exists"]:
                backup_path = os.path.join(directory, described["backup"])
                if os.path.lexists(backup_path):
                    os.remove(backup_path)
        cls._fsync_directory(directory)
        os.remove(recovery_path)
        cls._fsync_directory(directory)

    @classmethod
    def _diagnostic_target_verdict(
        cls, target: str, described: dict[str, Any] | None
    ) -> tuple[str, str]:
        """Classify one cache target for a read-only diagnostic.

        Both cache files are known to exist (a missing index or
        progress raises :class:`OSError` before verdicts are made).
        With no parseable record the verdict is only
        ``("present", "")``. With a record the verdict mirrors
        :meth:`_cache_target_state` but never raises: ``"new"``/
        ``"old"``/``"same"`` carry the matching version (``"old"``,
        ``"new"`` or ``"both"``), and bytes matching neither recorded
        digest are ``("unknown", "")``.
        """
        if described is None:
            return cls._RECOVERY_DIAGNOSTIC_TARGET_PRESENT, ""
        digest = cls._sha256_file(target)
        matches_new = hmac.compare_digest(
            digest, described["new_digest"]
        )
        matches_old = described["old_exists"] and hmac.compare_digest(
            digest, described["old_digest"]
        )
        if matches_new and matches_old:
            return (
                cls._RECOVERY_DIAGNOSTIC_TARGET_SAME,
                cls._RECOVERY_DIAGNOSTIC_MATCH_BOTH,
            )
        if matches_new:
            return (
                cls._RECOVERY_DIAGNOSTIC_TARGET_NEW,
                cls._RECOVERY_DIAGNOSTIC_MATCH_NEW,
            )
        if matches_old:
            return (
                cls._RECOVERY_DIAGNOSTIC_TARGET_OLD,
                cls._RECOVERY_DIAGNOSTIC_MATCH_OLD,
            )
        return cls._RECOVERY_DIAGNOSTIC_TARGET_UNKNOWN, ""

    @classmethod
    def _diagnostic_backup_status(
        cls, directory: str, described: dict[str, Any] | None
    ) -> dict[str, bool]:
        """Observe one target's recorded backup without touching it:
        ``{"present": ..., "intact": ...}``. With no valid record, or a
        target that did not previously exist, both flags are ``False``;
        otherwise an existing backup's bytes must equal the recorded old
        digest. A present backup that cannot be opened or read raises
        :class:`OSError`."""
        if described is None or not described["old_exists"]:
            return {"present": False, "intact": False}
        backup_path = os.path.join(directory, described["backup"])
        if not os.path.lexists(backup_path):
            return {"present": False, "intact": False}
        intact = hmac.compare_digest(
            cls._sha256_file(backup_path), described["old_digest"]
        )
        return {"present": True, "intact": intact}

    @classmethod
    def _diagnostic_rollback_allowed(
        cls, phase: str, states: dict[str, str]
    ) -> bool:
        """Whether ``phase`` plus the two target verdicts is a
        combination the publication/rollback protocol could have left
        behind that converges by rollback. This mirrors
        :meth:`_recover_cache_publication_locked` exactly, including the
        ``"same"`` bytes standing for both versions at once."""
        rollback_states = {
            cls._RECOVERY_CACHE_PHASE_PREPARED: (
                ("old", "old"),
                ("new", "old"),
            ),
            cls._RECOVERY_CACHE_PHASE_INDEX: (
                ("old", "old"),
                ("new", "old"),
                ("old", "new"),
                ("new", "new"),
            ),
        }
        if phase not in rollback_states:
            return False
        return any(
            (index_state, progress_state) in rollback_states[phase]
            for index_state in cls._cache_state_options(states["index"])
            for progress_state in cls._cache_state_options(
                states["progress"]
            )
        )

    @classmethod
    def _diagnostic_record_evidence(
        cls,
        index_path: str,
        progress_path: str,
        recovery_path: str,
    ) -> tuple[
        str,
        str,
        dict[str, Any] | None,
        dict[str, str],
        dict[str, str],
        dict[str, dict[str, bool]],
        str | None,
    ]:
        """Observe the three paths strictly read-only, returning
        ``(transaction, phase, record, index_verdict, progress_verdict,
        backups, corrupt_reason)``.

        A missing recovery record means no transaction
        (``transaction == ""``); a present but unparseable record's
        transaction is the SHA-256 of its raw bytes and
        ``corrupt_reason`` is ``"record_corrupt"`` -- its material is
        never echoed. A valid record's checksum is the transaction.
        Index/progress absence or any open/read failure raises
        :class:`OSError`; the method never writes or removes anything.
        """
        # The index and progress must be present and readable.
        for target in (index_path, progress_path):
            if not os.path.lexists(target):
                raise FileNotFoundError(
                    f"recovery diagnostic target is missing: {target!r}"
                )
            with open(target, "rb") as handle:
                while handle.read(cls._RECOVERY_CACHE_BACKUP_CHUNK):
                    pass

        record: dict[str, Any] | None = None
        corrupt_reason: str | None = None
        transaction = ""
        phase = ""
        if os.path.lexists(recovery_path):
            with open(recovery_path, "rb") as handle:
                raw = handle.read()
            try:
                record = cls._parse_cache_recovery_record(raw)
            except ValueError:
                transaction = hashlib.sha256(raw).hexdigest()
                corrupt_reason = (
                    cls._RECOVERY_DIAGNOSTIC_REASON_RECORD_CORRUPT
                )

        descriptions: dict[str, dict[str, Any] | None] = {
            "index": None,
            "progress": None,
        }
        if record is not None:
            transaction = record["checksum"]
            phase = record["phase"]
            descriptions["index"] = record["index"]
            descriptions["progress"] = record["progress"]

        targets = (
            ("index", index_path),
            ("progress", progress_path),
        )
        verdicts = {
            label: cls._diagnostic_target_verdict(
                target, descriptions[label]
            )
            for label, target in targets
        }
        states = {label: verdict[0] for label, verdict in verdicts.items()}
        matches = {
            label: verdict[1] for label, verdict in verdicts.items()
        }
        directory = os.path.dirname(os.path.abspath(recovery_path))
        backups = {
            label: cls._diagnostic_backup_status(
                directory, descriptions[label]
            )
            for label, _target in targets
        }
        return (
            transaction,
            phase,
            record,
            states,
            matches,
            backups,
            corrupt_reason,
        )

    @classmethod
    def _diagnostic_decide(
        cls,
        phase: str,
        record: dict[str, Any] | None,
        states: dict[str, str],
        backups: dict[str, dict[str, bool]],
    ) -> tuple[str, str]:
        """Decide ``(disposition, reason)`` from a valid record's
        evidence, projecting the exact rejection causes of
        :meth:`_recover_cache_publication_locked` without performing
        recovery.

        Both cache files are known to exist (their absence raises
        :class:`OSError` before this decision); a target holding bytes
        from an unknown version is ``rejected/target_mismatch``; a
        phase the target digests cannot realize is
        ``rejected/phase_contradiction``; both targets on the recorded
        new bytes with a committable phase is ``pending_commit``
        (backups may already have been removed by an interrupted
        cleanup); every other reachable combination is
        ``pending_rollback`` unless a backup a rollback would need is
        missing or a surviving backup is corrupt.
        """
        if any(
            state == cls._RECOVERY_DIAGNOSTIC_TARGET_UNKNOWN
            for state in states.values()
        ):
            return (
                cls._RECOVERY_DIAGNOSTIC_DISPOSITION_REJECTED,
                cls._RECOVERY_DIAGNOSTIC_REASON_TARGET_MISMATCH,
            )
        commit_allowed = phase in (
            cls._RECOVERY_CACHE_PHASE_INDEX,
            cls._RECOVERY_CACHE_PHASE_PROGRESS,
        ) and all(
            state
            in (
                cls._RECOVERY_DIAGNOSTIC_TARGET_NEW,
                cls._RECOVERY_DIAGNOSTIC_TARGET_SAME,
            )
            for state in states.values()
        )
        rollback_allowed = cls._diagnostic_rollback_allowed(
            phase, states
        )
        if not commit_allowed and not rollback_allowed:
            return (
                cls._RECOVERY_DIAGNOSTIC_DISPOSITION_REJECTED,
                cls._RECOVERY_DIAGNOSTIC_REASON_PHASE_CONTRADICTION,
            )
        if commit_allowed:
            return (
                cls._RECOVERY_DIAGNOSTIC_DISPOSITION_PENDING_COMMIT,
                cls._RECOVERY_DIAGNOSTIC_REASON_PENDING_COMMIT,
            )

        # Rollback territory: verify backups exactly as the recovery
        # machine does before any restoration.
        for label in ("index", "progress"):
            described = record[label]
            if not described["old_exists"]:
                continue
            status = backups[label]
            if states[label] == "new":
                if not status["present"]:
                    return (
                        cls._RECOVERY_DIAGNOSTIC_DISPOSITION_REJECTED,
                        cls._RECOVERY_DIAGNOSTIC_REASON_BACKUP_MISSING,
                    )
                if not status["intact"]:
                    return (
                        cls._RECOVERY_DIAGNOSTIC_DISPOSITION_REJECTED,
                        cls._RECOVERY_DIAGNOSTIC_REASON_BACKUP_CORRUPT,
                    )
            elif status["present"] and not status["intact"]:
                return (
                    cls._RECOVERY_DIAGNOSTIC_DISPOSITION_REJECTED,
                    cls._RECOVERY_DIAGNOSTIC_REASON_BACKUP_CORRUPT,
                )
        return (
            cls._RECOVERY_DIAGNOSTIC_DISPOSITION_PENDING_ROLLBACK,
            cls._RECOVERY_DIAGNOSTIC_REASON_PENDING_ROLLBACK,
        )

    @classmethod
    def recovery_diagnostic(
        cls,
        index_path: Any,
        progress_path: Any,
        recovery_path: Any,
        previous: Any = None,
    ) -> str:
        """Observe, strictly read-only, the cross-process recovery
        evidence for one two-cache publication and return a compact
        checksummed diagnostic string.

        The three paths name the segment index, the progress cursor and
        the recovery record of an existing recovery-enabled publication;
        they are only read -- never created, replaced or deleted, and no
        backup, the canonical chain, any generation directory or
        business state is touched, and no recovery is performed. Each
        path must be a non-empty :class:`str` (a non-``str`` raises
        :class:`TypeError`, an empty string :class:`ValueError`); the
        three paths must name distinct files -- even through aliases or
        hard links, with existing objects identified by device and
        inode and a failed identity probe raising :class:`OSError` --
        and must all live in one directory, otherwise
        :class:`ValueError`. A missing or unreadable index or progress,
        or any open/read failure, raises :class:`OSError`; corrupt
        recovery material is never raised but reported as a
        ``rejected`` diagnostic.

        ``previous`` continues one transaction's diagnostic sequence
        after its recovery record was cleaned up: ``None`` (the
        default) starts a sequence, and a :class:`str` must be exactly a
        diagnostic previously returned by this method -- anything else
        raises :class:`TypeError`, and an empty string, a non-canonical
        document, duplicate keys, a bad version/structure or a failed
        checksum raise :class:`ValueError`.

        With no recovery record and no predecessor the diagnostic is
        ``idle``; with no record but a predecessor it judges by the
        predecessor's summary -- a committed or rolled-back transaction
        stays so across repeated observations, while evidence that
        regresses or changes transaction is ``rejected``. A present
        record that cannot be parsed or authenticated identifies the
        transaction by the SHA-256 of its raw bytes (no material body is
        ever echoed) and is ``rejected/record_corrupt``; a valid
        record's checksum is the transaction identity. Against a valid
        record the two targets are judged against the recorded old/new
        digests and the backups' presence and integrity are observed:
        the disposition is ``pending_commit`` when both targets hold the
        new bytes at a committable phase, ``pending_rollback`` for every
        other combination the protocol can roll back, and ``rejected``
        with a stable reason (``phase_contradiction``,
        ``target_mismatch``, ``backup_missing`` or
        ``backup_corrupt``) otherwise; a durable record that reappears
        after an already disposed transaction is
        ``rejected/reprocessed``.

        The returned string is UTF-8 without a BOM, compact JSON with
        no trailing newline, binding format/version, transaction, the
        predecessor's digest, phase, both targets' verdicts, backup
        status, disposition and reason, sealed by a SHA-256 checksum;
        identical evidence always yields a byte-for-byte identical
        string. Sequences authenticate through
        :meth:`verify_recovery_diagnostics`.
        """
        for label, value in (
            ("index_path", index_path),
            ("progress_path", progress_path),
            ("recovery_path", recovery_path),
        ):
            if not isinstance(value, str):
                raise TypeError(
                    f"{label} must be a str, got {type(value).__name__}"
                )
            if not value:
                raise ValueError(f"{label} must be a non-empty str")
        if previous is not None and not isinstance(previous, str):
            raise TypeError(
                "previous must be None or a str, got "
                f"{type(previous).__name__}"
            )

        real_paths = {
            os.path.realpath(index_path),
            os.path.realpath(progress_path),
            os.path.realpath(recovery_path),
        }
        if len(real_paths) != 3:
            raise ValueError(
                "index_path, progress_path and recovery_path must each "
                "name a distinct file"
            )
        directories = {
            os.path.dirname(os.path.realpath(index_path)),
            os.path.dirname(os.path.realpath(progress_path)),
            os.path.dirname(os.path.realpath(recovery_path)),
        }
        if len(directories) != 1:
            raise ValueError(
                "index_path, progress_path and recovery_path must all "
                "live in the same directory"
            )

        # Distinct resolved path strings are not enough: two of the
        # three objects may be hard links to one underlying file, which
        # must not bypass the identity check. Existing objects are
        # identified by device and inode; an absent object (the
        # recovery record commonly is, which the idle semantics below
        # report) is simply not probed, while any other stat failure
        # propagates as OSError.
        identities: set[tuple[int, int]] = set()
        for probed in (index_path, progress_path, recovery_path):
            try:
                stat_result = os.stat(probed)
            except FileNotFoundError:
                continue
            identity = (stat_result.st_dev, stat_result.st_ino)
            if identity in identities:
                raise ValueError(
                    "index_path, progress_path and recovery_path must "
                    "not name the same file"
                )
            identities.add(identity)

        prev_document: dict[str, Any] | None = None
        prev_digest = cls._RECOVERY_DIAGNOSTIC_GENESIS_PREV
        if previous is not None:
            if not previous:
                raise ValueError("previous must be a non-empty str")
            prev_document = cls._parse_recovery_diagnostic(previous)
            prev_digest = prev_document["checksum"]

        (
            transaction,
            phase,
            record,
            states,
            matches,
            backups,
            corrupt_reason,
        ) = cls._diagnostic_record_evidence(
            index_path, progress_path, recovery_path
        )

        disposition = ""
        reason = ""
        prev_disposition = (
            prev_document["disposition"]
            if prev_document is not None
            else None
        )
        idle = cls._RECOVERY_DIAGNOSTIC_DISPOSITION_IDLE
        committed = cls._RECOVERY_DIAGNOSTIC_DISPOSITION_COMMITTED
        rolled_back = cls._RECOVERY_DIAGNOSTIC_DISPOSITION_ROLLED_BACK
        pending_commit = (
            cls._RECOVERY_DIAGNOSTIC_DISPOSITION_PENDING_COMMIT
        )
        pending_rollback = (
            cls._RECOVERY_DIAGNOSTIC_DISPOSITION_PENDING_ROLLBACK
        )

        if corrupt_reason is not None:
            disposition = cls._RECOVERY_DIAGNOSTIC_DISPOSITION_REJECTED
            reason = corrupt_reason
        elif record is not None:
            # A durable record exists: its own phase, target digests and
            # backups decide the observation.
            disposition, reason = cls._diagnostic_decide(
                phase, record, states, backups
            )
            # ...but a durable record for a transaction the predecessor
            # summary already disposed is the same publication being
            # reprocessed. That contradiction is only reported as
            # ``reprocessed`` when the present evidence itself forms a
            # processable publication; evidence already defective
            # (unknown bytes, an unreachable phase, a bad backup) keeps
            # its own stronger rejection reason.
            if (
                prev_disposition in (committed, rolled_back)
                and hmac.compare_digest(
                    prev_document["transaction"], transaction
                )
                and disposition
                in (pending_commit, pending_rollback)
            ):
                disposition = (
                    cls._RECOVERY_DIAGNOSTIC_DISPOSITION_REJECTED
                )
                reason = cls._RECOVERY_DIAGNOSTIC_REASON_REPROCESSED
        elif prev_document is None or prev_disposition == idle:
            # No record and no open transaction: idle.
            transaction = ""
            phase = ""
            disposition = idle
            reason = cls._RECOVERY_DIAGNOSTIC_REASON_IDLE
        elif prev_disposition in (committed, rolled_back):
            # The record was cleaned up after the transaction was
            # already observed as disposed: carry its identity and
            # judgment forward.
            transaction = prev_document["transaction"]
            phase = ""
            disposition = prev_disposition
            reason = prev_disposition
        elif prev_disposition in (pending_commit, pending_rollback):
            # The protocol removes the record only after commit cleanup
            # or after a complete rollback, so a record vanishing while
            # the transaction was pending settles its outcome: a
            # committable state was finished as committed, every other
            # reachable state as rolled back.
            transaction = prev_document["transaction"]
            phase = ""
            if prev_disposition == pending_commit:
                disposition = committed
                reason = committed
            else:
                disposition = rolled_back
                reason = rolled_back
        else:
            # A rejected transaction's record can never be removed by
            # the protocol, so its disappearance contradicts the
            # predecessor summary.
            transaction = prev_document["transaction"]
            phase = ""
            disposition = cls._RECOVERY_DIAGNOSTIC_DISPOSITION_REJECTED
            reason = (
                cls._RECOVERY_DIAGNOSTIC_REASON_TRANSACTION_CONTRADICTION
            )

        body = {
            "format": cls._RECOVERY_DIAGNOSTIC_FORMAT,
            "version": cls._RECOVERY_DIAGNOSTIC_VERSION,
            "transaction": transaction,
            "prev": prev_digest,
            "phase": phase,
            "index": {
                "state": states["index"],
                "matches": matches["index"],
            },
            "progress": {
                "state": states["progress"],
                "matches": matches["progress"],
            },
            "backups": {
                "index": dict(backups["index"]),
                "progress": dict(backups["progress"]),
            },
            "disposition": disposition,
            "reason": reason,
        }
        document = dict(body)
        document["checksum"] = hashlib.sha256(
            cls._canonical_json(body).encode("utf-8")
        ).hexdigest()
        return cls._canonical_json(document)

    @classmethod
    def _parse_recovery_diagnostic(cls, text: Any) -> dict[str, Any]:
        """Strictly parse and authenticate one diagnostic string,
        returning its normalized mapping. Any encoding, JSON,
        canonical-form, duplicate-key, field, type, version, domain,
        cross-field consistency or checksum defect raises
        :class:`ValueError`. The caller validates that ``text`` is a
        non-empty :class:`str`."""
        if not isinstance(text, str):
            raise TypeError(
                f"diagnostic must be a str, got {type(text).__name__}"
            )
        if text.startswith("\ufeff"):
            raise ValueError("diagnostic must be UTF-8 without a BOM")
        try:
            document = json.loads(
                text,
                object_pairs_hook=cls._reject_duplicate_json_keys,
            )
        except ValueError as exc:
            raise ValueError(f"diagnostic is not valid JSON: {exc}")
        if cls._canonical_json(document) != text:
            raise ValueError("diagnostic is not canonical compact JSON")
        if not isinstance(document, dict) or set(document) != set(
            cls._RECOVERY_DIAGNOSTIC_KEYS
        ):
            raise ValueError("diagnostic has bad top-level keys")
        if document["format"] != cls._RECOVERY_DIAGNOSTIC_FORMAT:
            raise ValueError("diagnostic has an unknown format")
        version = document["version"]
        if (
            isinstance(version, bool)
            or not isinstance(version, int)
            or version not in cls._RECOVERY_DIAGNOSTIC_SUPPORTED_VERSIONS
        ):
            raise ValueError("diagnostic has an unsupported version")

        def is_digest(value: Any, allow_empty: bool = False) -> bool:
            if not isinstance(value, str):
                return False
            if allow_empty and value == "":
                return True
            return len(value) == 64 and not (
                set(value) - cls._HEX_DIGITS
            )

        transaction = document["transaction"]
        if not is_digest(transaction, allow_empty=True):
            raise ValueError("diagnostic transaction is malformed")
        prev = document["prev"]
        if not is_digest(prev):
            raise ValueError("diagnostic prev digest is malformed")
        phase = document["phase"]
        if not isinstance(phase, str) or phase not in (
            cls._RECOVERY_DIAGNOSTIC_PHASES
        ):
            raise ValueError("diagnostic phase is unknown")

        def parse_target(value: Any) -> dict[str, str]:
            if not isinstance(value, dict) or set(value) != set(
                cls._RECOVERY_DIAGNOSTIC_TARGET_KEYS
            ):
                raise ValueError("diagnostic target has bad keys")
            state = value["state"]
            matches = value["matches"]
            if (
                not isinstance(state, str)
                or state not in cls._RECOVERY_DIAGNOSTIC_TARGET_STATES
            ):
                raise ValueError("diagnostic target state is unknown")
            if not isinstance(matches, str):
                raise ValueError("diagnostic target matches must be a str")
            allowed_matches = {
                cls._RECOVERY_DIAGNOSTIC_TARGET_PRESENT: {""},
                cls._RECOVERY_DIAGNOSTIC_TARGET_NEW: {
                    cls._RECOVERY_DIAGNOSTIC_MATCH_NEW
                },
                cls._RECOVERY_DIAGNOSTIC_TARGET_OLD: {
                    cls._RECOVERY_DIAGNOSTIC_MATCH_OLD
                },
                cls._RECOVERY_DIAGNOSTIC_TARGET_SAME: {
                    cls._RECOVERY_DIAGNOSTIC_MATCH_BOTH
                },
                cls._RECOVERY_DIAGNOSTIC_TARGET_UNKNOWN: {""},
            }
            if matches not in allowed_matches[state]:
                raise ValueError(
                    "diagnostic target state and matches disagree"
                )
            return {"state": state, "matches": matches}

        targets = {
            label: parse_target(document[label])
            for label in ("index", "progress")
        }

        backups_value = document["backups"]
        if not isinstance(backups_value, dict) or set(
            backups_value
        ) != set(cls._RECOVERY_DIAGNOSTIC_BACKUPS_KEYS):
            raise ValueError("diagnostic backups have bad keys")

        def parse_backup(value: Any) -> dict[str, bool]:
            if not isinstance(value, dict) or set(value) != set(
                cls._RECOVERY_DIAGNOSTIC_BACKUP_KEYS
            ):
                raise ValueError("diagnostic backup has bad keys")
            present = value["present"]
            intact = value["intact"]
            if not isinstance(present, bool) or not isinstance(
                intact, bool
            ):
                raise ValueError(
                    "diagnostic backup flags must be bools"
                )
            if not present and intact:
                raise ValueError(
                    "diagnostic backup cannot be intact when absent"
                )
            return {"present": present, "intact": intact}

        backups = {
            label: parse_backup(backups_value[label])
            for label in ("index", "progress")
        }

        disposition = document["disposition"]
        reason = document["reason"]
        if (
            not isinstance(disposition, str)
            or disposition not in cls._RECOVERY_DIAGNOSTIC_DISPOSITIONS
        ):
            raise ValueError("diagnostic disposition is unknown")
        if not isinstance(reason, str) or reason not in (
            cls._RECOVERY_DIAGNOSTIC_REASONS
        ):
            raise ValueError("diagnostic reason is unknown")
        checksum = document["checksum"]
        if not is_digest(checksum):
            raise ValueError("diagnostic checksum is malformed")

        idle = cls._RECOVERY_DIAGNOSTIC_DISPOSITION_IDLE
        terminal = {
            cls._RECOVERY_DIAGNOSTIC_DISPOSITION_COMMITTED,
            cls._RECOVERY_DIAGNOSTIC_DISPOSITION_ROLLED_BACK,
        }
        pending = {
            cls._RECOVERY_DIAGNOSTIC_DISPOSITION_PENDING_COMMIT,
            cls._RECOVERY_DIAGNOSTIC_DISPOSITION_PENDING_ROLLBACK,
        }
        rejected = cls._RECOVERY_DIAGNOSTIC_DISPOSITION_REJECTED
        target_states = {
            label: targets[label]["state"] for label in ("index", "progress")
        }
        no_backup = {"present": False, "intact": False}
        if disposition == idle:
            if (
                transaction != ""
                or phase != ""
                or reason != cls._RECOVERY_DIAGNOSTIC_REASON_IDLE
                or any(
                    target["state"]
                    != cls._RECOVERY_DIAGNOSTIC_TARGET_PRESENT
                    or target["matches"] != ""
                    for target in targets.values()
                )
                or any(
                    backup != no_backup for backup in backups.values()
                )
            ):
                raise ValueError("idle diagnostic is inconsistent")
        elif disposition in terminal:
            if (
                transaction == ""
                or phase != ""
                or reason != disposition
                or any(
                    target["state"]
                    != cls._RECOVERY_DIAGNOSTIC_TARGET_PRESENT
                    or target["matches"] != ""
                    for target in targets.values()
                )
                or any(
                    backup != no_backup for backup in backups.values()
                )
            ):
                raise ValueError(
                    "terminal diagnostic is inconsistent"
                )
        elif disposition in pending:
            if (
                transaction == ""
                or phase == ""
                or reason != disposition
                or any(
                    target["state"]
                    not in (
                        cls._RECOVERY_DIAGNOSTIC_TARGET_NEW,
                        cls._RECOVERY_DIAGNOSTIC_TARGET_OLD,
                        cls._RECOVERY_DIAGNOSTIC_TARGET_SAME,
                    )
                    for target in targets.values()
                )
            ):
                raise ValueError("pending diagnostic is inconsistent")
            # A pending observation must name a combination the
            # publication protocol can actually leave behind: a
            # committable one only at a committable phase with both
            # targets on the new bytes, a rollback one only in the
            # rollback-reachable phase/state combinations.
            observed_states = {
                label: targets[label]["state"]
                for label in ("index", "progress")
            }
            committable = phase in (
                cls._RECOVERY_CACHE_PHASE_INDEX,
                cls._RECOVERY_CACHE_PHASE_PROGRESS,
            ) and all(
                state
                in (
                    cls._RECOVERY_DIAGNOSTIC_TARGET_NEW,
                    cls._RECOVERY_DIAGNOSTIC_TARGET_SAME,
                )
                for state in observed_states.values()
            )
            rollbackable = cls._diagnostic_rollback_allowed(
                phase, observed_states
            )
            if disposition == (
                cls._RECOVERY_DIAGNOSTIC_DISPOSITION_PENDING_COMMIT
            ) and not committable:
                raise ValueError(
                    "pending_commit diagnostic is not committable"
                )
            if disposition == (
                cls._RECOVERY_DIAGNOSTIC_DISPOSITION_PENDING_ROLLBACK
            ) and (committable or not rollbackable):
                raise ValueError(
                    "pending_rollback diagnostic is not a reachable "
                    "rollback state"
                )
            # A present backup must be intact in any pending
            # observation, and a target still holding the new bytes in
            # a rollback scenario must have its recorded old bytes
            # backed up -- anything else is a rejection the producer
            # would have reported instead.
            for label in ("index", "progress"):
                status = backups[label]
                if status["present"] and not status["intact"]:
                    raise ValueError(
                        "pending diagnostic names a corrupt backup"
                    )
                if (
                    disposition
                    == cls._RECOVERY_DIAGNOSTIC_DISPOSITION_PENDING_ROLLBACK
                    and observed_states[label]
                    == cls._RECOVERY_DIAGNOSTIC_TARGET_NEW
                    and not status["present"]
                ):
                    raise ValueError(
                        "pending_rollback diagnostic is missing a "
                        "backup required to restore a new target"
                    )
        else:  # rejected
            assert disposition == rejected
            if reason in (
                cls._RECOVERY_DIAGNOSTIC_REASON_IDLE,
                cls._RECOVERY_DIAGNOSTIC_REASON_PENDING_COMMIT,
                cls._RECOVERY_DIAGNOSTIC_REASON_PENDING_ROLLBACK,
                cls._RECOVERY_DIAGNOSTIC_REASON_COMMITTED,
                cls._RECOVERY_DIAGNOSTIC_REASON_ROLLED_BACK,
            ):
                raise ValueError(
                    "rejected diagnostic carries a non-rejection reason"
                )
            neutral_states = {
                cls._RECOVERY_DIAGNOSTIC_TARGET_PRESENT,
            }
            evidence_states = {
                cls._RECOVERY_DIAGNOSTIC_TARGET_NEW,
                cls._RECOVERY_DIAGNOSTIC_TARGET_OLD,
                cls._RECOVERY_DIAGNOSTIC_TARGET_SAME,
                cls._RECOVERY_DIAGNOSTIC_TARGET_UNKNOWN,
            }
            decided_states = {
                cls._RECOVERY_DIAGNOSTIC_TARGET_NEW,
                cls._RECOVERY_DIAGNOSTIC_TARGET_OLD,
                cls._RECOVERY_DIAGNOSTIC_TARGET_SAME,
            }
            if reason == cls._RECOVERY_DIAGNOSTIC_REASON_RECORD_CORRUPT:
                # A corrupt record identifies the transaction by its raw
                # bytes but yields no parseable phase or target/backup
                # evidence.
                if (
                    transaction == ""
                    or phase != ""
                    or not all(
                        state in neutral_states
                        for state in target_states.values()
                    )
                    or any(
                        backup != no_backup for backup in backups.values()
                    )
                ):
                    raise ValueError(
                        "record_corrupt diagnostic is inconsistent"
                    )
            elif reason == (
                cls._RECOVERY_DIAGNOSTIC_REASON_TRANSACTION_CONTRADICTION
            ):
                # The record vanished or changed identity: no parseable
                # record evidence remains, but the predecessor named the
                # contradicted transaction.
                if transaction == "" or phase != "" or not all(
                    state in neutral_states
                    for state in target_states.values()
                ) or any(
                    backup != no_backup for backup in backups.values()
                ):
                    raise ValueError(
                        "transaction_contradiction diagnostic is "
                        "inconsistent"
                    )
            elif reason == cls._RECOVERY_DIAGNOSTIC_REASON_TARGET_MISMATCH:
                if (
                    transaction == ""
                    or phase == ""
                    or not all(
                        state in evidence_states
                        for state in target_states.values()
                    )
                    or not any(
                        state
                        == cls._RECOVERY_DIAGNOSTIC_TARGET_UNKNOWN
                        for state in target_states.values()
                    )
                ):
                    raise ValueError(
                        "target_mismatch diagnostic is inconsistent"
                    )
            elif reason == (
                cls._RECOVERY_DIAGNOSTIC_REASON_REPROCESSED
            ):
                if (
                    transaction == ""
                    or phase == ""
                    or not all(
                        state in decided_states
                        for state in target_states.values()
                    )
                ):
                    raise ValueError(
                        "reprocessed diagnostic is inconsistent"
                    )
            else:
                # phase_contradiction, backup_missing, backup_corrupt:
                # the record parsed and every target held a recorded
                # version.
                if (
                    transaction == ""
                    or phase == ""
                    or not all(
                        state in decided_states
                        for state in target_states.values()
                    )
                ):
                    raise ValueError(
                        f"{reason} diagnostic is inconsistent"
                    )
                if reason == (
                    cls._RECOVERY_DIAGNOSTIC_REASON_BACKUP_CORRUPT
                ) and not any(
                    status["present"] and not status["intact"]
                    for status in backups.values()
                ):
                    raise ValueError(
                        "backup_corrupt diagnostic names no corrupt "
                        "backup"
                    )

        body = {
            key: document[key] for key in cls._RECOVERY_DIAGNOSTIC_KEYS
            if key != "checksum"
        }
        expected = hashlib.sha256(
            cls._canonical_json(body).encode("utf-8")
        ).hexdigest()
        if not hmac.compare_digest(expected, checksum):
            raise ValueError("diagnostic checksum does not match")
        return {
            "format": document["format"],
            "version": version,
            "transaction": transaction,
            "prev": prev,
            "phase": phase,
            "index": targets["index"],
            "progress": targets["progress"],
            "backups": backups,
            "disposition": disposition,
            "reason": reason,
            "checksum": checksum,
        }

    @classmethod
    def verify_recovery_diagnostics(cls, diagnostics: Any) -> bool:
        """Authenticate a consecutive sequence of recovery diagnostics.

        ``diagnostics`` must be a :class:`tuple` of diagnostic strings
        produced by :meth:`recovery_diagnostic`; any other container
        type raises :class:`TypeError`, as does any element that is not
        a :class:`str`. An empty tuple returns ``True``. A string that
        is itself malformed -- non-canonical JSON, duplicate keys, a bad
        version or structure or a checksum that does not verify --
        raises :class:`ValueError`.

        Every record's predecessor digest must equal the sealed digest
        of the record before it (the first record roots at the genesis
        digest); once a transaction is named every later record must
        name that same one -- the idle prefix is the only place the
        identity is empty, so a new publication beginning (or any other
        identity change) mid-chain, including after the transaction was
        committed or rolled back, is a transaction mutation and returns
        ``False``. The observed phase may never move backwards. An idle
        sequence may open a transaction or observe a corrupt record; a
        pending transaction stays pending or settles deterministically
        -- ``pending_commit`` can only become ``committed`` and
        ``pending_rollback`` only ``rolled_back``; either pending state
        may turn into ``rejected``. ``committed`` and ``rolled_back``
        are strictly terminal and only carry themselves forward, so the
        same transaction being reprocessed after disposition (a
        ``reprocessed`` observation) or any other post-disposition
        change makes the sequence unverifiable; ``rejected`` only stays
        rejected. Evidence that regresses, a transaction that changes
        identity, or a durable record reprocessed after disposition
        returns ``False``. Verification is read-only and computes no
        recovery.
        """
        if not isinstance(diagnostics, tuple):
            raise TypeError(
                "diagnostics must be a tuple, got "
                f"{type(diagnostics).__name__}"
            )
        for item in diagnostics:
            if not isinstance(item, str):
                raise TypeError(
                    "every diagnostic must be a str, got "
                    f"{type(item).__name__}"
                )
        if not diagnostics:
            return True

        records = [
            cls._parse_recovery_diagnostic(item) for item in diagnostics
        ]
        if records[0]["prev"] != cls._RECOVERY_DIAGNOSTIC_GENESIS_PREV:
            return False

        idle = cls._RECOVERY_DIAGNOSTIC_DISPOSITION_IDLE
        pending_commit = (
            cls._RECOVERY_DIAGNOSTIC_DISPOSITION_PENDING_COMMIT
        )
        pending_rollback = (
            cls._RECOVERY_DIAGNOSTIC_DISPOSITION_PENDING_ROLLBACK
        )
        committed = cls._RECOVERY_DIAGNOSTIC_DISPOSITION_COMMITTED
        rolled_back = cls._RECOVERY_DIAGNOSTIC_DISPOSITION_ROLLED_BACK
        rejected = cls._RECOVERY_DIAGNOSTIC_DISPOSITION_REJECTED
        # A terminal disposition only arises by continuing from a
        # predecessor, so the first record in a sequence can never
        # already be committed or rolled back.
        if records[0]["disposition"] in (committed, rolled_back):
            return False
        allowed = {
            idle: {idle, pending_commit, pending_rollback, rejected},
            pending_commit: {
                pending_commit,
                committed,
                rejected,
            },
            pending_rollback: {
                pending_rollback,
                rolled_back,
                rejected,
            },
            committed: {committed},
            rolled_back: {rolled_back},
            rejected: {rejected},
        }
        phase_rank = {
            cls._RECOVERY_CACHE_PHASE_PREPARED: 0,
            cls._RECOVERY_CACHE_PHASE_INDEX: 1,
            cls._RECOVERY_CACHE_PHASE_PROGRESS: 2,
        }

        previous = records[0]
        for current in records[1:]:
            if not hmac.compare_digest(
                current["prev"], previous["checksum"]
            ):
                return False
            if previous["transaction"] != "" and not hmac.compare_digest(
                current["transaction"], previous["transaction"]
            ):
                return False
            if (
                current["disposition"]
                not in allowed[previous["disposition"]]
            ):
                return False
            if (
                current["phase"] != ""
                and previous["phase"] != ""
                and phase_rank[current["phase"]]
                < phase_rank[previous["phase"]]
            ):
                return False
            previous = current
        return True

    @classmethod
    def _authenticate_recovery_diagnostic_chains(
        cls, chains: Any
    ) -> list[dict[str, Any]]:
        """Validate a ledger of diagnostic chains and return one fresh
        summary mapping per chain, in input order.

        ``chains`` must be a :class:`tuple` of chains, each chain a
        non-empty :class:`tuple` of diagnostic strings; any other
        container or element type raises :class:`TypeError` and an
        empty chain raises :class:`ValueError`. Every structural check
        completes before any chain is authenticated. Each chain must
        then authenticate through :meth:`verify_recovery_diagnostics`
        -- a malformed record surfaces its own :class:`ValueError` and
        a chain that does not authenticate raises :class:`ValueError`
        -- and must name exactly one transaction identity: an all-idle
        chain that never opens a transaction raises :class:`ValueError`,
        as does a transaction identity shared by two chains. The
        returned summaries are new mappings that share nothing with the
        caller, and nothing is read from or written to the filesystem.
        """
        if not isinstance(chains, tuple):
            raise TypeError(
                "chains must be a tuple, got "
                f"{type(chains).__name__}"
            )
        # Plain structural validation of every chain and record
        # completes before any chain is authenticated.
        for chain in chains:
            if not isinstance(chain, tuple):
                raise TypeError(
                    "every diagnostic chain must be a tuple, got "
                    f"{type(chain).__name__}"
                )
            if not chain:
                raise ValueError("a diagnostic chain must not be empty")
            for record in chain:
                if not isinstance(record, str):
                    raise TypeError(
                        "every diagnostic record must be a str, got "
                        f"{type(record).__name__}"
                    )

        terminal_dispositions = (
            cls._RECOVERY_DIAGNOSTIC_DISPOSITION_COMMITTED,
            cls._RECOVERY_DIAGNOSTIC_DISPOSITION_ROLLED_BACK,
            cls._RECOVERY_DIAGNOSTIC_DISPOSITION_REJECTED,
        )
        summaries: list[dict[str, Any]] = []
        transactions: set[str] = set()
        for chain in chains:
            if not cls.verify_recovery_diagnostics(chain):
                raise ValueError(
                    "diagnostic chain does not authenticate as one "
                    "consecutive sequence"
                )
            documents = [
                cls._parse_recovery_diagnostic(record) for record in chain
            ]
            # A verified chain names at most one transaction identity:
            # the idle prefix is the only place it is empty.
            transaction = ""
            for document in documents:
                if document["transaction"]:
                    transaction = document["transaction"]
                    break
            if not transaction:
                raise ValueError(
                    "diagnostic chain never names a transaction"
                )
            if transaction in transactions:
                raise ValueError(
                    "transaction identity is duplicated across chains"
                )
            transactions.add(transaction)
            last = documents[-1]
            summaries.append(
                {
                    "transaction": transaction,
                    "first": documents[0]["disposition"],
                    "last": last["disposition"],
                    "count": len(chain),
                    "terminal": (
                        last["disposition"] in terminal_dispositions
                    ),
                    "reason": last["reason"],
                }
            )
        return summaries

    @classmethod
    def summarize_recovery_diagnostics(
        cls, chains: Any
    ) -> tuple[dict[str, Any], ...]:
        """Summarize, strictly read-only, a ledger of recovery
        diagnostic chains sampled over time.

        ``chains`` must be a :class:`tuple` of chains in sampling
        order, each chain a non-empty :class:`tuple` of diagnostic
        strings produced by :meth:`recovery_diagnostic`; any other
        container or a non-:class:`str` record raises
        :class:`TypeError`, and an empty inner chain raises
        :class:`ValueError`. An empty outer tuple returns an empty
        tuple. All structural validation completes before any chain is
        authenticated; a chain whose records are malformed, whose
        sequence does not authenticate through
        :meth:`verify_recovery_diagnostics`, that never names a
        transaction, or whose transaction identity duplicates another
        chain's raises :class:`ValueError` -- chains are never merged
        and no evidence is overwritten.

        Returns one new mapping per chain, in input order, with the
        keys ``transaction`` (the chain's transaction identity),
        ``first`` and ``last`` (the first and last records'
        dispositions), ``count`` (the number of records), ``terminal``
        (``True`` only when the last record is a committed, rolled-back
        or rejected terminal observation -- a pending transaction is
        never marked terminal early) and ``reason`` (the last record's
        stable reason, which for a pending transaction stays
        ``pending_commit`` or ``pending_rollback`` without speculating
        an outcome). Every value comes from authenticated records. The
        call modifies nothing: no diagnostic file, cache material,
        event graph, audit or idempotency state is touched, whether it
        succeeds or raises.
        """
        return tuple(
            cls._authenticate_recovery_diagnostic_chains(chains)
        )

    @classmethod
    def get_recovery_diagnostics(
        cls, chains: Any, transaction: Any
    ) -> tuple[dict[str, Any], tuple[str, ...]]:
        """Look up one transaction in a ledger of recovery diagnostic
        chains, strictly read-only.

        The ledger is validated completely first, exactly as
        :meth:`summarize_recovery_diagnostics` validates it: any
        illegal chain raises and no other transaction is returned --
        partial results are never produced. ``transaction`` must be a
        non-empty :class:`str` (a non-``str`` raises :class:`TypeError`,
        an empty string :class:`ValueError`); a transaction identity no
        chain names raises :class:`KeyError`.

        Returns ``(summary, records)``: a fresh copy of that chain's
        summary mapping (as :meth:`summarize_recovery_diagnostics`
        reports it) and the chain's own canonical tuple of diagnostic
        record strings. The returned levels share no mutable state with
        each other or across calls, and the call modifies nothing
        whether it succeeds or raises.
        """
        summaries = cls._authenticate_recovery_diagnostic_chains(chains)
        if not isinstance(transaction, str):
            raise TypeError(
                "transaction must be a str, got "
                f"{type(transaction).__name__}"
            )
        if not transaction:
            raise ValueError("transaction must be a non-empty str")
        for summary, chain in zip(summaries, chains):
            if summary["transaction"] == transaction:
                return (dict(summary), chain)
        raise KeyError(transaction)

    _DIAGNOSTIC_LEDGER_FORMAT = "branching-city-twin/diagnostic-ledger"
    _DIAGNOSTIC_LEDGER_VERSION = 1
    _DIAGNOSTIC_LEDGER_SUPPORTED_VERSIONS = (1,)
    _DIAGNOSTIC_LEDGER_KEYS = ("format", "version", "entries", "checksum")
    _DIAGNOSTIC_LEDGER_ENTRY_KEYS = ("at", "chain")
    _DIAGNOSTIC_LEDGER_CURSOR_FORMAT = (
        "branching-city-twin/diagnostic-ledger-cursor"
    )
    _DIAGNOSTIC_LEDGER_CURSOR_VERSION = 1
    _DIAGNOSTIC_LEDGER_CURSOR_SUPPORTED_VERSIONS = (1,)
    _DIAGNOSTIC_LEDGER_CURSOR_KEYS = (
        "format",
        "version",
        "identity",
        "digest",
        "start",
        "end",
        "transactions",
        "offset",
        "checksum",
    )
    _DIAGNOSTIC_LEDGER_IDENTITY_KEYS = ("device", "inode", "size")

    @classmethod
    def _validate_diagnostic_ledger_path(cls, path: Any) -> None:
        """Validate the ledger ``path`` argument shared by both ledger
        entry points: a non-:class:`str` raises :class:`TypeError` and
        an empty string raises :class:`ValueError`."""
        if not isinstance(path, str):
            raise TypeError(
                f"path must be a str, got {type(path).__name__}"
            )
        if not path:
            raise ValueError("path must be a non-empty str")

    @classmethod
    def _validate_diagnostic_ledger_entries(
        cls, entries: Any
    ) -> list[tuple[int, tuple[str, ...]]]:
        """Validate the ``entries`` argument of
        :meth:`save_diagnostic_ledger` and return it as a list of
        ``(time, chain)`` pairs in input order.

        ``entries`` must be a :class:`tuple` of ``(time, chain)``
        tuples; any other container or entry shape raises
        :class:`TypeError`, as does a non-:class:`int` (or
        :class:`bool`) time or a non-:class:`tuple` chain. A negative
        time or a time that steps backwards raises :class:`ValueError`.
        Chain authentication is the caller's next step, not checked
        here."""
        if not isinstance(entries, tuple):
            raise TypeError(
                "entries must be a tuple, got "
                f"{type(entries).__name__}"
            )
        pairs: list[tuple[int, tuple[str, ...]]] = []
        for entry in entries:
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise TypeError(
                    "every ledger entry must be a (time, chain) tuple"
                )
            at, chain = entry
            if isinstance(at, bool) or not isinstance(at, int):
                raise TypeError(
                    "ledger entry time must be an int, got "
                    f"{type(at).__name__}"
                )
            if at < 0:
                raise ValueError("ledger entry time must be non-negative")
            if not isinstance(chain, tuple):
                raise TypeError(
                    "ledger entry chain must be a tuple, got "
                    f"{type(chain).__name__}"
                )
            pairs.append((at, chain))
        previous_at: int | None = None
        for at, _chain in pairs:
            if previous_at is not None and at < previous_at:
                raise ValueError(
                    "ledger entry times must be non-decreasing"
                )
            previous_at = at
        return pairs

    @classmethod
    def _diagnostic_ledger_document(
        cls, pairs: list[tuple[int, tuple[str, ...]]]
    ) -> bytes:
        """Serialize the ledger document for ``pairs`` to its canonical
        bytes: compact UTF-8 JSON, no BOM, no trailing newline, with a
        SHA-256 checksum over the canonical body. ``pairs`` must
        already be authenticated and sorted."""
        document: dict[str, Any] = {
            "format": cls._DIAGNOSTIC_LEDGER_FORMAT,
            "version": cls._DIAGNOSTIC_LEDGER_VERSION,
            "entries": [
                {"at": at, "chain": list(chain)} for at, chain in pairs
            ],
        }
        body = dict(document)
        document["checksum"] = hashlib.sha256(
            cls._canonical_json(body).encode("utf-8")
        ).hexdigest()
        return cls._canonical_json(document).encode("utf-8")

    @classmethod
    def save_diagnostic_ledger(cls, path: Any, entries: Any) -> None:
        """Seal a snapshot of time-stamped recovery diagnostic chains
        into a persistent ledger at ``path``, replacing any previous
        ledger wholesale.

        ``path`` must be a non-empty :class:`str` (a non-:class:`str`
        raises :class:`TypeError`, an empty string :class:`ValueError`).
        ``entries`` must be a :class:`tuple` of ``(time, chain)``
        tuples -- any other shape or element type raises
        :class:`TypeError`. Each time must be a non-``bool``
        non-negative :class:`int` and the times must be non-decreasing;
        a negative or regressing time raises :class:`ValueError`. Each
        chain is authenticated exactly as
        :meth:`summarize_recovery_diagnostics` authenticates a ledger
        of chains: an empty chain, a malformed record, a sequence that
        does not authenticate, a chain that never names a transaction
        or a transaction identity duplicated across entries raises
        :class:`ValueError`. An empty ``entries`` tuple seals an empty
        ledger.

        The ledger is compact UTF-8 JSON with no BOM and no trailing
        newline, covered by a SHA-256 checksum over its canonical body,
        with entries stored sorted by time and then transaction
        identity. The document is written to a temporary file in the
        same directory, flushed, fsync-ed and atomically moved onto
        ``path``; any open, write, flush or replace failure raises
        :class:`OSError` and leaves a previous ledger byte-for-byte
        untouched.
        """
        cls._validate_diagnostic_ledger_path(path)
        pairs = cls._validate_diagnostic_ledger_entries(entries)
        chains = tuple(chain for _at, chain in pairs)
        summaries = cls._authenticate_recovery_diagnostic_chains(chains)
        order = sorted(
            range(len(pairs)),
            key=lambda index: (
                pairs[index][0],
                summaries[index]["transaction"],
            ),
        )
        sorted_pairs = [pairs[index] for index in order]
        data = cls._diagnostic_ledger_document(sorted_pairs)
        cls._replace_durable(path, data, ".diagnostic-ledger-")

    @classmethod
    def _read_diagnostic_ledger(cls, path: str) -> tuple[bytes, Any]:
        """Read the ledger file's raw bytes together with the
        :func:`os.fstat` result of the descriptor they were read from,
        so a cursor can bind the exact file identity behind one open.
        Missing, unreadable or otherwise failing files raise
        :class:`OSError`."""
        fd = os.open(path, os.O_RDONLY)
        try:
            stat_result = os.fstat(fd)
            handle = os.fdopen(fd, "rb")
        except BaseException:
            with contextlib.suppress(OSError):
                os.close(fd)
            raise
        with handle:
            raw = handle.read()
        return raw, stat_result

    @classmethod
    def _parse_diagnostic_ledger(
        cls, raw: bytes
    ) -> tuple[
        list[tuple[int, tuple[str, ...]]], list[dict[str, Any]], str
    ]:
        """Strictly parse and authenticate the ledger document bytes,
        returning ``(entries, summaries, digest)``: the ``(time,
        chain)`` pairs in canonical order, one fresh summary mapping
        per entry as :meth:`summarize_recovery_diagnostics` reports it,
        and the SHA-256 hex digest of the raw file bytes.

        Any encoding, canonical-form, duplicate-key, structure,
        version or checksum defect raises :class:`ValueError`, as does
        any embedded chain that does not authenticate, an entry time
        that is not a non-``bool`` non-negative :class:`int`, a
        duplicated transaction identity or entries stored out of
        canonical (time, transaction) order. No partial view of a
        defective ledger is ever produced."""
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"ledger is not valid UTF-8: {exc}"
            ) from exc
        if text.startswith("\ufeff"):
            raise ValueError("ledger must be UTF-8 without a BOM")
        if text.endswith("\n") or text.endswith("\r"):
            raise ValueError("ledger must not have a trailing newline")
        try:
            document = json.loads(
                text,
                object_pairs_hook=cls._reject_duplicate_json_keys,
            )
        except ValueError as exc:
            raise ValueError(f"ledger is not valid JSON: {exc}") from exc
        if cls._canonical_json(document) != text:
            raise ValueError("ledger is not canonical compact JSON")
        if not isinstance(document, dict) or set(document) != set(
            cls._DIAGNOSTIC_LEDGER_KEYS
        ):
            raise ValueError("ledger has bad top-level keys")
        if document["format"] != cls._DIAGNOSTIC_LEDGER_FORMAT:
            raise ValueError("ledger has an unknown format")
        version = document["version"]
        if (
            isinstance(version, bool)
            or not isinstance(version, int)
            or version not in cls._DIAGNOSTIC_LEDGER_SUPPORTED_VERSIONS
        ):
            raise ValueError("ledger has an unsupported version")
        checksum = document["checksum"]
        if (
            not isinstance(checksum, str)
            or len(checksum) != 64
            or set(checksum) - cls._HEX_DIGITS
        ):
            raise ValueError("ledger checksum is malformed")
        body = {
            key: value
            for key, value in document.items()
            if key != "checksum"
        }
        if not hmac.compare_digest(
            checksum,
            hashlib.sha256(
                cls._canonical_json(body).encode("utf-8")
            ).hexdigest(),
        ):
            raise ValueError("ledger checksum does not verify")
        raw_entries = document["entries"]
        if not isinstance(raw_entries, list):
            raise ValueError("ledger entries must be a list")
        pairs: list[tuple[int, tuple[str, ...]]] = []
        for raw_entry in raw_entries:
            if not isinstance(raw_entry, dict) or set(
                raw_entry
            ) != set(cls._DIAGNOSTIC_LEDGER_ENTRY_KEYS):
                raise ValueError("ledger entry has bad keys")
            at = raw_entry["at"]
            if (
                isinstance(at, bool)
                or not isinstance(at, int)
                or at < 0
            ):
                raise ValueError("ledger entry time is malformed")
            chain = raw_entry["chain"]
            if not isinstance(chain, list) or not chain:
                raise ValueError(
                    "ledger entry chain must be a non-empty list"
                )
            for record in chain:
                if not isinstance(record, str):
                    raise ValueError(
                        "ledger entry records must be str"
                    )
            pairs.append((at, tuple(chain)))
        chains = tuple(chain for _at, chain in pairs)
        summaries = cls._authenticate_recovery_diagnostic_chains(chains)
        order_keys = [
            (at, summaries[index]["transaction"])
            for index, (at, _chain) in enumerate(pairs)
        ]
        if order_keys != sorted(order_keys):
            raise ValueError(
                "ledger entries are not in canonical order"
            )
        digest = hashlib.sha256(raw).hexdigest()
        return pairs, summaries, digest

    @staticmethod
    def _validate_diagnostic_ledger_bound(
        value: Any, name: str
    ) -> int | None:
        """Validate one inclusive time bound: ``None`` (unbounded) or a
        non-``bool`` non-negative :class:`int`; anything else raises
        :class:`TypeError`."""
        if value is None:
            return None
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
        ):
            raise TypeError(
                f"{name} must be None or a non-negative int, got "
                f"{type(value).__name__}"
            )
        return value

    @staticmethod
    def _validate_diagnostic_ledger_transactions(
        transactions: Any,
    ) -> tuple[str, ...] | None:
        """Validate the transaction filter: ``None`` (match every
        transaction) or a :class:`tuple` of distinct non-empty
        :class:`str` identities -- an empty tuple matches nothing. A
        non-:class:`tuple` container or non-:class:`str` element raises
        :class:`TypeError`; an empty or duplicated identity raises
        :class:`ValueError`. Returns the identities sorted, so equal
        filters compare equal regardless of spelling order."""
        if transactions is None:
            return None
        if not isinstance(transactions, tuple):
            raise TypeError(
                "transactions must be None or a tuple, got "
                f"{type(transactions).__name__}"
            )
        seen: set[str] = set()
        for item in transactions:
            if not isinstance(item, str):
                raise TypeError(
                    "every transaction must be a str, got "
                    f"{type(item).__name__}"
                )
            if not item:
                raise ValueError(
                    "transactions must not contain an empty identity"
                )
            if item in seen:
                raise ValueError(
                    "transactions must not contain duplicates"
                )
            seen.add(item)
        return tuple(sorted(seen))

    @staticmethod
    def _validate_diagnostic_ledger_limit(limit: Any) -> int:
        """Validate the page size: a non-``bool`` :class:`int` of at
        least one; any other type raises :class:`TypeError` and a
        smaller value raises :class:`ValueError`."""
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise TypeError(
                f"limit must be an int, got {type(limit).__name__}"
            )
        if limit < 1:
            raise ValueError("limit must be >= 1")
        return limit

    @classmethod
    def _seal_diagnostic_ledger_cursor(
        cls,
        identity: dict[str, int],
        digest: str,
        start: int | None,
        end: int | None,
        transactions: tuple[str, ...] | None,
        offset: int,
    ) -> str:
        """Serialize a resumption cursor binding the ledger file's
        identity and content digest, the query's filter conditions and
        the next offset, checksummed like a diagnostic record."""
        document: dict[str, Any] = {
            "format": cls._DIAGNOSTIC_LEDGER_CURSOR_FORMAT,
            "version": cls._DIAGNOSTIC_LEDGER_CURSOR_VERSION,
            "identity": identity,
            "digest": digest,
            "start": start,
            "end": end,
            "transactions": (
                None if transactions is None else list(transactions)
            ),
            "offset": offset,
        }
        body = dict(document)
        document["checksum"] = hashlib.sha256(
            cls._canonical_json(body).encode("utf-8")
        ).hexdigest()
        return cls._canonical_json(document)

    @classmethod
    def _parse_diagnostic_ledger_cursor(cls, cursor: str) -> dict[str, Any]:
        """Strictly parse and authenticate a resumption cursor,
        returning its bound file identity, digest, filter conditions
        and offset. An empty string, non-canonical encoding, structural
        defect or checksum mismatch raises :class:`ValueError`. The
        caller validates that ``cursor`` is a :class:`str`."""
        if not cursor:
            raise ValueError("cursor must be a non-empty str")
        if cursor.startswith("\ufeff"):
            raise ValueError("cursor must be UTF-8 without a BOM")
        try:
            document = json.loads(
                cursor,
                object_pairs_hook=cls._reject_duplicate_json_keys,
            )
        except ValueError as exc:
            raise ValueError(f"cursor is not valid JSON: {exc}") from exc
        if cls._canonical_json(document) != cursor:
            raise ValueError("cursor is not canonical compact JSON")
        if not isinstance(document, dict) or set(document) != set(
            cls._DIAGNOSTIC_LEDGER_CURSOR_KEYS
        ):
            raise ValueError("cursor has bad top-level keys")
        if document["format"] != cls._DIAGNOSTIC_LEDGER_CURSOR_FORMAT:
            raise ValueError("cursor has an unknown format")
        version = document["version"]
        if (
            isinstance(version, bool)
            or not isinstance(version, int)
            or version
            not in cls._DIAGNOSTIC_LEDGER_CURSOR_SUPPORTED_VERSIONS
        ):
            raise ValueError("cursor has an unsupported version")
        identity = document["identity"]
        if not isinstance(identity, dict) or set(identity) != set(
            cls._DIAGNOSTIC_LEDGER_IDENTITY_KEYS
        ):
            raise ValueError("cursor identity has bad keys")
        for key in cls._DIAGNOSTIC_LEDGER_IDENTITY_KEYS:
            field = identity[key]
            if (
                isinstance(field, bool)
                or not isinstance(field, int)
                or field < 0
            ):
                raise ValueError("cursor identity is malformed")
        digest = document["digest"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or set(digest) - cls._HEX_DIGITS
        ):
            raise ValueError("cursor digest is malformed")

        def parse_bound(value: Any) -> int | None:
            if value is None:
                return None
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError("cursor time bound is malformed")
            return value

        start = parse_bound(document["start"])
        end = parse_bound(document["end"])
        if start is not None and end is not None and start > end:
            raise ValueError("cursor time bounds are reversed")
        raw_transactions = document["transactions"]
        transactions: tuple[str, ...] | None
        if raw_transactions is None:
            transactions = None
        else:
            if not isinstance(raw_transactions, list):
                raise ValueError("cursor transactions must be a list")
            for item in raw_transactions:
                if not isinstance(item, str) or not item:
                    raise ValueError(
                        "cursor transactions must be non-empty str"
                    )
            if len(set(raw_transactions)) != len(
                raw_transactions
            ) or raw_transactions != sorted(raw_transactions):
                raise ValueError(
                    "cursor transactions are not canonical"
                )
            transactions = tuple(raw_transactions)
        offset = document["offset"]
        if (
            isinstance(offset, bool)
            or not isinstance(offset, int)
            or offset < 0
        ):
            raise ValueError("cursor offset is malformed")
        checksum = document["checksum"]
        if (
            not isinstance(checksum, str)
            or len(checksum) != 64
            or set(checksum) - cls._HEX_DIGITS
        ):
            raise ValueError("cursor checksum is malformed")
        body = {
            key: value
            for key, value in document.items()
            if key != "checksum"
        }
        if not hmac.compare_digest(
            checksum,
            hashlib.sha256(
                cls._canonical_json(body).encode("utf-8")
            ).hexdigest(),
        ):
            raise ValueError("cursor checksum does not verify")
        return {
            "identity": dict(identity),
            "digest": digest,
            "start": start,
            "end": end,
            "transactions": transactions,
            "offset": offset,
        }

    @classmethod
    def page_diagnostic_ledger(
        cls,
        path: Any,
        start: Any,
        end: Any,
        transactions: Any,
        limit: Any,
        cursor: Any,
    ) -> dict[str, Any]:
        """Read one snapshot-consistent page from a sealed diagnostic
        ledger, strictly read-only.

        ``path`` must be a non-empty :class:`str` (a non-:class:`str`
        raises :class:`TypeError`, an empty string :class:`ValueError`);
        a missing, unreadable or unwritable-location file raises
        :class:`OSError`. When ``path`` names a directory ledger
        published by :meth:`append_diagnostic_ledger`, the visible
        manifest is authenticated and only the currently retained
        segments are paged, with the same filters, ordering and return
        structure; the cursor then binds the manifest's identity and
        content digest, so an append, rotation or truncation between
        pages raises :class:`RuntimeError` on resume, and a concurrent
        page only ever observes the complete snapshot from before or
        after a publish, never a mixture. A directory without a
        manifest raises :class:`OSError`. ``start`` and ``end`` are
        inclusive time
        bounds: each may be ``None`` (unbounded) or a non-``bool``
        non-negative :class:`int`, anything else raises
        :class:`TypeError`, and ``start`` after ``end`` raises
        :class:`ValueError`. ``transactions`` may be ``None`` (match
        every transaction) or a :class:`tuple` of distinct non-empty
        :class:`str` identities -- an empty tuple matches nothing; a
        bad container or element type raises :class:`TypeError`, an
        empty or duplicated identity :class:`ValueError`. Identities no
        entry names are simply never matched, never an error.
        ``limit`` must be a non-``bool`` positive :class:`int`
        (:class:`TypeError` otherwise, :class:`ValueError` when below
        one). ``cursor`` must be ``None`` (first page) or a cursor
        string from a previous page: any other type raises
        :class:`TypeError`; an empty, non-canonical or
        checksum-invalid string raises :class:`ValueError`, as does a
        cursor whose bound filter conditions differ from this call's.

        A cursor binds the ledger file's identity and content digest,
        so paging can resume across restarts; if the ledger was
        replaced, truncated, appended to or otherwise modified since
        the cursor was issued, the resume raises :class:`RuntimeError`
        rather than mixing versions. A ledger whose encoding, checksum,
        structure or embedded chain authentication is illegal raises
        :class:`ValueError` and no partial page is returned.

        Returns a new mapping with ``snapshot``, ``items`` and
        ``next_cursor`` in that order. ``snapshot`` identifies the
        ledger snapshot read (its content digest, entry count and the
        byte count of the authenticated manifest under ``bytes``) and is
        returned even when nothing matches. ``items`` is a
        tuple of entries ordered by time and then transaction identity,
        each a fresh mapping with the entry's time (``at``), its
        ``summary`` as :meth:`summarize_recovery_diagnostics` reports
        it and its ``records`` -- the chain's canonical tuple of
        diagnostic strings. ``next_cursor`` is the resumption cursor
        for the following page, or ``None`` on the last page. The
        returned levels share no mutable state with each other or
        across calls, and the call modifies nothing.
        """
        cls._validate_diagnostic_ledger_path(path)
        start_value = cls._validate_diagnostic_ledger_bound(
            start, "start"
        )
        end_value = cls._validate_diagnostic_ledger_bound(end, "end")
        if (
            start_value is not None
            and end_value is not None
            and start_value > end_value
        ):
            raise ValueError("start must not be after end")
        filter_transactions = (
            cls._validate_diagnostic_ledger_transactions(transactions)
        )
        page_limit = cls._validate_diagnostic_ledger_limit(limit)
        if cursor is not None and not isinstance(cursor, str):
            raise TypeError(
                "cursor must be None or a str, got "
                f"{type(cursor).__name__}"
            )
        bookmark = (
            None
            if cursor is None
            else cls._parse_diagnostic_ledger_cursor(cursor)
        )
        if os.path.isdir(path):
            return cls._page_diagnostic_ledger_directory(
                path,
                start_value,
                end_value,
                filter_transactions,
                page_limit,
                bookmark,
            )
        raw, stat_result = cls._read_diagnostic_ledger(path)
        identity = {
            "device": stat_result.st_dev,
            "inode": stat_result.st_ino,
            "size": stat_result.st_size,
        }
        digest = hashlib.sha256(raw).hexdigest()
        if bookmark is not None:
            cls._check_diagnostic_ledger_bookmark(
                bookmark,
                identity,
                digest,
                start_value,
                end_value,
                filter_transactions,
            )
        pairs, summaries, _file_digest = cls._parse_diagnostic_ledger(raw)
        return cls._diagnostic_ledger_page_result(
            pairs,
            summaries,
            identity,
            digest,
            stat_result.st_size,
            start_value,
            end_value,
            filter_transactions,
            page_limit,
            bookmark,
        )

    @classmethod
    def _check_diagnostic_ledger_bookmark(
        cls,
        bookmark: dict[str, Any],
        identity: dict[str, int],
        digest: str,
        start_value: int | None,
        end_value: int | None,
        filter_transactions: tuple[str, ...] | None,
    ) -> None:
        """Enforce a resumption cursor's snapshot and filter binding:
        a changed ledger raises :class:`RuntimeError` and a filter
        mismatch raises :class:`ValueError`."""
        if (
            bookmark["identity"] != identity
            or bookmark["digest"] != digest
        ):
            raise RuntimeError(
                "the ledger changed since the cursor was issued; "
                "restart the query instead of mixing versions"
            )
        if (
            bookmark["start"] != start_value
            or bookmark["end"] != end_value
            or bookmark["transactions"] != filter_transactions
        ):
            raise ValueError(
                "cursor does not match the requested filters"
            )

    @classmethod
    def _diagnostic_ledger_page_result(
        cls,
        pairs: list[tuple[int, tuple[str, ...]]],
        summaries: list[dict[str, Any]],
        identity: dict[str, int],
        digest: str,
        size: int,
        start_value: int | None,
        end_value: int | None,
        filter_transactions: tuple[str, ...] | None,
        page_limit: int,
        bookmark: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Assemble one page of the matched entries shared by the
        single-file and directory ledger readers. ``pairs`` must
        already be in canonical (time, transaction) order; every
        returned level is fresh."""
        wanted = (
            None
            if filter_transactions is None
            else set(filter_transactions)
        )
        matched = [
            (at, chain, summaries[index])
            for index, (at, chain) in enumerate(pairs)
            if (start_value is None or at >= start_value)
            and (end_value is None or at <= end_value)
            and (
                wanted is None
                or summaries[index]["transaction"] in wanted
            )
        ]
        offset = 0 if bookmark is None else bookmark["offset"]
        page = matched[offset : offset + page_limit]
        remaining = len(matched) - offset - len(page)
        next_cursor = None
        if remaining > 0:
            next_cursor = cls._seal_diagnostic_ledger_cursor(
                identity,
                digest,
                start_value,
                end_value,
                filter_transactions,
                offset + len(page),
            )
        items = tuple(
            {
                "at": at,
                "summary": dict(summary),
                "records": chain,
            }
            for at, chain, summary in page
        )
        snapshot = {
            "digest": digest,
            "entries": len(pairs),
            "bytes": size,
        }
        return {
            "snapshot": snapshot,
            "items": items,
            "next_cursor": next_cursor,
        }

    _DIAGNOSTIC_LEDGER_MANIFEST_FORMAT = (
        "branching-city-twin/diagnostic-ledger-manifest"
    )
    _DIAGNOSTIC_LEDGER_MANIFEST_VERSION = 1
    _DIAGNOSTIC_LEDGER_MANIFEST_SUPPORTED_VERSIONS = (1,)
    _DIAGNOSTIC_LEDGER_MANIFEST_KEYS = (
        "format",
        "version",
        "segments",
        "dropped",
        "evidence",
        "checksum",
    )
    _DIAGNOSTIC_LEDGER_MANIFEST_SEGMENT_KEYS = (
        "name",
        "digest",
        "entries",
        "first_at",
        "last_at",
    )
    _DIAGNOSTIC_LEDGER_DROPPED_KEYS = (
        "digest",
        "entries",
        "first_at",
        "last_at",
    )
    _DIAGNOSTIC_LEDGER_EVIDENCE_KEYS = ("digest",)
    #: Durable, authenticated evidence of every transaction identity
    #: ever rotated out of the retained segments. The evidence document
    #: binds the identity set to the removed prefix's last segment
    #: digest, its total entry count and time bounds, and authenticates
    #: itself with a SHA-256 checksum, so the prefix stays exactly
    #: remembered across restarts even after its segment files are gone.
    _DIAGNOSTIC_LEDGER_PREFIX_EVIDENCE_FORMAT = (
        "branching-city-twin/diagnostic-ledger-prefix-evidence"
    )
    _DIAGNOSTIC_LEDGER_PREFIX_EVIDENCE_VERSION = 1
    _DIAGNOSTIC_LEDGER_PREFIX_EVIDENCE_SUPPORTED_VERSIONS = (1,)
    _DIAGNOSTIC_LEDGER_PREFIX_EVIDENCE_KEYS = (
        "format",
        "version",
        "digest",
        "entries",
        "first_at",
        "last_at",
        "transactions",
        "checksum",
    )
    _DIAGNOSTIC_LEDGER_EVIDENCE_PREFIX = "dropped-evidence-"
    _DIAGNOSTIC_LEDGER_SEGMENT_FORMAT = (
        "branching-city-twin/diagnostic-ledger-segment"
    )
    _DIAGNOSTIC_LEDGER_SEGMENT_VERSION = 1
    _DIAGNOSTIC_LEDGER_SEGMENT_SUPPORTED_VERSIONS = (1,)
    _DIAGNOSTIC_LEDGER_SEGMENT_KEYS = (
        "format",
        "version",
        "index",
        "prev",
        "entries",
        "checksum",
    )
    _DIAGNOSTIC_LEDGER_MANIFEST_NAME = "manifest.json"
    _DIAGNOSTIC_LEDGER_LOCK_NAME = ".diagnostic-ledger.lock"
    _DIAGNOSTIC_LEDGER_GENESIS_PREV = "0" * 64
    #: How often a first-page directory read restarts after racing a
    #: publish before it reports the snapshot as unobservable.
    _DIAGNOSTIC_LEDGER_READ_ATTEMPTS = 3

    @classmethod
    def _validate_diagnostic_ledger_expected(cls, expected: Any) -> Any:
        """Validate the ``expected`` snapshot of
        :meth:`append_diagnostic_ledger`: ``None`` (only for the first
        publish) or a 64-character lowercase hex snapshot digest. Any
        other type raises :class:`TypeError`; a malformed string raises
        :class:`ValueError`."""
        if expected is None:
            return None
        if not isinstance(expected, str):
            raise TypeError(
                "expected must be None or a str, got "
                f"{type(expected).__name__}"
            )
        if len(expected) != 64 or set(expected) - cls._HEX_DIGITS:
            raise ValueError(
                "expected must be None or a snapshot digest of 64 "
                "lowercase hex characters"
            )
        return expected

    @staticmethod
    def _validate_diagnostic_ledger_positive(value: Any, name: str) -> int:
        """Validate a segment-limit or retention-count argument: a
        non-``bool`` :class:`int` of at least one; any other type raises
        :class:`TypeError` and a smaller value raises
        :class:`ValueError`."""
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                f"{name} must be an int, got {type(value).__name__}"
            )
        if value < 1:
            raise ValueError(f"{name} must be >= 1")
        return value

    @staticmethod
    def _diagnostic_ledger_segment_name(index: int) -> str:
        """Canonical file name of the segment with one-based ``index``."""
        return f"segment-{index:06d}.json"

    @staticmethod
    def _diagnostic_ledger_segment_index(name: Any) -> int | None:
        """The index encoded in a canonical segment file ``name``, or
        ``None`` when the name is malformed."""
        if not isinstance(name, str):
            return None
        if not name.startswith("segment-") or not name.endswith(".json"):
            return None
        digits = name[len("segment-") : -len(".json")]
        if len(digits) != 6 or not digits.isdigit():
            return None
        return int(digits)

    @classmethod
    def _diagnostic_ledger_evidence_name(cls, digest: str) -> str:
        """Content-addressed file name of the prefix evidence whose
        content has SHA-256 hex ``digest``. The full digest is used, so
        the name binds the exact content: a fresh rotation publishes a
        distinct name, the new evidence is in place before the manifest
        switch and the previous evidence only becomes garbage once that
        switch is durable."""
        return f"{cls._DIAGNOSTIC_LEDGER_EVIDENCE_PREFIX}{digest}.json"

    @classmethod
    def _diagnostic_ledger_evidence_files(cls, directory: str) -> list[str]:
        """The on-disk prefix-evidence file names present in
        ``directory`` (content-addressed, possibly including a stale
        generation awaiting garbage collection). A directory read
        failure raises :class:`OSError`."""
        names: list[str] = []
        prefix = cls._DIAGNOSTIC_LEDGER_EVIDENCE_PREFIX
        suffix = ".json"
        for name in os.listdir(directory):
            if (
                name.startswith(prefix)
                and name.endswith(suffix)
                and not (
                    set(name[len(prefix) : -len(suffix)]) - cls._HEX_DIGITS
                )
            ):
                names.append(name)
        return sorted(names)

    @classmethod
    def _require_diagnostic_ledger_digest(cls, value: Any, what: str) -> str:
        """Require a 64-character lowercase hex digest field inside a
        manifest or segment; anything else raises :class:`ValueError`."""
        if (
            not isinstance(value, str)
            or len(value) != 64
            or set(value) - cls._HEX_DIGITS
        ):
            raise ValueError(f"{what} is malformed")
        return value

    @classmethod
    def _require_diagnostic_ledger_time(cls, value: Any, what: str) -> int:
        """Require a non-``bool`` non-negative :class:`int` time field
        inside a manifest; anything else raises :class:`ValueError`."""
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{what} is malformed")
        return value

    @classmethod
    def _diagnostic_ledger_segment_document(
        cls,
        index: int,
        prev: str,
        pairs: list[tuple[int, tuple[str, ...]]],
    ) -> bytes:
        """Serialize one immutable segment to its canonical bytes:
        compact UTF-8 JSON, no BOM, no trailing newline, with a SHA-256
        checksum over the canonical body. ``pairs`` must already be
        authenticated and sorted."""
        document: dict[str, Any] = {
            "format": cls._DIAGNOSTIC_LEDGER_SEGMENT_FORMAT,
            "version": cls._DIAGNOSTIC_LEDGER_SEGMENT_VERSION,
            "index": index,
            "prev": prev,
            "entries": [
                {"at": at, "chain": list(chain)} for at, chain in pairs
            ],
        }
        body = dict(document)
        document["checksum"] = hashlib.sha256(
            cls._canonical_json(body).encode("utf-8")
        ).hexdigest()
        return cls._canonical_json(document).encode("utf-8")

    @classmethod
    def _diagnostic_ledger_manifest_document(
        cls,
        segments: list[dict[str, Any]],
        dropped: dict[str, Any] | None,
        evidence_digest: str | None,
    ) -> bytes:
        """Serialize the directory manifest to its canonical bytes,
        checksummed like a segment. ``segments`` carries one metadata
        mapping per retained segment in chain order; ``dropped`` is the
        continuity proof for the rotated-away prefix, or ``None``;
        ``evidence_digest`` is the content digest of the durable
        transaction-identity evidence covering that prefix (present
        exactly when ``dropped`` is), or ``None``."""
        document: dict[str, Any] = {
            "format": cls._DIAGNOSTIC_LEDGER_MANIFEST_FORMAT,
            "version": cls._DIAGNOSTIC_LEDGER_MANIFEST_VERSION,
            "segments": [
                {
                    key: meta[key]
                    for key in cls._DIAGNOSTIC_LEDGER_MANIFEST_SEGMENT_KEYS
                }
                for meta in segments
            ],
            "dropped": (
                None
                if dropped is None
                else {
                    key: dropped[key]
                    for key in cls._DIAGNOSTIC_LEDGER_DROPPED_KEYS
                }
            ),
            "evidence": (
                None
                if evidence_digest is None
                else {"digest": evidence_digest}
            ),
        }
        body = dict(document)
        document["checksum"] = hashlib.sha256(
            cls._canonical_json(body).encode("utf-8")
        ).hexdigest()
        return cls._canonical_json(document).encode("utf-8")

    @classmethod
    def _diagnostic_ledger_evidence_document(
        cls,
        identities: list[str],
        dropped: dict[str, Any],
    ) -> bytes:
        """Serialize the durable uniqueness evidence for the whole
        rotated-away prefix to canonical bytes. The evidence binds the
        exact set of rotated-away transaction identities to the
        prefix's last segment digest, its total entry count and its
        time bounds (the same facts the manifest's ``dropped`` proof
        carries), and authenticates itself with a SHA-256 checksum.
        ``identities`` must already be de-duplicated in prefix order;
        ``dropped`` describes the complete prefix covered."""
        document: dict[str, Any] = {
            "format": cls._DIAGNOSTIC_LEDGER_PREFIX_EVIDENCE_FORMAT,
            "version": cls._DIAGNOSTIC_LEDGER_PREFIX_EVIDENCE_VERSION,
            "digest": dropped["digest"],
            "entries": dropped["entries"],
            "first_at": dropped["first_at"],
            "last_at": dropped["last_at"],
            "transactions": identities,
        }
        body = dict(document)
        document["checksum"] = hashlib.sha256(
            cls._canonical_json(body).encode("utf-8")
        ).hexdigest()
        return cls._canonical_json(document).encode("utf-8")

    @classmethod
    def _parse_diagnostic_ledger_evidence(
        cls, raw: bytes
    ) -> dict[str, Any]:
        """Strictly parse and authenticate prefix-uniqueness evidence
        bytes, returning a fresh mapping with the covered prefix's last
        segment digest (``digest``), total entry count, time bounds and
        the exact tuple of remembered transaction identities.

        Every identity must be a 64-character lowercase hex digest and
        the list must hold no duplicates: identity judgement is exact,
        never probabilistic. Any encoding, canonical-form, structure,
        version, field, duplicate-identity or checksum defect raises
        :class:`ValueError`; no partial view is produced."""
        document = cls._parse_diagnostic_ledger_envelope(
            raw,
            cls._DIAGNOSTIC_LEDGER_PREFIX_EVIDENCE_KEYS,
            cls._DIAGNOSTIC_LEDGER_PREFIX_EVIDENCE_FORMAT,
            cls._DIAGNOSTIC_LEDGER_PREFIX_EVIDENCE_SUPPORTED_VERSIONS,
            "ledger prefix evidence",
        )
        digest = cls._require_diagnostic_ledger_digest(
            document["digest"], "ledger prefix evidence prefix digest"
        )
        entries = document["entries"]
        if (
            isinstance(entries, bool)
            or not isinstance(entries, int)
            or entries < 1
        ):
            raise ValueError(
                "ledger prefix evidence entry count is malformed"
            )
        first_at = cls._require_diagnostic_ledger_time(
            document["first_at"], "ledger prefix evidence time"
        )
        last_at = cls._require_diagnostic_ledger_time(
            document["last_at"], "ledger prefix evidence time"
        )
        if first_at > last_at:
            raise ValueError(
                "ledger prefix evidence time bounds are reversed"
            )
        raw_identities = document["transactions"]
        if not isinstance(raw_identities, list) or not raw_identities:
            raise ValueError(
                "ledger prefix evidence transactions must be a "
                "non-empty list"
            )
        seen: set[str] = set()
        identities: list[str] = []
        for identity in raw_identities:
            cls._require_diagnostic_ledger_digest(
                identity, "ledger prefix evidence transaction identity"
            )
            if identity in seen:
                raise ValueError(
                    "ledger prefix evidence repeats a transaction "
                    "identity"
                )
            seen.add(identity)
            identities.append(identity)
        if len(identities) != entries:
            raise ValueError(
                "ledger prefix evidence entry count does not match "
                "its transaction identities"
            )
        return {
            "digest": digest,
            "entries": entries,
            "first_at": first_at,
            "last_at": last_at,
            "transactions": tuple(identities),
        }

    @classmethod
    def _parse_diagnostic_ledger_envelope(
        cls,
        raw: bytes,
        keys: tuple[str, ...],
        format_value: str,
        supported_versions: tuple[int, ...],
        what: str,
    ) -> dict[str, Any]:
        """Strictly parse the shared envelope of a manifest or segment
        document: UTF-8 without BOM, no trailing newline, canonical
        compact JSON without duplicate keys, exact top-level keys, known
        format, supported version and a verifying SHA-256 checksum. Any
        defect raises :class:`ValueError`."""
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"{what} is not valid UTF-8: {exc}"
            ) from exc
        if text.startswith("\ufeff"):
            raise ValueError(f"{what} must be UTF-8 without a BOM")
        if text.endswith("\n") or text.endswith("\r"):
            raise ValueError(f"{what} must not have a trailing newline")
        try:
            document = json.loads(
                text,
                object_pairs_hook=cls._reject_duplicate_json_keys,
            )
        except ValueError as exc:
            raise ValueError(f"{what} is not valid JSON: {exc}") from exc
        if cls._canonical_json(document) != text:
            raise ValueError(f"{what} is not canonical compact JSON")
        if not isinstance(document, dict) or set(document) != set(keys):
            raise ValueError(f"{what} has bad top-level keys")
        if document["format"] != format_value:
            raise ValueError(f"{what} has an unknown format")
        version = document["version"]
        if (
            isinstance(version, bool)
            or not isinstance(version, int)
            or version not in supported_versions
        ):
            raise ValueError(f"{what} has an unsupported version")
        checksum = cls._require_diagnostic_ledger_digest(
            document["checksum"], f"{what} checksum"
        )
        body = {
            key: value
            for key, value in document.items()
            if key != "checksum"
        }
        if not hmac.compare_digest(
            checksum,
            hashlib.sha256(
                cls._canonical_json(body).encode("utf-8")
            ).hexdigest(),
        ):
            raise ValueError(f"{what} checksum does not verify")
        return document

    @classmethod
    def _parse_diagnostic_ledger_manifest(
        cls, raw: bytes
    ) -> tuple[list[dict[str, Any]], dict[str, Any] | None, str]:
        """Strictly parse and authenticate manifest bytes, returning
        ``(segments, dropped, digest)``: one fresh metadata mapping per
        retained segment in chain order, the rotated-prefix continuity
        proof (or ``None``) and the SHA-256 hex digest of the raw
        bytes. Any encoding, structure, checksum or field defect raises
        :class:`ValueError`."""
        document = cls._parse_diagnostic_ledger_envelope(
            raw,
            cls._DIAGNOSTIC_LEDGER_MANIFEST_KEYS,
            cls._DIAGNOSTIC_LEDGER_MANIFEST_FORMAT,
            cls._DIAGNOSTIC_LEDGER_MANIFEST_SUPPORTED_VERSIONS,
            "ledger manifest",
        )
        raw_segments = document["segments"]
        if not isinstance(raw_segments, list):
            raise ValueError("ledger manifest segments must be a list")
        segments: list[dict[str, Any]] = []
        previous_index: int | None = None
        for raw_meta in raw_segments:
            if not isinstance(raw_meta, dict) or set(raw_meta) != set(
                cls._DIAGNOSTIC_LEDGER_MANIFEST_SEGMENT_KEYS
            ):
                raise ValueError("ledger manifest segment has bad keys")
            name = raw_meta["name"]
            index = cls._diagnostic_ledger_segment_index(name)
            if index is None or index < 1:
                raise ValueError(
                    "ledger manifest segment name is malformed"
                )
            if previous_index is not None and index != previous_index + 1:
                raise ValueError(
                    "ledger manifest segments are not contiguous"
                )
            digest = cls._require_diagnostic_ledger_digest(
                raw_meta["digest"], "ledger manifest segment digest"
            )
            entries = raw_meta["entries"]
            if (
                isinstance(entries, bool)
                or not isinstance(entries, int)
                or entries < 1
            ):
                raise ValueError(
                    "ledger manifest segment entry count is malformed"
                )
            first_at = cls._require_diagnostic_ledger_time(
                raw_meta["first_at"], "ledger manifest segment time"
            )
            last_at = cls._require_diagnostic_ledger_time(
                raw_meta["last_at"], "ledger manifest segment time"
            )
            if first_at > last_at:
                raise ValueError(
                    "ledger manifest segment time bounds are reversed"
                )
            segments.append(
                {
                    "name": name,
                    "index": index,
                    "digest": digest,
                    "entries": entries,
                    "first_at": first_at,
                    "last_at": last_at,
                }
            )
            previous_index = index
        dropped = cls._parse_diagnostic_ledger_dropped(
            document["dropped"]
        )
        raw_evidence = document["evidence"]
        if raw_evidence is None:
            evidence = None
        else:
            if not isinstance(raw_evidence, dict) or set(
                raw_evidence
            ) != set(cls._DIAGNOSTIC_LEDGER_EVIDENCE_KEYS):
                raise ValueError("ledger manifest evidence has bad keys")
            evidence = {
                "digest": cls._require_diagnostic_ledger_digest(
                    raw_evidence["digest"],
                    "ledger manifest evidence digest",
                )
            }
        if (dropped is None) != (evidence is None):
            raise ValueError(
                "ledger manifest dropped proof and prefix evidence "
                "must agree"
            )
        digest = hashlib.sha256(raw).hexdigest()
        return segments, dropped, evidence, digest

    @classmethod
    def _parse_diagnostic_ledger_dropped(
        cls, raw_dropped: Any
    ) -> dict[str, Any] | None:
        """Validate the manifest's rotated-prefix continuity proof:
        ``None`` when nothing was rotated away, otherwise a mapping
        with the removed prefix's last segment digest, total entry
        count and time bounds. Any defect raises :class:`ValueError`."""
        if raw_dropped is None:
            return None
        if not isinstance(raw_dropped, dict) or set(raw_dropped) != set(
            cls._DIAGNOSTIC_LEDGER_DROPPED_KEYS
        ):
            raise ValueError("ledger manifest dropped proof has bad keys")
        digest = cls._require_diagnostic_ledger_digest(
            raw_dropped["digest"], "ledger manifest dropped digest"
        )
        entries = raw_dropped["entries"]
        if (
            isinstance(entries, bool)
            or not isinstance(entries, int)
            or entries < 1
        ):
            raise ValueError(
                "ledger manifest dropped entry count is malformed"
            )
        first_at = cls._require_diagnostic_ledger_time(
            raw_dropped["first_at"], "ledger manifest dropped time"
        )
        last_at = cls._require_diagnostic_ledger_time(
            raw_dropped["last_at"], "ledger manifest dropped time"
        )
        if first_at > last_at:
            raise ValueError(
                "ledger manifest dropped time bounds are reversed"
            )
        return {
            "digest": digest,
            "entries": entries,
            "first_at": first_at,
            "last_at": last_at,
        }

    @classmethod
    def _parse_diagnostic_ledger_segment(
        cls, raw: bytes
    ) -> tuple[int, str, list[tuple[int, tuple[str, ...]]], str]:
        """Strictly parse and authenticate segment bytes, returning
        ``(index, prev, entries, digest)``: the segment's one-based
        index, its predecessor link, its ``(time, chain)`` pairs in
        stored order and the SHA-256 hex digest of the raw bytes. Any
        encoding, structure, checksum or entry-shape defect raises
        :class:`ValueError`; chain authentication is the caller's next
        step."""
        document = cls._parse_diagnostic_ledger_envelope(
            raw,
            cls._DIAGNOSTIC_LEDGER_SEGMENT_KEYS,
            cls._DIAGNOSTIC_LEDGER_SEGMENT_FORMAT,
            cls._DIAGNOSTIC_LEDGER_SEGMENT_SUPPORTED_VERSIONS,
            "ledger segment",
        )
        index = document["index"]
        if isinstance(index, bool) or not isinstance(index, int) or index < 1:
            raise ValueError("ledger segment index is malformed")
        prev = cls._require_diagnostic_ledger_digest(
            document["prev"], "ledger segment predecessor link"
        )
        raw_entries = document["entries"]
        if not isinstance(raw_entries, list) or not raw_entries:
            raise ValueError(
                "ledger segment entries must be a non-empty list"
            )
        pairs: list[tuple[int, tuple[str, ...]]] = []
        for raw_entry in raw_entries:
            if not isinstance(raw_entry, dict) or set(raw_entry) != set(
                cls._DIAGNOSTIC_LEDGER_ENTRY_KEYS
            ):
                raise ValueError("ledger segment entry has bad keys")
            at = raw_entry["at"]
            if (
                isinstance(at, bool)
                or not isinstance(at, int)
                or at < 0
            ):
                raise ValueError("ledger segment entry time is malformed")
            chain = raw_entry["chain"]
            if not isinstance(chain, list) or not chain:
                raise ValueError(
                    "ledger segment entry chain must be a non-empty list"
                )
            for record in chain:
                if not isinstance(record, str):
                    raise ValueError(
                        "ledger segment entry records must be str"
                    )
            pairs.append((at, tuple(chain)))
        digest = hashlib.sha256(raw).hexdigest()
        return index, prev, pairs, digest

    @classmethod
    def _load_diagnostic_ledger_directory(
        cls, directory: str, manifest_raw: bytes
    ) -> dict[str, Any]:
        """Authenticate a directory ledger snapshot: parse the manifest
        bytes, then read, parse and authenticate every retained segment,
        the durable rotated-prefix uniqueness evidence and verify the
        manifest's digests, the inter-segment digest chain, the
        rotated-prefix continuity link, the evidence's identity set and
        prefix summary, per-segment canonical order, non-decreasing times
        across segment boundaries and ledger-wide chain authentication
        (including transaction uniqueness).

        Returns a mapping with ``segments`` (one mapping per retained
        segment: the manifest metadata plus its ``pairs`` and its
        ``_transactions`` in stored order), ``dropped``, ``evidence``
        (the parsed prefix evidence, or ``None``), ``evidence_name``
        (its file name, or ``None``), ``digest`` (the manifest's content
        digest), ``pairs`` and ``summaries`` (all entries in segment
        order with one summary each). A missing or unreadable segment
        or evidence file raises :class:`OSError`; any content defect
        raises :class:`ValueError` and no partial view is produced."""
        segments, dropped, evidence_ref, manifest_digest = (
            cls._parse_diagnostic_ledger_manifest(manifest_raw)
        )
        previous_digest = (
            cls._DIAGNOSTIC_LEDGER_GENESIS_PREV
            if dropped is None
            else dropped["digest"]
        )
        loaded: list[dict[str, Any]] = []
        all_pairs: list[tuple[int, tuple[str, ...]]] = []
        for meta in segments:
            segment_raw, _stat = cls._read_diagnostic_ledger(
                os.path.join(directory, meta["name"])
            )
            index, prev, pairs, digest = (
                cls._parse_diagnostic_ledger_segment(segment_raw)
            )
            if not hmac.compare_digest(digest, meta["digest"]):
                raise ValueError(
                    "ledger segment digest does not match the manifest"
                )
            if index != meta["index"]:
                raise ValueError(
                    "ledger segment index does not match the manifest"
                )
            if not hmac.compare_digest(prev, previous_digest):
                raise ValueError("ledger segment digest chain is broken")
            if len(pairs) != meta["entries"]:
                raise ValueError(
                    "ledger segment entry count does not match the "
                    "manifest"
                )
            if (
                pairs[0][0] != meta["first_at"]
                or pairs[-1][0] != meta["last_at"]
            ):
                raise ValueError(
                    "ledger segment time bounds do not match the "
                    "manifest"
                )
            previous_digest = digest
            loaded.append({**meta, "pairs": pairs})
            all_pairs.extend(pairs)
        chains = tuple(chain for _at, chain in all_pairs)
        summaries = cls._authenticate_recovery_diagnostic_chains(chains)
        offset = 0
        previous_at: int | None = None
        for segment in loaded:
            keys = [
                (at, summaries[offset + position]["transaction"])
                for position, (at, _chain) in enumerate(segment["pairs"])
            ]
            if keys != sorted(keys):
                raise ValueError(
                    "ledger segment entries are not in canonical order"
                )
            if (
                previous_at is not None
                and segment["pairs"][0][0] < previous_at
            ):
                raise ValueError(
                    "ledger segments are not in time order"
                )
            segment["_transactions"] = tuple(
                summaries[offset + position]["transaction"]
                for position in range(len(segment["pairs"]))
            )
            previous_at = segment["pairs"][-1][0]
            offset += len(segment["pairs"])
        evidence_name: str | None = None
        evidence: dict[str, Any] | None = None
        if evidence_ref is not None:
            evidence_name = cls._diagnostic_ledger_evidence_name(
                evidence_ref["digest"]
            )
            evidence_raw, _stat = cls._read_diagnostic_ledger(
                os.path.join(directory, evidence_name)
            )
            if not hmac.compare_digest(
                hashlib.sha256(evidence_raw).hexdigest(),
                evidence_ref["digest"],
            ):
                raise ValueError(
                    "ledger prefix evidence digest does not match the "
                    "manifest"
                )
            evidence = cls._parse_diagnostic_ledger_evidence(evidence_raw)
            evidence["self_digest"] = hashlib.sha256(
                evidence_raw
            ).hexdigest()
            assert dropped is not None
            if (
                evidence["digest"] != dropped["digest"]
                or evidence["entries"] != dropped["entries"]
                or evidence["first_at"] != dropped["first_at"]
                or evidence["last_at"] != dropped["last_at"]
            ):
                raise ValueError(
                    "ledger prefix evidence does not match the "
                    "dropped-prefix proof"
                )
            retained_transactions = {
                identity
                for segment in loaded
                for identity in segment["_transactions"]
            }
            remembered = set(evidence["transactions"])
            if retained_transactions & remembered:
                raise ValueError(
                    "ledger prefix evidence repeats a transaction "
                    "identity still retained"
                )
        return {
            "segments": loaded,
            "dropped": dropped,
            "evidence": evidence,
            "evidence_name": evidence_name,
            "digest": manifest_digest,
            "pairs": all_pairs,
            "summaries": summaries,
        }

    @classmethod
    def _write_diagnostic_ledger_file(
        cls, path: str, data: bytes, prefix: str
    ) -> None:
        """Write ``data`` to a temp file beside ``path``, flush and
        fsync it, sync the directory entry and atomically replace
        ``path``, then sync the directory again; on any failure the
        previous file is left byte-for-byte untouched and the temporary
        file is removed."""
        directory = os.path.dirname(os.path.abspath(path))
        fd, tmp_path = tempfile.mkstemp(
            prefix=prefix, suffix=".tmp", dir=directory
        )
        replaced = False
        try:
            try:
                handle = os.fdopen(fd, "wb")
            except BaseException:
                # fdopen only reaches here without taking ownership of fd.
                with contextlib.suppress(OSError):
                    os.close(fd)
                raise
            with handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            cls._fsync_directory(directory)
            os.replace(tmp_path, path)
            replaced = True
            cls._fsync_directory(directory)
        finally:
            if not replaced:
                with contextlib.suppress(OSError):
                    os.remove(tmp_path)

    @classmethod
    def append_diagnostic_ledger(
        cls,
        directory: Any,
        entries: Any,
        expected: Any,
        segment_limit: Any,
        keep: Any,
        timeout: Any,
    ) -> str:
        """Append time-stamped recovery diagnostic chains to a
        segmented directory ledger, durably and atomically publishing a
        new visible snapshot.

        ``directory`` must be a non-empty :class:`str` naming an
        existing directory (a non-:class:`str` raises
        :class:`TypeError`, an empty string :class:`ValueError`, a
        missing directory :class:`OSError`). ``entries`` follows the
        sealing rules of :meth:`save_diagnostic_ledger`: a
        :class:`tuple` of ``(time, chain)`` tuples -- any other shape
        or element type raises :class:`TypeError` -- with non-``bool``
        non-negative non-decreasing :class:`int` times and chains that
        authenticate exactly as
        :meth:`summarize_recovery_diagnostics` requires; a negative or
        regressing time, an empty chain, a malformed record, a sequence
        that does not authenticate, a chain that never names a
        transaction or a transaction identity duplicated within
        ``entries`` raises :class:`ValueError`. Uniqueness is one exact
        judgement spanning the retained segments, a durable,
        authenticated record of every identity already rotated away and
        the new batch: after the oldest segments leave retention their
        transaction identities are still precisely remembered (never
        probabilistically), so appending one of them again raises
        :class:`ValueError` rather than being accepted because the
        segment file is gone. Additionally, the first appended time must
        not be earlier than the ledger's current last entry.

        ``expected`` is the optimistic-concurrency token: ``None`` for
        the first publish (only accepted when no manifest exists yet),
        otherwise the current snapshot digest as returned by an earlier
        :meth:`append_diagnostic_ledger` call or reported by
        :meth:`page_diagnostic_ledger` as ``snapshot["digest"]``. Any
        other type raises :class:`TypeError`, a malformed string raises
        :class:`ValueError`, and a mismatch -- including ``None`` once
        the ledger exists -- raises :class:`RuntimeError`. ``segment_limit``
        and ``keep`` must be non-``bool`` :class:`int` values of at
        least one (any other type raises :class:`TypeError`, a smaller
        value :class:`ValueError`): each segment holds at most
        ``segment_limit`` entries and at most ``keep`` segments are
        retained. ``timeout`` bounds only the wait for the publish
        lock: ``None`` waits indefinitely, a non-``bool``
        :class:`int`/:class:`float` number of seconds bounds the wait
        (any other type raises :class:`TypeError`; a negative value,
        NaN or either infinity raises :class:`ValueError`) and an
        elapsed wait raises :class:`TimeoutError`.

        Publishes are serialized across processes by an exclusive
        operating-system file lock on a fixed coordination file inside
        the canonical (symlink- and alias-resolved) ledger directory,
        so equivalent spellings of the same directory compete for the
        same lock. The manifest, every retained segment, the
        rotated-prefix uniqueness evidence and the expected snapshot are
        re-authenticated inside the lock, never against state observed
        before it was taken, so concurrent callers holding the same old
        snapshot see at most one publish succeed: every other caller
        acquires the lock afterwards and raises :class:`RuntimeError`
        without writing a segment, manifest or evidence file. A corrupt
        manifest, segment, prefix evidence or digest chain raises
        :class:`ValueError`; lock, read, write, flush, replace or sync
        failures raise :class:`OSError`.

        The new entries are stored sorted by time and then transaction
        identity in new immutable segments of at most ``segment_limit``
        entries each, every segment linked to its predecessor's digest
        so the segments form a hash chain. When more than ``keep``
        segments would be retained, the oldest are rotated away; the
        manifest records the removed prefix's last segment digest,
        total entry count and time bounds, and a separate durable,
        checksummed evidence file binds the exact identities of every
        transaction in the whole removed prefix, so the ledger's
        continuity and transaction uniqueness stay provable and
        enforceable after the old segment files are gone. The new
        segments, the new evidence and the new manifest are durably
        written (flushed, fsync-ed, directory-synced) before the
        visible manifest is atomically replaced; a failure before that
        switch raises and leaves the old snapshot -- manifest, retained
        segments and evidence -- byte-for-byte readable with no partial
        result visible. The manifest switch, the removal of the
        rotated-away segment files, the reclamation of the superseded
        evidence file and the final directory sync are one rollback-able
        commit: the exact bytes of every doomed file are captured first,
        and a failure of the replacement, a removal or any directory sync
        restores the pre-publish state byte-for-byte -- the old
        manifest and every segment and evidence it references stay readable
        and re-authenticate -- leaves no newly staged segment,
        evidence, temporary or backup file behind, and raises
        :class:`OSError`; a failure encountered during that rollback
        itself raises :class:`OSError`, never masked as success by the
        original failure. Rotated-away segment files (and a superseded
        evidence file) are removed only after the new manifest is
        durable and only once the exclusive publish lock can be held,
        i.e. once every in-progress directory page's shared lock has
        drained; a reader that exits abruptly releases its shared lock
        at the operating-system level, so its occupancy is reclaimed
        automatically. Returns the new snapshot digest, which the next
        append passes as ``expected``.
        """
        cls._validate_diagnostic_ledger_path(directory)
        pairs = cls._validate_diagnostic_ledger_entries(entries)
        expected_value = cls._validate_diagnostic_ledger_expected(expected)
        segment_limit_value = cls._validate_diagnostic_ledger_positive(
            segment_limit, "segment_limit"
        )
        keep_value = cls._validate_diagnostic_ledger_positive(keep, "keep")
        cls._validate_chain_timeout(timeout)
        new_chains = tuple(chain for _at, chain in pairs)
        new_summaries = cls._authenticate_recovery_diagnostic_chains(
            new_chains
        )

        real_directory = os.path.realpath(directory)
        lock_path = os.path.join(
            real_directory, cls._DIAGNOSTIC_LEDGER_LOCK_NAME
        )
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            cls._acquire_chain_lock(fd, timeout)
            try:
                return cls._append_diagnostic_ledger_locked(
                    real_directory,
                    pairs,
                    new_summaries,
                    expected_value,
                    segment_limit_value,
                    keep_value,
                )
            finally:
                cls._release_chain_lock(fd)
        finally:
            os.close(fd)

    @classmethod
    def _append_diagnostic_ledger_locked(
        cls,
        directory: str,
        pairs: list[tuple[int, tuple[str, ...]]],
        new_summaries: list[dict[str, Any]],
        expected: str | None,
        segment_limit: int,
        keep: int,
    ) -> str:
        """Publish one appended snapshot under the directory lock and
        return its digest. The caller holds the lock and has validated
        every argument; the manifest, every retained segment, the
        rotated-prefix uniqueness evidence and the expected snapshot are
        (re-)authenticated here, inside the lock."""
        manifest_path = os.path.join(
            directory, cls._DIAGNOSTIC_LEDGER_MANIFEST_NAME
        )
        try:
            manifest_raw, _stat = cls._read_diagnostic_ledger(
                manifest_path
            )
        except FileNotFoundError:
            manifest_raw = None
        if manifest_raw is None:
            if expected is not None:
                raise RuntimeError(
                    "no diagnostic ledger exists yet, so the expected "
                    "snapshot must be None"
                )
            state: dict[str, Any] = {
                "segments": [],
                "dropped": None,
                "evidence": None,
                "evidence_name": None,
                "pairs": [],
                "summaries": [],
            }
        else:
            state = cls._load_diagnostic_ledger_directory(
                directory, manifest_raw
            )
            if expected is None:
                raise RuntimeError(
                    "the diagnostic ledger already exists, so the "
                    "current snapshot is required"
                )
            if not hmac.compare_digest(state["digest"], expected):
                raise RuntimeError(
                    "expected snapshot does not match the current "
                    "diagnostic ledger snapshot"
                )
        if state["pairs"] and pairs:
            last_at = state["pairs"][-1][0]
            if pairs[0][0] < last_at:
                raise ValueError(
                    "appended entry times must not be earlier than the "
                    "ledger's last entry"
                )
        # One exact uniqueness judgement spans the retained segments,
        # the durable evidence of every already rotated-away prefix and
        # the new batch: a transaction that left the retained segments
        # is still remembered and is rejected precisely.
        existing_transactions = {
            summary["transaction"] for summary in state["summaries"]
        }
        if state["evidence"] is not None:
            existing_transactions.update(state["evidence"]["transactions"])
        for summary in new_summaries:
            if summary["transaction"] in existing_transactions:
                raise ValueError(
                    "transaction identity is duplicated across the "
                    "ledger"
                )
        order = sorted(
            range(len(pairs)),
            key=lambda index: (
                pairs[index][0],
                new_summaries[index]["transaction"],
            ),
        )
        sorted_pairs = [pairs[index] for index in order]
        sorted_summaries = [new_summaries[index] for index in order]
        retained = list(state["segments"])
        next_index = retained[-1]["index"] + 1 if retained else 1
        if retained:
            previous_digest = retained[-1]["digest"]
        elif state["dropped"] is not None:
            previous_digest = state["dropped"]["digest"]
        else:
            previous_digest = cls._DIAGNOSTIC_LEDGER_GENESIS_PREV
        # Every file published before the atomic manifest switch is
        # unreferenced by the still-visible old manifest. If anything
        # fails before that switch, these are rolled back so the
        # directory holds no half-publish residue; only names that did
        # not exist when the publish began are removed.
        preexisting = set(os.listdir(directory))
        created: list[str] = []
        evidence_name: str | None = None
        manifest_evidence_digest: str | None
        try:
            position = 0
            for start in range(0, len(sorted_pairs), segment_limit):
                chunk = sorted_pairs[start : start + segment_limit]
                chunk_summaries = sorted_summaries[
                    position : position + len(chunk)
                ]
                data = cls._diagnostic_ledger_segment_document(
                    next_index, previous_digest, chunk
                )
                name = cls._diagnostic_ledger_segment_name(next_index)
                segment_path = os.path.join(directory, name)
                cls._write_diagnostic_ledger_file(
                    segment_path,
                    data,
                    ".diagnostic-ledger-segment-",
                )
                created.append(segment_path)
                previous_digest = hashlib.sha256(data).hexdigest()
                retained.append(
                    {
                        "name": name,
                        "index": next_index,
                        "digest": previous_digest,
                        "entries": len(chunk),
                        "first_at": chunk[0][0],
                        "last_at": chunk[-1][0],
                        "_transactions": tuple(
                            summary["transaction"]
                            for summary in chunk_summaries
                        ),
                    }
                )
                next_index += 1
                position += len(chunk)
            dropped = state["dropped"]
            removed: list[dict[str, Any]] = []
            excess = len(retained) - keep
            if excess > 0:
                removed = retained[:excess]
                retained = retained[excess:]
                dropped = {
                    "digest": removed[-1]["digest"],
                    "entries": (dropped["entries"] if dropped else 0)
                    + sum(meta["entries"] for meta in removed),
                    "first_at": (
                        dropped["first_at"]
                        if dropped
                        else removed[0]["first_at"]
                    ),
                    "last_at": removed[-1]["last_at"],
                }
            # The prefix evidence is rewritten (under a fresh,
            # content-addressed name) before the manifest switch,
            # carrying forward every previously remembered identity and
            # adding the identities of the segments removed here.
            if removed:
                identities: list[str] = []
                if state["evidence"] is not None:
                    identities.extend(state["evidence"]["transactions"])
                for meta in removed:
                    identities.extend(meta["_transactions"])
                evidence_data = cls._diagnostic_ledger_evidence_document(
                    identities, dropped
                )
                evidence_digest = hashlib.sha256(
                    evidence_data
                ).hexdigest()
                evidence_name = cls._diagnostic_ledger_evidence_name(
                    evidence_digest
                )
                evidence_path = os.path.join(directory, evidence_name)
                cls._write_diagnostic_ledger_file(
                    evidence_path,
                    evidence_data,
                    ".diagnostic-ledger-evidence-",
                )
                created.append(evidence_path)
                manifest_evidence_digest = evidence_digest
            else:
                # No new rotation: keep pointing at the existing
                # evidence, if any.
                manifest_evidence_digest = (
                    None
                    if state["evidence"] is None
                    else state["evidence"]["self_digest"]
                )
            manifest_data = cls._diagnostic_ledger_manifest_document(
                retained, dropped, manifest_evidence_digest
            )
        except BaseException as staging_error:
            # The visible manifest still names only the old snapshot, so
            # deleting the newly staged files (never the old ones, which
            # remain in ``preexisting``) restores the directory to its
            # pre-publish state. A cleanup failure is itself an OSError:
            # it must never hide behind the original failure as success.
            cleanup_errors: list[OSError] = []
            for staged in created:
                if os.path.basename(staged) not in preexisting:
                    try:
                        os.remove(staged)
                    except OSError as exc:
                        cleanup_errors.append(exc)
            try:
                cls._fsync_directory(directory)
            except OSError as exc:
                cleanup_errors.append(exc)
            if cleanup_errors:
                rollback_failure = OSError(
                    "a diagnostic ledger publish failed and its staging "
                    "cleanup failed as well"
                )
                raise rollback_failure from staging_error
            raise
        # The manifest switch, the rotated-segment reclamation, the
        # superseded/orphan-evidence reclamation and the final directory
        # sync are one rollback-able commit: any failure restores the
        # pre-publish snapshot byte-for-byte.
        keep_evidence_name = (
            None
            if manifest_evidence_digest is None
            else cls._diagnostic_ledger_evidence_name(
                manifest_evidence_digest
            )
        )
        cls._commit_diagnostic_ledger_publish(
            directory,
            manifest_path,
            manifest_data,
            manifest_raw,
            [meta["name"] for meta in removed],
            keep_evidence_name,
            created,
            preexisting,
        )
        return hashlib.sha256(manifest_data).hexdigest()

    @classmethod
    def _commit_diagnostic_ledger_publish(
        cls,
        directory: str,
        manifest_path: str,
        manifest_data: bytes,
        old_manifest_raw: bytes | None,
        removed_names: list[str],
        keep_evidence_name: str | None,
        staged: list[str],
        preexisting: set[str],
    ) -> None:
        """Switch the visible manifest and reclaim every segment and
        evidence file it supersedes as a single rollback-able commit.

        The doomed files are identified (a directory listing included),
        their exact bytes captured, the new manifest durably published,
        the old files removed and the directory synced, all inside one
        guarded sequence. Any failure rolls back to the pre-publish state
        (old manifest, segments and evidence byte-for-byte, every newly
        staged file removed) before the original exception is re-raised; a
        failure during that rollback itself raises :class:`OSError`. The
        caller holds the exclusive publish lock."""
        backups: dict[str, bytes] = {}
        try:
            victims = [
                os.path.join(directory, name) for name in removed_names
            ]
            # Evidence reclamation follows the same gate as the manifest's
            # rotation bookkeeping: only a publish that actually rotates
            # reconciles the evidence directory. This removes the
            # superseded evidence and any orphan left by a publish that
            # crashed after writing its evidence but before the manifest
            # switch. The caller's exclusive lock waits for every shared
            # reader lock to drain, so no in-progress reader is still
            # resolving these names; a reader that already opened a file
            # keeps its unlinked inode readable by the kernel.
            if removed_names:
                for name in cls._diagnostic_ledger_evidence_files(directory):
                    if name != keep_evidence_name:
                        victims.append(os.path.join(directory, name))
            for victim in victims:
                raw, _stat = cls._read_diagnostic_ledger(victim)
                backups[victim] = raw
            cls._write_diagnostic_ledger_file(
                manifest_path,
                manifest_data,
                ".diagnostic-ledger-manifest-",
            )
            for victim in victims:
                os.remove(victim)
            cls._fsync_directory(directory)
        except BaseException as original:
            cls._rollback_diagnostic_ledger_publish(
                directory,
                manifest_path,
                old_manifest_raw,
                backups,
                staged,
                preexisting,
                original,
            )
            raise

    @classmethod
    def _rollback_diagnostic_ledger_publish(
        cls,
        directory: str,
        manifest_path: str,
        old_manifest_raw: bytes | None,
        backups: dict[str, bytes],
        staged: list[str],
        preexisting: set[str],
        original: BaseException,
    ) -> None:
        """Restore the pre-publish snapshot after a failed publish
        commit: the old manifest bytes, every backed-up segment or
        evidence file and the removal of every newly staged file, with a
        final directory sync. Each step is attempted even after an
        earlier one fails. When anything here fails, an
        :class:`OSError` chained to the original failure is raised
        rather than letting the rollback fault stay hidden; when the
        rollback succeeds, control returns and the caller re-raises the
        original exception."""
        errors: list[OSError] = []

        def attempt(operation: Any) -> None:
            try:
                operation()
            except OSError as exc:
                errors.append(exc)

        # Re-establish the old visible view first.
        if old_manifest_raw is not None:
            attempt(
                lambda: cls._write_diagnostic_ledger_file(
                    manifest_path,
                    old_manifest_raw,
                    ".diagnostic-ledger-manifest-rollback-",
                )
            )
        elif os.path.exists(manifest_path):
            attempt(lambda: os.remove(manifest_path))
        for victim, raw in backups.items():
            attempt(
                lambda victim=victim, raw=raw: (
                    cls._write_diagnostic_ledger_file(
                        victim,
                        raw,
                        ".diagnostic-ledger-rollback-",
                    )
                )
            )
        for path in staged:
            if os.path.basename(path) not in preexisting:
                if os.path.exists(path):
                    attempt(lambda path=path: os.remove(path))
        attempt(lambda: cls._fsync_directory(directory))
        if errors:
            raise OSError(
                "a diagnostic ledger publish failed and its rollback "
                "failed as well; the ledger directory may need recovery"
            ) from original

    @classmethod
    def _page_diagnostic_ledger_directory(
        cls,
        path: str,
        start_value: int | None,
        end_value: int | None,
        filter_transactions: tuple[str, ...] | None,
        page_limit: int,
        bookmark: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Read one snapshot-consistent page from a directory ledger.

        The whole read runs under a *shared* lock on the ledger's
        coordination file, which excludes an exclusive publisher lock
        for the entire publish (manifest switch and rotated-segment
        garbage collection). A page therefore observes only the complete
        snapshot from before or after a switch -- never a mixture -- and
        a segment an in-progress reader may still open is never deleted.
        The shared lock is released when the call returns, and the
        operating system also releases it if the reader exits abruptly,
        so crashed readers leave no occupancy to block later publishers.
        As a second line of defence the manifest is re-read after the
        segments and compared, so a publish that somehow raced the lock
        is detected and the read restarted (a first page) or reported (a
        resumed page) rather than mixing two versions."""
        real_path = os.path.realpath(path)
        lock_path = os.path.join(
            real_path, cls._DIAGNOSTIC_LEDGER_LOCK_NAME
        )
        # Read-only, and deliberately without O_CREAT: a directory
        # ledger always carries its coordination file (the first
        # append creates it before the manifest), so a page never has
        # to write, and a directory without one -- hence without a
        # manifest -- raises OSError having created nothing.
        lock_fd = os.open(lock_path, os.O_RDONLY)
        try:
            cls._acquire_diagnostic_ledger_reader_lock(lock_fd)
            try:
                return cls._page_diagnostic_ledger_directory_locked(
                    real_path,
                    start_value,
                    end_value,
                    filter_transactions,
                    page_limit,
                    bookmark,
                )
            finally:
                cls._release_diagnostic_ledger_reader_lock(lock_fd)
        finally:
            os.close(lock_fd)

    @classmethod
    def _page_diagnostic_ledger_directory_locked(
        cls,
        path: str,
        start_value: int | None,
        end_value: int | None,
        filter_transactions: tuple[str, ...] | None,
        page_limit: int,
        bookmark: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Read one directory page under the shared reader lock. See
        :meth:`_page_diagnostic_ledger_directory` for the locking
        contract; the caller holds the shared lock and ``path`` is the
        canonical ledger directory."""
        manifest_path = os.path.join(
            path, cls._DIAGNOSTIC_LEDGER_MANIFEST_NAME
        )
        attempts = 0
        while True:
            attempts += 1
            raw, stat_result = cls._read_diagnostic_ledger(manifest_path)
            identity = {
                "device": stat_result.st_dev,
                "inode": stat_result.st_ino,
                "size": stat_result.st_size,
            }
            digest = hashlib.sha256(raw).hexdigest()
            if bookmark is not None:
                cls._check_diagnostic_ledger_bookmark(
                    bookmark,
                    identity,
                    digest,
                    start_value,
                    end_value,
                    filter_transactions,
                )
            state = cls._load_diagnostic_ledger_directory(path, raw)
            confirm_raw, confirm_stat = cls._read_diagnostic_ledger(
                manifest_path
            )
            confirm_identity = {
                "device": confirm_stat.st_dev,
                "inode": confirm_stat.st_ino,
                "size": confirm_stat.st_size,
            }
            if (
                confirm_identity == identity
                and hmac.compare_digest(
                    hashlib.sha256(confirm_raw).hexdigest(), digest
                )
            ):
                break
            if (
                bookmark is not None
                or attempts >= cls._DIAGNOSTIC_LEDGER_READ_ATTEMPTS
            ):
                raise RuntimeError(
                    "the ledger changed while the page was read; "
                    "restart the query instead of mixing versions"
                )
        pairs = state["pairs"]
        summaries = state["summaries"]
        order = sorted(
            range(len(pairs)),
            key=lambda index: (
                pairs[index][0],
                summaries[index]["transaction"],
            ),
        )
        return cls._diagnostic_ledger_page_result(
            [pairs[index] for index in order],
            [summaries[index] for index in order],
            identity,
            digest,
            stat_result.st_size,
            start_value,
            end_value,
            filter_transactions,
            page_limit,
            bookmark,
        )

    _DIAGNOSTIC_LEDGER_PROOF_FORMAT = (
        "branching-city-twin/diagnostic-ledger-proof"
    )
    _DIAGNOSTIC_LEDGER_PROOF_VERSION = 1
    _DIAGNOSTIC_LEDGER_PROOF_SUPPORTED_VERSIONS = (1,)
    _DIAGNOSTIC_LEDGER_PROOF_KEYS = (
        "format",
        "version",
        "filter",
        "snapshot",
        "manifest",
        "evidence",
        "segments",
        "checksum",
    )
    _DIAGNOSTIC_LEDGER_PROOF_FILTER_MODE_TRANSACTIONS = "transactions"
    _DIAGNOSTIC_LEDGER_PROOF_FILTER_MODE_TIME = "time"
    _DIAGNOSTIC_LEDGER_PROOF_FILTER_TXN_KEYS = ("mode", "transactions")
    _DIAGNOSTIC_LEDGER_PROOF_FILTER_TIME_KEYS = ("mode", "start", "end")
    _DIAGNOSTIC_LEDGER_PROOF_SNAPSHOT_KEYS = (
        "digest",
        "bytes",
        "segments",
        "entries",
        "dropped",
        "evidence",
    )
    _DIAGNOSTIC_LEDGER_PROOF_SEGMENT_KEYS = ("name", "document")

    @classmethod
    def _validate_diagnostic_ledger_proof_transactions(
        cls, transactions: Any
    ) -> tuple[str, ...]:
        """Validate the proof's transaction identity set: a non-empty
        :class:`tuple` of distinct non-empty :class:`str` identities,
        preserved in request order (the result keeps that order). A
        non-:class:`tuple` container or non-:class:`str` element raises
        :class:`TypeError`; an empty tuple, an empty identity or a
        duplicate identity raises :class:`ValueError`."""
        if not isinstance(transactions, tuple):
            raise TypeError(
                "transactions must be a tuple, got "
                f"{type(transactions).__name__}"
            )
        if not transactions:
            raise ValueError("transactions must be a non-empty tuple")
        seen: set[str] = set()
        identities: list[str] = []
        for item in transactions:
            if not isinstance(item, str):
                raise TypeError(
                    "every transaction must be a str, got "
                    f"{type(item).__name__}"
                )
            if not item:
                raise ValueError(
                    "transactions must not contain an empty identity"
                )
            if item in seen:
                raise ValueError(
                    "transactions must not contain duplicates"
                )
            seen.add(item)
            identities.append(item)
        return tuple(identities)

    @staticmethod
    def _validate_diagnostic_ledger_proof_bound(
        value: Any, name: str
    ) -> int | None:
        """Validate one proof time bound: ``None`` (only valid when the
        other bound is also ``None``) or a non-``bool`` non-negative
        :class:`int`. A wrong type raises :class:`TypeError`; a negative
        value raises :class:`ValueError` -- unlike the paging helper, a
        negative proof bound is an out-of-range value, not a type fault.
        """
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                f"{name} must be None or a non-negative int, got "
                f"{type(value).__name__}"
            )
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
        return value

    @classmethod
    def _resolve_diagnostic_ledger_proof_mode(
        cls,
        start_value: Any,
        end_value: Any,
        transactions: Any,
    ) -> dict[str, Any]:
        """Resolve and validate the proof's mutually exclusive filter
        modes: either both time bounds are ``None`` and ``transactions``
        is a non-empty tuple of distinct non-empty identities, or both
        bounds are non-``bool`` non-negative :class:`int` with
        ``start <= end`` and ``transactions`` is ``None``. Anything else
        raises :class:`TypeError` (bad element type) or
        :class:`ValueError` (mixed bounds, a negative or reversed range,
        empty or duplicated identities, neither mode or both modes)."""
        start_bound = cls._validate_diagnostic_ledger_proof_bound(
            start_value, "start"
        )
        end_bound = cls._validate_diagnostic_ledger_proof_bound(
            end_value, "end"
        )
        if (start_bound is None) != (end_bound is None):
            raise ValueError(
                "start and end must both be None or both be ints"
            )
        if start_bound is None:
            if transactions is None:
                raise ValueError(
                    "a proof requires either a time range or a "
                    "transaction identity set"
                )
            identities = (
                cls._validate_diagnostic_ledger_proof_transactions(
                    transactions
                )
            )
            return {
                "mode": cls._DIAGNOSTIC_LEDGER_PROOF_FILTER_MODE_TRANSACTIONS,
                "transactions": identities,
            }
        if transactions is not None:
            raise ValueError(
                "a proof must use either a time range or a transaction "
                "identity set, not both"
            )
        if start_bound > end_bound:
            raise ValueError("start must not be after end")
        return {
            "mode": cls._DIAGNOSTIC_LEDGER_PROOF_FILTER_MODE_TIME,
            "start": start_bound,
            "end": end_bound,
        }

    @classmethod
    def _diagnostic_ledger_proof_filter_document(
        cls, mode: dict[str, Any]
    ) -> dict[str, Any]:
        """Serialize the validated filter mode to its canonical proof
        object."""
        if (
            mode["mode"]
            == cls._DIAGNOSTIC_LEDGER_PROOF_FILTER_MODE_TRANSACTIONS
        ):
            return {
                "mode": cls._DIAGNOSTIC_LEDGER_PROOF_FILTER_MODE_TRANSACTIONS,
                "transactions": list(mode["transactions"]),
            }
        return {
            "mode": cls._DIAGNOSTIC_LEDGER_PROOF_FILTER_MODE_TIME,
            "start": mode["start"],
            "end": mode["end"],
        }

    @classmethod
    def _parse_diagnostic_ledger_proof_filter(
        cls, raw_filter: Any
    ) -> dict[str, Any]:
        """Strictly validate a proof package's filter object: the exact
        keys and value shapes of one of the two modes, with the same
        rules as export. Any defect raises :class:`ValueError`."""
        if not isinstance(raw_filter, dict):
            raise ValueError("ledger proof filter must be an object")
        mode = raw_filter.get("mode")
        if (
            mode
            == cls._DIAGNOSTIC_LEDGER_PROOF_FILTER_MODE_TRANSACTIONS
        ):
            if set(raw_filter) != set(
                cls._DIAGNOSTIC_LEDGER_PROOF_FILTER_TXN_KEYS
            ):
                raise ValueError("ledger proof filter has bad keys")
            raw_identities = raw_filter["transactions"]
            if not isinstance(raw_identities, list) or not raw_identities:
                raise ValueError(
                    "ledger proof transactions must be a non-empty list"
                )
            seen: set[str] = set()
            identities: list[str] = []
            for item in raw_identities:
                if not isinstance(item, str):
                    raise ValueError(
                        "ledger proof transactions must be str"
                    )
                if not item:
                    raise ValueError(
                        "ledger proof transactions must be non-empty"
                    )
                if item in seen:
                    raise ValueError(
                        "ledger proof transactions must not repeat"
                    )
                seen.add(item)
                identities.append(item)
            return {
                "mode": mode,
                "transactions": tuple(identities),
            }
        if mode == cls._DIAGNOSTIC_LEDGER_PROOF_FILTER_MODE_TIME:
            if set(raw_filter) != set(
                cls._DIAGNOSTIC_LEDGER_PROOF_FILTER_TIME_KEYS
            ):
                raise ValueError("ledger proof filter has bad keys")
            start = raw_filter["start"]
            end = raw_filter["end"]
            if (
                isinstance(start, bool)
                or not isinstance(start, int)
                or start < 0
                or isinstance(end, bool)
                or not isinstance(end, int)
                or end < 0
            ):
                raise ValueError("ledger proof time bounds are malformed")
            if start > end:
                raise ValueError(
                    "ledger proof time bounds are reversed"
                )
            return {"mode": mode, "start": start, "end": end}
        raise ValueError("ledger proof filter has an unknown mode")

    @classmethod
    def _diagnostic_ledger_proof_snapshot(
        cls,
        manifest_raw: bytes,
        manifest_document: dict[str, Any],
        segments: list[dict[str, Any]],
        evidence_digest: str | None,
    ) -> dict[str, Any]:
        """Assemble the manifest snapshot summary bound by a proof: the
        manifest's content digest and byte count, the retained segment
        and entry counts, the manifest's rotated-prefix continuity proof
        and the self-digest of the prefix evidence the manifest points
        at (or ``None``)."""
        return {
            "digest": hashlib.sha256(manifest_raw).hexdigest(),
            "bytes": len(manifest_raw),
            "segments": len(segments),
            "entries": sum(meta["entries"] for meta in segments),
            "dropped": (
                None
                if manifest_document["dropped"] is None
                else dict(manifest_document["dropped"])
            ),
            "evidence": evidence_digest,
        }

    @classmethod
    def _required_diagnostic_ledger_proof_segments(
        cls, segments: list[dict[str, Any]], mode: dict[str, Any]
    ) -> list[str]:
        """Names of the retained segments whose full bodies a proof must
        carry, in chain order.

        The transaction mode cannot rule any retained segment out from
        metadata alone (identity membership is invisible in the
        manifest), so every retained segment is required -- that is the
        only sound basis for the retained/rotated/absent judgement. The
        time mode needs exactly the segments whose manifest time bounds
        intersect the inclusive range; segments wholly before the start
        or wholly after the end are provably irrelevant by their
        manifest bounds and may have their bodies omitted."""
        if (
            mode["mode"]
            == cls._DIAGNOSTIC_LEDGER_PROOF_FILTER_MODE_TRANSACTIONS
        ):
            return [meta["name"] for meta in segments]
        start = mode["start"]
        end = mode["end"]
        return [
            meta["name"]
            for meta in segments
            if not (meta["last_at"] < start or meta["first_at"] > end)
        ]

    @staticmethod
    def _diagnostic_ledger_proof_range_meets_dropped(
        dropped: dict[str, Any] | None, start: int, end: int
    ) -> bool:
        """Whether the inclusive ``[start, end]`` range intersects the
        rotated-away prefix's recorded time bounds."""
        if dropped is None:
            return False
        return start <= dropped["last_at"] and end >= dropped["first_at"]

    @classmethod
    def _diagnostic_ledger_proof_items(
        cls,
        pairs: list[tuple[int, tuple[str, ...]]],
        summaries: list[dict[str, Any]],
        mode: dict[str, Any],
    ) -> tuple[dict[str, Any], ...]:
        """Build the matched retained entries in canonical (time,
        transaction) order, each a fresh ``at``/``summary``/``records``
        mapping exactly as :meth:`page_diagnostic_ledger` reports one."""
        if (
            mode["mode"]
            == cls._DIAGNOSTIC_LEDGER_PROOF_FILTER_MODE_TRANSACTIONS
        ):
            wanted = set(mode["transactions"])
            matched = [
                index
                for index, summary in enumerate(summaries)
                if summary["transaction"] in wanted
            ]
        else:
            start = mode["start"]
            end = mode["end"]
            matched = [
                index
                for index, (at, _chain) in enumerate(pairs)
                if start <= at <= end
            ]
        matched.sort(
            key=lambda index: (
                pairs[index][0],
                summaries[index]["transaction"],
            )
        )
        return tuple(
            {
                "at": pairs[index][0],
                "summary": dict(summaries[index]),
                "records": pairs[index][1],
            }
            for index in matched
        )

    @classmethod
    def export_diagnostic_ledger_proof(
        cls,
        directory: Any,
        start: Any,
        end: Any,
        transactions: Any,
    ) -> str:
        """Export an offline-verifiable historical proof from a
        segmented, possibly rotated directory diagnostic ledger,
        strictly read-only.

        ``directory`` must be a non-empty :class:`str` naming the
        directory ledger published by
        :meth:`append_diagnostic_ledger` (a non-:class:`str` raises
        :class:`TypeError`, an empty string :class:`ValueError`; a
        missing or non-directory path, a missing manifest, a read,
        locking or sync failure raises :class:`OSError`). Exactly one
        filter mode must be selected:

        * transaction mode -- ``start`` and ``end`` both ``None`` and
          ``transactions`` a non-empty :class:`tuple` of distinct
          non-empty :class:`str` identities. Every requested identity is
          classified deterministically as retained (a matching entry is
          returned), rotated away (remembered by the durable prefix
          evidence) or never seen;
        * time mode -- ``start`` and ``end`` both non-``bool``
          non-negative :class:`int` with ``start <= end`` and
          ``transactions`` ``None``. Only retained entries inside the
          inclusive range are returned, and the proof binds the complete
          set of segments intersecting the range, so verification can
          show no retained record in the range was omitted. If the range
          intersects the rotated-away prefix's time bounds,
          :class:`ValueError` is raised: the surviving prefix evidence
          only remembers aggregate time bounds, not per-entry times, so
          completeness cannot be claimed.

        A non-:class:`tuple` transaction container, a non-:class:`str`
        identity or a bad bound type raises :class:`TypeError`; mixed
        bounds, a reversed range, an empty tuple, an empty or repeated
        identity or selecting neither/both modes raises
        :class:`ValueError`. A corrupt manifest, retained segment,
        prefix evidence or digest chain raises :class:`ValueError`; no
        partial proof is ever produced.

        The export acquires a *shared* lock on the same coordination
        file append and rotation use exclusively, so it observes only a
        complete pre-publish or post-publish snapshot, never a mixture,
        and a segment being read is never garbage-collected. The proof
        is canonical compact UTF-8 JSON, checksummed like the ledger
        documents, and binds the manifest snapshot (its digest, byte
        count, retained segment/entry counts, dropped-prefix proof and
        evidence pointer), the full manifest, the prefix evidence (when
        one exists), the complete bodies of every required segment and
        the filter. Bodies of retained segments irrelevant to the
        conclusion are omitted; nothing about the ledger is modified.
        """
        cls._validate_diagnostic_ledger_path(directory)
        mode = cls._resolve_diagnostic_ledger_proof_mode(
            start, end, transactions
        )
        real_path = os.path.realpath(directory)
        if not os.path.isdir(real_path):
            raise OSError(
                f"diagnostic ledger directory not found: {directory!r}"
            )
        lock_path = os.path.join(
            real_path, cls._DIAGNOSTIC_LEDGER_LOCK_NAME
        )
        # Read-only and without O_CREAT, exactly like a directory page:
        # a directory ledger always carries its coordination file.
        lock_fd = os.open(lock_path, os.O_RDONLY)
        try:
            cls._acquire_diagnostic_ledger_reader_lock(lock_fd)
            try:
                return cls._export_diagnostic_ledger_proof_locked(
                    real_path, mode
                )
            finally:
                cls._release_diagnostic_ledger_reader_lock(lock_fd)
        finally:
            os.close(lock_fd)

    @classmethod
    def _export_diagnostic_ledger_proof_locked(
        cls, directory: str, mode: dict[str, Any]
    ) -> str:
        """Build the proof string under the shared reader lock. See
        :meth:`export_diagnostic_ledger_proof` for the contract; the
        caller holds the shared lock and ``directory`` is canonical."""
        manifest_path = os.path.join(
            directory, cls._DIAGNOSTIC_LEDGER_MANIFEST_NAME
        )
        attempts = 0
        while True:
            attempts += 1
            manifest_raw, stat_result = cls._read_diagnostic_ledger(
                manifest_path
            )
            identity = {
                "device": stat_result.st_dev,
                "inode": stat_result.st_ino,
                "size": stat_result.st_size,
            }
            state = cls._load_diagnostic_ledger_directory(
                directory, manifest_raw
            )
            confirm_raw, confirm_stat = cls._read_diagnostic_ledger(
                manifest_path
            )
            confirm_identity = {
                "device": confirm_stat.st_dev,
                "inode": confirm_stat.st_ino,
                "size": confirm_stat.st_size,
            }
            if (
                confirm_identity == identity
                and hmac.compare_digest(
                    hashlib.sha256(confirm_raw).hexdigest(),
                    hashlib.sha256(manifest_raw).hexdigest(),
                )
            ):
                break
            if attempts >= cls._DIAGNOSTIC_LEDGER_READ_ATTEMPTS:
                raise RuntimeError(
                    "the ledger changed while the proof was exported; "
                    "restart the query instead of mixing versions"
                )
        manifest_document = json.loads(
            manifest_raw.decode("utf-8"),
            object_pairs_hook=cls._reject_duplicate_json_keys,
        )
        retained = state["segments"]
        dropped = state["dropped"]
        if (
            mode["mode"] == cls._DIAGNOSTIC_LEDGER_PROOF_FILTER_MODE_TIME
            and cls._diagnostic_ledger_proof_range_meets_dropped(
                dropped, mode["start"], mode["end"]
            )
        ):
            raise ValueError(
                "the requested time range intersects the rotated-away "
                "prefix, whose per-entry times are not recoverable from "
                "the surviving prefix evidence"
            )
        required_names = (
            cls._required_diagnostic_ledger_proof_segments(retained, mode)
        )
        segment_documents: list[dict[str, Any]] = []
        for meta in retained:
            if meta["name"] not in required_names:
                continue
            segment_raw, _stat = cls._read_diagnostic_ledger(
                os.path.join(directory, meta["name"])
            )
            segment_documents.append(
                {
                    "name": meta["name"],
                    "document": json.loads(
                        segment_raw.decode("utf-8"),
                        object_pairs_hook=(
                            cls._reject_duplicate_json_keys
                        ),
                    ),
                }
            )
        evidence_document: dict[str, Any] | None = None
        evidence_digest: str | None = None
        if state["evidence"] is not None:
            evidence_raw, _stat = cls._read_diagnostic_ledger(
                os.path.join(directory, state["evidence_name"])
            )
            evidence_document = json.loads(
                evidence_raw.decode("utf-8"),
                object_pairs_hook=cls._reject_duplicate_json_keys,
            )
            evidence_digest = hashlib.sha256(evidence_raw).hexdigest()
        snapshot = cls._diagnostic_ledger_proof_snapshot(
            manifest_raw,
            manifest_document,
            retained,
            evidence_digest,
        )
        package: dict[str, Any] = {
            "format": cls._DIAGNOSTIC_LEDGER_PROOF_FORMAT,
            "version": cls._DIAGNOSTIC_LEDGER_PROOF_VERSION,
            "filter": cls._diagnostic_ledger_proof_filter_document(mode),
            "snapshot": snapshot,
            "manifest": manifest_document,
            "evidence": evidence_document,
            "segments": segment_documents,
        }
        body = dict(package)
        package["checksum"] = hashlib.sha256(
            cls._canonical_json(body).encode("utf-8")
        ).hexdigest()
        return cls._canonical_json(package)

    @classmethod
    def verify_diagnostic_ledger_proof(
        cls, proof: Any
    ) -> dict[str, Any]:
        """Authenticate an exported diagnostic ledger proof with no
        access to the original ledger.

        ``proof`` must be the canonical string returned by
        :meth:`export_diagnostic_ledger_proof`: a non-:class:`str`
        raises :class:`TypeError` and an empty string
        :class:`ValueError`. The proof is then fully re-authenticated:
        its encoding, canonical form, structure, version and checksum,
        the bound manifest, the inter-segment digest chain reconstructed
        from the manifest's ordered digests (anchored on the
        rotated-prefix proof or the genesis anchor), the prefix evidence,
        every included segment's digest, predecessor link, entry count,
        time bounds and embedded diagnostic chains, and the package
        checksum over the snapshot, manifest, evidence, complete required
        segment set and filter are all recomputed. The required segment
        set is independently derived from the filter and the manifest's
        segment bounds, so a missing or extra segment, a tampered
        digest, a cross-snapshot splice, a time range that reaches into
        the rotated prefix, an uncovered range or one identity falling
        into more than one class all raise :class:`ValueError`. No
        partial result is ever returned.

        On success returns a fresh isolated mapping with ``snapshot``,
        ``items``, ``rotated`` and ``absent`` in that order. ``items``
        is a tuple of matched retained entries in the ledger's canonical
        (time, transaction) order, each an ``at``/``summary``/
        ``records`` mapping as :meth:`page_diagnostic_ledger` reports
        one; in transaction mode ``rotated`` and ``absent`` are the
        requested identities found in the prefix evidence or nowhere,
        respectively, both in request order (in time mode both are
        empty tuples). The result shares no mutable state with the
        proof string or any other call.
        """
        _mode, snapshot, items, rotated, absent, _view = (
            cls._authenticate_diagnostic_ledger_proof_string(proof)
        )
        return {
            "snapshot": snapshot,
            "items": items,
            "rotated": rotated,
            "absent": absent,
        }

    @classmethod
    def _authenticate_diagnostic_ledger_proof_string(
        cls, proof: Any
    ) -> tuple[
        dict[str, Any],
        dict[str, Any],
        tuple[dict[str, Any], ...],
        tuple[str, ...],
        tuple[str, ...],
        dict[str, Any],
    ]:
        """Fully validate and authenticate one exported proof string and
        return ``(mode, snapshot, items, rotated, absent, view)``: the
        parsed filter mode, the authenticated snapshot summary, the
        matched items, the transaction-mode extra classes and the
        internal authenticated-history view described under
        :meth:`_authenticate_diagnostic_ledger_proof`. This is the
        complete validation pipeline of
        :meth:`verify_diagnostic_ledger_proof`; every defect raises
        exactly as documented there."""
        if not isinstance(proof, str):
            raise TypeError(
                f"proof must be a str, got {type(proof).__name__}"
            )
        if not proof:
            raise ValueError("proof must be a non-empty str")
        if proof.startswith("\ufeff"):
            raise ValueError("proof must be UTF-8 without a BOM")
        try:
            document = json.loads(
                proof,
                object_pairs_hook=cls._reject_duplicate_json_keys,
            )
        except ValueError as exc:
            raise ValueError(f"proof is not valid JSON: {exc}") from exc
        if cls._canonical_json(document) != proof:
            raise ValueError("proof is not canonical compact JSON")
        cls._parse_diagnostic_ledger_proof(document, proof)
        mode = cls._parse_diagnostic_ledger_proof_filter(
            document["filter"]
        )
        snapshot, items, rotated, absent, view = (
            cls._authenticate_diagnostic_ledger_proof(document, mode)
        )
        checksum = document["checksum"]
        body = {
            key: value
            for key, value in document.items()
            if key != "checksum"
        }
        if not hmac.compare_digest(
            checksum,
            hashlib.sha256(
                cls._canonical_json(body).encode("utf-8")
            ).hexdigest(),
        ):
            raise ValueError("proof checksum does not verify")
        return mode, snapshot, items, rotated, absent, view

    @classmethod
    def _parse_diagnostic_ledger_proof(
        cls, document: Any, proof: str
    ) -> None:
        """Validate the proof envelope's top-level shape, format,
        version and checksum field. The checksum itself is verified last
        by the caller once the package content has authenticated. Any
        defect raises :class:`ValueError`."""
        if proof.startswith("\ufeff"):
            raise ValueError("proof must be UTF-8 without a BOM")
        if proof.endswith("\n") or proof.endswith("\r"):
            raise ValueError("proof must not have a trailing newline")
        if not isinstance(document, dict) or set(document) != set(
            cls._DIAGNOSTIC_LEDGER_PROOF_KEYS
        ):
            raise ValueError("proof has bad top-level keys")
        if document["format"] != cls._DIAGNOSTIC_LEDGER_PROOF_FORMAT:
            raise ValueError("proof has an unknown format")
        version = document["version"]
        if (
            isinstance(version, bool)
            or not isinstance(version, int)
            or version
            not in cls._DIAGNOSTIC_LEDGER_PROOF_SUPPORTED_VERSIONS
        ):
            raise ValueError("proof has an unsupported version")
        checksum = document["checksum"]
        if (
            not isinstance(checksum, str)
            or len(checksum) != 64
            or set(checksum) - cls._HEX_DIGITS
        ):
            raise ValueError("proof checksum is malformed")
        snapshot = document["snapshot"]
        if not isinstance(snapshot, dict) or set(snapshot) != set(
            cls._DIAGNOSTIC_LEDGER_PROOF_SNAPSHOT_KEYS
        ):
            raise ValueError("proof snapshot has bad keys")

    @classmethod
    def _authenticate_diagnostic_ledger_proof(
        cls, document: dict[str, Any], mode: dict[str, Any]
    ) -> tuple[
        dict[str, Any],
        tuple[dict[str, Any], ...],
        tuple[str, ...],
        tuple[str, ...],
        dict[str, Any],
    ]:
        """Recompute every fact a proof claims from its embedded
        manifest, prefix evidence and required segment bodies and return
        ``(snapshot, items, rotated, absent, view)``. The caller has
        already validated the envelope and the package checksum is
        verified after this returns. Any inconsistency raises
        :class:`ValueError`; no partial result is produced.

        ``view`` is an internal fresh mapping with the authenticated
        history the proof binds: ``segments`` (one fresh metadata
        mapping per retained segment, in chain order), ``dropped`` (the
        rotated-prefix continuity proof, or ``None``), ``evidence`` (the
        rotated-away transaction identities in prefix order) and
        ``segment_entries`` (one tuple per retained segment, in chain
        order, of fresh ``at``/``summary``/``records`` entry mappings,
        or ``None`` for a segment whose body the filter did not
        require). It shares nothing with the proof document."""
        manifest_object = document["manifest"]
        if not isinstance(manifest_object, dict):
            raise ValueError("proof manifest must be an object")
        manifest_raw = cls._canonical_json(manifest_object).encode("utf-8")
        segments, dropped, evidence_ref, manifest_digest = (
            cls._parse_diagnostic_ledger_manifest(manifest_raw)
        )
        # Reconstruct the digest chain from the manifest's ordered
        # digests: the manifest is the published snapshot record and its
        # checksum covers this ordering. Every embedded body then proves
        # its own link against the same predecessor.
        expected_previous: dict[str, str] = {}
        previous_digest = (
            cls._DIAGNOSTIC_LEDGER_GENESIS_PREV
            if dropped is None
            else dropped["digest"]
        )
        for meta in segments:
            expected_previous[meta["name"]] = previous_digest
            previous_digest = meta["digest"]
        evidence_object = document["evidence"]
        evidence_digest: str | None = None
        evidence_transactions: set[str] = set()
        evidence_identities: tuple[str, ...] = ()
        if dropped is None:
            if evidence_object is not None:
                raise ValueError(
                    "proof carries prefix evidence without a dropped "
                    "prefix"
                )
        else:
            if not isinstance(evidence_object, dict):
                raise ValueError(
                    "proof is missing its rotated-prefix evidence"
                )
            if evidence_ref is None:
                raise ValueError(
                    "proof manifest dropped proof and evidence disagree"
                )
            evidence_raw = cls._canonical_json(
                evidence_object
            ).encode("utf-8")
            evidence_digest = hashlib.sha256(evidence_raw).hexdigest()
            if not hmac.compare_digest(
                evidence_digest, evidence_ref["digest"]
            ):
                raise ValueError(
                    "proof prefix evidence digest does not match the "
                    "manifest"
                )
            evidence = cls._parse_diagnostic_ledger_evidence(
                evidence_raw
            )
            if (
                evidence["digest"] != dropped["digest"]
                or evidence["entries"] != dropped["entries"]
                or evidence["first_at"] != dropped["first_at"]
                or evidence["last_at"] != dropped["last_at"]
            ):
                raise ValueError(
                    "proof prefix evidence does not match the dropped "
                    "prefix"
                )
            evidence_transactions = set(evidence["transactions"])
            evidence_identities = evidence["transactions"]
        required_names = (
            cls._required_diagnostic_ledger_proof_segments(segments, mode)
        )
        raw_embedded = document["segments"]
        if not isinstance(raw_embedded, list):
            raise ValueError("proof segments must be a list")
        metas_by_name = {meta["name"]: meta for meta in segments}
        embedded: dict[str, tuple[int, str, list[
            tuple[int, tuple[str, ...]]
        ]]] = {}
        for entry in raw_embedded:
            if not isinstance(entry, dict) or set(entry) != set(
                cls._DIAGNOSTIC_LEDGER_PROOF_SEGMENT_KEYS
            ):
                raise ValueError("proof segment entry has bad keys")
            name = entry["name"]
            if (
                not isinstance(name, str)
                or name not in metas_by_name
                or name in embedded
            ):
                raise ValueError("proof segment name is malformed")
            segment_document = entry["document"]
            if not isinstance(segment_document, dict):
                raise ValueError("proof segment document must be an object")
            segment_raw = cls._canonical_json(
                segment_document
            ).encode("utf-8")
            index, prev, pairs, digest = (
                cls._parse_diagnostic_ledger_segment(segment_raw)
            )
            meta = metas_by_name[name]
            if index != meta["index"]:
                raise ValueError(
                    "proof segment index does not match the manifest"
                )
            if not hmac.compare_digest(digest, meta["digest"]):
                raise ValueError(
                    "proof segment digest does not match the manifest"
                )
            if not hmac.compare_digest(prev, expected_previous[name]):
                raise ValueError("proof segment digest chain is broken")
            if len(pairs) != meta["entries"]:
                raise ValueError(
                    "proof segment entry count does not match the manifest"
                )
            if (
                pairs[0][0] != meta["first_at"]
                or pairs[-1][0] != meta["last_at"]
            ):
                raise ValueError(
                    "proof segment time bounds do not match the manifest"
                )
            embedded[name] = (index, prev, pairs)
        if set(embedded) != set(required_names):
            raise ValueError(
                "proof does not carry exactly the segments required by "
                "its filter"
            )
        # Splice the embedded bodies back together in manifest order and
        # authenticate the embedded chains, their per-segment canonical
        # ordering, non-decreasing times across embedded boundaries and
        # their disjointness from the remembered prefix identities.
        all_pairs: list[tuple[int, tuple[str, ...]]] = []
        previous_at: int | None = None
        for meta in segments:
            if meta["name"] not in embedded:
                previous_at = None
                continue
            _index, _prev, pairs = embedded[meta["name"]]
            if previous_at is not None and pairs[0][0] < previous_at:
                raise ValueError(
                    "proof segments are not in time order"
                )
            all_pairs.extend(pairs)
            previous_at = pairs[-1][0]
        chains = tuple(chain for _at, chain in all_pairs)
        summaries = cls._authenticate_recovery_diagnostic_chains(chains)
        offset = 0
        for meta in segments:
            if meta["name"] not in embedded:
                continue
            _index, _prev, pairs = embedded[meta["name"]]
            keys = [
                (at, summaries[offset + position]["transaction"])
                for position, (at, _chain) in enumerate(pairs)
            ]
            if keys != sorted(keys):
                raise ValueError(
                    "proof segment entries are not in canonical order"
                )
            offset += len(pairs)
        retained_transactions = {
            summary["transaction"] for summary in summaries
        }
        if retained_transactions & evidence_transactions:
            raise ValueError(
                "proof prefix evidence repeats a retained transaction"
            )
        if (
            mode["mode"] == cls._DIAGNOSTIC_LEDGER_PROOF_FILTER_MODE_TIME
            and cls._diagnostic_ledger_proof_range_meets_dropped(
                dropped, mode["start"], mode["end"]
            )
        ):
            raise ValueError(
                "proof time range intersects the rotated-away prefix, "
                "whose per-entry times are not recoverable"
            )
        items = cls._diagnostic_ledger_proof_items(
            all_pairs, summaries, mode
        )
        rotated: tuple[str, ...] = ()
        absent: tuple[str, ...] = ()
        if (
            mode["mode"]
            == cls._DIAGNOSTIC_LEDGER_PROOF_FILTER_MODE_TRANSACTIONS
        ):
            items_by_transaction: dict[str, int] = {}
            for item in items:
                identity = item["summary"]["transaction"]
                if identity in items_by_transaction:
                    raise ValueError(
                        "one transaction identity is classified as "
                        "retained more than once"
                    )
                items_by_transaction[identity] = 1
            rotated_identities: list[str] = []
            absent_identities: list[str] = []
            for identity in mode["transactions"]:
                classes = (
                    identity in items_by_transaction,
                    identity in evidence_transactions,
                )
                if sum(classes) > 1:
                    raise ValueError(
                        "one transaction identity falls into more than "
                        "one classification"
                    )
                if classes[0]:
                    continue
                if classes[1]:
                    rotated_identities.append(identity)
                else:
                    absent_identities.append(identity)
            rotated = tuple(rotated_identities)
            absent = tuple(absent_identities)
        snapshot = cls._diagnostic_ledger_proof_snapshot(
            manifest_raw,
            manifest_object,
            segments,
            evidence_digest,
        )
        if snapshot != document["snapshot"]:
            raise ValueError("proof snapshot does not match its manifest")
        segment_entries: list[tuple[dict[str, Any], ...] | None] = []
        offset = 0
        for meta in segments:
            embedded_entry = embedded.get(meta["name"])
            if embedded_entry is None:
                segment_entries.append(None)
                continue
            _index, _prev, pairs = embedded_entry
            segment_entries.append(
                tuple(
                    {
                        "at": pairs[position][0],
                        "summary": dict(summaries[offset + position]),
                        "records": pairs[position][1],
                    }
                    for position in range(len(pairs))
                )
            )
            offset += len(pairs)
        view = {
            "segments": [dict(meta) for meta in segments],
            "dropped": None if dropped is None else dict(dropped),
            "evidence": evidence_identities,
            "segment_entries": tuple(segment_entries),
        }
        return dict(snapshot), items, rotated, absent, view

    @classmethod
    def compare_diagnostic_ledger_proofs(
        cls, before: Any, after: Any
    ) -> dict[str, Any]:
        """Compare two exported diagnostic ledger proofs entirely
        offline and report how the ``after`` history relates to the
        ``before`` history, with no access to any ledger.

        The arguments are validated in ``before``, ``after`` order,
        each exactly as :meth:`verify_diagnostic_ledger_proof`
        requires: a non-:class:`str` raises :class:`TypeError`; an
        empty string, a non-canonical proof or any authentication
        failure raises :class:`ValueError`. Both proofs must then use
        the transaction-identity filter mode and request the *same*
        ordered identity set; a time-range proof or two different
        identity sets raise :class:`ValueError`. The comparison itself
        is pure: every fact comes from the two authenticated proof
        packages, nothing is read from or written to the filesystem,
        and the proofs, any ledger, the event graph, the business
        audit and the idempotency state are never modified, whether
        the call succeeds or raises.

        When the two authenticated snapshots are equivalent the
        relation is ``"same"`` and every change level is empty.
        Otherwise the histories are aligned: a rotated-away prefix
        carried by ``after`` must *exactly* inherit the history
        ``before`` authenticated -- the prefix's last segment digest,
        total entry count and time bounds must match the corresponding
        retained segments of ``before`` position by position, and its
        prefix evidence must hold exactly the identities ``before``
        remembered plus those of the newly covered segments. A digest,
        entry-count, time-bound or evidence contradiction at such a
        shared position is a history rewrite and raises
        :class:`ValueError`; so does a rotation that ran past the
        ``before`` proof's tail when the transition segments
        connecting the two ends are missing from both proofs (the
        aggregate prefix digest alone never stands in for them), an
        ``after`` proof that is a strict historical prefix of
        ``before`` (a rollback), and -- unless ``before`` is the empty
        ledger -- two non-empty proofs with no authenticatable common
        segment (a cross-ledger splice).

        The relation is ``"continued"`` when ``after`` extends the
        ``before`` tail (possibly after a verified rotation), and
        ``"forked"`` when both proofs carry two different legal
        successors of the common boundary.

        On success returns a fresh mapping with ``relation``,
        ``common``, ``before_only``, ``after_only`` and ``rotated`` in
        that order. ``common`` names the last segment the two sides
        agree on digest-by-digest: a mapping with its ``index``,
        ``digest`` and cumulative ``entries`` count, or ``None`` when
        the comparison starts from the empty ledger. ``before_only``
        and ``after_only`` are tuples of the segments each side alone
        carries after that boundary, in chain order, each a mapping
        with the segment ``index``, its ``digest`` and an isolated
        ``entries`` copy of its diagnostic entries (each an
        ``at``/``summary``/``records`` mapping as
        :meth:`page_diagnostic_ledger` reports one); a ``"continued"``
        result leaves the increment only in ``after_only``, a
        ``"forked"`` result carries both sides and a ``"same"`` result
        has two empty tuples. ``rotated`` is the tuple of requested
        identities that newly entered the rotated-away prefix, in
        request order -- only ever non-empty for ``"continued"``.
        Every returned level is detached from the proof strings and
        from every other call, and no level shares mutable state with
        another.
        """
        before_mode, before_snapshot, _b_items, before_rotated, \
            _b_absent, before_view = (
                cls._authenticate_diagnostic_ledger_proof_string(before)
            )
        if (
            before_mode["mode"]
            != cls._DIAGNOSTIC_LEDGER_PROOF_FILTER_MODE_TRANSACTIONS
        ):
            raise ValueError(
                "the before proof must use the transaction identity "
                "filter mode"
            )
        after_mode, after_snapshot, _a_items, after_rotated, \
            _a_absent, after_view = (
                cls._authenticate_diagnostic_ledger_proof_string(after)
            )
        if (
            after_mode["mode"]
            != cls._DIAGNOSTIC_LEDGER_PROOF_FILTER_MODE_TRANSACTIONS
        ):
            raise ValueError(
                "the after proof must use the transaction identity "
                "filter mode"
            )
        if before_mode["transactions"] != after_mode["transactions"]:
            raise ValueError(
                "both proofs must request the same ordered transaction "
                "identity set"
            )
        b_segments = before_view["segments"]
        a_segments = after_view["segments"]
        b_dropped = before_view["dropped"]
        a_dropped = after_view["dropped"]
        if before_snapshot == after_snapshot:
            return {
                "relation": "same",
                "common": cls._diagnostic_ledger_proof_chain_boundary(
                    before_view
                ),
                "before_only": (),
                "after_only": (),
                "rotated": (),
            }
        if not b_segments:
            # The empty ledger has no retained segment and (checked
            # here) no rotated prefix, so there is nothing to align:
            # any non-empty authenticated history continues it.
            if b_dropped is not None:
                raise ValueError(
                    "the before proof's rotated-prefix boundary cannot "
                    "be located without retained segments"
                )
            if not a_segments:
                raise ValueError(
                    "the after proof's rotated-prefix boundary cannot "
                    "be located without retained segments"
                )
            return {
                "relation": "continued",
                "common": None,
                "before_only": (),
                "after_only": cls._diagnostic_ledger_proof_segment_items(
                    after_view,
                    a_segments[0]["index"],
                    a_segments[-1]["index"],
                ),
                "rotated": after_rotated,
            }
        b_boundary = cls._diagnostic_ledger_proof_chain_start(
            before_view, "before"
        )
        a_boundary = cls._diagnostic_ledger_proof_chain_start(
            after_view, "after"
        )
        if a_dropped is None:
            if b_dropped is not None:
                raise ValueError(
                    "the after proof does not carry the rotated-prefix "
                    "continuity proof the before proof established"
                )
        elif b_dropped is None:
            cls._diagnostic_ledger_proof_prefix_inheritance(
                before_view, a_boundary, a_dropped, after_view["evidence"]
            )
        else:
            if a_boundary < b_boundary:
                raise ValueError(
                    "the after proof rotates away less history than "
                    "the before proof already rotated away"
                )
            if a_boundary == b_boundary:
                if (
                    a_dropped != b_dropped
                    or before_view["evidence"] != after_view["evidence"]
                ):
                    raise ValueError(
                        "the proofs disagree about the rotated-away "
                        "prefix they share"
                    )
            else:
                cls._diagnostic_ledger_proof_prefix_inheritance(
                    before_view,
                    a_boundary,
                    a_dropped,
                    after_view["evidence"],
                )
        b_start = b_segments[0]["index"]
        b_end = b_segments[-1]["index"]
        a_start = a_segments[0]["index"] if a_segments else 1
        a_end = a_segments[-1]["index"] if a_segments else 0
        divergence: int | None = None
        for position in range(max(b_start, a_start), min(b_end, a_end) + 1):
            if not hmac.compare_digest(
                b_segments[position - b_start]["digest"],
                a_segments[position - a_start]["digest"],
            ):
                divergence = position
                break
        if divergence is not None:
            boundary = divergence - 1
            if boundary >= a_start:
                common = cls._diagnostic_ledger_proof_boundary_at(
                    after_view, boundary
                )
            elif a_dropped is not None:
                # The boundary is the verified rotated-prefix boundary
                # segment both sides authenticated.
                common = {
                    "index": a_boundary,
                    "digest": a_dropped["digest"],
                    "entries": a_dropped["entries"],
                }
            else:
                raise ValueError(
                    "the two proofs share no authenticatable segment "
                    "and come from different ledgers"
                )
            return {
                "relation": "forked",
                "common": common,
                "before_only": (
                    cls._diagnostic_ledger_proof_segment_items(
                        before_view, divergence, b_end
                    )
                ),
                "after_only": cls._diagnostic_ledger_proof_segment_items(
                    after_view, divergence, a_end
                ),
                "rotated": (),
            }
        if a_end < b_end:
            raise ValueError(
                "the after proof is a strict historical prefix of the "
                "before proof (a rollback)"
            )
        if b_end >= a_start:
            common = cls._diagnostic_ledger_proof_boundary_at(
                before_view, b_end
            )
        else:
            # after rotated away exactly up to the before proof's tail,
            # so the shared boundary is the verified prefix boundary.
            common = {
                "index": a_boundary,
                "digest": a_dropped["digest"],
                "entries": a_dropped["entries"],
            }
        before_rotated_set = set(before_rotated)
        return {
            "relation": "continued",
            "common": common,
            "before_only": (),
            "after_only": cls._diagnostic_ledger_proof_segment_items(
                after_view, max(a_start, b_end + 1), a_end
            ),
            "rotated": tuple(
                identity
                for identity in after_rotated
                if identity not in before_rotated_set
            ),
        }

    @classmethod
    def _diagnostic_ledger_proof_chain_start(
        cls, view: dict[str, Any], side: str
    ) -> int:
        """The segment index of the rotated-prefix boundary of one
        authenticated proof view (0 when nothing was rotated away),
        after checking the retained chain starts where the prefix
        proof says it must. A boundary that cannot be located, or a
        retained chain that does not start at the genesis segment
        (without a prefix) or right after the boundary (with one),
        raises :class:`ValueError`."""
        segments = view["segments"]
        dropped = view["dropped"]
        if dropped is None:
            if segments and segments[0]["index"] != 1:
                raise ValueError(
                    f"the {side} proof's retained chain does not start "
                    "at the genesis segment"
                )
            return 0
        if not segments:
            raise ValueError(
                f"the {side} proof's rotated-prefix boundary cannot "
                "be located without retained segments"
            )
        if segments[0]["index"] < 2:
            raise ValueError(
                f"the {side} proof's rotated-prefix proof covers no "
                "segment"
            )
        return segments[0]["index"] - 1

    @classmethod
    def _diagnostic_ledger_proof_prefix_inheritance(
        cls,
        before_view: dict[str, Any],
        a_boundary: int,
        a_dropped: dict[str, Any],
        a_evidence: tuple[str, ...],
    ) -> None:
        """Verify that the ``after`` proof's rotated-away prefix
        exactly inherits the history the ``before`` proof
        authenticated: the boundary segment's digest, the cumulative
        entry count, the prefix time bounds and the prefix evidence's
        identity sequence must equal, position by position, what
        ``before_view`` retains (plus its own already-rotated prefix,
        if any). A rotation that ran past the ``before`` proof's tail
        without the transition segments connecting the two ends, and
        any digest, count, time-bound or identity contradiction at a
        shared position, raises :class:`ValueError`."""
        b_segments = before_view["segments"]
        b_entries = before_view["segment_entries"]
        b_dropped = before_view["dropped"]
        b_start = b_segments[0]["index"]
        b_end = b_segments[-1]["index"]
        first_covered = b_start if b_dropped is not None else 1
        if a_boundary > b_end:
            raise ValueError(
                "the after proof rotated away history past the before "
                "proof's tail and the transition segments connecting "
                "the two ends are missing"
            )
        covered: list[int] = []
        for position in range(first_covered, a_boundary + 1):
            if position < b_start:
                raise ValueError(
                    "the after proof rotated away segments the before "
                    "proof does not carry and the transition segments "
                    "connecting the two ends are missing"
                )
            covered.append(position - b_start)
        boundary_meta = b_segments[covered[-1]]
        if not hmac.compare_digest(
            boundary_meta["digest"], a_dropped["digest"]
        ):
            raise ValueError(
                "the after proof's rotated-prefix digest contradicts "
                "the before proof's retained segment at the boundary"
            )
        entries = sum(b_segments[offset]["entries"] for offset in covered)
        if b_dropped is not None:
            entries += b_dropped["entries"]
        if entries != a_dropped["entries"]:
            raise ValueError(
                "the after proof's rotated-prefix entry count "
                "contradicts the before proof's retained segments"
            )
        first_at = (
            b_segments[covered[0]]["first_at"]
            if b_dropped is None
            else b_dropped["first_at"]
        )
        if first_at != a_dropped["first_at"]:
            raise ValueError(
                "the after proof's rotated-prefix time bounds "
                "contradict the before proof's retained segments"
            )
        if boundary_meta["last_at"] != a_dropped["last_at"]:
            raise ValueError(
                "the after proof's rotated-prefix time bounds "
                "contradict the before proof's retained segments"
            )
        new_identities: list[str] = []
        for offset in covered:
            for entry in b_entries[offset]:
                new_identities.append(entry["summary"]["transaction"])
        if (
            tuple(before_view["evidence"]) + tuple(new_identities)
            != a_evidence
        ):
            raise ValueError(
                "the after proof's prefix evidence does not exactly "
                "inherit the before proof's history"
            )

    @classmethod
    def _diagnostic_ledger_proof_boundary_at(
        cls, view: dict[str, Any], index: int
    ) -> dict[str, Any]:
        """The common-boundary record (``index``, ``digest`` and
        cumulative ``entries`` count) for the retained segment at
        segment number ``index`` of one authenticated proof view."""
        segments = view["segments"]
        dropped = view["dropped"]
        entries = 0 if dropped is None else dropped["entries"]
        meta: dict[str, Any] | None = None
        for candidate in segments:
            if candidate["index"] > index:
                break
            entries += candidate["entries"]
            meta = candidate
        assert meta is not None and meta["index"] == index
        return {
            "index": index,
            "digest": meta["digest"],
            "entries": entries,
        }

    @classmethod
    def _diagnostic_ledger_proof_chain_boundary(
        cls, view: dict[str, Any]
    ) -> dict[str, Any] | None:
        """The common-boundary record for the tail of one
        authenticated proof view's chain, or ``None`` for the empty
        ledger."""
        segments = view["segments"]
        if not segments:
            return None
        return cls._diagnostic_ledger_proof_boundary_at(
            view, segments[-1]["index"]
        )

    @classmethod
    def _diagnostic_ledger_proof_segment_items(
        cls, view: dict[str, Any], first: int, last: int
    ) -> tuple[dict[str, Any], ...]:
        """One fresh ``index``/``digest``/``entries`` mapping per
        retained segment of ``view`` with a segment number in the
        inclusive ``first``..``last`` range, in chain order; an empty
        range yields an empty tuple."""
        if first > last:
            return ()
        segments = view["segments"]
        segment_entries = view["segment_entries"]
        start = segments[0]["index"]
        items: list[dict[str, Any]] = []
        for position in range(first, last + 1):
            offset = position - start
            meta = segments[offset]
            items.append(
                {
                    "index": meta["index"],
                    "digest": meta["digest"],
                    "entries": segment_entries[offset],
                }
            )
        return tuple(items)

    @classmethod
    def _publish_caches_recovery_locked(
        cls,
        index_path: str,
        index_tmp: str,
        progress_path: str,
        progress_tmp: str | None,
        recovery_path: str,
    ) -> None:
        """Publish the new caches under the cross-process recovery
        protocol. The caller holds the chain lock and has already
        flushed and fsync-ed every temp file; this routine owns the
        durable publication sequence and cleanup.

        ``progress_tmp`` is ``None`` for an index-only publication (no
        complete segment exists yet, so no new cursor was built): the
        progress target is then recorded unchanged -- its current bytes
        as both the old and the new version, or its continued absence --
        and never replaced, but it still appears in the record so no
        target is ever replaced outside the protocol.

        Every pre-existing target is streamed byte-for-byte into a
        durable backup in the shared directory first. The recovery
        record -- version, phase, both targets' old/new existence and
        SHA-256 digests and the backup names, covered by a checksum over
        every other field -- is then durably published *before* the
        first target is replaced. After each atomic replacement (and its
        directory syncs) the record is rewritten and synced with the
        advanced phase. Only once every replacement has landed are the
        old backups and, last, the record removed, so a process dying at
        any point leaves the next entry-point call enough durable
        evidence to commit the complete new version or restore the
        complete old one.

        Any in-process failure first converges the on-disk publication
        with the same recovery routine the next process would run, so
        the raising call never strands a mixed version; failures of that
        convergence propagate as :class:`OSError` with all evidence
        retained. The canonical chain and all generation state are
        untouched either way.
        """
        directory = os.path.dirname(os.path.abspath(index_path))
        prepared: list[dict[str, Any]] = []
        try:
            for target, tmp_path in (
                (index_path, index_tmp),
                (progress_path, progress_tmp),
            ):
                if os.path.lexists(target):
                    old_digest = cls._sha256_file(target)
                    backup_path = cls._backup_cache_file(target)
                    backup_name = os.path.basename(backup_path)
                    if not hmac.compare_digest(
                        cls._sha256_file(backup_path), old_digest
                    ):
                        raise ValueError(
                            "recovery backup does not match the target's "
                            "current bytes"
                        )
                else:
                    old_digest = ""
                    backup_path = None
                    backup_name = ""
                new_digest = (
                    cls._sha256_file(tmp_path)
                    if tmp_path is not None
                    else old_digest
                )
                prepared.append(
                    {
                        "target": target,
                        "tmp": tmp_path,
                        "old_exists": backup_path is not None,
                        "old_digest": old_digest,
                        "new_digest": new_digest,
                        "backup_path": backup_path,
                        "backup_name": backup_name,
                    }
                )

            def described() -> tuple[dict[str, Any], dict[str, Any]]:
                return (
                    {
                        "old_exists": prepared[0]["old_exists"],
                        "old_digest": prepared[0]["old_digest"],
                        "new_exists": True,
                        "new_digest": prepared[0]["new_digest"],
                        "backup": prepared[0]["backup_name"],
                    },
                    {
                        "old_exists": prepared[1]["old_exists"],
                        "old_digest": prepared[1]["old_digest"],
                        "new_exists": True,
                        "new_digest": prepared[1]["new_digest"],
                        "backup": prepared[1]["backup_name"],
                    },
                )

            index_desc, progress_desc = described()
            cls._write_cache_recovery_record(
                recovery_path,
                cls._cache_recovery_record_bytes(
                    cls._RECOVERY_CACHE_PHASE_PREPARED,
                    index_desc,
                    progress_desc,
                ),
            )
            phases = [
                (index_path, index_tmp, cls._RECOVERY_CACHE_PHASE_INDEX),
            ]
            if progress_tmp is not None:
                phases.append(
                    (
                        progress_path,
                        progress_tmp,
                        cls._RECOVERY_CACHE_PHASE_PROGRESS,
                    )
                )
            for target, tmp_path, phase in phases:
                cls._fsync_directory(directory)
                os.replace(tmp_path, target)
                cls._fsync_directory(directory)
                index_desc, progress_desc = described()
                cls._write_cache_recovery_record(
                    recovery_path,
                    cls._cache_recovery_record_bytes(
                        phase, index_desc, progress_desc
                    ),
                )
            for entry in prepared:
                if entry["backup_path"] is not None:
                    os.remove(entry["backup_path"])
            cls._fsync_directory(directory)
            os.remove(recovery_path)
            cls._fsync_directory(directory)
        except BaseException:
            if os.path.lexists(recovery_path):
                # A durable record exists, so one or both replacements
                # may have landed: converge exactly as a fresh process
                # would. If convergence itself fails its exception
                # (ValueError with all evidence retained for corrupt
                # material, OSError for a failed restore/cleanup/sync)
                # propagates instead of the original.
                cls._recover_cache_publication_locked(
                    index_path, progress_path, recovery_path
                )
            else:
                # The record was never durable, so no replacement could
                # have become visible: discard this call's backups and
                # temps without touching the targets.
                for entry in prepared:
                    backup_path = entry["backup_path"]
                    if backup_path is not None and os.path.lexists(
                        backup_path
                    ):
                        with contextlib.suppress(OSError):
                            os.remove(backup_path)
            for entry in prepared:
                tmp_path = entry["tmp"]
                if tmp_path is not None and os.path.lexists(tmp_path):
                    with contextlib.suppress(OSError):
                        os.remove(tmp_path)
            raise

    @classmethod
    def _new_segment_index_temp(cls, index_path: str) -> tuple[Any, str]:
        """Create the temp file a fresh segment index is built into."""
        directory = os.path.dirname(os.path.abspath(index_path))
        fd, tmp_path = tempfile.mkstemp(
            prefix=".recovery-chain-segments-", suffix=".tmp", dir=directory
        )
        return os.fdopen(fd, "wb"), tmp_path

    @classmethod
    def _new_progress_temp(cls, progress_path: str) -> tuple[Any, str]:
        """Create the temp file a fresh progress cursor is built into."""
        directory = os.path.dirname(os.path.abspath(progress_path))
        fd, tmp_path = tempfile.mkstemp(
            prefix=".recovery-chain-progress-", suffix=".tmp", dir=directory
        )
        return os.fdopen(fd, "wb"), tmp_path

    @classmethod
    def _full_segment_build_locked(
        cls,
        path: str,
        index_path: str,
        progress_path: str | None,
        segment_size: int,
        wanted: set[int],
        anchor_boundary_seq: int = 0,
    ) -> dict[str, Any]:
        """Stream the whole canonical chain exactly once, fully
        authenticating every frame, and build the complete segmented
        index temp plus (when a complete segment boundary exists and a
        progress path was given) the progress temp.

        Returns a mapping with ``version``, ``count``, ``head``,
        ``captured``, ``index_tmp``, ``index_fingerprint``,
        ``progress_tmp``, ``progress_payload`` (the normalized cursor
        written, or ``None``), ``anchor_digest`` and ``anchor_prefix``
        (the frame digest and raw-prefix digest at
        ``anchor_boundary_seq``, empty when zero). The caller holds the
        chain lock and owns publication and cleanup; nothing is replaced
        here.

        Segment records are emitted as their frames stream past, so no
        more than one frame, audit body or segment is ever materialized
        besides the wanted bodies. The emitted records feed both the
        temp file and two rolling fingerprints -- one over every
        segment and one over only the complete segments -- and a
        raw-byte hasher over the single scan is snapshotted whenever a
        frame completes a segment, yielding the raw authenticated
        prefix through the latest complete boundary without a second
        pass.
        """
        captured: dict[int, dict[str, Any]] = {}
        fingerprint_all = hashlib.sha256()
        fingerprint_complete = hashlib.sha256()
        raw_hasher = hashlib.sha256()
        prefix_digest = hashlib.sha256(b"").hexdigest()
        boundary_seq = 0
        boundary_offset = 0
        boundary_length = 0
        boundary_digest = cls._RECOVERY_CHAIN_GENESIS_PREV
        anchor_digest = ""
        anchor_prefix = ""
        last_frame = [0, 0, 0, ""]
        handle, index_tmp = cls._new_segment_index_temp(index_path)
        progress_tmp: str | None = None
        try:
            with handle:
                handle.write(cls._segments_envelope_prefix(segment_size))
                first = True

                def on_segment(segment: dict[str, Any]) -> None:
                    nonlocal first
                    canonical = cls._canonical_json(segment).encode("utf-8")
                    fingerprint_all.update(canonical)
                    if segment["last"] % segment_size == 0:
                        fingerprint_complete.update(canonical)
                    if not first:
                        handle.write(b",")
                    handle.write(canonical)
                    first = False

                segmenter = _ChainSegmenter(
                    segment_size, cls._canonical_json, on_segment
                )

                def on_frame(frame: dict[str, Any]) -> None:
                    last_frame[0] = frame["seq"]
                    last_frame[1] = frame["offset"]
                    last_frame[2] = frame["length"]
                    last_frame[3] = frame["digest"]
                    if frame["seq"] in wanted:
                        captured[frame["seq"]] = frame["body"]
                    segmenter.on_frame(frame)

                def on_authenticated(stream: Any) -> None:
                    nonlocal prefix_digest
                    nonlocal boundary_seq, boundary_offset, boundary_length
                    nonlocal boundary_digest, anchor_digest, anchor_prefix
                    seq = last_frame[0]
                    if seq % segment_size == 0:
                        prefix_digest = stream.consumed_digest()
                        boundary_seq = seq
                        boundary_offset = last_frame[1]
                        boundary_length = last_frame[2]
                        boundary_digest = last_frame[3]
                    if seq == anchor_boundary_seq:
                        anchor_digest = last_frame[3]
                        anchor_prefix = stream.consumed_digest()

                chain_file = cls._open_recovery_chain_locked(path)
                try:
                    version, count, head = cls._scan_recovery_chain(
                        chain_file,
                        on_frame,
                        raw_hasher=raw_hasher,
                        on_frame_authenticated=on_authenticated,
                    )
                finally:
                    chain_file.close()
                segmenter.finish()
                handle.write(cls._segments_trailer(version, count, head))
                handle.flush()
                os.fsync(handle.fileno())
            index_fingerprint = fingerprint_all.hexdigest()
            complete_fingerprint = fingerprint_complete.hexdigest()
            progress_payload: dict[str, Any] | None = None
            if progress_path is not None and boundary_seq >= 1:
                progress_payload = {
                    "segment_size": segment_size,
                    "frames": count,
                    "head": head,
                    "boundary": {
                        "seq": boundary_seq,
                        "offset": boundary_offset,
                        "length": boundary_length,
                        "digest": boundary_digest,
                    },
                    "prefix": prefix_digest,
                    "index": complete_fingerprint,
                }
                progress_handle, progress_tmp = cls._new_progress_temp(
                    progress_path
                )
                with progress_handle:
                    progress_handle.write(
                        cls._progress_document_bytes(progress_payload)
                    )
                    progress_handle.flush()
                    os.fsync(progress_handle.fileno())
        except BaseException:
            with contextlib.suppress(OSError):
                os.remove(index_tmp)
            if progress_tmp is not None:
                with contextlib.suppress(OSError):
                    os.remove(progress_tmp)
            raise
        return {
            "version": version,
            "count": count,
            "head": head,
            "captured": captured,
            "index_tmp": index_tmp,
            "index_fingerprint": index_fingerprint,
            "progress_tmp": progress_tmp,
            "progress_payload": progress_payload,
            "anchor_digest": anchor_digest,
            "anchor_prefix": anchor_prefix,
        }

    @classmethod
    def _scan_segment_entries(
        cls,
        stream: _CanonicalJsonStream,
        fingerprint: Any,
        on_segment: Any = None,
    ) -> tuple[int, int, str | None]:
        """Stream the segment index's ``segments`` array, validating
        each segment's shape, contiguous coverage from sequence number
        one, bounds and digest evidence, and return the segment count,
        the last covered sequence number and the last segment's trailing
        boundary digest. ``fingerprint`` is a SHA-256 hasher updated
        with each segment's canonical bytes, so the caller can compare
        the segments item for item against the authenticated chain.
        When given, ``on_segment`` is called in order with each
        validated segment's normalized mapping, so a resuming build can
        reuse the already-authenticated complete segments without
        materializing them. Any defect raises :class:`ValueError`, which
        the caller treats as a corrupt (rebuildable) segment index."""
        stream.expect(0x5B)  # '['
        count = 0
        expected_first = 1
        last_seq = 0
        last_digest: str | None = None
        if stream.peek() == 0x5D:  # ']'
            stream.take()
            return count, last_seq, last_digest
        while True:
            stream.expect(0x7B)  # '{'
            fields: dict[str, Any] = {}
            if stream.peek() == 0x7D:  # '}'
                stream.take()
            else:
                while True:
                    if stream.peek() != 0x22:  # '"'
                        raise ValueError("expected a segment key")
                    key = stream.parse_string()
                    if key in fields:
                        raise ValueError(
                            f"duplicate key {key!r} in JSON object"
                        )
                    stream.expect(0x3A)  # ':'
                    if key in ("first", "last"):
                        fields[key] = stream.parse_number()
                    elif key in ("first_digest", "last_digest", "digest"):
                        if stream.peek() != 0x22:  # '"'
                            raise ValueError("expected a string")
                        fields[key] = stream.parse_string()
                    else:
                        stream.skip_value()
                        fields[key] = None
                    byte = stream.take()
                    if byte == 0x2C:  # ','
                        continue
                    if byte == 0x7D:  # '}'
                        break
                    raise ValueError("unexpected content")
            if set(fields.keys()) != set(cls._RECOVERY_CHAIN_SEGMENT_KEYS):
                raise ValueError("bad segment keys")
            first = fields["first"]
            last = fields["last"]
            for name, value in (("first", first), ("last", last)):
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 1
                ):
                    raise ValueError("bad segment bounds")
            if last < first:
                raise ValueError("bad segment bounds")
            if first != expected_first:
                raise ValueError("bad segment continuity")
            digests = {}
            for name in ("first_digest", "last_digest", "digest"):
                value = fields[name]
                if len(value) != 64 or set(value) - cls._HEX_DIGITS:
                    raise ValueError("bad segment digest")
                digests[name] = value
            normalized_segment = {
                "first": first,
                "last": last,
                "first_digest": digests["first_digest"],
                "last_digest": digests["last_digest"],
                "digest": digests["digest"],
            }
            fingerprint.update(
                cls._canonical_json(normalized_segment).encode("utf-8")
            )
            if on_segment is not None:
                on_segment(normalized_segment)
            count += 1
            expected_first = last + 1
            last_seq = last
            last_digest = digests["last_digest"]
            byte = stream.take()
            if byte == 0x2C:  # ','
                continue
            if byte == 0x5D:  # ']'
                break
            raise ValueError("unexpected content")
        return count, last_seq, last_digest

    @classmethod
    def _read_recovery_progress(cls, progress_path: str) -> dict[str, Any] | None:
        """Read and fully validate a resumable-build progress cursor,
        returning its normalized binding or ``None`` when the file is
        missing, unreadable or defective in any way (truncated, wrong
        fields, bad types, mismatched inner shapes).

        Progress is a disposable cache, so no progress problem ever
        raises here: it only marks the call for one full rebuild from
        the canonical chain. The cursor is a fixed-size document and
        never carries an audit body.
        """
        try:
            with open(progress_path, "rb") as fileobj:
                stream = _CanonicalJsonStream(fileobj)
                if stream.peek() is None or stream.peek() == 0xEF:
                    return None
                stream.expect(0x7B)  # '{'
                seen: set[str] = set()
                values: dict[str, Any] = {}
                if stream.peek() == 0x7D:  # '}'
                    stream.take()
                else:
                    while True:
                        if stream.peek() != 0x22:  # '"'
                            raise ValueError("expected an object key")
                        key = stream.parse_string()
                        if key in seen:
                            raise ValueError(
                                f"duplicate key {key!r} in JSON object"
                            )
                        seen.add(key)
                        stream.expect(0x3A)  # ':'
                        if key in ("format", "head", "prefix", "index"):
                            if stream.peek() != 0x22:  # '"'
                                raise ValueError("expected a string")
                            values[key] = stream.parse_string()
                        elif key in (
                            "version",
                            "segment_size",
                            "frames",
                        ):
                            values[key] = stream.parse_number()
                        elif key == "boundary":
                            boundary = cls._read_progress_boundary(stream)
                            values["boundary"] = boundary
                        else:
                            stream.skip_value()
                            values[key] = None
                        byte = stream.take()
                        if byte == 0x2C:  # ','
                            continue
                        if byte == 0x7D:  # '}'
                            break
                        raise ValueError("unexpected content")
                if not stream.at_end():
                    return None
                if seen != set(cls._RECOVERY_CHAIN_PROGRESS_KEYS):
                    return None
                if values["format"] != cls._RECOVERY_CHAIN_PROGRESS_FORMAT:
                    return None
                version = values["version"]
                if (
                    isinstance(version, bool)
                    or not isinstance(version, int)
                    or version
                    not in cls._RECOVERY_CHAIN_PROGRESS_SUPPORTED_VERSIONS
                ):
                    return None
                segment_size = values["segment_size"]
                frames = values["frames"]
                if (
                    isinstance(segment_size, bool)
                    or not isinstance(segment_size, int)
                    or segment_size < 1
                ):
                    return None
                if isinstance(frames, bool) or not isinstance(frames, int):
                    return None
                if frames < 1:
                    return None
                head = values["head"]
                if (
                    not isinstance(head, str)
                    or len(head) != 64
                    or set(head) - cls._HEX_DIGITS
                ):
                    return None
                for name in ("prefix", "index"):
                    digest = values[name]
                    if (
                        not isinstance(digest, str)
                        or len(digest) != 64
                        or set(digest) - cls._HEX_DIGITS
                    ):
                        return None
                boundary = values["boundary"]
                bseq = boundary["seq"]
                if (
                    isinstance(bseq, bool)
                    or not isinstance(bseq, int)
                    or bseq < 1
                    or bseq > frames
                    or bseq % segment_size != 0
                ):
                    return None
                return {
                    "segment_size": segment_size,
                    "frames": frames,
                    "head": head,
                    "boundary": boundary,
                    "prefix": values["prefix"],
                    "index": values["index"],
                }
        except (OSError, ValueError, RecursionError):
            return None

    @classmethod
    def _read_progress_boundary(
        cls, stream: _CanonicalJsonStream
    ) -> dict[str, int | str]:
        """Parse and validate the nested ``boundary`` object of a
        progress cursor: exactly seq/offset/length/digest."""
        stream.expect(0x7B)  # '{'
        fields: dict[str, Any] = {}
        if stream.peek() == 0x7D:  # '}'
            stream.take()
        else:
            while True:
                if stream.peek() != 0x22:  # '"'
                    raise ValueError("expected a boundary key")
                key = stream.parse_string()
                if key in fields:
                    raise ValueError(
                        f"duplicate key {key!r} in JSON object"
                    )
                stream.expect(0x3A)  # ':'
                if key in ("seq", "offset", "length"):
                    fields[key] = stream.parse_number()
                elif key == "digest":
                    if stream.peek() != 0x22:  # '"'
                        raise ValueError("expected a string")
                    fields[key] = stream.parse_string()
                else:
                    stream.skip_value()
                    fields[key] = None
                byte = stream.take()
                if byte == 0x2C:  # ','
                    continue
                if byte == 0x7D:  # '}'
                    break
                raise ValueError("unexpected content")
        if set(fields.keys()) != set(
            cls._RECOVERY_CHAIN_PROGRESS_BOUNDARY_KEYS
        ):
            raise ValueError("bad boundary keys")
        for name in ("seq", "offset", "length"):
            value = fields[name]
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError("bad boundary number")
        if fields["seq"] < 1 or fields["offset"] < 0 or fields["length"] < 1:
            raise ValueError("bad boundary bounds")
        digest = fields["digest"]
        if len(digest) != 64 or set(digest) - cls._HEX_DIGITS:
            raise ValueError("bad boundary digest")
        return {
            "seq": fields["seq"],
            "offset": fields["offset"],
            "length": fields["length"],
            "digest": digest,
        }

    @classmethod
    def _read_recovery_chain_segments(
        cls,
        index_path: str,
        on_segment: Any = None,
    ) -> tuple[int, int, int, str, str] | None:
        """Read and fully validate the segment index, returning its
        binding -- ``(segment_size, chain_version, frames, head,
        segments_fingerprint)`` -- or ``None`` when the index is
        missing, unreadable or defective in any way. The fingerprint is
        the SHA-256 over every segment's canonical bytes in order, so
        the caller can check that each segment's bounds, boundary
        digests and frame-bytes digest match the authenticated chain
        item for item.

        The segment index is a disposable cache, so no index problem
        ever raises here: it only marks the index for a rebuild from the
        canonical chain. Only a fixed amount of memory is used
        regardless of the segment count.
        """
        try:
            with open(index_path, "rb") as fileobj:
                stream = _CanonicalJsonStream(fileobj)
                if stream.peek() is None or stream.peek() == 0xEF:
                    return None
                stream.expect(0x7B)  # '{'
                seen: set[str] = set()
                format_value: Any = None
                version_value: Any = None
                segment_size_value: Any = None
                chain_version: Any = None
                declared_frames: Any = None
                head: Any = None
                count = 0
                last_seq = 0
                last_digest: str | None = None
                fingerprint = hashlib.sha256()
                if stream.peek() == 0x7D:  # '}'
                    stream.take()
                else:
                    while True:
                        if stream.peek() != 0x22:  # '"'
                            raise ValueError("expected an object key")
                        key = stream.parse_string()
                        if key in seen:
                            raise ValueError(
                                f"duplicate key {key!r} in JSON object"
                            )
                        seen.add(key)
                        stream.expect(0x3A)  # ':'
                        if key == "segments":
                            count, last_seq, last_digest = (
                                cls._scan_segment_entries(
                                    stream,
                                    fingerprint,
                                    on_segment=on_segment,
                                )
                            )
                        elif key in ("format", "head"):
                            if stream.peek() != 0x22:  # '"'
                                raise ValueError("expected a string")
                            value = stream.parse_string()
                            if key == "format":
                                format_value = value
                            else:
                                head = value
                        elif key in (
                            "version",
                            "segment_size",
                            "chain_version",
                            "frames",
                        ):
                            value = stream.parse_number()
                            if key == "version":
                                version_value = value
                            elif key == "segment_size":
                                segment_size_value = value
                            elif key == "chain_version":
                                chain_version = value
                            else:
                                declared_frames = value
                        else:
                            stream.skip_value()
                        byte = stream.take()
                        if byte == 0x2C:  # ','
                            continue
                        if byte == 0x7D:  # '}'
                            break
                        raise ValueError("unexpected content")
                if not stream.at_end():
                    return None
                if seen != set(cls._RECOVERY_CHAIN_SEGMENTS_KEYS):
                    return None
                if format_value != cls._RECOVERY_CHAIN_SEGMENTS_FORMAT:
                    return None
                if (
                    isinstance(version_value, bool)
                    or not isinstance(version_value, int)
                    or version_value
                    not in cls._RECOVERY_CHAIN_SEGMENTS_SUPPORTED_VERSIONS
                ):
                    return None
                if (
                    isinstance(segment_size_value, bool)
                    or not isinstance(segment_size_value, int)
                    or segment_size_value < 1
                ):
                    return None
                if (
                    isinstance(chain_version, bool)
                    or not isinstance(chain_version, int)
                    or chain_version
                    not in cls._RECOVERY_CHAIN_SUPPORTED_VERSIONS
                ):
                    return None
                if (
                    isinstance(declared_frames, bool)
                    or not isinstance(declared_frames, int)
                    or declared_frames != last_seq
                    or count < 1
                ):
                    return None
                if (
                    not isinstance(head, str)
                    or len(head) != 64
                    or set(head) - cls._HEX_DIGITS
                    or head != last_digest
                ):
                    return None
                return (
                    segment_size_value,
                    chain_version,
                    declared_frames,
                    head,
                    fingerprint.hexdigest(),
                )
        except (OSError, ValueError, RecursionError):
            return None

    @classmethod
    def _scan_chain_resuming(
        cls,
        fileobj: Any,
        progress: dict[str, Any],
        on_tail_frame: Any,
    ) -> tuple[int, int, str]:
        """Authenticate a chain that only appended frames past a
        recorded segment boundary, parsing no prefix frame.

        The open file is streamed once like
        :meth:`_scan_recovery_chain`, but when the ``frames`` array is
        reached the raw bytes through the recorded boundary are crossed
        with a fixed buffer (fed to a raw-byte hasher), the single
        boundary frame is parsed and re-checked for sequence, span and
        digest, and only frames after it are fully authenticated and
        handed to ``on_tail_frame``. Each handed frame also carries its
        raw ``prefix`` digest -- the SHA-256 of the chain bytes through
        the end of that frame -- so a caller can anchor a new complete
        boundary without a second pass. The boundary raw prefix digest
        must equal the one recorded in progress; a sound-looking chain
        that forks from that prefix raises
        :class:`_RecoveryResumeMismatch`, while a malformed prefix or
        tail raises plain :class:`ValueError`.
        """
        boundary = progress["boundary"]
        stream = _CanonicalJsonStream(fileobj, hashlib.sha256())
        if stream.peek() is None:
            raise ValueError("recovery chain document is empty")
        if stream.peek() == 0xEF:
            bom = bytes((stream.take(), stream.take(), stream.take()))
            if bom == b"\xef\xbb\xbf":
                raise ValueError(
                    "recovery chain must be UTF-8 without a BOM"
                )
            raise ValueError(
                "recovery chain is not valid JSON: unexpected content"
            )
        stream.expect(0x7B)  # '{'
        seen: set[str] = set()
        format_value: Any = None
        version_value: Any = None
        count = boundary["seq"]
        previous_digest = boundary["digest"]
        head = boundary["digest"]
        frames_done = False
        if stream.peek() == 0x7D:  # '}'
            stream.take()
        else:
            while True:
                if stream.peek() != 0x22:  # '"'
                    raise ValueError(
                        "recovery chain is not valid JSON: expected an "
                        "object key"
                    )
                key = stream.parse_string()
                if key in seen:
                    raise ValueError(
                        f"duplicate key {key!r} in JSON object"
                    )
                seen.add(key)
                stream.expect(0x3A)  # ':'
                if key == "frames":
                    stream.expect(0x5B)  # '['
                    frames_start = stream.offset
                    if boundary["offset"] < frames_start:
                        raise _RecoveryResumeMismatch(
                            "boundary frame lies before the frames array"
                        )
                    stream.skip_raw(boundary["offset"] - frames_start)
                    boundary_start = stream.offset
                    fields = cls._parse_chain_frame_stream(stream)
                    boundary_length = stream.offset - boundary_start
                    seq = fields["seq"]
                    if (
                        isinstance(seq, bool)
                        or not isinstance(seq, int)
                        or seq != boundary["seq"]
                        or boundary_length != boundary["length"]
                    ):
                        raise _RecoveryResumeMismatch(
                            "boundary frame does not match progress"
                        )
                    prev = fields["prev"]
                    record = fields["record"]
                    digest = fields["digest"]
                    if len(prev) != 64 or set(prev) - cls._HEX_DIGITS:
                        raise _RecoveryResumeMismatch(
                            "bad boundary predecessor"
                        )
                    if (
                        len(digest) != 64
                        or set(digest) - cls._HEX_DIGITS
                        or not hmac.compare_digest(digest, boundary["digest"])
                        or not hmac.compare_digest(
                            digest,
                            cls._recovery_chain_frame_digest(
                                seq, prev, record
                            ),
                        )
                    ):
                        raise _RecoveryResumeMismatch(
                            "boundary frame digest does not match progress"
                        )
                    if not hmac.compare_digest(
                        stream.consumed_digest(), progress["prefix"]
                    ):
                        raise _RecoveryResumeMismatch(
                            "chain prefix digest does not match progress"
                        )
                    byte = stream.take()
                    if byte == 0x2C:  # ','
                        while True:
                            position = count + 1
                            frame_offset = stream.offset
                            tail_fields = cls._parse_chain_frame_stream(
                                stream
                            )
                            frame_length = stream.offset - frame_offset
                            body, tail_digest = (
                                cls._validate_chain_frame_fields(
                                    tail_fields,
                                    position,
                                    previous_digest,
                                )
                            )
                            count = position
                            previous_digest = tail_digest
                            head = tail_digest
                            on_tail_frame(
                                {
                                    "seq": count,
                                    "prev": tail_fields["prev"],
                                    "record": tail_fields["record"],
                                    "digest": tail_digest,
                                    "body": body,
                                    "offset": frame_offset,
                                    "length": frame_length,
                                    "prefix": stream.consumed_digest(),
                                }
                            )
                            separator = stream.take()
                            if separator == 0x2C:  # ','
                                continue
                            if separator == 0x5D:  # ']'
                                break
                            raise ValueError(
                                "recovery chain is not valid JSON: "
                                "unexpected content"
                            )
                    elif byte == 0x5D:  # ']'
                        pass
                    else:
                        raise ValueError(
                            "recovery chain is not valid JSON: unexpected "
                            "content"
                        )
                    frames_done = True
                elif key == "format":
                    if stream.peek() == 0x22:  # '"'
                        format_value = stream.parse_string()
                    else:
                        stream.skip_value()
                elif key == "version":
                    version_value = stream.parse_number()
                else:
                    stream.skip_value()
                byte = stream.take()
                if byte == 0x2C:  # ','
                    continue
                if byte == 0x7D:  # '}'
                    break
                raise ValueError(
                    "recovery chain is not valid JSON: unexpected content"
                )
        if not stream.at_end():
            raise ValueError(
                "recovery chain is not valid JSON: trailing data"
            )
        if not frames_done or seen != set(cls._RECOVERY_CHAIN_KEYS):
            raise ValueError(
                "recovery chain must contain exactly the keys 'format', "
                "'version' and 'frames'"
            )
        if format_value != cls._RECOVERY_CHAIN_FORMAT:
            raise ValueError("recovery chain has an unknown format")
        if isinstance(version_value, bool) or not isinstance(
            version_value, int
        ):
            raise ValueError("recovery chain 'version' must be an int")
        if version_value not in cls._RECOVERY_CHAIN_SUPPORTED_VERSIONS:
            raise ValueError(
                f"unsupported recovery chain version {version_value!r}"
            )
        if count < 1:
            raise ValueError("recovery chain document has no frames")
        return version_value, count, head

    @classmethod
    def _segmented_chain_scan(
        cls,
        path: str,
        index_path: str,
        segment_size: int,
        wanted: set[int],
        progress_path: str | None = None,
        force_publish: bool = True,
        recovery_path: str | None = None,
    ) -> tuple[int, str, dict[int, dict[str, Any]]]:
        """Authenticate the chain with bounded memory under the chain
        lock, keep the segment index (and, with a progress path, the
        progress cursor) fresh, and return the frame count, the head
        digest and the detached audit bodies of the ``wanted`` sequence
        numbers.

        Without a progress path this is one locked full chain scan plus
        one segment check, exactly as before. With a progress path the
        chain, segment index and progress cursor are all read inside
        this one lock hold and the canonical chain is scanned at most
        once: when the caches are sound and the chain only appended
        frames past the recorded complete-segment boundary, the
        authenticated prefix is crossed with a fixed buffer (its raw
        digest and the boundary frame checked against progress) and
        just the new frames are parsed and folded; anything missing,
        truncated or mismatching falls back to exactly one full rebuild
        from the canonical chain. A readable progress anchor that still
        disagrees with a fully rebuilt chain -- a truncated, rewritten,
        reordered or duplicated chain -- raises :class:`ValueError`;
        caches never answer by themselves and cannot mask a chain
        defect.

        With a ``recovery_path`` an interrupted two-cache publication
        left by a previous process is finished or rolled back inside
        this lock hold before any cache is read, so the scan only ever
        starts from a complete old or complete new version.
        """
        lock_path = cls._recovery_chain_lock_path(path)
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            cls._acquire_chain_lock(fd, None)
            try:
                if recovery_path is not None:
                    cls._recover_cache_publication_locked(
                        index_path, progress_path, recovery_path
                    )
                if progress_path is None and not force_publish:
                    return cls._query_segment_scan_locked(
                        path, index_path, segment_size, wanted
                    )
                if progress_path is None:
                    return cls._full_rebuild_publish_locked(
                        path,
                        index_path,
                        None,
                        segment_size,
                        wanted,
                        anchor=None,
                        force_publish=True,
                    )
                anchor = cls._read_recovery_progress(progress_path)
                boundary_seq = (
                    anchor["boundary"]["seq"] if anchor is not None else 0
                )
                can_continue = anchor is not None and (
                    anchor["segment_size"] == segment_size
                ) and all(w > boundary_seq for w in wanted)
                if can_continue:
                    try:
                        return cls._continue_segment_build_locked(
                            path,
                            index_path,
                            progress_path,
                            anchor,
                            segment_size,
                            wanted,
                            force_publish=force_publish,
                            recovery_path=recovery_path,
                        )
                    except ValueError:
                        # A sound chain that forks from the recorded
                        # prefix, or any stale/defective cache, gets one
                        # full rebuild; a genuinely malformed chain is
                        # rejected by that rebuild. Real I/O failures
                        # propagate as OSError instead.
                        pass
                return cls._full_rebuild_publish_locked(
                    path,
                    index_path,
                    progress_path,
                    segment_size,
                    wanted,
                    anchor=anchor if can_continue else None,
                    force_publish=force_publish,
                    recovery_path=recovery_path,
                )
            finally:
                cls._release_chain_lock(fd)
        finally:
            os.close(fd)

    @classmethod
    def _query_segment_scan_locked(
        cls,
        path: str,
        index_path: str,
        segment_size: int,
        wanted: set[int],
    ) -> tuple[int, str, dict[int, dict[str, Any]]]:
        """The progress-less, read-only-unless-stale batch path: one
        in-memory authenticated chain scan that materializes no temp
        file while the existing index is already fresh, and a single
        rollback-safe rebuild when it is missing, stale or corrupt. The
        caller holds the chain lock."""
        captured: dict[int, dict[str, Any]] = {}
        fingerprint = hashlib.sha256()

        def on_segment(segment: dict[str, Any]) -> None:
            fingerprint.update(
                cls._canonical_json(segment).encode("utf-8")
            )

        segmenter = _ChainSegmenter(
            segment_size, cls._canonical_json, on_segment
        )

        def on_frame(frame: dict[str, Any]) -> None:
            if frame["seq"] in wanted:
                captured[frame["seq"]] = frame["body"]
            segmenter.on_frame(frame)

        chain_file = cls._open_recovery_chain_locked(path)
        try:
            version, count, head = cls._scan_recovery_chain(
                chain_file, on_frame
            )
        finally:
            chain_file.close()
        segmenter.finish()
        expected = (
            segment_size,
            version,
            count,
            head,
            fingerprint.hexdigest(),
        )
        binding = cls._read_recovery_chain_segments(index_path)
        if binding != expected:
            cls._full_rebuild_publish_locked(
                path,
                index_path,
                None,
                segment_size,
                wanted,
                anchor=None,
                force_publish=True,
            )
        return count, head, captured

    @classmethod
    def _full_rebuild_publish_locked(
        cls,
        path: str,
        index_path: str,
        progress_path: str | None,
        segment_size: int,
        wanted: set[int],
        anchor: dict[str, Any] | None,
        force_publish: bool = True,
        recovery_path: str | None = None,
    ) -> tuple[int, str, dict[int, dict[str, Any]]]:
        """One full authenticated chain scan that builds and publishes
        the caches rollback-safely; when ``anchor`` is given it enforces
        that the chain only appended past the anchor's authenticated
        boundary. With ``force_publish`` false a query leaves already
        fresh caches byte-for-byte untouched and only rebuilds (once)
        when one is missing, stale or corrupt. The caller holds the
        chain lock."""
        anchor_boundary = anchor["boundary"]["seq"] if anchor else 0
        result = cls._full_segment_build_locked(
            path,
            index_path,
            progress_path,
            segment_size,
            wanted,
            anchor_boundary_seq=anchor_boundary,
        )
        publications: list[tuple[str, str]] = []
        index_tmp = result["index_tmp"]
        progress_tmp: str | None = result["progress_tmp"]
        try:
            binding = cls._read_recovery_chain_segments(index_tmp)
            if binding != (
                segment_size,
                result["version"],
                result["count"],
                result["head"],
                result["index_fingerprint"],
            ):
                raise ValueError(
                    "recovery chain segment index does not match the "
                    "authenticated chain"
                )
            if anchor is not None:
                if not result["anchor_digest"] or not hmac.compare_digest(
                    result["anchor_digest"], anchor["boundary"]["digest"]
                ) or not hmac.compare_digest(
                    result["anchor_prefix"], anchor["prefix"]
                ):
                    raise ValueError(
                        "recovery chain was truncated, rewritten, "
                        "reordered or duplicated behind the authenticated "
                        "boundary"
                    )
                if result["count"] < anchor["frames"] or (
                    result["count"] == anchor["frames"]
                    and not hmac.compare_digest(
                        result["head"], anchor["head"]
                    )
                ):
                    raise ValueError(
                        "recovery chain was truncated behind the "
                        "authenticated boundary"
                    )
            payload = result["progress_payload"]
            if progress_path is not None and not force_publish:
                # A query must not rewrite caches that are already
                # exactly the freshly authenticated ones.
                existing_index = cls._read_recovery_chain_segments(
                    index_path
                )
                existing_progress = cls._read_recovery_progress(
                    progress_path
                )
                index_fresh = existing_index == binding
                progress_fresh = (
                    progress_tmp is not None
                    and payload is not None
                    and existing_progress == payload
                )
                if index_fresh and progress_fresh:
                    with contextlib.suppress(OSError):
                        os.remove(index_tmp)
                    with contextlib.suppress(OSError):
                        os.remove(progress_tmp)
                    return (
                        result["count"],
                        result["head"],
                        result["captured"],
                    )
            publications.append((index_path, index_tmp))
            if progress_path is not None:
                if progress_tmp is None or payload is None:
                    # No complete segment exists yet; progress cannot
                    # anchor a resume, so publish only the index and
                    # leave any progress decision to a later append.
                    pass
                else:
                    refreshed = cls._read_recovery_progress(progress_tmp)
                    if refreshed != payload:
                        raise ValueError(
                            "recovery chain progress does not match the "
                            "authenticated chain"
                        )
                    publications.append((progress_path, progress_tmp))
            if recovery_path is not None:
                # Every cache publication with a recovery path goes
                # through the cross-process recovery protocol -- an
                # index-only update (no complete segment exists yet, so
                # no progress temp was built) included, with the
                # progress target recorded unchanged. Any earlier
                # two-cache publication was already resolved before the
                # caches were read.
                cls._publish_caches_recovery_locked(
                    index_path,
                    index_tmp,
                    progress_path,
                    progress_tmp,
                    recovery_path,
                )
            else:
                cls._publish_cache_files(publications)
        except BaseException:
            # Remove the temps this call built that never reached a
            # publication slot (a binding or anchor failure raises
            # before the targets are registered).
            published_temps = {tmp for _target, tmp in publications}
            for candidate in (index_tmp, progress_tmp):
                if candidate and candidate not in published_temps:
                    with contextlib.suppress(OSError):
                        os.remove(candidate)
            for _target, tmp_path in publications:
                with contextlib.suppress(OSError):
                    os.remove(tmp_path)
            raise
        return result["count"], result["head"], result["captured"]

    @classmethod
    def _continue_segment_build_locked(
        cls,
        path: str,
        index_path: str,
        progress_path: str,
        anchor: dict[str, Any],
        segment_size: int,
        wanted: set[int],
        force_publish: bool = True,
        recovery_path: str | None = None,
    ) -> tuple[int, str, dict[int, dict[str, Any]]]:
        """Build and publish both caches by appending past the progress
        boundary; raise :class:`ValueError` (typically
        :class:`_RecoveryResumeMismatch`) when the caches or chain do
        not support a continuation, so the caller performs its one full
        rebuild. The caller holds the chain lock and guarantees no
        wanted frame lies in the authenticated prefix. A query
        (``force_publish`` false) that finds the chain unchanged since
        the anchor leaves the already-fresh caches untouched.

        The complete segments already evidenced by the segment index
        are copied byte-for-byte into the new index temp (their
        canonical bytes seed both the all-segment and complete-segment
        fingerprints); the resuming scan parses only frames after the
        boundary and folds them through a fresh segmenter. A fixed
        buffer crosses the raw prefix, so memory stays bounded by the
        largest single frame regardless of the prefix length.
        """
        boundary = anchor["boundary"]
        captured: dict[int, dict[str, Any]] = {}
        all_hasher = hashlib.sha256()
        complete_hasher = hashlib.sha256()
        last_copied_digest = [""]
        handle, index_tmp = cls._new_segment_index_temp(index_path)
        progress_tmp: str | None = None
        try:
            with handle:
                handle.write(cls._segments_envelope_prefix(segment_size))
                copied = [0]

                def copy_complete(segment: dict[str, Any]) -> None:
                    if segment["last"] > boundary["seq"]:
                        return
                    canonical = cls._canonical_json(segment).encode(
                        "utf-8"
                    )
                    all_hasher.update(canonical)
                    complete_hasher.update(canonical)
                    last_copied_digest[0] = segment["last_digest"]
                    if copied[0]:
                        handle.write(b",")
                    handle.write(canonical)
                    copied[0] += 1

                index_binding = cls._read_recovery_chain_segments(
                    index_path, on_segment=copy_complete
                )
                if index_binding is None:
                    raise _RecoveryResumeMismatch(
                        "segment index is missing or corrupt"
                    )
                index_size = index_binding[0]
                if index_size != segment_size:
                    raise _RecoveryResumeMismatch(
                        "segment index size does not match progress"
                    )
                if copied[0] * segment_size != boundary["seq"]:
                    raise _RecoveryResumeMismatch(
                        "segment index does not cover the boundary"
                    )
                # The complete-segment prefix must be exactly the
                # evidence the progress cursor fingerprinted off the
                # authenticated chain, and its last segment must end on
                # the recorded boundary frame. A stale (or even
                # partially advanced) trailing segment is irrelevant --
                # it is rebuilt from the authenticated tail.
                if not hmac.compare_digest(
                    complete_hasher.hexdigest(), anchor["index"]
                ) or not hmac.compare_digest(
                    last_copied_digest[0], boundary["digest"]
                ):
                    raise _RecoveryResumeMismatch(
                        "segment index prefix does not match progress"
                    )

                new_boundary = {
                    "seq": boundary["seq"],
                    "offset": boundary["offset"],
                    "length": boundary["length"],
                    "digest": boundary["digest"],
                    "prefix": anchor["prefix"],
                }
                emitted = [copied[0] > 0]
                # Info for the most recently parsed tail frame and for
                # the frame before it. A segment is emitted either at
                # the start of the frame following its last frame (its
                # boundary info is then ``previous``) or by finish() for
                # the trailing segment (its boundary is ``current``).
                context = {"previous": [0, 0, ""], "current": [0, 0, ""]}
                finishing = [False]

                def emit_tail(segment: dict[str, Any]) -> None:
                    canonical = cls._canonical_json(segment).encode(
                        "utf-8"
                    )
                    all_hasher.update(canonical)
                    if segment["last"] % segment_size == 0:
                        complete_hasher.update(canonical)
                        ctx = (
                            context["current"]
                            if finishing[0]
                            else context["previous"]
                        )
                        new_boundary["seq"] = segment["last"]
                        new_boundary["digest"] = segment["last_digest"]
                        new_boundary["offset"] = ctx[0]
                        new_boundary["length"] = ctx[1]
                        new_boundary["prefix"] = ctx[2]
                    if emitted[0]:
                        handle.write(b",")
                    handle.write(canonical)
                    emitted[0] = True

                segmenter = _ChainSegmenter(
                    segment_size, cls._canonical_json, emit_tail
                )

                def on_tail_frame(frame: dict[str, Any]) -> None:
                    if frame["seq"] in wanted:
                        captured[frame["seq"]] = frame["body"]
                    context["previous"] = context["current"]
                    context["current"] = [
                        frame["offset"],
                        frame["length"],
                        frame["prefix"],
                    ]
                    segmenter.on_frame(frame)

                chain_file = cls._open_recovery_chain_locked(path)
                try:
                    version, count, head = cls._scan_chain_resuming(
                        chain_file, anchor, on_tail_frame
                    )
                finally:
                    chain_file.close()
                finishing[0] = True
                segmenter.finish()
                handle.write(cls._segments_trailer(version, count, head))
                handle.flush()
                os.fsync(handle.fileno())
            if not emitted[0]:
                raise _RecoveryResumeMismatch(
                    "the segment index held no segments to continue"
                )
            binding = cls._read_recovery_chain_segments(index_tmp)
            if binding != (
                segment_size,
                version,
                count,
                head,
                all_hasher.hexdigest(),
            ):
                raise ValueError(
                    "recovery chain segment index does not match the "
                    "authenticated chain"
                )
            payload: dict[str, Any] | None = None
            if new_boundary["seq"] >= 1:
                payload = {
                    "segment_size": segment_size,
                    "frames": count,
                    "head": head,
                    "boundary": {
                        "seq": new_boundary["seq"],
                        "offset": new_boundary["offset"],
                        "length": new_boundary["length"],
                        "digest": new_boundary["digest"],
                    },
                    "prefix": new_boundary["prefix"],
                    "index": complete_hasher.hexdigest(),
                }
                progress_handle, progress_tmp = cls._new_progress_temp(
                    progress_path
                )
                with progress_handle:
                    progress_handle.write(
                        cls._progress_document_bytes(payload)
                    )
                    progress_handle.flush()
                    os.fsync(progress_handle.fileno())
                refreshed = cls._read_recovery_progress(progress_tmp)
                if refreshed != payload:
                    raise ValueError(
                        "recovery chain progress does not match the "
                        "authenticated chain"
                    )
            publications = [(index_path, index_tmp)]
            if progress_tmp is not None:
                publications.append((progress_path, progress_tmp))
            if not force_publish:
                existing_index = cls._read_recovery_chain_segments(
                    index_path
                )
                existing_progress = (
                    cls._read_recovery_progress(progress_path)
                    if progress_tmp is not None
                    else None
                )
                if existing_index == binding and (
                    progress_tmp is None
                    or existing_progress == payload
                ):
                    with contextlib.suppress(OSError):
                        os.remove(index_tmp)
                    if progress_tmp is not None:
                        with contextlib.suppress(OSError):
                            os.remove(progress_tmp)
                    return count, head, captured
            if recovery_path is not None:
                cls._publish_caches_recovery_locked(
                    index_path,
                    index_tmp,
                    progress_path,
                    progress_tmp,
                    recovery_path,
                )
            else:
                cls._publish_cache_files(publications)
        except BaseException:
            with contextlib.suppress(OSError):
                os.remove(index_tmp)
            if progress_tmp is not None:
                with contextlib.suppress(OSError):
                    os.remove(progress_tmp)
            raise
        return count, head, captured

    @classmethod
    def build_recovery_audit_segments(
        cls,
        path: Any,
        index_path: Any,
        segment_size: Any,
        progress_path: Any = None,
        recovery_path: Any = None,
    ) -> str:
        """Build the disposable bounded-memory segmented index over a
        persistent recovery-audit chain and return the authenticated
        chain head digest.

        ``path`` and ``index_path`` must be non-empty :class:`str`
        values (a non-``str`` raises :class:`TypeError`, an empty string
        :class:`ValueError`), and ``index_path`` must not name the chain
        file itself, even through an alias (:class:`ValueError`).
        ``segment_size`` must be a non-``bool`` :class:`int` (anything
        else raises :class:`TypeError`) of at least one (a smaller value
        raises :class:`ValueError`). ``progress_path`` is optional:
        ``None`` (the default) keeps the call, result and authentication
        semantics exactly as before; otherwise it must be a non-empty
        :class:`str` (a non-``str`` raises :class:`TypeError`, an empty
        string :class:`ValueError`) naming neither the chain nor the
        index, even through an alias (:class:`ValueError`). A missing or
        unreadable chain file, a missing index/progress directory or any
        failed open, flush, replace or sync raises :class:`OSError` and
        publishes nothing.

        ``recovery_path`` is optional and validated after all the
        arguments above: ``None`` (the default) keeps every prior
        behavior; otherwise it must be a non-empty :class:`str` (a
        non-``str`` raises :class:`TypeError`, an empty string
        :class:`ValueError`) used together with a ``progress_path`` --
        omitting progress raises :class:`ValueError` -- and must name
        neither the chain, index nor progress file (even through an
        alias), while the three cache files must share one directory
        (otherwise :class:`ValueError`). Given a recovery path, every
        call first finishes an earlier process's interrupted
        publication: when the index and progress both carry the
        recorded new version and the recorded phase allows committing,
        the new version is completed and the old backups are removed,
        and every other reachable state is rolled back to the old
        version byte-for-byte (restoring absence when the old target did
        not exist); only then does authentication, a resuming build or a
        query begin. With no record present the two caches must form one
        complete version -- a mixed index/progress version no valid
        record explains raises :class:`ValueError`, as does a record
        phase that contradicts the target digests. An unparseable,
        malformed or checksum-bad recovery record, a target matching
        neither recorded digest or a missing or corrupt required backup
        raises :class:`ValueError` with all targets, backups and the
        record preserved as evidence; a missing parent directory or a
        failed open, read, write, flush, replace, delete or directory
        sync raises :class:`OSError`.

        The build competes with appends for the same chain lock and
        re-checks the chain file's identity inside the lock, so it only
        ever observes the complete chain from before or after an
        append. While holding the lock it streams the chain and fully
        authenticates it -- encoding, canonical form, duplicate keys,
        frame order, predecessor links, embedded record checksums and
        frame digests -- without ever materializing all frames or audit
        bodies at once; any chain defect raises :class:`ValueError` and
        publishes nothing. Consecutive frames are folded into segments
        of ``segment_size`` frames (the last segment may be short);
        every segment records its first and last sequence numbers, the
        boundary frame digests and the SHA-256 of the segment's
        canonical frame bytes. The document binds the chain's schema
        version, frame count, head digest and segment size; its bytes
        are UTF-8 without a BOM, compact JSON without a trailing
        newline, and identical for identical chains and segment sizes.

        With a ``progress_path`` the build is resumable: the chain,
        segment index and progress cursor are read inside the one chain
        lock and the canonical chain is scanned at most once. A sound
        progress cursor that matches the index and only lags an
        appended chain lets the build cross the already authenticated
        prefix with a fixed buffer (checking its raw digest and the
        boundary frame) and parse just the new frames, then complete the
        trailing segment; a missing, truncated or mismatching cursor or
        index triggers exactly one full rebuild from the canonical
        chain, and a cursor that still disagrees with a rebuilt chain
        -- a truncated, rewritten, reordered or duplicated chain --
        raises :class:`ValueError`. The compact progress document binds
        the segment size, authenticated boundary, frame count, head,
        raw prefix digest and complete-segment index digest, and never
        stores an audit body.

        Both caches are written to temp files in their target
        directories, flushed and fsync-ed, then published under a
        rollback-safe protocol that restores any replaced target
        byte-for-byte if a later step fails; concurrent readers see
        only a complete old or new cache. With a ``recovery_path`` the
        two targets are published under the chain lock with durable
        backups of both old targets made first, a checksummed recovery
        record (version, phase, each target's old/new existence and
        SHA-256 digest and the backup names) made durable before the
        first replacement and updated and directory-synced after each
        replacement, and the backups and record removed only after the
        complete new version is on disk; a process dying at any point
        therefore resumes across processes as described above. A
        crashed or failed build never rewrites the chain file and
        removes its temp and backup files. Whether the build succeeds
        or fails, nothing under the generation directory -- business
        state, audits, idempotency records or the journal attachment --
        is modified.
        """
        cls._validate_chain_path(path)
        cls._validate_index_path(index_path)
        cls._validate_index_target(path, index_path)
        cls._validate_segment_size(segment_size)
        cls._validate_progress_path(progress_path)
        if progress_path is not None:
            cls._validate_progress_target(path, index_path, progress_path)
        cls._validate_recovery_path(recovery_path)
        if recovery_path is not None:
            cls._validate_recovery_target(
                path, index_path, progress_path, recovery_path
            )
        _count, head, _captured = cls._segmented_chain_scan(
            path,
            index_path,
            segment_size,
            set(),
            progress_path,
            recovery_path=recovery_path,
        )
        return head

    @classmethod
    def diff_recovery_audit_ranges(
        cls,
        path: Any,
        expected_head: Any,
        ranges: Any,
        index_path: Any,
        segment_size: Any,
        progress_path: Any = None,
        recovery_path: Any = None,
    ) -> tuple[dict[str, Any], ...]:
        """Compare the audit records sealed at the endpoints of many
        chain ranges in one batch, strictly read-only.

        ``path`` and ``expected_head`` are authenticated exactly as for
        :meth:`diff_recovery_audit_range`: an invalid path raises
        :class:`TypeError`/:class:`ValueError`, a missing or unreadable
        file :class:`OSError`, any chain defect :class:`ValueError`, and
        a sound chain whose last frame does not match ``expected_head``
        raises :class:`ValueError` rather than reporting diffs over
        unauthenticated material.

        ``ranges`` must be a :class:`tuple` whose every element is a
        ``(start, end)`` :class:`tuple` of two non-``bool`` integers; a
        wrong container or element shape raises :class:`TypeError`, as
        does a non-integer endpoint. An endpoint below one, a start
        later than its end, a duplicated range or an end past the
        chain's last frame raises :class:`ValueError`. ``index_path``
        and ``segment_size`` are validated exactly as for
        :meth:`build_recovery_audit_segments`. ``progress_path`` is
        optional and validated in the existing argument order: ``None``
        (the default) keeps the call, result and full authentication
        semantics exactly as before; a non-``str`` raises
        :class:`TypeError`, an empty string :class:`ValueError`, and a
        path naming the chain or index (even through an alias) raises
        :class:`ValueError`. ``recovery_path`` is optional and validated
        last, exactly as for :meth:`build_recovery_audit_segments`:
        ``None`` keeps every prior behavior, a non-``str`` raises
        :class:`TypeError`, an empty string :class:`ValueError`, and the
        path may only be used together with a ``progress_path`` and only
        when it aliases none of the chain, index or progress files and
        the three cache files share one directory (otherwise
        :class:`ValueError`). Given a recovery path the batch first
        finishes or rolls back an earlier process's interrupted
        publication under the chain lock; success still returns exactly
        the per-range dictionaries described below.

        The whole batch performs at most one locked canonical-chain
        scan and one segment check: the chain is authenticated under
        the same chain lock that serializes appends, only the queried
        endpoint records are materialized, and the segment index at
        ``index_path`` is checked segment by segment against the
        authenticated chain and rebuilt once from it when missing,
        stale, truncated or corrupt (an index that still does not match
        after the rebuild raises :class:`ValueError`; a failed rebuild
        propagates :class:`OSError`). With a ``progress_path`` a sound
        cursor that only lags an appended chain lets the batch cross
        the authenticated prefix with a fixed buffer and parse just
        the new frames (at most one full rebuild on any cache
        divergence, and a readable cursor that disagrees with a rebuilt
        chain raises :class:`ValueError`); a queried endpoint inside
        the recorded prefix forces the full authenticated scan so the
        answer never comes from a cache. Memory grows only with the
        largest single frame and fixed buffers -- never with the total
        frame count or a range span. An empty ``ranges`` still
        completes the path, head, segment-size, progress, recovery,
        chain and segment-index authentication and then returns an
        empty tuple.

        The result is a tuple of fresh dictionaries, one per input range
        in input order, each carrying ``start``, ``end`` and ``changes``
        -- where ``changes`` is item-for-item identical to the
        :meth:`diff_recovery_audit_range` result for the same endpoints
        (an empty tuple when ``start`` equals ``end``). Nothing on disk
        or any business, audit or idempotency state is modified, and the
        returned dictionaries are detached from the parsed chain and
        from each other.
        """
        cls._validate_chain_path(path)
        cls._validate_head_digest(expected_head, allow_none=False)
        if not isinstance(ranges, tuple):
            raise TypeError(
                f"ranges must be a tuple, got {type(ranges).__name__}"
            )
        normalized: list[tuple[int, int]] = []
        seen_ranges: set[tuple[int, int]] = set()
        for item in ranges:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError(
                    "each range must be a (start, end) tuple"
                )
            start, end = item
            for name, value in (("start", start), ("end", end)):
                if isinstance(value, bool) or not isinstance(value, int):
                    raise TypeError(
                        f"{name} must be an int, got "
                        f"{type(value).__name__}"
                    )
            if start < 1 or end < 1:
                raise ValueError(
                    "range endpoints must be greater than or equal to 1"
                )
            if start > end:
                raise ValueError(
                    "range start must not be later than end"
                )
            if (start, end) in seen_ranges:
                raise ValueError(
                    f"duplicate range ({start}, {end})"
                )
            seen_ranges.add((start, end))
            normalized.append((start, end))
        cls._validate_index_path(index_path)
        cls._validate_index_target(path, index_path)
        cls._validate_segment_size(segment_size)
        cls._validate_progress_path(progress_path)
        if progress_path is not None:
            cls._validate_progress_target(path, index_path, progress_path)
        cls._validate_recovery_path(recovery_path)
        if recovery_path is not None:
            cls._validate_recovery_target(
                path, index_path, progress_path, recovery_path
            )
        wanted: set[int] = set()
        for start, end in normalized:
            wanted.add(start)
            wanted.add(end)
        last, head, captured = cls._segmented_chain_scan(
            path,
            index_path,
            segment_size,
            wanted,
            progress_path,
            force_publish=False,
            recovery_path=recovery_path,
        )
        if not hmac.compare_digest(head, expected_head):
            raise ValueError(
                "recovery chain head does not match the expected head "
                "digest"
            )
        results: list[dict[str, Any]] = []
        for start, end in normalized:
            if end > last:
                raise ValueError(
                    f"range end {end} is past the last frame {last}"
                )
            if start == end:
                changes: tuple[tuple[Any, ...], ...] = ()
            else:
                changes = cls._diff_recovery_bodies(
                    captured[start], captured[end]
                )
            results.append(
                {"start": start, "end": end, "changes": changes}
            )
        return tuple(results)

    @classmethod
    def _diff_recovery_bodies(
        cls, before: dict[str, Any], after: dict[str, Any]
    ) -> tuple[tuple[Any, ...], ...]:
        """Compute the ordered change tuples between a sealed audit
        body and the body of freshly gathered evidence. Both bodies
        must share the same record version."""
        changes: list[tuple[Any, ...]] = []
        if before["current"] != after["current"]:
            changes.append(
                (
                    "pointer",
                    None,
                    "changed",
                    before["current"],
                    after["current"],
                )
            )
        if before["selected"] != after["selected"]:
            changes.append(
                (
                    "chain",
                    None,
                    "changed",
                    before["selected"],
                    after["selected"],
                )
            )
        before_gens = {
            entry["name"]: entry for entry in before["generations"]
        }
        after_gens = {
            entry["name"]: entry for entry in after["generations"]
        }
        for name in sorted(
            set(before_gens) | set(after_gens),
            key=cls._generation_number,
        ):
            sealed = before_gens.get(name)
            fresh = after_gens.get(name)
            cls._append_recovery_diff(
                changes,
                "chain",
                name,
                cls._audit_chain_view(sealed),
                cls._audit_chain_view(fresh),
            )
            for category in ("checkpoint", "journal"):
                cls._append_recovery_diff(
                    changes,
                    category,
                    name,
                    sealed["files"][category] if sealed else None,
                    fresh["files"][category] if fresh else None,
                )
            cls._append_recovery_diff(
                changes,
                "replay",
                name,
                sealed["replay"] if sealed else None,
                fresh["replay"] if fresh else None,
            )
        return tuple(changes)

    @staticmethod
    def _audit_chain_view(
        entry: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """The chain evidence compared per generation: the dependency
        verdict, manifest digest, validity and ignore reason. ``None``
        for a generation absent on that side."""
        if entry is None:
            return None
        return {
            "valid": entry["valid"],
            "dependency": entry["dependency"],
            "manifest": entry["manifest"],
            "reason": entry["reason"],
        }

    @staticmethod
    def _append_recovery_diff(
        changes: list[tuple[Any, ...]],
        category: str,
        generation: str | None,
        before: Any,
        after: Any,
    ) -> None:
        """Append one change tuple unless both sides agree."""
        if before == after:
            return
        if before is None:
            change = "added"
        elif after is None:
            change = "removed"
        else:
            change = "changed"
        changes.append((category, generation, change, before, after))

    @classmethod
    def _parse_recovery_audit_record(cls, record: Any) -> dict[str, Any]:
        """Validate a sealed recovery-audit record and return its body.

        A non-``str`` raises :class:`TypeError`; an empty string,
        unparsable JSON, duplicate JSON keys, a non-canonical encoding,
        a structural or version violation, or a checksum that does not
        protect the record raises :class:`ValueError`. Both supported
        record versions are accepted. The returned body is the envelope
        without its ``checksum`` key, exactly as sealed.
        """
        if not isinstance(record, str):
            raise TypeError(
                f"record must be a str, got {type(record).__name__}"
            )
        if not record:
            raise ValueError("record must be a non-empty str")
        try:
            document = json.loads(
                record,
                object_pairs_hook=cls._reject_duplicate_json_keys,
            )
        except ValueError as exc:
            raise ValueError(f"record is not valid JSON: {exc}") from exc
        cls._validate_recovery_audit_document(document)
        if cls._canonical_json(document) != record:
            raise ValueError(
                "record is not canonical compact JSON"
            )
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
        expected_checksum = hashlib.sha256(
            cls._canonical_json(body).encode("utf-8")
        ).hexdigest()
        if not hmac.compare_digest(
            document["checksum"], expected_checksum
        ):
            raise ValueError("record checksum is invalid")
        return body

    @classmethod
    def _validate_recovery_audit_document(cls, document: Any) -> None:
        """Validate a parsed recovery-audit envelope's structure and
        field domains. Any deviation raises :class:`ValueError`."""
        if not isinstance(document, dict):
            raise ValueError(
                "audit record top-level JSON value must be an object"
            )
        if set(document.keys()) != set(cls._RECOVERY_AUDIT_KEYS):
            raise ValueError(
                "audit record must contain exactly the keys 'format', "
                "'version', 'current', 'selected', 'generations', "
                "'ignored' and 'checksum'"
            )
        if document["format"] != cls._RECOVERY_AUDIT_FORMAT:
            raise ValueError("audit record has an unknown format")
        version = document["version"]
        if isinstance(version, bool) or not isinstance(version, int):
            raise ValueError("audit record 'version' must be an int")
        if version not in cls._RECOVERY_AUDIT_SUPPORTED_VERSIONS:
            raise ValueError(
                f"unsupported audit record version {version!r}"
            )
        cls._validate_audit_pointer(document["current"], version)
        selected = document["selected"]
        if selected is not None and not cls._is_legal_generation_name(
            selected
        ):
            raise ValueError(
                "audit record 'selected' must be null or a legal "
                "generation name"
            )
        if not isinstance(document["generations"], list):
            raise ValueError("audit record 'generations' must be an array")
        for entry in document["generations"]:
            cls._validate_audit_generation(entry)
        if not isinstance(document["ignored"], list):
            raise ValueError("audit record 'ignored' must be an array")
        for entry in document["ignored"]:
            if not isinstance(entry, dict) or set(entry.keys()) != {
                "name",
                "reason",
            }:
                raise ValueError(
                    "each ignored entry must be an object with exactly "
                    "'name' and 'reason'"
                )
            if not isinstance(entry["name"], str) or not (
                cls._is_generation_entry_name(entry["name"])
            ):
                raise ValueError(
                    "each ignored entry's 'name' must match the "
                    "generation directory name pattern"
                )
            if not isinstance(entry["reason"], str) or not entry["reason"]:
                raise ValueError(
                    "each ignored entry's 'reason' must be a non-empty "
                    "string"
                )
        checksum = document["checksum"]
        if not isinstance(checksum, str) or len(checksum) != 64 or (
            set(checksum) - cls._HEX_DIGITS
        ):
            raise ValueError(
                "audit record 'checksum' must be a 64-character "
                "lowercase hex string"
            )

    @classmethod
    def _is_legal_generation_name(cls, value: str) -> bool:
        """Whether ``value`` is exactly a positive numbered generation
        directory name."""
        number = cls._generation_number(value)
        return number is not None and number >= 1

    @classmethod
    def _is_generation_entry_name(cls, value: str) -> bool:
        """Whether ``value`` matches the generation directory pattern,
        including the zero-numbered forensic name."""
        return cls._generation_number(value) is not None

    @classmethod
    def _validate_audit_pointer(cls, value: Any, version: int) -> None:
        expected_keys = {"state", "value"}
        if version >= 2:
            expected_keys = {"state", "value", "digest"}
        if not isinstance(value, dict) or set(value.keys()) != (
            expected_keys
        ):
            keys_description = (
                "'state', 'value' and 'digest'"
                if version >= 2
                else "'state' and 'value'"
            )
            raise ValueError(
                "audit record 'current' must be an object with exactly "
                + keys_description
            )
        state = value["state"]
        if state not in cls._POINTER_STATES:
            raise ValueError(
                "audit record pointer state must be one of 'missing', "
                "'ok' or 'corrupt'"
            )
        pointer_value = value["value"]
        if state == "ok":
            if not isinstance(pointer_value, str) or not (
                cls._is_legal_generation_name(pointer_value)
            ):
                raise ValueError(
                    "an 'ok' pointer value must be a legal generation "
                    "name"
                )
        elif pointer_value is not None:
            raise ValueError(
                "a missing or corrupt pointer value must be null"
            )
        if version >= 2:
            digest = value["digest"]
            if state == "corrupt":
                if (
                    not isinstance(digest, str)
                    or len(digest) != 64
                    or set(digest) - cls._HEX_DIGITS
                ):
                    raise ValueError(
                        "a corrupt pointer's 'digest' must be a "
                        "64-character lowercase hex string"
                    )
            elif digest is not None:
                raise ValueError(
                    "a missing or ok pointer's 'digest' must be null"
                )

    @classmethod
    def _validate_audit_generation(cls, entry: Any) -> None:
        if not isinstance(entry, dict) or set(entry.keys()) != {
            "name",
            "number",
            "valid",
            "dependency",
            "previous",
            "manifest",
            "files",
            "replay",
            "reason",
        }:
            raise ValueError(
                "each generation record must contain exactly the keys "
                "'name', 'number', 'valid', 'dependency', 'previous', "
                "'manifest', 'files', 'replay' and 'reason'"
            )
        name = entry["name"]
        if not isinstance(name, str) or not (
            cls._is_generation_entry_name(name)
        ):
            raise ValueError(
                "each generation record's 'name' must match the "
                "generation directory name pattern"
            )
        number = entry["number"]
        if isinstance(number, bool) or not isinstance(number, int):
            raise ValueError(
                "each generation record's 'number' must be an int"
            )
        if number < 0 or number != cls._generation_number(name):
            raise ValueError(
                "each generation record's 'number' must match its name"
            )
        if not isinstance(entry["valid"], bool):
            raise ValueError(
                "each generation record's 'valid' must be a bool"
            )
        if entry["dependency"] not in cls._DEPENDENCY_STATES:
            raise ValueError(
                "each generation record's 'dependency' must be one of "
                "'root', 'linked', 'broken' or 'invalid'"
            )
        for field in ("previous", "manifest"):
            value = entry[field]
            if value is not None and (
                not isinstance(value, str)
                or len(value) != 64
                or set(value) - cls._HEX_DIGITS
            ):
                raise ValueError(
                    f"each generation record's {field!r} must be null or "
                    "a 64-character lowercase hex string"
                )
        files = entry["files"]
        if not isinstance(files, dict) or set(files.keys()) != {
            "checkpoint",
            "journal",
        }:
            raise ValueError(
                "each generation record's 'files' must be an object "
                "with exactly 'checkpoint' and 'journal'"
            )
        for field in ("checkpoint", "journal"):
            value = files[field]
            if value is not None and (
                not isinstance(value, str)
                or len(value) != 64
                or set(value) - cls._HEX_DIGITS
            ):
                raise ValueError(
                    f"each generation record's file digest {field!r} "
                    "must be null or a 64-character lowercase hex string"
                )
        if not isinstance(entry["replay"], str) or not entry["replay"]:
            raise ValueError(
                "each generation record's 'replay' must be a non-empty "
                "string"
            )
        reason = entry["reason"]
        if reason is not None and (
            not isinstance(reason, str) or not reason
        ):
            raise ValueError(
                "each generation record's 'reason' must be null or a "
                "non-empty string"
            )


    @classmethod
    def _parse_generation_manifest(
        cls, data: bytes, name: str, number: int
    ) -> tuple[str | None, dict[str, Any] | None]:
        """Validate a generation manifest's encoding, JSON and structure.

        Returns ``(None, manifest)`` on success, or ``(reason, None)``
        naming the first structural defect found."""
        if data.startswith(b"\xef\xbb\xbf"):
            return "manifest must be UTF-8 without a BOM", None
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            return "manifest is not valid UTF-8", None
        try:
            document = json.loads(
                text, object_pairs_hook=cls._reject_duplicate_json_keys
            )
        except ValueError as exc:
            return f"manifest is not valid JSON: {exc}", None
        if not isinstance(document, dict):
            return "manifest top-level JSON value must be an object", None
        if set(document.keys()) != set(cls._GENERATION_MANIFEST_KEYS):
            return (
                "manifest must contain exactly the keys 'schema_version', "
                "'generation', 'number', 'previous', 'checkpoint', "
                "'journal' and 'complete'"
            ), None
        version = document["schema_version"]
        if isinstance(version, bool) or not isinstance(version, int):
            return "manifest 'schema_version' must be an int", None
        if version != cls._GENERATION_MANIFEST_VERSION:
            return (
                f"unsupported manifest schema_version {version!r}"
            ), None
        if document["generation"] != name:
            return (
                "manifest 'generation' does not name its directory"
            ), None
        num = document["number"]
        if isinstance(num, bool) or not isinstance(num, int):
            return "manifest 'number' must be an int", None
        if num != number:
            return (
                "manifest 'number' does not match its directory name"
            ), None
        previous = document["previous"]
        if previous is not None and (
            not isinstance(previous, str)
            or len(previous) != 64
            or set(previous) - cls._HEX_DIGITS
        ):
            return (
                "manifest 'previous' must be null or a 64-character "
                "lowercase hex string"
            ), None
        for field in ("checkpoint", "journal"):
            value = document[field]
            if not isinstance(value, str) or len(value) != 64 or (
                set(value) - cls._HEX_DIGITS
            ):
                return (
                    f"manifest {field!r} must be a 64-character "
                    "lowercase hex string"
                ), None
        return None, document

    def audit_merge(self, id: str) -> dict[str, object]:
        """Return an audit record for the merge event ``id``.

        The record is a fresh dict whose keys are ordered
        ``event_id, target, source, parents, at, changes``: the merge
        event id, the target and source branch names, the recorded
        parent ids (target head first), the non-negative timestamp and
        a copy of the merged changes. Unknown ids and ids of non-merge
        events raise :class:`KeyError`; mutating the returned dict or
        its ``changes`` never affects the store's records.
        """
        self._require_nonempty_str(id, "id")
        recorded = self._merges.get(id)
        if recorded is None:
            raise KeyError(id)
        target, source, at_value, changes = recorded
        return {
            "event_id": id,
            "target": target,
            "source": source,
            "parents": self._graph._parents[id],
            "at": at_value,
            "changes": dict(changes),
        }

    def audit_log(self) -> tuple[dict[str, object], ...]:
        """Return audit records for every event this store recorded.

        Covers exactly the branch appends and merges performed through
        this store -- events added directly to the graph are not
        included -- each exactly once, ordered by ``(at, event_id)``
        ascending (event ids compared by Unicode code point). Each
        record is a fresh dict whose keys are ordered
        ``event_id, kind, owner, peer, parents, at, changes``: for an
        append, ``owner`` is the branch name, ``peer`` is ``None`` and
        ``parents`` is the single-element parent id tuple; for a merge,
        ``owner`` is the target branch, ``peer`` is the source branch
        and ``parents`` is the two-element tuple with the target parent
        first. ``changes`` is a copy ordered by Unicode code point of
        its keys. The query is read-only: the graph, branch heads and
        records are never modified, and the returned tuple, dicts and
        changes are detached from internal state, so mutating them
        affects neither the store nor later calls, and repeated calls
        return item-wise equal results in a stable order.
        """
        records: list[dict[str, object]] = []
        for name, appends in self._appends.items():
            for event_id, (at_value, changes) in appends.items():
                records.append(
                    {
                        "event_id": event_id,
                        "kind": "append",
                        "owner": name,
                        "peer": None,
                        "parents": self._graph._parents[event_id],
                        "at": at_value,
                        "changes": {
                            key: changes[key] for key in sorted(changes)
                        },
                    }
                )
        for event_id, (target, source, at_value, changes) in (
            self._merges.items()
        ):
            records.append(
                {
                    "event_id": event_id,
                    "kind": "merge",
                    "owner": target,
                    "peer": source,
                    "parents": self._graph._parents[event_id],
                    "at": at_value,
                    "changes": {
                        key: changes[key] for key in sorted(changes)
                    },
                }
            )
        records.sort(
            key=lambda record: (record["at"], record["event_id"])
        )
        return tuple(records)

    def trace(self, name: str, key: str) -> tuple[str, ...]:
        """Return ids of events on ``name``'s head closure touching ``key``.

        Ids follow the graph's replay order and include zero-delta and
        merge events, each at most once.
        """
        self._require_nonempty_str(name, "name")
        self._require_nonempty_str(key, "key")
        self._require_known_branch(name)
        return self._graph.explain(self._heads[name], key)

    def explain_change(
        self, name: str, key: str
    ) -> tuple[dict[str, object], ...]:
        """Explain the causal value trajectory of ``key`` on a branch.

        The branch's current head ancestor closure is replayed in the
        graph's parents-before-children, ``(at, id)`` order. Every event
        whose changes contain ``key`` -- including zero-delta and merge
        events -- appears exactly once, with the key's integer value
        accumulated from an initial ``0``. Each record is a fresh dict
        whose keys are ordered ``event_id, at, parents, delta, before,
        after``: the event id, its non-negative timestamp, a copy of its
        recorded parent ids (for a merge, target head first, source head
        second), the integer delta the event applies to ``key`` and the
        values immediately before and after applying it. ``name`` and
        ``key`` are validated (in that order) before the branch is looked
        up: non-``str`` values raise :class:`TypeError`, empty strings
        raise :class:`ValueError`, unknown branches raise
        :class:`KeyError`. The query is read-only and its result is
        detached from internal state, so the graph, branch heads, audit
        and idempotency records are never modified, and repeated calls
        return item-wise equal results.
        """
        self._require_nonempty_str(name, "name")
        self._require_nonempty_str(key, "key")
        self._require_known_branch(name)

        order = self._graph._ordered_ancestors(self._heads[name])
        records: list[dict[str, object]] = []
        value = 0
        for event_id in order:
            changes = self._graph._changes[event_id]
            if key not in changes:
                continue
            delta = changes[key]
            before = value
            value += delta
            records.append(
                {
                    "event_id": event_id,
                    "at": self._graph._at[event_id],
                    "parents": (*self._graph._parents[event_id],),
                    "delta": delta,
                    "before": before,
                    "after": value,
                }
            )
        return tuple(records)

    def explain_impact(
        self, name: str, event_id: str
    ) -> tuple[dict[str, object], ...]:
        """Explain the forward causal impact of an event on a branch.

        Within the ancestor closure of the branch's current head, edges
        point parent-to-child as the impact direction. The strict
        descendants of ``event_id`` -- reachable events other than
        ``event_id`` itself -- are returned each exactly once, in the
        graph's parents-before-children, ``(at, id)`` order. Each record
        is a fresh dict whose keys are ordered ``event_id, at, path,
        keys``: the descendant id, its non-negative timestamp, the tuple
        of ids on the shortest path from ``event_id`` to it (both ends
        included), and the tuple of change keys its ``changes`` touch,
        sorted by Unicode code point (an empty tuple for events with no
        changes). When several paths exist, the one with the fewest edges
        wins, ties broken by the Unicode code point order of the complete
        id tuples; no descendants yields an empty tuple. ``name`` and
        ``event_id`` are validated (in that order) before the branch is
        looked up: non-``str`` values raise :class:`TypeError`, empty
        strings raise :class:`ValueError`, unknown branches raise
        :class:`KeyError`, and an ``event_id`` absent from the graph or
        outside the branch's current head ancestor closure raises
        :class:`KeyError`. The query is read-only and its result is
        detached from internal state, so the graph, branch heads, audit
        and idempotency records are never modified, and repeated calls
        return item-wise equal results.
        """
        self._require_nonempty_str(name, "name")
        self._require_nonempty_str(event_id, "event_id")
        self._require_known_branch(name)
        if event_id not in self._graph._at:
            raise KeyError(event_id)

        head = self._heads[name]
        order = self._graph._ordered_ancestors(head)
        closure = set(order)
        if event_id not in closure:
            raise KeyError(event_id)

        # Reverse the parent edges within the closure so children can be
        # enumerated parent-to-child.
        children: dict[str, list[str]] = {event: [] for event in closure}
        for current in closure:
            for parent in self._graph._parents[current]:
                children[parent].append(current)

        # Level-by-level breadth-first search gives the fewest-edge
        # paths. Within a level every candidate path has equal length, so
        # keeping the minimum full id tuple applies the Unicode code
        # point tie-break; children are claimed only after the whole
        # level is evaluated, so every shortest candidate is compared.
        paths: dict[str, tuple[str, ...]] = {event_id: (event_id,)}
        frontier = [event_id]
        while frontier:
            candidates: dict[str, tuple[str, ...]] = {}
            for current in frontier:
                current_path = paths[current]
                for child in children[current]:
                    if child in paths:
                        continue
                    candidate = (*current_path, child)
                    best = candidates.get(child)
                    if best is None or candidate < best:
                        candidates[child] = candidate
            for child, path in candidates.items():
                paths[child] = path
            frontier = list(candidates)

        # Descendants are reported in the closure's existing replay
        # order: parents before children, ready events by (at, id).
        records: list[dict[str, object]] = []
        for descendant in order:
            if descendant == event_id or descendant not in paths:
                continue
            changes = self._graph._changes[descendant]
            records.append(
                {
                    "event_id": descendant,
                    "at": self._graph._at[descendant],
                    "path": (*paths[descendant],),
                    "keys": tuple(sorted(changes)),
                }
            )
        return tuple(records)

    def diff_at(self, name_a: str, name_b: str, at: int) -> dict[str, tuple[int, int]]:
        """Diff two branches' current heads as of ``at``.

        Both names and ``at`` are validated (in signature order) before
        either branch is looked up; the diff itself is delegated to
        :meth:`EventGraph.diff_at` on the branches' current heads. The
        query is read-only: the graph, branch heads and records are
        never modified, and the returned dict and tuples are detached
        from internal state.
        """
        self._require_nonempty_str(name_a, "name_a")
        self._require_nonempty_str(name_b, "name_b")
        at_value = EventGraph._require_int(at, "at")
        if at_value < 0:
            raise ValueError("at must be a non-negative int")
        self._require_known_branch(name_a)
        self._require_known_branch(name_b)
        return self._graph.diff_at(
            self._heads[name_a], self._heads[name_b], at_value
        )

    def compare(self, name_a: str, name_b: str) -> dict[str, object]:
        """Compare the ancestor closures of two branches' current heads.

        Both names are validated (in signature order) before either
        branch is looked up: non-``str`` names raise :class:`TypeError`,
        empty names raise :class:`ValueError`, unknown branches raise
        :class:`KeyError`, and comparing a branch with itself raises
        :class:`ValueError`. Each head's ancestor closure is taken once,
        in the graph's parents-before-children, ``(at, id)`` replay
        order. The result is a fresh dict whose keys are ordered
        ``common, left_only, right_only, fork``: tuples of event ids for
        the intersection, the left-only and the right-only events
        (common and left-only in left replay order, right-only in right
        replay order), plus ``fork`` -- the last common id in left replay
        order, or ``None`` when the closures share no event. The query
        is read-only: the graph, branch heads and records are never
        modified, and the returned dict and tuples are detached from
        internal state.
        """
        self._require_nonempty_str(name_a, "name_a")
        self._require_nonempty_str(name_b, "name_b")
        self._require_known_branch(name_a)
        self._require_known_branch(name_b)
        if name_a == name_b:
            raise ValueError(f"cannot compare branch {name_a!r} with itself")

        left_order = self._graph._ordered_ancestors(self._heads[name_a])
        right_order = self._graph._ordered_ancestors(self._heads[name_b])
        left_ids = set(left_order)
        right_ids = set(right_order)

        common = tuple(
            event_id for event_id in left_order if event_id in right_ids
        )
        left_only = tuple(
            event_id for event_id in left_order if event_id not in right_ids
        )
        right_only = tuple(
            event_id for event_id in right_order if event_id not in left_ids
        )
        return {
            "common": common,
            "left_only": left_only,
            "right_only": right_only,
            "fork": common[-1] if common else None,
        }

    def attribute_divergence(
        self, name_a: str, name_b: str, key: str
    ) -> dict[str, object]:
        """Attribute the cross-branch divergence of ``key`` to its causes.

        Both names and ``key`` are validated (in signature order) before
        either branch is looked up: non-``str`` values raise
        :class:`TypeError`, empty strings raise :class:`ValueError`,
        unknown branches raise :class:`KeyError` (``name_a`` first), and
        naming the same branch twice raises :class:`ValueError`. Each
        branch's current head ancestor closure is replayed in the graph's
        parents-before-children, ``(at, id)`` order; a side on which
        ``key`` never appears contributes the value ``0``.

        The result is a fresh dict whose keys are ordered
        ``key, fork, left, right``: ``key`` is the queried key as given
        and ``fork`` is the last common event id in left replay order (or
        ``None`` when the closures share no event). ``left`` and ``right``
        are fresh dicts whose keys are ordered ``value, cause, path,
        affected``: the replayed integer value, the cause event id, and
        data derived from it. When the two values differ, a side's cause
        is the first event exclusive to that side whose changes contain
        ``key`` (zero deltas included) in that side's replay order, or
        ``None`` when no such event exists; when the values are equal,
        both causes are ``None``. ``path`` is the tuple of ids on the
        shortest parent-to-child path from the cause to that side's
        current head (both ends included), ties broken by the Unicode
        code point order of the complete id tuples, and ``affected`` is
        the tuple of the cause's strict descendant ids in that side's
        replay order; both are empty tuples when the cause is ``None``.
        The query is read-only: the graph, branch heads and records are
        never modified, and the returned dict and tuples are detached
        from internal state.
        """
        self._require_nonempty_str(name_a, "name_a")
        self._require_nonempty_str(name_b, "name_b")
        self._require_nonempty_str(key, "key")
        self._require_known_branch(name_a)
        self._require_known_branch(name_b)
        if name_a == name_b:
            raise ValueError(
                f"cannot attribute divergence for branch {name_a!r} "
                "against itself"
            )

        return self._attribute_divergence_for(
            self._heads[name_a], self._heads[name_b], key
        )

    def attribute_divergence_at(
        self,
        name_a: str,
        node_a: str,
        name_b: str,
        node_b: str,
        key: str,
    ) -> dict[str, object]:
        """Attribute the divergence of ``key`` at two historical nodes.

        This is the historical-node form of :meth:`attribute_divergence`:
        instead of each branch's current head, the left side replays the
        ancestor closure of ``node_a`` on branch ``name_a`` and the right
        side that of ``node_b`` on branch ``name_b``. All five parameters
        are validated in signature order (``name_a``, ``node_a``,
        ``name_b``, ``node_b``, ``key``) before either node is replayed:
        non-``str`` values raise :class:`TypeError`, empty strings raise
        :class:`ValueError`; the branches are then looked up in order
        (``name_a`` first), an unknown branch raises :class:`KeyError`,
        and naming the same branch twice raises :class:`ValueError`; a
        node absent from the graph, or outside its named branch's current
        head ancestor closure (``node_a`` checked before ``node_b``),
        raises :class:`KeyError`.

        Each node's ancestor closure is replayed in the graph's
        parents-before-children, ``(at, id)`` order; a side on which
        ``key`` never appears contributes the value ``0``. The result is
        a fresh dict whose keys are ordered ``key, fork, left, right``:
        ``key`` is the queried key as given, ``fork`` is the last common
        event id in left replay order (or ``None`` when the closures
        share no event), and ``left``/``right`` are fresh dicts whose
        keys are ordered ``value, cause, path, affected``. When the two
        values differ, a side's cause is the first event exclusive to
        that side whose changes contain ``key`` (zero deltas included)
        in that side's replay order, or ``None`` when no such event
        exists; when the values are equal, both causes are ``None``.
        ``path`` is the tuple of ids on the shortest parent-to-child
        path from the cause to that side's node (both ends included),
        ties broken by the Unicode code point order of the complete id
        tuples, and ``affected`` is the tuple of the cause's strict
        descendant ids within that side's closure in its replay order;
        both are empty tuples when the cause is ``None``. The query is
        read-only: success or failure never modifies the graph, branch
        heads, audit or idempotency records, and the returned dict and
        tuples are detached from internal state.
        """
        self._require_nonempty_str(name_a, "name_a")
        self._require_nonempty_str(node_a, "node_a")
        self._require_nonempty_str(name_b, "name_b")
        self._require_nonempty_str(node_b, "node_b")
        self._require_nonempty_str(key, "key")

        self._require_known_branch(name_a)
        self._require_known_branch(name_b)
        if name_a == name_b:
            raise ValueError(
                f"cannot attribute divergence for branch {name_a!r} "
                "against itself"
            )

        if node_a not in self._graph._at:
            raise KeyError(node_a)
        if node_a not in set(
            self._graph._ordered_ancestors(self._heads[name_a])
        ):
            raise KeyError(node_a)
        if node_b not in self._graph._at:
            raise KeyError(node_b)
        if node_b not in set(
            self._graph._ordered_ancestors(self._heads[name_b])
        ):
            raise KeyError(node_b)

        return self._attribute_divergence_for(node_a, node_b, key)

    def attribute_divergences_at(
        self,
        name_a: str,
        node_a: str,
        name_b: str,
        node_b: str,
        keys: tuple[str, ...],
    ) -> tuple[dict[str, object], ...]:
        """Attribute divergences for several keys at two historical nodes.

        The multi-key form of :meth:`attribute_divergence_at`: all five
        parameters are validated before either node is replayed. The four
        string parameters are checked in signature order (``name_a``,
        ``node_a``, ``name_b``, ``node_b``): non-``str`` values raise
        :class:`TypeError` and empty strings raise :class:`ValueError`.
        ``keys`` must be a tuple (else :class:`TypeError`); its elements
        are checked in input order, and a non-``str`` element raises
        :class:`TypeError`, while an empty or duplicated key raises
        :class:`ValueError`. The branches are then looked up in order
        (``name_a`` first): an unknown branch raises :class:`KeyError` and
        naming the same branch twice raises :class:`ValueError`; a node
        absent from the graph, or outside its named branch's current head
        ancestor closure (``node_a`` checked before ``node_b``), raises
        :class:`KeyError`. Branch and node checks run even when ``keys``
        is empty, in which case an empty tuple is returned.

        Every key is answered from one read-only historical view: both
        nodes' ancestor closures are taken once before results are built.
        The returned tuple has one fresh dict per key, ordered by the
        Unicode code point of the keys (independent of their input
        order); item-wise, each dict equals the result of
        :meth:`attribute_divergence_at` with the same first four
        arguments and that key, preserving its key order, types and full
        semantics. The dicts and tuples neither share objects with each
        other nor alias internal state. Success or failure never modifies
        the graph, branch heads, audit or idempotency records.
        """
        self._require_nonempty_str(name_a, "name_a")
        self._require_nonempty_str(node_a, "node_a")
        self._require_nonempty_str(name_b, "name_b")
        self._require_nonempty_str(node_b, "node_b")
        if not isinstance(keys, tuple):
            raise TypeError(
                f"keys must be a tuple, got {type(keys).__name__}"
            )
        seen_keys: set[str] = set()
        for key in keys:
            self._require_nonempty_str(key, "key")
            if key in seen_keys:
                raise ValueError(f"duplicate key {key!r}")
            seen_keys.add(key)

        self._require_known_branch(name_a)
        self._require_known_branch(name_b)
        if name_a == name_b:
            raise ValueError(
                f"cannot attribute divergence for branch {name_a!r} "
                "against itself"
            )

        # Capture the read-only historical view before any node is
        # replayed: the branches' current head closures decide node
        # membership, and the nodes' own closures answer every key.
        head_closure_a = set(
            self._graph._ordered_ancestors(self._heads[name_a])
        )
        head_closure_b = set(
            self._graph._ordered_ancestors(self._heads[name_b])
        )
        if node_a not in self._graph._at or node_a not in head_closure_a:
            raise KeyError(node_a)
        if node_b not in self._graph._at or node_b not in head_closure_b:
            raise KeyError(node_b)

        if not keys:
            return ()

        left_order = self._graph._ordered_ancestors(node_a)
        right_order = self._graph._ordered_ancestors(node_b)
        return tuple(
            self._attribute_divergence_on_orders(
                left_order, right_order, node_a, node_b, key
            )
            for key in sorted(keys)
        )

    def divergence_timeline(
        self,
        name_a: str,
        name_b: str,
        points: tuple[tuple[str, str], ...],
        keys: tuple[str, ...],
    ) -> tuple[tuple[dict[str, object], ...], ...]:
        """Attribute divergences for several node pairs and keys at once.

        The multi-point form of :meth:`attribute_divergences_at`:
        ``points`` is a tuple of ``(node_a, node_b)`` pairs, each played
        against the same two branches -- the left nodes live on
        ``name_a``'s current head ancestor closure and the right nodes on
        ``name_b``'s.

        All four parameters are validated in signature order
        (``name_a``, ``name_b``, ``points``, ``keys``); containers are
        walked in input order, and all of this finishes before any state
        is consulted. The names raise :class:`TypeError` when not
        ``str`` and :class:`ValueError` when empty. ``points`` and
        ``keys`` must be tuples (else :class:`TypeError`); a point that
        is not a length-2 tuple raises :class:`TypeError`, and within
        each point the left node is checked before the right one:
        non-``str`` nodes and keys raise :class:`TypeError`, empty
        strings raise :class:`ValueError`, and a repeated node pair or
        key (in input order) raises :class:`ValueError`.

        The branches are then looked up in order (``name_a`` first): an
        unknown branch raises :class:`KeyError` and naming the same
        branch twice raises :class:`ValueError`. Nodes are checked per
        pair, left before right, pairs in ``points`` order: a node
        absent from the graph or outside its named branch's current
        head ancestor closure raises :class:`KeyError`. With empty
        ``points`` the names, keys and branches are still validated and
        an empty tuple is returned; with empty ``keys`` every node is
        still validated and each point contributes an empty tuple.

        Every point and key is answered from one read-only historical
        view: both branches' current head closures and every node's own
        closure are taken once before results are built. The outer
        tuple follows ``points`` in its original order; each point
        contributes a tuple of one fresh dict per key, ordered by the
        Unicode code point of the keys (independent of their input
        order), item-wise equal to calling
        :meth:`attribute_divergences_at` with that node pair and
        ``keys`` -- preserving its dict key order, types and full
        attribution semantics. The tuples and dicts at every level are
        freshly built, neither sharing objects with each other nor
        aliasing internal state. Success or failure never modifies the
        graph, branch heads, audit or idempotency records.
        """
        # --- Parameters validated in signature order, containers in
        # input order, before any state is consulted. ---
        ordered_points, keys = self._divergence_inputs(
            name_a, name_b, points, keys
        )
        if ordered_points is None:
            return ()

        return tuple(
            tuple(
                self._attribute_divergence_on_orders(
                    left_order, right_order, node_a, node_b, key
                )
                for key in sorted(keys)
            )
            for left_order, right_order, node_a, node_b in ordered_points
        )

    def divergence_summary(
        self,
        name_a: str,
        name_b: str,
        points: tuple[tuple[str, str], ...],
        keys: tuple[str, ...],
    ) -> tuple[dict[str, object], ...]:
        """Summarize how each key diverges and converges across node pairs.

        Shares :meth:`divergence_timeline`'s parameters, validation order
        and errors, empty-container handling and branch/node closure rules
        exactly: the four parameters are validated in signature order
        (``name_a``, ``name_b``, ``points``, ``keys``), containers walked
        in input order, before any state is consulted; non-``str`` values
        raise :class:`TypeError`, empty strings :class:`ValueError`,
        ``points``/``keys`` that are not tuples or points that are not
        length-2 tuples raise :class:`TypeError`, and a repeated node pair
        or key raises :class:`ValueError`. Branches are then looked up in
        order (unknown branch :class:`KeyError`, same branch twice
        :class:`ValueError`), and nodes are checked per pair, left before
        right, pairs in ``points`` order (:class:`KeyError` when absent
        from the graph or the named branch's current head closure).

        Every point and key is answered from the same single read-only
        historical view as :meth:`divergence_timeline`: both branches'
        current head closures and every node's own closure are taken once
        before results are built. Points are indexed from ``0`` in their
        given order. For each key (ordered by Unicode code point,
        independent of input order) the result is a fresh dict whose keys
        are ordered ``key, first_diverged, transitions, last``:

        ``first_diverged`` is the index of the first point whose left and
        right replayed values for the key differ, or ``None`` when they are
        equal at every point. ``transitions`` lists, in point-index order,
        every state change between adjacent points (plus divergence at the
        first point): each item is a fresh dict whose keys are ordered
        ``index, kind, left_value, right_value, left_cause, right_cause``.
        ``kind`` is ``diverged`` when the first point already diverges or
        equality becomes divergence, ``converged`` when divergence becomes
        equality, and ``reattributed`` when adjacent points both diverge
        but either attributed cause changes; adjacent points with no such
        change record nothing. The values and causes are those of the
        attribution at ``index``. ``last`` is a detached deep copy of the
        complete attribution dict for the key at the final point (the
        ``key, fork, left, right`` structure of
        :meth:`attribute_divergences_at`), or ``None`` when ``points`` is
        empty.

        Empty ``points`` still validates names, keys and branches and
        yields an empty tuple; empty ``keys`` still validates every node
        and yields an empty tuple. The tuples and dicts at every level are
        freshly built, neither sharing objects with each other nor aliasing
        internal state. Success or failure never modifies the graph,
        branch heads, audit or idempotency records.
        """
        ordered_points, keys = self._divergence_inputs(
            name_a, name_b, points, keys
        )
        if ordered_points is None:
            return ()
        return self._divergence_summaries(ordered_points, keys)

    def divergence_matrix(
        self,
        reference: str,
        series: dict[str, tuple[tuple[str, str], ...]],
        keys: tuple[str, ...],
    ) -> tuple[dict[str, object], ...]:
        """Run :meth:`divergence_summary` for several branches at once.

        Each entry of ``series`` pairs a branch name with its own tuple of
        ``(reference node, series node)`` points, every point replayed
        against the same ``reference`` branch -- the left node must live on
        the reference branch's current head ancestor closure and the right
        node on the named series branch's.

        ``reference`` is validated first: a non-``str`` value raises
        :class:`TypeError` and an empty string :class:`ValueError`.
        ``series`` must be a dict (else :class:`TypeError`); its keys are
        checked in insertion order, a non-``str`` series name raising
        :class:`TypeError` and an empty or ``reference``-equal name
        raising :class:`ValueError`, and each series' points are walked in
        input order. ``keys`` is checked last, exactly as in
        :meth:`divergence_summary`: it must be a tuple (else
        :class:`TypeError`), a non-``str`` element raises :class:`TypeError`,
        and an empty or duplicated key raises :class:`ValueError`. The rest
        of the validation, errors, empty-container handling and
        branch/node closure rules are exactly
        :meth:`divergence_summary`'s for the corresponding
        ``(reference, series_name, points, keys)`` call: points must be a
        tuple of length-2 node-id tuples with no repeated pair, and all
        inputs finish validating before the branches are looked up,
        reference first and then the series in ``series`` insertion order
        (unknown branch :class:`KeyError`, a node absent from the graph or
        the named branch's current head closure :class:`KeyError`, left
        before right, pairs in points order).

        Every series, point and key is answered from one shared read-only
        historical view: the reference branch's and every series branch's
        current head closures and every node's own closure are taken once
        before results are built. The returned tuple follows ``series`` in
        its insertion order; each entry is a fresh dict whose keys are
        ordered ``branch, summaries, reconverged``: ``branch`` is the
        series branch name, ``summaries`` is the tuple of fresh summary
        dicts item-wise equal to :meth:`divergence_summary` for that series
        (keyed and ordered as there), and ``reconverged`` is a tuple of
        booleans aligned with ``summaries`` (keys in Unicode code point
        order) -- each is ``True`` exactly when the summary's
        ``first_diverged`` is not ``None`` and the two sides' values in its
        ``last`` attribution are equal. An empty ``series`` still
        validates ``reference`` and ``keys`` and looks the reference branch
        up, yielding an empty tuple; empty ``keys`` still validates every
        node, and each series then contributes empty
        ``summaries``/``reconverged`` tuples. The tuples and dicts at every
        level are freshly built, neither sharing objects with each other
        nor aliasing internal state. Success or failure never modifies the
        graph, branch heads, audit or idempotency records.
        """
        # --- All inputs finish validating before any branch is looked up,
        # mirroring divergence_summary's reference, branch, points, keys
        # order with series walked in insertion order. ---
        self._require_nonempty_str(reference, "reference")
        if not isinstance(series, dict):
            raise TypeError(
                f"series must be a dict, got {type(series).__name__}"
            )
        per_series: list[tuple[str, tuple[tuple[str, str], ...]]] = []
        for series_name, series_points in series.items():
            self._require_nonempty_str(series_name, "series branch")
            if series_name == reference:
                raise ValueError(
                    f"series branch {series_name!r} cannot equal reference "
                    f"branch {reference!r}"
                )
            self._validate_points(series_points)
            per_series.append((series_name, series_points))
        ordered_keys = self._validate_keys(keys)

        self._require_known_branch(reference)

        # Capture the shared read-only view before any result is built:
        # the reference head closure serves every series, and each series
        # branch's own closure plus its nodes' closures are taken once.
        reference_head_closure = set(
            self._graph._ordered_ancestors(self._heads[reference])
        )
        views: list[
            tuple[str, list[tuple[list[str], list[str], str, str]] | None]
        ] = []
        for series_name, series_points in per_series:
            self._require_known_branch(series_name)
            head_closure_b = set(
                self._graph._ordered_ancestors(self._heads[series_name])
            )
            for node_a, node_b in series_points:
                if (
                    node_a not in self._graph._at
                    or node_a not in reference_head_closure
                ):
                    raise KeyError(node_a)
                if (
                    node_b not in self._graph._at
                    or node_b not in head_closure_b
                ):
                    raise KeyError(node_b)
            ordered_points: list[
                tuple[list[str], list[str], str, str]
            ] | None = None
            if series_points:
                ordered_points = [
                    (
                        self._graph._ordered_ancestors(node_a),
                        self._graph._ordered_ancestors(node_b),
                        node_a,
                        node_b,
                    )
                    for node_a, node_b in series_points
                ]
            views.append((series_name, ordered_points))

        rows: list[dict[str, object]] = []
        for series_name, ordered_points in views:
            summaries = self._divergence_summaries(
                ordered_points, ordered_keys
            )
            reconverged = tuple(
                summary["first_diverged"] is not None
                and summary["last"]["left"]["value"]
                == summary["last"]["right"]["value"]
                for summary in summaries
            )
            rows.append(
                {
                    "branch": series_name,
                    "summaries": summaries,
                    "reconverged": reconverged,
                }
            )
        return tuple(rows)

    def rank_impacts(
        self,
        reference: str,
        series: dict[str, tuple[tuple[str, str], ...]],
        weights: dict[str, int | float],
    ) -> tuple[dict[str, object], ...]:
        """Rank branches by the weighted impact of their unconverged keys.

        The scheduling-impact companion of :meth:`divergence_matrix`: each
        entry of ``series`` pairs a branch name with its own tuple of
        ``(reference node, series node)`` points, replayed against the same
        ``reference`` branch exactly as there, and ``weights`` maps each
        examined state key to its non-negative weight.

        ``reference`` and ``series`` are validated first, in exactly
        :meth:`divergence_matrix`'s order and with its errors: a
        non-``str`` or empty ``reference`` raises :class:`TypeError`/
        :class:`ValueError`; ``series`` must be a dict (else
        :class:`TypeError`); its keys are checked in insertion order, a
        non-``str`` series name raising :class:`TypeError` and an empty or
        ``reference``-equal name raising :class:`ValueError`; and each
        series' points are walked in input order (a non-tuple container or
        non-length-2-tuple point raises :class:`TypeError`, non-``str`` or
        empty node ids raise :class:`TypeError`/:class:`ValueError`, a
        repeated node pair raises :class:`ValueError`). ``weights`` is
        checked last, before any branch is looked up: it must be a dict
        (else :class:`TypeError`); its entries are walked in insertion
        order, a non-``str`` key raising :class:`TypeError` and an empty
        key raising :class:`ValueError`; each weight must be a non-``bool``
        :class:`int` or :class:`float` (anything else raises
        :class:`TypeError`), and a negative, NaN or infinite weight raises
        :class:`ValueError`. The branches are then looked up, reference
        first and then the series in ``series`` insertion order (unknown
        branch :class:`KeyError`), and nodes are checked per pair, left
        before right, pairs in points order (:class:`KeyError` when absent
        from the graph or the named branch's current head closure).

        Every series, point and key is answered from one shared read-only
        historical view: the reference branch's and every series branch's
        current head closures and every node's own closure are taken once
        before results are built. For each series branch the examined
        keys are the ``weights`` keys, ordered by Unicode code point; their
        per-point divergences and final attributions are item-wise equal
        to :meth:`divergence_summary`'s for the same arguments. Each branch
        contributes a fresh dict whose keys are ordered ``branch, score,
        first_impact, unconverged, attributions``:

        ``score`` is the sum, over the keys whose left and right replayed
        values still differ at the final point, of the absolute value
        difference times the key's weight -- plain Python arithmetic
        yielding an :class:`int` or :class:`float` with no rounding, and a
        zero result is never negative zero. ``first_impact`` is the
        smallest ``first_diverged`` point index among the keys that
        diverged at least once, or ``None`` when no examined key ever
        diverged. ``unconverged`` is the tuple of the still-diverging keys
        ordered by Unicode code point, and ``attributions`` aligns with it:
        one fresh ``key, fork, left, right`` attribution dict per
        unconverged key, item-wise equal to that key's ``last`` attribution
        in :meth:`divergence_matrix`.

        The rows are sorted by descending ``score``, then ascending
        ``first_impact`` with ``None`` placed after every integer, then by
        the Unicode code point of the branch name. An empty ``series``
        still validates ``reference`` and ``weights`` and looks the
        reference branch up, yielding an empty tuple; empty ``weights``
        still validates every branch and node, and each series then
        contributes a zero score, ``None`` and empty ``unconverged`` and
        ``attributions`` tuples. The tuples and dicts at every level are
        freshly built, neither sharing objects with each other nor aliasing
        internal state. Success or failure never modifies the graph,
        branch heads, audit or idempotency records.
        """
        # --- All inputs finish validating before any branch is looked up,
        # mirroring divergence_matrix with weights in the keys position. ---
        self._require_nonempty_str(reference, "reference")
        if not isinstance(series, dict):
            raise TypeError(
                f"series must be a dict, got {type(series).__name__}"
            )
        per_series: list[tuple[str, tuple[tuple[str, str], ...]]] = []
        for series_name, series_points in series.items():
            self._require_nonempty_str(series_name, "series branch")
            if series_name == reference:
                raise ValueError(
                    f"series branch {series_name!r} cannot equal reference "
                    f"branch {reference!r}"
                )
            self._validate_points(series_points)
            per_series.append((series_name, series_points))
        ordered_keys = self._validate_weights(weights)

        self._require_known_branch(reference)

        # Capture the shared read-only view before any result is built,
        # exactly as in divergence_matrix.
        reference_head_closure = set(
            self._graph._ordered_ancestors(self._heads[reference])
        )
        views: list[
            tuple[str, list[tuple[list[str], list[str], str, str]] | None]
        ] = []
        for series_name, series_points in per_series:
            self._require_known_branch(series_name)
            head_closure_b = set(
                self._graph._ordered_ancestors(self._heads[series_name])
            )
            for node_a, node_b in series_points:
                if (
                    node_a not in self._graph._at
                    or node_a not in reference_head_closure
                ):
                    raise KeyError(node_a)
                if (
                    node_b not in self._graph._at
                    or node_b not in head_closure_b
                ):
                    raise KeyError(node_b)
            ordered_points: list[
                tuple[list[str], list[str], str, str]
            ] | None = None
            if series_points:
                ordered_points = [
                    (
                        self._graph._ordered_ancestors(node_a),
                        self._graph._ordered_ancestors(node_b),
                        node_a,
                        node_b,
                    )
                    for node_a, node_b in series_points
                ]
            views.append((series_name, ordered_points))

        rows: list[dict[str, object]] = []
        for series_name, ordered_points in views:
            summaries = self._divergence_summaries(
                ordered_points, ordered_keys
            )
            unconverged: list[str] = []
            attributions: list[dict[str, object]] = []
            diverged_indices: list[int] = []
            score: int | float = 0
            for summary in summaries:
                first_diverged = summary["first_diverged"]
                if first_diverged is not None:
                    diverged_indices.append(first_diverged)
                last = summary["last"]
                left_value = last["left"]["value"]
                right_value = last["right"]["value"]
                if left_value == right_value:
                    continue
                key = summary["key"]
                score += abs(left_value - right_value) * weights[key]
                unconverged.append(key)
                attributions.append(last)
            # A zero score must never surface as negative zero.
            if score == 0:
                score = 0 if isinstance(score, int) else 0.0
            rows.append(
                {
                    "branch": series_name,
                    "score": score,
                    "first_impact": (
                        min(diverged_indices) if diverged_indices else None
                    ),
                    "unconverged": tuple(unconverged),
                    "attributions": tuple(attributions),
                }
            )
        rows.sort(
            key=lambda row: (
                -row["score"],
                row["first_impact"] is None,
                (
                    row["first_impact"]
                    if row["first_impact"] is not None
                    else 0
                ),
                row["branch"],
            )
        )
        return tuple(rows)

    def impact_budget(
        self,
        reference: str,
        series: dict[str, tuple[tuple[str, str], ...]],
        weights: dict[str, int | float],
        total_budget: int | float,
        key_budgets: dict[str, int | float],
    ) -> tuple[dict[str, object], ...]:
        """Judge each branch's weighted risk against operating budgets.

        The read-only budget companion of :meth:`rank_impacts`: each entry
        of ``series`` pairs a branch name with its own tuple of
        ``(reference node, series node)`` checkpoints, replayed against the
        same ``reference`` branch exactly as there, ``weights`` maps each
        examined state key to its non-negative weight, ``total_budget`` is
        the one total-risk ceiling, and ``key_budgets`` maps a subset of
        the weight keys to per-key absolute-difference ceilings.

        ``reference``, ``series`` and ``weights`` are validated first, in
        exactly :meth:`rank_impacts`' order and with its errors: a
        non-``str`` or empty ``reference`` raises :class:`TypeError`/
        :class:`ValueError`; ``series`` must be a dict (else
        :class:`TypeError`); its keys are checked in insertion order, a
        non-``str`` series name raising :class:`TypeError` and an empty or
        ``reference``-equal name raising :class:`ValueError`; points are
        walked in input order (a non-tuple container or non-length-2-tuple
        point raises :class:`TypeError`, non-``str`` or empty node ids
        raise :class:`TypeError`/:class:`ValueError`, a repeated node pair
        raises :class:`ValueError`); ``weights`` must be a dict of
        non-empty ``str`` keys to non-negative, finite non-``bool``
        numbers. ``total_budget`` is checked next, then ``key_budgets``,
        all before any branch is looked up: each must be a non-``bool``
        :class:`int` or :class:`float` (anything else, including
        :class:`bool`, raises :class:`TypeError`), negative, NaN or
        infinite values raise :class:`ValueError`. ``key_budgets`` must be
        a dict (else :class:`TypeError`); a non-``str`` key raises
        :class:`TypeError` and an empty key :class:`ValueError`, and a key
        absent from ``weights`` raises :class:`ValueError`; omitting a
        weight key simply leaves that key without a per-key ceiling. The
        branches are then looked up, reference first and then the series
        in ``series`` insertion order (unknown branch :class:`KeyError`),
        and nodes are checked per pair, left before right, pairs in points
        order (:class:`KeyError` when absent from the graph or the named
        branch's current head closure).

        Every series, checkpoint and key is answered from one shared
        read-only historical view: the reference branch's and every series
        branch's current head closures and every node's own closure are
        taken once before results are built. At each checkpoint the total
        risk is the sum, over the weight keys, of the absolute difference
        between the two sides' replayed values times the key's weight --
        plain Python arithmetic yielding an :class:`int` or
        :class:`float`, never rounded and never negative zero; a side that
        never introduced a key contributes ``0``. A checkpoint breaches
        when its total risk is strictly greater than ``total_budget``, or
        when any configured key's absolute difference is strictly greater
        than that key's per-key budget.

        Each branch contributes a fresh dict whose keys are ordered
        ``branch, breached, first_breach, max_overrun, remaining,
        over_keys, attributions``: ``breached`` says whether any
        checkpoint ever breached; ``first_breach`` is the index of the
        first checkpoint that did, or ``None`` when none did;
        ``max_overrun`` is the largest breach magnitude seen -- the
        maximum, over every checkpoint, of the total risk above
        ``total_budget`` and of each breaching key's excess difference
        multiplied by that key's weight -- or ``0`` when nothing ever
        breached; ``remaining`` is ``total_budget`` minus the final
        checkpoint's risk (plain Python arithmetic, may be negative);
        ``over_keys`` is the tuple of keys still over their per-key
        budget at the final checkpoint, ordered by Unicode code point;
        and ``attributions`` aligns one-to-one with it, one fresh
        ``key, fork, left, right`` attribution dict per such key,
        item-wise equal to the existing divergence attribution at that
        same final checkpoint (left and right final values, cause events
        and both paths included), detached from every other returned
        object. A series with no checkpoints never breaches: its final
        risk is ``0`` and ``remaining`` equals ``total_budget``.

        The rows are sorted with every ever-breached branch first; within
        that grouping they are ordered by descending ``max_overrun``, then
        ascending ``first_breach`` with ``None`` placed after every
        integer, then by the Unicode code point of the branch name. An
        empty ``series`` still validates ``weights`` and both budgets and
        looks the reference branch up, yielding an empty tuple; empty
        ``weights`` still validates every branch and node and yields zero
        risk (``key_budgets`` must then be empty). The tuples and dicts at
        every level are freshly built, neither sharing objects with each
        other nor aliasing internal state. Success or failure never
        modifies the graph, branch heads, audit or idempotency records, or
        any existing divergence interface.
        """
        # --- All inputs finish validating before any branch is looked up,
        # mirroring rank_impacts with the two budgets after weights. ---
        self._require_nonempty_str(reference, "reference")
        if not isinstance(series, dict):
            raise TypeError(
                f"series must be a dict, got {type(series).__name__}"
            )
        per_series: list[tuple[str, tuple[tuple[str, str], ...]]] = []
        for series_name, series_points in series.items():
            self._require_nonempty_str(series_name, "series branch")
            if series_name == reference:
                raise ValueError(
                    f"series branch {series_name!r} cannot equal reference "
                    f"branch {reference!r}"
                )
            self._validate_points(series_points)
            per_series.append((series_name, series_points))
        ordered_keys = self._validate_weights(weights)
        self._require_finite_budget(total_budget, "total budget")
        if not isinstance(key_budgets, dict):
            raise TypeError(
                "key_budgets must be a dict, got "
                f"{type(key_budgets).__name__}"
            )
        for limit_key, limit in key_budgets.items():
            EventGraph._require_nonempty_str(
                limit_key, "per-key budget key"
            )
            self._require_finite_budget(
                limit, f"per-key budget for {limit_key!r}"
            )
            if limit_key not in weights:
                raise ValueError(
                    f"per-key budget key {limit_key!r} is absent from weights"
                )

        self._require_known_branch(reference)

        # Capture the shared read-only view before any result is built,
        # exactly as in rank_impacts.
        reference_head_closure = set(
            self._graph._ordered_ancestors(self._heads[reference])
        )
        views: list[
            tuple[str, list[tuple[list[str], list[str], str, str]] | None]
        ] = []
        for series_name, series_points in per_series:
            self._require_known_branch(series_name)
            head_closure_b = set(
                self._graph._ordered_ancestors(self._heads[series_name])
            )
            for node_a, node_b in series_points:
                if (
                    node_a not in self._graph._at
                    or node_a not in reference_head_closure
                ):
                    raise KeyError(node_a)
                if (
                    node_b not in self._graph._at
                    or node_b not in head_closure_b
                ):
                    raise KeyError(node_b)
            ordered_points: list[
                tuple[list[str], list[str], str, str]
            ] | None = None
            if series_points:
                ordered_points = [
                    (
                        self._graph._ordered_ancestors(node_a),
                        self._graph._ordered_ancestors(node_b),
                        node_a,
                        node_b,
                    )
                    for node_a, node_b in series_points
                ]
            views.append((series_name, ordered_points))

        rows: list[dict[str, object]] = []
        for series_name, ordered_points in views:
            breached = False
            first_breach: int | None = None
            max_overrun: int | float = 0
            final_risk: int | float = 0
            final_view: tuple[list[str], list[str], str, str] | None = None
            final_over_keys: list[str] = []

            points = ordered_points or ()
            for index, (left_order, right_order, node_a, node_b) in enumerate(
                points
            ):
                risk: int | float = 0
                diffs: dict[str, int] = {}
                for key in ordered_keys:
                    diff = abs(
                        self._replayed_value(left_order, key)
                        - self._replayed_value(right_order, key)
                    )
                    diffs[key] = diff
                    risk += diff * weights[key]
                # A zero risk must never surface as negative zero.
                if risk == 0:
                    risk = 0 if isinstance(risk, int) else 0.0

                key_breaches = [
                    key
                    for key in ordered_keys
                    if key in key_budgets and diffs[key] > key_budgets[key]
                ]
                if risk > total_budget or key_breaches:
                    if not breached:
                        breached = True
                        first_breach = index
                    # The checkpoint's breach magnitude is the greatest of
                    # its total overrun and every breaching key's weighted
                    # excess; no overrun contributes a negative amount.
                    point_overrun: int | float = (
                        risk - total_budget if risk > total_budget else 0
                    )
                    for key in key_breaches:
                        amount = (
                            diffs[key] - key_budgets[key]
                        ) * weights[key]
                        if amount > point_overrun:
                            point_overrun = amount
                    if point_overrun > max_overrun:
                        max_overrun = point_overrun

                final_risk = risk
                final_view = (left_order, right_order, node_a, node_b)
                final_over_keys = key_breaches

            final_attributions: list[dict[str, object]] = []
            if final_view is not None and final_over_keys:
                left_order, right_order, node_a, node_b = final_view
                final_over_keys = sorted(final_over_keys)
                # Attributions reuse the existing historical divergence
                # result at this same checkpoint, freshly copied so the
                # returned objects share nothing with each other.
                final_attributions = [
                    self._copy_attribution(
                        self._attribute_divergence_on_orders(
                            left_order, right_order, node_a, node_b, key
                        )
                    )
                    for key in final_over_keys
                ]
            remaining = total_budget - final_risk
            # Remaining must never surface as negative zero.
            if remaining == 0:
                remaining = 0 if isinstance(remaining, int) else 0.0
            rows.append(
                {
                    "branch": series_name,
                    "breached": breached,
                    "first_breach": first_breach,
                    "max_overrun": max_overrun,
                    "remaining": remaining,
                    "over_keys": tuple(final_over_keys),
                    "attributions": tuple(final_attributions),
                }
            )
        rows.sort(
            key=lambda row: (
                not row["breached"],
                -row["max_overrun"],
                row["first_breach"] is None,
                (
                    row["first_breach"]
                    if row["first_breach"] is not None
                    else 0
                ),
                row["branch"],
            )
        )
        return tuple(rows)

    def combination_budget(
        self,
        reference: str,
        series: dict[str, tuple[tuple[str, str], ...]],
        combinations: dict[str, tuple[str, ...]],
        weights: dict[str, int | float],
        total_budget: int | float,
        key_budgets: dict[str, int | float],
    ) -> tuple[dict[str, object], ...]:
        """Judge simultaneously applied branch combinations against budgets.

        The read-only portfolio companion of :meth:`impact_budget`: each
        entry of ``combinations`` names a set of series branches applied
        together, and their per-checkpoint divergences from ``reference``
        are aggregated before being weighed against the same two budgets.
        ``series``, ``weights``, ``total_budget`` and ``key_budgets`` keep
        exactly :meth:`impact_budget`'s numeric conventions and errors.

        ``reference`` and ``series`` are validated first, in exactly
        :meth:`impact_budget`'s order and with its errors: a non-``str``
        or empty ``reference`` raises :class:`TypeError`/
        :class:`ValueError`; ``series`` must be a dict (else
        :class:`TypeError`); its keys are checked in insertion order, a
        non-``str`` series name raising :class:`TypeError` and an empty or
        ``reference``-equal name raising :class:`ValueError`; points are
        walked in input order (a non-tuple container or non-length-2-tuple
        point raises :class:`TypeError`, non-``str`` or empty node ids
        raise :class:`TypeError`/:class:`ValueError`, a repeated node pair
        raises :class:`ValueError`). ``combinations`` is checked next: it
        must be a dict (else :class:`TypeError`) mapping non-empty
        combination names to member branch tuples; a non-``str``
        combination name raises :class:`TypeError` and an empty one
        :class:`ValueError`; members that are not tuples raise
        :class:`TypeError`; within a combination, walked in input order, a
        non-``str`` member raises :class:`TypeError`, and an empty,
        duplicated or non-series member name raises :class:`ValueError`.
        Every member of one combination must have the same number of
        checkpoints and the same reference node at each position, else
        :class:`ValueError`; an empty member tuple applies no branch at
        all. ``weights``, ``total_budget`` and ``key_budgets`` are checked
        last, before any branch is looked up, with exactly
        :meth:`impact_budget`'s contract and errors. The branches are then
        looked up, reference first and then the series in ``series``
        insertion order (unknown branch :class:`KeyError`), and nodes are
        checked per pair, left before right, pairs in points order
        (:class:`KeyError` when absent from the graph or the named
        branch's current head closure).

        Every combination, checkpoint and key is answered from one shared
        read-only historical view: the reference branch's and every series
        branch's current head closures and every node's own closure are
        taken once before results are built. At each aligned checkpoint
        the signed difference ``member value minus reference value`` is
        summed per key over the combination's members -- a side that never
        introduced a key contributes ``0`` -- and the total risk is the
        sum, over the weight keys, of the absolute aggregate difference
        times the key's weight: plain Python arithmetic yielding an
        :class:`int` or :class:`float`, never rounded, and a zero (integer
        or float) is never negative zero. A checkpoint breaches when its
        total risk is strictly greater than ``total_budget``, or when any
        configured key's absolute aggregate difference is strictly greater
        than that key's per-key budget.

        Each combination contributes a fresh dict whose keys are ordered
        ``combination, members, breached, first_breach, max_overrun,
        remaining, over_keys, contributions, attributions``:
        ``combination`` is the combination name and ``members`` its member
        branch names in member order; ``breached`` says whether any
        checkpoint ever breached; ``first_breach`` is the index of the
        first checkpoint that did, or ``None`` when none did;
        ``max_overrun`` is the largest breach magnitude seen -- the
        maximum, over every checkpoint, of the total risk above
        ``total_budget`` and of each breaching key's excess aggregate
        difference multiplied by that key's weight -- or ``0`` when
        nothing ever breached; ``remaining`` is ``total_budget`` minus the
        final checkpoint's risk (plain Python arithmetic, may be
        negative); ``over_keys`` is the tuple of keys still over their
        per-key budget at the final checkpoint, ordered by Unicode code
        point. ``contributions`` holds each member's marginal contribution
        at the final checkpoint -- the full combination's risk minus the
        risk of the combination with that member removed -- which may be
        negative and is returned in member order. ``attributions`` aligns
        one-to-one with ``over_keys``: one fresh dict per such key whose
        keys are ordered ``key, aggregate, diffs, attributions`` -- the
        key, its signed aggregate difference at the final checkpoint, the
        tuple of the members' signed branch differences for the key in
        member order, and the tuple of independent copies of the existing
        historical divergence attribution for each member at that same
        checkpoint (the ``key, fork, left, right`` structure of
        :meth:`attribute_divergence_at` between the shared reference node
        and the member's node), detached from every other returned object.
        A combination with no checkpoints -- an empty member tuple or
        members without points -- never breaches: its final risk is ``0``,
        ``remaining`` equals ``total_budget`` and every member's
        contribution is ``0``.

        The rows are sorted with every ever-breached combination first;
        within that grouping they are ordered by descending
        ``max_overrun``, then ascending ``first_breach`` with ``None``
        placed after every integer, then by the Unicode code point of the
        combination name. An empty ``combinations`` still validates every
        input and looks the reference branch up, yielding an empty tuple;
        empty ``weights`` still checks every member, branch and node and
        yields zero risk (``key_budgets`` must then be empty). The tuples
        and dicts at every level are freshly built, neither sharing
        objects with each other nor aliasing internal state. Success or
        failure never modifies the graph, branch heads, audit or
        idempotency records, or any existing divergence interface.
        """
        # --- All inputs finish validating before any branch is looked up,
        # mirroring impact_budget with combinations after series. ---
        self._require_nonempty_str(reference, "reference")
        if not isinstance(series, dict):
            raise TypeError(
                f"series must be a dict, got {type(series).__name__}"
            )
        per_series: list[tuple[str, tuple[tuple[str, str], ...]]] = []
        series_points_by_name: dict[str, tuple[tuple[str, str], ...]] = {}
        for series_name, series_points in series.items():
            self._require_nonempty_str(series_name, "series branch")
            if series_name == reference:
                raise ValueError(
                    f"series branch {series_name!r} cannot equal reference "
                    f"branch {reference!r}"
                )
            self._validate_points(series_points)
            per_series.append((series_name, series_points))
            series_points_by_name[series_name] = series_points
        if not isinstance(combinations, dict):
            raise TypeError(
                "combinations must be a dict, got "
                f"{type(combinations).__name__}"
            )
        per_combination: list[tuple[str, tuple[str, ...]]] = []
        for combination_name, members in combinations.items():
            self._require_nonempty_str(combination_name, "combination")
            if not isinstance(members, tuple):
                raise TypeError(
                    f"members must be a tuple, got {type(members).__name__}"
                )
            seen_members: set[str] = set()
            for member in members:
                self._require_nonempty_str(member, "member")
                if member in seen_members:
                    raise ValueError(f"duplicate member {member!r}")
                seen_members.add(member)
                if member not in series:
                    raise ValueError(
                        f"member {member!r} of combination "
                        f"{combination_name!r} is absent from series"
                    )
            if members:
                anchor = series_points_by_name[members[0]]
                for member in members[1:]:
                    member_points = series_points_by_name[member]
                    if len(member_points) != len(anchor) or any(
                        point[0] != anchor_point[0]
                        for point, anchor_point in zip(member_points, anchor)
                    ):
                        raise ValueError(
                            f"members of combination {combination_name!r} "
                            "must share checkpoint count and reference nodes"
                        )
            per_combination.append((combination_name, members))
        ordered_keys = self._validate_weights(weights)
        self._require_finite_budget(total_budget, "total budget")
        if not isinstance(key_budgets, dict):
            raise TypeError(
                "key_budgets must be a dict, got "
                f"{type(key_budgets).__name__}"
            )
        for limit_key, limit in key_budgets.items():
            EventGraph._require_nonempty_str(
                limit_key, "per-key budget key"
            )
            self._require_finite_budget(
                limit, f"per-key budget for {limit_key!r}"
            )
            if limit_key not in weights:
                raise ValueError(
                    f"per-key budget key {limit_key!r} is absent from weights"
                )

        self._require_known_branch(reference)
        if not combinations:
            return ()

        # Capture the shared read-only view before any result is built,
        # exactly as in impact_budget: every combination is answered from
        # the same historical closures.
        reference_head_closure = set(
            self._graph._ordered_ancestors(self._heads[reference])
        )
        views_by_name: dict[
            str, list[tuple[list[str], list[str], str, str]]
        ] = {}
        for series_name, series_points in per_series:
            self._require_known_branch(series_name)
            head_closure_b = set(
                self._graph._ordered_ancestors(self._heads[series_name])
            )
            for node_a, node_b in series_points:
                if (
                    node_a not in self._graph._at
                    or node_a not in reference_head_closure
                ):
                    raise KeyError(node_a)
                if (
                    node_b not in self._graph._at
                    or node_b not in head_closure_b
                ):
                    raise KeyError(node_b)
            views_by_name[series_name] = [
                (
                    self._graph._ordered_ancestors(node_a),
                    self._graph._ordered_ancestors(node_b),
                    node_a,
                    node_b,
                )
                for node_a, node_b in series_points
            ]

        rows: list[dict[str, object]] = []
        for combination_name, members in per_combination:
            member_views = [views_by_name[member] for member in members]
            point_count = len(member_views[0]) if member_views else 0

            breached = False
            first_breach: int | None = None
            max_overrun: int | float = 0
            final_risk: int | float = 0
            final_over_keys: list[str] = []
            final_aggregate: dict[str, int] = {}
            final_member_diffs: list[dict[str, int]] = []
            final_position: list[
                tuple[list[str], list[str], str, str]
            ] | None = None

            for index in range(point_count):
                position = [views[index] for views in member_views]
                # The reference node is shared across members, so its
                # replayed values are computed once per checkpoint.
                reference_order = position[0][0]
                reference_values = {
                    key: self._replayed_value(reference_order, key)
                    for key in ordered_keys
                }
                aggregate: dict[str, int] = {
                    key: 0 for key in ordered_keys
                }
                member_diffs: list[dict[str, int]] = []
                for _, right_order, _, _ in position:
                    diffs: dict[str, int] = {}
                    for key in ordered_keys:
                        diff = (
                            self._replayed_value(right_order, key)
                            - reference_values[key]
                        )
                        diffs[key] = diff
                        aggregate[key] += diff
                    member_diffs.append(diffs)

                risk: int | float = 0
                for key in ordered_keys:
                    risk += abs(aggregate[key]) * weights[key]
                # A zero risk must never surface as negative zero.
                if risk == 0:
                    risk = 0 if isinstance(risk, int) else 0.0

                key_breaches = [
                    key
                    for key in ordered_keys
                    if key in key_budgets
                    and abs(aggregate[key]) > key_budgets[key]
                ]
                if risk > total_budget or key_breaches:
                    if not breached:
                        breached = True
                        first_breach = index
                    # The checkpoint's breach magnitude is the greatest of
                    # its total overrun and every breaching key's weighted
                    # excess; no overrun contributes a negative amount.
                    point_overrun: int | float = (
                        risk - total_budget if risk > total_budget else 0
                    )
                    for key in key_breaches:
                        amount = (
                            abs(aggregate[key]) - key_budgets[key]
                        ) * weights[key]
                        if amount > point_overrun:
                            point_overrun = amount
                    if point_overrun > max_overrun:
                        max_overrun = point_overrun

                final_risk = risk
                final_over_keys = key_breaches
                final_aggregate = aggregate
                final_member_diffs = member_diffs
                final_position = position

            # Each member's marginal contribution at the final checkpoint
            # is the full combination's risk minus the risk without it.
            contributions: list[int | float] = []
            for member_index in range(len(members)):
                if final_position is None:
                    contributions.append(0)
                    continue
                excluded_risk: int | float = 0
                for key in ordered_keys:
                    excluded = (
                        final_aggregate[key]
                        - final_member_diffs[member_index][key]
                    )
                    excluded_risk += abs(excluded) * weights[key]
                if excluded_risk == 0:
                    excluded_risk = (
                        0 if isinstance(excluded_risk, int) else 0.0
                    )
                contribution = final_risk - excluded_risk
                # A zero contribution must never surface as negative zero.
                if contribution == 0:
                    contribution = (
                        0 if isinstance(contribution, int) else 0.0
                    )
                contributions.append(contribution)

            attributions: list[dict[str, object]] = []
            if final_position is not None and final_over_keys:
                for key in sorted(final_over_keys):
                    # Attributions reuse the existing historical divergence
                    # result at this same checkpoint, freshly copied so the
                    # returned objects share nothing with each other.
                    attributions.append(
                        {
                            "key": key,
                            "aggregate": final_aggregate[key],
                            "diffs": tuple(
                                diffs[key]
                                for diffs in final_member_diffs
                            ),
                            "attributions": tuple(
                                self._copy_attribution(
                                    self._attribute_divergence_on_orders(
                                        left_order,
                                        right_order,
                                        node_a,
                                        node_b,
                                        key,
                                    )
                                )
                                for (
                                    left_order,
                                    right_order,
                                    node_a,
                                    node_b,
                                ) in final_position
                            ),
                        }
                    )

            remaining = total_budget - final_risk
            # Remaining must never surface as negative zero.
            if remaining == 0:
                remaining = 0 if isinstance(remaining, int) else 0.0
            rows.append(
                {
                    "combination": combination_name,
                    "members": tuple(members),
                    "breached": breached,
                    "first_breach": first_breach,
                    "max_overrun": max_overrun,
                    "remaining": remaining,
                    "over_keys": tuple(final_over_keys),
                    "contributions": tuple(contributions),
                    "attributions": tuple(attributions),
                }
            )
        rows.sort(
            key=lambda row: (
                not row["breached"],
                -row["max_overrun"],
                row["first_breach"] is None,
                (
                    row["first_breach"]
                    if row["first_breach"] is not None
                    else 0
                ),
                row["combination"],
            )
        )
        return tuple(rows)

    def search_combinations(
        self,
        reference: str,
        series: dict[str, tuple[tuple[str, str], ...]],
        weights: dict[str, int | float],
        total_budget: int | float,
        key_budgets: dict[str, int | float],
        min_size: int,
        max_size: int,
        required: tuple[str, ...],
        exclusive_pairs: tuple[tuple[str, str], ...],
        limit: int,
    ) -> dict[str, object]:
        """Search budget-feasible subsets of series branches, read-only.

        Where :meth:`combination_budget` judges given combinations, this
        method actively enumerates subsets of the ``series`` pool within a
        size window and keeps the budget-feasible Pareto frontier.
        ``series``, ``weights``, ``total_budget`` and ``key_budgets`` keep
        exactly :meth:`combination_budget`'s numeric conventions, node
        closure rules and error types: series points are
        ``(reference node, series node)`` checkpoints, and every pool
        member must have the same checkpoint count with the same reference
        node at each position (else :class:`ValueError`).

        ``reference`` and ``series`` are validated first, in exactly
        :meth:`combination_budget`'s order and with its errors, followed by
        ``weights``, ``total_budget`` and ``key_budgets``. The search
        parameters are then validated in signature order: ``min_size`` and
        ``max_size`` must be non-``bool`` :class:`int` values (anything
        else, including :class:`bool`, raises :class:`TypeError`), with a
        negative ``min_size`` or a ``max_size`` smaller than ``min_size``
        or larger than the pool size raising :class:`ValueError`;
        ``required`` and ``exclusive_pairs`` must be tuples (else
        :class:`TypeError`), each exclusive entry a length-2 tuple (else
        :class:`TypeError`), and their member names are walked in input
        order -- a non-``str`` name raises :class:`TypeError`, and an empty
        name, a duplicate or pool-unknown member, a self-exclusive pair, a
        duplicate unordered exclusive pair, or two required members that
        are mutually exclusive raises :class:`ValueError`; ``limit`` comes
        last and must be a non-``bool`` :class:`int` no smaller than ``1``
        (else :class:`TypeError`/:class:`ValueError`). Only after every
        input has validated are branches looked up -- reference first,
        then the series in ``series`` insertion order (unknown branch
        :class:`KeyError`) -- and nodes checked per series, left before
        right, pairs in points order (:class:`KeyError` when absent from
        the graph or the named branch's current head closure).

        Subsets are enumerated by ascending member count and, within a
        size, by the Unicode code point order of the sorted member-name
        tuple. Enumerating more than ``limit`` candidates raises
        :class:`ValueError`; no partial result is returned. An empty pool
        accepts only the zero-to-zero size window, and the empty subset's
        risk is zero. Each candidate reuses the combination-budget
        accounting -- per-checkpoint signed member differences summed per
        key, total risk as the sum of absolute aggregates times weights,
        the same strictly-greater total/per-key breach tests -- all from
        one shared read-only historical view. The rejection first causes,
        in order, are: a missing required member
        (``reason="missing_required"``), a hit exclusive pair
        (``reason="exclusive_pair"``), the earliest checkpoint whose total
        risk is over budget (``reason="total_budget"``), and at that
        checkpoint the first key in Unicode order over its per-key budget
        (``reason="key_budget"``).

        The result is a fresh dict whose keys are ordered
        ``frontier, rejected``. Each rejected item is a fresh dict with
        keys ordered ``members, reason, checkpoint, key, overrun``: the
        sorted member-name tuple, the reason code, the breaching checkpoint
        index, the breaching state key and the overrun magnitude (total
        risk above the total budget, or the key's excess aggregate
        difference times its weight); values not applicable to the reason
        are ``None``. Rejected items keep enumeration order. Each feasible
        item is a fresh dict with keys ordered ``members, risk,
        contributions, attributions``: the sorted member tuple, the final
        checkpoint's risk, each member's marginal contribution at that
        checkpoint (full risk minus risk with that member removed, in
        member order), and the independent per-member attributions -- one
        fresh ``key, fork, left, right`` historical divergence attribution
        per member per weight key (members outer, keys in Unicode order
        inner), each detached from every other returned object; both
        tuples are empty for the empty subset and the no-checkpoint case.
        The frontier keeps exactly the feasible subsets no other feasible
        subset weakly dominates -- none with no fewer members and no higher
        risk that is strictly better on at least one -- and is sorted by
        descending member count, ascending final risk, then ascending
        member-name tuple. Every returned level is freshly built and
        detached from internal state. Success or failure never modifies
        the graph, branch heads, audit or idempotency records, and no
        existing combination-budget or divergence interface changes.
        """
        # --- All inputs finish validating before any branch is looked up,
        # mirroring combination_budget's reference, series, weights and
        # budget order. ---
        self._require_nonempty_str(reference, "reference")
        if not isinstance(series, dict):
            raise TypeError(
                f"series must be a dict, got {type(series).__name__}"
            )
        per_series: list[tuple[str, tuple[tuple[str, str], ...]]] = []
        anchor_points: tuple[tuple[str, str], ...] | None = None
        for series_name, series_points in series.items():
            self._require_nonempty_str(series_name, "series branch")
            if series_name == reference:
                raise ValueError(
                    f"series branch {series_name!r} cannot equal reference "
                    f"branch {reference!r}"
                )
            self._validate_points(series_points)
            if anchor_points is None:
                anchor_points = series_points
            elif len(series_points) != len(anchor_points) or any(
                point[0] != anchor_point[0]
                for point, anchor_point in zip(
                    series_points, anchor_points
                )
            ):
                raise ValueError(
                    "all series must share checkpoint count and reference "
                    "nodes"
                )
            per_series.append((series_name, series_points))
        ordered_keys = self._validate_weights(weights)
        self._require_finite_budget(total_budget, "total budget")
        if not isinstance(key_budgets, dict):
            raise TypeError(
                "key_budgets must be a dict, got "
                f"{type(key_budgets).__name__}"
            )
        for budget_key, budget_limit in key_budgets.items():
            EventGraph._require_nonempty_str(
                budget_key, "per-key budget key"
            )
            self._require_finite_budget(
                budget_limit, f"per-key budget for {budget_key!r}"
            )
            if budget_key not in weights:
                raise ValueError(
                    f"per-key budget key {budget_key!r} is absent from weights"
                )

        # Size bounds, required members and exclusive pairs are validated
        # in signature order; limit is last. Each integer parameter shares
        # the non-bool int contract, with its range checked in place.
        min_size_value = self._require_size_bound(min_size, "min_size")
        if min_size_value < 0:
            raise ValueError("min_size must be non-negative")
        max_size_value = self._require_size_bound(max_size, "max_size")
        if max_size_value < min_size_value:
            raise ValueError("max_size must be >= min_size")
        if max_size_value > len(per_series):
            raise ValueError("max_size must not exceed the number of series")

        if not isinstance(required, tuple):
            raise TypeError(
                f"required must be a tuple, got {type(required).__name__}"
            )
        required_members: list[str] = []
        seen_required: set[str] = set()
        for member in required:
            self._require_nonempty_str(member, "required member")
            if member in seen_required:
                raise ValueError(f"duplicate required member {member!r}")
            seen_required.add(member)
            if member not in series:
                raise ValueError(
                    f"required member {member!r} is absent from series"
                )
            required_members.append(member)

        if not isinstance(exclusive_pairs, tuple):
            raise TypeError(
                "exclusive_pairs must be a tuple, got "
                f"{type(exclusive_pairs).__name__}"
            )
        ordered_pairs: list[tuple[str, str]] = []
        seen_pairs: set[tuple[str, str]] = set()
        for pair in exclusive_pairs:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise TypeError(
                    "each exclusive pair must be a length-2 tuple of "
                    f"member names, got {pair!r} ({type(pair).__name__})"
                )
            member_a, member_b = pair
            self._require_nonempty_str(member_a, "exclusive member")
            self._require_nonempty_str(member_b, "exclusive member")
            if member_a == member_b:
                raise ValueError(
                    f"member {member_a!r} cannot be mutually exclusive "
                    "with itself"
                )
            if member_a not in series:
                raise ValueError(
                    f"exclusive member {member_a!r} is absent from series"
                )
            if member_b not in series:
                raise ValueError(
                    f"exclusive member {member_b!r} is absent from series"
                )
            normalized = tuple(sorted(pair))
            if normalized in seen_pairs:
                raise ValueError(f"duplicate exclusive pair {normalized!r}")
            seen_pairs.add(normalized)
            ordered_pairs.append((member_a, member_b))
        required_set = set(required_members)
        for member_a, member_b in ordered_pairs:
            if member_a in required_set and member_b in required_set:
                raise ValueError(
                    f"required members {member_a!r} and {member_b!r} are "
                    "mutually exclusive"
                )

        limit_value = self._require_size_bound(limit, "limit")
        if limit_value < 1:
            raise ValueError("limit must be >= 1")

        self._require_known_branch(reference)

        # Capture the single read-only historical view before any subset is
        # evaluated, exactly as in combination_budget.
        reference_head_closure = set(
            self._graph._ordered_ancestors(self._heads[reference])
        )
        views_by_name: dict[
            str, list[tuple[list[str], list[str], str, str]]
        ] = {}
        for series_name, series_points in per_series:
            self._require_known_branch(series_name)
            head_closure_b = set(
                self._graph._ordered_ancestors(self._heads[series_name])
            )
            for node_a, node_b in series_points:
                if (
                    node_a not in self._graph._at
                    or node_a not in reference_head_closure
                ):
                    raise KeyError(node_a)
                if (
                    node_b not in self._graph._at
                    or node_b not in head_closure_b
                ):
                    raise KeyError(node_b)
            views_by_name[series_name] = [
                (
                    self._graph._ordered_ancestors(node_a),
                    self._graph._ordered_ancestors(node_b),
                    node_a,
                    node_b,
                )
                for node_a, node_b in series_points
            ]

        # The Unicode-sorted pool fixes the canonical member tuple and the
        # code point enumeration order within each size.
        pool = tuple(sorted(name for name, _ in per_series))

        rejected: list[dict[str, object]] = []
        feasible: list[
            tuple[
                tuple[str, ...],
                int | float,
                tuple[int | float, ...],
                tuple[tuple[dict[str, object], ...], ...],
            ]
        ] = []
        candidate_count = 0
        for size in range(min_size_value, max_size_value + 1):
            for combo in itertools.combinations(pool, size):
                candidate_count += 1
                if candidate_count > limit_value:
                    raise ValueError(
                        "search candidate limit exceeded: more than "
                        f"{limit_value} candidates in size window"
                    )
                combo_set = set(combo)

                # Rejection first causes: required coverage, then an
                # exclusive pair, before any budget work is done.
                if not required_set.issubset(combo_set):
                    rejected.append(
                        {
                            "members": combo,
                            "reason": "missing_required",
                            "checkpoint": None,
                            "key": None,
                            "overrun": None,
                        }
                    )
                    continue
                hit_pair = next(
                    (
                        pair
                        for pair in ordered_pairs
                        if pair[0] in combo_set and pair[1] in combo_set
                    ),
                    None,
                )
                if hit_pair is not None:
                    rejected.append(
                        {
                            "members": combo,
                            "reason": "exclusive_pair",
                            "checkpoint": None,
                            "key": None,
                            "overrun": None,
                        }
                    )
                    continue

                outcome = self._evaluate_subset_view(
                    [views_by_name[member] for member in combo],
                    ordered_keys,
                    weights,
                    key_budgets,
                    total_budget,
                )
                if outcome[0] == "reject":
                    _, checkpoint, reason, breach_key, overrun = outcome
                    rejected.append(
                        {
                            "members": combo,
                            "reason": reason,
                            "checkpoint": checkpoint,
                            "key": breach_key,
                            "overrun": overrun,
                        }
                    )
                    continue

                # Feasible: the last checkpoint's aggregate answers the
                # final risk and the marginal contributions.
                (
                    _,
                    final_risk,
                    final_aggregate,
                    final_member_diffs,
                    final_position,
                ) = outcome
                contributions: list[int | float] = []
                for member_index in range(len(combo)):
                    if final_position is None:
                        contributions.append(0)
                        continue
                    excluded_risk: int | float = 0
                    for key in ordered_keys:
                        excluded = (
                            final_aggregate[key]
                            - final_member_diffs[member_index][key]
                        )
                        excluded_risk += abs(excluded) * weights[key]
                    if excluded_risk == 0:
                        excluded_risk = (
                            0
                            if isinstance(excluded_risk, int)
                            else 0.0
                        )
                    contribution = final_risk - excluded_risk
                    if contribution == 0:
                        contribution = (
                            0
                            if isinstance(contribution, int)
                            else 0.0
                        )
                    contributions.append(contribution)

                attributions: tuple[
                    tuple[dict[str, object], ...], ...
                ] = ()
                if final_position is not None:
                    # Independent copies of the historical divergence
                    # attribution at the final checkpoint, member outer and
                    # key inner, detached from one another and the store.
                    attributions = tuple(
                        tuple(
                            self._copy_attribution(
                                self._attribute_divergence_on_orders(
                                    left_order,
                                    right_order,
                                    node_a,
                                    node_b,
                                    key,
                                )
                            )
                            for key in ordered_keys
                        )
                        for (
                            left_order,
                            right_order,
                            node_a,
                            node_b,
                        ) in final_position
                    )

                feasible.append(
                    (
                        combo,
                        final_risk,
                        tuple(contributions),
                        attributions,
                    )
                )

        # Pareto frontier: drop a feasible subset when another has no fewer
        # members and no higher risk and is strictly better on one of them.
        frontier_entries: list[
            tuple[
                tuple[str, ...],
                int | float,
                tuple[int | float, ...],
                tuple[tuple[dict[str, object], ...], ...],
            ]
        ] = []
        for entry in feasible:
            members, risk, _, _ = entry
            dominated = any(
                len(other_members) >= len(members)
                and other_risk <= risk
                and (
                    len(other_members) > len(members)
                    or other_risk < risk
                )
                for other_members, other_risk, _, _ in feasible
            )
            if not dominated:
                frontier_entries.append(entry)
        frontier_entries.sort(
            key=lambda entry: (-len(entry[0]), entry[1], entry[0])
        )

        frontier = [
            {
                "members": members,
                "risk": risk,
                "contributions": tuple(contributions),
                "attributions": tuple(
                    tuple(member_attributions)
                    for member_attributions in attributions
                ),
            }
            for members, risk, contributions, attributions in frontier_entries
        ]
        return {"frontier": tuple(frontier), "rejected": tuple(rejected)}

    def frontier_sensitivity(
        self,
        reference: str,
        series: dict[str, tuple[tuple[str, str], ...]],
        scenarios: tuple[dict[str, object], ...],
        min_size: int,
        max_size: int,
        required: tuple[str, ...],
        exclusive_pairs: tuple[tuple[str, str], ...],
        limit: int,
    ) -> dict[str, object]:
        """Judge how stable the budget frontier is across budget scenarios.

        The read-only sensitivity companion of :meth:`search_combinations`:
        the same pool, size window, required members, exclusive pairs and
        candidate limit are searched once, but under several ordered
        ``scenarios`` -- each its own weights, total budget and per-key
        budgets -- answered from one shared frozen historical view.

        Parameters are validated in signature order (``reference``,
        ``series``, ``scenarios``, ``min_size``, ``max_size``, ``required``,
        ``exclusive_pairs``, ``limit``); none has a default. ``reference``
        and ``series`` keep exactly :meth:`search_combinations`' contract,
        including the shared-checkpoint/reference-node alignment error.
        ``scenarios`` must be a tuple (else :class:`TypeError`) of dicts
        (a non-dict item raises :class:`TypeError`), each containing
        exactly the keys ``name``, ``weights``, ``total_budget`` and
        ``key_budgets`` -- a missing or extra key raises
        :class:`ValueError`. A scenario ``name`` that is not a ``str``
        raises :class:`TypeError`; an empty or duplicated name (scenarios
        walked in input order) raises :class:`ValueError`. Each scenario's
        ``weights``, ``total_budget`` and ``key_budgets`` reuse
        :meth:`search_combinations`' numeric, finiteness and key-set rules
        and error types, scoped to that scenario's own weight keys.
        ``min_size``, ``max_size``, ``required``, ``exclusive_pairs`` and
        ``limit`` then keep exactly the existing search's validation order
        and errors. Only after every ordinary input and every scenario has
        validated are branches looked up -- reference first, then the
        series in ``series`` insertion order (unknown branch
        :class:`KeyError`) -- and nodes checked per series, left before
        right, pairs in points order (:class:`KeyError` when absent from
        the graph or the named branch's current head closure).

        Candidates are enumerated exactly once, by ascending member count
        and, within a size, by the Unicode code point order of the sorted
        member-name tuple, exactly as in :meth:`search_combinations`;
        enumerating more than ``limit`` candidates raises
        :class:`ValueError` and no partial result is returned. Each
        scenario evaluates every candidate on the same frozen read-only
        historical view with that scenario's budgets and weights; its
        frontier and rejection accounting are identical to one existing
        search with the same arguments. With an empty ``scenarios`` tuple
        the search-parameter validation, branch lookups and node closure
        checks still all run, and empty scenario and combination summaries
        are returned.

        The result is a fresh dict whose keys are ordered
        ``scenarios, combinations``. ``scenarios`` follows scenario input
        order; each entry is a fresh dict whose keys are ordered
        ``name, frontier, rejected``, with independent copies of exactly
        the frontier and rejected row structures
        :meth:`search_combinations` returns (frontier rows keyed
        ``members, risk, contributions, attributions`` and rejected rows
        keyed ``members, reason, checkpoint, key, overrun``).
        ``combinations`` follows candidate enumeration order; each row is a
        fresh dict whose keys are ordered ``members, first_entry,
        first_exit, dominators, risk_delta, member_delta``:
        ``first_entry`` is the index of the first scenario whose frontier
        contains the combination (``0`` when the first scenario does), or
        ``None`` when it never makes a frontier; ``first_exit`` is the
        index of the first later scenario whose frontier no longer
        contains it, or ``None`` when it never entered or stays on the
        frontier through the last scenario. ``dominators`` aligns one
        tuple per scenario: the feasible combinations that dominate this
        combination under the existing member-count and final-risk weak-
        domination rule, sorted in frontier order (descending member
        count, ascending risk, ascending member tuple); an infeasible or
        non-dominated scenario contributes an empty tuple. ``risk_delta``
        aligns one value per scenario: the final-risk difference from the
        previous scenario, or ``None`` at the first scenario or whenever
        this combination is infeasible in either scenario.
        ``member_delta`` mirrors it per member (member order): the marginal
        contribution differences between adjacent scenarios, or ``None``
        under the same conditions. Every returned level is freshly built,
        independent of the others and detached from internal state.
        Success or failure never modifies the graph, branch heads, audit
        or idempotency records, and no existing interface changes.
        """
        # --- Ordinary inputs and every scenario finish validating before
        # any branch is looked up. ---
        self._require_nonempty_str(reference, "reference")
        if not isinstance(series, dict):
            raise TypeError(
                f"series must be a dict, got {type(series).__name__}"
            )
        per_series: list[tuple[str, tuple[tuple[str, str], ...]]] = []
        anchor_points: tuple[tuple[str, str], ...] | None = None
        for series_name, series_points in series.items():
            self._require_nonempty_str(series_name, "series branch")
            if series_name == reference:
                raise ValueError(
                    f"series branch {series_name!r} cannot equal reference "
                    f"branch {reference!r}"
                )
            self._validate_points(series_points)
            if anchor_points is None:
                anchor_points = series_points
            elif len(series_points) != len(anchor_points) or any(
                point[0] != anchor_point[0]
                for point, anchor_point in zip(
                    series_points, anchor_points
                )
            ):
                raise ValueError(
                    "all series must share checkpoint count and reference "
                    "nodes"
                )
            per_series.append((series_name, series_points))

        parsed_scenarios = self._validate_sensitivity_scenarios(scenarios)

        # The size window, member constraints and limit reuse the existing
        # search's checks exactly.
        min_size_value = self._require_size_bound(min_size, "min_size")
        if min_size_value < 0:
            raise ValueError("min_size must be non-negative")
        max_size_value = self._require_size_bound(max_size, "max_size")
        if max_size_value < min_size_value:
            raise ValueError("max_size must be >= min_size")
        if max_size_value > len(per_series):
            raise ValueError("max_size must not exceed the number of series")

        if not isinstance(required, tuple):
            raise TypeError(
                f"required must be a tuple, got {type(required).__name__}"
            )
        required_members: list[str] = []
        seen_required: set[str] = set()
        for member in required:
            self._require_nonempty_str(member, "required member")
            if member in seen_required:
                raise ValueError(f"duplicate required member {member!r}")
            seen_required.add(member)
            if member not in series:
                raise ValueError(
                    f"required member {member!r} is absent from series"
                )
            required_members.append(member)

        if not isinstance(exclusive_pairs, tuple):
            raise TypeError(
                "exclusive_pairs must be a tuple, got "
                f"{type(exclusive_pairs).__name__}"
            )
        ordered_pairs: list[tuple[str, str]] = []
        seen_pairs: set[tuple[str, str]] = set()
        for pair in exclusive_pairs:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise TypeError(
                    "each exclusive pair must be a length-2 tuple of "
                    f"member names, got {pair!r} ({type(pair).__name__})"
                )
            member_a, member_b = pair
            self._require_nonempty_str(member_a, "exclusive member")
            self._require_nonempty_str(member_b, "exclusive member")
            if member_a == member_b:
                raise ValueError(
                    f"member {member_a!r} cannot be mutually exclusive "
                    "with itself"
                )
            if member_a not in series:
                raise ValueError(
                    f"exclusive member {member_a!r} is absent from series"
                )
            if member_b not in series:
                raise ValueError(
                    f"exclusive member {member_b!r} is absent from series"
                )
            normalized = tuple(sorted(pair))
            if normalized in seen_pairs:
                raise ValueError(f"duplicate exclusive pair {normalized!r}")
            seen_pairs.add(normalized)
            ordered_pairs.append((member_a, member_b))
        required_set = set(required_members)
        for member_a, member_b in ordered_pairs:
            if member_a in required_set and member_b in required_set:
                raise ValueError(
                    f"required members {member_a!r} and {member_b!r} are "
                    "mutually exclusive"
                )

        limit_value = self._require_size_bound(limit, "limit")
        if limit_value < 1:
            raise ValueError("limit must be >= 1")

        self._require_known_branch(reference)

        # Capture the single frozen read-only historical view every
        # scenario is answered from, exactly as in search_combinations.
        reference_head_closure = set(
            self._graph._ordered_ancestors(self._heads[reference])
        )
        views_by_name: dict[
            str, list[tuple[list[str], list[str], str, str]]
        ] = {}
        for series_name, series_points in per_series:
            self._require_known_branch(series_name)
            head_closure_b = set(
                self._graph._ordered_ancestors(self._heads[series_name])
            )
            for node_a, node_b in series_points:
                if (
                    node_a not in self._graph._at
                    or node_a not in reference_head_closure
                ):
                    raise KeyError(node_a)
                if (
                    node_b not in self._graph._at
                    or node_b not in head_closure_b
                ):
                    raise KeyError(node_b)
            views_by_name[series_name] = [
                (
                    self._graph._ordered_ancestors(node_a),
                    self._graph._ordered_ancestors(node_b),
                    node_a,
                    node_b,
                )
                for node_a, node_b in series_points
            ]

        # Candidates are enumerated once in the existing search order and
        # shared by every scenario evaluation. The count cap is a search
        # constraint, so it runs even with no scenarios.
        pool = tuple(sorted(name for name, _ in per_series))
        combos: list[tuple[str, ...]] = []
        candidate_count = 0
        for size in range(min_size_value, max_size_value + 1):
            for combo in itertools.combinations(pool, size):
                candidate_count += 1
                if candidate_count > limit_value:
                    raise ValueError(
                        "search candidate limit exceeded: more than "
                        f"{limit_value} candidates in size window"
                    )
                combos.append(combo)

        # Empty scenarios still ran the full search-parameter, branch,
        # node and candidate-count checks above; nothing drives a
        # scenario evaluation.
        if not parsed_scenarios:
            return {"scenarios": (), "combinations": ()}

        scenario_results: list[dict[str, object]] = []
        scenario_tables: list[dict[str, object]] = []
        for (
            name,
            weights,
            total_budget,
            key_budgets,
            ordered_keys,
        ) in parsed_scenarios:
            frontier_rows, rejected_rows, feasible_stats = (
                self._frontier_search_once(
                    combos,
                    views_by_name,
                    ordered_keys,
                    weights,
                    key_budgets,
                    total_budget,
                    required_set,
                    ordered_pairs,
                )
            )
            frontier_order = [row["members"] for row in frontier_rows]
            scenario_tables.append(
                {
                    "frontier": set(frontier_order),
                    "feasible": feasible_stats,
                }
            )
            scenario_results.append(
                {
                    "name": name,
                    "frontier": tuple(frontier_rows),
                    "rejected": tuple(rejected_rows),
                }
            )

        scenario_count = len(parsed_scenarios)
        combination_rows: list[dict[str, object]] = []
        for combo in combos:
            members = tuple(combo)
            first_entry: int | None = None
            for index in range(scenario_count):
                if members in scenario_tables[index]["frontier"]:
                    first_entry = index
                    break
            first_exit: int | None = None
            if first_entry is not None:
                for index in range(first_entry + 1, scenario_count):
                    if members not in scenario_tables[index]["frontier"]:
                        first_exit = index
                        break

            dominators: list[tuple[tuple[str, ...], ...]] = []
            risk_delta: list[int | float | None] = []
            member_delta: list[tuple[int | float, ...] | None] = []
            for index in range(scenario_count):
                table = scenario_tables[index]
                feasible_stats = table["feasible"]
                if members in feasible_stats:
                    risk = feasible_stats[members][0]
                    dominated_by = [
                        other
                        for other in feasible_stats
                        if len(other) >= len(members)
                        and feasible_stats[other][0] <= risk
                        and (
                            len(other) > len(members)
                            or feasible_stats[other][0] < risk
                        )
                    ]
                    dominated_by.sort(
                        key=lambda other: (
                            -len(other),
                            feasible_stats[other][0],
                            other,
                        )
                    )
                    dominators.append(
                        tuple(tuple(other) for other in dominated_by)
                    )
                else:
                    dominators.append(())

                if (
                    index == 0
                    or members not in feasible_stats
                    or members
                    not in scenario_tables[index - 1]["feasible"]
                ):
                    risk_delta.append(None)
                    member_delta.append(None)
                    continue

                previous_stats = scenario_tables[index - 1]["feasible"]
                risk_change = risk - previous_stats[members][0]
                if risk_change == 0:
                    risk_change = (
                        0
                        if isinstance(risk_change, int)
                        else 0.0
                    )
                risk_delta.append(risk_change)
                contribution_change: list[int | float] = []
                contributions = feasible_stats[members][1]
                previous_contributions = previous_stats[members][1]
                for member_index in range(len(members)):
                    change = (
                        contributions[member_index]
                        - previous_contributions[member_index]
                    )
                    if change == 0:
                        change = 0 if isinstance(change, int) else 0.0
                    contribution_change.append(change)
                member_delta.append(tuple(contribution_change))

            combination_rows.append(
                {
                    "members": members,
                    "first_entry": first_entry,
                    "first_exit": first_exit,
                    "dominators": tuple(dominators),
                    "risk_delta": tuple(risk_delta),
                    "member_delta": tuple(member_delta),
                }
            )

        return {
            "scenarios": tuple(scenario_results),
            "combinations": tuple(combination_rows),
        }

    def frontier_breakpoints(
        self,
        reference: str,
        series: dict[str, tuple[tuple[str, str], ...]],
        base_scenario: dict[str, object],
        axis: str,
        values: tuple[int | float, ...],
        min_size: int,
        max_size: int,
        required: tuple[str, ...],
        exclusive_pairs: tuple[tuple[str, str], ...],
        limit: int,
    ) -> dict[str, object]:
        """Locate the perturbation intervals at which the frontier turns.

        The read-only turning-point companion of
        :meth:`frontier_sensitivity`: where the sensitivity analysis runs
        several whole scenarios, this method holds one ``base_scenario``
        fixed and perturbs exactly one of its numbers -- the total budget
        or one existing weight key -- across an ordered series of values,
        re-running the existing combination search at each value on one
        frozen historical view, and reports one record per adjacent value
        pair whose feasibility, domination or frontier membership changes.

        Parameters are validated in signature order (``reference``,
        ``series``, ``base_scenario``, ``axis``, ``values``, ``min_size``,
        ``max_size``, ``required``, ``exclusive_pairs``, ``limit``); none
        has a default. ``reference`` and ``series`` keep exactly
        :meth:`search_combinations`' contract, including the shared-
        checkpoint/reference-node alignment error
        (:class:`TypeError`/:class:`ValueError`/:class:`KeyError`).
        ``base_scenario`` must be a dict (else :class:`TypeError`)
        containing exactly the keys ``weights``, ``total_budget`` and
        ``key_budgets`` -- a missing or extra key raises
        :class:`ValueError` -- and its three entries reuse
        :meth:`search_combinations`' numeric, finiteness and key-set rules
        and error types. ``axis`` must be a non-empty ``str``
        (:class:`TypeError`/:class:`ValueError`) equal to
        ``"total_budget"`` or to one of the base scenario's weight keys;
        any other string raises :class:`ValueError`. ``values`` must be a
        tuple (else :class:`TypeError`) of at least two non-``bool``
        :class:`int`/:class:`float` numbers; a :class:`bool` or
        non-number element raises :class:`TypeError`, while a negative,
        NaN or infinite value, or values that are not strictly increasing
        and unique (out of order or repeated), raise :class:`ValueError`.
        ``min_size``, ``max_size``, ``required``, ``exclusive_pairs`` and
        ``limit`` then keep exactly the existing search's validation order
        and errors. Only after every ordinary input has validated are
        branches looked up -- reference first, then the series in
        ``series`` insertion order (unknown branch :class:`KeyError`) --
        and nodes checked per series, left before right, pairs in points
        order (:class:`KeyError` when absent from the graph or the named
        branch's current head closure); misaligned checkpoints raise
        :class:`ValueError` during input validation.

        Each value replaces only the one number ``axis`` names -- a total
        budget or one weight -- leaving every other base-scenario entry
        untouched; candidates are enumerated once, in the existing
        ascending-size/code-point order on one frozen read-only view, and
        enumerating more than ``limit`` candidates raises
        :class:`ValueError` with no partial result. An empty ``series``
        pool still runs the size-window, node and candidate-count checks
        (the empty subset is its sole candidate).

        The result is a fresh dict whose keys are ordered
        ``points, breakpoints``. ``points`` follows the given value order;
        each entry is a fresh dict whose keys are ordered
        ``value, frontier, rejected``, holding independent copies of
        exactly the row structures :meth:`search_combinations` returns.
        ``breakpoints`` follows interval order: one fresh dict per
        adjacent value pair whose feasibility, dominators or frontier
        membership changes for any candidate, with keys ordered
        ``left_bound, right_bound, left, right, entered, exited,
        affected, member_deltas``. ``left`` and ``right`` are independent
        copies of the two sides' complete point results; ``entered`` and
        ``exited`` list the member tuples joining/leaving the frontier,
        and ``affected`` the combinations whose feasibility or
        domination relationships change -- each list in candidate
        enumeration order with a combination appearing at most once per
        record. ``member_deltas`` aligns one fresh ``members, delta`` dict
        per changed combination in that same order; ``delta`` is the
        right-minus-left difference of every member's marginal
        contribution in member order, or ``None`` whenever the
        combination is infeasible on either side, and a zero difference
        is never negative zero. Adjacent values with completely equal
        results generate no record. Every returned level is freshly
        built and detached from internal state. Success or failure never
        modifies the graph, branch heads, audit or idempotency records,
        and no existing interface changes.
        """
        # Every ordinary input, the base scenario and the value series
        # finish validating before any branch is looked up; the frozen
        # historical view and the once-enumerated candidates are shared by
        # both frontier_breakpoints and explain_frontier_breakpoints.
        prepared = self._prepare_frontier_scan(
            reference,
            series,
            base_scenario,
            axis,
            values,
            min_size,
            max_size,
            required,
            exclusive_pairs,
            limit,
        )
        combos = prepared["combos"]

        snapshots = [
            self._breakpoint_snapshot(prepared, value)
            for value in prepared["ordered_values"]
        ]

        # Independent point results first; breakpoint copies are built
        # separately so no object is shared between or within the levels.
        points = tuple(
            self._copy_breakpoint_point(snapshot) for snapshot in snapshots
        )

        breakpoints: list[dict[str, object]] = []
        for index in range(len(snapshots) - 1):
            left_snapshot = snapshots[index]
            right_snapshot = snapshots[index + 1]
            entered: list[tuple[str, ...]] = []
            exited: list[tuple[str, ...]] = []
            affected: list[tuple[str, ...]] = []
            member_deltas: list[dict[str, object]] = []
            for combo in combos:
                feasible_left = combo in left_snapshot["feasible"]
                feasible_right = combo in right_snapshot["feasible"]
                frontier_left = combo in left_snapshot["frontier"]
                frontier_right = combo in right_snapshot["frontier"]
                dominators_left = left_snapshot["dominators"][combo]
                dominators_right = right_snapshot["dominators"][combo]
                feasibility_changed = feasible_left != feasible_right
                domination_changed = dominators_left != dominators_right
                frontier_changed = frontier_left != frontier_right
                if not (
                    feasibility_changed
                    or domination_changed
                    or frontier_changed
                ):
                    continue

                if frontier_right and not frontier_left:
                    entered.append(combo)
                if frontier_left and not frontier_right:
                    exited.append(combo)
                if feasibility_changed or domination_changed:
                    affected.append(combo)

                if feasible_left and feasible_right:
                    left_contributions = left_snapshot["stats"][combo][1]
                    right_contributions = right_snapshot["stats"][combo][1]
                    changes: list[int | float] = []
                    for member_index in range(len(combo)):
                        change = (
                            right_contributions[member_index]
                            - left_contributions[member_index]
                        )
                        if change == 0:
                            change = (
                                0
                                if isinstance(change, int)
                                else 0.0
                            )
                        changes.append(change)
                    delta: tuple[int | float, ...] | None = tuple(changes)
                else:
                    delta = None
                member_deltas.append(
                    {"members": tuple(combo), "delta": delta}
                )

            if not entered and not exited and not affected:
                continue
            breakpoints.append(
                {
                    "left_bound": left_snapshot["value"],
                    "right_bound": right_snapshot["value"],
                    "left": self._copy_breakpoint_point(left_snapshot),
                    "right": self._copy_breakpoint_point(right_snapshot),
                    "entered": tuple(tuple(combo) for combo in entered),
                    "exited": tuple(tuple(combo) for combo in exited),
                    "affected": tuple(tuple(combo) for combo in affected),
                    "member_deltas": tuple(member_deltas),
                }
            )

        return {"points": points, "breakpoints": tuple(breakpoints)}

    def explain_frontier_breakpoints(
        self,
        reference: str,
        series: dict[str, tuple[tuple[str, str], ...]],
        base_scenario: dict[str, object],
        axis: str,
        values: tuple[int | float, ...],
        min_size: int,
        max_size: int,
        required: tuple[str, ...],
        exclusive_pairs: tuple[tuple[str, str], ...],
        limit: int,
    ) -> tuple[dict[str, object], ...]:
        """Explain which historical node and state key drives each frontier turn.

        The read-only attribution companion of :meth:`frontier_breakpoints`:
        it answers from the exact same arguments -- none has a default, and
        parameter validation, the branch/node lookup order, checkpoint
        alignment, candidate enumeration order and the candidate-count cap
        are exactly :meth:`frontier_breakpoints`' contract and errors
        (:class:`TypeError` for wrong container/name/number types, with
        :class:`bool` never accepted as an :class:`int`/:class:`float`;
        :class:`ValueError` for empty names, wrong field sets, an unknown
        axis, non-finite or out-of-order values, member-constraint conflicts,
        misaligned checkpoints or more than ``limit`` candidates;
        :class:`KeyError` for unknown reference/series branches or nodes,
        including a node outside its named branch's current head ancestor
        closure). Exceeding the candidate cap raises before any explanation
        is built, so no partial result or state change is possible.

        The existing turning-point result is first taken on one frozen
        historical view; every breakpoint interval is then explained for
        every combination that enters, exits or is otherwise affected. Each
        combination appears once per interval, in the existing candidate
        enumeration order (ascending member count, then the Unicode code
        point order of the sorted member tuple). The change type is decided
        in order: ``"feasibility"`` when feasibility flips, otherwise
        ``"domination"`` when the dominator set changes, otherwise
        ``"membership"`` for a frontier-membership-only change. A feasibility
        change is located at the first checkpoint whose budget verdict
        differs between the two sides; every other change uses the last
        checkpoint, the one deciding final risk.

        The returned tuple has one fresh dict per interval with keys ordered
        ``left_bound, right_bound, changes`` (the same interval bounds as
        :meth:`frontier_breakpoints`' records). ``changes`` holds one fresh
        evidence dict per changed combination, keys ordered ``members, kind,
        checkpoint, cause_key, attribution, left, right``: ``members`` is
        the member tuple; ``kind`` is the change type; ``checkpoint`` is the
        locating checkpoint index, or ``None`` when no checkpoint exists;
        ``cause_key`` is the responsible state key -- for the total-budget
        axis the non-zero key with the largest absolute contribution at that
        checkpoint (ties broken by the smallest Unicode code point), and for
        a weight axis the perturbed key itself, or ``None`` when that key has
        no contribution in any relevant member; ``attribution`` is an empty
        tuple when ``cause_key`` is ``None``, otherwise the per-member copies
        of the existing historical divergence attribution at that
        checkpoint, in member order. ``left`` and ``right`` are fresh dicts
        with keys ordered ``feasible, risk, dominators, overrun``: the
        feasibility flag, the final-checkpoint risk (``None`` when
        infeasible), the dominators in the existing deterministic frontier
        order, and the locating checkpoint's budget overrun (``None`` when
        no budget evidence exists; a zero is never substituted). With no
        turning-point intervals an empty tuple is returned. Every dict keeps
        the documented key order, and all levels are freshly built, share no
        objects with one another and alias no internal state. The query is
        read-only: success or failure never modifies the event graph,
        branch heads, audit or idempotency records, or any existing search,
        sensitivity or breakpoint result.
        """
        prepared = self._prepare_frontier_scan(
            reference,
            series,
            base_scenario,
            axis,
            values,
            min_size,
            max_size,
            required,
            exclusive_pairs,
            limit,
        )
        combos = prepared["combos"]

        snapshots = [
            self._breakpoint_snapshot(prepared, value)
            for value in prepared["ordered_values"]
        ]

        intervals: list[dict[str, object]] = []
        for index in range(len(snapshots) - 1):
            left_snapshot = snapshots[index]
            right_snapshot = snapshots[index + 1]
            changes: list[dict[str, object]] = []
            for combo in combos:
                evidence = self._breakpoint_change_evidence(
                    prepared,
                    combo,
                    left_snapshot,
                    right_snapshot,
                )
                if evidence is not None:
                    changes.append(evidence)
            if changes:
                intervals.append(
                    {
                        "left_bound": left_snapshot["value"],
                        "right_bound": right_snapshot["value"],
                        "changes": tuple(changes),
                    }
                )
        return tuple(intervals)

    def decision_cascade(
        self,
        reference: str,
        series: dict[str, tuple[tuple[str, str], ...]],
        base_scenario: dict[str, object],
        axis: str,
        values: tuple[int | float, ...],
        min_size: int,
        max_size: int,
        required: tuple[str, ...],
        exclusive_pairs: tuple[tuple[str, str], ...],
        limit: int,
    ) -> dict[str, object]:
        """Chain consecutive frontier-turn explanations into a decision chain.

        The read-only cascade companion of
        :meth:`explain_frontier_breakpoints`: it answers from the exact same
        arguments -- none has a default -- with identical parameter
        validation, the branch/node lookup order, checkpoint alignment,
        candidate enumeration order and the candidate-count cap
        (:class:`TypeError` for wrong container/name/number types, with
        :class:`bool` never accepted as an :class:`int`/:class:`float`;
        :class:`ValueError` for empty names, wrong field sets, an unknown
        axis, non-finite or out-of-order values, member-constraint
        conflicts, misaligned checkpoints or more than ``limit``
        candidates; :class:`KeyError` for unknown reference/series branches
        or nodes, including a node outside its named branch's current head
        ancestor closure). Exceeding the candidate cap raises before any
        node, edge or gap is built, so no partial cascade or leftover
        intermediate result is possible.

        The whole turning-point explanation is first taken on one frozen
        historical view; its intervals are then processed in interval order
        (adjacent value pairs, left before right). Every non-empty
        attribution -- one per changed combination per member, in the
        existing change order and member order -- contributes an evidence
        node describing its cause event, the state key, the locating
        checkpoint and the affected event range. Evidence with the same
        cause event and state key is one node even when it spans several
        intervals, checkpoints or branches: a cause event shared across
        branches or reintroduced after a merge is never split. A node's
        branch names are sorted by Unicode code point, its intervals are
        aggregated in interval order, its checkpoints by first appearance,
        and its affected events are deduplicated while
        preserving the existing replay order of the corresponding
        historical closures -- events outside those closures are never
        pulled in. A change whose cause key or cause event is missing is
        not forged into a node; it is recorded as a gap under its original
        interval and combination, preserving the existing candidate
        enumeration order.

        Directed edges are created only between cause nodes of adjacent
        intervals, from the earlier cause to the later one. When the later
        cause event is not a strict descendant of the earlier one no edge is
        created and the two stay independent evidence chains; otherwise the
        edge path is the shortest parent-to-child path containing both
        ends, ties broken by the Unicode code point order of the complete
        event id tuples. Merge events may form cross-branch convergence
        edges, but an edge with the same start, end and path appears once.

        The result is a fresh dict whose keys are ordered ``nodes, edges,
        gaps``. Each node is a fresh dict with keys ordered ``cause, key,
        intervals, checkpoints, branches, affected``: the cause event id,
        the state key, the tuple of turning-interval indices the evidence
        appears at (the first is the node's first interval), the tuple of
        locating checkpoint indices in first-appearance order (the first is
        the node's locating checkpoint; every node's checkpoints are
        integers, since evidence without a checkpoint becomes a gap), the
        associated member branch names sorted by Unicode code point, and
        the deduplicated affected event ids in the corresponding
        historical closures' existing replay order. Each edge is a fresh
        dict with keys ordered ``source, target, path``: the two endpoint
        nodes' positions in the returned node tuple and the shortest id
        path. Each gap is a fresh dict with keys ordered
        ``interval, members``: the turning-interval index and the
        combination members. Nodes are sorted by first interval, then the
        first-appearance checkpoint (a missing checkpoint sorts after
        every integer), then state key, then cause event id; edges by
        source node order, target node order and path; gaps keep interval
        order and the existing candidate enumeration order within each
        interval. With no turning-point intervals the result holds three
        empty tuples. Every level is freshly built, detached from internal
        state and unshared across levels. The query is read-only: success
        or failure never modifies the event graph, branch heads, audit or
        idempotency records, or any existing query result.
        """
        # The shared prepare step applies exactly the explanation entry's
        # validation, lookup order, alignment, enumeration and candidate
        # cap, and captures the single frozen view every interval uses.
        prepared = self._prepare_frontier_scan(
            reference,
            series,
            base_scenario,
            axis,
            values,
            min_size,
            max_size,
            required,
            exclusive_pairs,
            limit,
        )
        return self._build_decision_cascade(prepared)

    def _build_decision_cascade(
        self, prepared: dict[str, object]
    ) -> dict[str, object]:
        """Build the frozen cascade nodes, edges and gaps from one view.

        Shared by :meth:`decision_cascade` and :meth:`cascade_slice` so the
        slice reuses exactly the existing nodes, edges and gaps on the one
        frozen historical view :meth:`_prepare_frontier_scan` captured,
        never reinterpreting or widening its historical closure.
        """
        combos = prepared["combos"]

        snapshots = [
            self._breakpoint_snapshot(prepared, value)
            for value in prepared["ordered_values"]
        ]

        # Only the explanation entry's turning intervals (those with at
        # least one change) exist as cascade intervals; they are reindexed
        # densely in interval order, so consecutive turning intervals stay
        # adjacent even when raw adjacent value pairs changed nothing.
        intervals: list[list[dict[str, object]]] = []
        for index in range(len(snapshots) - 1):
            changes: list[dict[str, object]] = []
            for combo in combos:
                evidence = self._breakpoint_change_evidence(
                    prepared,
                    combo,
                    snapshots[index],
                    snapshots[index + 1],
                )
                if evidence is not None:
                    changes.append(evidence)
            if changes:
                intervals.append(changes)

        if not intervals:
            return {"nodes": (), "edges": (), "gaps": ()}

        # Aggregate evidence by (cause event, state key): one node even when
        # the same cause appears in several intervals, checkpoints or
        # branches (a shared event across branches, or one reintroduced by a
        # merge, is the same node).
        node_records: dict[tuple[str, str], dict[str, object]] = {}
        gaps: list[dict[str, object]] = []
        # The (cause, key) evidence present in each interval, in evidence
        # processing order; edges only ever join adjacent interval sets.
        interval_nodes: list[list[tuple[str, str]]] = []
        # Every member closure contributing affected evidence; their union
        # is ancestor-closed and supplies the merged replay order.
        contributing_orders: list[list[str]] = []

        for interval_index, changes in enumerate(intervals):
            present: list[tuple[str, str]] = []
            present_seen: set[tuple[str, str]] = set()
            for change in changes:
                members = change["members"]
                checkpoint = change["checkpoint"]
                attributions = change["attribution"]
                if change["cause_key"] is None or not attributions:
                    gaps.append(
                        {
                            "interval": interval_index,
                            "members": tuple(members),
                        }
                    )
                    continue
                produced = 0
                # Attribution copies align one-to-one with the members.
                # Each attribution names the cause on both sides -- the
                # reference closure and the member checkpoint closure --
                # and either side may carry the cause event. The member
                # checkpoint's own closure is captured in the frozen view,
                # so affected events never leave their side's closure.
                for member_index, (branch, attribution) in enumerate(
                    zip(members, attributions)
                ):
                    side_views = (
                        ("left", prepared["reference"], None),
                        ("right", branch, member_index),
                    )
                    for side, side_branch, side_index in side_views:
                        side_order = self._cascade_side_order(
                            prepared,
                            members,
                            side_index,
                            checkpoint,
                        )
                        node_key = self._collect_cascade_evidence(
                            node_records,
                            present,
                            present_seen,
                            interval_index,
                            checkpoint,
                            side_branch,
                            attribution["key"],
                            attribution[side],
                            side_order,
                            contributing_orders,
                        )
                        if node_key is not None:
                            produced += 1
                # A present cause key whose attributions carry no cause
                # event at all cannot become a node; record the gap once
                # under the original interval and combination.
                if produced == 0:
                    gaps.append(
                        {
                            "interval": interval_index,
                            "members": tuple(members),
                        }
                    )
            interval_nodes.append(present)

        # Deterministic node order: first interval, then the first
        # checkpoint (None after every integer), then state key, then
        # cause event id.
        def node_sort_key(node_key: tuple[str, str]) -> tuple[object, ...]:
            record = node_records[node_key]
            checkpoint = (
                record["checkpoints"][0] if record["checkpoints"] else None
            )
            return (
                record["intervals"][0],
                checkpoint is not None,
                checkpoint if checkpoint is not None else 0,
                record["key"],
                record["cause"],
            )

        ordered_node_keys = sorted(node_records, key=node_sort_key)
        node_index = {
            node_key: index for index, node_key in enumerate(ordered_node_keys)
        }

        # Merge every contributing member closure's replay order into one
        # ancestor-closed replay order, so a node's affected events -- each
        # originating inside one of those closures -- are deduplicated while
        # keeping the existing parents-before-children, (at, id) order.
        union_order = self._union_replay_order(contributing_orders)

        # A strict-descendant edge per adjacent interval pair only. Parent
        # edges are walked over the whole graph, so a merge can converge two
        # branches; identical (source, target, path) edges are deduplicated.
        # The children map is built once for every edge search.
        edge_records: set[tuple[int, int, tuple[str, ...]]] = set()
        if len(interval_nodes) >= 2:
            children_map = self._graph_children_map()
            for interval_index in range(len(interval_nodes) - 1):
                earlier_keys = interval_nodes[interval_index]
                later_keys = interval_nodes[interval_index + 1]
                for earlier in earlier_keys:
                    earlier_event = earlier[0]
                    for later in later_keys:
                        later_event = later[0]
                        if earlier == later or later_event == earlier_event:
                            continue
                        path = self._shortest_ancestor_path(
                            earlier_event,
                            later_event,
                            children_map,
                        )
                        if path is None:
                            continue
                        edge_records.add(
                            (
                                node_index[earlier],
                                node_index[later],
                                path,
                            )
                        )

        ordered_edges = sorted(edge_records)
        edges = tuple(
            {
                "source": source,
                "target": target,
                "path": tuple(path),
            }
            for source, target, path in ordered_edges
        )

        def node_affected(node_key: tuple[str, str]) -> tuple[str, ...]:
            affected: set[str] = set()
            for affected_set, _ in node_records[node_key]["affected_sets"]:
                affected.update(affected_set)
            return tuple(
                event_id
                for event_id in union_order
                if event_id in affected
            )

        nodes = tuple(
            {
                "cause": node_records[node_key]["cause"],
                "key": node_records[node_key]["key"],
                "intervals": tuple(node_records[node_key]["intervals"]),
                "checkpoints": tuple(
                    node_records[node_key]["checkpoints"]
                ),
                "branches": tuple(sorted(node_records[node_key]["branches"])),
                "affected": node_affected(node_key),
            }
            for node_key in ordered_node_keys
        )
        gap_tuple = tuple(
            {
                "interval": gap["interval"],
                "members": tuple(gap["members"]),
            }
            for gap in gaps
        )
        return {"nodes": nodes, "edges": edges, "gaps": gap_tuple}

    def cascade_slice(
        self,
        reference: str,
        series: dict[str, tuple[tuple[str, str], ...]],
        base_scenario: dict[str, object],
        axis: str,
        values: tuple[int | float, ...],
        min_size: int,
        max_size: int,
        required: tuple[str, ...],
        exclusive_pairs: tuple[tuple[str, str], ...],
        limit: int,
        causes: tuple[str, ...],
        direction: str,
        depth: int,
        node_limit: int,
    ) -> dict[str, object]:
        """Slice a bounded evidence subgraph out of a decision cascade.

        The read-only slice companion of :meth:`decision_cascade`: it is
        called with every one of that query's arguments -- none has a
        default -- followed by the cause-event set, the expansion
        direction, the edge-count depth and the selected-node cap. The
        ordinary cascade inputs are validated first in exactly
        :meth:`decision_cascade`'s order, including the branch and
        historical-node lookup order, checkpoint alignment and the
        candidate-count cap; the cascade is then generated on one frozen
        historical view, and the slice reuses its existing nodes, edges
        and gaps -- it never reinterprets or widens the historical
        closure.

        The slice-specific inputs are validated in order after the
        ordinary ones, all before any cascade node is inspected:
        ``causes`` must be a tuple (else :class:`TypeError`) of non-empty
        ``str`` ids walked in input order, a non-``str`` element raising
        :class:`TypeError` and an empty or duplicated id raising
        :class:`ValueError`; ``direction`` must be a ``str`` (else
        :class:`TypeError`) and exactly ``"forward"``, ``"backward"`` or
        ``"both"`` -- an empty string, any other value and every case
        variant raise :class:`ValueError`; ``depth`` and ``node_limit``
        must each be a non-``bool`` :class:`int` (else
        :class:`TypeError`), a negative ``depth`` or a ``node_limit``
        below one raising :class:`ValueError`.

        Cause events are handled in input order; when one event is the
        cause node of several state keys, every such node is a start.
        After the frozen cascade is built, a cause event absent from the
        cascade nodes raises :class:`KeyError` (the first absent one in
        input order). ``"forward"`` follows the directed edges from
        earlier cause to later cause, ``"backward"`` walks them in
        reverse, and ``"both"`` does both; cross-branch convergence edges
        stay usable because the walk follows the cascade's own directed
        edges, which may cross branches through merges. Depth counts
        edges: depth zero keeps only the start nodes, and no expansion
        ever leaves the existing cascade evidence. A node reached from
        several starts is selected once. The reachable set is counted
        only after the direction-and-depth expansion; when it exceeds
        ``node_limit`` a :class:`ValueError` is raised and no partial
        slice is returned. An empty ``causes`` tuple still completes
        every validation and the state lookup, then returns three empty
        tuples without applying the node cap.

        The result is a fresh dict with keys ordered ``nodes, edges,
        gaps``: nodes keep their full-cascade order (each start and
        reachable node exactly once), edges keep only records whose two
        ends are both selected, with endpoints remapped to slice
        positions while paths and the existing edge order are unchanged,
        and gaps keep only records whose interval is also an interval of
        a selected node, in the existing interval and candidate order.
        Nodes, edges and gaps at every level are freshly built copies
        that share no objects with each other or with any full-cascade
        result. The query is read-only: success or failure never modifies
        the event graph, branch heads, audit or idempotency records, and
        the existing cascade, turning-point explanation and ``status``
        entry behavior are unchanged.
        """
        # Ordinary inputs are validated first, in exactly the existing
        # order, but no branch or historical node is looked up yet.
        validated = self._validate_frontier_inputs(
            reference,
            series,
            base_scenario,
            axis,
            values,
            min_size,
            max_size,
            required,
            exclusive_pairs,
            limit,
        )

        # Slice inputs finish validating, in signature order, before any
        # state is queried: branch lookups, historical-node alignment and
        # the candidate cap all run afterwards.
        causes, direction, depth_value, node_limit_value = (
            self._validate_slice_inputs(causes, direction, depth, node_limit)
        )

        # Only now is state queried: the existing branch/node lookup order,
        # checkpoint alignment and the candidate cap run on the one frozen
        # view every result is taken from.
        prepared = self._capture_frontier_view(validated)
        return self._cascade_slice_on_prepared(
            prepared, causes, direction, depth_value, node_limit_value
        )

    def cascade_slice_diff(
        self,
        reference: str,
        series: dict[str, tuple[tuple[str, str], ...]],
        base_scenario: dict[str, object],
        axis: str,
        values: tuple[int | float, ...],
        min_size: int,
        max_size: int,
        required: tuple[str, ...],
        exclusive_pairs: tuple[tuple[str, str], ...],
        limit: int,
        before: int,
        after: int,
        causes: tuple[str, ...],
        direction: str,
        depth: int,
        node_limit: int,
    ) -> dict[str, object]:
        """Diff two frozen cascade slices taken at historical checkpoints.

        The read-only two-checkpoint companion of :meth:`cascade_slice`: it
        is called with every ordinary argument of that query -- none has a
        default -- followed by the ``before`` and ``after`` checkpoint
        indices and then the existing cause set, expansion direction,
        edge-count depth and selected-node cap. The two indices select
        historical positions aligned across every series plan; each side
        uses only the prefix of checkpoints from the first point up to and
        including its index, and both prefix slices are answered from one
        shared read-only historical view, reusing the existing cause,
        direction, depth and node-limit semantics exactly.

        The ordinary inputs are validated first in exactly
        :meth:`cascade_slice`'s order, before any state is consulted.
        ``before`` and ``after`` are then validated in that order: each
        must be a non-``bool`` :class:`int` (a :class:`bool` or any
        non-:class:`int` value raises :class:`TypeError`), a negative or
        out-of-range index raises :class:`ValueError`, and a ``before``
        greater than ``after`` raises :class:`ValueError`; equal indices
        are allowed. The range is the shared checkpoint count of the
        series plans (zero plans leave no valid index). The existing
        slice inputs -- ``causes``, ``direction``, ``depth`` and
        ``node_limit`` -- are validated afterwards in their existing
        order. Only then do the branch and historical-node lookups,
        checkpoint alignment and the candidate-count cap run, all on the
        one frozen view, keeping the existing :class:`KeyError`/
        :class:`ValueError` contracts and lookup order; any cap breach on
        either side raises before any result is returned.

        Cause events are handled in input order; each cause is first
        required on the ``before`` side and then on the ``after`` side,
        so a cause absent from the corresponding frozen prefix slice
        raises :class:`KeyError` on the first such side in that order.
        An empty ``causes`` tuple still completes every input and state
        check and returns two empty slices with an empty change set.

        The result is a fresh dict whose keys are ordered ``before,
        after, changes``: ``before`` and ``after`` are each a fresh
        ``nodes, edges, gaps`` slice dict item-wise equal to the existing
        :meth:`cascade_slice` result on that checkpoint prefix.
        ``changes`` is a tuple of fresh records grouped by nodes, then
        edges, then gaps; within each group the records run removed,
        added and changed, in that order. Each record is a fresh dict
        whose keys are ordered ``kind, identity, before, after``:
        ``kind`` is ``node_removed``/``node_added``/``node_changed``,
        ``edge_*`` or ``gap_*``; ``identity`` pairs the node's cause
        event with its state key, names an edge solely by the node
        identities of its two endpoints (slice position numbers are
        first remapped back to node identities, never used directly), or
        pairs a gap's interval with its member combination; the missing
        side is ``None`` and the present side is a fully isolated copy of
        that side's slice node, edge or gap record. Objects at every
        level are freshly built and share nothing with each other or
        internal state. The query is read-only: success or failure never
        modifies the event graph, branch heads, audit or idempotency
        records, or any existing query, and the existing cascade,
        single-point slice and turning-point explanation behavior is
        unchanged.
        """
        # Ordinary inputs are validated first, in exactly the existing
        # order, but no branch or historical node is looked up yet.
        validated = self._validate_frontier_inputs(
            reference,
            series,
            base_scenario,
            axis,
            values,
            min_size,
            max_size,
            required,
            exclusive_pairs,
            limit,
        )

        # The two checkpoint indices come next, each validated fully in
        # before/after order (type, then sign, then range), followed by the
        # before <= after cross-check, all before the existing slice inputs.
        # The shared checkpoint count is pure validated input data, so the
        # bounds need no state.
        point_count = self._frontier_point_count(validated)
        before_value = self._require_checkpoint_index(before, "before")
        if before_value < 0:
            raise ValueError("before checkpoint index must be non-negative")
        if before_value >= point_count:
            raise ValueError(
                f"before checkpoint index {before_value} is out of range "
                f"for {point_count} checkpoint(s)"
            )
        after_value = self._require_checkpoint_index(after, "after")
        if after_value < 0:
            raise ValueError("after checkpoint index must be non-negative")
        if after_value >= point_count:
            raise ValueError(
                f"after checkpoint index {after_value} is out of range "
                f"for {point_count} checkpoint(s)"
            )
        if before_value > after_value:
            raise ValueError(
                f"before checkpoint index {before_value} must not be greater "
                f"than after checkpoint index {after_value}"
            )

        causes, direction, depth_value, node_limit_value = (
            self._validate_slice_inputs(causes, direction, depth, node_limit)
        )

        # Capture the single read-only historical view once; both prefixes
        # are taken from it, so the two slices never disagree about history.
        prepared = self._capture_frontier_view(validated)
        before_prepared = self._prefix_frontier_view(
            prepared, before_value + 1
        )
        after_prepared = self._prefix_frontier_view(
            prepared, after_value + 1
        )

        if not causes:
            # Every input and state check is done; an empty set never trips
            # either side's node cap.
            return {
                "before": {"nodes": (), "edges": (), "gaps": ()},
                "after": {"nodes": (), "edges": (), "gaps": ()},
                "changes": (),
            }

        before_cascade = self._build_decision_cascade(before_prepared)
        after_cascade = self._build_decision_cascade(after_prepared)
        before_nodes = before_cascade["nodes"]
        after_nodes = after_cascade["nodes"]

        # Cause presence is interleaved by cause input order: the before
        # side is checked first for each cause, then the after side.
        before_positions: dict[str, list[int]] = {}
        for index, node in enumerate(before_nodes):
            before_positions.setdefault(node["cause"], []).append(index)
        after_positions: dict[str, list[int]] = {}
        for index, node in enumerate(after_nodes):
            after_positions.setdefault(node["cause"], []).append(index)
        before_starts: list[int] = []
        after_starts: list[int] = []
        for cause in causes:
            before_indices = before_positions.get(cause)
            if before_indices is None:
                raise KeyError(cause)
            after_indices = after_positions.get(cause)
            if after_indices is None:
                raise KeyError(cause)
            before_starts.extend(before_indices)
            after_starts.extend(after_indices)

        before_selected = self._expand_slice_nodes(
            before_nodes,
            before_cascade["edges"],
            before_starts,
            direction,
            depth_value,
        )
        after_selected = self._expand_slice_nodes(
            after_nodes,
            after_cascade["edges"],
            after_starts,
            direction,
            depth_value,
        )
        # Both reachable sets are known before the caps are enforced, so an
        # over-cap side (before first) returns no partial diff.
        if len(before_selected) > node_limit_value:
            raise ValueError(
                f"cascade slice node limit exceeded: {len(before_selected)} "
                f"nodes selected on the before slice, limit is "
                f"{node_limit_value}"
            )
        if len(after_selected) > node_limit_value:
            raise ValueError(
                f"cascade slice node limit exceeded: {len(after_selected)} "
                f"nodes selected on the after slice, limit is "
                f"{node_limit_value}"
            )

        before_slice = self._assemble_cascade_slice(
            before_cascade, before_selected
        )
        after_slice = self._assemble_cascade_slice(
            after_cascade, after_selected
        )
        changes = self._build_slice_changes(before_slice, after_slice)
        return {
            "before": before_slice,
            "after": after_slice,
            "changes": changes,
        }

    def cascade_slice_timeline(
        self,
        reference: str,
        series: dict[str, tuple[tuple[str, str], ...]],
        base_scenario: dict[str, object],
        axis: str,
        values: tuple[int | float, ...],
        min_size: int,
        max_size: int,
        required: tuple[str, ...],
        exclusive_pairs: tuple[tuple[str, str], ...],
        limit: int,
        indices: tuple[int, ...],
        causes: tuple[str, ...],
        direction: str,
        depth: int,
        node_limit: int,
        change_limit: int,
    ) -> dict[str, object]:
        """Snapshot frozen cascade slices at several historical checkpoints.

        The read-only multi-checkpoint companion of
        :meth:`cascade_slice_diff`: it is called with every ordinary
        argument of that query -- none has a default -- with ``indices``
        in place of the ``before``/``after`` pair and ``change_limit``
        appended after ``node_limit``. The indices select historical
        positions aligned across every series plan; each position uses
        only the prefix of checkpoints from the first point up to and
        including its index, and every prefix slice is answered from one
        shared read-only historical view, so the batch never reads
        different branch states per segment.

        The ordinary inputs are validated first in exactly
        :meth:`cascade_slice_diff`'s order, before any state is
        consulted. ``indices`` is validated next: it must be a tuple
        (else :class:`TypeError`) whose elements, walked in input order,
        are each a non-``bool`` :class:`int` (a :class:`bool` or any
        non-:class:`int` element raises :class:`TypeError`); a negative
        or out-of-range index, a duplicated index or a sequence that is
        not strictly increasing raises :class:`ValueError`. The range is
        the shared checkpoint count of the series plans (zero plans leave
        no valid index). The existing slice inputs -- ``causes``,
        ``direction``, ``depth`` and ``node_limit`` -- are validated
        afterwards in their existing order, and ``change_limit`` last:
        it must be a non-``bool`` :class:`int` (else :class:`TypeError`)
        and at least one (else :class:`ValueError`). Only then do the
        branch and historical-node lookups, checkpoint alignment and the
        candidate-count cap run, all on the one frozen view, keeping the
        existing :class:`KeyError`/:class:`ValueError` contracts and
        lookup order.

        Cause events are handled in input order; a cause event absent
        from every selected slice raises :class:`KeyError` (the first
        such cause in input order). Each position's reachable set is
        counted after the direction-and-depth expansion, and when any
        position's set exceeds ``node_limit`` a :class:`ValueError` is
        raised before any slice is assembled, so no partial timeline is
        returned. The change records of every segment are counted
        together; when their total exceeds ``change_limit`` a
        :class:`ValueError` is raised and nothing is returned. An empty
        ``indices`` still completes the ordinary-input, slice-input,
        branch and historical-node checks, then returns two empty
        tuples. An empty ``causes`` tuple still completes every input
        and state check and yields an empty slice at every selected
        position with an empty change set on every segment, never
        tripping the node cap.

        The result is a fresh dict whose keys are ordered ``snapshots,
        segments``. ``snapshots`` is a tuple with one fresh dict per
        index in its original order, whose keys are ordered ``index,
        slice``: the checkpoint index and a fresh ``nodes, edges, gaps``
        slice dict item-wise equal to the existing :meth:`cascade_slice`
        result on that checkpoint prefix. ``segments`` compares adjacent
        indices only: one fresh dict per adjacent pair in index order,
        whose keys are ordered ``before, after, changes`` -- the two
        adjacent checkpoint indices and the tuple of fresh change
        records between their slices, with identity, classification and
        ordering exactly as :meth:`cascade_slice_diff` produces them
        (nodes, then edges, then gaps; removed, added and changed within
        each group). Objects at every level are freshly built and share
        nothing with each other or internal state, so repeated calls
        return item-wise equal results. The query is read-only: success
        or failure never modifies the event graph, branch heads, audit
        or idempotency records, or any existing query, and the existing
        cascade, single-point slice, two-point diff and turning-point
        explanation behavior is unchanged.

        The trailing ``token`` is optional and defaults to ``None``;
        omitting it keeps the existing positional signature, validation
        order and return structure exactly as before. When a token from
        :meth:`create_snapshot` is given, every ordinary parameter is
        validated first (a non-``str`` token then raises
        :class:`TypeError`, an empty string :class:`ValueError`, and an
        unknown, released or foreign token :class:`KeyError`), one read
        of the token's allowance is reserved atomically, and the branch
        and historical-node checks run only inside the snapshot's frozen
        view: branches or events created after the token are invisible,
        so a missing branch or node keeps raising :class:`KeyError` and
        an out-of-range alignment or count keeps raising
        :class:`ValueError`. A failed state check never consumes the
        reserved read; when the allowance is exhausted the token is
        expired and further token queries raise :class:`RuntimeError`.
        """
        # Ordinary inputs are validated first, in exactly the existing
        # order, but no branch or historical node is looked up yet.
        validated = self._validate_frontier_inputs(
            reference,
            series,
            base_scenario,
            axis,
            values,
            min_size,
            max_size,
            required,
            exclusive_pairs,
            limit,
        )

        # The checkpoint indices come next, walked in input order (type,
        # then sign, then range, then duplicates and monotonicity). The
        # shared checkpoint count is pure validated input data, so the
        # bounds need no state.
        point_count = self._frontier_point_count(validated)
        index_values = self._validate_slice_indices(indices, point_count)

        causes, direction, depth_value, node_limit_value = (
            self._validate_slice_inputs(causes, direction, depth, node_limit)
        )
        change_limit_value = BranchStore._require_record_limit(
            change_limit, "change_limit"
        )

        # Capture the single read-only historical view once; every prefix
        # is taken from it, so the slices never disagree about history.
        prepared = self._capture_frontier_view(validated)
        snapshots, segments = self._cascade_slice_timeline_data(
            prepared,
            index_values,
            causes,
            direction,
            depth_value,
            node_limit_value,
            change_limit_value,
        )
        return {"snapshots": snapshots, "segments": segments}

    def cascade_slice_lifetimes(
        self,
        reference: str,
        series: dict[str, tuple[tuple[str, str], ...]],
        base_scenario: dict[str, object],
        axis: str,
        values: tuple[int | float, ...],
        min_size: int,
        max_size: int,
        required: tuple[str, ...],
        exclusive_pairs: tuple[tuple[str, str], ...],
        limit: int,
        indices: tuple[int, ...],
        causes: tuple[str, ...],
        direction: str,
        depth: int,
        node_limit: int,
        change_limit: int,
        lifetime_limit: int,
        token: str | None = None,
    ) -> dict[str, object]:
        """Summarize evidence identity lifetimes over a slice timeline.

        The read-only lifetime companion of
        :meth:`cascade_slice_timeline`: it is called with every argument
        of that query -- none has a default -- followed by
        ``lifetime_limit``. The ``indices`` window is the whole
        observation window: the query reads only the selected snapshots
        and the adjacent change segments of the one frozen timeline that
        :meth:`cascade_slice_timeline` assembles on the shared
        historical view, never any other historical position.

        Validation runs in exactly :meth:`cascade_slice_timeline`'s
        order -- the ordinary inputs, then ``indices``, then the
        existing slice inputs ``causes``, ``direction``, ``depth`` and
        ``node_limit``, then ``change_limit`` -- and ``lifetime_limit``
        is validated last: it must be a non-``bool`` :class:`int` (else
        :class:`TypeError`) and at least one (else :class:`ValueError`).
        Only then do the branch and historical-node lookups, checkpoint
        alignment and the candidate-count cap run, all on the one frozen
        view, keeping the existing :class:`KeyError`/:class:`ValueError`
        contracts and lookup order; the per-position node cap and the
        total change cap are enforced exactly as in
        :meth:`cascade_slice_timeline`.

        Nodes are identified by their cause event and state key, edges
        solely by the node identities of their two endpoints and gaps by
        their interval and member combination. The result is a fresh
        dict whose only key is ``lifetimes``: a tuple of fresh dicts,
        one per identity that appears in at least one selected snapshot
        -- an identity that never appears produces no record. Each
        record's keys are ordered ``type, identity, first_seen,
        last_seen, intervals, transitions``: ``type`` is ``node``,
        ``edge`` or ``gap``; ``first_seen`` and ``last_seen`` are the
        first and last selected checkpoint indices the identity appears
        at; ``intervals`` holds the closed checkpoint-index intervals of
        presence in time order, continuing only while adjacent selected
        snapshots both carry the identity -- a disappearance followed by
        a reappearance opens a new interval; ``transitions`` collects
        the identity's additions, removals and content changes segment
        by segment in time order, each a fresh dict keeping the original
        change classification and isolated copies of both sides'
        evidence. Evidence already present in the first selected
        snapshot counts toward the lifetime only; no addition change is
        fabricated for it. Records are ordered by ``node``, then
        ``edge``, then ``gap``; within each type they are stably sorted
        by first-appearance index, ties keeping the existing identity
        order.

        When the record count exceeds ``lifetime_limit`` a
        :class:`ValueError` is raised and no partial result is returned.
        An empty ``indices`` still completes every input and state
        check, then returns an empty tuple; an empty ``causes`` tuple
        does the same. Objects at every level are freshly built and
        share nothing with each other or internal state, so repeated
        calls return item-wise equal results. The query is read-only:
        success or failure never modifies the event graph, branch heads,
        audit or idempotency records, or any existing query, and the
        existing cascade, slice, diff, timeline and turning-point
        explanation behavior is unchanged.

        The trailing ``token`` is optional and defaults to ``None``;
        omitting it preserves the existing positional signature,
        validation order and return structure. When a snapshot token is
        given, all ordinary parameters are validated first, then the
        token (non-``str`` :class:`TypeError`, empty :class:`ValueError`,
        unknown/released/foreign :class:`KeyError`), one read is
        reserved atomically, and both windows' state checks run inside
        the frozen view -- later branch or event changes stay invisible
        while existing :class:`KeyError`/:class:`ValueError` contracts
        hold; a failed state check refunds the read, and an exhausted
        token raises :class:`RuntimeError`.
        """
        # Ordinary inputs are validated first, in exactly the existing
        # order, but no branch or historical node is looked up yet.
        validated = self._validate_frontier_inputs(
            reference,
            series,
            base_scenario,
            axis,
            values,
            min_size,
            max_size,
            required,
            exclusive_pairs,
            limit,
        )

        # The checkpoint indices come next, then the existing slice inputs
        # and change_limit in their existing order; lifetime_limit is
        # validated last, all before any state is queried.
        point_count = self._frontier_point_count(validated)
        index_values = self._validate_slice_indices(indices, point_count)

        causes, direction, depth_value, node_limit_value = (
            self._validate_slice_inputs(causes, direction, depth, node_limit)
        )
        change_limit_value = BranchStore._require_record_limit(
            change_limit, "change_limit"
        )
        lifetime_limit_value = BranchStore._require_record_limit(
            lifetime_limit, "lifetime_limit"
        )

        # Token checks come after every ordinary parameter. Without a token
        # the query runs on live state exactly as before; with one, the read
        # allowance is reserved atomically before any state check and given
        # back if a state check (KeyError/ValueError below) fails, so only
        # fully successful queries consume a read.
        view = self
        if token is not None:
            view = self._reserve_snapshot_read(token)
        try:
            # Capture the single read-only historical view once; the
            # selected snapshots and adjacent segments are taken from it,
            # so the lifetimes never read different branch states per
            # position. For a token query the view is the frozen store
            # captured at token creation, so later branch changes are
            # invisible.
            prepared = view._capture_frontier_view(validated)
            snapshots, segments = view._cascade_slice_timeline_data(
                prepared,
                index_values,
                causes,
                direction,
                depth_value,
                node_limit_value,
                change_limit_value,
            )
            lifetimes = view._build_slice_lifetimes(snapshots, segments)

            # The full record set is known before the cap is enforced, so an
            # over-cap call returns no partial lifetimes.
            if len(lifetimes) > lifetime_limit_value:
                raise ValueError(
                    f"cascade slice lifetimes lifetime limit exceeded: "
                    f"{len(lifetimes)} lifetime records, limit is "
                    f"{lifetime_limit_value}"
                )
        except BaseException:
            if token is not None:
                self._refund_snapshot_read(token)
            raise
        return {"lifetimes": lifetimes}

    def compare_lifetimes(
        self,
        reference: str,
        series: dict[str, tuple[tuple[str, str], ...]],
        base_scenario: dict[str, object],
        axis: str,
        values: tuple[int | float, ...],
        min_size: int,
        max_size: int,
        required: tuple[str, ...],
        exclusive_pairs: tuple[tuple[str, str], ...],
        limit: int,
        left_indices: tuple[int, ...],
        right_indices: tuple[int, ...],
        causes: tuple[str, ...],
        direction: str,
        depth: int,
        node_limit: int,
        change_limit: int,
        lifetime_limit: int,
        diff_limit: int,
        token: str | None = None,
    ) -> dict[str, object]:
        """Diff the identity lifetimes of two slice-timeline windows.

        The read-only window-comparison companion of
        :meth:`cascade_slice_lifetimes`: it is called with every argument
        of that query -- none has a default -- except that the single
        ``indices`` window is replaced by ``left_indices`` and
        ``right_indices`` and a ``diff_limit`` is appended after
        ``lifetime_limit``. Each window selects historical positions
        aligned across every series plan and summarizes only its own
        selected snapshots and adjacent change segments; both windows are
        answered from the one frozen historical view captured by
        :meth:`_capture_frontier_view`, so the two sides never read
        different branch states.

        Validation runs in exactly :meth:`cascade_slice_lifetimes`'s
        order -- the ordinary inputs first, then the two windows in
        left/right order, each checked exactly like ``indices`` (a tuple,
        else :class:`TypeError`; non-``bool`` :class:`int` elements
        walked in input order, a :class:`bool` or any non-:class:`int`
        element raising :class:`TypeError`; a negative or out-of-range
        index, a duplicate or a non-increasing sequence raising
        :class:`ValueError`, reported per window in input order), then
        the existing slice inputs ``causes``, ``direction``, ``depth``
        and ``node_limit``, then ``change_limit`` and ``lifetime_limit``
        -- and ``diff_limit`` is validated last: it must be a
        non-``bool`` :class:`int` (else :class:`TypeError`) and at least
        one (else :class:`ValueError`). Only then do the branch and
        historical-node lookups, checkpoint alignment and the
        candidate-count cap run, all on the one frozen view, keeping the
        existing :class:`KeyError`/:class:`ValueError` contracts and
        lookup order; the per-position node cap and the total change cap
        are enforced per window exactly as in
        :meth:`cascade_slice_timeline`, and ``lifetime_limit`` caps each
        window's record count, the left window first.

        Nodes are identified by their cause event and state key, edges
        solely by the node identities of their two endpoints and gaps by
        their interval and member combination, exactly as in
        :meth:`cascade_slice_lifetimes`. An identity found only in the
        right window is ``added``, one found only in the left window is
        ``removed``; an identity present in both whose first/last
        position, presence intervals or classification transitions
        differ is ``changed``, and an identity whose records are
        completely equal is omitted. The result is a fresh dict whose
        keys are ordered ``left, right, changes``: ``left`` and ``right``
        are isolated copies of each window's full lifetime tuple, and
        ``changes`` is a tuple of fresh records whose keys are ordered
        ``kind, identity, before, after`` -- ``kind`` is
        ``node_removed``/``node_added``/``node_changed``, ``edge_*`` or
        ``gap_*``, the missing side is ``None`` and each present side is
        an isolated copy of that window's lifetime record. Changes are
        grouped by nodes, then edges, then gaps; within each group the
        records run removed, added and changed, removed and changed
        records following the left window's identity order and added
        records the right window's, so the result is stable and
        repeatable.

        When either window's record count exceeds ``lifetime_limit``, or
        the change count exceeds ``diff_limit``, a :class:`ValueError` is
        raised and no partial result is returned. An empty window still
        completes every input and state check and yields an empty
        lifetime tuple on its side; when both windows are empty all three
        entries are empty tuples. Objects at every level are freshly
        built and share nothing with each other or internal state, so
        repeated calls return item-wise equal results. The query is
        read-only: success or failure never modifies the event graph,
        branch heads, audit or idempotency records, or any existing
        query, and the existing cascade, slice, diff, timeline, lifetime
        and turning-point explanation behavior is unchanged.

        The trailing ``token`` is optional and defaults to ``None``;
        omitting it preserves the existing positional signature,
        validation order and return structure. When a snapshot token is
        given, all ordinary parameters are validated first, then the
        token (non-``str`` :class:`TypeError`, empty :class:`ValueError`,
        unknown/released/foreign :class:`KeyError`), one read is
        reserved atomically, and every window's state checks run inside
        the frozen view -- later branch or event changes stay invisible
        while existing :class:`KeyError`/:class:`ValueError` contracts
        hold; a failed state check refunds the read, and an exhausted
        token raises :class:`RuntimeError`.
        """
        # Ordinary inputs are validated first, in exactly the existing
        # order, but no branch or historical node is looked up yet.
        validated = self._validate_frontier_inputs(
            reference,
            series,
            base_scenario,
            axis,
            values,
            min_size,
            max_size,
            required,
            exclusive_pairs,
            limit,
        )

        # The two windows come next in left/right order, each checked
        # exactly like the lifetime query's indices, then the existing
        # slice inputs and change_limit in their existing order;
        # lifetime_limit and diff_limit are validated last, all before
        # any state is queried.
        point_count = self._frontier_point_count(validated)
        left_index_values = self._validate_slice_indices(
            left_indices, point_count, "left_indices"
        )
        right_index_values = self._validate_slice_indices(
            right_indices, point_count, "right_indices"
        )

        causes, direction, depth_value, node_limit_value = (
            self._validate_slice_inputs(causes, direction, depth, node_limit)
        )
        change_limit_value = BranchStore._require_record_limit(
            change_limit, "change_limit"
        )
        lifetime_limit_value = BranchStore._require_record_limit(
            lifetime_limit, "lifetime_limit"
        )
        diff_limit_value = BranchStore._require_record_limit(
            diff_limit, "diff_limit"
        )

        # Token checks come after every ordinary parameter. Without a token
        # the query runs on live state exactly as before; with one, the read
        # allowance is reserved atomically before any state check and given
        # back if a state check (KeyError/ValueError below) fails, so only
        # fully successful queries consume a read. Both windows are then
        # answered from the token's one frozen view.
        view = self
        if token is not None:
            view = self._reserve_snapshot_read(token)
        try:
            # Capture the single read-only historical view once; both
            # windows select their snapshots and adjacent segments from it,
            # so the two sides never disagree about history.
            prepared = view._capture_frontier_view(validated)
            left_snapshots, left_segments = view._cascade_slice_timeline_data(
                prepared,
                left_index_values,
                causes,
                direction,
                depth_value,
                node_limit_value,
                change_limit_value,
            )
            right_snapshots, right_segments = view._cascade_slice_timeline_data(
                prepared,
                right_index_values,
                causes,
                direction,
                depth_value,
                node_limit_value,
                change_limit_value,
            )
            left_lifetimes = view._build_slice_lifetimes(
                left_snapshots, left_segments
            )
            right_lifetimes = view._build_slice_lifetimes(
                right_snapshots, right_segments
            )

            # Both full record sets are known before either cap is enforced,
            # so an over-cap call returns no partial lifetimes.
            if len(left_lifetimes) > lifetime_limit_value:
                raise ValueError(
                    f"compare lifetimes lifetime limit exceeded: "
                    f"{len(left_lifetimes)} lifetime records on the left "
                    f"window, limit is {lifetime_limit_value}"
                )
            if len(right_lifetimes) > lifetime_limit_value:
                raise ValueError(
                    f"compare lifetimes lifetime limit exceeded: "
                    f"{len(right_lifetimes)} lifetime records on the right "
                    f"window, limit is {lifetime_limit_value}"
                )

            changes = view._build_lifetime_changes(left_lifetimes, right_lifetimes)
            if len(changes) > diff_limit_value:
                raise ValueError(
                    f"compare lifetimes diff limit exceeded: "
                    f"{len(changes)} change records, limit is "
                    f"{diff_limit_value}"
                )
            result = {
                "left": tuple(
                    view._copy_lifetime_record(record)
                    for record in left_lifetimes
                ),
                "right": tuple(
                    view._copy_lifetime_record(record)
                    for record in right_lifetimes
                ),
                "changes": changes,
            }
        except BaseException:
            if token is not None:
                self._refund_snapshot_read(token)
            raise
        return result

    def lifetime_evolution(
        self,
        reference: str,
        series: dict[str, tuple[tuple[str, str], ...]],
        base_scenario: dict[str, object],
        axis: str,
        values: tuple[int | float, ...],
        min_size: int,
        max_size: int,
        required: tuple[str, ...],
        exclusive_pairs: tuple[tuple[str, str], ...],
        limit: int,
        windows: tuple[tuple[int, ...], ...],
        causes: tuple[str, ...],
        direction: str,
        depth: int,
        node_limit: int,
        change_limit: int,
        lifetime_limit: int,
        diff_limit: int,
        window_limit: int,
        total_diff_limit: int,
        token: str | None = None,
    ) -> dict[str, object]:
        """Follow identity-lifetime evolution across many slice windows.

        The read-only multi-window extension of :meth:`compare_lifetimes`:
        it is called with every ordinary argument of that query -- none
        has a default -- except that the ``left_indices``/
        ``right_indices`` pair is replaced by a single ``windows`` tuple
        and, after ``diff_limit``, a ``window_limit`` and a
        ``total_diff_limit`` are appended. ``windows`` is a tuple of
        index tuples in input order; each window independently selects
        the historical positions aligned across every series plan and is
        summarized with exactly :meth:`cascade_slice_lifetimes`'s
        lifetime semantics. Every window and every adjacent pair is
        answered from the one frozen historical view captured by
        :meth:`_capture_frontier_view`, so the whole batch reads the
        same branch state and never sees a head move mid-query.

        Validation runs in exactly :meth:`compare_lifetimes`'s order --
        the ordinary inputs first, then ``windows`` and each window in
        input order -- followed by the existing slice inputs ``causes``,
        ``direction``, ``depth`` and ``node_limit``, then
        ``change_limit``, ``lifetime_limit`` and ``diff_limit``, with
        ``window_limit`` and ``total_diff_limit`` validated last of all;
        each cap must be a non-``bool`` :class:`int` (else
        :class:`TypeError`) and at least one (else :class:`ValueError`).
        ``windows`` or any window that is not a tuple raises
        :class:`TypeError`; a :class:`bool` or any non-:class:`int`
        index raises :class:`TypeError`; a negative, out-of-range,
        duplicate or non-strictly-increasing index raises
        :class:`ValueError`. Every window error names the window's
        zero-based position in ``windows`` -- including a duplicated
        index -- and the checks hold only within a window, so identical
        windows may appear next to each other. Only then do the branch
        and historical-node lookups, checkpoint alignment and the
        candidate-count cap run, all on the one frozen view, keeping the
        existing :class:`KeyError`/:class:`ValueError` contracts and
        lookup order; the per-position node cap and the total change cap
        are enforced per window exactly as in
        :meth:`cascade_slice_timeline`, and ``lifetime_limit`` caps each
        window's record count.

        The result is a fresh dict whose keys are ordered ``windows,
        segments``, both fresh tuples. Each window contributes a fresh
        dict whose keys are ordered ``position, lifetimes``: ``position``
        is the window's zero-based index in ``windows`` and
        ``lifetimes`` is the window's complete, isolated lifetime tuple,
        identical to :meth:`cascade_slice_lifetimes` on that window.
        Each adjacent pair of windows forms one segment, one fresh dict
        per pair in window order whose keys are ordered ``left, right,
        changes``: the two window positions and the complete change
        tuple between their lifetimes, with node, edge and gap identity,
        classification and stable ordering exactly as
        :meth:`compare_lifetimes` produces. Two completely identical
        adjacent windows still keep their segment; its change tuple is
        simply empty.

        When the window count exceeds ``window_limit``, any window
        exceeds the existing node, change or lifetime cap, any segment's
        change count exceeds ``diff_limit``, or the total change count
        over every segment exceeds ``total_diff_limit``, a
        :class:`ValueError` is raised and no partial result is returned.
        Empty ``windows`` still completes every input and state check,
        then returns two empty tuples. Unknown branches, historical
        nodes or cause events raise :class:`KeyError`; checkpoint
        misalignment and the candidate cap raise :class:`ValueError`.
        Objects at every level are freshly built and share nothing with
        each other or internal state, so repeated calls return
        item-wise equal results. The query is read-only: success or
        failure never modifies the event graph, branch heads, audit or
        idempotency records, or any existing query.
        """
        # Ordinary inputs are validated first, in exactly the existing
        # order, but no branch or historical node is looked up yet.
        validated = self._validate_frontier_inputs(
            reference,
            series,
            base_scenario,
            axis,
            values,
            min_size,
            max_size,
            required,
            exclusive_pairs,
            limit,
        )

        # The windows batch and each window in input order come next,
        # each window checked exactly like the two-window query's index
        # tuples but labeled with its position in windows. The existing
        # slice inputs and the caps follow, window_limit and
        # total_diff_limit last, all before any state is queried.
        point_count = self._frontier_point_count(validated)
        index_windows = self._validate_lifetime_windows(windows, point_count)

        causes, direction, depth_value, node_limit_value = (
            self._validate_slice_inputs(causes, direction, depth, node_limit)
        )
        change_limit_value = BranchStore._require_record_limit(
            change_limit, "change_limit"
        )
        lifetime_limit_value = BranchStore._require_record_limit(
            lifetime_limit, "lifetime_limit"
        )
        diff_limit_value = BranchStore._require_record_limit(
            diff_limit, "diff_limit"
        )
        window_limit_value = BranchStore._require_record_limit(
            window_limit, "window_limit"
        )
        total_diff_limit_value = BranchStore._require_record_limit(
            total_diff_limit, "total_diff_limit"
        )
        if len(index_windows) > window_limit_value:
            raise ValueError(
                f"lifetime evolution window limit exceeded: "
                f"{len(index_windows)} windows, limit is "
                f"{window_limit_value}"
            )

        # Token checks come after every ordinary parameter. Without a token
        # the query runs on live state exactly as before; with one, the read
        # allowance is reserved atomically before any state check and given
        # back if a state check (KeyError/ValueError below) fails, so only
        # fully successful queries consume a read. Every window is then
        # answered from the token's one frozen view.
        view = self
        if token is not None:
            view = self._reserve_snapshot_read(token)
        try:
            # Capture the single read-only historical view once; every
            # window selects its snapshots and adjacent segments from it,
            # so the batch never reads different branch states per window.
            prepared = view._capture_frontier_view(validated)

            window_records: list[tuple[dict[str, object], ...]] = []
            for position, index_values in enumerate(index_windows):
                snapshots, segments = view._cascade_slice_timeline_data(
                    prepared,
                    index_values,
                    causes,
                    direction,
                    depth_value,
                    node_limit_value,
                    change_limit_value,
                )
                lifetimes = view._build_slice_lifetimes(snapshots, segments)
                if len(lifetimes) > lifetime_limit_value:
                    raise ValueError(
                        f"lifetime evolution lifetime limit exceeded: "
                        f"{len(lifetimes)} lifetime records on window at "
                        f"position {position}, limit is "
                        f"{lifetime_limit_value}"
                    )
                window_records.append(lifetimes)

            # Every segment's full change set is known before either cap is
            # enforced, so an over-cap batch returns no partial evolution.
            segment_changes: list[tuple[dict[str, object], ...]] = []
            total_changes = 0
            for position in range(len(window_records) - 1):
                changes = view._build_lifetime_changes(
                    window_records[position], window_records[position + 1]
                )
                if len(changes) > diff_limit_value:
                    raise ValueError(
                        f"lifetime evolution diff limit exceeded: "
                        f"{len(changes)} change records on segment between "
                        f"windows at positions {position} and {position + 1}, "
                        f"limit is {diff_limit_value}"
                    )
                total_changes += len(changes)
                segment_changes.append(changes)
            if total_changes > total_diff_limit_value:
                raise ValueError(
                    f"lifetime evolution total diff limit exceeded: "
                    f"{total_changes} change records over "
                    f"{len(segment_changes)} segment(s), limit is "
                    f"{total_diff_limit_value}"
                )

            result = {
                "windows": tuple(
                    {
                        "position": position,
                        "lifetimes": tuple(
                            view._copy_lifetime_record(record)
                            for record in lifetimes
                        ),
                    }
                    for position, lifetimes in enumerate(window_records)
                ),
                "segments": tuple(
                    {
                        "left": position,
                        "right": position + 1,
                        "changes": changes,
                    }
                    for position, changes in enumerate(segment_changes)
                ),
            }
        except BaseException:
            if token is not None:
                self._refund_snapshot_read(token)
            raise
        return result

    @staticmethod
    def _freeze_lifetime_value(value: object) -> object:
        """Return an isolated tuple/dict-only copy of a lifetime value."""
        if isinstance(value, dict):
            return {
                key: BranchStore._freeze_lifetime_value(item)
                for key, item in value.items()
            }
        if isinstance(value, (tuple, list)):
            return tuple(
                BranchStore._freeze_lifetime_value(item) for item in value
            )
        return value

    @staticmethod
    def _copy_lifetime_record(record: dict[str, object]) -> dict[str, object]:
        """Return an isolated copy of one lifetime record.

        The copy keeps the existing key order -- ``type, identity,
        first_seen, last_seen, intervals, transitions`` -- and shares no
        mutable object with the source record.
        """
        return {
            "type": record["type"],
            "identity": BranchStore._freeze_lifetime_value(
                record["identity"]
            ),
            "first_seen": record["first_seen"],
            "last_seen": record["last_seen"],
            "intervals": tuple(
                tuple(interval) for interval in record["intervals"]
            ),
            "transitions": tuple(
                {
                    "kind": transition["kind"],
                    "before": BranchStore._freeze_lifetime_value(
                        transition["before"]
                    ),
                    "after": BranchStore._freeze_lifetime_value(
                        transition["after"]
                    ),
                }
                for transition in record["transitions"]
            ),
        }

    @staticmethod
    def _build_lifetime_changes(
        left_lifetimes: tuple[dict[str, object], ...],
        right_lifetimes: tuple[dict[str, object], ...],
    ) -> tuple[dict[str, object], ...]:
        """Compare two windows' lifetime records type by type, by identity.

        Records are grouped by nodes, then edges, then gaps; each group
        runs removed, added and changed. Removed and changed records
        follow the left window's identity order, added records the right
        window's, so the output is stable. An identity present in both
        windows changes when its first/last position, presence intervals
        or classification transitions differ; a completely equal pair is
        omitted. Every present side is copied again, so the change
        records share no object with either returned lifetime tuple.
        """
        changes: list[dict[str, object]] = []
        for type_name in ("node", "edge", "gap"):
            left_records = [
                record
                for record in left_lifetimes
                if record["type"] == type_name
            ]
            right_records = [
                record
                for record in right_lifetimes
                if record["type"] == type_name
            ]
            left_by_identity = {
                record["identity"]: record for record in left_records
            }
            right_by_identity = {
                record["identity"]: record for record in right_records
            }
            for record in left_records:
                if record["identity"] not in right_by_identity:
                    changes.append(
                        {
                            "kind": f"{type_name}_removed",
                            "identity": BranchStore._freeze_lifetime_value(
                                record["identity"]
                            ),
                            "before": BranchStore._copy_lifetime_record(
                                record
                            ),
                            "after": None,
                        }
                    )
            for record in right_records:
                if record["identity"] not in left_by_identity:
                    changes.append(
                        {
                            "kind": f"{type_name}_added",
                            "identity": BranchStore._freeze_lifetime_value(
                                record["identity"]
                            ),
                            "before": None,
                            "after": BranchStore._copy_lifetime_record(
                                record
                            ),
                        }
                    )
            for record in left_records:
                other = right_by_identity.get(record["identity"])
                if other is None:
                    continue
                changed = (
                    record["first_seen"] != other["first_seen"]
                    or record["last_seen"] != other["last_seen"]
                    or record["intervals"] != other["intervals"]
                    or record["transitions"] != other["transitions"]
                )
                if changed:
                    changes.append(
                        {
                            "kind": f"{type_name}_changed",
                            "identity": BranchStore._freeze_lifetime_value(
                                record["identity"]
                            ),
                            "before": BranchStore._copy_lifetime_record(
                                record
                            ),
                            "after": BranchStore._copy_lifetime_record(
                                other
                            ),
                        }
                    )
        return tuple(changes)

    def _cascade_slice_timeline_data(
        self,
        prepared: dict[str, object],
        index_values: list[int],
        causes: tuple[str, ...],
        direction: str,
        depth_value: int,
        node_limit_value: int,
        change_limit_value: int,
    ) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
        """Build the frozen per-index snapshots and adjacent segments.

        Shared by :meth:`cascade_slice_timeline` and
        :meth:`cascade_slice_lifetimes` so both read the same selected
        snapshots and adjacent change segments from the one captured
        view. Every input and state check has already run; an empty
        index list selects nothing, an empty cause set yields an empty
        slice per position, and the per-position node cap and the total
        change cap are enforced before any result is assembled.
        """
        if not index_values:
            # Every input and state check is done; no position is selected.
            return (), ()

        if not causes:
            # Every input and state check is done; an empty set yields an
            # empty slice per position and never trips the node cap.
            return (
                tuple(
                    {
                        "index": index_value,
                        "slice": {"nodes": (), "edges": (), "gaps": ()},
                    }
                    for index_value in index_values
                ),
                tuple(
                    {"before": before, "after": after, "changes": ()}
                    for before, after in zip(index_values, index_values[1:])
                ),
            )

        cascades = [
            self._build_decision_cascade(
                self._prefix_frontier_view(prepared, index_value + 1)
            )
            for index_value in index_values
        ]

        # A cause event must appear in at least one position's selected
        # slice; the first cause absent everywhere raises. Start positions
        # follow cause input order, skipping positions whose prefix does
        # not carry the cause.
        positions_per_cascade: list[dict[str, list[int]]] = []
        for cascade in cascades:
            positions: dict[str, list[int]] = {}
            for position, node in enumerate(cascade["nodes"]):
                positions.setdefault(node["cause"], []).append(position)
            positions_per_cascade.append(positions)
        for cause in causes:
            if not any(
                cause in positions for positions in positions_per_cascade
            ):
                raise KeyError(cause)

        selected_per_cascade: list[set[int]] = []
        for cascade, positions in zip(cascades, positions_per_cascade):
            starts: list[int] = []
            for cause in causes:
                starts.extend(positions.get(cause, ()))
            selected_per_cascade.append(
                self._expand_slice_nodes(
                    cascade["nodes"],
                    cascade["edges"],
                    starts,
                    direction,
                    depth_value,
                )
            )

        # Every reachable set is known before any cap is enforced, so an
        # over-cap position returns no partial timeline.
        for index_value, selected in zip(index_values, selected_per_cascade):
            if len(selected) > node_limit_value:
                raise ValueError(
                    f"cascade slice node limit exceeded: {len(selected)} "
                    f"nodes selected on the slice at checkpoint index "
                    f"{index_value}, limit is {node_limit_value}"
                )

        slices = [
            self._assemble_cascade_slice(cascade, selected)
            for cascade, selected in zip(cascades, selected_per_cascade)
        ]
        snapshots = tuple(
            {"index": index_value, "slice": slice_result}
            for index_value, slice_result in zip(index_values, slices)
        )

        segments: list[dict[str, object]] = []
        change_count = 0
        for position in range(len(slices) - 1):
            changes = self._build_slice_changes(
                slices[position], slices[position + 1]
            )
            change_count += len(changes)
            segments.append(
                {
                    "before": index_values[position],
                    "after": index_values[position + 1],
                    "changes": changes,
                }
            )
        if change_count > change_limit_value:
            raise ValueError(
                f"cascade slice timeline change limit exceeded: "
                f"{change_count} change records, limit is "
                f"{change_limit_value}"
            )
        return snapshots, tuple(segments)

    @staticmethod
    def _build_slice_lifetimes(
        snapshots: tuple[dict[str, object], ...],
        segments: tuple[dict[str, object], ...],
    ) -> tuple[dict[str, object], ...]:
        """Aggregate per-identity lifetimes over frozen snapshots/segments.

        Identities never use slice position numbers: edge endpoints are
        remapped back to node identities, exactly as in
        :meth:`_build_slice_changes`. Presence is tracked per selected
        position, so an identity seen at two positions separated by an
        absence closes one interval and opens another. Transitions reuse
        the segments' own change records -- already isolated copies --
        keeping their classification and both sides' evidence; evidence
        present in the first selected snapshot has no segment before it,
        so it never gains a fabricated addition.
        """
        index_values = [snapshot["index"] for snapshot in snapshots]
        first_seen: dict[str, dict[object, int]] = {
            "node": {},
            "edge": {},
            "gap": {},
        }
        last_seen: dict[str, dict[object, int]] = {
            "node": {},
            "edge": {},
            "gap": {},
        }
        # Identity -> selected positions (ascending) where it appears.
        presence: dict[str, dict[object, list[int]]] = {
            "node": {},
            "edge": {},
            "gap": {},
        }
        transitions: dict[str, dict[object, list[dict[str, object]]]] = {
            "node": {},
            "edge": {},
            "gap": {},
        }

        for position, snapshot in enumerate(snapshots):
            index_value = snapshot["index"]
            slice_result = snapshot["slice"]
            nodes = slice_result["nodes"]
            typed_identities = {
                "node": [
                    (node["cause"], node["key"]) for node in nodes
                ],
                "edge": [
                    (
                        (
                            nodes[edge["source"]]["cause"],
                            nodes[edge["source"]]["key"],
                        ),
                        (
                            nodes[edge["target"]]["cause"],
                            nodes[edge["target"]]["key"],
                        ),
                    )
                    for edge in slice_result["edges"]
                ],
                "gap": [
                    (gap["interval"], tuple(gap["members"]))
                    for gap in slice_result["gaps"]
                ],
            }
            for type_name, identities in typed_identities.items():
                for identity in identities:
                    if identity not in first_seen[type_name]:
                        first_seen[type_name][identity] = index_value
                        presence[type_name][identity] = []
                    last_seen[type_name][identity] = index_value
                    presence[type_name][identity].append(position)

        for segment in segments:
            for change in segment["changes"]:
                type_name = change["kind"].split("_", 1)[0]
                transitions[type_name].setdefault(
                    change["identity"], []
                ).append(
                    {
                        "kind": change["kind"],
                        "before": change["before"],
                        "after": change["after"],
                    }
                )

        lifetimes: list[dict[str, object]] = []
        for type_name in ("node", "edge", "gap"):
            records: list[dict[str, object]] = []
            # Dict order is the first-encounter order, which is the
            # existing identity order within each snapshot sequence.
            for identity, first in first_seen[type_name].items():
                positions = presence[type_name][identity]
                intervals: list[tuple[int, int]] = []
                run_start = positions[0]
                previous = positions[0]
                for position in positions[1:]:
                    if position != previous + 1:
                        intervals.append(
                            (index_values[run_start], index_values[previous])
                        )
                        run_start = position
                    previous = position
                intervals.append(
                    (index_values[run_start], index_values[previous])
                )
                records.append(
                    {
                        "type": type_name,
                        "identity": identity,
                        "first_seen": first,
                        "last_seen": last_seen[type_name][identity],
                        "intervals": tuple(intervals),
                        "transitions": tuple(
                            transitions[type_name].get(identity, ())
                        ),
                    }
                )
            # Stable sort: ties on the first-appearance index keep the
            # existing identity order.
            records.sort(key=lambda record: record["first_seen"])
            lifetimes.extend(records)
        return tuple(lifetimes)

    @staticmethod
    def _validate_slice_indices(
        indices: Any, point_count: int, name: str = "indices"
    ) -> list[int]:
        """Validate one checkpoint window of the slice queries.

        Shared by :meth:`cascade_slice_timeline`,
        :meth:`cascade_slice_lifetimes` and :meth:`compare_lifetimes` so
        all apply identical checks in the same order: a tuple (else
        :class:`TypeError`) of non-``bool`` :class:`int` elements walked
        in input order, a negative or out-of-range index, a duplicate or
        a non-increasing sequence raising :class:`ValueError`. ``name``
        labels the window in every message.
        """
        if not isinstance(indices, tuple):
            raise TypeError(
                f"{name} must be a tuple, got {type(indices).__name__}"
            )
        index_values: list[int] = []
        seen_indices: set[int] = set()
        previous_index: int | None = None
        for index in indices:
            index_value = BranchStore._require_checkpoint_index(
                index, name
            )
            if index_value < 0:
                raise ValueError(
                    f"{name} checkpoint index must be non-negative"
                )
            if index_value >= point_count:
                raise ValueError(
                    f"{name} checkpoint index {index_value} is out of "
                    f"range for {point_count} checkpoint(s)"
                )
            if index_value in seen_indices:
                raise ValueError(
                    f"duplicate checkpoint index {index_value}"
                )
            if previous_index is not None and index_value <= previous_index:
                raise ValueError(
                    f"{name} checkpoint indices must be strictly increasing"
                )
            seen_indices.add(index_value)
            previous_index = index_value
            index_values.append(index_value)
        return index_values

    @staticmethod
    def _validate_lifetime_windows(
        windows: Any, point_count: int
    ) -> list[list[int]]:
        """Validate the batch of checkpoint windows for lifetime evolution.

        Used only by :meth:`lifetime_evolution`: ``windows`` must be a
        tuple (else :class:`TypeError`), walked in input order, whose
        elements are each an index tuple validated exactly like
        :meth:`_validate_slice_indices`'s ``indices`` -- a non-tuple
        window raises :class:`TypeError`, a :class:`bool` or any
        non-:class:`int` element raises :class:`TypeError`, and a
        negative, out-of-range, duplicate or non-strictly-increasing
        index raises :class:`ValueError`. Uniqueness and monotonicity
        hold only within a window: the same window may occur several
        times and two windows may share indices, so identical windows
        placed next to each other are accepted. Every message names the
        window's zero-based position in ``windows``, including the
        duplicate-index error, so a faulty window can never lose its
        position identifier.
        """
        if not isinstance(windows, tuple):
            raise TypeError(
                f"windows must be a tuple, got {type(windows).__name__}"
            )
        validated_windows: list[list[int]] = []
        for position, window in enumerate(windows):
            name = f"windows[{position}]"
            if not isinstance(window, tuple):
                raise TypeError(
                    f"{name} must be a tuple, got {type(window).__name__}"
                )
            index_values: list[int] = []
            seen_indices: set[int] = set()
            previous_index: int | None = None
            for index in window:
                index_value = BranchStore._require_checkpoint_index(
                    index, name
                )
                if index_value < 0:
                    raise ValueError(
                        f"{name} checkpoint index must be non-negative"
                    )
                if index_value >= point_count:
                    raise ValueError(
                        f"{name} checkpoint index {index_value} is out of "
                        f"range for {point_count} checkpoint(s)"
                    )
                if index_value in seen_indices:
                    raise ValueError(
                        f"{name} duplicate checkpoint index {index_value}"
                    )
                if previous_index is not None and index_value <= previous_index:
                    raise ValueError(
                        f"{name} checkpoint indices must be strictly "
                        f"increasing"
                    )
                seen_indices.add(index_value)
                previous_index = index_value
                index_values.append(index_value)
            validated_windows.append(index_values)
        return validated_windows

    @staticmethod
    def _require_record_limit(value: Any, name: str) -> int:
        """Validate one non-``bool`` record-count cap of at least one."""
        limit_value = BranchStore._require_size_bound(value, name)
        if limit_value < 1:
            raise ValueError(f"{name} must be >= 1")
        return limit_value

    @staticmethod
    def _require_checkpoint_index(value: Any, name: str) -> int:
        """Validate one non-``bool`` checkpoint index parameter."""
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                f"{name} checkpoint index must be an int, got "
                f"{type(value).__name__}"
            )
        return value

    @staticmethod
    def _frontier_point_count(validated: dict[str, object]) -> int:
        """Return the checkpoint count every series plan is aligned to."""
        per_series = validated["per_series"]
        if not per_series:
            return 0
        return len(per_series[0][1])

    @staticmethod
    def _prefix_frontier_view(
        prepared: dict[str, object], length: int
    ) -> dict[str, object]:
        """Restrict a captured frozen view to its first ``length`` checkpoints.

        Only the per-series checkpoint views are shortened; candidates,
        values, weights and budgets are unchanged, so building a cascade on
        the result equals running the existing query with series truncated
        to that checkpoint prefix. Both prefixes of a diff derive from the
        same captured view, merely taking shorter copies of its orders.
        """
        prefix_views: dict[
            str, list[tuple[list[str], list[str], str, str]]
        ] = {
            name: list(views[:length])
            for name, views in prepared["views_by_name"].items()
        }
        return {**prepared, "views_by_name": prefix_views}

    @staticmethod
    def _validate_slice_inputs(
        causes: Any,
        direction: Any,
        depth: Any,
        node_limit: Any,
    ) -> tuple[tuple[str, ...], str, int, int]:
        """Validate :meth:`cascade_slice`'s slice-specific inputs.

        Shared by :meth:`cascade_slice` and :meth:`cascade_slice_diff` so
        both apply identical checks in the same order: ``causes`` (a tuple
        of unique non-empty ``str`` ids), then ``direction``, ``depth`` and
        ``node_limit``.
        """
        if not isinstance(causes, tuple):
            raise TypeError(
                f"causes must be a tuple, got {type(causes).__name__}"
            )
        seen_causes: set[str] = set()
        for cause in causes:
            EventGraph._require_nonempty_str(cause, "cause")
            if cause in seen_causes:
                raise ValueError(f"duplicate cause {cause!r}")
            seen_causes.add(cause)
        if not isinstance(direction, str):
            raise TypeError(
                f"direction must be a str, got {type(direction).__name__}"
            )
        if direction not in ("forward", "backward", "both"):
            raise ValueError(
                "direction must be 'forward', 'backward' or 'both', got "
                f"{direction!r}"
            )
        depth_value = BranchStore._require_size_bound(depth, "depth")
        if depth_value < 0:
            raise ValueError("depth must be non-negative")
        node_limit_value = BranchStore._require_size_bound(
            node_limit, "node_limit"
        )
        if node_limit_value < 1:
            raise ValueError("node_limit must be >= 1")
        return causes, direction, depth_value, node_limit_value

    def _cascade_slice_on_prepared(
        self,
        prepared: dict[str, object],
        causes: tuple[str, ...],
        direction: str,
        depth_value: int,
        node_limit_value: int,
    ) -> dict[str, object]:
        """Build one frozen slice on a captured view using slice inputs."""
        cascade = self._build_decision_cascade(prepared)
        if not causes:
            # All validation and the state lookup are done; an empty set
            # never trips the node cap.
            return {"nodes": (), "edges": (), "gaps": ()}

        starts = self._slice_start_positions(cascade["nodes"], causes)
        selected = self._expand_slice_nodes(
            cascade["nodes"],
            cascade["edges"],
            starts,
            direction,
            depth_value,
        )

        # The cap is checked after the full reachable set is known, before
        # any slice object is built, so an over-cap call returns nothing.
        if len(selected) > node_limit_value:
            raise ValueError(
                f"cascade slice node limit exceeded: {len(selected)} nodes "
                f"selected, limit is {node_limit_value}"
            )
        return self._assemble_cascade_slice(cascade, selected)

    @staticmethod
    def _slice_start_positions(
        nodes: tuple[dict[str, object], ...],
        causes: tuple[str, ...],
    ) -> list[int]:
        """Map cause events (input order) to all their cascade node positions.

        One cause event may start several nodes (one per state key); the
        positions are appended in cascade node order. The first cause event
        absent from the cascade nodes raises :class:`KeyError`.
        """
        nodes_by_cause: dict[str, list[int]] = {}
        for index, node in enumerate(nodes):
            nodes_by_cause.setdefault(node["cause"], []).append(index)
        starts: list[int] = []
        for cause in causes:
            indices = nodes_by_cause.get(cause)
            if indices is None:
                raise KeyError(cause)
            starts.extend(indices)
        return starts

    @staticmethod
    def _expand_slice_nodes(
        nodes: tuple[dict[str, object], ...],
        edges: tuple[dict[str, object], ...],
        starts: list[int],
        direction: str,
        depth_value: int,
    ) -> set[int]:
        """Expand start node positions over the cascade's own directed edges.

        Adjacency comes only from the existing directed edge records, so the
        walk cannot leave the existing evidence. Levels count edges, a node
        reached more than once is claimed once, and depth zero keeps only
        the starts.
        """
        forward_neighbors: dict[int, list[int]] = {
            index: [] for index in range(len(nodes))
        }
        backward_neighbors: dict[int, list[int]] = {
            index: [] for index in range(len(nodes))
        }
        for edge in edges:
            forward_neighbors[edge["source"]].append(edge["target"])
            backward_neighbors[edge["target"]].append(edge["source"])

        selected = set(starts)
        if depth_value > 0:
            frontier = set(starts)
            for _ in range(depth_value):
                reached: set[int] = set()
                for index in frontier:
                    if direction in ("forward", "both"):
                        reached.update(forward_neighbors[index])
                    if direction in ("backward", "both"):
                        reached.update(backward_neighbors[index])
                reached.difference_update(selected)
                if not reached:
                    break
                selected.update(reached)
                frontier = reached
        return selected

    @staticmethod
    def _assemble_cascade_slice(
        cascade: dict[str, object],
        selected: set[int],
    ) -> dict[str, object]:
        """Build the detached ``nodes, edges, gaps`` slice for one selection.

        Nodes keep the full cascade's order and positions remap densely;
        edges keep only records whose two ends are selected, with endpoints
        remapped to slice positions while paths and edge order are
        unchanged; gaps keep only records whose interval is also an interval
        of a selected node. Every object is freshly copied.
        """
        nodes = cascade["nodes"]
        edges = cascade["edges"]
        gaps = cascade["gaps"]

        selected_positions = sorted(selected)
        remapped = {
            old: new for new, old in enumerate(selected_positions)
        }
        slice_nodes = tuple(
            {
                "cause": nodes[old]["cause"],
                "key": nodes[old]["key"],
                "intervals": tuple(nodes[old]["intervals"]),
                "checkpoints": tuple(nodes[old]["checkpoints"]),
                "branches": tuple(nodes[old]["branches"]),
                "affected": tuple(nodes[old]["affected"]),
            }
            for old in selected_positions
        )
        slice_edges = tuple(
            {
                "source": remapped[edge["source"]],
                "target": remapped[edge["target"]],
                "path": tuple(edge["path"]),
            }
            for edge in edges
            if edge["source"] in selected and edge["target"] in selected
        )
        selected_intervals: set[int] = set()
        for old in selected_positions:
            selected_intervals.update(nodes[old]["intervals"])
        slice_gaps = tuple(
            {
                "interval": gap["interval"],
                "members": tuple(gap["members"]),
            }
            for gap in gaps
            if gap["interval"] in selected_intervals
        )
        return {
            "nodes": slice_nodes,
            "edges": slice_edges,
            "gaps": slice_gaps,
        }

    @staticmethod
    def _build_slice_changes(
        before: dict[str, object],
        after: dict[str, object],
    ) -> tuple[dict[str, object], ...]:
        """Compare two assembled slices node/edge/gap by identity.

        Records are grouped by nodes, then edges, then gaps; each group runs
        removed, added and changed. Identities never use slice position
        numbers: edge endpoints are remapped back to node identities. Every
        present side is copied again, so the change records share no object
        with either returned slice.
        """
        changes: list[dict[str, object]] = []

        def copy_node(node: dict[str, object]) -> dict[str, object]:
            return {
                "cause": node["cause"],
                "key": node["key"],
                "intervals": tuple(node["intervals"]),
                "checkpoints": tuple(node["checkpoints"]),
                "branches": tuple(node["branches"]),
                "affected": tuple(node["affected"]),
            }

        def copy_edge(edge: dict[str, object]) -> dict[str, object]:
            return {
                "source": edge["source"],
                "target": edge["target"],
                "path": tuple(edge["path"]),
            }

        def copy_gap(gap: dict[str, object]) -> dict[str, object]:
            return {
                "interval": gap["interval"],
                "members": tuple(gap["members"]),
            }

        def record(
            kind: str,
            identity: object,
            old: object,
            new: object,
        ) -> None:
            changes.append(
                {
                    "kind": kind,
                    "identity": identity,
                    "before": old,
                    "after": new,
                }
            )

        # --- Nodes, identified by (cause event, state key). ---
        before_nodes = {
            (node["cause"], node["key"]): node
            for node in before["nodes"]
        }
        after_nodes = {
            (node["cause"], node["key"]): node
            for node in after["nodes"]
        }
        for node in before["nodes"]:
            identity = (node["cause"], node["key"])
            if identity not in after_nodes:
                record("node_removed", identity, copy_node(node), None)
        for node in after["nodes"]:
            identity = (node["cause"], node["key"])
            if identity not in before_nodes:
                record("node_added", identity, None, copy_node(node))
        for node in before["nodes"]:
            identity = (node["cause"], node["key"])
            other = after_nodes.get(identity)
            if other is None:
                continue
            changed = (
                node["intervals"] != other["intervals"]
                or node["checkpoints"] != other["checkpoints"]
                or node["branches"] != other["branches"]
                or node["affected"] != other["affected"]
            )
            if changed:
                record(
                    "node_changed",
                    identity,
                    copy_node(node),
                    copy_node(other),
                )

        # --- Edges, identified solely by endpoint node identities. Slice
        # position numbers are remapped back before any comparison. ---
        def edge_identity(
            nodes: tuple[dict[str, object], ...],
            edge: dict[str, object],
        ) -> tuple[tuple[str, str], tuple[str, str]]:
            source = nodes[edge["source"]]
            target = nodes[edge["target"]]
            return (
                (source["cause"], source["key"]),
                (target["cause"], target["key"]),
            )

        before_edges = {
            edge_identity(before["nodes"], edge): edge
            for edge in before["edges"]
        }
        after_edges = {
            edge_identity(after["nodes"], edge): edge
            for edge in after["edges"]
        }
        for edge in before["edges"]:
            identity = edge_identity(before["nodes"], edge)
            if identity not in after_edges:
                record("edge_removed", identity, copy_edge(edge), None)
        for edge in after["edges"]:
            identity = edge_identity(after["nodes"], edge)
            if identity not in before_edges:
                record("edge_added", identity, None, copy_edge(edge))
        for edge in before["edges"]:
            identity = edge_identity(before["nodes"], edge)
            other = after_edges.get(identity)
            if other is not None and edge["path"] != other["path"]:
                record(
                    "edge_changed",
                    identity,
                    copy_edge(edge),
                    copy_edge(other),
                )

        # --- Gaps, identified by (interval, member combination). Their
        # identity exhausts their content, so gaps never "change". ---
        before_gaps = {
            (gap["interval"], tuple(gap["members"])): gap
            for gap in before["gaps"]
        }
        after_gaps = {
            (gap["interval"], tuple(gap["members"])): gap
            for gap in after["gaps"]
        }
        for gap in before["gaps"]:
            identity = (gap["interval"], tuple(gap["members"]))
            if identity not in after_gaps:
                record("gap_removed", identity, copy_gap(gap), None)
        for gap in after["gaps"]:
            identity = (gap["interval"], tuple(gap["members"]))
            if identity not in before_gaps:
                record("gap_added", identity, None, copy_gap(gap))

        return tuple(changes)

    def _cascade_side_order(
        self,
        prepared: dict[str, object],
        members: tuple[str, ...],
        member_index: int | None,
        checkpoint: int | None,
    ) -> list[str] | None:
        """Return one attribution side's frozen closure at the locating point.

        The attributions inside a change evidence dict are ordered one per
        member, but do not themselves carry either side's replay order; the
        shared frozen view captured by the prepare step does, so affected
        events stay inside their side's historical closure.
        ``member_index`` is ``None`` for the reference side (every member
        shares the checkpoint's reference node) and the member's position
        for the diverging side. A combination with no locating checkpoint
        contributes no closure.
        """
        if checkpoint is None or not members:
            return None
        position = self._combo_checkpoint_position(
            prepared, tuple(members), checkpoint
        )
        if member_index is None:
            return position[0][0]
        return position[member_index][1]

    def _collect_cascade_evidence(
        self,
        node_records: dict[tuple[str, str], dict[str, object]],
        present: list[tuple[str, str]],
        present_seen: set[tuple[str, str]],
        interval_index: int,
        checkpoint: int | None,
        branch: str,
        key: str,
        side: dict[str, object],
        side_order: list[str] | None,
        contributing_orders: list[list[str]],
    ) -> tuple[str, str] | None:
        """Fold one attribution side into its shared cascade node record.

        Returns the node key when the side carries a cause event, otherwise
        ``None`` (the change counts as a gap only when neither side of any
        member carries one). Each record keeps every interval/checkpoint the
        evidence appears at, associated branch names in a set, and the raw
        affected event ids with their source closure -- the merged replay
        order is applied once at finalization. Aggregation always runs; the
        node is listed in the interval's edge set only the first time it
        appears there.
        """
        cause = side["cause"]
        if cause is None:
            # A present cause key without a cause event cannot become a
            # node; the caller records the combination once as a gap.
            return None
        node_key = (cause, key)
        record = node_records.get(node_key)
        if record is None:
            record = {
                "cause": cause,
                "key": key,
                "intervals": [],
                "seen_intervals": set(),
                "checkpoints": [],
                "seen_checkpoints": set(),
                "branches": {branch},
                "affected_sets": [],
            }
            node_records[node_key] = record
        else:
            record["branches"].add(branch)
        if interval_index not in record["seen_intervals"]:
            record["seen_intervals"].add(interval_index)
            record["intervals"].append(interval_index)
        if checkpoint is not None and checkpoint not in record[
            "seen_checkpoints"
        ]:
            record["seen_checkpoints"].add(checkpoint)
            record["checkpoints"].append(checkpoint)
        affected = side["affected"]
        if affected and side_order is not None:
            record["affected_sets"].append((set(affected), side_order))
            contributing_orders.append(side_order)
        if node_key not in present_seen:
            present_seen.add(node_key)
            present.append(node_key)
        return node_key

    def _union_replay_order(self, orders: list[list[str]]) -> list[str]:
        """Merge captured closure orders into one closure replay order.

        The union of ancestor-closed sets is itself ancestor-closed; the
        Kahn replay below -- parents before children, ready events by
        ``(at, id)`` -- is the same order the graph gives each input
        closure, so the merged order restricts back to each input's order.
        """
        union: set[str] = set()
        for order in orders:
            union.update(order)
        if not union:
            return []
        indegree = {event_id: 0 for event_id in union}
        children: dict[str, list[str]] = {
            event_id: [] for event_id in union
        }
        for event_id in union:
            for parent in self._graph._parents[event_id]:
                if parent in union:
                    indegree[event_id] += 1
                    children[parent].append(event_id)
        ready = [
            (self._graph._at[event_id], event_id)
            for event_id, degree in indegree.items()
            if degree == 0
        ]
        heapq.heapify(ready)
        order: list[str] = []
        while ready:
            _, event_id = heapq.heappop(ready)
            order.append(event_id)
            for child in children[event_id]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    heapq.heappush(
                        ready, (self._graph._at[child], child)
                    )
        return order

    def _graph_children_map(self) -> dict[str, list[str]]:
        """Build the whole-graph parent-to-child adjacency used by edges."""
        children: dict[str, list[str]] = {
            event_id: [] for event_id in self._graph._at
        }
        for event_id in self._graph._at:
            for parent in self._graph._parents[event_id]:
                children[parent].append(event_id)
        return children

    def _shortest_ancestor_path(
        self,
        ancestor: str,
        descendant: str,
        children: dict[str, list[str]],
    ) -> tuple[str, ...] | None:
        """Shortest parent-to-child path ``ancestor`` -> ``descendant``.

        Returns the fewest-edge id path containing both ends over the whole
        graph, ties broken by the Unicode code point order of the complete
        id tuples (level-by-level breadth-first search, as in
        :meth:`explain_impact`), or ``None`` when ``descendant`` is not a
        strict descendant of ``ancestor``. Unlike the closure-scoped
        impact queries, the walk is over the whole graph, so a merge event
        may produce a cross-branch convergence path.
        """
        if ancestor == descendant or descendant not in self._graph._at:
            return None

        paths: dict[str, tuple[str, ...]] = {ancestor: (ancestor,)}
        frontier = [ancestor]
        while frontier:
            if descendant in paths:
                return paths[descendant]
            candidates: dict[str, tuple[str, ...]] = {}
            for current in frontier:
                current_path = paths[current]
                for child in children[current]:
                    if child in paths:
                        continue
                    candidate = (*current_path, child)
                    best = candidates.get(child)
                    if best is None or candidate < best:
                        candidates[child] = candidate
            for child, path in candidates.items():
                paths[child] = path
            frontier = list(candidates)
        return None

    def _prepare_frontier_scan(
        self,
        reference: str,
        series: dict[str, tuple[tuple[str, str], ...]],
        base_scenario: dict[str, object],
        axis: str,
        values: tuple[int | float, ...],
        min_size: int,
        max_size: int,
        required: tuple[str, ...],
        exclusive_pairs: tuple[tuple[str, str], ...],
        limit: int,
    ) -> dict[str, object]:
        """Validate the shared breakpoint inputs and capture one frozen view.

        Shared by :meth:`frontier_breakpoints` and
        :meth:`explain_frontier_breakpoints` so both apply identical
        validation order, errors, branch/node lookup order, checkpoint
        alignment, enumeration order and the candidate-count cap, and answer
        from one frozen read-only historical view. Returns the parsed base
        scenario, axis and value tuple, the member constraints, the captured
        per-series checkpoint views and the once-enumerated candidate list.
        """
        validated = self._validate_frontier_inputs(
            reference,
            series,
            base_scenario,
            axis,
            values,
            min_size,
            max_size,
            required,
            exclusive_pairs,
            limit,
        )
        return self._capture_frontier_view(validated)

    def _validate_frontier_inputs(
        self,
        reference: str,
        series: dict[str, tuple[tuple[str, str], ...]],
        base_scenario: dict[str, object],
        axis: str,
        values: tuple[int | float, ...],
        min_size: int,
        max_size: int,
        required: tuple[str, ...],
        exclusive_pairs: tuple[tuple[str, str], ...],
        limit: int,
    ) -> dict[str, object]:
        """Validate the ordinary breakpoint inputs without touching state.

        The pure-validation half of :meth:`_prepare_frontier_scan`: every
        ordinary input, the base scenario and the value series are checked
        in the existing order, but no branch or historical node is looked
        up and no candidate is enumerated. :meth:`cascade_slice` validates
        its own extra inputs between this phase and
        :meth:`_capture_frontier_view`, so its causes, direction, depth and
        node limit are checked before any state query, while the ordinary
        inputs keep exactly their existing validation order.
        """
        # --- Every ordinary input, the base scenario and the value series
        # finish validating before any branch is looked up, mirroring
        # frontier_sensitivity's reference, series, scenario order. ---
        self._require_nonempty_str(reference, "reference")
        if not isinstance(series, dict):
            raise TypeError(
                f"series must be a dict, got {type(series).__name__}"
            )
        per_series: list[tuple[str, tuple[tuple[str, str], ...]]] = []
        anchor_points: tuple[tuple[str, str], ...] | None = None
        for series_name, series_points in series.items():
            self._require_nonempty_str(series_name, "series branch")
            if series_name == reference:
                raise ValueError(
                    f"series branch {series_name!r} cannot equal reference "
                    f"branch {reference!r}"
                )
            self._validate_points(series_points)
            if anchor_points is None:
                anchor_points = series_points
            elif len(series_points) != len(anchor_points) or any(
                point[0] != anchor_point[0]
                for point, anchor_point in zip(
                    series_points, anchor_points
                )
            ):
                raise ValueError(
                    "all series must share checkpoint count and reference "
                    "nodes"
                )
            per_series.append((series_name, series_points))

        weights, total_budget, key_budgets = (
            self._validate_breakpoint_scenario(base_scenario)
        )
        ordered_keys = tuple(sorted(weights))

        self._require_nonempty_str(axis, "axis")
        if axis != "total_budget" and axis not in weights:
            raise ValueError(f"unknown perturbation axis {axis!r}")
        ordered_values = self._validate_breakpoint_values(values)

        # The size window, member constraints and limit reuse the existing
        # search's checks exactly.
        min_size_value = self._require_size_bound(min_size, "min_size")
        if min_size_value < 0:
            raise ValueError("min_size must be non-negative")
        max_size_value = self._require_size_bound(max_size, "max_size")
        if max_size_value < min_size_value:
            raise ValueError("max_size must be >= min_size")
        if max_size_value > len(per_series):
            raise ValueError("max_size must not exceed the number of series")

        if not isinstance(required, tuple):
            raise TypeError(
                f"required must be a tuple, got {type(required).__name__}"
            )
        required_members: list[str] = []
        seen_required: set[str] = set()
        for member in required:
            self._require_nonempty_str(member, "required member")
            if member in seen_required:
                raise ValueError(f"duplicate required member {member!r}")
            seen_required.add(member)
            if member not in series:
                raise ValueError(
                    f"required member {member!r} is absent from series"
                )
            required_members.append(member)

        if not isinstance(exclusive_pairs, tuple):
            raise TypeError(
                "exclusive_pairs must be a tuple, got "
                f"{type(exclusive_pairs).__name__}"
            )
        ordered_pairs: list[tuple[str, str]] = []
        seen_pairs: set[tuple[str, str]] = set()
        for pair in exclusive_pairs:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise TypeError(
                    "each exclusive pair must be a length-2 tuple of "
                    f"member names, got {pair!r} ({type(pair).__name__})"
                )
            member_a, member_b = pair
            self._require_nonempty_str(member_a, "exclusive member")
            self._require_nonempty_str(member_b, "exclusive member")
            if member_a == member_b:
                raise ValueError(
                    f"member {member_a!r} cannot be mutually exclusive "
                    "with itself"
                )
            if member_a not in series:
                raise ValueError(
                    f"exclusive member {member_a!r} is absent from series"
                )
            if member_b not in series:
                raise ValueError(
                    f"exclusive member {member_b!r} is absent from series"
                )
            normalized = tuple(sorted(pair))
            if normalized in seen_pairs:
                raise ValueError(f"duplicate exclusive pair {normalized!r}")
            seen_pairs.add(normalized)
            ordered_pairs.append((member_a, member_b))
        required_set = set(required_members)
        for member_a, member_b in ordered_pairs:
            if member_a in required_set and member_b in required_set:
                raise ValueError(
                    f"required members {member_a!r} and {member_b!r} are "
                    "mutually exclusive"
                )

        limit_value = self._require_size_bound(limit, "limit")
        if limit_value < 1:
            raise ValueError("limit must be >= 1")

        return {
            "reference": reference,
            "per_series": per_series,
            "ordered_keys": ordered_keys,
            "weights": weights,
            "total_budget": total_budget,
            "key_budgets": key_budgets,
            "axis": axis,
            "ordered_values": ordered_values,
            "min_size_value": min_size_value,
            "max_size_value": max_size_value,
            "required_set": required_set,
            "ordered_pairs": ordered_pairs,
            "limit_value": limit_value,
        }

    def _capture_frontier_view(
        self, validated: dict[str, object]
    ) -> dict[str, object]:
        """Look up branches/nodes and capture the one frozen historical view.

        The state-touching half of :meth:`_prepare_frontier_scan`: it applies
        the existing branch lookup order (reference first, then the series
        in input order), checks every node against its named branch's
        current head ancestor closure, captures each checkpoint closure
        once, and enumerates the candidates with the count cap. It runs
        only after every ordinary input -- and, for
        :meth:`cascade_slice`, every slice input -- has been validated.
        """
        reference = validated["reference"]
        per_series = validated["per_series"]

        self._require_known_branch(reference)

        # Capture the single frozen read-only historical view every value is
        # answered from, exactly as in search_combinations.
        reference_head_closure = set(
            self._graph._ordered_ancestors(self._heads[reference])
        )
        views_by_name: dict[
            str, list[tuple[list[str], list[str], str, str]]
        ] = {}
        for series_name, series_points in per_series:
            self._require_known_branch(series_name)
            head_closure_b = set(
                self._graph._ordered_ancestors(self._heads[series_name])
            )
            for node_a, node_b in series_points:
                if (
                    node_a not in self._graph._at
                    or node_a not in reference_head_closure
                ):
                    raise KeyError(node_a)
                if (
                    node_b not in self._graph._at
                    or node_b not in head_closure_b
                ):
                    raise KeyError(node_b)
            views_by_name[series_name] = [
                (
                    self._graph._ordered_ancestors(node_a),
                    self._graph._ordered_ancestors(node_b),
                    node_a,
                    node_b,
                )
                for node_a, node_b in series_points
            ]

        # Candidates are enumerated once in the existing search order and
        # reused at every value; the count cap runs before any evaluation,
        # so exceeding it leaves no partial work behind.
        pool = tuple(sorted(name for name, _ in per_series))
        combos: list[tuple[str, ...]] = []
        candidate_count = 0
        min_size_value = validated["min_size_value"]
        max_size_value = validated["max_size_value"]
        limit_value = validated["limit_value"]
        for size in range(min_size_value, max_size_value + 1):
            for combo in itertools.combinations(pool, size):
                candidate_count += 1
                if candidate_count > limit_value:
                    raise ValueError(
                        "search candidate limit exceeded: more than "
                        f"{limit_value} candidates in size window"
                    )
                combos.append(combo)

        return {
            "reference": reference,
            "views_by_name": views_by_name,
            "ordered_keys": validated["ordered_keys"],
            "weights": validated["weights"],
            "total_budget": validated["total_budget"],
            "key_budgets": validated["key_budgets"],
            "axis": validated["axis"],
            "ordered_values": validated["ordered_values"],
            "required_set": validated["required_set"],
            "ordered_pairs": validated["ordered_pairs"],
            "combos": combos,
        }

    def _breakpoint_snapshot(
        self, prepared: dict[str, object], value: int | float
    ) -> dict[str, object]:
        """Evaluate one perturbation value on the shared frozen view.

        Only the one number the axis names is replaced; every other weight
        and both budgets keep their base-scenario values. Returns the
        frontier and rejected rows, the feasible/frontier sets, the
        deterministic dominator tuples and the per-combination
        ``(risk, contributions)`` stats, plus the per-combination checkpoint
        evidence used to explain a turn.
        """
        weights = prepared["weights"]
        total_budget = prepared["total_budget"]
        axis = prepared["axis"]
        if axis == "total_budget":
            point_weights = dict(weights)
            point_total = value
        else:
            point_weights = dict(weights)
            point_weights[axis] = value
            point_total = total_budget
        point_key_budgets = dict(prepared["key_budgets"])
        frontier_rows, rejected_rows, feasible_stats = (
            self._frontier_search_once(
                prepared["combos"],
                prepared["views_by_name"],
                prepared["ordered_keys"],
                point_weights,
                point_key_budgets,
                point_total,
                prepared["required_set"],
                prepared["ordered_pairs"],
            )
        )
        frontier_set = {tuple(row["members"]) for row in frontier_rows}
        dominators: dict[
            tuple[str, ...], tuple[tuple[str, ...], ...]
        ] = {}
        for combo in prepared["combos"]:
            if combo not in feasible_stats:
                dominators[combo] = ()
                continue
            risk = feasible_stats[combo][0]
            dominated_by = [
                other
                for other in feasible_stats
                if len(other) >= len(combo)
                and feasible_stats[other][0] <= risk
                and (
                    len(other) > len(combo)
                    or feasible_stats[other][0] < risk
                )
            ]
            dominated_by.sort(
                key=lambda other: (
                    -len(other),
                    feasible_stats[other][0],
                    other,
                )
            )
            dominators[combo] = tuple(dominated_by)

        return {
            "value": value,
            "weights": point_weights,
            "total_budget": point_total,
            "key_budgets": point_key_budgets,
            "frontier_rows": frontier_rows,
            "rejected_rows": rejected_rows,
            "feasible": set(feasible_stats),
            "frontier": frontier_set,
            "dominators": dominators,
            "stats": feasible_stats,
        }

    def _breakpoint_change_evidence(
        self,
        prepared: dict[str, object],
        combo: tuple[str, ...],
        left_snapshot: dict[str, object],
        right_snapshot: dict[str, object],
    ) -> dict[str, object] | None:
        """Build one combination's explanation for an interval, or ``None``.

        The change type is decided in order -- feasibility, then domination,
        then frontier membership -- exactly as the breakpoint record groups
        entered/exited/affected combinations.
        """
        feasible_left = combo in left_snapshot["feasible"]
        feasible_right = combo in right_snapshot["feasible"]
        frontier_left = combo in left_snapshot["frontier"]
        frontier_right = combo in right_snapshot["frontier"]
        dominators_left = left_snapshot["dominators"][combo]
        dominators_right = right_snapshot["dominators"][combo]
        feasibility_changed = feasible_left != feasible_right
        domination_changed = dominators_left != dominators_right
        frontier_changed = frontier_left != frontier_right
        if not (
            feasibility_changed
            or domination_changed
            or frontier_changed
        ):
            return None

        if feasibility_changed:
            kind = "feasibility"
        elif domination_changed:
            kind = "domination"
        else:
            kind = "membership"

        point_count = self._combo_point_count(prepared, combo)
        checkpoint: int | None
        if point_count == 0:
            checkpoint = None
        elif kind == "feasibility":
            checkpoint = self._first_verdict_diff_checkpoint(
                prepared,
                combo,
                left_snapshot,
                right_snapshot,
            )
        else:
            checkpoint = point_count - 1

        overrun_left, overrun_right = (
            self._checkpoint_overrun(
                prepared, combo, left_snapshot, checkpoint
            ),
            self._checkpoint_overrun(
                prepared, combo, right_snapshot, checkpoint
            ),
        )

        # Aggregate signed differences are weight-independent, and for the
        # total-budget axis both sides share the base weights; for a weight
        # axis the perturbed key is returned directly, so either side's
        # snapshot names the same cause.
        cause_key = (
            self._checkpoint_cause_key(
                prepared, combo, left_snapshot, checkpoint
            )
            if checkpoint is not None
            else None
        )
        attribution: tuple[dict[str, object], ...] = ()
        if cause_key is not None:
            attribution = self._checkpoint_attributions(
                prepared, combo, checkpoint, cause_key
            )

        risk_left = (
            left_snapshot["stats"][combo][0] if feasible_left else None
        )
        risk_right = (
            right_snapshot["stats"][combo][0] if feasible_right else None
        )
        return {
            "members": tuple(combo),
            "kind": kind,
            "checkpoint": checkpoint,
            "cause_key": cause_key,
            "attribution": attribution,
            "left": {
                "feasible": feasible_left,
                "risk": risk_left,
                "dominators": tuple(
                    tuple(other) for other in dominators_left
                ),
                "overrun": overrun_left,
            },
            "right": {
                "feasible": feasible_right,
                "risk": risk_right,
                "dominators": tuple(
                    tuple(other) for other in dominators_right
                ),
                "overrun": overrun_right,
            },
        }

    def _combo_point_count(
        self,
        prepared: dict[str, object],
        combo: tuple[str, ...],
    ) -> int:
        """Return the shared checkpoint count behind one combination."""
        views_by_name = prepared["views_by_name"]
        if not combo:
            # The empty subset has no member views; the pool alignment
            # guarantees every series shares one checkpoint count.
            views = next(iter(views_by_name.values()), None)
            return 0 if views is None else len(views)
        return len(views_by_name[combo[0]])

    def _combo_checkpoint_position(
        self,
        prepared: dict[str, object],
        combo: tuple[str, ...],
        checkpoint: int,
    ) -> list[tuple[list[str], list[str], str, str]]:
        """Capture every member's frozen view at one checkpoint.

        One ``(left_order, right_order, node_a, node_b)`` entry per member
        in member order. The empty subset contributes no entries; pool
        alignment guarantees that a checkpoint exists for every member.
        """
        views_by_name = prepared["views_by_name"]
        return [
            views_by_name[member][checkpoint] for member in combo
        ]

    def _checkpoint_accounting(
        self,
        prepared: dict[str, object],
        combo: tuple[str, ...],
        snapshot: dict[str, object],
        checkpoint: int,
    ) -> dict[str, object]:
        """Recompute one combination's accounting at one checkpoint.

        Returns the per-key signed aggregate, the per-member per-key signed
        differences, the weighted total risk and the total/per-key budget
        breach magnitudes under the snapshot's own weights and budgets.
        """
        ordered_keys = prepared["ordered_keys"]
        weights = snapshot["weights"]
        key_budgets = snapshot["key_budgets"]
        total_budget = snapshot["total_budget"]
        position = self._combo_checkpoint_position(
            prepared, combo, checkpoint
        )
        aggregate: dict[str, int | float] = {
            key: 0 for key in ordered_keys
        }
        member_diffs: list[dict[str, int | float]] = []
        if position:
            reference_order = position[0][0]
            reference_values = {
                key: self._replayed_value(reference_order, key)
                for key in ordered_keys
            }
            for _, right_order, _, _ in position:
                diffs: dict[str, int | float] = {}
                for key in ordered_keys:
                    diff = (
                        self._replayed_value(right_order, key)
                        - reference_values[key]
                    )
                    diffs[key] = diff
                    aggregate[key] += diff
                member_diffs.append(diffs)
        risk: int | float = 0
        for key in ordered_keys:
            risk += abs(aggregate[key]) * weights[key]
        if risk == 0:
            risk = 0 if isinstance(risk, int) else 0.0
        total_overrun: int | float | None = (
            risk - total_budget if risk > total_budget else None
        )
        key_overruns: dict[str, int | float] = {}
        for key in ordered_keys:
            if (
                key in key_budgets
                and abs(aggregate[key]) > key_budgets[key]
            ):
                key_overruns[key] = (
                    abs(aggregate[key]) - key_budgets[key]
                ) * weights[key]
        return {
            "aggregate": aggregate,
            "member_diffs": member_diffs,
            "risk": risk,
            "total_overrun": total_overrun,
            "key_overruns": key_overruns,
        }

    def _first_verdict_diff_checkpoint(
        self,
        prepared: dict[str, object],
        combo: tuple[str, ...],
        left_snapshot: dict[str, object],
        right_snapshot: dict[str, object],
    ) -> int | None:
        """Return the first checkpoint whose budget verdict differs.

        A checkpoint's verdict is its feasible/rejected outcome under each
        side's own weights and budgets; the comparison uses the same
        strictly-greater total and per-key breach tests as the existing
        search. Returns ``None`` only when no checkpoint verdict differs.
        """
        point_count = self._combo_point_count(prepared, combo)
        for checkpoint in range(point_count):
            left_accounting = self._checkpoint_accounting(
                prepared, combo, left_snapshot, checkpoint
            )
            right_accounting = self._checkpoint_accounting(
                prepared, combo, right_snapshot, checkpoint
            )
            left_breach = (
                left_accounting["total_overrun"] is not None
                or bool(left_accounting["key_overruns"])
            )
            right_breach = (
                right_accounting["total_overrun"] is not None
                or bool(right_accounting["key_overruns"])
            )
            if left_breach != right_breach:
                return checkpoint
        return None

    def _checkpoint_overrun(
        self,
        prepared: dict[str, object],
        combo: tuple[str, ...],
        snapshot: dict[str, object],
        checkpoint: int | None,
    ) -> int | float | None:
        """Return the locating checkpoint's breach magnitude for one side.

        Mirrors the existing rejected-row ``overrun``: the total risk above
        the total budget when the total budget is breached (the total test
        outranks per-key breaches), otherwise the first breaching key's
        weighted excess in Unicode key order. A checkpoint that holds, or no
        checkpoint at all, has no breach evidence and yields ``None`` rather
        than a substituted zero.
        """
        if checkpoint is None:
            return None
        accounting = self._checkpoint_accounting(
            prepared, combo, snapshot, checkpoint
        )
        if accounting["total_overrun"] is not None:
            return accounting["total_overrun"]
        key_overruns = accounting["key_overruns"]
        if key_overruns:
            return key_overruns[min(key_overruns)]
        return None

    def _checkpoint_cause_key(
        self,
        prepared: dict[str, object],
        combo: tuple[str, ...],
        snapshot: dict[str, object],
        checkpoint: int,
    ) -> str | None:
        """Pick the state key responsible at one checkpoint.

        For the total-budget axis this is the non-zero key with the largest
        absolute weighted contribution to the combination at the checkpoint,
        ties broken by the smallest Unicode code point; for a weight axis
        the perturbed key is used directly, unless it contributes nothing
        in every relevant member, in which case the result is ``None``.
        """
        axis = prepared["axis"]
        accounting = self._checkpoint_accounting(
            prepared, combo, snapshot, checkpoint
        )
        weights = snapshot["weights"]
        if axis != "total_budget":
            if any(
                member_diffs[axis] != 0
                for member_diffs in accounting["member_diffs"]
            ):
                return axis
            return None
        candidates = [
            key
            for key in prepared["ordered_keys"]
            if accounting["aggregate"][key] != 0
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda key: (
                -abs(accounting["aggregate"][key]) * weights[key],
                key,
            ),
        )

    def _checkpoint_attributions(
        self,
        prepared: dict[str, object],
        combo: tuple[str, ...],
        checkpoint: int,
        cause_key: str,
    ) -> tuple[dict[str, object], ...]:
        """Copy the existing divergence attribution at one checkpoint.

        One independent copy per member in member order, reusing
        :meth:`attribute_divergence_at`' ``key, fork, left, right`` result
        between the shared reference node and that member's node.
        """
        position = self._combo_checkpoint_position(
            prepared, combo, checkpoint
        )
        return tuple(
            self._copy_attribution(
                self._attribute_divergence_on_orders(
                    left_order,
                    right_order,
                    node_a,
                    node_b,
                    cause_key,
                )
            )
            for left_order, right_order, node_a, node_b in position
        )

    @staticmethod
    def _validate_breakpoint_scenario(
        scenario: Any,
    ) -> tuple[
        dict[str, int | float],
        int | float,
        dict[str, int | float],
    ]:
        """Validate ``frontier_breakpoints``' base scenario.

        A scenario without a scenario name: exactly the keys ``weights``,
        ``total_budget`` and ``key_budgets``, each obeying
        :meth:`search_combinations`' numeric, finiteness and key-set
        rules. A non-dict raises :class:`TypeError`; missing or extra
        keys raise :class:`ValueError`.
        """
        if not isinstance(scenario, dict):
            raise TypeError(
                "base_scenario must be a dict, got "
                f"{type(scenario).__name__}"
            )
        expected_keys = {"weights", "total_budget", "key_budgets"}
        if set(scenario.keys()) != expected_keys:
            raise ValueError(
                "base_scenario must contain exactly the keys 'weights', "
                "'total_budget' and 'key_budgets'"
            )
        weights = scenario["weights"]
        BranchStore._validate_weights(weights)
        total_budget = scenario["total_budget"]
        BranchStore._require_finite_budget(total_budget, "total budget")
        key_budgets = scenario["key_budgets"]
        if not isinstance(key_budgets, dict):
            raise TypeError(
                "key_budgets must be a dict, got "
                f"{type(key_budgets).__name__}"
            )
        for budget_key, budget_limit in key_budgets.items():
            EventGraph._require_nonempty_str(
                budget_key, "per-key budget key"
            )
            BranchStore._require_finite_budget(
                budget_limit, f"per-key budget for {budget_key!r}"
            )
            if budget_key not in weights:
                raise ValueError(
                    f"per-key budget key {budget_key!r} is absent from weights"
                )
        return weights, total_budget, key_budgets

    @staticmethod
    def _validate_breakpoint_values(
        values: Any,
    ) -> tuple[int | float, ...]:
        """Validate the strictly increasing perturbation value tuple.

        A non-tuple raises :class:`TypeError`; fewer than two values, a
        :class:`bool`/non-number element, a negative, NaN or infinite
        value, an out-of-order pair or a repeated value raise
        :class:`ValueError`.
        """
        if not isinstance(values, tuple):
            raise TypeError(
                f"values must be a tuple, got {type(values).__name__}"
            )
        if len(values) < 2:
            raise ValueError("values must contain at least two numbers")
        checked: list[int | float] = []
        previous: int | float | None = None
        for value in values:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(
                    "each value must be an int or float, got "
                    f"{type(value).__name__}"
                )
            if isinstance(value, float) and (
                math.isnan(value) or math.isinf(value)
            ):
                raise ValueError("values must be finite")
            if value < 0:
                raise ValueError("values must be non-negative")
            if previous is not None and value <= previous:
                raise ValueError(
                    "values must be strictly increasing and unique"
                )
            checked.append(value)
            previous = value
        return tuple(checked)

    @staticmethod
    def _copy_frontier_row(row: dict[str, object]) -> dict[str, object]:
        """Deep-copy one ``search_combinations`` frontier row."""
        return {
            "members": tuple(row["members"]),
            "risk": row["risk"],
            "contributions": tuple(row["contributions"]),
            "attributions": tuple(
                tuple(
                    BranchStore._copy_attribution(attribution)
                    for attribution in member_attributions
                )
                for member_attributions in row["attributions"]
            ),
        }

    @staticmethod
    def _copy_rejected_row(row: dict[str, object]) -> dict[str, object]:
        """Copy one ``search_combinations`` rejected row (scalar fields)."""
        return {
            "members": tuple(row["members"]),
            "reason": row["reason"],
            "checkpoint": row["checkpoint"],
            "key": row["key"],
            "overrun": row["overrun"],
        }

    def _copy_breakpoint_point(
        self, snapshot: dict[str, object]
    ) -> dict[str, object]:
        """Build one independent ``value, frontier, rejected`` point copy."""
        return {
            "value": snapshot["value"],
            "frontier": tuple(
                self._copy_frontier_row(row)
                for row in snapshot["frontier_rows"]
            ),
            "rejected": tuple(
                self._copy_rejected_row(row)
                for row in snapshot["rejected_rows"]
            ),
        }

    @staticmethod
    def _validate_sensitivity_scenarios(
        scenarios: Any,
    ) -> list[tuple[str, dict, int | float, dict, tuple[str, ...]]]:
        """Validate the ordered ``scenarios`` tuple of frontier_sensitivity.

        Returns one parsed tuple per scenario in input order:
        ``(name, weights, total_budget, key_budgets, ordered_weight_keys)``.
        The container and each item's type raise :class:`TypeError`; a
        scenario whose keys are not exactly ``name``, ``weights``,
        ``total_budget`` and ``key_budgets`` raises :class:`ValueError`. A
        name that is not a ``str`` raises :class:`TypeError`; an empty or
        duplicate name raises :class:`ValueError`. Numeric, finiteness and
        key-set rules mirror :meth:`search_combinations` per scenario.
        """
        if not isinstance(scenarios, tuple):
            raise TypeError(
                f"scenarios must be a tuple, got {type(scenarios).__name__}"
            )
        expected_keys = {"name", "weights", "total_budget", "key_budgets"}
        parsed: list[
            tuple[str, dict, int | float, dict, tuple[str, ...]]
        ] = []
        seen_names: set[str] = set()
        for scenario in scenarios:
            if not isinstance(scenario, dict):
                raise TypeError(
                    "each scenario must be a dict, got "
                    f"{type(scenario).__name__}"
                )
            if set(scenario.keys()) != expected_keys:
                raise ValueError(
                    "each scenario must contain exactly the keys 'name', "
                    "'weights', 'total_budget' and 'key_budgets'"
                )
            name = scenario["name"]
            if not isinstance(name, str):
                raise TypeError(
                    "scenario name must be a str, got "
                    f"{type(name).__name__}"
                )
            if not name:
                raise ValueError("scenario name must be a non-empty str")
            if name in seen_names:
                raise ValueError(f"duplicate scenario name {name!r}")
            seen_names.add(name)
            weights = scenario["weights"]
            ordered_keys = BranchStore._validate_weights(weights)
            total_budget = scenario["total_budget"]
            BranchStore._require_finite_budget(
                total_budget, f"total budget for scenario {name!r}"
            )
            key_budgets = scenario["key_budgets"]
            if not isinstance(key_budgets, dict):
                raise TypeError(
                    "key_budgets for scenario "
                    f"{name!r} must be a dict, got "
                    f"{type(key_budgets).__name__}"
                )
            for budget_key, budget_limit in key_budgets.items():
                EventGraph._require_nonempty_str(
                    budget_key, "per-key budget key"
                )
                BranchStore._require_finite_budget(
                    budget_limit,
                    f"per-key budget for {budget_key!r} in scenario "
                    f"{name!r}",
                )
                if budget_key not in weights:
                    raise ValueError(
                        f"per-key budget key {budget_key!r} is absent from "
                        f"the weights of scenario {name!r}"
                    )
            parsed.append(
                (name, weights, total_budget, key_budgets, ordered_keys)
            )
        return parsed

    def _frontier_search_once(
        self,
        combos: list[tuple[str, ...]],
        views_by_name: dict[
            str, list[tuple[list[str], list[str], str, str]]
        ],
        ordered_keys: tuple[str, ...],
        weights: dict[str, int | float],
        key_budgets: dict[str, int | float],
        total_budget: int | float,
        required_set: set[str],
        ordered_pairs: list[tuple[str, str]],
    ) -> tuple[
        list[dict[str, object]],
        list[dict[str, object]],
        dict[tuple[str, ...], tuple[int | float, tuple[int | float, ...]]],
    ]:
        """Run one existing-search evaluation over pre-enumerated candidates.

        Shares :meth:`search_combinations`' rejection first causes,
        budget accounting, frontier rule and output row shapes exactly,
        but enumerates nothing itself: the same candidate list is reused
        across scenarios. Returns the frontier rows (already in frontier
        order), the rejected rows (enumeration order) and a stats dict
        mapping every feasible combination's copied member tuple to its
        final-checkpoint risk and marginal contributions.
        """
        rejected: list[dict[str, object]] = []
        feasible: list[
            tuple[
                tuple[str, ...],
                int | float,
                tuple[int | float, ...],
                tuple[tuple[dict[str, object], ...], ...],
            ]
        ] = []
        for candidate in combos:
            # A copied member tuple keeps each scenario's rows detached
            # from the shared enumeration and from one another.
            combo = tuple(candidate)
            combo_set = set(combo)

            if not required_set.issubset(combo_set):
                rejected.append(
                    {
                        "members": combo,
                        "reason": "missing_required",
                        "checkpoint": None,
                        "key": None,
                        "overrun": None,
                    }
                )
                continue
            hit_pair = next(
                (
                    pair
                    for pair in ordered_pairs
                    if pair[0] in combo_set and pair[1] in combo_set
                ),
                None,
            )
            if hit_pair is not None:
                rejected.append(
                    {
                        "members": combo,
                        "reason": "exclusive_pair",
                        "checkpoint": None,
                        "key": None,
                        "overrun": None,
                    }
                )
                continue

            outcome = self._evaluate_subset_view(
                [views_by_name[member] for member in combo],
                ordered_keys,
                weights,
                key_budgets,
                total_budget,
            )
            if outcome[0] == "reject":
                _, checkpoint, reason, breach_key, overrun = outcome
                rejected.append(
                    {
                        "members": combo,
                        "reason": reason,
                        "checkpoint": checkpoint,
                        "key": breach_key,
                        "overrun": overrun,
                    }
                )
                continue

            (
                _,
                final_risk,
                final_aggregate,
                final_member_diffs,
                final_position,
            ) = outcome
            contributions: list[int | float] = []
            for member_index in range(len(combo)):
                if final_position is None:
                    contributions.append(0)
                    continue
                excluded_risk: int | float = 0
                for key in ordered_keys:
                    excluded = (
                        final_aggregate[key]
                        - final_member_diffs[member_index][key]
                    )
                    excluded_risk += abs(excluded) * weights[key]
                if excluded_risk == 0:
                    excluded_risk = (
                        0
                        if isinstance(excluded_risk, int)
                        else 0.0
                    )
                contribution = final_risk - excluded_risk
                if contribution == 0:
                    contribution = (
                        0
                        if isinstance(contribution, int)
                        else 0.0
                    )
                contributions.append(contribution)

            attributions: tuple[
                tuple[dict[str, object], ...], ...
            ] = ()
            if final_position is not None:
                attributions = tuple(
                    tuple(
                        self._copy_attribution(
                            self._attribute_divergence_on_orders(
                                left_order,
                                right_order,
                                node_a,
                                node_b,
                                key,
                            )
                        )
                        for key in ordered_keys
                    )
                    for (
                        left_order,
                        right_order,
                        node_a,
                        node_b,
                    ) in final_position
                )

            feasible.append(
                (combo, final_risk, tuple(contributions), attributions)
            )

        frontier_entries: list[
            tuple[
                tuple[str, ...],
                int | float,
                tuple[int | float, ...],
                tuple[tuple[dict[str, object], ...], ...],
            ]
        ] = []
        for entry in feasible:
            members, risk, _, _ = entry
            dominated = any(
                len(other_members) >= len(members)
                and other_risk <= risk
                and (
                    len(other_members) > len(members)
                    or other_risk < risk
                )
                for other_members, other_risk, _, _ in feasible
            )
            if not dominated:
                frontier_entries.append(entry)
        frontier_entries.sort(
            key=lambda entry: (-len(entry[0]), entry[1], entry[0])
        )

        frontier_rows = [
            {
                "members": members,
                "risk": risk,
                "contributions": tuple(contributions),
                "attributions": tuple(
                    tuple(member_attributions)
                    for member_attributions in attributions
                ),
            }
            for members, risk, contributions, attributions in (
                frontier_entries
            )
        ]
        feasible_stats = {
            members: (risk, tuple(contributions))
            for members, risk, contributions, _ in feasible
        }
        return frontier_rows, rejected, feasible_stats

    def _evaluate_subset_view(
        self,
        member_views: list[list[tuple[list[str], list[str], str, str]]],
        ordered_keys: tuple[str, ...],
        weights: dict[str, int | float],
        key_budgets: dict[str, int | float],
        total_budget: int | float,
    ) -> tuple[object, ...]:
        """Apply the combination-budget accounting to one enumerated subset.

        Returns a reject tuple ``("reject", checkpoint, reason, key,
        overrun)`` at the earliest breaching checkpoint -- the total budget
        before the first per-key breach in Unicode key order -- or a
        feasible tuple carrying the final checkpoint's risk, per-key
        aggregate, per-member signed differences and captured positions.
        An empty subset (or a pool without checkpoints) is feasible with
        zero risk. All inputs come from the caller's captured read-only
        view.
        """
        point_count = len(member_views[0]) if member_views else 0
        final_risk: int | float = 0
        final_aggregate: dict[str, int] = {}
        final_member_diffs: list[dict[str, int]] = []
        final_position: list[
            tuple[list[str], list[str], str, str]
        ] | None = None

        for index in range(point_count):
            position = [views[index] for views in member_views]
            # Pool alignment guarantees every member shares this checkpoint's
            # reference node, so its values are replayed once per point.
            reference_order = position[0][0]
            reference_values = {
                key: self._replayed_value(reference_order, key)
                for key in ordered_keys
            }
            aggregate: dict[str, int] = {key: 0 for key in ordered_keys}
            member_diffs: list[dict[str, int]] = []
            for _, right_order, _, _ in position:
                diffs: dict[str, int] = {}
                for key in ordered_keys:
                    diff = (
                        self._replayed_value(right_order, key)
                        - reference_values[key]
                    )
                    diffs[key] = diff
                    aggregate[key] += diff
                member_diffs.append(diffs)

            risk: int | float = 0
            for key in ordered_keys:
                risk += abs(aggregate[key]) * weights[key]
            if risk == 0:
                risk = 0 if isinstance(risk, int) else 0.0

            # At one checkpoint the total breach outranks every per-key
            # breach; the first breaching key by Unicode order is reported.
            if risk > total_budget:
                return (
                    "reject",
                    index,
                    "total_budget",
                    None,
                    risk - total_budget,
                )
            for key in ordered_keys:
                if (
                    key in key_budgets
                    and abs(aggregate[key]) > key_budgets[key]
                ):
                    return (
                        "reject",
                        index,
                        "key_budget",
                        key,
                        (abs(aggregate[key]) - key_budgets[key])
                        * weights[key],
                    )

            final_risk = risk
            final_aggregate = aggregate
            final_member_diffs = member_diffs
            final_position = position

        return (
            "feasible",
            final_risk,
            final_aggregate,
            final_member_diffs,
            final_position,
        )

    @staticmethod
    def _require_size_bound(value: Any, name: str) -> int:
        """Validate one non-``bool`` integer size/search parameter.

        ``min_size``, ``max_size`` and ``limit`` share the contract: a
        :class:`bool` or any non-:class:`int` value raises
        :class:`TypeError`; the value's range is the caller's concern.
        """
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                f"{name} must be an int, got {type(value).__name__}"
            )
        return value

    @staticmethod
    def _require_finite_budget(value: Any, name: str) -> None:
        """Validate one budget number, shared by the total and per-key maps.

        A :class:`bool` or any non-:class:`int`/:class:`float` value raises
        :class:`TypeError`; a NaN or infinite value raises
        :class:`ValueError` before the sign is checked, and a negative
        number raises :class:`ValueError`. Zero (int or float) is accepted.
        """
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(
                f"{name} must be an int or float, got {type(value).__name__}"
            )
        if isinstance(value, float) and (
            math.isnan(value) or math.isinf(value)
        ):
            raise ValueError(f"{name} must be finite")
        if value < 0:
            raise ValueError(f"{name} must be non-negative")

    @staticmethod
    def _validate_weights(weights: Any) -> tuple[str, ...]:
        """Validate ``weights`` and return its keys in Unicode order.

        The weight-map counterpart of :meth:`_validate_keys`: a non-dict
        raises :class:`TypeError`; entries are walked in insertion order,
        a non-``str`` key raising :class:`TypeError` and an empty key
        raising :class:`ValueError`; each weight must be a non-``bool``
        :class:`int` or :class:`float` (else :class:`TypeError`) and must
        not be negative, NaN or infinite (else :class:`ValueError`).
        """
        if not isinstance(weights, dict):
            raise TypeError(
                f"weights must be a dict, got {type(weights).__name__}"
            )
        for key, value in weights.items():
            EventGraph._require_nonempty_str(key, "weight key")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(
                    f"weight for {key!r} must be an int or float, got "
                    f"{type(value).__name__}"
                )
            if isinstance(value, float) and (
                math.isnan(value) or math.isinf(value)
            ):
                raise ValueError(f"weight for {key!r} must be finite")
            if value < 0:
                raise ValueError(f"weight for {key!r} must be non-negative")
        return tuple(sorted(weights))

    @staticmethod
    def _copy_attribution(
        record: dict[str, Any],
    ) -> dict[str, object]:
        """Deep-copy a ``key, fork, left, right`` attribution dict.

        The copied result shares no tuples or dicts with the source, so it
        stays detached from the per-call attributions and internal state.
        """
        def copy_side(side: dict[str, Any]) -> dict[str, object]:
            return {
                "value": side["value"],
                "cause": side["cause"],
                "path": tuple(side["path"]),
                "affected": tuple(side["affected"]),
            }

        return {
            "key": record["key"],
            "fork": record["fork"],
            "left": copy_side(record["left"]),
            "right": copy_side(record["right"]),
        }

    @staticmethod
    def _validate_points(points: Any) -> None:
        """Validate a tuple of length-2, unique node-id pairs.

        Shared by :meth:`divergence_timeline`, :meth:`divergence_summary`
        and :meth:`divergence_matrix` so points raise identical errors: a
        non-tuple container or non-length-2-tuple point raises
        :class:`TypeError`, non-``str`` or empty node ids raise
        :class:`TypeError`/:class:`ValueError` (left node before right),
        and a repeated node pair raises :class:`ValueError`.
        """
        if not isinstance(points, tuple):
            raise TypeError(
                f"points must be a tuple, got {type(points).__name__}"
            )
        seen_points: set[tuple[str, str]] = set()
        for point in points:
            if not isinstance(point, tuple) or len(point) != 2:
                raise TypeError(
                    "each point must be a length-2 tuple of node ids, "
                    f"got {point!r} ({type(point).__name__})"
                )
            node_a, node_b = point
            EventGraph._require_nonempty_str(node_a, "node_a")
            EventGraph._require_nonempty_str(node_b, "node_b")
            pair = (node_a, node_b)
            if pair in seen_points:
                raise ValueError(f"duplicate point {pair!r}")
            seen_points.add(pair)

    @staticmethod
    def _validate_keys(keys: Any) -> tuple[str, ...]:
        """Validate ``keys`` and return them in Unicode code point order.

        Shared by the divergence family: a non-tuple raises
        :class:`TypeError`, a non-``str`` element raises :class:`TypeError`,
        and an empty or duplicated key raises :class:`ValueError`. Summary
        and matrix rows always present keys by Unicode code point, so the
        sorted, duplicate-free tuple is returned once.
        """
        if not isinstance(keys, tuple):
            raise TypeError(
                f"keys must be a tuple, got {type(keys).__name__}"
            )
        seen_keys: set[str] = set()
        for key in keys:
            EventGraph._require_nonempty_str(key, "key")
            if key in seen_keys:
                raise ValueError(f"duplicate key {key!r}")
            seen_keys.add(key)
        return tuple(sorted(keys))

    def _divergence_inputs(
        self,
        name_a: str,
        name_b: str,
        points: tuple[tuple[str, str], ...],
        keys: tuple[str, ...],
    ) -> tuple[
        list[tuple[list[str], list[str], str, str]] | None,
        tuple[str, ...],
    ]:
        """Validate and capture the single read-only divergence view.

        Shared by :meth:`divergence_timeline` and
        :meth:`divergence_summary` so both apply identical validation
        order, errors, empty-container handling and branch/node closure
        rules, and answer every point and key from one historical view.
        Returns the keys in Unicode code point order alongside a list of
        one ``(left_order, right_order, node_a, node_b)`` entry per point
        in input order; the list is ``None`` when ``points`` is empty (the
        empty-tuple result case).
        """
        self._require_nonempty_str(name_a, "name_a")
        self._require_nonempty_str(name_b, "name_b")
        self._validate_points(points)
        ordered_keys = self._validate_keys(keys)

        self._require_known_branch(name_a)
        self._require_known_branch(name_b)
        if name_a == name_b:
            raise ValueError(
                f"cannot attribute divergence for branch {name_a!r} "
                "against itself"
            )

        if not points:
            return None, ordered_keys

        # Capture the single read-only historical view: the current head
        # closures decide node membership for every pair, and the nodes'
        # own closures answer every key.
        head_closure_a = set(
            self._graph._ordered_ancestors(self._heads[name_a])
        )
        head_closure_b = set(
            self._graph._ordered_ancestors(self._heads[name_b])
        )
        for node_a, node_b in points:
            if node_a not in self._graph._at or node_a not in head_closure_a:
                raise KeyError(node_a)
            if node_b not in self._graph._at or node_b not in head_closure_b:
                raise KeyError(node_b)

        return [
            (
                self._graph._ordered_ancestors(node_a),
                self._graph._ordered_ancestors(node_b),
                node_a,
                node_b,
            )
            for node_a, node_b in points
        ], ordered_keys

    def _divergence_summaries(
        self,
        ordered_points: list[tuple[list[str], list[str], str, str]]
        | None,
        ordered_keys: tuple[str, ...],
    ) -> tuple[dict[str, object], ...]:
        """Build :meth:`divergence_summary` rows from a captured view.

        ``ordered_points`` is one captured view entry per point in input
        order, or ``None``/empty when there are no points; ``ordered_keys``
        are already sorted by Unicode code point. Shared by
        :meth:`divergence_summary` and :meth:`divergence_matrix`, so a
        matrix row is item-wise equal to the corresponding standalone
        summary call. With no points or keys an empty tuple is returned.
        """
        if not ordered_points or not ordered_keys:
            return ()

        # All attributes per (point, key) come from the captured view.
        attributed: list[dict[str, dict[str, Any]]] = [
            {
                key: self._attribute_divergence_on_orders(
                    left_order, right_order, node_a, node_b, key
                )
                for key in ordered_keys
            }
            for left_order, right_order, node_a, node_b in ordered_points
        ]

        summaries: list[dict[str, object]] = []
        for key in ordered_keys:
            first_diverged: int | None = None
            transitions: list[dict[str, object]] = []
            prev_diverged: bool | None = None
            prev_left_cause: object = None
            prev_right_cause: object = None
            for index, per_key in enumerate(attributed):
                record = per_key[key]
                left = record["left"]
                right = record["right"]
                left_value = left["value"]
                right_value = right["value"]
                left_cause = left["cause"]
                right_cause = right["cause"]
                diverged = left_value != right_value
                if diverged and first_diverged is None:
                    first_diverged = index

                kind: str | None = None
                if prev_diverged is None:
                    if diverged:
                        kind = "diverged"
                elif not prev_diverged and diverged:
                    kind = "diverged"
                elif prev_diverged and not diverged:
                    kind = "converged"
                elif (
                    prev_diverged
                    and diverged
                    and (
                        prev_left_cause != left_cause
                        or prev_right_cause != right_cause
                    )
                ):
                    kind = "reattributed"
                if kind is not None:
                    transitions.append(
                        {
                            "index": index,
                            "kind": kind,
                            "left_value": left_value,
                            "right_value": right_value,
                            "left_cause": left_cause,
                            "right_cause": right_cause,
                        }
                    )

                prev_diverged = diverged
                prev_left_cause = left_cause
                prev_right_cause = right_cause

            summaries.append(
                {
                    "key": key,
                    "first_diverged": first_diverged,
                    "transitions": tuple(transitions),
                    "last": self._copy_attribution(attributed[-1][key]),
                }
            )
        return tuple(summaries)


    def _attribute_divergence_for(
        self, head_a: str, head_b: str, key: str
    ) -> dict[str, object]:
        """Replay-based divergence attribution for two closure heads.

        Shared by :meth:`attribute_divergence` (the branches' current
        heads) and :meth:`attribute_divergence_at` (arbitrary historical
        nodes); all argument validation belongs to the callers.
        """
        left_order = self._graph._ordered_ancestors(head_a)
        right_order = self._graph._ordered_ancestors(head_b)
        return self._attribute_divergence_on_orders(
            left_order, right_order, head_a, head_b, key
        )

    def _replayed_value(self, order: list[str], key: str) -> int:
        """Replay one ``key``'s integer value over a captured replay order.

        A closure on which ``key`` never appears contributes ``0``.
        """
        value = 0
        for event_id in order:
            changes = self._graph._changes[event_id]
            if key in changes:
                value += changes[key]
        return value

    def _attribute_divergence_on_orders(
        self,
        left_order: list[str],
        right_order: list[str],
        head_a: str,
        head_b: str,
        key: str,
    ) -> dict[str, object]:
        """Divergence attribution on closures captured by the caller.

        Sharing the precomputed replay orders lets
        :meth:`attribute_divergences_at` answer every key from one
        read-only historical view; results are identical to replaying
        the closures per key because replay order is deterministic.
        """
        left_ids = set(left_order)
        right_ids = set(right_order)

        left_value = self._replayed_value(left_order, key)
        right_value = self._replayed_value(right_order, key)

        # Common events form an ancestor-closed set, so the last common
        # event in either side's deterministic replay order is the fork.
        common = tuple(
            event_id
            for event_id in left_order
            if event_id in right_ids
        )
        fork = common[-1] if common else None

        if left_value != right_value:
            left_cause = next(
                (
                    event_id
                    for event_id in left_order
                    if event_id not in right_ids
                    and key in self._graph._changes[event_id]
                ),
                None,
            )
            right_cause = next(
                (
                    event_id
                    for event_id in right_order
                    if event_id not in left_ids
                    and key in self._graph._changes[event_id]
                ),
                None,
            )
        else:
            left_cause = None
            right_cause = None

        def cause_impact(
            order: list[str], head: str, cause: str | None
        ) -> tuple[tuple[str, ...], tuple[str, ...]]:
            if cause is None:
                return (), ()

            closure = set(order)
            # Reverse the parent edges within the closure so children can
            # be enumerated parent-to-child.
            children: dict[str, list[str]] = {
                event: [] for event in closure
            }
            for current in closure:
                for parent in self._graph._parents[current]:
                    if parent in closure:
                        children[parent].append(current)

            # Level-by-level breadth-first search gives the fewest-edge
            # paths; keeping the minimum full id tuple within a level
            # applies the Unicode code point tie-break, as in
            # explain_impact.
            paths: dict[str, tuple[str, ...]] = {cause: (cause,)}
            frontier = [cause]
            while frontier:
                candidates: dict[str, tuple[str, ...]] = {}
                for current in frontier:
                    current_path = paths[current]
                    for child in children[current]:
                        if child in paths:
                            continue
                        candidate = (*current_path, child)
                        best = candidates.get(child)
                        if best is None or candidate < best:
                            candidates[child] = candidate
                for child, path in candidates.items():
                    paths[child] = path
                frontier = list(candidates)

            path = tuple(paths[head])
            affected = tuple(
                event_id
                for event_id in order
                if event_id != cause and event_id in paths
            )
            return path, affected

        left_path, left_affected = cause_impact(
            left_order, head_a, left_cause
        )
        right_path, right_affected = cause_impact(
            right_order, head_b, right_cause
        )

        return {
            "key": key,
            "fork": fork,
            "left": {
                "value": left_value,
                "cause": left_cause,
                "path": left_path,
                "affected": left_affected,
            },
            "right": {
                "value": right_value,
                "cause": right_cause,
                "path": right_path,
                "affected": right_affected,
            },
        }

    def head(self, name: str) -> str:
        """Return the id of the branch's current head event."""
        self._require_nonempty_str(name, "name")
        self._require_known_branch(name)
        return self._heads[name]

    def replay(self, name: str) -> dict[str, int]:
        """Replay the branch's head via :meth:`EventGraph.replay`."""
        self._require_nonempty_str(name, "name")
        self._require_known_branch(name)
        return self._graph.replay(self._heads[name])

    def replay_at(self, name: str, at: int) -> dict[str, int]:
        """Replay the branch's head as of ``at`` via :meth:`EventGraph.replay_at`.

        ``name`` and ``at`` are validated (in that order) before the
        branch is looked up; the query is read-only, so the graph,
        branch heads and records are never modified.
        """
        self._require_nonempty_str(name, "name")
        at_value = EventGraph._require_int(at, "at")
        if at_value < 0:
            raise ValueError("at must be a non-negative int")
        self._require_known_branch(name)
        return self._graph.replay_at(self._heads[name], at_value)
