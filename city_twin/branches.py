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

    __slots__ = ("_file", "_buffer", "_start", "_pos", "_eof")

    def __init__(self, fileobj: Any) -> None:
        self._file = fileobj
        self._buffer = b""
        self._start = 0
        self._pos = 0
        self._eof = False

    @property
    def offset(self) -> int:
        """Absolute byte offset of the next unparsed byte."""
        return self._start + self._pos

    def _fill(self) -> None:
        if self._eof:
            return
        if self._pos:
            self._buffer = self._buffer[self._pos:]
            self._start += self._pos
            self._pos = 0
        chunk = self._file.read(self._CHUNK_SIZE)
        if chunk:
            self._buffer += chunk
        else:
            self._eof = True

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

    #: Largest single object :meth:`read_raw_object` will return. A
    #: legal segment record is a small fixed-shape object, so this bound
    #: is generous while keeping the raw read memory-bounded on a
    #: damaged document.
    _RAW_OBJECT_LIMIT = 65536

    def read_raw_object(self) -> bytes:
        """Read the JSON object starting at the current byte and
        return its exact bytes, braces included, without
        canonicalizing. Nested objects/arrays are tracked and string
        contents (with escapes) are skipped; an unexpected end or an
        object longer than :attr:`_RAW_OBJECT_LIMIT` raises
        :class:`ValueError`. The caller separately parses and
        canonicalizes the returned bytes, so only one small object is
        ever held."""
        raw = bytearray()
        raw.append(self.take())  # must be the opening brace
        if raw[-1] != 0x7B:  # '{'
            raise ValueError("expected a JSON object")
        depth = 1
        in_string = False
        escaped = False
        while depth:
            byte = self.take()
            raw.append(byte)
            if len(raw) > self._RAW_OBJECT_LIMIT:
                raise ValueError("JSON object is larger than the limit")
            if in_string:
                if escaped:
                    escaped = False
                elif byte == 0x5C:  # '\\'
                    escaped = True
                elif byte == 0x22:  # '"'
                    in_string = False
                continue
            if byte == 0x22:  # '"'
                in_string = True
            elif byte == 0x7B or byte == 0x5B:  # '{' or '['
                depth += 1
            elif byte == 0x7D or byte == 0x5D:  # '}' or ']'
                depth -= 1
        return bytes(raw)


class _HashingTee:
    """Forward :meth:`read`/:meth:`write` calls to a binary file while
    folding every passing byte into a SHA-256 hasher, so a streamed
    document can be digested without ever materializing its bytes."""

    __slots__ = ("_file", "_hasher")

    def __init__(self, fileobj: Any, hasher: Any) -> None:
        self._file = fileobj
        self._hasher = hasher

    @property
    def hexdigest(self) -> Any:
        return self._hasher.hexdigest

    def read(self, size: int = -1) -> bytes:
        data = self._file.read(size)
        self._hasher.update(data)
        return data

    def write(self, data: bytes) -> int:
        self._hasher.update(data)
        return self._file.write(data)

    def flush(self) -> None:
        self._file.flush()

    def fileno(self) -> int:
        return self._file.fileno()


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
        "_prefix_hasher",
        "_records_hasher",
        "_completed",
        "_completed_digest",
        "_completed_prefix_digest",
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
        # Running hash over every folded frame's canonical bytes; the
        # snapshot at a complete-segment boundary pins the exact chain
        # prefix a resuming build re-checks.
        self._prefix_hasher = hashlib.sha256()
        # Hash over the canonical bytes of every emitted segment record
        # in order, i.e. the fingerprint a freshly published index must
        # carry item for item.
        self._records_hasher = hashlib.sha256()
        # Sequence number and digest of the last full-size segment's
        # final frame (zero and the empty-prefix digest until one full
        # segment exists); a short trailing segment never advances the
        # resumable boundary.
        self._completed = 0
        self._completed_digest = ""
        self._completed_prefix_digest = hashlib.sha256(b"").hexdigest()

    @property
    def completed_boundary(self) -> int:
        """Sequence number of the last full-size segment boundary."""
        return self._completed

    @property
    def completed_boundary_digest(self) -> str:
        """Digest of the frame at :attr:`completed_boundary`."""
        return self._completed_digest

    @property
    def completed_prefix_digest(self) -> str:
        """SHA-256 of the canonical frame bytes folded through
        :attr:`completed_boundary` (the empty-input digest before the
        first full-size segment exists)."""
        return self._completed_prefix_digest

    @property
    def published_fingerprint(self) -> str:
        """SHA-256 over every emitted segment record's canonical bytes
        in emission order."""
        return self._records_hasher.hexdigest()

    def _reset(self) -> None:
        self._count = 0
        self._hasher = hashlib.sha256()

    def _emit(self) -> None:
        record = {
            "first": self._first,
            "last": self._last,
            "first_digest": self._first_digest,
            "last_digest": self._last_digest,
            "digest": self._hasher.hexdigest(),
        }
        self._on_segment(record)
        self._records_hasher.update(
            self._canonical(record).encode("utf-8")
        )
        # Only a full-size segment is a resumable boundary; a short
        # trailing segment is rebuilt wholesale on the next append.
        if self._count == self._size:
            self._completed = self._last
            self._completed_digest = self._last_digest
            self._completed_prefix_digest = (
                self._prefix_hasher.copy().hexdigest()
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
        frame_bytes = self._canonical(
            {
                "seq": frame["seq"],
                "prev": frame["prev"],
                "record": frame["record"],
                "digest": frame["digest"],
            }
        ).encode("utf-8")
        self._hasher.update(frame_bytes)
        self._prefix_hasher.update(frame_bytes)
        self._count += 1

    def open_segment_evidence(self) -> dict[str, Any] | None:
        """The segment record the accumulator would emit right now
        without emitting it, or ``None`` when the accumulator is empty.
        A resuming build compares this against the previous index's
        trailing (short) segment to prove the frames past the last
        complete boundary are byte-for-byte unchanged."""
        if not self._count:
            return None
        return {
            "first": self._first,
            "last": self._last,
            "first_digest": self._first_digest,
            "last_digest": self._last_digest,
            "digest": self._hasher.hexdigest(),
        }

    def finish(self) -> None:
        """Emit the trailing segment when one is partially filled."""
        if self._count:
            self._emit()


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

    #: Resumable build progress (see
    #: :meth:`build_recovery_audit_segments`). The progress document
    #: binds the segment size, the last complete-segment boundary, the
    #: authenticated frame count and chain head, and digests of the chain
    #: prefix through that boundary, the prefix's canonical frame bytes
    #: and the matching segment index's canonical bytes. It is a
    #: disposable cache like the segment index: it never stores audit
    #: bodies and the canonical chain stays the sole trusted source.
    _RECOVERY_CHAIN_PROGRESS_FORMAT = (
        "branching-city-twin/recovery-audit-chain-progress"
    )
    _RECOVERY_CHAIN_PROGRESS_VERSION = 1
    _RECOVERY_CHAIN_PROGRESS_SUPPORTED_VERSIONS = (1,)
    _RECOVERY_CHAIN_PROGRESS_KEYS = (
        "format",
        "version",
        "segment_size",
        "boundary",
        "boundary_digest",
        "frames",
        "head",
        "prefix_digest",
        "index_digest",
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
        """Validate a ``progress_path`` argument: only ``None`` or a
        non-empty :class:`str` is accepted. ``None`` opts out of
        resumable builds; a non-``str`` raises :class:`TypeError` and an
        empty string :class:`ValueError`."""
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
        """The progress file must never name the chain or index file,
        even through aliases, so publishing progress can never overwrite
        either cache or the canonical chain."""
        progress_real = os.path.realpath(progress_path)
        if progress_real == os.path.realpath(path):
            raise ValueError(
                "progress_path must not name the recovery chain file"
            )
        if progress_real == os.path.realpath(index_path):
            raise ValueError(
                "progress_path must not name the segment index file"
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
    def _validate_chain_frame_envelope(
        cls, fields: dict[str, Any], position: int, previous_digest: str
    ) -> str:
        """Validate one already-parsed frame's envelope without
        re-parsing its embedded audit record -- consecutive numbering
        from one, the predecessor link, the digest field's shape and the
        frame digest over the exact record bytes -- and return the
        frame's digest.

        This is the deliberately lighter check used for a prefix whose
        exact bytes the caller pins with a digest: the frame digest
        still covers the untouched record bytes, so any rewrite of an
        old frame breaks either the digest comparison, the predecessor
        chain or the caller's prefix digest. It is never used for frames
        that have not been authenticated before.
        """
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
        return digest

    @classmethod
    def _scan_recovery_chain(
        cls,
        fileobj: Any,
        on_frame: Any,
        *,
        lightweight_through: int = 0,
        parse_body_seqs: set[int] | None = None,
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

        ``lightweight_through`` names a previously authenticated
        boundary: frames at or below that sequence number still have
        their canonical encoding, consecutive numbering, predecessor
        links and frame digests verified frame by frame, but their
        embedded audit records are not parsed again (``body`` is
        ``None``). A caller may use this only while independently
        pinning those frames' exact bytes with a prefix digest, so the
        lighter pass cannot mask a changed or reordered prefix -- any
        altered byte breaks either a frame digest, the link chain or
        the pinned prefix digest. Frames past the boundary are always
        fully authenticated, embedded records included.
        """
        stream = _CanonicalJsonStream(fileobj)
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
                            if (
                                count <= lightweight_through
                                and not (
                                    parse_body_seqs
                                    and count in parse_body_seqs
                                )
                            ):
                                # Verify the frame envelope and its
                                # digest links without re-parsing the
                                # embedded record; the caller pins the
                                # prefix bytes separately.
                                digest = (
                                    cls._validate_chain_frame_envelope(
                                        fields, count, previous_digest
                                    )
                                )
                                body = None
                            else:
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

    @classmethod
    def _build_recovery_audit_segments_locked(
        cls, path: str, index_path: str, segment_size: int
    ) -> str:
        """Stream-authenticate the chain and durably publish its
        segmented index, returning the chain head digest; the caller
        holds the chain lock.

        Segment records are written as the segments complete while the
        frames stream past, so no more than one frame, audit body or
        segment is ever materialized. The segment bytes are UTF-8
        without a BOM, compact JSON without a trailing newline, and
        identical for identical chains and segment sizes. They are
        written to a temporary file in the index's directory, flushed,
        fsync-ed and atomically moved onto ``index_path`` with directory
        syncs; on any failure the temporary file is removed and neither
        the chain file nor a previous segment index is modified.
        """
        chain_file = cls._open_recovery_chain_locked(path)
        try:
            directory = os.path.dirname(os.path.abspath(index_path))
            fd, tmp_path = tempfile.mkstemp(
                prefix=".recovery-chain-segments-",
                suffix=".tmp",
                dir=directory,
            )
            published = False
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(
                        b'{"format":"'
                        + cls._RECOVERY_CHAIN_SEGMENTS_FORMAT.encode(
                            "ascii"
                        )
                        + b'","version":'
                        + str(cls._RECOVERY_CHAIN_SEGMENTS_VERSION).encode(
                            "ascii"
                        )
                        + b',"segment_size":'
                        + str(segment_size).encode("ascii")
                        + b',"segments":['
                    )
                    first = True

                    def on_segment(segment: dict[str, Any]) -> None:
                        nonlocal first
                        if not first:
                            handle.write(b",")
                        handle.write(
                            cls._canonical_json(segment).encode("utf-8")
                        )
                        first = False

                    segmenter = _ChainSegmenter(
                        segment_size, cls._canonical_json, on_segment
                    )
                    version, count, head = cls._scan_recovery_chain(
                        chain_file, segmenter.on_frame
                    )
                    segmenter.finish()
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
    def _scan_segment_entries(
        cls,
        stream: _CanonicalJsonStream,
        fingerprint: Any,
        on_record: Any = None,
    ) -> tuple[int, int, str | None]:
        """Stream the segment index's ``segments`` array, validating
        each segment's shape, contiguous coverage from sequence number
        one, bounds and digest evidence, and return the segment count,
        the last covered sequence number and the last segment's trailing
        boundary digest. ``fingerprint`` is a SHA-256 hasher updated
        with each segment's canonical bytes, so the caller can compare
        the segments item for item against the authenticated chain.
        When ``on_record`` is given it is invoked once per segment with
        the exact canonical record mapping (re-serialization of which
        reproduces the record's bytes); a resuming build uses it to copy
        the prefix records verbatim. Any defect raises
        :class:`ValueError`, which the caller treats as a corrupt
        (rebuildable) segment index."""
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
            record = {
                "first": first,
                "last": last,
                "first_digest": digests["first_digest"],
                "last_digest": digests["last_digest"],
                "digest": digests["digest"],
            }
            fingerprint.update(
                cls._canonical_json(record).encode("utf-8")
            )
            if on_record is not None:
                on_record(record)
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
    def _read_recovery_chain_segments(
        cls, index_path: str, *, detail: bool = False
    ) -> tuple[Any, ...] | None:
        """Read and fully validate the segment index, returning its
        binding -- ``(segment_size, chain_version, frames, head,
        segments_fingerprint)`` -- or ``None`` when the index is
        missing, unreadable or defective in any way. The fingerprint is
        the SHA-256 over every segment's canonical bytes in order, so
        the caller can check that each segment's bounds, boundary
        digests and frame-bytes digest match the authenticated chain
        item for item.

        With ``detail`` set, the result additionally carries the
        complete-segment boundary, the trailing short segment record
        (``None`` when the frame count is an exact multiple of the
        segment size) and the SHA-256 of the index file's exact bytes,
        which a resuming build uses to splice only the appended tail.

        The segment index is a disposable cache, so no index problem
        ever raises here: it only marks the index for a rebuild from
        the canonical chain. Only a fixed amount of memory is used
        regardless of the segment count.
        """
        file_hasher = hashlib.sha256()
        try:
            with open(index_path, "rb") as raw:
                tee = _HashingTee(raw, file_hasher)
                stream = _CanonicalJsonStream(tee)
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
                prefix_fingerprint = hashlib.sha256()
                boundary_digest: str | None = None
                tail_record: dict[str, Any] | None = None

                def on_record(record: dict[str, Any]) -> None:
                    nonlocal tail_record, boundary_digest
                    # Keys are ordered, so segment_size_value is known
                    # before the segments array streams past. Records at
                    # a complete-segment boundary anchor the prefix a
                    # resume re-checks; the final record is the only
                    # possibly short trailing segment.
                    if (
                        isinstance(segment_size_value, int)
                        and not isinstance(segment_size_value, bool)
                        and record["last"] % segment_size_value == 0
                    ):
                        prefix_fingerprint.update(
                            cls._canonical_json(record).encode("utf-8")
                        )
                        boundary_digest = record["last_digest"]
                        tail_record = None
                    else:
                        tail_record = record
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
                                    on_record if detail else None,
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
                binding: tuple[Any, ...] = (
                    segment_size_value,
                    chain_version,
                    declared_frames,
                    head,
                    fingerprint.hexdigest(),
                )
                if not detail:
                    return binding
                remainder = declared_frames % segment_size_value
                complete = declared_frames - remainder
                if remainder == 0:
                    tail: dict[str, Any] | None = None
                    if count < 1 or boundary_digest is None:
                        return None
                else:
                    if (
                        tail_record is None
                        or tail_record["first"] != complete + 1
                        or tail_record["last"] != declared_frames
                    ):
                        return None
                    tail = tail_record
                return (
                    *binding,
                    complete,
                    boundary_digest,
                    prefix_fingerprint.hexdigest(),
                    tail,
                    file_hasher.hexdigest(),
                )
        except (OSError, ValueError, RecursionError):
            return None

    @classmethod
    def _read_recovery_progress(
        cls, progress_path: str
    ) -> tuple[int, int, str, int, str, str, str] | None:
        """Read and strictly validate a resumable-build progress
        document, returning its binding -- ``(segment_size, boundary,
        boundary_digest, frames, head, prefix_digest, index_digest)``
        -- or ``None`` when the file is missing, truncated or defective
        in any way, which the caller treats as a one-shot rebuild from
        the canonical chain.

        A genuinely missing file is the normal first-build case and
        yields ``None``; every other open or read failure propagates as
        :class:`OSError`. The document binds the segment size, the last
        full-size segment boundary and its frame digest, the
        authenticated frame count and head, the SHA-256 of the prefix's
        canonical frame bytes through that boundary and the SHA-256 of
        the matching segment index's exact bytes; it never carries
        audit bodies.
        """
        try:
            with open(progress_path, "rb") as handle:
                raw = handle.read()
        except FileNotFoundError:
            return None
        if raw.startswith(b"\xef\xbb\xbf"):
            return None
        try:
            text = raw.decode("utf-8")
            document = json.loads(
                text, object_pairs_hook=cls._reject_duplicate_json_keys
            )
        except (UnicodeDecodeError, ValueError, RecursionError):
            return None
        if not isinstance(document, dict):
            return None
        if set(document.keys()) != set(cls._RECOVERY_CHAIN_PROGRESS_KEYS):
            return None
        if document["format"] != cls._RECOVERY_CHAIN_PROGRESS_FORMAT:
            return None
        version = document["version"]
        if (
            isinstance(version, bool)
            or not isinstance(version, int)
            or version
            not in cls._RECOVERY_CHAIN_PROGRESS_SUPPORTED_VERSIONS
        ):
            return None
        segment_size = document["segment_size"]
        boundary = document["boundary"]
        frames = document["frames"]
        for value in (segment_size, boundary, frames):
            if isinstance(value, bool) or not isinstance(value, int):
                return None
        if segment_size < 1 or boundary < 0 or frames < 1:
            return None
        if boundary % segment_size or boundary > frames:
            return None
        if frames - boundary >= segment_size:
            return None
        for name in (
            "boundary_digest",
            "head",
            "prefix_digest",
            "index_digest",
        ):
            value = document[name]
            if (
                not isinstance(value, str)
                or len(value) != 64
                or set(value) - cls._HEX_DIGITS
            ):
                return None
        if boundary == 0 and (
            document["boundary_digest"]
            != cls._RECOVERY_CHAIN_GENESIS_PREV
        ):
            return None
        return (
            segment_size,
            boundary,
            document["boundary_digest"],
            frames,
            document["head"],
            document["prefix_digest"],
            document["index_digest"],
        )

    @classmethod
    def _progress_bytes(
        cls,
        segment_size: int,
        boundary: int,
        boundary_digest: str,
        frames: int,
        head: str,
        prefix_digest: str,
        index_digest: str,
    ) -> bytes:
        """Deterministic compact JSON for a progress binding: UTF-8
        without a BOM, no whitespace and no trailing newline, identical
        byte for byte for identical bindings."""
        document = {
            "format": cls._RECOVERY_CHAIN_PROGRESS_FORMAT,
            "version": cls._RECOVERY_CHAIN_PROGRESS_VERSION,
            "segment_size": segment_size,
            "boundary": boundary,
            "boundary_digest": boundary_digest,
            "frames": frames,
            "head": head,
            "prefix_digest": prefix_digest,
            "index_digest": index_digest,
        }
        return cls._canonical_json(document).encode("utf-8")

    @classmethod
    def _publish_files_rollback(
        cls,
        publications: tuple[tuple[str, str], ...],
    ) -> None:
        """Durably publish ordered ``(temporary, target)`` pairs under a
        rollback protocol; each temporary already sits in its target's
        directory and is flushed and fsync-ed.

        Every pre-existing target is first moved aside to a unique
        same-directory backup; the temps are then replaced onto the
        targets one by one, with the containing directory synced after
        each move. If any replace or directory sync fails, every target
        is restored byte for byte from its backup (removed entirely
        when it did not exist before), the backups and unpublished
        temps are cleaned up, the directories are synced and the
        :class:`OSError` propagates -- so a failed publication leaves
        every previous target exactly as it was, never a partial set.
        On success the backups are removed and every directory synced.
        """
        backups: dict[str, str] = {}
        installed: list[str] = []

        def directory_of(target: str) -> str:
            return os.path.dirname(os.path.abspath(target))

        try:
            for tmp, target in publications:
                if os.path.exists(target):
                    backup = (
                        target
                        + f".recovery-backup-{uuid.uuid4().hex}.bak"
                    )
                    os.replace(target, backup)
                    backups[target] = backup
                os.replace(tmp, target)
                installed.append(target)
                cls._fsync_directory(directory_of(target))
        except OSError:
            for target in reversed(installed):
                with contextlib.suppress(OSError):
                    os.remove(target)
                backup = backups.get(target)
                if backup is not None:
                    with contextlib.suppress(OSError):
                        os.replace(backup, target)
            for target, backup in backups.items():
                if target not in installed and os.path.exists(backup):
                    with contextlib.suppress(OSError):
                        os.replace(backup, target)
            for tmp, target in publications:
                if target not in installed:
                    with contextlib.suppress(OSError):
                        os.remove(tmp)
            for backup in backups.values():
                with contextlib.suppress(OSError):
                    os.remove(backup)
            synced: set[str] = set()
            for _tmp, target in publications:
                directory = directory_of(target)
                if directory not in synced:
                    synced.add(directory)
                    with contextlib.suppress(OSError):
                        cls._fsync_directory(directory)
            raise
        for backup in backups.values():
            with contextlib.suppress(OSError):
                os.remove(backup)
        synced = set()
        for _tmp, target in publications:
            directory = directory_of(target)
            if directory not in synced:
                synced.add(directory)
                cls._fsync_directory(directory)

    @classmethod
    def _verify_published_segments(
        cls,
        index_path: str,
        progress_path: str | None,
        segment_size: int,
        boundary: int,
        boundary_digest: str,
        count: int,
        head: str,
        prefix_digest: str,
        segment_records_fingerprint: str,
        chain_version: int,
    ) -> None:
        """Re-read the just-published caches and prove they bind the
        authenticated chain: the index's segment fingerprint must
        equal the fingerprint of the records the authenticated scan
        emitted, and the progress (when present) must bind the same
        head, boundary, prefix and exact index bytes. Any mismatch
        raises :class:`ValueError`."""
        detail = cls._read_recovery_chain_segments(index_path, detail=True)
        if detail is None:
            raise ValueError(
                "recovery chain segment index is unusable after "
                "publication"
            )
        (
            idx_size,
            idx_version,
            idx_frames,
            idx_head,
            idx_fingerprint,
            idx_boundary,
            idx_boundary_digest,
            _idx_prefix_fingerprint,
            _idx_tail,
            idx_file_digest,
        ) = detail
        if (
            idx_size != segment_size
            or idx_version != chain_version
            or idx_frames != count
            or not hmac.compare_digest(idx_head, head)
            or idx_fingerprint != segment_records_fingerprint
            or idx_boundary != boundary
            or (
                boundary
                and not hmac.compare_digest(
                    idx_boundary_digest, boundary_digest
                )
            )
        ):
            raise ValueError(
                "recovery chain segment index does not match the "
                "authenticated chain after publication"
            )
        if progress_path is None:
            return
        progress = cls._read_recovery_progress(progress_path)
        if progress is None:
            raise ValueError(
                "recovery chain progress is unusable after publication"
            )
        if progress != (
            segment_size,
            boundary,
            boundary_digest,
            count,
            head,
            prefix_digest,
            idx_file_digest,
        ):
            raise ValueError(
                "recovery chain progress does not match the "
                "authenticated chain after publication"
            )

    @classmethod
    def _full_segment_build_locked(
        cls,
        path: str,
        index_path: str,
        segment_size: int,
        wanted: set[int],
        progress_path: str | None,
        extension_pin: tuple[int, int, str, int, str, str] | None = None,
    ) -> tuple[int, str, dict[int, dict[str, Any]]]:
        """Fully authenticate the chain from frame one, fold every
        frame into segments, durably publish the segment index (and,
        when given, the progress file) and return the frame count, head
        and wanted bodies; the caller holds the chain lock.

        This is the trusted path used for the first build and exactly
        once per call after an unusable progress/index pair. It
        streams the whole chain to the full authentication standard
        with bounded memory. Both caches are written as same-directory
        temps and published under the rollback protocol, so no
        open, write, flush, replace or sync failure can leave a
        previously valid cache byte-different; a post-publish re-read
        verifies the published index item for item against the
        authenticated chain.

        ``extension_pin`` carries ``(segment_size, boundary,
        boundary_digest, old_frames, old_head, prefix_digest)`` from a
        structurally valid progress whose index could not drive an
        incremental continuation: even on the fallback rebuild the
        chain must still be an append-only extension of that
        authenticated prefix (the boundary frame, the old head frame
        and the prefix-frame-bytes digest must all match and the chain
        may not be shorter), so a lost index can never legitimize a
        truncated or rewritten chain; a violation raises
        :class:`ValueError` rather than rebuilding over it.
        """
        directory = os.path.dirname(os.path.abspath(index_path))
        captured: dict[int, dict[str, Any]] = {}
        index_fd, index_tmp = tempfile.mkstemp(
            prefix=".recovery-chain-segments-",
            suffix=".tmp",
            dir=directory,
        )
        progress_tmp: str | None = None
        publications: list[tuple[str, str]] = [(index_tmp, index_path)]
        published = False
        pin_prefix: Any = None
        pin_boundary_frame = False
        pin_old_head_frame = False
        index_handle: Any = None
        if extension_pin is not None:
            (
                _pin_size,
                pin_boundary,
                pin_boundary_digest,
                pin_old_frames,
                pin_old_head,
                pin_prefix_digest,
            ) = extension_pin
            pin_prefix = hashlib.sha256()
        try:
            handle = os.fdopen(index_fd, "wb")
            index_handle = handle
            index_hasher = hashlib.sha256()
            tee = _HashingTee(handle, index_hasher)
            tee.write(
                b'{"format":"'
                + cls._RECOVERY_CHAIN_SEGMENTS_FORMAT.encode("ascii")
                + b'","version":'
                + str(cls._RECOVERY_CHAIN_SEGMENTS_VERSION).encode("ascii")
                + b',"segment_size":'
                + str(segment_size).encode("ascii")
                + b',"segments":['
            )
            first = True

            def on_segment(segment: dict[str, Any]) -> None:
                nonlocal first
                if not first:
                    tee.write(b",")
                tee.write(cls._canonical_json(segment).encode("utf-8"))
                first = False

            segmenter = _ChainSegmenter(
                segment_size, cls._canonical_json, on_segment
            )

            def on_frame(frame: dict[str, Any]) -> None:
                nonlocal pin_boundary_frame, pin_old_head_frame
                if frame["seq"] in wanted:
                    captured[frame["seq"]] = frame["body"]
                if extension_pin is not None:
                    if frame["seq"] <= pin_boundary:
                        pin_prefix.update(
                            cls._canonical_json(
                                {
                                    "seq": frame["seq"],
                                    "prev": frame["prev"],
                                    "record": frame["record"],
                                    "digest": frame["digest"],
                                }
                            ).encode("utf-8")
                        )
                    if pin_boundary and frame["seq"] == pin_boundary:
                        pin_boundary_frame = hmac.compare_digest(
                            frame["digest"], pin_boundary_digest
                        )
                    if frame["seq"] == pin_old_frames:
                        pin_old_head_frame = hmac.compare_digest(
                            frame["digest"], pin_old_head
                        )
                segmenter.on_frame(frame)

            chain_file = cls._open_recovery_chain_locked(path)
            try:
                # A fallback rebuild always fully authenticates every
                # frame -- embedded records included -- exactly like a
                # clean first build; the extension pin is an additional
                # append-only constraint on top of that standard, never
                # a lighter authentication.
                version, count, head = cls._scan_recovery_chain(
                    chain_file, on_frame
                )
            finally:
                chain_file.close()
            if extension_pin is not None:
                if count < pin_old_frames:
                    raise ValueError(
                        "recovery chain was truncated behind the "
                        "authenticated progress frame count"
                    )
                if pin_boundary and not pin_boundary_frame:
                    raise ValueError(
                        "recovery chain progress boundary frame is "
                        "missing or rewritten"
                    )
                if not pin_old_head_frame:
                    raise ValueError(
                        "recovery chain progress head frame is missing "
                        "or rewritten"
                    )
                if not hmac.compare_digest(
                    pin_prefix.hexdigest(), pin_prefix_digest
                ):
                    raise ValueError(
                        "recovery chain prefix does not match the "
                        "authenticated progress prefix digest"
                    )
            segmenter.finish()
            tee.write(
                (
                    '],"chain_version":%d,"frames":%d,"head":"%s"}'
                    % (version, count, head)
                ).encode("ascii")
            )
            tee.flush()
            os.fsync(tee.fileno())
            handle.close()
            index_digest = index_hasher.hexdigest()
            boundary = segmenter.completed_boundary
            boundary_digest = (
                segmenter.completed_boundary_digest
                if boundary
                else cls._RECOVERY_CHAIN_GENESIS_PREV
            )
            prefix_digest = segmenter.completed_prefix_digest
            if progress_path is not None:
                progress_directory = os.path.dirname(
                    os.path.abspath(progress_path)
                )
                progress_fd, progress_tmp = tempfile.mkstemp(
                    prefix=".recovery-chain-progress-",
                    suffix=".tmp",
                    dir=progress_directory,
                )
                with os.fdopen(progress_fd, "wb") as progress_handle:
                    progress_handle.write(
                        cls._progress_bytes(
                            segment_size,
                            boundary,
                            boundary_digest,
                            count,
                            head,
                            prefix_digest,
                            index_digest,
                        )
                    )
                    progress_handle.flush()
                    os.fsync(progress_handle.fileno())
                publications.append((progress_tmp, progress_path))
            cls._publish_files_rollback(tuple(publications))
            published = True
            cls._verify_published_segments(
                index_path,
                progress_path,
                segment_size,
                boundary,
                boundary_digest,
                count,
                head,
                prefix_digest,
                segmenter.published_fingerprint,
                version,
            )
            return count, head, captured
        finally:
            # A chain defect raised while the index temp is still open
            # must not leak the descriptor; closing twice is harmless.
            if index_handle is not None and not index_handle.closed:
                with contextlib.suppress(OSError):
                    index_handle.close()
            if not published:
                for tmp, _target in publications:
                    with contextlib.suppress(OSError):
                        os.remove(tmp)

    @classmethod
    def _locate_segments_splice(
        cls, index_path: str, boundary: int, segment_size: int
    ) -> int | None:
        """Stream the segment index only far enough to locate the byte
        offset immediately after the complete-segment record covering
        sequence ``boundary`` (or immediately after the opening ``[``
        when ``boundary`` is zero), using fixed buffers. Returns
        ``None`` when the records through the boundary are not a
        canonical, continuous run of full-size segment records. The
        index has already been validated by the caller in the same
        lock hold, so a genuine open/read failure propagates as
        :class:`OSError` rather than masquerading as a corrupt cache.
        """
        try:
            with open(index_path, "rb") as fileobj:
                stream = _CanonicalJsonStream(fileobj)
                if stream.peek() is None:
                    return None
                stream.expect(0x7B)  # '{'
                if stream.peek() == 0x7D:  # '}'
                    return None
                while True:
                    if stream.peek() != 0x22:  # '"'
                        return None
                    key = stream.parse_string()
                    stream.expect(0x3A)  # ':'
                    if key == "segments":
                        stream.expect(0x5B)  # '['
                        break
                    stream.skip_value()
                    byte = stream.take()
                    if byte != 0x2C:  # ','
                        return None
                if boundary == 0:
                    return stream.offset
                expected_first = 1
                while True:
                    raw = stream.read_raw_object()
                    try:
                        record = json.loads(
                            raw.decode("utf-8"),
                            object_pairs_hook=(
                                cls._reject_duplicate_json_keys
                            ),
                        )
                    except ValueError:
                        return None
                    if not isinstance(record, dict) or set(
                        record.keys()
                    ) != set(cls._RECOVERY_CHAIN_SEGMENT_KEYS):
                        return None
                    if cls._canonical_json(record).encode("utf-8") != raw:
                        return None
                    first = record["first"]
                    last = record["last"]
                    for value in (first, last):
                        if (
                            isinstance(value, bool)
                            or not isinstance(value, int)
                        ):
                            return None
                    if (
                        first != expected_first
                        or last - first + 1 != segment_size
                    ):
                        return None
                    point = stream.offset
                    byte = stream.take()
                    if last == boundary:
                        if byte not in (0x2C, 0x5D):  # ',' or ']'
                            return None
                        return point
                    expected_first = last + 1
                    if byte != 0x2C:  # ','
                        return None
        except OSError:
            raise
        except (ValueError, RecursionError):
            return None

    @classmethod
    def _resume_segment_build_locked(
        cls,
        path: str,
        index_path: str,
        progress_path: str,
        segment_size: int,
        wanted: set[int],
    ) -> tuple[int, str, dict[int, dict[str, Any]]]:
        """Continue a build from a mutually consistent progress/index
        pair, scanning the chain exactly once, and return the frame
        count, head and wanted bodies; the caller holds the chain lock.

        The progress and index are read and cross-checked before the
        chain is opened, so a missing or unusable cache pair falls
        through to one full rebuild without scanning the chain. Once
        the pair claims an authenticated prefix, a disagreement with
        the chain is a canonical-chain defect and raises
        :class:`ValueError`: a chain truncated behind the authenticated
        frame count, a rewritten, reordered or duplicated prefix
        frame, a broken boundary or head link, or a frame that will not
        canonically parse.

        Old frames through the previous frame count are checked with
        fixed buffers only (envelope shape, predecessor links, frame
        digests, the pinned prefix-frame-bytes digest, the boundary
        frame and the index's complete-segment and trailing-segment
        evidence); queried old endpoint records and every appended
        frame are fully parsed, so neither cache can ever answer over
        an unauthenticated record. The replacement tail segments are
        spliced onto the exact old prefix-index bytes and both caches
        are published under the rollback protocol and re-verified.
        """
        progress = cls._read_recovery_progress(progress_path)
        if progress is None:
            # No trustworthy prior binding: this is the one free full
            # rebuild (a missing or unparseable progress file).
            return cls._full_segment_build_locked(
                path, index_path, segment_size, wanted, progress_path
            )
        (
            prog_size,
            boundary,
            boundary_digest,
            old_frames,
            old_head,
            prefix_digest,
            index_digest,
        ) = progress
        extension_pin = (
            prog_size,
            boundary,
            boundary_digest,
            old_frames,
            old_head,
            prefix_digest,
        )
        detail: tuple[Any, ...] | None = None
        if prog_size == segment_size:
            detail = cls._read_recovery_chain_segments(
                index_path, detail=True
            )
        if detail is not None:
            (
                idx_size,
                idx_version,
                idx_frames,
                idx_head,
                _idx_fingerprint,
                idx_boundary,
                idx_boundary_digest,
                idx_prefix_fingerprint,
                idx_tail,
                idx_file_digest,
            ) = detail
            pair_consistent = (
                idx_size == segment_size
                and idx_frames == old_frames
                and hmac.compare_digest(idx_head, old_head)
                and idx_boundary == boundary
                and hmac.compare_digest(idx_file_digest, index_digest)
                and (
                    boundary == 0
                    or hmac.compare_digest(
                        idx_boundary_digest, boundary_digest
                    )
                )
            )
        else:
            pair_consistent = False
        if not pair_consistent:
            # The progress still binds a once-authenticated chain
            # prefix, so even the fallback rebuild must prove the chain
            # merely extends it; a lost or stale index can never
            # legitimise a truncated or rewritten chain.
            return cls._full_segment_build_locked(
                path,
                index_path,
                segment_size,
                wanted,
                progress_path,
                extension_pin=extension_pin,
            )

        captured: dict[int, dict[str, Any]] = {}
        prefix_frame_hasher = hashlib.sha256()
        prefix_index_hasher = hashlib.sha256()
        index_prefix_ok = boundary == 0
        tail_ok = idx_tail is None
        old_head_seen = False
        new_records: list[dict[str, Any]] = []

        def on_segment(segment: dict[str, Any]) -> None:
            nonlocal index_prefix_ok
            if segment["last"] <= boundary:
                prefix_index_hasher.update(
                    cls._canonical_json(segment).encode("utf-8")
                )
                if segment["last"] == boundary:
                    index_prefix_ok = True
            else:
                new_records.append(segment)

        segmenter = _ChainSegmenter(
            segment_size, cls._canonical_json, on_segment
        )

        def on_frame(frame: dict[str, Any]) -> None:
            nonlocal tail_ok, old_head_seen
            seq = frame["seq"]
            if seq in wanted:
                captured[seq] = frame["body"]
            if seq <= boundary:
                prefix_frame_hasher.update(
                    cls._canonical_json(
                        {
                            "seq": frame["seq"],
                            "prev": frame["prev"],
                            "record": frame["record"],
                            "digest": frame["digest"],
                        }
                    ).encode("utf-8")
                )
            segmenter.on_frame(frame)
            if boundary and seq == boundary and not hmac.compare_digest(
                frame["digest"], boundary_digest
            ):
                raise ValueError(
                    "recovery chain progress boundary digest does not "
                    "match the authenticated chain"
                )
            if seq == old_frames:
                if not hmac.compare_digest(frame["digest"], old_head):
                    raise ValueError(
                        "recovery chain progress head does not match "
                        "the frame at the authenticated frame count"
                    )
                old_head_seen = True
                if idx_tail is not None:
                    tail_ok = (
                        segmenter.open_segment_evidence() == idx_tail
                    )

        chain_file = cls._open_recovery_chain_locked(path)
        try:
            version, count, head = cls._scan_recovery_chain(
                chain_file,
                on_frame,
                lightweight_through=old_frames,
                parse_body_seqs=wanted,
            )
        finally:
            chain_file.close()
        if count < old_frames:
            raise ValueError(
                "recovery chain was truncated behind the authenticated "
                "progress frame count"
            )
        if not old_head_seen:
            raise ValueError(
                "recovery chain does not contain the authenticated "
                "progress head frame"
            )
        segmenter.finish()
        if not index_prefix_ok or (idx_tail is not None and not tail_ok):
            raise ValueError(
                "recovery chain prefix does not match the authenticated "
                "segment-index evidence"
            )
        if not hmac.compare_digest(
            prefix_frame_hasher.hexdigest(), prefix_digest
        ):
            raise ValueError(
                "recovery chain prefix does not match the authenticated "
                "progress prefix digest"
            )
        if not hmac.compare_digest(
            prefix_index_hasher.hexdigest(), idx_prefix_fingerprint
        ):
            raise ValueError(
                "recovery chain segment index prefix does not match "
                "its authenticated evidence"
            )
        if count == old_frames:
            if not hmac.compare_digest(head, old_head):
                raise ValueError(
                    "recovery chain head does not match the "
                    "authenticated progress head"
                )
            # Nothing was appended: the existing caches already bind
            # this exact chain, so nothing is published.
            return count, head, captured
        tail_records = [
            record for record in new_records if record["first"] > boundary
        ]
        if not tail_records:
            raise ValueError(
                "recovery chain continuation produced no replacement "
                "tail segment"
            )
        new_boundary = segmenter.completed_boundary
        new_boundary_digest = (
            segmenter.completed_boundary_digest
            if new_boundary
            else cls._RECOVERY_CHAIN_GENESIS_PREV
        )
        new_prefix_digest = segmenter.completed_prefix_digest
        splice_point = cls._locate_segments_splice(
            index_path, boundary, segment_size
        )
        if splice_point is None:
            raise ValueError(
                "recovery chain segment index cannot be spliced at the "
                "authenticated boundary"
            )
        directory = os.path.dirname(os.path.abspath(index_path))
        index_fd, index_tmp = tempfile.mkstemp(
            prefix=".recovery-chain-segments-",
            suffix=".tmp",
            dir=directory,
        )
        progress_tmp: str | None = None
        publications: list[tuple[str, str]] = [(index_tmp, index_path)]
        published = False
        try:
            with open(index_path, "rb") as old_index, os.fdopen(
                index_fd, "wb"
            ) as handle:
                index_hasher = hashlib.sha256()
                tee = _HashingTee(handle, index_hasher)
                remaining = splice_point
                while remaining:
                    chunk = old_index.read(min(remaining, 65536))
                    if not chunk:
                        raise ValueError(
                            "recovery chain segment index ended before "
                            "the authenticated splice point"
                        )
                    tee.write(chunk)
                    remaining -= len(chunk)
                if boundary:
                    tee.write(b",")
                tee.write(
                    b",".join(
                        cls._canonical_json(record).encode("utf-8")
                        for record in tail_records
                    )
                )
                if version != idx_version:
                    raise ValueError(
                        "recovery chain schema version changed without "
                        "a rebuilt segment index"
                    )
                tee.write(
                    (
                        '],"chain_version":%d,"frames":%d,"head":"%s"}'
                        % (idx_version, count, head)
                    ).encode("ascii")
                )
                tee.flush()
                os.fsync(tee.fileno())
                index_digest_new = index_hasher.hexdigest()
            progress_directory = os.path.dirname(
                os.path.abspath(progress_path)
            )
            progress_fd, progress_tmp = tempfile.mkstemp(
                prefix=".recovery-chain-progress-",
                suffix=".tmp",
                dir=progress_directory,
            )
            with os.fdopen(progress_fd, "wb") as progress_handle:
                progress_handle.write(
                    cls._progress_bytes(
                        segment_size,
                        new_boundary,
                        new_boundary_digest,
                        count,
                        head,
                        new_prefix_digest,
                        index_digest_new,
                    )
                )
                progress_handle.flush()
                os.fsync(progress_handle.fileno())
            publications.append((progress_tmp, progress_path))
            cls._publish_files_rollback(tuple(publications))
            published = True
            cls._verify_published_segments(
                index_path,
                progress_path,
                segment_size,
                new_boundary,
                new_boundary_digest,
                count,
                head,
                new_prefix_digest,
                segmenter.published_fingerprint,
                version,
            )
            return count, head, captured
        finally:
            if not published:
                for tmp, _target in publications:
                    with contextlib.suppress(OSError):
                        os.remove(tmp)

    @classmethod
    def _segmented_chain_scan(
        cls,
        path: str,
        index_path: str,
        segment_size: int,
        wanted: set[int],
        progress_path: str | None = None,
    ) -> tuple[int, str, dict[int, dict[str, Any]]]:
        """Authenticate the chain with bounded memory under the chain
        lock, keep the segment index (and, when given, the progress
        file) fresh and return the frame count, the head digest and the
        detached audit bodies of the ``wanted`` sequence numbers.

        Without ``progress_path`` the behaviour is exactly the original
        one: one locked full chain scan, one segment-index check and at
        most one rebuild of a missing, stale or corrupt index. With a
        progress path the progress, the segment index and the chain are
        read inside the same lock hold and the chain is scanned at most
        once: when the progress binds a prefix the chain only appends
        to, the old prefix is checked with fixed buffers (envelope
        links, frame digests, the pinned prefix digest, the boundary
        frame and the index's complete-segment and trailing-segment
        evidence) and only appended frames are fully parsed and folded
        into a replacement tail; a missing, truncated or mismatching
        progress/index pair triggers exactly one full rebuild from the
        canonical chain. Either way a post-publish re-read proves the
        index item for item against the authenticated chain, and
        :class:`ValueError` is raised if it cannot match; the chain
        stays the only trusted source, so neither cache can mask a
        chain defect.
        """
        lock_path = cls._recovery_chain_lock_path(path)
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            cls._acquire_chain_lock(fd, None)
            try:
                if progress_path is not None:
                    return cls._resume_segment_build_locked(
                        path,
                        index_path,
                        progress_path,
                        segment_size,
                        wanted,
                    )
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
                    cls._build_recovery_audit_segments_locked(
                        path, index_path, segment_size
                    )
                    binding = cls._read_recovery_chain_segments(index_path)
                    if binding != expected:
                        raise ValueError(
                            "recovery chain segment index does not match "
                            "the authenticated chain"
                        )
                return count, head, captured
            finally:
                cls._release_chain_lock(fd)
        finally:
            os.close(fd)

    @classmethod
    def build_recovery_audit_segments(
        cls,
        path: Any,
        index_path: Any,
        segment_size: Any,
        progress_path: Any = None,
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
        raises :class:`ValueError`). A missing or unreadable chain file,
        a missing index directory or any failed open, flush, replace or
        sync raises :class:`OSError` and publishes nothing.

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
        They are written to a temporary file in the index's directory,
        flushed, fsync-ed and atomically moved onto ``index_path`` with
        directory syncs, so concurrent readers see either the old index
        or the new one, never a partial document.

        The segment index is a disposable cache: the canonical chain
        stays the only trusted source, and
        :meth:`diff_recovery_audit_ranges` rebuilds it whenever it is
        missing, stale, truncated or corrupt. A crashed or failed build
        never rewrites the chain file, leaves any previous segment index
        valid or recognizably stale, and removes its temporary file.
        Whether the build succeeds or fails, nothing under the
        generation directory -- business state, audits, idempotency
        records or the journal attachment -- is modified.

        ``progress_path`` is optional and defaults to ``None``, which
        keeps the call, its result and its authentication semantics
        exactly as above. When given, it must be a non-empty
        :class:`str` (a non-``str``/``None`` raises :class:`TypeError`,
        an empty string :class:`ValueError`) and must not name the
        chain or index file, even through an alias
        (:class:`ValueError`); it is validated after the existing
        arguments, whose order is unchanged. The build then reads the
        progress, the segment index and the chain inside one chain-lock
        hold and scans the chain at most once: a progress binding a
        prefix the chain only appends to is continued from the last
        full-size segment boundary (the old prefix is checked with
        fixed buffers against the boundary frame and the pinned
        prefix/index digests, and only appended frames are fully
        parsed), while a missing, truncated or mismatching
        progress/index pair triggers exactly one full rebuild --
        though a structurally valid progress still forces the chain to
        be an append-only extension of its authenticated prefix, so a
        lost index can never legitimize a truncated or rewritten
        chain. The progress is deterministic compact JSON binding the
        segment size, boundary, frame count, chain head, prefix digest
        and index digest; it stores no audit bodies. The index and
        progress are written as same-directory temps and published
        together under a rollback protocol, so a failed open, read,
        write, flush, replace or directory sync raises
        :class:`OSError` (content mismatches raise
        :class:`ValueError`), returns no partial result and restores
        both previous caches byte for byte; only this call's temps and
        backups are cleaned up. After a successful continuation the new
        index and progress are jointly bound to the current chain
        head, so a restart resumes from the latest complete boundary.
        """
        cls._validate_chain_path(path)
        cls._validate_index_path(index_path)
        cls._validate_index_target(path, index_path)
        cls._validate_segment_size(segment_size)
        cls._validate_progress_path(progress_path)
        if progress_path is not None:
            cls._validate_progress_target(path, index_path, progress_path)
        lock_path = cls._recovery_chain_lock_path(path)
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            cls._acquire_chain_lock(fd, None)
            try:
                if progress_path is None:
                    return cls._build_recovery_audit_segments_locked(
                        path, index_path, segment_size
                    )
                _count, head, _captured = cls._resume_segment_build_locked(
                    path, index_path, progress_path, segment_size, set()
                )
                return head
            finally:
                cls._release_chain_lock(fd)
        finally:
            os.close(fd)

    @classmethod
    def diff_recovery_audit_ranges(
        cls,
        path: Any,
        expected_head: Any,
        ranges: Any,
        index_path: Any,
        segment_size: Any,
        progress_path: Any = None,
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
        :meth:`build_recovery_audit_segments`.

        The whole batch performs exactly one locked chain scan and one
        segment check: the chain is streamed and fully authenticated
        under the same chain lock that serializes appends, only the
        queried endpoint records are materialized, and the segment index
        at ``index_path`` is checked segment by segment against the
        authenticated chain and rebuilt once from it when missing,
        stale, truncated or corrupt (an index that still does not match
        after the rebuild raises :class:`ValueError`; a failed rebuild
        propagates :class:`OSError`). Memory grows only with the largest
        single frame and fixed buffers -- never with the total frame
        count or a range span. An empty ``ranges`` still completes the
        path, head, segment-size, chain and segment-index
        authentication and then returns an empty tuple.

        ``progress_path`` is optional and trails the existing
        arguments; ``None`` (the default) leaves the call, results and
        authentication semantics exactly as above. When given it is
        validated like the ``progress_path`` of
        :meth:`build_recovery_audit_segments` (non-empty :class:`str`,
        never aliasing the chain or index; :class:`TypeError`,
        :class:`ValueError` and :class:`OSError` follow the same rules),
        and the progress, segment index and chain are read together
        under the chain lock so the chain is scanned at most once per
        batch: a matching progress drives an incremental continuation
        from the latest complete segment boundary (the old prefix is
        checked with fixed buffers and only appended frames are parsed
        fully, while queried old endpoint records are always
        re-authenticated), and a missing, truncated or mismatching
        progress/index pair triggers exactly one full rebuild, with a
        structurally valid progress still requiring the chain to be an
        append-only extension of its authenticated prefix. The index
        and progress are published together under the rollback
        protocol, so any filesystem failure raises :class:`OSError`,
        returns no partial results and restores both previous caches
        byte for byte. Concurrent appends, builds and queries compete
        for the same chain lock and observe only the complete chain
        from before or after an append.

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
        wanted: set[int] = set()
        for start, end in normalized:
            wanted.add(start)
            wanted.add(end)
        last, head, captured = cls._segmented_chain_scan(
            path, index_path, segment_size, wanted, progress_path
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
