"""In-memory event DAG with deterministic replay."""

from __future__ import annotations

import heapq
import json
from typing import Any


class EventGraph:
    """A directed acyclic graph of immutable events.

    Edges only point from a new event to events added earlier, so the
    structure can never contain a cycle. Inputs to :meth:`add` are copied,
    so later mutation of caller objects cannot pollute recorded history.
    """

    def __init__(self) -> None:
        self._at: dict[str, int] = {}
        self._parents: dict[str, tuple[str, ...]] = {}
        self._changes: dict[str, dict[str, int]] = {}

    @staticmethod
    def _require_nonempty_str(value: Any, name: str) -> None:
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a str, got {type(value).__name__}")
        if not value:
            raise ValueError(f"{name} must be a non-empty str")

    @staticmethod
    def _require_int(value: Any, name: str) -> int:
        # bool is a subclass of int, but must not be accepted as one.
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an int, got {type(value).__name__}")
        return value

    def add(
        self,
        id: str,
        at: int,
        parents: tuple[str, ...],
        changes: dict[str, int],
    ) -> None:
        """Add an event.

        Re-adding an existing ``id`` with identical ``at``, ``parents`` and
        ``changes`` is a no-op; any difference is a conflict.
        """
        # --- Full validation happens before any mutation. ---
        self._require_nonempty_str(id, "id")
        at_value = self._require_int(at, "at")
        if at_value < 0:
            raise ValueError("at must be a non-negative int")

        if not isinstance(parents, tuple):
            raise TypeError(f"parents must be a tuple, got {type(parents).__name__}")
        seen: set[str] = set()
        for parent in parents:
            self._require_nonempty_str(parent, "parent")
            if parent in seen:
                raise ValueError(f"duplicate parent {parent!r}")
            seen.add(parent)

        if not isinstance(changes, dict):
            raise TypeError(f"changes must be a dict, got {type(changes).__name__}")
        for key, value in changes.items():
            self._require_nonempty_str(key, "change key")
            self._require_int(value, f"change value for {key!r}")

        for parent in parents:
            if parent not in self._at:
                raise KeyError(parent)

        if id in self._at:
            if (
                self._at[id] != at_value
                or self._parents[id] != parents
                or self._changes[id] != changes
            ):
                raise ValueError(f"event {id!r} already exists with different inputs")
            return

        # --- Commit: store copies of caller-owned inputs. ---
        self._at[id] = at_value
        self._parents[id] = tuple(parents)
        self._changes[id] = dict(changes)

    def _ordered_ancestors(self, head: str) -> list[str]:
        """Validate ``head`` and return its ancestor closure in the unique
        parents-before-children order, ties among ready events broken by
        ``(at, id)`` ascending."""
        self._require_nonempty_str(head, "head")
        if head not in self._at:
            raise KeyError(head)

        closure: set[str] = {head}
        stack = [head]
        while stack:
            current = stack.pop()
            for parent in self._parents[current]:
                if parent not in closure:
                    closure.add(parent)
                    stack.append(parent)

        indegree = {event_id: 0 for event_id in closure}
        children: dict[str, list[str]] = {event_id: [] for event_id in closure}
        for event_id in closure:
            for parent in self._parents[event_id]:
                indegree[event_id] += 1
                children[parent].append(event_id)

        ready = [
            (self._at[event_id], event_id)
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
                    heapq.heappush(ready, (self._at[child], child))
        return order

    def replay(self, head: str) -> dict[str, int]:
        """Accumulate changes over the ancestor closure of ``head``.

        Keys are kept (even at a zero value) once introduced, and the
        returned dict is ordered by Unicode code point of its keys.
        """
        order = self._ordered_ancestors(head)
        state: dict[str, int] = {}
        for event_id in order:
            for key, delta in self._changes[event_id].items():
                state[key] = state.get(key, 0) + delta
        return {key: state[key] for key in sorted(state)}

    def replay_at(self, head: str, at: int) -> dict[str, int]:
        """Replay ``head``'s ancestor closure as of timestamp ``at``.

        Only events whose own ``at`` is at most the given ``at`` are
        accumulated, in the same parents-before-children, ``(at, id)``
        order as :meth:`replay`. Keys are kept (even at a zero value)
        once introduced, and the returned dict is ordered by Unicode
        code point of its keys. The query is read-only: the graph and
        its events are never modified.
        """
        self._require_nonempty_str(head, "head")
        at_value = self._require_int(at, "at")
        if at_value < 0:
            raise ValueError("at must be a non-negative int")

        order = self._ordered_ancestors(head)
        state: dict[str, int] = {}
        for event_id in order:
            if self._at[event_id] > at_value:
                continue
            for key, delta in self._changes[event_id].items():
                state[key] = state.get(key, 0) + delta
        return {key: state[key] for key in sorted(state)}

    def diff_at(self, head_a: str, head_b: str, at: int) -> dict[str, tuple[int, int]]:
        """Compare two heads' ancestor closures as of timestamp ``at``.

        Each head is replayed with :meth:`replay_at`'s rules (only events
        whose own ``at`` is at most the given ``at``, parents-before-
        children in ``(at, id)`` order, zero-valued keys kept). The result
        is a fresh dict over the union of both states' keys, ordered by
        Unicode code point, mapping each key to a ``(left, right)`` tuple
        where a side missing the key contributes ``0``. All parameters
        are validated (in signature order) before either head is looked
        up; the query is read-only and its result is detached from the
        graph's internal state.
        """
        self._require_nonempty_str(head_a, "head_a")
        self._require_nonempty_str(head_b, "head_b")
        at_value = self._require_int(at, "at")
        if at_value < 0:
            raise ValueError("at must be a non-negative int")

        left = self.replay_at(head_a, at_value)
        right = self.replay_at(head_b, at_value)
        return {
            key: (left.get(key, 0), right.get(key, 0))
            for key in sorted(left.keys() | right.keys())
        }

    def explain(self, head: str, key: str) -> tuple[str, ...]:
        """Return ids of events touching ``key``, in replay order.

        Events carrying a zero delta for the key are included.
        """
        self._require_nonempty_str(key, "key")
        order = self._ordered_ancestors(head)
        return tuple(
            event_id for event_id in order if key in self._changes[event_id]
        )

    def to_json(self) -> str:
        """Serialize the graph to deterministic compact JSON.

        The top level has a single ``events`` key; events are ordered by
        the Unicode code points of their ids, and each event's keys are
        emitted as ``id, at, parents, changes``. Parent ids keep their
        recorded order, and change keys are sorted by Unicode code point.
        The output uses no whitespace and never ends with a newline.
        """
        events = []
        for event_id in sorted(self._at):
            changes = self._changes[event_id]
            events.append(
                {
                    "id": event_id,
                    "at": self._at[event_id],
                    "parents": list(self._parents[event_id]),
                    "changes": {key: changes[key] for key in sorted(changes)},
                }
            )
        return json.dumps(
            {"events": events},
            separators=(",", ":"),
            ensure_ascii=False,
        )

    @classmethod
    def from_json(cls, payload: str) -> "EventGraph":
        """Rebuild a graph from :meth:`to_json` output.

        Every layer is strictly validated before the new graph is built.
        A non-``str`` payload raises :class:`TypeError`; unparseable JSON
        or any structural problem (missing or extra top-level or event
        keys, wrong types, empty ids, a negative ``at``, booleans where
        ints are required, duplicate event or parent ids, unknown
        parents, cycles, or invalid change keys and values) raises
        :class:`ValueError`. On failure no graph is constructed, and the
        result never shares mutable state with the input.
        """
        if not isinstance(payload, str):
            raise TypeError(
                f"payload must be a str, got {type(payload).__name__}"
            )

        def no_duplicate_keys(pairs: list[tuple[Any, Any]]) -> dict[Any, Any]:
            result: dict[Any, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate key {key!r} in JSON object")
                result[key] = value
            return result

        try:
            data = json.loads(payload, object_pairs_hook=no_duplicate_keys)
        except ValueError as exc:  # json.JSONDecodeError subclasses ValueError
            raise ValueError(f"payload is not valid JSON: {exc}") from exc

        def require_nonempty_str(value: Any, what: str) -> str:
            if not isinstance(value, str):
                raise ValueError(f"{what} must be a str, got {type(value).__name__}")
            if not value:
                raise ValueError(f"{what} must be a non-empty str")
            return value

        def require_int(value: Any, what: str) -> int:
            # bool is a subclass of int, but must not be accepted as one.
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{what} must be an int, got {type(value).__name__}")
            return value

        if not isinstance(data, dict):
            raise ValueError("top level must be an object")
        if set(data.keys()) != {"events"}:
            raise ValueError("top level must contain only an 'events' key")
        raw_events = data["events"]
        if not isinstance(raw_events, list):
            raise ValueError("'events' must be an array")

        parsed: list[tuple[str, int, tuple[str, ...], dict[str, int]]] = []
        known_ids: set[str] = set()
        for index, raw_event in enumerate(raw_events):
            where = f"events[{index}]"
            if not isinstance(raw_event, dict):
                raise ValueError(f"{where} must be an object")
            if set(raw_event.keys()) != {"id", "at", "parents", "changes"}:
                raise ValueError(
                    f"{where} must have exactly the keys "
                    "'id', 'at', 'parents', 'changes'"
                )

            event_id = require_nonempty_str(raw_event["id"], f"{where}.id")
            if event_id in known_ids:
                raise ValueError(f"duplicate event id {event_id!r}")
            known_ids.add(event_id)

            at_value = require_int(raw_event["at"], f"{where}.at")
            if at_value < 0:
                raise ValueError(f"{where}.at must be a non-negative int")

            raw_parents = raw_event["parents"]
            if not isinstance(raw_parents, list):
                raise ValueError(f"{where}.parents must be an array")
            parents: list[str] = []
            seen_parents: set[str] = set()
            for raw_parent in raw_parents:
                parent_id = require_nonempty_str(raw_parent, f"{where} parent")
                if parent_id in seen_parents:
                    raise ValueError(
                        f"duplicate parent {parent_id!r} in event {event_id!r}"
                    )
                seen_parents.add(parent_id)
                parents.append(parent_id)

            raw_changes = raw_event["changes"]
            if not isinstance(raw_changes, dict):
                raise ValueError(f"{where}.changes must be an object")
            changes: dict[str, int] = {}
            for raw_key, raw_value in raw_changes.items():
                key = require_nonempty_str(raw_key, f"{where} change key")
                changes[key] = require_int(raw_value, f"change value for {key!r}")

            parsed.append((event_id, at_value, tuple(parents), changes))

        # Every parent must reference an event present in the snapshot.
        parent_map: dict[str, tuple[str, ...]] = {}
        for event_id, _, parents, _ in parsed:
            for parent_id in parents:
                if parent_id not in known_ids:
                    raise ValueError(
                        f"event {event_id!r} references unknown parent "
                        f"{parent_id!r}"
                    )
            parent_map[event_id] = parents

        # Reject cycles (including self-parentage) with iterative DFS;
        # the live graph can never contain one, so a snapshot that does
        # cannot have come from to_json.
        state: dict[str, int] = {}  # 0 = on stack, 1 = fully explored
        for root in parent_map:
            if root in state:
                continue
            stack: list[tuple[str, int]] = [(root, 0)]
            state[root] = 0
            while stack:
                node, next_edge = stack[-1]
                node_parents = parent_map[node]
                if next_edge < len(node_parents):
                    stack[-1] = (node, next_edge + 1)
                    child = node_parents[next_edge]
                    child_state = state.get(child)
                    if child_state == 0:
                        raise ValueError(f"cycle detected through event {child!r}")
                    if child_state is None:
                        state[child] = 0
                        stack.append((child, 0))
                else:
                    state[node] = 1
                    stack.pop()

        # --- Commit: construct and populate only after full validation. ---
        graph = cls()
        for event_id, at_value, parents, changes in parsed:
            graph._at[event_id] = at_value
            graph._parents[event_id] = tuple(parents)
            graph._changes[event_id] = dict(changes)
        return graph
