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

    def to_json(self) -> str:
        """Serialize the graph to a compact, deterministic JSON string.

        The top-level object contains only the ``events`` key. Events are
        ordered by the Unicode code point of their id; each event's keys
        are ordered ``id, at, parents, changes``, parent ids keep their
        recorded order, and change keys are sorted by Unicode code point.
        The output uses ``separators=(',', ':')``, ``ensure_ascii=False``
        and no trailing newline.
        """
        events: list[dict[str, object]] = []
        for event_id in sorted(self._at):
            events.append(
                {
                    "id": event_id,
                    "at": self._at[event_id],
                    "parents": list(self._parents[event_id]),
                    "changes": {
                        key: self._changes[event_id][key]
                        for key in sorted(self._changes[event_id])
                    },
                }
            )
        return json.dumps(
            {"events": events},
            separators=(",", ":"),
            ensure_ascii=False,
        )

    @classmethod
    def from_json(cls, payload: Any) -> "EventGraph":
        """Deserialize a string produced by :meth:`to_json`.

        Validation is strict and layered: a non-``str`` payload raises
        :class:`TypeError`; unparseable JSON, a wrong top-level shape,
        missing or extra fields, type errors, empty strings, negative
        timestamps, booleans posing as integers, duplicate event or
        parent ids, unknown parents, cycles and invalid change entries
        raise :class:`ValueError`. All validation finishes before the
        new graph is built, so a failure never leaves a partially
        populated object, and the result shares no mutable state with
        either the input string or the parsed JSON values.
        """
        if not isinstance(payload, str):
            raise TypeError(
                f"payload must be a str, got {type(payload).__name__}"
            )
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError(f"payload is not valid JSON: {exc}") from exc

        if not isinstance(data, dict):
            raise ValueError("top-level JSON value must be an object")
        if set(data.keys()) != {"events"}:
            raise ValueError(
                "top-level object must contain exactly the 'events' key"
            )
        raw_events = data["events"]
        if not isinstance(raw_events, list):
            raise ValueError("'events' must be an array")

        # Parse into fully local structures first; the EventGraph is only
        # constructed below, after every check has passed.
        at_values: dict[str, int] = {}
        parent_map: dict[str, tuple[str, ...]] = {}
        change_map: dict[str, dict[str, int]] = {}

        for index, event in enumerate(raw_events):
            if not isinstance(event, dict):
                raise ValueError(f"event at index {index} must be an object")
            if set(event.keys()) != {"id", "at", "parents", "changes"}:
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
            # bool is a subclass of int, but must not be accepted as one.
            if isinstance(at_value, bool) or not isinstance(at_value, int):
                raise ValueError(f"event {event_id!r}: at must be an int")
            if at_value < 0:
                raise ValueError(f"event {event_id!r}: at must be non-negative")

            raw_parents = event["parents"]
            if not isinstance(raw_parents, list):
                raise ValueError(
                    f"event {event_id!r}: parents must be an array"
                )
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

            raw_changes = event["changes"]
            if not isinstance(raw_changes, dict):
                raise ValueError(
                    f"event {event_id!r}: changes must be an object"
                )
            event_changes: dict[str, int] = {}
            for key, value in raw_changes.items():
                if not isinstance(key, str) or not key:
                    raise ValueError(
                        f"event {event_id!r}: every change key must be a "
                        "non-empty str"
                    )
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError(
                        f"event {event_id!r}: change value for {key!r} "
                        "must be an int"
                    )
                event_changes[key] = value

            at_values[event_id] = at_value
            parent_map[event_id] = tuple(event_parents)
            change_map[event_id] = event_changes

        for event_id, event_parents in parent_map.items():
            for parent in event_parents:
                if parent not in at_values:
                    raise ValueError(
                        f"event {event_id!r}: unknown parent {parent!r}"
                    )

        # Cycle detection: iterative DFS following child -> parent edges.
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

        graph = cls()
        for event_id in at_values:
            graph._at[event_id] = at_values[event_id]
            graph._parents[event_id] = parent_map[event_id]
            graph._changes[event_id] = dict(sorted(change_map[event_id].items()))
        return graph

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

    def _impact_descendants(
        self, head: str, start: str
    ) -> list[tuple[str, int, tuple[str, ...], tuple[str, ...]]]:
        """Return ``start``'s strict descendants inside ``head``'s closure.

        Edges are read parent-to-child, so only events causally *after*
        ``start`` are considered; ``start`` itself is excluded. Each entry
        is ``(id, at, path, keys)``: the descendant id, its non-negative
        timestamp, the id tuple of the chosen path from ``start`` to the
        descendant with both ends included, and the tuple of change keys
        the descendant touches, ordered by Unicode code point (an empty
        tuple for events with no changes).

        When several paths reach a descendant, the one with the fewest
        edges wins; same-length paths are broken by the lexicographically
        smallest full id tuple (compared by Unicode code point). Entries
        follow the unique parents-before-children, ``(at, id)`` order of
        :meth:`_ordered_ancestors`. Unknown ``start`` ids and ids outside
        ``head``'s ancestor closure raise :class:`KeyError`. The query is
        read-only and every returned tuple is freshly built.
        """
        order = self._ordered_ancestors(head)
        closure = set(order)
        if start not in self._at or start not in closure:
            raise KeyError(start)

        children: dict[str, list[str]] = {event_id: [] for event_id in closure}
        for event_id in closure:
            for parent in self._parents[event_id]:
                children[parent].append(event_id)

        # Breadth-first expansion, one edge-count level at a time, keeps
        # the first path found to each node shortest; within a level the
        # lexicographically smallest parent path wins (the child id is a
        # common suffix, so parent-path order decides the full tuple).
        best: dict[str, tuple[str, ...]] = {start: (start,)}
        frontier = [start]
        while frontier:
            candidates: dict[str, list[tuple[str, ...]]] = {}
            for event_id in frontier:
                for child in children[event_id]:
                    if child not in best:
                        candidates.setdefault(child, []).append(best[event_id])
            next_frontier: list[str] = []
            for child, parent_paths in candidates.items():
                best[child] = min(parent_paths) + (child,)
                next_frontier.append(child)
            frontier = next_frontier

        records: list[
            tuple[str, int, tuple[str, ...], tuple[str, ...]]
        ] = []
        for event_id in order:
            path = best.get(event_id)
            if path is None or event_id == start:
                continue
            records.append(
                (
                    event_id,
                    self._at[event_id],
                    path,
                    tuple(sorted(self._changes[event_id])),
                )
            )
        return records
