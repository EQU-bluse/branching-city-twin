"""In-memory event DAG with deterministic replay."""

from __future__ import annotations

import heapq
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

    def explain(self, head: str, key: str) -> tuple[str, ...]:
        """Return ids of events touching ``key``, in replay order.

        Events carrying a zero delta for the key are included.
        """
        self._require_nonempty_str(key, "key")
        order = self._ordered_ancestors(head)
        return tuple(
            event_id for event_id in order if key in self._changes[event_id]
        )
