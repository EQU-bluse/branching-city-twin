"""In-memory event DAG for reconstructing city state.

Events form a directed acyclic graph: each event records the state changes
it applies and the parent events it builds on. ``replay`` folds the changes
of a head's ancestor closure into a state snapshot; ``explain`` traces which
events touched a given key.
"""

import heapq

__all__ = ["EventGraph"]


def _check_name(value, kind):
    """Validate a non-empty ``str`` identifier (event id, head, or key)."""
    if not isinstance(value, str):
        raise TypeError(f"{kind} must be a str, got {type(value).__name__}")
    if not value:
        raise ValueError(f"{kind} must be a non-empty str")
    return value


def _check_at(at):
    # bool is a subclass of int but is not accepted as a timestamp.
    if type(at) is not int:
        raise TypeError(f"at must be an int, got {type(at).__name__}")
    if at < 0:
        raise ValueError("at must be non-negative")
    return at


def _check_parents(parents):
    if not isinstance(parents, tuple):
        raise TypeError(f"parents must be a tuple, got {type(parents).__name__}")
    seen = set()
    for parent in parents:
        _check_name(parent, "parent id")
        if parent in seen:
            raise ValueError(f"duplicate parent id: {parent!r}")
        seen.add(parent)
    return parents


def _check_changes(changes):
    if not isinstance(changes, dict):
        raise TypeError(f"changes must be a dict, got {type(changes).__name__}")
    for key, value in changes.items():
        _check_name(key, "change key")
        if type(value) is not int:
            raise TypeError(
                f"change value for {key!r} must be an int, got {type(value).__name__}"
            )
    return changes


class EventGraph:
    """An in-memory DAG of events with deterministic replay."""

    def __init__(self) -> None:
        # id -> (at, parents, changes); insertion-ordered by add time.
        self._events = {}

    def add(self, id, at, parents, changes) -> None:
        """Append an event. Idempotent when the id already exists with
        identical inputs; any other reuse of an id is a conflict.

        The graph is left untouched if validation fails, and the inputs are
        copied so later mutation of the caller's objects cannot alter history.
        """
        _check_name(id, "event id")
        _check_at(at)
        _check_parents(parents)
        _check_changes(changes)

        parents = tuple(parents)
        changes = dict(changes)

        for parent in parents:
            if parent not in self._events:
                raise KeyError(f"unknown parent event: {parent!r}")

        existing = self._events.get(id)
        if existing is not None:
            if existing == (at, parents, changes):
                return
            raise ValueError(f"conflicting event id: {id!r}")

        self._events[id] = (at, parents, changes)

    def _order(self, head):
        """Ancestor closure of ``head`` in deterministic replay order:
        parents before children, ready events by (at, id) ascending."""
        _check_name(head, "head")
        if head not in self._events:
            raise KeyError(f"unknown head event: {head!r}")

        closure = set()
        stack = [head]
        while stack:
            event_id = stack.pop()
            if event_id in closure:
                continue
            closure.add(event_id)
            stack.extend(self._events[event_id][1])

        # Kahn's algorithm over the closure; every parent of a member is
        # itself a member, so in-degree is just the parent count.
        children = {event_id: [] for event_id in closure}
        remaining = {}
        for event_id in closure:
            parents = self._events[event_id][1]
            remaining[event_id] = len(parents)
            for parent in parents:
                children[parent].append(event_id)

        ready = [
            (self._events[event_id][0], event_id)
            for event_id, count in remaining.items()
            if count == 0
        ]
        heapq.heapify(ready)

        order = []
        while ready:
            _, event_id = heapq.heappop(ready)
            order.append(event_id)
            for child_id in children[event_id]:
                remaining[child_id] -= 1
                if remaining[child_id] == 0:
                    heapq.heappush(ready, (self._events[child_id][0], child_id))
        return order

    def replay(self, head) -> dict:
        """Fold the changes of ``head``'s ancestor closure, starting from an
        empty state. Zero-valued keys are kept; result keys are inserted in
        ascending Unicode code point order. Sibling branches are excluded."""
        state = {}
        for event_id in self._order(head):
            for key, delta in self._events[event_id][2].items():
                state[key] = state.get(key, 0) + delta
        return {key: state[key] for key in sorted(state)}

    def explain(self, head, key) -> tuple:
        """Ids of events in ``head``'s ancestor closure whose changes mention
        ``key`` (zero deltas included), in replay order."""
        _check_name(key, "key")
        return tuple(
            event_id
            for event_id in self._order(head)
            if key in self._events[event_id][2]
        )
