"""Named branches over an :class:`~city_twin.event_graph.EventGraph`."""

from __future__ import annotations

from typing import Any

from city_twin.event_graph import EventGraph


class BranchStore:
    """Named append-only branches sharing one event graph.

    Each branch points at a current head event; :meth:`append` extends a
    branch by adding an event whose sole parent is that head. Branches
    never move one another's heads. Failed calls leave the graph, branch
    heads and idempotency records untouched.
    """

    def __init__(self, graph: EventGraph) -> None:
        if not isinstance(graph, EventGraph):
            raise TypeError(
                f"graph must be an EventGraph, got {type(graph).__name__}"
            )
        self._graph = graph
        self._heads: dict[str, str] = {}
        # Last successful append per branch: (id, at, changes).
        self._last_append: dict[str, tuple[str, int, dict[str, int]]] = {}

    @staticmethod
    def _require_nonempty_str(value: Any, name: str) -> None:
        EventGraph._require_nonempty_str(value, name)

    @classmethod
    def _validate_changes(cls, changes: Any) -> None:
        if not isinstance(changes, dict):
            raise TypeError(
                f"changes must be a dict, got {type(changes).__name__}"
            )
        for key, value in changes.items():
            cls._require_nonempty_str(key, "change key")
            EventGraph._require_int(value, f"change value for {key!r}")

    def create(self, name: str, from_event: str) -> None:
        """Create a branch rooted at an existing event.

        Re-creating an existing branch from the same event is a no-op;
        re-creating it from a different event is a conflict.
        """
        self._require_nonempty_str(name, "name")
        self._require_nonempty_str(from_event, "from_event")

        existing = self._heads.get(name)
        if existing is not None:
            if existing != from_event:
                raise ValueError(
                    f"branch {name!r} already exists at {existing!r}, "
                    f"cannot re-create at {from_event!r}"
                )
            return
        if from_event not in self._graph._at:
            raise KeyError(from_event)
        self._heads[name] = from_event

    def append(
        self,
        name: str,
        id: str,
        at: int,
        changes: dict[str, int],
    ) -> None:
        """Append an event to a branch using its current head as parent.

        Retrying the exact same ``(name, id, at, changes)`` call is a
        no-op; reusing an existing ``id`` with different inputs raises.
        """
        self._require_nonempty_str(name, "name")
        self._require_nonempty_str(id, "id")
        at_value = EventGraph._require_int(at, "at")
        if at_value < 0:
            raise ValueError("at must be a non-negative int")
        self._validate_changes(changes)

        head = self._heads.get(name)
        if head is None:
            raise KeyError(name)

        record = self._last_append.get(name)
        if record is not None and record[0] == id:
            if record[1] == at_value and record[2] == changes:
                return
            raise ValueError(
                f"event {id!r} already appended to branch {name!r} "
                "with different inputs"
            )

        self._graph.add(id, at_value, (head,), changes)

        # Commit only after the graph accepted the event.
        self._heads[name] = id
        self._last_append[name] = (id, at_value, dict(changes))

    def head(self, name: str) -> str:
        """Return the current head event id of a branch."""
        self._require_nonempty_str(name, "name")
        if name not in self._heads:
            raise KeyError(name)
        return self._heads[name]

    def replay(self, name: str) -> dict[str, int]:
        """Replay the branch at its current head."""
        self._require_nonempty_str(name, "name")
        if name not in self._heads:
            raise KeyError(name)
        return self._graph.replay(self._heads[name])
