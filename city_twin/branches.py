"""Named branches over a shared event graph."""

from __future__ import annotations

from typing import Any

from city_twin.event_graph import EventGraph


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

    def __init__(self, graph: EventGraph) -> None:
        if not isinstance(graph, EventGraph):
            raise TypeError(
                f"graph must be an EventGraph, got {type(graph).__name__}"
            )
        self._graph = graph
        self._heads: dict[str, str] = {}
        self._sources: dict[str, str] = {}
        # Appends record (at, changes); merges record (source, at, changes).
        self._appends: dict[str, dict[str, tuple[Any, ...]]] = {}

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

        if name in self._heads:
            if self._sources[name] != from_event:
                raise ValueError(
                    f"branch {name!r} already exists from "
                    f"{self._sources[name]!r}, not {from_event!r}"
                )
            return
        if from_event not in self._graph._at:
            raise KeyError(from_event)

        self._heads[name] = from_event
        self._sources[name] = from_event
        self._appends[name] = {}

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
        self._graph.add(id, at_value, (head,), changes)

        # --- Commit: only after the graph accepted the event. ---
        self._heads[name] = id
        self._appends[name][id] = (at_value, changes)

    def merge(
        self,
        target: str,
        source: str,
        id: str,
        at: int,
        changes: dict[str, int],
    ) -> None:
        """Merge ``source`` into ``target`` under a new two-parent event.

        The merge event's parents are ``(target head, source head)`` in
        that fixed order. On success only the target head moves; the
        source head is left untouched. Conflicting change keys among the
        events that are ancestors of exactly one of the two heads make
        the merge invalid. Repeating a merge with the same five inputs
        is a no-op; reusing the record's id or an id already in the
        graph with any different input is a conflict.
        """
        # --- Full validation happens before any state is consulted. ---
        self._require_nonempty_str(target, "target")
        self._require_nonempty_str(source, "source")
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

        self._require_known_branch(target)
        self._require_known_branch(source)
        if target == source:
            raise ValueError(f"cannot merge branch {target!r} into itself")

        recorded = self._appends[target].get(id)
        if recorded is not None:
            if recorded == (source, at_value, changes):
                return
            raise ValueError(
                f"event {id!r} already recorded on branch {target!r} "
                f"with different inputs"
            )

        def closure(head: str) -> set[str]:
            found = {head}
            stack = [head]
            while stack:
                current = stack.pop()
                for parent in self._graph._parents[current]:
                    if parent not in found:
                        found.add(parent)
                        stack.append(parent)
            return found

        target_head = self._heads[target]
        source_head = self._heads[source]
        t_closure = closure(target_head)
        s_closure = closure(source_head)
        common = t_closure & s_closure
        t_only = t_closure - common
        s_only = s_closure - common

        def change_keys(event_ids: set[str]) -> set[str]:
            keys: set[str] = set()
            for event_id in event_ids:
                keys.update(self._graph._changes[event_id])
            return keys

        conflicts = change_keys(t_only) & change_keys(s_only)
        if conflicts:
            raise ValueError(
                f"merge conflicts on change keys: "
                f"{', '.join(sorted(conflicts))}"
            )

        # graph.add enforces id idempotency/conflict for ids already in
        # the graph (e.g. appended by another branch) and parent validity.
        self._graph.add(id, at_value, (target_head, source_head), changes)

        # --- Commit: only after the graph accepted the event. ---
        self._heads[target] = id
        # The record is the five-tuple of inputs (target/id are the dict
        # keys); the heads captured by the merge are deliberately absent,
        # so the same inputs stay idempotent after either branch moves.
        self._appends[target][id] = (source, at_value, changes)

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
