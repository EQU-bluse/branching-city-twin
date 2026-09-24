"""Named branches over a shared event graph."""

from __future__ import annotations

import heapq
import itertools
import math
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
        self._appends: dict[str, dict[str, tuple[int, dict[str, int]]]] = {}
        self._merges: dict[str, tuple[str, str, int, dict[str, int]]] = {}

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

        self._graph.add(id, at_value, (target_head, source_head), changes)

        # --- Commit: only after the graph accepted the event. ---
        self._heads[target] = id
        self._merges[id] = (target, source, at_value, changes)

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

        # --- Slice inputs validated in signature order, before any state is
        # queried: branch lookups, historical-node alignment and the
        # candidate cap all run afterwards. ---
        if not isinstance(causes, tuple):
            raise TypeError(
                f"causes must be a tuple, got {type(causes).__name__}"
            )
        seen_causes: set[str] = set()
        for cause in causes:
            self._require_nonempty_str(cause, "cause")
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
        depth_value = self._require_size_bound(depth, "depth")
        if depth_value < 0:
            raise ValueError("depth must be non-negative")
        node_limit_value = self._require_size_bound(node_limit, "node_limit")
        if node_limit_value < 1:
            raise ValueError("node_limit must be >= 1")

        # Only now is state queried: the existing branch/node lookup order,
        # checkpoint alignment and the candidate cap run on the one frozen
        # view every result is taken from.
        prepared = self._capture_frontier_view(validated)
        return self._slice_cascade(
            prepared,
            causes,
            direction,
            depth_value,
            node_limit_value,
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
        """Diff two read-only frozen cascade slices at historical checkpoints.

        The read-only two-checkpoint companion of :meth:`cascade_slice`: it is
        called with every one of that query's ordinary arguments -- none has a
        default -- then two checkpoint indices ``before`` and ``after`` placed
        immediately before the existing cause set, and finally that query's
        ``causes``, ``direction``, ``depth`` and ``node_limit``. The ordinary
        inputs are validated first in exactly :meth:`cascade_slice`'s order;
        ``before`` and ``after`` are then validated before the existing slice
        range inputs, all before any state is queried. Each index must be a
        non-``bool`` :class:`int` (else :class:`TypeError`); a negative
        index, an out-of-range index, or ``before`` greater than ``after``
        raises :class:`ValueError`. The branch and historical-node lookup
        order, checkpoint alignment, the candidate-count cap and the slice
        node cap keep exactly :meth:`cascade_slice`'s
        :class:`KeyError`/:class:`ValueError` contracts and lookup order; any
        cap breach raises before any result is returned, so no partial diff is
        ever produced.

        The two indices select common, aligned historical positions: each side
        uses only the prefix of the checkpoint series from the first point up
        to and including that index to build its frozen slice, and both
        prefixes are taken from one read-only historical view captured once.
        Each side's slice reuses the existing cause, direction, depth and
        node-cap semantics exactly. A cause event absent from a frozen slice
        raises :class:`KeyError`: the causes are walked in their input order,
        the ``before`` side checked before the ``after`` side. An empty
        ``causes`` tuple still completes every input and state check and
        returns two empty slices and an empty change set.

        The result is a fresh dict whose keys are ordered
        ``before, after, changes``: the ``before`` and ``after`` entries are
        each item-wise equal to the existing slice result on its prefix (the
        ``nodes, edges, gaps`` structure). ``changes`` groups records by
        category -- nodes, then edges, then gaps -- and within each category
        lists removals, additions and content changes in that order. Each
        record is a fresh dict whose keys are ordered ``kind, identity,
        before, after``; ``kind`` is the category joined by an underscore
        to ``removed``, ``added`` or ``changed`` (``node_removed``,
        ``edge_added``, ``gap_changed`` ...). A node is identified by the
        pairing of its cause event id and state key; an edge by the node
        identities (cause/key pairs) of its two ends, never by the
        endpoint positions inside a slice -- positions are remapped back to node
        identities before comparison; a gap by its interval and its member
        combination. The missing side of a removal or addition is ``None``;
        the present side and both sides of a change are fully isolated
        copies. Every returned level is freshly built and detached from
        internal state and shares no objects across levels. The query is
        read-only: success or failure never modifies the event graph, branch
        heads, audit or idempotency records, or any existing query.
        """
        # Ordinary inputs are validated first, in exactly the existing order,
        # but no branch or historical node is looked up yet.
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

        # --- The two checkpoint indices are checked next, before the existing
        # slice range parameters and before any state is queried. ---
        before_value = self._require_size_bound(before, "before")
        if before_value < 0:
            raise ValueError("before must be non-negative")
        after_value = self._require_size_bound(after, "after")
        if after_value < 0:
            raise ValueError("after must be non-negative")
        if before_value > after_value:
            raise ValueError("before must be <= after")

        # --- Existing slice inputs, exactly cascade_slice's order. ---
        if not isinstance(causes, tuple):
            raise TypeError(
                f"causes must be a tuple, got {type(causes).__name__}"
            )
        seen_causes: set[str] = set()
        for cause in causes:
            self._require_nonempty_str(cause, "cause")
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
        depth_value = self._require_size_bound(depth, "depth")
        if depth_value < 0:
            raise ValueError("depth must be non-negative")
        node_limit_value = self._require_size_bound(node_limit, "node_limit")
        if node_limit_value < 1:
            raise ValueError("node_limit must be >= 1")

        # The shared frozen view decides checkpoint membership, alignment and the
        # candidate cap exactly as in cascade_slice; its checkpoint count sets
        # the two indices' bounds.
        prepared = self._capture_frontier_view(validated)
        checkpoint_count = self._prepared_checkpoint_count(prepared)
        if before_value >= checkpoint_count:
            raise ValueError(
                f"before checkpoint {before_value} out of range for "
                f"{checkpoint_count} checkpoints"
            )
        if after_value >= checkpoint_count:
            raise ValueError(
                f"after checkpoint {after_value} out of range for "
                f"{checkpoint_count} checkpoints"
            )

        before_cascade = self._build_decision_cascade(
            self._prefix_frontier_view(prepared, before_value)
        )
        after_cascade = self._build_decision_cascade(
            self._prefix_frontier_view(prepared, after_value)
        )

        if not causes:
            # All validation and the state lookup are done; an empty set
            # never trips either node cap.
            return {
                "before": {"nodes": (), "edges": (), "gaps": ()},
                "after": {"nodes": (), "edges": (), "gaps": ()},
                "changes": (),
            }

        # Cause presence is checked in cause-input order, the before side
        # before the after side for each cause, before any expansion runs.
        before_nodes = before_cascade["nodes"]
        after_nodes = after_cascade["nodes"]
        before_by_cause = self._nodes_by_cause(before_nodes)
        after_by_cause = self._nodes_by_cause(after_nodes)
        before_starts: list[int] = []
        after_starts: list[int] = []
        for cause in causes:
            before_indices = before_by_cause.get(cause)
            if before_indices is None:
                raise KeyError(cause)
            after_indices = after_by_cause.get(cause)
            if after_indices is None:
                raise KeyError(cause)
            before_starts.extend(before_indices)
            after_starts.extend(after_indices)

        # Expand both reachable sets before either cap is enforced or any
        # result object is built, so an over-cap side returns nothing.
        before_positions = self._slice_positions(
            before_cascade,
            before_starts,
            direction,
            depth_value,
            node_limit_value,
        )
        after_positions = self._slice_positions(
            after_cascade,
            after_starts,
            direction,
            depth_value,
            node_limit_value,
        )

        before_slice = self._build_slice(before_cascade, before_positions)
        after_slice = self._build_slice(after_cascade, after_positions)
        changes = self._diff_cascade_slices(before_slice, after_slice)
        return {
            "before": before_slice,
            "after": after_slice,
            "changes": changes,
        }

    @staticmethod
    def _prepared_checkpoint_count(prepared: dict[str, object]) -> int:
        """Return the common aligned checkpoint count of a captured view."""
        views_by_name = prepared["views_by_name"]
        if not views_by_name:
            # Pool alignment guarantees one checkpoint count; with no series
            # every series shares the empty points tuple.
            return 0
        first_views = next(iter(views_by_name.values()))
        return len(first_views)

    @staticmethod
    def _prefix_frontier_view(
        prepared: dict[str, object], checkpoint: int
    ) -> dict[str, object]:
        """Return a prepared view restricted to its first ``checkpoint+1`` points.

        The one frozen read-only historical view is shared: only the captured
        checkpoint entries are trimmed to the prefix ending at ``checkpoint``;
        the closures, nodes, enumerated candidates and scenario figures are the
        same objects. Building a cascade on the returned view is therefore
        exactly the existing slice over the prefix inputs -- the search reads
        each combination's checkpoint views only, and the candidate set never
        depends on checkpoint count.
        """
        prefix_views: dict[
            str, list[tuple[list[str], list[str], str, str]]
        ] = {
            name: views[: checkpoint + 1]
            for name, views in prepared["views_by_name"].items()
        }
        prefix_prepared = dict(prepared)
        prefix_prepared["views_by_name"] = prefix_views
        return prefix_prepared

    def _slice_cascade(
        self,
        prepared: dict[str, object],
        causes: tuple[str, ...],
        direction: str,
        depth_value: int,
        node_limit_value: int,
    ) -> dict[str, object]:
        """Build the existing frozen slice from an already validated view.

        Shared by :meth:`cascade_slice` and :meth:`cascade_slice_diff` so
        both apply the same cause-presence, direction/depth expansion and
        node-cap semantics. All inputs and the view are already validated;
        the returned dict and every level are freshly built copies of the
        cascade's existing nodes, edges and gaps.
        """
        cascade = self._build_decision_cascade(prepared)

        if not causes:
            # All validation and the state lookup are done; an empty set
            # never trips the node cap.
            return {"nodes": (), "edges": (), "gaps": ()}

        starts: list[int] = []
        by_cause = self._nodes_by_cause(cascade["nodes"])
        for cause in causes:
            indices = by_cause.get(cause)
            if indices is None:
                raise KeyError(cause)
            starts.extend(indices)

        positions = self._slice_positions(
            cascade,
            starts,
            direction,
            depth_value,
            node_limit_value,
        )
        return self._build_slice(cascade, positions)

    @staticmethod
    def _nodes_by_cause(
        nodes: tuple[dict[str, object], ...]
    ) -> dict[str, list[int]]:
        """Map each cause event id to all of its node positions, in order."""
        by_cause: dict[str, list[int]] = {}
        for index, node in enumerate(nodes):
            by_cause.setdefault(node["cause"], []).append(index)
        return by_cause

    def _slice_positions(
        self,
        cascade: dict[str, object],
        starts: list[int],
        direction: str,
        depth_value: int,
        node_limit_value: int,
    ) -> list[int]:
        """Return the sorted selected node positions after the bounded walk.

        Adjacency comes only from the cascade's existing directed edges, so
        the walk cannot leave the existing evidence. The cap is checked after
        the full reachable set is known; an over-cap call raises
        :class:`ValueError` before any slice object is built.
        """
        nodes = cascade["nodes"]
        edges = cascade["edges"]

        # Adjacency comes only from the cascade's existing directed edges,
        # so the walk cannot leave the existing evidence.
        forward_neighbors: dict[int, list[int]] = {
            index: [] for index in range(len(nodes))
        }
        backward_neighbors: dict[int, list[int]] = {
            index: [] for index in range(len(nodes))
        }
        for edge in edges:
            forward_neighbors[edge["source"]].append(edge["target"])
            backward_neighbors[edge["target"]].append(edge["source"])

        # Level-by-level expansion from every start at once: levels count
        # edges, a node reached more than once is claimed once, and depth
        # zero keeps only the starts.
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

        # The cap is checked after the full reachable set is known, before
        # any slice object is built, so an over-cap call returns nothing.
        if len(selected) > node_limit_value:
            raise ValueError(
                f"cascade slice node limit exceeded: {len(selected)} nodes "
                f"selected, limit is {node_limit_value}"
            )
        return sorted(selected)

    def _build_slice(
        self, cascade: dict[str, object], selected_positions: list[int]
    ) -> dict[str, object]:
        """Project a cascade onto selected node positions, remapping edges.

        Nodes keep the full cascade's order; edge endpoints remap densely;
        edges keep only records whose two ends are both selected, paths and
        the existing edge order unchanged; gaps keep only records whose
        interval is also an interval of a selected node. Every level is a
        fresh isolated copy.
        """
        nodes = cascade["nodes"]
        edges = cascade["edges"]
        gaps = cascade["gaps"]
        remapped = {
            old: new for new, old in enumerate(selected_positions)
        }
        slice_nodes = tuple(
            self._copy_slice_node(nodes[old]) for old in selected_positions
        )
        slice_edges = tuple(
            {
                "source": remapped[edge["source"]],
                "target": remapped[edge["target"]],
                "path": tuple(edge["path"]),
            }
            for edge in edges
            if edge["source"] in remapped and edge["target"] in remapped
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
    def _copy_slice_node(node: dict[str, object]) -> dict[str, object]:
        """Fresh isolated copy of one slice node record."""
        return {
            "cause": node["cause"],
            "key": node["key"],
            "intervals": tuple(node["intervals"]),
            "checkpoints": tuple(node["checkpoints"]),
            "branches": tuple(node["branches"]),
            "affected": tuple(node["affected"]),
        }

    @staticmethod
    def _copy_slice_edge(edge: dict[str, object]) -> dict[str, object]:
        """Fresh isolated copy of one slice edge record (positions kept)."""
        return {
            "source": edge["source"],
            "target": edge["target"],
            "path": tuple(edge["path"]),
        }

    @staticmethod
    def _copy_slice_gap(gap: dict[str, object]) -> dict[str, object]:
        """Fresh isolated copy of one slice gap record."""
        return {
            "interval": gap["interval"],
            "members": tuple(gap["members"]),
        }

    def _diff_cascade_slices(
        self,
        before: dict[str, object],
        after: dict[str, object],
    ) -> tuple[dict[str, object], ...]:
        """Classify the changes between two frozen slices.

        Nodes are keyed by the (cause event, state key) pairing, edges by
        the node identities of their two ends (positions are first mapped
        back to those identities, never compared by slice position), and
        gaps by their (interval, members) combination. Within each category
        removals, additions and content changes are emitted in that order.
        Every record and side value is a freshly isolated copy.
        """
        before_nodes = before["nodes"]
        after_nodes = after["nodes"]
        node_identity = lambda node: (node["cause"], node["key"])
        before_node_map = {
            node_identity(node): node for node in before_nodes
        }
        after_node_map = {
            node_identity(node): node for node in after_nodes
        }
        before_node_ids = set(before_node_map)
        after_node_ids = set(after_node_map)

        def node_change(
            kind: str,
            identity: tuple[str, str],
            before_node: object,
            after_node: object,
        ) -> dict[str, object]:
            return {
                "kind": kind,
                "identity": identity,
                "before": (
                    None
                    if before_node is None
                    else self._copy_slice_node(before_node)
                ),
                "after": (
                    None
                    if after_node is None
                    else self._copy_slice_node(after_node)
                ),
            }

        node_removed = [
            node_change(
                "node_removed",
                identity,
                before_node_map[identity],
                None,
            )
            for identity in sorted(before_node_ids - after_node_ids)
        ]
        node_added = [
            node_change(
                "node_added",
                identity,
                None,
                after_node_map[identity],
            )
            for identity in sorted(after_node_ids - before_node_ids)
        ]
        node_changed = [
            node_change(
                "node_changed",
                identity,
                before_node_map[identity],
                after_node_map[identity],
            )
            for identity in sorted(before_node_ids & after_node_ids)
            if before_node_map[identity] != after_node_map[identity]
        ]

        # Edges are compared by endpoint node identities only, never by
        # slice position: positions are remapped back to (cause, key)
        # node identities first. The path stays content, so the same
        # endpoints with a different path (or different remapped
        # positions) is an edge change, not a different edge.
        def edge_identity(
            edge: dict[str, object],
            slice_nodes: tuple[dict[str, object], ...],
        ) -> tuple[tuple[str, str], tuple[str, str]]:
            # Endpoint positions are first restored to node identities
            # (cause event, state key) before any comparison.
            source = node_identity(slice_nodes[edge["source"]])
            target = node_identity(slice_nodes[edge["target"]])
            return (source, target)

        before_edges: dict[
            tuple[object, ...], dict[str, object]
        ] = {}
        for edge in before["edges"]:
            before_edges[edge_identity(edge, before_nodes)] = edge
        after_edges: dict[
            tuple[object, ...], dict[str, object]
        ] = {}
        for edge in after["edges"]:
            after_edges[edge_identity(edge, after_nodes)] = edge
        before_edge_ids = set(before_edges)
        after_edge_ids = set(after_edges)

        def edge_record(
            kind: str,
            identity: tuple[object, ...],
            edge_before: object,
            edge_after: object,
        ) -> dict[str, object]:
            # The identity is only the two endpoint node identities
            # (source first), independent of either slice's positions.
            return {
                "kind": kind,
                "identity": identity,
                "before": (
                    None
                    if edge_before is None
                    else self._copy_slice_edge(edge_before)
                ),
                "after": (
                    None
                    if edge_after is None
                    else self._copy_slice_edge(edge_after)
                ),
            }

        edge_removed = [
            edge_record(
                "edge_removed",
                identity,
                before_edges[identity],
                None,
            )
            for identity in sorted(before_edge_ids - after_edge_ids)
        ]
        edge_added = [
            edge_record(
                "edge_added",
                identity,
                None,
                after_edges[identity],
            )
            for identity in sorted(after_edge_ids - before_edge_ids)
        ]
        # Common endpoint/path identities always carry equal content (the path
        # is part of the identity); a kept edge still changes record when
        # its endpoints' slice positions differ.
        edge_changed = [
            edge_record(
                "edge_changed",
                identity,
                before_edges[identity],
                after_edges[identity],
            )
            for identity in sorted(before_edge_ids & after_edge_ids)
            if before_edges[identity] != after_edges[identity]
        ]

        # Gaps are identified by interval and member combination only.
        def gap_identity(
            gap: dict[str, object]
        ) -> tuple[int, tuple[str, ...]]:
            return (gap["interval"], tuple(gap["members"]))

        before_gap_map = {
            gap_identity(gap): gap for gap in before["gaps"]
        }
        after_gap_map = {
            gap_identity(gap): gap for gap in after["gaps"]
        }
        before_gap_ids = set(before_gap_map)
        after_gap_ids = set(after_gap_map)

        def gap_record(
            kind: str,
            identity: tuple[int, tuple[str, ...]],
            gap_before: object,
            gap_after: object,
        ) -> dict[str, object]:
            return {
                "kind": kind,
                "identity": identity,
                "before": (
                    None
                    if gap_before is None
                    else self._copy_slice_gap(gap_before)
                ),
                "after": (
                    None
                    if gap_after is None
                    else self._copy_slice_gap(gap_after)
                ),
            }

        gap_removed = [
            gap_record(
                "gap_removed",
                identity,
                before_gap_map[identity],
                None,
            )
            for identity in sorted(before_gap_ids - after_gap_ids)
        ]
        gap_added = [
            gap_record(
                "gap_added",
                identity,
                None,
                after_gap_map[identity],
            )
            for identity in sorted(after_gap_ids - before_gap_ids)
        ]
        gap_changed = [
            gap_record(
                "gap_changed",
                identity,
                before_gap_map[identity],
                after_gap_map[identity],
            )
            for identity in sorted(before_gap_ids & after_gap_ids)
            if before_gap_map[identity] != after_gap_map[identity]
        ]

        return tuple(
            node_removed
            + node_added
            + node_changed
            + edge_removed
            + edge_added
            + edge_changed
            + gap_removed
            + gap_added
            + gap_changed
        )

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
