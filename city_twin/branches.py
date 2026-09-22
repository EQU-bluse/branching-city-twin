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

        def replayed_value(order: list[str]) -> int:
            value = 0
            for event_id in order:
                changes = self._graph._changes[event_id]
                if key in changes:
                    value += changes[key]
            return value

        left_value = replayed_value(left_order)
        right_value = replayed_value(right_order)

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
