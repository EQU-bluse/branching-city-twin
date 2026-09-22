import json
import unittest

from city_twin.branches import BranchStore
from city_twin.event_graph import EventGraph


def make_graph() -> EventGraph:
    graph = EventGraph()
    graph.add("root", 0, (), {"a": 1, "中": 3})
    graph.add("e1", 2, ("root",), {"b": 2})
    graph.add("e2", 2, ("root",), {"a": 3})
    graph.add("e3", 5, ("e1", "e2"), {"a": -1, "b": -2})
    graph.add("e4", 9, ("e3",), {"c": 7})
    return graph


class ToJsonTests(unittest.TestCase):
    def test_compact_deterministic_shape(self) -> None:
        payload = make_graph().to_json()
        self.assertNotIn("\n", payload)
        self.assertNotIn(", ", payload)
        self.assertNotIn(": ", payload)
        decoded = json.loads(payload)
        self.assertEqual(list(decoded.keys()), ["events"])
        self.assertEqual([e["id"] for e in decoded["events"]],
                         ["e1", "e2", "e3", "e4", "root"])
        for event in decoded["events"]:
            self.assertEqual(
                list(event.keys()), ["id", "at", "parents", "changes"]
            )
            self.assertEqual(
                list(event["changes"].keys()), sorted(event["changes"].keys())
            )

    def test_repeated_output_is_identical(self) -> None:
        graph = make_graph()
        self.assertEqual(graph.to_json(), graph.to_json())

    def test_empty_graph(self) -> None:
        self.assertEqual(EventGraph().to_json(), '{"events":[]}')

    def test_unicode_and_negative_values(self) -> None:
        graph = EventGraph()
        graph.add("z", 1, (), {"x": -5, "é": 1, "中": 2, "a": 0})
        event = json.loads(graph.to_json())["events"][0]
        self.assertEqual(list(event["changes"]), ["a", "x", "é", "中"])


class FromJsonRoundTripTests(unittest.TestCase):
    def test_round_trip_preserves_events_parents_and_changes(self) -> None:
        graph = make_graph()
        restored = EventGraph.from_json(graph.to_json())
        for event_id in ("root", "e1", "e2", "e3", "e4"):
            self.assertEqual(restored._at[event_id], graph._at[event_id])
            self.assertEqual(restored._parents[event_id], graph._parents[event_id])
            self.assertEqual(restored._changes[event_id], graph._changes[event_id])
        self.assertEqual(restored._parents["e3"], ("e1", "e2"))

    def test_round_trip_preserves_replay_queries(self) -> None:
        graph = make_graph()
        restored = EventGraph.from_json(graph.to_json())
        for head in ("root", "e1", "e2", "e3", "e4"):
            self.assertEqual(restored.replay(head), graph.replay(head))
            for at in range(11):
                self.assertEqual(
                    restored.replay_at(head, at), graph.replay_at(head, at)
                )
                self.assertEqual(
                    restored.explain(head, "a"), graph.explain(head, "a")
                )
        self.assertEqual(
            restored.diff_at("e1", "e2", 4), graph.diff_at("e1", "e2", 4)
        )

    def test_parents_may_appear_later_in_the_array(self) -> None:
        # Events are sorted by id in the payload, so parents need not
        # precede their children in array order.
        graph = EventGraph()
        graph.add("b", 0, (), {})
        graph.add("a", 0, (), {})
        graph.add("aa", 0, ("b",), {})
        restored = EventGraph.from_json(graph.to_json())
        self.assertEqual(restored._parents["aa"], ("b",))
        self.assertEqual(restored.replay("aa"), {})

    def test_branch_store_keeps_working_on_restored_graph(self) -> None:
        graph = EventGraph.from_json(make_graph().to_json())
        store = BranchStore(graph)
        store.create("main", "e4")
        store.append("main", "m1", 10, {"d": 4})
        self.assertEqual(store.head("main"), "m1")
        self.assertEqual(store.replay_at("main", 10)["d"], 4)
        other = BranchStore(graph)
        other.create("feat", "e2")
        self.assertEqual(other.trace("feat", "a"), graph.explain("e2", "a"))

    def test_does_not_share_mutable_state_with_input(self) -> None:
        payload = make_graph().to_json()
        decoded = json.loads(payload)
        restored = EventGraph.from_json(
            json.dumps(decoded, separators=(",", ":"), ensure_ascii=False)
        )
        decoded["events"][0]["changes"]["b"] = 999
        self.assertEqual(restored._changes["e1"]["b"], 2)
        restored._at["root"] = 42
        self.assertEqual(make_graph()._at["root"], 0)


class FromJsonValidationTests(unittest.TestCase):
    def event(self, **overrides: object) -> str:
        event = {"id": "e", "at": 1, "parents": [], "changes": {}}
        event.update(overrides)
        return json.dumps({"events": [event]}, separators=(",", ":"))

    def assertRejected(self, payload: object) -> None:
        with self.assertRaises(ValueError):
            EventGraph.from_json(payload)  # type: ignore[arg-type]

    def test_non_str_payload_is_type_error(self) -> None:
        for bad in (None, b'{"events":[]}', 1, [], {}):
            with self.assertRaises(TypeError):
                EventGraph.from_json(bad)  # type: ignore[arg-type]

    def test_unparseable_json(self) -> None:
        for bad in ("", "{", "[", "nul", "true", "42", '"x"',
                    "{'events': []}", '{"events":}'):
            self.assertRejected(bad)

    def test_top_level_shape(self) -> None:
        for bad in ("{}", '{"events":[],"x":1}', '{"x":[]}',
                    '{"events":{}}', "[]", '{"events":[{}]}'):
            self.assertRejected(bad)

    def test_event_keys_must_be_exact(self) -> None:
        self.assertRejected('{"events":[{"id":"e","at":1,"parents":[]}]}')
        self.assertRejected(
            '{"events":[{"id":"e","at":1,"parents":[],"changes":{},"x":1}]}'
        )

    def test_id_rules(self) -> None:
        self.assertRejected(self.event(id=""))
        self.assertRejected(self.event(id=1))
        self.assertRejected(self.event(id=True))
        one = {"id": "e", "at": 1, "parents": [], "changes": {}}
        self.assertRejected(
            json.dumps({"events": [one, one]}, separators=(",", ":"))
        )

    def test_at_rules(self) -> None:
        self.assertRejected(self.event(at=-1))
        self.assertRejected(self.event(at=True))
        self.assertRejected(self.event(at="1"))
        self.assertRejected(self.event(at=1.5))
        self.assertRejected(self.event(at=None))

    def test_parents_rules(self) -> None:
        self.assertRejected(self.event(parents=("root",)))
        self.assertRejected(self.event(parents=[""]))
        self.assertRejected(self.event(parents=[1]))
        self.assertRejected(self.event(parents=["e", "e"]))
        self.assertRejected(self.event(parents=["missing"]))
        self.assertRejected(self.event(parents=["e"]))  # self-parent

    def test_cycles_are_rejected(self) -> None:
        self.assertRejected(
            '{"events":['
            '{"id":"a","at":0,"parents":["b"],"changes":{}},'
            '{"id":"b","at":0,"parents":["a"],"changes":{}}]}'
        )
        self.assertRejected(
            '{"events":['
            '{"id":"a","at":0,"parents":["c"],"changes":{}},'
            '{"id":"b","at":0,"parents":["a"],"changes":{}},'
            '{"id":"c","at":0,"parents":["b"],"changes":{}}]}'
        )

    def test_changes_rules(self) -> None:
        self.assertRejected(self.event(changes={"": 1}))
        self.assertRejected(self.event(changes={"a": True}))
        self.assertRejected(self.event(changes={"a": "1"}))
        self.assertRejected(self.event(changes={"a": 1.0}))
        self.assertRejected(self.event(changes=[("a", 1)]))

    def test_duplicate_json_object_keys_rejected(self) -> None:
        self.assertRejected('{"events":[],"events":[]}')
        self.assertRejected(
            '{"events":[{"id":"e","id":"e","at":1,"parents":[],"changes":{}}]}'
        )
        self.assertRejected(
            '{"events":[{"id":"e","at":1,"parents":[],"changes":{"a":1,"a":2}}]}'
        )

    def test_negative_and_zero_change_values_are_allowed(self) -> None:
        graph = EventGraph.from_json(self.event(changes={"a": -7, "b": 0}))
        self.assertEqual(graph.replay("e"), {"a": -7, "b": 0})


if __name__ == "__main__":
    unittest.main()
