import copy
import unittest

from city_twin.branches import BranchStore
from city_twin.event_graph import EventGraph


def make_store() -> BranchStore:
    graph = EventGraph()
    graph.add("root", 0, (), {})
    store = BranchStore(graph)
    store.create("main", "root")
    store.create("feature", "root")
    store.append("main", "m1", 1, {"x": 1, "y": 2})
    store.append("feature", "f1", 2, {"x": 2})
    store.append("feature", "f2", 3, {"y": 5})
    store.append("main", "m2", 4, {"z": 7})
    return store


class DivergenceTimelineTests(unittest.TestCase):
    def test_result_shape_and_orders(self) -> None:
        store = make_store()
        points = (("m2", "f2"), ("root", "f1"), ("m1", "f2"))
        keys = ("y", "x", "z")
        result = store.divergence_timeline(
            "main", "feature", points, keys
        )
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)
        for point_result in result:
            self.assertIsInstance(point_result, tuple)
            self.assertEqual(len(point_result), 3)
            for record in point_result:
                self.assertEqual(
                    list(record), ["key", "fork", "left", "right"]
                )
                self.assertEqual(
                    list(record["left"]),
                    ["value", "cause", "path", "affected"],
                )
                self.assertEqual(
                    list(record["right"]),
                    ["value", "cause", "path", "affected"],
                )
        # Outer order follows points; inner records sort keys by code point.
        self.assertEqual(
            [[record["key"] for record in point] for point in result],
            [["x", "y", "z"], ["x", "y", "z"], ["x", "y", "z"]],
        )

    def test_each_item_equals_attribute_divergences_at(self) -> None:
        store = make_store()
        points = (("m2", "f2"), ("root", "f1"), ("m1", "f2"))
        keys = ("y", "x", "z")
        result = store.divergence_timeline(
            "main", "feature", points, keys
        )
        for (node_a, node_b), got in zip(points, result):
            expected = store.attribute_divergences_at(
                "main", node_a, "feature", node_b, keys
            )
            self.assertEqual(got, expected)

    def test_empty_points_still_validates_branches(self) -> None:
        store = make_store()
        self.assertEqual(
            store.divergence_timeline("main", "feature", (), ("x",)), ()
        )
        self.assertEqual(
            store.divergence_timeline("main", "feature", (), ()), ()
        )
        with self.assertRaises(KeyError):
            store.divergence_timeline("ghost", "feature", (), ())
        with self.assertRaises(KeyError):
            store.divergence_timeline("main", "ghost", (), ())
        with self.assertRaises(ValueError):
            store.divergence_timeline("main", "main", (), ("x",))

    def test_empty_keys_still_validates_all_nodes(self) -> None:
        store = make_store()
        self.assertEqual(
            store.divergence_timeline(
                "main", "feature", (("m1", "f1"), ("m2", "f2")), ()
            ),
            ((), ()),
        )
        with self.assertRaises(KeyError):
            store.divergence_timeline(
                "main", "feature", (("nope", "f1"),), ()
            )
        with self.assertRaises(KeyError):
            store.divergence_timeline(
                "main", "feature", (("m1", "nope"),), ()
            )
        # f2 is not on main's head closure; m2 is not on feature's.
        with self.assertRaises(KeyError):
            store.divergence_timeline(
                "main", "feature", (("f2", "m1"),), ()
            )
        with self.assertRaises(KeyError):
            store.divergence_timeline(
                "main", "feature", (("m1", "m2"),), ()
            )

    def test_containers_must_be_tuples(self) -> None:
        store = make_store()
        for bad in ([], None, [("m1", "f1")]):
            with self.assertRaises(TypeError):
                store.divergence_timeline(
                    "main", "feature", bad, ("x",)  # type: ignore[arg-type]
                )
        for bad in ([], None, ["x"]):
            with self.assertRaises(TypeError):
                store.divergence_timeline(
                    "main", "feature", (("m1", "f1"),), bad  # type: ignore[arg-type]
                )

    def test_points_must_be_length_two_tuples(self) -> None:
        store = make_store()
        for bad in [
            ("m1",),
            ("m1", "f1", "x"),
            ["m1", "f1"],
            "m1",
            None,
        ]:
            with self.assertRaises(TypeError):
                store.divergence_timeline(
                    "main",
                    "feature",
                    (bad,),  # type: ignore[arg-type]
                    ("x",),
                )

    def test_non_str_arguments_raise_type_error(self) -> None:
        store = make_store()
        for bad in (None, 1, 1.5, b"main", ["main"]):
            with self.assertRaises(TypeError):
                store.divergence_timeline(
                    bad,  # type: ignore[arg-type]
                    "feature",
                    (("m1", "f1"),),
                    ("x",),
                )
            with self.assertRaises(TypeError):
                store.divergence_timeline(
                    "main",
                    bad,  # type: ignore[arg-type]
                    (("m1", "f1"),),
                    ("x",),
                )
            with self.assertRaises(TypeError):
                store.divergence_timeline(
                    "main",
                    "feature",
                    ((bad, "f1"),),  # type: ignore[arg-type]
                    ("x",),
                )
            with self.assertRaises(TypeError):
                store.divergence_timeline(
                    "main",
                    "feature",
                    (("m1", bad),),  # type: ignore[arg-type]
                    ("x",),
                )
            with self.assertRaises(TypeError):
                store.divergence_timeline(
                    "main",
                    "feature",
                    (("m1", "f1"),),
                    (bad,),  # type: ignore[arg-type]
                )

    def test_empty_strings_raise_value_error(self) -> None:
        store = make_store()
        with self.assertRaises(ValueError):
            store.divergence_timeline(
                "", "feature", (("m1", "f1"),), ("x",)
            )
        with self.assertRaises(ValueError):
            store.divergence_timeline(
                "main", "", (("m1", "f1"),), ("x",)
            )
        with self.assertRaises(ValueError):
            store.divergence_timeline(
                "main", "feature", (("", "f1"),), ("x",)
            )
        with self.assertRaises(ValueError):
            store.divergence_timeline(
                "main", "feature", (("m1", ""),), ("x",)
            )
        with self.assertRaises(ValueError):
            store.divergence_timeline(
                "main", "feature", (("m1", "f1"),), ("",)
            )

    def test_duplicate_points_and_keys_raise_value_error(self) -> None:
        store = make_store()
        with self.assertRaises(ValueError):
            store.divergence_timeline(
                "main",
                "feature",
                (("m1", "f1"), ("m1", "f1")),
                ("x",),
            )
        with self.assertRaises(ValueError):
            store.divergence_timeline(
                "main",
                "feature",
                (("m1", "f1"),),
                ("x", "x"),
            )
        # A reversed pair is a distinct pair, not a duplicate. "side"
        # shares root/m1 with main, so both orderings are valid nodes.
        store.create("side", "m1")
        result = store.divergence_timeline(
            "main",
            "side",
            (("m1", "root"), ("root", "m1")),
            ("x",),
        )
        self.assertEqual(len(result), 2)

    def test_unknown_branch_and_same_name_errors(self) -> None:
        store = make_store()
        with self.assertRaises(KeyError):
            store.divergence_timeline(
                "ghost", "feature", (("m1", "f1"),), ("x",)
            )
        with self.assertRaises(KeyError):
            store.divergence_timeline(
                "main", "ghost", (("m1", "f1"),), ("x",)
            )
        with self.assertRaises(ValueError):
            store.divergence_timeline(
                "main", "main", (("m1", "m1"),), ("x",)
            )

    def test_validation_precedes_state_lookup(self) -> None:
        store = make_store()
        # Unknown branches raise KeyError before the same-name check.
        with self.assertRaises(KeyError):
            store.divergence_timeline(
                "ghost", "ghost", (("m1", "f1"),), ("x",)
            )
        # Invalid input is rejected before branches are consulted.
        with self.assertRaises(ValueError):
            store.divergence_timeline(
                "ghost", "ghost", (("m1", "f1"),), ("",)
            )
        with self.assertRaises(TypeError):
            store.divergence_timeline(
                "ghost", "ghost", (("m1", "f1"),), "x"  # type: ignore[arg-type]
            )
        # Nodes are checked pair by pair, left before right.
        with self.assertRaises(KeyError):
            store.divergence_timeline(
                "main",
                "feature",
                (("nope1", "f1"), ("m1", "nope2")),
                (),
            )
        with self.assertRaises(KeyError):
            store.divergence_timeline(
                "main",
                "feature",
                (("m1", "nope2"), ("nope1", "f1")),
                (),
            )

    def test_results_are_detached_and_unshared(self) -> None:
        store = make_store()
        first = store.divergence_timeline(
            "main",
            "feature",
            (("m2", "f2"), ("m1", "f1")),
            ("x", "y"),
        )
        # Distinct objects across points, keys and calls.
        self.assertIsNot(first[0][0], first[0][1])
        self.assertIsNot(first[0][0], first[1][0])
        self.assertIsNot(first[0][0]["left"], first[1][0]["left"])
        second = store.divergence_timeline(
            "main", "feature", (("m2", "f2"),), ("x",)
        )
        self.assertIsNot(first[0][0], second[0][0])
        for point in first:
            for record in point:
                record["key"] = "evil"
                record["left"]["path"] += ("evil",)
                record["left"]["affected"] += ("evil",)
        fresh = store.divergence_timeline(
            "main", "feature", (("m2", "f2"),), ("x",)
        )
        self.assertEqual(
            fresh,
            store.divergence_timeline(
                "main", "feature", (("m2", "f2"),), ("x",)
            ),
        )
        self.assertEqual(fresh[0][0]["key"], "x")
        self.assertEqual(fresh[0][0]["left"]["path"], ("m1", "m2"))

    def test_query_is_read_only(self) -> None:
        store = make_store()
        heads = dict(store._heads)
        appends = copy.deepcopy(store._appends)
        merges = copy.deepcopy(store._merges)
        replay_main = store.replay("main")
        replay_feature = store.replay("feature")
        audit = store.audit_log()
        store.divergence_timeline(
            "main",
            "feature",
            (("m2", "f2"), ("m1", "f1"), ("root", "root")),
            ("x", "y", "z"),
        )
        # A failing call must leave state untouched as well.
        with self.assertRaises(KeyError):
            store.divergence_timeline(
                "main",
                "feature",
                (("m1", "f1"), ("nope", "f2")),
                ("x", "y"),
            )
        self.assertEqual(dict(store._heads), heads)
        self.assertEqual(copy.deepcopy(store._appends), appends)
        self.assertEqual(copy.deepcopy(store._merges), merges)
        self.assertEqual(store.replay("main"), replay_main)
        self.assertEqual(store.replay("feature"), replay_feature)
        self.assertEqual(store.audit_log(), audit)


if __name__ == "__main__":
    unittest.main()
