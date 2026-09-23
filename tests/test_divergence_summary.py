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
    # feature catches up: x and y converge with main at f3.
    store.append("feature", "f3", 5, {"x": -1, "y": -3})
    return store


class DivergenceSummaryTests(unittest.TestCase):
    def test_result_shape_and_key_order(self) -> None:
        store = make_store()
        result = store.divergence_summary(
            "main",
            "feature",
            (
                ("root", "root"),
                ("m1", "f1"),
                ("m2", "f2"),
                ("m2", "f3"),
            ),
            ("y", "x", "z"),
        )
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)
        self.assertEqual([record["key"] for record in result], ["x", "y", "z"])
        for record in result:
            self.assertEqual(
                list(record), ["key", "first_diverged", "transitions", "last"]
            )
            self.assertIsInstance(record["transitions"], tuple)
            for transition in record["transitions"]:
                self.assertEqual(
                    list(transition),
                    [
                        "index",
                        "kind",
                        "left_value",
                        "right_value",
                        "left_cause",
                        "right_cause",
                    ],
                )
            self.assertEqual(
                list(record["last"]), ["key", "fork", "left", "right"]
            )

    def test_transitions_over_diverge_reattribute_converge(self) -> None:
        store = make_store()
        result = {
            record["key"]: record
            for record in store.divergence_summary(
                "main",
                "feature",
                (
                    ("root", "root"),
                    ("m1", "f1"),
                    ("m2", "f2"),
                    ("m2", "f3"),
                ),
                ("y", "x", "z"),
            )
        }
        # x: equal at 0 -> diverged at 1 (causes m1/f1) -> still diverged
        # at 2 with the same causes -> converged at 3.
        self.assertEqual(result["x"]["first_diverged"], 1)
        self.assertEqual(
            [
                (
                    t["index"],
                    t["kind"],
                    t["left_value"],
                    t["right_value"],
                    t["left_cause"],
                    t["right_cause"],
                )
                for t in result["x"]["transitions"]
            ],
            [
                (1, "diverged", 1, 2, "m1", "f1"),
                (3, "converged", 1, 1, None, None),
            ],
        )
        # y: diverged at 1 with no right-side cause yet, reattributed at 2
        # when f2 introduces y on the right, converged at 3.
        self.assertEqual(result["y"]["first_diverged"], 1)
        self.assertEqual(
            [
                (
                    t["index"],
                    t["kind"],
                    t["left_cause"],
                    t["right_cause"],
                )
                for t in result["y"]["transitions"]
            ],
            [
                (1, "diverged", "m1", None),
                (2, "reattributed", "m1", "f2"),
                (3, "converged", None, None),
            ],
        )
        # z appears only on the left at index 2 and stays diverged.
        self.assertEqual(result["z"]["first_diverged"], 2)
        self.assertEqual(
            [
                (t["index"], t["kind"], t["left_value"], t["right_value"])
                for t in result["z"]["transitions"]
            ],
            [(2, "diverged", 7, 0)],
        )

    def test_first_point_already_diverged(self) -> None:
        store = make_store()
        result = store.divergence_summary(
            "main",
            "feature",
            (("m1", "root"), ("m1", "f1")),
            ("x",),
        )
        self.assertEqual(result[0]["first_diverged"], 0)
        self.assertEqual(
            [
                (
                    t["index"],
                    t["kind"],
                    t["left_value"],
                    t["right_value"],
                    t["left_cause"],
                    t["right_cause"],
                )
                for t in result[0]["transitions"]
            ],
            [
                (0, "diverged", 1, 0, "m1", None),
                (1, "reattributed", 1, 2, "m1", "f1"),
            ],
        )

    def test_never_diverged_has_no_transitions(self) -> None:
        store = make_store()
        result = store.divergence_summary(
            "main", "feature", (("root", "root"), ("m1", "f3")), ("x",)
        )
        self.assertIsNone(result[0]["first_diverged"])
        self.assertEqual(result[0]["transitions"], ())

    def test_last_is_final_point_attribution(self) -> None:
        store = make_store()
        points = (("root", "root"), ("m1", "f1"), ("m2", "f3"))
        for record in store.divergence_summary(
            "main", "feature", points, ("y", "x", "z")
        ):
            self.assertEqual(
                record["last"],
                store.attribute_divergence_at(
                    "main", "m2", "feature", "f3", record["key"]
                ),
            )

    def test_empty_points_and_keys(self) -> None:
        store = make_store()
        self.assertEqual(
            store.divergence_summary("main", "feature", (), ("x",)), ()
        )
        self.assertEqual(
            store.divergence_summary("main", "feature", (), ()), ()
        )
        # Empty keys still validate every node.
        self.assertEqual(
            store.divergence_summary(
                "main", "feature", (("m1", "f1"), ("m2", "f3")), ()
            ),
            (),
        )
        with self.assertRaises(KeyError):
            store.divergence_summary(
                "main", "feature", (("nope", "f1"),), ()
            )

    def test_validation_matches_divergence_timeline(self) -> None:
        store = make_store()
        with self.assertRaises(TypeError):
            store.divergence_summary(
                "main", "feature", [("m1", "f1")], ("x",)  # type: ignore[arg-type]
            )
        with self.assertRaises(TypeError):
            store.divergence_summary(
                "main", "feature", (("m1", "f1"),), ["x"]  # type: ignore[arg-type]
            )
        for bad in (("m1",), ("m1", "f1", "x"), ["m1", "f1"], None):
            with self.assertRaises(TypeError):
                store.divergence_summary(
                    "main",
                    "feature",
                    (bad,),  # type: ignore[arg-type]
                    ("x",),
                )
        with self.assertRaises(ValueError):
            store.divergence_summary(
                "main",
                "feature",
                (("m1", "f1"), ("m1", "f1")),
                ("x",),
            )
        with self.assertRaises(ValueError):
            store.divergence_summary(
                "main", "feature", (("m1", "f1"),), ("x", "x")
            )
        with self.assertRaises(KeyError):
            store.divergence_summary(
                "ghost", "feature", (("m1", "f1"),), ("x",)
            )
        with self.assertRaises(KeyError):
            store.divergence_summary(
                "main", "ghost", (("m1", "f1"),), ("x",)
            )
        with self.assertRaises(ValueError):
            store.divergence_summary(
                "main", "main", (("m1", "m1"),), ("x",)
            )
        # Right nodes live on feature's closure, left nodes on main's.
        with self.assertRaises(KeyError):
            store.divergence_summary(
                "main", "feature", (("f2", "m1"),), ("x",)
            )

    def test_results_are_detached_and_unshared(self) -> None:
        store = make_store()
        first = store.divergence_summary(
            "main",
            "feature",
            (("m1", "f1"), ("m2", "f2")),
            ("x", "y"),
        )
        self.assertIsNot(first[0], first[1])
        self.assertIsNot(first[0]["last"], first[1]["last"])
        second = store.divergence_summary(
            "main", "feature", (("m1", "f1"),), ("x",)
        )
        self.assertIsNot(first[0]["last"], second[0]["last"])
        for record in first:
            record["key"] = "evil"
            record["transitions"][0]["left_cause"] = "evil"
            record["last"]["left"]["path"] += ("evil",)
        fresh = store.divergence_summary(
            "main",
            "feature",
            (("m1", "f1"), ("m2", "f2")),
            ("x", "y"),
        )
        self.assertEqual(
            [record["key"] for record in fresh], ["x", "y"]
        )
        self.assertEqual(fresh[0]["transitions"][0]["left_cause"], "m1")
        self.assertEqual(
            fresh[0]["last"]["left"]["path"], ("m1", "m2")
        )

    def test_query_is_read_only(self) -> None:
        store = make_store()
        heads = dict(store._heads)
        appends = copy.deepcopy(store._appends)
        merges = copy.deepcopy(store._merges)
        replay_main = store.replay("main")
        replay_feature = store.replay("feature")
        audit = store.audit_log()
        store.divergence_summary(
            "main",
            "feature",
            (("m1", "f1"), ("m2", "f3"), ("root", "root")),
            ("x", "y", "z"),
        )
        with self.assertRaises(KeyError):
            store.divergence_summary(
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
