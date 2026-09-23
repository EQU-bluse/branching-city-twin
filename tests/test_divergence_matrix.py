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


class DivergenceMatrixTests(unittest.TestCase):
    def test_result_shape_and_orders(self) -> None:
        store = make_store()
        # Insertion order deliberately differs from Unicode order.
        result = store.divergence_matrix(
            "main",
            {
                "feature": (("m2", "f2"), ("root", "f1"), ("m1", "f2")),
            },
            ("y", "x", "z"),
        )
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 1)
        row = result[0]
        self.assertEqual(list(row), ["branch", "summaries", "reconverged"])
        self.assertEqual(row["branch"], "feature")
        summaries = row["summaries"]
        self.assertIsInstance(summaries, tuple)
        self.assertEqual(len(summaries), 3)
        self.assertEqual([s["key"] for s in summaries], ["x", "y", "z"])
        for summary in summaries:
            self.assertEqual(
                list(summary),
                ["key", "first_diverged", "transitions", "last"],
            )
        reconverged = row["reconverged"]
        self.assertIsInstance(reconverged, tuple)
        self.assertEqual(len(reconverged), 3)
        for flag in reconverged:
            self.assertIsInstance(flag, bool)

    def test_rows_follow_series_insertion_order(self) -> None:
        store = make_store()
        store.create("side", "root")
        store.append("side", "s1", 5, {"x": 1})
        result = store.divergence_matrix(
            "main",
            {
                "side": (("m1", "s1"),),
                "feature": (("m1", "f1"),),
            },
            ("x",),
        )
        self.assertEqual([row["branch"] for row in result], ["side", "feature"])

    def test_summaries_equal_divergence_summary(self) -> None:
        store = make_store()
        store.create("side", "root")
        store.append("side", "s1", 5, {"x": 1})
        series = {
            "feature": (("m2", "f2"), ("root", "f1"), ("m1", "f2")),
            "side": (("m1", "s1"), ("root", "root")),
        }
        keys = ("y", "x", "z")
        result = store.divergence_matrix("main", series, keys)
        for row, (name, points) in zip(result, series.items()):
            self.assertEqual(
                row["summaries"],
                store.divergence_summary("main", name, points, keys),
            )

    def test_reconverged_semantics(self) -> None:
        store = make_store()
        # Diverge at point 0, equal at the final point -> reconverged.
        row = store.divergence_matrix(
            "main",
            {"feature": (("m1", "f1"), ("root", "root"))},
            ("x", "z"),
        )[0]
        self.assertEqual(
            [s["first_diverged"] for s in row["summaries"]], [0, None]
        )
        self.assertEqual(row["reconverged"], (True, False))

        # Diverged at the only point and still unequal -> not reconverged.
        row = store.divergence_matrix(
            "main", {"feature": (("m2", "f2"),)}, ("x",)
        )[0]
        self.assertEqual(row["summaries"][0]["first_diverged"], 0)
        self.assertEqual(row["reconverged"], (False,))

        # Equal at every point: first_diverged stays None even though the
        # final values are equal -> not reconverged.
        row = store.divergence_matrix(
            "main", {"feature": (("root", "root"),)}, ("x",)
        )[0]
        self.assertIsNone(row["summaries"][0]["first_diverged"])
        self.assertEqual(row["reconverged"], (False,))

        # Diverge, converge, diverge again: first_diverged set but the
        # final point is unequal -> not reconverged.
        row = store.divergence_matrix(
            "main",
            {"feature": (("m1", "f1"), ("root", "root"), ("m2", "f2"))},
            ("x",),
        )[0]
        self.assertEqual(row["summaries"][0]["first_diverged"], 0)
        last = row["summaries"][0]["last"]
        self.assertNotEqual(
            last["left"]["value"], last["right"]["value"]
        )
        self.assertEqual(row["reconverged"], (False,))

    def test_reconverged_aligned_with_summaries(self) -> None:
        store = make_store()
        # Keys given out of Unicode order; the booleans must line up with
        # the sorted-key summaries.
        row = store.divergence_matrix(
            "main",
            {"feature": (("m1", "f1"), ("root", "root"))},
            ("z", "y", "x"),
        )[0]
        self.assertEqual([s["key"] for s in row["summaries"]], ["x", "y", "z"])
        expected = tuple(
            s["first_diverged"] is not None
            and s["last"]["left"]["value"] == s["last"]["right"]["value"]
            for s in row["summaries"]
        )
        self.assertEqual(row["reconverged"], expected)
        self.assertEqual(expected, (True, True, False))

    def test_empty_series_still_validates_reference_and_keys(self) -> None:
        store = make_store()
        self.assertEqual(store.divergence_matrix("main", {}, ("x",)), ())
        self.assertEqual(store.divergence_matrix("main", {}, ()), ())
        with self.assertRaises(KeyError):
            store.divergence_matrix("ghost", {}, ("x",))
        with self.assertRaises(ValueError):
            store.divergence_matrix("", {}, ("x",))
        with self.assertRaises(TypeError):
            store.divergence_matrix(None, {}, ("x",))  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            store.divergence_matrix("main", {}, ("x", "x"))
        with self.assertRaises(TypeError):
            store.divergence_matrix(
                "main", {}, ["x"]  # type: ignore[arg-type]
            )

    def test_empty_keys_still_validates_nodes(self) -> None:
        store = make_store()
        result = store.divergence_matrix(
            "main",
            {"feature": (("m1", "f1"), ("m2", "f2"))},
            (),
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["summaries"], ())
        self.assertEqual(result[0]["reconverged"], ())
        with self.assertRaises(KeyError):
            store.divergence_matrix(
                "main", {"feature": (("nope", "f1"),)}, ()
            )
        with self.assertRaises(KeyError):
            store.divergence_matrix(
                "main", {"feature": (("m1", "nope"),)}, ()
            )
        # f2 is not on main's head closure; m2 is not on feature's.
        with self.assertRaises(KeyError):
            store.divergence_matrix(
                "main", {"feature": (("f2", "m1"),)}, ()
            )
        with self.assertRaises(KeyError):
            store.divergence_matrix(
                "main", {"feature": (("m1", "m2"),)}, ()
            )

    def test_series_must_be_dict(self) -> None:
        store = make_store()
        for bad in ([], None, (), (("feature", ()),), "feature"):
            with self.assertRaises(TypeError):
                store.divergence_matrix(
                    "main", bad, ("x",)  # type: ignore[arg-type]
                )

    def test_series_keys_validation_and_insertion_order(self) -> None:
        store = make_store()
        for bad in (None, 1, 1.5, b"feature", ["feature"]):
            with self.assertRaises(TypeError):
                store.divergence_matrix(
                    "main",
                    {bad: ()},  # type: ignore[dict-item]
                    ("x",),
                )
        with self.assertRaises(ValueError):
            store.divergence_matrix("main", {"": ()}, ("x",))
        with self.assertRaises(ValueError):
            store.divergence_matrix("main", {"main": ()}, ("x",))
        # Keys are checked in insertion order: the first bad key wins,
        # before any later series is examined.
        with self.assertRaises(TypeError):
            store.divergence_matrix(
                "main",
                {1: (), "feature": [("m1", "f1")]},  # type: ignore[dict-item]
                ("x",),
            )
        with self.assertRaises(ValueError):
            store.divergence_matrix(
                "main",
                {"main": (), 1: ()},  # type: ignore[dict-item]
                ("x",),
            )

    def test_points_validation_per_series(self) -> None:
        store = make_store()
        # Points containers must be tuples.
        with self.assertRaises(TypeError):
            store.divergence_matrix(
                "main", {"feature": [("m1", "f1")]}, ("x",)
            )
        # Points must be length-2 tuples.
        for bad in [("m1",), ("m1", "f1", "x"), ["m1", "f1"], "m1", None]:
            with self.assertRaises(TypeError):
                store.divergence_matrix(
                    "main",
                    {"feature": (bad,)},  # type: ignore[dict-item]
                    ("x",),
                )
        # Empty node ids and non-str nodes.
        with self.assertRaises(ValueError):
            store.divergence_matrix(
                "main", {"feature": (("", "f1"),)}, ("x",)
            )
        with self.assertRaises(TypeError):
            store.divergence_matrix(
                "main",
                {"feature": ((1, "f1"),)},  # type: ignore[dict-item]
                ("x",),
            )
        # Duplicate pairs within one series are rejected; the same pair in
        # two different series is allowed.
        with self.assertRaises(ValueError):
            store.divergence_matrix(
                "main",
                {"feature": (("m1", "f1"), ("m1", "f1"))},
                ("x",),
            )
        store.create("side", "root")
        result = store.divergence_matrix(
            "main",
            {"feature": (("root", "root"),), "side": (("root", "root"),)},
            ("x",),
        )
        self.assertEqual(len(result), 2)

    def test_keys_validation(self) -> None:
        store = make_store()
        for bad in ([], None, ["x"], "x"):
            with self.assertRaises(TypeError):
                store.divergence_matrix(
                    "main",
                    {"feature": (("m1", "f1"),)},
                    bad,  # type: ignore[arg-type]
                )
        with self.assertRaises(ValueError):
            store.divergence_matrix(
                "main", {"feature": (("m1", "f1"),)}, ("",)
            )
        with self.assertRaises(TypeError):
            store.divergence_matrix(
                "main",
                {"feature": (("m1", "f1"),)},
                (1,),  # type: ignore[arg-type]
            )
        with self.assertRaises(ValueError):
            store.divergence_matrix(
                "main", {"feature": (("m1", "f1"),)}, ("x", "x")
            )

    def test_all_inputs_validated_before_branch_lookups(self) -> None:
        store = make_store()
        # Bad input beats unknown branches and nodes in every position.
        with self.assertRaises(ValueError):
            store.divergence_matrix(
                "ghost", {"ghost2": (("m1", "f1"),)}, ("",)
            )
        with self.assertRaises(TypeError):
            store.divergence_matrix(
                "ghost",
                {"ghost2": (("m1", "f1"),)},
                "x",  # type: ignore[arg-type]
            )
        # A series name is checked before that series' points...
        with self.assertRaises(TypeError):
            store.divergence_matrix(
                "main",
                # First entry's name fails before its bad points are read.
                {1: [("m1", "f1")]},  # type: ignore[dict-item]
                ("x",),
            )
        # All series' points are walked before keys or branches.
        with self.assertRaises(TypeError):
            store.divergence_matrix(
                "main",
                {
                    "feature": [("m1", "f1")],
                    1: (),  # type: ignore[dict-item]
                },
                ("x",),
            )
        with self.assertRaises(ValueError):
            store.divergence_matrix(
                "main",
                {"feature": (("m1", "f1"),)},
                ("x", "x"),
            )
        # Reference is looked up before the series branches.
        with self.assertRaises(KeyError):
            store.divergence_matrix(
                "ghost", {"ghost2": ()}, ("x",)
            )
        # Series branches are looked up in insertion order.
        with self.assertRaises(KeyError):
            store.divergence_matrix(
                "main", {"ghost1": (), "ghost2": ()}, ("x",)
            )
        # Nodes of the first series are checked before a later series'
        # branch is looked up; left node before right within a pair.
        with self.assertRaises(KeyError):
            store.divergence_matrix(
                "main",
                {"feature": (("nope1", "f1"),), "ghost": ()},
                (),
            )
        with self.assertRaises(KeyError):
            store.divergence_matrix(
                "main",
                {"feature": (("m1", "nope2"), ("nope1", "f1"))},
                (),
            )

    def test_results_are_detached_and_unshared(self) -> None:
        store = make_store()
        store.create("side", "root")
        series = {
            "feature": (("m2", "f2"), ("m1", "f1")),
            "side": (("root", "root"),),
        }
        first = store.divergence_matrix("main", series, ("x", "y"))
        # Distinct objects across rows, keys and calls -- even when two
        # series carry identical points.
        same_points = {
            "feature": (("root", "root"),),
            "side": (("root", "root"),),
        }
        equal_rows = store.divergence_matrix("main", same_points, ("x",))
        self.assertEqual(
            equal_rows[0]["summaries"], equal_rows[1]["summaries"]
        )
        self.assertIsNot(
            equal_rows[0]["summaries"], equal_rows[1]["summaries"]
        )
        self.assertIsNot(
            equal_rows[0]["summaries"][0], equal_rows[1]["summaries"][0]
        )
        second = store.divergence_matrix(
            "main", {"feature": (("m2", "f2"), ("m1", "f1"))}, ("x", "y")
        )
        self.assertIsNot(
            first[0]["summaries"], second[0]["summaries"]
        )
        for row in first:
            for summary in row["summaries"]:
                summary["key"] = "evil"
                summary["transitions"] = ("evil",)  # type: ignore[assignment]
                summary["last"]["left"]["path"] += ("evil",)
                summary["last"]["left"]["affected"] += ("evil",)
        fresh = store.divergence_matrix(
            "main",
            {"feature": (("m2", "f2"), ("m1", "f1"))},
            ("x", "y"),
        )
        self.assertEqual(
            fresh,
            store.divergence_matrix(
                "main",
                {"feature": (("m2", "f2"), ("m1", "f1"))},
                ("x", "y"),
            ),
        )
        self.assertEqual(fresh[0]["summaries"][0]["key"], "x")
        # The final point is (m1, f1), so the last attribution's path
        # ends at m1, not m2.
        self.assertEqual(
            fresh[0]["summaries"][0]["last"]["left"]["path"], ("m1",)
        )

    def test_query_is_read_only(self) -> None:
        store = make_store()
        heads = dict(store._heads)
        appends = copy.deepcopy(store._appends)
        merges = copy.deepcopy(store._merges)
        replay_main = store.replay("main")
        replay_feature = store.replay("feature")
        audit = store.audit_log()
        store.divergence_matrix(
            "main",
            {
                "feature": (
                    ("m2", "f2"),
                    ("m1", "f1"),
                    ("root", "root"),
                )
            },
            ("x", "y", "z"),
        )
        # A failing call must leave state untouched as well.
        with self.assertRaises(KeyError):
            store.divergence_matrix(
                "main",
                {"feature": (("m1", "f1"), ("nope", "f2"))},
                ("x", "y"),
            )
        with self.assertRaises(ValueError):
            store.divergence_matrix(
                "main", {"main": (("root", "root"),)}, ("x",)
            )
        self.assertEqual(dict(store._heads), heads)
        self.assertEqual(copy.deepcopy(store._appends), appends)
        self.assertEqual(copy.deepcopy(store._merges), merges)
        self.assertEqual(store.replay("main"), replay_main)
        self.assertEqual(store.replay("feature"), replay_feature)
        self.assertEqual(store.audit_log(), audit)


if __name__ == "__main__":
    unittest.main()
