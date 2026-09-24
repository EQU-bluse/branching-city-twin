import copy
import math
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


class RankImpactsTests(unittest.TestCase):
    def test_result_shape_and_key_order(self) -> None:
        store = make_store()
        result = store.rank_impacts(
            "main",
            {"feature": (("m2", "f2"),)},
            {"y": 3, "x": 2, "z": 1},
        )
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 1)
        row = result[0]
        self.assertEqual(
            list(row),
            ["branch", "score", "first_impact", "unconverged", "attributions"],
        )
        self.assertEqual(row["branch"], "feature")
        self.assertIsInstance(row["unconverged"], tuple)
        self.assertIsInstance(row["attributions"], tuple)

    def test_score_is_weighted_sum_of_final_gaps(self) -> None:
        store = make_store()
        # At (m2, f2): x is 1 vs 2, y is 2 vs 5, z is 7 vs 0.
        row = store.rank_impacts(
            "main",
            {"feature": (("m2", "f2"),)},
            {"x": 2, "y": 3, "z": 1},
        )[0]
        self.assertEqual(row["score"], 1 * 2 + 3 * 3 + 7 * 1)
        self.assertIsInstance(row["score"], int)
        self.assertEqual(row["unconverged"], ("x", "y", "z"))
        self.assertEqual(row["first_impact"], 0)

    def test_score_type_follows_python_arithmetic(self) -> None:
        store = make_store()
        row = store.rank_impacts(
            "main", {"feature": (("m2", "f2"),)}, {"x": 0.5}
        )[0]
        self.assertEqual(row["score"], 0.5)
        self.assertIsInstance(row["score"], float)
        # A float weight on a converged key never enters the sum.
        row = store.rank_impacts(
            "main", {"feature": (("root", "root"),)}, {"x": 0.5}
        )[0]
        self.assertEqual(row["score"], 0)
        self.assertIsInstance(row["score"], int)

    def test_zero_score_is_never_negative_zero(self) -> None:
        store = make_store()
        row = store.rank_impacts(
            "main", {"feature": (("m2", "f2"),)}, {"x": -0.0}
        )[0]
        self.assertEqual(row["score"], 0.0)
        self.assertGreater(math.copysign(1.0, row["score"]), 0)

    def test_unweighted_keys_are_ignored(self) -> None:
        store = make_store()
        row = store.rank_impacts(
            "main", {"feature": (("m2", "f2"),)}, {"x": 1}
        )[0]
        self.assertEqual(row["score"], 1)
        self.assertEqual(row["unconverged"], ("x",))
        self.assertEqual(len(row["attributions"]), 1)

    def test_first_impact_is_earliest_divergence_across_keys(self) -> None:
        store = make_store()
        # z only diverges once m2 is the left node; x diverges at point 0.
        points = (("m1", "f1"), ("m2", "f2"))
        row = store.rank_impacts("main", {"feature": points}, {"z": 1})[0]
        self.assertEqual(row["first_impact"], 1)
        self.assertEqual(row["score"], 7)
        row = store.rank_impacts(
            "main", {"feature": points}, {"z": 1, "x": 1}
        )[0]
        self.assertEqual(row["first_impact"], 0)
        # Never diverging keys leave first_impact at None.
        row = store.rank_impacts(
            "main", {"feature": (("root", "root"),)}, {"x": 1}
        )[0]
        self.assertIsNone(row["first_impact"])

    def test_unconverged_only_from_final_point(self) -> None:
        store = make_store()
        # Diverged at point 0 but equal at the final point: no score, no
        # unconverged keys, yet first_impact records the past divergence.
        row = store.rank_impacts(
            "main",
            {"feature": (("m1", "f1"), ("root", "root"))},
            {"x": 5},
        )[0]
        self.assertEqual(row["score"], 0)
        self.assertEqual(row["first_impact"], 0)
        self.assertEqual(row["unconverged"], ())
        self.assertEqual(row["attributions"], ())

    def test_attributions_match_matrix_last(self) -> None:
        store = make_store()
        points = (("m2", "f2"), ("m1", "f1"))
        keys = ("x", "y", "z")
        matrix_row = store.divergence_matrix(
            "main", {"feature": points}, keys
        )[0]
        last_by_key = {
            summary["key"]: summary["last"]
            for summary in matrix_row["summaries"]
        }
        row = store.rank_impacts(
            "main", {"feature": points}, {"z": 1, "x": 1, "y": 1}
        )[0]
        self.assertEqual(
            len(row["attributions"]), len(row["unconverged"])
        )
        for key, attribution in zip(
            row["unconverged"], row["attributions"]
        ):
            self.assertEqual(attribution["key"], key)
            self.assertEqual(attribution, last_by_key[key])
            self.assertEqual(
                list(attribution), ["key", "fork", "left", "right"]
            )

    def test_rows_sorted_by_score_impact_then_name(self) -> None:
        store = make_store()
        store.create("late", "root")
        store.append("late", "l1", 5, {"x": 3})
        store.create("early", "root")
        store.append("early", "e1", 6, {"x": 3})
        store.create("recon", "root")
        store.append("recon", "r1", 7, {"x": 5})
        store.create("calm", "root")
        result = store.rank_impacts(
            "main",
            {
                "calm": (("root", "root"),),
                "late": (("root", "root"), ("m1", "l1")),
                "recon": (("m1", "r1"), ("root", "root")),
                "early": (("m1", "e1"),),
                "feature": (("m2", "f2"),),
            },
            {"x": 1},
        )
        # feature scores 1; early and late both score 2 (impact 0 beats 1);
        # recon and calm both score 0 (impact 0 beats None).
        self.assertEqual(
            [row["branch"] for row in result],
            ["early", "late", "feature", "recon", "calm"],
        )
        self.assertEqual(
            [row["score"] for row in result], [2, 2, 1, 0, 0]
        )
        self.assertEqual(
            [row["first_impact"] for row in result],
            [0, 1, 0, 0, None],
        )

    def test_name_tiebreak_is_unicode_code_point(self) -> None:
        store = make_store()
        store.create("b-branch", "root")
        store.create("a-branch", "root")
        result = store.rank_impacts(
            "main",
            {
                "b-branch": (("root", "root"),),
                "a-branch": (("root", "root"),),
            },
            {"x": 1},
        )
        self.assertEqual(
            [row["branch"] for row in result], ["a-branch", "b-branch"]
        )

    def test_empty_series_still_validates_reference_and_weights(self) -> None:
        store = make_store()
        self.assertEqual(store.rank_impacts("main", {}, {"x": 1}), ())
        self.assertEqual(store.rank_impacts("main", {}, {}), ())
        with self.assertRaises(KeyError):
            store.rank_impacts("ghost", {}, {"x": 1})
        with self.assertRaises(ValueError):
            store.rank_impacts("", {}, {"x": 1})
        with self.assertRaises(TypeError):
            store.rank_impacts(None, {}, {"x": 1})  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            store.rank_impacts(
                "main", {}, {"x": "1"}  # type: ignore[dict-item]
            )
        with self.assertRaises(ValueError):
            store.rank_impacts("main", {}, {"x": -1})

    def test_empty_weights_still_validates_branches_and_nodes(self) -> None:
        store = make_store()
        result = store.rank_impacts(
            "main", {"feature": (("m2", "f2"),)}, {}
        )
        self.assertEqual(len(result), 1)
        row = result[0]
        self.assertEqual(row["score"], 0)
        self.assertIsNone(row["first_impact"])
        self.assertEqual(row["unconverged"], ())
        self.assertEqual(row["attributions"], ())
        with self.assertRaises(KeyError):
            store.rank_impacts("main", {"ghost": ()}, {})
        with self.assertRaises(KeyError):
            store.rank_impacts(
                "main", {"feature": (("nope", "f1"),)}, {}
            )
        with self.assertRaises(KeyError):
            store.rank_impacts(
                "main", {"feature": (("m1", "nope"),)}, {}
            )
        with self.assertRaises(KeyError):
            store.rank_impacts(
                "main", {"feature": (("f2", "m1"),)}, {}
            )

    def test_weights_must_be_dict(self) -> None:
        store = make_store()
        for bad in ([], None, (), (("x", 1),), "x", 1):
            with self.assertRaises(TypeError):
                store.rank_impacts(
                    "main",
                    {"feature": (("m1", "f1"),)},
                    bad,  # type: ignore[arg-type]
                )

    def test_weight_keys_validation(self) -> None:
        store = make_store()
        for bad_key in (None, 1, 1.5, b"x", ("x",)):
            with self.assertRaises(TypeError):
                store.rank_impacts(
                    "main",
                    {"feature": (("m1", "f1"),)},
                    {bad_key: 1},  # type: ignore[dict-item]
                )
        with self.assertRaises(ValueError):
            store.rank_impacts(
                "main", {"feature": (("m1", "f1"),)}, {"": 1}
            )

    def test_weight_values_validation(self) -> None:
        store = make_store()
        for bad in (True, False, "1", None, (1,), [1], {"x": 1}):
            with self.assertRaises(TypeError):
                store.rank_impacts(
                    "main",
                    {"feature": (("m1", "f1"),)},
                    {"x": bad},  # type: ignore[dict-item]
                )
        for bad in (-1, -0.5, float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError):
                store.rank_impacts(
                    "main", {"feature": (("m1", "f1"),)}, {"x": bad}
                )
        # Zero weights, int or float, are accepted.
        result = store.rank_impacts(
            "main", {"feature": (("m2", "f2"),)}, {"x": 0, "y": 0.0}
        )
        self.assertEqual(result[0]["score"], 0)

    def test_all_inputs_validated_before_branch_lookups(self) -> None:
        store = make_store()
        # Bad weights beat unknown branches and nodes.
        with self.assertRaises(TypeError):
            store.rank_impacts(
                "ghost",
                {"ghost2": (("m1", "f1"),)},
                {"x": "1"},  # type: ignore[dict-item]
            )
        with self.assertRaises(ValueError):
            store.rank_impacts(
                "ghost", {"ghost2": (("m1", "f1"),)}, {"x": -1}
            )
        # Series errors beat weights errors.
        with self.assertRaises(ValueError):
            store.rank_impacts(
                "main",
                {"feature": (("m1", "f1"), ("m1", "f1"))},
                {"x": "1"},  # type: ignore[dict-item]
            )
        with self.assertRaises(ValueError):
            store.rank_impacts(
                "main", {"main": ()}, {"x": "1"}  # type: ignore[dict-item]
            )
        # Reference is looked up before the series branches.
        with self.assertRaises(KeyError):
            store.rank_impacts("ghost", {"ghost2": ()}, {"x": 1})
        # Series branches are looked up in insertion order.
        with self.assertRaises(KeyError):
            store.rank_impacts(
                "main", {"ghost1": (), "ghost2": ()}, {"x": 1}
            )
        # Nodes of the first series are checked before a later series'
        # branch is looked up; left node before right within a pair.
        with self.assertRaises(KeyError):
            store.rank_impacts(
                "main",
                {"feature": (("nope1", "f1"),), "ghost": ()},
                {"x": 1},
            )
        with self.assertRaises(KeyError):
            store.rank_impacts(
                "main",
                {"feature": (("m1", "nope2"), ("nope1", "f1"))},
                {"x": 1},
            )

    def test_results_are_detached_and_unshared(self) -> None:
        store = make_store()
        store.create("side", "root")
        store.append("side", "s1", 5, {"x": 1})
        series = {
            "feature": (("m2", "f2"),),
            "side": (("m2", "s1"),),
        }
        weights = {"x": 2, "y": 3, "z": 1}
        first = store.rank_impacts("main", series, weights)
        self.assertIsNot(first[0], first[1])
        self.assertIsNot(
            first[0]["attributions"], first[1]["attributions"]
        )
        for row in first:
            row["score"] = -1
            row["unconverged"] += ("evil",)  # type: ignore[assignment]
            for attribution in row["attributions"]:  # type: ignore[union-attr]
                attribution["key"] = "evil"
                attribution["left"]["path"] += ("evil",)
        fresh = store.rank_impacts("main", series, weights)
        self.assertEqual(fresh, store.rank_impacts("main", series, weights))
        self.assertEqual(fresh[0]["branch"], "feature")
        self.assertEqual(fresh[0]["score"], 18)
        self.assertEqual(fresh[0]["unconverged"], ("x", "y", "z"))

    def test_query_is_read_only(self) -> None:
        store = make_store()
        heads = dict(store._heads)
        appends = copy.deepcopy(store._appends)
        merges = copy.deepcopy(store._merges)
        replay_main = store.replay("main")
        replay_feature = store.replay("feature")
        audit = store.audit_log()
        store.rank_impacts(
            "main",
            {"feature": (("m2", "f2"), ("m1", "f1"), ("root", "root"))},
            {"x": 2, "y": 3, "z": 1},
        )
        # A failing call must leave state untouched as well.
        with self.assertRaises(KeyError):
            store.rank_impacts(
                "main",
                {"feature": (("m1", "f1"), ("nope", "f2"))},
                {"x": 1},
            )
        with self.assertRaises(ValueError):
            store.rank_impacts(
                "main", {"main": (("root", "root"),)}, {"x": 1}
            )
        with self.assertRaises(ValueError):
            store.rank_impacts(
                "main", {"feature": (("m1", "f1"),)}, {"x": float("nan")}
            )
        self.assertEqual(dict(store._heads), heads)
        self.assertEqual(copy.deepcopy(store._appends), appends)
        self.assertEqual(copy.deepcopy(store._merges), merges)
        self.assertEqual(store.replay("main"), replay_main)
        self.assertEqual(store.replay("feature"), replay_feature)
        self.assertEqual(store.audit_log(), audit)


if __name__ == "__main__":
    unittest.main()
