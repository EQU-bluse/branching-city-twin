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


class ImpactBudgetTests(unittest.TestCase):
    def test_result_shape_and_key_order(self) -> None:
        store = make_store()
        result = store.impact_budget(
            "main",
            {"feature": (("m2", "f2"),)},
            {"y": 3, "x": 2, "z": 1},
            100,
            {},
        )
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 1)
        row = result[0]
        self.assertEqual(
            list(row),
            [
                "branch",
                "breached",
                "first_breach",
                "max_overshoot",
                "remaining",
                "exceeded",
                "attributions",
            ],
        )
        self.assertEqual(row["branch"], "feature")
        self.assertIsInstance(row["exceeded"], tuple)
        self.assertIsInstance(row["attributions"], tuple)

    def test_under_budget_never_breaches(self) -> None:
        store = make_store()
        # At (m2, f2): x is 1 vs 2, y is 2 vs 5, z is 7 vs 0, so the
        # total risk is 1*2 + 3*3 + 7*1 = 18.
        row = store.impact_budget(
            "main",
            {"feature": (("m2", "f2"),)},
            {"x": 2, "y": 3, "z": 1},
            18,
            {"x": 1, "y": 3, "z": 7},
        )[0]
        # Equality is not a breach, for the total or any key.
        self.assertFalse(row["breached"])
        self.assertIsNone(row["first_breach"])
        self.assertEqual(row["max_overshoot"], 0)
        self.assertEqual(row["remaining"], 0)
        self.assertEqual(row["exceeded"], ())
        self.assertEqual(row["attributions"], ())

    def test_total_breach(self) -> None:
        store = make_store()
        row = store.impact_budget(
            "main",
            {"feature": (("m2", "f2"),)},
            {"x": 2, "y": 3, "z": 1},
            10,
            {},
        )[0]
        self.assertTrue(row["breached"])
        self.assertEqual(row["first_breach"], 0)
        self.assertEqual(row["max_overshoot"], 8)
        self.assertIsInstance(row["max_overshoot"], int)
        # The remaining budget may go negative.
        self.assertEqual(row["remaining"], -8)
        self.assertEqual(row["exceeded"], ())
        self.assertEqual(row["attributions"], ())

    def test_key_breach_is_independent(self) -> None:
        store = make_store()
        row = store.impact_budget(
            "main",
            {"feature": (("m2", "f2"),)},
            {"x": 2, "y": 3, "z": 1},
            100,
            {"y": 2},
        )[0]
        self.assertTrue(row["breached"])
        self.assertEqual(row["first_breach"], 0)
        # The per-key excess is (3 - 2) * weight 3 = 3.
        self.assertEqual(row["max_overshoot"], 3)
        self.assertEqual(row["remaining"], 82)
        self.assertEqual(row["exceeded"], ("y",))
        self.assertEqual(len(row["attributions"]), 1)

    def test_max_overshoot_combines_total_and_key_excess(self) -> None:
        store = make_store()
        # Total excess 18 - 10 = 8 beats the key excess (3 - 2) * 3 = 3.
        row = store.impact_budget(
            "main",
            {"feature": (("m2", "f2"),)},
            {"x": 2, "y": 3, "z": 1},
            10,
            {"y": 2},
        )[0]
        self.assertEqual(row["max_overshoot"], 8)
        # A key excess can in turn beat the total excess.
        row = store.impact_budget(
            "main",
            {"feature": (("m2", "f2"),)},
            {"x": 2, "y": 3, "z": 1},
            17,
            {"y": 0},
        )[0]
        self.assertEqual(row["max_overshoot"], 9)

    def test_risk_type_follows_python_arithmetic(self) -> None:
        store = make_store()
        row = store.impact_budget(
            "main", {"feature": (("m2", "f2"),)}, {"x": 0.5}, 10, {}
        )[0]
        self.assertEqual(row["remaining"], 9.5)
        self.assertIsInstance(row["remaining"], float)
        row = store.impact_budget(
            "main", {"feature": (("m2", "f2"),)}, {"x": 2}, 10.0, {}
        )[0]
        self.assertEqual(row["remaining"], 8.0)
        self.assertIsInstance(row["remaining"], float)

    def test_zero_risk_is_never_negative_zero(self) -> None:
        store = make_store()
        row = store.impact_budget(
            "main", {"feature": (("m2", "f2"),)}, {"x": -0.0}, 10, {}
        )[0]
        self.assertEqual(row["remaining"], 10.0)
        self.assertGreater(math.copysign(1.0, row["remaining"]), 0)
        self.assertEqual(row["max_overshoot"], 0)
        self.assertGreater(math.copysign(1.0, row["max_overshoot"]), 0)

    def test_first_breach_tracks_checkpoints(self) -> None:
        store = make_store()
        # Risks at (m1, f1), (m2, f2), (root, root) are 3, 11 and 0.
        points = (("m1", "f1"), ("m2", "f2"), ("root", "root"))
        row = store.impact_budget(
            "main", {"feature": points}, {"x": 1, "y": 1, "z": 1}, 5, {}
        )[0]
        self.assertTrue(row["breached"])
        self.assertEqual(row["first_breach"], 1)
        self.assertEqual(row["max_overshoot"], 6)
        # The final checkpoint is calm again.
        self.assertEqual(row["remaining"], 5)
        self.assertEqual(row["exceeded"], ())
        row = store.impact_budget(
            "main", {"feature": points}, {"x": 1, "y": 1, "z": 1}, 2, {}
        )[0]
        self.assertEqual(row["first_breach"], 0)

    def test_exceeded_keys_sorted_and_attributed(self) -> None:
        store = make_store()
        row = store.impact_budget(
            "main",
            {"feature": (("m2", "f2"),)},
            {"x": 1, "y": 1, "z": 1},
            100,
            {"z": 1, "x": 0},
        )[0]
        self.assertEqual(row["exceeded"], ("x", "z"))
        self.assertEqual(
            [attribution["key"] for attribution in row["attributions"]],
            ["x", "z"],
        )

    def test_exceeded_keys_only_from_final_point(self) -> None:
        store = make_store()
        # x exceeds its budget at (m1, f1) but not at (root, root).
        row = store.impact_budget(
            "main",
            {"feature": (("m1", "f1"), ("root", "root"))},
            {"x": 1},
            100,
            {"x": 0},
        )[0]
        self.assertTrue(row["breached"])
        self.assertEqual(row["first_breach"], 0)
        self.assertEqual(row["exceeded"], ())
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
        row = store.impact_budget(
            "main",
            {"feature": points},
            {"z": 1, "x": 1, "y": 1},
            100,
            {"y": 1, "x": 0},
        )[0]
        # At the final point (m1, f1) x differs by 1 and y by 2.
        self.assertEqual(row["exceeded"], ("x", "y"))
        self.assertEqual(
            len(row["attributions"]), len(row["exceeded"])
        )
        for key, attribution in zip(row["exceeded"], row["attributions"]):
            self.assertEqual(attribution["key"], key)
            self.assertEqual(attribution, last_by_key[key])
            self.assertEqual(
                list(attribution), ["key", "fork", "left", "right"]
            )

    def test_rows_sorted_breach_overshoot_impact_then_name(self) -> None:
        store = make_store()
        store.create("big", "root")
        store.append("big", "b1", 5, {"x": 10})
        store.create("early", "root")
        store.append("early", "e1", 6, {"x": 8})
        store.create("late", "root")
        store.append("late", "l1", 7, {"x": 8})
        store.create("small", "root")
        store.append("small", "s1", 8, {"x": 7})
        store.create("calm", "root")
        result = store.impact_budget(
            "main",
            {
                "calm": (("root", "root"),),
                "late": (("root", "root"), ("root", "l1")),
                "small": (("root", "s1"),),
                "early": (("root", "e1"),),
                "big": (("root", "b1"),),
            },
            {"x": 1},
            5,
            {},
        )
        # Breaching rows first: overshoot 5, then 3 (first_breach 0
        # beats 1), then 2; the calm row trails with overshoot 0.
        self.assertEqual(
            [row["branch"] for row in result],
            ["big", "early", "late", "small", "calm"],
        )
        self.assertEqual(
            [row["max_overshoot"] for row in result], [5, 3, 3, 2, 0]
        )
        self.assertEqual(
            [row["first_breach"] for row in result],
            [0, 0, 1, 0, None],
        )
        self.assertEqual(
            [row["breached"] for row in result],
            [True, True, True, True, False],
        )

    def test_name_tiebreak_is_unicode_code_point(self) -> None:
        store = make_store()
        store.create("b-branch", "root")
        store.create("a-branch", "root")
        result = store.impact_budget(
            "main",
            {
                "b-branch": (("root", "root"),),
                "a-branch": (("root", "root"),),
            },
            {"x": 1},
            5,
            {},
        )
        self.assertEqual(
            [row["branch"] for row in result], ["a-branch", "b-branch"]
        )

    def test_empty_points_give_zero_risk_row(self) -> None:
        store = make_store()
        row = store.impact_budget(
            "main", {"feature": ()}, {"x": 1}, 7, {}
        )[0]
        self.assertFalse(row["breached"])
        self.assertIsNone(row["first_breach"])
        self.assertEqual(row["max_overshoot"], 0)
        self.assertEqual(row["remaining"], 7)
        self.assertEqual(row["exceeded"], ())
        self.assertEqual(row["attributions"], ())

    def test_empty_series_still_validates_budgets_and_reference(self) -> None:
        store = make_store()
        self.assertEqual(store.impact_budget("main", {}, {"x": 1}, 1, {}), ())
        self.assertEqual(store.impact_budget("main", {}, {}, 0, {}), ())
        with self.assertRaises(KeyError):
            store.impact_budget("ghost", {}, {"x": 1}, 1, {})
        with self.assertRaises(ValueError):
            store.impact_budget("", {}, {"x": 1}, 1, {})
        with self.assertRaises(TypeError):
            store.impact_budget(None, {}, {"x": 1}, 1, {})  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            store.impact_budget(
                "main", {}, {"x": 1}, "1", {}  # type: ignore[arg-type]
            )
        with self.assertRaises(ValueError):
            store.impact_budget("main", {}, {"x": 1}, -1, {})
        with self.assertRaises(TypeError):
            store.impact_budget(
                "main", {}, {"x": 1}, 1, []  # type: ignore[arg-type]
            )

    def test_empty_weights_still_validates_branches_and_nodes(self) -> None:
        store = make_store()
        result = store.impact_budget(
            "main", {"feature": (("m2", "f2"),)}, {}, 10, {}
        )
        self.assertEqual(len(result), 1)
        row = result[0]
        self.assertFalse(row["breached"])
        self.assertIsNone(row["first_breach"])
        self.assertEqual(row["max_overshoot"], 0)
        self.assertEqual(row["remaining"], 10)
        self.assertEqual(row["exceeded"], ())
        self.assertEqual(row["attributions"], ())
        with self.assertRaises(KeyError):
            store.impact_budget("main", {"ghost": ()}, {}, 10, {})
        with self.assertRaises(KeyError):
            store.impact_budget(
                "main", {"feature": (("nope", "f1"),)}, {}, 10, {}
            )
        with self.assertRaises(KeyError):
            store.impact_budget(
                "main", {"feature": (("m1", "nope"),)}, {}, 10, {}
            )
        with self.assertRaises(KeyError):
            store.impact_budget(
                "main", {"feature": (("f2", "m1"),)}, {}, 10, {}
            )

    def test_total_budget_validation(self) -> None:
        store = make_store()
        for bad in (True, False, "1", None, (1,), [1], {"x": 1}):
            with self.assertRaises(TypeError):
                store.impact_budget(
                    "main",
                    {"feature": (("m1", "f1"),)},
                    {"x": 1},
                    bad,  # type: ignore[arg-type]
                    {},
                )
        for bad in (-1, -0.5, float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError):
                store.impact_budget(
                    "main",
                    {"feature": (("m1", "f1"),)},
                    {"x": 1},
                    bad,
                    {},
                )
        # Zero budgets, int or float, are accepted.
        result = store.impact_budget(
            "main", {"feature": (("root", "root"),)}, {"x": 1}, 0, {}
        )
        self.assertFalse(result[0]["breached"])
        result = store.impact_budget(
            "main", {"feature": (("root", "root"),)}, {"x": 1}, 0.0, {}
        )
        self.assertFalse(result[0]["breached"])

    def test_key_budgets_must_be_dict(self) -> None:
        store = make_store()
        for bad in ([], None, (), (("x", 1),), "x", 1):
            with self.assertRaises(TypeError):
                store.impact_budget(
                    "main",
                    {"feature": (("m1", "f1"),)},
                    {"x": 1},
                    1,
                    bad,  # type: ignore[arg-type]
                )

    def test_key_budget_keys_validation(self) -> None:
        store = make_store()
        for bad_key in (None, 1, 1.5, b"x", ("x",)):
            with self.assertRaises(TypeError):
                store.impact_budget(
                    "main",
                    {"feature": (("m1", "f1"),)},
                    {"x": 1},
                    1,
                    {bad_key: 1},  # type: ignore[dict-item]
                )
        with self.assertRaises(ValueError):
            store.impact_budget(
                "main",
                {"feature": (("m1", "f1"),)},
                {"x": 1},
                1,
                {"": 1},
            )

    def test_key_budget_values_validation(self) -> None:
        store = make_store()
        for bad in (True, False, "1", None, (1,), [1], {"x": 1}):
            with self.assertRaises(TypeError):
                store.impact_budget(
                    "main",
                    {"feature": (("m1", "f1"),)},
                    {"x": 1},
                    1,
                    {"x": bad},  # type: ignore[dict-item]
                )
        for bad in (-1, -0.5, float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError):
                store.impact_budget(
                    "main",
                    {"feature": (("m1", "f1"),)},
                    {"x": 1},
                    1,
                    {"x": bad},
                )
        # Zero key budgets, int or float, are accepted.
        row = store.impact_budget(
            "main",
            {"feature": (("root", "root"),)},
            {"x": 1},
            1,
            {"x": 0},
        )[0]
        self.assertFalse(row["breached"])

    def test_key_budgets_must_stay_within_weights(self) -> None:
        store = make_store()
        with self.assertRaises(ValueError):
            store.impact_budget(
                "main",
                {"feature": (("m1", "f1"),)},
                {"x": 1},
                1,
                {"y": 1},
            )
        with self.assertRaises(ValueError):
            store.impact_budget(
                "main",
                {"feature": (("m1", "f1"),)},
                {},
                1,
                {"x": 1},
            )
        # Omitting weighted keys is fine: they have no per-key cap.
        row = store.impact_budget(
            "main",
            {"feature": (("m2", "f2"),)},
            {"x": 1, "y": 1},
            100,
            {"x": 0},
        )[0]
        self.assertEqual(row["exceeded"], ("x",))

    def test_all_inputs_validated_before_branch_lookups(self) -> None:
        store = make_store()
        # Bad budgets beat unknown branches and nodes.
        with self.assertRaises(TypeError):
            store.impact_budget(
                "ghost",
                {"ghost2": (("m1", "f1"),)},
                {"x": 1},
                "1",  # type: ignore[arg-type]
                {},
            )
        with self.assertRaises(ValueError):
            store.impact_budget(
                "ghost", {"ghost2": (("m1", "f1"),)}, {"x": 1}, -1, {}
            )
        with self.assertRaises(ValueError):
            store.impact_budget(
                "ghost",
                {"ghost2": (("m1", "f1"),)},
                {"x": 1},
                1,
                {"y": 1},
            )
        # Weights errors beat budget errors.
        with self.assertRaises(TypeError):
            store.impact_budget(
                "main",
                {"feature": (("m1", "f1"),)},
                {"x": "1"},  # type: ignore[dict-item]
                "1",  # type: ignore[arg-type]
                {},
            )
        # Series errors beat weights and budget errors.
        with self.assertRaises(ValueError):
            store.impact_budget(
                "main",
                {"feature": (("m1", "f1"), ("m1", "f1"))},
                {"x": "1"},  # type: ignore[dict-item]
                "1",  # type: ignore[arg-type]
                {},
            )
        with self.assertRaises(ValueError):
            store.impact_budget(
                "main",
                {"main": ()},
                {"x": "1"},  # type: ignore[dict-item]
                "1",  # type: ignore[arg-type]
                {},
            )
        # Reference is looked up before the series branches.
        with self.assertRaises(KeyError):
            store.impact_budget("ghost", {"ghost2": ()}, {"x": 1}, 1, {})
        # Series branches are looked up in insertion order.
        with self.assertRaises(KeyError):
            store.impact_budget(
                "main", {"ghost1": (), "ghost2": ()}, {"x": 1}, 1, {}
            )
        # Nodes of the first series are checked before a later series'
        # branch is looked up; left node before right within a pair.
        with self.assertRaises(KeyError):
            store.impact_budget(
                "main",
                {"feature": (("nope1", "f1"),), "ghost": ()},
                {"x": 1},
                1,
                {},
            )
        with self.assertRaises(KeyError):
            store.impact_budget(
                "main",
                {"feature": (("m1", "nope2"), ("nope1", "f1"))},
                {"x": 1},
                1,
                {},
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
        first = store.impact_budget("main", series, weights, 1, {"x": 0})
        self.assertIsNot(first[0], first[1])
        self.assertIsNot(
            first[0]["attributions"], first[1]["attributions"]
        )
        for row in first:
            row["remaining"] = -1
            row["exceeded"] += ("evil",)  # type: ignore[assignment]
            for attribution in row["attributions"]:  # type: ignore[union-attr]
                attribution["key"] = "evil"
                attribution["left"]["path"] += ("evil",)
        fresh = store.impact_budget("main", series, weights, 1, {"x": 0})
        self.assertEqual(
            fresh, store.impact_budget("main", series, weights, 1, {"x": 0})
        )
        self.assertEqual(fresh[0]["branch"], "feature")
        self.assertEqual(fresh[0]["remaining"], -17)
        self.assertEqual(fresh[0]["exceeded"], ("x",))

    def test_query_is_read_only(self) -> None:
        store = make_store()
        heads = dict(store._heads)
        appends = copy.deepcopy(store._appends)
        merges = copy.deepcopy(store._merges)
        replay_main = store.replay("main")
        replay_feature = store.replay("feature")
        audit = store.audit_log()
        store.impact_budget(
            "main",
            {"feature": (("m2", "f2"), ("m1", "f1"), ("root", "root"))},
            {"x": 2, "y": 3, "z": 1},
            10,
            {"x": 0},
        )
        # A failing call must leave state untouched as well.
        with self.assertRaises(KeyError):
            store.impact_budget(
                "main",
                {"feature": (("m1", "f1"), ("nope", "f2"))},
                {"x": 1},
                1,
                {},
            )
        with self.assertRaises(ValueError):
            store.impact_budget(
                "main", {"main": (("root", "root"),)}, {"x": 1}, 1, {}
            )
        with self.assertRaises(ValueError):
            store.impact_budget(
                "main",
                {"feature": (("m1", "f1"),)},
                {"x": 1},
                float("nan"),
                {},
            )
        self.assertEqual(dict(store._heads), heads)
        self.assertEqual(copy.deepcopy(store._appends), appends)
        self.assertEqual(copy.deepcopy(store._merges), merges)
        self.assertEqual(store.replay("main"), replay_main)
        self.assertEqual(store.replay("feature"), replay_feature)
        self.assertEqual(store.audit_log(), audit)


if __name__ == "__main__":
    unittest.main()
