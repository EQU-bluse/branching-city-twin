import copy
import inspect
import math
import subprocess
import sys
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
    def test_public_signature_has_no_defaults(self) -> None:
        parameters = list(
            inspect.signature(BranchStore.impact_budget).parameters.values()
        )
        self.assertEqual(
            [p.name for p in parameters],
            [
                "self",
                "reference",
                "series",
                "weights",
                "total_budget",
                "key_budgets",
            ],
        )
        self.assertTrue(
            all(p.default is inspect.Parameter.empty for p in parameters)
        )

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
                "max_overrun",
                "remaining",
                "over_keys",
                "attributions",
            ],
        )
        self.assertEqual(row["branch"], "feature")
        self.assertIsInstance(row["breached"], bool)
        self.assertIsInstance(row["over_keys"], tuple)
        self.assertIsInstance(row["attributions"], tuple)

    def test_within_budget_values(self) -> None:
        store = make_store()
        # At (m2, f2): risk is 1*2 + 3*3 + 7*1 = 18.
        row = store.impact_budget(
            "main",
            {"feature": (("m2", "f2"),)},
            {"x": 2, "y": 3, "z": 1},
            18,
            {"x": 1, "y": 3},
        )[0]
        self.assertFalse(row["breached"])
        self.assertIsNone(row["first_breach"])
        self.assertEqual(row["max_overrun"], 0)
        self.assertIsInstance(row["max_overrun"], int)
        self.assertEqual(row["remaining"], 0)
        self.assertEqual(row["over_keys"], ())
        self.assertEqual(row["attributions"], ())

    def test_total_budget_breach(self) -> None:
        store = make_store()
        # Risk 18 against a budget of 10: overrun 8, remaining -8.
        row = store.impact_budget(
            "main",
            {"feature": (("m2", "f2"),)},
            {"x": 2, "y": 3, "z": 1},
            10,
            {},
        )[0]
        self.assertTrue(row["breached"])
        self.assertEqual(row["first_breach"], 0)
        self.assertEqual(row["max_overrun"], 8)
        self.assertEqual(row["remaining"], -8)
        self.assertEqual(row["over_keys"], ())
        self.assertEqual(row["attributions"], ())

    def test_per_key_breach_is_independent(self) -> None:
        store = make_store()
        # With a zero x weight the total risk stays at 0 < 100, but the x
        # difference of 1 exceeds its per-key budget of 0.
        row = store.impact_budget(
            "main",
            {"feature": (("m2", "f2"),)},
            {"x": 0},
            100,
            {"x": 0},
        )[0]
        self.assertTrue(row["breached"])
        self.assertEqual(row["first_breach"], 0)
        # A zero weight means the weighted excess is zero too.
        self.assertEqual(row["max_overrun"], 0)
        self.assertEqual(row["remaining"], 100)
        self.assertEqual(row["over_keys"], ("x",))
        self.assertEqual(len(row["attributions"]), 1)

    def test_max_overrun_takes_weighted_key_excess(self) -> None:
        store = make_store()
        # Risk 18 < total budget 100, but y differs by 3 against a per-key
        # budget of 1: weighted excess is (3 - 1) * 3 = 6.
        row = store.impact_budget(
            "main",
            {"feature": (("m2", "f2"),)},
            {"x": 2, "y": 3, "z": 1},
            100,
            {"y": 1},
        )[0]
        self.assertTrue(row["breached"])
        self.assertEqual(row["max_overrun"], 6)
        self.assertEqual(row["remaining"], 82)
        self.assertEqual(row["over_keys"], ("y",))

    def test_max_overrun_is_largest_across_checkpoints(self) -> None:
        store = make_store()
        # Point 0 (m1, f1): x differs by 1, y by 2, z equal -> risk 8,
        # still within the budget of 10; point 1 breaches with risk 18.
        row = store.impact_budget(
            "main",
            {"feature": (("m1", "f1"), ("m2", "f2"))},
            {"x": 2, "y": 3, "z": 1},
            10,
            {},
        )[0]
        self.assertTrue(row["breached"])
        self.assertEqual(row["first_breach"], 1)
        self.assertEqual(row["max_overrun"], 8)
        self.assertEqual(row["remaining"], -8)

    def test_first_breach_skips_early_compliant_points(self) -> None:
        store = make_store()
        row = store.impact_budget(
            "main",
            {"feature": (("m1", "f1"), ("m2", "f2"))},
            {"x": 2, "y": 3, "z": 1},
            10,
            {"z": 6},
        )[0]
        # Point 0 has risk 8 < 10 and no z divergence; point 1 breaches
        # both the total budget (18 > 10) and z's per-key budget (7 > 6).
        self.assertTrue(row["breached"])
        self.assertEqual(row["first_breach"], 1)
        self.assertEqual(row["max_overrun"], 8)
        self.assertEqual(row["over_keys"], ("z",))

    def test_float_arithmetic_and_zero_signs(self) -> None:
        store = make_store()
        row = store.impact_budget(
            "main",
            {"feature": (("m2", "f2"),)},
            {"x": 0.5},
            1.0,
            {},
        )[0]
        self.assertEqual(row["remaining"], 0.5)
        self.assertIsInstance(row["remaining"], float)
        self.assertEqual(row["max_overrun"], 0)
        self.assertIsInstance(row["max_overrun"], int)
        # Negative zero weights keep the zero risk an int; a float zero
        # remaining must never carry a negative sign.
        row = store.impact_budget(
            "main",
            {"feature": (("m2", "f2"),)},
            {"x": -0.0},
            0.0,
            {},
        )[0]
        self.assertFalse(row["breached"])
        self.assertEqual(row["remaining"], 0.0)
        self.assertIsInstance(row["remaining"], float)
        self.assertGreater(math.copysign(1.0, row["remaining"]), 0)
        self.assertEqual(row["max_overrun"], 0.0)
        self.assertGreater(math.copysign(1.0, row["max_overrun"]), 0)

    def test_over_keys_come_from_final_checkpoint_only(self) -> None:
        store = make_store()
        # Point 0 breaches z (7 > 6) on the left side against feature's 0;
        # point 1 has no per-key breach even though the total still exceeds.
        row = store.impact_budget(
            "main",
            {"feature": (("m2", "f2"), ("m1", "f1"))},
            {"x": 2, "y": 3, "z": 1},
            100,
            {"z": 6},
        )[0]
        self.assertTrue(row["breached"])
        self.assertEqual(row["first_breach"], 0)
        self.assertEqual(row["over_keys"], ())
        self.assertEqual(row["attributions"], ())

    def test_over_keys_sorted_by_unicode_code_point(self) -> None:
        store = make_store()
        row = store.impact_budget(
            "main",
            {"feature": (("m2", "f2"),)},
            {"z": 1, "y": 1, "x": 1},
            0,
            {"z": 0, "x": 0},
        )[0]
        self.assertEqual(row["over_keys"], ("x", "z"))
        self.assertEqual(
            [a["key"] for a in row["attributions"]], ["x", "z"]
        )

    def test_attributions_match_historical_divergence(self) -> None:
        store = make_store()
        expected = store.attribute_divergences_at(
            "main", "m2", "feature", "f2", ("z", "y", "x")
        )
        expected_by_key = {record["key"]: record for record in expected}
        row = store.impact_budget(
            "main",
            {"feature": (("m1", "f1"), ("m2", "f2"))},
            {"z": 1, "y": 1, "x": 1},
            0,
            {"z": 0, "x": 0},
        )[0]
        self.assertEqual(row["over_keys"], ("x", "z"))
        for key, attribution in zip(row["over_keys"], row["attributions"]):
            self.assertEqual(list(attribution), ["key", "fork", "left", "right"])
            self.assertEqual(attribution, expected_by_key[key])
            for side in ("left", "right"):
                self.assertEqual(
                    list(attribution[side]),
                    ["value", "cause", "path", "affected"],
                )

    def test_rows_sorted_breached_first_then_overrun_breach_name(self) -> None:
        store = make_store()
        store.create("late", "root")
        store.append("late", "l1", 5, {"x": 3})
        store.create("early", "root")
        store.append("early", "e1", 6, {"x": 3})
        store.create("calm", "root")
        store.create("calm2", "root")
        # Budget 0: any x divergence breaches by its weighted value.
        result = store.impact_budget(
            "main",
            {
                "calm2": (("root", "root"),),
                "late": (("root", "root"), ("m1", "l1")),
                "early": (("m1", "e1"),),
                "feature": (("m2", "f2"),),
                "calm": (("root", "root"),),
            },
            {"x": 1},
            0,
            {},
        )
        # early and late both overrun 2: breach index 0 beats 1; feature
        # overruns 1; calm/calm2 never breach and sort by name with None
        # after every integer.
        self.assertEqual(
            [row["branch"] for row in result],
            ["early", "late", "feature", "calm", "calm2"],
        )
        self.assertEqual(
            [(row["breached"], row["first_breach"], row["max_overrun"]) for row in result],
            [(True, 0, 2), (True, 1, 2), (True, 0, 1), (False, None, 0), (False, None, 0)],
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
            1,
            {},
        )
        self.assertEqual(
            [row["branch"] for row in result], ["a-branch", "b-branch"]
        )

    def test_empty_series_still_validates_budgets_and_reference(self) -> None:
        store = make_store()
        self.assertEqual(store.impact_budget("main", {}, {"x": 1}, 5, {}), ())
        self.assertEqual(store.impact_budget("main", {}, {}, 0, {}), ())
        with self.assertRaises(KeyError):
            store.impact_budget("ghost", {}, {"x": 1}, 5, {})
        with self.assertRaises(ValueError):
            store.impact_budget("main", {}, {"x": 1}, -1, {})
        with self.assertRaises(ValueError):
            store.impact_budget("main", {}, {"x": 1}, 5, {"z": 1})
        with self.assertRaises(TypeError):
            store.impact_budget(None, {}, {"x": 1}, 5, {})
        with self.assertRaises(TypeError):
            store.impact_budget("main", {}, {"x": 1}, True, {})
        with self.assertRaises(TypeError):
            store.impact_budget("main", {}, {"x": 1}, 5, ())

    def test_empty_weights_zero_risk_and_empty_key_budgets(self) -> None:
        store = make_store()
        result = store.impact_budget(
            "main", {"feature": (("m2", "f2"),)}, {}, 0, {}
        )
        row = result[0]
        self.assertFalse(row["breached"])
        self.assertIsNone(row["first_breach"])
        self.assertEqual(row["max_overrun"], 0)
        self.assertEqual(row["remaining"], 0)
        self.assertEqual(row["over_keys"], ())
        # Every branch and node is still validated.
        with self.assertRaises(KeyError):
            store.impact_budget("main", {"ghost": ()}, {}, 0, {})
        with self.assertRaises(KeyError):
            store.impact_budget(
                "main", {"feature": (("nope", "f1"),)}, {}, 0, {}
            )
        with self.assertRaises(KeyError):
            store.impact_budget(
                "main", {"feature": (("m1", "nope"),)}, {}, 0, {}
            )
        with self.assertRaises(KeyError):
            store.impact_budget(
                "main", {"feature": (("f2", "m1"),)}, {}, 0, {}
            )
        # A per-key budget with no weights to back it is still rejected.
        with self.assertRaises(ValueError):
            store.impact_budget(
                "main", {"feature": (("m2", "f2"),)}, {}, 0, {"x": 0}
            )

    def test_no_checkpoints_never_breaches(self) -> None:
        store = make_store()
        row = store.impact_budget(
            "main", {"feature": ()}, {"x": 1}, 5.5, {}
        )[0]
        self.assertFalse(row["breached"])
        self.assertIsNone(row["first_breach"])
        self.assertEqual(row["max_overrun"], 0)
        self.assertEqual(row["remaining"], 5.5)
        self.assertEqual(row["over_keys"], ())
        self.assertEqual(row["attributions"], ())

    def test_total_budget_validation(self) -> None:
        store = make_store()
        for bad in (True, False, "1", None, (1,), [1], {"x": 1}):
            with self.subTest(value=repr(bad)):
                with self.assertRaises(TypeError):
                    store.impact_budget(
                        "main",
                        {"feature": (("m1", "f1"),)},
                        {"x": 1},
                        bad,
                        {},
                    )
        for bad in (-1, -0.5, float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=repr(bad)):
                with self.assertRaises(ValueError):
                    store.impact_budget(
                        "main",
                        {"feature": (("m1", "f1"),)},
                        {"x": 1},
                        bad,
                        {},
                    )

    def test_key_budgets_validation(self) -> None:
        store = make_store()
        for bad in ([], None, (), (("x", 1),), "x", 1):
            with self.subTest(container=repr(bad)):
                with self.assertRaises(TypeError):
                    store.impact_budget(
                        "main",
                        {"feature": (("m1", "f1"),)},
                        {"x": 1},
                        5,
                        bad,
                    )
        for bad_key in (None, 1, 1.5, b"x", ("x",)):
            with self.subTest(key=repr(bad_key)):
                with self.assertRaises(TypeError):
                    store.impact_budget(
                        "main",
                        {"feature": (("m1", "f1"),)},
                        {"x": 1},
                        5,
                        {bad_key: 1},
                    )
        with self.assertRaises(ValueError):
            store.impact_budget(
                "main",
                {"feature": (("m1", "f1"),)},
                {"x": 1},
                5,
                {"": 1},
            )
        for bad in (True, False, "1", None, (1,), [1]):
            with self.subTest(value=repr(bad)):
                with self.assertRaises(TypeError):
                    store.impact_budget(
                        "main",
                        {"feature": (("m1", "f1"),)},
                        {"x": 1},
                        5,
                        {"x": bad},
                    )
        for bad in (-1, -0.5, float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=repr(bad)):
                with self.assertRaises(ValueError):
                    store.impact_budget(
                        "main",
                        {"feature": (("m1", "f1"),)},
                        {"x": 1},
                        5,
                        {"x": bad},
                    )
        with self.assertRaises(ValueError):
            store.impact_budget(
                "main",
                {"feature": (("m1", "f1"),)},
                {"x": 1},
                5,
                {"z": 1},
            )

    def test_all_inputs_validated_before_branch_lookups(self) -> None:
        store = make_store()
        # Bad budgets beat unknown branches and nodes.
        with self.assertRaises(TypeError):
            store.impact_budget(
                "ghost",
                {"ghost2": (("m1", "f1"),)},
                {"x": 1},
                True,
                {},
            )
        with self.assertRaises(ValueError):
            store.impact_budget(
                "ghost",
                {"ghost2": (("m1", "f1"),)},
                {"x": 1},
                5,
                {"z": 1},
            )
        # Series errors beat budget errors.
        with self.assertRaises(ValueError):
            store.impact_budget(
                "main",
                {"feature": (("m1", "f1"), ("m1", "f1"))},
                {"x": 1},
                True,
                {},
            )
        with self.assertRaises(ValueError):
            store.impact_budget(
                "main", {"main": ()}, {"x": 1}, True, {}
            )
        # Weights errors beat the total budget.
        with self.assertRaises(TypeError):
            store.impact_budget(
                "main",
                {"feature": (("m1", "f1"),)},
                {"x": "1"},
                True,
                {},
            )
        # Reference is looked up before the series branches.
        with self.assertRaises(KeyError) as caught:
            store.impact_budget("ghost", {"ghost2": ()}, {"x": 1}, 5, {})
        self.assertEqual(caught.exception.args, ("ghost",))
        with self.assertRaises(KeyError) as caught:
            store.impact_budget(
                "main", {"ghost1": (), "ghost2": ()}, {"x": 1}, 5, {}
            )
        self.assertEqual(caught.exception.args, ("ghost1",))
        # Nodes of the first series are checked before a later series'
        # branch is looked up; left node before right within a pair.
        with self.assertRaises(KeyError) as caught:
            store.impact_budget(
                "main",
                {"feature": (("nope1", "f1"),), "ghost": ()},
                {"x": 1},
                5,
                {},
            )
        self.assertEqual(caught.exception.args, ("nope1",))
        with self.assertRaises(KeyError) as caught:
            store.impact_budget(
                "main",
                {"feature": (("m1", "nope2"), ("nope1", "f1"))},
                {"x": 1},
                5,
                {},
            )
        self.assertEqual(caught.exception.args, ("nope2",))

    def test_results_are_detached_and_unshared(self) -> None:
        store = make_store()
        store.create("side", "root")
        store.append("side", "s1", 5, {"x": 1})
        series = {
            "feature": (("m2", "f2"),),
            "side": (("m2", "s1"),),
        }
        weights = {"x": 2, "y": 3, "z": 1}
        first = store.impact_budget(
            "main", series, weights, 0, {"x": 0, "z": 0}
        )
        self.assertIsNot(first[0], first[1])
        self.assertIsNot(first[0]["attributions"], first[1]["attributions"])
        pristine = copy.deepcopy(first)
        for row in first:
            row["remaining"] = -999
            row["over_keys"] += ("evil",)
            for attribution in row["attributions"]:
                attribution["key"] = "evil"
                attribution["left"]["path"] += ("evil",)
        fresh = store.impact_budget("main", series, weights, 0, {"x": 0, "z": 0})
        self.assertEqual(fresh, pristine)
        self.assertEqual(fresh[0]["branch"], "feature")
        self.assertEqual(fresh[0]["over_keys"], ("x", "z"))

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
        # Failing calls must leave state untouched as well.
        with self.assertRaises(KeyError):
            store.impact_budget(
                "main",
                {"feature": (("m1", "f1"), ("nope", "f2"))},
                {"x": 1},
                10,
                {},
            )
        with self.assertRaises(ValueError):
            store.impact_budget(
                "main", {"main": (("root", "root"),)}, {"x": 1}, 10, {}
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
        # The existing divergence interface keeps its previous results.
        matrix = store.divergence_matrix(
            "main",
            {"feature": (("m2", "f2"), ("m1", "f1"))},
            ("z", "x"),
        )
        store.impact_budget(
            "main",
            {"feature": (("m2", "f2"), ("m1", "f1"))},
            {"z": 1, "x": 1},
            0,
            {"x": 0},
        )
        self.assertEqual(
            store.divergence_matrix(
                "main",
                {"feature": (("m2", "f2"), ("m1", "f1"))},
                ("z", "x"),
            ),
            matrix,
        )

    def test_status_command_keeps_its_output(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-m", "city_twin", "status"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0)
        self.assertEqual(completed.stderr, "")
        self.assertEqual(
            completed.stdout.splitlines(),
            ['{"service": "branching-city-twin", "status": "ready", "version": "0.1.0"}'],
        )


if __name__ == "__main__":
    unittest.main()
