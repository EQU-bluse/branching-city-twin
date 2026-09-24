import copy
import inspect
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
    # A second plan branch with its own checkpoint axis against main.
    store.create("feature2", "root")
    store.append("feature2", "g0", 5, {"x": 1})
    store.append("feature2", "g1", 6, {"x": 2})
    # A branch whose x change cancels feature's at the final checkpoint.
    store.create("opp", "root")
    store.append("opp", "o1", 7, {"x": 0})
    return store


# At (m2, f2): feature's signed branch diffs are x +1, y +3, z -7.
# At (m2, g1): feature2's signed branch diffs are x +2, y -2, z -7.
# Together the aggregate diffs are x +3, y +1, z -14.
PLANS = {
    "feature": (("m2", "f2"),),
    "feature2": (("m2", "g1"),),
}
PLAN_ONE = {"feature": (("m2", "f2"),)}
WEIGHTS = {"x": 2, "y": 3, "z": 1}


class CombinationBudgetTests(unittest.TestCase):
    def test_public_signature_has_no_defaults(self) -> None:
        parameters = list(
            inspect.signature(BranchStore.combination_budget).parameters.values()
        )
        self.assertEqual(
            [p.name for p in parameters],
            [
                "self",
                "reference",
                "plans",
                "combinations",
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
        result = store.combination_budget(
            "main",
            PLANS,
            {"both": ("feature", "feature2"), "none": ()},
            WEIGHTS,
            100,
            {},
        )
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        row = result[0]
        self.assertEqual(
            list(row),
            [
                "combination",
                "members",
                "breached",
                "first_breach",
                "max_overrun",
                "remaining",
                "over_keys",
                "margins",
                "attributions",
            ],
        )
        self.assertEqual(row["combination"], "both")
        self.assertEqual(row["members"], ("feature", "feature2"))
        self.assertIsInstance(row["breached"], bool)
        self.assertIsInstance(row["margins"], tuple)
        self.assertIsInstance(row["attributions"], tuple)

    def test_signed_aggregate_risk_and_remaining(self) -> None:
        store = make_store()
        row = store.combination_budget(
            "main", PLANS, {"both": ("feature", "feature2")},
            WEIGHTS, 100, {},
        )[0]
        # Aggregate: x +3, y +1, z -14 -> risk 6 + 3 + 14 = 23.
        self.assertFalse(row["breached"])
        self.assertIsNone(row["first_breach"])
        self.assertEqual(row["max_overrun"], 0)
        self.assertIsInstance(row["max_overrun"], int)
        self.assertEqual(row["remaining"], 77)

    def test_total_budget_breach(self) -> None:
        store = make_store()
        row = store.combination_budget(
            "main", PLANS, {"both": ("feature", "feature2")},
            WEIGHTS, 20, {},
        )[0]
        self.assertTrue(row["breached"])
        self.assertEqual(row["first_breach"], 0)
        self.assertEqual(row["max_overrun"], 3)
        self.assertEqual(row["remaining"], -3)
        self.assertEqual(row["over_keys"], ())
        self.assertEqual(row["attributions"], ())

    def test_per_key_breach_uses_absolute_aggregate(self) -> None:
        store = make_store()
        # Total risk 23 < 100, but the x aggregate of 3 exceeds its
        # per-key budget of 2: weighted excess is (3 - 2) * 2 = 2.
        row = store.combination_budget(
            "main", PLANS, {"both": ("feature", "feature2")},
            WEIGHTS, 100, {"x": 2},
        )[0]
        self.assertTrue(row["breached"])
        self.assertEqual(row["first_breach"], 0)
        self.assertEqual(row["max_overrun"], 2)
        self.assertEqual(row["remaining"], 77)
        self.assertEqual(row["over_keys"], ("x",))

    def test_signed_opposition_cancels_in_aggregate(self) -> None:
        store = make_store()
        # feature's x diff is +1, opp's x diff is -1: together 0.
        plans = {
            "feature": (("m2", "f2"),),
            "opp": (("m2", "o1"),),
        }
        row = store.combination_budget(
            "main", plans, {"pair": ("feature", "opp"), "solo": ("opp",)},
            {"x": 2}, 10, {"x": 0},
        )
        pair = next(r for r in row if r["combination"] == "pair")
        solo = next(r for r in row if r["combination"] == "solo")
        self.assertFalse(pair["breached"])
        self.assertEqual(pair["remaining"], 10)
        self.assertEqual(pair["over_keys"], ())
        self.assertTrue(solo["breached"])
        self.assertEqual(solo["over_keys"], ("x",))

    def test_marginal_contributions_can_be_negative(self) -> None:
        store = make_store()
        plans = {
            "feature": (("m2", "f2"),),
            "opp": (("m2", "o1"),),
        }
        # Cancelling pair: full risk 0; either member alone risks 2.
        pair = store.combination_budget(
            "main", plans, {"pair": ("feature", "opp")}, {"x": 2}, 100, {},
        )[0]
        self.assertEqual(pair["margins"], (-2, -2))

        # Additive pair: full risk 23; without feature feature2 alone is
        # 17; without feature2 feature alone is 18.
        additive = store.combination_budget(
            "main", PLANS,
            {"both": ("feature", "feature2")}, WEIGHTS, 100, {},
        )[0]
        self.assertEqual(additive["margins"], (6, 5))

        # A single-member combination's margin equals its own risk.
        solo = store.combination_budget(
            "main", PLAN_ONE, {"solo": ("feature",)}, WEIGHTS, 100, {},
        )[0]
        self.assertEqual(solo["margins"], (18,))

    def test_margins_follow_member_order_not_plans_order(self) -> None:
        store = make_store()
        reversed_plans = {
            "feature2": (("m2", "g1"),),
            "feature": (("m2", "f2"),),
        }
        row = store.combination_budget(
            "main", reversed_plans,
            {"both": ("feature", "feature2")}, WEIGHTS, 100, {},
        )[0]
        # Members stay in combination order; margins align with members.
        self.assertEqual(row["members"], ("feature", "feature2"))
        self.assertEqual(row["margins"], (6, 5))

    def test_attribution_shape_values_and_history(self) -> None:
        store = make_store()
        row = store.combination_budget(
            "main", PLANS, {"both": ("feature", "feature2")},
            WEIGHTS, 100, {"x": 2, "z": 13},
        )[0]
        # z aggregate -14 exceeds 13 (excess 1 weighted), x aggregate 3
        # exceeds 2; over_keys are Unicode sorted at the final checkpoint.
        self.assertEqual(row["over_keys"], ("x", "z"))
        self.assertEqual(len(row["attributions"]), 2)
        by_key = {item["key"]: item for item in row["attributions"]}
        x_item = by_key["x"]
        self.assertEqual(list(x_item), ["key", "aggregate", "members", "history"])
        self.assertEqual(x_item["aggregate"], 3)
        self.assertEqual(x_item["members"], (1, 2))
        self.assertEqual(len(x_item["history"]), 2)
        z_item = by_key["z"]
        self.assertEqual(z_item["aggregate"], -14)
        self.assertEqual(z_item["members"], (-7, -7))

        expected_feature = store.attribute_divergences_at(
            "main", "m2", "feature", "f2", ("x", "z")
        )
        expected_feature2 = store.attribute_divergences_at(
            "main", "m2", "feature2", "g1", ("x", "z")
        )
        expected = {
            "feature": {r["key"]: r for r in expected_feature},
            "feature2": {r["key"]: r for r in expected_feature2},
        }
        for item in (x_item, z_item):
            key = item["key"]
            for member, history in zip(("feature", "feature2"), item["history"]):
                self.assertEqual(list(history), ["key", "fork", "left", "right"])
                self.assertEqual(history, expected[member][key])
                for side in ("left", "right"):
                    self.assertEqual(
                        list(history[side]),
                        ["value", "cause", "path", "affected"],
                    )

    def test_empty_combination_has_no_checkpoints(self) -> None:
        store = make_store()
        row = store.combination_budget(
            "main", PLANS, {"none": ()}, WEIGHTS, 5.5, {"x": 0},
        )[0]
        self.assertFalse(row["breached"])
        self.assertIsNone(row["first_breach"])
        self.assertEqual(row["max_overrun"], 0)
        self.assertIsInstance(row["max_overrun"], int)
        self.assertEqual(row["remaining"], 5.5)
        self.assertEqual(row["over_keys"], ())
        self.assertEqual(row["margins"], ())
        self.assertEqual(row["attributions"], ())

    def test_zero_checkpoint_members_never_breach(self) -> None:
        store = make_store()
        plans = {"feature": (), "feature2": ()}
        row = store.combination_budget(
            "main", plans, {"both": ("feature", "feature2")},
            WEIGHTS, 9, {"x": 0},
        )[0]
        self.assertFalse(row["breached"])
        self.assertIsNone(row["first_breach"])
        self.assertEqual(row["max_overrun"], 0)
        self.assertEqual(row["remaining"], 9)
        self.assertEqual(row["over_keys"], ())
        # No final checkpoint means no per-member detail either.
        self.assertEqual(row["margins"], ())
        self.assertEqual(row["attributions"], ())

    def test_first_breach_skips_early_compliant_points(self) -> None:
        store = make_store()
        plans = {
            "feature": (("m1", "f1"), ("m2", "f2")),
            "feature2": (("m1", "g0"), ("m2", "g1")),
        }
        # Point 0: feature (x +1, y -2) + feature2 (x 0, y -2) ->
        # aggregate x +1, y -4, risk 2 + 12 = 14 (z unseen) -> within 20.
        # Point 1: aggregate x +3, y +1, z -14 -> risk 23, breaches 20.
        row = store.combination_budget(
            "main", plans, {"both": ("feature", "feature2")},
            WEIGHTS, 20, {},
        )[0]
        self.assertTrue(row["breached"])
        self.assertEqual(row["first_breach"], 1)
        self.assertEqual(row["max_overrun"], 3)
        self.assertEqual(row["remaining"], -3)
        # Final margins use the final checkpoint.
        self.assertEqual(row["margins"], (6, 5))

    def test_float_arithmetic_and_zero_signs(self) -> None:
        store = make_store()
        row = store.combination_budget(
            "main", PLANS, {"both": ("feature", "feature2")},
            {"x": 0.5}, 10.0, {},
        )[0]
        self.assertEqual(row["remaining"], 8.5)
        self.assertIsInstance(row["remaining"], float)

        row = store.combination_budget(
            "main", PLAN_ONE, {"solo": ("feature",)},
            {"x": -0.0}, 0.0, {},
        )[0]
        self.assertEqual(row["remaining"], 0.0)
        self.assertGreater(math.copysign(1.0, row["remaining"]), 0)
        self.assertEqual(row["max_overrun"], 0.0)
        self.assertGreater(math.copysign(1.0, row["max_overrun"]), 0)
        self.assertEqual(row["margins"], (0.0,))
        self.assertGreater(math.copysign(1.0, row["margins"][0]), 0)

    def test_rows_sorted_breached_first_overrun_breach_name(self) -> None:
        store = make_store()
        # feature+feature2 risk 23; feature alone risk 18; the two empty
        # combinations never breach and tie-break by Unicode name.
        result = store.combination_budget(
            "main",
            PLANS,
            {
                "calm2": (),
                "zeta": ("feature",),
                "alpha": ("feature", "feature2"),
                "calm": (),
            },
            WEIGHTS,
            0,
            {},
        )
        names = [row["combination"] for row in result]
        self.assertEqual(names, ["alpha", "zeta", "calm", "calm2"])
        self.assertEqual(
            [
                (row["breached"], row["first_breach"], row["max_overrun"])
                for row in result
            ],
            [
                (True, 0, 23),
                (True, 0, 18),
                (False, None, 0),
                (False, None, 0),
            ],
        )

    def test_equal_overrun_orders_by_first_breach_ascending(self) -> None:
        # A two-checkpoint combination whose aggregate is 0 at point 0 and
        # -6 at point 1, versus a one-checkpoint combination with a -6
        # aggregate at point 0: equal max_overrun of 3 against a budget
        # of 3, so the earlier first breach wins regardless of insertion
        # order.
        graph = EventGraph()
        graph.add("root", 0, (), {})
        store = BranchStore(graph)
        store.create("ref", "root")
        store.create("p", "root")
        store.create("q", "root")
        store.create("r", "root")
        store.create("u", "root")
        store.append("ref", "r1", 1, {"x": 10})
        store.append("p", "p1", 2, {"x": 12})
        store.append("q", "q1", 2, {"x": 8})
        store.append("ref", "r2", 3, {"x": 10})
        store.append("p", "p2", 4, {"x": 6})
        store.append("q", "q2", 4, {"x": 8})
        store.append("r", "t1", 4, {"x": 14})
        store.append("u", "u1", 4, {"x": 20})
        # Cumulative at point 0: ref 10, p 12, q 8 -> aggregate 0.
        # Point 1: ref 20, p 18, q 16 -> aggregate -6.
        # Early: 14 - 20 = -6; calm: 20 - 20 = 0.
        plans = {
            "p": (("r1", "p1"), ("r2", "p2")),
            "q": (("r1", "q1"), ("r2", "q2")),
            "r": (("r2", "t1"),),
            "u": (("r2", "u1"),),
        }
        result = store.combination_budget(
            "ref",
            plans,
            {"late": ("p", "q"), "calm": ("u",), "early": ("r",)},
            {"x": 1},
            3,
            {},
        )
        self.assertEqual(
            [(row["combination"], row["first_breach"], row["max_overrun"])
             for row in result],
            [("early", 0, 3), ("late", 1, 3), ("calm", None, 0)],
        )

    def test_validation_type_errors(self) -> None:
        store = make_store()
        with self.assertRaises(TypeError):
            store.combination_budget(None, PLANS, {"none": ()}, WEIGHTS, 5, {})
        with self.assertRaises(TypeError):
            store.combination_budget("main", [], {"none": ()}, WEIGHTS, 5, {})
        with self.assertRaises(TypeError):
            store.combination_budget("main", PLANS, [], WEIGHTS, 5, {})
        with self.assertRaises(TypeError):
            store.combination_budget(
                "main", PLANS, {"both": ["feature"]}, WEIGHTS, 5, {}
            )
        with self.assertRaises(TypeError):
            store.combination_budget(
                "main", PLANS, {1: ("feature",)}, WEIGHTS, 5, {}
            )
        with self.assertRaises(TypeError):
            store.combination_budget(
                "main", PLANS, {"both": (1,)}, WEIGHTS, 5, {}
            )
        with self.assertRaises(TypeError):
            store.combination_budget(
                "main", PLANS, {"none": ()}, [], 5, {}
            )
        with self.assertRaises(TypeError):
            store.combination_budget(
                "main", PLANS, {"none": ()}, WEIGHTS, True, {}
            )
        with self.assertRaises(TypeError):
            store.combination_budget(
                "main", PLANS, {"none": ()}, WEIGHTS, 5, []
            )

    def test_validation_value_errors(self) -> None:
        store = make_store()
        with self.assertRaises(ValueError):
            store.combination_budget("", PLANS, {"none": ()}, WEIGHTS, 5, {})
        with self.assertRaises(ValueError):
            store.combination_budget("main", PLANS, {"": ()}, WEIGHTS, 5, {})
        with self.assertRaises(ValueError):
            store.combination_budget(
                "main", PLANS, {"both": ("",)}, WEIGHTS, 5, {}
            )
        with self.assertRaises(ValueError):
            store.combination_budget(
                "main", PLANS,
                {"both": ("feature", "feature")}, WEIGHTS, 5, {},
            )
        with self.assertRaises(ValueError):
            store.combination_budget(
                "main", PLANS,
                {"both": ("feature", "ghost")}, WEIGHTS, 5, {},
            )
        with self.assertRaises(ValueError):
            store.combination_budget(
                "main", {"main": ()}, {"solo": ("main",)}, WEIGHTS, 5, {}
            )
        with self.assertRaises(ValueError):
            # Repeated node pair within a plan keeps impact_budget's rule.
            store.combination_budget(
                "main",
                {"feature": (("m2", "f2"), ("m2", "f2"))},
                {"solo": ("feature",)},
                WEIGHTS, 5, {},
            )
        with self.assertRaises(ValueError):
            store.combination_budget(
                "main", PLANS, {"none": ()}, WEIGHTS, -1, {}
            )
        with self.assertRaises(ValueError):
            store.combination_budget(
                "main", PLANS, {"none": ()}, WEIGHTS, 5, {"w": 0}
            )

    def test_misaligned_checkpoints_raise_value_error(self) -> None:
        store = make_store()
        plans_counts = {
            "feature": (("m1", "f1"), ("m2", "f2")),
            "feature2": (("m2", "g1"),),
        }
        with self.assertRaises(ValueError):
            store.combination_budget(
                "main", plans_counts,
                {"both": ("feature", "feature2")}, WEIGHTS, 5, {},
            )
        plans_nodes = {
            "feature": (("m1", "f1"), ("m2", "f2")),
            "feature2": (("m2", "g1"), ("m2", "g1")),
        }
        with self.assertRaises(ValueError):
            store.combination_budget(
                "main", plans_nodes,
                {"both": ("feature", "feature2")}, WEIGHTS, 5, {},
            )
        # Equal counts with matching reference nodes are accepted.
        plans_ok = {
            "feature": (("m1", "f1"), ("m2", "f2")),
            "feature2": (("m1", "g0"), ("m2", "g1")),
        }
        self.assertEqual(
            len(
                store.combination_budget(
                    "main", plans_ok,
                    {"both": ("feature", "feature2")}, WEIGHTS, 1000, {},
                )
            ),
            1,
        )

    def test_all_inputs_validated_before_branch_lookups(self) -> None:
        store = make_store()
        # Bad budget beats unknown branches and nodes.
        with self.assertRaises(TypeError):
            store.combination_budget(
                "ghost",
                {"ghost2": (("m1", "f1"),)},
                {"solo": ("ghost2",)},
                WEIGHTS, True, {},
            )
        # Combination errors beat budget errors.
        with self.assertRaises(ValueError):
            store.combination_budget(
                "main", PLANS, {"both": ("ghost",)}, WEIGHTS, -1, {}
            )
        # Reference is looked up before any plan branch.
        with self.assertRaises(KeyError) as caught:
            store.combination_budget(
                "ghost", PLANS, {"solo": ("feature",)}, WEIGHTS, 5, {}
            )
        self.assertEqual(caught.exception.args, ("ghost",))

    def test_only_member_plans_are_consulted_in_plans_order(self) -> None:
        store = make_store()
        # feature2 is a member and unknown nodes on it are checked;
        # "ghost" is listed in plans but never combined, so its branch
        # and nodes must never be consulted.
        plans = {
            "feature": (("m2", "f2"),),
            "ghost": (("m2", "nope"),),
            "feature2": (("m2", "g1"),),
        }
        result = store.combination_budget(
            "main", plans,
            {"both": ("feature", "feature2"), "none": ()},
            WEIGHTS, 100, {},
        )
        self.assertEqual([r["combination"] for r in result], ["both", "none"])

        with self.assertRaises(KeyError) as caught:
            store.combination_budget(
                "main",
                {"ghost": (("m2", "f2"),)},
                {"solo": ("ghost",)},
                WEIGHTS, 5, {},
            )
        self.assertEqual(caught.exception.args, ("ghost",))

        # Member plan branches are looked up in plans insertion order.
        plans_two = {
            "ghost1": (("m2", "f2"),),
            "feature": (("m2", "f2"),),
        }
        with self.assertRaises(KeyError) as caught:
            store.combination_budget(
                "main", plans_two,
                {"both": ("ghost1", "feature")}, WEIGHTS, 5, {},
            )
        self.assertEqual(caught.exception.args, ("ghost1",))

    def test_node_closure_errors_left_before_right(self) -> None:
        store = make_store()
        with self.assertRaises(KeyError) as caught:
            store.combination_budget(
                "main",
                {"feature": (("nope1", "f2"),)},
                {"solo": ("feature",)},
                WEIGHTS, 5, {},
            )
        self.assertEqual(caught.exception.args, ("nope1",))
        with self.assertRaises(KeyError) as caught:
            store.combination_budget(
                "main",
                {"feature": (("m2", "nope2"),)},
                {"solo": ("feature",)},
                WEIGHTS, 5, {},
            )
        self.assertEqual(caught.exception.args, ("nope2",))
        # A series node outside the reference head closure, and a plan
        # node outside the plan's head closure, both raise KeyError.
        with self.assertRaises(KeyError):
            store.combination_budget(
                "main",
                {"feature": (("f2", "f2"),)},
                {"solo": ("feature",)},
                WEIGHTS, 5, {},
            )
        with self.assertRaises(KeyError):
            store.combination_budget(
                "main",
                {"feature": (("m2", "m2"),)},
                {"solo": ("feature",)},
                WEIGHTS, 5, {},
            )

    def test_empty_combinations_still_validates_and_checks_reference(self) -> None:
        store = make_store()
        self.assertEqual(
            store.combination_budget("main", PLANS, {}, WEIGHTS, 5, {}), ()
        )
        with self.assertRaises(KeyError):
            store.combination_budget("ghost", PLANS, {}, WEIGHTS, 5, {})
        with self.assertRaises(ValueError):
            store.combination_budget("main", PLANS, {}, WEIGHTS, -1, {})
        with self.assertRaises(TypeError):
            store.combination_budget("main", PLANS, None, WEIGHTS, 5, {})

    def test_empty_weights_still_checks_members_branches_and_nodes(self) -> None:
        store = make_store()
        result = store.combination_budget(
            "main", PLANS,
            {"both": ("feature", "feature2"), "none": ()},
            {}, 0, {},
        )
        for row in result:
            self.assertFalse(row["breached"])
            self.assertEqual(row["remaining"], 0)
        with self.assertRaises(KeyError):
            store.combination_budget(
                "main", {"ghost": ()}, {"solo": ("ghost",)}, {}, 0, {}
            )
        with self.assertRaises(KeyError):
            store.combination_budget(
                "main",
                {"feature": (("nope", "f2"),)},
                {"solo": ("feature",)},
                {}, 0, {},
            )
        with self.assertRaises(ValueError):
            store.combination_budget(
                "main", PLANS, {"none": ()}, {}, 0, {"x": 0}
            )

    def test_results_are_detached_and_unshared(self) -> None:
        store = make_store()
        combinations = {
            "both": ("feature", "feature2"),
            "solo": ("feature",),
        }
        first = store.combination_budget(
            "main", PLANS, combinations, WEIGHTS, 0, {"x": 0, "z": 0}
        )
        self.assertIsNot(first[0], first[1])
        self.assertIsNot(first[0]["attributions"], first[1]["attributions"])
        self.assertIsNot(
            first[0]["attributions"][0]["history"],
            first[0]["attributions"][1]["history"],
        )
        pristine = copy.deepcopy(first)
        for row in first:
            row["combination"] = "evil"
            row["members"] += ("evil",)
            row["over_keys"] += ("evil",)
            row["margins"] += (-999,)
            for item in row["attributions"]:
                item["aggregate"] = -999
                item["members"] += (-999,)
                for history in item["history"]:
                    history["key"] = "evil"
                    history["left"]["path"] += ("evil",)
        fresh = store.combination_budget(
            "main", PLANS, combinations, WEIGHTS, 0, {"x": 0, "z": 0}
        )
        self.assertEqual(fresh, pristine)

    def test_history_is_independent_from_live_divergence_queries(self) -> None:
        store = make_store()
        row = store.combination_budget(
            "main", PLANS, {"both": ("feature", "feature2")},
            WEIGHTS, 0, {"x": 0},
        )[0]
        history = row["attributions"][0]["history"]
        expected = store.attribute_divergences_at(
            "main", "m2", "feature", "f2", ("x",)
        )[0]
        self.assertEqual(history[0], expected)
        self.assertIsNot(history[0], expected)
        self.assertIsNot(history[0]["left"], expected["left"])
        self.assertIsNot(history[0]["left"]["path"], expected["left"]["path"])

    def test_query_is_read_only(self) -> None:
        store = make_store()
        heads = dict(store._heads)
        appends = copy.deepcopy(store._appends)
        merges = copy.deepcopy(store._merges)
        replay = {name: store.replay(name) for name in heads}
        audit = store.audit_log()
        matrix = store.divergence_matrix(
            "main", {"feature": (("m2", "f2"), ("m1", "f1"))}, ("z", "x")
        )
        budget = store.impact_budget(
            "main", {"feature": (("m2", "f2"),)}, {"x": 1}, 5, {}
        )
        store.combination_budget(
            "main",
            PLANS,
            {
                "none": (),
                "both": ("feature", "feature2"),
                "solo": ("feature",),
            },
            WEIGHTS,
            10,
            {"x": 0},
        )
        # Failing calls must leave state untouched as well.
        with self.assertRaises(KeyError):
            store.combination_budget(
                "main",
                {"feature": (("m2", "nope"),)},
                {"bad": ("feature",)}, WEIGHTS, 10, {},
            )
        with self.assertRaises(ValueError):
            store.combination_budget(
                "main", PLANS,
                {"bad": ("ghost",)}, WEIGHTS, 10, {},
            )
        with self.assertRaises(ValueError):
            store.combination_budget(
                "main", PLANS, {"dup": ("feature", "feature")},
                WEIGHTS, 10, {},
            )
        with self.assertRaises(ValueError):
            store.combination_budget(
                "main", PLANS, {"none": ()}, WEIGHTS, float("nan"), {}
            )
        self.assertEqual(dict(store._heads), heads)
        self.assertEqual(copy.deepcopy(store._appends), appends)
        self.assertEqual(copy.deepcopy(store._merges), merges)
        for name, state in replay.items():
            self.assertEqual(store.replay(name), state)
        self.assertEqual(store.audit_log(), audit)
        self.assertEqual(
            store.divergence_matrix(
                "main", {"feature": (("m2", "f2"), ("m1", "f1"))},
                ("z", "x"),
            ),
            matrix,
        )
        self.assertEqual(
            store.impact_budget(
                "main", {"feature": (("m2", "f2"),)}, {"x": 1}, 5, {}
            ),
            budget,
        )


if __name__ == "__main__":
    unittest.main()
