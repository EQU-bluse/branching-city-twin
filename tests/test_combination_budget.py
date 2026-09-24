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
    store.create("side", "root")
    store.append("main", "m1", 1, {"x": 1, "y": 2})
    store.append("feature", "f1", 2, {"x": 2})
    store.append("feature", "f2", 3, {"y": 5})
    store.append("main", "m2", 4, {"z": 7})
    store.append("side", "s1", 5, {"x": 3, "y": -1})
    store.append("side", "s2", 6, {"z": 4})
    return store


def make_series() -> dict:
    return {
        "feature": (("m1", "f1"), ("m2", "f2")),
        "side": (("m1", "s1"), ("m2", "s2")),
    }


class CombinationBudgetTests(unittest.TestCase):
    def test_public_signature_has_no_defaults(self) -> None:
        parameters = list(
            inspect.signature(
                BranchStore.combination_budget
            ).parameters.values()
        )
        self.assertEqual(
            [p.name for p in parameters],
            [
                "self",
                "reference",
                "series",
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
            make_series(),
            {"both": ("feature", "side")},
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
                "combination",
                "members",
                "breached",
                "first_breach",
                "max_overrun",
                "remaining",
                "over_keys",
                "contributions",
                "attributions",
            ],
        )
        self.assertEqual(row["combination"], "both")
        self.assertEqual(row["members"], ("feature", "side"))
        self.assertIsInstance(row["breached"], bool)
        self.assertIsInstance(row["over_keys"], tuple)
        self.assertIsInstance(row["contributions"], tuple)
        self.assertIsInstance(row["attributions"], tuple)

    def test_aggregate_risk_and_breach(self) -> None:
        store = make_store()
        # Point 0 (ref m1): diffs feature (1, -2, 0), side (2, -3, 0),
        # aggregate (3, -5, 0) -> risk 3*2 + 5*3 = 21 > 20, overrun 1.
        # Point 1 (ref m2): aggregate (3, 0, -10) -> risk 6 + 10 = 16.
        row = store.combination_budget(
            "main",
            make_series(),
            {"both": ("feature", "side")},
            {"x": 2, "y": 3, "z": 1},
            20,
            {},
        )[0]
        self.assertTrue(row["breached"])
        self.assertEqual(row["first_breach"], 0)
        self.assertEqual(row["max_overrun"], 1)
        self.assertEqual(row["remaining"], 4)
        self.assertEqual(row["over_keys"], ())
        self.assertEqual(row["attributions"], ())

    def test_contributions_can_be_negative_and_follow_member_order(self) -> None:
        store = make_store()
        # Final checkpoint risk is 16. Without feature the aggregate is
        # side's (2, -3, -3) -> risk 16, contribution 0; without side it
        # is feature's (1, 3, -7) -> risk 18, contribution 16 - 18 = -2.
        row = store.combination_budget(
            "main",
            make_series(),
            {"both": ("feature", "side")},
            {"x": 2, "y": 3, "z": 1},
            20,
            {},
        )[0]
        self.assertEqual(row["contributions"], (0, -2))
        # Member order is preserved in both members and contributions.
        row = store.combination_budget(
            "main",
            make_series(),
            {"rev": ("side", "feature")},
            {"x": 2, "y": 3, "z": 1},
            20,
            {},
        )[0]
        self.assertEqual(row["members"], ("side", "feature"))
        self.assertEqual(row["contributions"], (-2, 0))

    def test_signed_differences_cancel_across_members(self) -> None:
        store = make_store()
        # At the final checkpoint the y differences (+3 and -3) cancel, so
        # the combination's y risk is zero even though each member alone
        # diverges.
        row = store.combination_budget(
            "main",
            make_series(),
            {"both": ("feature", "side")},
            {"y": 1},
            0,
            {},
        )[0]
        # Point 0 aggregates y to -5 (risk 5 > 0); point 1 has zero risk.
        self.assertTrue(row["breached"])
        self.assertEqual(row["first_breach"], 0)
        self.assertEqual(row["max_overrun"], 5)
        self.assertEqual(row["remaining"], 0)
        solo = store.combination_budget(
            "main",
            make_series(),
            {"solo": ("feature",)},
            {"y": 1},
            0,
            {},
        )[0]
        # Alone, feature's y differences are -2 and +3.
        self.assertEqual(solo["max_overrun"], 3)
        self.assertEqual(solo["remaining"], -3)

    def test_per_key_breach_uses_absolute_aggregate(self) -> None:
        store = make_store()
        # Risk 16 < 100, but the aggregate z difference is -10 against a
        # per-key budget of 9: weighted excess is (10 - 9) * 1 = 1.
        row = store.combination_budget(
            "main",
            make_series(),
            {"both": ("feature", "side")},
            {"x": 2, "y": 3, "z": 1},
            100,
            {"z": 9},
        )[0]
        self.assertTrue(row["breached"])
        self.assertEqual(row["first_breach"], 1)
        self.assertEqual(row["max_overrun"], 1)
        self.assertEqual(row["remaining"], 84)
        self.assertEqual(row["over_keys"], ("z",))
        self.assertEqual(len(row["attributions"]), 1)

    def test_max_overrun_is_largest_across_checkpoints(self) -> None:
        store = make_store()
        # Risks 21 and 16 against a budget of 15: overruns 6 and 1.
        row = store.combination_budget(
            "main",
            make_series(),
            {"both": ("feature", "side")},
            {"x": 2, "y": 3, "z": 1},
            15,
            {},
        )[0]
        self.assertTrue(row["breached"])
        self.assertEqual(row["first_breach"], 0)
        self.assertEqual(row["max_overrun"], 6)
        self.assertEqual(row["remaining"], -1)

    def test_float_arithmetic_and_zero_signs(self) -> None:
        store = make_store()
        row = store.combination_budget(
            "main",
            make_series(),
            {"both": ("feature", "side")},
            {"x": 0.5},
            1.0,
            {},
        )[0]
        # Both checkpoints aggregate x to 3: risk 1.5, overrun 0.5.
        self.assertTrue(row["breached"])
        self.assertEqual(row["max_overrun"], 0.5)
        self.assertIsInstance(row["max_overrun"], float)
        self.assertEqual(row["remaining"], -0.5)
        self.assertIsInstance(row["remaining"], float)
        self.assertEqual(row["contributions"], (0.5, 1.0))
        # A negative zero weight keeps the zero risk a positive zero.
        row = store.combination_budget(
            "main",
            make_series(),
            {"both": ("feature", "side")},
            {"x": -0.0},
            0.0,
            {},
        )[0]
        self.assertFalse(row["breached"])
        self.assertEqual(row["remaining"], 0.0)
        self.assertIsInstance(row["remaining"], float)
        self.assertGreater(math.copysign(1.0, row["remaining"]), 0)
        self.assertEqual(row["max_overrun"], 0)
        self.assertGreater(math.copysign(1.0, row["max_overrun"]), 0)
        self.assertEqual(row["contributions"], (0.0, 0.0))
        for contribution in row["contributions"]:
            self.assertGreater(math.copysign(1.0, contribution), 0)

    def test_empty_member_tuple_never_breaches(self) -> None:
        store = make_store()
        row = store.combination_budget(
            "main",
            make_series(),
            {"none": ()},
            {"x": 1},
            5.5,
            {},
        )[0]
        self.assertEqual(row["members"], ())
        self.assertFalse(row["breached"])
        self.assertIsNone(row["first_breach"])
        self.assertEqual(row["max_overrun"], 0)
        self.assertIsInstance(row["max_overrun"], int)
        # Remaining keeps the input budget value.
        self.assertEqual(row["remaining"], 5.5)
        self.assertEqual(row["over_keys"], ())
        self.assertEqual(row["contributions"], ())
        self.assertEqual(row["attributions"], ())
        row = store.combination_budget(
            "main", make_series(), {"none": ()}, {"x": 1}, 5, {}
        )[0]
        self.assertEqual(row["remaining"], 5)
        self.assertIsInstance(row["remaining"], int)

    def test_no_checkpoints_never_breaches(self) -> None:
        store = make_store()
        row = store.combination_budget(
            "main",
            {"feature": (), "side": ()},
            {"both": ("feature", "side")},
            {"x": 1},
            5.5,
            {},
        )[0]
        self.assertFalse(row["breached"])
        self.assertIsNone(row["first_breach"])
        self.assertEqual(row["max_overrun"], 0)
        self.assertEqual(row["remaining"], 5.5)
        self.assertEqual(row["over_keys"], ())
        self.assertEqual(row["contributions"], (0, 0))
        self.assertEqual(row["attributions"], ())

    def test_empty_combinations_validates_inputs_and_reference(self) -> None:
        store = make_store()
        series = make_series()
        self.assertEqual(
            store.combination_budget("main", series, {}, {"x": 1}, 5, {}),
            (),
        )
        # Series branches are not looked up when no combination exists.
        self.assertEqual(
            store.combination_budget(
                "main", {"ghost": ()}, {}, {"x": 1}, 5, {}
            ),
            (),
        )
        with self.assertRaises(KeyError):
            store.combination_budget("ghost", series, {}, {"x": 1}, 5, {})
        with self.assertRaises(ValueError):
            store.combination_budget("main", series, {}, {"x": 1}, -1, {})
        with self.assertRaises(TypeError):
            store.combination_budget("main", series, {}, {"x": 1}, True, {})
        with self.assertRaises(ValueError):
            store.combination_budget(
                "main", series, {}, {"x": 1}, 5, {"z": 1}
            )
        with self.assertRaises(TypeError):
            store.combination_budget("main", series, None, {"x": 1}, 5, {})

    def test_empty_weights_zero_risk_and_empty_key_budgets(self) -> None:
        store = make_store()
        row = store.combination_budget(
            "main",
            make_series(),
            {"both": ("feature", "side")},
            {},
            0,
            {},
        )[0]
        self.assertFalse(row["breached"])
        self.assertIsNone(row["first_breach"])
        self.assertEqual(row["max_overrun"], 0)
        self.assertEqual(row["remaining"], 0)
        self.assertEqual(row["over_keys"], ())
        self.assertEqual(row["contributions"], (0, 0))
        # Every member, branch and node is still checked.
        with self.assertRaises(ValueError):
            store.combination_budget(
                "main", make_series(), {"bad": ("ghost",)}, {}, 0, {}
            )
        with self.assertRaises(KeyError):
            store.combination_budget(
                "main", {"ghost": ()}, {"g": ("ghost",)}, {}, 0, {}
            )
        with self.assertRaises(KeyError):
            store.combination_budget(
                "main",
                {"feature": (("nope", "f1"),)},
                {"c": ("feature",)},
                {},
                0,
                {},
            )
        with self.assertRaises(KeyError):
            store.combination_budget(
                "main",
                {"feature": (("m1", "nope"),)},
                {"c": ("feature",)},
                {},
                0,
                {},
            )
        # A per-key budget with no weights to back it is still rejected.
        with self.assertRaises(ValueError):
            store.combination_budget(
                "main",
                make_series(),
                {"both": ("feature", "side")},
                {},
                0,
                {"x": 0},
            )

    def test_over_keys_sorted_and_attribution_entry_shape(self) -> None:
        store = make_store()
        row = store.combination_budget(
            "main",
            make_series(),
            {"both": ("feature", "side")},
            {"z": 1, "y": 1, "x": 1},
            1000,
            {"z": 0, "x": 0},
        )[0]
        self.assertEqual(row["over_keys"], ("x", "z"))
        self.assertEqual(
            [entry["key"] for entry in row["attributions"]], ["x", "z"]
        )
        for entry in row["attributions"]:
            self.assertEqual(
                list(entry), ["key", "aggregate", "diffs", "attributions"]
            )
        x_entry, z_entry = row["attributions"]
        # Final checkpoint aggregates: x is 1 + 2 = 3, z is -7 + -3 = -10.
        self.assertEqual(x_entry["aggregate"], 3)
        self.assertEqual(x_entry["diffs"], (1, 2))
        self.assertEqual(z_entry["aggregate"], -10)
        self.assertEqual(z_entry["diffs"], (-7, -3))

    def test_attributions_match_historical_divergence(self) -> None:
        store = make_store()
        expected_feature = store.attribute_divergences_at(
            "main", "m2", "feature", "f2", ("z",)
        )[0]
        expected_side = store.attribute_divergences_at(
            "main", "m2", "side", "s2", ("z",)
        )[0]
        row = store.combination_budget(
            "main",
            make_series(),
            {"both": ("feature", "side")},
            {"x": 2, "y": 3, "z": 1},
            100,
            {"z": 9},
        )[0]
        self.assertEqual(row["over_keys"], ("z",))
        entry = row["attributions"][0]
        self.assertEqual(entry["attributions"][0], expected_feature)
        self.assertEqual(entry["attributions"][1], expected_side)
        for attribution in entry["attributions"]:
            self.assertEqual(
                list(attribution), ["key", "fork", "left", "right"]
            )
            for side in ("left", "right"):
                self.assertEqual(
                    list(attribution[side]),
                    ["value", "cause", "path", "affected"],
                )
        self.assertIsNot(
            entry["attributions"][0], entry["attributions"][1]
        )

    def test_rows_sorted_breached_first_then_overrun_breach_name(self) -> None:
        store = make_store()
        store.create("heavy", "m1")
        store.append("heavy", "h1", 7, {"z": 30})
        store.create("calm", "root")
        series = {
            "feature": (("m1", "f1"), ("m2", "f2")),
            "side": (("m1", "s1"), ("m2", "s2")),
            "heavy": (("m1", "m1"), ("m2", "h1")),
            "calm": (("root", "root"),),
        }
        combinations = {
            "both": ("feature", "side"),
            "heavy-solo": ("heavy",),
            "none": (),
            "calm-one": ("calm",),
        }
        result = store.combination_budget(
            "main", series, combinations, {"x": 2, "y": 3, "z": 1}, 20, {}
        )
        # heavy-solo overruns 3 at index 1 (z diff 23 vs budget 20), both
        # overruns 1 at index 0; the unbreached sort by name, None last.
        self.assertEqual(
            [row["combination"] for row in result],
            ["heavy-solo", "both", "calm-one", "none"],
        )
        self.assertEqual(
            [
                (row["breached"], row["first_breach"], row["max_overrun"])
                for row in result
            ],
            [(True, 1, 3), (True, 0, 1), (False, None, 0), (False, None, 0)],
        )

    def test_first_breach_orders_before_name_on_overrun_tie(self) -> None:
        graph = EventGraph()
        graph.add("root", 0, (), {})
        store = BranchStore(graph)
        store.create("main", "root")
        store.create("a", "root")
        store.append("a", "a1", 1, {"x": 5})
        store.create("b", "root")
        store.append("b", "b1", 2, {"x": 5})
        series = {
            "a": (("root", "root"), ("root", "a1")),
            "b": (("root", "b1"),),
        }
        result = store.combination_budget(
            "main",
            series,
            {"first-at-1": ("a",), "first-at-0": ("b",)},
            {"x": 1},
            0,
            {},
        )
        # Both overrun 5; the breach at index 0 beats the one at index 1.
        self.assertEqual(
            [row["combination"] for row in result],
            ["first-at-0", "first-at-1"],
        )
        self.assertEqual(
            [(row["first_breach"], row["max_overrun"]) for row in result],
            [(0, 5), (1, 5)],
        )

    def test_combinations_container_validation(self) -> None:
        store = make_store()
        series = make_series()
        for bad in ([], None, (), (("c", ()),), "c", 1):
            with self.subTest(container=repr(bad)):
                with self.assertRaises(TypeError):
                    store.combination_budget(
                        "main", series, bad, {"x": 1}, 5, {}
                    )
        for bad_members in ([], "feature", None, 1, {"feature": ()}):
            with self.subTest(members=repr(bad_members)):
                with self.assertRaises(TypeError):
                    store.combination_budget(
                        "main",
                        series,
                        {"c": bad_members},
                        {"x": 1},
                        5,
                        {},
                    )
        with self.assertRaises(TypeError):
            store.combination_budget(
                "main", series, {1: ()}, {"x": 1}, 5, {}
            )
        with self.assertRaises(ValueError):
            store.combination_budget(
                "main", series, {"": ()}, {"x": 1}, 5, {}
            )
        with self.assertRaises(TypeError):
            store.combination_budget(
                "main", series, {"c": (1,)}, {"x": 1}, 5, {}
            )
        with self.assertRaises(ValueError):
            store.combination_budget(
                "main", series, {"c": ("",)}, {"x": 1}, 5, {}
            )
        with self.assertRaises(ValueError):
            store.combination_budget(
                "main",
                series,
                {"c": ("feature", "feature")},
                {"x": 1},
                5,
                {},
            )
        with self.assertRaises(ValueError):
            store.combination_budget(
                "main", series, {"c": ("ghost",)}, {"x": 1}, 5, {}
            )

    def test_checkpoint_alignment_validation(self) -> None:
        store = make_store()
        # Different checkpoint counts across members.
        with self.assertRaises(ValueError):
            store.combination_budget(
                "main",
                {
                    "feature": (("m1", "f1"), ("m2", "f2")),
                    "side": (("m1", "s1"),),
                },
                {"both": ("feature", "side")},
                {"x": 1},
                5,
                {},
            )
        # Same count but different reference nodes at a position.
        with self.assertRaises(ValueError):
            store.combination_budget(
                "main",
                {
                    "feature": (("m2", "f2"),),
                    "side": (("m1", "s1"),),
                },
                {"both": ("feature", "side")},
                {"x": 1},
                5,
                {},
            )
        # Member nodes may differ freely; only reference nodes align.
        row = store.combination_budget(
            "main",
            {"feature": (("m2", "f2"),), "side": (("m2", "s1"),)},
            {"both": ("feature", "side")},
            {"x": 1},
            100,
            {},
        )[0]
        self.assertFalse(row["breached"])

    def test_budget_validation_matches_impact_budget(self) -> None:
        store = make_store()
        series = make_series()
        both = {"both": ("feature", "side")}
        for bad in (True, False, "1", None, (1,), [1], {"x": 1}):
            with self.subTest(value=repr(bad)):
                with self.assertRaises(TypeError):
                    store.combination_budget(
                        "main", series, both, {"x": 1}, bad, {}
                    )
        for bad in (-1, -0.5, float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=repr(bad)):
                with self.assertRaises(ValueError):
                    store.combination_budget(
                        "main", series, both, {"x": 1}, bad, {}
                    )
        for bad in ([], None, (), (("x", 1),), "x", 1):
            with self.subTest(container=repr(bad)):
                with self.assertRaises(TypeError):
                    store.combination_budget(
                        "main", series, both, {"x": 1}, 5, bad
                    )
        with self.assertRaises(ValueError):
            store.combination_budget(
                "main", series, both, {"x": 1}, 5, {"": 1}
            )
        with self.assertRaises(ValueError):
            store.combination_budget(
                "main", series, both, {"x": 1}, 5, {"z": 1}
            )
        with self.assertRaises(TypeError):
            store.combination_budget(
                "main", series, both, {"x": 1}, 5, {"x": True}
            )
        with self.assertRaises(ValueError):
            store.combination_budget(
                "main", series, both, {"x": 1}, 5, {"x": -1}
            )

    def test_all_inputs_validated_before_branch_lookups(self) -> None:
        store = make_store()
        series = make_series()
        # Series errors beat combination errors.
        with self.assertRaises(ValueError):
            store.combination_budget(
                "main", {"main": ()}, [], {"x": 1}, 5, {}
            )
        # Combination errors beat weight errors.
        with self.assertRaises(ValueError):
            store.combination_budget(
                "main",
                series,
                {"c": ("feature", "feature")},
                {"x": "1"},
                5,
                {},
            )
        # Weight errors beat budget errors.
        with self.assertRaises(TypeError):
            store.combination_budget(
                "main", series, {"c": ("feature",)}, {"x": "1"}, True, {}
            )
        # Budget errors beat unknown branches and nodes.
        with self.assertRaises(TypeError):
            store.combination_budget(
                "ghost", series, {"c": ("feature",)}, {"x": 1}, True, {}
            )
        with self.assertRaises(ValueError):
            store.combination_budget(
                "ghost", series, {"c": ("feature",)}, {"x": 1}, 5, {"z": 1}
            )
        # An unknown member is a validation error, not a lookup.
        with self.assertRaises(ValueError):
            store.combination_budget(
                "ghost", series, {"c": ("ghost-member",)}, {"x": 1}, 5, {}
            )

    def test_branch_and_node_lookup_order(self) -> None:
        store = make_store()
        series = make_series()
        # Reference is looked up before the series branches.
        with self.assertRaises(KeyError) as caught:
            store.combination_budget(
                "ghost", series, {"c": ("feature",)}, {"x": 1}, 5, {}
            )
        self.assertEqual(caught.exception.args, ("ghost",))
        with self.assertRaises(KeyError) as caught:
            store.combination_budget(
                "main",
                {"ghost1": (), "ghost2": ()},
                {"c": ("ghost1",)},
                {"x": 1},
                5,
                {},
            )
        self.assertEqual(caught.exception.args, ("ghost1",))
        # Nodes of the first series are checked before a later series'
        # branch is looked up; left node before right within a pair.
        with self.assertRaises(KeyError) as caught:
            store.combination_budget(
                "main",
                {"feature": (("nope1", "f1"),), "ghost": ()},
                {"c": ("feature",)},
                {"x": 1},
                5,
                {},
            )
        self.assertEqual(caught.exception.args, ("nope1",))
        with self.assertRaises(KeyError) as caught:
            store.combination_budget(
                "main",
                {"feature": (("m1", "nope2"), ("nope1", "f1"))},
                {"c": ("feature",)},
                {"x": 1},
                5,
                {},
            )
        self.assertEqual(caught.exception.args, ("nope2",))
        # A node outside its branch's head closure is rejected.
        with self.assertRaises(KeyError):
            store.combination_budget(
                "main",
                {"feature": (("f2", "f1"),)},
                {"c": ("feature",)},
                {"x": 1},
                5,
                {},
            )

    def test_results_are_detached_and_unshared(self) -> None:
        store = make_store()
        series = make_series()
        combinations = {
            "both": ("feature", "side"),
            "solo": ("feature",),
        }
        weights = {"x": 2, "y": 3, "z": 1}
        first = store.combination_budget(
            "main", series, combinations, weights, 0, {"x": 0, "z": 0}
        )
        self.assertIsNot(first[0], first[1])
        self.assertIsNot(first[0]["attributions"], first[1]["attributions"])
        self.assertIsNot(
            first[0]["attributions"][0]["attributions"],
            first[1]["attributions"][0]["attributions"],
        )
        pristine = copy.deepcopy(first)
        for row in first:
            row["remaining"] = -999
            row["over_keys"] += ("evil",)
            row["contributions"] += (999,)
            for entry in row["attributions"]:
                entry["aggregate"] = 0
                entry["diffs"] += (0,)
                for attribution in entry["attributions"]:
                    attribution["key"] = "evil"
                    attribution["left"]["path"] += ("evil",)
        fresh = store.combination_budget(
            "main", series, combinations, weights, 0, {"x": 0, "z": 0}
        )
        self.assertEqual(fresh, pristine)
        self.assertEqual(fresh[0]["combination"], "both")
        self.assertEqual(fresh[0]["over_keys"], ("x", "z"))

    def test_query_is_read_only(self) -> None:
        store = make_store()
        heads = dict(store._heads)
        appends = copy.deepcopy(store._appends)
        merges = copy.deepcopy(store._merges)
        replay_main = store.replay("main")
        replay_feature = store.replay("feature")
        audit = store.audit_log()
        series = make_series()
        store.combination_budget(
            "main",
            series,
            {"both": ("feature", "side"), "none": ()},
            {"x": 2, "y": 3, "z": 1},
            10,
            {"x": 0},
        )
        # Failing calls must leave state untouched as well.
        with self.assertRaises(KeyError):
            store.combination_budget(
                "main",
                {"feature": (("m1", "f1"), ("nope", "f2"))},
                {"c": ("feature",)},
                {"x": 1},
                10,
                {},
            )
        with self.assertRaises(ValueError):
            store.combination_budget(
                "main",
                series,
                {"c": ("feature", "side", "feature")},
                {"x": 1},
                10,
                {},
            )
        with self.assertRaises(ValueError):
            store.combination_budget(
                "main",
                series,
                {"c": ("feature",)},
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
        # The existing divergence and budget interfaces keep their results.
        matrix = store.divergence_matrix(
            "main",
            {"feature": (("m2", "f2"), ("m1", "f1"))},
            ("z", "x"),
        )
        budget = store.impact_budget(
            "main",
            {"feature": (("m2", "f2"),)},
            {"x": 2, "y": 3, "z": 1},
            10,
            {},
        )
        store.combination_budget(
            "main",
            series,
            {"both": ("feature", "side")},
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
        self.assertEqual(
            store.impact_budget(
                "main",
                {"feature": (("m2", "f2"),)},
                {"x": 2, "y": 3, "z": 1},
                10,
                {},
            ),
            budget,
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
