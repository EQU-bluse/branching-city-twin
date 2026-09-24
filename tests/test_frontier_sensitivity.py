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


WEIGHTS = {"x": 2, "y": 3, "z": 1}


def make_scenarios() -> tuple:
    return (
        {"name": "tight", "weights": WEIGHTS, "total_budget": 0,
         "key_budgets": {}},
        {"name": "mid", "weights": WEIGHTS, "total_budget": 17,
         "key_budgets": {}},
        {"name": "loose", "weights": WEIGHTS, "total_budget": 100,
         "key_budgets": {}},
    )


class FrontierSensitivityTests(unittest.TestCase):
    def test_public_signature_has_no_defaults(self) -> None:
        parameters = list(
            inspect.signature(
                BranchStore.frontier_sensitivity
            ).parameters.values()
        )
        self.assertEqual(
            [p.name for p in parameters],
            [
                "self",
                "reference",
                "series",
                "scenarios",
                "min_size",
                "max_size",
                "required",
                "exclusive_pairs",
                "limit",
            ],
        )
        self.assertTrue(
            all(p.default is inspect.Parameter.empty for p in parameters)
        )

    def test_result_shape_and_key_order(self) -> None:
        store = make_store()
        result = store.frontier_sensitivity(
            "main", make_series(), make_scenarios(), 0, 2, (), (), 10
        )
        self.assertIsInstance(result, dict)
        self.assertEqual(list(result), ["scenarios", "combinations"])
        self.assertIsInstance(result["scenarios"], tuple)
        self.assertIsInstance(result["combinations"], tuple)
        for scenario in result["scenarios"]:
            self.assertEqual(list(scenario), ["name", "frontier", "rejected"])
            self.assertIsInstance(scenario["frontier"], tuple)
            self.assertIsInstance(scenario["rejected"], tuple)
            for row in scenario["frontier"]:
                self.assertEqual(
                    list(row),
                    ["members", "risk", "contributions", "attributions"],
                )
            for item in scenario["rejected"]:
                self.assertEqual(
                    list(item),
                    ["members", "reason", "checkpoint", "key", "overrun"],
                )
        for combo in result["combinations"]:
            self.assertEqual(
                list(combo),
                [
                    "members",
                    "first_entry",
                    "first_exit",
                    "dominators",
                    "risk_delta",
                    "member_delta",
                ],
            )
            self.assertIsInstance(combo["members"], tuple)
            self.assertIsInstance(combo["dominators"], tuple)
            self.assertIsInstance(combo["risk_delta"], tuple)
            self.assertIsInstance(combo["member_delta"], tuple)

    def test_scenarios_follow_input_order_and_names_kept(self) -> None:
        store = make_store()
        result = store.frontier_sensitivity(
            "main", make_series(), make_scenarios(), 0, 2, (), (), 10
        )
        self.assertEqual(
            [scenario["name"] for scenario in result["scenarios"]],
            ["tight", "mid", "loose"],
        )

    def test_each_scenario_matches_standalone_search(self) -> None:
        store = make_store()
        series = make_series()
        scenarios = (
            {"name": "tight", "weights": WEIGHTS, "total_budget": 0,
             "key_budgets": {}},
            {"name": "mid", "weights": WEIGHTS, "total_budget": 17,
             "key_budgets": {"z": 100}},
            {"name": "weighted", "weights": {"x": 1}, "total_budget": 10,
             "key_budgets": {}},
        )
        result = store.frontier_sensitivity(
            "main", series, scenarios, 0, 2, (), (), 10
        )
        for scenario in result["scenarios"]:
            spec = next(
                item for item in scenarios if item["name"] == scenario["name"]
            )
            standalone = store.search_combinations(
                "main",
                series,
                spec["weights"],
                spec["total_budget"],
                spec["key_budgets"],
                0,
                2,
                (),
                (),
                10,
            )
            self.assertEqual(scenario["frontier"], standalone["frontier"])
            self.assertEqual(scenario["rejected"], standalone["rejected"])

    def test_combinations_follow_single_enumeration_order(self) -> None:
        store = make_store()
        result = store.frontier_sensitivity(
            "main", make_series(), make_scenarios(), 0, 2, (), (), 10
        )
        # Sizes ascending, member tuples by Unicode order within a size,
        # regardless of how many scenarios run.
        self.assertEqual(
            [combo["members"] for combo in result["combinations"]],
            [(), ("feature",), ("side",), ("feature", "side")],
        )

    def test_candidates_enumerated_once_regardless_of_scenario_count(self) -> None:
        store = make_store()
        result = store.frontier_sensitivity(
            "main", make_series(), make_scenarios(), 0, 2, (), (), 4
        )
        self.assertEqual(len(result["combinations"]), 4)
        # The same four candidates exceed a limit of three even though the
        # budget work happens three times: the limit bounds enumeration.
        with self.assertRaises(ValueError):
            store.frontier_sensitivity(
                "main", make_series(), make_scenarios(), 0, 2, (), (), 3
            )

    def test_first_entry_and_first_exit_track_frontier_membership(self) -> None:
        store = make_store()
        result = store.frontier_sensitivity(
            "main", make_series(), make_scenarios(), 0, 2, (), (), 10
        )
        by_members = {
            combo["members"]: combo for combo in result["combinations"]
        }
        # The empty subset is on every frontier (risk 0 is never dominated).
        self.assertEqual(by_members[()]["first_entry"], 0)
        self.assertIsNone(by_members[()]["first_exit"])
        # feature is infeasible at tight/mid and only feasible (dominated)
        # at loose, so it never enters a frontier.
        self.assertIsNone(by_members[("feature",)]["first_entry"])
        self.assertIsNone(by_members[("feature",)]["first_exit"])
        # side enters the mid frontier (index 1), then leaves at loose
        # (index 2) where the pair offers more members at the same risk.
        self.assertEqual(by_members[("side",)]["first_entry"], 1)
        self.assertEqual(by_members[("side",)]["first_exit"], 2)
        # The pair first reaches the frontier at loose.
        self.assertEqual(by_members[("feature", "side")]["first_entry"], 2)
        self.assertIsNone(by_members[("feature", "side")]["first_exit"])

    def test_dominators_none_when_infeasible_and_empty_on_frontier(self) -> None:
        store = make_store()
        result = store.frontier_sensitivity(
            "main", make_series(), make_scenarios(), 0, 2, (), (), 10
        )
        by_members = {
            combo["members"]: combo for combo in result["combinations"]
        }
        # feature is infeasible in the first two scenarios.
        self.assertEqual(
            by_members[("feature",)]["dominators"][:2], (None, None)
        )
        # Frontier combinations have no dominators.
        self.assertEqual(by_members[()]["dominators"], ((), (), ()))
        self.assertEqual(
            by_members[("feature", "side")]["dominators"],
            (None, None, ()),
        )

    def test_dominators_list_all_feasible_in_frontier_order(self) -> None:
        store = make_store()
        result = store.frontier_sensitivity(
            "main", make_series(), make_scenarios(), 0, 2, (), (), 10
        )
        by_members = {
            combo["members"]: combo for combo in result["combinations"]
        }
        # At loose, feature (risk 18) is dominated by the pair (more
        # members, risk 16) and by side (same size, lower risk 16). The
        # pair comes first in the frontier ordering (-size, risk, tuple).
        self.assertEqual(
            by_members[("feature",)]["dominators"][2],
            (("feature", "side"), ("side",)),
        )
        # side is itself on the loose frontier at index 1 but dominated at
        # loose (index 2) only by the larger pair.
        self.assertEqual(by_members[("side",)]["dominators"][1], ())
        self.assertEqual(
            by_members[("side",)]["dominators"][2], (("feature", "side"),)
        )

    def test_risk_and_member_deltas_between_adjacent_scenarios(self) -> None:
        store = make_store()
        result = store.frontier_sensitivity(
            "main", make_series(), make_scenarios(), 0, 2, (), (), 10
        )
        by_members = {
            combo["members"]: combo for combo in result["combinations"]
        }
        # The first scenario never has a previous scenario.
        self.assertIsNone(by_members[()]["risk_delta"][0])
        self.assertIsNone(by_members[()]["member_delta"][0])
        # Empty subset is feasible everywhere with constant zero risk.
        self.assertEqual(by_members[()]["risk_delta"], (None, 0, 0))
        self.assertEqual(by_members[()]["member_delta"], (None, 0, 0))
        # feature is infeasible until loose, so every delta is None.
        self.assertEqual(
            by_members[("feature",)]["risk_delta"], (None, None, None)
        )
        self.assertEqual(
            by_members[("feature",)]["member_delta"], (None, None, None)
        )
        # side is infeasible at tight (no delta into mid), feasible with
        # risk 16 at mid and again 16 at loose: a zero risk delta.
        self.assertEqual(
            by_members[("side",)]["risk_delta"], (None, None, 0)
        )
        self.assertEqual(
            by_members[("side",)]["member_delta"], (None, None, 0)
        )

    def test_risk_delta_reflects_weight_change(self) -> None:
        graph = EventGraph()
        graph.add("root", 0, (), {})
        store = BranchStore(graph)
        store.create("main", "root")
        store.create("a", "root")
        store.append("main", "r1", 1, {"x": 10})
        store.append("a", "a1", 1, {"x": 11})  # diff +1
        series = {"a": (("r1", "a1"),)}
        scenarios = (
            {"name": "w1", "weights": {"x": 1}, "total_budget": 100,
             "key_budgets": {}},
            {"name": "w10", "weights": {"x": 10}, "total_budget": 100,
             "key_budgets": {}},
        )
        result = store.frontier_sensitivity(
            "main", series, scenarios, 0, 1, (), (), 10
        )
        by_members = {
            combo["members"]: combo for combo in result["combinations"]
        }
        self.assertEqual(by_members[("a",)]["risk_delta"], (None, 9))
        self.assertEqual(by_members[("a",)]["member_delta"], (None, 0))

    def test_float_deltas_are_never_negative_zero(self) -> None:
        store = make_store()
        scenarios = (
            {"name": "a", "weights": {"x": 0.5}, "total_budget": 100.0,
             "key_budgets": {}},
            {"name": "b", "weights": {"x": 0.5}, "total_budget": 100.0,
             "key_budgets": {}},
        )
        result = store.frontier_sensitivity(
            "main", make_series(), scenarios, 0, 0, (), (), 10
        )
        empty = result["combinations"][0]
        # The empty subset never evaluates a checkpoint: integer zero.
        self.assertIs(empty["risk_delta"][1], 0)
        scenarios = (
            {"name": "a", "weights": {"x": 0.5}, "total_budget": 100.0,
             "key_budgets": {}},
            {"name": "b", "weights": {"x": 0.5}, "total_budget": 100.0,
             "key_budgets": {}},
        )
        result = store.frontier_sensitivity(
            "main", make_series(), scenarios, 1, 1, (), (), 10
        )
        feature = result["combinations"][0]
        delta = feature["risk_delta"][1]
        self.assertEqual(delta, 0.0)
        self.assertFalse(math.copysign(1.0, delta) < 0)

    def test_structural_rejects_repeat_in_every_scenario(self) -> None:
        store = make_store()
        scenarios = make_scenarios()
        result = store.frontier_sensitivity(
            "main",
            make_series(),
            scenarios,
            0,
            2,
            ("side",),
            (("feature", "side"),),
            10,
        )
        for scenario in result["scenarios"]:
            reasons = {
                item["members"]: item["reason"]
                for item in scenario["rejected"]
            }
            self.assertEqual(reasons[()], "missing_required")
            self.assertEqual(reasons[("feature",)], "missing_required")
            self.assertEqual(reasons[("feature", "side")], "exclusive_pair")

    def test_empty_scenarios_still_checks_constraints_branches_nodes(self) -> None:
        store = make_store()
        result = store.frontier_sensitivity(
            "main", make_series(), (), 0, 2, (), (), 10
        )
        self.assertEqual(result["scenarios"], ())
        self.assertEqual(
            [combo["members"] for combo in result["combinations"]],
            [(), ("feature",), ("side",), ("feature", "side")],
        )
        for combo in result["combinations"]:
            self.assertIsNone(combo["first_entry"])
            self.assertIsNone(combo["first_exit"])
            self.assertEqual(combo["dominators"], ())
            self.assertEqual(combo["risk_delta"], ())
            self.assertEqual(combo["member_delta"], ())
        # Limit and search constraints still apply.
        with self.assertRaises(ValueError):
            store.frontier_sensitivity(
                "main", make_series(), (), 0, 2, (), (), 3
            )
        with self.assertRaises(ValueError):
            store.frontier_sensitivity(
                "main", make_series(), (), 0, 3, (), (), 10
            )
        # Branches and nodes are still looked up.
        with self.assertRaises(KeyError):
            store.frontier_sensitivity(
                "ghost", make_series(), (), 0, 2, (), (), 10
            )
        with self.assertRaises(KeyError):
            store.frontier_sensitivity(
                "main",
                {"feature": (("m1", "nope"),), "side": (("m1", "s1"),)},
                (),
                0,
                2,
                (),
                (),
                10,
            )

    def test_no_checkpoints_feasible_with_integer_zero_deltas(self) -> None:
        store = make_store()
        series = {"feature": (), "side": ()}
        result = store.frontier_sensitivity(
            "main", series, make_scenarios()[:2], 0, 2, (), (), 10
        )
        pair = next(
            combo
            for combo in result["combinations"]
            if combo["members"] == ("feature", "side")
        )
        self.assertEqual(pair["first_entry"], 0)
        self.assertIsNone(pair["first_exit"])
        self.assertEqual(pair["risk_delta"], (None, 0))
        self.assertIs(pair["risk_delta"][1], 0)
        for scenario in result["scenarios"]:
            self.assertEqual(
                [row["members"] for row in scenario["frontier"]],
                [("feature", "side")],
            )
            self.assertEqual(scenario["frontier"][0]["risk"], 0)

    # --- scenario container / item validation ---

    def test_scenarios_container_must_be_tuple(self) -> None:
        store = make_store()
        for bad in ([], None, "x", 1, {"x"}, [{}]):
            with self.subTest(scenarios=repr(bad)):
                with self.assertRaises(TypeError):
                    store.frontier_sensitivity(
                        "main", make_series(), bad, 0, 2, (), (), 10
                    )

    def test_scenario_item_must_be_dict(self) -> None:
        store = make_store()
        for bad in (("x",), (1,), (None,), ((),)):
            with self.subTest(item=repr(bad)):
                with self.assertRaises(TypeError):
                    store.frontier_sensitivity(
                        "main", make_series(), bad, 0, 2, (), (), 10
                    )

    def test_scenario_missing_and_extra_keys_raise_value_error(self) -> None:
        store = make_store()

        def scenario(**overrides) -> dict:
            base = {
                "name": "s",
                "weights": WEIGHTS,
                "total_budget": 100,
                "key_budgets": {},
            }
            base.update(overrides)
            return base

        for drop in ("name", "weights", "total_budget", "key_budgets"):
            missing = scenario()
            del missing[drop]
            with self.subTest(drop=drop):
                with self.assertRaises(ValueError):
                    store.frontier_sensitivity(
                        "main", make_series(), (missing,), 0, 2, (), (), 10
                    )
        with self.assertRaises(ValueError):
            store.frontier_sensitivity(
                "main",
                make_series(),
                (scenario(extra=1),),
                0,
                2,
                (),
                (),
                10,
            )

    def test_scenario_name_validation(self) -> None:
        store = make_store()

        def scenario(name: object) -> dict:
            return {
                "name": name,
                "weights": WEIGHTS,
                "total_budget": 100,
                "key_budgets": {},
            }

        for bad in (1, 1.0, None, (), ["s"]):
            with self.subTest(name=repr(bad)):
                with self.assertRaises(TypeError):
                    store.frontier_sensitivity(
                        "main", make_series(), (scenario(bad),), 0, 2,
                        (), (), 10,
                    )
        with self.assertRaises(ValueError):
            store.frontier_sensitivity(
                "main", make_series(), (scenario(""),), 0, 2, (), (), 10
            )
        with self.assertRaises(ValueError):
            store.frontier_sensitivity(
                "main",
                make_series(),
                (scenario("s"), scenario("s")),
                0,
                2,
                (),
                (),
                10,
            )

    def test_scenario_weights_and_budgets_follow_search_rules(self) -> None:
        store = make_store()

        def scenario(**fields) -> dict:
            base = {
                "name": "s",
                "weights": WEIGHTS,
                "total_budget": 100,
                "key_budgets": {},
            }
            base.update(fields)
            return base

        with self.assertRaises(TypeError):
            store.frontier_sensitivity(
                "main",
                make_series(),
                (scenario(weights=[]),),
                0,
                2,
                (),
                (),
                10,
            )
        with self.assertRaises(ValueError):
            store.frontier_sensitivity(
                "main",
                make_series(),
                (scenario(weights={"x": -1}),),
                0,
                2,
                (),
                (),
                10,
            )
        for bad in (True, "1", None, (), []):
            with self.subTest(total=repr(bad)):
                with self.assertRaises(TypeError):
                    store.frontier_sensitivity(
                        "main",
                        make_series(),
                        (scenario(total_budget=bad),),
                        0,
                        2,
                        (),
                        (),
                        10,
                    )
        for bad in (-1, float("nan"), float("inf")):
            with self.subTest(total=repr(bad)):
                with self.assertRaises(ValueError):
                    store.frontier_sensitivity(
                        "main",
                        make_series(),
                        (scenario(total_budget=bad),),
                        0,
                        2,
                        (),
                        (),
                        10,
                    )
        with self.assertRaises(TypeError):
            store.frontier_sensitivity(
                "main",
                make_series(),
                (scenario(key_budgets=[]),),
                0,
                2,
                (),
                (),
                10,
            )
        with self.assertRaises(ValueError):
            store.frontier_sensitivity(
                "main",
                make_series(),
                (scenario(key_budgets={"ghost": 1}),),
                0,
                2,
                (),
                (),
                10,
            )
        with self.assertRaises(ValueError):
            store.frontier_sensitivity(
                "main",
                make_series(),
                (scenario(key_budgets={"x": -1}),),
                0,
                2,
                (),
                (),
                10,
            )

    def test_second_scenarios_errors_still_reported_in_order(self) -> None:
        store = make_store()
        good = {
            "name": "good",
            "weights": WEIGHTS,
            "total_budget": 100,
            "key_budgets": {},
        }
        bad_name = {
            "name": "",
            "weights": WEIGHTS,
            "total_budget": 100,
            "key_budgets": {},
        }
        with self.assertRaises(ValueError):
            store.frontier_sensitivity(
                "main", make_series(), (good, bad_name), 0, 2, (), (), 10
            )

    def test_search_parameter_validation_matches_search_combinations(self) -> None:
        store = make_store()
        series = make_series()
        scenarios = make_scenarios()
        for bad in (True, False, 1.0, "1", None):
            with self.subTest(min_size=repr(bad)):
                with self.assertRaises(TypeError):
                    store.frontier_sensitivity(
                        "main", series, scenarios, bad, 2, (), (), 10
                    )
            with self.subTest(limit=repr(bad)):
                with self.assertRaises(TypeError):
                    store.frontier_sensitivity(
                        "main", series, scenarios, 0, 2, (), (), bad
                    )
        with self.assertRaises(ValueError):
            store.frontier_sensitivity(
                "main", series, scenarios, -1, 2, (), (), 10
            )
        with self.assertRaises(ValueError):
            store.frontier_sensitivity(
                "main", series, scenarios, 2, 1, (), (), 10
            )
        with self.assertRaises(ValueError):
            store.frontier_sensitivity(
                "main", series, scenarios, 0, 3, (), (), 10
            )
        with self.assertRaises(TypeError):
            store.frontier_sensitivity(
                "main", series, scenarios, 0, 2, ["side"], (), 10
            )
        with self.assertRaises(ValueError):
            store.frontier_sensitivity(
                "main", series, scenarios, 0, 2, ("ghost",), (), 10
            )
        with self.assertRaises(TypeError):
            store.frontier_sensitivity(
                "main", series, scenarios, 0, 2, (), ["x"], 10
            )
        with self.assertRaises(ValueError):
            store.frontier_sensitivity(
                "main",
                series,
                scenarios,
                0,
                2,
                ("feature", "side"),
                (("feature", "side"),),
                10,
            )

    def test_all_inputs_validated_before_branch_lookups(self) -> None:
        store = make_store()
        scenarios = make_scenarios()
        # Scenario shape errors beat the unknown reference branch.
        with self.assertRaises(TypeError):
            store.frontier_sensitivity(
                "ghost", make_series(), [], 0, 2, (), (), 10
            )
        with self.assertRaises(ValueError):
            store.frontier_sensitivity(
                "ghost", make_series(), scenarios, 0, 2, ("ghost-m",), (), 10
            )
        # Limit errors beat unknown branches.
        with self.assertRaises(ValueError):
            store.frontier_sensitivity(
                "ghost", make_series(), scenarios, 0, 2, (), (), 0
            )

    def test_branch_and_node_lookup_order(self) -> None:
        store = make_store()
        scenarios = make_scenarios()
        with self.assertRaises(KeyError) as caught:
            store.frontier_sensitivity(
                "ghost", make_series(), scenarios, 0, 2, (), (), 10
            )
        self.assertEqual(caught.exception.args, ("ghost",))
        with self.assertRaises(KeyError) as caught:
            store.frontier_sensitivity(
                "main",
                {"ghost1": (), "ghost2": ()},
                scenarios,
                0,
                0,
                (),
                (),
                10,
            )
        self.assertEqual(caught.exception.args, ("ghost1",))
        with self.assertRaises(KeyError) as caught:
            store.frontier_sensitivity(
                "main",
                {"feature": (("nope1", "f1"),)},
                scenarios,
                0,
                1,
                (),
                (),
                10,
            )
        self.assertEqual(caught.exception.args, ("nope1",))
        with self.assertRaises(KeyError) as caught:
            store.frontier_sensitivity(
                "main",
                {"feature": (("m1", "nope2"),)},
                scenarios,
                0,
                1,
                (),
                (),
                10,
            )
        self.assertEqual(caught.exception.args, ("nope2",))
        with self.assertRaises(KeyError):
            store.frontier_sensitivity(
                "main",
                {"feature": (("f2", "f1"),)},
                scenarios,
                0,
                1,
                (),
                (),
                10,
            )

    def test_checkpoint_misalignment_raises_value_error(self) -> None:
        store = make_store()
        scenarios = make_scenarios()
        with self.assertRaises(ValueError):
            store.frontier_sensitivity(
                "main",
                {
                    "feature": (("m1", "f1"), ("m2", "f2")),
                    "side": (("m1", "s1"),),
                },
                scenarios,
                0,
                2,
                (),
                (),
                10,
            )
        with self.assertRaises(ValueError):
            store.frontier_sensitivity(
                "main",
                {
                    "feature": (("m2", "f2"),),
                    "side": (("m1", "s1"),),
                },
                scenarios,
                0,
                2,
                (),
                (),
                10,
            )

    def test_results_are_detached_and_unshared(self) -> None:
        store = make_store()
        series = make_series()
        scenarios = make_scenarios()
        first = store.frontier_sensitivity(
            "main", series, scenarios, 0, 2, (), (), 10
        )
        pristine = copy.deepcopy(first)
        for scenario in first["scenarios"]:
            for row in scenario["frontier"]:
                row["members"] += ("evil",)
                for member_group in row["attributions"]:
                    for attribution in member_group:
                        attribution["key"] = "evil"
                        attribution["left"]["path"] += ("evil",)
            for item in scenario["rejected"]:
                item["members"] += ("evil",)
        for combo in first["combinations"]:
            combo["members"] += ("evil",)
            # dominators is a tuple of (None | tuple); rebuild it mutably.
            combo["dominators"] = tuple(
                None if value is None else value + ("evil",)
                for value in combo["dominators"]
            )
        fresh = store.frontier_sensitivity(
            "main", series, scenarios, 0, 2, (), (), 10
        )
        self.assertEqual(fresh, pristine)
        # Independent scenario copies must not alias one another or the
        # summary's member tuples.
        self.assertIsNot(
            fresh["scenarios"][0]["frontier"],
            fresh["scenarios"][1]["frontier"],
        )

    def test_query_is_read_only(self) -> None:
        store = make_store()
        heads = dict(store._heads)
        appends = copy.deepcopy(store._appends)
        merges = copy.deepcopy(store._merges)
        replay_main = store.replay("main")
        audit = store.audit_log()
        series = make_series()
        scenarios = make_scenarios()
        store.frontier_sensitivity(
            "main", series, scenarios, 0, 2, (), (), 10
        )
        with self.assertRaises(ValueError):
            store.frontier_sensitivity(
                "main", series, scenarios, 0, 2, (), (), 1
            )
        with self.assertRaises(KeyError):
            store.frontier_sensitivity(
                "main",
                {"feature": (("m1", "nope"),), "side": (("m1", "s1"),)},
                scenarios,
                0,
                2,
                (),
                (),
                10,
            )
        with self.assertRaises(TypeError):
            store.frontier_sensitivity(
                "main", series, scenarios, True, 2, (), (), 10
            )
        self.assertEqual(dict(store._heads), heads)
        self.assertEqual(copy.deepcopy(store._appends), appends)
        self.assertEqual(copy.deepcopy(store._merges), merges)
        self.assertEqual(store.replay("main"), replay_main)
        self.assertEqual(store.audit_log(), audit)
        # The existing search interface keeps its behavior.
        before = store.search_combinations(
            "main", series, WEIGHTS, 17, {}, 0, 2, (), (), 10
        )
        store.frontier_sensitivity(
            "main", series, scenarios, 0, 2, (), (), 10
        )
        after = store.search_combinations(
            "main", series, WEIGHTS, 17, {}, 0, 2, (), (), 10
        )
        self.assertEqual(after, before)

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
