import copy
import inspect
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


def scenario(
    name: str,
    total_budget: int | float = 100,
    weights: dict | None = None,
    key_budgets: dict | None = None,
) -> dict:
    return {
        "name": name,
        "weights": WEIGHTS if weights is None else weights,
        "total_budget": total_budget,
        "key_budgets": {} if key_budgets is None else key_budgets,
    }


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
            "main",
            make_series(),
            (scenario("loose"), scenario("tight", 17)),
            0,
            2,
            (),
            (),
            10,
        )
        self.assertIsInstance(result, dict)
        self.assertEqual(list(result), ["scenarios", "combinations"])
        self.assertIsInstance(result["scenarios"], tuple)
        self.assertIsInstance(result["combinations"], tuple)
        self.assertEqual(len(result["scenarios"]), 2)
        for entry in result["scenarios"]:
            self.assertEqual(list(entry), ["name", "frontier", "rejected"])
            self.assertIsInstance(entry["frontier"], tuple)
            self.assertIsInstance(entry["rejected"], tuple)
            for row in entry["frontier"]:
                self.assertEqual(
                    list(row),
                    ["members", "risk", "contributions", "attributions"],
                )
            for row in entry["rejected"]:
                self.assertEqual(
                    list(row),
                    ["members", "reason", "checkpoint", "key", "overrun"],
                )
        # One combination row per enumerated candidate, in enumeration
        # order: sizes ascending, tuples by code point within a size.
        self.assertEqual(
            [row["members"] for row in result["combinations"]],
            [(), ("feature",), ("side",), ("feature", "side")],
        )
        for row in result["combinations"]:
            self.assertEqual(
                list(row),
                [
                    "members",
                    "first_entry",
                    "first_exit",
                    "dominators",
                    "risk_delta",
                    "member_delta",
                ],
            )
            self.assertEqual(len(row["dominators"]), 2)
            self.assertEqual(len(row["risk_delta"]), 2)
            self.assertEqual(len(row["member_delta"]), 2)

    def test_each_scenario_matches_one_existing_search(self) -> None:
        store = make_store()
        series = make_series()
        scenarios = (
            scenario("loose", 100),
            scenario("tight", 17),
            scenario("zero", 0),
        )
        result = store.frontier_sensitivity(
            "main", series, scenarios, 0, 2, (), (), 10
        )
        for entry, total in zip(
            result["scenarios"], (100, 17, 0)
        ):
            search = store.search_combinations(
                "main", series, WEIGHTS, total, {}, 0, 2, (), (), 10
            )
            self.assertEqual(
                [dict(row) for row in entry["frontier"]],
                [dict(row) for row in search["frontier"]],
            )
            self.assertEqual(
                [dict(row) for row in entry["rejected"]],
                [dict(row) for row in search["rejected"]],
            )

    def test_scenario_order_preserved_and_names_independent(self) -> None:
        store = make_store()
        scenarios = (
            scenario("zero", 0),
            scenario("loose", 100),
            scenario("tight", 17),
        )
        result = store.frontier_sensitivity(
            "main",
            make_series(),
            scenarios,
            0,
            2,
            (),
            (),
            10,
        )
        self.assertEqual(
            [entry["name"] for entry in result["scenarios"]],
            ["zero", "loose", "tight"],
        )

    def test_first_entry_and_first_exit(self) -> None:
        store = make_store()
        # Frontier memberships:
        # loose(100): () and pair; tight(17): () and side; zero(0): ().
        result = store.frontier_sensitivity(
            "main",
            make_series(),
            (scenario("loose", 100), scenario("tight", 17), scenario("zero", 0)),
            0,
            2,
            (),
            (),
            10,
        )
        by_members = {
            row["members"]: row for row in result["combinations"]
        }
        empty = by_members[()]
        self.assertEqual(empty["first_entry"], 0)
        self.assertIsNone(empty["first_exit"])
        # feature is feasible under loose but dominated by the pair there;
        # it never reaches a frontier.
        feature = by_members[("feature",)]
        self.assertIsNone(feature["first_entry"])
        self.assertIsNone(feature["first_exit"])
        side = by_members[("side",)]
        self.assertEqual(side["first_entry"], 1)
        self.assertEqual(side["first_exit"], 2)
        pair = by_members[("feature", "side")]
        self.assertEqual(pair["first_entry"], 0)
        self.assertEqual(pair["first_exit"], 1)

    def test_first_entry_zero_when_on_first_frontier(self) -> None:
        store = make_store()
        result = store.frontier_sensitivity(
            "main",
            make_series(),
            (scenario("loose", 100), scenario("zero", 0)),
            0,
            2,
            (),
            (),
            10,
        )
        pair = next(
            row
            for row in result["combinations"]
            if row["members"] == ("feature", "side")
        )
        self.assertEqual(pair["first_entry"], 0)
        self.assertEqual(pair["first_exit"], 1)

    def test_dominators_follow_existing_rule_and_frontier_order(self) -> None:
        store = make_store()
        result = store.frontier_sensitivity(
            "main",
            make_series(),
            (scenario("loose", 100), scenario("tight", 17), scenario("zero", 0)),
            0,
            2,
            (),
            (),
            10,
        )
        by_members = {
            row["members"]: row for row in result["combinations"]
        }
        # Scenario 0 (loose): feature is feasible but both the pair (more
        # members, risk 16 < 18) and side (same size, risk 16 < 18)
        # dominate it; frontier order lists the pair first, then side.
        # Empty subset is dominated by nothing (no feasible combo has
        # lower risk).
        self.assertEqual(
            by_members[("feature",)]["dominators"][0],
            (("feature", "side"), ("side",)),
        )
        # Scenario 1 (tight): feature is infeasible -> empty tuple.
        self.assertEqual(by_members[("feature",)]["dominators"][1], ())
        # Scenario 2 (zero): still infeasible.
        self.assertEqual(by_members[("feature",)]["dominators"][2], ())
        # side in the loose scenario: the pair has more members but equal
        # risk (16) -- strict-better on member count alone dominates.
        self.assertEqual(
            by_members[("side",)]["dominators"][0],
            (("feature", "side"),),
        )
        # On the tight frontier side dominates nobody dominates it, and
        # the pair is infeasible there.
        self.assertEqual(by_members[("side",)]["dominators"][1], ())
        # A frontier combination lists no dominators in that scenario.
        self.assertEqual(by_members[()]["dominators"], ((), (), ()))
        self.assertEqual(
            by_members[("feature", "side")]["dominators"][0], ()
        )

    def test_dominators_ordered_like_frontier(self) -> None:
        graph = EventGraph()
        graph.add("root", 0, (), {})
        store = BranchStore(graph)
        store.create("main", "root")
        for name in ("a", "b", "c"):
            store.create(name, "root")
        store.append("main", "r1", 1, {"x": 10})
        store.append("a", "a1", 1, {"x": 11})
        store.append("b", "b1", 1, {"x": 12})
        store.append("c", "c1", 1, {"x": 13})
        series = {
            name: (("r1", f"{name}1"),) for name in ("a", "b", "c")
        }
        # In this construction b(2) is dominated by a(1) (same size,
        # lower risk), c(3) by a(1), ab(3) by... nothing at size 2 with
        # lower risk? a alone dominates ab? fewer members -> no. abc(6),
        # ab(3), a(1), () are the frontier (as in the search test).
        # bc(5) is dominated by ab(3), and also by a(1) (fewer members
        # does not dominate since domination needs no FEWER members:
        # a has 1 member < 2, so it does not dominate bc).
        result = store.frontier_sensitivity(
            "main",
            series,
            (scenario("only", 100, weights={"x": 1}),),
            0,
            3,
            (),
            (),
            50,
        )
        by_members = {
            row["members"]: row for row in result["combinations"]
        }
        # b: dominated by a (same size, risk 1) and ab/abc (more members,
        # higher risk still dominates via member count since risk must be
        # no higher... ab risk 3 > 2, so no). Only a and combinations
        # with >=1 members and risk <=2: a(1). abc(6) no. ab(3) no.
        self.assertEqual(by_members[("b",)]["dominators"][0], (("a",),))
        # bc (risk 5): dominators need >=2 members, risk <= 5:
        # ab(3) yes; ac(4) yes; abc(6) no. Frontier order among them:
        # descending member count first (both size 2), then risk: ab(3)
        # before ac(4).
        self.assertEqual(
            by_members[("b", "c")]["dominators"][0],
            (("a", "b"), ("a", "c")),
        )
        # c (risk 3) is dominated by every feasible combo with no fewer
        # members and no higher risk, strictly better on one: the larger
        # ab (risk 3, more members), and the same-size a (risk 1) and b
        # (risk 2). b is itself off the frontier (a dominates it), yet it
        # still qualifies -- every feasible dominator is listed, sorted in
        # frontier order (descending count, ascending risk, tuple).
        self.assertEqual(
            by_members[("c",)]["dominators"][0],
            (("a", "b"), ("a",), ("b",)),
        )

    def test_risk_delta_adjacent_scenarios(self) -> None:
        store = make_store()
        # Weights change between scenarios, so even a feasible-everywhere
        # combination's final risk moves. Use key-only weights on x:
        # pair point-1 aggregate x = 3 -> risks 3 then 6.
        scenarios = (
            scenario("w1", 100, weights={"x": 1}),
            scenario("w2", 100, weights={"x": 2}),
        )
        result = store.frontier_sensitivity(
            "main", make_series(), scenarios, 2, 2, (), (), 10
        )
        pair = result["combinations"][0]
        self.assertEqual(pair["members"], ("feature", "side"))
        self.assertIsNone(pair["risk_delta"][0])
        self.assertEqual(pair["risk_delta"][1], 3)

    def test_risk_delta_none_when_infeasible_in_either_scenario(self) -> None:
        store = make_store()
        # feature is feasible loose (risk 18 <= 100), breaches tight
        # (18 > 17), feasible again at an even looser budget.
        result = store.frontier_sensitivity(
            "main",
            make_series(),
            (
                scenario("loose", 100),
                scenario("tight", 17),
                scenario("looser", 1000),
            ),
            0,
            2,
            (),
            (),
            10,
        )
        feature = next(
            row
            for row in result["combinations"]
            if row["members"] == ("feature",)
        )
        self.assertEqual(feature["risk_delta"], (None, None, None))

    def test_member_delta_adjacent_scenarios(self) -> None:
        store = make_store()
        scenarios = (
            scenario("w1", 100, weights={"x": 1}),
            scenario("w2", 100, weights={"x": 2}),
        )
        result = store.frontier_sensitivity(
            "main", make_series(), scenarios, 2, 2, (), (), 10
        )
        pair = result["combinations"][0]
        # First scenario contributes no predecessor.
        self.assertIsNone(pair["member_delta"][0])
        # Contributions under x-only weights at the final checkpoint:
        # feature final x diff 1, side final x diff 2, aggregate 3.
        # Without feature risk 2 -> contribution 1; without side risk 1
        # -> contribution 2; doubled weights give 2 and 4, deltas 1, 2.
        self.assertEqual(pair["member_delta"][1], (1, 2))

    def test_member_delta_none_without_previous_or_when_infeasible(self) -> None:
        store = make_store()
        result = store.frontier_sensitivity(
            "main",
            make_series(),
            (scenario("tight", 17), scenario("loose", 100)),
            1,
            1,
            (),
            (),
            10,
        )
        side = next(
            row
            for row in result["combinations"]
            if row["members"] == ("side",)
        )
        # side feasible in both scenarios: index 0 None (no predecessor),
        # index 1 a one-element delta tuple.
        self.assertIsNone(side["member_delta"][0])
        self.assertEqual(side["member_delta"][1], (0,))
        feature = next(
            row
            for row in result["combinations"]
            if row["members"] == ("feature",)
        )
        # feature breaches the tight first scenario.
        self.assertEqual(feature["member_delta"], (None, None))

    def test_empty_subset_member_delta_is_empty_tuple(self) -> None:
        store = make_store()
        result = store.frontier_sensitivity(
            "main",
            make_series(),
            (scenario("a", 100), scenario("b", 1000)),
            0,
            0,
            (),
            (),
            10,
        )
        empty = result["combinations"][0]
        self.assertEqual(empty["members"], ())
        self.assertIsNone(empty["risk_delta"][0])
        self.assertEqual(empty["risk_delta"][1], 0)
        self.assertIsNone(empty["member_delta"][0])
        self.assertEqual(empty["member_delta"][1], ())

    def test_empty_scenarios_still_checks_branches_nodes_and_limit(self) -> None:
        store = make_store()
        result = store.frontier_sensitivity(
            "main", make_series(), (), 0, 2, (), (), 10
        )
        self.assertEqual(result, {"scenarios": (), "combinations": ()})
        # Unknown reference branch is still looked up.
        with self.assertRaises(KeyError):
            store.frontier_sensitivity(
                "ghost", make_series(), (), 0, 2, (), (), 10
            )
        # Node closure is still checked.
        with self.assertRaises(KeyError):
            store.frontier_sensitivity(
                "main",
                {"feature": (("m1", "nope"),)},
                (),
                0,
                1,
                (),
                (),
                10,
            )
        # The candidate limit is a search constraint and still enforced.
        with self.assertRaises(ValueError):
            store.frontier_sensitivity(
                "main", make_series(), (), 0, 2, (), (), 3
            )

    def test_empty_scenarios_still_runs_member_validation(self) -> None:
        store = make_store()
        with self.assertRaises(ValueError):
            store.frontier_sensitivity(
                "main", make_series(), (), 0, 2, ("ghost",), (), 10
            )
        with self.assertRaises(TypeError):
            store.frontier_sensitivity(
                "main", make_series(), (), True, 2, (), (), 10
            )

    # --- Scenario container / item / field validation. ---

    def test_scenarios_container_must_be_tuple(self) -> None:
        store = make_store()
        for bad in ([], None, {}, "s", 1):
            with self.subTest(scenarios=repr(bad)):
                with self.assertRaises(TypeError):
                    store.frontier_sensitivity(
                        "main", make_series(), bad, 0, 2, (), (), 10
                    )

    def test_scenario_item_must_be_dict(self) -> None:
        store = make_store()
        for bad in (None, 1, "s", (), []):
            with self.subTest(item=repr(bad)):
                with self.assertRaises(TypeError):
                    store.frontier_sensitivity(
                        "main",
                        make_series(),
                        (bad,),
                        0,
                        2,
                        (),
                        (),
                        10,
                    )

    def test_scenario_fields_exact_set(self) -> None:
        store = make_store()
        base = scenario("s")
        for mutated in (
            {k: v for k, v in base.items() if k != "name"},
            {**base, "extra": 1},
            {},
        ):
            with self.subTest(fields=repr(mutated)):
                with self.assertRaises(ValueError):
                    store.frontier_sensitivity(
                        "main",
                        make_series(),
                        (mutated,),
                        0,
                        2,
                        (),
                        (),
                        10,
                    )

    def test_scenario_name_type_and_value(self) -> None:
        store = make_store()
        for bad_name in (1, None, (), []):
            with self.subTest(name=repr(bad_name)):
                with self.assertRaises(TypeError):
                    store.frontier_sensitivity(
                        "main",
                        make_series(),
                        (scenario(bad_name),),
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
                (scenario(""),),
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
                (scenario("dup"), scenario("dup")),
                0,
                2,
                (),
                (),
                10,
            )

    def test_scenario_weights_and_budgets_reuse_search_rules(self) -> None:
        store = make_store()
        series = make_series()
        # Non-dict weights.
        with self.assertRaises(TypeError):
            store.frontier_sensitivity(
                "main",
                series,
                (scenario("s", weights=()),),
                0,
                2,
                (),
                (),
                10,
            )
        # A bool/non-number weight.
        with self.assertRaises(TypeError):
            store.frontier_sensitivity(
                "main",
                series,
                (scenario("s", weights={"x": True}),),
                0,
                2,
                (),
                (),
                10,
            )
        # Negative / NaN / infinite weight.
        for bad in (-1, float("nan"), float("inf")):
            with self.subTest(weight=repr(bad)):
                with self.assertRaises(ValueError):
                    store.frontier_sensitivity(
                        "main",
                        series,
                        (scenario("s", weights={"x": bad}),),
                        0,
                        2,
                        (),
                        (),
                        10,
                    )
        # Bad total budgets.
        for bad in (True, "1", None):
            with self.subTest(total=repr(bad)):
                with self.assertRaises(TypeError):
                    store.frontier_sensitivity(
                        "main",
                        series,
                        (scenario("s", total_budget=bad),),
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
                        series,
                        (scenario("s", total_budget=bad),),
                        0,
                        2,
                        (),
                        (),
                        10,
                    )
        # key_budgets not a dict.
        with self.assertRaises(TypeError):
            store.frontier_sensitivity(
                "main",
                series,
                (scenario("s", key_budgets=()),),
                0,
                2,
                (),
                (),
                10,
            )
        # A per-key budget key outside the scenario's weights.
        with self.assertRaises(ValueError):
            store.frontier_sensitivity(
                "main",
                series,
                (
                    scenario(
                        "s",
                        weights={"x": 1},
                        key_budgets={"z": 1},
                    ),
                ),
                0,
                2,
                (),
                (),
                10,
            )
        # Per-key key sets are scoped to each scenario independently.
        store.frontier_sensitivity(
            "main",
            series,
            (
                scenario("a", weights={"x": 1}, key_budgets={"x": 1}),
                scenario("b", weights={"z": 2}, key_budgets={"z": 0}),
            ),
            0,
            2,
            (),
            (),
            10,
        )

    def test_scenarios_walked_in_input_order(self) -> None:
        store = make_store()
        # The first scenario is valid; the second carries the error.
        with self.assertRaises(TypeError):
            store.frontier_sensitivity(
                "main",
                make_series(),
                (scenario("ok"), None),
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
                (scenario("ok"), scenario("ok")),
                0,
                2,
                (),
                (),
                10,
            )

    # --- Search-parameter validation mirrors search_combinations. ---

    def test_search_parameters_keep_existing_validation(self) -> None:
        store = make_store()
        series = make_series()
        scenarios = (scenario("s"),)
        for bad in (True, 1.0, "1", None):
            with self.subTest(min_size=repr(bad)):
                with self.assertRaises(TypeError):
                    store.frontier_sensitivity(
                        "main", series, scenarios, bad, 2, (), (), 10
                    )
            with self.subTest(max_size=repr(bad)):
                with self.assertRaises(TypeError):
                    store.frontier_sensitivity(
                        "main", series, scenarios, 0, bad, (), (), 10
                    )
            with self.subTest(limit=repr(bad)):
                with self.assertRaises(TypeError):
                    store.frontier_sensitivity(
                        "main", series, scenarios, 0, 2, (), (), bad
                    )
        for call in (
            lambda: store.frontier_sensitivity(
                "main", series, scenarios, -1, 2, (), (), 10
            ),
            lambda: store.frontier_sensitivity(
                "main", series, scenarios, 2, 1, (), (), 10
            ),
            lambda: store.frontier_sensitivity(
                "main", series, scenarios, 0, 3, (), (), 10
            ),
            lambda: store.frontier_sensitivity(
                "main", series, scenarios, 0, 2, (), (), 0
            ),
            lambda: store.frontier_sensitivity(
                "main", series, scenarios, 0, 2, ("ghost",), (), 10
            ),
            lambda: store.frontier_sensitivity(
                "main",
                series,
                scenarios,
                0,
                2,
                (),
                (("feature", "ghost"),),
                10,
            ),
        ):
            with self.subTest(call=call):
                with self.assertRaises(ValueError):
                    call()

    def test_required_container_type(self) -> None:
        store = make_store()
        scenarios = (scenario("s"),)
        for bad in ([], None, "feature", 1):
            with self.subTest(required=repr(bad)):
                with self.assertRaises(TypeError):
                    store.frontier_sensitivity(
                        "main",
                        make_series(),
                        scenarios,
                        0,
                        2,
                        bad,
                        (),
                        10,
                    )

    def test_exclusive_pairs_container_and_shape(self) -> None:
        store = make_store()
        scenarios = (scenario("s"),)
        for bad in ([], None, "x", 1):
            with self.subTest(pairs=repr(bad)):
                with self.assertRaises(TypeError):
                    store.frontier_sensitivity(
                        "main",
                        make_series(),
                        scenarios,
                        0,
                        2,
                        (),
                        bad,
                        10,
                    )
        with self.assertRaises(TypeError):
            store.frontier_sensitivity(
                "main",
                make_series(),
                scenarios,
                0,
                2,
                (),
                (("feature",),),
                10,
            )

    def test_alignment_error_matches_search(self) -> None:
        store = make_store()
        scenarios = (scenario("s"),)
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

    def test_all_inputs_validated_before_branch_lookups(self) -> None:
        store = make_store()
        series = make_series()
        # Scenario numeric errors beat an unknown reference branch.
        with self.assertRaises(ValueError):
            store.frontier_sensitivity(
                "ghost",
                series,
                (scenario("s", total_budget=-1),),
                0,
                2,
                (),
                (),
                10,
            )
        # Size errors beat unknown branches.
        with self.assertRaises(TypeError):
            store.frontier_sensitivity(
                "ghost",
                series,
                (scenario("s"),),
                True,
                2,
                (),
                (),
                10,
            )
        # Member-constraint errors beat unknown branches.
        with self.assertRaises(ValueError):
            store.frontier_sensitivity(
                "ghost",
                series,
                (scenario("s"),),
                0,
                2,
                ("ghost-m",),
                (),
                10,
            )

    def test_branch_and_node_lookup_order(self) -> None:
        store = make_store()
        scenarios = (scenario("s"),)
        with self.assertRaises(KeyError) as caught:
            store.frontier_sensitivity(
                "ghost",
                make_series(),
                scenarios,
                0,
                2,
                (),
                (),
                10,
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

    def test_limit_counts_candidates_once_and_no_partial_results(self) -> None:
        store = make_store()
        # Window 0..2 over two members enumerates 4 candidates.
        result = store.frontier_sensitivity(
            "main",
            make_series(),
            (scenario("a"), scenario("b", 0)),
            0,
            2,
            (),
            (),
            4,
        )
        self.assertEqual(len(result["combinations"]), 4)
        with self.assertRaises(ValueError):
            store.frontier_sensitivity(
                "main",
                make_series(),
                (scenario("a"), scenario("b", 0)),
                0,
                2,
                (),
                (),
                3,
            )

    def test_results_detached_and_unshared_across_scenarios(self) -> None:
        store = make_store()
        series = make_series()
        scenarios = (
            scenario("loose", 100),
            scenario("tight", 17),
        )
        result = store.frontier_sensitivity(
            "main", series, scenarios, 0, 2, (), (), 10
        )
        pristine = copy.deepcopy(result)
        # Mutating every returned level must not affect a repeated call.
        for entry in result["scenarios"]:
            for row in entry["frontier"]:
                row["members"] += ("evil",)
                row["contributions"] += (999,)
                for group in row["attributions"]:
                    for attribution in group:
                        attribution["key"] = "evil"
                        attribution["left"]["path"] += ("evil",)
            for row in entry["rejected"]:
                row["members"] += ("evil",)
        for row in result["combinations"]:
            row["members"] += ("evil",)
            row["dominators"] = row["dominators"] + (("evil",),)
            row["risk_delta"] = row["risk_delta"] + (999,)
        fresh = store.frontier_sensitivity(
            "main", series, scenarios, 0, 2, (), (), 10
        )
        self.assertEqual(fresh, pristine)
        # Scenario rows never alias each other even when equal.
        loose = result and fresh["scenarios"][0]
        tight = fresh["scenarios"][1]
        empty_loose = next(
            row for row in loose["frontier"] if row["members"] == ()
        )
        empty_tight = next(
            row for row in tight["frontier"] if row["members"] == ()
        )
        self.assertIsNot(empty_loose, empty_tight)

    def test_inputs_are_not_retained(self) -> None:
        store = make_store()
        series = make_series()
        # Local weight maps: the call must copy what it needs, and later
        # caller mutation must not reach the already-built result.
        loose_weights = {"x": 2, "y": 3, "z": 1}
        tight_weights = dict(loose_weights)
        scenarios = [
            scenario("loose", 100, weights=loose_weights),
            scenario("tight", 17, weights=tight_weights),
        ]
        result = store.frontier_sensitivity(
            "main", series, tuple(scenarios), 0, 2, (), (), 10
        )
        scenarios[0]["name"] = "mutated"
        loose_weights["x"] = 999
        self.assertEqual(
            [entry["name"] for entry in result["scenarios"]],
            ["loose", "tight"],
        )
        # The loose scenario's pair risk was computed with the original
        # weight 2, not the mutated 999.
        loose = result["scenarios"][0]
        pair = next(
            row
            for row in loose["frontier"]
            if row["members"] == ("feature", "side")
        )
        self.assertEqual(pair["risk"], 16)

    def test_query_is_read_only(self) -> None:
        store = make_store()
        heads = dict(store._heads)
        appends = copy.deepcopy(store._appends)
        merges = copy.deepcopy(store._merges)
        replay_main = store.replay("main")
        audit = store.audit_log()
        series = make_series()
        scenarios = (scenario("loose", 100), scenario("tight", 17))
        store.frontier_sensitivity(
            "main", series, scenarios, 0, 2, (), (), 10
        )
        # Failing calls leave state untouched as well.
        with self.assertRaises(ValueError):
            store.frontier_sensitivity(
                "main", series, scenarios, 0, 2, (), (), 1
            )
        with self.assertRaises(KeyError):
            store.frontier_sensitivity(
                "main",
                {"feature": (("m1", "nope"),)},
                scenarios,
                0,
                1,
                (),
                (),
                10,
            )
        with self.assertRaises(TypeError):
            store.frontier_sensitivity(
                "main", series, None, 0, 2, (), (), 10
            )
        self.assertEqual(dict(store._heads), heads)
        self.assertEqual(copy.deepcopy(store._appends), appends)
        self.assertEqual(copy.deepcopy(store._merges), merges)
        self.assertEqual(store.replay("main"), replay_main)
        self.assertEqual(store.audit_log(), audit)
        # Existing interfaces keep their behavior.
        search = store.search_combinations(
            "main", series, WEIGHTS, 17, {}, 0, 2, (), (), 10
        )
        store.frontier_sensitivity(
            "main", series, scenarios, 0, 2, (), (), 10
        )
        self.assertEqual(
            store.search_combinations(
                "main", series, WEIGHTS, 17, {}, 0, 2, (), (), 10
            ),
            search,
        )

    def test_float_risk_delta_normalized_positive_zero(self) -> None:
        store = make_store()
        # Identical float-weighted scenarios keep risk equal; the delta
        # must not surface as negative zero.
        result = store.frontier_sensitivity(
            "main",
            make_series(),
            (
                scenario("a", 100, weights={"x": 0.5}),
                scenario("b", 100, weights={"x": 0.5}),
            ),
            2,
            2,
            (),
            (),
            10,
        )
        pair = result["combinations"][0]
        self.assertEqual(pair["risk_delta"][1], 0.0)
        self.assertFalse(str(pair["risk_delta"][1]).startswith("-"))

    def test_no_checkpoints_scenarios(self) -> None:
        store = make_store()
        series = {"feature": (), "side": ()}
        result = store.frontier_sensitivity(
            "main",
            series,
            (scenario("a", 0, key_budgets={"x": 0}), scenario("b", 100)),
            0,
            2,
            (),
            (),
            10,
        )
        # With zero risk everywhere the largest subset alone is on each
        # frontier; deltas across scenarios are all zero.
        for entry in result["scenarios"]:
            self.assertEqual(
                [row["members"] for row in entry["frontier"]],
                [("feature", "side")],
            )
            self.assertEqual(entry["rejected"], ())
        pair = result["combinations"][-1]
        self.assertEqual(pair["first_entry"], 0)
        self.assertIsNone(pair["first_exit"])
        self.assertEqual(pair["risk_delta"], (None, 0))
        self.assertEqual(pair["member_delta"], (None, (0, 0)))


if __name__ == "__main__":
    unittest.main()
