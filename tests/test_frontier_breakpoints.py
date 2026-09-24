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


def base_scenario(
    weights: dict | None = None,
    total_budget: int | float = 100,
    key_budgets: dict | None = None,
) -> dict:
    return {
        "weights": WEIGHTS if weights is None else weights,
        "total_budget": total_budget,
        "key_budgets": {} if key_budgets is None else key_budgets,
    }


class FrontierBreakpointsTests(unittest.TestCase):
    # --- Signature -------------------------------------------------------

    def test_public_signature_has_no_defaults(self) -> None:
        parameters = list(
            inspect.signature(
                BranchStore.frontier_breakpoints
            ).parameters.values()
        )
        self.assertEqual(
            [p.name for p in parameters],
            [
                "self",
                "reference",
                "series",
                "base_scenario",
                "axis",
                "values",
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

    # --- Shape -----------------------------------------------------------

    def test_result_shape_and_key_order(self) -> None:
        store = make_store()
        result = store.frontier_breakpoints(
            "main",
            make_series(),
            base_scenario(total_budget=17),
            "total_budget",
            (16, 17, 100),
            0,
            2,
            (),
            (),
            10,
        )
        self.assertIsInstance(result, dict)
        self.assertEqual(list(result), ["points", "breakpoints"])
        self.assertIsInstance(result["points"], tuple)
        self.assertIsInstance(result["breakpoints"], tuple)
        self.assertEqual(len(result["points"]), 3)
        for point in result["points"]:
            self.assertEqual(list(point), ["value", "frontier", "rejected"])
            self.assertIsInstance(point["frontier"], tuple)
            self.assertIsInstance(point["rejected"], tuple)
            for row in point["frontier"]:
                self.assertEqual(
                    list(row),
                    ["members", "risk", "contributions", "attributions"],
                )
            for row in point["rejected"]:
                self.assertEqual(
                    list(row),
                    ["members", "reason", "checkpoint", "key", "overrun"],
                )
        for record in result["breakpoints"]:
            self.assertEqual(
                list(record),
                [
                    "left_bound",
                    "right_bound",
                    "left",
                    "right",
                    "entered",
                    "exited",
                    "affected",
                    "member_deltas",
                ],
            )
            for delta_row in record["member_deltas"]:
                self.assertEqual(list(delta_row), ["members", "delta"])

    def test_points_follow_value_order_with_candidate_enumeration(self) -> None:
        store = make_store()
        values = (0, 15, 16, 17, 100)
        result = store.frontier_breakpoints(
            "main",
            make_series(),
            base_scenario(),
            "total_budget",
            values,
            0,
            2,
            (),
            (),
            10,
        )
        self.assertEqual(
            [point["value"] for point in result["points"]],
            list(values),
        )
        for point in result["points"]:
            # Rejected rows keep candidate enumeration order: sizes
            # ascending, tuples by code point within a size. (Feasible
            # but dominated candidates appear in neither tuple.)
            rejected_members = [row["members"] for row in point["rejected"]]
            self.assertEqual(
                rejected_members,
                sorted(
                    rejected_members,
                    key=lambda members: (len(members), members),
                ),
            )

    # --- Per-point equivalence with the existing search ------------------

    def test_each_budget_point_matches_one_existing_search(self) -> None:
        store = make_store()
        series = make_series()
        values = (0, 15, 16, 17, 100)
        result = store.frontier_breakpoints(
            "main",
            series,
            base_scenario(),
            "total_budget",
            values,
            0,
            2,
            (),
            (),
            10,
        )
        for value, point in zip(values, result["points"]):
            search = store.search_combinations(
                "main", series, WEIGHTS, value, {}, 0, 2, (), (), 10
            )
            self.assertEqual(
                [dict(row) for row in point["frontier"]],
                [dict(row) for row in search["frontier"]],
            )
            self.assertEqual(
                [dict(row) for row in point["rejected"]],
                [dict(row) for row in search["rejected"]],
            )

    def test_each_weight_point_replaces_only_the_one_weight(self) -> None:
        store = make_store()
        series = make_series()
        values = (0, 1, 2, 3.0)
        result = store.frontier_breakpoints(
            "main",
            series,
            base_scenario(total_budget=17),
            "x",
            values,
            0,
            2,
            (),
            (),
            10,
        )
        self.assertEqual(
            [point["value"] for point in result["points"]],
            [0, 1, 2, 3.0],
        )
        for value, point in zip(values, result["points"]):
            weights = dict(WEIGHTS)
            weights["x"] = value
            search = store.search_combinations(
                "main", series, weights, 17, {}, 0, 2, (), (), 10
            )
            self.assertEqual(
                [dict(row) for row in point["frontier"]],
                [dict(row) for row in search["frontier"]],
            )
            self.assertEqual(
                [dict(row) for row in point["rejected"]],
                [dict(row) for row in search["rejected"]],
            )
            # The other two weights and the budgets are untouched.
            pair = next(
                (
                    row
                    for row in (*point["frontier"], *point["rejected"])
                    if row["members"] == ("feature", "side")
                ),
                None,
            )
            self.assertIsNotNone(pair)

    # --- Breakpoint semantics --------------------------------------------

    def test_budget_interval_records(self) -> None:
        store = make_store()
        # Frontier memberships: 0 and 15 -> (); 16 and 17 -> side + ();
        # 100 -> pair + ().
        result = store.frontier_breakpoints(
            "main",
            make_series(),
            base_scenario(),
            "total_budget",
            (0, 15, 16, 17, 100),
            0,
            2,
            (),
            (),
            10,
        )
        self.assertEqual(
            [(r["left_bound"], r["right_bound"]) for r in result["breakpoints"]],
            [(15, 16), (17, 100)],
        )

        first = result["breakpoints"][0]
        self.assertEqual(first["entered"], (("side",),))
        self.assertEqual(first["exited"], ())
        self.assertEqual(first["affected"], (("side",),))
        self.assertEqual(
            first["member_deltas"],
            ({"members": ("side",), "delta": None},),
        )

        second = result["breakpoints"][1]
        # Enumeration order: feature, side, pair.
        self.assertEqual(second["entered"], (("feature", "side"),))
        self.assertEqual(second["exited"], (("side",),))
        self.assertEqual(
            second["affected"],
            (("feature",), ("side",), ("feature", "side")),
        )
        self.assertEqual(
            [row["members"] for row in second["member_deltas"]],
            [("feature",), ("side",), ("feature", "side")],
        )
        deltas = {
            row["members"]: row["delta"] for row in second["member_deltas"]
        }
        # feature infeasible on the left, the pair infeasible on the left.
        self.assertIsNone(deltas[("feature",)])
        self.assertIsNone(deltas[("feature", "side")])
        # side stays feasible; its marginal contribution does not depend
        # on the total budget.
        self.assertEqual(deltas[("side",)], (0,))

    def test_record_carries_complete_results_on_both_sides(self) -> None:
        store = make_store()
        result = store.frontier_breakpoints(
            "main",
            make_series(),
            base_scenario(),
            "total_budget",
            (15, 16),
            0,
            2,
            (),
            (),
            10,
        )
        self.assertEqual(len(result["breakpoints"]), 1)
        record = result["breakpoints"][0]
        self.assertEqual(
            list(record["left"]), ["value", "frontier", "rejected"]
        )
        self.assertEqual(
            list(record["right"]), ["value", "frontier", "rejected"]
        )
        self.assertEqual(record["left_bound"], 15)
        self.assertEqual(record["right_bound"], 16)
        self.assertEqual(record["left"]["value"], 15)
        self.assertEqual(record["right"]["value"], 16)
        self.assertEqual(
            [row["members"] for row in record["left"]["frontier"]],
            [()],
        )
        self.assertEqual(
            [row["members"] for row in record["right"]["frontier"]],
            [("side",), ()],
        )

    def test_feasibility_only_change_still_records(self) -> None:
        store = make_store()
        # Weight axis x at total budget 17: between 1 and 2 feature flips
        # feasible -> rejected while the frontier (pair + empty) and its
        # members stay put.
        result = store.frontier_breakpoints(
            "main",
            make_series(),
            base_scenario(total_budget=17),
            "x",
            (0, 1, 2, 3.0),
            0,
            2,
            (),
            (),
            10,
        )
        self.assertEqual(
            [(r["left_bound"], r["right_bound"]) for r in result["breakpoints"]],
            [(0, 1), (1, 2), (2, 3.0)],
        )
        middle = result["breakpoints"][1]
        self.assertEqual(middle["entered"], ())
        self.assertEqual(middle["exited"], ())
        self.assertEqual(middle["affected"], (("feature",),))
        # Infeasible on the right -> delta None.
        self.assertEqual(
            middle["member_deltas"],
            ({"members": ("feature",), "delta": None},),
        )

    def test_weight_interval_entered_exited_and_member_deltas(self) -> None:
        store = make_store()
        result = store.frontier_breakpoints(
            "main",
            make_series(),
            base_scenario(total_budget=17),
            "x",
            (0, 1),
            0,
            2,
            (),
            (),
            10,
        )
        record = result["breakpoints"][0]
        self.assertEqual(record["entered"], (("side",),))
        self.assertEqual(record["exited"], (("feature", "side"),))
        self.assertEqual(
            record["affected"],
            (("feature",), ("side",), ("feature", "side")),
        )
        deltas = {
            row["members"]: row["delta"] for row in record["member_deltas"]
        }
        # feature and side are feasible on both sides; the pair is not.
        self.assertEqual(deltas[("feature",)], (1,))
        self.assertEqual(deltas[("side",)], (2,))
        self.assertIsNone(deltas[("feature", "side")])

    def test_domination_only_change_still_records(self) -> None:
        # Feasibility and frontier membership stay identical, yet the
        # dominator set of one feasible combination changes: a record is
        # still generated, with empty entered/exited and the combination
        # listed under affected.
        graph = EventGraph()
        graph.add("root", 0, (), {})
        store = BranchStore(graph)
        store.create("main", "root")
        store.create("a", "root")
        store.create("b", "root")
        store.append("main", "r1", 1, {})
        store.append("a", "a1", 2, {"x": 10})
        store.append("b", "b1", 3, {"x": -4, "y": 10})
        series = {"a": (("r1", "a1"),), "b": (("r1", "b1"),)}
        result = store.frontier_breakpoints(
            "main",
            series,
            {
                "weights": {"x": 1, "y": 0.3},
                "total_budget": 100.0,
                "key_budgets": {},
            },
            "y",
            (0.3, 0.5),
            0,
            2,
            (),
            (),
            10,
        )
        self.assertEqual(len(result["breakpoints"]), 1)
        record = result["breakpoints"][0]
        # The pair, b and the empty subset stay on both frontiers.
        self.assertEqual(
            [row["members"] for row in record["left"]["frontier"]],
            [row["members"] for row in record["right"]["frontier"]],
        )
        self.assertEqual(record["entered"], ())
        self.assertEqual(record["exited"], ())
        self.assertEqual(record["affected"], (("a",),))
        # a is feasible on both sides; its own marginal contribution does
        # not move with y's weight, so the delta is positive-zero float.
        self.assertEqual(
            record["member_deltas"],
            ({"members": ("a",), "delta": (0.0,)},),
        )

    def test_float_member_delta_is_positive_zero(self) -> None:
        # A weight-axis perturbation changes one member's risk and the
        # domination structure, while another member's float marginal
        # contribution stays equal: its right-minus-left delta is 0.0,
        # never negative zero.
        graph = EventGraph()
        graph.add("root", 0, (), {})
        store = BranchStore(graph)
        store.create("main", "root")
        for name in ("a", "b", "c"):
            store.create(name, "root")
        store.append("main", "r1", 1, {})
        store.append("a", "a1", 2, {"x": 8})
        store.append("b", "b1", 3, {"x": -8, "z": 30})
        store.append("c", "c1", 4, {"x": 8})
        series = {
            name: (("r1", f"{name}1"),) for name in ("a", "b", "c")
        }
        result = store.frontier_breakpoints(
            "main",
            series,
            {
                "weights": {"x": 0.5, "z": 0.0},
                "total_budget": 100.0,
                "key_budgets": {},
            },
            "z",
            (0.0, 0.5),
            0,
            3,
            (),
            (),
            50,
        )
        record = result["breakpoints"][0]
        a_delta = next(
            row["delta"]
            for row in record["member_deltas"]
            if row["members"] == ("a",)
        )
        self.assertEqual(a_delta, (0.0,))
        self.assertFalse(str(a_delta[0]).startswith("-"))

    def test_no_breakpoint_when_results_identical(self) -> None:
        store = make_store()
        # Both 16 and 17 keep side feasible with the same frontier.
        result = store.frontier_breakpoints(
            "main",
            make_series(),
            base_scenario(),
            "total_budget",
            (16, 17),
            0,
            2,
            (),
            (),
            10,
        )
        self.assertEqual(result["breakpoints"], ())
        self.assertEqual(len(result["points"]), 2)

    def test_two_value_one_interval_half_open_records(self) -> None:
        store = make_store()
        result = store.frontier_breakpoints(
            "main",
            make_series(),
            base_scenario(),
            "total_budget",
            (0, 100),
            0,
            2,
            (),
            (),
            10,
        )
        self.assertEqual(
            [(r["left_bound"], r["right_bound"]) for r in result["breakpoints"]],
            [(0, 100)],
        )

    # --- Degenerate pools -------------------------------------------------

    def test_no_checkpoints_has_no_breakpoints(self) -> None:
        store = make_store()
        series = {"feature": (), "side": ()}
        result = store.frontier_breakpoints(
            "main",
            series,
            base_scenario(total_budget=0, key_budgets={"x": 0}),
            "total_budget",
            (0, 100),
            0,
            2,
            (),
            (),
            10,
        )
        self.assertEqual(result["breakpoints"], ())
        for point in result["points"]:
            self.assertEqual(
                [row["members"] for row in point["frontier"]],
                [("feature", "side")],
            )
            self.assertEqual(point["rejected"], ())

    def test_empty_pool_still_runs_size_node_and_limit_checks(self) -> None:
        store = make_store()
        result = store.frontier_breakpoints(
            "main",
            {},
            base_scenario(),
            "total_budget",
            (0, 100),
            0,
            0,
            (),
            (),
            10,
        )
        self.assertEqual(len(result["points"]), 2)
        for point in result["points"]:
            self.assertEqual(
                [row["members"] for row in point["frontier"]], [()]
            )
            self.assertEqual(point["rejected"], ())
        self.assertEqual(result["breakpoints"], ())
        # Unknown reference is still looked up.
        with self.assertRaises(KeyError):
            store.frontier_breakpoints(
                "ghost",
                {},
                base_scenario(),
                "total_budget",
                (0, 100),
                0,
                0,
                (),
                (),
                10,
            )
        # Member-constraint validation still runs.
        with self.assertRaises(ValueError):
            store.frontier_breakpoints(
                "main",
                {},
                base_scenario(),
                "total_budget",
                (0, 100),
                0,
                0,
                ("ghost",),
                (),
                10,
            )
        # Size window must fit the (empty) pool.
        with self.assertRaises(ValueError):
            store.frontier_breakpoints(
                "main",
                {},
                base_scenario(),
                "total_budget",
                (0, 100),
                0,
                1,
                (),
                (),
                10,
            )

    def test_limit_counts_candidates_once_across_values(self) -> None:
        store = make_store()
        # Four candidates regardless of how many values are probed.
        result = store.frontier_breakpoints(
            "main",
            make_series(),
            base_scenario(),
            "total_budget",
            (0, 100),
            0,
            2,
            (),
            (),
            4,
        )
        self.assertEqual(len(result["points"]), 2)
        with self.assertRaises(ValueError):
            store.frontier_breakpoints(
                "main",
                make_series(),
                base_scenario(),
                "total_budget",
                (0, 100),
                0,
                2,
                (),
                (),
                3,
            )

    # --- Value-series validation -----------------------------------------

    def test_values_container_must_be_tuple(self) -> None:
        store = make_store()
        for bad in ([0, 1], None, {0: 1}, "s", 1, range(2)):
            with self.subTest(values=repr(bad)):
                with self.assertRaises(TypeError):
                    store.frontier_breakpoints(
                        "main",
                        make_series(),
                        base_scenario(),
                        "total_budget",
                        bad,
                        0,
                        2,
                        (),
                        (),
                        10,
                    )

    def test_values_need_at_least_two(self) -> None:
        store = make_store()
        for bad in ((), (1,)):
            with self.subTest(values=repr(bad)):
                with self.assertRaises(ValueError):
                    store.frontier_breakpoints(
                        "main",
                        make_series(),
                        base_scenario(),
                        "total_budget",
                        bad,
                        0,
                        2,
                        (),
                        (),
                        10,
                    )

    def test_value_types(self) -> None:
        store = make_store()
        for bad in (True, False, "1", None, 1j, (1,)):
            with self.subTest(value=repr(bad)):
                with self.assertRaises(TypeError):
                    store.frontier_breakpoints(
                        "main",
                        make_series(),
                        base_scenario(),
                        "total_budget",
                        (0, bad),
                        0,
                        2,
                        (),
                        (),
                        10,
                    )

    def test_value_ranges(self) -> None:
        store = make_store()
        for bad in (
            (-1, 0),
            (0, float("nan")),
            (float("nan"), 1),
            (0, float("inf")),
            (float("-inf"), 0),
            (1, 0),
            (0, 0),
            (0, 1, 1),
            (0, 2, 2, 3),
        ):
            with self.subTest(values=repr(bad)):
                with self.assertRaises(ValueError):
                    store.frontier_breakpoints(
                        "main",
                        make_series(),
                        base_scenario(),
                        "total_budget",
                        bad,
                        0,
                        2,
                        (),
                        (),
                        10,
                    )

    def test_zero_and_floats_accepted(self) -> None:
        store = make_store()
        store.frontier_breakpoints(
            "main",
            make_series(),
            base_scenario(),
            "total_budget",
            (0, 0.5, 1.5, 2),
            0,
            2,
            (),
            (),
            10,
        )

    # --- Scenario validation ----------------------------------------------

    def test_base_scenario_container_must_be_dict(self) -> None:
        store = make_store()
        for bad in (None, 42, (), [], "s"):
            with self.subTest(scenario=repr(bad)):
                with self.assertRaises(TypeError):
                    store.frontier_breakpoints(
                        "main",
                        make_series(),
                        bad,
                        "total_budget",
                        (0, 1),
                        0,
                        2,
                        (),
                        (),
                        10,
                    )

    def test_base_scenario_fields_exact_set(self) -> None:
        store = make_store()
        base = base_scenario()
        for mutated in (
            {k: v for k, v in base.items() if k != "weights"},
            {**base, "name": "extra"},
            {},
        ):
            with self.subTest(fields=repr(mutated)):
                with self.assertRaises(ValueError):
                    store.frontier_breakpoints(
                        "main",
                        make_series(),
                        mutated,
                        "total_budget",
                        (0, 1),
                        0,
                        2,
                        (),
                        (),
                        10,
                    )

    def test_base_scenario_weights_and_budgets_reuse_search_rules(self) -> None:
        store = make_store()
        series = make_series()
        with self.assertRaises(TypeError):
            store.frontier_breakpoints(
                "main",
                series,
                base_scenario(weights=()),
                "total_budget",
                (0, 1),
                0,
                2,
                (),
                (),
                10,
            )
        with self.assertRaises(TypeError):
            store.frontier_breakpoints(
                "main",
                series,
                base_scenario(weights={"x": True}),
                "total_budget",
                (0, 1),
                0,
                2,
                (),
                (),
                10,
            )
        for bad in (-1, float("nan"), float("inf")):
            with self.subTest(weight=repr(bad)):
                with self.assertRaises(ValueError):
                    store.frontier_breakpoints(
                        "main",
                        series,
                        base_scenario(weights={"x": bad}),
                        "total_budget",
                        (0, 1),
                        0,
                        2,
                        (),
                        (),
                        10,
                    )
        for bad in (True, "1", None):
            with self.subTest(total=repr(bad)):
                with self.assertRaises(TypeError):
                    store.frontier_breakpoints(
                        "main",
                        series,
                        base_scenario(total_budget=bad),
                        "total_budget",
                        (0, 1),
                        0,
                        2,
                        (),
                        (),
                        10,
                    )
        for bad in (-1, float("nan"), float("inf")):
            with self.subTest(total=repr(bad)):
                with self.assertRaises(ValueError):
                    store.frontier_breakpoints(
                        "main",
                        series,
                        base_scenario(total_budget=bad),
                        "total_budget",
                        (0, 1),
                        0,
                        2,
                        (),
                        (),
                        10,
                    )
        with self.assertRaises(TypeError):
            store.frontier_breakpoints(
                "main",
                series,
                base_scenario(key_budgets=()),
                "total_budget",
                (0, 1),
                0,
                2,
                (),
                (),
                10,
            )
        with self.assertRaises(ValueError):
            store.frontier_breakpoints(
                "main",
                series,
                base_scenario(
                    weights={"x": 1}, key_budgets={"z": 1}
                ),
                "total_budget",
                (0, 1),
                0,
                2,
                (),
                (),
                10,
            )

    # --- Axis validation --------------------------------------------------

    def test_axis_type_and_empty(self) -> None:
        store = make_store()
        for bad in (1, None, (), [], 1.0):
            with self.subTest(axis=repr(bad)):
                with self.assertRaises(TypeError):
                    store.frontier_breakpoints(
                        "main",
                        make_series(),
                        base_scenario(),
                        bad,
                        (0, 1),
                        0,
                        2,
                        (),
                        (),
                        10,
                    )
        with self.assertRaises(ValueError):
            store.frontier_breakpoints(
                "main",
                make_series(),
                base_scenario(),
                "",
                (0, 1),
                0,
                2,
                (),
                (),
                10,
            )

    def test_unknown_axis_raises_value_error(self) -> None:
        store = make_store()
        for bad in ("nope", "X", "total"):
            with self.subTest(axis=bad):
                with self.assertRaises(ValueError):
                    store.frontier_breakpoints(
                        "main",
                        make_series(),
                        base_scenario(),
                        bad,
                        (0, 1),
                        0,
                        2,
                        (),
                        (),
                        10,
                    )

    def test_total_budget_axis_accepted(self) -> None:
        store = make_store()
        store.frontier_breakpoints(
            "main",
            make_series(),
            base_scenario(),
            "total_budget",
            (0, 1),
            0,
            2,
            (),
            (),
            10,
        )

    # --- Search-parameter validation mirrors search_combinations ---------

    def test_search_parameters_keep_existing_validation(self) -> None:
        store = make_store()
        series = make_series()
        scenario = base_scenario()
        for bad in (True, 1.0, "1", None):
            with self.subTest(min_size=repr(bad)):
                with self.assertRaises(TypeError):
                    store.frontier_breakpoints(
                        "main", series, scenario, "total_budget",
                        (0, 1), bad, 2, (), (), 10,
                    )
            with self.subTest(max_size=repr(bad)):
                with self.assertRaises(TypeError):
                    store.frontier_breakpoints(
                        "main", series, scenario, "total_budget",
                        (0, 1), 0, bad, (), (), 10,
                    )
            with self.subTest(limit=repr(bad)):
                with self.assertRaises(TypeError):
                    store.frontier_breakpoints(
                        "main", series, scenario, "total_budget",
                        (0, 1), 0, 2, (), (), bad,
                    )
        for call in (
            lambda: store.frontier_breakpoints(
                "main", series, scenario, "total_budget",
                (0, 1), -1, 2, (), (), 10,
            ),
            lambda: store.frontier_breakpoints(
                "main", series, scenario, "total_budget",
                (0, 1), 2, 1, (), (), 10,
            ),
            lambda: store.frontier_breakpoints(
                "main", series, scenario, "total_budget",
                (0, 1), 0, 3, (), (), 10,
            ),
            lambda: store.frontier_breakpoints(
                "main", series, scenario, "total_budget",
                (0, 1), 0, 2, (), (), 0,
            ),
            lambda: store.frontier_breakpoints(
                "main", series, scenario, "total_budget",
                (0, 1), 0, 2, ("ghost",), (), 10,
            ),
            lambda: store.frontier_breakpoints(
                "main",
                series,
                scenario,
                "total_budget",
                (0, 1),
                0,
                2,
                (),
                (("feature", "ghost"),),
                10,
            ),
        ):
            with self.assertRaises(ValueError):
                call()

    def test_required_and_exclusive_container_types(self) -> None:
        store = make_store()
        scenario = base_scenario()
        for bad in ([], None, "feature", 1):
            with self.subTest(required=repr(bad)):
                with self.assertRaises(TypeError):
                    store.frontier_breakpoints(
                        "main", make_series(), scenario, "total_budget",
                        (0, 1), 0, 2, bad, (), 10,
                    )
            with self.subTest(pairs=repr(bad)):
                with self.assertRaises(TypeError):
                    store.frontier_breakpoints(
                        "main", make_series(), scenario, "total_budget",
                        (0, 1), 0, 2, (), bad, 10,
                    )
        with self.assertRaises(TypeError):
            store.frontier_breakpoints(
                "main", make_series(), scenario, "total_budget",
                (0, 1), 0, 2, (), (("feature",),), 10,
            )

    def test_alignment_error_matches_search(self) -> None:
        store = make_store()
        scenario = base_scenario()
        with self.assertRaises(ValueError):
            store.frontier_breakpoints(
                "main",
                {
                    "feature": (("m1", "f1"), ("m2", "f2")),
                    "side": (("m1", "s1"),),
                },
                scenario,
                "total_budget",
                (0, 1),
                0,
                2,
                (),
                (),
                10,
            )
        with self.assertRaises(ValueError):
            store.frontier_breakpoints(
                "main",
                {
                    "feature": (("m2", "f2"),),
                    "side": (("m1", "s1"),),
                },
                scenario,
                "total_budget",
                (0, 1),
                0,
                2,
                (),
                (),
                10,
            )

    def test_all_inputs_validated_before_branch_lookups(self) -> None:
        store = make_store()
        series = make_series()
        # Numeric scenario errors beat an unknown reference branch.
        with self.assertRaises(ValueError):
            store.frontier_breakpoints(
                "ghost",
                series,
                base_scenario(total_budget=-1),
                "total_budget",
                (0, 1),
                0,
                2,
                (),
                (),
                10,
            )
        # Value-series errors beat an unknown reference branch.
        with self.assertRaises(ValueError):
            store.frontier_breakpoints(
                "ghost",
                series,
                base_scenario(),
                "total_budget",
                (1, 0),
                0,
                2,
                (),
                (),
                10,
            )
        # Size errors beat unknown branches.
        with self.assertRaises(TypeError):
            store.frontier_breakpoints(
                "ghost",
                series,
                base_scenario(),
                "total_budget",
                (0, 1),
                True,
                2,
                (),
                (),
                10,
            )
        # Member-constraint errors beat unknown branches.
        with self.assertRaises(ValueError):
            store.frontier_breakpoints(
                "ghost",
                series,
                base_scenario(),
                "total_budget",
                (0, 1),
                0,
                2,
                ("ghost-m",),
                (),
                10,
            )

    def test_branch_and_node_lookup_order(self) -> None:
        store = make_store()
        with self.assertRaises(KeyError) as caught:
            store.frontier_breakpoints(
                "ghost",
                make_series(),
                base_scenario(),
                "total_budget",
                (0, 1),
                0,
                2,
                (),
                (),
                10,
            )
        self.assertEqual(caught.exception.args, ("ghost",))
        with self.assertRaises(KeyError) as caught:
            store.frontier_breakpoints(
                "main",
                {"ghost1": (), "ghost2": ()},
                base_scenario(),
                "total_budget",
                (0, 1),
                0,
                0,
                (),
                (),
                10,
            )
        self.assertEqual(caught.exception.args, ("ghost1",))
        with self.assertRaises(KeyError) as caught:
            store.frontier_breakpoints(
                "main",
                {"feature": (("nope1", "f1"),)},
                base_scenario(),
                "total_budget",
                (0, 1),
                0,
                1,
                (),
                (),
                10,
            )
        self.assertEqual(caught.exception.args, ("nope1",))
        # A node outside the named branch's current head closure.
        with self.assertRaises(KeyError):
            store.frontier_breakpoints(
                "main",
                {"feature": (("f2", "f1"),)},
                base_scenario(),
                "total_budget",
                (0, 1),
                0,
                1,
                (),
                (),
                10,
            )

    # --- Detachment and read-only behavior --------------------------------

    def test_results_detached_and_unshared(self) -> None:
        store = make_store()
        series = make_series()
        result = store.frontier_breakpoints(
            "main",
            series,
            base_scenario(),
            "total_budget",
            (15, 16, 100),
            0,
            2,
            (),
            (),
            10,
        )
        pristine = copy.deepcopy(result)
        for point in result["points"]:
            for row in point["frontier"]:
                row["members"] += ("evil",)
                row["contributions"] += (999,)
                for group in row["attributions"]:
                    for attribution in group:
                        attribution["key"] = "evil"
                        attribution["left"]["path"] += ("evil",)
            for row in point["rejected"]:
                row["members"] += ("evil",)
        for record in result["breakpoints"]:
            for side in (record["left"], record["right"]):
                for row in side["frontier"]:
                    row["members"] += ("evil",)
            record["entered"] += (("evil",),)
            record["affected"] += (("evil",),)
            record["member_deltas"][0]["delta"] = (999,)
        fresh = store.frontier_breakpoints(
            "main",
            series,
            base_scenario(),
            "total_budget",
            (15, 16, 100),
            0,
            2,
            (),
            (),
            10,
        )
        self.assertEqual(fresh, pristine)

        # Point rows, left and right copies are independent objects even
        # when item-wise equal.
        record = fresh["breakpoints"][0]
        left_side = next(
            row
            for row in record["left"]["frontier"]
            if row["members"] == ()
        )
        right_side = next(
            row
            for row in record["right"]["frontier"]
            if row["members"] == ()
        )
        point_side = next(
            row
            for point in fresh["points"]
            if point["value"] == 15
            for row in point["frontier"]
            if row["members"] == ()
        )
        self.assertIsNot(left_side, right_side)
        self.assertIsNot(left_side, point_side)
        self.assertIsNot(right_side, point_side)

    def test_inputs_are_not_retained(self) -> None:
        store = make_store()
        weights = {"x": 2, "y": 3, "z": 1}
        scenario = base_scenario(weights=weights)
        result = store.frontier_breakpoints(
            "main",
            make_series(),
            scenario,
            "x",
            (0, 1, 2),
            0,
            2,
            (),
            (),
            10,
        )
        weights["y"] = 999
        scenario["total_budget"] = 0
        # The point risks were computed with the original weights/budget.
        first_pair = next(
            row
            for row in result["points"][0]["frontier"]
            if row["members"] == ("feature", "side")
        )
        self.assertEqual(first_pair["risk"], 10)

    def test_query_is_read_only(self) -> None:
        store = make_store()
        heads = dict(store._heads)
        appends = copy.deepcopy(store._appends)
        merges = copy.deepcopy(store._merges)
        replay_main = store.replay("main")
        audit = store.audit_log()
        series = make_series()
        scenario = base_scenario()
        store.frontier_breakpoints(
            "main", series, scenario, "total_budget", (0, 100),
            0, 2, (), (), 10,
        )
        # Failing calls leave state untouched as well.
        with self.assertRaises(ValueError):
            store.frontier_breakpoints(
                "main", series, scenario, "total_budget", (0, 100),
                0, 2, (), (), 1,
            )
        with self.assertRaises(KeyError):
            store.frontier_breakpoints(
                "main",
                {"feature": (("m1", "nope"),)},
                scenario,
                "total_budget",
                (0, 100),
                0,
                1,
                (),
                (),
                10,
            )
        with self.assertRaises(TypeError):
            store.frontier_breakpoints(
                "main", series, scenario, "total_budget", [0, 100],
                0, 2, (), (), 10,
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
        store.frontier_breakpoints(
            "main", series, scenario, "total_budget", (0, 100),
            0, 2, (), (), 10,
        )
        self.assertEqual(
            store.search_combinations(
                "main", series, WEIGHTS, 17, {}, 0, 2, (), (), 10
            ),
            search,
        )
        sensitivity = store.frontier_sensitivity(
            "main", series,
            (
                {
                    "name": "a",
                    "weights": WEIGHTS,
                    "total_budget": 17,
                    "key_budgets": {},
                },
            ),
            0,
            2,
            (),
            (),
            10,
        )
        self.assertEqual(
            [row["members"] for row in sensitivity["scenarios"][0]["frontier"]],
            [("side",), ()],
        )


if __name__ == "__main__":
    unittest.main()
