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
    weights: dict | None = None,
    total_budget: int | float = 100,
    key_budgets: dict | None = None,
) -> dict:
    return {
        "weights": WEIGHTS if weights is None else weights,
        "total_budget": total_budget,
        "key_budgets": {} if key_budgets is None else key_budgets,
    }


def call(store, **overrides):
    kwargs = dict(
        reference="main",
        series=make_series(),
        base_scenario=scenario(),
        axis="total_budget",
        values=(15, 16, 17, 100),
        min_size=0,
        max_size=2,
        required=(),
        exclusive_pairs=(),
        limit=10,
    )
    kwargs.update(overrides)
    return store.explain_frontier_breakpoints(**kwargs)


class ExplainFrontierBreakpointsSignatureTests(unittest.TestCase):
    def test_public_signature_has_no_defaults(self) -> None:
        parameters = list(
            inspect.signature(
                BranchStore.explain_frontier_breakpoints
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

    def test_returns_tuple_of_fresh_dicts(self) -> None:
        store = make_store()
        result = call(store)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        for record in result:
            self.assertIsInstance(record, dict)


class ExplainFrontierBreakpointsShapeTests(unittest.TestCase):
    def test_interval_record_key_order(self) -> None:
        store = make_store()
        result = call(store)
        self.assertEqual(
            [(r["left_bound"], r["right_bound"]) for r in result],
            [(15, 16), (17, 100)],
        )
        for record in result:
            self.assertEqual(
                list(record), ["left_bound", "right_bound", "changes"]
            )
            self.assertIsInstance(record["changes"], tuple)

    def test_change_and_side_key_order(self) -> None:
        store = make_store()
        result = call(store)
        for record in result:
            for change in record["changes"]:
                self.assertEqual(
                    list(change),
                    [
                        "members",
                        "change_type",
                        "checkpoint",
                        "cause_key",
                        "attributions",
                        "left",
                        "right",
                    ],
                )
                for side in (change["left"], change["right"]):
                    self.assertEqual(
                        list(side),
                        ["feasible", "risk", "dominators", "overrun"],
                    )

    def test_changes_follow_candidate_enumeration_once_each(self) -> None:
        store = make_store()
        result = call(store)
        # Enumeration order over the pool is: (), feature, side, pair.
        second = result[1]
        self.assertEqual(
            [change["members"] for change in second["changes"]],
            [("feature",), ("side",), ("feature", "side")],
        )
        members = [change["members"] for change in second["changes"]]
        self.assertEqual(len(members), len(set(members)))

    def test_changes_match_entered_exited_affected_union(self) -> None:
        store = make_store()
        explain = call(store)
        turning = store.frontier_breakpoints(
            "main",
            make_series(),
            scenario(),
            "total_budget",
            (15, 16, 17, 100),
            0,
            2,
            (),
            (),
            10,
        )
        self.assertEqual(len(explain), len(turning["breakpoints"]))
        for explained, breakpoint in zip(
            explain, turning["breakpoints"]
        ):
            self.assertEqual(
                explained["left_bound"], breakpoint["left_bound"]
            )
            self.assertEqual(
                explained["right_bound"], breakpoint["right_bound"]
            )
            expected = set(
                breakpoint["entered"]
                + breakpoint["exited"]
                + breakpoint["affected"]
            )
            self.assertEqual(
                {change["members"] for change in explained["changes"]},
                expected,
            )

    def test_no_turning_interval_returns_empty_tuple(self) -> None:
        store = make_store()
        result = call(store, values=(16, 17))
        self.assertEqual(result, ())

    def test_no_checkpoints_returns_empty_tuple(self) -> None:
        store = make_store()
        result = call(
            store,
            series={"feature": (), "side": ()},
            base_scenario=scenario(total_budget=0, key_budgets={"x": 0}),
            values=(0, 100),
        )
        self.assertEqual(result, ())


class ExplainFrontierBreakpointsChangeTypeTests(unittest.TestCase):
    def test_feasibility_flip_labeled_first_and_anchors_first_verdict(self) -> None:
        store = make_store()
        result = call(store)
        # 15 -> 16: side flips at checkpoint 1 (checkpoint 0 breaches at
        # both budgets), and the pair is infeasible throughout this band.
        change = result[0]["changes"][0]
        self.assertEqual(change["members"], ("side",))
        self.assertEqual(change["change_type"], "feasibility")
        self.assertEqual(change["checkpoint"], 1)

        # 17 -> 100: the pair becomes feasible at checkpoint 0, the first
        # checkpoint whose budget verdict differs across the interval.
        pair = next(
            change
            for change in result[1]["changes"]
            if change["members"] == ("feature", "side")
        )
        self.assertEqual(pair["change_type"], "feasibility")
        self.assertEqual(pair["checkpoint"], 0)
        feature = next(
            change
            for change in result[1]["changes"]
            if change["members"] == ("feature",)
        )
        self.assertEqual(feature["change_type"], "feasibility")
        self.assertEqual(feature["checkpoint"], 1)

    def test_domination_change_uses_last_checkpoint(self) -> None:
        store = make_store()
        result = call(store)
        side = next(
            change
            for change in result[1]["changes"]
            if change["members"] == ("side",)
        )
        # side stays feasible but the pair newly dominates it at the last
        # checkpoint, so feasibility does not change while domination does.
        self.assertEqual(side["change_type"], "domination")
        self.assertEqual(side["checkpoint"], 1)

    def test_frontier_flip_with_feasibility_flip_is_feasibility(self) -> None:
        store = make_store()
        result = call(store)
        # side enters the frontier between 15 and 16; its feasibility also
        # flips, so feasibility wins the type precedence.
        side = result[0]["changes"][0]
        self.assertFalse(side["left"]["feasible"])
        self.assertTrue(side["right"]["feasible"])
        self.assertEqual(side["change_type"], "feasibility")

    def test_domination_only_fixture(self) -> None:
        # Mirrors the breakpoint suite's domination-only scenario: the
        # frontier membership of 'a' never changes, but its dominator set
        # does, so the explanation labels it a domination change at the
        # only (last) checkpoint.
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
        result = store.explain_frontier_breakpoints(
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
        self.assertEqual(len(result), 1)
        changes = result[0]["changes"]
        self.assertEqual([c["members"] for c in changes], [("a",)])
        change = changes[0]
        self.assertEqual(change["change_type"], "domination")
        self.assertEqual(change["checkpoint"], 0)
        self.assertTrue(change["left"]["feasible"])
        self.assertTrue(change["right"]["feasible"])


class ExplainFrontierBreakpointsSideEvidenceTests(unittest.TestCase):
    def test_feasible_side_carries_risk_and_no_overrun(self) -> None:
        store = make_store()
        result = call(store)
        side = result[0]["changes"][0]
        right = side["right"]
        self.assertTrue(right["feasible"])
        self.assertEqual(right["risk"], 16)
        self.assertIsNone(right["overrun"])

    def test_infeasible_side_carries_overrun_and_no_risk(self) -> None:
        store = make_store()
        result = call(store)
        side = result[0]["changes"][0]
        left = side["left"]
        self.assertFalse(left["feasible"])
        self.assertIsNone(left["risk"])
        self.assertEqual(left["overrun"], 1)

    def test_dominators_follow_frontier_order(self) -> None:
        store = make_store()
        result = call(store)
        feature = next(
            change
            for change in result[1]["changes"]
            if change["members"] == ("feature",)
        )
        # Infeasible on the left: no dominators. On the right both the
        # larger pair and the same-size, lower-risk side dominate it, in
        # descending member count then ascending member tuple order.
        self.assertEqual(feature["left"]["dominators"], ())
        self.assertEqual(
            feature["right"]["dominators"],
            (("feature", "side"), ("side",)),
        )
        side = next(
            change
            for change in result[1]["changes"]
            if change["members"] == ("side",)
        )
        self.assertEqual(side["left"]["dominators"], ())
        self.assertEqual(
            side["right"]["dominators"], (("feature", "side"),)
        )

    def test_sides_match_turning_point_rows_when_present(self) -> None:
        store = make_store()
        explain = call(store)
        turning = store.frontier_breakpoints(
            "main",
            make_series(),
            scenario(),
            "total_budget",
            (15, 16, 17, 100),
            0,
            2,
            (),
            (),
            10,
        )
        for explained, breakpoint in zip(
            explain, turning["breakpoints"]
        ):
            points = {
                "left": breakpoint["left"],
                "right": breakpoint["right"],
            }
            for change in explained["changes"]:
                for side_name, point in points.items():
                    side = change[side_name]
                    frontier_row = next(
                        (
                            row
                            for row in point["frontier"]
                            if row["members"] == change["members"]
                        ),
                        None,
                    )
                    rejected_row = next(
                        (
                            row
                            for row in point["rejected"]
                            if row["members"] == change["members"]
                        ),
                        None,
                    )
                    if frontier_row is not None:
                        self.assertTrue(side["feasible"])
                        self.assertEqual(
                            side["risk"], frontier_row["risk"]
                        )
                        self.assertIsNone(side["overrun"])
                    if rejected_row is not None:
                        self.assertFalse(side["feasible"])
                        self.assertIsNone(side["risk"])
                        self.assertEqual(
                            side["overrun"], rejected_row["overrun"]
                        )

    def test_float_overrun_is_preserved_not_coerced(self) -> None:
        graph = EventGraph()
        graph.add("root", 0, (), {})
        store = BranchStore(graph)
        store.create("main", "root")
        store.create("a", "root")
        store.append("main", "r1", 1, {})
        store.append("a", "a1", 2, {"x": 10})
        series = {"a": (("r1", "a1"),)}
        result = store.explain_frontier_breakpoints(
            "main",
            series,
            {
                "weights": {"x": 0.5},
                "total_budget": 100.0,
                "key_budgets": {},
            },
            "total_budget",
            (4.0, 6.0),
            0,
            1,
            (),
            (),
            10,
        )
        change = result[0]["changes"][0]
        self.assertFalse(change["left"]["feasible"])
        self.assertEqual(change["left"]["overrun"], 1.0)
        self.assertTrue(change["right"]["feasible"])
        self.assertEqual(change["right"]["risk"], 5.0)


class ExplainFrontierBreakpointsCauseKeyTests(unittest.TestCase):
    def test_total_budget_axis_picks_largest_nonzero_contribution(self) -> None:
        store = make_store()
        result = call(store)
        side = result[0]["changes"][0]
        # At checkpoint 1 the reference m2 holds (x=1, y=2, z=7) and side
        # ends at s2 with (x=3, y=-1, z=4), so side's differences are
        # x=2, y=-3, z=-3. The cause key is chosen on the unweighted
        # aggregate contribution: y and z tie on magnitude 3, larger than
        # x's 2, and the smallest Unicode code point breaks the tie.
        self.assertEqual(side["cause_key"], "y")

    def test_total_budget_tie_breaks_by_smallest_code_point(self) -> None:
        # One member, one checkpoint, equal-magnitude non-zero aggregates
        # for x and y: the smallest code point wins.
        graph = EventGraph()
        graph.add("root", 0, (), {})
        store = BranchStore(graph)
        store.create("main", "root")
        store.create("a", "root")
        store.append("main", "r1", 1, {})
        store.append("a", "a1", 2, {"x": 2, "y": -2})
        series = {"a": (("r1", "a1"),)}
        result = store.explain_frontier_breakpoints(
            "main",
            series,
            {
                "weights": {"x": 1, "y": 1},
                "total_budget": 100,
                "key_budgets": {},
            },
            "total_budget",
            (3, 5),
            0,
            1,
            (),
            (),
            10,
        )
        change = result[0]["changes"][0]
        self.assertEqual(change["members"], ("a",))
        self.assertEqual(change["checkpoint"], 0)
        self.assertEqual(change["cause_key"], "x")

    def test_weight_axis_names_perturbed_key_when_a_member_contributes(self) -> None:
        store = make_store()
        result = call(
            store,
            base_scenario=scenario(total_budget=17),
            axis="x",
            values=(0, 1, 2),
        )
        pair = next(
            change
            for change in result[0]["changes"]
            if change["members"] == ("feature", "side")
        )
        self.assertEqual(pair["change_type"], "feasibility")
        self.assertEqual(pair["cause_key"], "x")
        feature = next(
            change
            for change in result[0]["changes"]
            if change["members"] == ("feature",)
        )
        self.assertEqual(feature["cause_key"], "x")

    def test_weight_axis_none_when_no_involved_member_contributes(self) -> None:
        # Perturbing y changes the frontier through x-only members: the
        # empty subset and 'a' have no y contribution anywhere, so their
        # cause key is None and no attribution is returned.
        graph = EventGraph()
        graph.add("root", 0, (), {})
        store = BranchStore(graph)
        store.create("main", "root")
        store.create("a", "root")
        store.create("b", "root")
        store.append("main", "r1", 1, {})
        store.append("a", "a1", 2, {"x": 10})
        store.append("b", "b1", 3, {"y": 10})
        series = {"a": (("r1", "a1"),), "b": (("r1", "b1"),)}
        result = store.explain_frontier_breakpoints(
            "main",
            series,
            {
                "weights": {"x": 1, "y": 1},
                "total_budget": 100,
                "key_budgets": {},
            },
            "y",
            (0, 1),
            0,
            2,
            (),
            (),
            10,
        )
        members = {
            change["members"]: change for change in result[0]["changes"]
        }
        self.assertEqual(set(members), {(), ("a",)})
        for change in members.values():
            self.assertEqual(change["change_type"], "domination")
            self.assertIsNone(change["cause_key"])
            self.assertEqual(change["attributions"], ())


class ExplainFrontierBreakpointsAttributionTests(unittest.TestCase):
    def test_weight_axis_first_verdict_difference_locates_checkpoint(self) -> None:
        # Two checkpoints: the weight perturbation does not change the
        # verdict at checkpoint 0 (both values pass), but flips the
        # total-budget verdict at checkpoint 1. The feasibility change
        # must anchor at 1, not at the last-only guess.
        graph = EventGraph()
        graph.add("root", 0, (), {})
        store = BranchStore(graph)
        store.create("main", "root")
        store.create("a", "root")
        store.append("main", "r1", 1, {})
        store.append("main", "r2", 2, {})
        store.append("a", "a1", 3, {"x": 10})
        store.append("a", "a2", 4, {"x": 10})
        series = {"a": (("r1", "a1"), ("r2", "a2"))}
        result = store.explain_frontier_breakpoints(
            "main",
            series,
            {
                "weights": {"x": 1},
                "total_budget": 15,
                "key_budgets": {},
            },
            "x",
            (0, 1),
            0,
            1,
            (),
            (),
            10,
        )
        change = next(
            change
            for change in result[0]["changes"]
            if change["members"] == ("a",)
        )
        self.assertEqual(change["members"], ("a",))
        self.assertEqual(change["change_type"], "feasibility")
        self.assertEqual(change["checkpoint"], 1)
        self.assertTrue(change["left"]["feasible"])
        self.assertFalse(change["right"]["feasible"])

    def test_key_budget_breach_is_stable_along_total_budget_axis(self) -> None:
        # A combination rejected by a per-key budget stays rejected at
        # every total-budget value: its verdict never differs along that
        # axis, so it must not be explained even while another combination
        # crosses the total-budget line. feature's checkpoint-1 z
        # difference is -7 (f2 has no z against m2's 7), so a z cap of 5
        # rejects it regardless of the total budget; side (|z|=3) still
        # turns on the total budget between 0 and 50.
        store = make_store()
        result = call(
            store,
            base_scenario=scenario(total_budget=100, key_budgets={"z": 5}),
            values=(0, 50, 100),
        )
        explained_members = {
            change["members"]
            for record in result
            for change in record["changes"]
        }
        self.assertIn(("side",), explained_members)
        self.assertNotIn(("feature",), explained_members)

    def test_one_attribution_per_member_in_member_order(self) -> None:
        store = make_store()
        result = call(store)
        pair = next(
            change
            for change in result[1]["changes"]
            if change["members"] == ("feature", "side")
        )
        self.assertEqual(len(pair["attributions"]), 2)
        for attribution in pair["attributions"]:
            self.assertEqual(
                list(attribution), ["key", "fork", "left", "right"]
            )
            self.assertEqual(attribution["key"], pair["cause_key"])

    def test_attributions_reuse_checkpoint_historical_attribution(self) -> None:
        store = make_store()
        series = make_series()
        result = store.explain_frontier_breakpoints(
            "main",
            series,
            scenario(total_budget=17),
            "x",
            (0, 1),
            0,
            2,
            (),
            (),
            10,
        )
        pair = next(
            change
            for change in result[0]["changes"]
            if change["members"] == ("feature", "side")
        )
        # A weight-axis feasibility flip for the pair anchors at the
        # first checkpoint; each attribution must equal the existing
        # historical divergence attribution for that member's nodes.
        feature_nodes = series["feature"][pair["checkpoint"]]
        side_nodes = series["side"][pair["checkpoint"]]
        feature_attribution = (
            store.attribute_divergences_at(
                "main", feature_nodes[0], "feature", feature_nodes[1], ("x",)
            )[0]
        )
        side_attribution = store.attribute_divergences_at(
            "main", side_nodes[0], "side", side_nodes[1], ("x",)
        )[0]
        self.assertEqual(pair["attributions"][0], feature_attribution)
        self.assertEqual(pair["attributions"][1], side_attribution)

    def test_total_axis_attribution_uses_selected_checkpoint_nodes(self) -> None:
        store = make_store()
        series = make_series()
        result = call(store)
        # The side feasibility flip between 15 and 16 anchors at cp1, so
        # the reused attribution is the cp1 (m2 vs s2) one.
        side = result[0]["changes"][0]
        expected = store.attribute_divergences_at(
            "main", "m2", "side", "s2", (side["cause_key"],)
        )[0]
        self.assertEqual(side["attributions"][0], expected)

    def test_attributions_are_independent_detached_copies(self) -> None:
        store = make_store()
        result_a = call(store)
        result_b = call(store)
        pair_a = next(
            change
            for change in result_a[1]["changes"]
            if change["members"] == ("feature", "side")
        )
        pair_b = next(
            change
            for change in result_b[1]["changes"]
            if change["members"] == ("feature", "side")
        )
        self.assertEqual(pair_a["attributions"], pair_b["attributions"])
        for first, second in zip(
            pair_a["attributions"], pair_b["attributions"]
        ):
            self.assertIsNot(first, second)
            self.assertIsNot(first["left"], second["left"])
            self.assertIsNot(first["left"]["path"], second["left"]["path"])
        # The two members' attributions never share containers either.
        first, second = pair_a["attributions"]
        self.assertIsNot(first, second)
        self.assertIsNot(first["left"], second["left"])


class ExplainFrontierBreakpointsNoneEvidenceTests(unittest.TestCase):
    def test_missing_risk_and_overrun_are_none_not_zero(self) -> None:
        store = make_store()
        result = call(store)
        for record in result:
            for change in record["changes"]:
                for side in (change["left"], change["right"]):
                    self.assertIsInstance(side["feasible"], bool)
                    if side["feasible"]:
                        self.assertIsNotNone(side["risk"])
                        self.assertIsNone(side["overrun"])
                    else:
                        self.assertIsNone(side["risk"])
                        self.assertIsNotNone(side["overrun"])
                    self.assertIsInstance(side["dominators"], tuple)

    def test_empty_attributions_when_cause_key_none(self) -> None:
        graph = EventGraph()
        graph.add("root", 0, (), {})
        store = BranchStore(graph)
        store.create("main", "root")
        store.create("a", "root")
        store.create("b", "root")
        store.append("main", "r1", 1, {})
        store.append("a", "a1", 2, {"x": 10})
        store.append("b", "b1", 3, {"y": 10})
        series = {"a": (("r1", "a1"),), "b": (("r1", "b1"),)}
        result = store.explain_frontier_breakpoints(
            "main",
            series,
            {"weights": {"x": 1, "y": 1}, "total_budget": 100, "key_budgets": {}},
            "y",
            (0, 1),
            0,
            2,
            (),
            (),
            10,
        )
        for record in result:
            for change in record["changes"]:
                if change["cause_key"] is None:
                    self.assertEqual(change["attributions"], ())


class ExplainFrontierBreakpointsValidationTests(unittest.TestCase):
    """Validation must raise exactly what frontier_breakpoints raises."""

    def _breakpoint_kwargs(self, **overrides):
        kwargs = dict(
            reference="main",
            series=make_series(),
            base_scenario=scenario(),
            axis="total_budget",
            values=(0, 1),
            min_size=0,
            max_size=2,
            required=(),
            exclusive_pairs=(),
            limit=10,
        )
        kwargs.update(overrides)
        return kwargs

    def _assert_parity(self, **overrides) -> None:
        kwargs = self._breakpoint_kwargs(**overrides)
        with self.assertRaises(Exception) as turning_expected:
            make_store().frontier_breakpoints(**kwargs)
        with self.assertRaises(type(turning_expected.exception)):
            make_store().explain_frontier_breakpoints(**kwargs)

    def test_reference_type_and_empty(self) -> None:
        for bad in (1, None, ()):
            self._assert_parity(reference=bad)
        self._assert_parity(reference="")

    def test_series_container(self) -> None:
        for bad in ([], None, (), "s", 1):
            self._assert_parity(series=bad)

    def test_base_scenario_container_and_fields(self) -> None:
        for bad in (None, 42, (), [], "s"):
            self._assert_parity(base_scenario=bad)
        base = scenario()
        self._assert_parity(
            base_scenario={k: v for k, v in base.items() if k != "weights"}
        )
        self._assert_parity(base_scenario={**base, "extra": 1})

    def test_bool_is_not_a_number(self) -> None:
        self._assert_parity(values=(0, True))
        self._assert_parity(min_size=True)
        self._assert_parity(max_size=1.0)
        self._assert_parity(limit=False)
        self._assert_parity(
            base_scenario=scenario(total_budget=True)
        )

    def test_values_need_two_finite_ordered(self) -> None:
        for bad in (
            [0, 1],
            (),
            (1,),
            (True, 1),
            (-1, 0),
            (0, float("nan")),
            (0, float("inf")),
            (1, 0),
            (0, 0),
        ):
            self._assert_parity(values=bad)

    def test_illegal_axis(self) -> None:
        for bad in (1, None, 1.0, ()):
            self._assert_parity(axis=bad)
        self._assert_parity(axis="")
        self._assert_parity(axis="nope")

    def test_member_constraint_conflicts(self) -> None:
        self._assert_parity(required=("ghost",))
        self._assert_parity(exclusive_pairs=(("feature", "ghost"),))
        self._assert_parity(exclusive_pairs=(("feature", "feature"),))
        self._assert_parity(
            required=("feature", "side"),
            exclusive_pairs=(("feature", "side"),),
        )

    def test_size_window_and_limit(self) -> None:
        self._assert_parity(min_size=-1)
        self._assert_parity(max_size=3)
        self._assert_parity(max_size=0, min_size=1)
        self._assert_parity(limit=0)
        self._assert_parity(limit=3)

    def test_misaligned_checkpoints(self) -> None:
        self._assert_parity(
            series={
                "feature": (("m1", "f1"), ("m2", "f2")),
                "side": (("m1", "s1"),),
            }
        )
        self._assert_parity(
            series={
                "feature": (("m2", "f2"),),
                "side": (("m1", "s1"),),
            }
        )

    def test_unknown_reference_branch_is_key_error(self) -> None:
        kwargs = self._breakpoint_kwargs(reference="ghost")
        with self.assertRaises(KeyError) as caught:
            make_store().explain_frontier_breakpoints(**kwargs)
        self.assertEqual(caught.exception.args, ("ghost",))

    def test_unknown_series_branch_is_key_error(self) -> None:
        kwargs = self._breakpoint_kwargs(
            series={"ghost1": (), "ghost2": ()}, max_size=0
        )
        with self.assertRaises(KeyError) as caught:
            make_store().explain_frontier_breakpoints(**kwargs)
        self.assertEqual(caught.exception.args, ("ghost1",))

    def test_unknown_node_is_key_error(self) -> None:
        kwargs = self._breakpoint_kwargs(
            series={"feature": (("nope1", "f1"),)}, max_size=1
        )
        with self.assertRaises(KeyError) as caught:
            make_store().explain_frontier_breakpoints(**kwargs)
        self.assertEqual(caught.exception.args, ("nope1",))

    def test_node_outside_current_head_closure_is_key_error(self) -> None:
        kwargs = self._breakpoint_kwargs(
            series={"feature": (("f2", "f1"),)}, max_size=1
        )
        with self.assertRaises(KeyError):
            make_store().explain_frontier_breakpoints(**kwargs)

    def test_inputs_validated_before_branch_lookups(self) -> None:
        # Scenario and value errors beat an unknown reference branch.
        with self.assertRaises(ValueError):
            make_store().explain_frontier_breakpoints(
                "ghost",
                make_series(),
                scenario(total_budget=-1),
                "total_budget",
                (0, 1),
                0,
                2,
                (),
                (),
                10,
            )
        with self.assertRaises(ValueError):
            make_store().explain_frontier_breakpoints(
                "ghost",
                make_series(),
                scenario(),
                "total_budget",
                (1, 0),
                0,
                2,
                (),
                (),
                10,
            )

    def test_limit_exceeded_raises_value_error_with_no_partial_result(self) -> None:
        store = make_store()
        with self.assertRaises(ValueError):
            call(store, values=(0, 100), limit=3)
        # State and other queries still answer exactly as before.
        again = call(store, values=(0, 100), limit=4)
        self.assertIsInstance(again, tuple)


class ExplainFrontierBreakpointsDetachmentTests(unittest.TestCase):
    def test_results_detached_and_unshared(self) -> None:
        store = make_store()
        result = call(store, values=(15, 16, 100))
        pristine = copy.deepcopy(result)
        for record in result:
            record["changes"][0]["members"] += ("evil",)
            for change in record["changes"]:
                change["left"]["dominators"] += (("evil",),)
                for attribution in change["attributions"]:
                    attribution["key"] = "evil"
                    attribution["left"]["path"] += ("evil",)
            record["changes"][0]["attributions"] += (
                {"key": "evil", "fork": None, "left": {}, "right": {}},
            )
        fresh = call(store, values=(15, 16, 100))
        self.assertEqual(fresh, pristine)

    def test_left_and_right_sides_are_independent_objects(self) -> None:
        store = make_store()
        result = call(store, values=(15, 16))
        change = result[0]["changes"][0]
        self.assertIsNot(change["left"], change["right"])

    def test_repeated_calls_are_itemwise_equal(self) -> None:
        store = make_store()
        first = call(store)
        second = call(store)
        self.assertEqual(first, second)


class ExplainFrontierBreakpointsReadOnlyTests(unittest.TestCase):
    def test_success_and_failure_change_nothing(self) -> None:
        store = make_store()
        heads = dict(store._heads)
        appends = copy.deepcopy(store._appends)
        merges = copy.deepcopy(store._merges)
        replay_main = store.replay("main")
        audit = store.audit_log()
        series = make_series()
        base = scenario()

        call(store)
        with self.assertRaises(ValueError):
            call(store, limit=1)
        with self.assertRaises(KeyError):
            call(
                store,
                series={"feature": (("m1", "nope"),)},
                max_size=1,
            )
        with self.assertRaises(TypeError):
            call(store, values=[0, 100])

        self.assertEqual(dict(store._heads), heads)
        self.assertEqual(copy.deepcopy(store._appends), appends)
        self.assertEqual(copy.deepcopy(store._merges), merges)
        self.assertEqual(store.replay("main"), replay_main)
        self.assertEqual(store.audit_log(), audit)

    def test_existing_search_and_sensitivity_results_unchanged(self) -> None:
        store = make_store()
        series = make_series()
        search = store.search_combinations(
            "main", series, WEIGHTS, 17, {}, 0, 2, (), (), 10
        )
        sensitivity = store.frontier_sensitivity(
            "main",
            series,
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
        call(store)
        self.assertEqual(
            store.search_combinations(
                "main", series, WEIGHTS, 17, {}, 0, 2, (), (), 10
            ),
            search,
        )
        self.assertEqual(
            store.frontier_sensitivity(
                "main",
                series,
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
            ),
            sensitivity,
        )

    def test_audit_records_are_not_created(self) -> None:
        store = make_store()
        before = store.audit_log()
        call(store)
        self.assertEqual(store.audit_log(), before)


if __name__ == "__main__":
    unittest.main()
