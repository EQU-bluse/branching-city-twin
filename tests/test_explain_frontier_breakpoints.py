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


def call(store: BranchStore, **overrides) -> object:
    kwargs = dict(
        reference="main",
        series=make_series(),
        base_scenario=base_scenario(),
        axis="total_budget",
        values=(0, 15, 16, 17, 100),
        min_size=0,
        max_size=2,
        required=(),
        exclusive_pairs=(),
        limit=10,
    )
    kwargs.update(overrides)
    return store.explain_frontier_breakpoints(**kwargs)


class ExplainFrontierBreakpointsTests(unittest.TestCase):
    # --- Signature -------------------------------------------------------

    def test_public_signature_matches_turning_point_query(self) -> None:
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

    # --- Shape and key order ---------------------------------------------

    def test_result_shape_and_key_order(self) -> None:
        store = make_store()
        result = call(store, base_scenario=base_scenario(total_budget=17))
        self.assertIsInstance(result, tuple)
        for interval in result:
            self.assertEqual(
                list(interval), ["left_bound", "right_bound", "changes"]
            )
            self.assertIsInstance(interval["changes"], tuple)
            for change in interval["changes"]:
                self.assertEqual(
                    list(change),
                    [
                        "members",
                        "kind",
                        "checkpoint",
                        "cause_key",
                        "attribution",
                        "left",
                        "right",
                    ],
                )
                self.assertIsInstance(change["members"], tuple)
                self.assertIsInstance(change["attribution"], tuple)
                for side in (change["left"], change["right"]):
                    self.assertEqual(
                        list(side),
                        ["feasible", "risk", "dominators", "overrun"],
                    )
                    self.assertIsInstance(side["feasible"], bool)
                    self.assertIsInstance(side["dominators"], tuple)
                    for dominators in side["dominators"]:
                        self.assertIsInstance(dominators, tuple)
                for attribution in change["attribution"]:
                    self.assertEqual(
                        list(attribution), ["key", "fork", "left", "right"]
                    )

    def test_interval_bounds_match_the_turning_point_query(self) -> None:
        store = make_store()
        explained = call(store)
        breakpoints = store.frontier_breakpoints(
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
            [
                (interval["left_bound"], interval["right_bound"])
                for interval in explained
            ],
            [
                (record["left_bound"], record["right_bound"])
                for record in breakpoints["breakpoints"]
            ],
        )

    def test_changes_follow_candidate_enumeration_order(self) -> None:
        store = make_store()
        result = call(store, base_scenario=base_scenario(total_budget=17))
        for interval in result:
            members = [change["members"] for change in interval["changes"]]
            self.assertEqual(
                members,
                sorted(members, key=lambda combo: (len(combo), combo)),
            )
            # Each combination appears at most once per interval.
            self.assertEqual(len(members), len(set(members)))

    def test_changes_cover_entered_exited_and_affected(self) -> None:
        store = make_store()
        explained = call(store)
        breakpoints = store.frontier_breakpoints(
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
        for interval, record in zip(
            explained, breakpoints["breakpoints"]
        ):
            expected = list(record["entered"])
            for combo in record["exited"]:
                if combo not in expected:
                    expected.append(combo)
            for combo in record["affected"]:
                if combo not in expected:
                    expected.append(combo)
            expected.sort(key=lambda combo: (len(combo), combo))
            self.assertEqual(
                [change["members"] for change in interval["changes"]],
                expected,
            )

    # --- Change kind priority --------------------------------------------

    def test_feasibility_kind_for_a_flip(self) -> None:
        store = make_store()
        # The side combo flips feasible -> rejected on the first interval.
        result = call(
            store,
            base_scenario=base_scenario(total_budget=17),
            axis="x",
            values=(0, 1),
        )
        pair = next(
            change
            for interval in result
            for change in interval["changes"]
            if change["members"] == ("feature", "side")
        )
        self.assertEqual(pair["kind"], "feasibility")
        self.assertTrue(pair["left"]["feasible"])
        self.assertFalse(pair["right"]["feasible"])
        # A feasibility turn locates the first checkpoint whose verdict
        # differs; the pair already breaches total budget at checkpoint 0.
        self.assertEqual(pair["checkpoint"], 0)

    def test_domination_kind_when_only_dominators_move(self) -> None:
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
        (change,) = result[0]["changes"]
        self.assertEqual(change["members"], ("a",))
        self.assertEqual(change["kind"], "domination")
        self.assertTrue(change["left"]["feasible"])
        self.assertTrue(change["right"]["feasible"])
        self.assertEqual(
            change["left"]["dominators"], (("a", "b"), ("b",))
        )
        self.assertEqual(change["right"]["dominators"], (("b",),))

    def test_feasibility_outranks_a_simultaneous_frontier_flip(self) -> None:
        # 15 -> 16: side joins the frontier while flipping infeasible ->
        # feasible. Both a feasibility and a membership change happen; the
        # decision order labels it feasibility.
        store = make_store()
        result = call(store, values=(15, 16))
        self.assertEqual(len(result), 1)
        (change,) = result[0]["changes"]
        self.assertEqual(change["members"], ("side",))
        self.assertEqual(change["kind"], "feasibility")
        self.assertFalse(change["left"]["feasible"])
        self.assertTrue(change["right"]["feasible"])
        # The other-change locating rule does not apply: feasibility locates
        # the first checkpoint whose verdict differs.
        self.assertEqual(change["checkpoint"], 1)

    # --- Locating checkpoint ---------------------------------------------

    def test_feasibility_uses_first_differing_checkpoint(self) -> None:
        store = make_store()
        # feature's final-checkpoint risk is 18: it stays rejected at total
        # 17 and becomes feasible at 100. Checkpoint 0 (risk 2) holds on
        # both sides, so the first verdict difference is checkpoint 1.
        result = call(store, values=(17, 100))
        feature = next(
            change
            for change in result[0]["changes"]
            if change["members"] == ("feature",)
        )
        self.assertEqual(feature["kind"], "feasibility")
        self.assertEqual(feature["checkpoint"], 1)

    def test_other_changes_use_the_last_checkpoint(self) -> None:
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
        self.assertEqual(result[0]["changes"][0]["checkpoint"], 0)

    # --- Cause key --------------------------------------------------------

    def test_total_budget_axis_picks_largest_nonzero_contribution(self) -> None:
        store = make_store()
        result = call(store, values=(15, 16))
        (change,) = result[0]["changes"]
        # side's final aggregate: x diff 2 (weight 2 -> 4), y diff -3
        # (weight 3 -> 9), z diff -3 (weight 1 -> 3). y contributes most.
        self.assertEqual(change["cause_key"], "y")

    def test_cause_key_tie_breaks_by_smallest_code_point(self) -> None:
        graph = EventGraph()
        graph.add("root", 0, (), {})
        store = BranchStore(graph)
        store.create("main", "root")
        store.create("a", "root")
        store.append("main", "r1", 1, {})
        # Equal absolute diffs and equal weights for a and b.
        store.append("a", "a1", 2, {"a": 5, "b": -5})
        series = {"a": (("r1", "a1"),)}
        result = store.explain_frontier_breakpoints(
            "main",
            series,
            {
                "weights": {"a": 1, "b": 1},
                "total_budget": 10,
                "key_budgets": {},
            },
            "total_budget",
            (0, 10),
            0,
            1,
            (),
            (),
            10,
        )
        change = result[0]["changes"][0]
        self.assertEqual(change["cause_key"], "a")

    def test_weight_axis_uses_perturbed_key(self) -> None:
        store = make_store()
        result = call(
            store,
            base_scenario=base_scenario(total_budget=17),
            axis="x",
            values=(1, 2),
        )
        change = next(
            change
            for interval in result
            for change in interval["changes"]
            if change["members"] == ("feature",)
        )
        self.assertEqual(change["cause_key"], "x")

    def test_weight_axis_none_cause_when_key_has_no_contribution(self) -> None:
        graph = EventGraph()
        graph.add("root", 0, (), {})
        store = BranchStore(graph)
        store.create("main", "root")
        store.create("a", "root")
        store.create("b", "root")
        store.append("main", "r1", 1, {})
        # a never touches y; its dominators move when b's y weight changes.
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
        change = result[0]["changes"][0]
        self.assertEqual(change["members"], ("a",))
        self.assertIsNone(change["cause_key"])
        self.assertEqual(change["attribution"], ())

    # --- Left/right evidence ---------------------------------------------

    def test_risk_none_for_infeasible_side(self) -> None:
        store = make_store()
        result = call(store, values=(15, 16))
        change = result[0]["changes"][0]
        self.assertFalse(change["left"]["feasible"])
        self.assertIsNone(change["left"]["risk"])
        self.assertTrue(change["right"]["feasible"])
        self.assertEqual(change["right"]["risk"], 16)

    def test_overrun_magnitude_for_rejecting_side(self) -> None:
        store = make_store()
        result = call(store, values=(15, 16))
        change = result[0]["changes"][0]
        # side final-checkpoint risk 16 against total budget 15: overrun 1.
        self.assertEqual(change["left"]["overrun"], 1)
        self.assertIsNone(change["right"]["overrun"])

    def test_always_binding_key_budget_keeps_the_verdict_constant(self) -> None:
        graph = EventGraph()
        graph.add("root", 0, (), {})
        store = BranchStore(graph)
        store.create("main", "root")
        store.create("a", "root")
        store.append("main", "r1", 1, {})
        store.append("a", "a1", 2, {"x": 4})
        series = {"a": (("r1", "a1"),)}
        # The x key budget of 2 binds at every total-budget value, so the
        # member is rejected on both sides and no interval is reported.
        result = store.explain_frontier_breakpoints(
            "main",
            series,
            {
                "weights": {"x": 1},
                "total_budget": 2,
                "key_budgets": {"x": 2},
            },
            "total_budget",
            (2, 100),
            0,
            1,
            (),
            (),
            10,
        )
        self.assertEqual(result, ())

    def test_dominators_follow_frontier_order(self) -> None:
        store = make_store()
        result = call(store)
        # feature on the last interval (17 -> 100) is feasible on the right
        # and dominated by the pair and side; frontier order puts the pair
        # (2 members) first, then side.
        feature = next(
            change
            for interval in result
            for change in interval["changes"]
            if change["members"] == ("feature",)
        )
        self.assertEqual(
            feature["right"]["dominators"],
            (("feature", "side"), ("side",)),
        )

    # --- Attribution copies ----------------------------------------------

    def test_attribution_reuses_historical_divergence_in_member_order(self) -> None:
        store = make_store()
        result = call(store, values=(15, 16))
        change = result[0]["changes"][0]
        self.assertEqual(change["members"], ("side",))
        (attribution,) = change["attribution"]
        existing = store.attribute_divergence_at(
            "main", "m2", "side", "s2", "y"
        )
        self.assertEqual(attribution, existing)
        self.assertIsNot(attribution, existing)

    def test_pair_attribution_has_one_copy_per_member(self) -> None:
        store = make_store()
        result = call(store)
        pair = next(
            change
            for interval in result
            for change in interval["changes"]
            if change["members"] == ("feature", "side")
        )
        self.assertEqual(len(pair["attribution"]), 2)
        feature_attr, side_attr = pair["attribution"]
        self.assertEqual(
            feature_attr,
            store.attribute_divergence_at(
                "main", "m1", "feature", "f1", pair["cause_key"]
            ),
        )
        self.assertEqual(
            side_attr,
            store.attribute_divergence_at(
                "main", "m1", "side", "s1", pair["cause_key"]
            ),
        )
        self.assertIsNot(feature_attr, side_attr)

    # --- Empty/degenerate cases ------------------------------------------

    def test_no_intervals_returns_empty_tuple(self) -> None:
        store = make_store()
        result = call(store, values=(16, 17))
        self.assertEqual(result, ())

    def test_no_checkpoints_returns_empty_tuple(self) -> None:
        store = make_store()
        result = call(
            store,
            series={"feature": (), "side": ()},
            base_scenario=base_scenario(
                total_budget=0, key_budgets={"x": 0}
            ),
            values=(0, 100),
        )
        self.assertEqual(result, ())

    # --- Validation parity with frontier_breakpoints ---------------------

    def test_validation_errors_match_the_turning_point_query(self) -> None:
        store = make_store()
        series = make_series()
        scenario = base_scenario()

        def explain(**overrides):
            return call(store, **overrides)

        def breakpoints(**overrides):
            kwargs = dict(
                reference="main",
                series=series,
                base_scenario=scenario,
                axis="total_budget",
                values=(0, 100),
                min_size=0,
                max_size=2,
                required=(),
                exclusive_pairs=(),
                limit=10,
            )
            kwargs.update(overrides)
            return store.frontier_breakpoints(**kwargs)

        # Every invalid input must raise the same exception type.
        invalid = [
            dict(reference=1),
            dict(reference=""),
            dict(reference="ghost"),
            dict(series=()),
            dict(series={"": ()}),
            dict(series={"ghost": ()}),
            dict(base_scenario=()),
            dict(base_scenario={}),
            dict(axis=1),
            dict(axis=""),
            dict(axis="nope"),
            dict(values=[0, 1]),
            dict(values=(0,)),
            dict(values=(1, 0)),
            dict(values=(True, 1)),
            dict(min_size=True),
            dict(max_size=1.0),
            dict(required=["feature"]),
            dict(required=("ghost",)),
            dict(exclusive_pairs=((),)),
            dict(limit=0),
        ]
        for overrides in invalid:
            with self.subTest(**overrides):
                with self.assertRaises(Exception) as explain_caught:
                    explain(**overrides)
                with self.assertRaises(type(explain_caught.exception)):
                    breakpoints(**overrides)

    def test_booleans_never_accepted_as_numbers(self) -> None:
        store = make_store()
        for bad_kwargs in (
            dict(values=(0, True)),
            dict(min_size=True),
            dict(max_size=False),
            dict(limit=True),
        ):
            with self.subTest(**bad_kwargs):
                with self.assertRaises(TypeError):
                    call(store, **bad_kwargs)

    def test_unknown_branch_and_node_raise_key_error(self) -> None:
        store = make_store()
        with self.assertRaises(KeyError):
            call(store, reference="ghost")
        with self.assertRaises(KeyError):
            call(
                store,
                series={"feature": (("nope1", "f1"),)},
                max_size=1,
            )
        with self.assertRaises(KeyError):
            call(
                store,
                series={"feature": (("f2", "f1"),)},
                max_size=1,
            )

    def test_candidate_limit_exceeded_raises_value_error(self) -> None:
        store = make_store()
        with self.assertRaises(ValueError):
            call(store, limit=3)

    def test_misaligned_checkpoints_raise_value_error(self) -> None:
        store = make_store()
        with self.assertRaises(ValueError):
            call(
                store,
                series={
                    "feature": (("m1", "f1"), ("m2", "f2")),
                    "side": (("m1", "s1"),),
                },
            )

    # --- Read-only and detachment ----------------------------------------

    def test_query_is_read_only(self) -> None:
        store = make_store()
        heads = dict(store._heads)
        appends = copy.deepcopy(store._appends)
        merges = copy.deepcopy(store._merges)
        replay_main = store.replay("main")
        audit = store.audit_log()
        series = make_series()
        scenario = base_scenario()
        call(store)
        # Failing calls leave state untouched as well.
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
        # Existing search and sensitivity results are unaffected.
        search = store.search_combinations(
            "main", series, WEIGHTS, 17, {}, 0, 2, (), (), 10
        )
        call(store)
        self.assertEqual(
            store.search_combinations(
                "main", series, WEIGHTS, 17, {}, 0, 2, (), (), 10
            ),
            search,
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
        self.assertEqual(
            [
                row["members"]
                for row in sensitivity["scenarios"][0]["frontier"]
            ],
            [("side",), ()],
        )

    def test_results_detached_and_unshared(self) -> None:
        store = make_store()
        result = call(store)
        pristine = copy.deepcopy(result)
        for interval in result:
            for change in interval["changes"]:
                change["members"] += ("evil",)
                change["cause_key"] = "evil"
                change["left"]["dominators"] += (("evil",),)
                for attribution in change["attribution"]:
                    attribution["key"] = "evil"
                    attribution["left"]["path"] += ("evil",)
        self.assertEqual(call(store), pristine)

    def test_no_object_sharing_between_evidence_levels(self) -> None:
        store = make_store()
        result = call(store)
        # The outer dominator containers are freshly built per side; the
        # immutable member tuples inside may be shared (as in the existing
        # search/sensitivity rows), but no mutable object is aliased.
        dominator_tuples = [
            dominators
            for interval in result
            for change in interval["changes"]
            for dominators in (
                change["left"]["dominators"],
                change["right"]["dominators"],
            )
            if dominators
        ]
        self.assertEqual(
            len(dominator_tuples), len({id(d) for d in dominator_tuples})
        )
        # Attribution copies are distinct dicts across members.
        attribution_ids = [
            id(attribution)
            for interval in result
            for change in interval["changes"]
            for attribution in change["attribution"]
        ]
        self.assertEqual(
            len(attribution_ids), len(set(attribution_ids))
        )

    def test_repeated_calls_are_equal(self) -> None:
        store = make_store()
        first = call(store)
        second = call(store)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
