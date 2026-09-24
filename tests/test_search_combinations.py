import copy
import inspect
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


class SearchCombinationsTests(unittest.TestCase):
    def test_public_signature_has_no_defaults(self) -> None:
        parameters = list(
            inspect.signature(
                BranchStore.search_combinations
            ).parameters.values()
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
        result = store.search_combinations(
            "main", make_series(), WEIGHTS, 100, {}, 0, 2, (), (), 10
        )
        self.assertIsInstance(result, dict)
        self.assertEqual(list(result), ["frontier", "rejected"])
        self.assertIsInstance(result["frontier"], tuple)
        self.assertIsInstance(result["rejected"], tuple)
        row = result["frontier"][0]
        self.assertEqual(
            list(row),
            ["members", "risk", "contributions", "attributions"],
        )
        self.assertIsInstance(row["members"], tuple)
        self.assertIsInstance(row["contributions"], tuple)
        self.assertIsInstance(row["attributions"], tuple)

    def test_rejected_item_shape_and_inapplicable_fields_are_none(self) -> None:
        store = make_store()
        result = store.search_combinations(
            "main", make_series(), WEIGHTS, 0, {}, 0, 2, (), (), 10
        )
        for item in result["rejected"]:
            self.assertEqual(
                list(item),
                ["members", "reason", "checkpoint", "key", "overrun"],
            )
        # Budget rejections name a checkpoint; structural ones do not.
        by_members = {item["members"]: item for item in result["rejected"]}
        total = by_members[("feature",)]
        self.assertEqual(total["reason"], "total_budget")
        self.assertEqual(total["checkpoint"], 0)
        self.assertIsNone(total["key"])

        result = store.search_combinations(
            "main",
            make_series(),
            WEIGHTS,
            1000,
            {},
            0,
            2,
            ("side",),
            (),
            10,
        )
        missing = next(
            item
            for item in result["rejected"]
            if item["reason"] == "missing_required"
        )
        self.assertIsNone(missing["checkpoint"])
        self.assertIsNone(missing["key"])
        self.assertIsNone(missing["overrun"])

    def test_enumeration_size_then_tuple_code_point_order(self) -> None:
        store = make_store()
        # A zero total budget rejects every non-empty subset; the empty
        # subset is feasible, so rejected enumeration order is fully
        # observable: sizes ascending, tuples by code point within a size.
        result = store.search_combinations(
            "main", make_series(), WEIGHTS, 0, {}, 0, 2, (), (), 10
        )
        self.assertEqual(
            [item["members"] for item in result["rejected"]],
            [("feature",), ("side",), ("feature", "side")],
        )
        self.assertEqual(
            [item["members"] for item in result["frontier"]],
            [()],
        )

    def test_risk_and_breach_match_combination_budget(self) -> None:
        store = make_store()
        # Point 0 aggregate (x, y, z) = (3, -5, 0) -> risk 21; point 1 is
        # (3, 0, -10) -> risk 16. Total budget 17: the pair breaches at
        # point 0 with overrun 4; feature (final risk 18) breaches at
        # point 1 with overrun 1; side (final risk 16) stays feasible.
        result = store.search_combinations(
            "main", make_series(), WEIGHTS, 17, {}, 0, 2, (), (), 10
        )
        by_members = {item["members"]: item for item in result["rejected"]}
        pair = by_members[("feature", "side")]
        self.assertEqual(pair["reason"], "total_budget")
        self.assertEqual(pair["checkpoint"], 0)
        self.assertIsNone(pair["key"])
        self.assertEqual(pair["overrun"], 4)
        feature = by_members[("feature",)]
        self.assertEqual(feature["reason"], "total_budget")
        self.assertEqual(feature["checkpoint"], 1)
        self.assertEqual(feature["overrun"], 1)

        # Side dominates the equal-size, higher-risk feature; the empty
        # subset survives because nothing feasible has a lower zero risk.
        frontier_members = {item["members"] for item in result["frontier"]}
        self.assertEqual(frontier_members, {("side",), ()})
        side = next(
            item for item in result["frontier"] if item["members"] == ("side",)
        )
        self.assertEqual(side["risk"], 16)

        # Cross-check the final risks against combination_budget.
        rows = store.combination_budget(
            "main",
            make_series(),
            {
                "none": (),
                "feature": ("feature",),
                "side": ("side",),
                "both": ("feature", "side"),
            },
            WEIGHTS,
            17,
            {},
        )
        risks = {row["members"]: 17 - row["remaining"] for row in rows}
        self.assertEqual(risks[()], 0)
        self.assertEqual(risks[("feature",)], 18)
        self.assertEqual(risks[("side",)], 16)
        self.assertEqual(risks[("feature", "side")], 16)

    def test_per_key_breach_reports_first_key_and_weighted_excess(self) -> None:
        store = make_store()
        # feature alone: point 1 aggregate z = -7; budgets z: 6 make it the
        # breach with overrun (7 - 6) * weight 1 = 1. x never breaches.
        result = store.search_combinations(
            "main",
            make_series(),
            {"z": 1, "x": 2},
            1000,
            {"x": 100, "z": 6},
            0,
            1,
            (),
            (),
            10,
        )
        by_members = {item["members"]: item for item in result["rejected"]}
        feature = by_members[("feature",)]
        self.assertEqual(feature["reason"], "key_budget")
        self.assertEqual(feature["checkpoint"], 1)
        self.assertEqual(feature["key"], "z")
        self.assertEqual(feature["overrun"], 1)

    def test_rejection_first_cause_order(self) -> None:
        store = make_store()
        # zero total budget would breach every non-empty subset, but the
        # required/exclusive filters must take precedence.
        result = store.search_combinations(
            "main",
            make_series(),
            WEIGHTS,
            0,
            {},
            0,
            2,
            ("side",),
            (("feature", "side"),),
            10,
        )
        by_members = {item["members"]: item for item in result["rejected"]}
        self.assertEqual(
            by_members[()]["reason"], "missing_required"
        )
        self.assertEqual(
            by_members[("feature",)]["reason"], "missing_required"
        )
        # Has the required member but hits the exclusive pair: that wins
        # over the (certain) total-budget breach.
        self.assertEqual(
            by_members[("feature", "side")]["reason"], "exclusive_pair"
        )
        self.assertEqual(by_members[("side",)]["reason"], "total_budget")

    def test_frontier_keeps_pareto_optimal_and_sorts(self) -> None:
        graph = EventGraph()
        graph.add("root", 0, (), {})
        store = BranchStore(graph)
        store.create("main", "root")
        for name in ("a", "b", "c"):
            store.create(name, "root")
        store.append("main", "r1", 1, {"x": 10})
        store.append("a", "a1", 1, {"x": 11})  # diff +1
        store.append("b", "b1", 1, {"x": 12})  # diff +2
        store.append("c", "c1", 1, {"x": 13})  # diff +3
        series = {
            name: (("r1", f"{name}1"),) for name in ("a", "b", "c")
        }
        result = store.search_combinations(
            "main", series, {"x": 1}, 100, {}, 0, 3, (), (), 50
        )
        self.assertEqual(result["rejected"], ())
        # Risks are the aggregate diffs: b(2) is dominated by a(1) at the
        # same size, ac(4)/bc(5) by ab(3), abc(6) stands; () risk 0 stands.
        self.assertEqual(
            [(row["members"], row["risk"]) for row in result["frontier"]],
            [
                (("a", "b", "c"), 6),
                (("a", "b"), 3),
                (("a",), 1),
                ((), 0),
            ],
        )

    def test_size_window_restricts_enumeration_and_required(self) -> None:
        store = make_store()
        # A zero budget rejects both singletons, so the window's full
        # enumeration is observable without Pareto filtering.
        result = store.search_combinations(
            "main", make_series(), WEIGHTS, 0, {}, 1, 1, (), (), 10
        )
        members_seen = {
            row["members"] for row in result["frontier"]
        } | {item["members"] for item in result["rejected"]}
        self.assertEqual(members_seen, {("feature",), ("side",)})

        # With the window fixed at size 1, requiring feature makes side
        # infeasible even though its risk alone would fit.
        result = store.search_combinations(
            "main",
            make_series(),
            WEIGHTS,
            100,
            {},
            1,
            1,
            ("feature",),
            (),
            10,
        )
        self.assertEqual(
            [item["members"] for item in result["rejected"]],
            [("side",)],
        )
        self.assertEqual(
            [row["members"] for row in result["frontier"]],
            [("feature",)],
        )

    def test_limit_counts_all_candidates_and_rejects_partial_results(self) -> None:
        store = make_store()
        # Window 0..2 over a 2-member pool enumerates exactly 4 subsets; a
        # zero budget rejects the three non-empty ones, so all four appear.
        result = store.search_combinations(
            "main", make_series(), WEIGHTS, 0, {}, 0, 2, (), (), 4
        )
        self.assertEqual(
            len(result["frontier"]) + len(result["rejected"]),
            4,
        )
        with self.assertRaises(ValueError):
            store.search_combinations(
                "main", make_series(), WEIGHTS, 0, {}, 0, 2, (), (), 3
            )

    def test_empty_pool_only_allows_zero_window_and_empty_risk_zero(self) -> None:
        store = make_store()
        result = store.search_combinations(
            "main", {}, {}, 0, {}, 0, 0, (), (), 1
        )
        self.assertEqual(len(result["frontier"]), 1)
        only = result["frontier"][0]
        self.assertEqual(only["members"], ())
        self.assertEqual(only["risk"], 0)
        self.assertEqual(only["contributions"], ())
        self.assertEqual(only["attributions"], ())
        self.assertEqual(result["rejected"], ())
        for bad_max in (1, 2):
            with self.assertRaises(ValueError):
                store.search_combinations(
                    "main", {}, {}, 0, {}, 0, bad_max, (), (), 10
                )

    def test_contributions_and_attributions_on_feasible_subset(self) -> None:
        store = make_store()
        # Budget 100 makes every subset feasible; final point aggregates:
        # feature (1, 3, -7) -> risk 18, side (2, -3, -3) -> risk 16.
        result = store.search_combinations(
            "main", make_series(), WEIGHTS, 100, {}, 2, 2, (), (), 10
        )
        pair = result["frontier"][0]
        self.assertEqual(pair["members"], ("feature", "side"))
        self.assertEqual(pair["risk"], 16)
        # Without feature: side risk 16 -> contribution 0.
        # Without side: feature risk 18 -> contribution -2.
        self.assertEqual(pair["contributions"], (0, -2))
        # One attribution group per member (sorted member order).
        self.assertEqual(len(pair["attributions"]), 2)
        for member_group in pair["attributions"]:
            self.assertEqual(len(member_group), 3)
            for attribution in member_group:
                self.assertEqual(
                    list(attribution), ["key", "fork", "left", "right"]
                )
        expected_feature = store.attribute_divergences_at(
            "main", "m2", "feature", "f2", ("x", "y", "z")
        )
        expected_side = store.attribute_divergences_at(
            "main", "m2", "side", "s2", ("x", "y", "z")
        )
        self.assertEqual(
            tuple(pair["attributions"][0]), expected_feature
        )
        self.assertEqual(tuple(pair["attributions"][1]), expected_side)

    def test_no_checkpoints_is_feasible_with_zero_risk(self) -> None:
        store = make_store()
        series = {"feature": (), "side": ()}
        result = store.search_combinations(
            "main", series, WEIGHTS, 0, {"x": 0}, 0, 2, (), (), 10
        )
        self.assertEqual(result["rejected"], ())
        # Every size carries zero risk, so Pareto dominance keeps only the
        # largest enumerated subset.
        self.assertEqual(
            [row["members"] for row in result["frontier"]],
            [("feature", "side")],
        )
        pair = result["frontier"][0]
        self.assertEqual(pair["risk"], 0)
        self.assertEqual(pair["contributions"], (0, 0))
        self.assertEqual(pair["attributions"], ())

    def test_float_arithmetic_and_positive_zero(self) -> None:
        store = make_store()
        result = store.search_combinations(
            "main",
            make_series(),
            {"x": 0.5},
            0.0,
            {},
            0,
            1,
            (),
            (),
            10,
        )
        # The empty subset never evaluates a checkpoint: its zero risk is
        # the integer 0, exactly as combination_budget's empty combination.
        empty = next(
            row for row in result["frontier"] if row["members"] == ()
        )
        self.assertEqual(empty["risk"], 0)
        self.assertIsInstance(empty["risk"], int)
        rejected = {
            item["members"]: item for item in result["rejected"]
        }
        # Earliest breaches: feature's point-0 x diff is 1 (overrun 0.5),
        # side's is 2 (overrun 1.0).
        self.assertEqual(rejected[("feature",)]["overrun"], 0.5)
        self.assertIsInstance(rejected[("feature",)]["overrun"], float)
        self.assertEqual(rejected[("side",)]["overrun"], 1.0)

    def test_size_parameters_reject_non_int_and_bool(self) -> None:
        store = make_store()
        series = make_series()
        for bad in (True, False, 1.0, "1", None, (1,), [1]):
            with self.subTest(min_size=repr(bad)):
                with self.assertRaises(TypeError):
                    store.search_combinations(
                        "main", series, WEIGHTS, 100, {}, bad, 2, (), (), 10
                    )
            with self.subTest(max_size=repr(bad)):
                with self.assertRaises(TypeError):
                    store.search_combinations(
                        "main", series, WEIGHTS, 100, {}, 0, bad, (), (), 10
                    )
            with self.subTest(limit=repr(bad)):
                with self.assertRaises(TypeError):
                    store.search_combinations(
                        "main", series, WEIGHTS, 100, {}, 0, 2, (), (), bad
                    )

    def test_size_range_validation(self) -> None:
        store = make_store()
        series = make_series()
        with self.assertRaises(ValueError):
            store.search_combinations(
                "main", series, WEIGHTS, 100, {}, -1, 2, (), (), 10
            )
        with self.assertRaises(ValueError):
            store.search_combinations(
                "main", series, WEIGHTS, 100, {}, 2, 1, (), (), 10
            )
        with self.assertRaises(ValueError):
            store.search_combinations(
                "main", series, WEIGHTS, 100, {}, 0, 3, (), (), 10
            )
        with self.assertRaises(ValueError):
            store.search_combinations(
                "main", series, WEIGHTS, 100, {}, 0, 2, (), (), 0
            )

    def test_required_container_and_member_validation(self) -> None:
        store = make_store()
        series = make_series()
        for bad in ([], None, "feature", 1, {"feature"}):
            with self.subTest(required=repr(bad)):
                with self.assertRaises(TypeError):
                    store.search_combinations(
                        "main", series, WEIGHTS, 100, {}, 0, 2, bad, (), 10
                    )
        with self.assertRaises(TypeError):
            store.search_combinations(
                "main", series, WEIGHTS, 100, {}, 0, 2, (1,), (), 10
            )
        with self.assertRaises(ValueError):
            store.search_combinations(
                "main", series, WEIGHTS, 100, {}, 0, 2, ("",), (), 10
            )
        with self.assertRaises(ValueError):
            store.search_combinations(
                "main",
                series,
                WEIGHTS,
                100,
                {},
                0,
                2,
                ("feature", "feature"),
                (),
                10,
            )
        with self.assertRaises(ValueError):
            store.search_combinations(
                "main", series, WEIGHTS, 100, {}, 0, 2, ("ghost",), (), 10
            )

    def test_exclusive_pairs_container_and_shape_validation(self) -> None:
        store = make_store()
        series = make_series()
        for bad in ([], None, "x", 1, {"x"}):
            with self.subTest(pairs=repr(bad)):
                with self.assertRaises(TypeError):
                    store.search_combinations(
                        "main", series, WEIGHTS, 100, {}, 0, 2, (), bad, 10
                    )
        for bad_pair in (
            ("feature",),
            ("feature", "side", "x"),
            ["feature", "side"],
            "fs",
            1,
        ):
            with self.subTest(pair=repr(bad_pair)):
                with self.assertRaises(TypeError):
                    store.search_combinations(
                        "main",
                        series,
                        WEIGHTS,
                        100,
                        {},
                        0,
                        2,
                        (),
                        (bad_pair,),
                        10,
                    )
        with self.assertRaises(TypeError):
            store.search_combinations(
                "main",
                series,
                WEIGHTS,
                100,
                {},
                0,
                2,
                (),
                ((1, "side"),),
                10,
            )

    def test_exclusive_pair_value_validation(self) -> None:
        store = make_store()
        series = make_series()
        # Empty member name.
        with self.assertRaises(ValueError):
            store.search_combinations(
                "main",
                series,
                WEIGHTS,
                100,
                {},
                0,
                2,
                (),
                (("", "side"),),
                10,
            )
        # Self-exclusive.
        with self.assertRaises(ValueError):
            store.search_combinations(
                "main",
                series,
                WEIGHTS,
                100,
                {},
                0,
                2,
                (),
                (("feature", "feature"),),
                10,
            )
        # Unknown member.
        with self.assertRaises(ValueError):
            store.search_combinations(
                "main",
                series,
                WEIGHTS,
                100,
                {},
                0,
                2,
                (),
                (("feature", "ghost"),),
                10,
            )
        # Duplicate unordered pair (given in both orientations).
        with self.assertRaises(ValueError):
            store.search_combinations(
                "main",
                series,
                WEIGHTS,
                100,
                {},
                0,
                2,
                (),
                (("side", "feature"), ("feature", "side")),
                10,
            )
        # Both members required while mutually exclusive.
        with self.assertRaises(ValueError):
            store.search_combinations(
                "main",
                series,
                WEIGHTS,
                100,
                {},
                0,
                2,
                ("feature", "side"),
                (("feature", "side"),),
                10,
            )
        # The same pair is fine when only one side is required.
        store.search_combinations(
            "main",
            series,
            WEIGHTS,
            100,
            {},
            0,
            2,
            ("feature",),
            (("feature", "side"),),
            10,
        )

    def test_budget_validation_matches_combination_budget(self) -> None:
        store = make_store()
        series = make_series()
        for bad in (True, False, "1", None, (1,), [1], {"x": 1}):
            with self.subTest(total=repr(bad)):
                with self.assertRaises(TypeError):
                    store.search_combinations(
                        "main", series, WEIGHTS, bad, {}, 0, 2, (), (), 10
                    )
        for bad in (-1, float("nan"), float("inf")):
            with self.subTest(total=repr(bad)):
                with self.assertRaises(ValueError):
                    store.search_combinations(
                        "main", series, WEIGHTS, bad, {}, 0, 2, (), (), 10
                    )
        with self.assertRaises(TypeError):
            store.search_combinations(
                "main", series, {"x": 1}, 5, [], 0, 2, (), (), 10
            )
        with self.assertRaises(ValueError):
            store.search_combinations(
                "main", series, {"x": 1}, 5, {"z": 1}, 0, 2, (), (), 10
            )

    def test_checkpoint_alignment_validation(self) -> None:
        store = make_store()
        with self.assertRaises(ValueError):
            store.search_combinations(
                "main",
                {
                    "feature": (("m1", "f1"), ("m2", "f2")),
                    "side": (("m1", "s1"),),
                },
                WEIGHTS,
                100,
                {},
                0,
                2,
                (),
                (),
                10,
            )
        with self.assertRaises(ValueError):
            store.search_combinations(
                "main",
                {
                    "feature": (("m2", "f2"),),
                    "side": (("m1", "s1"),),
                },
                WEIGHTS,
                100,
                {},
                0,
                2,
                (),
                (),
                10,
            )

    def test_all_inputs_validated_before_branch_lookups(self) -> None:
        store = make_store()
        series = make_series()
        # Size/type errors beat unknown branches.
        with self.assertRaises(TypeError):
            store.search_combinations(
                "ghost", series, WEIGHTS, True, {}, 0, 2, (), (), 10
            )
        # Member-shape errors beat unknown branches.
        with self.assertRaises(ValueError):
            store.search_combinations(
                "ghost", series, WEIGHTS, 100, {}, 0, 2, ("ghost-m",), (), 10
            )
        # Budget errors beat unknown branches.
        with self.assertRaises(ValueError):
            store.search_combinations(
                "ghost", series, {"x": 1}, 100, {"z": 1}, 0, 2, (), (), 10
            )

    def test_branch_and_node_lookup_order(self) -> None:
        store = make_store()
        series = make_series()
        with self.assertRaises(KeyError) as caught:
            store.search_combinations(
                "ghost", series, WEIGHTS, 100, {}, 0, 2, (), (), 10
            )
        self.assertEqual(caught.exception.args, ("ghost",))
        # Reference looked up before series.
        with self.assertRaises(KeyError) as caught:
            store.search_combinations(
                "main",
                {"ghost1": (), "ghost2": ()},
                WEIGHTS,
                100,
                {},
                0,
                0,
                (),
                (),
                10,
            )
        self.assertEqual(caught.exception.args, ("ghost1",))
        # Left node before right within a pair.
        with self.assertRaises(KeyError) as caught:
            store.search_combinations(
                "main",
                {"feature": (("nope1", "f1"),)},
                WEIGHTS,
                100,
                {},
                0,
                1,
                (),
                (),
                10,
            )
        self.assertEqual(caught.exception.args, ("nope1",))
        with self.assertRaises(KeyError) as caught:
            store.search_combinations(
                "main",
                {"feature": (("m1", "nope2"),)},
                WEIGHTS,
                100,
                {},
                0,
                1,
                (),
                (),
                10,
            )
        self.assertEqual(caught.exception.args, ("nope2",))
        # A node outside the branch's current head closure.
        with self.assertRaises(KeyError):
            store.search_combinations(
                "main",
                {"feature": (("f2", "f1"),)},
                WEIGHTS,
                100,
                {},
                0,
                1,
                (),
                (),
                10,
            )

    def test_results_are_detached_and_unshared(self) -> None:
        store = make_store()
        series = make_series()
        first = store.search_combinations(
            "main", series, WEIGHTS, 100, {}, 0, 2, (), (), 10
        )
        self.assertIsNot(first["frontier"], first["rejected"])
        pristine = copy.deepcopy(first)
        for row in first["frontier"]:
            row["members"] += ("evil",)
            row["contributions"] += (999,)
            for member_group in row["attributions"]:
                for attribution in member_group:
                    attribution["key"] = "evil"
                    attribution["left"]["path"] += ("evil",)
        for item in first["rejected"]:
            item["members"] += ("evil",)
        fresh = store.search_combinations(
            "main", series, WEIGHTS, 100, {}, 0, 2, (), (), 10
        )
        self.assertEqual(fresh, pristine)

    def test_query_is_read_only(self) -> None:
        store = make_store()
        heads = dict(store._heads)
        appends = copy.deepcopy(store._appends)
        merges = copy.deepcopy(store._merges)
        replay_main = store.replay("main")
        audit = store.audit_log()
        series = make_series()
        store.search_combinations(
            "main", series, WEIGHTS, 17, {}, 0, 2, (), (), 10
        )
        # Failing calls must leave state untouched as well.
        with self.assertRaises(ValueError):
            store.search_combinations(
                "main", series, WEIGHTS, 17, {}, 0, 2, (), (), 1
            )
        with self.assertRaises(KeyError):
            store.search_combinations(
                "main",
                {"feature": (("m1", "nope"),)},
                WEIGHTS,
                17,
                {},
                0,
                1,
                (),
                (),
                10,
            )
        with self.assertRaises(TypeError):
            store.search_combinations(
                "main", series, WEIGHTS, 17, {}, True, 2, (), (), 10
            )
        self.assertEqual(dict(store._heads), heads)
        self.assertEqual(copy.deepcopy(store._appends), appends)
        self.assertEqual(copy.deepcopy(store._merges), merges)
        self.assertEqual(store.replay("main"), replay_main)
        self.assertEqual(store.audit_log(), audit)
        # Existing interfaces keep their behavior.
        rows = store.combination_budget(
            "main",
            series,
            {"both": ("feature", "side")},
            WEIGHTS,
            20,
            {},
        )
        store.search_combinations(
            "main", series, WEIGHTS, 17, {}, 0, 2, (), (), 10
        )
        self.assertEqual(
            store.combination_budget(
                "main",
                series,
                {"both": ("feature", "side")},
                WEIGHTS,
                20,
                {},
            ),
            rows,
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
