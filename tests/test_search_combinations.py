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
                "exclusive",
                "limit",
            ],
        )
        self.assertTrue(
            all(p.default is inspect.Parameter.empty for p in parameters)
        )

    def test_result_shape_and_key_order(self) -> None:
        store = make_store()
        result = store.search_combinations(
            "main", make_series(), {"x": 2, "y": 3, "z": 1}, 20, {},
            0, 2, (), (), 10,
        )
        self.assertIsInstance(result, dict)
        self.assertEqual(list(result), ["frontier", "rejected"])
        self.assertIsInstance(result["frontier"], tuple)
        self.assertIsInstance(result["rejected"], tuple)
        for entry in result["frontier"]:
            self.assertEqual(
                list(entry),
                ["members", "risk", "contributions", "attributions"],
            )
        for entry in result["rejected"]:
            self.assertEqual(
                list(entry),
                ["members", "reason", "checkpoint", "key", "overrun"],
            )

    def test_frontier_and_rejected_basic(self) -> None:
        store = make_store()
        # (feature,) risks 8 and 18; (side,) 13 and 16; (feature,side)
        # aggregates (3, -5, 0) at point 0 -> risk 21 > 20, overrun 1.
        result = store.search_combinations(
            "main", make_series(), {"x": 2, "y": 3, "z": 1}, 20, {},
            0, 2, (), (), 10,
        )
        self.assertEqual(
            [entry["members"] for entry in result["frontier"]],
            [("side",), ()],
        )
        self.assertEqual(
            [entry["risk"] for entry in result["frontier"]], [16, 0]
        )
        self.assertEqual(result["frontier"][0]["contributions"], (16,))
        self.assertEqual(result["frontier"][0]["attributions"], ())
        self.assertEqual(result["frontier"][1]["contributions"], ())
        # (feature,) is dominated by (side,): same size, lower risk.
        rejected = result["rejected"]
        self.assertEqual(len(rejected), 1)
        self.assertEqual(
            rejected[0],
            {
                "members": ("feature", "side"),
                "reason": "total_budget",
                "checkpoint": 0,
                "key": None,
                "overrun": 1,
            },
        )

    def test_rejection_reason_priority(self) -> None:
        store = make_store()
        series = make_series()
        # A candidate missing a required member is rejected before any
        # budget evaluation, and an exclusive hit before budget checks.
        result = store.search_combinations(
            "main", series, {"x": 2, "y": 3, "z": 1}, 20, {},
            0, 2, ("feature",), (), 10,
        )
        self.assertEqual(
            [
                (entry["members"], entry["reason"])
                for entry in result["rejected"]
            ],
            [
                ((), "missing_required"),
                (("side",), "missing_required"),
                (("feature", "side"), "total_budget"),
            ],
        )
        for entry in result["rejected"][:2]:
            self.assertIsNone(entry["checkpoint"])
            self.assertIsNone(entry["key"])
            self.assertIsNone(entry["overrun"])
        result = store.search_combinations(
            "main", series, {"x": 2, "y": 3, "z": 1}, 20, {},
            2, 2, (), (("feature", "side"),), 10,
        )
        self.assertEqual(
            result["rejected"][0],
            {
                "members": ("feature", "side"),
                "reason": "exclusive_pair",
                "checkpoint": None,
                "key": None,
                "overrun": None,
            },
        )
        self.assertEqual(result["frontier"], ())

    def test_key_budget_rejection_reports_first_key_in_order(self) -> None:
        store = make_store()
        # (feature,) breaches only the z budget at checkpoint 1: excess
        # (7 - 5) * weight 1 = 2. (feature,side) aggregates z to -10 at
        # checkpoint 1: excess (10 - 5) * 1 = 5. Neither total is over 100.
        result = store.search_combinations(
            "main", make_series(), {"x": 2, "y": 3, "z": 1}, 100, {"z": 5},
            1, 2, (), (), 10,
        )
        self.assertEqual(
            [entry["members"] for entry in result["frontier"]],
            [("side",)],
        )
        self.assertEqual(
            [
                (
                    entry["members"],
                    entry["reason"],
                    entry["checkpoint"],
                    entry["key"],
                    entry["overrun"],
                )
                for entry in result["rejected"]
            ],
            [
                (("feature",), "key_budget", 1, "z", 2),
                (("feature", "side"), "key_budget", 1, "z", 5),
            ],
        )

    def test_earliest_breaching_checkpoint_is_reported(self) -> None:
        store = make_store()
        # (feature,) risks 8 and 18, (side,) 13 and 16; against a budget
        # of 15 both breach only at checkpoint 1, by 3 and 1, and the
        # rejected entries keep their enumeration order.
        result = store.search_combinations(
            "main", make_series(), {"x": 2, "y": 3, "z": 1}, 15, {},
            1, 1, (), (), 10,
        )
        self.assertEqual(
            [
                (entry["members"], entry["checkpoint"], entry["overrun"])
                for entry in result["rejected"]
            ],
            [(("feature",), 1, 3), (("side",), 1, 1)],
        )
        self.assertEqual(result["frontier"], ())

    def test_frontier_dominance_by_size_and_risk(self) -> None:
        store = make_store()
        # The y differences cancel inside (feature,side): final risk 0
        # with two members dominates every smaller feasible combination.
        result = store.search_combinations(
            "main", make_series(), {"y": 1}, 10, {},
            0, 2, (), (), 10,
        )
        self.assertEqual(
            [entry["members"] for entry in result["frontier"]],
            [("feature", "side")],
        )
        self.assertEqual(result["frontier"][0]["risk"], 0)
        self.assertEqual(result["rejected"], ())

    def test_frontier_keeps_equal_size_and_risk_ties(self) -> None:
        graph = EventGraph()
        graph.add("root", 0, (), {})
        store = BranchStore(graph)
        store.create("main", "root")
        store.create("a", "root")
        store.append("a", "a1", 1, {"x": 5})
        store.create("b", "root")
        store.append("b", "b1", 2, {"x": 5})
        series = {"a": (("root", "a1"),), "b": (("root", "b1"),)}
        result = store.search_combinations(
            "main", series, {"x": 1}, 100, {}, 0, 2, (), (), 10,
        )
        self.assertEqual(
            [
                (entry["members"], entry["risk"])
                for entry in result["frontier"]
            ],
            [
                (("a", "b"), 10),
                (("a",), 5),
                (("b",), 5),
                ((), 0),
            ],
        )

    def test_contributions_follow_combination_budget(self) -> None:
        store = make_store()
        # Final checkpoint risk is 16; without feature it is 16, without
        # side it is 18 -- matching combination_budget exactly.
        result = store.search_combinations(
            "main", make_series(), {"x": 2, "y": 3, "z": 1}, 100, {},
            2, 2, (), (), 10,
        )
        self.assertEqual(len(result["frontier"]), 1)
        entry = result["frontier"][0]
        self.assertEqual(entry["members"], ("feature", "side"))
        self.assertEqual(entry["risk"], 16)
        self.assertEqual(entry["contributions"], (0, -2))
        row = store.combination_budget(
            "main",
            make_series(),
            {"both": ("feature", "side")},
            {"x": 2, "y": 3, "z": 1},
            100,
            {},
        )[0]
        self.assertEqual(entry["contributions"], row["contributions"])
        self.assertEqual(entry["risk"], 100 - row["remaining"])

    def test_size_bounds_restrict_enumeration(self) -> None:
        store = make_store()
        # Only the two singletons are candidates; (feature,) is feasible
        # but dominated by the equal-size, lower-risk (side,).
        result = store.search_combinations(
            "main", make_series(), {"x": 2, "y": 3, "z": 1}, 100, {},
            1, 1, (), (), 10,
        )
        self.assertEqual(
            [entry["members"] for entry in result["frontier"]],
            [("side",)],
        )
        self.assertEqual(result["rejected"], ())

    def test_limit_counts_every_candidate(self) -> None:
        store = make_store()
        series = make_series()
        # Sizes 0..2 over two branches give 1 + 2 + 1 = 4 candidates.
        with self.assertRaises(ValueError):
            store.search_combinations(
                "main", series, {"x": 1}, 100, {}, 0, 2, (), (), 3,
            )
        # A limit equal to the candidate count is accepted.
        result = store.search_combinations(
            "main", series, {"x": 1}, 100, {}, 0, 2, (), (), 4,
        )
        self.assertEqual(result["rejected"], ())
        # (side,) is dominated by the equal-size, lower-risk (feature,).
        self.assertEqual(
            [entry["members"] for entry in result["frontier"]],
            [("feature", "side"), ("feature",), ()],
        )

    def test_empty_series_allows_only_zero_to_zero(self) -> None:
        store = make_store()
        result = store.search_combinations(
            "main", {}, {"x": 1}, 5, {}, 0, 0, (), (), 10,
        )
        self.assertEqual(result["rejected"], ())
        self.assertEqual(len(result["frontier"]), 1)
        entry = result["frontier"][0]
        self.assertEqual(entry["members"], ())
        self.assertEqual(entry["risk"], 0)
        self.assertIsInstance(entry["risk"], int)
        self.assertEqual(entry["contributions"], ())
        self.assertEqual(entry["attributions"], ())
        with self.assertRaises(ValueError):
            store.search_combinations(
                "main", {}, {"x": 1}, 5, {}, 0, 1, (), (), 10,
            )
        with self.assertRaises(ValueError):
            store.search_combinations(
                "main", {}, {"x": 1}, 5, {}, 1, 1, (), (), 10,
            )

    def test_size_and_limit_validation(self) -> None:
        store = make_store()
        series = make_series()
        for bad in (True, False, "1", None, 1.0, (1,), [1]):
            with self.subTest(value=repr(bad)):
                with self.assertRaises(TypeError):
                    store.search_combinations(
                        "main", series, {"x": 1}, 5, {}, bad, 2, (), (), 10,
                    )
                with self.assertRaises(TypeError):
                    store.search_combinations(
                        "main", series, {"x": 1}, 5, {}, 0, bad, (), (), 10,
                    )
                with self.assertRaises(TypeError):
                    store.search_combinations(
                        "main", series, {"x": 1}, 5, {}, 0, 2, (), (), bad,
                    )
        with self.assertRaises(ValueError):
            store.search_combinations(
                "main", series, {"x": 1}, 5, {}, -1, 2, (), (), 10,
            )
        with self.assertRaises(ValueError):
            store.search_combinations(
                "main", series, {"x": 1}, 5, {}, 2, 1, (), (), 10,
            )
        with self.assertRaises(ValueError):
            store.search_combinations(
                "main", series, {"x": 1}, 5, {}, 0, 3, (), (), 10,
            )
        with self.assertRaises(ValueError):
            store.search_combinations(
                "main", series, {"x": 1}, 5, {}, 0, 2, (), (), 0,
            )

    def test_required_and_exclusive_validation(self) -> None:
        store = make_store()
        series = make_series()
        for bad in ([], None, "feature", 1, {"feature": ()}):
            with self.subTest(container=repr(bad)):
                with self.assertRaises(TypeError):
                    store.search_combinations(
                        "main", series, {"x": 1}, 5, {}, 0, 2, bad, (), 10,
                    )
                with self.assertRaises(TypeError):
                    store.search_combinations(
                        "main", series, {"x": 1}, 5, {}, 0, 2, (), bad, 10,
                    )
        with self.assertRaises(TypeError):
            store.search_combinations(
                "main", series, {"x": 1}, 5, {}, 0, 2, (1,), (), 10,
            )
        with self.assertRaises(ValueError):
            store.search_combinations(
                "main", series, {"x": 1}, 5, {}, 0, 2, ("",), (), 10,
            )
        with self.assertRaises(ValueError):
            store.search_combinations(
                "main", series, {"x": 1}, 5, {}, 0, 2,
                ("feature", "feature"), (), 10,
            )
        with self.assertRaises(ValueError):
            store.search_combinations(
                "main", series, {"x": 1}, 5, {}, 0, 2, ("ghost",), (), 10,
            )
        # Exclusive pair shapes.
        for bad_pair in (("feature",), "feature", ("feature", "side", "x"),
                         None, 1):
            with self.subTest(pair=repr(bad_pair)):
                with self.assertRaises(TypeError):
                    store.search_combinations(
                        "main", series, {"x": 1}, 5, {}, 0, 2,
                        (), (bad_pair,), 10,
                    )
        with self.assertRaises(TypeError):
            store.search_combinations(
                "main", series, {"x": 1}, 5, {}, 0, 2,
                (), ((1, "side"),), 10,
            )
        with self.assertRaises(ValueError):
            store.search_combinations(
                "main", series, {"x": 1}, 5, {}, 0, 2,
                (), (("", "side"),), 10,
            )
        with self.assertRaises(ValueError):
            store.search_combinations(
                "main", series, {"x": 1}, 5, {}, 0, 2,
                (), (("ghost", "side"),), 10,
            )
        with self.assertRaises(ValueError):
            store.search_combinations(
                "main", series, {"x": 1}, 5, {}, 0, 2,
                (), (("feature", "feature"),), 10,
            )
        # A repeated pair in either member order is a conflict.
        with self.assertRaises(ValueError):
            store.search_combinations(
                "main", series, {"x": 1}, 5, {}, 0, 2,
                (), (("feature", "side"), ("side", "feature")), 10,
            )
        # A required member cannot be exclusive.
        with self.assertRaises(ValueError):
            store.search_combinations(
                "main", series, {"x": 1}, 5, {}, 0, 2,
                ("feature",), (("feature", "side"),), 10,
            )

    def test_series_checkpoint_alignment_validation(self) -> None:
        store = make_store()
        with self.assertRaises(ValueError):
            store.search_combinations(
                "main",
                {
                    "feature": (("m1", "f1"), ("m2", "f2")),
                    "side": (("m1", "s1"),),
                },
                {"x": 1}, 5, {}, 0, 2, (), (), 10,
            )
        with self.assertRaises(ValueError):
            store.search_combinations(
                "main",
                {
                    "feature": (("m2", "f2"),),
                    "side": (("m1", "s1"),),
                },
                {"x": 1}, 5, {}, 0, 2, (), (), 10,
            )

    def test_shared_input_validation_matches_combination_budget(self) -> None:
        store = make_store()
        series = make_series()
        with self.assertRaises(TypeError):
            store.search_combinations(1, series, {"x": 1}, 5, {}, 0, 2, (), (), 10)
        with self.assertRaises(ValueError):
            store.search_combinations("", series, {"x": 1}, 5, {}, 0, 2, (), (), 10)
        with self.assertRaises(TypeError):
            store.search_combinations("main", [], {"x": 1}, 5, {}, 0, 2, (), (), 10)
        with self.assertRaises(ValueError):
            store.search_combinations(
                "main", {"main": ()}, {"x": 1}, 5, {}, 0, 0, (), (), 10,
            )
        with self.assertRaises(TypeError):
            store.search_combinations("main", series, None, 5, {}, 0, 2, (), (), 10)
        with self.assertRaises(TypeError):
            store.search_combinations(
                "main", series, {"x": True}, 5, {}, 0, 2, (), (), 10,
            )
        with self.assertRaises(ValueError):
            store.search_combinations(
                "main", series, {"x": -1}, 5, {}, 0, 2, (), (), 10,
            )
        with self.assertRaises(TypeError):
            store.search_combinations(
                "main", series, {"x": 1}, True, {}, 0, 2, (), (), 10,
            )
        with self.assertRaises(ValueError):
            store.search_combinations(
                "main", series, {"x": 1}, -1, {}, 0, 2, (), (), 10,
            )
        with self.assertRaises(TypeError):
            store.search_combinations(
                "main", series, {"x": 1}, 5, [], 0, 2, (), (), 10,
            )
        with self.assertRaises(ValueError):
            store.search_combinations(
                "main", series, {"x": 1}, 5, {"z": 1}, 0, 2, (), (), 10,
            )

    def test_all_inputs_validated_before_branch_lookups(self) -> None:
        store = make_store()
        series = make_series()
        # Search-control errors beat unknown branches.
        with self.assertRaises(ValueError):
            store.search_combinations(
                "ghost", series, {"x": 1}, 5, {}, 0, 3, (), (), 10,
            )
        with self.assertRaises(ValueError):
            store.search_combinations(
                "ghost", series, {"x": 1}, 5, {}, 0, 2, ("ghost",), (), 10,
            )
        with self.assertRaises(ValueError):
            store.search_combinations(
                "ghost", series, {"x": 1}, 5, {}, 0, 2, (), (), 0,
            )
        # Budget errors beat search-control errors.
        with self.assertRaises(TypeError):
            store.search_combinations(
                "main", series, {"x": 1}, True, {}, -1, 2, (), (), 10,
            )

    def test_branch_and_node_lookup_order(self) -> None:
        store = make_store()
        series = make_series()
        with self.assertRaises(KeyError) as caught:
            store.search_combinations(
                "ghost", series, {"x": 1}, 5, {}, 0, 2, (), (), 10,
            )
        self.assertEqual(caught.exception.args, ("ghost",))
        with self.assertRaises(KeyError) as caught:
            store.search_combinations(
                "main",
                {"ghost1": (), "ghost2": ()},
                {"x": 1}, 5, {}, 0, 2, (), (), 10,
            )
        self.assertEqual(caught.exception.args, ("ghost1",))
        with self.assertRaises(KeyError) as caught:
            store.search_combinations(
                "main",
                {"feature": (("nope1", "f1"),), "ghost": (("nope1", "m1"),)},
                {"x": 1}, 5, {}, 0, 2, (), (), 10,
            )
        self.assertEqual(caught.exception.args, ("nope1",))
        with self.assertRaises(KeyError) as caught:
            store.search_combinations(
                "main",
                {"feature": (("m1", "nope2"), ("nope1", "f1"))},
                {"x": 1}, 5, {}, 0, 1, (), (), 10,
            )
        self.assertEqual(caught.exception.args, ("nope2",))
        # A node outside its branch's head closure is rejected.
        with self.assertRaises(KeyError):
            store.search_combinations(
                "main",
                {"feature": (("f2", "f1"),)},
                {"x": 1}, 5, {}, 0, 1, (), (), 10,
            )

    def test_results_are_detached_and_unshared(self) -> None:
        store = make_store()
        series = make_series()
        first = store.search_combinations(
            "main", series, {"x": 2, "y": 3, "z": 1}, 20, {},
            0, 2, (), (), 10,
        )
        self.assertIsNot(first["frontier"], first["rejected"])
        pristine = copy.deepcopy(first)
        for entry in first["frontier"]:
            entry["members"] += ("evil",)
            entry["contributions"] += (999,)
            entry["attributions"] += ({},)
        for entry in first["rejected"]:
            entry["members"] += ("evil",)
            entry["overrun"] = -999
        fresh = store.search_combinations(
            "main", series, {"x": 2, "y": 3, "z": 1}, 20, {},
            0, 2, (), (), 10,
        )
        self.assertEqual(fresh, pristine)

    def test_float_arithmetic_and_zero_signs(self) -> None:
        store = make_store()
        result = store.search_combinations(
            "main", make_series(), {"x": 0.5}, 1.0, {},
            1, 2, (), (), 10,
        )
        # (feature,) and (side,) aggregate x to 1 and 2 at both points:
        # risks 0.5 and 1.0 stay feasible; (feature,side) risks 1.5 > 1.0.
        rejected = result["rejected"]
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["reason"], "total_budget")
        self.assertEqual(rejected[0]["overrun"], 0.5)
        self.assertIsInstance(rejected[0]["overrun"], float)
        # (side,) is feasible but dominated by the equal-size, lower-risk
        # (feature,), so the frontier holds only (feature,).
        self.assertEqual(
            [entry["members"] for entry in result["frontier"]],
            [("feature",)],
        )
        entry = result["frontier"][0]
        self.assertEqual(entry["risk"], 0.5)
        self.assertIsInstance(entry["risk"], float)
        self.assertEqual(entry["contributions"], (0.5,))
        # A negative zero weight keeps zero risks positive zero.
        result = store.search_combinations(
            "main", make_series(), {"x": -0.0}, 0.0, {},
            0, 2, (), (), 10,
        )
        for entry in result["frontier"]:
            self.assertGreater(math.copysign(1.0, entry["risk"]), 0)
            for contribution in entry["contributions"]:
                self.assertGreater(math.copysign(1.0, contribution), 0)

    def test_query_is_read_only(self) -> None:
        store = make_store()
        heads = dict(store._heads)
        appends = copy.deepcopy(store._appends)
        merges = copy.deepcopy(store._merges)
        replay_main = store.replay("main")
        replay_feature = store.replay("feature")
        audit = store.audit_log()
        series = make_series()
        store.search_combinations(
            "main", series, {"x": 2, "y": 3, "z": 1}, 10, {"x": 0},
            0, 2, (), (), 10,
        )
        # Failing calls must leave state untouched as well.
        with self.assertRaises(KeyError):
            store.search_combinations(
                "main",
                {"feature": (("m1", "f1"), ("nope", "f2"))},
                {"x": 1}, 10, {}, 0, 1, (), (), 10,
            )
        with self.assertRaises(ValueError):
            store.search_combinations(
                "main", series, {"x": 1}, 10, {}, 0, 2, (), (), 2,
            )
        with self.assertRaises(ValueError):
            store.search_combinations(
                "main", series, {"x": 1}, float("nan"), {}, 0, 2, (), (), 10,
            )
        self.assertEqual(dict(store._heads), heads)
        self.assertEqual(copy.deepcopy(store._appends), appends)
        self.assertEqual(copy.deepcopy(store._merges), merges)
        self.assertEqual(store.replay("main"), replay_main)
        self.assertEqual(store.replay("feature"), replay_feature)
        self.assertEqual(store.audit_log(), audit)
        # The existing combination budget and divergence interfaces keep
        # their results.
        combination = store.combination_budget(
            "main", series, {"both": ("feature", "side")},
            {"x": 2, "y": 3, "z": 1}, 10, {"x": 0},
        )
        matrix = store.divergence_matrix(
            "main", {"feature": (("m2", "f2"), ("m1", "f1"))}, ("z", "x"),
        )
        store.search_combinations(
            "main", series, {"z": 1, "x": 1}, 0, {"x": 0},
            0, 2, ("feature",), (), 10,
        )
        self.assertEqual(
            store.combination_budget(
                "main", series, {"both": ("feature", "side")},
                {"x": 2, "y": 3, "z": 1}, 10, {"x": 0},
            ),
            combination,
        )
        self.assertEqual(
            store.divergence_matrix(
                "main", {"feature": (("m2", "f2"), ("m1", "f1"))}, ("z", "x"),
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
