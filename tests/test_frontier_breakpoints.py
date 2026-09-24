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


def base(
    name: str = "base",
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


def call(
    store: BranchStore,
    axis: str = "total_budget",
    values: tuple = (15, 16, 100),
    base_scenario: dict | None = None,
    **overrides,
):
    arguments = {
        "reference": "main",
        "series": make_series(),
        "base_scenario": base() if base_scenario is None else base_scenario,
        "axis": axis,
        "values": values,
        "min_size": 0,
        "max_size": 2,
        "required": (),
        "exclusive_pairs": (),
        "limit": 10,
    }
    arguments.update(overrides)
    return store.frontier_breakpoints(**arguments)


class FrontierBreakpointsTests(unittest.TestCase):
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

    def test_result_shape_and_key_order(self) -> None:
        store = make_store()
        result = call(store)
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
                    "left",
                    "right",
                    "left_result",
                    "right_result",
                    "entered",
                    "exited",
                    "affected",
                    "member_delta",
                ],
            )

    def test_each_point_matches_one_existing_search(self) -> None:
        store = make_store()
        series = make_series()
        values = (15, 16, 100)
        result = call(store, values=values)
        for value, point in zip(values, result["points"]):
            self.assertEqual(point["value"], value)
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

    def test_total_budget_axis_breakpoints(self) -> None:
        store = make_store()
        # Final-checkpoint risks: () 0, feature 18, side 16, pair 16 --
        # but the pair breaches any budget below 21 at the first
        # checkpoint, so it is infeasible at 15 and 16.
        # budget 15: only () feasible -> frontier [()].
        # budget 16: side joins the feasible set -> frontier [(), side].
        # budget 100: all feasible; the pair dominates side and feature
        # -> frontier [(), pair].
        result = call(store, values=(15, 16, 100))
        self.assertEqual(len(result["breakpoints"]), 2)

        first, second = result["breakpoints"]
        self.assertEqual(first["left"], 15)
        self.assertEqual(first["right"], 16)
        self.assertEqual(first["entered"], (("side",),))
        self.assertEqual(first["exited"], ())
        self.assertEqual(first["affected"], ())
        self.assertEqual(first["member_delta"], {})

        self.assertEqual(second["left"], 16)
        self.assertEqual(second["right"], 100)
        self.assertEqual(second["entered"], (("feature", "side"),))
        self.assertEqual(second["exited"], (("side",),))
        # feature becomes feasible but stays dominated off the frontier.
        self.assertEqual(second["affected"], (("feature",),))
        self.assertEqual(second["member_delta"], {("feature",): None})

    def test_weight_axis_breakpoint_with_member_delta(self) -> None:
        store = make_store()
        # Perturbing the z weight: feature risk 11+7w, side 13+3w,
        # pair 6+10w. At w=1 the frontier is [pair, ()]; at w=2 side
        # joins the frontier and feature's dominator set changes.
        result = call(store, axis="z", values=(1, 2))
        self.assertEqual(len(result["breakpoints"]), 1)
        record = result["breakpoints"][0]
        self.assertEqual(record["left"], 1)
        self.assertEqual(record["right"], 2)
        self.assertEqual(record["entered"], (("side",),))
        self.assertEqual(record["exited"], ())
        # feature's dominators change from (pair, side) to (side,); each
        # combination appears in exactly one of the record's lists.
        self.assertEqual(record["affected"], (("feature",),))
        self.assertEqual(record["member_delta"], {("feature",): (7,)})

    def test_identical_adjacent_results_generate_no_record(self) -> None:
        store = make_store()
        # Budgets 16 and 17 leave feasibility, dominators and frontier
        # untouched (feature's risk 18 breaches both).
        result = call(store, values=(16, 17))
        self.assertEqual(result["breakpoints"], ())
        self.assertEqual(len(result["points"]), 2)

    def test_record_results_are_independent_copies(self) -> None:
        store = make_store()
        result = call(store, values=(15, 16))
        record = result["breakpoints"][0]
        left_point, right_point = result["points"]
        self.assertEqual(
            record["left_result"], _plain(left_point)
        )
        self.assertEqual(
            record["right_result"], _plain(right_point)
        )
        self.assertIsNot(record["left_result"], left_point)
        self.assertIsNot(
            record["left_result"]["frontier"], left_point["frontier"]
        )
        self.assertIsNot(
            record["right_result"]["frontier"], right_point["frontier"]
        )

    def test_result_detached_from_internal_state(self) -> None:
        store = make_store()
        result = call(store, values=(15, 16, 100))
        snapshot = copy.deepcopy(_plain(result))
        # Mutating every returned level must not change later calls.
        for point in result["points"]:
            point["frontier"][0]["contributions"] += (999,)
            point["rejected"] += ({"members": ()},)
        for record in result["breakpoints"]:
            record["entered"] += (("bogus",),)
            record["member_delta"][("nope",)] = None
        again = call(store, values=(15, 16, 100))
        self.assertEqual(_plain(again), snapshot)

    def test_read_only_success_and_failure(self) -> None:
        store = make_store()
        heads_before = {
            name: store.head(name) for name in ("main", "feature", "side")
        }
        audit_before = store.audit_log()
        call(store, values=(15, 16, 100))
        with self.assertRaises(ValueError):
            call(store, values=(2, 1))
        with self.assertRaises(KeyError):
            call(store, reference="unknown")
        self.assertEqual(
            heads_before,
            {name: store.head(name) for name in heads_before},
        )
        self.assertEqual(audit_before, store.audit_log())
        self.assertEqual(store.replay("main"), {"x": 1, "y": 2, "z": 7})

    def test_empty_pool_still_validates_and_runs(self) -> None:
        graph = EventGraph()
        graph.add("root", 0, (), {})
        store = BranchStore(graph)
        store.create("main", "root")
        result = store.frontier_breakpoints(
            "main", {}, base(), "total_budget", (0, 5), 0, 0, (), (), 10
        )
        self.assertEqual(len(result["points"]), 2)
        self.assertEqual(result["breakpoints"], ())
        # Size-window checks still run against the empty pool.
        with self.assertRaises(ValueError):
            store.frontier_breakpoints(
                "main", {}, base(), "total_budget", (0, 5), 0, 1, (), (), 10
            )
        # The reference branch is still looked up.
        with self.assertRaises(KeyError):
            store.frontier_breakpoints(
                "unknown", {}, base(), "total_budget", (0, 5), 0, 0, (), (), 10
            )

    def test_candidate_limit_raises_without_partial_result(self) -> None:
        store = make_store()
        with self.assertRaises(ValueError):
            call(store, limit=3)

    def test_unknown_branch_and_node_errors(self) -> None:
        store = make_store()
        with self.assertRaises(KeyError):
            call(store, reference="unknown")
        with self.assertRaises(KeyError):
            call(
                store,
                series={"unknown": (("m1", "f1"), ("m2", "f2"))},
                max_size=1,
            )
        with self.assertRaises(KeyError):
            call(
                store,
                series={"feature": (("m1", "nope"), ("m2", "f2"))},
                max_size=1,
            )
        with self.assertRaises(KeyError):
            call(
                store,
                series={"feature": (("nope", "f1"), ("m2", "f2"))},
                max_size=1,
            )
        # A node outside the named branch's head closure.
        with self.assertRaises(KeyError):
            call(
                store,
                series={"feature": (("f1", "f1"), ("m2", "f2"))},
                max_size=1,
            )

    def test_misaligned_checkpoints_raise(self) -> None:
        store = make_store()
        series = {
            "feature": (("m1", "f1"), ("m2", "f2")),
            "side": (("m2", "s1"), ("m2", "s2")),
        }
        with self.assertRaises(ValueError):
            call(store, series=series)

    def test_base_scenario_validation(self) -> None:
        store = make_store()
        with self.assertRaises(TypeError):
            call(store, base_scenario="base")
        with self.assertRaises(ValueError):
            bad = base()
            del bad["weights"]
            call(store, base_scenario=bad)
        with self.assertRaises(ValueError):
            bad = base()
            bad["extra"] = 1
            call(store, base_scenario=bad)
        with self.assertRaises(TypeError):
            call(store, base_scenario=base(name=1))
        with self.assertRaises(ValueError):
            call(store, base_scenario=base(name=""))
        with self.assertRaises(TypeError):
            call(store, base_scenario=base(weights={"x": True}))
        with self.assertRaises(ValueError):
            call(store, base_scenario=base(weights={"x": -1}))
        with self.assertRaises(ValueError):
            call(store, base_scenario=base(total_budget=float("nan")))
        with self.assertRaises(ValueError):
            call(store, base_scenario=base(key_budgets={"unknown": 1}))

    def test_axis_validation(self) -> None:
        store = make_store()
        with self.assertRaises(TypeError):
            call(store, axis=1)
        with self.assertRaises(ValueError):
            call(store, axis="")
        with self.assertRaises(ValueError):
            call(store, axis="unknown")
        # Every weight key is a valid axis.
        result = call(store, axis="x", values=(1, 2))
        self.assertEqual(len(result["points"]), 2)

    def test_values_validation(self) -> None:
        store = make_store()
        with self.assertRaises(TypeError):
            call(store, values=[1, 2])
        with self.assertRaises(ValueError):
            call(store, values=())
        with self.assertRaises(ValueError):
            call(store, values=(1,))
        with self.assertRaises(TypeError):
            call(store, values=(1, True))
        with self.assertRaises(TypeError):
            call(store, values=(1, "2"))
        with self.assertRaises(ValueError):
            call(store, values=(-1, 2))
        with self.assertRaises(ValueError):
            call(store, values=(1, float("nan")))
        with self.assertRaises(ValueError):
            call(store, values=(1, float("inf")))
        with self.assertRaises(ValueError):
            call(store, values=(2, 1))
        with self.assertRaises(ValueError):
            call(store, values=(1, 1))
        # Zero and mixed int/float values are accepted.
        result = call(store, values=(0, 0.5, 2))
        self.assertEqual(len(result["points"]), 3)

    def test_search_parameter_validation_reused(self) -> None:
        store = make_store()
        with self.assertRaises(TypeError):
            call(store, min_size=True)
        with self.assertRaises(ValueError):
            call(store, min_size=-1)
        with self.assertRaises(ValueError):
            call(store, min_size=2, max_size=1)
        with self.assertRaises(ValueError):
            call(store, max_size=3)
        with self.assertRaises(TypeError):
            call(store, required=["feature"])
        with self.assertRaises(ValueError):
            call(store, required=("unknown",))
        with self.assertRaises(TypeError):
            call(store, exclusive_pairs=[("feature", "side")])
        with self.assertRaises(ValueError):
            call(
                store,
                required=("feature", "side"),
                exclusive_pairs=(("feature", "side"),),
            )
        with self.assertRaises(ValueError):
            call(store, limit=0)

    def test_required_and_exclusive_constraints_apply_per_point(self) -> None:
        store = make_store()
        result = call(
            store,
            values=(15, 100),
            required=("side",),
            exclusive_pairs=(),
        )
        for point in result["points"]:
            reasons = {
                tuple(row["members"]): row["reason"]
                for row in point["rejected"]
            }
            self.assertEqual(reasons[()], "missing_required")
            self.assertEqual(reasons[("feature",)], "missing_required")


def _plain(value):
    """Deep-convert nested dict/tuple structures for equality snapshots."""
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(_plain(item) for item in value)
    return value


if __name__ == "__main__":
    unittest.main()
