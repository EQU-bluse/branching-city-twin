import copy
import inspect
import unittest

from city_twin.branches import BranchStore
from tests.test_lifetime_evolution import (
    compare_kwargs,
    diamond_store,
    evolution_kwargs,
)


def churn_kwargs(store_args, **overrides) -> dict:
    kwargs = evolution_kwargs(store_args)
    kwargs["churn_limit"] = 50
    kwargs.update(overrides)
    return kwargs


def churn_call(store, store_args, **overrides):
    return store.lifetime_churn(**churn_kwargs(store_args, **overrides))


class LifetimeChurnSignatureTests(unittest.TestCase):
    def test_public_signature_matches_evolution_plus_churn_limit(self):
        parameters = list(
            inspect.signature(BranchStore.lifetime_churn).parameters.values()
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
                "windows",
                "causes",
                "direction",
                "depth",
                "node_limit",
                "change_limit",
                "lifetime_limit",
                "diff_limit",
                "window_limit",
                "total_diff_limit",
                "churn_limit",
                "token",
            ],
        )
        self.assertEqual(
            [p.default for p in parameters[:-1]],
            [inspect.Parameter.empty] * (len(parameters) - 1),
        )
        self.assertIsNone(parameters[-1].default)

    def test_result_is_a_dict_with_identities_and_totals(self):
        store, args = diamond_store()
        result = churn_call(store, args)
        self.assertIsInstance(result, dict)
        self.assertEqual(list(result), ["identities", "totals"])
        self.assertIsInstance(result["identities"], tuple)
        for record in result["identities"]:
            self.assertIsInstance(record, tuple)
            self.assertEqual(len(record), 7)
            self.assertIn(record[0], ("node", "edge", "gap"))
            self.assertIsInstance(record[2], tuple)
            self.assertIsInstance(record[6], tuple)
            for change in record[6]:
                self.assertIsInstance(change, tuple)
                self.assertEqual(len(change), 5)
                self.assertEqual(change[0] + 1, change[1])
                self.assertTrue(change[2].startswith(record[0]))
        self.assertEqual(result["totals"], (3, 2, 0, 4))


class LifetimeChurnResultTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        return churn_call(self.store, self.args, **overrides)

    def test_records_follow_the_fixture(self):
        result = self.call()
        summaries = [
            (record[0], record[1], record[2], record[3], record[4], record[5])
            for record in result["identities"]
        ]
        self.assertEqual(
            summaries,
            [
                ("node", ("a0", "x"), (0, 1, 2), 0, 0, 2),
                ("node", ("W", "y"), (1, 2), 1, 0, 1),
                ("edge", (("a0", "x"), ("W", "y")), (1, 2), 1, 0, 1),
            ],
        )

    def test_present_uses_zero_based_window_positions(self):
        result = self.call(windows=((0, 2), (1,), (0, 1, 2)))
        present = {
            (record[0], record[1]): record[2] for record in result["identities"]
        }
        self.assertEqual(present[("node", ("a0", "x"))], (0, 1, 2))
        self.assertEqual(present[("node", ("W", "y"))], (0, 1, 2))
        self.assertEqual(present[("edge", (("a0", "x"), ("W", "y")))], (0, 1, 2))

    def test_changes_keep_segment_endpoints_kind_and_evidence(self):
        result = self.call()
        evolution = self.store.lifetime_evolution(**evolution_kwargs(self.args))
        for record in result["identities"]:
            for segment_position, to_position, kind, before, after in record[6]:
                self.assertEqual(to_position, segment_position + 1)
                segment = evolution["segments"][segment_position]
                matches = [
                    change
                    for change in segment["changes"]
                    if change["kind"] == kind
                    and change["identity"] == record[1]
                ]
                self.assertEqual(len(matches), 1)
                self.assertEqual(before, matches[0]["before"])
                self.assertEqual(after, matches[0]["after"])

    def test_changes_are_in_segment_order(self):
        result = self.call()
        a0 = next(
            record
            for record in result["identities"]
            if record[1] == ("a0", "x")
        )
        self.assertEqual(
            [(change[0], change[1], change[2]) for change in a0[6]],
            [(0, 1, "node_changed"), (1, 2, "node_changed")],
        )

    def test_ordering_counter_sum_desc_then_first_presence_asc(self):
        result = self.call()
        records = result["identities"]
        scores = [record[3] + record[4] + record[5] for record in records]
        self.assertEqual(scores, sorted(scores, reverse=True))
        firsts = [record[2][0] for record in records]
        # a0 (first 0) outranks W and the edge (first 1) on the score tie.
        self.assertEqual(firsts[0], 0)

    def test_ties_keep_type_and_identity_order(self):
        # Window (2,) then (0,): W and the edge are removed, a0 shifts: all
        # three score one and first appear in window 0.
        result = self.call(windows=((2,), (0,)))
        self.assertEqual(
            [(record[0], record[1]) for record in result["identities"]],
            [
                ("node", ("a0", "x")),
                ("node", ("W", "y")),
                ("edge", (("a0", "x"), ("W", "y"))),
            ],
        )
        self.assertEqual(result["totals"], (3, 0, 2, 1))

    def test_identical_adjacent_windows_create_no_changes(self):
        result = self.call(windows=((0, 1), (0, 1), (0, 1)))
        for record in result["identities"]:
            self.assertEqual(record[3:6], (0, 0, 0))
            self.assertEqual(record[6], ())
            self.assertEqual(record[2], (0, 1, 2))
        self.assertEqual(result["totals"], (3, 0, 0, 0))

    def test_single_window_has_presence_without_changes(self):
        result = self.call(windows=((0, 1, 2),))
        self.assertEqual(result["totals"][0], 3)
        self.assertEqual(result["totals"][1:], (0, 0, 0))
        for record in result["identities"]:
            self.assertEqual(record[2], (0,))

    def test_empty_windows_return_empty_identities_and_four_zeros(self):
        result = self.call(windows=())
        self.assertEqual(result, {"identities": (), "totals": (0, 0, 0, 0)})

    def test_empty_causes_return_empty_identities_and_four_zeros(self):
        result = self.call(causes=())
        self.assertEqual(result, {"identities": (), "totals": (0, 0, 0, 0)})

    def test_matches_pairwise_evolution_aggregation(self):
        windows = ((0,), (2,), (0, 1))
        result = self.call(windows=windows)
        evolution = self.store.lifetime_evolution(
            **evolution_kwargs(self.args, windows=windows)
        )
        expected_present = {}
        counters = {}
        collected = {}
        for position, window in enumerate(evolution["windows"]):
            for lifetime in window["lifetimes"]:
                key = (lifetime["type"], lifetime["identity"])
                expected_present.setdefault(key, []).append(position)
                collected.setdefault(key, [])
        for segment in evolution["segments"]:
            for change in segment["changes"]:
                type_name = change["kind"].split("_", 1)[0]
                key = (type_name, change["identity"])
                counters.setdefault(key, [0, 0, 0])
                suffix = change["kind"].split("_", 1)[1]
                counters[key][["added", "removed", "changed"].index(suffix)] += 1
                collected.setdefault(key, []).append(
                    (segment["left"], segment["right"], change["kind"])
                )
        for record in result["identities"]:
            key = (record[0], record[1])
            self.assertEqual(record[2], tuple(expected_present[key]))
            self.assertEqual(record[3:6], tuple(counters.get(key, [0, 0, 0])))
            self.assertEqual(
                [(c[0], c[1], c[2]) for c in record[6]], collected[key]
            )


class LifetimeChurnLimitValidationTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        overrides.setdefault("causes", ("a0",))
        return churn_call(self.store, self.args, **overrides)

    def test_churn_limit_must_be_a_non_bool_positive_int(self):
        for bad in (True, False, 1.0, "1", None):
            with self.subTest(churn_limit=bad):
                with self.assertRaises(TypeError):
                    self.call(churn_limit=bad)
        with self.assertRaises(ValueError):
            self.call(churn_limit=0)
        with self.assertRaises(ValueError):
            self.call(churn_limit=-3)

    def test_churn_limit_validated_after_total_diff_limit(self):
        # total_diff_limit is fully checked (type then range) before
        # churn_limit's own checks run.
        with self.assertRaises(ValueError):
            self.call(total_diff_limit=0, churn_limit="x")
        with self.assertRaises(TypeError):
            self.call(total_diff_limit=1, churn_limit="x")
        with self.assertRaises(TypeError):
            self.call(total_diff_limit="x", churn_limit=0)
        with self.assertRaises(TypeError):
            self.call(total_diff_limit="x", churn_limit=1)
        with self.assertRaises(ValueError):
            self.call(total_diff_limit=1, churn_limit=0)

    def test_churn_limit_validated_before_state_lookups(self):
        with self.assertRaises(TypeError):
            self.call(reference="ghost", churn_limit="x")
        with self.assertRaises(ValueError):
            self.call(reference="ghost", churn_limit=0)
        with self.assertRaises(KeyError):
            self.call(reference="ghost")

    def test_churn_cap_runs_after_full_aggregation_with_no_partial_result(self):
        with self.assertRaises(ValueError) as caught:
            self.call(churn_limit=2)
        self.assertIn("lifetime churn limit", str(caught.exception))
        result = self.call(churn_limit=3)
        self.assertEqual(len(result["identities"]), 3)

    def test_existing_caps_keep_their_order(self):
        with self.assertRaises(ValueError):
            self.call(window_limit=2)
        with self.assertRaises(ValueError):
            self.call(diff_limit=2)
        with self.assertRaises(ValueError):
            self.call(total_diff_limit=5)
        with self.assertRaises(ValueError):
            self.call(lifetime_limit=2, windows=((0,), (0, 1, 2)))
        with self.assertRaises(ValueError):
            self.call(change_limit=1, windows=((0, 1),))
        with self.assertRaises(ValueError):
            self.call(limit=1)

    def test_window_validation_keeps_existing_contracts(self):
        for bad_windows in ("x", ((0,), "x"), ((True,),), ((-1,),), ((3,),)):
            with self.subTest(windows=bad_windows):
                with self.assertRaises((TypeError, ValueError)):
                    self.call(windows=bad_windows)


class LifetimeChurnStateCheckTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        overrides.setdefault("causes", ("a0",))
        return churn_call(self.store, self.args, **overrides)

    def test_unknown_branch_history_node_and_cause_raise_key_error(self):
        with self.assertRaises(KeyError):
            self.call(reference="ghost")
        with self.assertRaises(KeyError):
            self.call(
                series={
                    "a": (("r0", "a0"), ("r1", "a0"), ("r2", "a0")),
                    "b": (("r0", "a0"), ("r1", "W"), ("r2", "nope2")),
                }
            )
        with self.assertRaises(KeyError) as caught:
            self.call(causes=("zzz",))
        self.assertEqual(caught.exception.args[0], "zzz")

    def test_misaligned_series_points_raise_value_error(self):
        with self.assertRaises(ValueError):
            self.call(
                series={
                    "a": (("r0", "a0"), ("r1", "a0"), ("r2", "a0")),
                    "b": (("r1", "a0"), ("r1", "W"), ("r2", "W")),
                }
            )

    def test_empty_windows_still_run_every_check(self):
        with self.assertRaises(KeyError):
            self.call(windows=(), reference="ghost")
        with self.assertRaises(ValueError):
            self.call(windows=(), limit=1)
        with self.assertRaises(ValueError):
            self.call(windows=(), churn_limit=0)
        with self.assertRaises(TypeError):
            self.call(windows=(), churn_limit=True)


class LifetimeChurnIsolationTests(unittest.TestCase):
    def test_result_levels_are_mutually_unshared(self):
        store, args = diamond_store()
        result = churn_call(store, args)
        seen_ids = []

        def collect(value):
            if isinstance(value, dict):
                seen_ids.append(id(value))
                for item in value.values():
                    collect(item)
            elif isinstance(value, tuple):
                for item in value:
                    collect(item)

        collect(result)
        self.assertEqual(len(seen_ids), len(set(seen_ids)))

    def test_mutating_result_never_affects_later_calls(self):
        store, args = diamond_store()
        first = churn_call(store, args)
        pristine = copy.deepcopy(first)
        for record in first["identities"]:
            for change in record[6]:
                for evidence in (change[3], change[4]):
                    if evidence is not None:
                        evidence["first_seen"] = 99
                        if "branches" in evidence:
                            evidence["branches"] += ("evil",)
        second = churn_call(store, args)
        self.assertEqual(second, pristine)

    def test_repeated_calls_are_equal(self):
        store, args = diamond_store()
        self.assertEqual(churn_call(store, args), churn_call(store, args))


class LifetimeChurnReadOnlyTests(unittest.TestCase):
    def test_success_and_failure_change_nothing(self):
        store, args = diamond_store()
        heads = dict(store._heads)
        appends = copy.deepcopy(store._appends)
        merges = copy.deepcopy(store._merges)
        audit = store.audit_log()

        churn_call(store, args)
        failures = [
            dict(windows="x"),
            dict(windows=("x",)),
            dict(windows=((-1,),)),
            dict(windows=((9,),)),
            dict(causes=[1]),
            dict(direction="up"),
            dict(node_limit=0),
            dict(change_limit=0),
            dict(lifetime_limit=0),
            dict(diff_limit=0),
            dict(window_limit=0),
            dict(total_diff_limit=0),
            dict(churn_limit=0),
            dict(churn_limit="x"),
            dict(churn_limit=True),
            dict(churn_limit=2),
            dict(total_diff_limit=1),
            dict(causes=("zzz",)),
            dict(limit=1),
            dict(reference="ghost"),
        ]
        for overrides in failures:
            with self.subTest(overrides=overrides):
                with self.assertRaises(Exception):
                    churn_call(store, args, causes=("a0",), **overrides)

        self.assertEqual(dict(store._heads), heads)
        self.assertEqual(copy.deepcopy(store._appends), appends)
        self.assertEqual(copy.deepcopy(store._merges), merges)
        self.assertEqual(store.audit_log(), audit)

    def test_existing_queries_are_unaffected(self):
        store, args = diamond_store()
        evolution = store.lifetime_evolution(**evolution_kwargs(args))
        compared = store.compare_lifetimes(
            **compare_kwargs(args, left_indices=(0, 1), right_indices=(1, 2))
        )
        churn_call(store, args)
        self.assertEqual(
            store.lifetime_evolution(**evolution_kwargs(args)), evolution
        )
        self.assertEqual(
            store.compare_lifetimes(
                **compare_kwargs(args, left_indices=(0, 1), right_indices=(1, 2))
            ),
            compared,
        )


class LifetimeChurnTokenTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, token=None, **overrides):
        return self.store.lifetime_churn(
            token=token, **churn_kwargs(self.args, **overrides)
        )

    def test_successful_call_consumes_one_read(self):
        token = self.store.create_snapshot(max_reads=2)
        self.call(token=token)
        self.call(token=token)
        with self.assertRaises(RuntimeError):
            self.call(token=token)

    def test_failed_call_refunds_the_read(self):
        token = self.store.create_snapshot(max_reads=1)
        with self.assertRaises(ValueError):
            self.call(token=token, churn_limit=0)
        result = self.call(token=token)
        self.assertEqual(result["totals"], (3, 2, 0, 4))
        with self.assertRaises(RuntimeError):
            self.call(token=token)

    def test_parameter_failure_never_reserves_a_read(self):
        token = self.store.create_snapshot(max_reads=1)
        with self.assertRaises(TypeError):
            self.call(token=token, churn_limit=True)
        self.call(token=token)

    def test_unknown_and_released_tokens_raise_key_error(self):
        with self.assertRaises(KeyError):
            self.call(token="nope")
        token = self.store.create_snapshot(max_reads=2)
        self.store.release_snapshot(token)
        with self.assertRaises(KeyError):
            self.call(token=token)

    def test_results_come_from_the_frozen_view(self):
        token = self.store.create_snapshot(max_reads=3)
        before = self.call(token=token)
        # Extend heads after the snapshot exactly like FrozenViewTests.
        self.store.append("a", "a2", 7, {"x": -1})
        self.store.create("d", "root")
        self.store.append("d", "d0", 8, {"z": 3})
        self.store.merge("a", "d", "M", 9, {"z": 1})
        self.store.append("main", "r3", 10, {"x": 1})

        # The base plans do not reference the new events, so tokenized
        # results keep the exact pre-mutation instant.
        self.assertEqual(self.call(token=token), before)

        # A plan on the new head event serves live but is unknown inside the
        # frozen view: both go through the same churn aggregation.
        new_head_series = {
            "a": (("r0", "a2"), ("r1", "a2"), ("r2", "a2")),
            "b": (("r0", "a0"), ("r1", "W"), ("r2", "W")),
        }
        self.call(series=new_head_series)
        with self.assertRaises(KeyError):
            self.call(token=token, series=new_head_series)


if __name__ == "__main__":
    unittest.main()
