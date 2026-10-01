import copy
import inspect
import unittest

from city_twin.branches import BranchStore

from tests.test_lifetime_churn import (
    churn_call,
    churn_kwargs,
    diamond_store,
)


def waves_kwargs(store_args, **overrides) -> dict:
    kwargs = churn_kwargs(store_args, **overrides)
    kwargs["streak_limit"] = overrides.pop("streak_limit", 50)
    kwargs["min_identities"] = overrides.pop("min_identities", 1)
    kwargs["wave_limit"] = overrides.pop("wave_limit", 50)
    return kwargs


def waves_call(store, store_args, **overrides):
    return store.lifetime_churn_waves(
        **waves_kwargs(store_args, **overrides)
    )


def reference_waves(
    evolution: dict[str, object], min_identities: int
) -> dict[str, object]:
    """Independently group a lifetime_evolution result into waves."""
    type_order = {"node": 0, "edge": 1, "gap": 2}
    segment_changes = [
        segment["changes"] for segment in evolution["segments"]
    ]

    runs = []
    run = []
    for position, changes in enumerate(segment_changes):
        distinct = {
            (change["kind"].split("_", 1)[0], change["identity"])
            for change in changes
        }
        if len(distinct) >= min_identities:
            run.append(position)
        elif run:
            runs.append(run)
            run = []
    if run:
        runs.append(run)

    waves = []
    all_identities = set()
    for positions in runs:
        members = {}
        encounter = {}
        first_segment = {}
        evidence = []
        for position in positions:
            for change in segment_changes[position]:
                key = (
                    change["kind"].split("_", 1)[0],
                    change["identity"],
                )
                if key not in members:
                    members[key] = {
                        "segments": set(),
                        "added": 0,
                        "removed": 0,
                        "changed": 0,
                    }
                    encounter[key] = len(encounter)
                    first_segment[key] = position
                member = members[key]
                member["segments"].add(position)
                if change["kind"].endswith("_added"):
                    member["added"] += 1
                elif change["kind"].endswith("_removed"):
                    member["removed"] += 1
                else:
                    member["changed"] += 1
                evidence.append(
                    {
                        "from": position,
                        "to": position + 1,
                        "kind": change["kind"],
                        "before": change["before"],
                        "after": change["after"],
                    }
                )
        ordered = sorted(
            members,
            key=lambda key: (
                -len(members[key]["segments"]),
                -(
                    members[key]["added"]
                    + members[key]["removed"]
                    + members[key]["changed"]
                ),
                first_segment[key],
                type_order[key[0]],
                encounter[key],
            ),
        )
        all_identities.update(members)
        waves.append(
            {
                "from": positions[0],
                "to": positions[-1] + 1,
                "segments": len(positions),
                "identities": len(members),
                "added": sum(m["added"] for m in members.values()),
                "removed": sum(m["removed"] for m in members.values()),
                "changed": sum(m["changed"] for m in members.values()),
                "members": tuple(
                    {
                        "type": key[0],
                        "identity": key[1],
                        "segments": len(members[key]["segments"]),
                        "added": members[key]["added"],
                        "removed": members[key]["removed"],
                        "changed": members[key]["changed"],
                    }
                    for key in ordered
                ),
                "changes": tuple(evidence),
            }
        )

    waves.sort(key=lambda wave: (wave["from"], wave["to"]))
    return {
        "waves": tuple(waves),
        "totals": {
            "waves": len(waves),
            "identities": len(all_identities),
            "added": sum(w["added"] for w in waves),
            "removed": sum(w["removed"] for w in waves),
            "changed": sum(w["changed"] for w in waves),
        },
    }


class LifetimeChurnWavesSignatureTests(unittest.TestCase):
    def test_public_signature_appends_min_identities_and_wave_limit(self):
        parameters = list(
            inspect.signature(
                BranchStore.lifetime_churn_waves
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
                "streak_limit",
                "min_identities",
                "wave_limit",
                "token",
            ],
        )
        self.assertEqual(
            [p.default for p in parameters[:-1]],
            [inspect.Parameter.empty] * (len(parameters) - 1),
        )
        self.assertIsNone(parameters[-1].default)

    def test_result_shape_and_key_order(self):
        store, args = diamond_store()
        result = waves_call(store, args)
        self.assertIsInstance(result, dict)
        self.assertEqual(list(result), ["waves", "totals"])
        self.assertIsInstance(result["waves"], tuple)
        self.assertEqual(
            list(result["totals"]),
            ["waves", "identities", "added", "removed", "changed"],
        )
        for record in result["waves"]:
            self.assertIsInstance(record, dict)
            self.assertEqual(
                list(record),
                [
                    "from",
                    "to",
                    "segments",
                    "identities",
                    "added",
                    "removed",
                    "changed",
                    "members",
                    "changes",
                ],
            )
            self.assertIsInstance(record["members"], tuple)
            self.assertIsInstance(record["changes"], tuple)
            for member in record["members"]:
                self.assertEqual(
                    list(member),
                    ["type", "identity", "segments",
                     "added", "removed", "changed"],
                )
            for change in record["changes"]:
                self.assertEqual(
                    list(change), ["from", "to", "kind", "before", "after"]
                )


class LifetimeChurnWavesResultTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        return waves_call(self.store, self.args, **overrides)

    def evolution(self, **overrides):
        kwargs = churn_kwargs(self.args, **overrides)
        kwargs.pop("churn_limit")
        kwargs.pop("streak_limit", None)
        kwargs.pop("min_identities", None)
        kwargs.pop("wave_limit", None)
        return self.store.lifetime_evolution(**kwargs)

    def test_matches_independent_grouping_of_evolution(self):
        for windows, min_identities in (
            (((0,), (1,), (2,)), 1),
            (((0,), (1,), (2,)), 2),
            (((0,), (1,), (2,)), 3),
            (((0,), (1,), (2,)), 4),
            (((0, 2), (1,), (0, 1, 2)), 1),
            (((0, 2), (1,), (0, 1, 2)), 2),
            (((2,), (0,)), 1),
            (((1, 2), (0, 1)), 3),
            (((0, 1), (0, 1), (0, 1)), 1),
            (((0,), (1,), (1,), (2,)), 1),
            (((0,), (1,), (1,), (2,)), 3),
            (((0, 1, 2),), 1),
        ):
            with self.subTest(windows=windows,
                              min_identities=min_identities):
                self.assertEqual(
                    self.call(windows=windows,
                              min_identities=min_identities),
                    reference_waves(
                        self.evolution(windows=windows), min_identities
                    ),
                )

    def test_diamond_fixture_all_segments_form_one_wave(self):
        result = self.call()
        self.assertEqual(len(result["waves"]), 1)
        wave = result["waves"][0]
        self.assertEqual((wave["from"], wave["to"]), (0, 2))
        self.assertEqual(wave["segments"], 2)
        self.assertEqual(wave["identities"], 3)
        self.assertEqual(
            (wave["added"], wave["removed"], wave["changed"]), (2, 0, 4)
        )
        self.assertEqual(len(wave["members"]), 3)
        self.assertEqual(len(wave["changes"]), 6)
        self.assertEqual(
            result["totals"],
            {"waves": 1, "identities": 3, "added": 2, "removed": 0,
             "changed": 4},
        )

    def test_min_identities_above_segment_count_yields_no_waves(self):
        result = self.call(min_identities=4)
        self.assertEqual(result["waves"], ())
        self.assertEqual(
            result["totals"],
            {"waves": 0, "identities": 0, "added": 0, "removed": 0,
             "changed": 0},
        )

    def test_quiet_segment_cuts_the_wave(self):
        # The middle window repeats, so the middle segment has no
        # changes and splits the batch into two one-segment waves.
        result = self.call(windows=((0,), (1,), (1,), (2,)))
        self.assertEqual(len(result["waves"]), 2)
        first, second = result["waves"]
        self.assertEqual((first["from"], first["to"]), (0, 1))
        self.assertEqual((second["from"], second["to"]), (2, 3))
        self.assertEqual(first["segments"], 1)
        self.assertEqual(second["segments"], 1)
        # Totals count distinct identities across every wave once.
        self.assertEqual(result["totals"]["waves"], 2)
        self.assertEqual(result["totals"]["identities"], 3)
        self.assertEqual(
            result["totals"]["added"],
            first["added"] + second["added"],
        )

    def test_empty_windows_causes_and_single_window_return_zeros(self):
        for overrides in (
            {"windows": ()},
            {"causes": ()},
            {"windows": ((0, 1, 2),)},
        ):
            with self.subTest(overrides=overrides):
                result = self.call(**overrides)
                self.assertEqual(result["waves"], ())
                self.assertEqual(
                    result["totals"],
                    {"waves": 0, "identities": 0, "added": 0,
                     "removed": 0, "changed": 0},
                )

    def test_members_sort_by_segments_changes_segment_type_encounter(self):
        result = self.call()
        for wave in result["waves"]:
            keys = [
                (
                    -m["segments"],
                    -(m["added"] + m["removed"] + m["changed"]),
                    {"node": 0, "edge": 1, "gap": 2}[m["type"]],
                )
                for m in wave["members"]
            ]
            self.assertEqual(keys, sorted(keys))
            self.assertEqual(
                wave["identities"], len(wave["members"])
            )

    def test_waves_sort_by_from_then_to(self):
        result = self.call(windows=((0,), (1,), (1,), (2,)))
        keys = [(w["from"], w["to"]) for w in result["waves"]]
        self.assertEqual(keys, sorted(keys))

    def test_change_evidence_keeps_lifetime_diff_semantics(self):
        result = self.call()
        evolution = self.evolution()
        segment_changes = {
            (segment["left"], segment["right"]): segment["changes"]
            for segment in evolution["segments"]
        }
        for wave in result["waves"]:
            for change in wave["changes"]:
                match = next(
                    candidate
                    for candidate in segment_changes[
                        (change["from"], change["to"])
                    ]
                    if candidate["kind"] == change["kind"]
                    and candidate["before"] == change["before"]
                    and candidate["after"] == change["after"]
                )
                self.assertEqual(change["before"], match["before"])
                self.assertEqual(change["after"], match["after"])
                if change["kind"].endswith("_added"):
                    self.assertIsNone(change["before"])
                elif change["kind"].endswith("_removed"):
                    self.assertIsNone(change["after"])


class LifetimeChurnWavesGroupingTests(unittest.TestCase):
    @staticmethod
    def _lifetime(identity):
        return {
            "type": "node",
            "identity": identity,
            "first_seen": 0,
            "last_seen": 0,
            "intervals": ((0, 0),),
            "transitions": (),
        }

    def _change(self, kind, identity, before, after):
        return {
            "kind": kind,
            "identity": identity,
            "before": (
                None if before is None else self._lifetime(before)
            ),
            "after": None if after is None else self._lifetime(after),
        }

    def test_threshold_cuts_runs_and_members_aggregate(self):
        x = ("X", "k")
        y = ("Y", "k")
        z = ("Z", "k")
        segments = [
            # Two distinct identities: qualifies for min_identities=2.
            (
                self._change("node_changed", x, x, x),
                self._change("node_removed", y, y, None),
            ),
            # Only one distinct identity: cuts the run.
            (self._change("node_added", z, None, z),),
            # Two distinct identities again: a new wave opens.
            (
                self._change("node_added", y, None, y),
                self._change("node_changed", x, x, x),
            ),
        ]
        result = BranchStore._build_lifetime_churn_waves_result(
            segments, 2, 100
        )
        self.assertEqual(len(result["waves"]), 2)

        first, second = result["waves"]
        self.assertEqual(
            (first["from"], first["to"], first["segments"]), (0, 1, 1)
        )
        self.assertEqual(
            (second["from"], second["to"], second["segments"]), (2, 3, 1)
        )
        self.assertEqual(first["identities"], 2)
        self.assertEqual(second["identities"], 2)
        self.assertEqual(
            (first["added"], first["removed"], first["changed"]),
            (0, 1, 1),
        )
        self.assertEqual(
            (second["added"], second["removed"], second["changed"]),
            (1, 0, 1),
        )
        # The cut segment's single-identity change appears nowhere.
        all_changes = [
            change
            for wave in result["waves"]
            for change in wave["changes"]
        ]
        self.assertEqual(len(all_changes), 4)
        self.assertEqual(
            [c["kind"] for c in all_changes],
            ["node_changed", "node_removed",
             "node_added", "node_changed"],
        )

        # Members: x and y each appear on one segment of their wave.
        first_members = {
            m["identity"]: m for m in first["members"]
        }
        self.assertEqual(
            (first_members[x]["segments"], first_members[x]["changed"]),
            (1, 1),
        )
        self.assertEqual(
            (first_members[y]["segments"], first_members[y]["removed"]),
            (1, 1),
        )

        # Totals: z never qualified, so it is absent everywhere.
        self.assertEqual(
            result["totals"],
            {"waves": 2, "identities": 2, "added": 1, "removed": 1,
             "changed": 2},
        )

    def test_min_identities_of_one_admits_every_changed_segment(self):
        x = ("X", "k")
        segments = [
            (self._change("node_changed", x, x, x),),
            (),
            (self._change("node_removed", x, x, None),),
        ]
        result = BranchStore._build_lifetime_churn_waves_result(
            segments, 1, 100
        )
        self.assertEqual(len(result["waves"]), 2)
        self.assertEqual(
            [(w["from"], w["to"]) for w in result["waves"]],
            [(0, 1), (2, 3)],
        )
        self.assertEqual(result["totals"]["identities"], 1)


class LifetimeChurnWavesLimitValidationTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        overrides.setdefault("causes", ("a0",))
        return waves_call(self.store, self.args, **overrides)

    def test_min_identities_must_be_a_non_bool_positive_int(self):
        for bad in (True, False, 1.0, "1", None):
            with self.subTest(min_identities=bad):
                with self.assertRaises(TypeError):
                    self.call(min_identities=bad)
        with self.assertRaises(ValueError):
            self.call(min_identities=0)
        with self.assertRaises(ValueError):
            self.call(min_identities=-4)

    def test_wave_limit_must_be_a_non_bool_positive_int(self):
        for bad in (True, False, 1.0, "1", None):
            with self.subTest(wave_limit=bad):
                with self.assertRaises(TypeError):
                    self.call(wave_limit=bad)
        with self.assertRaises(ValueError):
            self.call(wave_limit=0)
        with self.assertRaises(ValueError):
            self.call(wave_limit=-4)

    def test_new_limits_validated_in_order_after_streak_limit(self):
        # streak_limit fails before min_identities.
        with self.assertRaises(ValueError):
            self.call(streak_limit=0, min_identities=0)
        with self.assertRaises(TypeError):
            self.call(streak_limit="x", min_identities=0)
        # min_identities fails before wave_limit.
        with self.assertRaises(ValueError):
            self.call(min_identities=0, wave_limit=0)
        with self.assertRaises(TypeError):
            self.call(min_identities="x", wave_limit=0)
        # Once the earlier parameters pass, wave_limit's own error fires.
        with self.assertRaises(TypeError):
            self.call(min_identities=1, wave_limit="x")
        # Both still precede every state lookup and the token lookup.
        with self.assertRaises(ValueError):
            self.call(reference="ghost", min_identities=0)
        with self.assertRaises(ValueError):
            self.call(token="whatever", wave_limit=0)

    def test_wave_count_beyond_wave_limit_raises_value_error(self):
        # Two separated one-segment waves trip a limit of one.
        with self.assertRaises(ValueError) as caught:
            self.call(windows=((0,), (1,), (1,), (2,)), wave_limit=1)
        self.assertIn("wave limit", str(caught.exception))
        result = self.call(windows=((0,), (1,), (1,), (2,)), wave_limit=2)
        self.assertEqual(len(result["waves"]), 2)

    def test_churn_and_streak_limits_are_validated_but_not_enforced(self):
        with self.assertRaises(ValueError):
            self.call(churn_limit=0)
        with self.assertRaises(ValueError):
            self.call(streak_limit=0)
        with self.assertRaises(TypeError):
            self.call(churn_limit="x")
        with self.assertRaises(TypeError):
            self.call(streak_limit="x")
        # Limits of one would cap churn/streaks queries; waves ignore them.
        result = self.call(churn_limit=1, streak_limit=1)
        self.assertEqual(len(result["waves"]), 1)
        self.assertEqual(result["totals"]["identities"], 3)

    def test_batch_caps_still_fire_ahead_of_wave_cap(self):
        # Six total changes over five trips the batch cap, not wave cap.
        with self.assertRaises(ValueError) as caught:
            self.call(total_diff_limit=5, wave_limit=1)
        self.assertIn("total diff", str(caught.exception))
        # Empty batches never trip the wave cap.
        self.assertEqual(
            self.call(windows=(), wave_limit=1)["waves"], ()
        )

    def test_shared_validation_errors_match_streaks(self):
        with self.assertRaises(TypeError):
            self.call(windows="x")
        with self.assertRaises(ValueError):
            self.call(windows=((-1,),))
        with self.assertRaises(ValueError):
            self.call(direction="sideways")
        with self.assertRaises(ValueError):
            self.call(window_limit=2, windows=((0,), (1,), (2,)))


class LifetimeChurnWavesStateTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        overrides.setdefault("causes", ("a0",))
        return waves_call(self.store, self.args, **overrides)

    def test_unknown_branch_history_node_and_cause_raise(self):
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

    def test_empty_windows_still_run_every_state_check(self):
        with self.assertRaises(KeyError):
            self.call(windows=(), reference="ghost")
        with self.assertRaises(ValueError):
            self.call(windows=(), limit=1)


class LifetimeChurnWavesIsolationTests(unittest.TestCase):
    def test_result_levels_are_mutually_unshared(self):
        store, args = diamond_store()
        result = waves_call(store, args)
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
        first = waves_call(store, args)
        pristine = copy.deepcopy(first)
        for wave in first["waves"]:
            wave["changes"] += (
                {"from": 4, "to": 5, "kind": "evil",
                 "before": None, "after": None},
            )
            for member in wave["members"]:
                member["segments"] = 99
            for change in wave["changes"]:
                for side in ("before", "after"):
                    evidence = change[side]
                    if evidence is not None:
                        evidence["first_seen"] = 99
                        evidence["intervals"] += ((99, 99),)
        self.assertEqual(waves_call(store, args), pristine)

    def test_success_and_failure_change_nothing(self):
        store, args = diamond_store()
        heads = dict(store._heads)
        appends = copy.deepcopy(store._appends)
        merges = copy.deepcopy(store._merges)
        audit = store.audit_log()

        waves_call(store, args)
        failures = [
            dict(windows="x"),
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
            dict(streak_limit=0),
            dict(min_identities=0),
            dict(min_identities="x"),
            dict(wave_limit=0),
            dict(wave_limit="x"),
            dict(wave_limit=1, windows=((0,), (1,), (1,), (2,))),
            dict(causes=("zzz",)),
            dict(limit=1),
            dict(reference="ghost"),
        ]
        for overrides in failures:
            with self.subTest(overrides=overrides):
                with self.assertRaises(Exception):
                    waves_call(store, args, causes=("a0",), **overrides)

        self.assertEqual(dict(store._heads), heads)
        self.assertEqual(copy.deepcopy(store._appends), appends)
        self.assertEqual(copy.deepcopy(store._merges), merges)
        self.assertEqual(store.audit_log(), audit)

    def test_existing_public_entries_are_unaffected(self):
        store, args = diamond_store()
        waves_call(store, args)
        churn = churn_call(store, args)
        self.assertEqual(churn_call(store, args), churn)


class LifetimeChurnWavesTokenTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, token=None, **overrides):
        return waves_call(self.store, self.args, token=token, **overrides)

    def test_success_consumes_one_read_and_failure_refunds(self):
        token = self.store.create_snapshot(2)
        self.call(token=token)
        with self.assertRaises(ValueError):
            self.call(token=token, wave_limit=1,
                      windows=((0,), (1,), (1,), (2,)))
        with self.assertRaises(KeyError):
            self.call(token=token, causes=("zzz",))
        self.call(token=token)
        with self.assertRaises(RuntimeError):
            self.call(token=token)

    def test_results_come_from_the_frozen_view(self):
        before = copy.deepcopy(self.call())
        token = self.store.create_snapshot(4)
        self.store.append("a", "a2", 7, {"x": -1})
        self.store.create("d", "root")
        self.store.append("d", "d0", 8, {"z": 3})
        self.store.merge("a", "d", "M", 9, {"z": 1})
        self.store.append("main", "r3", 10, {"x": 1})
        self.assertEqual(self.call(token=token), before)


if __name__ == "__main__":
    unittest.main()
