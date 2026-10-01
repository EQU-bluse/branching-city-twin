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
    kwargs["min_identities"] = overrides.pop("min_identities", 2)
    kwargs["wave_limit"] = overrides.pop("wave_limit", 50)
    return kwargs


def waves_call(store, store_args, **overrides):
    return store.lifetime_churn_waves(**waves_kwargs(store_args, **overrides))


def reference_waves(
    evolution: dict[str, object], min_identities: int
) -> dict[str, object]:
    """Independently group a lifetime_evolution result into waves."""
    type_order = {"node": 0, "edge": 1, "gap": 2}
    encounter: dict[tuple, int] = {}
    next_encounter = 0
    segment_keys: list[list[tuple]] = []
    segment_counters: list[dict[tuple, list[int]]] = []
    segment_evidence: list[list[dict[str, object]]] = []
    for segment in evolution["segments"]:
        keys: list[tuple] = []
        counters: dict[tuple, list[int]] = {}
        evidence = []
        seen = set()
        for change in segment["changes"]:
            type_name = change["kind"].split("_", 1)[0]
            key = (type_name, change["identity"])
            if key not in encounter:
                encounter[key] = next_encounter
                next_encounter += 1
            if key not in seen:
                seen.add(key)
                keys.append(key)
                counters[key] = [0, 0, 0]
            if change["kind"].endswith("_added"):
                counters[key][0] += 1
            elif change["kind"].endswith("_removed"):
                counters[key][1] += 1
            else:
                counters[key][2] += 1
            evidence.append(
                {
                    "from": segment["left"],
                    "to": segment["right"],
                    "kind": change["kind"],
                    "before": change["before"],
                    "after": change["after"],
                }
            )
        segment_keys.append(keys)
        segment_counters.append(counters)
        segment_evidence.append(evidence)

    bounds: list[list[int]] = []
    member_accumulators: list[dict[tuple, dict[str, object]]] = []
    active = None
    for position, keys in enumerate(segment_keys):
        if len(keys) >= min_identities:
            if active is None:
                active = {}
                member_accumulators.append(active)
                bounds.append([position, position + 1])
            bounds[-1][1] = position + 1
            for key in keys:
                added, removed, changed = segment_counters[position][key]
                accumulator = active.setdefault(
                    key,
                    {"segments": set(), "added": 0, "removed": 0,
                     "changed": 0},
                )
                accumulator["segments"].add(position)
                accumulator["added"] += added
                accumulator["removed"] += removed
                accumulator["changed"] += changed
        else:
            active = None

    waves = []
    all_identities = set()
    total_added = total_removed = total_changed = 0
    for (from_position, to_position), members in zip(
        bounds, member_accumulators
    ):
        ordered = sorted(
            members.items(),
            key=lambda item: (
                -len(item[1]["segments"]),
                -(
                    item[1]["added"]
                    + item[1]["removed"]
                    + item[1]["changed"]
                ),
                min(item[1]["segments"]),
                type_order[item[0][0]],
                encounter[item[0]],
            ),
        )
        member_records = []
        wave_added = wave_removed = wave_changed = 0
        for (type_name, identity), accumulator in ordered:
            added, removed, changed = (
                accumulator["added"],
                accumulator["removed"],
                accumulator["changed"],
            )
            wave_added += added
            wave_removed += removed
            wave_changed += changed
            member_records.append(
                {
                    "type": type_name,
                    "identity": identity,
                    "segments": len(accumulator["segments"]),
                    "added": added,
                    "removed": removed,
                    "changed": changed,
                }
            )
        changes = []
        for position in range(from_position, to_position):
            changes.extend(segment_evidence[position])
        total_added += wave_added
        total_removed += wave_removed
        total_changed += wave_changed
        all_identities.update(members)
        waves.append(
            {
                "from": from_position,
                "to": to_position,
                "segments": to_position - from_position,
                "identities": len(members),
                "added": wave_added,
                "removed": wave_removed,
                "changed": wave_changed,
                "members": tuple(member_records),
                "changes": tuple(changes),
            }
        )
    waves.sort(key=lambda wave: (wave["from"], wave["to"]))
    return {
        "waves": tuple(waves),
        "totals": {
            "waves": len(waves),
            "identities": len(all_identities),
            "added": total_added,
            "removed": total_removed,
            "changed": total_changed,
        },
    }


class LifetimeChurnWavesSignatureTests(unittest.TestCase):
    def test_public_signature_is_streaks_plus_two_caps(self):
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
        for wave in result["waves"]:
            self.assertIsInstance(wave, dict)
            self.assertEqual(
                list(wave),
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
            self.assertIsInstance(wave["members"], tuple)
            self.assertIsInstance(wave["changes"], tuple)
            for member in wave["members"]:
                self.assertIsInstance(member, dict)
                self.assertEqual(
                    list(member),
                    [
                        "type",
                        "identity",
                        "segments",
                        "added",
                        "removed",
                        "changed",
                    ],
                )
            for change in wave["changes"]:
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
        return self.store.lifetime_evolution(**kwargs)

    def test_matches_independent_grouping_of_evolution(self):
        windows_set = (
            ((0,), (1,), (2,)),
            ((0, 2), (1,), (0, 1, 2)),
            ((2,), (0,)),
            ((1, 2), (0, 1)),
            ((0, 1), (0, 1), (0, 1)),
            ((0, 1, 2),),
            (),
        )
        for windows in windows_set:
            for minimum in (1, 2, 3, 4):
                with self.subTest(windows=windows, min_identities=minimum):
                    evolution = self.evolution(windows=windows)
                    self.assertEqual(
                        self.call(
                            windows=windows, min_identities=minimum
                        ),
                        reference_waves(evolution, minimum),
                    )

    def test_diamond_fixture_one_wave_spanning_both_segments(self):
        result = self.call()
        self.assertEqual(len(result["waves"]), 1)
        wave = result["waves"][0]
        self.assertEqual((wave["from"], wave["to"]), (0, 2))
        self.assertEqual(wave["segments"], 2)
        self.assertEqual(wave["identities"], 3)
        self.assertEqual(
            (wave["added"], wave["removed"], wave["changed"]), (2, 0, 4)
        )
        self.assertEqual(len(wave["changes"]), 6)
        self.assertEqual(
            [(m["type"], m["identity"]) for m in wave["members"]],
            [
                ("node", ("W", "y")),
                ("node", ("a0", "x")),
                ("edge", (("a0", "x"), ("W", "y"))),
            ],
        )
        by_identity = {
            (m["type"], m["identity"]): m for m in wave["members"]
        }
        a0 = by_identity[("node", ("a0", "x"))]
        w = by_identity[("node", ("W", "y"))]
        edge = by_identity[("edge", (("a0", "x"), ("W", "y")))]
        self.assertEqual(
            (a0["segments"], a0["added"], a0["removed"], a0["changed"]),
            (2, 0, 0, 2),
        )
        self.assertEqual(
            (w["segments"], w["added"], w["removed"], w["changed"]),
            (2, 1, 0, 1),
        )
        self.assertEqual(
            (edge["segments"], edge["added"], edge["removed"],
             edge["changed"]),
            (2, 1, 0, 1),
        )
        self.assertEqual(
            result["totals"],
            {"waves": 1, "identities": 3, "added": 2, "removed": 0,
             "changed": 4},
        )

    def test_changes_keep_segment_position_and_original_order(self):
        result = self.call()
        evolution = self.evolution()
        segment_changes = {
            (segment["left"], segment["right"]): segment["changes"]
            for segment in evolution["segments"]
        }
        for wave in result["waves"]:
            expected = []
            for position in range(wave["from"], wave["to"]):
                for raw in segment_changes[(position, position + 1)]:
                    expected.append((position, position + 1, raw))
            self.assertEqual(
                [(c["from"], c["to"], c["kind"]) for c in wave["changes"]],
                [
                    (left, right, raw["kind"])
                    for left, right, raw in expected
                ],
            )
            for change, (_, _, match) in zip(wave["changes"], expected):
                self.assertEqual(change["before"], match["before"])
                self.assertEqual(change["after"], match["after"])
                if change["kind"].endswith("_added"):
                    self.assertIsNone(change["before"])
                elif change["kind"].endswith("_removed"):
                    self.assertIsNone(change["after"])

    def test_high_threshold_cuts_every_segment(self):
        result = self.call(min_identities=4)
        self.assertEqual(result["waves"], ())
        self.assertEqual(
            result["totals"],
            {"waves": 0, "identities": 0, "added": 0, "removed": 0,
             "changed": 0},
        )

    def test_empty_windows_causes_and_single_window_return_zeros(self):
        for overrides in (
            {"windows": ()},
            {"causes": ()},
            {"windows": ((0, 1, 2),)},
            {"windows": ((0, 1), (0, 1), (0, 1))},
        ):
            with self.subTest(overrides=overrides):
                result = self.call(**overrides)
                self.assertEqual(result["waves"], ())
                self.assertEqual(
                    result["totals"],
                    {"waves": 0, "identities": 0, "added": 0,
                     "removed": 0, "changed": 0},
                )

    def test_threshold_one_qualifies_every_nonempty_segment(self):
        result = self.call(windows=((2,), (0,)), min_identities=1)
        self.assertEqual(len(result["waves"]), 1)
        wave = result["waves"][0]
        self.assertEqual((wave["from"], wave["to"]), (0, 1))
        self.assertEqual(wave["segments"], 1)

    def test_waves_sort_by_from_to_and_totals_sum_records(self):
        result = self.call()
        bounds = [(w["from"], w["to"]) for w in result["waves"]]
        self.assertEqual(bounds, sorted(bounds))
        self.assertEqual(
            result["totals"],
            {
                "waves": len(result["waves"]),
                "identities": len(
                    {
                        (m["type"], m["identity"])
                        for wave in result["waves"]
                        for m in wave["members"]
                    }
                ),
                "added": sum(w["added"] for w in result["waves"]),
                "removed": sum(w["removed"] for w in result["waves"]),
                "changed": sum(w["changed"] for w in result["waves"]),
            },
        )


class LifetimeChurnWaveGroupingTests(unittest.TestCase):
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

    def _change(self, kind, identity):
        type_name = kind.split("_", 1)[0]
        before = None if kind.endswith("_added") else self._lifetime(identity)
        after = None if kind.endswith("_removed") else self._lifetime(identity)
        return {
            "kind": kind,
            "identity": identity,
            "before": before,
            "after": after,
        }

    def test_subthreshold_segment_cuts_and_later_qualifying_opens_new_wave(
        self,
    ):
        x = ("X", "k")
        y = ("Y", "k")
        edge = ("e",)
        # seg0: three identities; seg1: two; seg2: one (cuts); seg3: two.
        segments = [
            (
                self._change("node_changed", x),
                self._change("node_removed", y),
                self._change("edge_added", edge),
            ),
            (
                self._change("node_added", y),
                self._change("edge_changed", edge),
            ),
            (self._change("node_added", x),),
            (
                self._change("node_changed", x),
                self._change("node_changed", y),
            ),
        ]
        result = BranchStore._build_lifetime_churn_waves_result(segments, 2, 50)
        self.assertEqual(len(result["waves"]), 2)
        first, second = result["waves"]

        self.assertEqual((first["from"], first["to"]), (0, 2))
        self.assertEqual(first["segments"], 2)
        self.assertEqual(first["identities"], 3)
        self.assertEqual(
            (first["added"], first["removed"], first["changed"]),
            (2, 1, 2),
        )
        # Members sort by participated segments then total changes; the
        # two two-segment members tie on both and fall back to type
        # (node y before edge) and encounter order.
        self.assertEqual(
            [(m["type"], m["identity"], m["segments"],
              m["added"] + m["removed"] + m["changed"])
             for m in first["members"]],
            [
                ("node", y, 2, 2),
                ("edge", edge, 2, 2),
                ("node", x, 1, 1),
            ],
        )
        kinds = [(c["from"], c["kind"]) for c in first["changes"]]
        self.assertEqual(
            kinds,
            [
                (0, "node_changed"),
                (0, "node_removed"),
                (0, "edge_added"),
                (1, "node_added"),
                (1, "edge_changed"),
            ],
        )

        self.assertEqual((second["from"], second["to"]), (3, 4))
        self.assertEqual(second["segments"], 1)
        self.assertEqual(second["identities"], 2)
        self.assertEqual(
            (second["added"], second["removed"], second["changed"]),
            (0, 0, 2),
        )
        self.assertEqual(
            [(m["type"], m["identity"]) for m in second["members"]],
            [("node", x), ("node", y)],
        )

        # x and y appear in both waves but count once in totals.
        self.assertEqual(
            result["totals"],
            {"waves": 2, "identities": 3, "added": 2, "removed": 1,
             "changed": 4},
        )

    def test_identity_with_two_records_counts_once_for_threshold(self):
        x = ("X", "k")
        y = ("Y", "k")
        segments = [
            (
                self._change("node_added", x),
                self._change("node_changed", x),
                self._change("node_changed", y),
            ),
        ]
        result = BranchStore._build_lifetime_churn_waves_result(segments, 2, 50)
        self.assertEqual(len(result["waves"]), 1)
        wave = result["waves"][0]
        self.assertEqual(wave["identities"], 2)
        # All three records count toward the counters and evidence.
        self.assertEqual(
            (wave["added"], wave["removed"], wave["changed"]), (1, 0, 2)
        )
        self.assertEqual(len(wave["changes"]), 3)
        member_x = next(
            m for m in wave["members"] if m["identity"] == x
        )
        self.assertEqual(
            (member_x["segments"], member_x["added"], member_x["changed"]),
            (1, 1, 1),
        )

    def test_subthreshold_threshold_sees_no_waves(self):
        x = ("X", "k")
        segments = [(self._change("node_changed", x),)]
        result = BranchStore._build_lifetime_churn_waves_result(segments, 2, 50)
        self.assertEqual(
            result,
            {
                "waves": (),
                "totals": {"waves": 0, "identities": 0, "added": 0,
                           "removed": 0, "changed": 0},
            },
        )


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

    def test_new_caps_validated_in_order_after_streak_limit(self):
        # streak_limit fails before min_identities, which fails before
        # wave_limit.
        with self.assertRaises(ValueError):
            self.call(streak_limit=0, min_identities=0, wave_limit=0)
        with self.assertRaises(TypeError):
            self.call(streak_limit="x", min_identities=0)
        with self.assertRaises(ValueError):
            self.call(streak_limit=1, min_identities=0, wave_limit="x")
        with self.assertRaises(TypeError):
            self.call(streak_limit=1, min_identities=1, wave_limit="x")
        # churn_limit still fails first.
        with self.assertRaises(ValueError):
            self.call(
                churn_limit=0, streak_limit=0, min_identities=0,
                wave_limit=0,
            )
        # Earlier batch caps precede all of them.
        with self.assertRaises(ValueError):
            self.call(total_diff_limit=0, min_identities="x")
        with self.assertRaises(TypeError):
            self.call(total_diff_limit=1, min_identities="x")
        # Both still precede every state lookup and the token lookup.
        with self.assertRaises(ValueError):
            self.call(reference="ghost", min_identities=0)
        with self.assertRaises(ValueError):
            self.call(token="whatever", wave_limit=0)

    def test_wave_count_beyond_wave_limit_raises_value_error(self):
        x = ("X", "k")
        y = ("Y", "k")
        segments = [
            (
                {"kind": "node_changed", "identity": x,
                 "before": None, "after": None},
                {"kind": "node_changed", "identity": y,
                 "before": None, "after": None},
            ),
            (
                {"kind": "node_added", "identity": x,
                 "before": None, "after": None},
            ),
            (
                {"kind": "node_changed", "identity": x,
                 "before": None, "after": None},
                {"kind": "node_changed", "identity": y,
                 "before": None, "after": None},
            ),
        ]
        with self.assertRaises(ValueError) as caught:
            BranchStore._build_lifetime_churn_waves_result(segments, 2, 1)
        self.assertIn("wave limit", str(caught.exception))
        result = BranchStore._build_lifetime_churn_waves_result(
            segments, 2, 2
        )
        self.assertEqual(len(result["waves"]), 2)

    def test_churn_and_streak_limits_are_validated_but_do_not_bound(self):
        # The diamond batch has three identities and three streaks, so
        # caps of one would fail churn and streaks queries; waves still
        # returns every wave.
        result = self.call(
            churn_limit=1, streak_limit=1, min_identities=2
        )
        self.assertEqual(len(result["waves"]), 1)
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
            wave["members"] += (
                {"type": "node", "identity": ("evil", "z"), "segments": 9,
                 "added": 9, "removed": 9, "changed": 9},
            )
            wave["changes"] += (
                {"from": 4, "to": 5, "kind": "evil",
                 "before": None, "after": None},
            )
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
        streaks = store.lifetime_churn_streaks(
            **{**churn_kwargs(args), "streak_limit": 50}
        )
        self.assertEqual(
            store.lifetime_churn_streaks(
                **{**churn_kwargs(args), "streak_limit": 50}
            ),
            streaks,
        )


class LifetimeChurnWavesTokenTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, token=None, **overrides):
        return waves_call(self.store, self.args, token=token, **overrides)

    def test_success_consumes_one_read_and_failure_refunds(self):
        token = self.store.create_snapshot(2)
        self.call(token=token)
        with self.assertRaises(ValueError):
            self.call(token=token, wave_limit=0)
        with self.assertRaises(ValueError):
            self.call(token=token, min_identities=0)
        with self.assertRaises(KeyError):
            self.call(token=token, causes=("zzz",))
        self.call(token=token)
        with self.assertRaises(RuntimeError):
            self.call(token=token)

    def test_cap_validation_failure_precedes_token_lookup(self):
        with self.assertRaises(ValueError):
            self.call(token="whatever", min_identities=0)
        with self.assertRaises(TypeError):
            self.call(token="whatever", wave_limit="x")

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
