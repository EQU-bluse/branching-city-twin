import copy
import inspect
import unittest

from city_twin.branches import BranchStore

from tests.test_lifetime_churn import churn_kwargs, diamond_store
from tests.test_lifetime_churn_recurrences import recurrences_kwargs
from tests.test_lifetime_churn_waves import waves_call


def periodicities_kwargs(store_args, **overrides) -> dict:
    max_jitter = overrides.pop("max_jitter", 0)
    periodicity_limit = overrides.pop("periodicity_limit", 50)
    kwargs = recurrences_kwargs(store_args, **overrides)
    kwargs["max_jitter"] = max_jitter
    kwargs["periodicity_limit"] = periodicity_limit
    return kwargs


def periodicities_call(store, store_args, **overrides):
    return store.lifetime_churn_periodicities(
        **periodicities_kwargs(store_args, **overrides)
    )


SIX_WINDOWS = ((0,), (1,), (1,), (2,), (2,), (0, 1, 2))


class LifetimeChurnPeriodicitiesSignatureTests(unittest.TestCase):
    def test_public_signature_appends_max_jitter_and_periodicity_limit(self):
        parameters = list(
            inspect.signature(
                BranchStore.lifetime_churn_periodicities
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
                "min_waves",
                "recurrence_limit",
                "max_jitter",
                "periodicity_limit",
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
        result = periodicities_call(
            store, args, windows=SIX_WINDOWS, max_jitter=0
        )
        self.assertIsInstance(result, dict)
        self.assertEqual(list(result), ["periodicities", "totals"])
        self.assertIsInstance(result["periodicities"], tuple)
        self.assertEqual(
            list(result["totals"]),
            ["periodicities", "waves", "segments",
             "added", "removed", "changed"],
        )
        for record in result["periodicities"]:
            self.assertIsInstance(record, dict)
            self.assertEqual(
                list(record),
                [
                    "type",
                    "identity",
                    "waves",
                    "first",
                    "last",
                    "span",
                    "segments",
                    "added",
                    "removed",
                    "changed",
                    "appearances",
                    "intervals",
                    "period",
                    "jitter",
                ],
            )
            self.assertIsInstance(record["appearances"], tuple)
            self.assertIsInstance(record["intervals"], tuple)
            self.assertEqual(len(record["intervals"]), record["waves"] - 1)
            for appearance in record["appearances"]:
                self.assertEqual(
                    list(appearance),
                    ["wave", "from", "to", "segments",
                     "added", "removed", "changed"],
                )


class LifetimeChurnPeriodicitiesResultTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        return periodicities_call(self.store, self.args, **overrides)

    def evolution(self, **overrides):
        kwargs = churn_kwargs(self.args, **overrides)
        kwargs.pop("churn_limit")
        kwargs.pop("streak_limit", None)
        kwargs.pop("min_identities", None)
        kwargs.pop("wave_limit", None)
        kwargs.pop("min_waves", None)
        kwargs.pop("recurrence_limit", None)
        kwargs.pop("max_jitter", None)
        kwargs.pop("periodicity_limit", None)
        return self.store.lifetime_evolution(**kwargs)

    def test_three_evenly_spaced_waves_on_the_diamond_fixture(self):
        result = self.call(windows=SIX_WINDOWS, max_jitter=0)
        self.assertEqual(len(result["periodicities"]), 3)
        by_identity = {
            (r["type"], r["identity"]): r
            for r in result["periodicities"]
        }
        node_w = by_identity[("node", ("W", "y"))]
        self.assertEqual(
            (node_w["waves"], node_w["first"], node_w["last"],
             node_w["span"], node_w["segments"]),
            (3, 0, 2, 3, 3),
        )
        self.assertEqual(node_w["intervals"], (1, 1))
        self.assertEqual(node_w["period"], 1)
        self.assertEqual(node_w["jitter"], 0)
        self.assertEqual(
            [a["wave"] for a in node_w["appearances"]], [0, 1, 2]
        )
        self.assertEqual(
            (node_w["added"], node_w["removed"], node_w["changed"]),
            (1, 0, 2),
        )
        for record in result["periodicities"]:
            waves = [a["wave"] for a in record["appearances"]]
            self.assertEqual(waves, sorted(waves))
            intervals = [b - a for a, b in zip(waves, waves[1:])]
            self.assertEqual(list(record["intervals"]), intervals)
            self.assertEqual(
                record["jitter"], max(intervals) - min(intervals)
            )
        self.assertEqual(
            result["totals"],
            {"periodicities": 3, "waves": 9, "segments": 9,
             "added": 2, "removed": 0, "changed": 7},
        )

    def test_two_waves_never_suffice_even_with_a_generous_jitter(self):
        result = self.call(
            windows=((0,), (1,), (1,), (2,)), max_jitter=100
        )
        self.assertEqual(result["periodicities"], ())
        self.assertEqual(
            result["totals"],
            {"periodicities": 0, "waves": 0, "segments": 0,
             "added": 0, "removed": 0, "changed": 0},
        )

    def test_empty_windows_causes_and_single_window_return_zeros(self):
        for overrides in (
            {"windows": ()},
            {"causes": ()},
            {"windows": ((0, 1, 2),)},
        ):
            with self.subTest(overrides=overrides):
                result = self.call(**overrides)
                self.assertEqual(result["periodicities"], ())
                self.assertEqual(
                    result["totals"],
                    {"periodicities": 0, "waves": 0, "segments": 0,
                     "added": 0, "removed": 0, "changed": 0},
                )

    def test_jitter_band_filters_end_to_end_on_even_waves(self):
        # Even spacing means only max_jitter zero keeps the records;
        # the band is inclusive of the boundary one interval width up.
        tight = self.call(windows=SIX_WINDOWS, max_jitter=0)
        self.assertEqual(len(tight["periodicities"]), 3)

    def test_appearances_match_the_wave_members_and_boundaries(self):
        waves = waves_call(
            self.store, self.args, windows=SIX_WINDOWS
        )["waves"]
        result = self.call(windows=SIX_WINDOWS, max_jitter=0)
        for record in result["periodicities"]:
            for appearance in record["appearances"]:
                wave = waves[appearance["wave"]]
                self.assertEqual(
                    (appearance["from"], appearance["to"]),
                    (wave["from"], wave["to"]),
                )
                member = next(
                    member
                    for member in wave["members"]
                    if member["type"] == record["type"]
                    and member["identity"] == record["identity"]
                )
                self.assertEqual(
                    (appearance["segments"], appearance["added"],
                     appearance["removed"], appearance["changed"]),
                    (member["segments"], member["added"],
                     member["removed"], member["changed"]),
                )

    def test_records_sort_jitter_waves_segments_changes_first_type(self):
        result = self.call(windows=SIX_WINDOWS, max_jitter=0)
        keys = [
            (
                r["jitter"],
                -r["waves"],
                -r["segments"],
                -(r["added"] + r["removed"] + r["changed"]),
                r["first"],
                {"node": 0, "edge": 1, "gap": 2}[r["type"]],
            )
            for r in result["periodicities"]
        ]
        self.assertEqual(keys, sorted(keys))


class LifetimeChurnPeriodicitiesGroupingTests(unittest.TestCase):
    @staticmethod
    def _member(kind, identity, segments, added, removed, changed):
        return {
            "type": kind,
            "identity": identity,
            "segments": segments,
            "added": added,
            "removed": removed,
            "changed": changed,
        }

    def _wave(self, serial, members):
        return {
            "from": serial * 2,
            "to": serial * 2 + 1,
            "segments": 1,
            "identities": len(members),
            "added": sum(m["added"] for m in members),
            "removed": sum(m["removed"] for m in members),
            "changed": sum(m["changed"] for m in members),
            "members": tuple(members),
            "changes": (),
        }

    def _fixture_waves(self):
        # A node: waves 0,1,2   intervals (1,1) jitter 0, segments 3
        # B node: waves 0,2,4   intervals (2,2) jitter 0, segments 6
        # C node: waves 1,2,4   intervals (1,2) jitter 1, first 1
        # D edge: waves 0,2,5   intervals (2,3) jitter 1, first 0
        # E node: waves 0,1     only two appearances, always dropped
        # F node: waves 0,1,2,3 intervals (1,1,1) jitter 0, waves 4
        a0 = self._member("node", "A", 1, 1, 0, 0)
        a1 = self._member("node", "A", 1, 0, 0, 1)
        a2 = self._member("node", "A", 1, 0, 0, 1)
        b0 = self._member("node", "B", 2, 1, 0, 0)
        b2 = self._member("node", "B", 2, 0, 0, 1)
        b4 = self._member("node", "B", 2, 0, 0, 1)
        c1 = self._member("node", "C", 1, 1, 0, 0)
        c2 = self._member("node", "C", 1, 0, 0, 0)
        c4 = self._member("node", "C", 1, 0, 0, 0)
        d0 = self._member("edge", "D", 1, 0, 1, 0)
        d2 = self._member("edge", "D", 1, 0, 0, 0)
        d5 = self._member("edge", "D", 1, 0, 0, 0)
        e0 = self._member("node", "E", 1, 1, 0, 0)
        e1 = self._member("node", "E", 1, 0, 0, 0)
        f0 = self._member("node", "F", 1, 0, 0, 1)
        f1 = self._member("node", "F", 1, 0, 0, 1)
        f2 = self._member("node", "F", 1, 0, 0, 1)
        f3 = self._member("node", "F", 1, 0, 0, 1)
        return (
            self._wave(0, (a0, b0, d0, e0, f0)),
            self._wave(1, (a1, c1, e1, f1)),
            self._wave(2, (a2, b2, c2, d2, f2)),
            self._wave(3, (f3,)),
            self._wave(4, (b4, c4)),
            self._wave(5, (d5,)),
        )

    def test_jitter_filter_intervals_period_and_sort_order(self):
        waves = self._fixture_waves()
        result = BranchStore._build_lifetime_churn_periodicities_result(
            waves, 1, 100
        )
        by_identity = {
            (r["type"], r["identity"]): r
            for r in result["periodicities"]
        }
        # F (four waves) leads the jitter-zero band; B's six segments
        # outrank A's three; D's earlier first wave outranks C.
        self.assertEqual(
            [(r["type"], r["identity"]) for r in result["periodicities"]],
            [("node", "F"), ("node", "B"), ("node", "A"),
             ("edge", "D"), ("node", "C")],
        )
        self.assertNotIn(("node", "E"), by_identity)

        self.assertEqual(by_identity[("node", "A")]["intervals"], (1, 1))
        self.assertEqual(by_identity[("node", "A")]["period"], 1)
        self.assertEqual(by_identity[("node", "A")]["jitter"], 0)

        self.assertEqual(by_identity[("node", "B")]["intervals"], (2, 2))
        self.assertEqual(by_identity[("node", "B")]["period"], 2)
        self.assertEqual(by_identity[("node", "B")]["jitter"], 0)
        self.assertEqual(by_identity[("node", "B")]["segments"], 6)

        self.assertEqual(by_identity[("node", "F")]["waves"], 4)
        self.assertEqual(by_identity[("node", "F")]["intervals"], (1, 1, 1))
        self.assertEqual(by_identity[("node", "F")]["period"], 1)

        self.assertEqual(by_identity[("edge", "D")]["intervals"], (2, 3))
        self.assertEqual(by_identity[("edge", "D")]["period"], 2)
        self.assertEqual(by_identity[("edge", "D")]["jitter"], 1)
        self.assertEqual(by_identity[("edge", "D")]["first"], 0)

        self.assertEqual(by_identity[("node", "C")]["intervals"], (1, 2))
        self.assertEqual(by_identity[("node", "C")]["period"], 1)
        self.assertEqual(by_identity[("node", "C")]["jitter"], 1)
        self.assertEqual(by_identity[("node", "C")]["first"], 1)

        self.assertEqual(
            result["totals"],
            {"periodicities": 5, "waves": 16, "segments": 19,
             "added": 3, "removed": 1, "changed": 8},
        )

    def test_zero_band_keeps_only_perfectly_even_spacing(self):
        waves = self._fixture_waves()
        result = BranchStore._build_lifetime_churn_periodicities_result(
            waves, 0, 100
        )
        self.assertEqual(
            [(r["type"], r["identity"]) for r in result["periodicities"]],
            [("node", "F"), ("node", "B"), ("node", "A")],
        )
        self.assertEqual(
            result["totals"],
            {"periodicities": 3, "waves": 10, "segments": 13,
             "added": 2, "removed": 0, "changed": 8},
        )

    def test_median_takes_the_smaller_middle_for_even_interval_counts(self):
        # X spans waves 0,2,5,9,14 -> intervals (2,3,4,5), even count,
        # median is the smaller middle value 3 and jitter is 3.
        # Y spans waves 0,2,5,9 -> intervals (2,3,4), odd count,
        # median is 3 and jitter is 2.
        filler = self._member("node", "Q", 1, 0, 0, 0)
        x = self._member("node", "X", 1, 0, 0, 0)
        y = self._member("gap", "Y", 1, 0, 0, 0)
        waves = []
        for serial in range(15):
            members = []
            if serial in (0, 2, 5, 9, 14):
                members.append(x)
            if serial in (0, 2, 5, 9):
                members.append(y)
            if not members:
                members.append(filler)
            waves.append(self._wave(serial, tuple(members)))
        result = BranchStore._build_lifetime_churn_periodicities_result(
            tuple(waves), 10, 100
        )
        by_identity = {
            (r["type"], r["identity"]): r
            for r in result["periodicities"]
        }
        record_x = by_identity[("node", "X")]
        self.assertEqual(record_x["intervals"], (2, 3, 4, 5))
        self.assertEqual(record_x["period"], 3)
        self.assertEqual(record_x["jitter"], 3)
        record_y = by_identity[("gap", "Y")]
        self.assertEqual(record_y["intervals"], (2, 3, 4))
        self.assertEqual(record_y["period"], 3)
        self.assertEqual(record_y["jitter"], 2)

    def test_repeated_segments_within_one_wave_count_once(self):
        x0 = self._member("node", "X", 3, 1, 0, 2)
        x1 = self._member("node", "X", 2, 0, 0, 2)
        x2 = self._member("node", "X", 4, 0, 1, 3)
        waves = (
            self._wave(0, (x0,)),
            self._wave(1, (x1,)),
            self._wave(2, (x2,)),
        )
        result = BranchStore._build_lifetime_churn_periodicities_result(
            waves, 0, 100
        )
        record = result["periodicities"][0]
        self.assertEqual(record["waves"], 3)
        self.assertEqual(len(record["appearances"]), 3)
        self.assertEqual(record["intervals"], (1, 1))
        self.assertEqual(record["period"], 1)
        self.assertEqual(record["jitter"], 0)
        self.assertEqual(record["segments"], 9)
        self.assertEqual(
            (record["added"], record["removed"], record["changed"]),
            (1, 1, 7),
        )

    def test_empty_wave_set_yields_zeros(self):
        result = BranchStore._build_lifetime_churn_periodicities_result(
            (), 0, 100
        )
        self.assertEqual(result["periodicities"], ())
        self.assertEqual(
            result["totals"],
            {"periodicities": 0, "waves": 0, "segments": 0,
             "added": 0, "removed": 0, "changed": 0},
        )


class LifetimeChurnPeriodicitiesLimitValidationTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        overrides.setdefault("causes", ("a0",))
        return periodicities_call(self.store, self.args, **overrides)

    def test_max_jitter_must_be_a_non_bool_non_negative_int(self):
        for bad in (True, False, 1.0, "0", None):
            with self.subTest(max_jitter=bad):
                with self.assertRaises(TypeError):
                    self.call(max_jitter=bad)
        with self.assertRaises(ValueError):
            self.call(max_jitter=-1)
        # Zero is a valid perfectly-even band.
        self.call(max_jitter=0)

    def test_periodicity_limit_must_be_a_non_bool_positive_int(self):
        for bad in (True, False, 1.0, "1", None):
            with self.subTest(periodicity_limit=bad):
                with self.assertRaises(TypeError):
                    self.call(periodicity_limit=bad)
        with self.assertRaises(ValueError):
            self.call(periodicity_limit=0)
        with self.assertRaises(ValueError):
            self.call(periodicity_limit=-4)

    def test_new_limits_validated_in_order_after_recurrence_limit(self):
        # recurrence_limit fails before max_jitter.
        with self.assertRaises(ValueError):
            self.call(recurrence_limit=0, max_jitter=-1)
        with self.assertRaises(TypeError):
            self.call(recurrence_limit="x", max_jitter=-1)
        # max_jitter fails before periodicity_limit.
        with self.assertRaises(ValueError):
            self.call(max_jitter=-1, periodicity_limit=0)
        with self.assertRaises(TypeError):
            self.call(max_jitter="x", periodicity_limit=0)
        # Once max_jitter passes, periodicity_limit's error fires.
        with self.assertRaises(TypeError):
            self.call(max_jitter=0, periodicity_limit="x")
        with self.assertRaises(ValueError):
            self.call(max_jitter=0, periodicity_limit=0)
        # Both still precede every state lookup and the token lookup.
        with self.assertRaises(ValueError):
            self.call(reference="ghost", max_jitter=-1)
        with self.assertRaises(ValueError):
            self.call(token="whatever", periodicity_limit=0)

    def test_periodicity_count_beyond_limit_raises_value_error(self):
        with self.assertRaises(ValueError) as caught:
            self.call(
                windows=SIX_WINDOWS, max_jitter=0, periodicity_limit=2
            )
        self.assertIn("periodicity limit", str(caught.exception))
        result = self.call(
            windows=SIX_WINDOWS, max_jitter=0, periodicity_limit=3
        )
        self.assertEqual(len(result["periodicities"]), 3)
        # An empty result never trips the cap.
        result = self.call(max_jitter=0, periodicity_limit=1)
        self.assertEqual(result["periodicities"], ())

    def test_wave_limit_is_still_enforced_ahead_of_periodicities(self):
        with self.assertRaises(ValueError) as caught:
            self.call(
                windows=SIX_WINDOWS,
                wave_limit=1,
                periodicity_limit=10,
            )
        self.assertIn("wave limit", str(caught.exception))

    def test_earlier_limits_are_validated_but_not_enforced(self):
        with self.assertRaises(ValueError):
            self.call(churn_limit=0)
        with self.assertRaises(ValueError):
            self.call(streak_limit=0)
        with self.assertRaises(ValueError):
            self.call(min_waves=0)
        with self.assertRaises(ValueError):
            self.call(recurrence_limit=0)
        # Bounds of one (or a demanding min_waves) would cap the earlier
        # queries; periodicities ignores them once validation passes.
        result = self.call(
            churn_limit=1, streak_limit=1, min_waves=10,
            recurrence_limit=1, windows=SIX_WINDOWS, max_jitter=0,
        )
        self.assertEqual(len(result["periodicities"]), 3)

    def test_batch_caps_still_fire_ahead_of_periodicity_cap(self):
        with self.assertRaises(ValueError) as caught:
            self.call(total_diff_limit=5, periodicity_limit=1)
        self.assertIn("total diff", str(caught.exception))
        # Empty batches never trip the periodicity cap.
        self.assertEqual(
            self.call(windows=(), periodicity_limit=1)["periodicities"],
            (),
        )

    def test_shared_validation_errors_match_recurrences(self):
        with self.assertRaises(TypeError):
            self.call(windows="x")
        with self.assertRaises(ValueError):
            self.call(windows=((-1,),))
        with self.assertRaises(ValueError):
            self.call(direction="sideways")
        with self.assertRaises(ValueError):
            self.call(window_limit=2, windows=((0,), (1,), (2,)))


class LifetimeChurnPeriodicitiesStateTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        overrides.setdefault("causes", ("a0",))
        return periodicities_call(self.store, self.args, **overrides)

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


class LifetimeChurnPeriodicitiesIsolationTests(unittest.TestCase):
    def test_result_levels_are_mutually_unshared(self):
        store, args = diamond_store()
        result = periodicities_call(
            store, args, windows=SIX_WINDOWS, max_jitter=0
        )
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
        first = periodicities_call(
            store, args, windows=SIX_WINDOWS, max_jitter=0
        )
        pristine = copy.deepcopy(first)
        for record in first["periodicities"]:
            record["intervals"] += (99,)
            record["period"] = 99
            record["jitter"] = 99
            record["waves"] = 99
            for appearance in record["appearances"]:
                appearance["segments"] = 99
        self.assertEqual(
            periodicities_call(
                store, args, windows=SIX_WINDOWS, max_jitter=0
            ),
            pristine,
        )

    def test_success_and_failure_change_nothing(self):
        store, args = diamond_store()
        heads = dict(store._heads)
        appends = copy.deepcopy(store._appends)
        merges = copy.deepcopy(store._merges)
        audit = store.audit_log()

        periodicities_call(store, args, windows=SIX_WINDOWS, max_jitter=0)
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
            dict(min_waves=0),
            dict(min_waves="x"),
            dict(recurrence_limit=0),
            dict(recurrence_limit="x"),
            dict(max_jitter=-1),
            dict(max_jitter="x"),
            dict(periodicity_limit=0),
            dict(periodicity_limit="x"),
            dict(wave_limit=1, windows=SIX_WINDOWS),
            dict(periodicity_limit=1, windows=SIX_WINDOWS, max_jitter=0),
            dict(causes=("zzz",)),
            dict(limit=1),
            dict(reference="ghost"),
        ]
        for overrides in failures:
            with self.subTest(overrides=overrides):
                with self.assertRaises(Exception):
                    periodicities_call(
                        store, args, causes=("a0",), **overrides
                    )

        self.assertEqual(dict(store._heads), heads)
        self.assertEqual(copy.deepcopy(store._appends), appends)
        self.assertEqual(copy.deepcopy(store._merges), merges)
        self.assertEqual(store.audit_log(), audit)

    def test_existing_public_entries_are_unaffected(self):
        store, args = diamond_store()
        periodicities_call(store, args, windows=SIX_WINDOWS, max_jitter=0)
        waves = waves_call(store, args)
        self.assertEqual(waves_call(store, args), waves)


class LifetimeChurnPeriodicitiesTokenTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, token=None, **overrides):
        return periodicities_call(
            self.store, self.args, token=token, **overrides
        )

    def test_success_consumes_one_read_and_failure_refunds(self):
        token = self.store.create_snapshot(2)
        self.call(token=token, windows=SIX_WINDOWS, max_jitter=0)
        # Over the periodicity cap: failure refunds.
        with self.assertRaises(ValueError):
            self.call(
                token=token,
                windows=SIX_WINDOWS,
                max_jitter=0,
                periodicity_limit=1,
            )
        # Over the wave cap inside the shared batch: failure refunds.
        with self.assertRaises(ValueError):
            self.call(
                token=token,
                windows=SIX_WINDOWS,
                wave_limit=1,
            )
        with self.assertRaises(KeyError):
            self.call(token=token, causes=("zzz",))
        self.call(token=token)
        with self.assertRaises(RuntimeError):
            self.call(token=token)

    def test_results_come_from_the_frozen_view(self):
        before = copy.deepcopy(
            self.call(windows=SIX_WINDOWS, max_jitter=0)
        )
        token = self.store.create_snapshot(4)
        self.store.append("a", "a2", 7, {"x": -1})
        self.store.create("d", "root")
        self.store.append("d", "d0", 8, {"z": 3})
        self.store.merge("a", "d", "M", 9, {"z": 1})
        self.store.append("main", "r3", 10, {"x": 1})
        self.assertEqual(
            self.call(token=token, windows=SIX_WINDOWS, max_jitter=0),
            before,
        )


if __name__ == "__main__":
    unittest.main()
