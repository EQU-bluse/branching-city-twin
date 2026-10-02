import copy
import inspect
import unittest
from fractions import Fraction

from city_twin.branches import BranchStore

from tests.test_lifetime_churn import churn_kwargs, diamond_store
from tests.test_lifetime_churn_waves import reference_waves
from tests.test_lifetime_churn_forecast_backtests import (
    backtests_kwargs,
    reference_backtests,
)


def scorecards_kwargs(store_args, **overrides) -> dict:
    min_resolved = overrides.pop("min_resolved", 1)
    scorecard_limit = overrides.pop("scorecard_limit", 50)
    kwargs = backtests_kwargs(store_args, **overrides)
    kwargs["min_resolved"] = min_resolved
    kwargs["scorecard_limit"] = scorecard_limit
    return kwargs


def scorecards_call(store, store_args, **overrides):
    return store.lifetime_churn_forecast_scorecards(
        **scorecards_kwargs(store_args, **overrides)
    )


def reference_scorecards(
    evolution: dict[str, object],
    min_identities: int,
    max_jitter: int,
    horizon: int,
    cutoffs: tuple[int, ...],
    min_resolved: int,
) -> dict[str, object]:
    """Independently aggregate reference backtest outcomes per object."""
    backtests = reference_backtests(
        evolution, min_identities, max_jitter, horizon, cutoffs
    )
    waves = reference_waves(evolution, min_identities)["waves"]
    encounter: dict[tuple[str, object], int] = {}
    for wave in waves:
        for member in wave["members"]:
            key = (member["type"], member["identity"])
            if key not in encounter:
                encounter[key] = len(encounter)

    grouped: dict[tuple[str, object], dict[str, object]] = {}
    for backtest in backtests["backtests"]:
        for outcome in backtest["outcomes"]:
            key = (outcome["type"], outcome["identity"])
            card = grouped.setdefault(
                key,
                {
                    "type": outcome["type"],
                    "identity": outcome["identity"],
                    "predictions": 0,
                    "hit": 0,
                    "early": 0,
                    "late": 0,
                    "missed": 0,
                    "unresolved": 0,
                    "offset_sum": 0,
                    "offset_count": 0,
                    "max_abs_offset": None,
                },
            )
            card["predictions"] += 1
            card[outcome["status"]] += 1
            if outcome["actual"] is not None:
                offset = outcome["actual"] - outcome["predicted"]
                card["offset_sum"] += offset
                card["offset_count"] += 1
                magnitude = abs(offset)
                if (
                    card["max_abs_offset"] is None
                    or magnitude > card["max_abs_offset"]
                ):
                    card["max_abs_offset"] = magnitude

    def reduced(numerator: int, denominator: int) -> tuple[int, int]:
        fraction = Fraction(numerator, denominator)
        return (fraction.numerator, fraction.denominator)

    scorecards = []
    totals = {
        "identities": 0,
        "predictions": 0,
        "resolved": 0,
        "unresolved": 0,
        "hit": 0,
        "early": 0,
        "late": 0,
        "missed": 0,
        "observed": 0,
    }
    for key in sorted(grouped, key=lambda item: encounter[item]):
        card = grouped[key]
        resolved = card["predictions"] - card["unresolved"]
        if resolved < min_resolved:
            continue
        observed = card["hit"] + card["early"] + card["late"]
        mean_offset = (
            reduced(card["offset_sum"], card["offset_count"])
            if card["offset_count"]
            else None
        )
        scorecards.append(
            {
                "type": card["type"],
                "identity": card["identity"],
                "predictions": card["predictions"],
                "resolved": resolved,
                "unresolved": card["unresolved"],
                "hit": card["hit"],
                "early": card["early"],
                "late": card["late"],
                "missed": card["missed"],
                "observed": observed,
                "exact_rate": reduced(card["hit"], resolved),
                "window_rate": reduced(observed, resolved),
                "mean_offset": mean_offset,
                "max_abs_offset": card["max_abs_offset"],
            }
        )
        totals["identities"] += 1
        totals["predictions"] += card["predictions"]
        totals["resolved"] += resolved
        totals["unresolved"] += card["unresolved"]
        totals["hit"] += card["hit"]
        totals["early"] += card["early"]
        totals["late"] += card["late"]
        totals["missed"] += card["missed"]
        totals["observed"] += observed
    return {"scorecards": tuple(scorecards), "totals": totals}


class LifetimeChurnForecastScorecardsSignatureTests(unittest.TestCase):
    def test_public_signature_appends_min_resolved_and_scorecard_limit(self):
        parameters = list(
            inspect.signature(
                BranchStore.lifetime_churn_forecast_scorecards
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
                "horizon",
                "forecast_limit",
                "cutoffs",
                "backtest_limit",
                "min_resolved",
                "scorecard_limit",
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
        result = scorecards_call(
            store,
            args,
            windows=(
                (0,), (1,), (1,), (2,), (2,), (0,), (0,), (1,),
            ),
            max_jitter=2,
            horizon=5,
            cutoffs=(2, 3),
        )
        self.assertIsInstance(result, dict)
        self.assertEqual(list(result), ["scorecards", "totals"])
        self.assertIsInstance(result["scorecards"], tuple)
        self.assertEqual(
            list(result["totals"]),
            [
                "identities",
                "predictions",
                "resolved",
                "unresolved",
                "hit",
                "early",
                "late",
                "missed",
                "observed",
            ],
        )
        for scorecard in result["scorecards"]:
            self.assertEqual(
                list(scorecard),
                [
                    "type",
                    "identity",
                    "predictions",
                    "resolved",
                    "unresolved",
                    "hit",
                    "early",
                    "late",
                    "missed",
                    "observed",
                    "exact_rate",
                    "window_rate",
                    "mean_offset",
                    "max_abs_offset",
                ],
            )
            self.assertIsInstance(scorecard["exact_rate"], tuple)
            self.assertIsInstance(scorecard["window_rate"], tuple)


class LifetimeChurnForecastScorecardsResultTests(unittest.TestCase):
    EXTENDED = (
        (0,), (1,), (1,), (2,), (2,), (0,), (0,), (1,),
    )

    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        return scorecards_call(self.store, self.args, **overrides)

    def evolution(self, **overrides):
        kwargs = churn_kwargs(self.args, **overrides)
        for name in (
            "churn_limit",
            "streak_limit",
            "min_identities",
            "wave_limit",
            "min_waves",
            "recurrence_limit",
            "max_jitter",
            "periodicity_limit",
            "horizon",
            "forecast_limit",
        ):
            kwargs.pop(name, None)
        return self.store.lifetime_evolution(**kwargs)

    def test_matches_independent_aggregation_of_evolution(self):
        cases = (
            self.EXTENDED,
            ((0,), (1,), (1,), (2,), (2,), (0,)),
            ((0,), (1,), (2,)),
            ((0, 2), (1,), (1,), (0, 1, 2)),
            (),
        )
        for windows in cases:
            evolution = self.evolution(windows=windows)
            waves = reference_waves(evolution, 1)["waves"]
            cutoff_choices = tuple(range(max(len(waves) - 1, 0)))
            for max_jitter in (0, 2):
                for horizon in (1, 3, 6):
                    for min_resolved in (1, 2):
                        with self.subTest(
                            windows=windows,
                            max_jitter=max_jitter,
                            horizon=horizon,
                            cutoffs=cutoff_choices,
                            min_resolved=min_resolved,
                        ):
                            self.assertEqual(
                                self.call(
                                    windows=windows,
                                    max_jitter=max_jitter,
                                    horizon=horizon,
                                    cutoffs=cutoff_choices,
                                    backtest_limit=10_000,
                                    min_resolved=min_resolved,
                                ),
                                reference_scorecards(
                                    evolution,
                                    1,
                                    max_jitter,
                                    horizon,
                                    cutoff_choices,
                                    min_resolved,
                                ),
                            )

    def test_diamond_fixture_cards_aggregate_both_cutoffs(self):
        result = self.call(
            windows=self.EXTENDED,
            max_jitter=2,
            horizon=5,
            cutoffs=(2, 3),
        )
        waves = reference_waves(
            self.evolution(windows=self.EXTENDED), 1
        )["waves"]
        encounter: dict[tuple[str, object], int] = {}
        for wave in waves:
            for member in wave["members"]:
                key = (member["type"], member["identity"])
                if key not in encounter:
                    encounter[key] = len(encounter)
        keys = [(card["type"], card["identity"]) for card in
                result["scorecards"]]
        self.assertEqual(keys, sorted(keys, key=encounter.__getitem__))
        self.assertEqual(len(result["scorecards"]), 3)
        for scorecard in result["scorecards"]:
            self.assertEqual(
                scorecard,
                {
                    "type": scorecard["type"],
                    "identity": scorecard["identity"],
                    "predictions": 10,
                    "resolved": 1,
                    "unresolved": 9,
                    "hit": 1,
                    "early": 0,
                    "late": 0,
                    "missed": 0,
                    "observed": 1,
                    "exact_rate": (1, 1),
                    "window_rate": (1, 1),
                    "mean_offset": (0, 1),
                    "max_abs_offset": 0,
                },
            )
        self.assertEqual(
            result["totals"],
            {
                "identities": 3,
                "predictions": 30,
                "resolved": 3,
                "unresolved": 27,
                "hit": 3,
                "early": 0,
                "late": 0,
                "missed": 0,
                "observed": 3,
            },
        )

    def test_single_cutoff_counts_each_prediction_once(self):
        result = self.call(
            windows=self.EXTENDED,
            max_jitter=2,
            horizon=5,
            cutoffs=(2,),
        )
        self.assertEqual(len(result["scorecards"]), 3)
        for scorecard in result["scorecards"]:
            self.assertEqual(scorecard["predictions"], 5)
            self.assertEqual(scorecard["resolved"], 1)
            self.assertEqual(scorecard["unresolved"], 4)
            self.assertEqual(scorecard["hit"], 1)

    def test_min_resolved_drops_under_resolved_objects_and_totals(self):
        # Across cutoffs 2 and 3 every object resolves exactly once, so
        # min_resolved two retains no card at all.
        result = self.call(
            windows=self.EXTENDED,
            max_jitter=2,
            horizon=5,
            cutoffs=(2, 3),
            min_resolved=2,
        )
        self.assertEqual(result["scorecards"], ())
        self.assertEqual(
            result["totals"],
            {
                "identities": 0,
                "predictions": 0,
                "resolved": 0,
                "unresolved": 0,
                "hit": 0,
                "early": 0,
                "late": 0,
                "missed": 0,
                "observed": 0,
            },
        )

    def test_empty_cutoffs_return_empty_scorecards_and_zero_totals(self):
        result = self.call(cutoffs=())
        self.assertEqual(result["scorecards"], ())
        self.assertEqual(
            result["totals"],
            {
                "identities": 0,
                "predictions": 0,
                "resolved": 0,
                "unresolved": 0,
                "hit": 0,
                "early": 0,
                "late": 0,
                "missed": 0,
                "observed": 0,
            },
        )

    def test_empty_cutoffs_still_run_every_state_check(self):
        with self.assertRaises(KeyError):
            self.call(cutoffs=(), reference="ghost")
        with self.assertRaises(ValueError):
            self.call(cutoffs=(), limit=1)


class LifetimeChurnForecastScorecardsScoringTests(unittest.TestCase):
    @staticmethod
    def _make_wave(*members):
        return (
            {
                "members": tuple(
                    {"type": type_name, "identity": identity}
                    for type_name, identity in members
                )
            },
        )

    @staticmethod
    def _make_outcome(
        type_name,
        identity,
        status,
        predicted,
        actual=None,
        ordinal=1,
    ):
        return {
            "type": type_name,
            "identity": identity,
            "ordinal": ordinal,
            "predicted": predicted,
            "earliest": predicted,
            "latest": predicted,
            "actual": actual,
            "status": status,
        }

    @staticmethod
    def _make_cutoff(*outcomes):
        return {"cutoff": 0, "outcomes": tuple(outcomes), "totals": {}}

    def test_rates_offsets_and_totals_over_duplicate_predictions(self):
        # A: hit at 3, early at 4 (actual 3), unresolved once;
        # B: late once (actual 7), missed once. Every prediction,
        # including the duplicated object across cutoffs, counts.
        waves = self._make_wave(("node", "A"), ("node", "B"))
        cutoffs = (
            self._make_cutoff(
                self._make_outcome("node", "A", "hit", 3, actual=3),
                self._make_outcome("node", "B", "late", 6, actual=7),
            ),
            self._make_cutoff(
                self._make_outcome("node", "A", "early", 4, actual=3),
                self._make_outcome("node", "A", "unresolved", 9),
                self._make_outcome("node", "B", "missed", 2),
            ),
        )
        result = BranchStore._build_lifetime_churn_forecast_scorecards(
            cutoffs, waves, 1, 50
        )
        card_a, card_b = result["scorecards"]
        self.assertEqual(card_a["type"], "node")
        self.assertEqual(card_a["identity"], "A")
        self.assertEqual(card_a["predictions"], 3)
        self.assertEqual(card_a["resolved"], 2)
        self.assertEqual(card_a["unresolved"], 1)
        self.assertEqual(card_a["hit"], 1)
        self.assertEqual(card_a["early"], 1)
        self.assertEqual(card_a["late"], 0)
        self.assertEqual(card_a["missed"], 0)
        self.assertEqual(card_a["observed"], 2)
        # 1/2 exact, 2/2 window; offsets 0 and -1 average to -1/2 and
        # the largest absolute offset is one.
        self.assertEqual(card_a["exact_rate"], (1, 2))
        self.assertEqual(card_a["window_rate"], (1, 1))
        self.assertEqual(card_a["mean_offset"], (-1, 2))
        self.assertEqual(card_a["max_abs_offset"], 1)
        self.assertEqual(card_b["predictions"], 2)
        self.assertEqual(card_b["resolved"], 2)
        self.assertEqual(card_b["unresolved"], 0)
        self.assertEqual(card_b["hit"], 0)
        self.assertEqual(card_b["early"], 0)
        self.assertEqual(card_b["late"], 1)
        self.assertEqual(card_b["missed"], 1)
        self.assertEqual(card_b["observed"], 1)
        # 0/2 reduces to 0/1; the mean of a single offset is 1/1.
        self.assertEqual(card_b["exact_rate"], (0, 1))
        self.assertEqual(card_b["window_rate"], (1, 2))
        self.assertEqual(card_b["mean_offset"], (1, 1))
        self.assertEqual(card_b["max_abs_offset"], 1)
        self.assertEqual(
            result["totals"],
            {
                "identities": 2,
                "predictions": 5,
                "resolved": 4,
                "unresolved": 1,
                "hit": 1,
                "early": 1,
                "late": 1,
                "missed": 1,
                "observed": 3,
            },
        )

    def test_mean_offset_reduces_and_max_uses_magnitude(self):
        # Offsets +2 (late), -4 (early): mean is -2/2 = -1/1 and the
        # largest magnitude is four.
        waves = self._make_wave(("node", "A"))
        cutoffs = (
            self._make_cutoff(
                self._make_outcome("node", "A", "late", 3, actual=5),
                self._make_outcome("node", "A", "early", 9, actual=5),
            ),
        )
        card = BranchStore._build_lifetime_churn_forecast_scorecards(
            cutoffs, waves, 1, 50
        )["scorecards"][0]
        self.assertEqual(card["mean_offset"], (-1, 1))
        self.assertEqual(card["max_abs_offset"], 4)

    def test_no_actual_appearance_leaves_both_offset_fields_none(self):
        waves = self._make_wave(("node", "A"))
        cutoffs = (
            self._make_cutoff(
                self._make_outcome("node", "A", "missed", 3),
                self._make_outcome("node", "A", "unresolved", 8),
            ),
        )
        card = BranchStore._build_lifetime_churn_forecast_scorecards(
            cutoffs, waves, 1, 50
        )["scorecards"][0]
        self.assertEqual(card["resolved"], 1)
        self.assertEqual(card["observed"], 0)
        self.assertEqual(card["exact_rate"], (0, 1))
        self.assertEqual(card["window_rate"], (0, 1))
        self.assertIsNone(card["mean_offset"])
        self.assertIsNone(card["max_abs_offset"])

    def test_only_unresolved_outcomes_are_filtered_out(self):
        waves = self._make_wave(("node", "A"), ("node", "B"))
        cutoffs = (
            self._make_cutoff(
                self._make_outcome("node", "A", "unresolved", 4),
                self._make_outcome("node", "B", "hit", 4, actual=4),
            ),
        )
        result = BranchStore._build_lifetime_churn_forecast_scorecards(
            cutoffs, waves, 1, 50
        )
        self.assertEqual(
            [(c["type"], c["identity"]) for c in result["scorecards"]],
            [("node", "B")],
        )
        # A's unresolved prediction is absent from every total.
        self.assertEqual(result["totals"]["predictions"], 1)
        self.assertEqual(result["totals"]["unresolved"], 0)

    def test_min_resolved_threshold_is_inclusive(self):
        waves = self._make_wave(("node", "A"))
        cutoffs = (
            self._make_cutoff(
                self._make_outcome("node", "A", "hit", 3, actual=3),
                self._make_outcome("node", "A", "unresolved", 9),
            ),
        )
        # One resolved outcome fails a threshold of two: the card is
        # dropped, leaving an empty set rather than an error.
        filtered = BranchStore._build_lifetime_churn_forecast_scorecards(
            cutoffs, waves, 2, 50
        )
        self.assertEqual(filtered["scorecards"], ())
        self.assertEqual(filtered["totals"]["identities"], 0)
        # A threshold of one retains it (the bound is inclusive).
        card = BranchStore._build_lifetime_churn_forecast_scorecards(
            cutoffs, waves, 1, 50
        )["scorecards"][0]
        self.assertEqual(card["resolved"], 1)

    def test_scorecards_order_by_first_wave_appearance(self):
        # Outcomes mention B before A, but A appears first in the waves.
        waves = (
            self._make_wave(("node", "A"))[0],
            self._make_wave(("edge", "B"))[0],
        )
        cutoffs = (
            self._make_cutoff(
                self._make_outcome("edge", "B", "hit", 2, actual=2),
                self._make_outcome("node", "A", "hit", 2, actual=2),
            ),
        )
        result = BranchStore._build_lifetime_churn_forecast_scorecards(
            cutoffs, waves, 1, 50
        )
        self.assertEqual(
            [(c["type"], c["identity"]) for c in result["scorecards"]],
            [("node", "A"), ("edge", "B")],
        )

    def test_scorecard_limit_raises_without_partial_cards(self):
        waves = self._make_wave(("node", "A"), ("node", "B"))
        cutoffs = (
            self._make_cutoff(
                self._make_outcome("node", "A", "hit", 2, actual=2),
                self._make_outcome("node", "B", "hit", 2, actual=2),
            ),
        )
        with self.assertRaises(ValueError) as caught:
            BranchStore._build_lifetime_churn_forecast_scorecards(
                cutoffs, waves, 1, 1
            )
        self.assertIn("scorecard limit", str(caught.exception))
        result = BranchStore._build_lifetime_churn_forecast_scorecards(
            cutoffs, waves, 1, 2
        )
        self.assertEqual(len(result["scorecards"]), 2)

    def test_identity_is_an_isolated_copy(self):
        identity = ("left", ("right", "leaf"))
        waves = self._make_wave(("gap", identity))
        cutoffs = (
            self._make_cutoff(
                self._make_outcome("gap", identity, "hit", 3, actual=3)
            ),
        )
        card = BranchStore._build_lifetime_churn_forecast_scorecards(
            cutoffs, waves, 1, 50
        )["scorecards"][0]
        self.assertEqual(card["identity"], identity)
        self.assertIsNot(card["identity"], identity)


class LifetimeChurnForecastScorecardsValidationTests(unittest.TestCase):
    EXTENDED = (
        (0,), (1,), (1,), (2,), (2,), (0,), (0,), (1,),
    )

    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        overrides.setdefault("causes", ("a0",))
        return scorecards_call(self.store, self.args, **overrides)

    def extended(self, **overrides):
        overrides.setdefault("windows", self.EXTENDED)
        overrides.setdefault("horizon", 5)
        overrides.setdefault("max_jitter", 2)
        overrides.setdefault("cutoffs", (2,))
        overrides.setdefault("backtest_limit", 100)
        overrides.setdefault("min_resolved", 1)
        overrides.setdefault("scorecard_limit", 50)
        return self.call(**overrides)

    def test_new_limits_must_be_non_bool_positive_ints(self):
        for name in ("min_resolved", "scorecard_limit"):
            for bad in (True, False, 1.0, "1", None):
                with self.subTest(name=name, bad=bad):
                    with self.assertRaises(TypeError):
                        self.extended(**{name: bad})
            with self.assertRaises(ValueError):
                self.extended(**{name: 0})
            with self.assertRaises(ValueError):
                self.extended(**{name: -3})

    def test_new_parameters_validated_in_order_after_backtest_limit(self):
        # backtest_limit fails before min_resolved.
        with self.assertRaises(ValueError):
            self.extended(backtest_limit=0, min_resolved="x")
        with self.assertRaises(TypeError):
            self.extended(backtest_limit="x", min_resolved="x")
        # min_resolved fails before scorecard_limit.
        with self.assertRaises(ValueError):
            self.extended(min_resolved=0, scorecard_limit="x")
        with self.assertRaises(TypeError):
            self.extended(min_resolved="x", scorecard_limit="x")
        # Once min_resolved passes, scorecard_limit's error fires.
        with self.assertRaises(TypeError):
            self.extended(scorecard_limit="x")
        with self.assertRaises(ValueError):
            self.extended(scorecard_limit=0)
        # All parameter checks still precede every state lookup.
        with self.assertRaises(ValueError):
            self.call(
                reference="ghost", cutoffs=(0,), min_resolved=0
            )
        with self.assertRaises(TypeError):
            self.call(
                reference="ghost", cutoffs=(0,), min_resolved="x"
            )
        with self.assertRaises(ValueError):
            self.call(
                token="whatever", cutoffs=(0,), scorecard_limit=0
            )

    def test_scorecard_limit_counts_retained_cards_after_filtering(self):
        # Cutoffs 2 and 3 retain three cards with min_resolved one.
        with self.assertRaises(ValueError) as caught:
            self.extended(cutoffs=(2, 3), scorecard_limit=2)
        self.assertIn("scorecard limit", str(caught.exception))
        # Raising min_resolved filters every card, so the same cap of
        # two no longer trips.
        result = self.extended(
            cutoffs=(2, 3), min_resolved=2, scorecard_limit=2
        )
        self.assertEqual(result["scorecards"], ())
        # The complete set is built before the cap: a passing call
        # returns every card, never a truncated prefix.
        result = self.extended(cutoffs=(2, 3), scorecard_limit=3)
        self.assertEqual(len(result["scorecards"]), 3)

    def test_backtest_limit_still_bounds_the_outcome_set(self):
        # Cutoff 2 alone makes fifteen predictions although only three
        # cards survive; the backtest cap still fires.
        with self.assertRaises(ValueError) as caught:
            self.extended(cutoffs=(2,), backtest_limit=14)
        self.assertIn("backtest limit", str(caught.exception))
        result = self.extended(cutoffs=(2,), backtest_limit=15)
        self.assertEqual(result["totals"]["identities"], 3)

    def test_cutoff_range_is_still_checked_against_the_actual_waves(self):
        with self.assertRaises(ValueError) as caught:
            self.extended(cutoffs=(4,))
        self.assertIn("out of range", str(caught.exception))

    def test_empty_cutoffs_never_trip_either_cap(self):
        result = self.call(
            cutoffs=(), backtest_limit=1, scorecard_limit=1
        )
        self.assertEqual(result["scorecards"], ())
        self.assertEqual(result["totals"]["predictions"], 0)

    def test_wave_limit_is_still_enforced_ahead_of_scoring(self):
        with self.assertRaises(ValueError) as caught:
            self.extended(wave_limit=1)
        self.assertIn("wave limit", str(caught.exception))

    def test_batch_caps_still_fire_ahead_of_scoring(self):
        with self.assertRaises(ValueError) as caught:
            self.call(
                total_diff_limit=5,
                cutoffs=(0,),
                backtest_limit=100,
                min_resolved=1,
                scorecard_limit=50,
            )
        self.assertIn("total diff", str(caught.exception))

    def test_shared_validation_errors_match_backtests(self):
        with self.assertRaises(TypeError):
            self.call(windows="x")
        with self.assertRaises(ValueError):
            self.call(windows=((-1,),))
        with self.assertRaises(ValueError):
            self.call(direction="sideways")


class LifetimeChurnForecastScorecardsStateTests(unittest.TestCase):
    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, **overrides):
        overrides.setdefault("causes", ("a0",))
        return scorecards_call(self.store, self.args, **overrides)

    def test_unknown_branch_history_node_and_cause_raise(self):
        with self.assertRaises(KeyError):
            self.call(
                reference="ghost",
                cutoffs=(0,),
                backtest_limit=1,
                min_resolved=1,
                scorecard_limit=1,
            )
        with self.assertRaises(KeyError):
            self.call(
                series={
                    "a": (("r0", "a0"), ("r1", "a0"), ("r2", "a0")),
                    "b": (("r0", "a0"), ("r1", "W"), ("r2", "nope2")),
                },
                cutoffs=(0,),
                backtest_limit=1,
                min_resolved=1,
                scorecard_limit=1,
            )
        with self.assertRaises(KeyError) as caught:
            self.call(
                causes=("zzz",),
                cutoffs=(0,),
                backtest_limit=1,
                min_resolved=1,
                scorecard_limit=1,
            )
        self.assertEqual(caught.exception.args[0], "zzz")

    def test_range_error_fires_after_the_state_checks(self):
        with self.assertRaises(KeyError):
            self.call(
                reference="ghost",
                cutoffs=(99,),
                backtest_limit=1,
                min_resolved=1,
                scorecard_limit=1,
            )


class LifetimeChurnForecastScorecardsIsolationTests(unittest.TestCase):
    EXTENDED = (
        (0,), (1,), (1,), (2,), (2,), (0,), (0,), (1,),
    )

    def test_result_levels_are_mutually_unshared(self):
        store, args = diamond_store()
        result = scorecards_call(
            store,
            args,
            windows=self.EXTENDED,
            max_jitter=2,
            horizon=5,
            cutoffs=(2, 3),
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
            elif isinstance(value, list):
                for item in value:
                    collect(item)

        collect(result)
        self.assertEqual(len(seen_ids), len(set(seen_ids)))

    def test_mutating_result_never_affects_later_calls(self):
        store, args = diamond_store()
        first = scorecards_call(
            store,
            args,
            windows=self.EXTENDED,
            max_jitter=2,
            horizon=5,
            cutoffs=(2, 3),
        )
        pristine = copy.deepcopy(first)
        for scorecard in first["scorecards"]:
            scorecard["predictions"] = 99
            scorecard["resolved"] = 99
            scorecard["hit"] = 99
            scorecard["exact_rate"] = (9, 9)
            scorecard["mean_offset"] = (9, 9)
            scorecard["max_abs_offset"] = 99
        self.assertEqual(
            scorecards_call(
                store,
                args,
                windows=self.EXTENDED,
                max_jitter=2,
                horizon=5,
                cutoffs=(2, 3),
            ),
            pristine,
        )

    def test_success_and_failure_change_nothing(self):
        store, args = diamond_store()
        heads = dict(store._heads)
        appends = copy.deepcopy(store._appends)
        merges = copy.deepcopy(store._merges)
        audit = store.audit_log()

        scorecards_call(
            store,
            args,
            windows=self.EXTENDED,
            max_jitter=2,
            horizon=5,
            cutoffs=(2,),
        )
        failures = [
            dict(windows="x"),
            dict(windows=((-1,),)),
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
            dict(min_waves=0),
            dict(min_waves="x"),
            dict(recurrence_limit=0),
            dict(recurrence_limit="x"),
            dict(max_jitter=-1),
            dict(max_jitter="x"),
            dict(periodicity_limit=0),
            dict(periodicity_limit="x"),
            dict(horizon=0),
            dict(horizon="x"),
            dict(forecast_limit=0),
            dict(forecast_limit="x"),
            dict(cutoffs="x"),
            dict(cutoffs=(True,)),
            dict(cutoffs=(-1,)),
            dict(cutoffs=(1, 1)),
            dict(cutoffs=(2, 1)),
            dict(backtest_limit=0),
            dict(backtest_limit="x"),
            dict(min_resolved=0),
            dict(min_resolved="x"),
            dict(min_resolved=True),
            dict(scorecard_limit=0),
            dict(scorecard_limit="x"),
            dict(scorecard_limit=False),
            dict(wave_limit=1, cutoffs=(2,)),
            dict(cutoffs=(99,)),
            dict(forecast_limit=1, cutoffs=(2,)),
            dict(cutoffs=(2,), backtest_limit=1),
            dict(cutoffs=(2, 3), scorecard_limit=2),
            dict(causes=("zzz",), cutoffs=(2,)),
            dict(limit=1, cutoffs=(2,)),
            dict(reference="ghost", cutoffs=(2,)),
        ]
        for overrides in failures:
            with self.subTest(overrides=overrides):
                with self.assertRaises(Exception):
                    scorecards_call(
                        store,
                        args,
                        causes=("a0",),
                        windows=self.EXTENDED,
                        max_jitter=2,
                        horizon=5,
                        **overrides,
                    )

        self.assertEqual(dict(store._heads), heads)
        self.assertEqual(copy.deepcopy(store._appends), appends)
        self.assertEqual(copy.deepcopy(store._merges), merges)
        self.assertEqual(store.audit_log(), audit)

    def test_existing_public_entries_are_unaffected(self):
        store, args = diamond_store()
        scorecards_call(
            store,
            args,
            windows=self.EXTENDED,
            max_jitter=2,
            horizon=5,
            cutoffs=(2, 3),
        )
        from tests.test_lifetime_churn_waves import waves_call
        from tests.test_lifetime_churn_forecasts import forecasts_call
        from tests.test_lifetime_churn_forecast_backtests import (
            backtests_call,
        )
        from tests.test_lifetime_churn_periodicities import (
            periodicities_call,
        )

        waves = waves_call(store, args, windows=self.EXTENDED)
        forecasts = forecasts_call(
            store, args, windows=self.EXTENDED, max_jitter=2, horizon=5
        )
        backtests = backtests_call(
            store,
            args,
            windows=self.EXTENDED,
            max_jitter=2,
            horizon=5,
            cutoffs=(2, 3),
        )
        periodicities = periodicities_call(
            store, args, windows=self.EXTENDED, max_jitter=2
        )
        self.assertEqual(
            waves_call(store, args, windows=self.EXTENDED), waves
        )
        self.assertEqual(
            forecasts_call(
                store, args, windows=self.EXTENDED,
                max_jitter=2, horizon=5
            ),
            forecasts,
        )
        self.assertEqual(
            backtests_call(
                store,
                args,
                windows=self.EXTENDED,
                max_jitter=2,
                horizon=5,
                cutoffs=(2, 3),
            ),
            backtests,
        )
        self.assertEqual(
            periodicities_call(
                store, args, windows=self.EXTENDED, max_jitter=2
            ),
            periodicities,
        )


class LifetimeChurnForecastScorecardsTokenTests(unittest.TestCase):
    EXTENDED = (
        (0,), (1,), (1,), (2,), (2,), (0,), (0,), (1,),
    )

    def setUp(self):
        self.store, self.args = diamond_store()

    def call(self, token=None, **overrides):
        return scorecards_call(
            self.store, self.args, token=token, **overrides
        )

    def test_success_consumes_one_read_and_failure_refunds(self):
        token = self.store.create_snapshot(2)
        self.call(
            token=token,
            windows=self.EXTENDED,
            max_jitter=2,
            horizon=5,
            cutoffs=(2, 3),
        )
        # Over the scorecard cap: failure refunds.
        with self.assertRaises(ValueError):
            self.call(
                token=token,
                windows=self.EXTENDED,
                max_jitter=2,
                horizon=5,
                cutoffs=(2, 3),
                scorecard_limit=1,
            )
        # Over the backtest cap: failure refunds.
        with self.assertRaises(ValueError):
            self.call(
                token=token,
                windows=self.EXTENDED,
                max_jitter=2,
                horizon=5,
                cutoffs=(2,),
                backtest_limit=1,
            )
        # Out-of-range cutoff: failure refunds.
        with self.assertRaises(ValueError):
            self.call(
                token=token,
                windows=self.EXTENDED,
                cutoffs=(99,),
            )
        # State error: failure refunds.
        with self.assertRaises(KeyError):
            self.call(
                token=token,
                causes=("zzz",),
                cutoffs=(2,),
            )
        # Parameter validation runs before the token is touched.
        with self.assertRaises(ValueError):
            self.call(token=token, cutoffs=(-1,))
        with self.assertRaises(TypeError):
            self.call(token=token, min_resolved="x")
        # An empty-cutoff success still performs one read.
        self.call(token=token, cutoffs=())
        with self.assertRaises(RuntimeError):
            self.call(token=token, cutoffs=())

    def test_results_come_from_the_frozen_view(self):
        before = self.call(
            windows=self.EXTENDED,
            max_jitter=2,
            horizon=5,
            cutoffs=(2, 3),
        )
        token = self.store.create_snapshot(4)
        self.store.append("a", "a2", 7, {"x": -1})
        self.store.create("d", "root")
        self.store.append("d", "d0", 8, {"z": 3})
        self.store.merge("a", "d", "M", 9, {"z": 1})
        self.store.append("main", "r3", 10, {"x": 1})
        self.assertEqual(
            self.call(
                token=token,
                windows=self.EXTENDED,
                max_jitter=2,
                horizon=5,
                cutoffs=(2, 3),
            ),
            before,
        )


if __name__ == "__main__":
    unittest.main()
