import copy
import threading
import unittest

from city_twin.branches import BranchStore
from city_twin.event_graph import EventGraph


def diamond_store() -> tuple[BranchStore, tuple]:
    """Three aligned checkpoints over the standard merge fixture."""
    graph = EventGraph()
    graph.add("root", 0, (), {})
    store = BranchStore(graph)
    store.create("main", "root")
    store.create("a", "root")
    store.append("main", "r0", 1, {})
    store.append("a", "a0", 2, {"x": 2})
    store.append("main", "r1", 3, {})
    store.create("b", "a0")
    store.append("b", "bu", 4, {})
    store.create("c", "a0")
    store.append("c", "cv", 4, {})
    store.merge("b", "c", "W", 5, {"y": 8})
    store.append("main", "r2", 6, {})
    series = {
        "a": (("r0", "a0"), ("r1", "a0"), ("r2", "a0")),
        "b": (("r0", "a0"), ("r1", "W"), ("r2", "W")),
    }
    scenario = {
        "weights": {"x": 1, "y": 1},
        "total_budget": 20,
        "key_budgets": {},
    }
    args = ("main", series, scenario, "total_budget", (0, 10, 20), 0, 2, (), (), 10)
    return store, args


_FIXTURE_ARGS = diamond_store()[1]
_, _SERIES, _SCENARIO, _AXIS, _VALUES, _MN, _MX, _REQ, _EXCL, _LIMIT = (
    _FIXTURE_ARGS
)


def slice_kwargs(**overrides):
    kwargs = dict(
        reference="main",
        series=_SERIES,
        base_scenario=_SCENARIO,
        axis=_AXIS,
        values=_VALUES,
        min_size=_MN,
        max_size=_MX,
        required=_REQ,
        exclusive_pairs=_EXCL,
        limit=_LIMIT,
        indices=(0, 1),
        causes=("a0",),
        direction="both",
        depth=99,
        node_limit=50,
        change_limit=50,
        lifetime_limit=50,
    )
    kwargs.update(overrides)
    return kwargs


def compare_kwargs():
    kwargs = slice_kwargs()
    del kwargs["indices"]
    kwargs.update(
        left_indices=(0, 1),
        right_indices=(1, 2),
        diff_limit=50,
    )
    return kwargs


def evolution_kwargs():
    kwargs = slice_kwargs()
    del kwargs["indices"]
    kwargs.update(
        windows=((0,), (1,), (2,)),
        diff_limit=50,
        window_limit=50,
        total_diff_limit=50,
    )
    return kwargs


class CreateSnapshotTests(unittest.TestCase):
    def test_returns_unique_nonempty_str_tokens(self):
        store, _ = diamond_store()
        first = store.create_snapshot(3)
        second = store.create_snapshot(3)
        self.assertIsInstance(first, str)
        self.assertIsInstance(second, str)
        self.assertTrue(first)
        self.assertTrue(second)
        self.assertNotEqual(first, second)

    def test_max_reads_rejects_bools_and_non_ints(self):
        store, _ = diamond_store()
        for bad in (True, False, 1.0, "1", None, (1,), [1]):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    store.create_snapshot(bad)

    def test_max_reads_rejects_values_below_one(self):
        store, _ = diamond_store()
        for bad in (0, -1, -100):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    store.create_snapshot(bad)

    def test_cap_is_32_unreleased_tokens_without_eviction(self):
        store, _ = diamond_store()
        tokens = [store.create_snapshot(1) for _ in range(32)]
        with self.assertRaises(RuntimeError):
            store.create_snapshot(1)

        # Expired tokens are still unreleased: they keep their slot.
        store.cascade_slice_lifetimes(token=tokens[0], **slice_kwargs())
        with self.assertRaises(RuntimeError):
            store.create_snapshot(1)

        # Freeing one slot -- including the expired token's -- makes room.
        self.assertIsNone(store.release_snapshot(tokens[0]))
        fresh = store.create_snapshot(1)
        self.assertIsInstance(fresh, str)
        with self.assertRaises(RuntimeError):
            store.create_snapshot(1)

        # The oldest live token was never evicted: it still answers.
        store.cascade_slice_lifetimes(token=tokens[1], **slice_kwargs())

    def test_failed_cap_creation_changes_nothing(self):
        store, _ = diamond_store()
        tokens = [store.create_snapshot(1) for _ in range(32)]
        with self.assertRaises(RuntimeError):
            store.create_snapshot(1)
        # Every previously issued token still has its full allowance.
        store.cascade_slice_lifetimes(token=tokens[-1], **slice_kwargs())
        with self.assertRaises(RuntimeError):
            store.cascade_slice_lifetimes(token=tokens[-1], **slice_kwargs())

    def test_creation_is_read_only_over_live_state(self):
        store, _ = diamond_store()
        graph_ids = set(store._graph._at)
        heads = dict(store._heads)
        audit = store.audit_log()
        store.create_snapshot(5)
        self.assertEqual(set(store._graph._at), graph_ids)
        self.assertEqual(store._heads, heads)
        self.assertEqual(store.audit_log(), audit)


class ReleaseSnapshotTests(unittest.TestCase):
    def test_release_live_token_returns_none(self):
        store, _ = diamond_store()
        token = store.create_snapshot(3)
        self.assertIsNone(store.release_snapshot(token))

    def test_release_expired_token_succeeds_and_frees_slot(self):
        store, _ = diamond_store()
        tokens = [store.create_snapshot(1) for _ in range(31)]
        dying = store.create_snapshot(1)
        store.cascade_slice_lifetimes(token=dying, **slice_kwargs())
        with self.assertRaises(RuntimeError):
            store.create_snapshot(1)
        self.assertIsNone(store.release_snapshot(dying))
        store.create_snapshot(1)
        for other in tokens:
            store.cascade_slice_lifetimes(token=other, **slice_kwargs())

    def test_release_validates_type_and_emptiness_first(self):
        store, _ = diamond_store()
        with self.assertRaises(TypeError):
            store.release_snapshot(123)
        with self.assertRaises(TypeError):
            store.release_snapshot(False)
        with self.assertRaises(ValueError):
            store.release_snapshot("")

    def test_unknown_released_and_foreign_tokens_raise_key_error(self):
        store, _ = diamond_store()
        other, _ = diamond_store()
        foreign = other.create_snapshot(2)
        with self.assertRaises(KeyError):
            store.release_snapshot("unknown-token")
        with self.assertRaises(KeyError):
            store.release_snapshot(foreign)
        token = store.create_snapshot(2)
        store.release_snapshot(token)
        with self.assertRaises(KeyError):
            store.release_snapshot(token)


class TokenValidationInQueriesTests(unittest.TestCase):
    def setUp(self):
        self.store, _ = diamond_store()

    def queries(self, token):
        return (
            lambda: self.store.cascade_slice_lifetimes(
                token=token, **slice_kwargs()
            ),
            lambda: self.store.compare_lifetimes(
                token=token, **compare_kwargs()
            ),
            lambda: self.store.lifetime_evolution(
                token=token, **evolution_kwargs()
            ),
        )

    def test_non_str_token_raises_type_error(self):
        for bad in (123, True, b"x", (), 1.5):
            for query in self.queries(bad):
                with self.subTest(bad=bad):
                    with self.assertRaises(TypeError):
                        query()

    def test_empty_token_raises_value_error(self):
        for query in self.queries(""):
            with self.assertRaises(ValueError):
                query()

    def test_unknown_released_and_foreign_tokens_raise_key_error(self):
        other, _ = diamond_store()
        foreign = other.create_snapshot(2)
        token = self.store.create_snapshot(5)
        self.store.release_snapshot(token)
        for bad in ("never-issued", token, foreign):
            for query in self.queries(bad):
                with self.subTest(bad=bad[:4]):
                    with self.assertRaises(KeyError):
                        query()

    def test_ordinary_parameters_are_validated_before_token(self):
        # A non-str reference fails before a non-str token is inspected.
        kwargs = slice_kwargs(reference=123)
        with self.assertRaises(TypeError):
            self.store.cascade_slice_lifetimes(token=456, **kwargs)
        # A value error in an ordinary parameter likewise wins.
        kwargs = slice_kwargs()
        kwargs["indices"] = (9,)
        with self.assertRaises(ValueError):
            self.store.cascade_slice_lifetimes(token="", **kwargs)

    def test_each_successful_query_consumes_one_read_across_types(self):
        token = self.store.create_snapshot(3)
        self.store.cascade_slice_lifetimes(token=token, **slice_kwargs())
        self.store.compare_lifetimes(token=token, **compare_kwargs())
        self.store.lifetime_evolution(token=token, **evolution_kwargs())
        for query in self.queries(token):
            with self.assertRaises(RuntimeError):
                query()

    def test_failed_state_checks_do_not_consume_reads(self):
        token = self.store.create_snapshot(1)

        # KeyError: branch absent from the frozen view.
        with self.assertRaises(KeyError):
            self.store.cascade_slice_lifetimes(
                token=token, **slice_kwargs(reference="ghost")
            )
        # ValueError: checkpoint index out of range (3 checkpoints: 0..2).
        with self.assertRaises(ValueError):
            self.store.cascade_slice_lifetimes(
                token=token, **slice_kwargs(indices=(7,))
            )
        # KeyError: cause event absent from every selected slice.
        with self.assertRaises(KeyError):
            self.store.cascade_slice_lifetimes(
                token=token, **slice_kwargs(causes=("no-such-cause",))
            )
        # ValueError: record cap tripped inside the snapshot.
        with self.assertRaises(ValueError):
            self.store.cascade_slice_lifetimes(
                token=token, **{**slice_kwargs(indices=(0, 1, 2)),
                                "lifetime_limit": 1}
            )

        # The single read is still available: one query succeeds and the
        # next one proves the token has now expired.
        self.store.cascade_slice_lifetimes(token=token, **slice_kwargs())
        with self.assertRaises(RuntimeError):
            self.store.cascade_slice_lifetimes(token=token, **slice_kwargs())

    def test_ordinary_input_failure_does_not_touch_allowance(self):
        token = self.store.create_snapshot(1)
        kwargs = slice_kwargs()
        kwargs["indices"] = 3  # tuple required -> TypeError before token
        with self.assertRaises(TypeError):
            self.store.cascade_slice_lifetimes(token=token, **kwargs)
        self.store.cascade_slice_lifetimes(token=token, **slice_kwargs())
        with self.assertRaises(RuntimeError):
            self.store.cascade_slice_lifetimes(token=token, **slice_kwargs())


class FrozenViewTests(unittest.TestCase):
    def setUp(self):
        self.store, _ = diamond_store()
        self.before_slice = copy.deepcopy(
            self.store.cascade_slice_lifetimes(**slice_kwargs())
        )
        self.before_compare = copy.deepcopy(
            self.store.compare_lifetimes(**compare_kwargs())
        )
        self.before_evolution = copy.deepcopy(
            self.store.lifetime_evolution(**evolution_kwargs())
        )
        self.token = self.store.create_snapshot(8)

    def mutate(self):
        # Extend an existing branch, create a new one and merge it in after
        # the snapshot was taken.
        self.store.append("a", "a2", 7, {"x": -1})
        self.store.create("d", "root")
        self.store.append("d", "d0", 8, {"z": 3})
        self.store.merge("a", "d", "M", 9, {"z": 1})
        self.store.append("main", "r3", 10, {"x": 1})

    def test_token_queries_keep_the_creation_instant_view(self):
        self.mutate()
        self.assertEqual(
            self.store.cascade_slice_lifetimes(
                token=self.token, **slice_kwargs()
            ),
            self.before_slice,
        )
        self.assertEqual(
            self.store.compare_lifetimes(
                token=self.token, **compare_kwargs()
            ),
            self.before_compare,
        )
        self.assertEqual(
            self.store.lifetime_evolution(
                token=self.token, **evolution_kwargs()
            ),
            self.before_evolution,
        )

    def test_new_branch_is_invisible_to_token_but_visible_live(self):
        self.mutate()
        series_with_d = {
            "a": (("r0", "a0"), ("r1", "a0"), ("r2", "a0")),
            "b": (("r0", "a0"), ("r1", "W"), ("r2", "W")),
            "d": (("r0", "d0"), ("r1", "d0"), ("r2", "d0")),
        }
        # The live store serves the new member without error.
        live_kwargs = slice_kwargs(series=series_with_d, max_size=3)
        self.store.cascade_slice_lifetimes(**live_kwargs)

        # The frozen view never learned branch d: original KeyError contract.
        frozen_kwargs = slice_kwargs(series=series_with_d)
        frozen_kwargs["max_size"] = 3
        with self.assertRaises(KeyError):
            self.store.cascade_slice_lifetimes(
                token=self.token, **frozen_kwargs
            )

    def test_new_head_event_is_unknown_inside_the_snapshot(self):
        self.mutate()
        # a2 exists live and is on branch a's head closure there ...
        series_new_node = {
            "a": (("r0", "a2"), ("r1", "a2"), ("r2", "a2")),
            "b": (("r0", "a0"), ("r1", "W"), ("r2", "W")),
        }
        self.store.cascade_slice_lifetimes(
            **slice_kwargs(series=series_new_node)
        )
        # ... but the snapshot predates a2, so the node lookup raises.
        with self.assertRaises(KeyError):
            self.store.cascade_slice_lifetimes(
                token=self.token, **slice_kwargs(series=series_new_node)
            )

    def test_token_queries_do_not_mutate_live_state_or_each_other(self):
        self.mutate()
        audit_before = self.store.audit_log()
        heads_before = dict(self.store._heads)
        first = self.store.cascade_slice_lifetimes(
            token=self.token, **slice_kwargs()
        )
        first["lifetimes"][0]["transitions"] = ("tampered",)
        second = self.store.cascade_slice_lifetimes(
            token=self.token, **slice_kwargs()
        )
        self.assertEqual(second, self.before_slice)
        self.assertEqual(self.store.audit_log(), audit_before)
        self.assertEqual(self.store._heads, heads_before)
        self.assertEqual(self.store.head("main"), "r3")
        self.assertEqual(self.store.head("a"), "M")

    def test_token_results_match_live_results_at_creation_time(self):
        # Even before mutation the two paths must agree exactly.
        self.assertEqual(
            self.store.cascade_slice_lifetimes(
                token=self.token, **slice_kwargs()
            ),
            self.store.cascade_slice_lifetimes(**slice_kwargs()),
        )


class SnapshotConcurrencyTests(unittest.TestCase):
    def test_concurrent_queries_share_the_read_atomically(self):
        store, _ = diamond_store()
        max_reads = 7
        token = store.create_snapshot(max_reads)
        threads_count = 25
        barrier = threading.Barrier(threads_count)
        results = []
        lock = threading.Lock()

        def worker():
            barrier.wait()
            try:
                store.cascade_slice_lifetimes(
                    token=token, **slice_kwargs()
                )
            except RuntimeError:
                outcome = "expired"
            except BaseException as exc:  # pragma: no cover - failure signal
                outcome = f"other:{type(exc).__name__}"
            else:
                outcome = "ok"
            with lock:
                results.append(outcome)

        threads = [threading.Thread(target=worker) for _ in range(threads_count)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(results.count("ok"), max_reads)
        self.assertEqual(results.count("expired"), threads_count - max_reads)
        self.assertEqual(
            [outcome for outcome in results if outcome.startswith("other:")],
            [],
        )
        # Exactly max_reads successes: the token is now expired.
        with self.assertRaises(RuntimeError):
            store.cascade_slice_lifetimes(token=token, **slice_kwargs())

    def test_concurrent_failures_never_consume_reads(self):
        store, _ = diamond_store()
        threads_count = 16
        # Every thread must be able to reserve a read at once; each reserve
        # is refunded when the state check fails, leaving the full allowance.
        token = store.create_snapshot(threads_count)
        barrier = threading.Barrier(threads_count)
        outcomes = []
        lock = threading.Lock()

        def worker():
            barrier.wait()
            kwargs = slice_kwargs(reference="ghost")
            try:
                store.cascade_slice_lifetimes(token=token, **kwargs)
            except KeyError:
                outcome = "key_error"
            except RuntimeError:  # pragma: no cover - would be a bug
                outcome = "expired"
            with lock:
                outcomes.append(outcome)

        threads = [threading.Thread(target=worker) for _ in range(threads_count)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(outcomes, ["key_error"] * threads_count)
        # All reserves were refunded: exactly the full allowance remains.
        for _ in range(threads_count):
            store.cascade_slice_lifetimes(token=token, **slice_kwargs())
        with self.assertRaises(RuntimeError):
            store.cascade_slice_lifetimes(token=token, **slice_kwargs())


if __name__ == "__main__":
    unittest.main()
