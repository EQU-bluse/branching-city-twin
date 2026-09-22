import copy
import inspect

import pytest

from city_twin.branches import BranchStore
from city_twin.event_graph import EventGraph


def build_store():
    g = EventGraph()
    g.add("r", 0, (), {})
    g.add("a", 1, ("r",), {"x": 1})
    g.add("b", 2, ("r",), {"y": 2})
    g.add("m", 3, ("a", "b"), {})
    g.add("c", 4, ("m",), {"z": 3, "é": 1, "a": 0})
    g.add("d", 5, ("r", "c"), {"q": 5})  # direct edge r->d beats long path
    g.add("x", 6, ("r",), {"k": 9})      # in graph, not ancestor of h
    s = BranchStore(g)
    s.create("br", "d")
    s.append("br", "h", 7, {})  # head h -> d
    return g, s


def test_basic_descendants_and_order():
    g, s = build_store()
    res = s.explain_impact("br", "r")
    ids = [r["event_id"] for r in res]
    # parents-first, (at,id): a(1), b(2), m(3), c(4), d(5), h(7); x excluded
    assert ids == ["a", "b", "m", "c", "d", "h"]
    assert all(list(r.keys()) == ["event_id", "at", "path", "keys"] for r in res)
    by = {r["event_id"]: r for r in res}
    assert by["a"]["at"] == 1
    assert by["a"]["path"] == ("r", "a")
    assert by["a"]["keys"] == ("x",)
    assert by["m"]["path"] == ("r", "a", "m")  # tie: (r,a,m) < (r,b,m)
    assert by["m"]["keys"] == ()
    assert by["c"]["path"] == ("r", "a", "m", "c")
    assert by["c"]["keys"] == ("a", "z", "é")  # codepoint order
    assert by["d"]["path"] == ("r", "d")       # fewest edges wins
    assert by["h"]["path"] == ("r", "d", "h")


def test_start_excluded_and_partial_views():
    g, s = build_store()
    assert [r["event_id"] for r in s.explain_impact("br", "a")] == ["m", "c", "d", "h"]
    by = {r["event_id"]: r for r in s.explain_impact("br", "a")}
    assert by["d"]["path"] == ("a", "m", "c", "d")
    assert [r["event_id"] for r in s.explain_impact("br", "m")] == ["c", "d", "h"]
    assert s.explain_impact("br", "h") == ()


def test_validation_order():
    g, s = build_store()
    with pytest.raises(TypeError):
        s.explain_impact(123, "r")
    with pytest.raises(ValueError):
        s.explain_impact("", "r")
    with pytest.raises(TypeError):
        s.explain_impact("br", 456)
    with pytest.raises(ValueError):
        s.explain_impact("br", "")
    # event_id validated before branch lookup
    with pytest.raises(TypeError):
        s.explain_impact("nope", 456)
    with pytest.raises(ValueError):
        s.explain_impact("nope", "")
    with pytest.raises(KeyError):
        s.explain_impact("nope", "r")
    with pytest.raises(KeyError):
        s.explain_impact("br", "ghost")   # not in graph
    with pytest.raises(KeyError):
        s.explain_impact("br", "x")       # in graph, not in head closure
    # bool is not an acceptable str
    with pytest.raises(TypeError):
        s.explain_impact(True, "r")


def test_no_defaults():
    sig = inspect.signature(BranchStore.explain_impact)
    params = sig.parameters
    assert list(params) == ["self", "name", "event_id"]
    assert params["name"].default is inspect.Parameter.empty
    assert params["event_id"].default is inspect.Parameter.empty
    # branches.py uses `from __future__ import annotations`
    assert sig.return_annotation == "tuple[dict[str, object], ...]"


def test_isolation_and_readonly():
    g, s = build_store()
    before_json = g.to_json()
    before_audit = copy.deepcopy(s.audit_log())
    before_heads = {n: s.head(n) for n in ["br"]}

    r1 = s.explain_impact("br", "r")
    # detach: mutate returned objects
    r1[0]["keys"].__class__  # tuple, immutable
    r1[0]["path"] = ("hacked",)
    r1[0]["event_id"] = "hacked"
    r2 = s.explain_impact("br", "r")
    assert r2[0]["event_id"] == "a"
    assert r2[0]["path"] == ("r", "a")
    assert r1 != r2  # mutation only affected local copy
    r3 = s.explain_impact("br", "r")
    assert r2 == r3

    # failures change nothing
    for call in (
        lambda: s.explain_impact(1, "r"),
        lambda: s.explain_impact("br", ""),
        lambda: s.explain_impact("nope", "r"),
        lambda: s.explain_impact("br", "ghost"),
        lambda: s.explain_impact("br", "x"),
    ):
        with pytest.raises((TypeError, ValueError, KeyError)):
            call()

    assert g.to_json() == before_json
    assert s.audit_log() == before_audit
    assert {n: s.head(n) for n in ["br"]} == before_heads
    # idempotency records still work after queries
    s.append("br", "h", 7, {})


def test_diamond_tiebreak_deeper():
    g = EventGraph()
    g.add("r", 0, (), {})
    # level 1
    g.add("p", 1, ("r",), {})
    g.add("q", 1, ("r",), {})
    # level 2 both funnel into z via different middlemen, same length
    g.add("mp", 2, ("p",), {})
    g.add("mq", 2, ("q",), {})
    g.add("z", 3, ("mp", "mq"), {})
    s = BranchStore(g)
    s.create("br", "z")
    rec = {r["event_id"]: r for r in s.explain_impact("br", "r")}
    # (r,p,mp,z) vs (r,q,mq,z): p<q decides
    assert rec["z"]["path"] == ("r", "p", "mp", "z")
    assert [r["event_id"] for r in s.explain_impact("br", "r")] == [
        "p", "q", "mp", "mq", "z"
    ]
