"""Offline unit tests for SelectDenoise Levers 1 & 2 (mock LLMs, no API)."""
import asyncio, types
import multi_agent_v2 as M


class _Resp:
    def __init__(self, content): self.content = content


class _MockLLM:
    def __init__(self, fn): self._fn = fn
    def invoke(self, prompt): return _Resp(self._fn(prompt))


def _patch_llm(obj_name, fn):
    """Replace module-level coder_llm / llm with a mock (ChatOpenAI is a frozen
    pydantic model, so we swap the object rather than its method)."""
    setattr(M, obj_name, _MockLLM(fn))


def test_deanchor_boundaries_and_types():
    """Lever 1: de-anchored ATF must keep dirty boundaries, adopt model types,
    and the type-masked prompt must NOT leak the dirty type."""
    tokens = ["Beijing", "beat", "Arsenal"]
    dirty  = ["B-PER", "O", "B-PER"]          # ATF: both types flipped to PER
    seen_prompts = []

    def coder_fn(prompt):
        seen_prompts.append(prompt)
        # model correctly re-derives: Beijing=LOC, Arsenal=ORG
        return '["B-LOC","O","B-ORG"]'
    _patch_llm("coder_llm", coder_fn)
    M._init_deer("conll2003")

    state = {"tokens": tokens, "dirty_tags": dirty, "dataset_name": "conll2003",
             "noise_type": "ATF", "deanchor_atf": True}
    out = asyncio.run(M.coder_node(state))
    paths = out["candidate_paths"]
    # boundaries (B/I/O structure) preserved from dirty; types from model
    for p in paths:
        assert [t[0] if t != "O" else "O" for t in p] == ["B", "O", "B"], p
    assert all(p == ["B-LOC", "O", "B-ORG"] for p in paths), paths
    # prompt must present masked ENT spans, not the dirty tag list with types
    assert "B-ENT" in seen_prompts[0]
    assert "Dirty IOB2 Tags:" not in seen_prompts[0]
    assert str(dirty) not in seen_prompts[0]   # the ['B-PER','O','B-PER'] repr
    print("[1] de-anchor OK: boundaries kept, types re-derived, no type leak")


def test_verifier_selects_and_falls_back():
    tokens = ["California"]
    dirty = ["B-ORG"]
    # contested: two distinct candidate paths
    cands = [["B-ORG"], ["B-ORG"], ["B-LOC"]]
    weights = [0.3, 0.3, 0.9]
    base_state = {"tokens": tokens, "dirty_tags": dirty, "candidate_paths": cands,
                  "rag_weights": weights, "dataset_name": "conll2003",
                  "use_verifier": True, "verify_all": False, "verifier_topk": 4}
    M._init_deer("conll2003")

    # verifier picks the correct minority path B-LOC
    _patch_llm("llm", lambda p: '["B-LOC"]')
    out = asyncio.run(M.verifier_node(dict(base_state)))
    assert out["current_tags"] == ["B-LOC"], out
    print("[2] verifier selects correct minority path OK")

    # malformed output -> vote fallback (never crashes)
    _patch_llm("llm", lambda p: "garbage not json")
    out = asyncio.run(M.verifier_node(dict(base_state)))
    assert len(out["current_tags"]) == 1, out
    print("[3] verifier fallback-to-vote OK")


def test_verifier_trigger_gating():
    """Uncontested sentence (all paths identical) must NOT call the LLM."""
    called = {"n": 0}
    def spy(p):
        called["n"] += 1
        return '["O"]'
    _patch_llm("llm", spy)
    M._init_deer("conll2003")
    state = {"tokens": ["hello"], "dirty_tags": ["O"],
             "candidate_paths": [["O"], ["O"], ["O"]], "rag_weights": [1, 1, 1],
             "dataset_name": "conll2003", "use_verifier": True,
             "verify_all": False, "verifier_topk": 4}
    out = asyncio.run(M.verifier_node(state))
    assert called["n"] == 0, "verifier fired on an uncontested sentence!"
    assert out["current_tags"] == ["O"], out
    print("[4] verifier trigger gating OK (no LLM call on uncontested)")


def test_noise_aware_legalization():
    """Final legalization must be noise-aware: on IF a residual dangling I- is a
    failed-merge artifact and must be DEMOTED to O (promoting it would invent a
    false-positive entity); on BT/ATF it must be PROMOTED to B- (recover a
    dropped B-). The promote path must stay byte-identical to enforce_iob2_syntax."""
    from utils import enforce_iob2_syntax, legalize_noise_aware
    V = {"PER", "LOC", "ORG"}

    # utils-level: dangling I- handling differs only by policy
    assert legalize_noise_aware(["O", "I-PER"], V, "demote") == ["O", "O"]
    assert legalize_noise_aware(["O", "I-PER"], V, "promote") == ["O", "B-PER"]
    # a LEGAL I- (valid same-type predecessor) is untouched by both policies
    assert legalize_noise_aware(["B-PER", "I-PER"], V, "demote") == ["B-PER", "I-PER"]
    # promote wrapper == historical behavior
    assert legalize_noise_aware(["I-LOC", "O", "I-ORG"], V, "promote") \
        == enforce_iob2_syntax(["I-LOC", "O", "I-ORG"], V)

    # verifier_node-level: same dangling input, opposite policy per noise_type.
    # Driven through the no-candidates fallback, which legalizes dirty_tags
    # directly. (The BT/IF *decode* path can no longer exercise this: since the
    # noise-adaptive change, _base_decode runs a global IOB2-constrained
    # Viterbi there, so it emits legal sequences by construction and never
    # leaves a dangling I- for the policy to fix.)
    _patch_llm("llm", lambda p: (_ for _ in ()).throw(
        AssertionError("verifier must not call LLM here")))
    M._init_deer("conll2003")
    fallback = dict(tokens=["a", "b"], dirty_tags=["O", "I-PER"],
                    candidate_paths=[], rag_weights=[],
                    dataset_name="conll2003", use_verifier=True,
                    verify_all=False, verifier_topk=4)
    out_if = asyncio.run(M.verifier_node(dict(fallback, noise_type="IF")))
    out_bt = asyncio.run(M.verifier_node(dict(fallback, noise_type="BT")))
    assert out_if["current_tags"] == ["O", "O"], out_if       # demote: FP killed
    assert out_bt["current_tags"] == ["O", "B-PER"], out_bt    # promote: recovered

    # The BT/IF global decode is legal by construction (SER=0 with no repair):
    # an all-dangling candidate pool must still yield a legal sequence.
    from metrics import compute_ser
    for nt in ("BT", "IF"):
        out = asyncio.run(M.verifier_node(dict(
            tokens=["Peter", "Smith"], dirty_tags=["O", "I-PER"],
            candidate_paths=[["O", "I-PER"]] * 3, rag_weights=[1, 1, 1],
            dataset_name="conll2003", use_verifier=True, verify_all=False,
            verifier_topk=4, noise_type=nt)))
        assert compute_ser([out["current_tags"]]) == 0.0, (nt, out)
    print("[5] noise-aware legalization OK (IF demote, BT/ATF promote, SER=0)")


if __name__ == "__main__":
    test_deanchor_boundaries_and_types()
    test_verifier_selects_and_falls_back()
    test_verifier_trigger_gating()
    test_noise_aware_legalization()
    print("\nAll SelectDenoise unit tests passed.")
