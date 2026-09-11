import shutil

import datasets
from datasets import Dataset

import multi_agent_v2 as M


def test_cache_only_deer_builds_from_arrow_without_high_level_loader(
    tmp_path, monkeypatch
):
    source = Dataset.from_dict(
        {
            "id": ["0", "1"],
            "tokens": [["Alice", "works"], ["Paris"]],
            "ner_tags": [[1, 0], [5]],
        }
    )
    saved = tmp_path / "saved"
    source.save_to_disk(str(saved))
    source_arrow = next(saved.glob("*.arrow"))
    cache_root = tmp_path / "datasets"
    cached_arrow = (
        cache_root
        / "msra_ner"
        / "msra_ner"
        / "1.0.0"
        / "fixture-hash"
        / "msra_ner-train.arrow"
    )
    cached_arrow.parent.mkdir(parents=True)
    shutil.copy2(source_arrow, cached_arrow)

    def forbid_high_level_loader(*_args, **_kwargs):
        raise AssertionError("cache-only DEER must not call datasets.load_dataset")

    monkeypatch.setenv("HF_DATASETS_CACHE", str(cache_root))
    monkeypatch.setattr(datasets, "load_dataset", forbid_high_level_loader)
    monkeypatch.setattr(M, "_deer_stats", {})
    monkeypatch.setattr(M, "_deer_retriever", {})

    M._init_deer("msra", cache_only=True)

    assert M._deer_stats["msra"].entity_prob("Alice") > 0
    examples = M._deer_retriever["msra"].retrieve(["Alice"], top_k=1)
    assert len(examples) == 1
    assert examples[0][2:] == (["Alice", "works"], ["B-PER", "O"])
