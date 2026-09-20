"""The per-query scores must agree with the aggregates WorkRB recorded beside them."""

import json
from pathlib import Path

import numpy as np
import pytest

from tiny_skill_linker.perquery import SCHEMA

ROOT = Path(__file__).resolve().parent.parent
# The aggregate and the mean of the per-query scores agree to floating-point
# noise, or one of the two has changed.
TOLERANCE = 1e-6
METRICS = {"rp@5": "rp@5", "rr": "mrr"}


def results_with_perquery() -> list[tuple[Path, Path]]:
    pairs = []
    for split in ("test", "val"):
        for result in sorted((ROOT / "results" / split).glob("*.json")):
            sibling = result.parent / "perquery" / result.name
            if sibling.exists():
                pairs.append((result, sibling))
    return pairs


CASES = results_with_perquery()


def test_some_results_are_covered():
    assert CASES, "no results/<split>/<label>.json has a per-query sibling"


@pytest.mark.parametrize("result,sibling", CASES, ids=lambda p: p.stem)
def test_perquery_means_match_workrb(result: Path, sibling: Path):
    aggregate = json.loads(result.read_text())["scores"]
    perquery = json.loads(sibling.read_text())["tasks"]
    assert set(perquery) == set(aggregate)
    for task, scores in perquery.items():
        for key, metric in METRICS.items():
            assert np.mean(scores[key]) == pytest.approx(aggregate[task][metric], abs=TOLERANCE)


@pytest.mark.parametrize("result,sibling", CASES, ids=lambda p: p.stem)
def test_perquery_declares_a_schema(result: Path, sibling: Path):
    schema = json.loads(sibling.read_text())["schema"]
    assert 1 <= schema <= SCHEMA


def test_gold_ranks_agree_with_the_hit_counts():
    """Schema 2 carries each gold skill's rank, which must reproduce RP@5."""
    for _, sibling in CASES:
        payload = json.loads(sibling.read_text())
        if payload["schema"] < 2:
            continue
        for scores in payload["tasks"].values():
            for rp, gold, ranks in zip(
                scores["rp@5"], scores["gold"], scores["gold_ranks"], strict=True
            ):
                hits = sum(1 for r in ranks if r < 5)
                assert rp == pytest.approx(hits / min(5, len(gold)), abs=TOLERANCE)
