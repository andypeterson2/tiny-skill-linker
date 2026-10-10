"""The reduction the export checks itself with, and the one a browser repeats.

Every published per-query record is replayed through it: candidates rebuilt from the
recorded gold ranks must reduce back to the recorded RP@5. A reduction that disagreed
would let a bad export through, since the export trusts this to refuse one.
"""

import json
from pathlib import Path

import pytest

from tiny_skill_linker.demo import check_top, rp_at_k

ROOT = Path(__file__).resolve().parent.parent
DEPTH = 10


def test_a_single_gold_skill_scores_all_or_nothing():
    assert rp_at_k([7], [7, 1, 2, 3, 4]) == 1.0
    assert rp_at_k([7], [1, 2, 3, 4, 5, 7]) == 0.0


def test_several_gold_skills_score_in_fractions():
    """Two gold skills, one in the top five: half, not a hit."""
    assert rp_at_k([7, 9], [7, 1, 2, 3, 4, 9]) == 0.5


def test_more_gold_skills_than_the_cutoff_divide_by_the_cutoff():
    """Six gold skills cannot all fit in five slots, so the divisor is five."""
    assert rp_at_k([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5]) == 1.0


def perquery_records() -> list[Path]:
    return sorted((ROOT / "results" / "test" / "perquery").glob("*.json"))


CASES = perquery_records()


def test_some_records_are_covered():
    assert CASES, "no results/test/perquery/*.json to replay"


@pytest.mark.parametrize("record", CASES, ids=lambda p: p.stem)
def test_candidates_rebuilt_from_a_published_run_pass_the_check(record: Path):
    payload = json.loads(record.read_text())
    if payload["schema"] < 2:
        pytest.skip(f"{record.name} holds no gold ranks")
    for scores in payload["tasks"].values():
        ranked = []
        for gold, ranks in zip(scores["gold"], scores["gold_ranks"], strict=True):
            # Fill the depth with placeholders no gold skill uses, then seat each
            # gold skill at the rank the published run gave it.
            row = [-1 - i for i in range(DEPTH)]
            for g, rank in zip(gold, ranks, strict=True):
                if rank < DEPTH:
                    row[rank] = g
            ranked.append(row)
        check_top(scores["gold"], ranked, scores)


def test_the_check_rejects_a_misplaced_gold_skill():
    reference = {"gold": [[4]], "gold_ranks": [[0]], "rp@5": [1.0]}
    with pytest.raises(ValueError, match="rank 0"):
        check_top([[4]], [[9, 4, 1, 2, 3]], reference)


def test_the_check_rejects_a_score_the_published_run_disagrees_with():
    """Right candidates, wrong published value: the record and the export disagree."""
    reference = {"gold": [[4]], "gold_ranks": [[0]], "rp@5": [0.5]}
    with pytest.raises(ValueError, match="RP@5"):
        check_top([[4]], [[4, 1, 2, 3, 9]], reference)
