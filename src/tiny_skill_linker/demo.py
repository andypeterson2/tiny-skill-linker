"""Reduce ranked candidates to RP@5, and check them against a published run.

The website's benchmark viewer recomputes RP@5 in the browser from the candidates
it is served. The same reduction lives here so an export is refused unless it
reproduces the per-query numbers already committed under results/.
"""

K = 5


def rp_at_k(gold: list[int], ranked: list[int], k: int = K) -> float:
    """Share of a query's gold skills found in the first k candidates.

    WorkRB's r-precision at k divides by min(k, number of gold skills), so a query
    with one gold skill scores 1.0 or 0.0 and a query with six scores in fifths.
    """
    hits = sum(1 for g in gold if g in ranked[:k])
    return hits / min(k, len(gold))


def check_top(gold: list[list[int]], ranked: list[list[int]], reference: dict) -> None:
    """Raise unless candidates agree with a results/<split>/perquery/ record.

    Indices are the task's own target-space positions, the same space the published
    record uses. Two things must hold: a gold skill the published run placed inside
    the exported depth sits at exactly that position here, and the RP@5 reduced from
    these candidates equals the published per-query value. A separate code path wrote
    those numbers, so an export that disagrees with the published result is refused.
    """
    depth = len(ranked[0])
    for q, (golds, candidates) in enumerate(zip(gold, ranked, strict=True)):
        for g, rank in zip(reference["gold"][q], reference["gold_ranks"][q], strict=True):
            if rank < depth and candidates[rank] != g:
                raise ValueError(
                    f"query {q}: published run has gold {g} at rank {rank}, "
                    f"export has {candidates[rank]}"
                )
        got = rp_at_k(golds, candidates)
        want = reference["rp@5"][q]
        if abs(got - want) > 1e-9:
            raise ValueError(f"query {q}: RP@5 {got} from the export, {want} published")
