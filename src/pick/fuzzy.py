"""Subsequence fuzzy matching for brawler names and map names.

Implements the algorithm described in docs/misc_notes.md:
  - subsequence match: all query chars must appear in target in order (case-insensitive)
  - tiebreak: prefer shortest normalized name, then lexicographic order
"""


def _normalize(s: str) -> str:
    """Uppercase and strip non-alphanumeric chars for matching."""
    return "".join(c for c in s.upper() if c.isalpha() or c.isdigit())


def _subseq_match(query: str, target: str) -> bool:
    """Return True if every char of query (already normalized) appears in target in order."""
    ti = 0
    for ch in query:
        while ti < len(target) and target[ti] != ch:
            ti += 1
        if ti >= len(target):
            return False
        ti += 1
    return True


def fuzzy_find(query: str, candidates: list[str]) -> str | None:
    """Return the best candidate matching query as a subsequence.

    Among all matches: prefer shortest normalized name; tiebreak alphabetically.
    Returns None if no candidate matches.
    """
    norm_query = _normalize(query)
    if not norm_query:
        return None
    matches = [c for c in candidates if _subseq_match(norm_query, _normalize(c))]
    if not matches:
        return None
    return min(matches, key=lambda c: (len(_normalize(c)), c))
