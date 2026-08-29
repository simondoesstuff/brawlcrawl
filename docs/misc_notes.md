### Fast fuzzy unique brawler name lookup:

- 4 characters is the minimum prefix length to uniquely identify all brawler names.
  The only collision at 3 characters is COL → COLT vs COLETTE. Every other name is already unique at 3 chars or fewer.

- Sequence matching + shortest-name fallback
  Eg: matching "CLT" to COLT and COLETTE, but preferring COLT
  - Max input: 3 chars (no 4-char cases at all)
  - Avg: 1.82 chars vs 2.44 for variable prefix (trie-based)
  - All 106 names fully resolvable, no exceptions
  - The only subtlety: the user needs to know which short code reliably resolves to their target. For most names the canonical input is just an obvious 1-2 char fragment. The tricky part is names that win only via tiebreaker — a user typing CO intending COLT has to know that COLT wins, not COLETTE. That's a mental model burden.
    - 6 names (BO, COLT, TARA, CARL, OTIS, MOE) have no unique subsequence without shortest-name fallback
