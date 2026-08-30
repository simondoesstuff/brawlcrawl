"""Tests for pick.fuzzy — subsequence matching with shortest-name tiebreak."""

import pytest
from pick.fuzzy import fuzzy_find, _normalize, _subseq_match

BRAWLER_NAMES = [
    "COLT", "COLETTE", "SHELLY", "BROCK", "RICO",
    "BULL", "BO", "TARA", "CARL", "EMZ", "LARRY & LAWRIE",
]

MAP_NAMES = [
    "Shooting Star", "Hard Rock Mine", "Dry Season", "Safe Zone",
    "Safe(r) Zone", "Deathcap Trap", "Gem Fort", "Sneaky Fields",
]


# ── normalize ────────────────────────────────────────────────────────────────

def test_normalize_strips_spaces():
    assert _normalize("Hard Rock Mine") == "HARDROCKMINEMINE"[: len("HARDROCKMINE")]
    assert _normalize("Hard Rock Mine") == "HARDROCKMINE"

def test_normalize_strips_ampersand():
    assert _normalize("LARRY & LAWRIE") == "LARRYLAWRIE"

def test_normalize_uppercase():
    assert _normalize("colt") == "COLT"


# ── subseq_match ─────────────────────────────────────────────────────────────

def test_subseq_match_exact():
    assert _subseq_match("COLT", "COLT")

def test_subseq_match_prefix():
    assert _subseq_match("COL", "COLT")

def test_subseq_match_scattered():
    assert _subseq_match("CT", "COLT")

def test_subseq_match_no_match():
    assert not _subseq_match("XY", "COLT")

def test_subseq_match_order_matters():
    assert not _subseq_match("TL", "COLT")  # T comes after L in "COLT"? No: C,O,L,T — T is at end, L before T
    # Actually "TL" requires T then L: in "COLT" T is at index 3, no L after that → no match
    assert not _subseq_match("TL", "COLT")


# ── fuzzy_find ───────────────────────────────────────────────────────────────

def test_fuzzy_exact():
    assert fuzzy_find("COLT", BRAWLER_NAMES) == "COLT"

def test_fuzzy_col_prefers_colt_over_colette():
    """The key tiebreak case from docs/misc_notes.md: COL → COLT (shorter than COLETTE)."""
    result = fuzzy_find("COL", BRAWLER_NAMES)
    assert result == "COLT"

def test_fuzzy_case_insensitive():
    assert fuzzy_find("col", BRAWLER_NAMES) == "COLT"

def test_fuzzy_no_match():
    assert fuzzy_find("XYZ", BRAWLER_NAMES) is None

def test_fuzzy_empty_query():
    assert fuzzy_find("", BRAWLER_NAMES) is None

def test_fuzzy_single_char_bo():
    # "B" matches BO, BROCK, BULL, EMZ? No — only those starting with B via subsequence
    # "B" matches BO (B at 0), BROCK (B at 0), BULL (B at 0), LARRY & LAWRIE (no B)
    # Shortest is "BO" (NORMALIZE="BO" len 2) vs "BULL"(4) vs "BROCK"(5)
    assert fuzzy_find("B", BRAWLER_NAMES) == "BO"

def test_fuzzy_map_abbreviation():
    """hrm → Hard Rock Mine via subsequence H..R..M."""
    assert fuzzy_find("hrm", MAP_NAMES) == "Hard Rock Mine"

def test_fuzzy_map_safe_zone_vs_safer():
    """'sz' matches Safe Zone and Safe(r) Zone; prefers shorter Safe Zone."""
    result = fuzzy_find("sz", MAP_NAMES)
    assert result == "Safe Zone"

def test_fuzzy_map_longer_query_disambiguates():
    """'sfr' adds R which only Safe(r) Zone has after S and F."""
    # SAFERZONE has S,A,F,E,R,Z,O,N,E — sfr = S..F..R ✓ for "SAFERZONE"
    # SAFEZONE = S,A,F,E,Z,O,N,E — no R → no match
    result = fuzzy_find("sfr", MAP_NAMES)
    assert result == "Safe(r) Zone"

def test_fuzzy_larry_and_lawrie():
    """Ampersand is stripped; LARRYLAWRIE is the normalized form."""
    result = fuzzy_find("ll", BRAWLER_NAMES)
    # "LL" must match a name with L..L; LARRYLAWRIE has L at 0 and 5 ✓
    # Also COLT has no two L's; SHELLY? not in BRAWLER_NAMES; COLETTE? no two L's
    # Actually BULL: B,U,L,L → "LL" = L..L matches at positions 2,3 ✓  (len 4 < 11)
    # So "ll" → "BULL" (shorter)
    assert result == "BULL"

def test_fuzzy_larry_specific():
    """'lrl' uniquely resolves to LARRY & LAWRIE."""
    # LARRYLAWRIE: L(0)..R(2)..L(5) → matches
    # BULL: B,U,L,L — only one R? No R at all → no match
    result = fuzzy_find("lrl", BRAWLER_NAMES)
    assert result == "LARRY & LAWRIE"


def test_fuzzy_slash_returns_none():
    """'/' normalizes to empty string and must return None — it's used as the filter toggle."""
    assert fuzzy_find("/", BRAWLER_NAMES) is None


def test_fuzzy_filter_partial_names():
    """Multiple partial names each resolve independently via fuzzy_find."""
    partials = ["sh", "bro", "col"]
    resolved = [fuzzy_find(p, BRAWLER_NAMES) for p in partials]
    assert resolved == ["SHELLY", "BROCK", "COLT"]
