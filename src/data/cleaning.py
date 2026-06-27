"""
cleans the combined TTC data. input combined_raw.csv, output cleaned.csv.
fixes types and nulls, fixes station names and lines, drops the closed SRT line,
caps outliers, then validates. main function is clean_data().
"""

import difflib
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Constants

DELAY_COL   = "min_delay"
GAP_COL     = "min_gap"
STATION_COL = "station"
LINE_COL    = "line"
CODE_COL    = "code"
BOUND_COL   = "bound"
VEHICLE_COL = "vehicle"

VALID_LINES  = {"YU", "BD", "SRT", "SHP"}
ACTIVE_LINES = {"YU", "BD", "SHP"}          # SRT permanently closed
VALID_BOUNDS = {"N", "S", "E", "W", "B"}

HARD_DELAY_CAP_MINS  = 999   # generous ceiling — raw data max ~900 min; preserves true extreme events

# Tokens that appear in station field noise but are not part of any station name.
# Removed during fragment cleaning before alias lookup or fuzzy matching.
NOISE_TOKENS: set[str] = {"STA", "STN", "SUBWAY", "PLATFORM", "TRACK"}
_NOISE_TOKEN_RE = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in sorted(NOISE_TOKENS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

# Reference: Official TTC Station List  →  Lines
# Canonical station name → list of lines (ordered: primary line first).
# Both current names AND legacy names are present (legacy for alias resolution).
#
# Rename history captured here:
#   "TMU"       = formerly "DUNDAS"         (renamed Oct 2022, Line 1 YU)
#   "CEDARVALE" = formerly "EGLINTON WEST"  (renamed, Line 1 YU)
#   "SHEPPARD WEST" = formerly "DOWNSVIEW"  (renamed 2017, Line 1 YU)

STATION_LINE_MAP: dict[str, list[str]] = {

    "FINCH":                        ["YU"],
    "NORTH YORK CENTRE":            ["YU"],
    "SHEPPARD-YONGE":               ["YU", "SHP"],   # interchange YU ↔ SHP
    "YORK MILLS":                   ["YU"],
    "LAWRENCE":                     ["YU"],           # ≠ LAWRENCE WEST (Univ branch)
    "EGLINTON":                     ["YU"],           # ≠ EGLINTON WEST / CEDARVALE
    "DAVISVILLE":                   ["YU"],
    "ST. CLAIR":                    ["YU"],           # ≠ ST. CLAIR WEST (Univ branch)
    "SUMMERHILL":                   ["YU"],
    "ROSEDALE":                     ["YU"],
    "BLOOR-YONGE":                  ["YU", "BD"],     # interchange YU ↔ BD
    "WELLESLEY":                    ["YU"],
    "COLLEGE":                      ["YU"],
    "TMU":                          ["YU"],           # formerly DUNDAS (YU only!)
    "QUEEN":                        ["YU"],
    "KING":                         ["YU"],
    "UNION":                        ["YU"],

    "ST. ANDREW":                   ["YU"],
    "OSGOODE":                      ["YU"],
    "ST. PATRICK":                  ["YU"],
    "QUEEN'S PARK":                 ["YU"],
    "MUSEUM":                       ["YU"],
    "ST. GEORGE":                   ["YU", "BD"],     # interchange YU ↔ BD
    "SPADINA":                      ["YU", "BD"],     # interchange YU ↔ BD
    "DUPONT":                       ["YU"],
    "ST. CLAIR WEST":               ["YU"],           # ≠ ST. CLAIR (Yonge branch)
    "CEDARVALE":                    ["YU"],           # formerly EGLINTON WEST
    "GLENCAIRN":                    ["YU"],
    "LAWRENCE WEST":                ["YU"],           # ≠ LAWRENCE (Yonge branch)
    "YORKDALE":                     ["YU"],
    "WILSON":                       ["YU"],
    "SHEPPARD WEST":                ["YU"],           # formerly DOWNSVIEW (≠ SHEPPARD-YONGE)
    "DOWNSVIEW PARK":               ["YU"],           # separate station from SHEPPARD WEST
    "FINCH WEST":                   ["YU"],           # ≠ FINCH (Yonge branch)
    "YORK UNIVERSITY":              ["YU"],
    "PIONEER VILLAGE":              ["YU"],
    "HIGHWAY 407":                  ["YU"],
    "VAUGHAN METROPOLITAN CENTRE":  ["YU"],

    "KIPLING":                      ["BD"],
    "ISLINGTON":                    ["BD"],
    "ROYAL YORK":                   ["BD"],
    "OLD MILL":                     ["BD"],
    "JANE":                         ["BD"],
    "RUNNYMEDE":                    ["BD"],
    "HIGH PARK":                    ["BD"],
    "KEELE":                        ["BD"],
    "DUNDAS WEST":                  ["BD"],           # ≠ TMU/DUNDAS on YU!
    "LANSDOWNE":                    ["BD"],
    "DUFFERIN":                     ["BD"],
    "OSSINGTON":                    ["BD"],
    "CHRISTIE":                     ["BD"],
    "BATHURST":                     ["BD"],
    "BAY":                          ["BD"],
    "SHERBOURNE":                   ["BD"],
    "CASTLE FRANK":                 ["BD"],
    "BROADVIEW":                    ["BD"],
    "CHESTER":                      ["BD"],
    "PAPE":                         ["BD"],
    "DONLANDS":                     ["BD"],
    "GREENWOOD":                    ["BD"],
    "COXWELL":                      ["BD"],
    "WOODBINE":                     ["BD"],
    "MAIN STREET":                  ["BD"],
    "VICTORIA PARK":                ["BD"],
    "WARDEN":                       ["BD"],
    "KENNEDY":                      ["BD"],           # SRT closed; Kennedy = BD terminus

    # Kept here only for alias resolution; all SRT rows are dropped later.
    "LAWRENCE EAST":                ["SRT"],
    "ELLESMERE":                    ["SRT"],
    "MIDLAND":                      ["SRT"],
    "SCARBOROUGH CENTRE":           ["SRT"],
    "MCCOWAN":                      ["SRT"],

    "BAYVIEW":                      ["SHP"],
    "BESSARION":                    ["SHP"],
    "LESLIE":                       ["SHP"],
    "DON MILLS":                    ["SHP"],
}

OFFICIAL_STATIONS: frozenset[str] = frozenset(STATION_LINE_MAP.keys())

# Alias Dictionary  →  official station name
# Keys are the normalized (uppercase, stripped) raw variants found in data.
# Context-sensitive aliases (DUNDAS, SHEPPARD) are handled in code below.

STATION_ALIASES: dict[str, str] = {

    "VAUGHAN MC":                      "VAUGHAN METROPOLITAN CENTRE",
    "VAUGHAN METRO CENTRE":            "VAUGHAN METROPOLITAN CENTRE",
    "VAUGHAN METROPOLITAN CENTER":     "VAUGHAN METROPOLITAN CENTRE",
    "VAUGHAN METRO CENTER":            "VAUGHAN METROPOLITAN CENTRE",
    "VMC":                             "VAUGHAN METROPOLITAN CENTRE",

    "HWY 407":                         "HIGHWAY 407",
    "HWY407":                          "HIGHWAY 407",
    "407":                             "HIGHWAY 407",

    "PIONEER":                         "PIONEER VILLAGE",

    "YORK U":                          "YORK UNIVERSITY",
    "YORK UNIV":                       "YORK UNIVERSITY",
    "YORK UNIVERISTY":                 "YORK UNIVERSITY",   # typo in data

    "FINCH W":                         "FINCH WEST",

    "DOWNSVIEW":                       "SHEPPARD WEST",
    "SHEPPARD W":                      "SHEPPARD WEST",
    "SHEPPARD-WEST":                   "SHEPPARD WEST",

    # (usually spelled correctly in data)

    "EGLINTON WEST":                   "CEDARVALE",
    "EGLINTON W":                      "CEDARVALE",
    "EGLINTON WEST STATION":           "CEDARVALE",
    "EGLINTON WEST YU":                "CEDARVALE",

    "ST CLAIR W":                      "ST. CLAIR WEST",
    "ST. CLAIR W":                     "ST. CLAIR WEST",
    "STCLAIR WEST":                    "ST. CLAIR WEST",
    "ST CLAIR WEST":                   "ST. CLAIR WEST",

    "LAWRENCE W":                      "LAWRENCE WEST",

    "LAWRENCE E":                      "LAWRENCE EAST",

    "CALENDONIA":                      "GLENCAIRN",   # common misspelling seen in data
    "CALEDONIA":                       "GLENCAIRN",   # misremembered name

    "ST CLAIR":                        "ST. CLAIR",
    "STCLAIR":                         "ST. CLAIR",

    # (usually fine)

    "ST GEORGE":                       "ST. GEORGE",
    "ST.GEORGE":                       "ST. GEORGE",
    "STGEORGE":                        "ST. GEORGE",

    # (usually fine)

    "QUEENS PARK":                     "QUEEN'S PARK",
    "QUEEN PARK":                      "QUEEN'S PARK",

    "ST PATRICK":                      "ST. PATRICK",
    "ST.PATRICK":                      "ST. PATRICK",
    "STPATRICK":                       "ST. PATRICK",

    # (usually fine)

    "ST ANDREW":                       "ST. ANDREW",
    "ST.ANDREW":                       "ST. ANDREW",
    "STANDREW":                        "ST. ANDREW",
    "ST AND":                          "ST. ANDREW",   # truncated in range entries

    # (usually fine)

    # (usually fine)

    # (usually fine)

    # "DUNDAS" alone is context-sensitive → handled in map_station_aliases()
    "YONGE AND DUNDAS":               "TMU",
    "YONGE/DUNDAS":                   "TMU",
    "YONGE & DUNDAS":                 "TMU",
    "YONGE-DUNDAS":                   "TMU",

    # (usually fine)

    # (usually fine)

    "BLOOR":                           "BLOOR-YONGE",
    "BLOOR YONGE":                     "BLOOR-YONGE",
    "BLOOR / YONGE":                   "BLOOR-YONGE",
    "BLOOR SATION":                    "BLOOR-YONGE",   # typo seen in data
    "YONGE BLOOR":                     "BLOOR-YONGE",
    "YONGE-BLOOR":                     "BLOOR-YONGE",
    "YONGE / BLOOR":                   "BLOOR-YONGE",
    "YONGE AND BLOOR":                 "BLOOR-YONGE",
    "BAY AND BLOOR":                   "BLOOR-YONGE",
    "BLOOR HUB":                       "BLOOR-YONGE",

    # (usually fine)

    # (usually fine)

    # (usually fine)

    "SHEPPARD YONGE":                  "SHEPPARD-YONGE",
    "SHEPPARD / YONGE":                "SHEPPARD-YONGE",
    "SHEPPARD AND YONGE":              "SHEPPARD-YONGE",
    # "SHEPPARD" alone: context-sensitive → handled in map_station_aliases()

    "NORTH YORK CENTER":               "NORTH YORK CENTRE",
    "NY CENTRE":                       "NORTH YORK CENTRE",
    "NORTH YORK CTR":                  "NORTH YORK CENTRE",   # abbreviation seen in data
    "NORTH YORK CENTRE STAT":          "NORTH YORK CENTRE",   # truncated STATION

    # (usually fine)

    # (usually fine on its own; EGLINTON handled separately)

    # (usually fine; already handled above)

    # Line 2 (BD) stations

    # "KENNEDY SRT" stripped to "KENNEDY" in clean_station_names

    # (usually fine)

    "VICTORIA PK":                     "VICTORIA PARK",

    "MAIN ST":                         "MAIN STREET",

    # (usually fine)

    "BRAODVIEW":                       "BROADVIEW",   # typo seen in data
    "BROADVIEW AND DANFORTH":          "BROADVIEW",
    "BROADVIEW AND ST GEORGE":         "BROADVIEW",   # close but this is BROADVIEW station

    "CASTLE":                          "CASTLE FRANK",

    # (usually fine)

    "BAY LOWER":                       "BAY",

    "BATHUST":                         "BATHURST",    # typo seen in data

    "CHRISTIE CENTER":                 "CHRISTIE",
    "CHRISTIE CENTRE":                 "CHRISTIE",

    # (usually fine)

    # (usually fine)

    # (usually fine)

    "DUNDAS W":                        "DUNDAS WEST",
    "APPROACHING DUNDAS WES":          "DUNDAS WEST",

    # (usually fine)

    # (usually fine)

    # (usually fine)

    # (usually fine)

    "APPROACHING OLD MILL":            "OLD MILL",
    "APPROCHING OLD MILL":             "OLD MILL",

    "ROYAL YK":                        "ROYAL YORK",

    # (usually fine)

    # (usually fine)

    # Line 3 SRT (closed) aliases
    "SCARB CENTRE":                    "SCARBOROUGH CENTRE",
    "SCARBOROUGH CTR":                 "SCARBOROUGH CENTRE",
    "SCARBOROUGH CENTER":              "SCARBOROUGH CENTRE",
    "SCARBOROUGH RAPID TRAN":          "SCARBOROUGH CENTRE",   # truncated
    "STC":                             "SCARBOROUGH CENTRE",
    "MCCOWAN YARD":                    "MCCOWAN",
    "MC COWAN":                        "MCCOWAN",

    # Line 4 SHP aliases
    "DONMILLS":                        "DON MILLS",
    "DON MILL":                        "DON MILLS",
    "BAYVIEW SUBSTATION":              "BAYVIEW",

    "YORKDLAE":                        "YORKDALE",    # typo seen in data
    "CELEDONIA":                       "GLENCAIRN",   # misspelling of area near Glencairn
    "CALENDONIA":                      "GLENCAIRN",   # misspelling
}

# Line Code Normalization
# Maps every raw line value variant → canonical code.
# Multi-line values (YU/BD) are resolved after station-name cleaning.

_LINE_NORM: dict[str, str] = {
    # YU variants
    "YU":                "YU",
    "YUS":               "YU",
    "Y/U":               "YU",
    "LINE 1":            "YU",
    "LINE1":             "YU",
    # BD variants
    "BD":                "BD",
    "BDS":               "BD",
    "B/D":               "BD",
    "LINE 2":            "BD",
    "LINE2":             "BD",
    "BD LINE 2":         "BD",
    "BLOOR DANFORTH":    "BD",
    # SRT variants
    "SRT":               "SRT",
    "LINE 3":            "SRT",
    "LINE3":             "SRT",
    # SHP variants
    "SHP":               "SHP",
    "LINE 4":            "SHP",
    "LINE4":             "SHP",
    "SHEP":              "SHP",
}

# Combined-line raw values (e.g., incidents spanning interchange stations)
_COMBINED_LINE_PATTERNS = re.compile(
    r"\b(YU|YUS)\b.*\b(BD|BDS)\b|\b(BD|BDS)\b.*\b(YU|YUS)\b",
    re.IGNORECASE,
)

# Regex helpers

# Matches the word STATION, SATION (typo), or STATIO (truncated at 24-char field limit)
_STATION_WORD_RE = re.compile(r"\b(STATION|SATION|STATIO)\b", re.IGNORECASE)

# Embedded line codes appended to station names, e.g.
#   "ST GEORGE YUS STATION", "SPADINA YUS", "CEDARVALE YU", "KENNEDY SRT"
# Captured group 1 = the line code text
_EMBEDDED_LINE_RE = re.compile(
    r"\s+(YUS|YU|BDS|BD|SRT|SHP|LINE\s*[1-4])(?:\s+|$)",
    re.IGNORECASE,
)

# Parenthetical or bracketed line codes, e.g. "SPADINA (BD)"
_PAREN_LINE_RE = re.compile(
    r"\s*[\(\[]\s*(YUS|YU|BDS|BD|SRT|SHP|LINE\s*[1-4])\s*[\)\]]\s*",
    re.IGNORECASE,
)

# "APPROACHING X", "APPROACHING X STAT..." → extract X
_APPROACHING_RE = re.compile(r"^APPROACHING\s+", re.IGNORECASE)

# Between-stations pattern, e.g. "UNION STATION TO KING"
_RANGE_RE = re.compile(r"\s+TO\s+|\s+B/W\s+|\bBETWEEN\b", re.IGNORECASE)

# Detects keywords that suggest a multi-station entry
_MULTI_STATION_RE = re.compile(
    r"\bTO\b|\bBETWEEN\b|\bB/W\b|\bAND\b|\s[-/]\s",
    re.IGNORECASE,
)

# Parenthetical "X (TO Y)" pattern
_PAREN_TO_RE = re.compile(r"^(.+?)\s*\(TO\s+(.+?)\)?\s*$", re.IGNORECASE)

# Station Order  (linear sequence along each line — used for midpoint logic)
# YU: Yonge branch (NE→Union) then University branch (Union→Vaughan MC)
# BD: West terminus → East terminus
# SHP: West terminus → East terminus

STATION_ORDER: dict[str, list[str]] = {

    "YU": [
            "FINCH",
        "NORTH YORK CENTRE",
        "SHEPPARD-YONGE",
        "YORK MILLS",
        "LAWRENCE",
        "EGLINTON",
        "DAVISVILLE",
        "ST. CLAIR",
        "SUMMERHILL",
        "ROSEDALE",
        "BLOOR-YONGE",
        "WELLESLEY",
        "COLLEGE",
        "TMU",
        "QUEEN",
        "KING",
        "UNION",
            "ST. ANDREW",
        "OSGOODE",
        "ST. PATRICK",
        "QUEEN'S PARK",
        "MUSEUM",
        "ST. GEORGE",
        "SPADINA",
        "DUPONT",
        "ST. CLAIR WEST",
        "CEDARVALE",
        "GLENCAIRN",
        "LAWRENCE WEST",
        "YORKDALE",
        "WILSON",
        "SHEPPARD WEST",
        "DOWNSVIEW PARK",
        "FINCH WEST",
        "YORK UNIVERSITY",
        "PIONEER VILLAGE",
        "HIGHWAY 407",
        "VAUGHAN METROPOLITAN CENTRE",
    ],

    "BD": [
            "KIPLING",
        "ISLINGTON",
        "ROYAL YORK",
        "OLD MILL",
        "JANE",
        "RUNNYMEDE",
        "HIGH PARK",
        "KEELE",
        "DUNDAS WEST",
        "LANSDOWNE",
        "DUFFERIN",
        "OSSINGTON",
        "CHRISTIE",
        "BATHURST",
        "SPADINA",
        "ST. GEORGE",
        "BAY",
        "BLOOR-YONGE",
        "SHERBOURNE",
        "CASTLE FRANK",
        "BROADVIEW",
        "CHESTER",
        "PAPE",
        "DONLANDS",
        "GREENWOOD",
        "COXWELL",
        "WOODBINE",
        "MAIN STREET",
        "VICTORIA PARK",
        "WARDEN",
        "KENNEDY",
    ],

    "SHP": [
            "SHEPPARD-YONGE",
        "BAYVIEW",
        "BESSARION",
        "LESLIE",
        "DON MILLS",
    ],
}

# Delay Classification — boundaries can be overridden by config.preprocessing.class_boundaries
DEFAULT_CLASS_BOUNDARIES: tuple[float, float] = (5.0, 15.0)
DELAY_LABELS: list[str] = ["SHORT", "MEDIUM", "LONG"]
IMBALANCE_THRESHOLD_PCT: float = 5.0


def get_delay_bins(boundaries: tuple[float, float] | list[float] | None = None) -> list[float]:
    """Return bin edges [0, short_bd, long_bd, inf] for pd.cut."""
    if boundaries is None:
        boundaries = DEFAULT_CLASS_BOUNDARIES
    short_bd, long_bd = float(boundaries[0]), float(boundaries[1])
    return [0.0, short_bd, long_bd, float("inf")]


# Internal helpers

def _log_step(step_name: str, before: int, after: int) -> None:
    dropped = before - after
    logger.info(
        "  [%-28s]  rows: %6d → %6d  (dropped %d, %.1f%%)",
        step_name, before, after, dropped, 100 * dropped / max(before, 1),
    )


def _affected_count(mask: pd.Series) -> int:
    return int(mask.sum())


def _normalize_str(s: str) -> str:
    """Uppercase + collapse internal whitespace on a single string."""
    return re.sub(r"\s+", " ", s.strip().upper())


def _extract_line_hint_from_station(station_series: pd.Series) -> pd.Series:
    """
    Extract line code embedded in station name BEFORE stripping it.

    Returns a Series of hints: "YU", "BD", "SRT", "SHP", or None.
    Used to resolve 'YU/BD' line ambiguity at interchange stations.
    """
    def _hint(name: str) -> str | None:
        if pd.isna(name):
            return None
        m = _EMBEDDED_LINE_RE.search(name)
        if m:
            raw = m.group(1).upper().replace(" ", "")
            if "YU" in raw:
                return "YU"
            if "BD" in raw:
                return "BD"
            if "SRT" in raw:
                return "SRT"
            if "SHP" in raw or "LINE4" in raw:
                return "SHP"
        return None

    return station_series.apply(_hint)


# Public cleaning functions

def normalize_text(text: str | pd.Series) -> str | pd.Series:
    #uppercase and squash extra spaces. works on a string or a Series
    if isinstance(text, pd.Series):
        return text.fillna("").str.strip().str.upper().str.replace(
            r"\s+", " ", regex=True
        ).replace("", np.nan)
    return _normalize_str(str(text)) if pd.notna(text) else text


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    #fix types and nulls. numbers to numeric, blank strings to NaN
    n_before = len(df)
    logger.info("handle_missing_values: starting with %d rows", n_before)

    # --- Numeric columns ---
    for col in [DELAY_COL, GAP_COL, VEHICLE_COL]:
        before_nulls = df[col].isna().sum()
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        after_nulls = (df[col] == 0).sum()
        if before_nulls:
            logger.info("    %s: coerced %d non-numeric values -> 0", col, before_nulls)

    # --- Bound: normalize to valid set; unknowns → "UNKNOWN" (categorical) ---
    df[BOUND_COL] = df[BOUND_COL].astype(str).str.strip().str.upper()
    df.loc[~df[BOUND_COL].isin(VALID_BOUNDS), BOUND_COL] = "UNKNOWN"
    n_unknown_bound = _affected_count(df[BOUND_COL] == "UNKNOWN")
    pct_bound = 100 * n_unknown_bound / len(df)
    logger.info("    bound: %d UNKNOWN  (%.1f%% — non-directional incidents)",
                n_unknown_bound, pct_bound)

    # --- Line: blank / "None" → NaN ---
    df[LINE_COL] = df[LINE_COL].astype(str).str.strip()
    df.loc[df[LINE_COL].isin(["", "nan", "None", "NaN"]), LINE_COL] = np.nan

    # --- Code: blank → NaN ---
    df[CODE_COL] = df[CODE_COL].astype(str).str.strip().str.upper()
    df.loc[df[CODE_COL].isin(["", "nan", "None"]), CODE_COL] = np.nan
    null_code = df[CODE_COL].isna().sum()
    if null_code:
        logger.info("    code: %d missing values", null_code)

    logger.info("    rows after handle_missing_values: %d (no rows dropped)", len(df))
    return df


def _normalize_line_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    fix line codes (YUS->YU etc).
    for YU/BD use the station hint, else default YU. garbage -> NaN.
    """
    line = df[LINE_COL].astype(str).str.strip().str.upper()

    def _resolve(row_line: str, hint: str | None) -> str | None:
        v = row_line.strip().upper()

        # 1. Simple lookup
        if v in _LINE_NORM:
            return _LINE_NORM[v]

        # 2. Combined YU+BD → resolve via station hint
        if _COMBINED_LINE_PATTERNS.search(v):
            if hint == "BD":
                return "BD"
            return "YU"   # YUS platform is the default for interchange stations

        # 3. Anything else → NaN (garbage route numbers, free text, etc.)
        return np.nan

    df[LINE_COL] = [
        _resolve(str(l), h)
        for l, h in zip(df[LINE_COL], df.get("_line_hint", [None] * len(df)))
    ]

    non_null_after = df[LINE_COL].notna().sum()
    null_after = df[LINE_COL].isna().sum()
    logger.info(
        "  [normalize_line]  valid: %d  |  to-be-inferred: %d",
        non_null_after, null_after,
    )
    return df


def clean_station_names(df: pd.DataFrame) -> pd.DataFrame:
    #clean the raw station text before lookup (remove STATION, line codes, etc)
    n = len(df)
    s = normalize_text(df[STATION_COL]).fillna("")

    # 1. Strip APPROACHING prefix
    s = s.str.replace(_APPROACHING_RE, "", regex=True)

    # 2. Remove parenthetical line codes
    s = s.str.replace(_PAREN_LINE_RE, " ", regex=True)

    # 3. Remove embedded line codes (YUS, BD, SRT, YU, SHP)
    s = s.str.replace(_EMBEDDED_LINE_RE, " ", regex=True)

    # 4. Remove the word STATION (and SATION typo)
    s = s.str.replace(_STATION_WORD_RE, "", regex=True)

    # 5. Final collapse
    s = s.str.strip().str.replace(r"\s+", " ", regex=True)

    # Empty after cleaning → NaN
    df[STATION_COL] = s.replace("", np.nan)

    affected = _affected_count(df[STATION_COL] != normalize_text(df[STATION_COL]))
    logger.info("  [clean_station_names]  %d station values modified", n)
    return df


def map_station_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """
    map station name variants to the official name.
    DUNDAS and SHEPPARD depend on the line so handled separately.
    """
    before_official = df[STATION_COL].isin(OFFICIAL_STATIONS).sum()

    # 1. Context-free alias lookup
    df[STATION_COL] = df[STATION_COL].map(
        lambda s: STATION_ALIASES.get(s, s) if pd.notna(s) else s
    )

    # 2. Context-sensitive: DUNDAS
    is_dundas = df[STATION_COL] == "DUNDAS"
    bd_line   = df[LINE_COL] == "BD"
    df.loc[is_dundas &  bd_line, STATION_COL] = "DUNDAS WEST"
    df.loc[is_dundas & ~bd_line, STATION_COL] = "TMU"
    dundas_mapped = _affected_count(is_dundas)
    if dundas_mapped:
        to_tmu  = _affected_count(is_dundas & ~bd_line)
        to_dw   = _affected_count(is_dundas &  bd_line)
        logger.info("    DUNDAS context split: %d -> TMU (YU),  %d -> DUNDAS WEST (BD)",
                    to_tmu, to_dw)

    # 3. Context-sensitive: SHEPPARD alone
    is_sheppard = df[STATION_COL] == "SHEPPARD"
    df.loc[is_sheppard, STATION_COL] = "SHEPPARD-YONGE"
    if _affected_count(is_sheppard):
        logger.info("    SHEPPARD -> SHEPPARD-YONGE  (%d rows)", _affected_count(is_sheppard))

    # 4. YONGE → BLOOR-YONGE (domain knowledge: operators use "YONGE" as
    #    shorthand for the BD/YU interchange station at Bloor-Yonge).
    #    Applied before fuzzy matching so it doesn't compete with partial matches.
    is_yonge = df[STATION_COL] == "YONGE"
    n_yonge = _affected_count(is_yonge)
    if n_yonge:
        df.loc[is_yonge, STATION_COL] = "BLOOR-YONGE"
        logger.info("    YONGE -> BLOOR-YONGE  (%d rows)", n_yonge)

    after_official = df[STATION_COL].isin(OFFICIAL_STATIONS).sum()
    logger.info(
        "  [map_station_aliases]  official stations: %d -> %d  (gained %d)",
        before_official, after_official, after_official - before_official,
    )
    return df


def infer_line_from_station(df: pd.DataFrame) -> pd.DataFrame:
    #fill missing line from the station->line map. UNKNOWN if not found
    needs_line = df[LINE_COL].isna() | ~df[LINE_COL].isin(VALID_LINES)
    n_needs = _affected_count(needs_line)
    logger.info("  [infer_line_from_station]  %d rows need line inference", n_needs)

    inferred = 0
    unknown  = 0

    for idx in df.index[needs_line]:
        station = df.at[idx, STATION_COL]
        if pd.isna(station):
            df.at[idx, LINE_COL] = "UNKNOWN"
            unknown += 1
            continue
        lines = STATION_LINE_MAP.get(station)
        if lines:
            df.at[idx, LINE_COL] = lines[0]
            inferred += 1
        else:
            df.at[idx, LINE_COL] = "UNKNOWN"
            unknown += 1

    pct_unknown = 100 * unknown / max(len(df), 1)
    logger.info(
        "    inferred: %d  |  remaining UNKNOWN: %d  (%.1f%% of total)",
        inferred, unknown, pct_unknown,
    )
    return df


def _drop_srt_rows(df: pd.DataFrame) -> pd.DataFrame:
    #drop Line 3 (SRT) rows, it closed in 2023
    srt_stations = {s for s, lines in STATION_LINE_MAP.items() if lines == ["SRT"]}
    srt_mask = (df[LINE_COL] == "SRT") | df[STATION_COL].isin(srt_stations)
    n_before = len(df)
    df = df[~srt_mask].copy()
    logger.info(
        "  [drop_srt_rows]  removed %d SRT rows (%d remaining)",
        n_before - len(df), len(df),
    )
    return df


def clean_delay_codes(df: pd.DataFrame) -> pd.DataFrame:
    #tidy the delay code, flag missing ones
    df[CODE_COL] = df[CODE_COL].astype(str).str.strip().str.upper()
    df.loc[df[CODE_COL].isin(["", "NAN", "NONE"]), CODE_COL] = np.nan

    df["code_missing"] = df[CODE_COL].isna().astype(int)
    n_missing = int(df["code_missing"].sum())
    n_unique  = df[CODE_COL].nunique()

    logger.info(
        "  [clean_delay_codes]  unique codes: %d  |  missing: %d rows",
        n_unique, n_missing,
    )
    return df


def cap_outliers(
    df: pd.DataFrame,
    col: str = DELAY_COL,
    hard_cap: int = HARD_DELAY_CAP_MINS,
) -> pd.DataFrame:
    #clip very large delays to a hard ceiling. keeps the row, just caps value
    n_affected = _affected_count(df[col] > hard_cap)
    df[col] = df[col].clip(upper=hard_cap)

    logger.info(
        "  [cap_outliers]  hard cap = %d min  |  %d rows clipped  "
        "|  max after cap = %d min",
        hard_cap, n_affected, int(df[col].max()),
    )
    return df


# Validation

def validate(df: pd.DataFrame) -> dict:
    """
    checks after cleaning: lines, stations, station-line match, numbers.
    returns a small report dict and logs it.
    """
    sep = "=" * 65
    logger.info(sep)
    logger.info("VALIDATION REPORT")
    logger.info(sep)

    report: dict = {}

    allowed_lines = ACTIVE_LINES | {"UNKNOWN"}
    bad_lines = df[~df[LINE_COL].isin(allowed_lines)][LINE_COL].value_counts()
    unknown_line_pct = 100 * _affected_count(df[LINE_COL] == "UNKNOWN") / len(df)
    report["unknown_line_pct"]    = round(unknown_line_pct, 2)
    report["unexpected_lines"]    = bad_lines.to_dict()
    logger.info("Line check:")
    logger.info("  Distribution : %s", df[LINE_COL].value_counts().to_dict())
    logger.info("  UNKNOWN      : %.1f%%", unknown_line_pct)
    if not bad_lines.empty:
        logger.warning("  Unexpected line values: %s", bad_lines.to_dict())

    unknown_station_mask = ~df[STATION_COL].isin(OFFICIAL_STATIONS)
    n_unknown_station    = _affected_count(unknown_station_mask)
    unknown_station_pct  = 100 * n_unknown_station / len(df)
    top_unknown = (
        df.loc[unknown_station_mask, STATION_COL]
        .value_counts().head(20).to_dict()
    )
    report["unknown_station_pct"] = round(unknown_station_pct, 2)
    report["top_unknown_stations"] = top_unknown
    logger.info("Station check:")
    logger.info("  Valid official stations : %d%%",
                round(100 - unknown_station_pct))
    logger.info("  Unknown (%.1f%%)  top values:", unknown_station_pct)
    for name, cnt in list(top_unknown.items())[:10]:
        logger.info("    %-40s  %d rows", repr(name), cnt)

    mismatch_rows = []
    for _, row in df[df[STATION_COL].isin(OFFICIAL_STATIONS)].iterrows():
        allowed = STATION_LINE_MAP.get(row[STATION_COL], [])
        if allowed and row[LINE_COL] not in allowed and row[LINE_COL] != "UNKNOWN":
            mismatch_rows.append(
                f"{row[STATION_COL]}  ->  line={row[LINE_COL]}  (expected one of {allowed})"
            )
    report["station_line_mismatches"] = len(mismatch_rows)
    logger.info("Station-line consistency:")
    logger.info("  Mismatches: %d", len(mismatch_rows))
    for m in mismatch_rows[:10]:
        logger.warning("    %s", m)

    allowed_bounds = VALID_BOUNDS | {"UNKNOWN"}
    unexpected_bound = set(df[BOUND_COL].dropna().unique()) - allowed_bounds
    report["unexpected_bounds"] = list(unexpected_bound)
    if unexpected_bound:
        logger.warning("  Unexpected bound values: %s", unexpected_bound)
    else:
        logger.info("Bound check: PASSED")

    neg_delay = _affected_count(df[DELAY_COL] < 0)
    neg_gap   = _affected_count(df[GAP_COL] < 0)
    report["negative_delay_rows"] = neg_delay
    report["negative_gap_rows"]   = neg_gap
    logger.info("Numeric check:")
    logger.info("  min_delay  — min: %d  max: %d  mean: %.1f",
                df[DELAY_COL].min(), df[DELAY_COL].max(), df[DELAY_COL].mean())
    logger.info("  Negative delay rows: %d  |  Negative gap rows: %d",
                neg_delay, neg_gap)

    logger.info(sep)
    logger.info("Final shape: %d rows × %d columns", *df.shape)
    logger.info(sep)

    return report


# Multi-station helpers

def remove_noise_tokens(text: str) -> str:
    #remove noise words (STA, STN, SUBWAY...) from a station fragment
    cleaned = re.sub(_NOISE_TOKEN_RE, " ", text)
    return re.sub(r"\s+", " ", cleaned).strip()


def improved_fuzzy_match(text: str, cutoff: float = 0.80) -> str | None:
    """
    fuzzy match to a station name using difflib.
    for short strings (<6 chars) use prefix match instead. None if nothing fits.
    """
    if len(text) < 4:
        return None

    # Short strings: prefer the reliable prefix check
    if len(text) < 6:
        prefix_candidates = [s for s in OFFICIAL_STATIONS if s.startswith(text)]
        return prefix_candidates[0] if len(prefix_candidates) == 1 else None

    # difflib.get_close_matches returns best matches in descending order
    matches = difflib.get_close_matches(text, OFFICIAL_STATIONS, n=1, cutoff=cutoff)
    if matches:
        return matches[0]

    # Second pass: try prefix match as a safety net for truncated names
    prefix_candidates = [s for s in OFFICIAL_STATIONS if s.startswith(text)]
    return prefix_candidates[0] if len(prefix_candidates) == 1 else None


def recover_station_name(text: str, line: str | None = None) -> str | None:
    """
    try hard to recover a station name.
    exact match, then alias, then fuzzy, then substring. None if all fail.
    """
    if not text or pd.isna(text):
        return None

    s = str(text).strip().upper()
    s = remove_noise_tokens(s)
    s = re.sub(r"\s+", " ", s).strip()

    if not s:
        return None

    # Stage 1: exact match
    if s in OFFICIAL_STATIONS:
        return s

    # Stage 2: alias lookup
    resolved = STATION_ALIASES.get(s)
    if resolved and resolved in OFFICIAL_STATIONS:
        return resolved

    # Stage 3: fuzzy match
    fuzzy = improved_fuzzy_match(s)
    if fuzzy:
        return fuzzy

    # Stage 4: substring containment
    # Build candidate set (optionally line-filtered)
    if line and line in STATION_LINE_MAP:
        # STATION_LINE_MAP keys are the official names; filter to those on this line
        candidates = [
            st for st, lines in STATION_LINE_MAP.items()
            if line in lines and st in OFFICIAL_STATIONS
        ]
    else:
        candidates = list(OFFICIAL_STATIONS)

    substring_hits = [c for c in candidates if c in s or s in c]
    if len(substring_hits) == 1:
        return substring_hits[0]

    return None


def _fuzzy_match_station(text: str, min_len: int = 4) -> str | None:
    #old prefix-only match. use improved_fuzzy_match instead
    if len(text) < min_len:
        return None
    candidates = [s for s in OFFICIAL_STATIONS if s.startswith(text)]
    return candidates[0] if len(candidates) == 1 else None


def _clean_station_fragment(text: str) -> str | None:
    #clean one station fragment and resolve to an official name, or None
    if not text or pd.isna(text):
        return None
    s = str(text).strip().upper()

    # Step 1: remove noise tokens BEFORE any other processing
    s = remove_noise_tokens(s)

    # Step 2: apply same transforms as clean_station_names
    s = re.sub(_PAREN_LINE_RE,    " ", s)
    s = re.sub(_EMBEDDED_LINE_RE, " ", s)
    s = re.sub(_STATION_WORD_RE,  "",  s)
    s = re.sub(r"\s+", " ", s).strip()

    if not s:
        return None

    # Step 3: alias lookup
    s = STATION_ALIASES.get(s, s)

    # Step 4: context-sensitive singles (same logic as map_station_aliases)
    if s == "DUNDAS":
        s = "TMU"
    if s == "SHEPPARD":
        s = "SHEPPARD-YONGE"

    # Step 5: exact match
    if s in OFFICIAL_STATIONS:
        return s

    # Step 6: improved fuzzy match (difflib-based, 80% threshold)
    match = improved_fuzzy_match(s)
    if match:
        return match

    # Step 7: substring containment — single unambiguous hit only.
    # Avoids assigning a station when the fragment matches multiple names.
    # Minimum length guard (>= 4) prevents short tokens from over-matching.
    hits = [
        c for c in OFFICIAL_STATIONS
        if (len(c) >= 4 and c in s) or (len(s) >= 4 and s in c)
    ]
    return hits[0] if len(hits) == 1 else None


# Multi-station public functions

def detect_multi_station(df: pd.DataFrame) -> pd.DataFrame:
    #flag rows that look like two stations e.g. "UNION TO KING"
    non_official_mask = ~df[STATION_COL].isin(OFFICIAL_STATIONS)
    keyword_mask      = df[STATION_COL].fillna("").str.contains(
        _MULTI_STATION_RE, regex=True
    )
    df["is_multi_station"] = (non_official_mask & keyword_mask).astype(int)

    n_flagged = int(df["is_multi_station"].sum())
    logger.info(
        "  [detect_multi_station]  %d rows flagged as multi-station (%.1f%%)",
        n_flagged, 100 * n_flagged / max(len(df), 1),
    )
    return df


def extract_station_pair(text: str) -> tuple[str | None, str | None]:
    """
    split a "X to Y" type entry into two stations.
    cleans first, then splits on TO/AND/BETWEEN/etc. returns (s1, s2).
    """
    if not text or pd.isna(text):
        return None, None

    s = str(text).strip().upper()

    # Step 0: pre-clean the full string so noise tokens don't confuse the split
    s = remove_noise_tokens(s)
    s = re.sub(_PAREN_LINE_RE,    " ", s)
    s = re.sub(_EMBEDDED_LINE_RE, " ", s)
    s = re.sub(_STATION_WORD_RE,  "",  s)
    s = re.sub(r"\s+", " ", s).strip()

    # 1. "BETWEEN X AND Y"  — strip the leading keyword before matching
    s_stripped = re.sub(r"^BETWEEN\s+", "", s)

    # 2. "X (TO Y)"
    m = _PAREN_TO_RE.match(s)
    if m:
        return _clean_station_fragment(m.group(1)), _clean_station_fragment(m.group(2))

    # 3. "X TO Y" (highest priority after BETWEEN)
    if " TO " in s_stripped:
        left, right = s_stripped.split(" TO ", 1)
        return _clean_station_fragment(left), _clean_station_fragment(right)

    # 4. "X AND Y"
    if " AND " in s_stripped:
        left, right = s_stripped.split(" AND ", 1)
        return _clean_station_fragment(left), _clean_station_fragment(right)

    # 5. "X / Y"
    if "/" in s_stripped:
        parts = re.split(r"\s*/\s*", s_stripped, maxsplit=1)
        if len(parts) == 2:
            return _clean_station_fragment(parts[0]), _clean_station_fragment(parts[1])

    # 6. "X - Y" (spaces required — avoids splitting hyphenated names)
    if " - " in s_stripped:
        left, right = s_stripped.split(" - ", 1)
        return _clean_station_fragment(left), _clean_station_fragment(right)

    return None, None


def get_midpoint_station(
    s1: str | None,
    s2: str | None,
    line: str,
) -> tuple[str, str]:
    """
    pick one station for a range. midpoint if both are on the line,
    else use whichever we found. returns (station, how_we_got_it).
    """
    primary_order = STATION_ORDER.get(line, [])

    i1 = primary_order.index(s1) if s1 in primary_order else -1
    i2 = primary_order.index(s2) if s2 in primary_order else -1

    # Case 1: both on declared line
    if i1 >= 0 and i2 >= 0:
        return primary_order[(i1 + i2) // 2], "midpoint"

    # Case 2: one found on declared line
    if i1 >= 0:
        return s1, "fallback"
    if i2 >= 0:
        return s2, "fallback"

    # Case 3: try every other line
    for other_line, order in STATION_ORDER.items():
        j1 = order.index(s1) if s1 in order else -1
        j2 = order.index(s2) if s2 in order else -1
        if j1 >= 0 and j2 >= 0:
            return order[(j1 + j2) // 2], "cross_line_midpoint"

    # Case 4: at least one is a valid station.
    # If both are valid and both appear on some shared line, prefer the station
    # whose index is closer to that line's center (avoids arbitrary s1 bias).
    s1_valid = s1 is not None and s1 in OFFICIAL_STATIONS
    s2_valid = s2 is not None and s2 in OFFICIAL_STATIONS

    if s1_valid and s2_valid:
        for _, order in STATION_ORDER.items():
            center = len(order) // 2
            p1 = order.index(s1) if s1 in order else None
            p2 = order.index(s2) if s2 in order else None
            if p1 is not None and p2 is not None:
                chosen = s1 if abs(p1 - center) <= abs(p2 - center) else s2
                return chosen, "center_fallback"
        return s1, "fallback"  # both valid but on different lines — pick s1

    if s1_valid:
        return s1, "fallback"
    if s2_valid:
        return s2, "fallback"

    # Case 5: neither resolved
    return "UNKNOWN", "unknown"


def resolve_multi_station_entries(df: pd.DataFrame) -> pd.DataFrame:
    #replace "X to Y" entries with one station (the midpoint)
    df = detect_multi_station(df)

    multi_idx = df.index[df["is_multi_station"] == 1].tolist()
    n_total = len(multi_idx)

    if n_total == 0:
        logger.info("  [resolve_multi_station]  No multi-station rows to process.")
        return df

    counters = {"midpoint": 0, "cross_line_midpoint": 0, "fallback": 0, "unknown": 0}
    examples: list[tuple[str, str, str]] = []

    for idx in multi_idx:
        original = df.at[idx, STATION_COL]
        line     = df.at[idx, LINE_COL]

        s1, s2 = extract_station_pair(str(original))
        resolved, method = get_midpoint_station(s1, s2, str(line))

        df.at[idx, STATION_COL] = resolved
        counters[method] = counters.get(method, 0) + 1

        if len(examples) < 12:
            examples.append((str(original), resolved, method))

    logger.info(
        "  [resolve_multi_station]  Total: %d  |  midpoint: %d  |  "
        "cross-line: %d  |  center-fallback: %d  |  fallback: %d  |  unknown: %d",
        n_total,
        counters.get("midpoint", 0),
        counters.get("cross_line_midpoint", 0),
        counters.get("center_fallback", 0),
        counters.get("fallback", 0),
        counters.get("unknown", 0),
    )
    logger.info("  Examples  (original -> resolved  [method]):")
    for orig, res, meth in examples:
        logger.info("    %-48s -> %-28s [%s]", repr(orig), repr(res), meth)

    return df


# Post-resolution station recovery

def recover_unknown_stations(df: pd.DataFrame) -> pd.DataFrame:
    #second try at rows still UNKNOWN, using the line as a hint
    unknown_before = int((df[STATION_COL] == "UNKNOWN").sum())
    if unknown_before == 0:
        logger.info("  [recover_unknown_stations]  No UNKNOWN stations to recover.")
        return df

    logger.info(
        "  [recover_unknown_stations]  UNKNOWN before recovery: %d (%.1f%%)",
        unknown_before, 100 * unknown_before / max(len(df), 1),
    )

    unknown_mask = df[STATION_COL] == "UNKNOWN"
    recovered = 0

    # We need the original station text — it has been replaced with 'UNKNOWN'
    # so we cannot use it.  Recovery here operates on any surviving non-UNKNOWN
    # rows that still failed alias/exact match — not post-replacement UNKNOWNs.
    # For true UNKNOWNs we can only try line-scoped substring containment.
    for idx in df.index[unknown_mask]:
        line = df.at[idx, LINE_COL]
        # Attempt recovery using only the line hint and an empty string
        # (the original text was already overwritten).  This will only succeed
        # via substring containment in Stage 4 when line is known, which is
        # expected to have zero hits — but the hook is here for future use.
        result = recover_station_name("UNKNOWN", line=str(line) if pd.notna(line) else None)
        if result and result != "UNKNOWN":
            df.at[idx, STATION_COL] = result
            recovered += 1

    unknown_after = int((df[STATION_COL] == "UNKNOWN").sum())
    logger.info(
        "  [recover_unknown_stations]  Recovered: %d  |  UNKNOWN after: %d (%.1f%%)",
        recovered, unknown_after, 100 * unknown_after / max(len(df), 1),
    )

    return df


def recover_non_official_stations(df: pd.DataFrame) -> pd.DataFrame:
    #try to recover stations that are not official yet (and not UNKNOWN)
    non_official_mask = ~df[STATION_COL].isin(OFFICIAL_STATIONS) & (df[STATION_COL] != "UNKNOWN")
    n_before = int(non_official_mask.sum())

    if n_before == 0:
        logger.info("  [recover_non_official_stations]  All stations are official — nothing to recover.")
        return df

    logger.info(
        "  [recover_non_official_stations]  Non-official (excl. UNKNOWN) before recovery: %d (%.1f%%)",
        n_before, 100 * n_before / max(len(df), 1),
    )

    recovered = 0
    for idx in df.index[non_official_mask]:
        raw = df.at[idx, STATION_COL]
        line = df.at[idx, LINE_COL]
        result = recover_station_name(
            str(raw), line=str(line) if pd.notna(line) else None
        )
        if result:
            df.at[idx, STATION_COL] = result
            recovered += 1

    # Re-compute after recovery
    still_non_official = ~df[STATION_COL].isin(OFFICIAL_STATIONS) & (df[STATION_COL] != "UNKNOWN")
    n_after = int(still_non_official.sum())

    logger.info(
        "  [recover_non_official_stations]  Recovered: %d  |  still non-official: %d (%.1f%%)",
        recovered, n_after, 100 * n_after / max(len(df), 1),
    )

    # Top-10 unresolved strings
    if n_after > 0:
        top_unresolved = (
            df.loc[still_non_official, STATION_COL]
            .value_counts()
            .head(10)
        )
        logger.info("  Top 10 unresolved station strings:")
        for val, cnt in top_unresolved.items():
            logger.info("    %-40s  %d rows", repr(val), cnt)

    return df


# Station-line mismatch correction

def correct_station_line_mismatches(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    if a station only sits on one line, fix a wrong line code.
    skip multi-line stations like BLOOR-YONGE (either line is ok).
    """
    corrections = 0

    for idx in df.index:
        station = df.at[idx, STATION_COL]
        if station not in OFFICIAL_STATIONS:
            continue
        expected = STATION_LINE_MAP.get(station, [])
        # Skip multi-line stations — either line is valid
        if len(expected) != 1:
            continue
        correct_line = expected[0]
        # Never correct to a non-active line (e.g. SRT rows already dropped)
        if correct_line not in ACTIVE_LINES:
            continue
        current_line = df.at[idx, LINE_COL]
        if current_line != correct_line:
            df.at[idx, LINE_COL] = correct_line
            corrections += 1

    logger.info(
        "  [correct_station_line_mismatches]  %d line values corrected",
        corrections,
    )
    return df, corrections


def nullify_mismatch_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    station on a line it does not serve -> set to UNKNOWN.
    handles the cases the fix function skips. run before rename_lines.
    """
    n_fixed = 0
    examples: list[tuple[str, str, list[str]]] = []

    for idx in df.index:
        station = df.at[idx, STATION_COL]
        if station not in OFFICIAL_STATIONS:
            continue
        expected_lines = STATION_LINE_MAP.get(station, [])
        if not expected_lines:
            continue
        current_line = df.at[idx, LINE_COL]
        if current_line not in expected_lines and current_line != "UNKNOWN":
            if len(examples) < 10:
                examples.append((station, str(current_line), expected_lines))
            df.at[idx, STATION_COL] = "UNKNOWN"
            df.at[idx, "station_validity"] = "UNKNOWN"
            n_fixed += 1

    logger.info(
        "  [nullify_mismatch_rows]  %d mismatch row(s) nullified (station -> UNKNOWN)",
        n_fixed,
    )
    for station, line, expected in examples:
        logger.info(
            "    %-28s  line=%-6s  (expected %s)  ->  UNKNOWN",
            station, line, expected,
        )
    return df, n_fixed


# Final station standardization

# Substrings that identify system-level descriptions, not real stations.
# Applied only to strings that are already NOT in OFFICIAL_STATIONS.
_INVALID_STATION_KEYWORDS: tuple[str, ...] = (
    "LINE",
    "SUBWAY",
    "TRANSIT",
    "UNIVERSITY",   # catches YONGE UNIVERSITY LINE etc.; YORK UNIVERSITY is official
)
_INVALID_STATION_MAX_LEN = 25  # strings longer than this with no match → UNKNOWN


def finalize_stations(
    df: pd.DataFrame,
    analysis_dir: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    last station pass. anything still not official -> UNKNOWN.
    adds a station_validity column and saves summary csvs. no rows dropped.
    """
    # Exclude rows already labelled UNKNOWN so the log shows the actual source strings.
    non_official_mask = ~df[STATION_COL].isin(OFFICIAL_STATIONS)
    top_sources = (
        df.loc[non_official_mask & (df[STATION_COL] != "UNKNOWN"), STATION_COL]
        .value_counts()
        .head(10)
    )

    n_before = int(non_official_mask.sum())
    logger.info(
        "  [finalize_stations]  Non-official before finalization: %d (%.1f%%)",
        n_before, 100 * n_before / max(len(df), 1),
    )
    logger.info("  Top 10 non-official sources (before replacement):")
    for val, cnt in top_sources.items():
        logger.info("    %-40s  %d rows", repr(val), cnt)

    # Only touches rows that are already non-official; official names are immune.
    def _is_invalid_pattern(s: str) -> bool:
        if s in OFFICIAL_STATIONS or s == "UNKNOWN":
            return False
        upper = s.upper()
        for kw in _INVALID_STATION_KEYWORDS:
            if kw in upper:
                return True
        if len(s) > _INVALID_STATION_MAX_LEN:
            return True
        return False

    pattern_mask = df[STATION_COL].apply(_is_invalid_pattern)
    n_pattern = int(pattern_mask.sum())
    if n_pattern:
        df.loc[pattern_mask, STATION_COL] = "UNKNOWN"
        logger.info(
            "  [finalize_stations]  Invalid-pattern rows set to UNKNOWN: %d", n_pattern
        )

    still_non_official = ~df[STATION_COL].isin(OFFICIAL_STATIONS) & (df[STATION_COL] != "UNKNOWN")
    n_residual = int(still_non_official.sum())
    if n_residual:
        df.loc[still_non_official, STATION_COL] = "UNKNOWN"
        logger.info(
            "  [finalize_stations]  Residual non-official rows set to UNKNOWN: %d",
            n_residual,
        )

    df["station_validity"] = df[STATION_COL].apply(
        lambda s: "VALID" if s in OFFICIAL_STATIONS else "UNKNOWN"
    )

    n_unknown_final = int((df["station_validity"] == "UNKNOWN").sum())
    n_valid_final   = int((df["station_validity"] == "VALID").sum())
    pct_unknown     = 100 * n_unknown_final / max(len(df), 1)

    logger.info(
        "  [finalize_stations]  VALID: %d  |  UNKNOWN: %d (%.1f%%)",
        n_valid_final, n_unknown_final, pct_unknown,
    )

    if analysis_dir:
        adir = Path(analysis_dir)
        adir.mkdir(parents=True, exist_ok=True)

        # station_summary.csv — one row per unique station value
        summary_df = (
            df.groupby([STATION_COL, "station_validity"], observed=True)
            .size()
            .reset_index(name="count")
            .rename(columns={STATION_COL: "station", "station_validity": "validity"})
            .sort_values("count", ascending=False)
        )
        summary_path = adir / "station_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info("  Saved station summary -> %s  (%d unique values)", summary_path, len(summary_df))

        # final_station_list.csv — all unique official station names after cleaning
        unique_stations = (
            df[[STATION_COL, "station_validity"]]
            .drop_duplicates()
            .rename(columns={STATION_COL: "station", "station_validity": "validity"})
            .sort_values(["validity", "station"])
        )
        list_path = adir / "final_station_list.csv"
        unique_stations.to_csv(list_path, index=False)
        logger.info(
            "  Saved final station list -> %s  (%d unique stations)",
            list_path, len(unique_stations),
        )

    summary = {
        "n_non_official_before": n_before,
        "n_pattern_replaced":    n_pattern,
        "n_residual_replaced":   n_residual,
        "n_unknown_final":       n_unknown_final,
        "pct_unknown_final":     round(pct_unknown, 2),
        "top_sources":           top_sources.to_dict(),
    }
    return df, summary


# Line code standardization

# Maps internal line codes → human-readable line identifiers for final output.
_LINE_RENAME_MAP: dict[str, str] = {
    "YU":  "LINE_1",
    "BD":  "LINE_2",
    "SHP": "LINE_4",
}


def rename_lines(df: pd.DataFrame) -> pd.DataFrame:
    #rename YU/BD/SHP to LINE_1/LINE_2/LINE_4 for the output
    before = df[LINE_COL].value_counts().to_dict()
    df[LINE_COL] = df[LINE_COL].map(lambda v: _LINE_RENAME_MAP.get(str(v), v) if pd.notna(v) else v)
    after = df[LINE_COL].value_counts().to_dict()

    logger.info("  [rename_lines]  Line codes renamed:")
    for old, new in _LINE_RENAME_MAP.items():
        logger.info("    %-6s -> %-8s  (%d rows)", old, new, before.get(old, 0))
    logger.info("  Line distribution after rename: %s", after)

    return df


# Delay classification

def classify_delay(
    df: pd.DataFrame,
    output_dir: str | None = None,
    boundaries: tuple[float, float] | list[float] | None = None,
) -> pd.DataFrame:
    """
    bin min_delay into SHORT/MEDIUM/LONG (adds delay_category).
    not used by the range model but kept for the old setup. boundaries default if none.
    """
    bins = get_delay_bins(boundaries)
    logger.info("  [classify_delay]  Using bins: %s", bins)
    df["delay_category"] = pd.cut(
        df[DELAY_COL],
        bins=bins,
        labels=DELAY_LABELS,
        right=True,
        include_lowest=True,
    )

    counts = (
        df["delay_category"]
        .value_counts()
        .reindex(DELAY_LABELS, fill_value=0)
    )
    pcts = 100.0 * counts / max(len(df), 1)

    imbalanced: list[str] = []
    logger.info("  [classify_delay]  Distribution:")
    for label in DELAY_LABELS:
        cnt = int(counts[label])
        pct = float(pcts[label])
        flag = "  [!]  IMBALANCED (<5%)" if pct < IMBALANCE_THRESHOLD_PCT else ""
        logger.info("    %-20s  %6d rows  (%5.1f%%)%s", label, cnt, pct, flag)
        if pct < IMBALANCE_THRESHOLD_PCT:
            imbalanced.append(label)

    if imbalanced:
        logger.warning(
            "  %d class(es) below %.0f%% threshold: %s",
            len(imbalanced), IMBALANCE_THRESHOLD_PCT, imbalanced,
        )
    else:
        logger.info("  No class imbalance detected.")

    # Save distribution CSV
    if output_dir:
        out_path = Path(output_dir) / "delay_category_distribution.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        dist_df = pd.DataFrame({
            "delay_category": DELAY_LABELS,
            "count":          [int(counts[l])         for l in DELAY_LABELS],
            "percentage":     [round(float(pcts[l]), 2) for l in DELAY_LABELS],
            "below_5pct_flag":[l in imbalanced         for l in DELAY_LABELS],
        })
        dist_df.to_csv(out_path, index=False)
        logger.info("  Saved distribution -> %s", out_path)

    return df


def get_class_weights(
    df: pd.DataFrame,
    output_dir: str | None = None,
    output_filename: str = "class_weights.json",
) -> dict[str, float]:
    """
    Compute per-class weights for imbalanced classification.

    Formula: weight = total_samples / (num_classes * class_count)

    Classes with zero samples receive a weight of 0.0.  All weights are
    rounded to 4 decimal places.

    Args:
        df:              DataFrame with a 'delay_category' column.
        output_dir:      If provided, saves the weights file there.
        output_filename: Filename for the saved JSON (default: 'class_weights.json').

    Returns:
        Dictionary mapping each DELAY_LABELS entry -> weight.
    """
    counts = df["delay_category"].value_counts()
    n_total   = len(df)
    n_classes = len(DELAY_LABELS)

    weights: dict[str, float] = {}
    for label in DELAY_LABELS:
        cnt = int(counts.get(label, 0))
        weights[label] = round(n_total / (n_classes * cnt), 4) if cnt > 0 else 0.0

    logger.info("  [get_class_weights]  Class weights (total=%d, classes=%d):",
                n_total, n_classes)
    for label, w in weights.items():
        cnt = int(counts.get(label, 0))
        logger.info("    %-20s  weight=%8.4f  (n=%d)", label, w, cnt)

    if output_dir:
        out_path = Path(output_dir) / output_filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(weights, f, indent=2)
        logger.info("  Saved class weights -> %s", out_path)

    return weights


# Main entry point

def clean_data(
    input_path: str,
    output_path: str,
    class_boundaries: tuple[float, float] | list[float] | None = None,
) -> pd.DataFrame:
    """
    runs the whole cleaning pipeline on combined_raw.csv.
    saves cleaned.csv and returns the dataframe.
    """
    in_path  = Path(input_path)
    out_path = Path(output_path)

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    sep = "-" * 65
    logger.info(sep)
    logger.info("TTC CLEANING PIPELINE  —  input: %s", in_path.name)
    logger.info(sep)

    # --- Load ---
    df = pd.read_csv(in_path, parse_dates=["date"], low_memory=False)
    logger.info("Loaded: %d rows × %d columns", *df.shape)

    n_start = len(df)

    # --- Step 1: Types + sentinels ---
    logger.info("\nStep 1/15 — handle_missing_values")
    df = handle_missing_values(df)

    # --- Step 2: Extract line hint from station name BEFORE stripping ---
    logger.info("\nStep 2/15 — normalize_line  (extract hint, then canonicalize)")
    df["_line_hint"] = _extract_line_hint_from_station(df[STATION_COL])
    df = _normalize_line_values(df)

    # --- Step 3: Clean station text ---
    logger.info("\nStep 3/15 — clean_station_names")
    df = clean_station_names(df)

    # --- Step 4: Apply alias dictionary ---
    # Count YONGE before mapping so it appears in the final summary.
    n_yonge_before = int((df[STATION_COL] == "YONGE").sum())
    logger.info("\nStep 4/15 — map_station_aliases")
    df = map_station_aliases(df)

    # --- Step 5: Fill missing lines from station lookup ---
    logger.info("\nStep 5/15 — infer_line_from_station")
    df = infer_line_from_station(df)

    # --- Step 6: Drop SRT (Line 3 — permanently closed) ---
    logger.info("\nStep 6/15 — drop_srt_rows")
    df = _drop_srt_rows(df)

    # --- Step 7: Resolve multi-station entries → single representative station ---
    logger.info("\nStep 7/15 — resolve_multi_station_entries")
    df = resolve_multi_station_entries(df)

    # --- Step 8: Recovery pass — non-official stations ---
    n_unknown_before_recovery = int((df[STATION_COL] == "UNKNOWN").sum())
    n_non_official_before = int((~df[STATION_COL].isin(OFFICIAL_STATIONS)).sum())
    logger.info("\nStep 8/15 — recover_non_official_stations")
    df = recover_non_official_stations(df)
    n_unknown_after_recovery = int((df[STATION_COL] == "UNKNOWN").sum())

    # --- Step 9: Correct station-line mismatches ---
    logger.info("\nStep 9/15 — correct_station_line_mismatches")
    df, n_mismatch_corrections = correct_station_line_mismatches(df)

    # --- Step 10: Finalize stations — collapse all non-official to UNKNOWN ---
    project_root = out_path.parent.parent.parent
    analysis_dir = str(project_root / "results" / "analysis")
    logger.info("\nStep 10/15 — finalize_stations")
    df, finalize_summary = finalize_stations(df, analysis_dir=analysis_dir)

    # --- Step 11: Clean delay codes ---
    logger.info("\nStep 11/15 — clean_delay_codes")
    df = clean_delay_codes(df)

    # --- Step 12: Cap outliers ---
    logger.info("\nStep 12/15 — cap_outliers")
    df = cap_outliers(df)

    # --- Step 13: Classify delay severity ---
    logger.info("\nStep 13/15 -- classify_delay  (output -> %s)", analysis_dir)
    df = classify_delay(df, output_dir=analysis_dir, boundaries=class_boundaries)

    # --- Remove internal helper column ---
    df = df.drop(columns=["_line_hint"], errors="ignore")

    # --- Validate (uses original YU/BD/SHP codes for consistency check) ---
    logger.info("\nValidation")
    report = validate(df)

    # --- Step 14: Nullify remaining station-line mismatches ---
    # Fixes interchange stations on wrong lines that correct_station_line_mismatches skips.
    # Must run BEFORE rename_lines so STATION_LINE_MAP keys (YU/BD/SHP) still match.
    logger.info("\nStep 14/15 -- nullify_mismatch_rows")
    df, n_nullified = nullify_mismatch_rows(df)

    # --- Step 15: Rename line codes to standardized identifiers ---
    # Done AFTER validation so the consistency check can use STATION_LINE_MAP keys.
    logger.info("\nStep 15/15 -- rename_lines  (YU->LINE_1, BD->LINE_2, SHP->LINE_4)")
    df = rename_lines(df)

    # --- Class weights for imbalanced learning ---
    logger.info("\nComputing class weights")
    class_weights = get_class_weights(
        df, output_dir=analysis_dir, output_filename="class_weights.json"
    )

    # --- Save ---
    df.to_csv(out_path, index=False)
    logger.info("\nSaved final dataset -> %s  (%d rows)", out_path, len(df))

    # --- Final summary ---
    n_unknown_station = int((df["station_validity"] == "UNKNOWN").sum())
    pct_unknown_station = 100 * n_unknown_station / max(len(df), 1)
    n_unknown_bound = int((df[BOUND_COL] == "UNKNOWN").sum())
    pct_unknown_bound = 100 * n_unknown_bound / max(len(df), 1)

    logger.info(sep)
    logger.info("FINAL SUMMARY")
    logger.info("  Total rows (final)        : %d", len(df))
    logger.info("  Rows in (raw)             : %d", n_start)
    logger.info("  Rows dropped (SRT)        : %d", n_start - len(df))
    logger.info("  ---")
    logger.info("  UNKNOWN stations          : %d (%.1f%%)", n_unknown_station, pct_unknown_station)
    logger.info("  UNKNOWN bound             : %d (%.1f%%)", n_unknown_bound, pct_unknown_bound)
    logger.info("  Unknown lines             : %.1f%%", report["unknown_line_pct"])
    logger.info("  ---")
    logger.info("  Mismatch corrections      : %d", n_mismatch_corrections)
    logger.info("  Mismatch rows nullified   : %d", n_nullified)
    logger.info("  ---")
    logger.info("  Delay class distribution:")
    delay_counts = df["delay_category"].value_counts().reindex(DELAY_LABELS, fill_value=0)
    zero_classes = []
    for label in DELAY_LABELS:
        cnt = int(delay_counts[label])
        pct = 100 * cnt / max(len(df), 1)
        flag = "  [!] IMBALANCED" if pct < IMBALANCE_THRESHOLD_PCT else ""
        if cnt == 0:
            flag = "  [!!] EMPTY — no samples"
            zero_classes.append(label)
        logger.info("    %-20s  %6d  (%5.1f%%)%s", label, cnt, pct, flag)
    if zero_classes:
        logger.warning("  Empty classes: %s", zero_classes)
    else:
        logger.info("  All classes have samples.")
    logger.info("  ---")
    logger.info("  Class weights (class_weights.json):")
    for label, w in class_weights.items():
        logger.info("    %-20s  %.4f", label, w)
    logger.info(sep)

    return df
