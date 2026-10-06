"""
Tool functions the agent can call, backed by the WAR CSVs and the enriched
ball-by-ball parquet built by data_layer.py.

Every function returns a plain JSON-serialisable dict (no NaN, no numpy
scalars, no DataFrames) so it can go straight into an Anthropic tool_result
block or be rendered in Streamlit.

Run `python data_layer.py` first to produce the CSVs/parquet these read.
"""

import difflib
import functools

import numpy as np
import pandas as pd

import config

VALID_METRICS     = {"batting_war", "bowling_war", "fielding_war", "total_war"}
VALID_BOWLER_TYPE = {"pace", "spin"}
VALID_BATTER_HAND = {"LHB", "RHB"}


# ---------------------------------------------------------------------------
# Data loading (cached — the files don't change during a session)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _batting_war() -> pd.DataFrame:
    return pd.read_csv(config.BATTING_WAR_CSV)


@functools.lru_cache(maxsize=1)
def _bowling_war() -> pd.DataFrame:
    return pd.read_csv(config.BOWLING_WAR_CSV)


@functools.lru_cache(maxsize=1)
def _fielding_war() -> pd.DataFrame:
    return pd.read_csv(config.FIELDING_WAR_CSV)


@functools.lru_cache(maxsize=1)
def _deliveries() -> pd.DataFrame:
    return pd.read_parquet(config.ENRICHED_DELIVERIES_PARQUET)


@functools.lru_cache(maxsize=1)
def _all_player_names() -> list:
    names = set(_batting_war()["striker"]) | set(_bowling_war()["bowler"]) | set(_fielding_war()["player"])
    return sorted(names)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _appearance_counts() -> dict:
    """Rows per player across the three WAR tables (used to break name ties)."""
    counts = {}
    for df, col in ((_batting_war(), "striker"), (_bowling_war(), "bowler"), (_fielding_war(), "player")):
        for name, n in df[col].value_counts().items():
            counts[name] = counts.get(name, 0) + int(n)
    return counts


class PlayerNotFoundError(ValueError):
    pass


def resolve_player(name: str) -> str:
    """
    Map a user-typed name (any case, possibly partial, e.g. "kohli" or
    "dhoni") to the exact CricSheet spelling used across the WAR tables.
    Tries: exact (case-insensitive) -> unique substring -> fuzzy match.
    Raises PlayerNotFoundError with suggestions if nothing is close.
    """
    candidates = _all_player_names()
    name_l = name.strip().lower()

    for c in candidates:
        if c.lower() == name_l:
            return c

    # "Virat Kohli" -> "V Kohli", "Jasprit Bumrah" -> "JJ Bumrah": CricSheet
    # stores initials + surname, so match surname plus the first initial.
    parts = name_l.split()
    if len(parts) >= 2:
        first, last = parts[0], parts[-1]
        initial_hits = [
            c for c in candidates
            if c.split()[0].isupper()                 # candidate uses initials
            and len(c.split()) >= 2
            and c.lower().split()[-1] == last
            and c.lower().split()[0].startswith(first[0])
        ]
        if len(initial_hits) == 1:
            return initial_hits[0]
        if len(initial_hits) > 1:
            # e.g. "Rohit Sharma" fits both "R Sharma" and "RG Sharma"; take
            # the player with clearly the most appearances in the WAR tables.
            counts = _appearance_counts()
            ranked = sorted(initial_hits, key=lambda c: counts.get(c, 0), reverse=True)
            if counts.get(ranked[0], 0) > counts.get(ranked[1], 0):
                return ranked[0]
            raise PlayerNotFoundError(
                f"'{name}' matches multiple players: {', '.join(initial_hits[:8])}. "
                "Please be more specific."
            )

    substring_hits = [c for c in candidates if name_l in c.lower()]
    if len(substring_hits) == 1:
        return substring_hits[0]
    if len(substring_hits) > 1:
        # Prefer a hit where the substring matches the surname/last token
        exact_token = [c for c in substring_hits if c.lower().split()[-1] == name_l]
        if len(exact_token) == 1:
            return exact_token[0]
        raise PlayerNotFoundError(
            f"'{name}' matches multiple players: {', '.join(substring_hits[:8])}. "
            "Please be more specific."
        )

    close = difflib.get_close_matches(name, candidates, n=5, cutoff=0.6)
    if close:
        raise PlayerNotFoundError(
            f"No player named '{name}' found. Did you mean: {', '.join(close)}?"
        )
    raise PlayerNotFoundError(f"No player named '{name}' found in the WAR data.")


def _clean(record: dict) -> dict:
    """Replace NaN/NaT and numpy scalar types with plain JSON-safe values."""
    out = {}
    for k, v in record.items():
        if isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = None if np.isnan(v) else round(float(v), 3)
        elif isinstance(v, float) and np.isnan(v):
            out[k] = None
        else:
            out[k] = v
    return out


def _season_filter(df: pd.DataFrame, season) -> pd.DataFrame:
    if season is None:
        return df
    return df[df["season"] == int(season)]


# ---------------------------------------------------------------------------
# 1. get_player_war
# ---------------------------------------------------------------------------

def get_player_war(player: str, season: int = None) -> dict:
    """
    Combined batting/bowling/fielding WAR for a player.
    If season is None, returns career totals (sum across seasons) plus a
    per-season breakdown.
    """
    name = resolve_player(player)

    bat = _season_filter(_batting_war()[_batting_war()["striker"] == name], season)
    bowl = _season_filter(_bowling_war()[_bowling_war()["bowler"] == name], season)
    field = _season_filter(_fielding_war()[_fielding_war()["player"] == name], season)

    def summarize(df, war_col):
        if df.empty:
            return None
        if season is not None:
            return _clean(df.iloc[0].to_dict())
        # Career: sum the additive WAR components, keep per-season rows too.
        total = float(df[war_col].sum())
        return {
            "career_" + war_col: round(total, 3),
            "seasons_played": sorted(int(s) for s in df["season"].unique()),
            "by_season": [_clean(r) for r in df.to_dict("records")],
        }

    batting_summary  = summarize(bat, "batting_war")
    bowling_summary  = summarize(bowl, "bowling_war")
    fielding_summary = summarize(field, "fielding_war")

    def war_value(summary, war_col):
        if summary is None:
            return 0.0
        return summary.get(war_col, summary.get("career_" + war_col, 0.0))

    total_war = (
        war_value(batting_summary, "batting_war")
        + war_value(bowling_summary, "bowling_war")
        + war_value(fielding_summary, "fielding_war")
    )

    return {
        "player": name,
        "season": season,
        "batting": batting_summary,
        "bowling": bowling_summary,
        "fielding": fielding_summary,
        "total_war": round(total_war, 3),
    }


# ---------------------------------------------------------------------------
# 2. get_batting_splits
# ---------------------------------------------------------------------------

def get_batting_splits(player: str, bowler_type: str, season: int = None) -> dict:
    """Batter's performance against pace or spin bowling."""
    name = resolve_player(player)
    bowler_type = bowler_type.strip().lower()
    if bowler_type not in VALID_BOWLER_TYPE:
        raise ValueError(f"bowler_type must be one of {sorted(VALID_BOWLER_TYPE)}, got '{bowler_type}'")

    df = _deliveries()
    df = df[(df["striker"] == name) & (df["bowler_type"] == bowler_type)]
    df = _season_filter(df, season)

    balls = int(df["legal_ball"].sum())
    runs = int(df.loc[df["legal_ball"], "runs_off_bat"].sum())
    dismissals = int(df["is_wicket"].sum())

    return {
        "player": name,
        "bowler_type": bowler_type,
        "season": season,
        "runs": runs,
        "balls_faced": balls,
        "dismissals": dismissals,
        "strike_rate": round(runs / balls * 100, 2) if balls else None,
        "average": round(runs / dismissals, 2) if dismissals else None,
    }


# ---------------------------------------------------------------------------
# 3. get_matchup_stats
# ---------------------------------------------------------------------------

def get_matchup_stats(bowler: str, batter_hand: str, season: int = None) -> dict:
    """Bowler's performance against left-handed or right-handed batters."""
    name = resolve_player(bowler)
    batter_hand = batter_hand.strip().upper()
    if batter_hand not in VALID_BATTER_HAND:
        raise ValueError(f"batter_hand must be one of {sorted(VALID_BATTER_HAND)}, got '{batter_hand}'")

    df = _deliveries()
    df = df[(df["bowler"] == name) & (df["batter_hand"] == batter_hand)]
    df = _season_filter(df, season)

    legal_balls = int(df["legal_ball_bowler"].sum())
    runs_conceded = float(df["bowler_runs"].sum())
    wickets = int(df["is_bowler_wicket"].sum())

    return {
        "bowler": name,
        "batter_hand": batter_hand,
        "season": season,
        "runs_conceded": runs_conceded,
        "legal_balls": legal_balls,
        "overs": round(legal_balls / 6, 1),
        "wickets": wickets,
        "economy": round(runs_conceded / legal_balls * 6, 2) if legal_balls else None,
    }


# ---------------------------------------------------------------------------
# 4. get_player_comparison
# ---------------------------------------------------------------------------

def get_player_comparison(player1: str, player2: str, season: int = None) -> dict:
    """Side-by-side WAR comparison of two players."""
    p1 = get_player_war(player1, season)
    p2 = get_player_war(player2, season)

    if p1["total_war"] > p2["total_war"]:
        leader = p1["player"]
    elif p2["total_war"] > p1["total_war"]:
        leader = p2["player"]
    else:
        leader = None

    return {
        "season": season,
        "player1": p1,
        "player2": p2,
        "war_leader": leader,
        "war_gap": round(abs(p1["total_war"] - p2["total_war"]), 3),
    }


# ---------------------------------------------------------------------------
# 5. get_season_leaderboard
# ---------------------------------------------------------------------------

def get_season_leaderboard(metric: str, season: int, top_n: int = 10) -> dict:
    """Top N players in a season by batting_war / bowling_war / fielding_war / total_war."""
    metric = metric.strip().lower()
    if metric not in VALID_METRICS:
        raise ValueError(f"metric must be one of {sorted(VALID_METRICS)}, got '{metric}'")
    season = int(season)
    top_n = max(1, min(int(top_n), 50))

    if metric == "batting_war":
        df = _batting_war()
        df = df[df["season"] == season].sort_values("batting_war", ascending=False).head(top_n)
        cols = ["season", "striker", "total_runs", "total_balls", "batting_sr", "batting_war"]
        rows = [_clean(r) for r in df[cols].rename(columns={"striker": "player"}).to_dict("records")]

    elif metric == "bowling_war":
        df = _bowling_war()
        df = df[df["season"] == season].sort_values("bowling_war", ascending=False).head(top_n)
        cols = ["season", "bowler", "overall_overs", "total_wickets", "overall_economy", "bowling_war"]
        rows = [_clean(r) for r in df[cols].rename(columns={"bowler": "player"}).to_dict("records")]

    elif metric == "fielding_war":
        df = _fielding_war()
        df = df[df["season"] == season].sort_values("fielding_war", ascending=False).head(top_n)
        cols = ["season", "player", "matches", "caught", "run_out", "stumped", "fielding_war"]
        rows = [_clean(r) for r in df[cols].to_dict("records")]

    else:  # total_war
        bat  = _batting_war()[_batting_war()["season"] == season][["striker", "batting_war"]].rename(columns={"striker": "player"})
        bowl = _bowling_war()[_bowling_war()["season"] == season][["bowler", "bowling_war"]].rename(columns={"bowler": "player"})
        field = _fielding_war()[_fielding_war()["season"] == season][["player", "fielding_war"]]

        merged = bat.merge(bowl, on="player", how="outer").merge(field, on="player", how="outer")
        for col in ["batting_war", "bowling_war", "fielding_war"]:
            merged[col] = merged[col].fillna(0.0)
        merged["total_war"] = merged["batting_war"] + merged["bowling_war"] + merged["fielding_war"]
        merged = merged.sort_values("total_war", ascending=False).head(top_n)
        merged.insert(0, "season", season)
        rows = [_clean(r) for r in merged.to_dict("records")]

    return {"metric": metric, "season": season, "top_n": top_n, "leaderboard": rows}


# ---------------------------------------------------------------------------
# 6. get_keeper_stats
# ---------------------------------------------------------------------------

def get_keeper_stats(player: str, season: int = None) -> dict:
    """Wicketkeeper-relevant fielding numbers: byes conceded, stumpings, catches."""
    name = resolve_player(player)
    df = _season_filter(_fielding_war()[_fielding_war()["player"] == name], season)

    if df.empty:
        return {"player": name, "season": season, "found": False}

    if season is not None:
        row = _clean(df.iloc[0].to_dict())
        matches = row.get("matches", 0)
        row["byes_per_match"] = round(row["bye"] / matches, 2) if matches else None
        row["is_primary_keeper"] = row.get("stumped", 0) > 0
        row["found"] = True
        return {"player": name, "season": season, **row}

    totals = df[["matches", "bye", "caught", "run_out", "stumped", "fielding_war"]].sum()
    matches = int(totals["matches"])
    return {
        "player": name,
        "season": None,
        "found": True,
        "seasons_played": sorted(int(s) for s in df["season"].unique()),
        "matches": matches,
        "byes_conceded": int(totals["bye"]),
        "byes_per_match": round(totals["bye"] / matches, 2) if matches else None,
        "stumpings": int(totals["stumped"]),
        "catches": int(totals["caught"]),
        "run_outs": int(totals["run_out"]),
        "career_fielding_war": round(float(totals["fielding_war"]), 3),
        "is_primary_keeper": bool(totals["stumped"] > 0),
    }
