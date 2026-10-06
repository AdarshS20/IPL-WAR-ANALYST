"""
IPL Bowling WAR — v1 (runs saved above average, phase-weighted, wicket credit)
===============================================================================
Data source: CricSheet ball-by-ball IPL data
  Download: https://cricsheet.org/downloads/ipl_csv2.zip
  Format:   CSV2 (one row per delivery, single header row)

Bowling WAR logic
-----------------
A bowler creates value in two ways:
  1. Conceding fewer runs than the average bowler would in that phase/venue
     (economy rate contribution)
  2. Taking wickets — each dismissal directly credited to the bowler
     removes a batter and resets the partnership

Runs charged to bowler = runs_off_bat + wides + no-balls
  (byes and legbyes are the wicketkeeper/fielding side's fault, not the bowler)

Wickets credited to bowler = all wicket_type dismissals EXCEPT run-outs
  (run-outs are fielding credit, handled in fielding WAR)

Build order
-----------
1. Load & clean deliveries  (reuses same CricSheet files as batting WAR)
2. Compute venue/phase economy rate benchmarks
3. Compute per-bowler phase contributions
4. Apply wicket credit per dismissal
5. Compare to replacement baseline
6. Convert to WAR

Dependencies: pip install pandas numpy
"""

import pandas as pd
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------------

DATA_DIR = Path("ipl_csv2")

# Same constant as batting WAR — must match for combined WAR to be additive
RUNS_PER_WIN = 10.0   # tune to match your calibrated batting WAR constant

# Replacement level: bottom 20% of qualified bowlers
REPLACEMENT_PERCENTILE = 0.20

# Minimum balls bowled to qualify (filters out part-time fillers)
MIN_BALLS_BOWLED = 60  # ~10 overs across the season

# Over ranges for each phase
PHASES = {
    "powerplay": (1, 6),
    "middle":    (7, 15),
    "death":     (16, 20),
}

# Phase leverage multipliers for bowlers — mirrors batting multipliers.
# Death-over bowling is highest leverage (defending a total / attacking a chase)
PHASE_MULTIPLIERS = {
    "powerplay": 1.1,
    "middle":    1.0,
    "death":     1.3,
}

# Wicket types that credit the BOWLER.
# Run-outs are fielding credit; obstructions etc. are rare and excluded.
BOWLER_WICKET_TYPES = {
    "bowled",
    "caught",
    "caught and bowled",
    "lbw",
    "stumped",
    "hit wicket",
}

# Run value awarded to bowler per wicket taken, by phase.
# A powerplay wicket is worth more — it removes a set batter early,
# disrupts the innings structure, and has more balls remaining to benefit from.
WICKET_CREDIT = {
    "powerplay": 7.0,
    "middle":    5.0,
    "death":     3.0,
}


# ---------------------------------------------------------------------------
# 2. LOAD & CLEAN
# ---------------------------------------------------------------------------

def assign_phase(over: int) -> str:
    """Map an over number (1-indexed) to its phase label."""
    for phase, (start, end) in PHASES.items():
        if start <= over <= end:
            return phase
    return "middle"


def load_deliveries(data_dir: Path) -> pd.DataFrame:
    """
    Read all CricSheet ball-by-ball CSVs (skipping _info files).
    Identical to batting WAR loader — if running both models, you can
    call this once and pass the DataFrame to both pipelines.
    """
    frames = []
    for path in sorted(data_dir.glob("*.csv")):
        if path.stem.endswith("_info"):
            continue
        try:
            df = pd.read_csv(path)
            frames.append(df)
        except Exception as e:
            print(f"Skipping {path.name}: {e}")

    if not frames:
        raise FileNotFoundError(
            f"No CSV files found in '{data_dir}'.\n"
            "Download ipl_csv2.zip from https://cricsheet.org/downloads/ "
            "and unzip it into a folder called 'ipl_csv2'."
        )

    raw = pd.concat(frames, ignore_index=True)
    return clean_deliveries(raw)


def clean_deliveries(df: pd.DataFrame) -> pd.DataFrame:
    """Derive helper columns needed for the bowling WAR model."""

    # Over number (1-indexed)
    df["over_num"] = df["ball"].apply(
        lambda b: int(str(b).split(".")[0]) + 1
    )

    # Phase label
    df["phase"] = df["over_num"].apply(assign_phase)

    # Season from start_date — avoids CricSheet's ambiguous season labels
    # e.g. "2020/21" and "2021" would both map to 2021 via string parsing,
    # but start_date correctly separates the two tournaments.
    df["season"] = pd.to_datetime(df["start_date"]).dt.year

    # Ensure numeric columns
    for col in ["runs_off_bat", "extras", "wides", "noballs", "byes", "legbyes"]:
        df[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0)

    # Runs charged to the BOWLER:
    #   runs off bat + wides + no-balls
    #   byes & legbyes are NOT the bowler's fault — exclude them
    df["bowler_runs"] = df["runs_off_bat"] + df["wides"] + df["noballs"]

    # Legal ball for bowler economy: wides and no-balls do NOT count
    # as a legal delivery (bowler must bowl it again)
    df["is_wide"]   = df["wides"] > 0
    df["is_noball"] = df["noballs"] > 0
    df["legal_ball_bowler"] = ~(df["is_wide"] | df["is_noball"])

    # Bowler wicket: only credit dismissals the bowler caused
    df["is_bowler_wicket"] = df["wicket_type"].str.lower().isin(BOWLER_WICKET_TYPES)

    return df


# ---------------------------------------------------------------------------
# 3. VENUE/PHASE ECONOMY BENCHMARKS
# ---------------------------------------------------------------------------

def compute_venue_phase_economy(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (venue, phase), compute average runs per legal ball.
    This is the benchmark each bowler is compared against.

    Note: we use ALL deliveries (including wides/no-balls) for total runs,
    but only legal balls for the denominator — matching how economy rate
    is conventionally computed.
    """
    vp = (
        df.groupby(["venue", "phase"])
        .agg(
            total_bowler_runs=("bowler_runs",      "sum"),
            total_legal_balls=("legal_ball_bowler", "sum"),
        )
        .reset_index()
    )
    # Economy rate = runs per over = runs per ball × 6
    vp["avg_economy"] = vp["total_bowler_runs"] / vp["total_legal_balls"] * 6
    return vp


def compute_league_phase_economy(df: pd.DataFrame) -> dict:
    """League-wide average economy per phase — fallback for thin venue samples."""
    lg = (
        df.groupby("phase")
        .agg(
            total_bowler_runs=("bowler_runs",       "sum"),
            total_legal_balls=("legal_ball_bowler",  "sum"),
        )
        .reset_index()
    )
    lg["league_avg_economy"] = lg["total_bowler_runs"] / lg["total_legal_balls"] * 6
    return lg.set_index("phase")["league_avg_economy"].to_dict()


# ---------------------------------------------------------------------------
# 4. PER-BOWLER PHASE CONTRIBUTIONS
# ---------------------------------------------------------------------------

def compute_bowler_phase_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate each bowler's runs conceded, legal balls bowled, and
    bowler-credited wickets per (season, bowler, venue, phase).
    """
    stats = (
        df.groupby(["season", "bowler", "venue", "phase"])
        .agg(
            bowler_runs  =("bowler_runs",       "sum"),
            legal_balls  =("legal_ball_bowler",  "sum"),
            wickets      =("is_bowler_wicket",   "sum"),
        )
        .reset_index()
    )
    # Economy rate for this bowler in this phase/venue
    stats["economy"] = stats["bowler_runs"] / stats["legal_balls"] * 6
    return stats


def merge_benchmarks(
    bowler_stats: pd.DataFrame,
    venue_phase_eco: pd.DataFrame,
    league_phase_eco: dict,
) -> pd.DataFrame:
    """
    Join venue-phase benchmark economy onto bowler stats.
    Fall back to league average when venue sample < 50 legal balls.
    """
    merged = bowler_stats.merge(
        venue_phase_eco[["venue", "phase", "avg_economy", "total_legal_balls"]],
        on=["venue", "phase"],
        how="left",
        suffixes=("", "_venue"),
    )

    merged["benchmark_economy"] = np.where(
        merged["total_legal_balls"] >= 50,
        merged["avg_economy"],
        merged["phase"].map(league_phase_eco),
    )
    return merged


# ---------------------------------------------------------------------------
# 5. BOWLING RUNS SAVED ABOVE AVERAGE  (with wicket credit)
# ---------------------------------------------------------------------------

def compute_bowling_raa(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Bowling RAA per bowler-season-phase:

        runs_saved   = legal_balls × (benchmark_economy − actual_economy) / 6
                       positive = bowler was more economical than average
                       negative = bowler was more expensive than average

        wicket_value = wickets × phase_wicket_credit
                       each bowler wicket adds run value on top of economy

        weighted_raa = (runs_saved + wicket_value) × phase_multiplier

    The wicket credit and economy savings are additive — a bowler who
    is both economical AND takes wickets is doubly valuable.
    """
    merged = merged.copy()

    # Economy contribution (positive = good for the bowler)
    merged["runs_saved"] = (
        merged["legal_balls"]
        * (merged["benchmark_economy"] - merged["economy"])
        / 6
    )

    # Wicket credit
    merged["wicket_value"] = merged["wickets"] * merged["phase"].map(WICKET_CREDIT)

    # Phase leverage
    merged["phase_mult"]   = merged["phase"].map(PHASE_MULTIPLIERS)
    merged["weighted_raa"] = (merged["runs_saved"] + merged["wicket_value"]) * merged["phase_mult"]

    return merged


def aggregate_to_season(raa_df: pd.DataFrame) -> pd.DataFrame:
    """Sum weighted RAA across all phases and venues per (bowler, season)."""
    season_stats = (
        raa_df.groupby(["season", "bowler"])
        .agg(
            total_bowler_runs=("bowler_runs",   "sum"),
            total_legal_balls=("legal_balls",   "sum"),
            total_wickets    =("wickets",        "sum"),
            total_wicket_val =("wicket_value",  "sum"),
            bowling_raa      =("weighted_raa",  "sum"),
        )
        .reset_index()
    )
    season_stats["overall_economy"] = (
        season_stats["total_bowler_runs"] / season_stats["total_legal_balls"] * 6
    )
    season_stats["overall_overs"] = season_stats["total_legal_balls"] / 6
    return season_stats


# ---------------------------------------------------------------------------
# 6. REPLACEMENT BASELINE & WAR
# ---------------------------------------------------------------------------

def compute_bowling_war(season_stats: pd.DataFrame) -> pd.DataFrame:
    """
    1. Qualify bowlers by minimum legal balls bowled.
    2. Replacement rate = bowling RAA per ball at the REPLACEMENT_PERCENTILE.
       Note: for bowling, higher RAA = better, same sign convention as batting.
    3. replacement_runs = bowler's legal balls × replacement_rate
    4. WAR = (bowling_raa − replacement_runs) / RUNS_PER_WIN
    """
    qualified = season_stats[
        season_stats["total_legal_balls"] >= MIN_BALLS_BOWLED
    ].copy()

    replacement_rate = (
        qualified["bowling_raa"].quantile(REPLACEMENT_PERCENTILE)
        / qualified["total_legal_balls"].mean()
    )

    qualified["replacement_runs"]       = qualified["total_legal_balls"] * replacement_rate
    qualified["runs_above_replacement"] = qualified["bowling_raa"] - qualified["replacement_runs"]
    qualified["bowling_war"]            = qualified["runs_above_replacement"] / RUNS_PER_WIN

    return qualified.sort_values("bowling_war", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 7. MAIN PIPELINE
# ---------------------------------------------------------------------------

def run_pipeline(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    print("Loading deliveries...")
    df = load_deliveries(data_dir)
    print(f"  {len(df):,} deliveries | "
          f"{df['match_id'].nunique()} matches | "
          f"seasons: {sorted(df['season'].unique())}")

    print("Computing venue/phase economy benchmarks...")
    venue_phase_eco  = compute_venue_phase_economy(df)
    league_phase_eco = compute_league_phase_economy(df)

    print("Computing bowler phase stats...")
    bowler_stats = compute_bowler_phase_stats(df)

    print("Merging benchmarks...")
    merged = merge_benchmarks(bowler_stats, venue_phase_eco, league_phase_eco)

    print("Computing bowling RAA with wicket credit...")
    raa_df       = compute_bowling_raa(merged)
    season_stats = aggregate_to_season(raa_df)

    print("Computing bowling WAR...")
    war_df = compute_bowling_war(season_stats)

    return war_df


# ---------------------------------------------------------------------------
# 8. QUICK VALIDATION
# ---------------------------------------------------------------------------

def validate(war_df: pd.DataFrame, season: int = None) -> None:
    """
    Print top/bottom 10 for a season.
    Healthy distribution:
      - Best bowler in a season: ~3-6 WAR
      - Solid regular: ~1-2 WAR
      - Below replacement: negative
    """
    df           = war_df if season is None else war_df[war_df["season"] == season]
    season_label = season or "all seasons"

    cols = ["season", "bowler", "overall_overs", "total_wickets",
            "overall_economy", "total_wicket_val", "bowling_raa", "bowling_war"]

    print(f"\n=== Top 10 bowling WAR — {season_label} ===")
    print(df[cols].head(10).to_string(index=False))

    print(f"\n=== Bottom 10 bowling WAR — {season_label} ===")
    print(df[cols].tail(10).to_string(index=False))

    above = (df["bowling_war"] > 0).mean()
    print(f"\nBowlers above replacement : {above:.1%}")
    print(f"Mean WAR                  : {df['bowling_war'].mean():.2f}")
    print(f"Max WAR                   : {df['bowling_war'].max():.2f}")
    print(f"Avg wicket value          : {df['total_wicket_val'].mean():.1f} runs")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    war_df = run_pipeline()
    latest_season = int(war_df["season"].max())
    validate(war_df, season=latest_season)

    out = Path("ipl_bowling_war.csv")
    war_df.to_csv(out, index=False)
    print(f"\nFull results saved to {out}")