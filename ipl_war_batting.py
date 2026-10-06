"""
IPL Batting WAR — v2 (runs above average, phase-weighted, wicket penalty)
==========================================================================
Data source: CricSheet ball-by-ball IPL data
  Download: https://cricsheet.org/downloads/ipl_csv2.zip
  Format:   CSV2 (one row per delivery, single header row)

Columns used from CricSheet:
  match_id, season, start_date, venue, innings,
  ball, striker, runs_off_bat, extras, wides,
  wicket_type, player_dismissed

Build order
-----------
1. Load & clean deliveries
2. Compute venue-adjusted run environment per phase
3. Compute each batter's phase contributions
4. Apply wicket penalty per dismissal
5. Compare to phase-specific replacement baseline
6. Convert to WAR using runs-per-win constant

Dependencies: pip install pandas numpy
"""

import pandas as pd
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------------

# Adjust to wherever you unzipped the CricSheet CSV2 files
DATA_DIR = Path("ipl_csv2")

# Runs-per-win constant for T20 — empirically ~10, tune after validation
RUNS_PER_WIN = 10.0

# Replacement level: bottom 20% of qualified players
REPLACEMENT_PERCENTILE = 0.20

# Minimum balls faced to qualify (filters out cameos)
MIN_BALLS_BATTED = 60   # ~4 typical innings worth of balls

# Over ranges for each phase
PHASES = {
    "powerplay": (1, 6),
    "middle":    (7, 15),
    "death":     (16, 20),
}

# Phase leverage multipliers — death overs count more because
# each run is higher-leverage. Calibrate against WPA later.
PHASE_MULTIPLIERS = {
    "powerplay": 1.1,
    "middle":    1.0,
    "death":     1.3,
}

# Wicket penalty — runs subtracted per dismissal.
# A powerplay wicket is most damaging (many balls remain for a new batter
# to settle); a death-overs wicket is least damaging (few balls left).
# These are reasonable starting values — tune empirically using WPA data.
WICKET_PENALTY = {
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
    return "middle"  # fallback


def load_deliveries(data_dir: Path) -> pd.DataFrame:
    """
    Read all CricSheet ball-by-ball CSVs (skipping _info files)
    and concatenate into one DataFrame.

    CricSheet CSV2 format has a single header row and one row per delivery.
    Columns already include: match_id, season, venue, striker, runs_off_bat,
    wides, wicket_type etc — no transformation of headers needed.
    """
    frames = []
    for path in sorted(data_dir.glob("*.csv")):
        # Skip metadata files — they have a different structure
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
    """Derive helper columns needed for the WAR model."""

    # Over number (1-indexed): CricSheet 'ball' column is over.delivery
    # e.g. 0.1 = over 1 ball 1,  5.6 = over 6 ball 6
    df["over_num"] = df["ball"].apply(
        lambda b: int(str(b).split(".")[0]) + 1
    )

    # Phase label per delivery
    df["phase"] = df["over_num"].apply(assign_phase)

    # Legal delivery: wides don't count toward batter strike rate
    df["is_wide"]    = df["wides"].notna() & (df["wides"] > 0)
    df["legal_ball"] = ~df["is_wide"]

    # Wicket flag — True only when the striker is dismissed
    # (excludes run-outs of the non-striker which appear in other_wicket_type)
    df["is_wicket"] = df["wicket_type"].notna() & (df["wicket_type"] != "")

    # Derive season from start_date rather than the 'season' field.
    # CricSheet's season labels are inconsistent for IPL: "2020/21" refers
    # to the 2020 tournament (played Sep-Nov 2020 in the UAE), while "2021"
    # refers to the separate 2021 tournament. Parsing the label directly
    # causes both to collapse onto the same year. Using the calendar year
    # of start_date is unambiguous and matches each match to its actual
    # IPL edition.
    df["season"] = pd.to_datetime(df["start_date"]).dt.year

    # Ensure numeric run columns
    df["runs_off_bat"] = pd.to_numeric(df["runs_off_bat"], errors="coerce").fillna(0)
    df["extras"]       = pd.to_numeric(df["extras"],       errors="coerce").fillna(0)

    return df


# ---------------------------------------------------------------------------
# 3. VENUE-ADJUSTED RUN ENVIRONMENT
# ---------------------------------------------------------------------------

def compute_venue_phase_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (venue, phase), compute average strike rate (runs per 100
    legal balls). This is the benchmark each batter is compared against.
    """
    legal = df[df["legal_ball"]].copy()

    vp = (
        legal.groupby(["venue", "phase"])
        .agg(
            total_runs =("runs_off_bat", "sum"),
            total_balls=("legal_ball",   "count"),
        )
        .reset_index()
    )
    vp["avg_sr"] = vp["total_runs"] / vp["total_balls"] * 100
    return vp


def compute_league_phase_averages(df: pd.DataFrame) -> dict:
    """
    League-wide average SR per phase — used as fallback when a venue
    has too few balls in that phase to be reliable.
    """
    legal = df[df["legal_ball"]].copy()

    lg = (
        legal.groupby("phase")
        .agg(
            total_runs =("runs_off_bat", "sum"),
            total_balls=("legal_ball",   "count"),
        )
        .reset_index()
    )
    lg["league_avg_sr"] = lg["total_runs"] / lg["total_balls"] * 100
    return lg.set_index("phase")["league_avg_sr"].to_dict()


# ---------------------------------------------------------------------------
# 4. PER-BATTER PHASE CONTRIBUTIONS
# ---------------------------------------------------------------------------

def compute_batter_phase_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate each batter's runs, balls faced, and dismissals per
    (season, striker, venue, phase).
    Dismissals are counted here so the wicket penalty can be applied
    at the same phase level as the run contributions.
    """
    legal = df[df["legal_ball"]].copy()

    stats = (
        legal.groupby(["season", "striker", "venue", "phase"])
        .agg(
            runs       =("runs_off_bat", "sum"),
            balls      =("legal_ball",   "count"),
            dismissals =("is_wicket",    "sum"),   # <-- new
        )
        .reset_index()
    )
    stats["sr"] = stats["runs"] / stats["balls"] * 100
    return stats


def merge_benchmarks(
    batter_stats: pd.DataFrame,
    venue_phase_avg: pd.DataFrame,
    league_phase_avg: dict,
) -> pd.DataFrame:
    """
    Join venue-phase average SR onto batter stats, falling back to the
    league average when a venue has fewer than 50 legal balls in that phase.
    """
    merged = batter_stats.merge(
        venue_phase_avg[["venue", "phase", "avg_sr", "total_balls"]],
        on=["venue", "phase"],
        how="left",
        suffixes=("", "_venue"),
    )

    merged["benchmark_sr"] = np.where(
        merged["total_balls"] >= 50,
        merged["avg_sr"],
        merged["phase"].map(league_phase_avg),
    )
    return merged


# ---------------------------------------------------------------------------
# 5. BATTING RUNS ABOVE AVERAGE  (with wicket penalty)
# ---------------------------------------------------------------------------

def compute_batting_raa(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Batting RAA per batter-season-phase:

        raw_raa      = balls_faced x (player_SR - benchmark_SR) / 100
        wicket_cost  = dismissals x phase_wicket_penalty
        weighted_raa = (raw_raa - wicket_cost) x phase_leverage_multiplier

    The wicket penalty captures the run value lost when a batter is
    dismissed — the team loses remaining potential contribution and
    absorbs the cost of a new batter settling in.

    Penalty is phase-specific because a powerplay wicket (many balls
    remaining) is more damaging than a death-overs wicket.
    """
    merged = merged.copy()

    # Strike-rate contribution
    merged["raw_raa"] = merged["balls"] * (merged["sr"] - merged["benchmark_sr"]) / 100

    # Wicket penalty — subtracted before phase multiplier
    merged["wicket_cost"] = merged["dismissals"] * merged["phase"].map(WICKET_PENALTY)

    # Phase leverage applied to the net figure
    merged["phase_mult"]   = merged["phase"].map(PHASE_MULTIPLIERS)
    merged["weighted_raa"] = (merged["raw_raa"] - merged["wicket_cost"]) * merged["phase_mult"]

    return merged


def aggregate_to_season(raa_df: pd.DataFrame) -> pd.DataFrame:
    """
    Sum weighted RAA across all phases and venues to get one
    batting RAA figure per (batter, season).
    """
    season_stats = (
        raa_df.groupby(["season", "striker"])
        .agg(
            total_runs  =("runs",        "sum"),
            total_balls =("balls",       "sum"),
            total_disms =("dismissals",  "sum"),   # <-- carried through for inspection
            wicket_cost =("wicket_cost", "sum"),   # <-- total runs lost to dismissals
            batting_raa =("weighted_raa","sum"),
        )
        .reset_index()
    )
    season_stats["batting_sr"] = (
        season_stats["total_runs"] / season_stats["total_balls"] * 100
    )
    return season_stats


# ---------------------------------------------------------------------------
# 6. REPLACEMENT BASELINE & WAR
# ---------------------------------------------------------------------------

def compute_batting_war(season_stats: pd.DataFrame) -> pd.DataFrame:
    """
    1. Keep only players who faced >= MIN_BALLS_BATTED (removes cameos).
    2. Replacement rate = RAA per ball at the REPLACEMENT_PERCENTILE
       across all qualified players.
    3. replacement_runs = player's balls x replacement_rate
    4. WAR = (batting_raa - replacement_runs) / RUNS_PER_WIN
    """
    qualified = season_stats[
        season_stats["total_balls"] >= MIN_BALLS_BATTED
    ].copy()

    replacement_rate = (
        qualified["batting_raa"].quantile(REPLACEMENT_PERCENTILE)
        / qualified["total_balls"].mean()
    )

    qualified["replacement_runs"]       = qualified["total_balls"] * replacement_rate
    qualified["runs_above_replacement"] = qualified["batting_raa"] - qualified["replacement_runs"]
    qualified["batting_war"]            = qualified["runs_above_replacement"] / RUNS_PER_WIN

    return qualified.sort_values("batting_war", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 7. MAIN PIPELINE
# ---------------------------------------------------------------------------

def run_pipeline(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    print("Loading deliveries...")
    df = load_deliveries(data_dir)
    print(f"  {len(df):,} deliveries | "
          f"{df['match_id'].nunique()} matches | "
          f"seasons: {sorted(df['season'].unique())}")

    print("Computing venue/phase benchmarks...")
    venue_phase_avg  = compute_venue_phase_averages(df)
    league_phase_avg = compute_league_phase_averages(df)

    print("Computing batter phase stats...")
    batter_stats = compute_batter_phase_stats(df)

    print("Merging benchmarks...")
    merged = merge_benchmarks(batter_stats, venue_phase_avg, league_phase_avg)

    print("Computing batting RAA with wicket penalty...")
    raa_df       = compute_batting_raa(merged)
    season_stats = aggregate_to_season(raa_df)

    print("Computing WAR...")
    war_df = compute_batting_war(season_stats)

    return war_df


# ---------------------------------------------------------------------------
# 8. QUICK VALIDATION
# ---------------------------------------------------------------------------

def validate(war_df: pd.DataFrame, season: int = None) -> None:
    """
    Print top/bottom 10 for a season plus basic sanity checks.
    A well-calibrated model should show:
      - 3-5 players above 2.0 WAR per season (franchise-level contributors)
      - ~80% of qualifiers above replacement by construction
    """
    df           = war_df if season is None else war_df[war_df["season"] == season]
    season_label = season or "all seasons"

    cols = ["season", "striker", "total_runs", "total_balls", "total_disms",
            "batting_sr", "wicket_cost", "batting_raa", "batting_war"]

    print(f"\n=== Top 10 batting WAR — {season_label} ===")
    print(df[cols].head(10).to_string(index=False))

    print(f"\n=== Bottom 10 batting WAR — {season_label} ===")
    print(df[cols].tail(10).to_string(index=False))

    above = (df["batting_war"] > 0).mean()
    print(f"\nPlayers above replacement : {above:.1%}")
    print(f"Mean WAR                  : {df['batting_war'].mean():.2f}")
    print(f"Max WAR                   : {df['batting_war'].max():.2f}")
    print(f"Avg wicket cost per player: {df['wicket_cost'].mean():.1f} runs")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    war_df = run_pipeline()
    # Default to the most recent season in the data
    latest_season = int(war_df["season"].max())
    validate(war_df, season=latest_season)

    out = Path("ipl_batting_war.csv")
    war_df.to_csv(out, index=False)
    print(f"\nFull results saved to {out}")