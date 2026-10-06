"""
Data layer for the IPL WAR agent.

Two jobs:
  1. Make sure the three WAR CSVs exist (by running the existing pipelines
     in ipl_war_batting.py / ipl_war_bowling.py / ipl_war_fielding.py).
  2. Build one enriched ball-by-ball table — deliveries joined with phase,
     bowler_type (pace/spin) and batter_hand (LHB/RHB) from player_lookup.py
     — that the splits/matchup tools query directly instead of re-parsing
     ~1,200 CricSheet CSVs on every question.

Run directly to (re)build everything: `python data_layer.py [--force]`
"""

import argparse
import sys

import pandas as pd

import config
from player_lookup import get_lookup_df

sys.path.insert(0, str(config.PROJECT_ROOT))

import ipl_war_batting as batting_pipeline
import ipl_war_bowling as bowling_pipeline
import ipl_war_fielding as fielding_pipeline


# ---------------------------------------------------------------------------
# 1. WAR CSVs
# ---------------------------------------------------------------------------

def ensure_batting_war(force: bool = False) -> pd.DataFrame:
    if not force and config.BATTING_WAR_CSV.exists():
        return pd.read_csv(config.BATTING_WAR_CSV)
    print("Computing batting WAR...")
    war_df = batting_pipeline.run_pipeline(config.CSV2_DIR)
    war_df.to_csv(config.BATTING_WAR_CSV, index=False)
    return war_df


def ensure_bowling_war(force: bool = False) -> pd.DataFrame:
    if not force and config.BOWLING_WAR_CSV.exists():
        return pd.read_csv(config.BOWLING_WAR_CSV)
    print("Computing bowling WAR...")
    war_df = bowling_pipeline.run_pipeline(config.CSV2_DIR)
    war_df.to_csv(config.BOWLING_WAR_CSV, index=False)
    return war_df


def ensure_fielding_war(force: bool = False) -> pd.DataFrame:
    if not force and config.FIELDING_WAR_CSV.exists():
        return pd.read_csv(config.FIELDING_WAR_CSV)
    print("Computing fielding WAR...")
    war_df = fielding_pipeline.run_pipeline(config.JSON_DIR, config.CSV2_DIR)
    war_df.to_csv(config.FIELDING_WAR_CSV, index=False)
    return war_df


# ---------------------------------------------------------------------------
# 2. Enriched deliveries (phase + bowler_type + batter_hand)
# ---------------------------------------------------------------------------

def build_enriched_deliveries(force: bool = False) -> pd.DataFrame:
    if not force and config.ENRICHED_DELIVERIES_PARQUET.exists():
        return pd.read_parquet(config.ENRICHED_DELIVERIES_PARQUET)

    print("Loading and cleaning deliveries for enrichment...")
    # bowling_pipeline's clean_deliveries is a superset of batting's: it adds
    # bowler_runs / legal_ball_bowler / is_bowler_wicket on top of the same
    # over_num / phase / season / legal_ball fields batting needs, so one
    # load covers both batting-side and bowling-side splits.
    df = bowling_pipeline.load_deliveries(config.CSV2_DIR)

    # Batting-side fields (mirrors ipl_war_batting.clean_deliveries)
    df["legal_ball"] = ~df["is_wide"]
    df["is_wicket"] = df["wicket_type"].notna() & (df["wicket_type"] != "")

    print("Joining bowler_type / batter_hand lookup...")
    lookup = get_lookup_df()

    batter_lookup = lookup.rename(
        columns={"player": "striker", "batting_hand": "batter_hand"}
    )[["striker", "batter_hand"]]
    df = df.merge(batter_lookup, on="striker", how="left")
    df["batter_hand"] = df["batter_hand"].fillna("unknown")

    bowler_lookup = lookup.rename(
        columns={"player": "bowler", "bowling_type": "bowler_type"}
    )[["bowler", "bowler_type"]]
    df = df.merge(bowler_lookup, on="bowler", how="left")
    df["bowler_type"] = df["bowler_type"].fillna("unknown")

    keep_cols = [
        "match_id", "season", "venue", "innings", "over_num", "phase",
        "batting_team", "bowling_team",
        "striker", "batter_hand", "bowler", "bowler_type",
        "runs_off_bat", "extras", "bowler_runs",
        "legal_ball", "legal_ball_bowler",
        "is_wicket", "is_bowler_wicket", "wicket_type", "player_dismissed",
    ]
    df = df[keep_cols]

    df.to_parquet(config.ENRICHED_DELIVERIES_PARQUET, index=False)
    print(f"  {len(df):,} deliveries -> {config.ENRICHED_DELIVERIES_PARQUET}")
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_all(force: bool = False) -> None:
    ensure_batting_war(force)
    ensure_bowling_war(force)
    ensure_fielding_war(force)
    build_enriched_deliveries(force)
    print("\nData layer ready.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build/refresh the WAR data layer.")
    parser.add_argument("--force", action="store_true",
                         help="Recompute everything even if outputs already exist.")
    args = parser.parse_args()
    build_all(force=args.force)
