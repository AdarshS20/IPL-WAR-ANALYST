"""Shared paths and constants for the WAR data layer, tools, and agent."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

CSV2_DIR = PROJECT_ROOT / "ipl_csv2"
JSON_DIR = PROJECT_ROOT / "ipl_json"

BATTING_WAR_CSV  = PROJECT_ROOT / "ipl_batting_war.csv"
BOWLING_WAR_CSV  = PROJECT_ROOT / "ipl_bowling_war.csv"
FIELDING_WAR_CSV = PROJECT_ROOT / "ipl_fielding_war.csv"

ENRICHED_DELIVERIES_PARQUET = PROJECT_ROOT / "ipl_deliveries_enriched.parquet"

# Must match ipl_war_batting.py / ipl_war_bowling.py
RUNS_PER_WIN = 10.0
MIN_BALLS    = 60
