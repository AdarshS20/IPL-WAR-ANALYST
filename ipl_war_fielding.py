"""
IPL Fielding WAR — v4 (JSON primary, Statsguru fallback)
=========================================================
Tries CricSheet JSON data first (more granular, offline, phase-aware).
Falls back to ESPNcricinfo Statsguru scraping if JSON data is unavailable.

Primary source — CricSheet JSON (ipl_json/)
  Download: https://cricsheet.org/downloads/ipl_json.zip
  Pros: delivery-level, exact fielder names, phase-aware, offline
  Cons: requires one-time download (~50MB)

Fallback source — ESPNcricinfo Statsguru (scraped live)
  URL: https://stats.espncricinfo.com
  Pros: no download needed, always current
  Cons: season totals only (no phase), slower, needs internet

Fielding events credited
------------------------
  caught           → fielder who held the catch
  stumped          → wicketkeeper
  run out          → all named fielders (split credit equally)
  byes conceded    → wicketkeeper (negative — from CricSheet CSV2)

Run values
----------
  catch / stumping / run-out : 5.0 runs each
  bye conceded               : -0.5 runs per bye

Dependencies: pip install requests beautifulsoup4 pandas numpy
"""

from __future__ import annotations

import json
import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from bs4 import BeautifulSoup


# ---------------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------------

JSON_DIR = Path("ipl_json")
CSV_DIR  = Path("ipl_csv2")

RUNS_PER_WIN           = 16.0
REPLACEMENT_PERCENTILE = 0.20
MIN_MATCHES            = 5

FIELDING_RUN_VALUES = {
    "caught":   5.0,
    "stumped":  5.0,
    "run_out":  5.0,
    "bye":     -0.5,
}

# Dismissal types that credit the BOWLER not the fielder — skip these
SKIP_DISMISSAL_TYPES = {
    "caught and bowled", "bowled", "lbw", "hit wicket",
    "obstructing the field", "handled the ball",
    "retired hurt", "retired out", "timed out",
}

# Statsguru config (fallback only)
STATSGURU_URL = (
    "https://stats.espncricinfo.com/ci/engine/stats/index.html"
    "?class=6;filter=advanced;type=fielding;trophy=117"
    ";template=results;view=innings;size=200;season={season}"
)
STATSGURU_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://stats.espncricinfo.com/",
}
IPL_SEASONS = {y: f"{y};{y}" for y in range(2008, 2027)}


# ===========================================================================
# PRIMARY SOURCE — CricSheet JSON
# ===========================================================================

def parse_match_json(path: Path) -> list[dict]:
    """
    Parse one CricSheet JSON match file.
    Returns a list of fielding event records — one row per (fielder, event).

    CricSheet JSON wicket structure:
      "wickets": [{
          "player_out": "SC Ganguly",
          "kind": "caught",
          "fielders": [{"name": "BB McCullum"}]
      }]
    """
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception as e:
        print(f"  Skipping {path.name}: {e}")
        return []

    info          = data.get("info", {})
    match_id      = path.stem
    dates         = info.get("dates", ["0000-00-00"])
    season        = pd.to_datetime(dates[0]).year
    venue         = info.get("venue", "Unknown")
    teams         = info.get("teams", [])

    records = []

    for innings in data.get("innings", []):
        batting_team  = innings.get("team", "")
        fielding_team = next((t for t in teams if t != batting_team), "")

        for over_data in innings.get("overs", []):
            over_num = over_data.get("over", 0) + 1  # CricSheet is 0-indexed

            for delivery in over_data.get("deliveries", []):

                # --- Byes (keeper penalty) ---
                extras = delivery.get("extras", {})
                byes   = extras.get("byes", 0)
                if byes > 0:
                    records.append({
                        "match_id":     match_id,
                        "season":       season,
                        "venue":        venue,
                        "fielding_team":fielding_team,
                        "fielder":      "KEEPER",
                        "event":        "bye",
                        "run_value":    byes * FIELDING_RUN_VALUES["bye"],
                        "over_num":     over_num,
                    })

                # --- Dismissals ---
                for wicket in delivery.get("wickets", []):
                    kind          = wicket.get("kind", "").lower()
                    fielders      = wicket.get("fielders", [])
                    fielder_names = [f["name"] for f in fielders if "name" in f]

                    if kind in SKIP_DISMISSAL_TYPES:
                        continue

                    if kind == "caught":
                        for fname in fielder_names:
                            records.append({
                                "match_id":     match_id,
                                "season":       season,
                                "venue":        venue,
                                "fielding_team":fielding_team,
                                "fielder":      fname,
                                "event":        "caught",
                                "run_value":    FIELDING_RUN_VALUES["caught"],
                                "over_num":     over_num,
                            })

                    elif kind == "stumped":
                        for fname in fielder_names:
                            records.append({
                                "match_id":     match_id,
                                "season":       season,
                                "venue":        venue,
                                "fielding_team":fielding_team,
                                "fielder":      fname,
                                "event":        "stumped",
                                "run_value":    FIELDING_RUN_VALUES["stumped"],
                                "over_num":     over_num,
                            })

                    elif kind == "run out":
                        if fielder_names:
                            share = FIELDING_RUN_VALUES["run_out"] / len(fielder_names)
                            for fname in fielder_names:
                                records.append({
                                    "match_id":     match_id,
                                    "season":       season,
                                    "venue":        venue,
                                    "fielding_team":fielding_team,
                                    "fielder":      fname,
                                    "event":        "run_out",
                                    "run_value":    share,
                                    "over_num":     over_num,
                                })
    return records


def load_json_source(json_dir: Path) -> pd.DataFrame:
    """Load and parse all CricSheet JSON match files."""
    files = sorted(json_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON files found in '{json_dir}'.")

    all_records = []
    for path in files:
        all_records.extend(parse_match_json(path))

    if not all_records:
        raise ValueError("No fielding events parsed from JSON files.")

    return pd.DataFrame(all_records)


def resolve_keepers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace 'KEEPER' placeholder in bye records with the actual keeper name.
    Keeper identified as the player with most stumpings per (season, team).
    """
    stumpings = df[df["event"] == "stumped"]
    keeper_map = (
        stumpings.groupby(["season", "fielding_team", "fielder"])
        .size()
        .reset_index(name="n")
        .sort_values("n", ascending=False)
        .drop_duplicates(subset=["season", "fielding_team"])
        .set_index(["season", "fielding_team"])["fielder"]
        .to_dict()
    )

    def get_keeper(row):
        if row["fielder"] != "KEEPER":
            return row["fielder"]
        return keeper_map.get((row["season"], row["fielding_team"]), "KEEPER (Unknown)")

    df = df.copy()
    df["fielder"] = df.apply(get_keeper, axis=1)
    return df


def aggregate_json(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate delivery-level fielding events to player-season totals."""
    counts = (
        df.groupby(["season", "fielder", "event"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for col in ["caught", "stumped", "run_out", "bye"]:
        if col not in counts.columns:
            counts[col] = 0

    totals = (
        df.groupby(["season", "fielder"])
        .agg(
            fielding_runs=("run_value", "sum"),
            matches      =("match_id",  "nunique"),
        )
        .reset_index()
    )
    return totals.merge(counts, on=["season", "fielder"], how="left").rename(
        columns={"fielder": "player"}
    )


# ===========================================================================
# FALLBACK SOURCE — ESPNcricinfo Statsguru
# ===========================================================================

def scrape_season_statsguru(year: int, season_param: str,
                             retries: int = 3) -> pd.DataFrame:
    """Scrape one IPL season from Statsguru."""
    url = STATSGURU_URL.format(season=season_param)

    for attempt in range(retries):
        try:
            r = requests.get(url, headers=STATSGURU_HEADERS, timeout=20)
            if r.status_code == 200:
                break
            print(f"  [{year}] HTTP {r.status_code} — retrying ({attempt+1}/{retries})")
            time.sleep(2 ** attempt)
        except requests.RequestException as e:
            print(f"  [{year}] Request error: {e} — retrying")
            time.sleep(2 ** attempt)
    else:
        print(f"  [{year}] Failed — skipping")
        return pd.DataFrame()

    soup   = BeautifulSoup(r.text, "html.parser")
    tables = soup.find_all("table", class_="engineTable")

    for table in tables:
        rows = table.find_all("tr")
        if len(rows) < 2:
            continue
        headers = [h.text.strip() for h in rows[0].find_all(["th", "td"])]
        fielding_indicators = {"Ct", "St", "RO", "Catches", "Stumpings"}
        if not ("Player" in headers and fielding_indicators & set(headers)):
            continue

        data = []
        for row in rows[1:]:
            cells = [td.text.strip() for td in row.find_all("td")]
            if not cells:
                continue
            if len(cells) < len(headers):
                cells += [""] * (len(headers) - len(cells))
            elif len(cells) > len(headers):
                cells = cells[:len(headers)]
            row_dict = dict(zip(headers, cells))
            if row_dict.get("Player", "").strip() in ("", "Player"):
                continue
            data.append(row_dict)

        if data:
            df = pd.DataFrame(data)
            df["season"] = year
            print(f"  [{year}] {len(df)} players scraped from Statsguru")
            return df

    print(f"  [{year}] Could not parse Statsguru table")
    return pd.DataFrame()


def scrape_all_statsguru(seasons: dict = IPL_SEASONS,
                          delay: float = 1.5) -> pd.DataFrame:
    """Scrape all IPL seasons from Statsguru."""
    frames = []
    for year, param in sorted(seasons.items()):
        df = scrape_season_statsguru(year, param)
        if not df.empty:
            frames.append(df)
        time.sleep(delay)

    if not frames:
        raise ValueError("No data scraped from Statsguru.")
    return pd.concat(frames, ignore_index=True)


def clean_statsguru(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise Statsguru column names and types."""
    print(f"  Statsguru columns: {list(df.columns)}")
    col_map = {
        "Player": "player", "Mat": "matches", "Matches": "matches",
        "Ct": "catches_total", "St": "stumpings", "Stumpings": "stumpings",
        "Ct Wk": "ct_wk", "Ct wk": "ct_wk", "CtWk": "ct_wk",
        "Ct non-Wk": "ct_nonwk", "Ct non-wk": "ct_nonwk",
        "RO": "run_outs", "Run outs": "run_outs",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    num_cols = ["matches", "catches_total", "stumpings",
                "ct_wk", "ct_nonwk", "run_outs"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace("-", "0").str.strip(),
                errors="coerce"
            ).fillna(0).astype(int)

    if "player" in df.columns:
        df["player"] = df["player"].str.split("(").str[0].str.strip()

    if "ct_wk" not in df.columns and "ct_nonwk" not in df.columns:
        df["ct_wk"]    = 0
        df["ct_nonwk"] = df.get("catches_total", pd.Series(0, index=df.index))
    elif "ct_wk" not in df.columns:
        df["ct_wk"] = 0
    elif "ct_nonwk" not in df.columns:
        ct = df.get("catches_total", df["ct_wk"])
        df["ct_nonwk"] = (ct - df["ct_wk"]).clip(lower=0)

    for col in ["run_outs", "stumpings", "matches"]:
        if col not in df.columns:
            df[col] = 0

    # Convert Statsguru counts to fielding_runs
    df["fielding_runs"] = (
        df.get("ct_wk",    0) * FIELDING_RUN_VALUES["caught"]
      + df.get("ct_nonwk", 0) * FIELDING_RUN_VALUES["caught"]
      + df.get("stumpings",0) * FIELDING_RUN_VALUES["stumped"]
      + df.get("run_outs", 0) * FIELDING_RUN_VALUES["run_out"]
    )

    required = ["season", "player", "matches", "fielding_runs"]
    for col in required:
        if col not in df.columns:
            df[col] = 0
    return df[required].copy()


# ===========================================================================
# SHARED — WAR COMPUTATION
# ===========================================================================

def compute_fielding_war(season_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Replacement level = bottom REPLACEMENT_PERCENTILE of fielding runs
    per match among qualified players (>= MIN_MATCHES).
    WAR = (fielding_runs - replacement_runs) / RUNS_PER_WIN
    """
    qualified = season_stats[season_stats["matches"] >= MIN_MATCHES].copy()

    if qualified.empty:
        print("  WARNING: No players met the minimum matches threshold.")
        return qualified

    replacement_rate = (
        qualified["fielding_runs"].quantile(REPLACEMENT_PERCENTILE)
        / qualified["matches"].mean()
    )

    qualified["replacement_runs"]       = qualified["matches"] * replacement_rate
    qualified["runs_above_replacement"] = qualified["fielding_runs"] - qualified["replacement_runs"]
    qualified["fielding_war"]           = qualified["runs_above_replacement"] / RUNS_PER_WIN

    return qualified.sort_values("fielding_war", ascending=False).reset_index(drop=True)


# ===========================================================================
# MAIN PIPELINE — tries JSON first, falls back to Statsguru
# ===========================================================================

def run_pipeline(json_dir: Path = JSON_DIR,
                 csv_dir:  Path = CSV_DIR) -> pd.DataFrame:
    """
    Full pipeline with automatic source selection:
      1. If ipl_json/ exists and has .json files → use CricSheet JSON (primary)
      2. Otherwise → scrape ESPNcricinfo Statsguru (fallback)
    """

    # ── Try primary source: CricSheet JSON ──────────────────────────────────
    if json_dir.exists() and any(json_dir.glob("*.json")):
        print(f"CricSheet JSON data found in '{json_dir}' — using primary source.")
        print("Parsing match files...")
        raw_df = load_json_source(json_dir)
        print(f"  {len(raw_df):,} fielding events | "
              f"seasons: {sorted(raw_df['season'].unique())}")

        print("Resolving wicketkeepers...")
        raw_df = resolve_keepers(raw_df)

        print("Aggregating to player-season...")
        season_stats = aggregate_json(raw_df)

        print("Computing fielding WAR...")
        return compute_fielding_war(season_stats)

    # ── Fallback: Statsguru scraper ──────────────────────────────────────────
    print(f"'{json_dir}' not found — falling back to ESPNcricinfo Statsguru.")
    print("Tip: download ipl_json.zip from https://cricsheet.org/downloads/")
    print("     and extract to 'ipl_json/' for better fielding data.\n")

    print("Scraping Statsguru fielding stats (this takes ~30 seconds)...")
    raw_df = scrape_all_statsguru()

    print("Cleaning Statsguru data...")
    season_stats = clean_statsguru(raw_df)

    print("Computing fielding WAR...")
    return compute_fielding_war(season_stats)


# ===========================================================================
# VALIDATION
# ===========================================================================

def validate(war_df: pd.DataFrame, season: int = None) -> None:
    df           = war_df if season is None else war_df[war_df["season"] == season]
    season_label = season or "all seasons"

    if df.empty:
        print(f"No data for {season_label}.")
        return

    # Show whatever columns exist
    preferred = ["season", "player", "matches", "caught", "stumped",
                 "run_out", "bye", "fielding_runs", "fielding_war"]
    cols = [c for c in preferred if c in df.columns]

    print(f"\n{'='*70}")
    print(f"  TOP 15 — Fielding WAR — {season_label}")
    print(f"{'='*70}")
    print(df[cols].head(15).to_string(index=False))

    print(f"\n{'='*70}")
    print(f"  BOTTOM 10 — Fielding WAR — {season_label}")
    print(f"{'='*70}")
    print(df[cols].tail(10).to_string(index=False))

    above = (df["fielding_war"] > 0).mean()
    print(f"\nPlayers above replacement : {above:.1%}")
    print(f"Mean fielding WAR         : {df['fielding_war'].mean():.2f}")
    print(f"Max fielding WAR          : {df['fielding_war'].max():.2f}  "
          f"({df.iloc[0]['player']})")

    if "stumped" in df.columns:
        keepers  = df[df["stumped"] > 0]
        outfield = df[df["stumped"] == 0]
        print(f"\nKeepers   : {len(keepers)} | avg WAR = {keepers['fielding_war'].mean():.2f}")
        print(f"Outfield  : {len(outfield)} | avg WAR = {outfield['fielding_war'].mean():.2f}")


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    war_df = run_pipeline(json_dir=JSON_DIR, csv_dir=CSV_DIR)
    latest_season = int(war_df["season"].max())
    validate(war_df, season=latest_season)

    out = Path("ipl_fielding_war.csv")
    war_df.to_csv(out, index=False)
    print(f"\nFielding WAR saved to {out}")