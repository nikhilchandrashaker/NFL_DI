#!/usr/bin/env python3
"""checks.py -- internal-consistency checks on the generated data. Exits non-zero on failure.
Usage: python3 checks.py --data data --raw raw"""
import argparse, os, sys
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data"); ap.add_argument("--raw", default="raw")
a = ap.parse_args()
ev = pd.read_csv(os.path.join(a.data, "dint_events.csv"))
ps = pd.read_csv(os.path.join(a.data, "dint_player_seasons.csv"))
ts = pd.read_csv(os.path.join(a.data, "dint_team_seasons.csv"))
reg = ev[ev.season_type == "REG"]
ok = True


def check(name, cond, detail=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name + (f"  {detail}" if detail else ""))
    ok &= bool(cond)


check("player-season INT sum equals REG INT events", ps["int"].sum() == (reg.event_type == "INT").sum(),
      f"{ps['int'].sum()} vs {(reg.event_type == 'INT').sum()}")
check("player-season MIC sum equals REG MIC events", ps["mic"].sum() == (reg.event_type == "MIC").sum())
check("team-season sums equal player-season sums", ts["int"].sum() == ps["int"].sum() and ts["mic"].sum() == ps["mic"].sum())
check("chances = int + mic", (ps.chances == ps["int"] + ps.mic).all())
check("convert rate within [0,1]", ps.convert_rate.between(0, 1).all() and ps.convert_rate_eb.between(0, 1).all())
check("no DL in rated tables", not (ps.pos_group == "DL").any())
check("every MIC event is on a DB/LB", ev[ev.event_type == "MIC"].pos_group.isin(["CB", "S", "DB", "LB"]).all())
check("no duplicate player-seasons", not ps.duplicated(["player_id", "season"]).any())

# reconcile INT events with raw play-by-play interception counts, by season
for y in sorted(ps.season.unique()):
    p = pd.read_csv(os.path.join(a.raw, f"pbp_{y}.csv.gz"), low_memory=False,
                    usecols=["season_type", "pass_attempt", "sack", "qb_spike", "interception", "interception_player_id"])
    p = p[(p.season_type == "REG") & (p.pass_attempt == 1) & (p.sack != 1) & (p.qb_spike != 1) & (p.interception == 1)]
    n_raw = int(p.interception_player_id.notna().sum())
    n_ours = int(((reg.season == y) & (reg.event_type == "INT")).sum())
    check(f"{y}: INT events vs raw credited INTs", n_ours <= n_raw and n_raw - n_ours < 0.12 * n_raw,
          f"ours {n_ours}, raw {n_raw} (difference = DL and unmatched-position INTs)")

# ---------------- long-run checks ----------------
L = os.path.join(a.data, "longrun")
lps = pd.read_csv(os.path.join(L, "lr_player_seasons.csv")); lts = pd.read_csv(os.path.join(L, "lr_team_seasons.csv"))
lev = pd.read_csv(os.path.join(L, "lr_events.csv.gz")); lreg = lev[lev.season_type == "REG"]
check("long-run: player-season INT/PD sums equal REG events",
      lps["int"].sum() == (lreg.event_type == "INT").sum() and lps["pd"].sum() == (lreg.event_type == "PD").sum(),
      f"INT {lps['int'].sum()}, PD {lps['pd'].sum()}")
check("long-run: team sums equal player sums", lts["int"].sum() == lps["int"].sum() and lts["pd"].sum() == lps["pd"].sum())
check("long-run: no duplicate player-seasons", not lps.duplicated(["player_id", "season"]).any())
check("long-run: no DL rows", not (lps.pos_group == "DL").any())
check("long-run: covers 1999 through latest", lps.season.min() == 1999 and lps.season.max() >= 2026)
m = ps[~ps.partial_season].merge(lps[["player_id", "season", "int"]], on=["player_id", "season"], suffixes=("", "_l"))
check("overlap 2022-2025: INT counts identical in both measures", (m["int"] == m.int_l).all(), f"{len(m)} player-seasons")
for y in sorted(lps.season.unique()):
    p = pd.read_csv(os.path.join(a.raw, f"pbp_{y}.csv.gz"), low_memory=False,
                    usecols=["season_type", "pass_attempt", "sack", "qb_spike", "interception", "interception_player_id"])
    p = p[(p.season_type == "REG") & (p.pass_attempt == 1) & (p.sack != 1) & (p.qb_spike != 1) & (p.interception == 1)]
    n_raw = int(p.interception_player_id.notna().sum()); n_l = int(lps[lps.season == y]["int"].sum())
    check(f"long-run {y}: INT vs raw credited", n_l <= n_raw and n_l >= 0.88 * n_raw, f"ours {n_l}, raw {n_raw}")
sys.exit(0 if ok else 1)
