#!/usr/bin/env python3
"""
build_dataset.py -- builds the Dropped / Missed Interception (DINT) dataset
from open nflverse data (play-by-play + FTN charting + rosters).

Usage:
    python3 build_dataset.py --raw raw --out data [--download] [--seasons 2022 2026]

Definitions (see paper Section 3):
  INT  : interception credited to a defender in nflverse play-by-play.
  MIC  : Missed-INT Chance. A pass attempt that FTN charters flagged
         `is_interception_worthy`, that was NOT intercepted, was incomplete,
         and in which a DB or LB was credited with a pass defense
         (pass_defense_1/2_player_id). This is a PROXY for a dropped
         interception: it also contains legitimate break-ups on throws that
         were interception-worthy. It is an upper bound on true drops.
  Chances      = INT + MIC
  convert_rate = INT / Chances
  mic_rate     = MIC / Chances   ("drop-proxy rate")
"""
import argparse, os, sys, urllib.request
import numpy as np
import pandas as pd
from scipy import optimize, special

BASE = "https://github.com/nflverse/nflverse-data/releases/download"
POS_GROUP = {
    "CB": "CB", "DB": "DB", "S": "S", "FS": "S", "SS": "S",
    "LB": "LB", "ILB": "LB", "OLB": "LB", "MLB": "LB",
    "DE": "DL", "DT": "DL", "NT": "DL", "DL": "DL",
}
COVERAGE_GROUPS = {"CB", "DB", "S", "LB"}   # eligible for MIC credit


def download(raw, seasons):
    os.makedirs(raw, exist_ok=True)
    for y in seasons:
        for name, url in [
            (f"pbp_{y}.csv.gz", f"{BASE}/pbp/play_by_play_{y}.csv.gz"),
            (f"ftn_{y}.csv", f"{BASE}/ftn_charting/ftn_charting_{y}.csv"),
            (f"roster_{y}.csv", f"{BASE}/rosters/roster_{y}.csv"),
        ]:
            path = os.path.join(raw, name)
            if not os.path.exists(path):
                print("downloading", url)
                urllib.request.urlretrieve(url, path)


def load_season(raw, y):
    p = pd.read_csv(
        os.path.join(raw, f"pbp_{y}.csv.gz"), low_memory=False,
        usecols=["game_id", "play_id", "season", "season_type", "week", "posteam",
                 "defteam", "pass_attempt", "sack", "qb_spike", "interception",
                 "complete_pass", "incomplete_pass", "interception_player_id",
                 "interception_player_name", "pass_defense_1_player_id",
                 "pass_defense_1_player_name", "pass_defense_2_player_id",
                 "pass_defense_2_player_name", "passer_player_name",
                 "receiver_player_name", "air_yards", "desc"])
    f = pd.read_csv(
        os.path.join(raw, f"ftn_{y}.csv"),
        usecols=["nflverse_game_id", "nflverse_play_id", "is_interception_worthy",
                 "is_catchable_ball", "is_contested_ball", "is_drop"])
    m = p.merge(f, left_on=["game_id", "play_id"],
                right_on=["nflverse_game_id", "nflverse_play_id"],
                how="left", validate="1:1")
    m["ftn_charted"] = m["nflverse_game_id"].notna()
    for c in ["is_interception_worthy", "is_catchable_ball", "is_contested_ball", "is_drop"]:
        m[c] = m[c].fillna(False).astype(bool)
    r = pd.read_csv(os.path.join(raw, f"roster_{y}.csv"),
                    usecols=["gsis_id", "full_name", "position", "depth_chart_position", "team"]).dropna(subset=["gsis_id"])
    # depth_chart_position separates CB / FS / SS / ILB / OLB; fall back to coarse position
    r["position"] = r["depth_chart_position"].fillna(r["position"])
    r["pos_group"] = r["position"].map(POS_GROUP)
    return m, r


def build_events(pbp, roster_pos):
    """Long table: one row per INT or MIC credit."""
    pa = pbp[(pbp.pass_attempt == 1) & (pbp.sack != 1) & (pbp.qb_spike != 1)].copy()
    common = ["game_id", "play_id", "season", "season_type", "week", "posteam", "defteam",
              "passer_player_name", "receiver_player_name", "air_yards", "desc",
              "is_interception_worthy", "is_catchable_ball", "is_contested_ball"]

    ints = pa[pa.interception == 1].copy()
    ints["event_type"] = "INT"
    ints["player_id"] = ints["interception_player_id"]
    ints["player_name_pbp"] = ints["interception_player_name"]
    ints = ints.dropna(subset=["player_id"])

    mic_base = pa[pa.is_interception_worthy & (pa.interception != 1) & (pa.incomplete_pass == 1)]
    mics = []
    for slot in ("1", "2"):
        t = mic_base.dropna(subset=[f"pass_defense_{slot}_player_id"]).copy()
        t["player_id"] = t[f"pass_defense_{slot}_player_id"]
        t["player_name_pbp"] = t[f"pass_defense_{slot}_player_name"]
        t["event_type"] = "MIC"
        t["pd_slot"] = int(slot)
        mics.append(t)
    mics = pd.concat(mics)

    ev = pd.concat([ints, mics], ignore_index=True)
    ev = ev.merge(roster_pos[["player_id", "season", "pos_group", "position"]],
                  on=["player_id", "season"], how="left")
    # defender-side players with no roster match: leave pos_group NaN -> excluded from MIC
    # DL are excluded entirely: their pass-defense credits are batted balls, and their INTs have
    # no matching "chance" denominator, so a conversion rate is undefined for them.
    n_dl = int((ev.pos_group == "DL").sum())
    n_unk = int(ev.pos_group.isna().sum())
    keep = ev.pos_group.isin(COVERAGE_GROUPS)
    ev = ev[keep].copy()
    ev["is_int_worthy_int"] = (ev.event_type == "INT") & ev.is_interception_worthy
    cols = ["season", "season_type", "week", "game_id", "play_id", "event_type", "player_id",
            "player_name_pbp", "position", "pos_group", "defteam", "posteam", "passer_player_name",
            "receiver_player_name", "air_yards", "is_catchable_ball", "is_contested_ball", "desc"]
    return ev[cols].reset_index(drop=True), n_dl, n_unk


def roster_positions(rosters):
    r = pd.concat(rosters, ignore_index=True)
    r = r.dropna(subset=["position"])
    out = (r.groupby(["gsis_id", "season"])
             .agg(position=("position", lambda s: s.mode().iat[0]),
                  full_name=("full_name", lambda s: s.mode().iat[0]),
                  roster_team=("team", lambda s: s.mode().iat[0]))
             .reset_index().rename(columns={"gsis_id": "player_id"}))
    out["pos_group"] = out["position"].map(POS_GROUP)
    return out


def fit_beta_binomial(k, n):
    """MLE of Beta(a,b) prior for k successes out of n trials (n>0)."""
    k = np.asarray(k, float); n = np.asarray(n, float)

    def nll(theta):
        a, b = np.exp(theta)
        ll = (special.gammaln(n + 1) - special.gammaln(k + 1) - special.gammaln(n - k + 1)
              + special.betaln(k + a, n - k + b) - special.betaln(a, b))
        return -ll.sum()
    res = optimize.minimize(nll, x0=np.log([2.0, 2.0]), method="Nelder-Mead")
    return tuple(np.exp(res.x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="raw")
    ap.add_argument("--out", default="data")
    ap.add_argument("--first", type=int, default=2022)
    ap.add_argument("--last", type=int, default=2026)
    ap.add_argument("--download", action="store_true")
    a = ap.parse_args()
    seasons = list(range(a.first, a.last + 1))
    if a.download:
        download(a.raw, seasons)
    os.makedirs(a.out, exist_ok=True)

    pbps, rosters = [], []
    for y in seasons:
        m, r = load_season(a.raw, y)
        r["season"] = y
        pbps.append(m); rosters.append(r)
    pbp = pd.concat(pbps, ignore_index=True)
    rp = roster_positions(rosters)

    events, n_dl, n_unk = build_events(pbp, rp)
    events.to_csv(os.path.join(a.out, "dint_events.csv"), index=False)

    # ---------- player-season (regular season only) ----------
    reg = events[events.season_type == "REG"]
    g = reg.groupby(["player_id", "season"])
    ps = g.agg(
        int_=("event_type", lambda s: (s == "INT").sum()),
        mic=("event_type", lambda s: (s == "MIC").sum()),
        team=("defteam", lambda s: s.mode().iat[0]),
        teams=("defteam", lambda s: "/".join(sorted(set(s)))),
    ).reset_index().rename(columns={"int_": "int"})
    ps = ps.merge(rp[["player_id", "season", "full_name", "position", "pos_group"]],
                  on=["player_id", "season"], how="left")
    names = reg.groupby("player_id")["player_name_pbp"].agg(lambda s: s.mode().iat[0])
    ps["player"] = ps["full_name"].fillna(ps["player_id"].map(names))
    ps["pos_group"] = ps["pos_group"].fillna("UNK")
    ps["chances"] = ps["int"] + ps["mic"]
    ps["convert_rate"] = ps["int"] / ps["chances"]
    ps["mic_rate"] = ps["mic"] / ps["chances"]
    ps["partial_season"] = ps["season"] == pbp["season"].max()

    # league conversion rate per season x pos group (for INT-over-expected)
    lg = (ps.groupby(["season", "pos_group"])
            .apply(lambda d: d["int"].sum() / d["chances"].sum(), include_groups=False)
            .rename("lg_convert_rate").reset_index())
    ps = ps.merge(lg, on=["season", "pos_group"], how="left")
    ps["exp_int"] = ps["chances"] * ps["lg_convert_rate"]
    ps["int_over_exp"] = ps["int"] - ps["exp_int"]

    # empirical-Bayes shrunk conversion rate (beta-binomial, per position group, complete seasons)
    complete = ps[~ps.partial_season]
    prior = {}
    for grp, d in complete.groupby("pos_group"):
        if len(d) < 30:
            continue
        prior[grp] = fit_beta_binomial(d["int"], d["chances"])
    pooled = fit_beta_binomial(complete["int"], complete["chances"])
    def shrink(row):
        a_, b_ = prior.get(row.pos_group, pooled)
        return (a_ + row["int"]) / (a_ + b_ + row["chances"])
    ps["convert_rate_eb"] = ps.apply(shrink, axis=1)
    ps["mic_rate_eb"] = 1 - ps["convert_rate_eb"]

    ps = ps.drop(columns=["full_name"]).sort_values(["season", "chances"], ascending=[True, False])
    cols = ["player_id", "player", "position", "pos_group", "season", "team", "teams", "int", "mic",
            "chances", "convert_rate", "mic_rate", "convert_rate_eb", "mic_rate_eb",
            "exp_int", "int_over_exp", "lg_convert_rate", "partial_season"]
    ps[cols].round(4).to_csv(os.path.join(a.out, "dint_player_seasons.csv"), index=False)

    # ---------- career (2022-latest) ----------
    car = (ps.groupby("player_id")
             .agg(player=("player", "last"), position=("position", "last"),
                  pos_group=("pos_group", "last"), last_team=("team", "last"),
                  seasons=("season", "nunique"), int=("int", "sum"), mic=("mic", "sum"),
                  int_over_exp=("int_over_exp", "sum")).reset_index())
    car["chances"] = car["int"] + car["mic"]
    car["convert_rate"] = car["int"] / car["chances"]
    car["mic_rate"] = car["mic"] / car["chances"]
    a_, b_ = pooled
    car["convert_rate_eb"] = [(prior.get(g_, pooled)[0] + i) / (sum(prior.get(g_, pooled)) + c)
                              for g_, i, c in zip(car.pos_group, car["int"], car["chances"])]
    car.sort_values("chances", ascending=False).round(4).to_csv(
        os.path.join(a.out, "dint_player_career.csv"), index=False)

    # ---------- team-season ----------
    ts = (reg.groupby(["defteam", "season"])
             .agg(int=("event_type", lambda s: (s == "INT").sum()),
                  mic=("event_type", lambda s: (s == "MIC").sum())).reset_index()
             .rename(columns={"defteam": "team"}))
    ts["chances"] = ts["int"] + ts["mic"]
    ts["convert_rate"] = ts["int"] / ts["chances"]
    ts.round(4).to_csv(os.path.join(a.out, "dint_team_seasons.csv"), index=False)

    # ---------- chart-quality audit ----------
    pa = pbp[(pbp.pass_attempt == 1) & (pbp.sack != 1) & (pbp.qb_spike != 1) & (pbp.season_type == "REG")]
    qa = (pa.groupby("season").agg(pass_attempts=("play_id", "size"),
                                   ftn_coverage=("ftn_charted", "mean"),
                                   int_worthy=("is_interception_worthy", "sum"),
                                   interceptions=("interception", "sum")).reset_index())
    qa["int_worthy_rate"] = qa["int_worthy"] / qa["pass_attempts"]
    qa["int_rate"] = qa["interceptions"] / qa["pass_attempts"]
    qa.round(5).to_csv(os.path.join(a.out, "dint_qa_by_season.csv"), index=False)

    pd.DataFrame([{"pos_group": k, "alpha": v[0], "beta": v[1],
                   "prior_mean": v[0] / (v[0] + v[1]), "prior_strength": v[0] + v[1]}
                  for k, v in {**prior, "ALL": pooled}.items()]).round(4).to_csv(
        os.path.join(a.out, "dint_eb_priors.csv"), index=False)

    print(f"events: {len(events):,}  player-seasons: {len(ps):,}  players: {car.shape[0]:,}")
    print(f"event credits excluded (INT or pass-defense): DL={n_dl}, unmatched-position={n_unk}")
    print(qa.to_string(index=False))
    print("EB priors:", {k: tuple(round(x, 2) for x in v) for k, v in {**prior, 'ALL': pooled}.items()})


if __name__ == "__main__":
    sys.exit(main())
