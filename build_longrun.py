#!/usr/bin/env python3
"""
build_longrun.py -- long-run (1999-latest) companion measure, built from play-by-play + rosters only.

FTN's interception-worthy flag exists only from 2022, so MIC cannot be computed earlier. This script
builds the best measure available for every season since 1999:

  INT   : interception credited to a defender.
  PD    : pass defense credited to a DB or LB on an incomplete pass that was not intercepted
          (pass_defense_1 and pass_defense_2 slots; one credit per credited defender).
  PD chances  = INT + PD
  int_share   = INT / (INT + PD)       ("how often a ball-hawking play ends in a catch")

PD is a much looser stand-in for "had a chance" than MIC: most passes defended were never catchable.
So int_share is NOT comparable to MIC conversion in level (about 17% vs about 60%). Section 8 of the paper
validates it against MIC on the 2022-2025 overlap.

Usage: python3 build_longrun.py --raw raw --out data/longrun --first 1999 --last 2026
"""
import argparse, os, sys
import numpy as np, pandas as pd
from scipy import optimize, special

POS_GROUP = {"CB": "CB", "DB": "DB", "S": "S", "FS": "S", "SS": "S", "SAF": "S",
             "LB": "LB", "ILB": "LB", "OLB": "LB", "MLB": "LB",
             "DE": "DL", "DT": "DL", "NT": "DL", "DL": "DL"}
COVERAGE_GROUPS = {"CB", "DB", "S", "LB"}


def mode0(s):
    m = s.dropna().mode()
    return m.iat[0] if len(m) else None


def load_roster(raw, y):
    r = pd.read_csv(os.path.join(raw, f"roster_{y}.csv"))
    r = r.dropna(subset=["gsis_id"])
    pos = r["depth_chart_position"] if "depth_chart_position" in r else pd.Series(index=r.index, dtype=object)
    r["pos"] = pos.fillna(r["position"])
    out = (r.groupby("gsis_id").agg(position=("pos", mode0), name=("full_name", mode0))
             .reset_index().rename(columns={"gsis_id": "player_id"}))
    out["pos_group"] = out["position"].map(POS_GROUP)
    out["season"] = y
    return out


def load_events(raw, y, ros):
    p = pd.read_csv(os.path.join(raw, f"pbp_{y}.csv.gz"), low_memory=False,
                    usecols=["game_id", "play_id", "season", "season_type", "week", "posteam", "defteam", "pass_attempt",
                             "sack", "qb_spike", "interception", "incomplete_pass", "interception_player_id",
                             "pass_defense_1_player_id", "pass_defense_2_player_id"])
    pa = p[(p.pass_attempt == 1) & (p.sack != 1) & (p.qb_spike != 1)]
    keep = ["game_id", "play_id", "season", "season_type", "week", "defteam"]
    a = pa[pa.interception == 1].dropna(subset=["interception_player_id"]).copy()
    a["player_id"] = a["interception_player_id"]; a["event_type"] = "INT"
    parts = [a[keep + ["player_id", "event_type"]]]
    b = pa[(pa.interception != 1) & (pa.incomplete_pass == 1)]
    for slot in ("1", "2"):
        t = b.dropna(subset=[f"pass_defense_{slot}_player_id"]).copy()
        t["player_id"] = t[f"pass_defense_{slot}_player_id"]; t["event_type"] = "PD"
        parts.append(t[keep + ["player_id", "event_type"]])
    ev = pd.concat(parts, ignore_index=True)
    n_total = len(ev)
    ev = ev.merge(ros[["player_id", "position", "pos_group", "name"]], on="player_id", how="left")
    n_dl = int((ev.pos_group == "DL").sum()); n_unk = int(ev.pos_group.isna().sum())
    ev = ev[ev.pos_group.isin(COVERAGE_GROUPS)]
    return ev, dict(season=y, credits=n_total, dl=n_dl, unmatched=n_unk, kept=len(ev))


def fit_bb(k, n):
    k = np.asarray(k, float); n = np.asarray(n, float)

    def nll(t):
        a, b = np.exp(t)
        return -(special.gammaln(n + 1) - special.gammaln(k + 1) - special.gammaln(n - k + 1)
                 + special.betaln(k + a, n - k + b) - special.betaln(a, b)).sum()
    return tuple(np.exp(optimize.minimize(nll, np.log([1.0, 4.0]), method="Nelder-Mead").x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="raw"); ap.add_argument("--out", default="data/longrun")
    ap.add_argument("--first", type=int, default=1999); ap.add_argument("--last", type=int, default=2026)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    evs, audit, ros_all = [], [], []
    for y in range(a.first, a.last + 1):
        ros = load_roster(a.raw, y); ros_all.append(ros)
        ev, au = load_events(a.raw, y, ros); evs.append(ev); audit.append(au)
        print(y, au, flush=True)
    ev = pd.concat(evs, ignore_index=True)
    last = ev.season.max()

    slim = ev[["season", "season_type", "week", "game_id", "play_id", "event_type", "player_id", "pos_group", "defteam"]]
    slim.to_csv(os.path.join(a.out, "lr_events.csv.gz"), index=False)
    pd.DataFrame(audit).to_csv(os.path.join(a.out, "lr_credit_audit.csv"), index=False)

    reg = ev[ev.season_type == "REG"]
    g = reg.groupby(["player_id", "season"])
    ps = g.agg(int_=("event_type", lambda s: (s == "INT").sum()), pd_=("event_type", lambda s: (s == "PD").sum()),
               team=("defteam", lambda s: s.mode().iat[0]), pos_group=("pos_group", lambda s: s.mode().iat[0]),
               position=("position", mode0), name=("name", mode0)
               ).reset_index().rename(columns={"int_": "int", "pd_": "pd"})
    ps["chances"] = ps["int"] + ps["pd"]
    ps["int_share"] = ps["int"] / ps["chances"]
    ps["partial_season"] = ps["season"] == last
    lg = (ps.groupby(["season", "pos_group"]).apply(lambda d: d["int"].sum() / d["chances"].sum(), include_groups=False)
            .rename("lg_int_share").reset_index())
    ps = ps.merge(lg, on=["season", "pos_group"])
    ps["exp_int"] = ps["chances"] * ps["lg_int_share"]
    ps["int_over_exp"] = ps["int"] - ps["exp_int"]

    comp = ps[~ps.partial_season]
    prior = {gname: fit_bb(d["int"], d["chances"]) for gname, d in comp.groupby("pos_group") if len(d) >= 200}
    pooled = fit_bb(comp["int"], comp["chances"])
    pri = lambda gname: prior.get(gname, pooled)
    ps["int_share_eb"] = [(pri(g_)[0] + i) / (sum(pri(g_)) + c) for g_, i, c in zip(ps.pos_group, ps["int"], ps.chances)]
    ps = ps.rename(columns={"name": "player"})
    cols = ["player_id", "player", "position", "pos_group", "season", "team", "int", "pd", "chances", "int_share",
            "int_share_eb", "lg_int_share", "exp_int", "int_over_exp", "partial_season"]
    ps.sort_values(["season", "chances"], ascending=[True, False])[cols].round(4).to_csv(
        os.path.join(a.out, "lr_player_seasons.csv"), index=False)

    car = (ps.sort_values("season").groupby("player_id")
             .agg(player=("player", "last"), position=("position", "last"), pos_group=("pos_group", "last"),
                  first_season=("season", "min"), last_season=("season", "max"), last_team=("team", "last"),
                  seasons=("season", "nunique"), int=("int", "sum"), pd=("pd", "sum"), int_over_exp=("int_over_exp", "sum"))
             .reset_index())
    car["chances"] = car["int"] + car["pd"]; car["int_share"] = car["int"] / car["chances"]
    car["int_share_eb"] = [(pri(g_)[0] + i) / (sum(pri(g_)) + c) for g_, i, c in zip(car.pos_group, car["int"], car.chances)]
    car.sort_values("int", ascending=False).round(4).to_csv(os.path.join(a.out, "lr_player_career.csv"), index=False)

    # league-season trend table (all passes, for rates)
    rows = []
    for y in range(a.first, a.last + 1):
        p = pd.read_csv(os.path.join(a.raw, f"pbp_{y}.csv.gz"), low_memory=False,
                        usecols=["season_type", "pass_attempt", "sack", "qb_spike", "interception"])
        p = p[(p.season_type == "REG") & (p.pass_attempt == 1) & (p.sack != 1) & (p.qb_spike != 1)]
        s = ps[ps.season == y]
        rows.append(dict(season=y, pass_attempts=len(p), interceptions=int(p.interception.sum()),
                         int_rate=p.interception.mean(), db_lb_int=int(s["int"].sum()), db_lb_pd=int(s["pd"].sum()),
                         int_share=s["int"].sum() / (s["int"].sum() + s["pd"].sum()), partial_season=y == last))
    pd.DataFrame(rows).round(5).to_csv(os.path.join(a.out, "lr_league_seasons.csv"), index=False)

    ts = (reg.groupby(["defteam", "season"]).agg(int=("event_type", lambda s: (s == "INT").sum()),
                                                 pd=("event_type", lambda s: (s == "PD").sum())).reset_index()
            .rename(columns={"defteam": "team"}))
    ts["chances"] = ts["int"] + ts["pd"]; ts["int_share"] = ts["int"] / ts["chances"]
    ts.round(4).to_csv(os.path.join(a.out, "lr_team_seasons.csv"), index=False)
    pd.DataFrame([{"pos_group": k, "alpha": v[0], "beta": v[1], "prior_mean": v[0] / sum(v), "prior_strength": sum(v)}
                  for k, v in {**prior, "ALL": pooled}.items()]).round(4).to_csv(os.path.join(a.out, "lr_eb_priors.csv"), index=False)
    print("player-seasons", len(ps), "players", len(car), "REG events", len(reg))
    print(ps.pos_group.value_counts().to_dict())
    print({k: tuple(round(x, 2) for x in v) for k, v in {**prior, "ALL": pooled}.items()})


if __name__ == "__main__":
    sys.exit(main())
