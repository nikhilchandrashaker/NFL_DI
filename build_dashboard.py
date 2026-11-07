#!/usr/bin/env python3
"""build_dashboard.py -- embeds the CSVs into one self-contained HTML dashboard.
Usage: python3 build_dashboard.py --data data --raw raw --results results.json --out dashboard
Writes dashboard/index.html (standalone) and dashboard/page_fragment.html (for hosts that add their own skeleton).
"""
import argparse, json, os
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data"); ap.add_argument("--raw", default="raw")
    ap.add_argument("--results", default="results.json"); ap.add_argument("--lr-results", default="results_longrun.json"); ap.add_argument("--out", default="dashboard")
    ap.add_argument("--template", default=os.path.join(os.path.dirname(__file__), "dashboard_template.html"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    R = json.load(open(a.results)); RL = json.load(open(a.lr_results))
    lps = pd.read_csv(os.path.join(a.data, "longrun", "lr_player_seasons.csv"))
    lts = pd.read_csv(os.path.join(a.data, "longrun", "lr_team_seasons.csv"))
    lpri = pd.read_csv(os.path.join(a.data, "longrun", "lr_eb_priors.csv"))
    ps = pd.read_csv(os.path.join(a.data, "dint_player_seasons.csv"))
    ev = pd.read_csv(os.path.join(a.data, "dint_events.csv"))
    ts = pd.read_csv(os.path.join(a.data, "dint_team_seasons.csv"))
    pri = pd.read_csv(os.path.join(a.data, "dint_eb_priors.csv"))
    ev = ev[ev.season_type == "REG"]

    last = pd.read_csv(os.path.join(a.raw, "pbp_2026.csv.gz"), usecols=["week", "season_type"])
    last_week = int(last[last.season_type == "REG"].week.max())

    P = [dict(id=r.player_id, n=r.player, pos=r.position, g=r.pos_group, s=int(r.season), t=r.team,
              i=int(r.int), m=int(r.mic), x=round(float(r.int_over_exp), 2)) for r in ps.itertuples()]
    names = ps.drop_duplicates("player_id").set_index("player_id").player.to_dict()
    E = []
    for r in ev.itertuples():
        d = str(r.desc)
        E.append(dict(id=r.player_id, s=int(r.season), w=int(r.week), k=r.event_type, o=r.posteam,
                      qb=(r.passer_player_name if isinstance(r.passer_player_name, str) else ""),
                      rc=(r.receiver_player_name if isinstance(r.receiver_player_name, str) else ""),
                      d=(d[:150] + "…") if len(d) > 150 else d))
    T = [dict(t=r.team, s=int(r.season), i=int(r.int), m=int(r.mic)) for r in ts.itertuples()]
    PRI = {r.pos_group: [round(r.alpha, 3), round(r.beta, 3)] for r in pri.itertuples()}
    pr = R["pos_group_rates"]
    sh = R["split_half"]["6"]; pm = R["prediction"]; tm = R["team"]; fp = R["funnel_plot"]
    out95 = sum(v["outside_95"] for v in fp.values()); npl = sum(v["players"] for v in fp.values())
    m_by = {m["model"]: m["deviance"] for m in pm["models"]}
    findings = [
        f"Conversion differs by position: safeties convert {pr['S']['convert']:.0%} of their chances, cornerbacks {pr['CB']['convert']:.0%}, linebackers {pr['LB']['convert']:.0%}.",
        f"In four seasons, conversion rate within a position cannot be told apart from luck. Splitting each defender's chances into odd and even weeks gives a correlation of {sh['r']:.2f} (95% CI {sh['ci'][0]:.2f} to {sh['ci'][1]:.2f}, {sh['players']} defenders with 6+ chances), and only {out95} of {npl} defenders fall outside the 95% funnel limits. The 1999\u20132026 view below has enough seasons to find a small stable component.",
        f"Chances predict next-season INTs better than INTs alone (held-out deviance {m_by['Last-year chances (INT+MIC)']:.3f} vs {m_by['Last-year INT']:.3f}). Last year's missed chances carry about {pm['ratio_mic_to_int']:.1f} times the weight of a pick (95% CI {pm['ratio_ci'][0]:.1f} to {pm['ratio_ci'][1]:.1f}).",
        f"Team INT totals did not persist from one season to the next in this window (r = {tm['r_int']:.2f}); neither did team chances (r = {tm['r_chances']:.2f}), across {tm['n']} team-season pairs.",
    ]
    lpl = lps[~lps.partial_season & lps.pos_group.isin(["CB", "S", "LB"])].groupby("pos_group").agg(k=("int", "sum"), n=("chances", "sum"))
    pos_rates_l = {g: round(float(r.k / r.n), 4) for g, r in lpl.iterrows()}
    LP = [dict(id=r.player_id, n=r.player, pos=r.position, g=r.pos_group, s=int(r.season), t=r.team,
               i=int(r.int), m=int(r.pd), x=round(float(r.int_over_exp), 2)) for r in lps.itertuples()]
    LT = [dict(t=r.team, s=int(r.season), i=int(r.int), m=int(r.pd)) for r in lts.itertuples()]
    PRIL = {r.pos_group: [round(r.alpha, 3), round(r.beta, 3)] for r in lpri.itertuples()}
    tr = RL["trend"]; yy = RL["year_over_year"]["8"]; lp = RL["prediction"]; lt = RL["team"]; lv = RL["validation"]
    pe = RL["pos_era"]
    lm = {m["model"]: m for m in lp["models"]}
    import numpy as np
    wins = int((np.array(lm["Last-year chances (INT+PD)"]["by_season"]) < np.array(lm["Last-year INT"]["by_season"])).sum())
    findings_l = [
        f"Interceptions got rarer: {tr['int_rate_first5']:.1%} of pass attempts in 1999\u20132003 against {tr['int_rate_last5']:.1%} in 2021\u20132025, while breakups per attempt moved from {tr['pd_rate_first5']:.1%} to {tr['pd_rate_last5']:.1%}. The INT share of INT plus PD fell from {tr['share_first5']:.0%} to {tr['share_last5']:.0%}.",
        f"Position matters: safeties convert about {pe['S_2020_2025']['share']:.0%} of INT-plus-breakup plays, linebackers {pe['LB_2020_2025']['share']:.0%}, cornerbacks {pe['CB_2020_2025']['share']:.0%} (2020\u20132025). The cornerback share has fallen from {pe['CB_1999_2005']['share']:.0%} in 1999\u20132005.",
        f"A small stable skill exists. With 8+ chances in each of two consecutive seasons, INT share (relative to the league rate) correlates {yy['r']:.2f} year to year (95% CI {yy['lo']:.2f} to {yy['hi']:.2f}, {yy['n']:,} pairs). That is weak, and it takes about 20 to 45 chances to reach 50% reliability.",
        f"Chances predict next-season INTs better than INTs alone in {wins} of 26 held-out seasons. A breakup carries about {lp['ratio']:.2f} times the predictive weight of a pick (95% CI {lp['ratio_ci'][0]:.2f} to {lp['ratio_ci'][1]:.2f}).",
        f"Team INT totals persist modestly once the league-wide trend is removed (r = {lt['rd_int']:.2f}, 95% CI {lt['rd_int_ci'][0]:.2f} to {lt['rd_int_ci'][1]:.2f}, {lt['n']} team-season pairs).",
        f"Breakups line up with charted chances: across player-seasons in 2022\u20132025 the two correlate {lv['r_counts']:.2f}, and {lv['mic_share_of_pd']:.0%} of breakups were charted as interception-worthy.",
    ]
    META = dict(pos_rates={k: round(v["convert"], 4) for k, v in pr.items()}, last_week=last_week,
                n_events=int(len(ev)), n_players=int(ps.player_id.nunique()), findings=findings,
                pos_rates_l=pos_rates_l, n_events_l=int(RL["counts"]["events"]), n_players_l=int(RL["counts"]["players"]), findings_l=findings_l)
    payload = json.dumps(dict(P=P, E=E, T=T, PRI=PRI, META=META, L=LP, LT=LT, PRIL=PRIL), separators=(",", ":"), ensure_ascii=False)
    payload = payload.replace("</", "<\\/")

    frag = open(a.template, encoding="utf-8").read().replace("/*DATA*/", payload)
    open(os.path.join(a.out, "page_fragment.html"), "w", encoding="utf-8").write(frag)
    wrapper = ('<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
               '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">'
               '<style>:root{color-scheme:light}body{margin:0;font:14px system-ui}[hidden]{display:none!important}</style>\n'
               + frag + "\n</html>\n")
    open(os.path.join(a.out, "index.html"), "w", encoding="utf-8").write(wrapper)
    print("dashboard bytes:", len(wrapper), "players:", len(P), "long-run rows:", len(LP), "events:", len(E), "last week:", last_week)
    for f in findings + findings_l:
        print("-", f)


if __name__ == "__main__":
    main()
