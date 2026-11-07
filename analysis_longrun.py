#!/usr/bin/env python3
"""
analysis_longrun.py -- statistics and figures for the 1999-2026 long-run proxy (INT vs passes defended).
Reads data/longrun/*, data/ (MIC tables) and raw/ rosters. Writes figures/fig10..fig14 and results_longrun.json.
Usage: python3 analysis_longrun.py --raw raw --data data --fig figures --out results_longrun.json
"""
import argparse, json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats, special
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_poisson_deviance

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "font.size": 9.5, "axes.titlesize": 10.5,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False})
RNG = np.random.default_rng(11)
R = {}


def save(fig, path):
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


# ------------------------------------------------------------------ A. league trends
def trends(ls, path):
    c = ls[~ls.partial_season].copy()
    c["pd_per_att"] = c.db_lb_pd / c.pass_attempts
    c["int_per_att"] = c.interceptions / c.pass_attempts
    fit = stats.linregress(c.season, c.int_per_att)
    R["trend"] = dict(int_rate_1999=float(c.int_per_att.iloc[0]), int_rate_2025=float(c.int_per_att.iloc[-1]),
                      int_rate_first5=float(c.int_per_att.head(5).mean()), int_rate_last5=float(c.int_per_att.tail(5).mean()),
                      slope_per_year=float(fit.slope), slope_p=float(fit.pvalue),
                      pd_rate_first5=float(c.pd_per_att.head(5).mean()), pd_rate_last5=float(c.pd_per_att.tail(5).mean()),
                      share_first5=float(c.int_share.head(5).mean()), share_last5=float(c.int_share.tail(5).mean()),
                      min_int_rate_season=int(c.loc[c.int_per_att.idxmin(), "season"]),
                      min_int_rate=float(c.int_per_att.min()))
    fig, axs = plt.subplots(1, 3, figsize=(7.8, 2.9))
    for ax, col, ttl, mult in [(axs[0], "int_per_att", "Interceptions per pass attempt", 100),
                               (axs[1], "pd_per_att", "DB/LB passes defended per attempt", 100),
                               (axs[2], "int_share", "INT share of INT + PD", 100)]:
        ax.plot(c.season, c[col] * mult, color=BLUE, lw=2)
        ax.scatter([c.season.iloc[-1]], [c[col].iloc[-1] * mult], s=28, color=BLUE, zorder=3, edgecolor=SURF)
        ax.set_title(ttl, fontsize=9); ax.set_xlim(1998, 2026); ax.set_xticks([2000, 2010, 2020])
        ax.set_ylim(0, c[col].max() * mult * 1.25)
    axs[0].set_ylabel("%")
    fig.text(0.01, -0.04, "Regular season, 1999-2025. Pass attempts exclude sacks and spikes.", fontsize=8, color=INK2)
    fig.tight_layout(); save(fig, path)


# ------------------------------------------------------------------ B. position x era
ERAS = [(1999, 2005), (2006, 2012), (2013, 2019), (2020, 2025)]


def position_era(ps, path):
    d = ps[~ps.partial_season & ps.pos_group.isin(["CB", "S", "LB"])]
    rows = {}
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    cols = {"CB": BLUE, "S": ORANGE, "LB": AQUA}
    xs = np.arange(len(ERAS))
    for j, (g, c) in enumerate(cols.items()):
        ys, lo, hi = [], [], []
        for a, b in ERAS:
            x = d[(d.pos_group == g) & d.season.between(a, b)]
            k, n = x["int"].sum(), x.chances.sum()
            l, h = wilson(k, n); ys.append(k / n); lo.append(l); hi.append(h)
            rows[f"{g}_{a}_{b}"] = dict(int=int(k), chances=int(n), share=float(k / n), lo=float(l), hi=float(h))
        off = (j - 1) * 0.18
        ys, lo, hi = map(np.array, (ys, lo, hi))
        ax.errorbar(xs + off, ys * 100, yerr=[(ys - lo) * 100, (hi - ys) * 100], fmt="o", color=c, ecolor=c,
                    capsize=3, ms=6, mec=SURF, mew=1.2, label=g)
    ax.set_xticks(xs, [f"{a}-{b}" for a, b in ERAS]); ax.set_ylim(0, 50)
    ax.set_ylabel("INT share of INT + PD (%)"); ax.legend(ncol=3, loc="upper right")
    ax.set_title("Conversion by position and era (95% CI)")
    ax.grid(axis="x", visible=False); save(fig, path)
    R["pos_era"] = rows


# ------------------------------------------------------------------ C. reliability
def betabin_ll(k, n, a, b):
    return (special.gammaln(n + 1) - special.gammaln(k + 1) - special.gammaln(n - k + 1)
            + special.betaln(k + a, n - k + b) - special.betaln(a, b))


def overdispersion(ps, pri):
    out = {}
    d = ps[~ps.partial_season]
    for g in ["CB", "S", "LB"]:
        x = d[d.pos_group == g]; k, n = x["int"].values, x.chances.values
        a, b = pri.loc[g, "alpha"], pri.loc[g, "beta"]
        p0 = k.sum() / n.sum()
        ll_bin = (special.gammaln(n + 1) - special.gammaln(k + 1) - special.gammaln(n - k + 1)
                  + k * np.log(p0) + (n - k) * np.log(1 - p0)).sum()
        lrt = max(2 * (betabin_ll(k, n, a, b).sum() - ll_bin), 0)
        out[g] = dict(n=int(len(x)), alpha=float(a), beta=float(b), strength=float(a + b), mean=float(a / (a + b)),
                      lrt=float(lrt), p=float(0.5 * stats.chi2.sf(lrt, 1)),
                      talent_sd=float(np.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))), median_chances=float(np.median(n)))
    return out


def corr_ci(x, y, B=500):
    r = np.corrcoef(x, y)[0, 1]; idx = np.arange(len(x)); bs = []
    for _ in range(B):
        s = RNG.choice(idx, len(idx)); bs.append(np.corrcoef(x[s], y[s])[0, 1])
    lo, hi = np.nanpercentile(bs, [2.5, 97.5]); return float(r), float(lo), float(hi)


def split_half(ev, ps):
    e = ev[(ev.season_type == "REG") & (ev.season < ev.season.max())].copy()
    e["half"] = e.week % 2
    t = e.groupby(["player_id", "season", "half"]).agg(k=("event_type", lambda s: (s == "INT").sum()), n=("event_type", "size")).unstack(fill_value=0)
    t.columns = [f"{a}{b}" for a, b in t.columns]; t = t.reset_index()
    t = t.merge(ps[["player_id", "season", "pos_group", "lg_int_share"]], on=["player_id", "season"])
    res = {}
    for thr in (8, 12):
        for g in ["CB", "S", "LB"]:
            x = t[(t.pos_group == g) & ((t.n0 + t.n1) >= thr) & (t.n0 > 0) & (t.n1 > 0)]
            r0 = (x.k0 / x.n0 - x.lg_int_share).values; r1 = (x.k1 / x.n1 - x.lg_int_share).values
            r, lo, hi = corr_ci(r0, r1)
            res[f"{g}_{thr}"] = dict(n=int(len(x)), r=r, lo=lo, hi=hi)
    return res


def year_over_year(ps):
    d = ps[~ps.partial_season]
    a = d[["player_id", "season", "pos_group", "int", "chances", "lg_int_share"]]
    b = a.assign(season=a.season - 1)[["player_id", "season", "int", "chances", "lg_int_share"]]
    m = a.merge(b, on=["player_id", "season"], suffixes=("", "_n"))
    res = {}
    for thr in (8, 12):
        x = m[(m.chances >= thr) & (m.chances_n >= thr) & m.pos_group.isin(["CB", "S", "LB"])]
        r0 = (x["int"] / x.chances - x.lg_int_share).values; r1 = (x.int_n / x.chances_n - x.lg_int_share_n).values
        r, lo, hi = corr_ci(r0, r1)
        res[str(thr)] = dict(n=int(len(x)), r=r, lo=lo, hi=hi)
    return res


def fig_reliability(ov, split, yoy, path):
    fig, axs = plt.subplots(1, 2, figsize=(7.8, 3.0), gridspec_kw=dict(width_ratios=[1, 1.1]))
    n = np.arange(1, 81); ax = axs[0]
    for g, c in {"CB": BLUE, "S": ORANGE, "LB": AQUA}.items():
        s = ov[g]["strength"]; ax.plot(n, n / (n + s), color=c, lw=2)
        off = {"CB": 0.0, "S": 0.035, "LB": -0.035}[g]
        ax.text(81.5, 80 / (80 + s) + off, g, va="center", fontsize=9, fontweight="bold")
    ax.axvline(4, color=MUTED, lw=1, ls=(0, (3, 3)))
    ax.text(5.5, 0.92, "median season\n(4 chances)", fontsize=8, color=INK2, va="top")
    ax.set_xlim(1, 92); ax.set_ylim(0, 1)
    ax.set_xlabel("Chances (INT + PD)"); ax.set_ylabel("Share of rate variance that is skill")
    ax.set_title("Reliability by sample size")
    ax = axs[1]
    labs, vals, los, his = [], [], [], []
    for g in ["CB", "S", "LB"]:
        v = split[f"{g}_8"]; labs.append(f"{g}, split-half\n(n = {v['n']:,})"); vals.append(v["r"]); los.append(v["lo"]); his.append(v["hi"])
    v = yoy["8"]; labs.append(f"Year to year\n(n = {v['n']:,})"); vals.append(v["r"]); los.append(v["lo"]); his.append(v["hi"])
    y = np.arange(len(labs))[::-1]; vals, los, his = map(np.array, (vals, los, his))
    ax.errorbar(vals, y, xerr=[vals - los, his - vals], fmt="o", color=BLUE, ecolor=INK2, capsize=4, ms=7, mec=SURF, mew=1.5)
    ax.axvline(0, color=MUTED, lw=1); ax.set_yticks(y, labs); ax.set_xlim(-0.2, 0.6)
    ax.set_xlabel("Correlation (8+ chances), residual from league rate"); ax.set_title("Is conversion stable?")
    ax.grid(axis="y", visible=False); fig.tight_layout(); save(fig, path)


# ------------------------------------------------------------------ D. prediction
def season_scales(ls):
    c = ls[~ls.partial_season]
    return (c.db_lb_int.mean() / c.set_index("season").db_lb_int).to_dict(), (c.db_lb_pd.mean() / c.set_index("season").db_lb_pd).to_dict()


def build_pairs(ps, raw, ls, first, last):
    ki, kp = season_scales(ls)
    on = []
    for y in range(first + 1, last + 1):
        r = pd.read_csv(os.path.join(raw, f"roster_{y}.csv"), usecols=["gsis_id"]).dropna()
        on.append(pd.DataFrame({"player_id": r.gsis_id.unique(), "season_next": y}))
    on = pd.concat(on)
    b = ps[~ps.partial_season & ps.pos_group.isin(["CB", "S", "LB"])][["player_id", "season", "pos_group", "int", "pd", "chances"]].copy()
    b["season_next"] = b.season + 1
    b = b.merge(on, on=["player_id", "season_next"])
    n = ps[["player_id", "season", "int"]].rename(columns={"season": "season_next", "int": "int_next"})
    b = b.merge(n, on=["player_id", "season_next"], how="left"); b["int_next"] = b.int_next.fillna(0)
    b = b[b.season_next <= last - 1 + 0]
    b = b[b.season_next < ps.season.max()]
    b["int_a"] = b["int"] * b.season.map(ki); b["pd_a"] = b["pd"] * b.season.map(kp)
    b["ch_a"] = b.int_a + b.pd_a
    b["y"] = b.int_next * b.season_next.map(ki)
    return b.reset_index(drop=True)


def X(d, cols):
    x = pd.get_dummies(d.pos_group)[["S", "LB"]].astype(float)
    for c in cols: x[c] = d[c].astype(float)
    return x


def fitm(d, cols):
    m = PoissonRegressor(alpha=1e-6, max_iter=2000); m.fit(X(d, cols), d.y); return m


def prediction(d):
    models = {"Position only": [], "Last-year INT": ["int_a"], "Last-year PD": ["pd_a"],
              "Last-year chances (INT+PD)": ["ch_a"], "INT and PD separately": ["int_a", "pd_a"]}
    seasons = sorted(d.season.unique()); rows = []
    for name, cols in models.items():
        devs = []
        for s in seasons:
            tr, te = d[d.season != s], d[d.season == s]
            m = fitm(tr, cols); devs.append(mean_poisson_deviance(te.y, np.clip(m.predict(X(te, cols)), 1e-6, None)))
        rows.append(dict(model=name, deviance=float(np.mean(devs)), wins_vs_pos=None, by_season=[float(x) for x in devs]))
    base = np.array(rows[0]["by_season"])
    for r in rows: r["wins_vs_pos"] = int((np.array(r["by_season"]) < base).sum())
    cols = ["int_a", "pd_a"]; m = fitm(d, cols); co = dict(zip(X(d, cols).columns, m.coef_))
    ids = d.player_id.unique(); grp = {i: g for i, g in d.groupby("player_id")}; bs = []
    for _ in range(200):
        pick = RNG.choice(ids, len(ids)); b = pd.concat([grp[i] for i in pick], ignore_index=True)
        mb = fitm(b, cols); cb = dict(zip(X(b, cols).columns, mb.coef_)); bs.append((cb["int_a"], cb["pd_a"]))
    bs = np.array(bs)
    return dict(n=int(len(d)), n_pairs=int(len(seasons)), models=rows, coef_int=float(co["int_a"]), coef_pd=float(co["pd_a"]),
                ci_int=[float(x) for x in np.percentile(bs[:, 0], [2.5, 97.5])],
                ci_pd=[float(x) for x in np.percentile(bs[:, 1], [2.5, 97.5])],
                ratio=float(co["pd_a"] / co["int_a"]), ratio_ci=[float(x) for x in np.nanpercentile(bs[:, 1] / bs[:, 0], [2.5, 97.5])])


def fig_prediction(pr, path):
    fig, axs = plt.subplots(1, 2, figsize=(7.8, 3.0), gridspec_kw=dict(width_ratios=[1.3, 1]))
    ax = axs[0]; names = [m["model"] for m in pr["models"]]; dev = [m["deviance"] for m in pr["models"]]
    y = np.arange(len(names))[::-1]; lo = min(dev) - 0.03; hi = max(dev) + 0.03
    ax.hlines(y, lo, dev, color=GRID, lw=1.5); ax.scatter(dev, y, s=60, color=BLUE, zorder=3, edgecolor=SURF, linewidth=1.5)
    for yi, v in zip(y, dev): ax.text(v, yi + 0.28, f"{v:.3f}", ha="center", fontsize=8.5)
    ax.set_yticks(y, names); ax.set_xlim(lo, hi); ax.set_ylim(-0.5, len(names) - 0.1)
    ax.set_xlabel("Held-out Poisson deviance (lower is better)"); ax.set_title(f"Predicting next-season INTs ({pr['n_pairs']} season pairs)")
    ax.grid(axis="y", visible=False)
    ax = axs[1]
    for i, (lab, c, ci, col) in enumerate([("INT last year", pr["coef_int"], pr["ci_int"], BLUE), ("PD last year", pr["coef_pd"], pr["ci_pd"], ORANGE)]):
        ax.errorbar(c, 1 - i, xerr=[[c - ci[0]], [ci[1] - c]], fmt="o", color=col, ecolor=INK2, capsize=4, ms=7, mec=SURF, mew=1.5)
    ax.axvline(0, color=MUTED, lw=1); ax.set_yticks([1, 0], ["INT last year", "PD last year"]); ax.set_ylim(-0.6, 1.6)
    ax.set_xlabel("Poisson coefficient (log INT per extra event)"); ax.set_title("What carries the signal?"); ax.grid(axis="y", visible=False)
    fig.tight_layout(); save(fig, path)


def team_persist(ts):
    a = ts[ts.season < ts.season.max()].copy()
    n = a.merge(a.assign(season=a.season - 1)[["team", "season", "int", "pd", "chances"]], on=["team", "season"], suffixes=("", "_n"))
    out = dict(n=int(len(n)))
    for c in ["int", "pd", "chances"]:
        out[f"r_{c}"] = float(np.corrcoef(n[c], n["int_n"])[0, 1])
    # detrend by season (league mean) so era shifts do not drive correlations
    z = n.copy()
    for c in ["int", "pd", "chances", "int_n"]:
        z[c] = z[c] - z.groupby("season")[c].transform("mean") if c != "int_n" else z[c] - z.groupby("season")["int_n"].transform("mean")
    for c in ["int", "pd", "chances"]:
        r = float(np.corrcoef(z[c], z["int_n"])[0, 1]); idx = np.arange(len(z)); bs = []
        for _ in range(500):
            s = RNG.choice(idx, len(idx)); bs.append(np.corrcoef(z[c].values[s], z["int_n"].values[s])[0, 1])
        out[f"rd_{c}"] = r; out[f"rd_{c}_ci"] = [float(x) for x in np.percentile(bs, [2.5, 97.5])]
    return out


# ------------------------------------------------------------------ E. validation against MIC
def validate(lps, mps, path):
    m = mps[~mps.partial_season].merge(lps[["player_id", "season", "pd", "int", "pos_group"]].rename(columns={"int": "int_l", "pos_group": "g_l"}),
                                      on=["player_id", "season"], how="inner")
    r_n, lo_n, hi_n = corr_ci(m["pd"].values.astype(float), m["mic"].values.astype(float))
    big = m[m["pd"] >= 3]
    tot_pd = lps[(lps.season.between(2022, 2025))]["pd"].sum(); tot_mic = mps[~mps.partial_season]["mic"].sum()
    by_g = {}
    for g in ["CB", "S", "LB"]:
        x = lps[(lps.season.between(2022, 2025)) & (lps.pos_group == g)]; y = mps[(~mps.partial_season) & (mps.pos_group == g)]
        by_g[g] = dict(pd=int(x["pd"].sum()), mic=int(y["mic"].sum()), share=float(y["mic"].sum() / x["pd"].sum()))
    cs = m[(m.chances >= 6) & (m.pos_group.isin(["CB", "S", "LB"]))]
    cs_pd = cs["int"] / (cs["int"] + cs["pd"])
    r_rate, lo_r, hi_r = corr_ci(cs.convert_rate.values, cs_pd.values)
    R["validation"] = dict(n=int(len(m)), r_counts=r_n, r_counts_ci=[lo_n, hi_n], mic_share_of_pd=float(tot_mic / tot_pd),
                           mic_total=int(tot_mic), pd_total=int(tot_pd), by_group=by_g, n_rate=int(len(cs)), r_rate=r_rate,
                           r_rate_ci=[lo_r, hi_r], int_match=bool((m["int"] == m.int_l).mean() > 0.99),
                           mean_pd_mic_slope=float(np.polyfit(m["pd"], m["mic"], 1)[0]))
    fig, axs = plt.subplots(1, 2, figsize=(7.8, 3.1))
    ax = axs[0]
    jit = RNG.uniform(-0.2, 0.2, len(m))
    ax.scatter(m["pd"] + jit, m["mic"] + RNG.uniform(-0.2, 0.2, len(m)), s=10, color=BLUE, alpha=0.4, lw=0)
    xs = np.linspace(0, m["pd"].max(), 20); b, a0 = np.polyfit(m["pd"], m["mic"], 1)
    ax.plot(xs, b * xs + a0, color=ORANGE, lw=2)
    ax.set_xlabel("Passes defended (PD), player-season"); ax.set_ylabel("Charted missed-INT chances (MIC)")
    ax.set_title(f"PD tracks charted chances (r = {r_n:.2f})")
    ax = axs[1]
    gs = ["CB", "S", "LB"]; v = [by_g[g]["share"] * 100 for g in gs]
    ax.bar(gs, v, width=0.5, color=BLUE)
    for i, x in enumerate(v): ax.text(i, x + 0.4, f"{x:.0f}%", ha="center", fontsize=9)
    ax.set_ylabel("MIC as % of passes defended"); ax.set_ylim(0, max(v) * 1.3)
    ax.set_title("Most breakups were not interception-worthy"); ax.grid(axis="x", visible=False)
    fig.tight_layout(); save(fig, path)


# ------------------------------------------------------------------ F. leaders
def fig_leaders(car, path):
    d = car.sort_values("int", ascending=False).head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 4.4)); y = np.arange(len(d))
    ax.barh(y, d["int"], height=0.6, color=BLUE, label="Interceptions")
    ax.barh(y, d["pd"], left=d["int"], height=0.6, color=ORANGE, label="Other passes defended", edgecolor=SURF, linewidth=1.5)
    ax.set_yticks(y, [f"{p}  ({int(a)}-{int(b) % 100:02d})" for p, a, b in zip(d.player, d.first_season, d.last_season)])
    for yi, (_, r) in zip(y, d.iterrows()): ax.text(r.chances + 2, yi, f"{r.int_share:.0%}", va="center", fontsize=8.5)
    ax.set_xlabel("Interceptions + passes defended, regular season 1999-2026 (label = INT share)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.14), ncol=2); ax.grid(axis="y", visible=False)
    ax.set_title("Career interception leaders and their other breakups"); save(fig, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="raw"); ap.add_argument("--data", default="data")
    ap.add_argument("--fig", default="figures"); ap.add_argument("--out", default="results_longrun.json")
    a = ap.parse_args(); L = os.path.join(a.data, "longrun")
    ps = pd.read_csv(os.path.join(L, "lr_player_seasons.csv")); car = pd.read_csv(os.path.join(L, "lr_player_career.csv"))
    ls = pd.read_csv(os.path.join(L, "lr_league_seasons.csv")); ts = pd.read_csv(os.path.join(L, "lr_team_seasons.csv"))
    pri = pd.read_csv(os.path.join(L, "lr_eb_priors.csv")).set_index("pos_group")
    ev = pd.read_csv(os.path.join(L, "lr_events.csv.gz")); au = pd.read_csv(os.path.join(L, "lr_credit_audit.csv"))
    mps = pd.read_csv(os.path.join(a.data, "dint_player_seasons.csv"))
    first, last = int(ls.season.min()), int(ls[~ls.partial_season].season.max())
    comp = ps[~ps.partial_season]
    R["counts"] = dict(first=first, last_complete=last, seasons=int(comp.season.nunique()), player_seasons=int(len(ps)),
                       players=int(len(car)), events=int(len(ev)), int_events=int((ev.event_type == "INT").sum()),
                       pd_events=int((ev.event_type == "PD").sum()), median_chances=float(comp.chances.median()),
                       share_le2=float((comp.chances <= 2).mean()),
                       db_share_early=float(comp[comp.season <= 2001].query("pos_group=='DB'").chances.sum() / comp[comp.season <= 2001].chances.sum()),
                       db_share_all=float(comp.query("pos_group=='DB'").chances.sum() / comp.chances.sum()),
                       dl_credit_share=float(au.dl.sum() / au.credits.sum()), overall_int_share=float(comp["int"].sum() / comp.chances.sum()))
    trends(ls, f"{a.fig}/fig10_trends.png")
    position_era(ps, f"{a.fig}/fig11_position_era.png")
    R["overdispersion"] = overdispersion(ps, pri); R["split_half"] = split_half(ev, ps); R["year_over_year"] = year_over_year(ps)
    fig_reliability(R["overdispersion"], R["split_half"], R["year_over_year"], f"{a.fig}/fig12_longrun_reliability.png")
    d = build_pairs(ps, a.raw, ls, first, last + 1)
    R["prediction"] = prediction(d); fig_prediction(R["prediction"], f"{a.fig}/fig13_longrun_prediction.png")
    R["team"] = team_persist(ts)
    validate(ps[ps.season.between(2022, 2025)], mps, f"{a.fig}/fig14_validation.png")
    fig_leaders(car, f"{a.fig}/fig15_longrun_leaders.png")
    top = car.head(10)[["player", "position", "first_season", "last_season", "int", "pd", "chances", "int_share", "int_share_eb"]]
    R["top_int"] = top.round(3).to_dict(orient="records")
    q = car[car.chances >= 80].sort_values("int_share_eb")
    cols = ["player", "position", "first_season", "last_season", "int", "pd", "chances", "int_share", "int_share_eb"]
    R["low_eb"] = q.head(8)[cols].round(3).to_dict(orient="records"); R["high_eb"] = q.tail(8).iloc[::-1][cols].round(3).to_dict(orient="records")
    R["lowest_int_season"] = None
    with open(a.out, "w") as f: json.dump(R, f, indent=1, default=float)
    print(json.dumps({k: R[k] for k in ["counts", "trend", "overdispersion", "split_half", "year_over_year", "team", "validation"]}, indent=1, default=float))
    print(json.dumps({k: {kk: vv for kk, vv in R["prediction"].items() if kk != "models"} for k in ["p"]}, indent=1)[:900])
    print([(m["model"], round(m["deviance"], 4), m["wins_vs_pos"]) for m in R["prediction"]["models"]])
    print(R["low_eb"][:5]); print(R["high_eb"][:5])


if __name__ == "__main__":
    main()
