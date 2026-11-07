#!/usr/bin/env python3
"""
analysis.py -- statistics, models and figures for the DINT paper.
Reads data/ (from build_dataset.py) and raw/ (nflverse). Writes figures/ and results.json.
Usage: python3 analysis.py --raw raw --data data --fig figures --first 2022 --last 2026
"""
import argparse, json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy import stats, special
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_poisson_deviance

# ---- reference palette (light surface) ----
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False,
})
RNG = np.random.default_rng(7)
R = {}   # results dict -> results.json


def save(fig, path):
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


# ------------------------------------------------------------------ funnel counts
def funnel_counts(raw, first, last):
    rows = []
    for y in range(first, last + 1):
        p = pd.read_csv(os.path.join(raw, f"pbp_{y}.csv.gz"), low_memory=False,
                        usecols=["game_id", "play_id", "season", "season_type", "pass_attempt", "sack",
                                 "qb_spike", "interception", "complete_pass", "incomplete_pass",
                                 "pass_defense_1_player_id", "pass_defense_2_player_id"])
        f = pd.read_csv(os.path.join(raw, f"ftn_{y}.csv"),
                        usecols=["nflverse_game_id", "nflverse_play_id", "is_interception_worthy"])
        r = pd.read_csv(os.path.join(raw, f"roster_{y}.csv"),
                        usecols=["gsis_id", "position", "depth_chart_position"]).dropna(subset=["gsis_id"])
        r["pos"] = r["depth_chart_position"].fillna(r["position"])
        r = r.groupby("gsis_id")["pos"].agg(lambda s: s.mode().iat[0])
        dl = {"DE", "DT", "NT", "DL"}
        m = p.merge(f, left_on=["game_id", "play_id"], right_on=["nflverse_game_id", "nflverse_play_id"], how="left")
        m = m[(m.season_type == "REG") & (m.pass_attempt == 1) & (m.sack != 1) & (m.qb_spike != 1)]
        m = m[m.is_interception_worthy == True]
        pos1 = m.pass_defense_1_player_id.map(r); pos2 = m.pass_defense_2_player_id.map(r)
        has_cov = (pos1.notna() & ~pos1.isin(dl)) | (pos2.notna() & ~pos2.isin(dl))
        has_any = m.pass_defense_1_player_id.notna() | m.pass_defense_2_player_id.notna()
        inc = (m.incomplete_pass == 1) & (m.interception != 1)
        rows.append(dict(season=y, iw=len(m), intercepted=int((m.interception == 1).sum()),
                         mic_plays=int((inc & has_cov).sum()),
                         dl_only=int((inc & has_any & ~has_cov).sum()),
                         no_credit=int((inc & ~has_any).sum()),
                         completed=int((m.complete_pass == 1).sum())))
    return pd.DataFrame(rows)


def fig_funnel(fc, path):
    tot = fc[fc.season <= 2025][["iw", "intercepted", "mic_plays", "dl_only", "no_credit", "completed"]].sum()
    R["funnel_2022_2025"] = {k: int(v) for k, v in tot.items()}
    fig, ax = plt.subplots(figsize=(7.4, 3.3))
    ax.set_xlim(0, 100); ax.set_ylim(0, 10); ax.axis("off"); ax.grid(False)
    def box(x, y, w, h, txt, fc_, tc="white", fs=9):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.5",
                                    fc=fc_, ec="none"))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", color=tc, fontsize=fs, linespacing=1.3)
    N = int(tot.iw)
    box(1, 3.9, 20, 2.4, f"FTN \"interception-\nworthy\" throws\nn = {N:,}", "#184f95")
    items = [("Intercepted", tot.intercepted, BLUE, "white", "INT"),
             ("Incomplete, DB/LB\ncredited with PD", tot.mic_plays, ORANGE, "white", "MIC  (drop proxy)"),
             ("Incomplete, DL\ncredited (batted)", tot.dl_only, "#f0efec", INK, "excluded"),
             ("Incomplete,\nno PD credit", tot.no_credit, "#f0efec", INK, "excluded"),
             ("Completed", tot.completed, "#f0efec", INK, "excluded")]
    ys = [8.0, 6.1, 4.2, 2.3, 0.4]
    for (lab, v, col, tc, tag), y in zip(items, ys):
        ax.plot([21, 38], [5.1, y + 0.8], color=MUTED, lw=1)
        box(38, y, 27, 1.6, f"{lab}", col, tc, 8.5)
        ax.text(67, y + 0.8, f"{int(v):,}  ({v / N:.0%})", va="center", fontsize=9, color=INK)
        ax.text(99, y + 0.8, tag, va="center", ha="right", fontsize=8.5,
                color=(ORANGE if tag.startswith("MIC") else (BLUE if tag == "INT" else MUTED)), fontweight="bold")
    ax.set_title("Where interception-worthy throws end up (regular season, 2022–2025)", loc="left", y=1.02)
    save(fig, path)


def fig_drift(qa, path):
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.9))
    seasons = qa.season.astype(str).tolist()
    seasons[-1] = seasons[-1] + "*"
    axs[0].plot(seasons, qa.int_worthy_rate * 100, color=BLUE, lw=2, marker="o", ms=5)
    axs[0].set_title("FTN interception-worthy rate"); axs[0].set_ylabel("% of pass attempts")
    axs[1].plot(seasons, qa.int_rate * 100, color=BLUE, lw=2, marker="o", ms=5)
    axs[1].set_title("Actual interception rate"); axs[1].set_ylabel("% of pass attempts")
    for a in axs:
        a.set_ylim(0, a.get_ylim()[1] * 1.05)
    axs[0].set_ylim(0, 6); axs[1].set_ylim(0, 6)
    fig.text(0.01, -0.03, "* 2026 is a partial season. Interception rate is stable; the charted rate falls by about 40%.",
             fontsize=8, color=INK2)
    fig.tight_layout()
    save(fig, path)


# ------------------------------------------------------------------ position-group rates
def fig_pos_rates(ps, path):
    d = ps[~ps.partial_season & ps.pos_group.isin(["CB", "S", "LB"])]
    g = d.groupby("pos_group").agg(k=("int", "sum"), n=("chances", "sum"), mic=("mic", "sum"))
    g = g.loc[["CB", "S", "LB"]]
    g["rate"] = g.k / g.n
    g["lo"], g["hi"] = wilson(g.k, g.n)
    R["pos_group_rates"] = {i: dict(int=int(r.k), mic=int(r.mic), chances=int(r.n), convert=float(r.rate),
                                    lo=float(r.lo), hi=float(r.hi)) for i, r in g.iterrows()}
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    x = np.arange(3)
    ax.bar(x, g.rate * 100, width=0.5, color=BLUE)
    ax.errorbar(x, g.rate * 100, yerr=[(g.rate - g.lo) * 100, (g.hi - g.rate) * 100], fmt="none",
                ecolor=INK2, elinewidth=1.2, capsize=4)
    for xi, (_, r) in zip(x, g.iterrows()):
        ax.text(xi + 0.3, r.rate * 100, f"{r.rate:.0%}", va="center", fontsize=9, color=INK)
    ax.set_xticks(x, [f"{i}\n(n = {int(g.loc[i, 'n']):,})" for i in g.index])
    ax.set_ylabel("Chances converted to INT (%)"); ax.set_ylim(0, 100); ax.grid(axis="x", visible=False)
    ax.set_title("Conversion rate by position, 2022–2025 (95% CI)")
    save(fig, path)


# ------------------------------------------------------------------ reliability
def betabin_ll(k, n, a, b):
    return (special.gammaln(n + 1) - special.gammaln(k + 1) - special.gammaln(n - k + 1)
            + special.betaln(k + a, n - k + b) - special.betaln(a, b))


def overdispersion(ps, pri):
    out = {}
    d = ps[~ps.partial_season]
    for g in ["CB", "S", "LB"]:
        x = d[d.pos_group == g]
        k, n = x["int"].values, x["chances"].values
        a, b = pri.loc[g, "alpha"], pri.loc[g, "beta"]
        p0 = k.sum() / n.sum()
        ll_bin = (special.gammaln(n + 1) - special.gammaln(k + 1) - special.gammaln(n - k + 1)
                  + k * np.log(p0) + (n - k) * np.log(1 - p0)).sum()
        ll_bb = betabin_ll(k, n, a, b).sum()
        lrt = max(2 * (ll_bb - ll_bin), 0)
        out[g] = dict(n_player_seasons=int(len(x)), alpha=float(a), beta=float(b),
                      prior_strength=float(a + b), lrt=float(lrt), p=float(0.5 * stats.chi2.sf(lrt, 1)),
                      true_talent_sd=float(np.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))),
                      median_chances=float(np.median(n)))
    return out


def split_half(ev):
    ev = ev[ev.season_type == "REG"].copy()
    ev["half"] = ev.week % 2
    t = ev.groupby(["player_id", "half"]).agg(k=("event_type", lambda s: (s == "INT").sum()),
                                              n=("event_type", "size")).unstack(fill_value=0)
    t.columns = [f"{a}{b}" for a, b in t.columns]
    res = {}
    for thr in (4, 6, 8):
        x = t[(t.n0 + t.n1) >= thr]
        x = x[(x.n0 > 0) & (x.n1 > 0)]
        r0, r1 = x.k0 / x.n0, x.k1 / x.n1
        r = np.corrcoef(r0, r1)[0, 1]
        boots = []
        idx = np.arange(len(x))
        for _ in range(1000):
            s = RNG.choice(idx, len(idx))
            boots.append(np.corrcoef(r0.values[s], r1.values[s])[0, 1])
        lo, hi = np.nanpercentile(boots, [2.5, 97.5])
        res[str(thr)] = dict(players=int(len(x)), r=float(r), ci=[float(lo), float(hi)],
                             spearman_brown=float(2 * r / (1 + r)) if r > -0.99 else None)
    return res


def fig_reliability(over, path):
    n = np.arange(1, 31)
    fig, ax = plt.subplots(figsize=(5.4, 3.1))
    cols = {"CB": BLUE, "S": ORANGE, "LB": AQUA}
    for g, c in cols.items():
        s = over[g]["prior_strength"]
        ax.plot(n, n / (n + s), color=c, lw=2)
        ax.text(30.4, 30 / (30 + s), g, color=INK, va="center", fontsize=9, fontweight="bold")
    ax.axvline(2, color=MUTED, lw=1, ls=(0, (3, 3)))
    ax.text(2.4, 0.96, "median player-season\n(2 chances)", fontsize=8, color=INK2, va="top")
    ax.set_xlim(1, 33); ax.set_ylim(0, 1)
    ax.set_xlabel("Chances in the sample (INT + MIC)"); ax.set_ylabel("Share of rate variance that is skill")
    ax.set_title("How many chances before conversion rate means something?")
    save(fig, path)


# ------------------------------------------------------------------ funnel plot
def fig_funnel_plot(car, path):
    groups = ["CB", "S", "LB"]
    fig, axs = plt.subplots(1, 3, figsize=(7.6, 3.0), sharey=True)
    summ = {}
    for ax, g in zip(axs, groups):
        d = car[car.pos_group == g]
        p0 = d["int"].sum() / d["chances"].sum()
        nmax = int(d.chances.max()) + 2
        nn = np.arange(1, nmax + 1)
        lo95 = stats.binom.ppf(0.025, nn, p0) / nn; hi95 = stats.binom.ppf(0.975, nn, p0) / nn
        lo99 = stats.binom.ppf(0.001, nn, p0) / nn; hi99 = stats.binom.ppf(0.999, nn, p0) / nn
        ax.fill_between(nn, lo99, hi99, step="mid", color=BLUE, alpha=0.10, lw=0)
        ax.fill_between(nn, lo95, hi95, step="mid", color=BLUE, alpha=0.16, lw=0)
        ax.axhline(p0, color=BLUE, lw=1.4)
        rate = d["int"] / d.chances
        jit = RNG.uniform(-0.25, 0.25, len(d))
        out95 = (d["int"] < stats.binom.ppf(0.025, d.chances, p0)) | (d["int"] > stats.binom.ppf(0.975, d.chances, p0))
        # "outside" uses the strict inequality on discrete limits
        ax.scatter(d.chances + jit, rate, s=10, color=BLUE, alpha=0.55, lw=0)
        ax.scatter((d.chances + jit)[out95], rate[out95], s=16, color=ORANGE, lw=0, zorder=3)
        ax.set_title(f"{g}  (mean {p0:.0%})"); ax.set_xlabel("Chances, 2022–2026")
        ax.set_ylim(-0.03, 1.03)
        summ[g] = dict(players=int(len(d)), mean=float(p0), outside_95=int(out95.sum()),
                       share_outside_95=float(out95.mean()), outside_names=d.loc[out95, "player"].tolist())
        for _, r in d.sort_values("chances", ascending=False).head(0).iterrows():
            pass
    axs[0].set_ylabel("Conversion rate")
    fig.suptitle("Funnel plots: do converters differ from luck?  (bands: 95% and 99.8% binomial limits)",
                 x=0.01, ha="left", fontsize=10.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, path)
    return summ


# ------------------------------------------------------------------ prediction
def build_pairs(ps, raw, first, last):
    ros = []
    for y in range(first, last + 1):
        r = pd.read_csv(os.path.join(raw, f"roster_{y}.csv"),
                        usecols=["gsis_id", "position", "depth_chart_position"]).dropna(subset=["gsis_id"])
        r["pos"] = r["depth_chart_position"].fillna(r["position"]); r["season"] = y
        ros.append(r.groupby(["gsis_id", "season"])["pos"].agg(lambda s: s.mode().iat[0]).reset_index())
    ros = pd.concat(ros).rename(columns={"gsis_id": "player_id"})
    base = ps[~ps.partial_season][["player_id", "player", "season", "pos_group", "int", "mic", "chances"]]
    nxt = ps[["player_id", "season", "int"]].rename(columns={"season": "season_next", "int": "int_next"})
    base = base.assign(season_next=base.season + 1)
    d = base.merge(ros.assign(season_next=ros.season)[["player_id", "season_next"]],
                   on=["player_id", "season_next"], how="inner")          # still on a roster next year
    d = d.merge(nxt, on=["player_id", "season_next"], how="left")
    d["int_next"] = d["int_next"].fillna(0)
    d = d[d.season_next <= 2025]                                         # complete outcome seasons only
    d = d[d.pos_group.isin(["CB", "S", "LB"])].reset_index(drop=True)
    return d


def design(d, cols):
    X = pd.get_dummies(d.pos_group)[["S", "LB"]].astype(float)
    for c in cols:
        X[c] = d[c].astype(float)
    return X


def fit(d, cols):
    m = PoissonRegressor(alpha=1e-6, max_iter=1000)
    m.fit(design(d, cols), d.int_next)
    return m


def prediction(d):
    models = {"Position only": [], "Last-year INT": ["int"], "Last-year chances (INT+MIC)": ["chances"],
              "INT and MIC separately": ["int", "mic"]}
    seasons = sorted(d.season.unique())
    rows = []
    for name, cols in models.items():
        devs = []
        for s in seasons:
            tr, te = d[d.season != s], d[d.season == s]
            m = fit(tr, cols)
            pred = np.clip(m.predict(design(te, cols)), 1e-6, None)
            devs.append(mean_poisson_deviance(te.int_next, pred))
        rows.append(dict(model=name, deviance=float(np.mean(devs)), by_season=[float(x) for x in devs]))
    # coefficients with player-cluster bootstrap
    m = fit(d, ["int", "mic"])
    coef = dict(zip(design(d, ["int", "mic"]).columns, m.coef_))
    ids = d.player_id.unique()
    groups = {i: g for i, g in d.groupby("player_id")}
    bs = []
    for _ in range(400):
        pick = RNG.choice(ids, len(ids))
        b = pd.concat([groups[i] for i in pick], ignore_index=True)
        mb = fit(b, ["int", "mic"])
        cb = dict(zip(design(b, ["int", "mic"]).columns, mb.coef_))
        bs.append((cb["int"], cb["mic"]))
    bs = np.array(bs)
    ratio = bs[:, 1] / bs[:, 0]
    res = dict(n=int(len(d)), pairs=[f"{s}->{s + 1}" for s in seasons], models=rows,
               coef_int=float(coef["int"]), coef_mic=float(coef["mic"]),
               ci_int=[float(x) for x in np.percentile(bs[:, 0], [2.5, 97.5])],
               ci_mic=[float(x) for x in np.percentile(bs[:, 1], [2.5, 97.5])],
               ratio_mic_to_int=float(coef["mic"] / coef["int"]),
               ratio_ci=[float(x) for x in np.nanpercentile(ratio, [2.5, 97.5])],
               rate_ratio_int=float(np.exp(coef["int"])), rate_ratio_mic=float(np.exp(coef["mic"])))
    return res


def fig_prediction(pr, path):
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.0), gridspec_kw=dict(width_ratios=[1.25, 1]))
    ax = axs[0]
    names = [m["model"] for m in pr["models"]]; dev = [m["deviance"] for m in pr["models"]]
    y = np.arange(len(names))[::-1]
    ax.hlines(y, 1.40, dev, color=GRID, lw=1.5)
    ax.scatter(dev, y, s=60, color=BLUE, zorder=3, edgecolor=SURF, linewidth=1.5)
    for yi, v in zip(y, dev):
        ax.text(v, yi + 0.28, f"{v:.3f}", va="bottom", ha="center", fontsize=8.5)
    ax.set_yticks(y, names); ax.set_xlim(1.40, 1.66); ax.set_ylim(-0.5, len(names) - 0.1)
    ax.set_xlabel("Held-out Poisson deviance (lower is better)")
    ax.set_title("Predicting next-season INTs"); ax.grid(axis="y", visible=False)
    ax = axs[1]
    pts = [("INT last year", pr["coef_int"], pr["ci_int"]), ("MIC last year", pr["coef_mic"], pr["ci_mic"])]
    for i, (lab, c, ci) in enumerate(pts):
        ax.errorbar(c, 1 - i, xerr=[[c - ci[0]], [ci[1] - c]], fmt="o", color=BLUE if i == 0 else ORANGE,
                    ecolor=INK2, capsize=4, ms=7, mec=SURF, mew=1.5)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks([1, 0], [p[0] for p in pts]); ax.set_ylim(-0.6, 1.6)
    ax.set_xlabel("Poisson coefficient (log INT per extra event)")
    ax.set_title("Is a miss worth a pick?"); ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, path)


# ------------------------------------------------------------------ teams
def team_analysis(ts):
    a = ts[ts.season <= 2025].copy()
    n = a.merge(a.assign(season=a.season - 1)[["team", "season", "int", "chances", "mic"]],
                on=["team", "season"], suffixes=("", "_next"))
    r_int = np.corrcoef(n["int"], n["int_next"])[0, 1]
    r_ch = np.corrcoef(n["chances"], n["int_next"])[0, 1]
    r_mic = np.corrcoef(n["mic"], n["int_next"])[0, 1]
    diffs, idx = [], np.arange(len(n))
    for _ in range(2000):
        s = RNG.choice(idx, len(idx)); x = n.iloc[s]
        diffs.append(np.corrcoef(x["chances"], x["int_next"])[0, 1] - np.corrcoef(x["int"], x["int_next"])[0, 1])
    return n, dict(n=int(len(n)), r_int=float(r_int), r_chances=float(r_ch), r_mic=float(r_mic),
                   diff=float(r_ch - r_int), diff_ci=[float(x) for x in np.percentile(diffs, [2.5, 97.5])])


def fig_team(n, tr, path):
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.2), sharey=True)
    for ax, col, lab, r in [(axs[0], "int", "Team INTs, season t", tr["r_int"]),
                            (axs[1], "chances", "Team chances (INT + MIC), season t", tr["r_chances"])]:
        ax.scatter(n[col], n["int_next"], s=22, color=BLUE, alpha=0.7, lw=0)
        m, b = np.polyfit(n[col], n["int_next"], 1)
        xs = np.linspace(n[col].min(), n[col].max(), 20)
        ax.plot(xs, m * xs + b, color=INK2, lw=1.4)
        ax.set_xlabel(lab); ax.set_title(f"r = {r:.2f}")
    axs[0].set_ylabel("Team INTs, season t+1")
    fig.suptitle("Team-level persistence (96 team-season pairs)", x=0.01, ha="left", fontsize=10.5,
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, path)


# ------------------------------------------------------------------ leaderboard
def fig_leaders(car, path, k=14):
    d = car.sort_values(["mic", "chances"], ascending=False).head(k).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    y = np.arange(len(d))
    ax.barh(y, d["int"], height=0.6, color=BLUE, label="Interceptions")
    ax.barh(y, d["mic"], left=d["int"], height=0.6, color=ORANGE, label="Missed-INT chances (MIC)",
            edgecolor=SURF, linewidth=1.5)
    ax.set_yticks(y, [f"{p}  ({pos}, {t})" for p, pos, t in zip(d.player, d.position, d.last_team)])
    for yi, (_, r) in zip(y, d.iterrows()):
        ax.text(r.chances + 0.3, yi, f"{r.convert_rate:.0%}", va="center", fontsize=8.5, color=INK)
    ax.set_xlabel("Chances, 2022–2026 (label = conversion rate)"); ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.14), ncol=2)
    ax.set_title("Most missed-INT chances since 2022")
    save(fig, path)


def fig_shrink(car, path):
    d = car[car.chances >= 10].copy()
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    cols = {"CB": BLUE, "S": ORANGE, "LB": AQUA}
    for g, c in cols.items():
        x = d[d.pos_group == g]
        ax.scatter(x.convert_rate, x.convert_rate_eb, s=24, color=c, lw=0, alpha=0.85, label=g)
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=(0, (3, 3)))
    ax.set_xlim(0.2, 1.02); ax.set_ylim(0.4, 0.95)
    ax.set_xlabel("Raw conversion rate"); ax.set_ylabel("Empirical-Bayes conversion rate")
    ax.set_title("Shrinkage: what is left after removing noise  (≥10 chances)")
    ax.legend(loc="upper left", ncol=3)
    save(fig, path)
    return int(len(d)), float(d.convert_rate.std()), float(d.convert_rate_eb.std())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="raw"); ap.add_argument("--data", default="data")
    ap.add_argument("--fig", default="figures"); ap.add_argument("--first", type=int, default=2022)
    ap.add_argument("--last", type=int, default=2026); ap.add_argument("--out", default="results.json")
    a = ap.parse_args()
    os.makedirs(a.fig, exist_ok=True)
    ps = pd.read_csv(os.path.join(a.data, "dint_player_seasons.csv"))
    car = pd.read_csv(os.path.join(a.data, "dint_player_career.csv"))
    ts = pd.read_csv(os.path.join(a.data, "dint_team_seasons.csv"))
    qa = pd.read_csv(os.path.join(a.data, "dint_qa_by_season.csv"))
    pri = pd.read_csv(os.path.join(a.data, "dint_eb_priors.csv")).set_index("pos_group")
    ev = pd.read_csv(os.path.join(a.data, "dint_events.csv"))

    R["counts"] = dict(events=int(len(ev)), int_events=int((ev.event_type == "INT").sum()),
                       mic_events=int((ev.event_type == "MIC").sum()), player_seasons=int(len(ps)),
                       players=int(len(car)), teams=int(ts.team.nunique()),
                       median_chances_season=float(ps[~ps.partial_season].chances.median()),
                       share_one_chance=float((ps[~ps.partial_season].chances == 1).mean()))
    R["qa"] = qa.to_dict(orient="records")
    fc = funnel_counts(a.raw, a.first, a.last)
    fig_funnel(fc, f"{a.fig}/fig1_funnel.png")
    fig_drift(qa, f"{a.fig}/fig2_drift.png")
    fig_pos_rates(ps, f"{a.fig}/fig3_position.png")
    R["overdispersion"] = overdispersion(ps, pri)
    R["split_half"] = split_half(ev)
    fig_reliability(R["overdispersion"], f"{a.fig}/fig4_reliability.png")
    R["funnel_plot"] = fig_funnel_plot(car, f"{a.fig}/fig5_funnelplot.png")
    d = build_pairs(ps, a.raw, a.first, a.last)
    R["prediction"] = prediction(d)
    fig_prediction(R["prediction"], f"{a.fig}/fig6_prediction.png")
    n, tr = team_analysis(ts)
    R["team"] = tr
    fig_team(n, tr, f"{a.fig}/fig7_team.png")
    fig_leaders(car, f"{a.fig}/fig8_leaders.png")
    R["shrink"] = dict(zip(["n", "sd_raw", "sd_eb"], fig_shrink(car, f"{a.fig}/fig9_shrink.png")))
    top = car.sort_values("mic", ascending=False).head(10)[["player", "position", "last_team", "int", "mic", "chances", "convert_rate", "convert_rate_eb"]]
    R["top_mic"] = top.round(3).to_dict(orient="records")
    best = car[car.chances >= 12].sort_values("convert_rate_eb", ascending=False).head(8)[["player", "position", "last_team", "int", "mic", "chances", "convert_rate", "convert_rate_eb"]]
    R["top_eb"] = best.round(3).to_dict(orient="records")
    anchors = car[car.player.isin(["Darious Williams", "Harrison Smith"])][["player", "position", "seasons", "int", "mic", "chances", "convert_rate"]]
    R["anchors"] = anchors.round(3).to_dict(orient="records")
    with open(a.out, "w") as f:
        json.dump(R, f, indent=1, default=float)
    print(json.dumps({k: R[k] for k in ["counts", "funnel_2022_2025", "pos_group_rates", "overdispersion",
                                        "split_half", "funnel_plot", "prediction", "team", "shrink", "anchors"]},
                     indent=1, default=float))


if __name__ == "__main__":
    main()
