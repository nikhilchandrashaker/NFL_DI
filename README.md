# Missed Interception Chances (DINT) — NFL, 1999–2026

An open, reproducible data set, dashboard and research paper on **dropped interceptions by defenders**, a statistic the NFL does not record.

No public source records true dropped interceptions (PFF charts them behind a paywall). This project builds two **proxies** from open nflverse data and tests what they can and cannot tell us.

| Measure | Years | Definition | Strength |
|---|---|---|---|
| **MIC** (missed-INT chance) | 2022–2026 | A throw FTN charters flagged *interception-worthy* that was **incomplete**, with a **DB or LB credited with the pass defense** | Closer to a real chance; an **upper bound** on true drops (also includes legitimate break-ups) |
| **PD** (pass defended) | 1999–2026 | A DB/LB pass defense on an incomplete, non-intercepted pass. **INT share** = INT / (INT + PD) | Covers 27 seasons, but most breakups were never catchable, so levels are low (~22%) and only relative comparisons are safe |

FTN's interception-worthy flag only exists from 2022, so MIC cannot go further back. PD is the best available measure before that.

## Headline findings

| Question | Result |
|---|---|
| Where do 2,841 interception-worthy throws (REG 2022–25) end up? | 49% intercepted, 30% MIC, 3% DL batted, 15% no pass-defense credit, 3% completed |
| Does conversion differ by position? | Yes, in both measures. Charted: safeties 73%, cornerbacks 62%, linebackers 54%. INT share 1999–2025: S 31%, CB 20%, LB 19% |
| Is conversion a stable skill? | **Small but real.** 2022–25 alone cannot tell it from luck (split-half r = 0.08, CI −0.05 to 0.20). Over 27 seasons, year-to-year r = 0.16 (CI 0.10 to 0.21, 1,147 pairs); needs ~22 to 46 chances for half reliability |
| Does counting misses predict next year's INTs? | Yes. Chances beat INT alone in 22 of 26 held-out seasons; an extra pass defended is worth ~0.45 of an interception (CI 0.38–0.54); a charted MIC ~0.7 (CI 0.2–1.4) |
| Do team totals persist? | 2022–25: no (r ≈ 0.00, 96 pairs). 1999–2025 after removing league trend: r = 0.14 (CI 0.08–0.22, 829 pairs) |
| What happened to interceptions? | Rate fell **30%**: 3.3% of attempts (1999–2003) to 2.3% (2021–2025), while breakups per attempt barely moved |
| Does PD track charted chances? | Correlation 0.37 across defender-seasons; only 14% of breakups were charted interception-worthy |

**Implication.** Track and publish **Chances** (INT + drops). A bare drop rate would mostly rank defenders by noise: the median defender-season has 4 chances. If a rate is shown, show it by position and shrunk toward the position mean (`convert_rate_eb` / `int_share_eb`), and publish the season's league rate beside it.

## What's here

```
README.md
paper/paper.pdf, paper.tex      18-page research paper (figures, models, tables)
dashboard/index.html            single-file dashboard with search and a measure toggle
data/                           charted (2022-26) data set, CSVs
data/longrun/                   long-run (1999-2026) data set
figures/                        all paper figures (PNG)
results.json                    every number from the charted analysis
results_longrun.json            every number from the long-run analysis
src/                            code that regenerates everything
```

## Reproduce

Requires Python 3 with `pandas numpy scipy matplotlib scikit-learn`, and `pdflatex` for the paper.

```bash
# charted era (2022-2026)
python3 src/build_dataset.py --raw raw --out data --download      # downloads nflverse files
python3 src/analysis.py --raw raw --data data --fig figures --out results.json
python3 src/fig_pipeline.py figures/fig0_pipeline.png

# long-run era (1999-2026): needs pbp and rosters for 1999-2021 in raw/ (~450 MB)
python3 src/build_longrun.py --raw raw --out data/longrun
python3 src/analysis_longrun.py --raw raw --data data --fig figures --out results_longrun.json

python3 src/build_paper.py --results results.json --lr-results results_longrun.json --data data --out paper
python3 src/build_dashboard.py --data data --raw raw --results results.json --lr-results results_longrun.json --out dashboard
python3 src/checks.py --data data --raw raw                       # 47 consistency checks
```

`build_longrun.py` expects the raw files already downloaded (`pbp_YYYY.csv.gz`, `roster_YYYY.csv` from the nflverse-data releases). The paper's numbers are filled in from the two results files, so it cannot drift from the analysis. 2026 is partial and nflverse updates weekly, so re-running later will change 2026 figures.

## Data dictionary

### Charted (2022–2026), in `data/`

**`dint_events.csv`**: one row per INT or MIC credit (DB/LB only; regular and postseason)

| column | meaning |
|---|---|
| season, season_type, week | REG or POST |
| game_id, play_id | nflverse keys |
| event_type | `INT` or `MIC` |
| player_id | GSIS id of the defender |
| player_name_pbp | name as printed in play-by-play |
| position, pos_group | roster depth-chart position and its group (CB, S, DB, LB) |
| defteam, posteam | defense and offense |
| passer_player_name, receiver_player_name, air_yards | from play-by-play |
| is_catchable_ball, is_contested_ball | FTN flags |
| desc | play description |

**`dint_player_seasons.csv`**: regular season only

| column | meaning |
|---|---|
| player_id, player, position, pos_group | identity (name from roster) |
| season, team, teams | primary team and all teams that season |
| int, mic, chances | counts; chances = int + mic |
| convert_rate, mic_rate | int / chances; 1 − convert_rate |
| convert_rate_eb, mic_rate_eb | empirical-Bayes shrunk rates (priors per position, fit on complete seasons) |
| lg_convert_rate, exp_int, int_over_exp | league rate for season × position, expected INTs, INTs minus expected |
| partial_season | True for 2026 |

Also: `dint_player_career.csv`, `dint_team_seasons.csv`, `dint_qa_by_season.csv` (FTN coverage and the charting drift), `dint_eb_priors.csv`.

### Long-run (1999–2026), in `data/longrun/`

| file | contents |
|---|---|
| `lr_events.csv.gz` | one row per INT or PD credit to a DB/LB: season, season_type, week, game_id, play_id, event_type (`INT`/`PD`), player_id, pos_group, defteam |
| `lr_player_seasons.csv` | regular season: player_id, player, position, pos_group, season, team, `int`, `pd`, `chances` (= int + pd), `int_share`, `int_share_eb` (shrunk), `lg_int_share`, `exp_int`, `int_over_exp`, partial_season |
| `lr_player_career.csv` | 1999–2026 totals with first/last season, `int_share`, `int_share_eb` |
| `lr_team_seasons.csv` | team, season, int, pd, chances, int_share |
| `lr_league_seasons.csv` | pass attempts, interceptions, DB/LB INT and PD, interception rate, INT share per season |
| `lr_credit_audit.csv` | credits per season, and how many were dropped as DL or unmatched |
| `lr_eb_priors.csv` | Beta-Binomial prior per position group |

`pos_group` is CB, S, LB, or DB (a defender listed only as "DB" on early rosters: 11% of chances in 1999–2001, 2% overall).

## Caveats

- **Both measures are proxies.** MIC mixes true drops and legitimate break-ups. PD is looser still: most breakups were never catchable. Measurement error attenuates correlations, so a null result is a statement about the proxy, not proof that hands do not matter.
- **Persistent component may be role, not hands.** The long-run correlation (~0.15) cannot separate catching ability from coverage role or scheme.
- **FTN's interception-worthy flag fell ~38% between 2022 and 2025** while the interception rate stayed flat, so charted counts are not comparable across seasons. PD shows no such drift.
- **Small samples per defender.** Median defender-season has 4 chances (2 for linebackers).
- **Positions** come from rosters, so edge rushers listed as OLB count as linebackers. Defensive linemen are excluded (13% of credits).
- **No coverage denominator** (targets or snaps in coverage), so chances are counts, not rates per opportunity.
- The face-validity check against PFF's public figures (Darious Williams, Harrison Smith) is directional only: their windows start in 2020 and their drops are judged from film.

## Sources and attribution

- nflverse play-by-play and rosters: https://github.com/nflverse/nflverse-data
- FTN charting data, distributed through nflverse (check nflverse for license and attribution terms before redistributing the FTN-derived columns)
- C. Swartz, *10 Statistics That Would Make the NFL Better*, Bleacher Report, 17 March 2009 (proposed the stat)
- Pro Football Focus articles on dropped interceptions (cited in the paper)

Analysis code was drafted with Claude.
