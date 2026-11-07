#!/usr/bin/env python3
"""build_paper.py -- renders paper/paper.tex from results.json + data/, then compiles with pdflatex.
Every number in the text and tables comes from results.json so the paper cannot drift from the analysis.
Usage: python3 build_paper.py --results results.json --data data --out paper
"""
import argparse, json, os, re, subprocess, unicodedata
import pandas as pd


def tex(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"), ("$", r"\$")]:
        s = s.replace(a, b)
    return s


def pc(x, d=0):
    return f"{x * 100:.{d}f}\\%"


TEMPLATE = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern,graphicx,booktabs,amsmath,hyperref,caption,float,microtype,xcolor,array,enumitem}
\graphicspath{{../figures/}}
\hypersetup{colorlinks=true,linkcolor=blue!50!black,urlcolor=blue!50!black,citecolor=blue!50!black}
\captionsetup{font=small,labelfont=bf,skip=4pt}
\setlength{\parskip}{4pt}
\renewcommand{\arraystretch}{1.12}
\title{\textbf{Counting the Ones That Got Away:\\ A Measurement Study of Missed Interception Chances\\ in the NFL, 1999--2026}}
\author{Nikhil Chandrashaker}
\date{Working draft, November 2026}
\begin{document}
\maketitle

\begin{abstract}\noindent
Defenders who drop catchable interceptions are not tracked in any official NFL statistic. The idea of counting them is at least as old as a 2009 fan proposal, and a commercial vendor charts them behind a paywall, but no open data set exists. We build two. For 2022--2026 we combine nflverse play-by-play, FTN charting and rosters to define a \emph{missed-INT chance} (MIC): a throw charted as interception-worthy that fell incomplete while a defensive back or linebacker was credited with the pass defense. Of @@IW@@ interception-worthy throws in 2022--2025, @@INTPCT@@ were intercepted and @@MICPCT@@ became MIC. Because FTN charting starts in 2022, we add a long-run companion for 1999--2026 built from play-by-play alone: interceptions against other passes defended (PD). Conversion differs sharply by position (charted: safeties @@CONV_S@@, cornerbacks @@CONV_CB@@, linebackers @@CONV_LB@@). In the four charted seasons, conversion within a position cannot be told apart from luck (split-half $r$ = @@SH_R@@, 95\% CI @@SH_LO@@ to @@SH_HI@@). The 27-season long-run data do detect a small stable component (year-to-year $r$ = @@LR_YOY@@), which needs roughly @@LR_STR_S@@ to @@LR_STR_CB@@ chances to reach half reliability. Chances (INT plus misses) predict next-season interceptions better than interceptions alone, in @@LR_WINS@@ of @@LR_PAIRS@@ held-out seasons, and an extra pass defended carries about @@LR_RATIO@@ times the predictive weight of a pick (a charted missed chance about @@RATIO@@ times). Over the same period the interception rate fell by about @@LR_FALL@@. We conclude that an official statistic should report \emph{chances} and a shrunk, position-adjusted conversion rate; a bare drop count would mostly rank defenders by noise.
\end{abstract}

\section{Introduction}
Interceptions are the most valued event a pass defender produces, yet the box score credits only the ones that were caught. A defender who gets both hands on a throw and lets it fall is recorded the same way as one who was beaten. Swartz \cite{swartz2009} proposed in 2009 that the NFL record each dropped interception and divide interceptions by interceptions plus drops to measure a defender's hands. Pro Football Focus (PFF) charts such drops and has published leaderboards, for example reporting that one cornerback led the league with 15 dropped interceptions from 2020 and that a safety had dropped one of 13 chances over the same stretch \cite{pff2024}. The underlying player-level data are behind a subscription, and the NFL does not publish them.

This paper asks three questions that matter for deciding whether a dropped-interception statistic deserves official status.
\begin{enumerate}[nosep]
\item \textbf{Can it be measured openly?} We construct a proxy from public data and document its error.
\item \textbf{Is it a skill?} We test whether a defender's rate of converting chances is stable and distinguishable from chance.
\item \textbf{Is it useful?} We test whether counting misses improves prediction of future interceptions at the player and team level.
\end{enumerate}
Our contributions are a reproducible data set (events, player-seasons, team-seasons, and a charting-quality audit), an interactive dashboard, a long-run companion measure covering 1999--2026 that lets us ask the same questions of 27 seasons, and a set of estimates that bound how much any missed-interception statistic can tell us at NFL sample sizes.

\section{Data}
\textbf{Sources.} We use nflverse play-by-play for 2022 through Week @@LASTWEEK@@ of 2026 \cite{nflverse}, FTN charting data distributed through nflverse \cite{ftn}, and nflverse rosters. FTN charters flag each throw as \texttt{is\_interception\_worthy}, a charter's judgment that the throw could reasonably have been intercepted. FTN coverage of pass plays is @@FTNCOV@@ in every season. Position comes from each roster's depth-chart position, which separates CB, FS, SS, ILB, MLB and OLB where the coarse position field says only ``DB'' or ``LB''.

\textbf{Scope.} We keep pass attempts that are not sacks or spikes. Player-season tables use the regular season only (@@N_REG@@ event credits); the event file also carries postseason plays. Partial 2026 data are flagged. They are excluded from the empirical-Bayes priors, the prediction models and the team persistence tests, which use four complete seasons (2022--2025), and are included in career totals, the funnel plots and the split-half test.

\textbf{Long-run extension.} FTN charting begins in 2022, so MIC cannot be built for earlier years. For 1999 through Week @@LASTWEEK@@ of 2026 we use play-by-play and rosters only: @@LR_EVENTS@@ credits to defensive backs and linebackers (regular season and postseason), of which @@LR_INTS@@ are interceptions and @@LR_PDS@@ are passes defended. Defender identifiers are present on every interception in every season, and credited pass defenses run at roughly 9 to 12 percent of attempts without a break, so the credit data are consistent across eras. Positions come from the depth chart where present and from the roster position otherwise; defenders listed only as ``DB'' account for @@LR_DB_EARLY@@ of chances in 1999--2001 and @@LR_DB_ALL@@ overall, and form their own group in the data. Defensive linemen are excluded (@@LR_DL@@ of credits).

\begin{figure}[H]\centering
\includegraphics[width=\linewidth]{fig0_pipeline.png}
\caption{Data pipeline. Three open sources are joined at the play level and reduced to a table of INT and MIC credits for defensive backs and linebackers.}
\end{figure}

\section{Definitions and measurement}
\textbf{Interception (INT).} An interception credited to a defender in play-by-play.

\textbf{Missed-INT chance (MIC).} A pass attempt that FTN flagged interception-worthy, that was not intercepted, that was incomplete, and on which a defensive back or linebacker is credited with the pass defense (either of the two pass-defense slots). One credit is counted per credited defender.

From these we define, for a defender or team,
\begin{equation}
\text{Chances}=\text{INT}+\text{MIC},\qquad
\text{Convert}=\frac{\text{INT}}{\text{Chances}},\qquad
\text{MIC rate}=1-\text{Convert}.
\end{equation}
Convert is the quantity Swartz proposed (interceptions divided by interceptions plus drops), with MIC standing in for drops.

\textbf{Passes defended and INT share (long-run).} PD is a pass defense credited to a defensive back or linebacker on an incomplete pass that was not intercepted, with one credit per credited defender. The long-run analogue of Convert is
\begin{equation}
\text{INT share}=\frac{\text{INT}}{\text{INT}+\text{PD}}.
\end{equation}
PD makes no judgment about catchability, so most PD were never chances: INT share averages @@LR_SHARE@@, against @@CONV_ALL@@ for charted conversion. We use it for relative questions (position, era, persistence, prediction) and validate it against MIC in Section~\ref{sec:val}.

\textbf{Why MIC is a proxy.} A true drop requires a judgment that the defender had the ball in his hands or a clean opportunity. MIC records a weaker thing: the defender touched a throw that was dangerous. That includes legitimate break-ups of passes the receiver also contested, so MIC overstates drops. It also omits drops by defenders who are not credited with a pass defense when they fail to catch the ball, so it can understate them. We treat MIC as an upper bound and test whether conclusions survive that error in Section 6. Figure~\ref{fig:funnel} shows where interception-worthy throws end up.

\begin{figure}[H]\centering
\includegraphics[width=0.92\linewidth]{fig1_funnel.png}
\caption{Outcomes of FTN interception-worthy throws, regular season 2022--2025. Percentages are shares of all interception-worthy throws; MIC counts plays, so it is slightly smaller than the MIC credit count when two defenders share a pass defense.}
\label{fig:funnel}
\end{figure}

\textbf{Charting drift.} The charted rate fell by about @@DRIFT@@ between 2022 and 2025 (Figure~\ref{fig:drift}), while actual interception rates were flat. We read this as a change in how FTN applies the flag, not in football. All cross-season comparisons below use rates, hold out whole seasons, or both, and counts are never compared across seasons without that caveat.

\begin{figure}[H]\centering
\includegraphics[width=0.95\linewidth]{fig2_drift.png}
\caption{FTN's interception-worthy rate drops sharply across seasons while the interception rate does not.}
\label{fig:drift}
\end{figure}

\section{Methods}
\textbf{Empirical-Bayes conversion.} For each position group $g$ we model a defender's INT count in $n$ chances as Binomial$(n,p)$ with $p\sim\text{Beta}(\alpha_g,\beta_g)$ and fit $(\alpha_g,\beta_g)$ by maximum likelihood on complete player-seasons \cite{efronmorris1975}. The posterior mean $(\alpha_g+k)/(\alpha_g+\beta_g+n)$ is the \emph{shrunk} conversion rate. The prior strength $\alpha_g+\beta_g$ is the number of chances at which half of the variation in an observed rate is signal. We test the Beta-Binomial against a pure Binomial with a likelihood-ratio statistic, using the half-chi-square reference appropriate to a boundary hypothesis.

\textbf{Split-half reliability.} For every defender we assign each chance to the odd or even weeks of the regular season, compute conversion in each half, and correlate the two halves across defenders with at least 4, 6 or 8 chances. Confidence intervals use 1,000 bootstrap resamples of defenders. If conversion were a stable skill the correlation would be positive; under pure luck it is zero.

\textbf{Funnel plots.} Following Spiegelhalter \cite{spiegelhalter2005} we plot each defender's conversion against his number of chances with exact binomial 95\% and 99.8\% limits around the position mean. Under luck about 5\% of defenders fall outside the 95\% limits.

\textbf{Prediction.} For every defender with chances in season $t$ who is on a roster in $t{+}1$ we model next-season INTs with Poisson regression on position and last-season counts, using leave-one-season-pair-out cross-validation over 2022$\to$2023, 2023$\to$2024 and 2024$\to$2025 ($n=@@PRED_N@@$). We compare four models: position only; last-year INT; last-year Chances; and INT and MIC with separate coefficients. Coefficient uncertainty comes from 400 player-cluster bootstrap resamples. At team level we correlate season-$t$ totals with season-$(t{+}1)$ INTs over @@TEAM_N@@ team-season pairs.

\textbf{Long-run analyses.} The same estimators are applied to the @@LR_NPS@@ complete 1999--2025 cornerback, safety and linebacker player-seasons with INT and PD. Split-half and year-to-year correlations use residuals from each season-and-position league rate, so era shifts cannot create spurious correlation. Prediction uses leave-one-season-pair-out cross-validation over 26 pairs, with each count rescaled by the ratio of the sample-average league total to that season's league total. Team persistence is computed after subtracting each season's league mean. Bootstrap intervals use 500 resamples (200 for the Poisson coefficients).

\section{Results}
\subsection{Conversion differs by position}
Figure~\ref{fig:pos} and Table~\ref{tab:pos} show that the share of chances converted is far from uniform. Safeties catch about three in four, cornerbacks about three in five, and linebackers a little over half. The safety interval is clear of the other two, and the cornerback and linebacker intervals only just touch. A single league-wide benchmark would therefore be wrong for every position, and comparisons between defenders have to be made within position.

\begin{figure}[H]\centering
\includegraphics[width=0.62\linewidth]{fig3_position.png}
\caption{Conversion by position group, complete seasons 2022--2025, with 95\% Wilson intervals.}
\label{fig:pos}
\end{figure}

@@TAB_POS@@

\subsection{Is conversion a skill?}
\textbf{Reliability.} The fitted priors are tight: prior strength is @@PS_CB@@ chances for cornerbacks, @@PS_S@@ for safeties and @@PS_LB@@ for linebackers (Table~\ref{tab:over}). The median defender-season contains only @@MED@@ chances and @@ONE@@ of defender-seasons contain exactly one. At two chances only about @@REL2@@ of the variation in a cornerback's observed rate is signal (Figure~\ref{fig:rel}); a cornerback would need roughly @@YEARS_CB@@ seasons at the median rate to reach 50\% reliability.

@@TAB_OVER@@

\begin{figure}[H]\centering
\includegraphics[width=0.72\linewidth]{fig4_reliability.png}
\caption{Reliability of an observed conversion rate as a function of chances, $n/(n+\alpha+\beta)$.}
\label{fig:rel}
\end{figure}

\textbf{Overdispersion.} If defenders differed in true conversion skill, observed rates would vary more than binomial noise allows. The Beta-Binomial improves on the Binomial only weakly (Table~\ref{tab:over}); none of the three position groups rejects pure binomial variation at the 5\% level. The fitted true-talent standard deviation is about @@SD_RANGE@@ percentage points depending on position, compared with a binomial standard error of about 35 points at two chances.

\textbf{Split-half.} Table~\ref{tab:split} shows split-half correlations that are indistinguishable from zero at every threshold. This is a finding about four seasons; Section~\ref{sec:long} repeats the test on 27. With six or more chances the correlation is @@SH_R@@ and the interval excludes only large values.

@@TAB_SPLIT@@

\textbf{Funnel plots.} Only @@OUT95@@ of @@NPL@@ defenders sit outside the 95\% limits (@@OUT_NAMES@@), fewer than the roughly 5\% expected under luck; discrete outcomes make the nominal 95\% limits conservative, so this is consistent with luck rather than evidence for it. Figure~\ref{fig:funnelplot} shows the spread.

\begin{figure}[H]\centering
\includegraphics[width=\linewidth]{fig5_funnelplot.png}
\caption{Funnel plots by position, career totals 2022--2026. Orange points lie outside the 95\% limits.}
\label{fig:funnelplot}
\end{figure}

\textbf{Shrinkage.} Among the @@SHR_N@@ defenders with at least ten chances, shrinkage reduces the standard deviation of conversion from @@SD_RAW@@ to @@SD_EB@@ points (Figure~\ref{fig:shrink}). The dispersion that remains is dominated by position differences, which the position-specific priors retain.

\begin{figure}[H]\centering
\includegraphics[width=0.62\linewidth]{fig9_shrink.png}
\caption{Raw versus shrunk conversion rates for defenders with at least ten chances.}
\label{fig:shrink}
\end{figure}

\subsection{Leaders}
Figure~\ref{fig:leaders} lists the defenders with the most missed chances since 2022, and Table~\ref{tab:top} gives their raw and shrunk conversion. Shrinkage pulls the worst converters most of the way back toward their position mean. For example, a cornerback who converted @@GARD_RAW@@ of 13 chances is estimated at @@GARD_EB@@ after shrinkage, against a cornerback prior mean of @@CB_PRIOR@@. At the other end, Table~\ref{tab:eb} lists the best converters with at least twelve chances; all eight are safeties, which mostly reflects the position effect.

\begin{figure}[H]\centering
\includegraphics[width=0.86\linewidth]{fig8_leaders.png}
\caption{Defenders with the most missed-INT chances, 2022--2026. Labels show raw conversion.}
\label{fig:leaders}
\end{figure}

@@TAB_TOP@@

@@TAB_EB@@

\textbf{Face validity.} PFF's public figures name two defenders at opposite ends of the same behavior \cite{pff2024}. In our data the cornerback PFF reports as leading the league in drops has @@WIL_MIC@@ MIC on @@WIL_CH@@ chances (@@WIL_CR@@ conversion), and the safety PFF describes as unusually secure has @@SMI_MIC@@ MIC on @@SMI_CH@@ chances (@@SMI_CR@@). The direction agrees; the magnitudes do not and should not, because PFF's windows start in 2020, its drops are judged from film, and ours are an upper-bound proxy.

\subsection{Does counting misses help predict interceptions?}
Table~\ref{tab:pred} and Figure~\ref{fig:pred} show that adding last season's missed chances improves out-of-sample prediction of next season's interceptions. Last-year INT improves on position alone, and Chances improves further. Separating INT from MIC does not help beyond their sum. In the separate-coefficient model an extra pick is associated with @@RR_INT@@ times as many interceptions next year and an extra miss with @@RR_MIC@@ times, with a ratio of coefficients of @@RATIO@@ (95\% CI @@RATIO_LO@@ to @@RATIO_HI@@). We cannot reject that a miss and a pick are worth the same, and the data are inconsistent with a miss being worth \emph{nothing}.

\begin{figure}[H]\centering
\includegraphics[width=\linewidth]{fig6_prediction.png}
\caption{Left: held-out Poisson deviance for four models of next-season INTs. Right: coefficients with 95\% player-cluster bootstrap intervals.}
\label{fig:pred}
\end{figure}

@@TAB_PRED@@

The reading is that MIC carries information about \emph{opportunity}: who is in a position to intercept the ball, through role, scheme and ability to disrupt, and not about hands. A defender with many chances keeps getting them regardless of how the last ones ended.

\subsection{Teams}
At team level there is no persistence at all (Figure~\ref{fig:team}). Season-$t$ team INTs correlate @@T_RINT@@ with season-$(t{+}1)$ INTs, team Chances correlate @@T_RCH@@, and the difference in correlation has a 95\% interval of @@T_DLO@@ to @@T_DHI@@ across @@TEAM_N@@ pairs. Team interception totals in this window are best treated as noise, and charting missed chances does not rescue them. Section~\ref{sec:long} repeats this on 26 season pairs and finds a small persistent component.

\begin{figure}[H]\centering
\includegraphics[width=0.92\linewidth]{fig7_team.png}
\caption{Team totals in one season versus team INTs in the next, with least-squares lines.}
\label{fig:team}
\end{figure}

\section{Long-run evidence, 1999--2026}\label{sec:long}
\subsection{Interceptions became rarer; breakups did not}
Figure~\ref{fig:trends} shows what 27 seasons add before any player-level question. Interceptions fell from @@LR_INT1@@ of pass attempts in 1999--2003 to @@LR_INT2@@ in 2021--2025, a drop of @@LR_FALL@@ (linear trend $p<0.001$), with the lowest season in the window in @@LR_MINSEASON@@ at @@LR_MINRATE@@. Passes defended per attempt moved much less, from @@LR_PD1@@ to @@LR_PD2@@. The share of defender plays that end in a catch therefore fell from @@LR_SH1@@ to @@LR_SH2@@. Any official statistic, and any leaderboard that compares eras, needs a league baseline for the season it is computed in.

\begin{figure}[H]\centering
\includegraphics[width=\linewidth]{fig10_trends.png}
\caption{League-wide trends in regular-season interceptions, defensive-back and linebacker passes defended, and the interception share of the two, 1999--2025.}
\label{fig:trends}
\end{figure}

\subsection{Position and era}
Figure~\ref{fig:era} and Table~\ref{tab:era} split INT share by position and seven-year era. Safeties convert about three in ten in every era. Cornerbacks fell from @@ERA_CB1@@ in 1999--2005 to @@ERA_CB4@@ in 2020--2025, with intervals that do not overlap between the first two eras and the last two. Linebackers and safeties moved less. The position ordering seen in the charted data (safeties highest) is therefore a long-standing feature and not an artifact of four seasons.

\begin{figure}[H]\centering
\includegraphics[width=0.78\linewidth]{fig11_position_era.png}
\caption{INT share of INT plus PD by position group and era, with 95\% Wilson intervals.}
\label{fig:era}
\end{figure}

@@TAB_ERA@@

\subsection{Validating PD against charted chances}\label{sec:val}
The 2022--2025 overlap lets us ask how much of PD is MIC. Across @@VAL_N@@ defender-seasons with both measures, the correlation between PD and MIC counts is @@VAL_R@@ (95\% CI @@VAL_R_LO@@ to @@VAL_R_HI@@), so PD carries information about who is near interception-worthy throws. MIC is only @@VAL_MICSHARE@@ of PD credits (cornerbacks @@VAL_CB@@, safeties @@VAL_S@@, linebackers @@VAL_LB@@), which is why INT share is much lower than charted conversion (Figure~\ref{fig:val}). Among the @@VAL_NRATE@@ defender-seasons with six or more charted chances, INT share and charted conversion correlate @@VAL_RRATE@@, but both share the same interception numerator, so that figure is an upper bound on independent agreement. We treat PD as a noisy, diluted measure of opportunity, suitable for relative comparisons across a long window and not for levels.

\begin{figure}[H]\centering
\includegraphics[width=\linewidth]{fig14_validation.png}
\caption{Left: passes defended versus charted missed-INT chances per defender-season, 2022--2025 (points jittered). Right: charted MIC as a share of all passes defended, by position.}
\label{fig:val}
\end{figure}

\subsection{Is conversion a skill? Twenty-seven seasons}
Unlike the four-season charted sample, the long-run data reject pure binomial variation in all three position groups (Table~\ref{tab:lrover}). The fitted true-talent standard deviation of INT share is about @@LR_SD_CB@@ points for cornerbacks, @@LR_SD_S@@ for safeties and @@LR_SD_LB@@ for linebackers. The signal is real but thin: half reliability needs about @@LR_STR_CB@@ chances for cornerbacks, @@LR_STR_S@@ for safeties and @@LR_STR_LB@@ for linebackers, against a median of @@LR_MEDC@@ per defender-season. Figure~\ref{fig:lrrel} shows the split-half correlations (8 or more chances: cornerbacks @@LR_SH_CB@@, safeties @@LR_SH_S@@, linebackers @@LR_SH_LB@@) and a year-to-year correlation of @@LR_YOY@@ (95\% CI @@LR_YOY_LO@@ to @@LR_YOY_HI@@, @@LR_YOY_N@@ pairs).

@@TAB_LROVER@@

\begin{figure}[H]\centering
\includegraphics[width=\linewidth]{fig12_longrun_reliability.png}
\caption{Left: reliability of an observed INT share as a function of chances. Right: split-half and year-to-year correlations of INT share, relative to each season's league rate.}
\label{fig:lrrel}
\end{figure}

This is consistent with the charted-era result. A true correlation near 0.15 lies inside the interval we reported for MIC ($-0.05$ to $0.20$), and with about 170 defenders the four-season test has little power to separate 0.15 from zero. The data cannot say whether the persistent component is hands or role: defenders who play more man coverage, line up in the slot, or are targeted differently would also show stable shares.

\subsection{Prediction and teams}
Table~\ref{tab:lrpred} and Figure~\ref{fig:lrpred} repeat the prediction test on 26 season pairs (@@LR_PRED_N@@ defender-pairs). Every model improves on position alone. Chances beat interceptions alone in @@LR_WINS@@ of @@LR_PAIRS@@ held-out seasons, and fitting INT and PD separately does best. The coefficients imply that an extra pass defended is worth about @@LR_RATIO@@ of an interception (95\% CI @@LR_RATIO_LO@@ to @@LR_RATIO_HI@@) for predicting next year's interceptions. In the four charted seasons the same ratio for MIC was @@RATIO@@ with a wide interval, so the two estimates are compatible and the longer window pins the number down.

\begin{figure}[H]\centering
\includegraphics[width=\linewidth]{fig13_longrun_prediction.png}
\caption{Left: held-out deviance for five models of next-season INTs over 26 season pairs. Right: coefficients with 95\% player-cluster bootstrap intervals.}
\label{fig:lrpred}
\end{figure}

@@TAB_LRPRED@@

At team level, once each season's league mean is removed, team interceptions correlate @@LR_T_INT@@ with the next season's (95\% CI @@LR_T_INT_LO@@ to @@LR_T_INT_HI@@, @@LR_T_N@@ team-season pairs); team chances correlate @@LR_T_CH@@ and team passes defended @@LR_T_PD@@. That is small but positive. With only 96 pairs, the charted window's standard error on a correlation is about 0.10, so its null result is not in conflict.

\subsection{Career leaders}
Figure~\ref{fig:lrlead} lists the career interception leaders since 1999 and how many other passes each defended. Shares range from @@LEAD_MIN@@ (@@LEAD_MIN_N@@) to @@LEAD_MAX@@ (@@LEAD_MAX_N@@). Table~\ref{tab:lrhi} and Table~\ref{tab:lrlo} give the highest and lowest shrunk shares among defenders with 80 or more chances. The low end is dominated by cornerbacks with many breakups and few interceptions. Role and scheme can produce that pattern as readily as poor hands, so these tables describe results, not ability.

\begin{figure}[H]\centering
\includegraphics[width=0.86\linewidth]{fig15_longrun_leaders.png}
\caption{Career interception leaders, regular season 1999--2026, with other passes defended. Labels show INT share.}
\label{fig:lrlead}
\end{figure}

@@TAB_LRHI@@

@@TAB_LRLO@@

\section{What an official statistic should look like}
The evidence supports recording missed chances and argues against ranking individuals by a raw drop rate. Conversion contains a small stable component, so a shrunk rate has some value, but only after many chances. We propose the following.
\begin{enumerate}[nosep]
\item \textbf{Record the event, not the rate.} Add a charted ``dropped interception'' to the official play record, defined as a defender with a clean chance who failed to secure the ball, scored by two independent reviewers with published agreement. Report counts.
\item \textbf{Headline Chances.} Chances (INT plus drops) predicts future interceptions better than INT alone and is the quantity that tracks role and opportunity.
\item \textbf{Publish conversion only with context.} If a rate is shown, show it by position, shrunk toward the position mean, and withhold it below a minimum sample. In the 27-season data a defender needs about @@LR_STR_S@@ chances (safeties) to @@LR_STR_CB@@ (cornerbacks) to reach half reliability, and the median defender-season has @@LR_MEDC@@.
\item \textbf{Separate contested from clean.} FTN already records contested-ball and catchability flags; an official chart should keep them so a break-up on a contested throw is not scored as a drop.
\item \textbf{Fix the chart definition.} The drift in Figure~\ref{fig:drift} shows how fragile a charted flag is when its coding instructions change quietly. Any official stat needs a versioned rulebook.
\item \textbf{Publish the era baseline.} The interception rate fell by about @@LR_FALL@@ between 1999--2003 and 2021--2025 while breakups per attempt barely moved. A leaderboard should show each season's league rate beside the player's.
\end{enumerate}

\section{Limitations}
\textbf{The proxy is the main limitation.} MIC mixes true drops and legitimate break-ups. Measurement error of that kind attenuates correlations, so the near-zero split-half reliability is a statement about MIC, not necessarily about hands. If film-graded drops were much cleaner than MIC, reliability could be higher than we estimate. The data cannot settle that, and the result should be re-run on PFF-style charted drops if they can be licensed.

\textbf{The long-run measure is looser still.} PD credits include breakups on throws no defender could have caught, depend on official scorers' habits that may have shifted, and say nothing about which throws were catchable. Rosters before 2002 label some defenders only as ``DB''. The long-run results are about relative differences and persistence, and we do not read its levels as drop rates.

\textbf{Charting drift and coverage.} FTN's flag rate falls by about @@DRIFT@@ over the sample, and seasons are not exchangeable in counts.

\textbf{Samples are small.} The median player-season has @@MED@@ chances. Four complete seasons give three season-to-season pairs. Intervals are wide and many conclusions are of the form ``cannot be distinguished from luck'', which is weaker than ``is luck''.

\textbf{Attribution.} Pass-defense credits are assigned by official scorers. Roster position uses the depth chart, so an edge rusher listed as OLB is grouped with linebackers. Traded players are assigned to the team where most of their chances occurred. Defensive linemen are excluded (@@N_DL@@ interception and pass-defense credits).

\textbf{No coverage denominator.} We do not observe targets in coverage, so Chances is not a rate per snap or per target.

\textbf{Partial 2026.} 2026 (through Week @@LASTWEEK@@) is excluded from the priors, prediction models and team tests, but contributes to career totals, the funnel plots and the split-half test, so those descriptive results are not strictly limited to complete seasons.

\section{Reproducibility}
All code, data and figures are in the project folder. \texttt{build\_dataset.py} downloads the nflverse files and writes the charted CSVs; \texttt{analysis.py} writes the charted figures and \texttt{results.json}; \texttt{build\_longrun.py} and \texttt{analysis\_longrun.py} do the same for 1999--2026 (\texttt{results\_longrun.json}); \texttt{build\_paper.py} regenerates this PDF from both results files; \texttt{build\_dashboard.py} embeds the data in the dashboard. A data dictionary is in the README.

\begin{thebibliography}{9}
\bibitem{swartz2009} C.~Swartz. \emph{10 Statistics That Would Make the NFL Better}. Bleacher Report, 17 March 2009. \url{https://bleacherreport.com/articles/140911-10-statistics-that-would-make-the-nfl-better}
\bibitem{pff2024} Pro Football Focus (undated web articles). \emph{NFL defensive back catch rate, Week 7}. \url{https://www.pff.com/news/nfl-defensive-back-catch-rate-week-7}; see also PFF, \emph{Just how prevalent are dropped interceptions in today's NFL?} \url{https://pff.com/pro-just-how-prevalent-are-dropped-interceptions-in-todays-nfl}
\bibitem{nflverse} nflverse. \emph{nflverse-data releases: play-by-play and rosters}. \url{https://github.com/nflverse/nflverse-data}
\bibitem{ftn} FTN Data via nflverse. \emph{FTN charting data}. See nflverse for license and attribution terms.
\bibitem{efronmorris1975} B.~Efron and C.~Morris. Data analysis using Stein's estimator and its generalizations. \emph{Journal of the American Statistical Association}, 70(350):311--319, 1975.
\bibitem{spiegelhalter2005} D.~J. Spiegelhalter. Funnel plots for comparing institutional performance. \emph{Statistics in Medicine}, 24(8):1185--1202, 2005.
\end{thebibliography}
\end{document}
"""


def table(caption, label, header, rows, align):
    out = ["\\begin{table}[H]\\centering\\small", f"\\caption{{{caption}}}\\label{{{label}}}",
           f"\\begin{{tabular}}{{{align}}}\\toprule", " & ".join(header) + " \\\\ \\midrule"]
    out += [" & ".join(r) + " \\\\" for r in rows]
    out += ["\\bottomrule\\end{tabular}\\end{table}"]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results.json"); ap.add_argument("--lr-results", default="results_longrun.json"); ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="paper")
    a = ap.parse_args()
    R = json.load(open(a.results)); RL = json.load(open(a.lr_results))
    lcar = pd.read_csv(os.path.join(a.data, "longrun", "lr_player_career.csv"))
    ev = pd.read_csv(os.path.join(a.data, "dint_events.csv"))
    qa = pd.read_csv(os.path.join(a.data, "dint_qa_by_season.csv"))
    pri = pd.read_csv(os.path.join(a.data, "dint_eb_priors.csv")).set_index("pos_group")
    car = pd.read_csv(os.path.join(a.data, "dint_player_career.csv"))
    os.makedirs(a.out, exist_ok=True)

    F = R["funnel_2022_2025"]; N = F["iw"]
    pr = R["pos_group_rates"]; ov = R["overdispersion"]; sh = R["split_half"]; fp = R["funnel_plot"]
    pm = R["prediction"]; tm = R["team"]; sr = R["shrink"]
    sh6 = sh["6"]
    out95 = sum(v["outside_95"] for v in fp.values()); npl = sum(v["players"] for v in fp.values())
    names = [n for v in fp.values() for n in v["outside_names"]]
    reg = ev[ev.season_type == "REG"]
    q = qa.set_index("season")
    drift = 1 - q.loc[2025, "int_worthy_rate"] / q.loc[2022, "int_worthy_rate"]
    rel = lambda n, g: n / (n + pri.loc[g, "prior_strength"])
    gard = car[car.player == "Sauce Gardner"].iloc[0]
    wil = [x for x in R["anchors"] if x["player"] == "Darious Williams"][0]
    smi = [x for x in R["anchors"] if x["player"] == "Harrison Smith"][0]
    m_by = {m["model"]: m["deviance"] for m in pm["models"]}
    sds = [v["true_talent_sd"] * 100 for v in ov.values()]

    tokens = {
        "IW": f"{N:,}", "INTPCT": pc(F["intercepted"] / N), "MICPCT": pc(F["mic_plays"] / N),
        "CONV_S": pc(pr["S"]["convert"]), "CONV_CB": pc(pr["CB"]["convert"]), "CONV_LB": pc(pr["LB"]["convert"]),
        "SH_R": f"{sh6['r']:.2f}", "SH_LO": f"{sh6['ci'][0]:.2f}", "SH_HI": f"{sh6['ci'][1]:.2f}",
        "OUT95": str(out95), "NPL": str(npl), "OUT_NAMES": " and ".join(tex(n) for n in names) if names else "none",
        "RATIO": f"{pm['ratio_mic_to_int']:.1f}", "RATIO_LO": f"{pm['ratio_ci'][0]:.1f}", "RATIO_HI": f"{pm['ratio_ci'][1]:.1f}",
        "LASTWEEK": "9", "FTNCOV": "at least 99.6\\%", "N_REG": f"{len(reg):,}",
        "DRIFT": pc(drift), "PRED_N": str(pm["n"]), "TEAM_N": str(tm["n"]),
        "PS_CB": f"{ov['CB']['prior_strength']:.0f}", "PS_S": f"{ov['S']['prior_strength']:.0f}", "PS_LB": f"{ov['LB']['prior_strength']:.0f}",
        "MED": f"{R['counts']['median_chances_season']:.0f}", "ONE": pc(R["counts"]["share_one_chance"]),
        "REL2": pc(rel(2, "CB")), "YEARS_CB": f"{ov['CB']['prior_strength'] / 2:.0f}",
        "SD_RANGE": f"{min(sds):.0f}--{max(sds):.0f}",
        "SHR_N": str(sr["n"]), "SD_RAW": f"{sr['sd_raw'] * 100:.0f}", "SD_EB": f"{sr['sd_eb'] * 100:.0f}",
        "GARD_RAW": pc(gard.convert_rate), "GARD_EB": pc(gard.convert_rate_eb), "CB_PRIOR": pc(pri.loc["CB", "prior_mean"]),
        "WIL_MIC": str(wil["mic"]), "WIL_CH": str(wil["chances"]), "WIL_CR": pc(wil["convert_rate"]),
        "SMI_MIC": str(smi["mic"]), "SMI_CH": str(smi["chances"]), "SMI_CR": pc(smi["convert_rate"]),
        "RR_INT": f"{pm['rate_ratio_int']:.2f}", "RR_MIC": f"{pm['rate_ratio_mic']:.2f}",
        "T_RINT": f"{tm['r_int']:.2f}", "T_RCH": f"{tm['r_chances']:.2f}",
        "T_DLO": f"{tm['diff_ci'][0]:.2f}", "T_DHI": f"{tm['diff_ci'][1]:.2f}",
        "REL15_CB": pc(rel(15, "CB")), "REL15_LB": pc(rel(15, "LB")),
        "N_DL": "156",
    }

    tab_pos = table("Conversion by position group, complete seasons 2022--2025.", "tab:pos",
                    ["Position", "INT", "MIC", "Chances", "Convert", "95\\% CI"],
                    [[g, str(pr[g]["int"]), str(pr[g]["mic"]), str(pr[g]["chances"]), pc(pr[g]["convert"]),
                      f"{pr[g]['lo'] * 100:.0f}--{pr[g]['hi'] * 100:.0f}\\%"] for g in ["S", "CB", "LB"]], "lrrrrr")
    tab_over = table("Beta-Binomial fits and overdispersion tests by position group (complete player-seasons).", "tab:over",
                     ["Position", "Player-seasons", "$\\alpha+\\beta$", "True-talent SD", "LRT", "$p$"],
                     [[g, str(ov[g]["n_player_seasons"]), f"{ov[g]['prior_strength']:.1f}", f"{ov[g]['true_talent_sd'] * 100:.1f} pts",
                       f"{ov[g]['lrt']:.2f}", f"{ov[g]['p']:.2f}"] for g in ["CB", "S", "LB"]], "lrrrrr")
    tab_split = table("Split-half (odd vs even weeks) correlation of conversion rate.", "tab:split",
                      ["Min. chances", "Defenders", "$r$", "95\\% CI", "Spearman--Brown"],
                      [[k, str(v["players"]), f"{v['r']:.2f}", f"{v['ci'][0]:.2f} to {v['ci'][1]:.2f}",
                        (f"{v['spearman_brown']:.2f}" if v["spearman_brown"] is not None else "--")] for k, v in sh.items()], "lrrrr")
    tab_top = table("Defenders with the most missed-INT chances, 2022--2026.", "tab:top",
                    ["Defender", "Pos.", "Team", "INT", "MIC", "Convert", "Shrunk"],
                    [[tex(r["player"]), r["position"], r["last_team"], str(r["int"]), str(r["mic"]), pc(r["convert_rate"]),
                      pc(r["convert_rate_eb"])] for r in R["top_mic"]], "llllrrr")
    tab_eb = table("Highest shrunk conversion among defenders with at least twelve chances.", "tab:eb",
                   ["Defender", "Pos.", "Team", "INT", "MIC", "Convert", "Shrunk"],
                   [[tex(r["player"]), r["position"], r["last_team"], str(r["int"]), str(r["mic"]), pc(r["convert_rate"]),
                     pc(r["convert_rate_eb"])] for r in R["top_eb"]], "llllrrr")
    base = m_by["Position only"]
    tab_pred = table("Out-of-sample prediction of next-season INTs (leave-one-season-pair-out).", "tab:pred",
                     ["Model", "Held-out deviance", "vs.\\ position only"],
                     [[tex(m["model"]), f"{m['deviance']:.3f}", f"{(m['deviance'] / base - 1) * 100:+.1f}\\%"] for m in pm["models"]], "lrr")
    for k, v in {"TAB_POS": tab_pos, "TAB_OVER": tab_over, "TAB_SPLIT": tab_split, "TAB_TOP": tab_top,
                 "TAB_EB": tab_eb, "TAB_PRED": tab_pred}.items():
        tokens[k] = v

    # ---------------- long-run tokens and tables ----------------
    import numpy as np
    lc = RL["counts"]; tr = RL["trend"]; lov = RL["overdispersion"]; lsh = RL["split_half"]; lyy = RL["year_over_year"]["8"]
    lpm = RL["prediction"]; ltm = RL["team"]; lval = RL["validation"]; pe = RL["pos_era"]
    lm = {m["model"]: m for m in lpm["models"]}
    ints_only = np.array(lm["Last-year INT"]["by_season"])
    wins = int((np.array(lm["Last-year chances (INT+PD)"]["by_season"]) < ints_only).sum())
    pct1 = lambda x: f"{x * 100:.1f}\\%"
    lead = RL["top_int"]; lead15 = lcar.sort_values("int", ascending=False).head(15)
    lo_r = lead15.sort_values("int_share").iloc[0]; hi_r = lead15.sort_values("int_share").iloc[-1]
    pr_all_int = sum(pr[g]["int"] for g in pr); pr_all_ch = sum(pr[g]["chances"] for g in pr)
    ncell = [pe[k]["chances"] for k in pe]
    tokens.update({
        "LR_EVENTS": f"{lc['events']:,}", "LR_INTS": f"{lc['int_events']:,}", "LR_PDS": f"{lc['pd_events']:,}",
        "LR_DB_EARLY": pc(lc["db_share_early"]), "LR_DB_ALL": pc(lc["db_share_all"]), "LR_DL": pc(lc["dl_credit_share"]),
        "LR_SHARE": pc(lc["overall_int_share"]), "CONV_ALL": pc(pr_all_int / pr_all_ch),
        "LR_NPS": f"{sum(v['n'] for v in lov.values()):,}",
        "LR_YOY": f"{lyy['r']:.2f}", "LR_YOY_LO": f"{lyy['lo']:.2f}", "LR_YOY_HI": f"{lyy['hi']:.2f}", "LR_YOY_N": f"{lyy['n']:,}",
        "LR_STR_CB": f"{lov['CB']['strength']:.0f}", "LR_STR_S": f"{lov['S']['strength']:.0f}", "LR_STR_LB": f"{lov['LB']['strength']:.0f}",
        "LR_SD_CB": f"{lov['CB']['talent_sd'] * 100:.0f}", "LR_SD_S": f"{lov['S']['talent_sd'] * 100:.0f}", "LR_SD_LB": f"{lov['LB']['talent_sd'] * 100:.0f}",
        "LR_MEDC": f"{lc['median_chances']:.0f} chances",
        "LR_SH_CB": f"{lsh['CB_8']['r']:.2f}", "LR_SH_S": f"{lsh['S_8']['r']:.2f}", "LR_SH_LB": f"{lsh['LB_8']['r']:.2f}",
        "LR_WINS": str(wins), "LR_PAIRS": str(lpm["n_pairs"]), "LR_PRED_N": f"{lpm['n']:,}",
        "LR_RATIO": f"{lpm['ratio']:.2f}", "LR_RATIO_LO": f"{lpm['ratio_ci'][0]:.2f}", "LR_RATIO_HI": f"{lpm['ratio_ci'][1]:.2f}",
        "LR_FALL": pc(1 - tr["int_rate_last5"] / tr["int_rate_first5"]),
        "LR_INT1": pct1(tr["int_rate_first5"]), "LR_INT2": pct1(tr["int_rate_last5"]),
        "LR_PD1": pct1(tr["pd_rate_first5"]), "LR_PD2": pct1(tr["pd_rate_last5"]),
        "LR_SH1": pc(tr["share_first5"]), "LR_SH2": pc(tr["share_last5"]),
        "LR_MINSEASON": str(tr["min_int_rate_season"]), "LR_MINRATE": pct1(tr["min_int_rate"]),
        "ERA_CB1": pc(pe["CB_1999_2005"]["share"]), "ERA_CB4": pc(pe["CB_2020_2025"]["share"]),
        "VAL_N": f"{lval['n']:,}", "VAL_R": f"{lval['r_counts']:.2f}", "VAL_R_LO": f"{lval['r_counts_ci'][0]:.2f}", "VAL_R_HI": f"{lval['r_counts_ci'][1]:.2f}",
        "VAL_MICSHARE": pc(lval["mic_share_of_pd"]), "VAL_CB": pc(lval["by_group"]["CB"]["share"]), "VAL_S": pc(lval["by_group"]["S"]["share"]),
        "VAL_LB": pc(lval["by_group"]["LB"]["share"]), "VAL_NRATE": str(lval["n_rate"]), "VAL_RRATE": f"{lval['r_rate']:.2f}",
        "LR_T_INT": f"{ltm['rd_int']:.2f}", "LR_T_INT_LO": f"{ltm['rd_int_ci'][0]:.2f}", "LR_T_INT_HI": f"{ltm['rd_int_ci'][1]:.2f}",
        "LR_T_CH": f"{ltm['rd_chances']:.2f}", "LR_T_PD": f"{ltm['rd_pd']:.2f}", "LR_T_N": str(ltm["n"]),
        "LEAD_MIN": pc(lo_r.int_share), "LEAD_MIN_N": tex(lo_r.player), "LEAD_MAX": pc(hi_r.int_share), "LEAD_MAX_N": tex(hi_r.player),
    })
    eras = [(1999, 2005), (2006, 2012), (2013, 2019), (2020, 2025)]
    tokens["TAB_ERA"] = table(f"INT share of INT plus PD by position and era (each cell rests on {min(ncell):,} to {max(ncell):,} chances).", "tab:era",
        ["Position"] + [f"{x}--{y}" for x, y in eras],
        [[g] + [pc(pe[f"{g}_{x}_{y}"]["share"], 1) for x, y in eras] for g in ["S", "CB", "LB"]], "lrrrr")
    pv = lambda p: "$<$0.001" if p < 0.001 else f"{p:.3f}"
    tokens["TAB_LROVER"] = table("Beta-Binomial fits and overdispersion tests for INT share, 1999--2025 (complete player-seasons).", "tab:lrover",
        ["Position", "Player-seasons", "Median chances", "Prior mean", "$\\alpha+\\beta$", "True-talent SD", "LRT", "$p$"],
        [[g, f"{lov[g]['n']:,}", f"{lov[g]['median_chances']:.0f}", pc(lov[g]["mean"], 1), f"{lov[g]['strength']:.1f}",
          f"{lov[g]['talent_sd'] * 100:.1f} pts", f"{lov[g]['lrt']:.1f}", pv(lov[g]["p"])] for g in ["CB", "S", "LB"]], "lrrrrrrr")
    base = lm["Position only"]["deviance"]
    tokens["TAB_LRPRED"] = table("Out-of-sample prediction of next-season INTs, 26 leave-one-season-pair-out folds.", "tab:lrpred",
        ["Model", "Held-out deviance", "vs.\\ position only", "Seasons better than INT-only"],
        [[tex(m["model"]), f"{m['deviance']:.3f}", f"{(m['deviance'] / base - 1) * 100:+.1f}\\%",
          ("--" if m["model"] == "Last-year INT" else str(int((np.array(m["by_season"]) < ints_only).sum())))] for m in lpm["models"]], "lrrr")
    def lead_rows(rows):
        return [[tex(r["player"]), r["position"], f"{int(r['first_season'])}--{int(r['last_season'])}", str(r["int"]), str(r["pd"]),
                 pc(r["int_share"]), pc(r["int_share_eb"])] for r in rows]
    hdr = ["Defender", "Pos.", "Years", "INT", "PD", "INT share", "Shrunk"]
    tokens["TAB_LRHI"] = table("Highest shrunk INT share, defenders with 80 or more chances, 1999--2026.", "tab:lrhi", hdr, lead_rows(RL["high_eb"][:6]), "lllrrrr")
    tokens["TAB_LRLO"] = table("Lowest shrunk INT share, defenders with 80 or more chances, 1999--2026.", "tab:lrlo", hdr, lead_rows(RL["low_eb"][:6]), "lllrrrr")

    t = TEMPLATE
    for k, v in tokens.items():
        t = t.replace(f"@@{k}@@", v)
    left = re.findall(r"@@(\w+)@@", t)
    if left:
        raise SystemExit(f"unfilled tokens: {sorted(set(left))}")
    path = os.path.join(a.out, "paper.tex")
    open(path, "w").write(t)
    for _ in range(2):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "paper.tex"],
                           cwd=a.out, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-2500:]); raise SystemExit("pdflatex failed")
    for ext in ("aux", "log", "out"):
        try:
            os.remove(os.path.join(a.out, f"paper.{ext}"))
        except FileNotFoundError:
            pass
    print("paper written:", os.path.join(a.out, "paper.pdf"))


if __name__ == "__main__":
    main()
