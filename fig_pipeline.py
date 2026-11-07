#!/usr/bin/env python3
"""Pipeline diagram for the paper (figures/fig0_pipeline.png)."""
import sys, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, SURF, PALE = "#0b0b0b", "#52514e", "#8a8984", "#fcfcfb", "#eef1f6"
out = sys.argv[1] if len(sys.argv) > 1 else "figures/fig0_pipeline.png"
os.makedirs(os.path.dirname(out), exist_ok=True)

fig, ax = plt.subplots(figsize=(7.6, 3.5))
fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis("off")


def box(x, y, w, h, lines, fc=PALE, tc=INK, fs=8.2, bold_first=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=1.2", fc=fc, ec="none"))
    n = len(lines)
    for k, t in enumerate(lines):
        yy = y + h / 2 + (n - 1) * 1.7 - k * 3.4
        ax.text(x + w / 2, yy, t, ha="center", va="center", color=tc, fontsize=fs,
                fontweight="bold" if (k == 0 and bold_first) else "normal")


def arrow(x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2, shrinkA=0, shrinkB=0))


# column 1: sources
box(1, 31, 20, 11, ["nflverse", "play-by-play", "defender credits"])
box(1, 17.5, 20, 11, ["FTN charting", "interception-worthy", "flag per throw"])
box(1, 4, 20, 11, ["nflverse rosters", "depth-chart position", "(CB / S / LB / DL)"])
# column 2: join + events
box(29, 17.5, 21, 11, ["Join + filter", "game_id, play_id", "pass attempts, REG"])
box(58, 28, 19, 11, ["Event table", "INT  or  MIC", "(DB / LB only)"], fc=BLUE, tc="white")
box(58, 8, 19, 11, ["Excluded", "DL batted balls,", "uncredited misses"], fc="#f0efec", tc=INK2)
# column 4: tables
box(80.5, 28, 19.5, 11, ["Player-season", "INT, MIC, chances", "EB conversion"])
box(80.5, 8, 19.5, 11, ["Models", "reliability, funnel,", "prediction, teams"])
for y in (36.5, 23, 9.5):
    arrow(21.4, y, 28.6, 23)
arrow(50.4, 23, 57.6, 33)
arrow(50.4, 23, 57.6, 14)
arrow(77.4, 33.5, 80.1, 33.5)
arrow(90.25, 27.6, 90.25, 19.4)
ax.text(1, 44.5, "MIC = interception-worthy, incomplete, and a DB or LB credited with the pass defense",
        fontsize=8.2, color=ORANGE, fontweight="bold", va="center")
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURF)
