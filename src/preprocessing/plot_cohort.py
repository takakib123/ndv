"""Dataset composition figure: participant race and gender.

Both panels count the same 77 participants, so they share one x scale and stay directly
comparable. Race and gender are NOMINAL, so every bar in a panel wears one colour -- shading
bars by their own height would double-encode length as hue and say nothing new.

Race labels come from extract_races.py and were reviewed before use; the Uncertain flag that
run produced is a review aid, not a property of the finished labels, so it is not drawn here.

    python -m src.preprocessing.plot_cohort
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch

from src.config import ARTIFACTS_DIR, GENDER_CSV, RACE_CSV, RACE_CLASSES

OUT_DIR = ARTIFACTS_DIR / "outputs" / "figures"

BAR_COLOR = "#2a78d6"     # blue step 450, validated against the light surface
SURFACE   = "#fcfcfb"
INK       = "#0b0b0b"
INK_2     = "#52514e"
GRID      = "#e6e5e1"

BAR_PX    = 22.0          # bar thickness, <= 24px
RADIUS_PX = 4.0           # rounded data-end


def _px_to_data(ax, px_x: float, px_y: float) -> tuple[float, float]:
    """Convert a pixel offset to x/y data units, so a corner renders round, not oval."""
    inv = ax.transData.inverted()
    x0, y0 = inv.transform((0, 0))
    x1, y1 = inv.transform((px_x, px_y))
    return abs(x1 - x0), abs(y1 - y0)


def rounded_end_bar(ax, x0: float, x1: float, y: float, h: float, color: str):
    """Horizontal bar, square at the baseline, 4px-rounded at the data end."""
    if x1 <= x0:
        return
    rx, ry = _px_to_data(ax, RADIUS_PX, RADIUS_PX)
    rx, ry = min(rx, x1 - x0), min(ry, h / 2.0)
    yb, yt = y - h / 2.0, y + h / 2.0

    if rx <= 0 or ry <= 0:
        verts = [(x0, yb), (x1, yb), (x1, yt), (x0, yt), (x0, yb)]
        codes = [MplPath.MOVETO] + [MplPath.LINETO] * 3 + [MplPath.CLOSEPOLY]
    else:
        verts = [
            (x0, yb),
            (x1 - rx, yb), (x1, yb), (x1, yb + ry),      # bottom-right corner
            (x1, yt - ry), (x1, yt), (x1 - rx, yt),      # top-right corner
            (x0, yt),
            (x0, yb),
        ]
        codes = [
            MplPath.MOVETO,
            MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3,
            MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3,
            MplPath.LINETO,
            MplPath.CLOSEPOLY,
        ]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color, edgecolor="none", zorder=3))


def load_cohort() -> pd.DataFrame:
    race = pd.read_csv(RACE_CSV)
    gender = pd.read_csv(GENDER_CSV)
    for df in (race, gender):
        df.columns = df.columns.str.strip()
    gender = gender.dropna(subset=["Subject", "Gender"])
    gender["Subject"] = gender["Subject"].str.strip()
    gender["Gender"] = gender["Gender"].str.strip().str.lower()
    race["Subject"] = race["Subject"].str.strip()
    return race.merge(gender[["Subject", "Gender"]], on="Subject", how="left")


def style_panel(ax, xmax: float):
    ax.set_xlim(0, xmax)
    ax.xaxis.grid(True, color=GRID, linewidth=1.0, linestyle="-", zorder=0)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(axis="y", length=0, labelsize=10, colors=INK)
    ax.tick_params(axis="x", length=0, labelsize=9, colors=INK_2)
    ax.set_facecolor(SURFACE)


def draw_panel(fig, ax, labels, counts, xmax, show_x_labels: bool):
    ax.set_ylim(-0.6, len(labels) - 0.4)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([l.capitalize() for l in labels])
    ax.invert_yaxis()
    style_panel(ax, xmax)
    if not show_x_labels:
        ax.tick_params(axis="x", labelbottom=False)   # shared scale is labelled once, below
    fig.canvas.draw()   # transData must be final before pixel->data conversion

    _, bar_h = _px_to_data(ax, 0, BAR_PX)
    for i, v in enumerate(counts):
        rounded_end_bar(ax, 0, v, i, bar_h, BAR_COLOR)
        ax.text(v + xmax * 0.015, i, f"{v}", va="center", ha="left",
                fontsize=9.5, color=INK_2)


def main():
    df = load_cohort()
    n = len(df)

    race_order = [r for r in sorted(RACE_CLASSES, key=lambda r: -(df["Race"] == r).sum())
                  if (df["Race"] == r).sum() > 0]
    race_counts = [int((df["Race"] == r).sum()) for r in race_order]

    gender_order = ["female", "male"]
    gender_counts = [int((df["Gender"] == g).sum()) for g in gender_order]

    xmax = max(max(race_counts), max(gender_counts)) * 1.12

    fig = plt.figure(figsize=(7.4, 4.5), dpi=200, facecolor=SURFACE)
    gs = fig.add_gridspec(2, 1, height_ratios=[len(race_order), len(gender_order)],
                          hspace=0.42, left=0.13, right=0.97, top=0.79, bottom=0.11)
    ax_r, ax_g = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    draw_panel(fig, ax_r, race_order, race_counts, xmax, show_x_labels=False)
    draw_panel(fig, ax_g, gender_order, gender_counts, xmax, show_x_labels=True)

    fig.text(0.13, 0.945, "Dataset composition", fontsize=14, color=INK,
             fontweight="semibold", ha="left")
    fig.text(0.13, 0.893, f"Naturalistic driving dataset · {n} participants · counts share one scale",
             fontsize=9.5, color=INK_2, ha="left")

    # Each panel is a single series, so no legend: the panel title names what is plotted.
    ax_r.set_title("Race  (perceived)", fontsize=10.5, color=INK, loc="left", pad=8)
    ax_g.set_title("Gender", fontsize=10.5, color=INK, loc="left", pad=8)
    ax_g.set_xlabel("Participants", fontsize=9.5, color=INK_2, labelpad=6)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = OUT_DIR / f"cohort_composition.{ext}"
        fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
        print(f"Saved {path}")

    print(f"\nRace:   {dict(zip(race_order, race_counts))}")
    print(f"Gender: {dict(zip(gender_order, gender_counts))}")


if __name__ == "__main__":
    main()
