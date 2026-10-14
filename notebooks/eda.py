import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

# This script lives at <root>/notebooks/eda.py; resolve paths relative to the repo root.
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "data" / "DEFADSMasterTableV1.csv"
OUTPUT_DIR = BASE_DIR / "artifacts" / "eda_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120


# ── Load ──────────────────────────────────────────────────────────────────────

df = pd.read_csv(DATA_FILE)
df.columns = df.columns.str.strip()

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Shape          : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Columns        : {list(df.columns)}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nDuplicates     : {df.duplicated().sum()}")


# ── Parse time columns ────────────────────────────────────────────────────────

def mmss_to_seconds(t):
    """Convert MM:SS.f string to float seconds."""
    try:
        parts = str(t).split(":")
        return float(parts[0]) * 60 + float(parts[1])
    except Exception:
        return np.nan

df["StartSec"] = df["StartTime"].apply(mmss_to_seconds)
df["EndSec"]   = df["EndTime"].apply(mmss_to_seconds)
df["DurationSec"] = df["EndSec"] - df["StartSec"]
df["DurationFrames"] = df["EndFrame"] - df["StartFrame"]


# ── Basic statistics ──────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("CATEGORICAL DISTRIBUTIONS")
print("=" * 60)
for col in ["VideoType", "Subject", "Camera", "Label"]:
    vc = df[col].value_counts()
    print(f"\n{col} ({vc.shape[0]} unique):\n{vc.to_string()}")

print("\n" + "=" * 60)
print("NUMERIC STATISTICS (duration / frames)")
print("=" * 60)
print(df[["StartSec", "EndSec", "DurationSec", "StartFrame", "EndFrame", "DurationFrames"]].describe().round(2))


# ── Figure 1 – Label distribution ─────────────────────────────────────────────

label_counts = df.groupby("Label")["DurationSec"].count().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(12, 5))
label_counts.plot(kind="bar", ax=ax, color=sns.color_palette("muted", len(label_counts)))
ax.set_title("Annotation Count per Label")
ax.set_xlabel("Label")
ax.set_ylabel("Count")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "label_counts.png")
plt.close()
print("\nSaved -> artifacts/eda_output/label_counts.png")


# ── Figure 2 – Duration distribution per label ────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 5))
order = df.groupby("Label")["DurationSec"].median().sort_values(ascending=False).index
sns.boxplot(data=df, x="Label", y="DurationSec", order=order, ax=ax, palette="muted")
ax.set_title("Annotation Duration (s) per Label")
ax.set_xlabel("Label")
ax.set_ylabel("Duration (s)")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "duration_by_label.png")
plt.close()
print("Saved -> artifacts/eda_output/duration_by_label.png")


# ── Figure 3 – Duration distribution per camera ───────────────────────────────

fig, ax = plt.subplots(figsize=(7, 4))
sns.boxplot(data=df, x="Camera", y="DurationSec", ax=ax, palette="muted")
ax.set_title("Annotation Duration (s) by Camera")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "duration_by_camera.png")
plt.close()
print("Saved -> artifacts/eda_output/duration_by_camera.png")


# ── Figure 4 – Annotation count per subject ───────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 4))
sub_counts = df.groupby(["Subject", "Camera"]).size().unstack(fill_value=0)
sub_counts.plot(kind="bar", ax=ax, colormap="tab10")
ax.set_title("Annotation Count per Subject × Camera")
ax.set_xlabel("Subject")
ax.set_ylabel("Count")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
ax.legend(title="Camera", bbox_to_anchor=(1.01, 1), loc="upper left")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "count_subject_camera.png")
plt.close()
print("Saved -> artifacts/eda_output/count_subject_camera.png")


# ── Figure 5 – Label heatmap per subject ─────────────────────────────────────

pivot = df.pivot_table(index="Subject", columns="Label", values="DurationSec",
                       aggfunc="count", fill_value=0)
fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns) * 0.6), max(5, len(pivot) * 0.4)))
sns.heatmap(pivot, annot=True, fmt="d", cmap="YlOrRd", linewidths=0.4, ax=ax)
ax.set_title("Annotation Count — Subject × Label")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "heatmap_subject_label.png")
plt.close()
print("Saved -> artifacts/eda_output/heatmap_subject_label.png")


# ── Figure 6 – VideoType split ────────────────────────────────────────────────

vtype_counts = df["VideoType"].value_counts()
fig, ax = plt.subplots(figsize=(5, 4))
ax.pie(vtype_counts, labels=vtype_counts.index, autopct="%1.1f%%",
       colors=sns.color_palette("muted", len(vtype_counts)))
ax.set_title("VideoType Distribution")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "videotype_pie.png")
plt.close()
print("Saved -> artifacts/eda_output/videotype_pie.png")


# ── Figure 7 – Duration histogram ─────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(df["DurationSec"].dropna(), bins=30, color=sns.color_palette("muted")[0], edgecolor="white")
ax.set_title("Distribution of Annotation Durations")
ax.set_xlabel("Duration (s)")
ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "duration_histogram.png")
plt.close()
print("Saved -> artifacts/eda_output/duration_histogram.png")


# ── Figure 8 – Total annotated time per label ────────────────────────────────

total_dur = df.groupby("Label")["DurationSec"].sum().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(12, 5))
total_dur.plot(kind="bar", ax=ax, color=sns.color_palette("muted", len(total_dur)))
ax.set_title("Total Annotated Time (s) per Label")
ax.set_xlabel("Label")
ax.set_ylabel("Total Duration (s)")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "total_duration_by_label.png")
plt.close()
print("Saved -> artifacts/eda_output/total_duration_by_label.png")


# ── Summary table ─────────────────────────────────────────────────────────────

summary = (
    df.groupby("Label")["DurationSec"]
    .agg(count="count", mean="mean", median="median", std="std", min="min", max="max")
    .round(2)
    .sort_values("count", ascending=False)
)
summary.to_csv(OUTPUT_DIR / "label_duration_summary.csv")
print("\nSaved -> artifacts/eda_output/label_duration_summary.csv")
print("\nLabel Duration Summary:\n")
print(summary.to_string())

print("\n" + "=" * 60)
print("EDA complete. All outputs written to artifacts/eda_output/")
print("=" * 60)


# ═══════════════════════════════════════════════════════════════════════════════
# PER-VIDEOTYPE EDA  (Action vs HeadPose — labels differ between the two)
# ═══════════════════════════════════════════════════════════════════════════════

TYPES = {
    "Action":   {"color": "steelblue",  "palette": "Blues_d"},
    "HeadPose": {"color": "darkorange", "palette": "Oranges_d"},
}

for vtype, style in TYPES.items():
    sub = df[df["VideoType"] == vtype].copy()
    out = OUTPUT_DIR / vtype.lower()
    out.mkdir(exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"VIDEO TYPE: {vtype}  ({len(sub):,} rows)")
    print(f"{'=' * 60}")

    labels = sorted(sub["Label"].unique())
    print(f"Labels ({len(labels)}): {labels}")
    print(f"\nSubjects: {sorted(sub['Subject'].unique())}")
    print(f"Cameras : {sorted(sub['Camera'].unique())}")

    print(f"\nDuration stats:\n{sub['DurationSec'].describe().round(2)}")

    # ── A. Label count bar ────────────────────────────────────────────────────
    lc = sub["Label"].value_counts()
    fig, ax = plt.subplots(figsize=(max(8, len(lc) * 0.7), 5))
    lc.plot(kind="bar", ax=ax, color=style["color"], edgecolor="white")
    ax.set_title(f"[{vtype}] Annotation Count per Label")
    ax.set_xlabel("Label")
    ax.set_ylabel("Count")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out / "label_counts.png")
    plt.close()

    # ── B. Duration boxplot per label ─────────────────────────────────────────
    order = sub.groupby("Label")["DurationSec"].median().sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.8), 5))
    sns.boxplot(data=sub, x="Label", y="DurationSec", order=order,
                ax=ax, palette=style["palette"])
    ax.set_title(f"[{vtype}] Duration (s) per Label")
    ax.set_xlabel("Label")
    ax.set_ylabel("Duration (s)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out / "duration_by_label.png")
    plt.close()

    # ── C. Total annotated time per label ─────────────────────────────────────
    tot = sub.groupby("Label")["DurationSec"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.8), 5))
    tot.plot(kind="bar", ax=ax, color=style["color"], edgecolor="white")
    ax.set_title(f"[{vtype}] Total Annotated Time (s) per Label")
    ax.set_xlabel("Label")
    ax.set_ylabel("Total Duration (s)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out / "total_duration_by_label.png")
    plt.close()

    # ── D. Heatmap — subject × label annotation count ─────────────────────────
    pivot = sub.pivot_table(index="Subject", columns="Label", values="DurationSec",
                            aggfunc="count", fill_value=0)
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.7), max(4, len(pivot) * 0.45)))
    cmap = "Blues" if vtype == "Action" else "Oranges"
    sns.heatmap(pivot, annot=True, fmt="d", cmap=cmap, linewidths=0.4, ax=ax)
    ax.set_title(f"[{vtype}] Annotation Count — Subject × Label")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out / "heatmap_subject_label.png")
    plt.close()

    # ── E. Per-camera label distribution ──────────────────────────────────────
    cam_label = sub.groupby(["Camera", "Label"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.7), 5))
    cam_label.T.plot(kind="bar", ax=ax, colormap="tab10")
    ax.set_title(f"[{vtype}] Annotation Count — Label × Camera")
    ax.set_xlabel("Label")
    ax.set_ylabel("Count")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(title="Camera", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out / "label_by_camera.png")
    plt.close()

    # ── F. Duration histogram ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(sub["DurationSec"].dropna(), bins=25, color=style["color"], edgecolor="white")
    ax.set_title(f"[{vtype}] Distribution of Annotation Durations")
    ax.set_xlabel("Duration (s)")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(out / "duration_histogram.png")
    plt.close()

    # ── G. Per-subject label stacked bar ──────────────────────────────────────
    subj_label = sub.groupby(["Subject", "Label"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(max(10, len(subj_label) * 0.8), 5))
    subj_label.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
    ax.set_title(f"[{vtype}] Annotation Count per Subject (stacked by Label)")
    ax.set_xlabel("Subject")
    ax.set_ylabel("Count")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(title="Label", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out / "subject_stacked_label.png")
    plt.close()

    # ── Summary CSV ───────────────────────────────────────────────────────────
    smry = (
        sub.groupby("Label")["DurationSec"]
        .agg(count="count", mean="mean", median="median", std="std", min="min", max="max")
        .round(2)
        .sort_values("count", ascending=False)
    )
    smry.to_csv(out / "label_duration_summary.csv")

    print(f"\nLabel duration summary:\n{smry.to_string()}")
    print(f"\nAll {vtype} plots saved -> artifacts/eda_output/{vtype.lower()}/")

print("\n" + "=" * 60)
print("Per-VideoType EDA complete.")
print("=" * 60)
