
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "reports" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METHODS = ["CLIP", "ResNet-50", "Color Histogram"]


def make_precision_chart(title, filename, values, ci_lower, ci_upper):
    values = np.array(values)
    ci_lower = np.array(ci_lower)
    ci_upper = np.array(ci_upper)

    # Matplotlib expects distances from each estimate to its CI bounds.
    errors = np.vstack([
        values - ci_lower,
        ci_upper - values,
    ])

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(METHODS, values, yerr=errors, capsize=6)

    ax.set_title(title)
    ax.set_ylabel("Precision@5 (%)")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    for bar, value in zip(bars, values):
        ax.annotate(
            f"{value:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=10,
        )

    fig.tight_layout()
    output_path = OUTPUT_DIR / filename
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")


make_precision_chart(
    title="Visual Similarity: Precision@5",
    filename="visual_precision_at_5.png",
    values=[87.333, 71.333, 46.000],
    ci_lower=[79.253, 60.922, 33.232],
    ci_upper=[95.414, 81.744, 58.768],
)

make_precision_chart(
    title="Room Consistency: Precision@5",
    filename="room_consistent_precision_at_5.png",
    values=[66.000, 42.667, 15.333],
    ci_lower=[54.700, 31.121, 8.381],
    ci_upper=[77.300, 54.212, 22.286],
)
