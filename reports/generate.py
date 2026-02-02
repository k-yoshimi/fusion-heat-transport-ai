"""Report generation: CSV and markdown summary."""

import csv
import os


def write_csv(results: list[dict], path: str) -> None:
    """Write benchmark results to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not results:
        return
    keys = list(results[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)


def write_markdown(results: list[dict], best_name: str, path: str) -> None:
    """Write markdown summary table."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("# Benchmark Results\n\n")
        if not results:
            f.write("No results.\n")
            return
        keys = list(results[0].keys())
        # Header
        f.write("| " + " | ".join(keys) + " | best |\n")
        f.write("| " + " | ".join("---" for _ in keys) + " | --- |\n")
        for r in results:
            vals = []
            for k in keys:
                v = r[k]
                if isinstance(v, float):
                    vals.append(f"{v:.6g}")
                else:
                    vals.append(str(v))
            is_best = "✓" if r.get("name") == best_name else ""
            f.write("| " + " | ".join(vals) + f" | {is_best} |\n")
        f.write(f"\nBest solver: **{best_name}**\n")
