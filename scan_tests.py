"""Scan all test folders and report which have freq_set_hz column."""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(
    "/Users/oskarjohnbruunsmith/Library/CloudStorage/"
    "OneDrive-DanmarksTekniskeUniversitet/"
    "AttoPump - steps towards commercialisation/"
    "Tests + documentation/All_tests"
)

results = []

for d in sorted(ROOT.iterdir()):
    if not d.is_dir() or d.name.startswith("."):
        continue

    csvs = list(d.glob("*.csv"))
    csv_names = [c.name for c in csvs]

    has_freq_col = False
    freq_values = []
    source_file = None

    for csv_file in csvs:
        try:
            with open(csv_file, "r", encoding="utf-8") as f:
                first_line = f.readline()
            delim = ";" if first_line.count(";") > first_line.count(",") else ","
            df = pd.read_csv(csv_file, delimiter=delim, nrows=1000)
            df.columns = df.columns.str.strip().str.lower()

            if "freq_set_hz" in df.columns:
                has_freq_col = True
                freq_values = sorted(df["freq_set_hz"].dropna().unique().tolist())
                source_file = csv_file.name
                break
        except Exception:
            pass

    results.append(
        {
            "folder": d.name,
            "csvs": csv_names,
            "has_freq": has_freq_col,
            "freq_vals": freq_values,
            "source": source_file,
        }
    )

# Print
with_freq = [r for r in results if r["has_freq"]]
without_freq = [r for r in results if not r["has_freq"]]

print(f"TOTAL: {len(results)} test folders\n")

print(f"=== WITH freq_set_hz ({len(with_freq)}) ===")
for r in with_freq:
    n = len(r["freq_vals"])
    vals = r["freq_vals"][:8]
    tag = "CONSTANT" if n == 1 else f"SWEEP ({n} freqs)"
    extra = f" ... +{n-8} more" if n > 8 else ""
    print(f"  {r['folder']}")
    print(f"    [{tag}] freqs={vals}{extra}  (from {r['source']})")

print(f"\n=== WITHOUT freq_set_hz ({len(without_freq)}) ===")
for r in without_freq:
    csv_str = ", ".join(r["csvs"]) if r["csvs"] else "(no CSVs)"
    print(f"  {r['folder']}")
    print(f"    CSVs: {csv_str}")
