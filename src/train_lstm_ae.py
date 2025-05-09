from pathlib import Path
import pandas as pd, numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW  = ROOT / "data" / "raw_normal.csv"
OUT  = ROOT / "data" / "annotated.csv"

# ---------- read GridLAB‑D output ------------------------------------------
raw = pd.read_csv(RAW, comment="#", names=["timestamp", "voltage_C"])

raw["timestamp"] = (
    pd.to_datetime(raw["timestamp"].str.replace(" EST", ""), errors="coerce")
      .dt.tz_localize("US/Eastern")
)
raw["time"] = (raw["timestamp"] - raw["timestamp"].min()).dt.total_seconds()

raw["voltage_C"] = raw["voltage_C"].apply(
    lambda x: complex(x) if isinstance(x, str) and "+" in x else np.nan
)
raw["P_node1"] = raw["voltage_C"].apply(lambda z: z.real * 0.1 if pd.notnull(z) else np.nan)
raw["Q_node1"] = raw["voltage_C"].apply(lambda z: z.imag * 0.1 if pd.notnull(z) else np.nan)
raw["V_node1"] = raw["voltage_C"].apply(lambda z: abs(z)       if pd.notnull(z) else np.nan)

# ---------- inject artificial anomalies ------------------------------------
np.random.seed(42)
rows = np.random.choice(raw.index, size=int(0.02 * len(raw)), replace=False)
raw.loc[rows, "P_node1"] *= np.random.uniform(1.2, 1.5, len(rows))
# fast-feedback: 3-second packet-loss windows at 30 s, 60 s, 90 s
for t in (30, 60, 90):
    mask = (raw["time"] >= t) & (raw["time"] < t + 3)
    raw.loc[mask, ["P_node1", "Q_node1", "V_node1"]] = np.nan


raw["label"] = raw[["P_node1", "Q_node1", "V_node1"]].isna().any(axis=1) | raw.index.isin(rows)

cols = ["timestamp", "voltage_C", "P_node1", "Q_node1", "V_node1", "time", "label"]
raw[cols].to_csv(OUT, index=False)
print("✓ anomalies injected → data/annotated.csv")
