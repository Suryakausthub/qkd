
from pathlib import Path

# --- import order matters ---------------------------------------------------
import numpy as np            # <- import numpy FIRST
import torch
import pandas as pd
from sklearn.metrics import classification_report
from inject_anomalies import AE, COLS, SEQ
# ---------------------------------------------------------------------------

# Project root
ROOT = Path(__file__).resolve().parents[1]

# --- Load the trained model safely -----------------------------------------
state_dict = torch.load(
    ROOT / "model" / "lstm_ae.pt",
    map_location="cpu",
    weights_only=True     # safer loading in PyTorch 2.x
)
net = AE(len(COLS))
net.load_state_dict(state_dict)
net.eval()

# ---------- Prepare dataset ------------------------------------------------
df = pd.read_csv(ROOT / "data" / "annotated.csv").ffill()

# extract real/imag parts of complex voltage
df["V_real"] = df["voltage_C"].str.extract(r"([-+]?\d*\.?\d+)").astype(float)
df["V_imag"] = df["voltage_C"].str.extract(r"\+(\d*\.?\d+)j").astype(float)

# map string labels to integers (0/1)
df["label"] = df["label"].astype(str).map({"False": 0, "True": 1})

# build sequential tensor data
data = torch.tensor(df[COLS].values, dtype=torch.float32)
seqs = torch.stack([data[i:i+SEQ] for i in range(len(data) - SEQ)])

# ---------- Compute reconstruction errors ----------------------------------
with torch.no_grad():
    recon = net(seqs)
errors = ((recon - seqs) ** 2).mean(dim=(1, 2)).cpu().numpy()

# dynamic threshold = mean + 3*std
thr = errors.mean() + 3 * errors.std()

# predictions and ground truth as ints
pred_int = (errors > thr).astype(int)
true_int = df["label"].iloc[SEQ:].values.astype(int)

# diagnostic prints (uncomment if needed)
# print("unique true:", np.unique(true_int), "dtype:", true_int.dtype)
# print("unique pred:", np.unique(pred_int), "dtype:", pred_int.dtype)

# ---------- Report results -------------------------------------------------
print(f"threshold = {thr:.4e}")
print(classification_report(
    true_int,
    pred_int,
    digits=4,
    zero_division=0       # avoid undefined metric warnings
))
