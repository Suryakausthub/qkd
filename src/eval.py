from pathlib import Path

# --- import order matters ---------------------------------------------------
import numpy as np            # <- import numpy FIRST
import torch, pandas as pd
from sklearn.metrics import classification_report
from inject_anomalies import AE, COLS, SEQ
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

state_dict = torch.load(ROOT / "model" / "lstm_ae.pt", map_location="cpu")
net = AE(len(COLS)); net.load_state_dict(state_dict); net.eval()

# ---------- prep dataset ----------------------------------------------------
df = pd.read_csv(ROOT / "data" / "annotated.csv").ffill()
df["V_real"] = df["voltage_C"].str.extract(r"([-+]?\d*\.?\d+)").astype(float)
df["V_imag"] = df["voltage_C"].str.extract(r"\+(\d*\.?\d+)j").astype(float)

x = torch.tensor(df[COLS].values, dtype=torch.float32)
seqs = torch.stack([x[i:i+SEQ] for i in range(len(x)-SEQ)])

# ---------- reconstruction error --------------------------------------------
with torch.no_grad():
    recon = net(seqs)
errors = ((recon - seqs)**2).mean(dim=(1,2)).cpu().numpy()

thr   = errors.mean() + 3*errors.std()
pred  = errors > thr
true  = df["label"].iloc[SEQ:].values

print(f"threshold = {thr:.4e}")
print(classification_report(true, pred, digits=4))
