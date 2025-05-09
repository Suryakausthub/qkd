from pathlib import Path
import torch, torch.nn as nn, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
df   = pd.read_csv(ROOT / "data" / "annotated.csv").ffill()

SEQ = 60
if len(df) <= SEQ:
    print(f"⚠ Data too short ({len(df)} rows) — shrinking SEQ → {len(df)-1}")
    SEQ = max(1, len(df)-1)

df["V_real"] = df["voltage_C"].str.extract(r"([-+]?\d*\.?\d+)").astype(float)
df["V_imag"] = df["voltage_C"].str.extract(r"\+(\d*\.?\d+)j").astype(float)

COLS = ["V_real", "V_imag", "time"]
x     = torch.tensor(df[COLS].values, dtype=torch.float32)
seqs  = torch.stack([x[i:i+SEQ] for i in range(len(x)-SEQ)])
train = seqs[: int(0.8 * len(seqs))]

device = "cuda" if torch.cuda.is_available() else "cpu"

class AE(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.enc = nn.LSTM(d, 64, batch_first=True)
        self.dec = nn.LSTM(64, d, batch_first=True)
    def forward(self, x):
        _, (h, _) = self.enc(x)
        h = h.repeat(x.size(1), 1, 1).transpose(0, 1)
        out, _ = self.dec(h)
        return out

net  = AE(len(COLS)).to(device)
opt  = torch.optim.Adam(net.parameters(), 1e-3)
crit = nn.MSELoss()

def train_loop():
    for ep in range(10):
        net.train(); tot = 0
        for i in range(0, len(train), 128):
            b = train[i:i+128].to(device)
            opt.zero_grad()
            l = crit(net(b), b); l.backward(); opt.step()
            tot += l.item() * len(b)
        print(f"epoch {ep}: {tot/len(train):.4e}")

if __name__ == "__main__":
    train_loop()
    (ROOT / "model").mkdir(exist_ok=True)
    torch.save(net.state_dict(), ROOT / "model" / "lstm_ae.pt")
    print("✓ model saved → model/lstm_ae.pt")
