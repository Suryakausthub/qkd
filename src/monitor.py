#!/usr/bin/env python3
import sys, os, time, json, torch, pandas as pd
from pathlib import Path
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from inject_anomalies import AE, COLS, SEQ

# ─── Configuration ──────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[1]
CSV       = ROOT / "data" / "annotated.csv"
KEYS      = ROOT / "keys"
WINDOW    = 10             # look at last 10 samples (~10 s)
THRESHOLD = 2e4            # lower threshold so you *will* see something
# ─────────────────────────────────────────────────────────────────────────────

def get_latest_key():
    files = sorted(KEYS.glob("*.bin"))
    if not files:
        raise FileNotFoundError("No QKD keys found in keys/")
    return files[-1].read_bytes()

# load only the weights
state = torch.load(
    ROOT / "model" / "lstm_ae.pt",
    map_location="cpu",
    weights_only=True
)
net = AE(len(COLS))
net.load_state_dict(state)
net.eval()

buff           = []
lines_consumed = 0

# banner on stderr
print("📡  monitoring …  Ctrl-C to stop", file=sys.stderr, flush=True)

while True:
    # read only newly appended lines
    with open(CSV, "r") as f:
        lines = f.readlines()[lines_consumed:]
    lines_consumed += len(lines)

    for ln in lines:
        ln = ln.strip()
        if not ln or ln.startswith("timestamp"):
            continue

        parts = [p.strip() for p in ln.split(",")]
        if len(parts) != 7:
            continue
        ts_raw, vc, p_s, q_s, v_s, sec_s, label = parts

        # convert P/Q/time, treat blank or “nan” as 0.0
        try:
            p   = float(p_s) if p_s and p_s.lower() != "nan" else 0.0
            q   = float(q_s) if q_s and q_s.lower() != "nan" else 0.0
            sec = float(sec_s)
        except ValueError:
            continue

        # sliding window
        buff.append([p, q, sec])
        if len(buff) > WINDOW:
            buff.pop(0)

        if len(buff) == WINDOW:
            x   = torch.tensor(buff, dtype=torch.float32).unsqueeze(0)
            err = ((net(x) - x) ** 2).mean().item()

            # always print this so you can watch err evolve
            print(f"[debug] t={sec:.1f}s  err={err:.2e}",
                  file=sys.stderr, flush=True)

            if err > THRESHOLD:
                # you *will* see this now
                print(f"[!!! ALERT_TRIPPED !!!] err={err:.2e} > THR={THRESHOLD:.2e}",
                      file=sys.stderr, flush=True)

                key   = get_latest_key()
                aes   = AESGCM(key)
                nonce = os.urandom(12)
                payload = json.dumps({
                    "timestamp": ts_raw,
                    "error": err
                }).encode()
                ct     = aes.encrypt(nonce, payload, None)
                packet = nonce + ct

                # **this** line is what the listener sees
                print(f"enc_alert={packet.hex()}", flush=True)

    time.sleep(1)
