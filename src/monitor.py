#!/usr/bin/env python3
import sys, os, time, json, torch, pandas as pd
from pathlib import Path
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from inject_anomalies import AE, COLS, SEQ


# ─── Configuration ──────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[1]
CSV       = ROOT / "data" / "annotated.csv"
KEYS      = ROOT / "keys"
WINDOW    = 10
THRESHOLD = 1e+00
# ─────────────────────────────────────────────────────────────────────────────

def get_latest_key_path():
    files = sorted(KEYS.glob("*.bin"))
    if not files:
        raise FileNotFoundError("No QKD keys found")
    return files[-1]

print("[DEBUG] loading model…", file=sys.stderr)
state = torch.load(ROOT / "model" / "lstm_ae.pt", map_location="cpu")
net = AE(len(COLS)); net.load_state_dict(state); net.eval()
print("[DEBUG] model ready", file=sys.stderr)

buff = []
lines_consumed = 0
print("📡  monitoring …  Ctrl‑C to stop", file=sys.stderr, flush=True)

while True:
    with open(CSV, "r") as f:
        lines = f.readlines()[lines_consumed:]
    lines_consumed += len(lines)
    if lines:
        print(f"[DEBUG] {len(lines)} new lines", file=sys.stderr)

    for ln in lines:
        ln = ln.strip()
        if not ln or ln.startswith("timestamp"): continue

        parts = [p.strip() for p in ln.split(",")]
        if len(parts) != 7:
            print(f"[DEBUG] parse error on line: {ln}", file=sys.stderr)
            continue

        ts_raw, vc, p_s, q_s, v_s, sec_s, label = parts
        try:
            p = float(p_s) if p_s and p_s.lower() != "nan" else 0.0
            q = float(q_s) if q_s and q_s.lower() != "nan" else 0.0
            sec = float(sec_s)
        except ValueError:
            continue

        buff.append([p, q, sec])
        if len(buff) > WINDOW:
            buff.pop(0)

        if len(buff) == WINDOW:
            x = torch.tensor(buff, dtype=torch.float32).unsqueeze(0)
            err = ((net(x) - x) ** 2).mean().item()

            print(f"[debug] t={sec:.1f}s  err={err:.2e}", file=sys.stderr, flush=True)

            if err > THRESHOLD:
                print(f"[!!! ALERT_TRIPPED !!!] err={err:.2e} > {THRESHOLD:.2e}", file=sys.stderr)

                key_path = get_latest_key_path()
                key = key_path.read_bytes()
                aes = AESGCM(key)
                nonce = os.urandom(12)
                payload = json.dumps({
                    "timestamp": ts_raw,
                    "error": err,
                    "key_file": key_path.name
                }).encode()

                try:
                    ct = aes.encrypt(nonce, payload, None)
                    packet = nonce + ct
                    sys.stdout.write(f"enc_alert={packet.hex()}\n")
                    sys.stdout.flush()
                    print(f"[DEBUG] encryption succeeded with {key_path.name}", file=sys.stderr)
                except Exception as e:
                    print(f"[ERROR] encryption failed: {e}", file=sys.stderr)

    time.sleep(1)