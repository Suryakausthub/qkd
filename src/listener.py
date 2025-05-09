#!/usr/bin/env python3
import sys
import json
from pathlib import Path
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

KEYS = Path(__file__).resolve().parents[1] / "keys"

def get_latest_key():
    files = sorted(KEYS.glob("*.bin"))
    if not files:
        raise FileNotFoundError("No QKD keys found")
    return files[-1].read_bytes()

aes = AESGCM(get_latest_key())

for line in sys.stdin:
    line = line.strip()
    if not line.startswith("enc_alert="):
        continue
    packet = bytes.fromhex(line.split("=", 1)[1])
    nonce, ct = packet[:12], packet[12:]
    try:
        pt = aes.decrypt(nonce, ct, None)
        alert = json.loads(pt)
        print(f"🚨 Decrypted alert: {alert}", flush=True)
    except Exception as e:
        print(f"❌ Decrypt failed: {e}", flush=True)
