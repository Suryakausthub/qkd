#!/usr/bin/env python3
import time
import secrets
from pathlib import Path

# ─── Configuration ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
KEY_DIR = ROOT / "keys"
KEY_DIR.mkdir(exist_ok=True)
# ─────────────────────────────────────────────────────────────────────────────

print("🔑  QKD‐simulator key producer starting… Ctrl-C to stop")
while True:
    key = secrets.token_bytes(32)           # 256-bit key
    fname = f"{int(time.time())}.bin"
    (KEY_DIR / fname).write_bytes(key)
    print(f"  • wrote key → {KEY_DIR / fname}")
    time.sleep(2)
