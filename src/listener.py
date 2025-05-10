#!/usr/bin/env python3
import sys
import json
from pathlib import Path
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

KEYS = Path(__file__).resolve().parents[1] / "keys"
LOG_PATH = Path(__file__).resolve().parents[1] / "listener_output.log"

# line-buffered file logging
log_file = open(LOG_PATH, "w", buffering=1, encoding="utf-8")
def log(msg):
    print(msg)
    print(msg, file=log_file)

def load_key(key_name: str) -> bytes:
    key_file = KEYS / key_name
    if not key_file.exists():
        raise FileNotFoundError(f"Key file not found: {key_name}")
    return key_file.read_bytes()

log("[listener] ready and listening for input...")

for line in sys.stdin:
    line = line.strip()
    log(f"[listener] raw line: {line}")

    if not line.startswith("enc_alert="):
        log(f"[listener] skipped (not alert): {line}")
        continue

    try:
        packet = bytes.fromhex(line.split("=", 1)[1])
        nonce, ct = packet[:12], packet[12:]

        log(f"Raw packet: {packet.hex()}")
        log(f"Nonce: {nonce.hex()}")
        log(f"Ciphertext: {ct.hex()}")

        # Try all keys to find correct one
        for key_path in sorted(KEYS.glob("*.bin"), reverse=True):
            try:
                key = key_path.read_bytes()
                aes = AESGCM(key)
                pt = aes.decrypt(nonce, ct, None)
                alert = json.loads(pt)
                log(f"[listener] decrypted with {key_path.name}")
                break
            except Exception:
                continue
        else:
            raise ValueError("❌ No valid key found to decrypt the alert.")

        log("🚨 Decrypted alert:")
        log(f"   Timestamp : {alert['timestamp']}")
        log(f"   Error     : {alert['error']:.2f}")
        log(f"   Key File  : {alert.get('key_file', 'unknown')}")

    except Exception as e:
        log(f"❌ Decrypt failed: {e}")