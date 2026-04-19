#!/usr/bin/env python3
"""
download_model.py
Run this ONCE with internet to download all offline dependencies.
After this, the badminton trainer works 100% offline.

Usage:
    python3 download_model.py
"""

import os
import sys
import json
import urllib.request
import urllib.error

# ── TF.js libraries ───────────────────────────────────────────
FILES_TFJS = [
    (
        "tf.min.js",
        [
            "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.15.0/dist/tf.min.js",
            "https://unpkg.com/@tensorflow/tfjs@4.15.0/dist/tf.min.js",
        ]
    ),
    (
        "pose-detection.min.js",
        [
            "https://cdn.jsdelivr.net/npm/@tensorflow-models/pose-detection@2.1.3/dist/pose-detection.min.js",
            "https://unpkg.com/@tensorflow-models/pose-detection@2.1.3/dist/pose-detection.min.js",
        ]
    ),
]

# ── Model base URLs (tried in order) ─────────────────────────
MODEL_BASES = [
    "https://storage.googleapis.com/tfjs-models/savedmodel/movenet/singlepose/lightning/4/",
    "https://tfhub.dev/google/tfjs-model/movenet/singlepose/lightning/4/",
    "https://cdn.jsdelivr.net/gh/tensorflow/tfjs-models/pose-detection/src/movenet/",
]

# ── Helpers ───────────────────────────────────────────────────
def dl(url, dest):
    print(f"    {url}")
    print(f"    → {dest} ...", end=" ", flush=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r, open(dest, "wb") as f:
            f.write(r.read())
        size = os.path.getsize(dest)
        if size < 100:
            os.remove(dest)
            print(f"✗ (too small: {size} bytes, likely an error page)")
            return False
        print(f"✓ ({size // 1024} KB)")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

def dl_first(name, urls):
    """Try each URL in turn, return True if any succeeds."""
    for url in urls:
        if dl(url, name):
            return True
    return False

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    print("\n=== Badminton Trainer — Offline Setup ===\n")

    # ── Step 1: TF.js libraries ───────────────────────────────
    print("Step 1: Downloading TensorFlow.js libraries...")
    for name, urls in FILES_TFJS:
        print(f"\n  {name}")
        if not dl_first(name, urls):
            print(f"  ✗ Could not download {name}. Check internet connection.")

    # ── Step 2: MoveNet model ─────────────────────────────────
    print("\nStep 2: Downloading MoveNet model...")
    os.makedirs("model", exist_ok=True)

    model_json_path = os.path.join("model", "model.json")
    downloaded_base = None

    for base in MODEL_BASES:
        url = base + "model.json"
        print(f"\n  Trying base: {base}")
        if dl(url, model_json_path):
            downloaded_base = base
            break

    if not downloaded_base:
        print("\n✗ Could not download model.json from any mirror.")
        print("\nTry manually:")
        print("  1. Open this URL in your browser:")
        print("     https://storage.googleapis.com/tfjs-models/savedmodel/movenet/singlepose/lightning/4/model.json")
        print("  2. Save the file as  model/model.json")
        print("  3. Run this script again — it will find the shards from model.json\n")
        sys.exit(1)

    # ── Step 3: Weight shards ─────────────────────────────────
    print("\nStep 3: Downloading weight shards...")
    try:
        with open(model_json_path) as f:
            model_data = json.load(f)
    except Exception as e:
        print(f"✗ Could not parse model.json: {e}")
        sys.exit(1)

    shards = []
    for manifest in model_data.get("weightsManifest", []):
        for path in manifest.get("paths", []):
            shards.append(path)

    if not shards:
        print("✗ No weight shards found in model.json — file may be corrupt.")
        sys.exit(1)

    print(f"  Found {len(shards)} shard(s).")
    for shard in shards:
        dest = os.path.join("model", shard)
        os.makedirs(os.path.dirname(dest), exist_ok=True)

        urls = [downloaded_base + shard] + [b + shard for b in MODEL_BASES if b != downloaded_base]
        print(f"\n  {shard}")
        if not dl_first(dest, urls):
            print(f"  ✗ Could not download shard: {shard}")

    print("\n=== All done! ===")
    print("\nTo run the trainer:")
    print("  python3 -m http.server 8080")
    print("  Then open: http://localhost:8080\n")

if __name__ == "__main__":
    main()