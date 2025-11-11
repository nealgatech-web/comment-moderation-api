"""Minimal example for a game server to moderate chat lines.

This script reads lines from stdin and prints them if non-toxic.
Set API_BASE env var to point to your running API.
"""
import os, sys, json, httpx

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")

def analyze(text: str) -> float:
    with httpx.Client(timeout=5) as h:
        r = h.post(f"{API_BASE}/analyze", json={"texts":[text]})
        r.raise_for_status()
        return r.json()["results"][0]["toxicity"]

def main():
    print("Enter chat lines. Ctrl-C to quit.")
    for line in sys.stdin:
        line = line.rstrip("\n")
        if not line:
            continue
        tox = analyze(line)
        if tox >= 0.8:
            print(f"[blocked toxic={tox:.2f}]")
        else:
            print(line)

if __name__ == "__main__":
    main()
