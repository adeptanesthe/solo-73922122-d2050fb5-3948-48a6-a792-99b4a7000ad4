#!/usr/bin/env python3
"""Run the demo beats against a live service.

Usage:
    python demo/run_demo.py [base_url]

Defaults to http://127.0.0.1:8000 (the docker compose port).
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://127.0.0.1:8000"
HERE = Path(__file__).resolve().parent


def call(method: str, path: str, payload: dict | None = None) -> tuple[int, dict]:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        BASE + path, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def load(name: str) -> dict:
    return json.loads((HERE / name).read_text())


def show(label: str, status: int, body: dict) -> None:
    print(f"\n=== {label} -> HTTP {status}, kind={body.get('kind')}, version={body.get('version')}")
    print(body.get("verdict", ""))
    for possibility in body.get("possibilities", []):
        print(f"  hypothesis: {possibility['hypothesis']}")
        for take in possibility["takes"]:
            gains = {k: round(v, 4) for k, v in take["gains"].items()}
            print(f"    take {take['segment']}: keys={take['pressed_keys']} gains={gains}")
    evidence = body.get("evidence") or {}
    if "key_groups" in evidence:
        print(f"  equivalence groups: {evidence.get('group_count')} "
              f"(sizes {[g['size'] for g in evidence['key_groups']]})")
        for model, diag in (evidence.get("models") or {}).items():
            if "model_size" in diag:
                print(f"    {model}: aggregated {diag['model_size']} "
                      f"vs per-key {diag['per_key_model_size']}")


def main() -> None:
    print("health:", call("GET", "/healthz"))

    for name, label in [
        ("01_same.json", "same key set"),
        ("02_different.json", "different key sets"),
        ("03_impossible.json", "conflicting evidence"),
        ("04_ambiguous.json", "both cases possible"),
    ]:
        status, body = call("POST", "/analyses", load(name))
        show(label, status, body)

    # Editor flow on the ambiguous analysis: correct take 0's key B.
    status, body = call("POST", "/analyses", load("04_ambiguous.json"))
    analysis_id = body["analysis_id"]
    version = body["version"]
    status, updated = call(
        "POST", f"/analyses/{analysis_id}/confirmations", load("04_confirmation.json")
    )
    show("after editor confirmation", status, updated)
    print("  confirmations:", updated.get("confirmations"))

    status, rejected = call(
        "POST", f"/analyses/{analysis_id}/confirmations", load("04_confirmation.json")
    )
    print("\n=== replayed stale confirmation ->", status, rejected.get("error", {}).get("code"))

    status, original = call("GET", f"/analyses/{analysis_id}?version={version}")
    print(f"=== original version {version} still queryable -> HTTP {status}, kind={original.get('kind')}")

    # Equivalence-group beat: six keys share one template and gain interval.
    # The solver aggregates them (no per-key binaries for unconfirmed members),
    # yet every witness still names concrete keys and per-key gains.
    status, grouped = call("POST", "/analyses", load("05_equivalence_group.json"))
    show("equivalence group (6 identical keys)", status, grouped)
    group_id = grouped["analysis_id"]
    group_version = grouped["version"]

    # Pin K2 to opposite states across the two takes: same-set becomes
    # infeasible and a different-set witness with real key names is returned.
    status, edited = call(
        "POST", f"/analyses/{group_id}/confirmations", load("05_confirmation.json")
    )
    show("group confirmation (K2 pressed only in take 0)", status, edited)
    print("  confirmations:", edited.get("confirmations"))

    # Replaying the same expected version is now stale; version 1 survives.
    status, stale = call(
        "POST", f"/analyses/{group_id}/confirmations", load("05_confirmation.json")
    )
    print("\n=== replayed group confirmation ->", status, stale.get("error", {}).get("code"))
    status, first = call("GET", f"/analyses/{group_id}?version={group_version}")
    print(
        f"=== group report version {group_version} still queryable -> HTTP {status}, "
        f"kind={first.get('kind')}"
    )


if __name__ == "__main__":
    main()
