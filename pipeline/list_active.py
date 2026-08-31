#!/usr/bin/env python3
"""list_active.py — ACTIVE 상태 알파 조회 (WQ API)

Usage:
    python list_active.py
    python list_active.py --json   # raw JSON 출력
"""
import argparse, json, os, sys, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from engine.ace_lib import brain_api_url
from _login_helper import get_session

logging.basicConfig(level=logging.WARNING)

def main():
    p = argparse.ArgumentParser(description="List ACTIVE WQ alphas")
    p.add_argument("--json", action="store_true", help="Output raw JSON")
    args = p.parse_args()
    # .env 처리
    env_path = Path(".env")
    if not env_path.exists():
        env_path.write_text("BRAIN_ID=\nBRAIN_PASSWORD=\n", encoding="utf-8")
        print(json.dumps({"message": ".env created — fill in BRAIN_ID and BRAIN_PASSWORD"}, indent=2))
        return
    for line in env_path.read_text(encoding="utf-8").strip().splitlines():
        if "=" in line:
            k, v = line.strip().split("=", 1)
            if k and v:
                os.environ[k] = v

    LOG_DIR = Path(__file__).parent / "logs"
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    session = get_session()
    limit = 50
    alphas = []

    # Paginated fetch via GET /users/self/alphas
    for offset in range(0, 5000, limit):
        url = f"{brain_api_url}/users/self/alphas?limit={limit}&offset={offset}"
        resp = session.get(url)
        if resp.status_code == 429:
            break
        data = resp.json()
        results = data.get("results", [])
        if not results:
            break
        alphas.extend(results)
        if len(results) < limit:
            break

    # Filter ACTIVE
    active = [a for a in alphas if a.get("status", "").upper() == "ACTIVE"]

    if args.json:
        print(json.dumps(active, indent=2, ensure_ascii=False, default=str))
        return

    if not active:
        print(json.dumps({"count": 0, "alphas": []}, indent=2, ensure_ascii=False))
        return

    summary = []
    for a in active:
        summary.append({
            "alpha_id": a.get("id"),
            "name": a.get("name", ""),
            "fitness": a.get("fitness"),
            "sharpe": a.get("sharpe"),
            "expression": str(a.get("regular", {}).get("code", a.get("regular", "")))[:80],
            "created": a.get("created"),
            "submitted": a.get("submitted"),
        })

    # logs/active_alphas.json 저장
    output_path = LOG_DIR / "active_alphas.json"
    output_path.write_text(
        json.dumps({"count": len(summary), "alphas": summary}, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8"
    )

    print(json.dumps({"count": len(summary), "alphas": summary}, indent=2, ensure_ascii=False, default=str))

if __name__ == "__main__":
    main()
