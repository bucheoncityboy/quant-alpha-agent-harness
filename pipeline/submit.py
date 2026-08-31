#!/usr/bin/env python3
"""submit.py — 알파 제출 + submit_log.json 누적

Usage:
    python submit.py <alpha_id>
"""
import argparse, json, os, sys
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from engine.ace_lib import submit_alpha as wq_submit
from _login_helper import get_session

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / "submit_log.json"

def load_log():
    if LOG_PATH.exists():
        with open(LOG_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {"entries": [], "next_seq": 1}

def save_log(log):
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False, default=str)

def main():
    p = argparse.ArgumentParser(description="WQ Alpha Submitter")
    p.add_argument("alpha_id", type=str, help="Alpha ID to submit")
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

    log = load_log()
    seq = log["next_seq"]
    entry = {"seq": seq, "timestamp": datetime.now().isoformat(), "alpha_id": args.alpha_id}

    session = get_session()
    try:
        resp = wq_submit(session, args.alpha_id)
        status = resp.get("status", "") if isinstance(resp, dict) else str(resp)
        entry["response"] = resp if isinstance(resp, dict) else {"raw": str(resp)}
        if status.upper() in ("ACTIVE", "SUCCESS", "OK"):
            entry["status"] = "success"
        else:
            entry["status"] = "failed"
            entry["reason"] = resp.get("message", resp.get("error", str(resp))) if isinstance(resp, dict) else str(resp)
    except Exception as e:
        entry["status"] = "failed"
        entry["reason"] = str(e)

    log["entries"].append(entry)
    log["next_seq"] = seq + 1
    save_log(log)

    print(json.dumps(entry, indent=2, ensure_ascii=False, default=str))

if __name__ == "__main__":
    main()
