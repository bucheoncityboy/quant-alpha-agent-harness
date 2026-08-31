#!/usr/bin/env python3
"""simulate.py — 알파 시뮬레이션 + simulation_log.json 누적

Usage:
    python simulate.py "ts_zscore(close, 20)"
    python simulate.py --file list.txt --parallel 5
"""
import argparse, json, os, sys, time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from engine.ace_lib import simulate_single_alpha, get_simulation_result_json
from _login_helper import get_session

DEFAULT_SETTINGS = {
    "universe": "TOP3000", "decay": 5, "neutralization": "SUBINDUSTRY",
    "truncation": 0.08, "pasteurization": "ON", "nanHandling": "ON",
    "dateFrom": "2019-01-01", "dateTo": "2023-12-31",
}
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / "simulation_log.json"

def load_log():
    if LOG_PATH.exists():
        with open(LOG_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {"entries": [], "next_seq": 1}

def save_log(log):
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False, default=str)

def run_one(session, expression, settings, seq):
    sim_data = {"type": "REGULAR", "regular": expression, "settings": {**settings, "visualization": False}}
    entry = {"seq": seq, "timestamp": datetime.now().isoformat(), "expression": expression}
    try:
        result = simulate_single_alpha(session, sim_data)
        aid = result.get("alpha_id")
        if not aid:
            entry["error"] = "No alpha_id"
            return entry
        entry["alpha_id"] = aid
        # WQ raw stats
        raw = get_simulation_result_json(session, aid)
        if raw and "is" in raw:
            is_data = raw["is"]
            entry["fitness"] = is_data.get("fitness")
            entry["sharpe"] = is_data.get("sharpe")
            entry["turnover"] = is_data.get("turnover")
            entry["returns"] = is_data.get("returns")
            entry["std"] = is_data.get("std")
        entry["success"] = True
    except Exception as e:
        entry["error"] = str(e)
    return entry

def main():
    p = argparse.ArgumentParser(description="WQ Alpha Simulator")
    p.add_argument("expression", type=str, nargs="?", help="WQ alpha expression")
    p.add_argument("--file", type=str, help="File with expressions (one per line)")
    p.add_argument("--parallel", type=int, default=3, help="Max parallel (default: 3)")
    p.add_argument("--settings", type=str, help="JSON settings override")
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

    if not args.expression and not args.file:
        p.print_help()
        return

    settings = dict(DEFAULT_SETTINGS)
    if args.settings:
        settings.update(json.loads(args.settings))

    if args.file:
        with open(args.file, encoding="utf-8") as f:
            expressions = [line.strip() for line in f if line.strip()]
    else:
        expressions = [args.expression]

    log = load_log()
    session = get_session()
    entries = []
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {pool.submit(run_one, session, expr, settings, log["next_seq"] + i): (expr, i)
                   for i, expr in enumerate(expressions)}
        for future in as_completed(futures):
            entries.append(future.result())

    entries.sort(key=lambda e: e.get("seq", 0))
    log["entries"].extend(entries)
    log["next_seq"] += len(expressions)
    save_log(log)

    print(json.dumps({"count": len(entries), "results": entries}, indent=2, ensure_ascii=False, default=str))

if __name__ == "__main__":
    main()
