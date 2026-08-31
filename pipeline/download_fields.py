#!/usr/bin/env python3
"""download_fields.py — 데이터필드 + 카테고리 다운로드 (WQ API → CSV)

Usage:
    python download_fields.py
    python download_fields.py --region USA --delay 1
"""
import argparse, json, os, sys, time, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from engine.ace_lib import get_datasets, get_datafields
from _login_helper import get_session

logging.basicConfig(level=logging.WARNING)
DATA_DIR = Path(__file__).parent / "Data and operators"

def main():
    p = argparse.ArgumentParser(description="Download WQ datafields to CSV")
    p.add_argument("--region", default="USA")
    p.add_argument("--delay", type=int, default=1)
    p.add_argument("--universe", default="TOP3000")
    p.add_argument("--instrument-type", default="EQUITY")
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

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    session = get_session()

    # 1. Get all datasets
    print(json.dumps({"step": "fetching datasets", "params": vars(args)}))
    ds_df = get_datasets(session, instrument_type=args.instrument_type, region=args.region,
                         delay=args.delay, universe=args.universe)
    if ds_df.empty:
        print(json.dumps({"error": "No datasets found"}))
        return

    datasets = ds_df.to_dict("records")
    print(json.dumps({"datasets_found": len(datasets)}))

    # 2. For each dataset, download fields
    all_fields = []
    for i, ds in enumerate(datasets):
        ds_id = ds.get("id", "")
        ds_name = ds.get("name", f"dataset_{i}")
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in ds_name).lower()[:60]

        print(json.dumps({"step": "fetching fields", "dataset": ds_name, "id": ds_id, "progress": f"{i+1}/{len(datasets)}"}))

        try:
            fields_df = get_datafields(session, instrument_type=args.instrument_type, region=args.region,
                                       delay=args.delay, universe=args.universe, dataset_id=ds_id)
            if fields_df.empty:
                continue

            # Add dataset info
            fields_df["dataset_id"] = ds_id
            fields_df["dataset_name"] = ds_name
            all_fields.append(fields_df)

            # Save per-dataset CSV
            csv_path = DATA_DIR / f"{safe_name}.csv"
            fields_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        except Exception as e:
            print(json.dumps({"warning": f"Failed for dataset {ds_name}: {e}"}))

        time.sleep(1)  # rate limit courtesy

    # 3. Save combined fields
    if all_fields:
        import pandas as pd
        combined = pd.concat(all_fields, ignore_index=True)
        combined.to_csv(DATA_DIR / "_all_fields.csv", index=False, encoding="utf-8-sig")

    print(json.dumps({
        "datasets": len(datasets),
        "datasets_with_fields": len(all_fields),
        "total_fields": sum(len(df) for df in all_fields) if all_fields else 0,
        "output_dir": str(DATA_DIR),
    }))

if __name__ == "__main__":
    main()
