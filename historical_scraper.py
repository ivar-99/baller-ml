"""
Step 1: Scrape historical bonus round data from Tracksino.
Tracksino archives months of Monopoly Big Baller results.
Saves raw data to historical_data.json
"""

import requests
import json
import time
from datetime import datetime, timezone, timedelta

IST = timezone(timedelta(hours=5, minutes=30))
OUTPUT_FILE = "historical_data.json"

# Tracksino API endpoint for Monopoly Big Baller history
TRACKSINO_URL = "https://tracksino.com/api/history/monopoly-big-baller"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
    "Referer": "https://tracksino.com/monopoly-big-baller"
}

def fetch_page(page: int) -> list:
    """Fetch one page of results from Tracksino."""
    try:
        params = {"page": page, "per_page": 100, "sort_by": "time", "sort_desc": True}
        resp = requests.get(TRACKSINO_URL, headers=HEADERS, params=params, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            # Tracksino returns {data: [...], total: N}
            return data.get("data", []) or data.get("results", []) or []
        else:
            print(f"  ⚠️  Page {page} returned status {resp.status_code}")
            return []
    except Exception as e:
        print(f"  ⚠️  Page {page} error: {e}")
        return []

def parse_record(record: dict) -> dict | None:
    """
    Parse a Tracksino record into our standard format.
    Only keep 3 Rolls and 5 Rolls bonus events.
    """
    # Tracksino field names vary — try common ones
    result = (
        record.get("result") or
        record.get("outcome") or
        record.get("bonus") or
        record.get("spin_result") or ""
    ).lower()

    bonus_type = None
    if "5 roll" in result or "5roll" in result:
        bonus_type = "5 Rolls"
    elif "3 roll" in result or "3roll" in result:
        bonus_type = "3 Rolls"

    if not bonus_type:
        return None

    # Parse timestamp
    ts_raw = (
        record.get("time") or
        record.get("timestamp") or
        record.get("created_at") or
        record.get("date") or ""
    )
    utc_dt = None
    try:
        if isinstance(ts_raw, (int, float)):
            utc_dt = datetime.fromtimestamp(ts_raw, tz=timezone.utc)
        elif isinstance(ts_raw, str) and ts_raw:
            # Try ISO format
            utc_dt = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
    except Exception:
        pass

    ist_dt = utc_dt.astimezone(IST) if utc_dt else None

    # Extract multiplier
    multiplier = record.get("multiplier") or record.get("payout") or record.get("total_win")
    if multiplier:
        try:
            multiplier = int(str(multiplier).replace("x", "").strip())
        except Exception:
            multiplier = None

    return {
        "bonus_type":  bonus_type,
        "utc_time":    utc_dt.isoformat() if utc_dt else None,
        "ist_time":    ist_dt.strftime("%d %b %Y, %I:%M %p IST") if ist_dt else None,
        "ist_hour":    ist_dt.hour if ist_dt else None,
        "ist_weekday": ist_dt.weekday() if ist_dt else None,  # 0=Mon, 6=Sun
        "multiplier":  multiplier,
        "raw":         record
    }

def scrape_historical(max_pages: int = 50):
    """
    Scrape up to max_pages pages from Tracksino.
    Each page has 100 records → up to 5000 records total.
    """
    print(f"🌐 Scraping Tracksino historical data (up to {max_pages} pages)...")
    all_bonus = []
    empty_pages = 0

    for page in range(1, max_pages + 1):
        print(f"  📄 Page {page}/{max_pages}...", end=" ")
        records = fetch_page(page)

        if not records:
            empty_pages += 1
            print("empty")
            if empty_pages >= 3:
                print("  3 empty pages in a row — stopping.")
                break
            time.sleep(2)
            continue

        empty_pages = 0
        found = 0
        for rec in records:
            parsed = parse_record(rec)
            if parsed:
                all_bonus.append(parsed)
                found += 1

        print(f"found {found} bonus rounds (total so far: {len(all_bonus)})")
        time.sleep(0.5)  # be polite to the server

    print(f"\n✅ Total bonus rounds collected: {len(all_bonus)}")
    return all_bonus

def run():
    print("🚀 HISTORICAL DATA SCRAPER STARTING...")

    # Load existing data
    try:
        with open(OUTPUT_FILE, "r") as f:
            existing = json.load(f)
        existing_records = existing.get("records", [])
        print(f"📂 Loaded {len(existing_records)} existing records.")
    except FileNotFoundError:
        existing_records = []
        print("📂 No existing data. Starting fresh.")

    # Scrape new
    new_records = scrape_historical(max_pages=50)

    # Merge and deduplicate by utc_time + bonus_type
    existing_keys = {(r.get("utc_time"), r.get("bonus_type")) for r in existing_records}
    added = 0
    for r in new_records:
        key = (r.get("utc_time"), r.get("bonus_type"))
        if key not in existing_keys:
            existing_records.append(r)
            existing_keys.add(key)
            added += 1

    print(f"➕ Added {added} new records. Total: {len(existing_records)}")

    # Sort by time
    existing_records.sort(key=lambda x: x.get("utc_time") or "")

    output = {
        "last_scraped": datetime.now(timezone.utc).isoformat(),
        "total_records": len(existing_records),
        "records": existing_records
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"💾 Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    run()
