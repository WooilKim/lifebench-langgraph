"""Fetch Korean name ranking data from koreanname.me and save as JSON.

API: GET https://koreanname.me/api/rank/{startYear}/{endYear}/{page}
Returns: {male: [{name, rank, count}], female: [...], maleHasNext, femaleHasNext}

Usage:
    python scripts/fetch_korean_names.py
    → saves data/korean_names.json
"""
import json
import time
import urllib.request
from pathlib import Path


API_BASE   = "https://koreanname.me/api/rank"
START_YEAR = 2008
END_YEAR   = 2024
MAX_PAGES  = 20   # 100 names/page × 20 pages = 2,000 names per gender (충분)
OUT_PATH   = Path(__file__).parent.parent / "data" / "korean_names.json"


def fetch_page(page: int) -> dict:
    url = f"{API_BASE}/{START_YEAR}/{END_YEAR}/{page}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_all():
    male_names   = []
    female_names = []

    for page in range(1, MAX_PAGES + 1):
        try:
            data = fetch_page(page)
        except Exception as e:
            print(f"  Page {page} error: {e}. Stopping.")
            break

        m = data.get("male", [])
        f = data.get("female", [])
        male_names.extend(m)
        female_names.extend(f)

        m_has_next = data.get("maleHasNext", False)
        f_has_next = data.get("femaleHasNext", False)

        print(f"  Page {page}: +{len(m)} male, +{len(f)} female "
              f"(total {len(male_names)}M / {len(female_names)}F) "
              f"hasNext={m_has_next}/{f_has_next}")

        if not m_has_next and not f_has_next:
            print("  No more pages.")
            break

        time.sleep(0.3)  # 서버 부하 방지

    return male_names, female_names


def main():
    print(f"Fetching Korean name rankings from koreanname.me ({START_YEAR}–{END_YEAR})...")
    male_names, female_names = fetch_all()

    result = {
        "source":     "koreanname.me",
        "start_year": START_YEAR,
        "end_year":   END_YEAR,
        "male":   sorted(male_names,   key=lambda x: x["rank"]),
        "female": sorted(female_names, key=lambda x: x["rank"]),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(male_names)} male + {len(female_names)} female names → {OUT_PATH}")
    print(f"Top 5 male:   {[n['name'] for n in male_names[:5]]}")
    print(f"Top 5 female: {[n['name'] for n in female_names[:5]]}")


if __name__ == "__main__":
    main()
