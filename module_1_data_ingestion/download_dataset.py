"""
Dataset Downloader for Project 12
===================================
Causal Feature Attribution for High-Stakes Tabular Decisions with Actionable Recourse

This script downloads the best available credit risk datasets
directly — no Kaggle login, no competition sign-up required.

Run:
    python3 download_dataset.py

Datasets available:
    1. FICO HELOC          — 10,459 rows, 23 numerical features (BEST for this project)
    2. YBI Credit Default  — 1,000 rows, simple numerical features (good for quick testing)
    3. OpenML Credit Card  — 30,000 rows, 23 numerical features (large alternative)
"""

import os
import sys

# ── Install dependencies if missing ──────────────────────────────────────────
def install(package):
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])

try:
    import requests
except ImportError:
    print("📦 Installing 'requests'...")
    install("requests")
    import requests

try:
    import pandas as pd
except ImportError:
    print("📦 Installing 'pandas'...")
    install("pandas")
    import pandas as pd

# ── Dataset definitions ───────────────────────────────────────────────────────

DATASETS = {
    "1": {
        "name": "FICO HELOC (Recommended ⭐)",
        "url": "https://raw.githubusercontent.com/benoitparis/explainable-challenge/master/heloc_dataset_v1.csv",
        "filename": "fico_heloc.csv",
        "description": "10,459 rows | 23 numerical features | XAI gold-standard benchmark",
        "target_col": "RiskPerformance",
        "notes": "All features are numerical credit bureau variables (delinquency counts, "
                 "trade history, utilization). Perfect for causal graph learning."
    },
    "2": {
        "name": "YBI Credit Default",
        "url": "https://raw.githubusercontent.com/YBIFoundation/Dataset/main/Credit%20Default.csv",
        "filename": "ybi_credit_default.csv",
        "description": "1,000 rows | Simple numerical features | Good for quick testing",
        "target_col": "default",
        "notes": "Smaller dataset. Good for prototyping and testing the pipeline quickly."
    },
    "3": {
        "name": "Default of Credit Card Clients (UCI)",
        "url": "https://raw.githubusercontent.com/dsrscientist/dataset1/master/default-of-credit-card-clients.csv",
        "filename": "uci_credit_card_default.csv",
        "description": "30,000 rows | 23 numerical features | Large & comprehensive",
        "target_col": "default.payment.next.month",
        "notes": "Taiwan credit card data. Features include payment history, bill amounts, "
                 "credit limit — all numerical and causal-inference friendly."
    },
}

BACKUP_URLS = {
    "1": [
        "https://raw.githubusercontent.com/interpretml/interpret/master/docs/interpret/python/examples/data/heloc_dataset_v1.csv",
    ],
    "3": [
        "https://raw.githubusercontent.com/MJeremy2017/Machine-Learning-Models/master/Credit%20Card%20Default/UCI_Credit_Card.csv",
    ]
}

# ── Helper functions ──────────────────────────────────────────────────────────

def download_csv(url, filename, backup_urls=None):
    """Download CSV from URL with fallback support."""
    all_urls = [url] + (backup_urls or [])

    for attempt, u in enumerate(all_urls, 1):
        try:
            print(f"   🌐 Trying URL {attempt}/{len(all_urls)}...")
            headers = {"User-Agent": "Mozilla/5.0 (compatible; DatasetDownloader/1.0)"}
            response = requests.get(u, headers=headers, timeout=30)
            response.raise_for_status()

            with open(filename, "wb") as f:
                f.write(response.content)

            print(f"   ✅ Downloaded successfully!")
            return True

        except requests.exceptions.HTTPError as e:
            print(f"   ❌ HTTP Error: {e}")
        except requests.exceptions.ConnectionError:
            print(f"   ❌ Connection failed (check your internet)")
        except requests.exceptions.Timeout:
            print(f"   ❌ Request timed out")
        except Exception as e:
            print(f"   ❌ Error: {e}")

    return False


def validate_and_preview(filename, target_col):
    """Load the CSV, show shape, columns, dtypes, and sample rows."""
    try:
        df = pd.read_csv(filename)

        print(f"\n{'='*60}")
        print(f"  📊 DATASET SUMMARY")
        print(f"{'='*60}")
        print(f"  Rows        : {df.shape[0]:,}")
        print(f"  Columns     : {df.shape[1]}")
        print(f"  File size   : {os.path.getsize(filename) / 1024:.1f} KB")
        print(f"{'='*60}")

        print(f"\n  📋 COLUMNS & DATA TYPES:")
        print(f"  {'Column':<45} {'Type':<15} {'Nulls'}")
        print(f"  {'-'*70}")
        for col in df.columns:
            null_count = df[col].isnull().sum()
            marker = " ⬅ TARGET" if col == target_col else ""
            print(f"  {col:<45} {str(df[col].dtype):<15} {null_count}{marker}")

        print(f"\n  🔍 SAMPLE ROWS (first 3):")
        print(df.head(3).to_string(index=False))

        # Quick class balance check
        if target_col in df.columns:
            counts = df[target_col].value_counts()
            print(f"\n  ⚖️  TARGET CLASS BALANCE ('{target_col}'):")
            for val, count in counts.items():
                pct = count / len(df) * 100
                print(f"     {val}: {count:,} rows ({pct:.1f}%)")

        return df

    except Exception as e:
        print(f"   ⚠️  Could not validate file: {e}")
        return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  📥 DATASET DOWNLOADER — Project 12")
    print("  Causal Feature Attribution with Actionable Recourse")
    print("="*60)

    print("\n  Available datasets:\n")
    for key, ds in DATASETS.items():
        print(f"  [{key}] {ds['name']}")
        print(f"       {ds['description']}")
        print()

    # ── Let user choose ──
    print("  Enter dataset number to download (or 'all' to download all): ", end="")
    choice = input().strip().lower()

    if choice == "all":
        choices = list(DATASETS.keys())
    elif choice in DATASETS:
        choices = [choice]
    else:
        print(f"\n  ❌ Invalid choice '{choice}'. Please enter 1, 2, 3, or 'all'.")
        sys.exit(1)

    # ── Download chosen datasets ──
    downloaded = []
    for key in choices:
        ds = DATASETS[key]
        print(f"\n{'─'*60}")
        print(f"  ⬇️  Downloading: {ds['name']}")
        print(f"  📝 Note: {ds['notes']}")
        print(f"{'─'*60}")

        backup = BACKUP_URLS.get(key, [])
        success = download_csv(ds["url"], ds["filename"], backup)

        if success:
            df = validate_and_preview(ds["filename"], ds["target_col"])
            if df is not None:
                downloaded.append((ds["filename"], df))
        else:
            print(f"\n  ❌ Failed to download '{ds['name']}'.")
            print(f"     💡 Try manually visiting:")
            print(f"     {ds['url']}")

    # ── Final summary ──
    print(f"\n{'='*60}")
    print(f"  ✅ DOWNLOAD COMPLETE")
    print(f"{'='*60}")
    for fname, df in downloaded:
        path = os.path.abspath(fname)
        print(f"\n  📁 File   : {fname}")
        print(f"  📍 Path   : {path}")
        print(f"  📊 Shape  : {df.shape[0]:,} rows × {df.shape[1]} columns")

    if not downloaded:
        print("\n  ⚠️  No files were downloaded. Check your internet connection.")
    else:
        print(f"\n  💡 NEXT STEP: Load in your project with:")
        print(f"     import pandas as pd")
        for fname, _ in downloaded:
            varname = fname.replace(".csv", "").replace("-", "_")
            print(f"     {varname} = pd.read_csv('{fname}')")

    print()


if __name__ == "__main__":
    main()
