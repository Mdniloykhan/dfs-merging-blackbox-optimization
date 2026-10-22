#!/usr/bin/env python3
# ============================================================
# GITHUB ADVISORY DATABASE BULK PARSER
# Downloads complete GHSA database and extracts
# supply chain relevant incidents
# No API key needed. No rate limits.
# ============================================================
# Usage:
#   pip install requests pandas tqdm
#   python parse_ghsa_db.py --input oss_threat_dataset_beta.csv
#
# Step 1: Downloads advisory-database-main.zip (~500MB)
# Step 2: Extracts and parses all JSON advisories
# Step 3: Filters supply chain relevant ones
# Step 4: Labels with rule-based classifier
# Step 5: Merges with your existing dataset
# Output: oss_threat_dataset_ghsa.csv
# ============================================================

import requests
import pandas as pd
import json
import os
import zipfile
import hashlib
import argparse
from tqdm import tqdm
from collections import Counter
from pathlib import Path

VALID_LABELS = ["AV-200", "AV-300", "AV-400", "AV-410", "AV-509"]

SUPPLY_CHAIN_KEYWORDS = [
    "malicious", "backdoor", "typosquat", "dependency confusion",
    "supply chain", "pipeline", "postinstall", "setup.py",
    "unicode bidirectional", "trojan source", "malware", "trojan",
    "credential steal", "exfiltrat", "cryptominer", "reverse shell",
    "ci/cd", "github action", "workflow inject", "package hijack",
    "namespace confusion", "brandjack", "impersonat", "mimic",
    "fake package", "malicious package", "npm install", "pip install",
    "install hook", "lifecycle hook", "dropper", "obfuscat",
    "token steal", "secret steal", "environment variable steal"
]

def is_supply_chain(text):
    t = text.lower()
    return any(kw in t for kw in SUPPLY_CHAIN_KEYWORDS)

def classify_rule_based(title, description):
    text = (title + " " + description).lower()

    # AV-300 — Trojan Source / Unicode
    if any(kw in text for kw in [
        "unicode bidirectional","bidi character","trojan source",
        "invisible character","zero-width","homoglyph",
        "bidirectional control","rtlo","right-to-left override",
        "cve-2021-42574"]):
        return "AV-300", 0.95

    # AV-509 — Dependency Confusion
    if any(kw in text for kw in [
        "dependency confusion","namespace confusion",
        "internal package","private package","public registry override",
        "registry priority","internal dependency","private dependency",
        "confused deputy","private repo","internal repo"]):
        return "AV-509", 0.90

    # AV-410 — Pipeline Poisoning
    if any(kw in text for kw in [
        "github actions","workflow inject","poisoned pipeline",
        "ppe attack","pipeline poison","ci/cd compromise",
        "workflow file","runner worker","actions secret",
        "pipeline execution","workflow token","pull request injection",
        "cicd secret","build pipeline compromis","github action workflow"]):
        return "AV-410", 0.90

    # AV-200 — Typosquatting
    if any(kw in text for kw in [
        "typosquat","typo-squat","misspell","impersonat",
        "mimic","look-alike","lookalike","character substitut",
        "brandjack","name squatt","one character","one-character",
        "namespace impersonat","fake package","package impersonat",
        "copied package","similar name"]):
        return "AV-200", 0.88

    # AV-400 — Malicious Builds (strong signals)
    if any(kw in text for kw in [
        "postinstall","setup.py","install script","lifecycle hook",
        "malicious code execut","code execut on install",
        "backdoor inject","reverse shell","cryptominer",
        "credential steal","exfiltrat","malicious payload",
        "dropper","npm install malicious","pip install malicious",
        "malicious npm","malicious pypi","malicious package"]):
        return "AV-400", 0.85

    # Broader patterns
    if any(kw in text for kw in ["ci/cd","pipeline","workflow","continuous integration"]):
        return "AV-410", 0.70
    if any(kw in text for kw in ["malicious","backdoor","trojan","malware","obfuscat"]):
        return "AV-400", 0.70

    return "UNCERTAIN", 0.0


def download_ghsa_database(output_path="advisory-database-main.zip"):
    """Download the complete GitHub Advisory Database."""
    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / 1024 / 1024
        print(f"  ZIP already exists ({size_mb:.1f} MB). Skipping download.")
        return output_path

    print("  Downloading GitHub Advisory Database (~500MB)...")
    print("  This may take 5-15 minutes depending on your connection.")

    url = "https://github.com/github/advisory-database/archive/refs/heads/main.zip"

    try:
        resp = requests.get(url, stream=True, timeout=300)
        total = int(resp.headers.get('content-length', 0))

        with open(output_path, 'wb') as f:
            with tqdm(total=total, unit='B', unit_scale=True,
                     desc="Downloading") as pbar:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))

        size_mb = os.path.getsize(output_path) / 1024 / 1024
        print(f"  Downloaded {size_mb:.1f} MB")
        return output_path

    except Exception as e:
        print(f"  Download failed: {e}")
        return None


def parse_ghsa_database(zip_path, max_entries=20000):
    """Parse all advisory JSON files from the ZIP."""
    print(f"\nParsing advisory database...")
    entries = []
    skipped = 0
    relevant = 0

    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            # Get all advisory JSON files
            json_files = [f for f in z.namelist()
                         if f.endswith('.json')
                         and '/advisories/' in f
                         and not f.endswith('index.json')]

            print(f"  Found {len(json_files)} advisory JSON files")
            print(f"  Filtering for supply chain relevant entries...")

            for fname in tqdm(json_files[:max_entries]):
                try:
                    with z.open(fname) as f:
                        data = json.load(f)

                    # Extract fields
                    ghsa_id = data.get("id", "")
                    summary = data.get("summary", "")
                    details = data.get("details", "")
                    severity = data.get("severity", "")
                    published = data.get("published", "")[:4]

                    # Get affected packages
                    affected = data.get("affected", [])
                    ecosystem = ""
                    pkg_name = ""
                    if affected:
                        pkg = affected[0].get("package", {})
                        ecosystem = pkg.get("ecosystem", "")
                        pkg_name = pkg.get("name", "")

                    # Get references
                    refs = data.get("references", [])
                    source_url = ""
                    for ref in refs:
                        if ref.get("type") == "WEB":
                            source_url = ref.get("url", "")
                            break
                    if not source_url:
                        source_url = f"https://github.com/advisories/{ghsa_id}"

                    # Filter for supply chain relevance
                    combined = (summary + " " + (details or "") + " " + pkg_name)
                    if not is_supply_chain(combined):
                        skipped += 1
                        continue

                    # Build title
                    if pkg_name:
                        title = f"{pkg_name}: {summary[:150]}"
                    else:
                        title = summary[:200]

                    entries.append({
                        "id": ghsa_id.lower().replace("-", ""),
                        "title": title[:200],
                        "description": (details or summary)[:1000],
                        "label": "",
                        "predicted_label": "",
                        "source": source_url,
                        "ecosystem": ecosystem,
                        "year": published,
                        "raw_source": "GHSA-DB"
                    })
                    relevant += 1

                except Exception:
                    continue

    except Exception as e:
        print(f"  Error parsing ZIP: {e}")
        return []

    print(f"  Relevant entries found: {relevant}")
    print(f"  Non-supply-chain skipped: {skipped}")
    return entries


def deduplicate(entries):
    print(f"\nDeduplicating {len(entries)} entries...")
    seen_ids = set()
    seen_sources = set()
    seen_hashes = set()
    unique = []

    for entry in entries:
        eid = str(entry.get("id", "")).strip()
        src = str(entry.get("source", "")).strip()
        title = str(entry.get("title", "")).lower().strip()
        thash = hashlib.md5(title.encode()).hexdigest()

        if eid and eid in seen_ids:
            continue
        if src and src in seen_sources:
            continue
        if thash in seen_hashes:
            continue
        if len(str(entry.get("description", ""))) < 30:
            continue
        if not title:
            continue

        if eid: seen_ids.add(eid)
        if src: seen_sources.add(src)
        seen_hashes.add(thash)
        unique.append(entry)

    print(f"After dedup: {len(unique)} unique entries")
    return unique


def label_entries(entries):
    print(f"\nApplying rule-based classification...")
    labeled = []
    uncertain = []

    for entry in entries:
        if entry.get("label") in VALID_LABELS:
            labeled.append(entry)
            continue

        label, conf = classify_rule_based(
            entry.get("title", ""),
            entry.get("description", "")
        )

        if label != "UNCERTAIN" and conf >= 0.70:
            entry["label"] = label
            entry["predicted_label"] = label
            labeled.append(entry)
        else:
            entry["label"] = "REVIEW_NEEDED"
            uncertain.append(entry)

    print(f"  Labeled: {len(labeled)}")
    print(f"  Uncertain: {len(uncertain)}")
    return labeled, uncertain


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='oss_threat_dataset_beta.csv')
    parser.add_argument('--merged', default='oss_threat_dataset_merged.csv')
    parser.add_argument('--output', default='oss_threat_dataset_ghsa.csv')
    parser.add_argument('--zip', default='advisory-database-main.zip')
    parser.add_argument('--max', type=int, default=50000)
    args = parser.parse_args()

    print("=" * 60)
    print("GITHUB ADVISORY DATABASE BULK PARSER")
    print("=" * 60)

    # Load existing datasets
    existing = []
    for csv_path in [args.input, args.merged]:
        try:
            df = pd.read_csv(csv_path)
            existing.extend(df.to_dict('records'))
            print(f"Loaded {csv_path}: {len(df)} entries")
        except FileNotFoundError:
            print(f"Not found: {csv_path}")

    print(f"Total existing: {len(existing)}")

    # Download GHSA database
    print(f"\n[1/3] Downloading GitHub Advisory Database...")
    zip_path = download_ghsa_database(args.zip)
    if not zip_path:
        print("Download failed. Exiting.")
        return

    # Parse database
    print(f"\n[2/3] Parsing advisories...")
    new_entries = parse_ghsa_database(zip_path, args.max)
    print(f"Parsed {len(new_entries)} supply chain entries")

    # Combine all
    all_entries = existing + new_entries
    all_entries = deduplicate(all_entries)

    # Label
    print(f"\n[3/3] Labeling...")
    labeled, uncertain = label_entries(all_entries)

    # Stats
    dist = Counter(e["label"] for e in labeled
                   if e.get("label") in VALID_LABELS)
    print(f"\n--- FINAL RESULTS ---")
    print(f"Total labeled entries: {len(labeled)}")
    print(f"\nLabel distribution:")
    for label in VALID_LABELS:
        count = dist.get(label, 0)
        pct = count / len(labeled) * 100 if labeled else 0
        print(f"  {label}: {count} ({pct:.1f}%)")

    # Save
    cols = ["id", "title", "description", "label",
            "predicted_label", "source", "ecosystem", "year"]
    df_out = pd.DataFrame(labeled)
    for col in cols:
        if col not in df_out.columns:
            df_out[col] = ""
    df_out[cols].to_csv(args.output, index=False)
    print(f"\nSaved {len(labeled)} entries to {args.output}")

    if uncertain:
        rev_path = args.output.replace('.csv', '_review.csv')
        df_rev = pd.DataFrame(uncertain)
        for col in cols:
            if col not in df_rev.columns:
                df_rev[col] = ""
        df_rev[cols].to_csv(rev_path, index=False)
        print(f"Saved {len(uncertain)} uncertain to {rev_path}")

    target = 5000
    if len(labeled) >= target:
        print(f"\n✅ Target of {target} reached!")
    else:
        print(f"\n⚠️ Got {len(labeled)} of {target} target")
        print("Next: Manually verify 200 entries for IRR score")

    print("\nDone.")

if __name__ == "__main__":
    main()
